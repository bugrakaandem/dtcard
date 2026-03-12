import argparse
import os
import json
import copy
import pickle
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm

from utils.logger import setup_logger
from games.hearts import HeartsGame
from games.whist import WhistGame
from games.spades import SpadesGame

from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from agents.whist_rule_based_agent import WhistRuleBasedAgent
from agents.spades_rule_based_agent import SpadesRuleBasedAgent
from agents.dqn_agent import QNetwork, DQNAgentWrapper

from models.decision_transformer import DecisionTransformer
from models.decision_convformer import DecisionConvformer

ACT_DIM = 52


HYPERPARAM_SCHEMES = {
    "baseline": {"n_layer": 2, "n_head": 2, "hidden_size": 64, "batch_size": 32, "lr": 5e-4, "dropout": .1},
    "compact": {"n_layer": 3, "n_head": 4, "hidden_size": 128, "batch_size": 64, "lr": 1e-4, "dropout": .1},
    "large": {"n_layer": 6, "n_head": 8, "hidden_size": 256, "batch_size": 64, "lr": 1e-4}, "dropout": .1,
    "custom": {"n_layer": 6, "n_head": 8, "hidden_size": 512, "batch_size": 104, "lr": 1e-4, "dropout": .1},
}


class TrajectoryDataset(Dataset):
    def __init__(self, data_path: Optional[str] = None, data_buffer: Optional[Dict[str, Any]] = None, context_len: int = 20):
        self.context_len = int(context_len)

        if data_buffer is not None:
            data = data_buffer
        elif data_path is not None:
            with open(data_path, "rb") as f:
                data = pickle.load(f)
        else:
            raise ValueError("TrajectoryDataset requires data_path or data_buffer")

        self.observations = np.asarray(data["observations"], dtype=np.float32)
        self.actions = np.asarray(data["actions"], dtype=np.int64)
        self.rtg = np.asarray(data["rtg"], dtype=np.float32)
        self.done_idxs = np.asarray(data["done_idxs"], dtype=np.int64)

        self.state_dim = int(self.observations.shape[-1])
        self.n_steps = int(self.actions.shape[0])

        if "legal_masks" in data and data["legal_masks"] is not None:
            lm = np.asarray(data["legal_masks"], dtype=np.uint8)
            if lm.ndim == 2 and lm.shape[0] == self.n_steps and lm.shape[1] == ACT_DIM:
                self.legal_masks = lm
            else:
                self.legal_masks = None
        else:
            self.legal_masks = None

        self._ep_ends = self.done_idxs
        self._ep_starts = np.concatenate([[0], self.done_idxs[:-1]])

    def __len__(self) -> int:
        return self.n_steps

    def __getitem__(self, idx: int):
        ep = int(np.searchsorted(self._ep_ends, idx, side="right"))
        ep_start = int(self._ep_starts[ep])

        start_idx = max(ep_start, idx - self.context_len + 1)
        seq_len = idx - start_idx + 1
        pad_len = self.context_len - seq_len

        s = self.observations[start_idx: idx + 1]
        a = self.actions[start_idx: idx + 1]
        r = self.rtg[start_idx: idx + 1]
        t = np.arange(start_idx - ep_start, idx - ep_start + 1, dtype=np.int64)

        states = np.zeros((self.context_len, self.state_dim), dtype=np.float32)
        actions = np.zeros((self.context_len,), dtype=np.int64)
        rtgs = np.zeros((self.context_len, 1), dtype=np.float32)
        timesteps = np.zeros((self.context_len,), dtype=np.int64)
        mask = np.zeros((self.context_len,), dtype=np.float32)

        states[pad_len:] = s
        actions[pad_len:] = a
        rtgs[pad_len:, 0] = r
        timesteps[pad_len:] = t
        mask[pad_len:] = 1.0

        if self.legal_masks is None:
            legal = np.ones((self.context_len, ACT_DIM), dtype=np.float32)
        else:
            lm = self.legal_masks[start_idx: idx + 1].astype(np.float32)
            legal = np.ones((self.context_len, ACT_DIM), dtype=np.float32)
            legal[pad_len:] = lm

        return (
            torch.from_numpy(states),
            torch.from_numpy(actions),
            torch.from_numpy(rtgs),
            torch.from_numpy(timesteps),
            torch.from_numpy(mask),
            torch.from_numpy(legal),
        )


class DecisionTransformerAgent:
    def __init__(self, model: DecisionTransformer, context_len: int, device: str, target_return: float = 0.0, state_dim: Optional[int] = None):
        self.model = model
        self.context_len = int(context_len)
        self.device = device
        self.target_return = float(target_return)
        self.state_dim = int(state_dim) if state_dim is not None else int(getattr(model, "state_dim", 0))
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.timesteps = 0
        self.running_rtg = float(self.target_return)

    def update_rtg(self, reward: float):
        self.running_rtg = float(self.running_rtg) - float(reward)

    def _push_state(self, obs: np.ndarray):
        obs = _adapt_obs(obs, self.state_dim)
        self.states.append(torch.tensor(obs, device=self.device, dtype=torch.float32))

    def select_action(self, obs: np.ndarray, legal_moves, temperature: float = 0.0, return_info: bool = False):
        self._push_state(obs)

        states = torch.stack(self.states[-self.context_len:]).unsqueeze(0)
        K = states.shape[1]

        rtgs = torch.full((1, K, 1), float(self.running_rtg), device=self.device, dtype=torch.float32)
        timesteps = torch.arange(max(0, self.timesteps - K + 1), self.timesteps + 1, device=self.device, dtype=torch.long).unsqueeze(0)

        if len(self.actions) > 0:
            actions = torch.stack(self.actions[-self.context_len:]).unsqueeze(0)
            actions = torch.cat([actions, torch.zeros((1, 1), device=self.device, dtype=torch.long)], dim=1)
            actions = actions[:, -K:]
        else:
            actions = torch.zeros((1, K), device=self.device, dtype=torch.long)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(states, actions, rtgs, timesteps)[0, -1, :]

        top1 = int(torch.argmax(logits).item())
        legal_ids = set(_as_action_id(m) for m in legal_moves)
        illegal_top1 = 1 if top1 not in legal_ids else 0

        masked_logits = torch.full_like(logits, float("-inf"))
        for mid in legal_ids:
            if 0 <= mid < ACT_DIM:
                masked_logits[mid] = logits[mid]

        if temperature > 0:
            probs = F.softmax(masked_logits / temperature, dim=-1)
            try:
                chosen = int(torch.multinomial(probs, num_samples=1).item())
            except RuntimeError:
                chosen = int(torch.argmax(masked_logits).item())
        else:
            chosen = int(torch.argmax(masked_logits).item())

        self.actions.append(torch.tensor(chosen, device=self.device, dtype=torch.long))
        self.timesteps += 1

        if return_info:
            return chosen, {"illegal_top1": illegal_top1, "top1": top1}
        return chosen


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _as_action_id(a: Any) -> int:
    return int(a) if isinstance(a, (int, np.integer)) else int(getattr(a, "id", a))


def _build_game(game_name: str):
    if game_name == "hearts":
        return HeartsGame()
    if game_name == "whist":
        return WhistGame()
    if game_name == "spades":
        return SpadesGame()
    raise ValueError(f"Unknown game: {game_name}")


def _adapt_obs(obs: np.ndarray, state_dim: int) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    if obs.shape[0] == state_dim:
        return obs
    if obs.shape[0] > state_dim:
        return obs[:state_dim]
    pad = np.zeros((state_dim - obs.shape[0],), dtype=np.float32)
    return np.concatenate([obs, pad], axis=0)


def select_target_return(dataset_dict: Dict[str, Any], percentile: float, logger=None) -> float:
    done_idxs = np.asarray(dataset_dict["done_idxs"], dtype=np.int64)
    rtg = np.asarray(dataset_dict["rtg"], dtype=np.float32)

    starts = np.concatenate([[0], done_idxs[:-1]])
    if len(starts) == 0:
        return 0.0

    episode_returns = rtg[starts]
    if len(episode_returns) == 0:
        return 0.0

    vals = np.sort(episode_returns)
    target = float(np.percentile(vals, percentile))

    uniq = np.unique(vals)
    if len(uniq) > 1 and np.isclose(target, uniq[-1]):
        if logger:
            logger.info(f"Target clamped from max {target:.2f} to second-max {uniq[-2]:.2f} to avoid OOD RTG.")
        target = float(uniq[-2])

    if logger:
        logger.info(f"Dataset Return Stats: Mean={float(np.mean(episode_returns)):.4f}, Max={float(np.max(episode_returns)):.4f}")
        logger.info(f"Selected Target ({percentile}%): {target:.4f}")

    return float(target)


def _build_opponents(game_name: str, oppo_type: str):
    if game_name == "whist":
        return [WhistRuleBasedAgent(i) if oppo_type == "rule_based" else RandomAgent(i) for i in range(1, 4)]
    if game_name == "spades":
        return [SpadesRuleBasedAgent(i) if oppo_type == "rule_based" else RandomAgent(i) for i in range(1, 4)]
    return [RuleBasedAgent(i) if oppo_type == "rule_based" else RandomAgent(i) for i in range(1, 4)]


def evaluate_in_simulation(
    game_name: str,
    model: DecisionTransformer,
    game,
    logger,
    num_games: int = 2000,
    device: str = "cpu",
    target_return: float = 0.0,
    context_len: int = 20,
    oppo_type: str = "random",
    log_diagnostics: bool = False,
    return_details: bool = False,
    custom_opponents: list = None
):
    dt_agent = DecisionTransformerAgent(model, context_len, device, target_return=float(target_return), state_dim=int(getattr(model, "state_dim", 0)))
    opponents = custom_opponents if custom_opponents is not None else _build_opponents(game_name, oppo_type)

    wins = 0
    total_score = 0.0
    illegal_top1 = 0
    decisions = 0
    debug_rtg_trace = []

    for g in tqdm(range(num_games)):
        obs = game.reset()
        dt_agent.reset()
        done = False
        steps = 0
        game_rewards = [0.0, 0.0, 0.0, 0.0]

        while not done:
            curr = int(game.current_player_idx)
            legal = game.get_legal_moves(curr)

            if curr == 0:
                if log_diagnostics and g == 0:
                    debug_rtg_trace.append(f"S{steps}:RTG={dt_agent.running_rtg:.2f}")
                action, info = dt_agent.select_action(obs, legal, temperature=0.0, return_info=True)
                illegal_top1 += int(info["illegal_top1"])
                decisions += 1
            else:
                action = opponents[curr - 1].select_action(obs, legal)

            obs, rewards, done, _ = game.step(action)

            if rewards is not None and float(rewards[0]) != 0.0:
                dt_agent.update_rtg(float(rewards[0]))

            for i in range(4):
                game_rewards[i] += float(rewards[i])
            steps += 1

        if game_rewards[0] == max(game_rewards):
            wins += 1
        total_score += float(game_rewards[0])

    if log_diagnostics:
        logger.info(f"[DIAGNOSTIC] Trace: {debug_rtg_trace[:5]} ... {debug_rtg_trace[-3:]}")

    win_rate = (wins / max(1, num_games)) * 100.0
    avg_score = total_score / max(1, num_games)
    illegal_rate = (illegal_top1 / max(1, decisions)) * 100.0

    logger.info(f"Eval ({num_games} games): Win Rate: {win_rate:.1f}% | Avg Score: {avg_score:.2f} | Illegal(top1): {illegal_rate:.2f}%")

    if return_details:
        return {
            "win_rate": float(win_rate),
            "avg_score": float(avg_score),
            "illegal_rate": float(illegal_rate),
            "eval_games": int(num_games),
            "decisions": int(decisions),
        }
    return float(win_rate)


def collect_experience(
        game_name: str,
        model: DecisionTransformer,
        game,
        num_games: int = 2000,
        device: str = "cpu",
        target_return: float = 0.0,
        context_len: int = 20,
        oppo_type: str = "random",
        good_threshold: Optional[float] = None,
        custom_opponents: list = None,
        logger=None
):
    dt_agent = DecisionTransformerAgent(model, context_len, device, target_return=float(target_return),
                                        state_dim=int(getattr(model, "state_dim", 0)))
    opponents = custom_opponents if custom_opponents is not None else _build_opponents(game_name, oppo_type)

    good_trajectories = []

    for _ in range(num_games):
        obs = game.reset()
        dt_agent.reset()

        done = False
        game_rewards = [0.0, 0.0, 0.0, 0.0]
        traj_obs, traj_act, traj_rew = [], [], []

        while not done:
            curr = int(game.current_player_idx)
            legal = game.get_legal_moves(curr)

            if curr == 0:
                action = dt_agent.select_action(obs, legal, temperature=0.0, return_info=False)
                traj_obs.append(_adapt_obs(obs, int(getattr(model, "state_dim", 0))))
                traj_act.append(int(action))
                traj_rew.append(0.0)
            else:
                action = opponents[curr - 1].select_action(obs, legal)

            obs, rewards, done, _ = game.step(action)

            if rewards is not None and float(rewards[0]) != 0.0:
                if len(traj_rew) > 0:
                    traj_rew[-1] += float(rewards[0])
                dt_agent.update_rtg(float(rewards[0]))

            for i in range(4):
                game_rewards[i] += float(rewards[i])

        is_win = (game_rewards[0] == max(game_rewards))
        is_good = True if good_threshold is None else (float(game_rewards[0]) >= float(good_threshold))

        if is_win and is_good and len(traj_obs) > 0:
            rtgs = np.zeros((len(traj_rew),), dtype=np.float32)
            running = 0.0
            for t in reversed(range(len(traj_rew))):
                running += float(traj_rew[t])
                rtgs[t] = running

            opponents_scores = game_rewards[1:4]
            margin = float(game_rewards[0]) - float(max(opponents_scores))

            good_trajectories.append({
                "obs": traj_obs, "act": traj_act, "rew": traj_rew, "rtg": rtgs.tolist(), "margin": margin
            })

    if len(good_trajectories) == 0:
        return None, 0

    margins = [t["margin"] for t in good_trajectories]
    median_margin = float(np.median(margins))

    narrow_wins = [t for t in good_trajectories if t["margin"] <= median_margin]
    decisive_wins = [t for t in good_trajectories if t["margin"] > median_margin]

    min_len = min(len(narrow_wins), len(decisive_wins))
    if min_len > 0:
        narrow_wins = narrow_wins[:min_len]
        decisive_wins = decisive_wins[:min_len]
        balanced_trajectories = narrow_wins + decisive_wins
    else:
        balanced_trajectories = good_trajectories

    if logger:
        logger.info(
            f"[EI] Collected {len(good_trajectories)} raw wins. Stratified into {len(narrow_wins)} narrow and {len(decisive_wins)} decisive wins.")

    new_data = {"observations": [], "actions": [], "rtg": [], "rewards": [], "done_idxs": []}
    running_len = 0

    for traj in balanced_trajectories:
        new_data["observations"].extend(traj["obs"])
        new_data["actions"].extend(traj["act"])
        new_data["rewards"].extend(traj["rew"])
        new_data["rtg"].extend(traj["rtg"])

        running_len += len(traj["obs"])
        new_data["done_idxs"].append(running_len)

    new_data["observations"] = np.array(new_data["observations"], dtype=np.float32)
    new_data["actions"] = np.array(new_data["actions"], dtype=np.int64)
    new_data["rtg"] = np.array(new_data["rtg"], dtype=np.float32)
    new_data["rewards"] = np.array(new_data["rewards"], dtype=np.float32)
    new_data["done_idxs"] = np.array(new_data["done_idxs"], dtype=np.int64)

    return new_data, len(balanced_trajectories)


def _masked_ce_loss(logits, actions, active_mask, legal_masks=None, train_action_mask: bool = False):
    B, K, C = logits.shape
    active = active_mask.reshape(-1) > 0

    if train_action_mask and legal_masks is not None:
        masked_logits = logits.clone()
        masked_logits[legal_masks <= 0.0] = -1e9
        flat_logits = masked_logits.reshape(-1, C)[active]
    else:
        flat_logits = logits.reshape(-1, C)[active]

    flat_actions = actions.reshape(-1)[active]
    if flat_logits.shape[0] == 0:
        return None
    return F.cross_entropy(flat_logits, flat_actions)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--game", type=str, default="hearts", choices=["hearts", "whist", "spades"])
    parser.add_argument("--dataset", type=str, default="data/hearts_random_1000.pkl")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--config", type=str, default="custom", choices=list(HYPERPARAM_SCHEMES.keys()))

    parser.add_argument("--model", type=str, default="dt", choices=["dt", "dc"],
                        help="Sequence model backbone: dt=DecisionTransformer, dc=DecisionConvFormer")
    parser.add_argument("--dc_kernel", type=int, default=3,
                        help="Decision ConvFormer causal conv kernel size (only used when --model dc).")

    parser.add_argument("--pretrain_epochs", type=int, default=3)
    parser.add_argument("--expert_iters", type=int, default=10)
    parser.add_argument("--expert_epochs", type=int, default=2)
    parser.add_argument("--test_game_count", type=int, default=1_000)
    parser.add_argument("--target_quantile", type=float, default=75.0)

    parser.add_argument("--oppo_type", type=str, default="rule_based", choices=["random", "rule_based"])
    parser.add_argument("--dqn_ckpt", type=str, default=None, help="Path to DQN checkpoint")
    parser.add_argument("--dqn_hidden", type=int, default=256, help="Hidden size for DQN model")

    parser.add_argument("--train_action_mask", action="store_true",
                        help="Train-time legality masking using dataset['legal_masks'] (Table IV ablation).")
    parser.add_argument("--context_len", type=int, default=13,
                        help="Override context length (history).")
    parser.add_argument("--accept_tolerance", type=float, default=0.0)
    parser.add_argument("--max_expert_steps", type=int, default=200)

    args = parser.parse_args()

    _ensure_dir(args.output_dir)
    logger = setup_logger(args.output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading dataset...")
    with open(args.dataset, "rb") as f:
        dataset_dict = pickle.load(f)

    state_dim = int(np.asarray(dataset_dict["observations"]).shape[-1])
    act_dim = ACT_DIM

    scheme = HYPERPARAM_SCHEMES[args.config]
    context_len = args.context_len

    target_return = select_target_return(dataset_dict, args.target_quantile, logger=logger)

    ret_max = float(dataset_dict.get("return_max", float(np.max(dataset_dict.get("rtg", [0.0])))))
    base = 5.0
    good_threshold = min(ret_max, max(base, float(target_return)))

    logger.info(f"[DATA] dataset_path = {args.dataset}")
    logger.info(f"[DATA] transitions = {len(dataset_dict['actions'])}, episodes = {len(dataset_dict['done_idxs'])}")
    logger.info(f"[DATA] action_min = {int(np.min(dataset_dict['actions']))}, action_max = {int(np.max(dataset_dict['actions']))}")
    logger.info(f"[DATA] rtg_mean = {float(np.mean(dataset_dict['rtg'])):.4f}, rtg_max = {float(np.max(dataset_dict['rtg'])):.4f}")
    logger.info(f"[MODEL] model = {args.model}")
    logger.info(f"[RUN] game={args.game} oppo_type={args.oppo_type} config={args.config} context_len={context_len} batch={scheme['batch_size']}")

    if args.model == "dt":
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            hidden_size=int(scheme["hidden_size"]),
            max_length=context_len,
            max_ep_len=100,
            n_layer=int(scheme["n_layer"]),
            n_head=int(scheme["n_head"]),
            n_inner=4,
            dropout=float(scheme["dropout"]),
        ).to(device)
    elif args.model == "dc":
        model = DecisionConvformer(
            state_dim=state_dim,
            act_dim=act_dim,
            hidden_size=int(scheme["hidden_size"]),
            max_length=context_len,
            max_ep_len=100,
            n_layer=int(scheme["n_layer"]),
            conv_kernel=int(args.dc_kernel),
            dropout=float(scheme["dropout"]),
        ).to(device)
    else:
        raise ValueError(f"Unknown --model: {args.model}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(scheme["lr"]))

    offline_ds = TrajectoryDataset(data_buffer=dataset_dict, context_len=context_len)
    loader = DataLoader(offline_ds, batch_size=int(scheme["batch_size"]), shuffle=True, drop_last=True)

    best_model_path = os.path.join(args.output_dir, "best_model.pt")
    best_model_phase1_path = os.path.join(args.output_dir, "best_model_phase1.pt")

    experiment_metrics = {
        "game": args.game,
        "oppo_type": args.oppo_type,
        "context_len": context_len,
        "train_action_mask": bool(args.train_action_mask),
        "model_scheme": args.config,
        "model_hparams": {
            "state_dim": int(state_dim),
            "act_dim": int(act_dim),
            "hidden_size": int(scheme["hidden_size"]),
            "max_length": int(context_len),
            "n_layer": int(scheme["n_layer"]),
            "n_head": int(scheme["n_head"]),
            "dropout": float(scheme["dropout"]),
        },
        "target_return": float(target_return),
        "good_threshold": float(good_threshold),
        "eval_games": int(args.test_game_count),
        "model_type": args.model,
        "dc_kernel": int(args.dc_kernel) if args.model == "dc" else None,
        "phase1": {"history": []},
        "phase2": {"history": []},
    }

    logger.info("--- PHASE 1: OFFLINE TRAINING ---")
    best_win_rate = -1e9
    best_avg_score = -1e9
    best_illegal = 0.0

    game = _build_game(args.game)

    custom_opponents = None
    if args.oppo_type == "dqn":
        logger.info(f"Loading DQN opponents from {args.dqn_ckpt}")
        if not args.dqn_ckpt or not os.path.exists(args.dqn_ckpt):
            raise FileNotFoundError(f"DQN checkpoint not found at {args.dqn_ckpt}")

        temp_obs = game.reset()
        state_dim_dqn = int(np.asarray(temp_obs, dtype=np.float32).reshape(-1).shape[0])

        dqn_model = QNetwork(state_dim_dqn, ACT_DIM, args.dqn_hidden).to(device)
        dqn_model.load_state_dict(torch.load(args.dqn_ckpt, map_location=device))
        dqn_model.eval()

        custom_opponents = [DQNAgentWrapper(i, dqn_model, device) for i in range(1, 4)]

    for epoch in range(int(args.pretrain_epochs)):
        model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}")
        for (states, actions, rtgs, timesteps, mask, legal_masks) in pbar:
            states = states.to(device)
            actions = actions.to(device)
            rtgs = rtgs.to(device)
            timesteps = timesteps.to(device)
            mask = mask.to(device)
            legal_masks = legal_masks.to(device)

            preds = model(states, actions, rtgs, timesteps)

            loss = _masked_ce_loss(preds, actions, mask, legal_masks, train_action_mask=bool(args.train_action_mask))
            if loss is None:
                continue

            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().item())
            pbar.set_postfix({"loss": float(loss.detach().item())})

        avg_loss = total_loss / max(1, len(loader))
        logger.info(f"Phase 1 Epoch {epoch + 1}")

        details = evaluate_in_simulation(
            game_name=args.game,
            model=model,
            game=game,
            logger=logger,
            num_games=int(args.test_game_count),
            device=device,
            target_return=target_return,
            context_len=context_len,
            oppo_type=args.oppo_type,
            log_diagnostics=(epoch == 0),
            return_details=True,
            custom_opponents=custom_opponents,
        )
        experiment_metrics["phase1"]["history"].append({"epoch": int(epoch + 1), **details, "loss": float(avg_loss)})

        if details["win_rate"] > best_win_rate:
            best_win_rate = float(details["win_rate"])
            best_avg_score = float(details["avg_score"])
            best_illegal = float(details["illegal_rate"])
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model (Phase 1) saved! Win Rate: {best_win_rate:.1f}%")

    if os.path.exists(best_model_path):
        state = torch.load(best_model_path, map_location="cpu")
        torch.save(state, best_model_phase1_path)

    experiment_metrics["phase1"]["best"] = {
        "win_rate": float(best_win_rate),
        "avg_score": float(best_avg_score),
        "illegal_rate": float(best_illegal),
        "checkpoint": os.path.basename(best_model_phase1_path),
    }

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    logger.info("--- PHASE 2: SELECTIVE EXPERT ITERATION ---")
    online_memory = deque([], maxlen=10)
    accepted = 0

    half_batch_size = max(1, int(scheme["batch_size"]) // 2)
    offline_ft_loader = DataLoader(offline_ds, batch_size=half_batch_size, shuffle=True, drop_last=True)

    for itr in range(int(args.expert_iters)):
        logger.info(f"Iteration {itr + 1}: Collecting Data...")
        new_data, success_count = collect_experience(
            game_name=args.game,
            model=model,
            game=game,
            num_games=int(args.test_game_count),
            device=device,
            target_return=target_return,
            context_len=context_len,
            oppo_type=args.oppo_type,
            good_threshold=good_threshold,
            custom_opponents=custom_opponents,
            logger=logger
        )

        itr_record = {"iter": int(itr + 1), "good_games": int(success_count)}

        if new_data is None or success_count == 0:
            itr_record.update({"skipped": True})
            experiment_metrics["phase2"]["history"].append(itr_record)
            continue

        online_memory.append(new_data)

        online_sets = [TrajectoryDataset(data_buffer=b, context_len=context_len) for b in list(online_memory)]
        online_ds = ConcatDataset(online_sets)

        if len(online_ds) == 0:
            itr_record.update({"skipped": True})
            experiment_metrics["phase2"]["history"].append(itr_record)
            continue

        online_ft_loader = DataLoader(online_ds, batch_size=half_batch_size, shuffle=True, drop_last=False)

        backup_state = copy.deepcopy(model.state_dict())
        model.train()

        steps_taken = 0
        total_ft_loss = 0.0
        pbar = tqdm(total=int(args.max_expert_steps), desc=f"Phase2 FT {itr + 1}")

        offline_iter = iter(offline_ft_loader)
        online_iter = iter(online_ft_loader)

        for _ in range(int(args.max_expert_steps)):
            try:
                off_batch = next(offline_iter)
            except StopIteration:
                offline_iter = iter(offline_ft_loader)
                off_batch = next(offline_iter)

            try:
                on_batch = next(online_iter)
            except StopIteration:
                online_iter = iter(online_ft_loader)
                on_batch = next(online_iter)

            states = torch.cat([off_batch[0], on_batch[0]], dim=0).to(device)
            actions = torch.cat([off_batch[1], on_batch[1]], dim=0).to(device)
            rtgs = torch.cat([off_batch[2], on_batch[2]], dim=0).to(device)
            timesteps = torch.cat([off_batch[3], on_batch[3]], dim=0).to(device)
            mask = torch.cat([off_batch[4], on_batch[4]], dim=0).to(device)
            legal_masks = torch.cat([off_batch[5], on_batch[5]], dim=0).to(device)
            preds = model(states, actions, rtgs, timesteps)
            loss = _masked_ce_loss(preds, actions, mask, legal_masks, train_action_mask=bool(args.train_action_mask))

            if loss is None:
                continue

            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            loss.backward()
            optimizer.step()

            total_ft_loss += float(loss.detach().item())
            steps_taken += 1
            pbar.update(1)
            pbar.set_postfix({"loss": float(loss.detach().item())})

        pbar.close()
        avg_ft_loss = total_ft_loss / max(1, steps_taken)

        cand = evaluate_in_simulation(
            game_name=args.game,
            model=model,
            game=game,
            logger=logger,
            num_games=int(args.test_game_count),
            device=device,
            target_return=target_return,
            context_len=context_len,
            oppo_type=args.oppo_type,
            return_details=True,
            custom_opponents=custom_opponents,
        )
        itr_record.update({"finetune_loss": float(avg_ft_loss), **cand})

        if cand["win_rate"] >= (best_win_rate - float(args.accept_tolerance)):
            accepted += 1
            best_win_rate = float(max(best_win_rate, cand["win_rate"]))
            best_avg_score = float(max(best_avg_score, cand["avg_score"]))
            best_illegal = float(cand["illegal_rate"])
            torch.save(model.state_dict(), best_model_path)
            online_memory.append(new_data)
            itr_record["accepted"] = True
            logger.info(f"Accepted candidate model. Best Win Rate: {best_win_rate:.1f}%")
        else:
            model.load_state_dict(backup_state)
            itr_record["accepted"] = False

        experiment_metrics["phase2"]["history"].append(itr_record)

    experiment_metrics["phase2"]["accepted"] = int(accepted)
    experiment_metrics["phase2"]["total_iters"] = int(args.expert_iters)
    experiment_metrics["phase2"]["acceptance_ratio"] = float(accepted) / float(max(1, int(args.expert_iters)))

    experiment_metrics["final_best"] = {
        "win_rate": float(best_win_rate),
        "avg_score": float(best_avg_score),
        "illegal_rate": float(best_illegal),
        "checkpoint": os.path.basename(best_model_path),
    }

    metrics_path = os.path.join(args.output_dir, "experiment_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(experiment_metrics, f, indent=2)

    logger.info(f"Training complete. Best Win Rate: {best_win_rate:.1f}%")
    logger.info(f"Best model saved at: {best_model_path}")


if __name__ == "__main__":
    main()
