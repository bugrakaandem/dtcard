import os
import pickle
import sys
import numpy as np
import argparse
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from games.hearts import HeartsGame
from games.whist import WhistGame
from games.spades import SpadesGame

from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from agents.whist_rule_based_agent import WhistRuleBasedAgent
from agents.spades_rule_based_agent import SpadesRuleBasedAgent


def _legal_mask_from_moves(legal_moves, act_dim: int = 52) -> np.ndarray:
    mask = np.zeros((act_dim,), dtype=np.uint8)
    for m in legal_moves:
        mid = int(m) if isinstance(m, (int, np.integer)) else int(getattr(m, "id", m))
        if 0 <= mid < act_dim:
            mask[mid] = 1
    return mask


def create_dataset(
        num_games: int = 1000,
        output_path: str = "data/hearts_random_1k.pkl",
        agent_type: str = "random",
        game_name: str = "hearts",
        store_legal_masks: bool = True,
        dt_model=None,
        dqn_model=None,
        device: str = "cpu",
        target_return: float = 1.0,
        context_len: int = 13,
        dt_temperature: float = 0.5
):
    print(f"Generating {num_games} games for {game_name.upper()} using {agent_type.upper()} agents...")

    all_obs = []
    all_actions = []
    all_rewards = []
    all_done_idxs = []
    all_legal_masks = []
    total_wins = 0

    if game_name == "hearts":
        game = HeartsGame()
    elif game_name == "whist":
        game = WhistGame()
    elif game_name == "spades":
        game = SpadesGame()
    else:
        raise ValueError(f"Unknown game: {game_name}")

    pbar = tqdm(range(num_games), desc="Simulating Games")
    prev_len = 0

    for _ in pbar:
        obs = game.reset()

        p0_is_dt = False
        if agent_type == "random":
            agents = [RandomAgent(i) for i in range(4)]
        elif agent_type == "rule_based":
            if game_name == "whist":
                agents = [WhistRuleBasedAgent(i) for i in range(4)]
            elif game_name == "spades":
                agents = [SpadesRuleBasedAgent(i) for i in range(4)]
            else:
                agents = [RuleBasedAgent(i) for i in range(4)]
        elif agent_type == "mixed":
            if np.random.rand() < 0.8:
                if game_name == "whist":
                    agents = [WhistRuleBasedAgent(i) for i in range(4)]
                elif game_name == "spades":
                    agents = [SpadesRuleBasedAgent(i) for i in range(4)]
                else:
                    agents = [RuleBasedAgent(i) for i in range(4)]
            else:
                agents = [RandomAgent(i) for i in range(4)]
        elif agent_type == "dt_mixed":
            if dt_model is None:
                raise ValueError("dt_model must be provided for dt_mixed")

            from train_pipeline import DecisionTransformerAgent
            dt_agent = DecisionTransformerAgent(dt_model, context_len, device, target_return)

            rule_agents = [WhistRuleBasedAgent(i) if game_name == "whist" else SpadesRuleBasedAgent(
                i) if game_name == "spades" else RuleBasedAgent(i) for i in range(4)]
            rand_agents = [RandomAgent(i) for i in range(4)]
            agents = [None] * 4

            rand_val = np.random.rand()
            if rand_val < 0.4:
                agents[0] = dt_agent
                p0_is_dt = True
            elif rand_val < 0.8:
                agents[0] = rule_agents[0]
            else:
                agents[0] = rand_agents[0]

            for i in range(1, 4):
                agents[i] = rule_agents[i] if np.random.rand() < 0.7 else rand_agents[i]
        elif agent_type == "dqn":
            if dqn_model is None:
                raise ValueError("dqn_model must be provided for dqn")
            from agents.dqn_agent import DQNAgentWrapper
            agents = [DQNAgentWrapper(i, dqn_model, device) for i in range(4)]

        elif agent_type == "dqn_mixed":
            if dqn_model is None:
                raise ValueError("dqn_model must be provided for dqn_mixed")
            from agents.dqn_agent import DQNAgentWrapper
            dqn_agent = DQNAgentWrapper(0, dqn_model, device)

            rule_agents = [WhistRuleBasedAgent(i) if game_name == "whist" else SpadesRuleBasedAgent(
                i) if game_name == "spades" else RuleBasedAgent(i) for i in range(4)]
            rand_agents = [RandomAgent(i) for i in range(4)]
            agents = [None] * 4

            rand_val = np.random.rand()
            if rand_val < 0.4:
                agents[0] = dqn_agent
            elif rand_val < 0.8:
                agents[0] = rule_agents[0]
            else:
                agents[0] = rand_agents[0]

            for i in range(1, 4):
                agents[i] = rule_agents[i] if np.random.rand() < 0.7 else rand_agents[i]
        else:
            raise ValueError(f"Unknown agent_type: {agent_type}")

        traj_obs = []
        traj_actions = []
        traj_rewards = []
        traj_legals = []

        done = False
        info = {}

        while not done:
            current_player = int(game.current_player_idx)
            agent = agents[current_player]
            legal_moves = game.get_legal_moves(current_player)

            if current_player == 0 and p0_is_dt:
                action = agent.select_action(obs, legal_moves, temperature=dt_temperature)
            else:
                action = agent.select_action(obs, legal_moves)

            next_obs, rewards, done, info = game.step(action)

            if current_player == 0:
                traj_obs.append(obs)
                traj_actions.append(
                    int(action) if isinstance(action, (int, np.integer)) else int(getattr(action, "id", action)))
                traj_rewards.append(0.0)
                if store_legal_masks:
                    traj_legals.append(_legal_mask_from_moves(legal_moves))
            if rewards is not None and len(rewards) > 0 and float(rewards[0]) != 0.0:
                if len(traj_rewards) > 0:
                    traj_rewards[-1] += float(rewards[0])
                if p0_is_dt:
                    agents[0].update_rtg(float(rewards[0]))

            obs = next_obs

        current_len = len(traj_actions)
        all_obs.extend(traj_obs)
        all_actions.extend(traj_actions)
        all_rewards.extend(traj_rewards)
        if store_legal_masks:
            all_legal_masks.extend(traj_legals)

        all_done_idxs.append(prev_len + current_len)
        prev_len += current_len

        win = False
        if hasattr(game, "tricks_won") and isinstance(game.tricks_won, (list, tuple)) and len(game.tricks_won) == 4:
            win = (game.tricks_won[0] == max(game.tricks_won))
        elif hasattr(game, "total_scores") and game.total_scores:
            win = (game.total_scores[0] == min(game.total_scores))
        elif isinstance(info, dict) and "winner" in info:
            win = (int(info["winner"]) == 0)

        if win:
            total_wins += 1

    rewards_arr = np.array(all_rewards, dtype=np.float32)
    rtg = np.zeros_like(rewards_arr, dtype=np.float32)

    start = 0
    for end in all_done_idxs:
        end = int(end)
        if end <= start:
            continue
        running = 0.0
        for t in reversed(range(start, end)):
            running += float(rewards_arr[t])
            rtg[t] = running
        start = end

    dataset = {
        "observations": np.array(all_obs, dtype=np.float32),
        "actions": np.array(all_actions, dtype=np.int64),
        "rewards": rewards_arr,
        "rtg": rtg,
        "done_idxs": np.array(all_done_idxs, dtype=np.int64),
        "game_type": game_name,
        "agent_type": agent_type,
    }
    if store_legal_masks:
        dataset["legal_masks"] = np.array(all_legal_masks, dtype=np.uint8)

    episode_returns = []
    start = 0
    for end in dataset["done_idxs"]:
        end = int(end)
        if start < len(rtg):
            episode_returns.append(float(rtg[start]))
        start = end

    if len(episode_returns) > 0:
        dataset["return_mean"] = float(np.mean(episode_returns))
        dataset["return_max"] = float(np.max(episode_returns))
        dataset["return_p50"] = float(np.quantile(episode_returns, 0.5))
        dataset["return_p75"] = float(np.quantile(episode_returns, 0.75))
        dataset["return_p90"] = float(np.quantile(episode_returns, 0.9))
    else:
        dataset["return_mean"] = 0.0
        dataset["return_max"] = 0.0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"\nDataset saved to {output_path}")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="hearts", choices=["hearts", "whist", "spades"])
    parser.add_argument("--num_games", type=int, default=1000)
    parser.add_argument("--agent", type=str, default="random", choices=["random", "rule_based", "mixed", "dt_mixed"])
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--no_legal_masks", action="store_true",
                        help="Do NOT store Player0 legal masks in the dataset.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.out is None or str(args.out).strip() == "":
        suffix = "random" if args.agent == "random" else args.agent
        args.out = f"data/{args.game}_{suffix}_{args.num_games}.pkl"

    create_dataset(
        num_games=args.num_games,
        output_path=args.out,
        agent_type=args.agent,
        game_name=args.game,
        store_legal_masks=(not args.no_legal_masks),
    )
