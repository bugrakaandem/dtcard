import os
import random
from dataclasses import dataclass
from typing import Deque, Tuple
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from games.hearts import HeartsGame
from games.whist import WhistGame
from games.spades import SpadesGame

from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from agents.whist_rule_based_agent import WhistRuleBasedAgent
from agents.spades_rule_based_agent import SpadesRuleBasedAgent

from agents.dqn_agent import QNetwork, ACT_DIM


def _build_game(game_name: str):
    if game_name == "hearts":
        return HeartsGame()
    if game_name == "whist":
        return WhistGame()
    if game_name == "spades":
        return SpadesGame()
    raise ValueError(game_name)


def _rule_agent_for_game(game_name: str, idx: int):
    if game_name == "whist":
        return WhistRuleBasedAgent(idx)
    if game_name == "spades":
        return SpadesRuleBasedAgent(idx)
    return RuleBasedAgent(idx)


@dataclass
class DQNTrainCfg:
    state_dim: int
    hidden: int = 512
    gamma: float = 0.95
    lr: float = 2.5e-4
    batch_size: int = 256
    buffer_size: int = 200_000
    learning_starts: int = 5_000
    train_every: int = 4
    target_update_every: int = 2_000
    max_steps: int = 500_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 150_000
    use_double_dqn: bool = True
    grad_clip_norm: float = 10.0
    eval_every_steps: int = 25_000
    eval_games: int = 2_000
    save_best: bool = True


def _eps(cfg: DQNTrainCfg, step: int) -> float:
    if step >= cfg.eps_decay_steps:
        return cfg.eps_end
    frac = step / float(cfg.eps_decay_steps)
    return cfg.eps_start - frac * (cfg.eps_start - cfg.eps_end)


def _legal_mask(legal_moves) -> np.ndarray:
    m = np.zeros((ACT_DIM,), dtype=np.bool_)
    for a in legal_moves:
        a = int(a)
        if 0 <= a < ACT_DIM:
            m[a] = True
    return m


def _masked_argmax(q: torch.Tensor, legal_mask_t: torch.Tensor) -> int:
    masked = q.masked_fill(~legal_mask_t, float("-inf"))
    return int(torch.argmax(masked).item())


def _masked_max(q: torch.Tensor, legal_mask_t: torch.Tensor) -> torch.Tensor:
    masked = q.masked_fill(~legal_mask_t, float("-inf"))
    return masked.max(dim=1).values


def _advance_until_player0_turn(
    game,
    obs,
    opponents,
    device: str,
    q_model: nn.Module,
    eps: float,
):
    assert int(game.current_player_idx) == 0

    s0 = np.asarray(obs, dtype=np.float32)

    legal0 = game.get_legal_moves(0)
    legal0_mask = _legal_mask(legal0)

    if np.random.rand() < eps:
        a0 = int(np.random.choice(legal0))
    else:
        with torch.no_grad():
            x = torch.tensor(s0, device=device).view(1, -1)
            qv = q_model(x)[0]
            a0 = _masked_argmax(qv, torch.tensor(legal0_mask, device=device))

    obs, rewards, done, _ = game.step(a0)
    env_steps_used = 1
    acc_r0 = float(rewards[0])

    while (not done) and int(game.current_player_idx) != 0:
        cur = int(game.current_player_idx)
        legal = game.get_legal_moves(cur)
        a = opponents[cur - 1].select_action(obs, legal)
        obs, rewards, done, _ = game.step(a)
        env_steps_used += 1
        acc_r0 += float(rewards[0])

    s1 = np.asarray(obs, dtype=np.float32)
    if done:
        legal1_mask = np.zeros((ACT_DIM,), dtype=np.bool_)
    else:
        assert int(game.current_player_idx) == 0
        legal1_mask = _legal_mask(game.get_legal_moves(0))

    return s0, int(a0), float(acc_r0), s1, bool(done), legal1_mask, env_steps_used


def _evaluate_dqn(
    game_name: str,
    q_model: nn.Module,
    device: str,
    oppo_type: str,
    num_games: int,
):
    game = _build_game(game_name)
    if oppo_type == "mixed":
        opponents = [
            _rule_agent_for_game(game_name, 1),
            RandomAgent(2),
            _rule_agent_for_game(game_name, 3)
        ]
    else:
        opponents = [
            (_rule_agent_for_game(game_name, i) if oppo_type == "rule" else RandomAgent(i))
            for i in range(1, 4)
        ]

    wins0 = 0
    total0 = 0.0

    q_model.eval()
    with torch.no_grad():
        for _ in range(int(num_games)):
            obs = game.reset()
            done = False
            game_rewards = [0.0, 0.0, 0.0, 0.0]

            while not done:
                cur = int(game.current_player_idx)
                legal = game.get_legal_moves(cur)

                if cur == 0:
                    s = torch.tensor(np.asarray(obs, dtype=np.float32), device=device).view(1, -1)
                    qv = q_model(s)[0]
                    lm = torch.tensor(_legal_mask(legal), device=device)
                    action = _masked_argmax(qv, lm)
                else:
                    action = opponents[cur - 1].select_action(obs, legal)

                obs, rewards, done, _ = game.step(action)
                for i in range(4):
                    game_rewards[i] += float(rewards[i])

            if game_rewards[0] == max(game_rewards):
                wins0 += 1
            total0 += float(game_rewards[0])

    q_model.train()
    win_rate = 100.0 * wins0 / max(1, int(num_games))
    avg_score = total0 / max(1, int(num_games))
    return float(win_rate), float(avg_score)


def train_dqn(
    game_name: str,
    oppo_type: str,
    output_dir: str,
    cfg: DQNTrainCfg,
    seed: int = 0,
):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    q = QNetwork(cfg.state_dim, ACT_DIM, cfg.hidden).to(device)
    tq = QNetwork(cfg.state_dim, ACT_DIM, cfg.hidden).to(device)
    tq.load_state_dict(q.state_dict())
    opt = torch.optim.AdamW(q.parameters(), lr=cfg.lr)

    buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]] = deque(maxlen=cfg.buffer_size)

    game = _build_game(game_name)
    opponents = [
        (_rule_agent_for_game(game_name, i) if oppo_type == "rule" else RandomAgent(i))
        for i in range(1, 4)
    ]

    obs = game.reset()
    step = 0

    best_wr = float("-inf")
    best_path = os.path.join(output_dir, "dqn_best.pt")
    last_path = os.path.join(output_dir, "dqn_last.pt")

    while int(game.current_player_idx) != 0:
        cur = int(game.current_player_idx)
        legal = game.get_legal_moves(cur)
        a = opponents[cur - 1].select_action(obs, legal)
        obs, _, done, _ = game.step(a)
        step += 1
        if done:
            obs = game.reset()

    while step < cfg.max_steps:
        assert int(game.current_player_idx) == 0
        eps = _eps(cfg, step)

        s0, a0, r0, s1, done, legal1_mask, env_steps_used = _advance_until_player0_turn(
            game=game,
            obs=obs,
            opponents=opponents,
            device=device,
            q_model=q,
            eps=eps,
        )

        buf.append((s0, a0, r0, s1, done, legal1_mask))

        obs = s1
        step += env_steps_used

        if done:
            obs = game.reset()
            while int(game.current_player_idx) != 0:
                cur = int(game.current_player_idx)
                legal = game.get_legal_moves(cur)
                a = opponents[cur - 1].select_action(obs, legal)
                obs, _, d2, _ = game.step(a)
                step += 1
                if d2:
                    obs = game.reset()

        if step >= cfg.learning_starts and len(buf) >= cfg.batch_size and (step % cfg.train_every == 0):
            batch = random.sample(buf, cfg.batch_size)
            s, a, r, ns, d, nlegal = zip(*batch)

            s = torch.tensor(np.stack(s), device=device, dtype=torch.float32)
            a = torch.tensor(a, device=device, dtype=torch.long)
            r = torch.tensor(r, device=device, dtype=torch.float32)
            ns = torch.tensor(np.stack(ns), device=device, dtype=torch.float32)
            d = torch.tensor(d, device=device, dtype=torch.float32)

            nlegal = torch.tensor(np.stack(nlegal), device=device, dtype=torch.bool)  # (B, ACT_DIM)

            qsa = q(s).gather(1, a.view(-1, 1)).squeeze(1)

            with torch.no_grad():
                if cfg.use_double_dqn:
                    q_next_online = q(ns)  # (B, ACT_DIM)
                    q_next_online_masked = q_next_online.masked_fill(~nlegal, float("-inf"))
                    a_star = torch.argmax(q_next_online_masked, dim=1)  # (B,)

                    q_next_target = tq(ns).gather(1, a_star.view(-1, 1)).squeeze(1)
                else:
                    q_next_target_all = tq(ns)  # (B, ACT_DIM)
                    q_next_target = _masked_max(q_next_target_all, nlegal)

                target = r + cfg.gamma * (1.0 - d) * q_next_target

            loss = F.smooth_l1_loss(qsa, target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), cfg.grad_clip_norm)
            opt.step()

        if step % cfg.target_update_every == 0:
            tq.load_state_dict(q.state_dict())

        if cfg.eval_every_steps > 0 and (step % int(cfg.eval_every_steps) == 0) and step > 0:
            wr, sc = _evaluate_dqn(game_name, q, device, oppo_type, int(cfg.eval_games))
            print(f"[DQN][EVAL] step={step} wr={wr:.1f}% avg_score={sc:.2f} oppo={oppo_type} eps={eps:.3f}")

            torch.save(q.state_dict(), last_path)

            if cfg.save_best and wr > best_wr:
                best_wr = wr
                torch.save(q.state_dict(), best_path)
                print(f"[DQN][BEST] step={step} best_wr={best_wr:.1f}% -> {best_path}")

    if not os.path.exists(best_path):
        torch.save(q.state_dict(), best_path)
    torch.save(q.state_dict(), last_path)
    return best_path
