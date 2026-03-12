import argparse
import sys
import os
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger
from training.dataset import create_dataset
from training.train_pipeline import main as run_training_pipeline

from games.hearts import HeartsGame
from games.whist import WhistGame
from games.spades import SpadesGame


def _build_game(game_name: str):
    if game_name == "hearts":
        return HeartsGame()
    if game_name == "whist":
        return WhistGame()
    if game_name == "spades":
        return SpadesGame()
    raise ValueError(f"Unknown game: {game_name}")


def main():
    parser = argparse.ArgumentParser(description="Unified Card Gaming Framework")

    parser.add_argument("--task", type=str, default="play",
                        choices=["train", "generate", "play", "dqn_train"])

    parser.add_argument("--game", type=str, default="spades",
                        choices=["hearts", "whist", "spades", "all"])

    parser.add_argument("--model", type=str, default="dt", choices=["dt", "dc"])

    parser.add_argument("--agent_type", type=str, default="rule_based",
                        choices=["random", "rule_based", "mixed", "dqn", "dt_mixed", "dqn_mixed"])

    parser.add_argument("--dt_temperature", type=float, default=0.5, help="Temperature for DT during dataset generation (exploration)")
    parser.add_argument("--eval_games", type=int, default=10_000)
    parser.add_argument("--test_game_count", type=int, default=10_000)
    parser.add_argument("--num_games", type=int, default=200_000)
    parser.add_argument("--run_dir", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="", help="Path to existing dataset (.pkl)")
    parser.add_argument("--dc_kernel", type=int, default=3)

    # dqn args
    parser.add_argument("--dqn_ckpt", type=str, default="", help="Path to DQN checkpoint (dqn_best.pt)")
    parser.add_argument("--dqn_hidden", type=int, default=512)
    parser.add_argument("--dqn_eval_every_steps", type=int, default=1_000)
    parser.add_argument("--dqn_eval_games", type=int, default=10_000)

    # training forward args
    parser.add_argument("--play_agent_path", type=str, default="")
    parser.add_argument("--config", type=str, default=None, choices=[None, "baseline", "compact", "large", "custom"])
    parser.add_argument("--pretrain_epochs", type=int, default=None)
    parser.add_argument("--expert_iters", type=int, default=None)
    parser.add_argument("--expert_epochs", type=int, default=None)
    parser.add_argument("--target_quantile", type=float, default=None)

    args = parser.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    logger = setup_logger(args.run_dir)

    logger.info(f"Starting Task: {args.task.upper()} | Game: {args.game.upper()}")
    logger.info(f"dataset_path: {args.dataset_path}")

    if args.task == "generate":
        output_pkl = args.dataset_path
        logger.info(f"Generating {args.agent_type} dataset for {args.game}...")

        dt_model = None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inferred_context_len = 13

        dqn_model = None
        if args.agent_type == "dt_mixed":
            from training.train_pipeline import HYPERPARAM_SCHEMES
            from models.decision_transformer import DecisionTransformer

            logger.info(f"Loading trained DT model from {args.play_agent_path}...")
            state_dict = torch.load(args.play_agent_path, map_location=device)

            config_name = args.config if args.config else "custom"
            scheme = HYPERPARAM_SCHEMES[config_name]

            if "blocks.0.attn.bias" in state_dict:
                bias_shape = state_dict["blocks.0.attn.bias"].shape
                if len(bias_shape) == 4:
                    inferred_context_len = bias_shape[2] // 3

            game = _build_game(args.game)
            obs0 = game.reset()
            inferred_state_dim = int(np.asarray(obs0, dtype=np.float32).reshape(-1).shape[0])

            dt_model = DecisionTransformer(
                state_dim=inferred_state_dim,
                act_dim=52,
                hidden_size=int(scheme["hidden_size"]),
                max_length=inferred_context_len,
                max_ep_len=100,
                n_layer=int(scheme["n_layer"]),
                n_head=int(scheme["n_head"]),
                n_inner=4,
                dropout=float(scheme["dropout"])
            )
            dt_model.load_state_dict(state_dict)
            dt_model.to(device)
            dt_model.eval()
        elif args.agent_type in ["dqn", "dqn_mixed"]:
            from agents.dqn_agent import QNetwork
            logger.info(f"Loading trained DQN model from {args.dqn_ckpt}...")

            game = _build_game(args.game)
            obs0 = game.reset()
            inferred_state_dim = int(np.asarray(obs0, dtype=np.float32).reshape(-1).shape[0])

            dqn_model = QNetwork(inferred_state_dim, 52, args.dqn_hidden).to(device)
            dqn_model.load_state_dict(torch.load(args.dqn_ckpt, map_location=device))
            dqn_model.eval()

        create_dataset(
            num_games=args.num_games,
            output_path=output_pkl,
            agent_type=args.agent_type,
            game_name=args.game,
            dt_model=dt_model,
            dqn_model=dqn_model,
            device=device,
            target_return=1.0,
            context_len=inferred_context_len,
            dt_temperature=args.dt_temperature
        )
        logger.info(f"Dataset generated at: {output_pkl}")

    elif args.task == "train":
        logger.info("Initializing Training Pipeline...")

        if args.dataset_path is None:
            potentials = [
                f"data/{args.game}_{args.agent_type}_{args.num_games}.pkl",
                f"data/{args.game}_random_1000.pkl",
            ]
            for p in potentials:
                if os.path.exists(p):
                    args.dataset_path = p
                    break
            if not args.dataset_path:
                logger.error("No dataset found. Run --task generate first!")
                return

        os.makedirs(args.run_dir, exist_ok=True)

        sys_argv = [
            "train_pipeline.py",
            "--game", args.game,
            "--dataset", args.dataset_path,
            "--output_dir", args.run_dir,
            "--model", args.model,
            "--oppo_type", args.agent_type,
            "--test_game_count", str(args.eval_games)
        ]
        if args.model == "dc":
            sys_argv += ["--dc_kernel", str(args.dc_kernel)]
        if args.config is not None:
            sys_argv += ["--config", args.config]
        if args.pretrain_epochs is not None:
            sys_argv += ["--pretrain_epochs", str(args.pretrain_epochs)]
        if args.expert_iters is not None:
            sys_argv += ["--expert_iters", str(args.expert_iters)]
        if args.expert_epochs is not None:
            sys_argv += ["--expert_epochs", str(args.expert_epochs)]
        if args.target_quantile is not None:
            sys_argv += ["--target_quantile", str(args.target_quantile)]
        if args.agent_type == "dqn":
            if not args.dqn_ckpt or not os.path.exists(args.dqn_ckpt):
                logger.error("DQN checkpoint not provided or not found! Use --dqn_ckpt")
                return
            sys_argv += ["--dqn_ckpt", args.dqn_ckpt]
            sys_argv += ["--dqn_hidden", str(args.dqn_hidden)]

        sys.argv = sys_argv
        run_training_pipeline()

    elif args.task == "play":
        from training.train_pipeline import evaluate_in_simulation, HYPERPARAM_SCHEMES
        from agents.dqn_agent import QNetwork, DQNAgentWrapper, ACT_DIM

        if args.game == "all":
            raise ValueError("play expects a single game, not 'all'.")

        ckpt_path = args.play_agent_path
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt_name = os.path.basename(ckpt_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading checkpoint weights from: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)

        config_name = args.config if args.config else "custom"
        scheme = HYPERPARAM_SCHEMES[config_name]

        inferred_context_len = 20
        if "blocks.0.attn.bias" in state_dict:
            bias_shape = state_dict["blocks.0.attn.bias"].shape
            if len(bias_shape) == 4:
                inferred_context_len = bias_shape[2] // 3

        inferred_state_dim = None
        if "embed_state.weight" in state_dict:
            inferred_state_dim = state_dict["embed_state.weight"].shape[1]

        game = _build_game(args.game)
        if inferred_state_dim is None:
            obs0 = game.reset()
            inferred_state_dim = int(np.asarray(obs0, dtype=np.float32).reshape(-1).shape[0])

        logger.info(
            f"Auto-Inferred Architecture: Model={args.model.upper()}, Config={config_name}, K={inferred_context_len}, StateDim={inferred_state_dim}")

        if args.model == "dt":
            from models.decision_transformer import DecisionTransformer
            model = DecisionTransformer(
                state_dim=inferred_state_dim,
                act_dim=ACT_DIM,
                hidden_size=int(scheme["hidden_size"]),
                max_length=inferred_context_len,
                max_ep_len=100,
                n_layer=int(scheme["n_layer"]),
                n_head=int(scheme["n_head"]),
                n_inner=4,
                dropout=float(scheme["dropout"])
            )
        elif args.model == "dc":
            from models.decision_convformer import DecisionConvformer
            model = DecisionConvformer(
                state_dim=inferred_state_dim,
                act_dim=ACT_DIM,
                hidden_size=int(scheme["hidden_size"]),
                max_length=inferred_context_len,
                max_ep_len=100,
                n_layer=int(scheme["n_layer"]),
                conv_kernel=args.dc_kernel,
                dropout=float(scheme["dropout"])
            )
        else:
            raise ValueError(f"Unknown model type: {args.model}")

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        custom_opps = None
        if args.agent_type == "dqn":
            if not args.dqn_ckpt or not os.path.exists(args.dqn_ckpt):
                raise ValueError(f"DQN path not valid: {args.dqn_ckpt}")

            logger.info(f"Loading DQN opponents from {args.dqn_ckpt}")
            q_model = QNetwork(inferred_state_dim, ACT_DIM, args.dqn_hidden).to(device)
            q_model.load_state_dict(torch.load(args.dqn_ckpt, map_location=device))
            q_model.eval()
            custom_opps = [DQNAgentWrapper(i, q_model, device) for i in range(1, 4)]
            oppo_type = "dqn"
        elif args.agent_type == "random":
            oppo_type = "random"
        else:
            oppo_type = "rule_based"

        target_ret = 1.0

        logger.info(f"[PLAY] ckpt={ckpt_name} | game={args.game} | opponents={oppo_type} | K={inferred_context_len}")

        det = evaluate_in_simulation(
            game_name=args.game,
            model=model,
            game=game,
            logger=logger,
            num_games=int(args.eval_games),
            device=device,
            target_return=target_ret,
            context_len=inferred_context_len,
            oppo_type=oppo_type,
            return_details=True,
            custom_opponents=custom_opps
        )

        logger.info("--- PLAY Results (Player0 = Saved Model) ---")
        logger.info(
            f"Win Rate: {det['win_rate']:.1f}% | Avg Score: {det['avg_score']:.2f} | "
            f"Illegal(top1): {det['illegal_rate']:.2f}% | Games: {det['eval_games']} | Decisions: {det['decisions']}"
        )

    elif args.task == "dqn_train":
        from training.dqn_train import train_dqn, DQNTrainCfg

        if args.game == "all":
            raise ValueError("dqn_train expects a single game, not 'all'.")

        game = _build_game(args.game)
        obs0 = game.reset()
        state_dim = int(np.asarray(obs0, dtype=np.float32).reshape(-1).shape[0])

        oppo_type = "rule"
        if args.agent_type == "random":
            oppo_type = "random"

        logger.info(f"[DQN_TRAIN] game={args.game} state_dim={state_dim} oppo_type={oppo_type} run_dir={args.run_dir}")

        cfg = DQNTrainCfg(
            state_dim=state_dim,
            hidden=int(args.dqn_hidden),
            eval_every_steps=int(args.dqn_eval_every_steps),
            eval_games=int(args.dqn_eval_games),
        )

        ckpt = train_dqn(
            game_name=args.game,
            oppo_type=oppo_type,
            output_dir=args.run_dir,
            cfg=cfg,
            seed=0,
        )

        logger.info(f"[DQN_TRAIN] Saved checkpoint: {ckpt}")
        print(ckpt)

    logger.info("Task completed.")


if __name__ == "__main__":
    main()
