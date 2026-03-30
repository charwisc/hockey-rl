"""Self-play PPO training for 2v2 hockey.

Usage (on RunPod RTX 4090):
    pip install -r requirements-train.txt
    python train.py --total-steps 100000000 --n-envs 16

Resume from checkpoint:
    python train.py --total-steps 100000000 --n-envs 16 --resume /workspace/checkpoints/step_50000000.zip

Checkpoints saved to /workspace/checkpoints/ every 30 min wall-time.
TensorBoard logs saved to /workspace/tb_logs/.
Monitor: tensorboard --logdir /workspace/tb_logs/
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList

from training.self_play_callback import SelfPlayPoolCallback
from training.checkpoint_callback import WallTimeCheckpointCallback
from training.tb_callback import TensorBoardCustomCallback

CHECKPOINT_DIR = "/workspace/checkpoints"  # per D-09: hardcoded RunPod volume
TB_LOG_DIR = "/workspace/tb_logs"
POOL_DIR = os.path.join(CHECKPOINT_DIR, "pool")


def make_env(agent_idx: int):
    """Factory returning a callable that creates a HockeyEnv in a subprocess.

    Each env starts with opponent_path=None (random opponent).
    SelfPlayPoolCallback updates opponent_path via set_attr during training.
    """
    def _init():
        from env.hockey_env import HockeyEnv
        return HockeyEnv(agent_idx=agent_idx, time_limit=60.0)
    return _init


def parse_args():
    p = argparse.ArgumentParser(description="Self-play PPO training for 2v2 hockey")
    p.add_argument("--total-steps", type=int, default=100_000_000,
                   help="Total training timesteps (default: 100M)")
    p.add_argument("--n-envs", type=int, default=16,
                   help="Number of parallel environments (default: 16, range: 8-16)")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to SB3 checkpoint .zip to resume from")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(POOL_DIR, exist_ok=True)
    os.makedirs(TB_LOG_DIR, exist_ok=True)

    # Create vectorized environment
    # All envs control agent_idx=0 (single shared policy, egocentric obs)
    env = SubprocVecEnv([make_env(agent_idx=0) for _ in range(args.n_envs)])

    if args.resume:
        # Resume: load VecNormalize stats, then load model (per D-11, D-12)
        vecnorm_path = args.resume.replace(".zip", "_vecnorm.pkl")
        if os.path.exists(vecnorm_path):
            env = VecNormalize.load(vecnorm_path, env)
            env.training = True  # continue updating stats
            env.norm_reward = True
        else:
            print(f"[Warning] VecNormalize file not found: {vecnorm_path}")
            env = VecNormalize(env, norm_obs=True, norm_reward=True)

        model = PPO.load(args.resume, env=env, device="cuda",
                         tensorboard_log=TB_LOG_DIR)
        print(f"[Resume] Loaded checkpoint from {args.resume}")
    else:
        # Fresh start: wrap with VecNormalize (per D-11)
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

        # PPO hyperparameters (Claude's discretion area)
        # n_steps=512 * n_envs=16 = 8192 step rollout buffer
        # batch_size=256 divides 8192 evenly (32 mini-batches)
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=512,
            batch_size=256,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,     # slight entropy bonus for exploration in sparse reward
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=TB_LOG_DIR,
            device="cuda",
        )

    # Compose callbacks (per TRAIN-01, TRAIN-03, TRAIN-04)
    callback = CallbackList([
        SelfPlayPoolCallback(
            pool_dir=POOL_DIR,
            pool_update_freq=500_000,   # per D-02
            max_pool_size=20,           # per D-03
            verbose=1,
        ),
        WallTimeCheckpointCallback(
            checkpoint_dir=CHECKPOINT_DIR,
            interval_minutes=30.0,      # per TRAIN-03
            verbose=1,
        ),
        TensorBoardCustomCallback(verbose=0),
    ])

    print(f"[Train] Starting: total_steps={args.total_steps}, n_envs={args.n_envs}")
    print(f"[Train] Checkpoints -> {CHECKPOINT_DIR}")
    print(f"[Train] TensorBoard -> {TB_LOG_DIR}")
    print(f"[Train] Self-play pool -> {POOL_DIR}")

    model.learn(
        total_timesteps=args.total_steps,
        callback=callback,
        reset_num_timesteps=not bool(args.resume),
    )

    # Final checkpoint save
    final_path = os.path.join(CHECKPOINT_DIR, f"step_{model.num_timesteps}.zip")
    final_vecnorm = os.path.join(CHECKPOINT_DIR, f"step_{model.num_timesteps}_vecnorm.pkl")
    model.save(final_path)
    vec_norm = model.get_vec_normalize_env()
    if vec_norm is not None:
        vec_norm.save(final_vecnorm)
    print(f"[Train] Complete. Final checkpoint: {final_path}")

    env.close()


if __name__ == "__main__":
    main()
