import argparse

from robotron2084gym.robotron import RobotronEnv
from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import QRDQN
from utils import WandBVideoRecorderWrapper
from wandb.integration.sb3 import WandbCallback
import wandb


def main(args):
    config = {
        "env": {
            "always_move": True,
            "config_path": "game_config.yaml",
            "level": 2,
            "lives": 0,
        },
        "model": {
            "batch_size": args.batch_size, 
            "buffer_size": args.buffer_size, 
            "exploration_final_eps": args.exploration_final_eps, 
            "exploration_fraction": args.exploration_fraction, 
            "gamma": args.gamma, 
            "learning_rate": args.learning_rate, 
            "learning_starts": args.learning_starts, 
            "max_grad_norm": args.max_grad_norm,
            "policy": "CnnPolicy",
            "target_update_interval": args.target_update_interval,
        },
        "total_timesteps": args.total_timesteps,
    }

    run = wandb.init(
        project="robotron",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )
    run.log_code()
    run.log_code(name="game_config", include_fn=lambda x: x.endswith(".yaml"))

    env = RobotronEnv(**config['env'])
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, (123, 166))
    env = Monitor(env, info_keywords=('score', 'level'))
    env = DummyVecEnv([lambda: env])
    env = WandBVideoRecorderWrapper(env, record_video_trigger=lambda x: x % 2000 == 0, video_length=200)
    env = VecFrameStack(env, 4, channels_order='first')

    env.reset()
    model = QRDQN(env=env, verbose=1, tensorboard_log=f"runs/{run.id}", **config['model'])
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_freq=500_000,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )

    run.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--buffer_size", type=int, default=None)
    parser.add_argument("--exploration_final_eps", type=float, default=None)
    parser.add_argument("--exploration_fraction", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--learning_starts", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--target_update_interval", type=float, default=None)
    parser.add_argument("--total_timesteps", type=float, default=None)
    args = parser.parse_args()
    main(args)
