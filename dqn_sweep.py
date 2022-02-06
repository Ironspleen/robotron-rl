import argparse

from robotron2084gym.robotron import RobotronEnv
from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import QRDQN
from utils import WandBVideoRecorderWrapper
from wandb.integration.sb3 import WandbCallback
import wandb


def main(batch_size, buffer_size, exploration_final_eps, exploration_fraction, gamma, learning_rate, learning_starts, target_update_interval, total_timesteps):
    config = {
        "env": {
            "config_path": "game_config.yaml",
            "level": 2,
            "lives": 0,
            "always_move": True,
        },
        "model": {
            "batch_size": batch_size, 
            "buffer_size": buffer_size, 
            "exploration_final_eps": exploration_final_eps, 
            "exploration_fraction": exploration_fraction, 
            "gamma": gamma, 
            "learning_rate": learning_rate, 
            "learning_starts": learning_starts, 
            "policy": "CnnPolicy",
            "target_update_interval": target_update_interval, 
        },
        "total_timesteps": total_timesteps,
    }
    device = "cuda:0"

    run = wandb.init(
        project="robotron",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
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
    model = QRDQN(env=env, verbose=1, tensorboard_log=f"runs/{run.id}", device=device, **config['model'])
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
    parser.add_argument("--target_update_interval", type=float, default=None)
    parser.add_argument("--total_timesteps", type=float, default=None)
    args = parser.parse_args()
    main(args.batch_size, args.buffer_size, args.exploration_final_eps, args.exploration_fraction, args.gamma, args.learning_rate, args.learning_starts, args.target_update_interval, args.total_timesteps)
