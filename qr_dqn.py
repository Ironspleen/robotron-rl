"""
Run the robotron environment using Stable Baselines 3 
Does not appear to actually train anything meaningful.
"""

from robotron2084gym.robotron import RobotronEnv
from wandb.integration.sb3 import WandbCallback
import wandb
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import QRDQN
from gym.wrappers import GrayScaleObservation, ResizeObservation


def main():
    config = {
        "policy_type": "CnnPolicy",
        "total_timesteps":5_500_000,
        "learning_rate": 0.00025,
        "batch_size": 32,
        "train_freq": 4,
        "target_update_interval": 10_000,
        "learning_starts": 200_000,
        "buffer_size": 500_000,
        "max_grad_norm": 10,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.01,
    }

    run = wandb.init(
        project="robotron",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    env = RobotronEnv(level=2, lives=0, fps=0, always_move=True)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, (168, 168))

    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')
    env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 2000 == 0, video_length=200)

    env.reset()

    model = QRDQN(
            "CnnPolicy",
            env,
            verbose=1,
            learning_rate=0.00025,
            batch_size=32,
            train_freq=4,
            target_update_interval=10_000,
            learning_starts=200_000,
            buffer_size=500_000,
            max_grad_norm=10,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            device="cuda",
            tensorboard_log=f"runs/{run.id}")

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    run.finish()


if __name__ == "__main__":
    main()
