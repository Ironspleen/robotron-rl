"""
Run the robotron environment using Stable Baselines 3 
Does not appear to actually train anything meaningful.
"""

from robotron2084gym.robotron import RobotronEnv
from wandb.integration.sb3 import WandbCallback
import wandb
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from gym.wrappers import GrayScaleObservation


def main():
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 5_500_000,
        "lr": 0.0001,
        "env_name": "robowork",
    }

    run = wandb.init(
        project="robotron",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    env = RobotronEnv()
    env = GrayScaleObservation(env, keep_dim=True)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')
    env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 2000 == 0, video_length=200)

    env.reset()

    model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}", learning_rate=config["lr"])
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
