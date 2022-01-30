"""
Run the robotron environment using Stable Baselines 3 and PPO
"""

from robotron2084gym.robotron import RobotronEnv
from wandb.integration.sb3 import WandbCallback
import wandb
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from gym.wrappers import GrayScaleObservation, ResizeObservation


def main():
    config = {
        "policy_type": "CnnPolicy",
        "total_timesteps": 5_500_000,
        "lr": 0.0001,
        "env_name": "robotron",
        "start_level": 2,
        "lives": 0,
        "fps": 0,
        "always_move": True,
    }

    resume_path = None  # "models/27rgmhsk/model.zip"

    run = wandb.init(
        project="robotron",
        group="ppo_new_rewards",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    run.log_code('robotron2084gym/robotron/engine/config.yaml')

    env = RobotronEnv(level=config["start_level"],
                      lives=config["lives"],
                      fps=config["fps"],
                      always_move=config["always_move"])

    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, (123, 166))
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')
    env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 2000 == 0, video_length=200)

    env.reset()

    if resume_path:
        model = PPO.load(resume_path, env, verbose=1, tensorboard_log=f"runs/{run.id}", learning_rate=config["lr"])
    else:
        model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}", learning_rate=config["lr"])

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_freq=500_000,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )

    run.finish()


if __name__ == "__main__":
    main()
