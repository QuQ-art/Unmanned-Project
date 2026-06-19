import glob
import os

import torch
import yaml
from envs.train_env import TrainEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from utils.callback import BestMeanRewardCallback, RewardComponentsCallback


def linear_schedule(initial_value):
    def schedule(progress_remaining):
        return progress_remaining * initial_value

    return schedule


def latest_checkpoint(model_dir, config):
    resume_checkpoint = config.get("resume_checkpoint")
    if resume_checkpoint:
        checkpoint_path = os.path.join(model_dir, resume_checkpoint)
        if os.path.exists(checkpoint_path):
            return checkpoint_path
        raise FileNotFoundError(f"指定的 checkpoint 不存在: {checkpoint_path}")

    interrupted = os.path.join(model_dir, "interrupted_model.zip")
    if os.path.exists(interrupted):
        return interrupted

    checkpoints = glob.glob(os.path.join(model_dir, "simple_attack_*_steps.zip"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


def continue_training():
    version = "simple_attack_v1"
    config_path = "./config/envs.yaml"
    log_dir = f"./logs_{version}/"
    model_dir = f"./model_{version}/"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    checkpoint_path = latest_checkpoint(model_dir, config)

    if checkpoint_path is None:
        raise FileNotFoundError(
            f"没有找到 {model_dir} 下的 simple_attack checkpoint，请先运行 main.py。"
        )

    base_env = TrainEnv(config_path=config_path)
    env = Monitor(base_env, filename=os.path.join(log_dir, "monitor_continue.csv"))

    model = PPO.load(checkpoint_path, env=env, device="auto")
    model.learning_rate = linear_schedule(1.2e-4)
    model.lr_schedule = linear_schedule(1.2e-4)
    model.n_epochs = 6
    model.clip_range = lambda _: 0.12
    model.ent_coef = 0.0
    model.target_kl = 0.01
    if hasattr(model.policy, "log_std"):
        with torch.no_grad():
            model.policy.log_std.fill_(-0.6)
    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

    callback = CallbackList(
        [
            CheckpointCallback(
                save_freq=25_000,
                save_path=model_dir,
                name_prefix="simple_attack",
            ),
            RewardComponentsCallback(
                csv_path=os.path.join(log_dir, "reward_components_finish.csv")
            ),
            BestMeanRewardCallback(
                save_path=model_dir,
                window_size=50,
                verbose=1,
            ),
        ]
    )

    print("=" * 80)
    print(f"继续训练: {checkpoint_path}")
    print("=" * 80)

    try:
        model.learn(
            total_timesteps=500_000,
            progress_bar=True,
            reset_num_timesteps=False,
            log_interval=1,
            callback=callback,
        )
    except (ConnectionError, ConnectionResetError, OSError) as exc:
        interrupted_model_path = os.path.join(model_dir, "interrupted_model")
        model.save(interrupted_model_path)
        print(f"平台连接中断，已保存当前模型到 {interrupted_model_path}.zip")
        print(f"错误信息: {exc}")
        return

    final_model_path = os.path.join(model_dir, "final_model")
    model.save(final_model_path)
    print(f"继续训练完成，模型已保存到 {final_model_path}.zip")


if __name__ == "__main__":
    continue_training()
