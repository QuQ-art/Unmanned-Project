"""
统一训练脚本
- 支持从头训练或迁移学习
- 通过命令行参数控制
"""
import os
import sys

from envs.train_env import TrainEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from torch import nn

from utils.callback import BestMeanRewardCallback, RewardComponentsCallback


def linear_schedule(initial_value):
    def schedule(progress_remaining):
        return progress_remaining * initial_value

    return schedule


def main():
    # 命令行参数：python main.py [version_name] [total_steps] [pretrained_model]
    version = sys.argv[1] if len(sys.argv) > 1 else "junior_dynamics_v1"
    total_timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 2_000_000
    pretrained_model = sys.argv[3] if len(sys.argv) > 3 else None

    log_dir = f"./logs_{version}/"
    model_dir = f"./model_{version}/"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    base_env = TrainEnv(config_path="./config/envs.yaml")
    env = Monitor(base_env, filename=os.path.join(log_dir, "monitor.csv"))

    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
        ortho_init=False,
    )

    # 根据是否提供预训练模型来决定训练方式
    if pretrained_model:
        print(f"从 {pretrained_model} 加载预训练模型...")

        # 传入custom_objects解决反序列化问题
        custom_objects = {
            "learning_rate": linear_schedule(2e-4),
            "clip_range": 0.22,
            "lr_schedule": linear_schedule(2e-4),
        }

        model = PPO.load(
            pretrained_model,
            env=env,
            device="cpu",
            custom_objects=custom_objects
        )
    else:
        print("从头开始训练...")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=linear_schedule(3.5e-4),
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.22,
            ent_coef=0.005,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.04,
            policy_kwargs=policy_kwargs,
            device="cpu",
        )

    model.set_logger(logger)

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=model_dir,
        name_prefix="model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    reward_callback = RewardComponentsCallback(
        csv_path=os.path.join(log_dir, "reward_components.csv")
    )
    best_callback = BestMeanRewardCallback(
        save_path=model_dir,
        window_size=50,
        verbose=1,
    )
    callback = CallbackList([checkpoint_callback, reward_callback, best_callback])

    print("=" * 80)
    print(f"版本: {version}")
    print(f"总步数: {total_timesteps:,}")
    print(f"预训练模型: {pretrained_model if pretrained_model else '无（从头训练）'}")
    print(f"日志目录: {log_dir}")
    print(f"模型目录: {model_dir}")
    print("=" * 80)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True,
            reset_num_timesteps=True,
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
    print(f"训练完成，模型已保存到 {final_model_path}.zip")


if __name__ == "__main__":
    main()
