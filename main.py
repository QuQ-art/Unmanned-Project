"""
main函数，用于训练模型并保存
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
from envs.train_env import TrainEnv
from stable_baselines3 import PPO, SAC, TD3
from torch import nn
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from utils.callback import RewardComponentsCallback

def main():
    # 版本号：v2_refined表示v2微调版（平衡的奖励系统）
    version = "v2_refined"
    log_dir = f"./logs_{version}/"
    model_dir = f"./model_{version}/"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    # 创建Checkpoint回调
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=model_dir,
        name_prefix="model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # 创建奖励分量回调
    reward_callback = RewardComponentsCallback(
        csv_path=os.path.join(log_dir, "reward_components.csv")
    )

    # 合并回调
    callback = CallbackList([checkpoint_callback, reward_callback])

    base_env = TrainEnv(config_path='./config/envs.yaml')
    env = Monitor(base_env)

    # 网络结构：可以尝试更深的网络
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),  # 3层网络
        ortho_init=False,
    )

    # 选择算法：PPO / SAC / TD3
    # 这里使用PPO，你也可以尝试SAC或TD3
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,  # 学习率
        n_steps=2048,  # 每次更新收集的步数
        batch_size=512,  # 批次大小
        n_epochs=10,  # 每次更新的epoch数
        gamma=0.99,  # 折扣因子
        gae_lambda=0.95,  # GAE lambda
        clip_range=0.2,  # PPO clip range
        ent_coef=0.01,  # 熵系数，鼓励探索
        vf_coef=0.5,  # 值函数系数
        max_grad_norm=0.5,  # 梯度裁剪
        target_kl=0.02,  # KL散度早停阈值
        policy_kwargs=policy_kwargs,
        device="cuda",  # 使用GPU加速
    )

    # 如果要使用SAC，可以取消下面的注释：
    # model = SAC(
    #     policy="MlpPolicy",
    #     env=env,
    #     verbose=1,
    #     tensorboard_log=log_dir,
    #     learning_rate=3e-4,
    #     buffer_size=1000000,
    #     learning_starts=10000,
    #     batch_size=256,
    #     tau=0.005,
    #     gamma=0.99,
    #     train_freq=1,
    #     gradient_steps=1,
    #     ent_coef='auto',
    #     policy_kwargs=policy_kwargs,
    #     device="cuda",
    # )

    # 如果要使用TD3，可以取消下面的注释：
    # model = TD3(
    #     policy="MlpPolicy",
    #     env=env,
    #     verbose=1,
    #     tensorboard_log=log_dir,
    #     learning_rate=3e-4,
    #     buffer_size=1000000,
    #     learning_starts=10000,
    #     batch_size=256,
    #     tau=0.005,
    #     gamma=0.99,
    #     train_freq=1,
    #     gradient_steps=1,
    #     policy_kwargs=policy_kwargs,
    #     device="cuda",
    # )

    model.set_logger(logger)

    # 开始训练
    print("=" * 80)
    print("开始训练...")
    print(f"算法: PPO")
    print(f"总步数: 1000000")
    print(f"网络结构: {policy_kwargs['net_arch']}")
    print("=" * 80)

    model.learn(
        total_timesteps=1000000,
        progress_bar=True,
        reset_num_timesteps=False,
        log_interval=1,
        callback=callback
    )

    # 保存最终模型
    final_model_path = os.path.join(model_dir, "final_model")
    model.save(final_model_path)
    print(f"训练完成！模型已保存到 {final_model_path}.zip")
    print(f"日志保存在: {log_dir}")
    print(f"检查点保存在: {model_dir}")

if __name__ == "__main__":
    main()