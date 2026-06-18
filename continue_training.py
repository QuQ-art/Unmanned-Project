"""
从checkpoint继续训练
"""
import os
from envs.train_env import TrainEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from utils.callback import RewardComponentsCallback

def continue_training():
    # 配置路径
    version = "v2_refined"
    log_dir = f"./logs_{version}/"
    model_dir = f"./model_{version}/"

    # 加载已有模型（从150K继续）
    checkpoint_path = f"{model_dir}model_150000_steps.zip"

    print("=" * 80)
    print(f"从checkpoint继续训练: {checkpoint_path}")
    print(f"新规则: 距离限制400单位 + 伤害奖励×100 + 击杀奖励10000")
    print(f"允许战术机动：智能体可以先飞离再回头攻击")
    print("=" * 80)

    # 创建环境
    base_env = TrainEnv(config_path='./config/envs.yaml')
    env = Monitor(base_env)

    # 加载模型
    print("加载模型...")
    model = PPO.load(checkpoint_path, env=env, device="cuda")

    # 配置logger（继续使用同一个日志目录）
    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)

    # 创建回调
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=model_dir,
        name_prefix="model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    reward_callback = RewardComponentsCallback(
        csv_path=os.path.join(log_dir, "reward_components.csv")
    )

    callback = CallbackList([checkpoint_callback, reward_callback])

    print("继续训练...")
    print(f"当前步数: 150,000")
    print(f"目标步数: 1,000,000")
    print(f"剩余步数: 850,000")
    print("=" * 80)

    # 继续训练（从150K到100万）
    model.learn(
        total_timesteps=850000,  # 剩余85万步
        progress_bar=True,
        reset_num_timesteps=False,  # 不重置timesteps计数
        log_interval=1,
        callback=callback
    )

    # 保存最终模型
    final_model_path = os.path.join(model_dir, "final_model")
    model.save(final_model_path)
    print(f"训练完成！模型已保存到 {final_model_path}.zip")

if __name__ == "__main__":
    continue_training()
