"""
从v7的35万步checkpoint继续训练，使用新的finish_kill奖励机制
"""
import os
from envs.train_env import TrainEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from utils.callback import RewardComponentsCallback

def continue_training():
    # 新版本配置
    version = "v8_from_v7_finish_kill"
    log_dir = f"./logs_{version}/"
    model_dir = f"./model_{version}/"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 从v7的35万步加载
    checkpoint_path = "model_v7_fresh_ultra_relaxed/model_350000_steps.zip"

    print("=" * 80)
    print(f"从v7的35万步checkpoint继续训练")
    print(f"加载: {checkpoint_path}")
    print(f"v7_35万步表现: 敌方HP打到0.4，几乎击杀成功！")
    print(f"新机制: finish_kill奖励 - 残血时鼓励留在附近完成击杀")
    print(f"  • 敌方HP < 5 且距离 < 300m: +50奖励")
    print(f"  • 敌方HP < 5 且距离 > 300m: -20惩罚")
    print(f"  • 击杀奖励提高: 2000 → 5000")
    print(f"  • 初始速度降低: 20 → 10")
    print(f"目标: 继续训练到100万步")
    print("=" * 80)

    # 创建环境
    base_env = TrainEnv(config_path='./config/envs.yaml')
    env = Monitor(base_env)

    # 加载模型
    print("加载模型...")
    model = PPO.load(checkpoint_path, env=env, device="cuda")

    # 配置logger
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
    print(f"当前步数: 350,000")
    print(f"目标步数: 500,000")
    print(f"剩余步数: 150,000")
    print("=" * 80)

    # 继续训练
    model.learn(
        total_timesteps=150000,  # 先训练15万步到50万，看看效果
        progress_bar=True,
        reset_num_timesteps=False,  # 不重置timesteps计数
        log_interval=10,
        callback=callback
    )

    # 保存最终模型
    final_model_path = os.path.join(model_dir, "final_model")
    model.save(final_model_path)
    print(f"训练完成！模型已保存到 {final_model_path}.zip")

if __name__ == "__main__":
    continue_training()
