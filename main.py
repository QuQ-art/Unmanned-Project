import os

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
    version = "simple_attack_v1"
    log_dir = f"./logs_{version}/"
    model_dir = f"./model_{version}/"
    total_timesteps = 1_500_000

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
        ent_coef=0.003,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.04,
        policy_kwargs=policy_kwargs,
        device="auto",
    )
    model.set_logger(logger)

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=model_dir,
        name_prefix="simple_attack",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    reward_callback = RewardComponentsCallback(
        csv_path=os.path.join(log_dir, "reward_components_finish.csv")
    )
    best_callback = BestMeanRewardCallback(
        save_path=model_dir,
        window_size=50,
        verbose=1,
    )
    callback = CallbackList([checkpoint_callback, reward_callback, best_callback])

    print("=" * 80)
    print("开始训练 Simple 战机攻击固定靶机")
    print("算法: PPO")
    print(f"总步数: {total_timesteps}")
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
