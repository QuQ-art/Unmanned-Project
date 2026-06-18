"""
修改gym的callback函数，实现训练过程中保存模型和日志
"""
from stable_baselines3.common.callbacks import BaseCallback
import os
import pandas as pd

class TrainLoggerCallback(BaseCallback):
    def __init__(self, save_freq=10, save_path="../logs/single_training/", verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.rewards = []
        self.timesteps = []
        self.episode_reward = 0
        self.episode_idx = 0


    def _on_step(self) -> bool:
        # 解包 Monitor/Venv 包裹
        env = self.training_env.envs[0]
        while hasattr(env, "env"):
            env = env.env

        # 累计当前 step 的 reward
        reward = self.locals["rewards"][0]
        self.episode_reward += reward

        # 监测 episode 是否结束
        done = self.locals["dones"][0]
        if done:
            self.episode_idx += 1
            self.rewards.append(self.episode_reward)
            self.timesteps.append(self.num_timesteps)

            if self.verbose > 0:
                print(f"[Episode {self.episode_idx}] Reward: {self.episode_reward:.2f} | Step: {self.num_timesteps}")

            self.episode_reward = 0

            # 保存日志
            if self.episode_idx % self.save_freq == 0:
                self._save()

        return True

    def _on_training_end(self) -> None:
        self._save()

    def _save(self):
        df = pd.DataFrame({
            "timesteps": self.timesteps,
            "rewards": self.rewards
        })
        csv_path = os.path.join(self.save_path, "training_log.csv")
        df.to_csv(csv_path, index=False)
        if self.verbose > 0:
            print(f"[Logger] Saved training log to {csv_path}")


class RewardComponentsCallback(BaseCallback):
    """
    记录每个step的奖励分量，用于分析训练过程
    """
    def __init__(self, csv_path, verbose=0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.data = []

    def _on_step(self) -> bool:
        # 获取info中的奖励分量
        infos = self.locals.get("infos", [])
        if len(infos) > 0 and "reward_comps" in infos[0]:
            comps = infos[0]["reward_comps"]
            # 添加timestep
            row = {"timesteps": self.num_timesteps}
            row.update({f"r/{k}": v for k, v in comps.items() if k != "total"})
            self.data.append(row)

            # 每1000步保存一次
            if len(self.data) % 1000 == 0:
                self._save()

        return True

    def _on_training_end(self) -> None:
        self._save()

    def _save(self):
        if len(self.data) > 0:
            df = pd.DataFrame(self.data)
            # 追加模式写入CSV
            if os.path.exists(self.csv_path):
                df.to_csv(self.csv_path, mode='a', header=False, index=False)
            else:
                df.to_csv(self.csv_path, index=False)
            self.data = []  # 清空已保存的数据
            if self.verbose > 0:
                print(f"[RewardLogger] Saved to {self.csv_path}")
