"""
测试训练好的模型
"""
import numpy as np
from envs.train_env import TrainEnv
from stable_baselines3 import PPO

def test_model(model_path, num_episodes=5):
    """
    测试模型性能

    Args:
        model_path: 模型路径
        num_episodes: 测试的episode数量
    """
    print("=" * 80)
    print(f"测试模型: {model_path}")
    print("=" * 80)

    # 创建环境
    env = TrainEnv(config_path='./config/envs.yaml')

    # 加载模型
    model = PPO.load(model_path, device="cuda")

    # 测试多个episode
    episode_rewards = []
    episode_lengths = []
    kill_count = 0

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        done = False

        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")

        while not done:
            # 使用模型预测动作
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            step_count += 1
            done = terminated or truncated

            # 每50步输出一次状态
            if step_count % 50 == 0:
                my_state = env.unwrapped.my_state
                enemy_state = env.unwrapped.enemy_state
                distance = np.linalg.norm(my_state[0:3] - enemy_state[0:3])
                print(f"  Step {step_count}: 距离={distance*10:.0f}m, "
                      f"我方HP={my_state[12]:.1f}, 敌方HP={enemy_state[12]:.1f}, "
                      f"奖励={reward:.2f}")

        # Episode结束
        my_state = env.unwrapped.my_state
        enemy_state = env.unwrapped.enemy_state

        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)

        if enemy_state[12] <= 0:
            kill_count += 1
            print(f"\n✓ 击杀成功！")
        else:
            print(f"\n✗ 未击杀 (敌方剩余HP: {enemy_state[12]:.1f})")

        print(f"Episode总奖励: {episode_reward:.2f}")
        print(f"Episode长度: {step_count}步")

    # 统计结果
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    print(f"平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"平均长度: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"击杀成功率: {kill_count}/{num_episodes} ({kill_count/num_episodes*100:.1f}%)")
    print(f"最高奖励: {max(episode_rewards):.2f}")
    print(f"最低奖励: {min(episode_rewards):.2f}")
    print("=" * 80)

    env.close()

if __name__ == "__main__":
    # 测试v7的35万步模型
    test_model("model_v7_fresh_ultra_relaxed/model_350000_steps.zip", num_episodes=5)
