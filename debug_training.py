"""
调试脚本，详细打印训练过程
"""
import numpy as np
from envs.train_env import TrainEnv

def debug_env():
    env = TrainEnv(config_path='./config/envs.yaml')

    print("=" * 80)
    print("开始调试训练环境")
    print("=" * 80)

    # Reset环境
    obs, info = env.reset()
    print(f"\n初始观测空间: {obs}")
    print(f"初始我方状态: {env.my_state}")
    print(f"初始敌方状态: {env.enemy_state}")

    my_pos = env.my_state[0:3]
    enemy_pos = env.enemy_state[0:3]
    distance = np.linalg.norm(my_pos - enemy_pos)
    print(f"\n初始距离: {distance:.2f} 单位 ({distance*10:.2f}米)")
    print(f"我方位置: x={my_pos[0]:.2f}, y={my_pos[1]:.2f}, z={my_pos[2]:.2f}")
    print(f"敌方位置: x={enemy_pos[0]:.2f}, y={enemy_pos[1]:.2f}, z={enemy_pos[2]:.2f}")
    print(f"我方速度: u={env.my_state[6]:.2f}, v={env.my_state[7]:.2f}, w={env.my_state[8]:.2f}")

    # 执行几步
    print("\n" + "=" * 80)
    print("执行10步，观察状态变化")
    print("=" * 80)

    for step in range(10):
        # 测试动作：全油门向前飞
        action = np.array([1.0, 0.0, 0.0, 0.0])  # 油门最大，其他为0

        print(f"\n执行Step {step+1}前:")
        print(f"  发送动作: {action}")

        obs, reward, terminated, truncated, info = env.step(action)

        print(f"  terminated={terminated}, truncated={truncated}")

        my_pos = env.my_state[0:3]
        enemy_pos = env.enemy_state[0:3]
        distance = np.linalg.norm(my_pos - enemy_pos)

        print(f"\nStep {step+1}:")
        print(f"  动作: throttle={action[0]:.2f}, pitch={action[1]:.2f}, roll={action[2]:.2f}, yaw={action[3]:.2f}")
        print(f"  我方位置: x={my_pos[0]:.2f}, y={my_pos[1]:.2f}, z={my_pos[2]:.2f}")
        print(f"  我方速度: u={env.my_state[6]:.2f}, v={env.my_state[7]:.2f}, w={env.my_state[8]:.2f}")
        print(f"  距离: {distance:.2f} 单位 ({distance*10:.2f}米)")
        print(f"  奖励: {reward:.3f}")
        print(f"  奖励分量: {info.get('reward_comps', {})}")
        print(f"  我方血量: {env.my_state[12]:.0f}")
        print(f"  敌方血量: {env.enemy_state[12]:.0f}")

        if terminated:
            print(f"  >>> Episode终止！")
            break
        if truncated:
            print(f"  >>> Episode被截断！")
            break

    print("\n" + "=" * 80)
    print("调试完成")
    print("=" * 80)

if __name__ == "__main__":
    debug_env()
