import numpy as np


def generate_initial_state():
    """
    初始化战机和靶机的状态
    返回格式：[我方战机12个参数, 敌方靶机12个参数]

    注意：这里只需要12个参数，不包含血量
    每个战机的12个参数：
    - [0-2]: x, y, z 坐标（单位：1单位=10m）
    - [3-5]: 欧拉角 φ, θ, ψ (roll, pitch, yaw) 单位：弧度
    - [6-8]: 线速度 u, v, w
    - [9-11]: 角速度 ω, β, η

    血量由服务器初始化，不在初始化包中

    要求：初始距离 >= 1000m，即 >= 100单位
    """

    # 我方战机初始状态（12个参数）
    my_initial_state = np.zeros(12, dtype=np.float64)

    # 我方战机位置：设置在原点附近
    my_initial_state[0] = 0.0      # x坐标
    my_initial_state[1] = 0.0      # y坐标
    my_initial_state[2] = 50.0     # z坐标（高度500m）

    # 我方战机姿态：水平飞行，朝向+x方向
    my_initial_state[3] = 0.0      # roll (φ)
    my_initial_state[4] = 0.0      # pitch (θ)
    my_initial_state[5] = 0.0      # yaw (ψ) - 朝向+x方向

    # 我方战机线速度：初始有一定速度
    my_initial_state[6] = 20.0     # u - 前向速度
    my_initial_state[7] = 0.0      # v
    my_initial_state[8] = 0.0      # w

    # 我方战机角速度：初始为0
    my_initial_state[9] = 0.0      # ω
    my_initial_state[10] = 0.0     # β
    my_initial_state[11] = 0.0     # η

    # 敌方靶机初始状态
    enemy_initial_state = np.zeros(12, dtype=np.float64)

    # 敌方靶机位置：固定在前方1000-1200m（符合作业要求）
    # 作业要求：初始距离 >= 1000m (即 >= 100单位)
    enemy_initial_state[0] = 100.0    # x坐标（1000m，最小合规距离）
    enemy_initial_state[1] = 0.0      # y坐标
    enemy_initial_state[2] = 50.0     # z坐标（与我方同高度500m）

    # 敌方靶机姿态：固定朝向（面对我方）
    enemy_initial_state[3] = 0.0      # roll
    enemy_initial_state[4] = 0.0      # pitch
    enemy_initial_state[5] = np.pi    # yaw（朝向-x方向）

    # 敌方靶机线速度：完全静止（符合"静止靶机"要求）
    enemy_initial_state[6] = 0.0      # u
    enemy_initial_state[7] = 0.0      # v
    enemy_initial_state[8] = 0.0      # w

    # 敌方靶机角速度：基本为0
    enemy_initial_state[9] = 0.0      # ω
    enemy_initial_state[10] = 0.0     # β
    enemy_initial_state[11] = 0.0     # η

    # 合并返回
    initial_state = np.append(my_initial_state, enemy_initial_state)
    return initial_state
