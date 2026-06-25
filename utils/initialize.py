import numpy as np


def generate_initial_state(config=None):
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

    config = config or {}
    random_initial = bool(config.get("random_initial", True))
    rng = np.random.default_rng()

    my_initial_state = np.zeros(12, dtype=np.float64)

    # 我方战机位置：设置在原点附近
    my_initial_state[0] = 0.0      # x坐标
    my_initial_state[1] = 0.0      # y坐标
    my_initial_state[2] = 60.0     # z坐标（高度600m，junior_dynamics需要更高以防掉高度）

    # 我方战机姿态：水平飞行，朝向+x方向
    my_initial_state[3] = 0.0      # roll (φ)
    my_initial_state[4] = 0.0      # pitch (θ)
    my_initial_state[5] = 0.0      # yaw (ψ) - 朝向+x方向

    # 我方战机线速度：junior_dynamics需要更高初速度以产生足够升力
    my_initial_state[6] = float(config.get("my_initial_speed", 25.0))  # u - 前向速度（提高到25，避免失速）
    my_initial_state[7] = 0.0      # v
    my_initial_state[8] = 0.0      # w

    # 我方战机角速度：初始为0
    my_initial_state[9] = 0.0      # ω
    my_initial_state[10] = 0.0     # β
    my_initial_state[11] = 0.0     # η

    enemy_initial_state = np.zeros(12, dtype=np.float64)

    if random_initial:
        # 初始包会被转成 int32，所以这里使用整数位置，避免小数被截断后分布异常。
        x_min = int(config.get("enemy_x_min", 100))
        x_max = int(config.get("enemy_x_max", 115))  # 100-115单位，即1000-1150m
        y_abs = int(config.get("enemy_y_abs", 5))   # y范围 -50m到+50m，即-5到+5单位
        z_min = int(config.get("enemy_z_min", 58))  # 提高高度到580m
        z_max = int(config.get("enemy_z_max", 61))  # 提高高度到610m

        enemy_initial_state[0] = float(rng.integers(x_min, x_max))
        enemy_initial_state[1] = float(rng.integers(-y_abs, y_abs + 1))
        enemy_initial_state[2] = float(rng.integers(z_min, z_max))
    else:
        enemy_initial_state[0] = 100.0
        enemy_initial_state[1] = 0.0
        enemy_initial_state[2] = 60.0  # 匹配我方高度

    # 敌方靶机姿态：固定朝向（面对我方）
    enemy_initial_state[3] = 0.0      # roll
    enemy_initial_state[4] = 0.0      # pitch
    enemy_initial_state[5] = np.pi    # yaw（朝向-x方向）

    # 敌方靶机线速度：完全静止（符合作业要求：静止靶机）
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
