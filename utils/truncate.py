import numpy as np


def check_truncation(my_state, enemy_state):
    """
    检查是否需要提前终止当前episode

    终止条件：
    1. 我方战机高度过低（坠毁）
    2. 我方战机高度过高（超出作战区域）
    3. 距离过远（超出作战区域）
    4. 我方战机姿态异常（失控）

    输入：
    - my_state: [13] 我方战机状态 (x,y,z, roll,pitch,yaw, u,v,w, ω,β,η, 血量)
    - enemy_state: [13] 敌方靶机状态

    返回：
    - True: 需要终止
    - False: 继续
    """

    debug = False

    # 1. 高度检查
    my_altitude = my_state[2]  # z坐标，单位为10m
    if my_altitude < 5.0:  # 低于50m，即将坠毁
        if debug:
            print(f"Truncate: 高度过低 {my_altitude:.2f}")
        return True
    if my_altitude > 300.0:  # 高于3000m，超出作战区域
        if debug:
            print(f"Truncate: 高度过高 {my_altitude:.2f}")
        return True

    # 2. 距离检查 - 进一步放宽，允许战术机动和回头攻击
    distance = np.linalg.norm(my_state[0:3] - enemy_state[0:3])
    if distance > 500.0:  # 距离超过5000m才终止
        if debug:
            print(f"Truncate: 距离过远 {distance:.2f}")
        return True

    # 3. 姿态检查（防止失控）- 放宽限制允许更大机动
    my_roll = abs(my_state[3])
    my_pitch = abs(my_state[4])

    # 翻滚角或俯仰角过大，可能失控 - 大幅放宽
    if my_roll > np.pi * 1.2:  # 超过216度（之前162度太严格）
        if debug:
            print(f"Truncate: 翻滚角过大 {my_roll:.2f}")
        return True
    if my_pitch > np.pi * 0.8:  # 超过144度（之前108度太严格）
        if debug:
            print(f"Truncate: 俯仰角过大 {my_pitch:.2f}")
        return True

    # 4. 速度检查 - 放宽限制，初始速度为0是正常的
    my_speed = np.linalg.norm(my_state[6:9])
    # 移除速度过低的检查，因为初始时速度可能为0
    # if my_speed < 1.0:  # 速度过低，可能失速
    #     if debug:
    #         print(f"Truncate: 速度过低 {my_speed:.2f}")
    #     return True
    if my_speed > 100.0:  # 速度过高，不现实
        if debug:
            print(f"Truncate: 速度过高 {my_speed:.2f}")
        return True

    # 5. 位置检查（防止飞出边界）
    x, y, z = my_state[0:3]
    if abs(x) > 500.0 or abs(y) > 500.0:  # 水平距离原点超过5000m
        if debug:
            print(f"Truncate: 飞出边界 x={x:.2f}, y={y:.2f}")
        return True

    return False
