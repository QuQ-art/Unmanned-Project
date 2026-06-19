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

    # 调试：打印检查过程
    debug = True  # 改为True可以看到详细检查过程

    # 1. 高度检查 - 放宽下限，允许低空攻击
    my_altitude = my_state[2]  # z坐标，单位为10m
    if my_altitude < 2.0:  # 低于20m才终止（之前50m太严格）
        if debug:
            print(f"Truncate: 高度过低 {my_altitude:.2f}")
        return True
    if my_altitude > 300.0:  # 高于3000m，超出作战区域
        if debug:
            print(f"Truncate: 高度过高 {my_altitude:.2f}")
        return True

    # 2. 距离检查 - 进一步放宽，允许战术机动和回头攻击
    distance = np.linalg.norm(my_state[0:3] - enemy_state[0:3])
    if distance > 400.0:  # 距离超过4000m才终止（300→400）
        if debug:
            print(f"Truncate: 距离过远 {distance:.2f}")
        return True

    # 3. 姿态检查（防止失控）- 进一步大幅放宽，几乎不限制
    my_roll = abs(my_state[3])
    my_pitch = abs(my_state[4])

    # 只在极端失控情况下才终止（接近倒飞）
    if my_roll > np.pi * 1.8:  # 超过324度，几乎倒立
        if debug:
            print(f"Truncate: 翻滚角过大 {my_roll:.2f}")
        return True
    if my_pitch > np.pi * 1.5:  # 超过270度，几乎垂直向下
        if debug:
            print(f"Truncate: 俯仰角过大 {my_pitch:.2f}")
        return True

    # 4. 速度检查 - 移除速度限制
    # 作业没有要求速度限制，战斗机可以高速飞行
    # my_speed = np.linalg.norm(my_state[6:9])
    # if my_speed > 100.0:
    #     if debug:
    #         print(f"Truncate: 速度过高 {my_speed:.2f}")
    #     return True

    # 5. 位置检查（防止飞出边界）
    x, y, z = my_state[0:3]
    if abs(x) > 500.0 or abs(y) > 500.0:  # 水平距离原点超过5000m
        if debug:
            print(f"Truncate: 飞出边界 x={x:.2f}, y={y:.2f}")
        return True

    return False
