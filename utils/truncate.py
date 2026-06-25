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
    debug = False  # 关闭调试输出

    # 1. 高度检查 - junior_dynamics需要更宽松的高度范围
    my_altitude = my_state[2]  # z坐标，单位为10m
    if my_altitude < 3.0:  # 低于30m，即将坠毁（原5.0，进一步降低）
        if debug:
            print(f"Truncate: 高度过低 {my_altitude:.2f}")
        return True
    if my_altitude > 350.0:  # 高于3500m，超出作战区域（原300.0，放宽）
        if debug:
            print(f"Truncate: 高度过高 {my_altitude:.2f}")
        return True

    # 2. 距离检查 - 敌机会盘旋，需要更大范围
    distance = np.linalg.norm(my_state[0:3] - enemy_state[0:3])
    if distance > 600.0:  # 距离超过6000m才终止（原400.0，进一步放宽）
        if debug:
            print(f"Truncate: 距离过远 {distance:.2f}")
        return True

    # 3. 姿态检查（防止失控）- junior需要更大俯仰角来爬升
    my_roll = abs(my_state[3])
    my_pitch = abs(my_state[4])

    # 只在极端失控情况下才终止
    if my_roll > np.pi * 1.8:  # 超过324度，几乎倒立
        if debug:
            print(f"Truncate: 翻滚角过大 {my_roll:.2f}")
        return True
    if my_pitch > np.pi * 1.5:  # 超过270度，几乎垂直向下（原来限制太严）
        if debug:
            print(f"Truncate: 俯仰角过大 {my_pitch:.2f}")
        return True

    # 4. 速度检查 - 移除失速检查
    # junior_dynamics中，服务器可能不会立即应用初始速度
    # 让智能体自己学习如何避免失速，而不是强制终止
    my_speed = np.linalg.norm(my_state[6:9])
    # if my_speed < 5.0:  # 注释掉失速检查
    #     if debug:
    #         print(f"Truncate: 速度过低可能失速 {my_speed:.2f}")
    #     return True
    # 移除速度上限，允许高速飞行

    # 5. 位置检查（防止飞出边界）
    x, y, z = my_state[0:3]
    if abs(x) > 500.0 or abs(y) > 500.0:  # 水平距离原点超过5000m
        if debug:
            print(f"Truncate: 飞出边界 x={x:.2f}, y={y:.2f}")
        return True

    return False
