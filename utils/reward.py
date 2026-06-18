# This is the reward calculation function. We provide current state and previous state for you.
def reward_components(prev_my_state, prev_enemy_state, my_state, enemy_state):
    """
    奖励函数设计

    设计思路（参考作业提示）：
    1. 距离奖励：两架飞机距离越近，奖励越高（势能函数形式）
    2. 角度奖励：战机朝向越接近指向靶机的朝向，奖励越高
    3. 血量奖励：击中靶机，靶机血量减少，给予奖励
    4. 回合惩罚：每回合固定惩罚，鼓励快速击落
    5. 存活奖励：保持存活给予小奖励
    6. 速度奖励：鼓励保持合理速度

    输入：
    - prev_my_state: [13] 上一步我方状态
    - prev_enemy_state: [13] 上一步敌方状态
    - my_state: [13] 当前我方状态
    - enemy_state: [13] 当前敌方状态

    状态格式：(x,y,z, roll,pitch,yaw, u,v,w, ω,β,η, 血量)
    """
    import numpy as np

    comps = {}

    # ============ 1. 距离奖励（势能函数形式）============
    # 当前距离
    current_distance = np.linalg.norm(my_state[0:3] - enemy_state[0:3])
    # 上一步距离
    prev_distance = np.linalg.norm(prev_my_state[0:3] - prev_enemy_state[0:3])

    # 距离减少给予奖励（势能函数）
    distance_improvement = prev_distance - current_distance
    comps["distance"] = distance_improvement * 3.0  # 适中提高（v2的2.0→3.0）

    # 距离越近，基础奖励越高
    distance_proximity_reward = np.exp(-current_distance / 50.0) * 8.0  # 适中提高（v2的5.0→8.0）
    comps["proximity"] = distance_proximity_reward

    # 逃离惩罚 - 适度惩罚
    if current_distance > prev_distance:
        comps["escape_penalty"] = -5.0  # 适中惩罚（v2的-3.0→-5.0，v3的-10.0太重）
    else:
        comps["escape_penalty"] = 0.0


    # ============ 2. 角度奖励 ============
    # 计算我方朝向向量
    my_yaw = my_state[5]
    my_pitch = my_state[4]
    my_direction = np.array([
        np.cos(my_pitch) * np.cos(my_yaw),
        np.cos(my_pitch) * np.sin(my_yaw),
        np.sin(my_pitch)
    ])

    # 计算指向目标的向量
    relative_pos = enemy_state[0:3] - my_state[0:3]
    target_direction = relative_pos / (current_distance + 1e-6)

    # 计算夹角的cos值（点积）
    cos_angle = np.dot(my_direction, target_direction)

    # 角度奖励：适度提高，保持连续性
    comps["angle"] = (cos_angle + 1.0) * 2.5  # v2微调（2.0→2.5）

    # 背离目标的惩罚
    if cos_angle < 0:  # 朝向与目标夹角>90度
        comps["wrong_direction"] = cos_angle * 3.0  # 适度惩罚（v2的2.0→3.0）
    else:
        comps["wrong_direction"] = 0.0


    # ============ 3. 血量奖励 ============
    # 敌方血量减少 = 击中敌人 - 这是核心目标！
    # 注意：敌方初始血量是1000，不是100！
    enemy_hp_loss = prev_enemy_state[12] - enemy_state[12]
    if enemy_hp_loss > 0:
        comps["damage"] = enemy_hp_loss * 100.0  # 大幅提高！（10.0→100.0）
    else:
        comps["damage"] = 0.0

    # 敌方被击毁 - 终极目标
    if enemy_state[12] <= 0:
        comps["kill"] = 10000.0  # 巨额奖励！（800→10000）
    else:
        comps["kill"] = 0.0


    # ============ 4. 回合惩罚 ============
    # 适度时间惩罚
    comps["time_penalty"] = -0.15  # 适中（v2的-0.2→-0.15）


    # ============ 5. 存活奖励 ============
    # 我方血量减少 = 被击中（惩罚）
    my_hp_loss = prev_my_state[12] - my_state[12]
    if my_hp_loss > 0:
        comps["self_damage"] = -my_hp_loss * 0.05  # 受伤惩罚
    else:
        comps["self_damage"] = 0.0

    # 我方坠毁
    if my_state[12] <= 0:
        comps["death"] = -50.0  # 自己被击毁，大惩罚
    else:
        comps["death"] = 0.0


    # ============ 6. 速度奖励 ============
    # 鼓励保持合理速度（不要太慢也不要失速）
    my_speed = np.linalg.norm(my_state[6:9])
    if my_speed < 10.0:
        comps["speed"] = -0.5  # 速度太慢惩罚
    elif my_speed > 15.0 and my_speed < 40.0:
        comps["speed"] = 0.2  # 合理速度范围内奖励
    else:
        comps["speed"] = 0.0


    # ============ 7. 高度惩罚 ============
    # 防止飞机飞得太低或太高
    my_altitude = my_state[2]
    if my_altitude < 10.0:  # 低于100m
        comps["altitude"] = -1.0
    elif my_altitude > 200.0:  # 高于2000m
        comps["altitude"] = -0.5
    else:
        comps["altitude"] = 0.0


    # ============ 8. 姿态惩罚 ============
    # 防止过度机动（roll和pitch角度过大）
    my_roll = abs(my_state[3])
    my_pitch_abs = abs(my_state[4])

    if my_roll > np.pi/2:  # 翻滚超过90度
        comps["attitude"] = -0.5
    elif my_pitch_abs > np.pi/3:  # 俯仰超过60度
        comps["attitude"] = -0.3
    else:
        comps["attitude"] = 0.0


    # ============ 9. 连续命中奖励 ============
    # 适度提高连续命中奖励
    prev_enemy_hp_loss = 0.0 if prev_enemy_state[12] == 100.0 else 1.0
    if enemy_hp_loss > 0 and prev_enemy_hp_loss > 0:
        comps["combo"] = 10.0  # 适中（v2的5.0→10.0）
    else:
        comps["combo"] = 0.0


    # ============ 总奖励 ============
    comps["total"] = sum(comps.values())

    return comps


def calculate_reward(prev_my_state, prev_enemy_state, my_state, enemy_state):
    return reward_components(prev_my_state, prev_enemy_state, my_state, enemy_state)["total"]
