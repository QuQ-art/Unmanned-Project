import numpy as np


def _body_axes(roll, pitch, yaw):
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    forward = np.array([cp * cy, cp * sy, sp], dtype=np.float64)
    return forward


def _attack_geometry(my_state, enemy_state):
    relative_pos = enemy_state[0:3] - my_state[0:3]
    distance = np.linalg.norm(relative_pos) + 1e-6
    line_of_sight = relative_pos / distance

    forward = _body_axes(my_state[3], my_state[4], my_state[5])
    forward_dist = np.dot(relative_pos, forward)
    lateral_vec = relative_pos - forward_dist * forward
    lateral_error = np.linalg.norm(lateral_vec)
    alignment = np.dot(forward, line_of_sight)

    return distance, forward_dist, lateral_error, alignment, line_of_sight


def reward_components(prev_my_state, prev_enemy_state, my_state, enemy_state):
    """
    Simple 战机的奖励函数。

    目标顺序：
    1. 先把敌机留在机头前方；
    2. 再缩小侧向误差，进入攻击走廊；
    3. 命中和击落给主要奖励。
    """
    comps = {}

    prev_distance, _, _, prev_alignment, _ = _attack_geometry(prev_my_state, prev_enemy_state)
    distance, forward_dist, lateral_error, alignment, line_of_sight = _attack_geometry(my_state, enemy_state)

    # 接近目标：限制单步奖励范围，避免高速穿越时数值过大。
    distance_delta = np.clip(prev_distance - distance, -5.0, 5.0)
    if distance > 120.0:
        comps["approach"] = 0.7 * distance_delta
    elif distance > 70.0:
        comps["approach"] = 0.25 * distance_delta
    else:
        comps["approach"] = 0.0

    # 机头对准目标：目标在前方时奖励更高，背向目标时给惩罚。
    comps["alignment"] = 2.0 * alignment
    comps["turning"] = 1.0 * np.clip(alignment - prev_alignment, -0.5, 0.5)

    # 攻击走廊：敌机在机头前方、侧向误差小、距离合适时给密集奖励。
    in_front = 1.0 if forward_dist > 0.0 else 0.0
    range_score = np.exp(-abs(distance - 55.0) / 75.0)
    lateral_score = np.exp(-lateral_error / 14.0)
    corridor_score = in_front * range_score * lateral_score * max(alignment, 0.0)
    comps["attack_corridor"] = 4.0 * corridor_score

    if forward_dist < -5.0:
        comps["overshoot"] = -0.35
    else:
        comps["overshoot"] = 0.0

    relative_vel = enemy_state[6:9] - my_state[6:9]
    closing_speed = -np.dot(relative_vel, line_of_sight)
    comps["closing"] = 0.04 * np.clip(closing_speed, -40.0, 40.0)

    # 命中和击落是核心目标。
    enemy_hp_loss = max(prev_enemy_state[12] - enemy_state[12], 0.0)
    comps["damage"] = 520.0 * enemy_hp_loss
    if prev_enemy_state[12] <= 220.0 and enemy_hp_loss > 0.0:
        comps["finish_damage"] = 600.0 * enemy_hp_loss
    else:
        comps["finish_damage"] = 0.0

    if prev_enemy_state[12] > 0.0 and enemy_state[12] <= 0.0:
        comps["kill"] = 10000.0
    else:
        comps["kill"] = 0.0

    my_hp_loss = max(prev_my_state[12] - my_state[12], 0.0)
    comps["self_damage"] = -8.0 * my_hp_loss

    if prev_my_state[12] > 0.0 and my_state[12] <= 0.0:
        comps["death"] = -1500.0
    else:
        comps["death"] = 0.0

    my_speed = np.linalg.norm(my_state[6:9])
    # junior_dynamics需要更高速度维持升力
    if my_speed < 15.0:  # 提高下限（原8.0）
        comps["speed"] = -2.0  # 加重惩罚，避免失速
    elif 18.0 <= my_speed <= 35.0:  # 调整最佳速度区间
        comps["speed"] = 0.5
    else:
        comps["speed"] = -0.2

    if distance < 120.0 and my_speed > 40.0:  # 放宽上限（原32.0）
        comps["close_fast"] = -0.9
    else:
        comps["close_fast"] = 0.0

    controlled_speed = max(0.0, 1.0 - abs(my_speed - 22.0) / 22.0)  # 目标速度提高到22（原18）
    controlled_closing = max(0.0, 1.0 - abs(closing_speed) / 18.0)
    hold_score = max(alignment, 0.0) * np.exp(-lateral_error / 12.0) * controlled_speed * controlled_closing
    if 25.0 <= distance <= 95.0 and forward_dist > 0.0:
        comps["hold_fire"] = 5.0 * hold_score
    else:
        comps["hold_fire"] = 0.0

    altitude = my_state[2]
    if altitude < 10.0:  # 提高下限（原8.0），防止触地
        comps["altitude"] = -5.0  # 加重惩罚
    elif altitude > 280.0:  # 放宽上限（原260.0）
        comps["altitude"] = -1.0
    else:
        comps["altitude"] = 0.0

    # 每步小惩罚，促使更快击落。
    comps["time"] = -0.05

    comps["total"] = float(sum(comps.values()))
    return comps


def calculate_reward(prev_my_state, prev_enemy_state, my_state, enemy_state):
    return reward_components(prev_my_state, prev_enemy_state, my_state, enemy_state)["total"]
