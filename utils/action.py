import numpy as np


def _body_axes(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    forward = np.array([cp * cy, cp * sy, sp], dtype=np.float64)
    right = np.array(
        [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp],
        dtype=np.float64,
    )
    up = np.array(
        [-cr * sp * cy - sr * sy, -cr * sp * sy + sr * cy, cr * cp],
        dtype=np.float64,
    )
    return forward, right, up


def _target_geometry(my_state, enemy_state):
    relative_pos = enemy_state[0:3] - my_state[0:3]
    distance = np.linalg.norm(relative_pos) + 1e-6
    forward, right, up = _body_axes(my_state[3], my_state[4], my_state[5])
    rel_body = np.array(
        [
            np.dot(relative_pos, forward),
            np.dot(relative_pos, right),
            np.dot(relative_pos, up),
        ],
        dtype=np.float64,
    )
    alignment = rel_body[0] / distance
    return distance, rel_body, alignment


def marshal_action(action, my_state=None, enemy_state=None, config=None):
    """
    处理智能体输出的动作，转换为平台可接受的控制指令

    输入：
    - action: [4] 智能体输出的归一化动作，范围[-1, 1]
      - action[0]: 油门控制
      - action[1]: 俯仰角控制 (pitch)
      - action[2]: 滚转角控制 (roll)
      - action[3]: 偏航角控制 (yaw)

    输出：
    - real_action: [4] 实际控制指令
      - [0]: throttle, 范围[0, 1]
      - [1]: pitch, 范围[-1, 1]
      - [2]: roll, 范围[-1, 1]
      - [3]: yaw, 范围[-1, 1]

    动作处理策略：
    1. 油门：远距离保留接近速度，近距离自动限油门以延长命中时间
    2. 俯仰/滚转/偏航：保持[-1, 1]范围，可以添加限制
    """

    real_action = np.zeros(4, dtype=np.float64)

    # 根据配置文件的 dynamics_type 判断
    cfg = config or {}
    dynamics_type = cfg.get("dynamics_type", "simple")  # 默认simple
    use_high_throttle = (dynamics_type == "junior" or dynamics_type == "jsbsim")

    if use_high_throttle:
        # junior/jsbsim_dynamics: 需要更高油门对抗重力
        throttle = 0.35 + 0.35 * (action[0] + 1.0)  # 范围[0.35, 1.05]
        throttle_low = 0.25
        throttle_high = 0.95
    else:
        # simple_dynamics: 原始较低油门
        throttle = 0.15 + 0.325 * (action[0] + 1.0)  # 范围[0.15, 0.80]
        throttle_low = 0.15
        throttle_high = 0.80
    distance = None
    rel_body = None
    alignment = None
    if my_state is not None and enemy_state is not None:
        distance, rel_body, alignment = _target_geometry(my_state, enemy_state)
        speed = np.linalg.norm(my_state[6:9])

        # 根据距离调整油门范围
        if use_high_throttle:
            # junior_dynamics: 更高的油门范围
            if distance < 55.0:
                throttle_low, throttle_high = 0.10, 0.25
            elif distance < 70.0:
                throttle_low, throttle_high = 0.15, 0.35
            elif distance < 95.0:
                throttle_low, throttle_high = 0.20, 0.50
            elif distance < 130.0:
                throttle_low, throttle_high = 0.30, 0.75
        else:
            # simple_dynamics: 原始油门范围
            if distance < 55.0:
                throttle_low, throttle_high = 0.00, 0.10
            elif distance < 70.0:
                throttle_low, throttle_high = 0.00, 0.15
            elif distance < 95.0:
                throttle_low, throttle_high = 0.03, 0.25
            elif distance < 130.0:
                throttle_low, throttle_high = 0.15, 0.60

        if distance < 130.0 and speed > 35.0:
            throttle_high = min(throttle_high, 0.25 if use_high_throttle else 0.12)

    real_action[0] = np.clip(throttle, throttle_low, throttle_high)

    pitch = action[1]
    roll = action[2]
    yaw = action[3]

    cfg = config or {}
    if (
        cfg.get("aim_assist_enabled", True)
        and distance is not None
        and distance < 110.0
        and alignment > 0.25
    ):
        pitch_sign = float(cfg.get("aim_assist_pitch_sign", 1.0))
        yaw_sign = float(cfg.get("aim_assist_yaw_sign", 1.0))
        assist_gain = float(cfg.get("aim_assist_gain", 0.55))

        aim_pitch = np.clip(pitch_sign * rel_body[2] / 16.0, -0.8, 0.8)
        aim_yaw = np.clip(yaw_sign * rel_body[1] / 16.0, -0.8, 0.8)

        blend = assist_gain
        if distance > 80.0:
            blend *= 0.55
        elif distance < 55.0:
            blend = min(0.75, blend + 0.15)

        pitch = (1.0 - blend) * pitch + blend * aim_pitch
        yaw = (1.0 - blend) * yaw + blend * aim_yaw
        roll *= 0.25

    real_action[1] = np.clip(pitch, -1.0, 1.0)
    real_action[2] = np.clip(roll, -1.0, 1.0)
    real_action[3] = np.clip(yaw, -1.0, 1.0)

    return real_action
