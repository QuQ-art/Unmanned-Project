import numpy as np


OBSERVATION_SIZE = 29


def _body_axes(roll, pitch, yaw):
    """Return forward/right/up unit vectors in world coordinates."""
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


def marshal_observation(my_state, enemy_state):
    """
    将平台状态转成智能体观测。

    重点给智能体“目标在机头前方多少、偏左/右多少、偏上/下多少”的信息，
    这样它更容易学到把敌机放进攻击走廊。
    """
    agent_state = np.zeros(shape=[OBSERVATION_SIZE], dtype=np.float32)

    relative_pos = enemy_state[0:3] - my_state[0:3]
    relative_vel = enemy_state[6:9] - my_state[6:9]
    distance = np.linalg.norm(relative_pos) + 1e-6
    line_of_sight = relative_pos / distance

    roll, pitch, yaw = my_state[3], my_state[4], my_state[5]
    forward, right, up = _body_axes(roll, pitch, yaw)
    rel_body = np.array(
        [
            np.dot(relative_pos, forward),
            np.dot(relative_pos, right),
            np.dot(relative_pos, up),
        ],
        dtype=np.float64,
    )

    closing_speed = -np.dot(relative_vel, line_of_sight)
    forward_alignment = np.dot(forward, line_of_sight)
    lateral_error = abs(rel_body[1])
    vertical_error = abs(rel_body[2])

    agent_state[0:3] = relative_pos / 200.0
    agent_state[3:6] = np.array([rel_body[0] / 200.0, rel_body[1] / 80.0, rel_body[2] / 80.0])
    agent_state[6] = np.tanh(distance / 150.0)
    agent_state[7] = forward_alignment
    agent_state[8] = np.tanh(lateral_error / 20.0)
    agent_state[9] = np.tanh(vertical_error / 15.0)
    agent_state[10] = np.clip(closing_speed / 60.0, -1.0, 1.0)

    agent_state[11:14] = my_state[6:9] / 60.0
    agent_state[14:17] = enemy_state[6:9] / 60.0
    agent_state[17:20] = relative_vel / 60.0

    agent_state[20] = np.clip(roll / np.pi, -1.0, 1.0)
    agent_state[21] = np.clip(pitch / np.pi, -1.0, 1.0)
    agent_state[22] = np.sin(yaw)
    agent_state[23] = np.cos(yaw)
    agent_state[24] = np.tanh(my_state[9])
    agent_state[25] = np.tanh(my_state[10])
    agent_state[26] = np.tanh(my_state[11])

    agent_state[27] = np.clip(my_state[12] / 1000.0, 0.0, 1.0)
    agent_state[28] = np.clip(enemy_state[12] / 1000.0, 0.0, 1.0)

    return np.clip(agent_state, -1.0, 1.0).astype(np.float32)
