import numpy as np


def marshal_action(action):
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
    1. 油门：从[-1,1]映射到[0.3, 1.0]，避免油门过低导致失速
    2. 俯仰/滚转/偏航：保持[-1, 1]范围，可以添加限制
    """

    real_action = np.zeros(4, dtype=np.float64)

    # 1. 油门控制：从[-1,1]映射到[0.3, 1.0]
    # 避免油门为0或过低导致失速
    # throttle = 0.3 + 0.7 * (action[0] + 1) / 2
    # 简化：throttle = 0.3 + 0.35 * (action[0] + 1)
    throttle = 0.3 + 0.35 * (action[0] + 1.0)
    real_action[0] = np.clip(throttle, 0.3, 1.0)

    # 2. 俯仰角控制 (pitch)
    # 直接使用，但可以限制最大值避免过度机动
    pitch = action[1]
    real_action[1] = np.clip(pitch, -1.0, 1.0)

    # 3. 滚转角控制 (roll)
    roll = action[2]
    real_action[2] = np.clip(roll, -1.0, 1.0)

    # 4. 偏航角控制 (yaw)
    yaw = action[3]
    real_action[3] = np.clip(yaw, -1.0, 1.0)

    return real_action
