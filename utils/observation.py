import numpy as np


# This is the observation processing function. Remember to modify the declarations in trainenv.py correspondingly.
def marshal_observation(my_state, enemy_state):
    """
    处理观测数据，将原始状态转换为智能体的观测空间

    输入：
    - my_state: [13] 我方战机状态 (x,y,z, roll,pitch,yaw, u,v,w, ω,β,η, 血量)
    - enemy_state: [13] 敌方靶机状态 (x,y,z, roll,pitch,yaw, u,v,w, ω,β,η, 血量)

    输出：
    - agent_state: [15] 归一化后的观测向量

    观测设计思路：
    1. 相对位置信息（归一化）
    2. 相对距离
    3. 我方朝向与目标方向的夹角
    4. 我方姿态角
    5. 我方速度信息
    6. 敌方血量信息
    """

    agent_state = np.zeros(shape=[15], dtype=np.float64)

    # 1. 相对位置向量 (3维)
    relative_pos = enemy_state[0:3] - my_state[0:3]
    distance = np.linalg.norm(relative_pos) + 1e-6  # 避免除零

    # 归一化相对位置 (除以最大可能距离，如2000单位)
    agent_state[0:3] = relative_pos / 200.0  # 索引0,1,2

    # 2. 相对距离（归一化）
    agent_state[3] = np.tanh(distance / 100.0)  # 索引3，使用tanh限制在[-1,1]

    # 3. 我方朝向与目标方向的夹角
    # 计算我方朝向向量（基于yaw和pitch）
    my_yaw = my_state[5]
    my_pitch = my_state[4]
    my_direction = np.array([
        np.cos(my_pitch) * np.cos(my_yaw),
        np.cos(my_pitch) * np.sin(my_yaw),
        np.sin(my_pitch)
    ])

    # 目标方向（归一化的相对位置）
    target_direction = relative_pos / distance

    # 计算夹角的cos值（点积）
    cos_angle = np.dot(my_direction, target_direction)
    agent_state[4] = cos_angle  # 索引4，范围[-1,1]

    # 4. 我方姿态角（归一化）
    agent_state[5] = my_state[3] / np.pi  # roll，索引5
    agent_state[6] = my_state[4] / np.pi  # pitch，索引6
    agent_state[7] = my_state[5] / np.pi  # yaw，索引7

    # 5. 我方线速度（归一化）
    my_speed = np.linalg.norm(my_state[6:9])
    agent_state[8] = np.tanh(my_speed / 50.0)  # 索引8，速度量级
    agent_state[9] = my_state[6] / 50.0   # u，索引9
    agent_state[10] = my_state[7] / 50.0  # v，索引10
    agent_state[11] = my_state[8] / 50.0  # w，索引11

    # 6. 我方角速度（归一化）
    agent_state[12] = np.tanh(my_state[9])   # ω，索引12
    agent_state[13] = np.tanh(my_state[10])  # β，索引13

    # 7. 敌方血量（归一化）
    agent_state[14] = enemy_state[12] / 1000.0  # 索引14，血量/1000

    # 限制观测值在[-1, 1]范围内
    agent_state = np.clip(agent_state, -1.0, 1.0)

    return agent_state
