# Train_Single
This is the code template for students training agent using python. Beta test.

transformer_option_agent.py:
- The agent is trained using the transformer option agent.
- Support SAC, PPO, TD3 algorithms.

algs.yaml:
- configure algorithm parameters

envs.yaml:
- configure environment parameters

custom_model.py:
- define custom model

callback.py:
- define callback functions to save the reward
- More information can be found on the use of stable baselines3

visualizer.py:
- visualize the flight path

Dependencies:
- anaconda
- python 3.9
- pytorch 2.2.1 with cuda 12.1: conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia
- stable baselines 3: pip install stable-baselines3[extra]
- gymnasium: pip install gymnasium

---

## 当前使用方法

### 配置文件说明

在 `config/envs.yaml` 中配置动力学模型类型：

**Simple Dynamics（第一题）：**
```yaml
port: 1015              # simple房间端口
dynamics_type: simple   # 动力学类型
my_initial_speed: 16    # 初始速度
```

**Junior Dynamics（第二题）：**
```yaml
port: 1040              # junior房间端口
dynamics_type: junior   # 动力学类型
my_initial_speed: 25    # 初始速度
```

### 训练

**从头训练：**
```bash
python main.py [版本名] [总步数]
```

**从预训练模型迁移学习：**
```bash
python main.py [版本名] [总步数] [预训练模型路径]
```

**示例：**
```bash
# Junior从头训练200万步
python main.py junior_v1 2000000

# Junior从simple模型迁移学习
python main.py junior_v1 2000000 correct_answer/best_mean_model.zip
```

### 测试

**测试模型性能：**
```bash
python test_model.py [模型路径] [测试次数]
```

**示例：**
```bash
# 测试simple模型（确保envs.yaml配置为simple）
python test_model.py correct_answer/best_mean_model.zip 3

# 测试junior模型（确保envs.yaml配置为junior）
python test_model.py model_junior_v1/best_mean_model.zip 3
```

**重要：测试前必须：**
1. 创建对应的房间（simple或junior）
2. 在 `config/envs.yaml` 中设置正确的 `port` 和 `dynamics_type`
3. 确保模型类型和房间类型匹配

### 关键区别

| 配置项 | Simple Dynamics | Junior Dynamics |
|--------|----------------|-----------------|
| 动力学特性 | 无重力、无升力 | 有重力、有升力 |
| 油门范围 | [0.15, 0.80] | [0.35, 1.05] |
| 初始速度 | 16 | 25 |
| 训练难度 | 较简单 | 较困难 |
| 建议步数 | 150万 | 200万+ |

代码会根据 `dynamics_type` 自动调整油门、奖励函数等参数。