# Chinese Standard Mahjong — Supervised Learning Bot

国标麻将监督学习 Bot，包含数据预处理、训练、评估、对战全流程。

## 目录结构

```
Chinese-Standard-Mahjong/
├── Sl/                        # 监督学习（Supervised Learning）
│   ├── agent.py               # Agent 基类：事件→状态→动作
│   ├── feature.py             # FeatureAgent：特征提取 + 合法动作生成
│   ├── model.py               # 神经网络模型定义
│   ├── preprocess.py          # 数据预处理：原始对局→训练样本
│   ├── dataset.py             # 数据集加载（支持延迟加载 + 数据增强）
│   ├── supervised.py          # 训练主程序（含逐类准确率 + 混淆矩阵）
│   ├── battle.py              # ★ 对战评估：加载训练好的模型进行多局对战
│   ├── __main__.py            # Botzone 平台交互入口
│   ├── train.py               # 早期训练脚本（已废弃，用 supervised.py）
│   ├── data/                  # 预处理后的数据
│   ├── log/                   # 训练日志和 checkpoint
│   └── battle_results/        # 对战结果
├── RL/                        # 强化学习（Reinforcement Learning）
│   ├── env.py                 # 完整游戏模拟器（洗牌→发牌→回合→结算）
│   ├── agent.py               # RL Agent 基类
│   ├── feature.py             # RL 特征提取
│   ├── model.py               # RL 模型（含 value head）
│   ├── actor.py               # Actor：自对弈收集数据
│   ├── learner.py             # Learner：训练更新
│   ├── replay_buffer.py       # 经验回放
│   ├── model_pool.py          # 模型池
│   └── train.py               # RL 训练入口
└── 国标麻将监督学习.pdf        # 课程说明
```

## 环境准备

```bash
# 安装 PyTorch
pip install torch numpy

# 安装算番库（必需）
pip install git+https://github.com/ailab-pku/PyMahjongGB.git

# 可选：图表输出
pip install matplotlib
```

## 完整工作流

### 第一步：数据预处理

将原始对局数据转换为训练样本：

```bash
cd Sl
python preprocess.py
```

- 输入：`data/data.txt`（98209 局专家自对弈数据）
- 输出：`data/0.npz, 1.npz, ...`（每 128 局一个文件）+ `data/count.json`

### 第二步：训练模型

```bash
cd Sl
python supervised.py
```

- 自动划分 90% 训练 / 10% 验证
- 每个 epoch 保存 checkpoint 到 `log/run_<timestamp>/checkpoint/`
- 输出：
  - `training_log.csv` — 训练/验证 loss、top-1/top-3 准确率
  - `per_type_accuracy.csv` — 每种动作类型的逐 epoch 准确率
  - `confusion_matrix.png` — 最终混淆矩阵热力图
  - `training_curves.png` — 训练曲线图
  - `model_architecture.txt` — 模型结构 + 参数量

### 第三步：改进模型

根据需要修改以下文件，然后重新训练：

| 文件 | 改进内容 |
|------|---------|
| `feature.py` | 扩展特征通道（6 → 145），加入其他玩家的副露、弃牌历史等 |
| `model.py` | 替换网络结构（如加深为 ResNet、加入 BatchNorm） |
| `dataset.py` | 添加数据增强（花色互换、对称互换）、延迟加载 |
| `supervised.py` | 调整超参数、checkpoint 续训 |

每次改进后运行 `python supervised.py` 获得新模型权重。

### 第四步：对战评估

用 `battle.py` 让新旧模型直接对战，量化改进效果。

---

## 对战系统使用指南

`battle.py` 使用 RL/env.py 中的完整游戏模拟器，加载 SL 训练出的模型权重，进行多局全自动对战。

### 基本用法

```bash
cd Sl

# 模型 A vs 模型 B，各 500 局
python battle.py --models log/run_A/checkpoint/19.pkl log/run_B/checkpoint/19.pkl --games 500
```

两个模型交叉坐位（A 坐 0/2 号位，B 坐 1/3 号位），每局轮换座位消除位置偏差。

### 不同网络结构对战（核心功能）

当你改进了模型结构，需要对比新旧架构：

```bash
python battle.py \
    --models log/run_baseline/checkpoint/19.pkl log/run_resnet/checkpoint/19.pkl \
    --model-classes model.CNNModel model.ResNetModel \
    --games 500
```

### 消融实验：单独验证某项改进

```bash
# 基线（6通道 + 浅CNN）
python supervised.py   # → log/run_baseline/19.pkl

# 仅改进特征（145通道 + 浅CNN）
# 修改 feature.py → python supervised.py → log/run_rich_feat/19.pkl

# 仅改进网络（6通道 + ResNet）
# 修改 model.py → python supervised.py → log/run_resnet/19.pkl

# 全量改进（145通道 + ResNet + 数据增强）
# 修改全部 → python supervised.py → log/run_full/19.pkl

# 对战对比
python battle.py --models log/run_baseline/19.pkl log/run_rich_feat/19.pkl --games 300
python battle.py --models log/run_baseline/19.pkl log/run_resnet/19.pkl --games 300
python battle.py --models log/run_rich_feat/19.pkl log/run_full/19.pkl --games 300
```

### 完整参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--models` | 路径列表 | 必填 | 模型 checkpoint 路径 |
| `--model-classes` | 类名列表 | `model.CNNModel` | 每个模型对应的类（`module.ClassName`） |
| `--seats` | 4个整数 | 自动 | 座位 0/1/2/3 分别用第几个模型 |
| `--games` | int | 200 | 对局总数 |
| `--device` | str | `cpu` | 推理设备（`cpu` / `cuda`） |
| `--selfplay` | flag | False | 自对弈模式（4家同一模型） |
| `--no-rotate` | flag | False | 禁止座位轮换 |
| `--verbose` | flag | False | 打印每步动作（调试用） |
| `--report-interval` | int | 50 | 每 N 局打印一次进度 |
| `--output` | 路径 | 自动 | 输出目录 |

### 输出文件

对战结束后在 `battle_results/run_<timestamp>/` 下生成：

| 文件 | 内容 |
|------|------|
| `battle_report.txt` | 完整文字报告 |
| `battle_summary.csv` | 一行一模型汇总表 |
| `battle_stats.json` | 完整统计数据（含每局得分序列） |
| `battle_charts.png` | 排名分布 + 得分箱线图 + 累计得分曲线 |
| `battle_config.json` | 对战配置记录 |

### 报告解读

```
Model                          Games    Win%  SelfDr%  DealIn%  AvgScore  AvgFan  AvgRank    Elo
------------------------------------------------------------------------------------------------
CNNModel@run_baseline/19       1000   22.3%    11.5%    24.1%     -2.34    18.2    2.561   1452
ResNetModel@run_resnet/19      1000   27.7%    14.2%    20.3%      2.34    20.5    2.439   1548
```

| 指标 | 含义 | 判断标准 |
|------|------|---------|
| Win% | 胡牌率 | 越高越好 |
| SelfDr% | 自摸率（占总局数） | 越高越好 |
| DealIn% | 点炮率 | 越低越好 |
| AvgScore | 每局平均得分 | 正值为佳 |
| AvgRank | 平均排名 (1-4) | < 2.5 表示强于平均水平 |
| Elo | 简化 Elo 评分 | 相对值，差值 > 50 通常意味着显著差异 |

### 自定义模型类

`--model-classes` 接受 `module.ClassName` 格式。只要模型满足以下接口即可：

```python
class YourModel(nn.Module):
    def __init__(self): ...           # 无参构造
    def forward(self, input_dict):    # 标准接口
        # input_dict = {
        #     'is_training': False,
        #     'obs': {
        #         'observation': tensor(B, C, 4, 9),
        #         'action_mask': tensor(B, 235),
        #     }
        # }
        # return tensor(B, 235)       # masked logits
        ...
```

示例：
```bash
# 从 my_models/resnet.py 导入 BigResNet
python battle.py \
    --models a.pkl b.pkl \
    --model-classes model.CNNModel my_models.resnet.BigResNet \
    --games 500
```

---

## 注意事项

1. **特征维度一致性**：模型 checkpoint 的输入通道数必须与 RL FeatureAgent 的特征维度（当前 OBS_SIZE=6）匹配。如果你扩展了特征通道，需要同步修改 RL 的 FeatureAgent，或在 battle.py 中替换 agent_clz。
2. **座位轮换**：默认开启，确保每个模型经历所有座位，消除位置偏差。仅在调试时可关闭。
3. **随机性**：麻将本身随机性大，建议至少 200 局以上才具有统计意义，500 局以上结论较可靠。
4. **无效动作**：如果模型频繁产生非法动作（报 Invalid），说明模型训练不充分或特征/网络有问题。
