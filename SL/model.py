# Model part
import torch
from torch import nn
import torch.nn.functional as F

class CNNModel(nn.Module):

    def __init__(self, in_channels=6):
        nn.Module.__init__(self)
        self._tower = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * 4 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, 235)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input_dict):
        self.train(mode = input_dict.get("is_training", False))
        obs = input_dict["obs"]["observation"].float()
        action_logits = self._tower(obs)
        action_mask = input_dict["obs"]["action_mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        return action_logits + inf_mask


class ResBlock2d(nn.Module):
    """
    2D 残差块 — 参考 IJCAI 第4名架构。

    结构: x → Conv3×3 → BN → ReLU → Conv3×3 → BN → +x → ReLU

    channels 全程不变（所有 ResBlock 统一 64 通道），因此不需要 1×1 投影。
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual          # 残差连接
        out = F.relu(out)
        return out


class ResNetModel(nn.Module):
    """
    参考 IJCAI 第4名架构的 ResNet 麻将模型。

    结构:
      stem:  Conv2d(C→128,3×3) → Conv2d(128→64,3×3) → BN → ReLU
      body:  ResBlock2d(64) × 9
      head:  Flatten → FC(64×4×9, 235)

    输入/输出签名与 CNNModel 完全一致。
    """

    def __init__(self, in_channels=6, num_resblocks=9):
        """
        in_channels:  特征通道数，默认 6；扩展特征后可设为 145
        num_resblocks: 残差块数量，参考模型为 9
        """
        super().__init__()

        # ── stem ──
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.Conv2d(128, 64, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ── body ──
        self.resblocks = nn.Sequential(
            *[ResBlock2d(64) for _ in range(num_resblocks)]
        )

        # ── head ──
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 9, 235),
        )

        # ── 权重初始化 ──
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_dict):
        self.train(mode=input_dict.get("is_training", False))
        obs = input_dict["obs"]["observation"].float()
        action_mask = input_dict["obs"]["action_mask"].float()

        x = self.stem(obs)         # (B, 64, 4, 9)
        x = self.resblocks(x)      # (B, 64, 4, 9)
        logits = self.head(x)      # (B, 235)

        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        return logits + inf_mask