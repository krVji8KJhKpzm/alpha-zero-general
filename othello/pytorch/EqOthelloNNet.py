# models/e2cnn_othello.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from e2cnn import gspaces
from e2cnn import nn as enn

class EqOthelloNet(nn.Module):
    """
    D4 等变 Othello 网络：
      - 输入: [B, C_in, 8, 8]，通常 C_in=2 或 3（我方/对手/占位或先手）
      - 输出:
          policy: [B, 64] 或 [B, 65]（含 pass），与棋盘 D4 变换严格等变
          value:  [B, 1]，与棋盘 D4 变换不变
    """
    def __init__(
        self,
        in_planes: int = 2,
        feat: int = 32,
        num_blocks: int = 5,
        include_pass: bool = True,
    ):
        super().__init__()
        self.include_pass = include_pass

        # --- 群空间：二维平面上的 4 阶旋转 + 翻转（D4）
        self.gspace = gspaces.FlipRot2dOnR2(N=4)

        # --- 输入类型（trivial 表示每个通道在群作用下“标量”，常规图像通道）
        in_type  = enn.FieldType(self.gspace, in_planes * [self.gspace.trivial_repr])

        # --- 我们用 Regular 表示来“携带”8 个朝向；feat 是“每个位置的 Regular 通道数”
        reg_type = enn.FieldType(self.gspace, feat * [self.gspace.regular_repr])

        # ===== Backbone：lifting (trivial -> regular) + 若干 group conv =====
        layers = []
        # lifting：把普通平面特征“抬升”到含朝向的群通道
        layers += [
            enn.R2Conv(in_type, reg_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(reg_type),
            enn.ReLU(reg_type, inplace=True),
        ]
        # group-to-group 卷积堆叠
        for _ in range(num_blocks - 1):
            layers += [
                enn.R2Conv(reg_type, reg_type, kernel_size=3, padding=1, bias=False),
                enn.InnerBatchNorm(reg_type),
                enn.ReLU(reg_type, inplace=True),
            ]
        self.backbone = enn.SequentialModule(*layers)

        # ===== Policy Head（保持等变，输出 1×8×8 的标量场）=====
        # 先做 1×1 等变卷积整合通道，再导出 trivial 标量场（仍等变，空间上会随输入旋转翻转）
        self.pi_conv1 = enn.R2Conv(reg_type, reg_type, kernel_size=1, bias=False)
        self.pi_bn1   = enn.InnerBatchNorm(reg_type)
        self.pi_relu1 = enn.ReLU(reg_type, inplace=True)
        pi_out_type   = enn.FieldType(self.gspace, 1 * [self.gspace.trivial_repr])
        self.pi_out   = enn.R2Conv(reg_type, pi_out_type, kernel_size=1, bias=True)

        # ===== Value Head（对群作用不变）=====
        # 对群通道做 GroupPooling（把 8 个朝向“池化”到不变），再全局空间平均+MLP
        self.gpool = enn.GroupPooling(reg_type)  # 输出类型依然是 FieldType，只是表示换成不变的
        # GroupPooling 后的通道数 = feat（每个 regular -> 1 个不变标量）
        v_in_ch = feat
        self.v_fc1 = nn.Linear(v_in_ch, 64)
        self.v_fc2 = nn.Linear(64, 1)

        # 可选：为“pass”动作单独给一个 logit（用全局特征）
        if self.include_pass:
            self.pass_fc = nn.Linear(v_in_ch, 1)

        # 记录一些元数据
        self.in_planes = in_planes
        self.feat = feat
        self.num_blocks = num_blocks

    def forward(self, x: torch.Tensor):
        """
        x: [B, C_in, 8, 8]
        return:
            pi: [B, 64] 或 [B, 65]
            v:  [B, 1]
        """
        B = x.size(0)
        # 包装成 GeometricTensor 才能过等变模块
        x = enn.GeometricTensor(x, enn.FieldType(self.gspace, self.in_planes * [self.gspace.trivial_repr]))

        # Backbone
        h = self.backbone(x)  # GeometricTensor, type=reg_type

        # --- Policy map（等变）---
        p = self.pi_conv1(h)
        p = self.pi_bn1(p)
        p = self.pi_relu1(p)
        p = self.pi_out(p)         # trivial（标量场），空间维仍是 8×8
        p_map = p.tensor.squeeze(1)  # [B, 8, 8] —— 每个位置一个 logit
        pi_flat = p_map.view(B, -1)  # -> [B, 64]

        if self.include_pass:
            # --- 全局不变特征（用于 pass）---
            g = self.gpool(h)              # 变成不变通道
            g = g.tensor.mean(dim=(2, 3))  # [B, feat] 全局空间平均
            pass_logit = self.pass_fc(g)   # [B, 1]
            pi = torch.cat([pi_flat, pass_logit], dim=1)  # [B, 65]
        else:
            pi = pi_flat

        # --- Value ---
        gv = self.gpool(h).tensor.mean(dim=(2, 3))  # [B, feat]
        v = torch.tanh(self.v_fc2(F.relu(self.v_fc1(gv), inplace=True)))  # [B, 1]

        return pi, v
