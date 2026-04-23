"""SO(3) 鲁棒轴融合。

将多个预测轴矩阵 (N, 3, 3) 看作 SO(3) 上的一组采样，用 **Iteratively
Reweighted Least Squares (IRLS) + Cauchy 核** 求几何中位数 (robust
Fréchet mean)，等价于：

    min_{R*}  sum_i rho( d(R_i^src, R*)^2 + d(R_i^tgt, R*)^2 )

其中 d(R1, R2) = || log(R1 R2^{-1}) ||，rho 为 Cauchy 鲁棒核；偏差大的
输入在迭代中自动降权，等效于 "软剔除" 离群点，不需要事先挑阈值。
"""

import torch

from tqdm import trange
from typing import Optional


def _skew(v: torch.Tensor) -> torch.Tensor:
    """向量 -> 反对称矩阵。``v: (..., 3) -> (..., 3, 3)``"""
    zero = torch.zeros_like(v[..., 0])
    row0 = torch.stack([zero, -v[..., 2], v[..., 1]], dim=-1)
    row1 = torch.stack([v[..., 2], zero, -v[..., 0]], dim=-1)
    row2 = torch.stack([-v[..., 1], v[..., 0], zero], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _so3_exp(omega: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """so(3) -> SO(3) 的 Rodrigues 指数映射。``omega: (..., 3) -> (..., 3, 3)``"""
    theta = omega.norm(dim=-1)  # (...,)
    batch_shape = omega.shape[:-1]
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(
        *batch_shape, 3, 3
    )

    theta_safe = theta.clamp(min=eps)
    axis = omega / theta_safe.unsqueeze(-1)
    K = _skew(axis)  # (..., 3, 3)

    sin_t = torch.sin(theta)[..., None, None]
    cos_t = torch.cos(theta)[..., None, None]
    R = eye + sin_t * K + (1.0 - cos_t) * (K @ K)

    # theta ~ 0 时直接退化成单位矩阵，避免 0/0
    small = (theta < eps)[..., None, None]
    return torch.where(small, eye, R)


def _so3_log(R: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """SO(3) -> so(3) 的对数映射。``R: (..., 3, 3) -> (..., 3)``

    theta 接近 0 用 Taylor 展开；theta 接近 pi 的极端情形由于我们从
    chordal 均值初始化 + 迭代收缩，在实际输入里几乎不会出现，这里沿用
    主分支公式 + clamp 保证数值稳定。
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)  # (...,)
    vee = torch.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        dim=-1,
    )  # (..., 3), 幅度 = 2 sin(theta)

    sin_theta = torch.sin(theta)
    small = theta < 1e-4
    coef_small = 0.5 + theta * theta / 12.0  # Taylor: theta/(2 sin theta)
    coef_regular = theta / (2.0 * sin_theta.clamp(min=eps))
    coef = torch.where(small, coef_small, coef_regular)
    return coef.unsqueeze(-1) * vee


def _chordal_mean(R_all: torch.Tensor) -> torch.Tensor:
    """闭式 L2 (chordal) 投影均值，作为 IRLS 的热启动。

    M = mean(R_i)，对 M 做 SVD 得 U S V^T，投影到 SO(3) 为
    ``U diag(1, 1, det(U V^T)) V^T``。
    """
    M = R_all.mean(dim=0)  # (3, 3)
    U, _, Vh = torch.linalg.svd(M)
    d = torch.det(U @ Vh)
    D = torch.diag(
        torch.stack(
            [
                torch.ones_like(d),
                torch.ones_like(d),
                d,
            ]
        )
    )
    return U @ D @ Vh


def _collect_rotations(
    src_axis_world: torch.Tensor,
    tgt_axis_world: Optional[torch.Tensor],
) -> torch.Tensor:
    """把 src / (可选) tgt 堆成 ``(M, 3, 3)`` 并提升到 float32。"""
    if src_axis_world.ndim != 3 or src_axis_world.shape[-2:] != (3, 3):
        raise ValueError(
            f'[fuseAxisWorld] src_axis_world shape must be (N, 3, 3), '
            f'got {tuple(src_axis_world.shape)}'
        )

    stacks = [src_axis_world]
    if tgt_axis_world is not None:
        if tgt_axis_world.shape != src_axis_world.shape:
            raise ValueError(
                f'[fuseAxisWorld] tgt shape {tuple(tgt_axis_world.shape)} '
                f'!= src shape {tuple(src_axis_world.shape)}'
            )
        stacks.append(tgt_axis_world)

    R_all = torch.cat(stacks, dim=0)
    if R_all.numel() == 0:
        raise ValueError('[fuseAxisWorld] empty rotation stack')
    return R_all.to(torch.float32)


def fuseAxisWorld(
    src_axis_world: torch.Tensor,
    tgt_axis_world: Optional[torch.Tensor] = None,
    iters: int = 20,
    cauchy_c: float = 0.1,
    tol: float = 1e-6,
    return_weights: bool = False,
) -> torch.Tensor:
    """在 SO(3) 上求一组轴矩阵的鲁棒 Fréchet 均值 (几何中位数)。

    参数
    ----
    src_axis_world : ``(N, 3, 3)`` 旋转矩阵 (行=方向，front/left/up)，
        期望元素近似属于 SO(3)。
    tgt_axis_world : 可选的配对侧旋转矩阵，形状同 src；非 ``None`` 时
        参与同一个投票池 (相当于每个 i 贡献两票)。
    iters : IRLS 最大迭代次数，默认 20。
    cauchy_c : Cauchy 核尺度，单位为弧度 (典型 0.05~0.2)。越小越激进地
        剔除离群点；越大越接近普通 L2 均值。默认 0.1。
    tol : 当本轮更新量 ``||delta||`` 小于该阈值时提前停止。
    return_weights : 若为 True，额外返回 ``(R*, weights)``。weights 形
        状与拼接后的输入一致 (``(M,)``，前 N 个对应 src，后 N 个对应
        tgt)，可供调用方查看谁被降权。

    返回
    ----
    默认返回融合后的 ``(3, 3)`` 旋转矩阵 (float32，与输入相同 device)。
    若 ``return_weights=True``，返回 ``(R*, weights)``。
    """
    R_all = _collect_rotations(src_axis_world, tgt_axis_world)  # (M, 3, 3)
    device = R_all.device

    R_star = _chordal_mean(R_all)  # (3, 3)

    c2 = float(cauchy_c) * float(cauchy_c)
    weights: Optional[torch.Tensor] = None

    print('[INFO][axis_fusion::fuseAxisWorld]')
    print('\t start fuse axis world...')
    for _ in trange(iters):
        rel = R_all @ R_star.transpose(-1, -2)  # (M, 3, 3)
        r = _so3_log(rel)  # (M, 3)

        r_sq = (r * r).sum(dim=-1)  # (M,)
        w = 1.0 / (1.0 + r_sq / c2)  # Cauchy 权重
        weights = w

        w_sum = w.sum().clamp(min=1e-8)
        delta = (w.unsqueeze(-1) * r).sum(dim=0) / w_sum  # (3,)

        R_star = _so3_exp(delta) @ R_star

        if float(delta.norm()) < tol:
            break

    # 最后再做一次 SVD 投影，保证严格落在 SO(3) 上
    U, _, Vh = torch.linalg.svd(R_star)
    d = torch.det(U @ Vh)
    D = torch.diag(
        torch.stack(
            [torch.ones_like(d), torch.ones_like(d), d]
        )
    )
    R_star = (U @ D @ Vh).to(device=device)

    if return_weights:
        assert weights is not None
        return R_star, weights  # type: ignore[return-value]
    return R_star
