"""Optimisation drivers: pypose LM over per-frame SE3 poses."""

from __future__ import annotations

import time

import numpy as np
import pypose as pp
import torch
from pypose.autograd.function import psjac
from torch import nn

from .residuals import (huber_weight, lambda_min, patch_covariances,
                        plane_from_covariance)


@psjac
def _point_plane_residual(poses, points_local, normals, offsets, weights):
    """One residual per point: w * (n^T (T p) - d). Touches a single pose each.

    Batched over points, so every row is independent -- the contract psjac
    requires. The robust weight is applied inside the traced function so the
    sparse-Jacobian trace terminates on a supported op.
    """
    pts_world = poses @ points_local
    res = (normals * pts_world).sum(dim=-1) - offsets
    return (res * weights).unsqueeze(-1)


@psjac
def _odom_residual(pose_i, pose_j, rel_init, weight):
    """Deviation of consecutive relative motion from the LIO estimate.

    Without this, a frame whose voxels give weak geometric support can drift
    almost freely; the LIO odometry is locally accurate, so anchoring relative
    motion costs nothing and bounds those frames. Touches two poses per row,
    which psjac supports (same shape as a pose-graph edge).
    """
    rel_est = pose_i.Inv() @ pose_j
    return (rel_init.Inv() @ rel_est).Log().tensor() * weight


class PointPlaneGraph(nn.Module):
    """Alternating point-to-plane BA. Frame 0 is held fixed to fix the gauge."""

    def __init__(self, poses: pp.LieTensor):
        super().__init__()
        self.register_buffer('pose_fixed', poses[:1].clone())
        self.poses_free = pp.Parameter(poses[1:].clone(), sjac=True)

    def all_poses(self) -> pp.LieTensor:
        return torch.cat([self.pose_fixed, self.poses_free], dim=0)

    def forward(self, points_local, frame_id, normals, offsets, weights,
                odom_i=None, odom_j=None, odom_rel=None, odom_w=0.0):
        poses = self.all_poses()
        plane = _point_plane_residual(poses[frame_id], points_local, normals,
                                      offsets, weights)
        if odom_i is None or odom_w <= 0.0:
            return plane
        odom = _odom_residual(poses[odom_i], poses[odom_j], odom_rel, odom_w)
        return torch.cat([plane, odom.reshape(-1, 1)], dim=0)


class LambdaMinGraph(nn.Module):
    """BALM cost: sqrt(N * lambda_min) per patch, plane marginalised.

    One residual couples every pose observing the patch, which is a batch
    reduction -- so this deliberately does NOT use psjac (see its contract) and
    runs with the dense-Jacobian LM path.
    """

    def __init__(self, poses: pp.LieTensor, num_patches: int):
        super().__init__()
        self.register_buffer('pose_fixed', poses[:1].clone())
        self.poses_free = pp.Parameter(poses[1:].clone())
        self.num_patches = num_patches

    def all_poses(self) -> pp.LieTensor:
        return torch.cat([self.pose_fixed, self.poses_free], dim=0)

    def forward(self, points_local, frame_id, patch_id, counts,
                odom_i=None, odom_j=None, odom_rel=None, odom_w=0.0):
        poses = self.all_poses()
        pts_world = poses[frame_id] @ points_local
        _, cov = patch_covariances(pts_world, patch_id, self.num_patches)
        # Symmetrise defensively: index_add_ accumulation is not exactly
        # symmetric in floating point, and eigh assumes it is.
        cov = 0.5 * (cov + cov.transpose(-1, -2))
        lam = lambda_min(cov).clamp(min=0.0)
        plane = torch.sqrt(counts * lam + 1e-12).unsqueeze(-1)
        if odom_i is None or odom_w <= 0.0:
            return plane
        # Same relative-motion prior as the point2plane path: without it,
        # frames with weak planar support drift freely.
        odom = _odom_residual(poses[odom_i], poses[odom_j], odom_rel, odom_w)
        return torch.cat([plane, odom.reshape(-1, 1)], dim=0)


def make_se3(quat: np.ndarray, trans: np.ndarray, device, dtype) -> pp.LieTensor:
    """Build an SE3 LieTensor from TUM-style translation + (qx,qy,qz,qw)."""
    data = np.concatenate([trans, quat], axis=1)
    return pp.SE3(torch.tensor(data, dtype=dtype, device=device))
