"""Solver loop: alternating point-to-plane, or direct BALM lambda_min."""

from __future__ import annotations

import contextlib
import io
import time

import numpy as np
import torch
from bae.optim import LM
from bae.utils.pysolvers import PCG
from pypose.optim import LevenbergMarquardt as PPLM
from pypose.optim.scheduler import StopOnPlateau

from .optimize import LambdaMinGraph, PointPlaneGraph, make_se3
from .residuals import huber_weight, patch_covariances, plane_from_covariance
from .voxels import build_patches, transform_points


def _to_quat_trans(poses) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        data = poses.tensor().detach().cpu().numpy()
    return data[:, 3:7].copy(), data[:, 0:3].copy()


def _health_warnings(num_patches: int, num_frames: int) -> list[str]:
    """Flag problem setups that would otherwise fail silently."""
    warn = []
    dof = 6 * max(num_frames - 1, 1)
    ratio = num_patches / dof
    if ratio < 0.15:
        warn.append(
            f'约束严重不足:只有 {num_patches} 个平面块,却要约束 {dof} 个位姿自由度\n'
            f'     (比值 {ratio:.3f},需要 > 0.3)。位姿在无约束方向上会自由乱跑,\n'
            f'     优化结果不可信。请减小 --voxel-size(室内通常用 0.2~0.4,不是 1.0)。')
    elif ratio < 0.3:
        warn.append(
            f'约束偏少:{num_patches} 个平面块对 {dof} 个位姿自由度'
            f'(比值 {ratio:.3f})。建议减小 --voxel-size。')
    return warn


def solve(scans, args) -> tuple[np.ndarray, np.ndarray]:
    dev, dtype = args.device, torch.float64
    quat, trans = scans.quat.copy(), scans.trans.copy()

    pts_all = scans.points
    fid_all = scans.frame_id

    # Relative-motion prior from the LIO trajectory, held fixed throughout.
    odom = None
    if args.odom_weight > 0:
        init = make_se3(quat, trans, dev, dtype)
        n = scans.num_frames
        oi = torch.arange(n - 1, device=dev)
        oj = oi + 1
        with torch.no_grad():
            rel = init[oi].Inv() @ init[oj]
        odom = (oi, oj, rel)

    graph = None
    rms_first = None
    for outer in range(args.outer):
        t0 = time.time()

        # Rebuild patches from the current poses: as poses improve, previously
        # rejected voxels can become planar and vice versa.
        pw = transform_points(pts_all, fid_all, quat, trans)
        patches = build_patches(
            pw, fid_all, voxel_size=args.voxel_size,
            min_points=args.min_points, min_frames=args.min_frames,
            planarity=args.planarity, max_thickness=args.max_thickness,
            max_points_per_patch=args.max_points_per_patch)

        # A handful of patches leaves the normal equations so degenerate that
        # the sparse solver itself breaks (bae's diagonal_op_ hits an
        # UnboundLocalError). Fail with an explanation instead.
        if patches.num_patches < 10:
            raise SystemExit(
                f'\n只筛出 {patches.num_patches} 个平面块,无法构成优化问题。\n'
                f'  当前 --voxel-size={args.voxel_size} --every={args.every}。\n'
                f'  抽帧太狠会让每个体素覆盖的帧数不够(--min-frames'
                f'={args.min_frames}),\n'
                f'  体素太大则筛不出平面。请减小 --every 或减小 --voxel-size。')

        idx = patches.point_idx
        pl = torch.tensor(pts_all[idx], dtype=dtype, device=dev)
        fid = torch.tensor(fid_all[idx], dtype=torch.long, device=dev)
        pid = torch.tensor(patches.patch_id, dtype=torch.long, device=dev)

        if args.cost == 'point2plane':
            if graph is None:
                graph = PointPlaneGraph(make_se3(quat, trans, dev, dtype)).to(dev)
        else:
            # The residual count equals the patch count, which changes between
            # outer iterations, so rebuild from the current poses rather than
            # mutating a graph whose LM state assumes a fixed residual length.
            graph = LambdaMinGraph(make_se3(quat, trans, dev, dtype),
                                   patches.num_patches).to(dev)

        if args.cost == 'point2plane':
            with torch.no_grad():
                pw_t = graph.all_poses()[fid] @ pl
                cent, cov = patch_covariances(pw_t, pid, patches.num_patches)
                cov = 0.5 * (cov + cov.transpose(-1, -2))
                nrm = plane_from_covariance(cov)
                normals = nrm[pid]
                offsets = (nrm * cent).sum(-1)[pid]
                resid = (normals * pw_t).sum(-1) - offsets
                weights = huber_weight(resid, args.huber)
            inputs = {'points_local': pl, 'frame_id': fid, 'normals': normals,
                      'offsets': offsets, 'weights': weights}
            if odom is not None:
                inputs.update(odom_i=odom[0], odom_j=odom[1],
                              odom_rel=odom[2], odom_w=args.odom_weight)
            solver = PCG(tol=1e-6)
        else:
            counts = torch.zeros(patches.num_patches, dtype=dtype, device=dev)
            counts.index_add_(0, pid, torch.ones_like(pid, dtype=dtype))
            inputs = {'points_local': pl, 'frame_id': fid, 'patch_id': pid,
                      'counts': counts}
            if odom is not None:
                inputs.update(odom_i=odom[0], odom_j=odom[1],
                              odom_rel=odom[2], odom_w=args.odom_weight)
            solver = None

        if solver is not None:
            # bae's LM is sparse-only: it calls bae.autograd.graph.jacobian
            # unconditionally, which needs the psjac trace on the parameters.
            opt = LM(graph, solver=solver, min=1e-12, reject=20)
        else:
            # lambda_min is a per-patch reduction, so psjac does not apply and
            # bae's LM cannot be used at all -- fall back to pypose's own LM,
            # which has a real dense path.
            opt = PPLM(graph, min=1e-12, reject=20, vectorize=False)
        sched = StopOnPlateau(opt, steps=args.inner, patience=3,
                              decreasing=1e-9, verbose=False)
        if getattr(args, 'verbose_lm', False):
            sched.optimize(input=inputs)
        else:
            # LM prints a loss/damping line per trial step; useful when
            # debugging, pure noise otherwise. The per-outer summary below
            # carries the same information.
            with contextlib.redirect_stdout(io.StringIO()):
                sched.optimize(input=inputs)

        quat, trans = _to_quat_trans(graph.all_poses())

        # Always report the same quantity -- RMS point-to-plane distance -- so
        # the two cost formulations are directly comparable. The raw residual
        # vector is not: lambda_min rows are per-patch and mix in odom rows.
        with torch.no_grad():
            pw_t = graph.all_poses()[fid] @ pl
            cent_e, cov_e = patch_covariances(pw_t, pid, patches.num_patches)
            cov_e = 0.5 * (cov_e + cov_e.transpose(-1, -2))
            nrm_e = plane_from_covariance(cov_e)
            dist = ((nrm_e[pid] * pw_t).sum(-1)
                    - (nrm_e * cent_e).sum(-1)[pid])
            rms = float(torch.sqrt((dist ** 2).mean()))
        if outer == 0:
            rms_first = rms
            for w in _health_warnings(patches.num_patches, scans.num_frames):
                print(f'\n  !! {w}\n')
        print(f'  第 {outer:2d} 轮: 平面块={patches.num_patches:6d}  '
              f'参与点={patches.num_points:8d}  '
              f'点面残差={rms * 100:.3f} cm  ({time.time() - t0:.1f}s)')

    return quat, trans
