"""Quantitative before/after metrics for the BA result."""

from __future__ import annotations

import numpy as np

from .voxels import transform_points


def plane_thickness(points_world: np.ndarray, frame_id: np.ndarray, *,
                    voxel_size: float = 1.0, min_points: int = 50,
                    min_frames: int = 5, planarity: float = 0.1
                    ) -> np.ndarray:
    """1-sigma thickness of each planar voxel, in metres.

    This is the main quality signal: a real surface should be as thin as the
    sensor's range noise, so any excess is pose inconsistency.
    """
    key = np.floor(points_world / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True,
                                   return_counts=True)
    inverse = inverse.ravel()
    order = np.argsort(inverse, kind='stable')
    sorted_vox = inverse[order]
    starts = np.searchsorted(sorted_vox, np.arange(len(counts)), side='left')

    out = []
    for v in np.where(counts >= min_points)[0]:
        members = order[starts[v]:starts[v] + counts[v]]
        if len(np.unique(frame_id[members])) < min_frames:
            continue
        q = points_world[members]
        centred = q - q.mean(axis=0)
        ev = np.linalg.eigvalsh(centred.T @ centred / len(q))
        if ev[1] <= 1e-9 or ev[0] / ev[1] > planarity:
            continue
        out.append(np.sqrt(max(ev[0], 0.0)))
    return np.asarray(out)


def pose_delta(quat_a: np.ndarray, trans_a: np.ndarray,
               quat_b: np.ndarray, trans_b: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame translation (m) and rotation (deg) change between two paths."""
    from scipy.spatial.transform import Rotation

    dt = np.linalg.norm(trans_b - trans_a, axis=1)
    ra = Rotation.from_quat(quat_a)
    rb = Rotation.from_quat(quat_b)
    dr = np.degrees((rb * ra.inv()).magnitude())
    return dt, dr


def report(label: str, points_local: np.ndarray, frame_id: np.ndarray,
           quat: np.ndarray, trans: np.ndarray, *, voxel_size: float = 1.0
           ) -> dict:
    """Compute and print the metric set for one trajectory."""
    pw = transform_points(points_local, frame_id, quat, trans)
    th = plane_thickness(pw, frame_id, voxel_size=voxel_size)
    step = np.linalg.norm(np.diff(trans, axis=0), axis=1)
    stats = {
        'n_planar_voxels': len(th),
        'thickness_median_cm': float(np.median(th) * 100) if len(th) else float('nan'),
        'thickness_mean_cm': float(th.mean() * 100) if len(th) else float('nan'),
        'thickness_p90_cm': float(np.percentile(th, 90) * 100) if len(th) else float('nan'),
        'start_end_dist_m': float(np.linalg.norm(trans[-1] - trans[0])),
        'traj_length_m': float(step.sum()),
    }
    print(f'[{label}] 平面体素={stats["n_planar_voxels"]}  '
          f'平面厚度 中位={stats["thickness_median_cm"]:.3f} cm  '
          f'均值={stats["thickness_mean_cm"]:.3f} cm  '
          f'p90={stats["thickness_p90_cm"]:.3f} cm')
    print(f'[{label}] 首尾距离={stats["start_end_dist_m"]*100:.2f} cm  '
          f'轨迹长度={stats["traj_length_m"]:.2f} m')
    return stats
