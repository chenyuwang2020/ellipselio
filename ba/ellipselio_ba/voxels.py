"""Voxel construction and planar-patch selection for LiDAR bundle adjustment.

Points from all frames are transformed to the world frame with the current pose
estimate, bucketed into a uniform voxel grid, and each voxel is tested for
planarity. Surviving voxels become the BA constraints: their points should be
coplanar, and any deviation is attributed to pose error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class VoxelPatches:
    """Planar voxels selected as BA constraints.

    point_idx: (M,) int64  -- index into the flat point array, grouped by patch
    patch_ptr: (K+1,) int64 -- CSR-style offsets; patch k owns
                               point_idx[patch_ptr[k]:patch_ptr[k+1]]
    patch_id:  (M,) int32  -- patch index per entry (expanded form of patch_ptr)
    """

    point_idx: np.ndarray
    patch_ptr: np.ndarray
    patch_id: np.ndarray

    @property
    def num_patches(self) -> int:
        return len(self.patch_ptr) - 1

    @property
    def num_points(self) -> int:
        return len(self.point_idx)


def transform_points(points: np.ndarray, frame_id: np.ndarray,
                     quat: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Map per-frame LiDAR-frame points into the world frame."""
    rot = Rotation.from_quat(quat).as_matrix()  # (N, 3, 3)
    return np.einsum('pij,pj->pi', rot[frame_id], points) + trans[frame_id]


def build_patches(points_world: np.ndarray, frame_id: np.ndarray, *,
                  voxel_size: float = 1.0,
                  min_points: int = 50,
                  min_frames: int = 5,
                  planarity: float = 0.1,
                  max_thickness: float = 0.20,
                  max_points_per_patch: int = 300,
                  rng: np.random.Generator | None = None) -> VoxelPatches:
    """Bucket world points into voxels and keep the planar, multi-frame ones.

    A voxel becomes a constraint only when it holds enough points seen from
    enough distinct frames -- a voxel observed by a single frame constrains
    nothing, it just adds cost. Planarity is lambda0/lambda1 of the point
    covariance; max_thickness rejects voxels that are planar in shape but too
    thick to be a real surface (foliage, wires).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    key = np.floor(points_world / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True,
                                   return_counts=True)
    inverse = inverse.ravel()

    # Group point indices by voxel via a single sort.
    order = np.argsort(inverse, kind='stable')
    sorted_vox = inverse[order]
    starts = np.searchsorted(sorted_vox, np.arange(len(counts)), side='left')
    ends = starts + counts

    sel_idx, sel_ptr = [], [0]
    for v in np.where(counts >= min_points)[0]:
        members = order[starts[v]:ends[v]]

        if len(np.unique(frame_id[members])) < min_frames:
            continue

        q = points_world[members]
        centred = q - q.mean(axis=0)
        eigval = np.linalg.eigvalsh(centred.T @ centred / len(q))
        if eigval[1] <= 1e-9 or eigval[0] / eigval[1] > planarity:
            continue
        if np.sqrt(max(eigval[0], 0.0)) > max_thickness:
            continue

        # Cap patch size so a few dense voxels cannot dominate the cost.
        if len(members) > max_points_per_patch:
            members = rng.choice(members, max_points_per_patch, replace=False)

        sel_idx.append(members)
        sel_ptr.append(sel_ptr[-1] + len(members))

    if not sel_idx:
        raise RuntimeError(
            'no planar voxels found -- try a larger voxel_size or looser '
            'min_points/min_frames/planarity thresholds')

    point_idx = np.concatenate(sel_idx)
    patch_ptr = np.asarray(sel_ptr, dtype=np.int64)
    patch_id = np.repeat(np.arange(len(sel_idx), dtype=np.int32),
                         np.diff(patch_ptr))
    return VoxelPatches(point_idx=point_idx, patch_ptr=patch_ptr,
                        patch_id=patch_id)
