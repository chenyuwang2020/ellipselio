"""Load EllipseLIO results: per-scan clouds (LiDAR frame) + TUM trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Field layout of ellipselio's PointXYZNRGBIT, written by savePCDFileBinary.
_PT_DTYPE = np.dtype([
    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
    ('bin_idx', '<u4'), ('scan_idx', '<u4'),
    ('has_rgb', '<u4'), ('prim_type', '<u4'),
    ('rgb', '<f4'), ('intensity', '<f4'),
    ('time_secs', '<u4'), ('time_nsecs', '<u4'),
])

# prim_type = (saliency_idx + 1) * 85 in map_processing.cpp.
# NOTE: this is always 0 in the per-scan dumps -- it is only assigned during
# tensor voting on the accumulated map cloud, which happens after the per-scan
# save point. Planarity is therefore derived from voxel geometry here.
PRIM_PLANE = 85
PRIM_LINE = 170
PRIM_BALL = 255


@dataclass
class Scans:
    """Per-scan point clouds in the LiDAR frame, concatenated.

    points: (P, 3) float64 -- all points from all frames
    frame_id: (P,) int32   -- which frame each point came from
    quat: (N, 4) float64   -- initial poses, qx qy qz qw
    trans: (N, 3) float64
    stamps: (N,) float64
    """

    points: np.ndarray
    frame_id: np.ndarray
    prim_type: np.ndarray
    quat: np.ndarray
    trans: np.ndarray
    stamps: np.ndarray

    @property
    def num_frames(self) -> int:
        return len(self.trans)


def read_pcd_binary(path: Path) -> np.ndarray:
    """Read a binary PCD written with the EllipseLioPoint layout."""
    with open(path, 'rb') as f:
        header, blob = b'', b''
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f'{path}: truncated header')
            header += line
            if line.strip().startswith(b'DATA'):
                if b'binary' not in line:
                    raise ValueError(f'{path}: only binary PCD supported, got {line!r}')
                blob = f.read()
                break

    npts = 0
    for line in header.decode('ascii', 'ignore').splitlines():
        if line.startswith('POINTS'):
            npts = int(line.split()[1])

    if npts == 0:
        return np.empty(0, dtype=_PT_DTYPE)

    need = npts * _PT_DTYPE.itemsize
    if len(blob) < need:
        raise ValueError(f'{path}: expected {need} data bytes, got {len(blob)}')
    return np.frombuffer(blob[:need], dtype=_PT_DTYPE)


def load_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a TUM file. Returns (stamps, trans, quat) with quat as qx qy qz qw."""
    raw = np.loadtxt(path)
    if raw.ndim != 2 or raw.shape[1] != 8:
        raise ValueError(f'{path}: expected 8 columns (TUM), got shape {raw.shape}')
    return raw[:, 0].copy(), raw[:, 1:4].copy(), raw[:, 4:8].copy()


def load_results(result_dir: str | Path, every: int = 1,
                 max_frames: int | None = None,
                 min_range: float = 0.0) -> Scans:
    """Load an ellipselio result directory into a single flat point array.

    Args:
        result_dir: directory holding lidar_pose_tum.txt and scans/.
        every: keep every Nth frame (1 = all).
        max_frames: stop after this many kept frames.
        min_range: drop points closer than this to the sensor (removes
            self-hits on the operator carrying the rig).
    """
    root = Path(result_dir)
    stamps, trans, quat = load_trajectory(root / 'lidar_pose_tum.txt')

    scan_files = sorted((root / 'scans').glob('*.pcd'))
    if len(scan_files) != len(trans):
        raise ValueError(
            f'{len(scan_files)} scans but {len(trans)} poses -- these must '
            'correspond one-to-one; was the run interrupted?')

    keep = list(range(0, len(scan_files), every))
    if max_frames is not None:
        keep = keep[:max_frames]

    pts_all, fid_all, prim_all = [], [], []
    for new_id, i in enumerate(keep):
        rec = read_pcd_binary(scan_files[i])
        if not len(rec):
            continue
        xyz = np.stack([rec['x'], rec['y'], rec['z']], axis=1).astype(np.float64)
        if min_range > 0.0:
            xyz_ok = np.linalg.norm(xyz, axis=1) >= min_range
            xyz = xyz[xyz_ok]
            rec = rec[xyz_ok]
        if not len(xyz):
            continue
        pts_all.append(xyz)
        fid_all.append(np.full(len(xyz), new_id, dtype=np.int32))
        prim_all.append(rec['prim_type'].astype(np.uint32))

    return Scans(
        points=np.concatenate(pts_all),
        frame_id=np.concatenate(fid_all),
        prim_type=np.concatenate(prim_all),
        quat=quat[keep],
        trans=trans[keep],
        stamps=stamps[keep],
    )
