"""Synthetic check: perturb known poses, confirm BA recovers them."""

from __future__ import annotations

import numpy as np
import pypose as pp
import torch
from bae.optim import LM
from bae.utils.pysolvers import PCG
from pypose.optim.scheduler import StopOnPlateau
from scipy.spatial.transform import Rotation

from ellipselio_ba.optimize import PointPlaneGraph, make_se3


def make_scene(n_frames=12, seed=0):
    """Three orthogonal walls seen from a moving sensor."""
    rng = np.random.default_rng(seed)
    trans = np.stack([np.linspace(0, 2, n_frames),
                      np.zeros(n_frames), np.zeros(n_frames)], axis=1)
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (n_frames, 1))

    walls = []
    for axis, offset in [(0, 6.0), (1, 5.0), (2, -1.5)]:
        p = rng.uniform(-4, 4, size=(900, 3))
        p[:, axis] = offset
        walls.append(p)
    world = np.concatenate(walls)

    pts_local, frame_id = [], []
    for i in range(n_frames):
        R = Rotation.from_quat(quat[i]).as_matrix()
        pts_local.append((world - trans[i]) @ R)
        frame_id.append(np.full(len(world), i))
    return (np.concatenate(pts_local), np.concatenate(frame_id),
            quat, trans, world)


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64
    pts_local, frame_id, quat_gt, trans_gt, _ = make_scene()

    # Perturb all but frame 0 (the gauge-fixed one).
    rng = np.random.default_rng(1)
    trans_init = trans_gt.copy()
    quat_init = quat_gt.copy()
    trans_init[1:] += rng.normal(0, 0.05, trans_init[1:].shape)
    for i in range(1, len(quat_init)):
        dq = Rotation.from_rotvec(rng.normal(0, 0.01, 3))
        quat_init[i] = (dq * Rotation.from_quat(quat_init[i])).as_quat()

    err0 = np.linalg.norm(trans_init - trans_gt, axis=1)
    print(f'initial translation error: mean {err0.mean()*100:.2f} cm, '
          f'max {err0.max()*100:.2f} cm')

    from ellipselio_ba.voxels import build_patches, transform_points
    pw = transform_points(pts_local, frame_id, quat_init, trans_init)
    patches = build_patches(pw, frame_id, voxel_size=2.0, min_points=30,
                           min_frames=3, planarity=0.05)
    print(f'patches: {patches.num_patches}, points: {patches.num_points}')

    idx = patches.point_idx
    pl = torch.tensor(pts_local[idx], dtype=dtype, device=dev)
    fid = torch.tensor(frame_id[idx], dtype=torch.long, device=dev)
    pid = torch.tensor(patches.patch_id, dtype=torch.long, device=dev)

    poses = make_se3(quat_init, trans_init, dev, dtype)
    graph = PointPlaneGraph(poses).to(dev)

    from ellipselio_ba.residuals import patch_covariances, plane_from_covariance
    for outer in range(25):
        with torch.no_grad():
            pw_t = graph.all_poses()[fid] @ pl
            cent, cov = patch_covariances(pw_t, pid, patches.num_patches)
            cov = 0.5 * (cov + cov.transpose(-1, -2))
            nrm = plane_from_covariance(cov)
            normals = nrm[pid]
            offsets = (nrm * cent).sum(-1)[pid]
            w = torch.ones_like(offsets)

        opt = LM(graph, solver=PCG(tol=1e-6), min=1e-12, reject=20)
        sched = StopOnPlateau(opt, steps=12, patience=3, decreasing=1e-9,
                              verbose=False)
        sched.optimize(input={'points_local': pl, 'frame_id': fid,
                              'normals': normals, 'offsets': offsets,
                              'weights': w})

        with torch.no_grad():
            t_now = graph.all_poses().translation().cpu().numpy()
        e = np.linalg.norm(t_now - trans_gt, axis=1)
        print(f'  outer {outer}: translation error mean {e.mean()*100:.3f} cm, '
              f'max {e.max()*100:.3f} cm')

    ok = e.mean() < err0.mean() * 0.3
    print('RESULT:', 'PASS' if ok else 'FAIL',
          f'(error reduced {err0.mean()/max(e.mean(),1e-12):.1f}x)')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
