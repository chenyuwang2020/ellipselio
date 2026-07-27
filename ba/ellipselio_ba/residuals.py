"""Residual formulations for LiDAR bundle adjustment, built on pypose LieTensor.

Two costs are provided, both minimising the same underlying quantity:

point2plane
    Each patch's plane (normal, centroid) is fixed from the current poses, then
    every point contributes n^T (T_k p - c). Each residual touches exactly one
    pose, giving a perfectly sparse Jacobian. The plane is refreshed between
    outer iterations, making this an alternating (variable-projection style)
    scheme.

lambda_min
    The BALM cost proper: sqrt(N * lambda_min(cov)) per patch, where the plane
    is marginalised analytically rather than held fixed. One residual couples
    every pose observing the patch, so the Jacobian is denser, but no
    alternation is needed.

For a fixed optimal normal the two agree numerically:
    sum_i (n^T (p_i - c))^2 == n^T A n * N == N * lambda_min(A).
"""

from __future__ import annotations

import torch


def _segment_mean(values: torch.Tensor, patch_id: torch.Tensor,
                  num_patches: int) -> torch.Tensor:
    """Per-patch mean of (M, D) values, returned as (K, D)."""
    dim = values.shape[1]
    total = torch.zeros(num_patches, dim, dtype=values.dtype,
                        device=values.device)
    total.index_add_(0, patch_id, values)
    count = torch.zeros(num_patches, dtype=values.dtype, device=values.device)
    count.index_add_(0, patch_id, torch.ones_like(patch_id, dtype=values.dtype))
    return total / count.clamp(min=1.0).unsqueeze(1)


def patch_covariances(points_world: torch.Tensor, patch_id: torch.Tensor,
                      num_patches: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (centroids (K,3), covariances (K,3,3)) for each patch."""
    centroid = _segment_mean(points_world, patch_id, num_patches)
    centred = points_world - centroid[patch_id]
    outer = centred.unsqueeze(2) * centred.unsqueeze(1)  # (M, 3, 3)
    cov = torch.zeros(num_patches, 3, 3, dtype=points_world.dtype,
                      device=points_world.device)
    cov.index_add_(0, patch_id, outer)
    count = torch.zeros(num_patches, dtype=points_world.dtype,
                        device=points_world.device)
    count.index_add_(0, patch_id, torch.ones_like(patch_id,
                                                  dtype=points_world.dtype))
    cov = cov / count.clamp(min=1.0).view(-1, 1, 1)
    return centroid, cov


def plane_from_covariance(cov: torch.Tensor) -> torch.Tensor:
    """Unit normal of each patch: eigenvector of the smallest eigenvalue."""
    # eigh returns ascending eigenvalues, so column 0 is the normal.
    _, vecs = torch.linalg.eigh(cov)
    return vecs[..., 0]


class _LambdaMin(torch.autograd.Function):
    """Smallest eigenvalue of a symmetric 3x3, with a degeneracy-safe gradient.

    Differentiating torch.linalg.eigh brings in 1/(lambda_i - lambda_j) terms
    from the eigenvector derivatives. For a planar patch lambda_1 ~= lambda_2
    (the two in-plane directions), so those blow up to nan.

    The eigenvalue alone needs no such term -- by first-order perturbation
    theory d(lambda_min)/dA = n n^T, where n is the corresponding eigenvector.
    That is all this implements; everything upstream of A (pose composition,
    extrinsics, covariance accumulation) is left to autograd, so new
    optimisation variables need no gradient code of their own.
    """

    @staticmethod
    def forward(ctx, cov):
        eigval, eigvec = torch.linalg.eigh(cov)
        normal = eigvec[..., 0]
        ctx.save_for_backward(normal)
        return eigval[..., 0]

    @staticmethod
    def backward(ctx, grad_out):
        (normal,) = ctx.saved_tensors
        outer = normal.unsqueeze(-1) * normal.unsqueeze(-2)
        return grad_out[..., None, None] * outer


def lambda_min(cov: torch.Tensor) -> torch.Tensor:
    """Smallest eigenvalue per patch, safe to differentiate."""
    return _LambdaMin.apply(cov)


def transform_points(poses, points_local: torch.Tensor,
                     frame_id: torch.Tensor) -> torch.Tensor:
    """Apply each point's own frame pose. poses is an SE3 LieTensor of (N,)."""
    return poses[frame_id] @ points_local


def huber_weight(residual: torch.Tensor, delta: float) -> torch.Tensor:
    """sqrt of the Huber weight, for use as a residual multiplier.

    Scaling a residual r by sqrt(w) makes the squared cost match Huber's, so the
    least-squares solver inherits robustness without needing a loss hook.
    """
    if delta <= 0:
        return torch.ones_like(residual)
    absr = residual.abs()
    return torch.where(absr <= delta, torch.ones_like(absr),
                       torch.sqrt(delta / absr.clamp(min=1e-12)))
