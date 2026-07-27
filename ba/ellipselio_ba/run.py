"""Offline LiDAR bundle adjustment for EllipseLIO results.

Reads a lio_res/ellipselio directory (per-scan clouds in the LiDAR frame plus a
TUM trajectory), refines the per-frame poses so that co-observed surfaces become
consistent, and writes the refined trajectory plus a rebuilt map.

Usage:
    uv run python -m ellipselio_ba.run --result-dir <dir> --cost point2plane
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import torch

# PyTorch/warp print kernel-override and beta-feature notices on first use that
# would otherwise bury the setup warnings below.
warnings.filterwarnings('ignore', message='.*Sparse BSR tensor support.*')
warnings.filterwarnings('ignore', message='.*Overriding a previously registered kernel.*')
warnings.filterwarnings('ignore', message='.*Warning only once for all operators.*')

# warp prints a multi-line device banner on init; not a Python warning.
try:
    import warp as _warp
    _warp.config.log_level = _warp.LOG_WARNING
except Exception:
    pass

from .config import PRESETS, resolve, show_presets
from .io_ellipselio import load_results
from .metrics import pose_delta, report
from .voxels import build_patches, transform_points


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument('--config', type=Path, default=None,
                   help='YAML 配置文件路径')
    p.add_argument('--preset', choices=sorted(PRESETS),
                   help='场景预设: ' + '; '.join(
                       f'{k}={v["_desc"]}' for k, v in PRESETS.items()))
    p.add_argument('--print-config', action='store_true',
                   help='只打印最终生效的参数,不运行优化')
    p.add_argument('--show-presets', action='store_true',
                   help='打印所有场景预设的参数对照表及取值理由,然后退出')

    g = p.add_argument_group('输入输出')
    g.add_argument('--result-dir', type=Path,
                   help='ellipselio 结果目录(含 lidar_pose_tum.txt 和 scans/)')
    g.add_argument('--out-dir', type=Path,
                   help='输出目录(默认 <result-dir>/../ellipselio_ba)')
    g.add_argument('--every', type=int, help='每 N 帧取一帧(调参用)')
    g.add_argument('--max-frames', type=int, help='最多用多少帧')
    g.add_argument('--min-range', type=float, help='丢弃近于此距离的点(米)')
    g.add_argument('--rebuild-map', action='store_true', default=None,
                   help='导出优化前后的拼接地图')
    g.add_argument('--map-voxel', type=float,
                   help='导出地图的降采样体素(米,0=不降采样)')

    g = p.add_argument_group('平面块构建')
    g.add_argument('--voxel-size', type=float,
                   help='体素边长(米)。最关键的参数,必须匹配场景尺度')
    g.add_argument('--min-points', type=int, help='体素最少点数')
    g.add_argument('--min-frames', type=int, help='体素最少覆盖帧数')
    g.add_argument('--planarity', type=float, help='平面性阈值 lambda0/lambda1')
    g.add_argument('--max-thickness', type=float, help='平面最大厚度(米)')
    g.add_argument('--max-points-per-patch', type=int, help='单个平面块最多点数')

    g = p.add_argument_group('优化')
    g.add_argument('--cost', choices=['point2plane', 'lambda_min'],
                   help='残差形式')
    g.add_argument('--outer', type=int, help='外层迭代次数(重算平面)')
    g.add_argument('--inner', type=int, help='每轮 LM 步数')
    g.add_argument('--huber', type=float, help='Huber 鲁棒核阈值(米,0=关闭)')
    g.add_argument('--odom-weight', type=float,
                   help='里程计相对运动先验权重(0=关闭)。约束几何支撑弱的帧')
    g.add_argument('--device', help='cuda 或 cpu')
    g.add_argument('--verbose-lm', action='store_true', default=None,
                   help='显示 LM 每步的 loss/damping(调试用)')

    args = p.parse_args(argv)

    # Informational only; needs no --result-dir.
    if args.show_presets:
        show_presets()
        raise SystemExit(0)

    cli = {k: v for k, v in vars(args).items()
           if k not in ('config', 'preset', 'print_config', 'show_presets')}
    cfg, origin = resolve(cli, args.config, args.preset)

    if cfg.result_dir is None:
        p.error('必须提供 --result-dir(命令行或配置文件)')
    if cfg.device is None:
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return cfg, origin, args.print_config


def write_tum(path: Path, stamps, trans, quat) -> None:
    with open(path, 'w') as f:
        f.write('# timestamp tx ty tz qx qy qz qw\n')
        for s, t, q in zip(stamps, trans, quat):
            f.write(f'{s:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} '
                    f'{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n')


def write_map(path: Path, points: np.ndarray, voxel: float = 0.0) -> None:
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    if voxel > 0:
        pc = pc.voxel_down_sample(voxel)
    o3d.io.write_point_cloud(str(path), pc, write_ascii=False, compressed=False)
    print(f'  已写入 {path.name}: {len(pc.points)} 个点')


def print_verdict(before: dict, after: dict) -> None:
    """Summarise all three quality signals, not just thickness.

    Thickness alone can look flat while the trajectory is being pulled apart,
    so trajectory length and loop drift are checked too.
    """
    tb, ta = before['thickness_median_cm'], after['thickness_median_cm']
    print('\n===== 优化结果 =====')
    thin_bad = False
    if tb == tb and ta == ta and tb > 0:
        d = 100.0 * (ta - tb) / tb
        mark = '变好' if d < -1 else '基本不变' if d < 1 else '变差 <<<'
        thin_bad = d > 1
        print(f'  平面厚度(中位)  {tb:7.3f} -> {ta:7.3f} cm  {d:+6.1f}%   {mark}')
    lb, la = before['traj_length_m'], after['traj_length_m']
    dl = 100.0 * (la - lb) / lb if lb else 0.0
    print(f'  轨迹长度        {lb:7.2f} -> {la:7.2f} m   {dl:+6.1f}%   '
          f'{"正常" if abs(dl) < 1 else "异常:位姿被拉散 <<<"}')
    sb, sa = before['start_end_dist_m'] * 100, after['start_end_dist_m'] * 100
    print(f'  首尾距离        {sb:7.2f} -> {sa:7.2f} cm            '
          f'{"正常" if sa <= sb + 1 else "异常:漂移增大 <<<"}')

    if thin_bad or abs(dl) > 1:
        print('\n  ***** 优化失败:地图没有变好 *****')
        print('  最常见的原因是 --voxel-size 相对场景太大。请看上面的')
        print('  「约束严重不足」警告,并减小 --voxel-size 重跑。')
        print('  室内(场景 < 10 m)用 0.2,室外街道尺度用 1.0。')
    else:
        print('\n  优化成功。')


def main(argv=None) -> int:
    args, origin, print_only = parse_args(argv)

    print('参数来源: ' + (' -> '.join(origin) if origin else '全部使用默认值'))
    print(f'  voxel_size={args.voxel_size}  huber={args.huber}  '
          f'outer={args.outer}  odom_weight={args.odom_weight}  '
          f'cost={args.cost}  device={args.device}')
    if print_only:
        from .config import DEFAULTS
        print('\n完整参数:')
        for k in sorted(DEFAULTS):
            print(f'  {k}: {getattr(args, k)}')
        return 0

    out_dir = args.out_dir or args.result_dir.parent / 'ellipselio_ba'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n读取 {args.result_dir}')
    t0 = time.time()
    scans = load_results(args.result_dir, every=args.every,
                         max_frames=args.max_frames, min_range=args.min_range)
    print(f'  {scans.num_frames} 帧, {len(scans.points)} 个点 '
          f'({time.time() - t0:.1f}s)')

    extent = scans.trans.max(axis=0) - scans.trans.min(axis=0)
    span = float(np.linalg.norm(extent))
    print(f'  场景范围 {extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f} m'
          f'   当前 --voxel-size {args.voxel_size} m')
    if span < 20.0 and args.voxel_size > 0.5:
        print(f'\n  !! 场景很小(对角线 {span:.1f} m)但体素很大({args.voxel_size} m)。')
        print(f'     室内数据通常需要 --voxel-size 0.2~0.4,否则能通过平面')
        print(f'     筛选的体素太少,优化会失败。建议加上 --voxel-size 0.2\n')

    quat0, trans0 = scans.quat.copy(), scans.trans.copy()

    print('\n--- 优化前 ---')
    before = report('优化前', scans.points, scans.frame_id, quat0, trans0,
                    voxel_size=args.voxel_size)

    from .solve import solve
    quat1, trans1 = solve(scans, args)

    print('\n--- 优化后 ---')
    after = report('优化后', scans.points, scans.frame_id, quat1, trans1,
                   voxel_size=args.voxel_size)

    dt, dr = pose_delta(quat0, trans0, quat1, trans1)
    print(f'\n位姿改动量: 平移 平均 {dt.mean()*100:.2f} cm '
          f'最大 {dt.max()*100:.2f} cm | 旋转 平均 {dr.mean():.4f} 度 '
          f'最大 {dr.max():.4f} 度')

    print_verdict(before, after)

    write_tum(out_dir / 'lidar_pose_tum_ba.txt', scans.stamps, trans1, quat1)
    print(f'\n已写入 {out_dir / "lidar_pose_tum_ba.txt"}')

    if args.rebuild_map:
        write_map(out_dir / 'map_before.pcd',
                  transform_points(scans.points, scans.frame_id, quat0, trans0),
                  args.map_voxel)
        write_map(out_dir / 'map_after.pcd',
                  transform_points(scans.points, scans.frame_id, quat1, trans1),
                  args.map_voxel)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
