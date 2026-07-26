#!/usr/bin/env python3
"""Convert a ROS1 bag with livox_ros_driver(2) CustomMsg into a rosbag2 usable by EllipseLIO.

CustomMsg is expanded into the PointCloud2 layout that livox_ros_driver2 emits with
xfer_format=0, i.e. 26-byte packed points:

    x,y,z      float32  @ 0, 4, 8
    intensity  float32  @ 12   (from CustomPoint.reflectivity)
    tag        uint8    @ 16
    line       uint8    @ 17
    timestamp  float64  @ 18   ABSOLUTE epoch nanoseconds

The timestamp field must be absolute: EllipseLIO reads it via
rclcpp::Time(in_pt.timestamp) in LidarProcess::SetPoint and discards the header
stamp entirely, so per-point time is written as timebase + offset_time.

Other topics are passed through unchanged.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.stores.ros2_humble import (
    builtin_interfaces__msg__Time as Time,
)
from rosbags.typesys.stores.ros2_humble import (
    sensor_msgs__msg__PointCloud2 as PointCloud2,
)
from rosbags.typesys.stores.ros2_humble import (
    sensor_msgs__msg__PointField as PointField,
)
from rosbags.typesys.stores.ros2_humble import (
    std_msgs__msg__Header as Header,
)

CUSTOM_MSG_TYPES = ('livox_ros_driver/msg/CustomMsg', 'livox_ros_driver2/msg/CustomMsg')

POINT_STEP = 26
_FLOAT32, _FLOAT64, _UINT8 = 7, 8, 2

# (name, offset, datatype) matching livox_ros_driver2 lddc.cpp InitPointcloud2MsgHeader
_FIELDS = (
    ('x', 0, _FLOAT32),
    ('y', 4, _FLOAT32),
    ('z', 8, _FLOAT32),
    ('intensity', 12, _FLOAT32),
    ('tag', 16, _UINT8),
    ('line', 17, _UINT8),
    ('timestamp', 18, _FLOAT64),
)

_PT = struct.Struct('<ffffBBd')
assert _PT.size == POINT_STEP, _PT.size

_NP_PT = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4'),
                   ('tag', 'u1'), ('line', 'u1'), ('timestamp', '<f8')])
assert _NP_PT.itemsize == POINT_STEP, _NP_PT.itemsize


def make_fields() -> list[PointField]:
    return [
        PointField(name=name, offset=off, datatype=dt, count=1)
        for name, off, dt in _FIELDS
    ]


def custom_to_pointcloud2(msg, frame_id: str | None,
                          drop_zero: bool) -> tuple[PointCloud2, int, int]:
    """Expand a Livox CustomMsg into a PointCloud2.

    Returns (msg, header_stamp_ns, dropped_count).
    """
    timebase = int(msg.timebase)
    pts = msg.points

    buf = bytearray(POINT_STEP * len(pts))
    pack = _PT.pack_into
    off = 0
    dropped = 0
    for p in pts:
        # Zero-range points are Livox no-return placeholders. EllipseLIO's filter is
        # `range < min_range`, so with min_range: 0.0 they would all survive and pile
        # into range bin 0, skewing the adaptive bin statistics.
        if drop_zero and p.x == 0.0 and p.y == 0.0 and p.z == 0.0:
            dropped += 1
            continue
        # offset_time is relative to timebase in CustomMsg; EllipseLIO needs absolute.
        pack(buf, off, p.x, p.y, p.z, float(p.reflectivity), p.tag, p.line,
             float(timebase + p.offset_time))
        off += POINT_STEP

    num = off // POINT_STEP
    del buf[off:]

    # Anchor the header on timebase so header stamp and per-point times agree.
    stamp_ns = timebase
    header = Header(
        stamp=Time(sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000),
        frame_id=frame_id if frame_id is not None else msg.header.frame_id,
    )

    return PointCloud2(
        header=header,
        height=1,
        width=num,
        fields=make_fields(),
        is_bigendian=False,
        point_step=POINT_STEP,
        row_step=POINT_STEP * num,
        data=np.frombuffer(bytes(buf), dtype=np.uint8),
        is_dense=True,
    ), stamp_ns, dropped


def header_stamp_ns(msg) -> int | None:
    hdr = getattr(msg, 'header', None)
    if hdr is None:
        return None
    return hdr.stamp.sec * 1_000_000_000 + hdr.stamp.nanosec


def convert(src: Path, dst: Path, *, use_header_time: bool, frame_id: str | None,
            topics: list[str] | None, drop_zero: bool) -> None:
    hts = get_typestore(Stores.ROS2_HUMBLE)

    if dst.exists():
        sys.exit(f'error: output {dst} already exists, refusing to overwrite')

    with AnyReader([src]) as reader, Writer(dst, version=8) as writer:
        conns = [
            c for c in reader.connections
            if topics is None or c.topic in topics
        ]
        if not conns:
            sys.exit('error: no matching topics found')

        out_conns = {}
        for c in conns:
            msgtype = 'sensor_msgs/msg/PointCloud2' if c.msgtype in CUSTOM_MSG_TYPES \
                else c.msgtype
            if msgtype not in hts.types:
                print(f'  skip {c.topic} ({c.msgtype}): unknown in ROS2 typestore')
                continue
            out_conns[c.topic] = (
                writer.add_connection(c.topic, msgtype, typestore=hts), msgtype)
            kind = 'CustomMsg -> PointCloud2' if c.msgtype in CUSTOM_MSG_TYPES \
                else 'passthrough'
            print(f'  {c.topic:24s} {msgtype:32s} n={c.msgcount:<7} [{kind}]')

        conns = [c for c in conns if c.topic in out_conns]
        total = sum(c.msgcount for c in conns)
        print(f'\nconverting {total} messages ...')

        n = 0
        kept = dropped = 0
        for con, bag_ts, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, con.msgtype)
            out_con, msgtype = out_conns[con.topic]

            if con.msgtype in CUSTOM_MSG_TYPES:
                msg, stamp_ns, drop = custom_to_pointcloud2(msg, frame_id, drop_zero)
                dropped += drop
                kept += msg.width
            else:
                stamp_ns = header_stamp_ns(msg)
                if frame_id is not None and getattr(msg, 'header', None) is not None \
                        and con.topic.endswith(('/lidar', '/imu')):
                    msg.header.frame_id = frame_id

            ts = stamp_ns if (use_header_time and stamp_ns) else bag_ts
            writer.write(out_con, ts, hts.serialize_cdr(msg, msgtype))

            n += 1
            if n % 2000 == 0:
                print(f'    {n}/{total}', flush=True)

    if kept or dropped:
        pct = 100.0 * dropped / (kept + dropped)
        print(f'lidar points: kept {kept}, dropped {dropped} zero-range ({pct:.1f}%)')
    print(f'done: {n} messages -> {dst}')


def verify(bag: Path, lidar_topic: str, imu_topic: str, nframes: int = 3) -> int:
    """Sanity-check per-point timestamps in the converted bag. Returns exit code."""
    hts = get_typestore(Stores.ROS2_HUMBLE)
    ok = True

    with AnyReader([bag]) as reader:
        lconns = [c for c in reader.connections if c.topic == lidar_topic]
        if not lconns:
            print(f'verify: no {lidar_topic} in output')
            return 1
        if lconns[0].msgtype != 'sensor_msgs/msg/PointCloud2':
            print(f'verify: {lidar_topic} is {lconns[0].msgtype}, expected PointCloud2')
            return 1

        print(f'\nverify: {lidar_topic}')
        seen = 0
        prev_end = None
        for con, bag_ts, raw in reader.messages(connections=lconns):
            m = reader.deserialize(raw, con.msgtype)
            names = {f.name: (f.offset, f.datatype) for f in m.fields}
            if 'timestamp' not in names:
                print('  FAIL: no per-point "timestamp" field')
                return 1
            toff, tdt = names['timestamp']
            if tdt != _FLOAT64 or m.point_step != POINT_STEP:
                print(f'  FAIL: timestamp datatype={tdt} point_step={m.point_step}')
                return 1

            data = memoryview(m.data)
            ts = [struct.unpack_from('<d', data, i * m.point_step + toff)[0]
                  for i in range(m.width)]
            t0, t1 = min(ts), max(ts)
            hdr = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            span = (t1 - t0) / 1e9
            skew = t0 / 1e9 - hdr

            arr = np.frombuffer(bytes(m.data), dtype=_NP_PT)
            rng = np.sqrt(arr['x'] ** 2 + arr['y'] ** 2 + arr['z'] ** 2)
            nzero = int((rng == 0).sum())

            # Absolute epoch ns: must be a plausible wall-clock time, not ~0.
            plausible = t0 > 1e18
            good = plausible and 0.0 < span < 0.5 and abs(skew) < 0.05
            ok &= good
            monotonic = prev_end is None or t0 >= prev_end - 1e6
            ok &= monotonic
            prev_end = t1

            print(f'  pts={m.width:<6} span={span * 1e3:6.2f}ms  '
                  f'first-pt minus hdr={skew * 1e3:+7.3f}ms  '
                  f'rng<={rng.max():5.1f}m zero={nzero}  '
                  f'epoch_ok={plausible} mono={monotonic} -> '
                  f'{"OK" if good and monotonic else "FAIL"}')
            seen += 1
            if seen >= nframes:
                break

        iconns = [c for c in reader.connections if c.topic == imu_topic]
        if iconns:
            accs = []
            for con, bag_ts, raw in reader.messages(connections=iconns):
                m = reader.deserialize(raw, con.msgtype)
                a = m.linear_acceleration
                accs.append((a.x ** 2 + a.y ** 2 + a.z ** 2) ** 0.5)
                if len(accs) >= 200:
                    break
            mean = sum(accs) / len(accs)
            unit = 'g' if mean < 3 else 'm/s^2'
            print(f'verify: {imu_topic} mean |acc| = {mean:.3f} ({unit}; '
                  f'EllipseLIO normalises either)')
        else:
            print(f'verify: WARNING no {imu_topic} in output')
            ok = False

    print('verify: PASS' if ok else 'verify: FAIL')
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', type=Path, help='input ROS1 .bag')
    ap.add_argument('dst', type=Path, nargs='?',
                    help='output rosbag2 dir (default: <src stem>_ros2 beside src)')
    ap.add_argument('--topics', nargs='+', default=None,
                    help='only convert these topics (default: all)')
    ap.add_argument('--frame-id', default=None,
                    help='override frame_id on lidar/imu (default: keep)')
    ap.add_argument('--bag-time', action='store_true',
                    help='keep original bag record times instead of header stamps')
    ap.add_argument('--keep-zero-range', action='store_true',
                    help='keep (0,0,0) no-return points; by default they are dropped')
    ap.add_argument('--lidar-topic', default='/livox/lidar')
    ap.add_argument('--imu-topic', default='/livox/imu')
    ap.add_argument('--verify-only', action='store_true',
                    help='skip conversion, only verify an existing rosbag2 at dst')
    args = ap.parse_args()

    dst = args.dst or args.src.parent / f'{args.src.stem}_ros2'

    if not args.verify_only:
        if not args.src.exists():
            sys.exit(f'error: {args.src} not found')
        print(f'{args.src}  ->  {dst}')
        convert(args.src, dst, use_header_time=not args.bag_time,
                frame_id=args.frame_id, topics=args.topics,
                drop_zero=not args.keep_zero_range)

    return verify(dst, args.lidar_topic, args.imu_topic)


if __name__ == '__main__':
    sys.exit(main())
