"""Configuration: YAML file + presets, overridable on the command line.

Precedence, lowest to highest:
    built-in defaults  <  preset  <  YAML file  <  command-line flags

So `--config indoor.yaml --outer 30` runs the file's settings with 30 outer
iterations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Every tunable, with its default. Also the whitelist used to reject typos in
# YAML files -- a silently ignored key is worse than an error.
DEFAULTS: dict[str, Any] = {
    # input / output
    'result_dir': None,
    'out_dir': None,
    'every': 1,
    'max_frames': None,
    'min_range': 0.5,
    # patch construction
    'voxel_size': 1.0,
    'min_points': 50,
    'min_frames': 5,
    'planarity': 0.1,
    'max_thickness': 0.20,
    'max_points_per_patch': 300,
    # optimisation
    'cost': 'point2plane',
    'outer': 10,
    'inner': 12,
    'huber': 0.06,
    'odom_weight': 3.0,
    'device': None,
    # map export
    'rebuild_map': False,
    'map_voxel': 0.0,
    # logging
    'verbose_lm': False,
}

# Scene-scale presets. The voxel must sit inside one flat surface, so it scales
# with the scene; huber tracks the range noise at typical working distance.
PRESETS: dict[str, dict[str, Any]] = {
    'indoor': {
        'voxel_size': 0.2,
        'huber': 0.02,
        'outer': 6,
        'map_voxel': 0.01,
        '_desc': '室内单个房间(场景对角线 < 10 m)',
    },
    'building': {
        'voxel_size': 0.4,
        'huber': 0.04,
        'outer': 8,
        'map_voxel': 0.02,
        '_desc': '楼层/走廊(10~30 m)',
    },
    'outdoor': {
        'voxel_size': 1.0,
        'huber': 0.06,
        'outer': 8,
        'map_voxel': 0.05,
        '_desc': '室外街道尺度(> 50 m)',
    },
}

# One-line rationale per tunable, shown by --show-presets.
PARAM_HELP: dict[str, tuple[str, str]] = {
    'voxel_size': ('体素边长(m)', '最关键。要小到能装进一个平面内部,大到能容纳多帧点'),
    'huber': ('鲁棒核阈值(m)', '约 3 倍测距噪声。太大则不起作用,太小会误剔有效点'),
    'outer': ('外层迭代次数', '每轮重算平面。收敛后再加无意义'),
    'map_voxel': ('导出地图降采样(m)', '按场景精度取,室内比室外细'),
    'min_points': ('体素最少点数', '点太少时平面拟合不可靠'),
    'min_frames': ('体素最少覆盖帧数', '单帧看到的体素提供不了位姿约束'),
    'planarity': ('平面性阈值', 'lambda0/lambda1,越小越严'),
    'max_thickness': ('平面最大厚度(m)', '剔除树叶、电线这类形状像平面但很厚的结构'),
    'max_points_per_patch': ('单块最多点数', '防止个别密集体素主导整个代价'),
    'min_range': ('最近距离(m)', '丢掉打到操作者身上的点'),
    'odom_weight': ('里程计先验权重', '约束几何支撑弱的帧。超过 10 会冻结轨迹'),
    'cost': ('残差形式', 'point2plane 稀疏快;lambda_min 是严格 BALM 但慢'),
    'inner': ('每轮 LM 步数', '内层优化步数'),
    'every': ('抽帧间隔', '调参时设 10 可快速试'),
}


def _w(text: str) -> int:
    """Display width, counting CJK characters as two terminal columns."""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1
               for c in text)


def _pad(text: str, width: int) -> str:
    """Left-align to a display width (str.ljust miscounts CJK)."""
    return text + ' ' * max(width - _w(text), 0)


def _rpad(text: str, width: int) -> str:
    return ' ' * max(width - _w(text), 0) + text


def show_presets() -> None:
    """Print every preset value side by side, with defaults and rationale."""
    names = list(PRESETS)
    keys = ['voxel_size', 'huber', 'outer', 'map_voxel']

    print('\n场景预设对照表')
    print('=' * 76)
    for name in names:
        print(f'  {_pad(name, 10)}{PRESETS[name]["_desc"]}')
    print()

    head = _pad('参数', 24) + _rpad('默认', 8)
    head += ''.join(_rpad(n, 11) for n in names)
    print(head)
    print('-' * 76)
    for k in keys:
        label, _ = PARAM_HELP[k]
        row = _pad(label, 24) + _rpad(str(DEFAULTS[k]), 8)
        for n in names:
            val = PRESETS[n].get(k, DEFAULTS[k])
            mark = ' *' if val != DEFAULTS[k] else '  '
            row += _rpad(f'{val}{mark}', 11)
        print(row)
    print('-' * 76)
    print('  * = 该预设覆盖了默认值\n')

    print('预设不改动的参数(三种场景通用)')
    print('-' * 76)
    for k in ['min_points', 'min_frames', 'planarity', 'max_thickness',
              'max_points_per_patch', 'min_range', 'odom_weight', 'cost',
              'inner', 'every']:
        label, why = PARAM_HELP[k]
        print(f'  {_pad(label, 24)}{_pad(str(DEFAULTS[k]), 14)}{why}')
    print()

    print('各参数为什么这样取')
    print('-' * 76)
    for k in keys:
        label, why = PARAM_HELP[k]
        print(f'  {label}: {why}')
    print()
    print('  实测(室内 717 帧,4296 个位姿自由度):体素 0.2 m 得到 3252 个平面块')
    print('  (比值 0.76,优化 -27.8%);体素 1.0 m 只剩 161 个(比值 0.04),欠定')
    print('  27 倍,优化失败(厚度反而 +2.1%,轨迹被拉长 4%)。')
    print()


class Config:
    """Resolved settings, accessible as attributes (cfg.voxel_size)."""

    def __init__(self, values: dict[str, Any]):
        self._values = values
        for key, val in values.items():
            setattr(self, key, val)

    def __repr__(self) -> str:
        return f'Config({self._values})'

    def describe(self, keys: list[str]) -> str:
        return '  '.join(f'{k}={getattr(self, k)}' for k in keys)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a config file, rejecting unknown keys."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f'找不到配置文件: {path}')
    with open(path, encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f'{path}: 配置文件顶层必须是键值对')

    # Accept 'preset' here so a file can build on a preset.
    allowed = set(DEFAULTS) | {'preset'}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise SystemExit(
            f'{path}: 无法识别的配置项 {unknown}\n'
            f'可用配置项: {sorted(DEFAULTS)}')
    return raw


def resolve(cli: dict[str, Any], config_path: str | Path | None,
            preset: str | None) -> tuple[Config, list[str]]:
    """Merge defaults, preset, YAML, and explicit CLI flags.

    `cli` should contain only flags the user actually passed (others None), so
    that defaults are not mistaken for overrides.
    Returns the config and a human-readable list of where values came from.
    """
    values = dict(DEFAULTS)
    origin: list[str] = []

    file_cfg = load_yaml(config_path) if config_path else {}

    # A preset named on the command line wins over one named in the file.
    name = preset or file_cfg.get('preset')
    if name:
        if name not in PRESETS:
            raise SystemExit(
                f'未知预设 "{name}"。可用: {", ".join(PRESETS)}')
        chosen = {k: v for k, v in PRESETS[name].items()
                  if not k.startswith('_')}
        values.update(chosen)
        origin.append(f'预设 {name}({PRESETS[name]["_desc"]})')

    file_rest = {k: v for k, v in file_cfg.items() if k != 'preset'}
    if file_rest:
        values.update(file_rest)
        origin.append(f'配置文件 {config_path}')

    explicit = {k: v for k, v in cli.items() if v is not None}
    if explicit:
        values.update(explicit)
        origin.append(f'命令行 ({", ".join(sorted(explicit))})')

    # YAML gives plain strings; argparse gives Path. Normalise so callers can
    # rely on Path semantics regardless of where the value came from.
    for key in ('result_dir', 'out_dir'):
        if values.get(key) is not None:
            values[key] = Path(values[key]).expanduser()

    return Config(values), origin
