# ellipselio_ba — EllipseLIO 离线 BA 优化

对 EllipseLIO 的结果做离线 BA, 通过读取 <path-to-ellipselio-results-path> ellipselio 的结果保存目录(雷达局部系的逐帧点云 + TUM 轨迹), 调整每帧位姿使多帧共同观测到的平面变得一致,输出优化后的轨迹和重建地图。

代码放在 ellipselio 仓库里,但是一个**独立的 uv 项目** —— 靠 `COLCON_IGNORE` 让 colcon 跳过它,而且完全不 import 任何 ROS 模块。

## 环境

```sh
uv sync
export CUDA_HOME=/usr/local/cuda-12.1 PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="7.5"        # RTX 2080 Ti = sm_75,只编这一个架构快很多
uv pip install --no-build-isolation -e <path-to-bae-repo>
```

注意:PyPI 上的 pypose 0.9.5 要求 `bae==0.2` 严格相等,会拒绝 0.2.5。
`pyproject.toml` 里钉的是 git 版 pypose,放宽了这个检查。

## 用法

选一个匹配场景的预设,通常这就是唯一需要调的东西:

```sh
cd <path-to-ellipselio-ba>

# 方式一:配置文件(推荐)
PYTHONPATH=. uv run --no-sync python -m ellipselio_ba.run --config configs/indoor.yaml

# 方式二:命令行给预设和路径
PYTHONPATH=. uv run --no-sync python -m ellipselio_ba.run \
  --preset indoor --result-dir /media/.../lio_res/ellipselio --rebuild-map
```

| 预设 | 场景 | voxel_size | huber | outer | map_voxel |
|---|---|---|---|---|---|
| `indoor` | 单个房间,< 10 m | 0.2 | 0.02 | 6 | 0.01 |
| `building` | 楼层/走廊,10~30 m | 0.4 | 0.04 | 8 | 0.02 |
| `outdoor` | 室外街道尺度,> 50 m | 1.0 | 0.06 | 8 | 0.05 |

查看所有预设的完整参数、默认值和取值理由:

```sh
PYTHONPATH=. uv run --no-sync python -m ellipselio_ba.run --show-presets
```

`configs/*.yaml` 里也把预设展开成了注释行,想改某一项取消注释即可,不用查文档。

参数优先级,后者覆盖前者:

```
内置默认值  <  preset  <  YAML 配置文件  <  命令行参数
```

所以 `--config configs/indoor.yaml --outer 30` = 用配置文件的设置但跑 30 轮。
每次运行会打印参数来源;`--print-config` 只打印最终参数不运行。
YAML 里拼错的配置项会**报错**,不会被静默忽略。

`configs/full_example.yaml` 列出了全部选项及默认值。

### 残差形式

`--cost point2plane`(默认)和 `--cost lambda_min`(严格的 BALM 代价)两种。
**默认值不需要改** —— 实测 lambda_min 慢 10~50 倍且效果更差,只作对比用。

## 输出

写到 `<result-dir>/../ellipselio_ba/`:

- `lidar_pose_tum_ba.txt` — 优化后位姿(TUM 格式)
- `map_before.pcd` / `map_after.pcd` — 优化前后的拼接地图(需 `rebuild_map: true`)

调参时想快速试:加 `--every 10` 抽帧。





  
