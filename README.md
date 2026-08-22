# NitroGen-IEG · 通用游戏智能体评测（Hollow Knight）

NitroGen 视觉→手柄动作基础模型的 zero-shot 评测工程。目标是复现"取原视频片段 → 切帧 → 推理 → 与真值对齐 → 算指标"整条链路，并评估模型在《空洞骑士》(Hollow Knight) 上的表现。

> 仓库只含**代码、脚本与文档**。数据集、视频、切帧画面、ffmpeg 等大文件按 `.gitignore` 不入库，需按本说明自行准备。

---

## 一、环境

| 项 | 版本/要求 |
|---|---|
| 系统 | Windows |
| Python | **3.10**（唯一版本，评测+推理共用） |
| 依赖 | `pip install pandas pyarrow matplotlib requests numpy zmq opencv-python` 与 torch cu128（`pip install torch --index-url https://download.pytorch.org/whl/cu128`） |
| 外部工具 | **ffmpeg 9.0.1**（含 ffprobe，切帧/校验）、**yt-dlp**（下载视频） |
| 代理 | 下载 twitch 视频需代理，地址由 `download_videos.py` 的 `--proxy` 参数指定 |

> **不同电脑适配**：代码本身兼容 Python 3.10–3.12。关键在 torch 的 CUDA 版本需匹配显卡架构——安装时用对应 index-url（cu128 见上），其余依赖任意 3.10+ 均可 `pip install`。

## 二、目录职责

```
NitroGen-IEG/
├── probes/               # 脚本（按功能分目录）
│   ├── eval/             #   评测主链路（run_eval、plan_test_frames）
│   ├── extract/          #   数据抽取/下载（extract_hk_test_chunks、download_videos、extract_frames、download_shard0）
│   ├── analysis/         #   统计分析（probe_hk_idle_stats、n2_stats_v4、analyze_button_mistakes）
│   └── visual/           #   图表与演示材料（make_compare_material、find_compare_cases、gen_module_diagram）
├── shards/               # 数据区（大文件不入库，见 .gitignore）
│   ├── actions/          #   数据集原始包 SHARD_0000.tar.gz（需自行下载）
│   ├── hk_chunks/        #   抽取出的测试分块标注 parquet、选帧清单 csv
│   ├── videos/           #   测试视频片段 mp4（yt-dlp 下载）
│   ├── frames/           #   切帧画面 PNG（ffmpeg 生成）
│   └── eval/             #   评测结果 csv（raw_predictions、metrics_by_group/overall）
```
> 另含 `docs/`（脚本说明）等文档目录。

> 模型仓库（`serve.py`、`ng.pt` 权重）在独立的 `NitroGen\` 目录（与本仓库同级），不在本仓库。

## 三、按流程复现

### 第 1 步：准备测试标注（若尚未抽取）

```bash
python probes\extract\download_shard0.py          # 下载数据集分片 SHARD_0000.tar.gz
python probes\extract\extract_hk_test_chunks.py   # 从 tar.gz 抽 3 个测试分块 parquet
```

### 第 2 步：下载测试视频片段（走代理）

```bash
python probes\extract\download_videos.py --video-id v946202192 --start 60 --end 80 \
    --out test_v946202192_chunk_0003.mp4
```

> 代理地址用 `--proxy` 指定。下载后用 ffprobe 校验帧数/色域。

### 第 3 步：生成选帧清单

```bash
python probes\eval\plan_test_frames.py   # 输出 test_frames_plan.csv（3 组 × 200 帧）
```

### 第 4 步：切帧

```bash
python probes\extract\extract_frames.py  # 按帧号从视频切出 PNG
```

### 第 5 步：启动推理服务（模型仓库目录）

```bash
# 进入模型仓库目录（与本仓库同级；换成你的实际路径）
cd ../NitroGen
py -3.10 scripts\serve.py checkpoints\ng.pt --port 5555
```

- 需权重 `checkpoints\ng.pt` 与 SigLIP 视觉编码器（`hf_cache\hub\...\model.safetensors`）就绪；
- 看到 `Server running on port 5555` 即成功。

### 第 6 步：跑评测（需第 5 步服务在跑）

```bash
python probes\eval\run_eval.py           # 输出 raw_predictions.csv + 指标表
```

### 第 7 步：分析/出图（可选）

```bash
python probes\analysis\analyze_button_mistakes.py  # 漏按/多按分析
python probes\visual\make_compare_material.py      # 生成对比表/对比图
```

## 四、环境变量

无必须的环境变量。代理地址在 `download_videos.py` 里用 `--proxy` 参数控制。

> 脚本内数据路径为绝对路径，若目录与本机不同，请将脚本开头的路径常量改为你的实际路径。

## 五、停止

- 推理服务：在运行 `serve.py` 的终端按 `Ctrl+C`，显示 `Shutting down server...` 即正常退出。

## 六、忽略规则

见 `.gitignore`。忽略的大致类别：数据集包、视频、切帧画面、ffmpeg、缓存、日志、内部记录——均可按本说明重新获得或为过程文件。

## 七、脚本说明

各脚本的职责与用法详见 [docs/脚本说明.md](docs/脚本说明.md)。
