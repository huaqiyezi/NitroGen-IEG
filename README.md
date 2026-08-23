# NitroGen-IEG · 通用游戏智能体评测（Hollow Knight）

NitroGen 视觉→手柄动作基础模型的 zero-shot 评测工程。目标是复现"取原视频片段 → 切帧 → 推理 → 与真值对齐 → 算指标"整条链路，并评估模型在《空洞骑士》(Hollow Knight) 上的表现。

> 仓库只含**代码、脚本与文档**。数据集、视频、切帧画面、ffmpeg 等大文件按 `.gitignore` 不入库，需按本说明自行准备。

---

## 一、环境

| 项 | 版本/要求 |
|---|---|
| 系统 | Windows |
| Python | **3.12**（唯一版本，推理与评测/分析共用；torch cu128） |
| 依赖 | pandas、pyarrow、numpy、matplotlib、requests、pyzmq、opencv-python、torch（安装命令见下方） |
| 外部工具 | **ffmpeg 9.0.1**（含 ffprobe，切帧/校验）、**yt-dlp**（下载视频） |
| 代理 | 下载 twitch 视频需代理；`download_videos.py` 自动探测代理地址，`--proxy` 可手动覆盖 |

**依赖安装命令**（Python 3.12）：

```bash
# 普通依赖（评测/分析/可视化；用国内 PyPI 镜像加速，如清华）
pip install pandas pyarrow matplotlib requests numpy pyzmq opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple

# torch（CUDA 版本需匹配显卡架构，本仓库验证用 cu128；从对应 CUDA 源安装，勿加 PyPI 镜像）
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

> **不同电脑适配**：代码本身兼容 Python 3.10–3.12，本仓库统一使用 Python 3.12（推理与评测/分析共用）。torch 的 CUDA 版本需匹配显卡架构，常见选择：Blackwell 新架构（如 RTX 50 系）用 cu128；30/40 系用 cu121/cu124；无 GPU 或 A 卡用 CPU 版（`pip install torch`）。不确定时用 `nvidia-smi` 查看驱动支持的 CUDA 版本，选 ≤ 该版本的 wheel。

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
│   ├── eval/             #   评测结果 csv（raw_predictions、metrics_by_group/overall）
│   └── out/              #   分析/可视化输出（对比图、模块图、漏按排行等）
```
> 另含 `docs/`（脚本说明）等文档目录。

> 模型仓库（`serve.py`、`ng.pt` 权重）在独立的 `NitroGen\` 目录（与本仓库同级），不在本仓库。获取方式：
> - 仓库：`git clone https://github.com/MineDojo/NitroGen.git`
> - 权重 `ng.pt`：从 HuggingFace 下载 `nvidia/NitroGen`（`hf download nvidia/NitroGen ng.pt`，放入 `checkpoints\`）
> - 还需 SigLIP 视觉编码器权重（首次运行自动从 HuggingFace 拉取，需能访问 HF）
>
> 本仓库**不执行 `pip install -e .`**：它会按 `pyproject.toml` 装全量依赖（含 `play` 一组的实机游玩库：`dxcam`、`vgamepad`、`xspeedhack`、`pywin32` 等），而本课题只用 `serve.py` 做推理、不跑实机游玩，这些依赖用不到且易引发版本冲突。只需进入模型仓库目录直接运行 `serve.py`（或设置 `PYTHONPATH` 指向模型仓库目录）即可。

## 三、按流程复现

### 第 1 步：准备测试标注（若尚未抽取）

```bash
python probes\extract\download_shard0.py          # 下载数据集分片 SHARD_0000.tar.gz
python probes\extract\extract_hk_test_chunks.py   # 抽测试分块 parquet + metadata.json
```

### 第 2 步：下载测试视频片段（需代理）

```bash
# 开启代理后直接运行：自动读步骤 1 的 metadata 取参数 + 自动探测代理
python probes\extract\download_videos.py

# 若自动探测不到代理，可手动指定
python probes\extract\download_videos.py --proxy http://127.0.0.1:7890
```

> 下载 twitch 视频需先开启代理（脚本自动探测，探测不到用 `--proxy` 手动指定）。下载后用 ffprobe 校验帧数/色域。

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
# 用 HF 镜像源：部分环境直连 huggingface.co 会证书校验失败，镜像源可规避且下载更稳；缓存会自动下载到默认位置
$env:HF_ENDPOINT="https://hf-mirror.com"
py -3.12 scripts\serve.py checkpoints\ng.pt --port 5555
```

- 需权重 `checkpoints\ng.pt`；
- **镜像源会自动下载 SigLIP 视觉编码器缓存**（约 3.5GB，首次运行需几分钟）；
- 可选：若不想占系统盘空间，可另设缓存位置 `$env:HF_HOME="<你的缓存目录>"`（如 `D:\hf_cache`）；
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

> 脚本内数据路径已用相对路径（基于脚本位置自动计算仓库根），拷到任意位置即可运行，无需改路径。

## 五、停止

- 推理服务：在运行 `serve.py` 的终端按 `Ctrl+C`，显示 `Shutting down server...` 即正常退出。

## 六、忽略规则

见 `.gitignore`。忽略的大致类别：数据集包、视频、切帧画面、ffmpeg、缓存、日志、内部记录——均可按本说明重新获得或为过程文件。

## 七、脚本说明

各脚本的职责与用法详见 [docs/脚本说明.md](docs/脚本说明.md)。
