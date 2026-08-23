# NitroGen-IEG · 通用游戏智能体评测（Hollow Knight）

NitroGen 视觉→手柄动作基础模型的 zero-shot 评测工程。目标是复现"取原视频片段 → 切帧 → 推理 → 与真值对齐 → 算指标"整条链路，并评估模型在《空洞骑士》(Hollow Knight) 上的表现。

**获取本仓库**（需先安装 git）：

```bash
git clone https://github.com/huaqiyezi/NitroGen-IEG.git
cd NitroGen-IEG
```

> 仓库只含**代码、脚本与文档**。数据集、视频、切帧画面、ffmpeg 等大文件按 `.gitignore` 不入库，需按本说明自行准备。

---

## 一、环境

| 项 | 版本/要求 |
|---|---|
| 系统 | Windows |
| Python | **必须 Python 3.12**（唯一版本，推理与评测/分析共用；旧版本如 3.8 无法运行） |
| 依赖 | pandas、pyarrow、numpy、matplotlib、requests、pyzmq、opencv-python、torch、torchvision（安装命令见下方） |
| 外部工具 | **git**（clone 模型仓库）、**ffmpeg 9.0.1**（含 ffprobe，切帧/校验）、**yt-dlp**（下载视频） |
| 代理 | 下载 twitch 视频需代理；`download_videos.py` 自动探测代理地址，`--proxy` 可手动覆盖 |

**确认/安装 Python 3.12**：

```bash
# 查看本机已装的 Python 版本（Windows 建议用 py launcher）
py -0

# 若无 3.12，前往 python.org/downloads 下载 Python 3.12 安装包，
# 安装时务必勾选 "Add python.exe to PATH"。装好后用 py -3.12 指定。
py -3.12 --version
```

> 注意：命令统一用 `py -3.12`（不要用默认 `python`，它可能指向旧版本如 3.8，会导致依赖安装失败或脚本语法错误）。

**外部工具安装**：

> winget 仅为便捷方式，**非必需**；若 winget 不可用，一律用官网/pip 手动安装。

```bash
# git（clone 模型仓库用）：winget 安装；若 winget 失败（常见于权限/网络），
# 改用官网 https://git-scm.com/download/win 下载 64-bit 安装包手动安装
winget install Git.Git

# ffmpeg（必需：下载视频片段裁剪 + 切帧/校验都要用）：官网 https://www.gyan.dev/ffmpeg/builds/ 下载 release-essentials
# 解压到某目录，然后用一行命令把 bin 加入用户 PATH（PowerShell，把 <你的ffmpeg目录>\bin 换成实际路径）：
[Environment]::SetEnvironmentVariable("Path", "$env:Path;<你的ffmpeg目录>\bin", "User")
#   （winget 若不可靠，一律用官网手动装；ffmpeg 是绿色软件，无需安装器）
# winget install Gyan.FFmpeg

# yt-dlp（下载视频用，需先装好 Python 3.12）：
py -3.12 -m pip install yt-dlp
```

> 装完 git/ffmpeg 后需**新开终端**使其生效（PATH 刷新）；若命令仍找不到，检查 PATH 是否已加入对应 bin 目录。

**装好后验证**（需新开终端）：

```bash
git --version
ffmpeg -version
yt-dlp --version
```

> 三条命令都能显示版本号即为装好；若某条报"无法识别"，新开终端重试，仍不行则检查该工具是否装对/ PATH 是否正确。

**依赖安装命令**（Python 3.12，用 `py -3.12 -m pip`）：

```bash
# 0) 可选：升级 pip（旧版可能装不上新依赖）
py -3.12 -m pip install --upgrade pip

# 1) 普通依赖（评测/分析/可视化；用国内 PyPI 镜像加速，如清华）
py -3.12 -m pip install pandas pyarrow matplotlib requests numpy pyzmq opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2) torch：装 CUDA 版（把 cu128 换成匹配你显卡的版本，如 cu121/cu124）
py -3.12 -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

> **不同电脑适配**：统一使用 Python 3.12（推理与评测/分析共用）。torch 的 CUDA 版本需匹配显卡架构：Blackwell 新架构（如 RTX 50 系）用 cu128；30/40 系用 cu121/cu124。不确定时用 `nvidia-smi` 查看驱动支持的 CUDA 版本。

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

> 模型仓库（`serve.py`、`ng.pt` 权重）在独立目录（如 `D:\NitroGen`），不在本仓库，路径自定。获取方式（需先安装 git）：
> - 仓库：`git clone https://github.com/MineDojo/NitroGen.git`
> - 权重 `ng.pt`：从 HuggingFace 下载 `nvidia/NitroGen`（`hf download nvidia/NitroGen ng.pt`，放入 `checkpoints\`）
> - 还需 SigLIP 视觉编码器权重（首次运行自动从 HuggingFace 拉取，需能访问 HF）
>
> 本仓库**不执行 `pip install -e .`**：它会按 `pyproject.toml` 装全量依赖（含 `play` 一组的实机游玩库：`dxcam`、`vgamepad`、`xspeedhack`、`pywin32` 等），而本课题只用 `serve.py` 做推理、不跑实机游玩，这些依赖用不到且易引发版本冲突。只需进入模型仓库目录直接运行 `serve.py`（或设置 `PYTHONPATH` 指向模型仓库目录）即可。

## 三、按流程复现

> **重要**：以下所有命令默认在**仓库根目录**（clone 出来的 `NitroGen-IEG` 文件夹）执行。若提示 `No such file or directory`，说明当前不在仓库目录，请先：
> ```powershell
> cd <你的NitroGen-IEG路径>   # 把 <你的NitroGen-IEG路径> 换成 clone 出来的文件夹位置
> ```
> 第 5 步会按你填的 `$model_repo` 路径切到模型仓库目录，跑完第 5 步后**第 6、7 步需先回到仓库根目录**再执行。

### 第 1 步：准备测试标注（若尚未抽取）

```bash
py -3.12 probes\extract\download_shard0.py          # 下载数据集分片 SHARD_0000.tar.gz
py -3.12 probes\extract\extract_hk_test_chunks.py   # 抽测试分块 parquet + metadata.json
```

### 第 2 步：下载测试视频片段（需代理）

```bash
py -3.12 probes\extract\download_videos.py                            # 自动读 metadata 取参数 + 自动探测代理
py -3.12 probes\extract\download_videos.py --proxy http://127.0.0.1:7890   # 探测不到代理时手动指定
py -3.12 probes\extract\download_videos.py --no-tls-verify             # 报证书校验失败时用（见下方说明）
```

> 下载 twitch 视频需先开启代理（脚本自动探测，探测不到用 `--proxy` 手动指定）。下载后用 ffprobe 校验帧数/色域。
> **`--no-tls-verify` 模式**：会先完整下载每个视频（约 2GB，3 个共需约 6GB 磁盘空间，耗时较长），再用 ffmpeg 对本地文件裁剪出目标片段，裁剪后自动删除完整临时文件。

### 第 3 步：生成选帧清单

```bash
py -3.12 probes\eval\plan_test_frames.py   # 输出 test_frames_plan.csv（3 组 × 200 帧）
```

### 第 4 步：切帧

```bash
py -3.12 probes\extract\extract_frames.py  # 按帧号从视频切出 PNG
```

### 第 5 步：启动推理服务（模型仓库目录）

> **路径**：把下面 `$model_repo = "D:\NitroGen"` 换成你的实际路径（只需改这一处，后续命令自动跟随）；填错会立刻报错停下。
>
> **补丁命令**（首次启动前执行一次）：修复官方 `nitrogen.py` 与 transformers 5.x 的兼容问题（第 186 行改为 `hasattr` 写法）。幂等，重复运行无副作用。
>
> **HF 镜像源**：`$env:HF_ENDPOINT` 走 `hf-mirror.com`，规避部分环境直连 huggingface.co 的证书校验失败，下载更稳。

```powershell
$model_repo = "D:\NitroGen"   # ← 改成你的实际路径
Set-Location $model_repo -ErrorAction Stop
py -3.12 -c "import io;f='nitrogen/flow_matching_transformer/nitrogen.py';s=io.open(f,encoding='utf-8').read();io.open(f,'w',encoding='utf-8',newline='').write(s if 'if hasattr(model, \x22vision_model\x22)' in s else s.replace('self.vision_encoder = model.vision_model','self.vision_encoder = model.vision_model if hasattr(model, \x22vision_model\x22) else model'))"
$env:HF_ENDPOINT="https://hf-mirror.com"
py -3.12 scripts\serve.py checkpoints\ng.pt --port 5555
```

- 需权重 `checkpoints\ng.pt`；
- **镜像源会自动下载 SigLIP 视觉编码器缓存**（约 3.5GB，首次运行需几分钟）；
- 可选：若不想占系统盘空间，可另设缓存位置 `$env:HF_HOME="<你的缓存目录>"`（如 `D:\hf_cache`）；
- 看到 `Server running on port 5555` 即成功。

### 第 6 步：跑评测（需第 5 步服务在跑）

```bash
# 先回到仓库根目录（第 5 步在模型仓库目录）
cd <你的NitroGen-IEG路径>   # 换成 clone 出来的文件夹位置
py -3.12 probes\eval\run_eval.py           # 输出 raw_predictions.csv + 指标表
```

### 第 7 步：分析/出图（可选）

```bash
# 仍在仓库根目录
py -3.12 probes\analysis\analyze_button_mistakes.py  # 漏按/多按分析
py -3.12 probes\visual\make_compare_material.py      # 生成对比表/对比图
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
