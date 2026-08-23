# extract_frames.py - 切帧：按帧号清单分批从测试视频片段抽取画面帧 PNG
# 输入：test_frames_plan.csv（块/帧号清单）
# 输出：shards/frames/<block>/<absolute_frame>.png
# 说明：块内帧号 = 视频帧号（下载片段从块起始帧开始）；分批 select 避免过滤器内存溢出
import subprocess, os, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # probes/extract -> 仓库根
FFMPEG = os.path.join(ROOT, "tools", "ffmpeg-9.0.1-essentials_build", "bin", "ffmpeg.exe")
VIDEO_DIR = os.path.join(ROOT, "shards", "videos")
OUT_DIR = os.path.join(ROOT, "shards", "frames")
PLAN = os.path.join(ROOT, "shards", "hk_chunks", "test_frames_plan.csv")
BATCH = 25  # 每批帧号数

df = pd.read_csv(PLAN)
total = 0

for block, g in df.groupby("block"):
    # 视频文件名 = {block}.mp4（与步骤 3 download_videos.py 输出一致），自动跟随分块
    video = os.path.join(VIDEO_DIR, f"{block}.mp4")
    if not os.path.isfile(video):
        print(f"[跳过] 缺少视频: {video}（请先运行 download_videos.py）", flush=True)
        continue
    outdir = os.path.join(OUT_DIR, block)
    os.makedirs(outdir, exist_ok=True)

    g = g.sort_values("frame_in_chunk")
    pairs = list(zip(g["frame_in_chunk"].tolist(), g["absolute_frame"].tolist()))
    n_frames = len(pairs)

    # 分批
    for bi in range(0, n_frames, BATCH):
        batch = pairs[bi:bi+BATCH]
        frames = [p[0] for p in batch]
        abs_frames = [p[1] for p in batch]
        sel = "+".join(f"between(n\\,{i}\\,{i})" for i in frames)

        tmp_pattern = os.path.join(outdir, "tmp_%d.png")
        cmd = [FFMPEG, "-y", "-i", video, "-vf", f"select='{sel}'",
               "-fps_mode", "vfr", tmp_pattern]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[{block} 批{bi//BATCH}] ffmpeg 出错: {r.stderr[-400:]}", flush=True)
            sys.exit(1)

        # 重命名 tmp_<k>.png -> <absframe>.png
        produced = sorted([f for f in os.listdir(outdir) if f.startswith("tmp_")],
                          key=lambda x: int(x[4:-4]))
        for k, fname in enumerate(produced):
            if k < len(abs_frames):
                dst = os.path.join(outdir, f"{abs_frames[k]}.png")
                os.replace(os.path.join(outdir, fname), dst)
        total += len(produced)
        print(f"[{block}] 批{bi//BATCH+1}/{(n_frames+BATCH-1)//BATCH}: {len(produced)} 帧", flush=True)

    n_out = len([f for f in os.listdir(outdir) if f.endswith(".png") and not f.startswith("tmp_")])
    print(f"[{block}] 完成: {n_out} 帧", flush=True)

print(f"\n全部切帧完成，共 {total} 帧")
