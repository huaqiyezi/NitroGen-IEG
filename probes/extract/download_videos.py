# download_videos.py - 用 yt-dlp 按时间区间下载测试视频片段（走代理）
# 输入：--video-id (twitch 视频 ID) + 起始/结束秒 或 metadata 帧号
# 输出：shards/videos/test_<video_id>_chunk_<n>.mp4
# 用法：
#   python download_videos.py --video-id v946202192 --start 60 --end 80 \
#       --out test_v946202192_chunk_0003.mp4 [--proxy http://127.0.0.1:7890]
# 前置：本机已安装 yt-dlp；下载 twitch 视频需先开启代理（Clash 等，默认 127.0.0.1:7890）。
# 说明：twitch 视频可能无法直接下载（登录/风控），此脚本提供走代理的标准做法。
import argparse, subprocess, sys, os

def main():
    ap = argparse.ArgumentParser(description="用 yt-dlp 下载 twitch 测试视频片段（走代理）")
    ap.add_argument("--video-id", required=True, help="twitch 视频 ID，如 v946202192")
    ap.add_argument("--start", type=float, required=True, help="起始秒")
    ap.add_argument("--end", type=float, required=True, help="结束秒")
    ap.add_argument("--out", required=True, help="输出文件名，如 test_v946202192_chunk_0003.mp4")
    ap.add_argument("--proxy", default="http://127.0.0.1:7890", help="代理地址（默认走 Clash 127.0.0.1:7890）")
    args = ap.parse_args()

    out_dir = r"D:\Projects\NitroGen-IEG\shards\videos"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.out)
    url = f"https://www.twitch.tv/videos/{args.video_id}"

    cmd = [
        "yt-dlp",
        "--proxy", args.proxy,
        "--download-sections", f"*{args.start}-{args.end}",
        "-f", "best[height<=1080]",
        "-o", out_path,
        url,
    ]
    print("运行:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("\n[失败] yt-dlp 返回非零。可能原因：", flush=True)
        print("  1) 代理未开启或端口不对（默认 127.0.0.1:7890，可用 --proxy 改）", flush=True)
        print("  2) twitch 视频需要登录或触发风控（--cookies 指定浏览器 cookie）", flush=True)
        print("  3) yt-dlp 未安装或过旧（pip install -U yt-dlp）", flush=True)
        sys.exit(1)

    print(f"\n完成: {out_path}")
    print("下一步: 用 ffprobe 校验帧数（60fps × 片段秒数 ≈ 帧数）与色域（bt709）")

if __name__ == "__main__":
    main()
