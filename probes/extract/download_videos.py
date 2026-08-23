# download_videos.py - 用 yt-dlp 下载步骤 2 抽出的测试视频片段（走代理）
# 输入：自动读取步骤 2 抽出的分块 metadata.json，无需手动填参数
# 输出：shards/videos/test_<chunk>.mp4
# 用法：python download_videos.py [--proxy <地址>]
# 前置：本机已安装 yt-dlp；下载 twitch 视频需先开启代理（脚本自动探测，--proxy 可手动覆盖）。
# 说明：分块由步骤 2 的 metadata.json 自动决定，改步骤 2 换分块时自动跟随；用户零手动参数。
import argparse, subprocess, sys, os, json, re, socket, winreg, glob

# 分块：自动扫描步骤 2 抽出的 metadata.json（chunk = 文件名，video-id/区间从 metadata 取）
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # probes/extract -> 仓库根
HK_DIR = os.path.join(ROOT, "shards", "hk_chunks")
OUT_DIR = os.path.join(ROOT, "shards", "videos")
FPS = 60

def detect_proxy():
    """自动探测本机代理地址。优先级：环境变量 > Windows 系统代理 > 常见默认端口。找不到返回 None。"""
    # 1) 环境变量
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        v = os.environ.get(var, "").strip()
        if v:
            return v
    # 2) Windows 注册表系统代理
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings") as key:
            enabled, _ = winreg.QueryValueEx(key, "ProxyEnable")
            server, _ = winreg.QueryValueEx(key, "ProxyServer")
            if enabled and server:
                return f"http://{server}" if not server.startswith("http") else server
    except Exception:
        pass
    # 3) 常见本地代理端口（试连）
    for port in (7890, 7897, 10809, 1080, 8118):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return f"http://127.0.0.1:{port}"
        except Exception:
            continue
    return None

def main():
    ap = argparse.ArgumentParser(description="自动下载评测所需的 3 个测试视频片段")
    ap.add_argument("--proxy", default=None, help="手动指定代理地址（默认自动探测）")
    args = ap.parse_args()

    proxy = args.proxy or detect_proxy()
    if not proxy:
        print("[提示] 未自动探测到代理。请先开启代理，或手动指定 --proxy <地址>。", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    meta_files = sorted(glob.glob(os.path.join(HK_DIR, "test_*_metadata.json")))
    if not meta_files:
        print("[提示] 未找到步骤 2 抽出的 metadata.json，请先运行 extract_hk_test_chunks.py。", flush=True)
        return
    for meta_path in meta_files:
        chunk = os.path.basename(meta_path).replace("_metadata.json", "")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        ov = meta["original_video"]
        url = ov["url"]
        m = re.search(r"/videos/(\d+)", url)
        if not m:
            print(f"[失败] 无法从 url 提取 video-id: {url}", flush=True)
            continue
        video_id = m.group(1)
        # 优先用 metadata 直接给出的秒级区间（start_time/end_time），最可靠
        start_s = ov.get("start_time", ov["start_frame"] / FPS)
        end_s = ov.get("end_time", ov["end_frame"] / FPS)
        out_path = os.path.join(OUT_DIR, f"{chunk}.mp4")

        print(f"\n[{chunk}] 下载 {video_id} 片段 {start_s:.1f}s-{end_s:.1f}s (代理: {proxy or '无'})", flush=True)
        cmd = ["yt-dlp",
               "--download-sections", f"*{start_s}-{end_s}",
               "-f", "best[height<=1080]", "-o", out_path, url]
        if proxy:
            cmd = ["yt-dlp", "--proxy", proxy] + cmd[1:]
        print("运行:", " ".join(cmd), flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"  [失败] 下载失败。可能原因：代理未开/端口不对、twitch 风控、yt-dlp 过旧", flush=True)
            continue
        print(f"  完成: {out_path}", flush=True)

    print("\n全部完成。下一步: 用 ffprobe 校验帧数（60fps × 片段秒数）与色域（bt709）")

if __name__ == "__main__":
    main()
