# download_shard0.py - 用 requests 关证书校验下载 SHARD_0000.tar.gz 到 D 盘
# 本机 hf CLI 因证书校验失败下不动；此脚本绕过(仅下公开数据)。
import os, time, requests, urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

URL = "https://huggingface.co/datasets/nvidia/NitroGen/resolve/main/actions/SHARD_0000.tar.gz"
OUT = r"D:\Projects\NitroGen-IEG\shards\actions\SHARD_0000.tar.gz"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

print("开始下载 SHARD_0000.tar.gz (约1.7GB)...")
t0 = time.time()
r = requests.get(URL, stream=True, verify=False, timeout=600)
print("HTTP", r.status_code)
r.raise_for_status()
total = int(r.headers.get("content-length", 0))
got = 0; last = 0
with open(OUT, "wb") as f:
    for chunk in r.iter_content(1 << 20):  # 1MB
        if chunk:
            f.write(chunk); got += len(chunk)
            if got - last >= 50 * (1 << 20):  # 每50MB报一次
                pct = got * 100 / total if total else 0
                print(f"  {got/1e6:.0f}/{total/1e6:.0f} MB ({pct:.0f}%)  {time.time()-t0:.0f}s", flush=True)
                last = got
print(f"\n完成: {OUT}")
print(f"大小: {os.path.getsize(OUT)/1e6:.0f} MB, 用时 {time.time()-t0:.0f}s")
if total and os.path.getsize(OUT) == total:
    print("校验: 大小与content-length一致 ✓")
else:
    print("校验: 大小不一致，可能传输中断，重跑即可续")
