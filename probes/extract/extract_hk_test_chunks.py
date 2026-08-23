# extract_hk_test_chunks.py - 抽取 N3 测试集的 3 个 processed 分块
# 范围：仅本地官方数据集 SHARD_0000.tar.gz，抽取 3 个指定分块到 hk_chunks\
import tarfile, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # probes/extract -> 仓库根
TAR = os.path.join(ROOT, "shards", "actions", "SHARD_0000.tar.gz")
OUT_DIR = os.path.join(ROOT, "shards", "hk_chunks")
os.makedirs(OUT_DIR, exist_ok=True)

# 目标分块 -> 输出文件名（与实际评测测试集一致：v946202192 的 chunk_0003/0065/0096）
WANT = {
    "v946202192_chunk_0003": "test_v946202192_chunk_0003_actions_processed.parquet",
    "v946202192_chunk_0065": "test_v946202192_chunk_0065_actions_processed.parquet",
    "v946202192_chunk_0096": "test_v946202192_chunk_0096_actions_processed.parquet",
}

tf = tarfile.open(TAR, "r|gz")
cur_dir = None
cur_processed = None
cur_meta = None
saved = 0

for m in tf:
    name = m.name
    d = name.rsplit("/", 1)[0] if "/" in name else ""
    base = name.rsplit("/", 1)[-1]
    if d != cur_dir:
        cur_dir = d
        cur_processed = None
        cur_meta = None
    if base == "actions_processed.parquet":
        try:
            cur_processed = tf.extractfile(m).read()
        except Exception:
            cur_processed = None
        continue
    if base == "metadata.json":
        try:
            cur_meta = json.loads(tf.extractfile(m).read())
        except Exception:
            cur_meta = None
        if cur_meta is None or cur_processed is None:
            continue
        if cur_meta.get("game") != "hollow_knight":
            cur_processed = None
            cur_meta = None
            continue
        chunk = d.rsplit("/", 1)[-1]
        if chunk in WANT:
            local = os.path.join(OUT_DIR, WANT[chunk])
            open(local, "wb").write(cur_processed)
            # 顺带保存 metadata.json，供步骤 3 下载视频自动读取参数
            meta_local = os.path.join(OUT_DIR, f"test_{chunk}_metadata.json")
            with open(meta_local, "w", encoding="utf-8") as f:
                json.dump(cur_meta, f, ensure_ascii=False, indent=2)
            ov = cur_meta["original_video"]
            print(f"抽出: {local} ({len(cur_processed)} bytes) | "
                  f"url={ov['url']} | 帧 {ov['start_frame']}-{ov['end_frame']}", flush=True)
            print(f"  metadata: {meta_local}", flush=True)
            saved += 1
            if saved >= len(WANT):
                break
        cur_processed = None
        cur_meta = None
tf.close()
print(f"\n完成: 抽 {saved}/{len(WANT)} 个测试分块")
