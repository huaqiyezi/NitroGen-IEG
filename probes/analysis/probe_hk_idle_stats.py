# probe_hk_idle_stats.py - 只读统计：候选视频的 HK processed 分块空闲率等参数
# 范围：仅本地官方数据集 SHARD_0000.tar.gz，只读，不落盘
# 输出：每块的空闲率/有效帧数/按键活跃度/摇杆活跃度 -> CSV
import tarfile, json, time, io, csv

import pandas as pd

TAR = r"D:\Projects\NitroGen-IEG\shards\actions\SHARD_0000.tar.gz"
WANT_VIDEOS = {"v2135647078", "v946202192"}

# 17 按钮列
BTN_COLS = ["dpad_down","dpad_left","dpad_right","dpad_up",
            "left_shoulder","left_thumb","left_trigger",
            "right_shoulder","right_thumb","right_trigger",
            "south","west","east","north","back","start","guide"]

t0 = time.time()
tf = tarfile.open(TAR, "r|gz")
cur_dir = None
cur_parquet = None
cur_meta = None
rows = []
n_blocks = 0

for m in tf:
    name = m.name
    d = name.rsplit("/", 1)[0] if "/" in name else ""
    base = name.rsplit("/", 1)[-1]
    if d != cur_dir:
        cur_dir = d
        cur_parquet = None
        cur_meta = None
    if base == "actions_processed.parquet":
        try:
            cur_parquet = tf.extractfile(m).read()
        except Exception:
            cur_parquet = None
        continue
    if base == "metadata.json":
        try:
            cur_meta = json.loads(tf.extractfile(m).read())
        except Exception:
            cur_meta = None
        if cur_meta is None or cur_parquet is None:
            continue
        if cur_meta.get("game") != "hollow_knight":
            cur_parquet = None
            cur_meta = None
            continue
        vid = cur_meta.get("original_video", {}).get("video_id")
        if vid not in WANT_VIDEOS:
            cur_parquet = None
            cur_meta = None
            continue
        chunk = d.rsplit("/", 1)[-1]
        try:
            df = pd.read_parquet(io.BytesIO(cur_parquet))
        except Exception as e:
            print(f"[解析失败] {chunk}: {e}", flush=True)
            cur_parquet = None
            cur_meta = None
            continue
        n_blocks += 1
        # 构建 17 按钮矩阵
        btns = df[BTN_COLS].fillna(False).astype(bool)
        any_btn = btns.any(axis=1).values
        # 摇杆近零判断
        jl = df["j_left"].apply(lambda v: (float(v[0]), float(v[1])) if v is not None else (0.0, 0.0))
        jr = df["j_right"].apply(lambda v: (float(v[0]), float(v[1])) if v is not None else (0.0, 0.0))
        jlx = jl.apply(lambda p: p[0]); jly = jl.apply(lambda p: p[1])
        jrx = jr.apply(lambda p: p[0]); jry = jr.apply(lambda p: p[1])
        joy_act = (jlx.abs() + jly.abs() + jrx.abs() + jry.abs()) > 0.01
        idle = (~any_btn) & (~joy_act.values)
        idle_rate = float(idle.mean())
        # 按键活跃度：每帧按下的按钮数均值
        btn_press_mean = float(btns.sum(axis=1).mean())
        # 摇杆活跃度：4 维绝对值均值
        joy_abs_mean = float((jlx.abs() + jly.abs() + jrx.abs() + jry.abs()).mean())
        rows.append({
            "chunk": chunk, "video": vid,
            "idle_rate": round(idle_rate, 4),
            "valid_frames": int((~idle).sum()),
            "btn_press_mean": round(btn_press_mean, 3),
            "joy_abs_mean": round(joy_abs_mean, 4),
        })
        cur_parquet = None
        cur_meta = None
        if n_blocks % 50 == 0:
            print(f"  已统计 {n_blocks} 块, 用时 {time.time()-t0:.0f}s", flush=True)
tf.close()

# 写 CSV
OUT = r"D:\Projects\NitroGen-IEG\shards\hk_chunks\hk_idle_stats.csv"
df_out = pd.DataFrame(rows)
df_out.to_csv(OUT, index=False, encoding="utf-8-sig")

print(f"\n=== 统计完成（{len(rows)} 块, 用时 {time.time()-t0:.0f}s）===")
print(f"汇总已写: {OUT}")
print(f"\n空闲率最低的前 20 块：")
low = df_out.sort_values("idle_rate").head(20)
for _, r in low.iterrows():
    print(f"  {r['chunk']}: 空闲率={r['idle_rate']*100:.1f}% 有效帧={r['valid_frames']} 按键均值={r['btn_press_mean']} 摇杆均值={r['joy_abs_mean']}")
