# plan_test_frames.py - 第一步：IDLE 过滤 + 每组 200 帧选帧清单
# 输入：3 个测试分块 parquet（本地）
# 输出：每块 200 个非空闲帧的帧号清单 CSV（不切帧、不下视频）
import pandas as pd
import os

BASE = r"D:\Projects\NitroGen-IEG\shards\hk_chunks"
FRAMES = 200

# 17 按钮列
BTN_COLS = ["dpad_down","dpad_left","dpad_right","dpad_up",
            "left_shoulder","left_thumb","left_trigger",
            "right_shoulder","right_thumb","right_trigger",
            "south","west","east","north","back","start","guide"]

BLOCKS = [
    {"name": "test_v946202192_chunk_0003", "start_frame": 3600},
    {"name": "test_v946202192_chunk_0065", "start_frame": 78000},
    {"name": "test_v946202192_chunk_0096", "start_frame": 115200},
]

rows = []
for b in BLOCKS:
    path = os.path.join(BASE, f"{b['name']}_actions_processed.parquet")
    df = pd.read_parquet(path)
    assert len(df) == 1200, f"{b['name']}: 行数 {len(df)} != 1200"

    btns = df[BTN_COLS].fillna(False).astype(bool)
    any_btn = btns.any(axis=1).values

    jl = df["j_left"].apply(lambda v: (float(v[0]), float(v[1])) if v is not None else (0.0, 0.0))
    jr = df["j_right"].apply(lambda v: (float(v[0]), float(v[1])) if v is not None else (0.0, 0.0))
    joy_act = (jl.apply(lambda p: abs(p[0])) + jl.apply(lambda p: abs(p[1]))
               + jr.apply(lambda p: abs(p[0])) + jr.apply(lambda p: abs(p[1]))) > 0.01

    idle = (~any_btn) & (~joy_act.values)
    valid = [i for i in range(1200) if not idle[i]]
    valid_count = len(valid)

    # 均匀抽 200 帧
    chosen = []
    if valid_count >= FRAMES:
        step = valid_count / FRAMES
        for k in range(FRAMES):
            chosen.append(valid[min(int(k * step), valid_count - 1)])
    else:
        chosen = valid[:]  # 有效帧不足则全取

    for i in chosen:
        rows.append({
            "block": b["name"],
            "frame_in_chunk": i,
            "absolute_frame": b["start_frame"] + i,
            "idle": False,
        })

    print(f"[{b['name']}] 有效帧={valid_count}/1200, 抽选 {len(chosen)} 帧", flush=True)
    # 简要：前 10 帧 + 分布
    print(f"   前10帧(块内): {chosen[:10]}", flush=True)

OUT = r"D:\Projects\NitroGen-IEG\shards\hk_chunks\test_frames_plan.csv"
pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"\n完成: 帧号清单已写 {OUT}（共 {len(rows)} 帧 = 3 组 × {FRAMES} 帧）")
