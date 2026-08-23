# run_eval.py - N3 zero-shot 评测：切好的帧 -> serve.py 推理 -> 与 parquet 真值比对
# 输出：raw_predictions.csv + metrics_by_group.csv + metrics_overall.csv
# 用法: py -3.12 run_eval.py [--limit N]   （--limit 用于小规模验证）
import argparse, os, pickle, time, json
import numpy as np
import pandas as pd
import zmq
from PIL import Image
import cv2

# ---------- 配置 ----------
SERVE_PORT = 5555
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # probes/eval -> 仓库根
FRAME_DIR = os.path.join(ROOT, "shards", "frames")
HK_DIR = os.path.join(ROOT, "shards", "hk_chunks")
EVAL_DIR = os.path.join(ROOT, "shards", "eval")
PLAN = os.path.join(HK_DIR, "test_frames_plan.csv")

BTN_COLS = ["dpad_down","dpad_left","dpad_right","dpad_up",
            "left_shoulder","left_thumb","left_trigger",
            "right_shoulder","right_thumb","right_trigger",
            "south","west","east","north","back","start","guide"]

# 模型 21 维按钮 -> 数据集 17 列映射（剔除 RIGHT_BOTTOM/LEFT/RIGHT/UP 4 个右摇杆方向维）
# BUTTON_ACTION_TOKENS = [BACK, DPAD_DOWN, DPAD_LEFT, DPAD_RIGHT, DPAD_UP, EAST, GUIDE,
#   LEFT_SHOULDER, LEFT_THUMB, LEFT_TRIGGER, NORTH, RIGHT_BOTTOM, RIGHT_LEFT, RIGHT_RIGHT,
#   RIGHT_SHOULDER, RIGHT_THUMB, RIGHT_TRIGGER, RIGHT_UP, SOUTH, START, WEST]
MODEL_TO_DATASET = {
    0: "back", 1: "dpad_down", 2: "dpad_left", 3: "dpad_right", 4: "dpad_up",
    5: "east", 6: "guide", 7: "left_shoulder", 8: "left_thumb", 9: "left_trigger",
    10: "north", 14: "right_shoulder", 15: "right_thumb", 16: "right_trigger",
    18: "south", 19: "start", 20: "west",
}
assert len(MODEL_TO_DATASET) == 17

def preprocess_img(path, size=256):
    """与 play.py 一致：RGB -> resize(256,256) INTER_AREA -> RGB ndarray"""
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"无法读取图片: {path}")
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 帧(0=全部)")
    args = ap.parse_args()

    os.makedirs(EVAL_DIR, exist_ok=True)
    plan = pd.read_csv(PLAN)
    if args.limit:
        plan = plan.head(args.limit)
    print(f"评测帧数: {len(plan)}")

    # 连接 serve
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 180000)
    sock.connect(f"tcp://localhost:{SERVE_PORT}")
    sock.send(pickle.dumps({"type": "reset"}))
    sock.recv()

    # 预加载真值 parquet
    gt_cache = {}
    for block in plan["block"].unique():
        p = os.path.join(HK_DIR, f"{block}_actions_processed.parquet")
        gt_cache[block] = pd.read_parquet(p)

    rows = []
    t0 = time.time()
    for idx, r in plan.iterrows():
        block, abs_frame, frame_in_chunk = r["block"], int(r["absolute_frame"]), int(r["frame_in_chunk"])
        img_path = os.path.join(FRAME_DIR, block, f"{abs_frame}.png")
        image = preprocess_img(img_path)

        sock.send(pickle.dumps({"type": "predict", "image": image}))
        resp = pickle.loads(sock.recv())
        if resp["status"] != "ok":
            raise RuntimeError(f"推理失败 frame {abs_frame}: {resp}")
        pred = resp["pred"]

        # 取动作块第 1 步
        p_buttons = np.array(pred["buttons"])[0]          # (21,)
        p_jleft = np.array(pred["j_left"])[0]             # (2,)
        p_jright = np.array(pred["j_right"])[0]           # (2,)

        # 17 维按钮映射（模型已 >0.5 阈值化成 0/1）
        gt = gt_cache[block].iloc[frame_in_chunk]
        row = {"block": block, "absolute_frame": abs_frame, "frame_in_chunk": frame_in_chunk}
        idx_of = {v: k for k, v in MODEL_TO_DATASET.items()}
        for btn in BTN_COLS:
            row[f"gt_{btn}"] = int(gt[btn])
            row[f"pred_{btn}"] = int(p_buttons[idx_of[btn]])
        # 摇杆
        row["gt_jl_x"], row["gt_jl_y"] = float(gt["j_left"][0]), float(gt["j_left"][1])
        row["gt_jr_x"], row["gt_jr_y"] = float(gt["j_right"][0]), float(gt["j_right"][1])
        row["pred_jl_x"], row["pred_jl_y"] = float(p_jleft[0]), float(p_jleft[1])
        row["pred_jr_x"], row["pred_jr_y"] = float(p_jright[0]), float(p_jright[1])
        # IDLE 判定
        gt_btns = np.array([row[f"gt_{b}"] for b in BTN_COLS])
        joy_abs = abs(row["gt_jl_x"]) + abs(row["gt_jl_y"]) + abs(row["gt_jr_x"]) + abs(row["gt_jr_y"])
        row["idle"] = bool((gt_btns.sum() == 0) and (joy_abs < 0.01))
        rows.append(row)

        if (idx + 1) % 25 == 0 or (args.limit and idx + 1 == args.limit):
            el = time.time() - t0
            print(f"  已推理 {idx+1}/{len(plan)} 帧, 用时 {el:.0f}s ({el/(idx+1):.2f}s/帧)", flush=True)

    sock.close(); ctx.term()

    df = pd.DataFrame(rows)
    raw_path = os.path.join(EVAL_DIR, "raw_predictions.csv")
    df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"\n原始数据已写: {raw_path} ({len(df)} 帧)")

    # ---------- 计算指标 ----------
    def metrics(sub):
        if len(sub) == 0:
            return {}
        btns = np.array([[r[f"gt_{b}"] for b in BTN_COLS] for _, r in sub.iterrows()])
        preds = np.array([[r[f"pred_{b}"] for b in BTN_COLS] for _, r in sub.iterrows()])
        match = (btns == preds).all(axis=1)
        btn_acc = match.mean()
        per_dim = (btns == preds).mean()
        # 摇杆 Pearson 4 维取平均
        corrs = []
        for key in ["jl_x", "jl_y", "jr_x", "jr_y"]:
            g = sub[f"gt_{key}"].astype(float).values
            p = sub[f"pred_{key}"].astype(float).values
            if np.std(g) < 1e-9 or np.std(p) < 1e-9:
                corrs.append(0.0)
            else:
                corrs.append(np.corrcoef(g, p)[0, 1])
        joy_corr = float(np.mean(corrs))
        return {"btn_acc": float(btn_acc), "per_dim_acc": float(per_dim), "joy_corr": joy_corr}

    # 按组（IDLE 过滤后）
    group_rows = []
    for block, sub in df.groupby("block"):
        sub_valid = sub[~sub["idle"]]
        m = metrics(sub_valid)
        group_rows.append({
            "block": block, "frames_total": len(sub), "frames_valid": len(sub_valid),
            "idle_rate": float(sub["idle"].mean()),
            "btn_acc": m.get("btn_acc"), "per_dim_acc": m.get("per_dim_acc"), "joy_corr": m.get("joy_corr"),
        })
    df_g = pd.DataFrame(group_rows)
    # 总体
    valid = df[~df["idle"]]
    m_all = metrics(valid)
    df_all = pd.DataFrame([{"block": "总体", "frames_total": len(df), "frames_valid": len(valid),
                            "idle_rate": float(df["idle"].mean()), **m_all}])
    g_path = os.path.join(EVAL_DIR, "metrics_by_group.csv")
    o_path = os.path.join(EVAL_DIR, "metrics_overall.csv")
    df_g.to_csv(g_path, index=False, encoding="utf-8-sig")
    df_all.to_csv(o_path, index=False, encoding="utf-8-sig")
    print(f"指标表已写: {g_path} / {o_path}")
    print("\n=== 三组指标(IDLE过滤后) ===")
    print(df_g.to_string(index=False))
    print("\n=== 总体指标 ===")
    print(df_all.to_string(index=False))

if __name__ == "__main__":
    main()
