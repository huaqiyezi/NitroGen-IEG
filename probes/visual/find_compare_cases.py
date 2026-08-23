# find_compare_cases.py - 只读分析：从 raw_predictions.csv 找"模型输出 vs 标注"对比案例
# 范围：仅读本地评测明细，只打印不落盘
# 输出：4 类案例（A 按键命中 / B 组合按错 / C 摇杆同向 / D 摇杆反向）的数量与候选帧
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # probes/visual -> 仓库根
CSV = os.path.join(ROOT, "shards", "eval", "raw_predictions.csv")
BTN = ["dpad_down","dpad_left","dpad_right","dpad_up",
       "left_shoulder","left_thumb","left_trigger",
       "right_shoulder","right_thumb","right_trigger",
       "south","west","east","north","back","start","guide"]

df = pd.read_csv(CSV)
print(f"共 {len(df)} 帧\n")

gt_btn_cols = [f"gt_{b}" for b in BTN]
pred_btn_cols = [f"pred_{b}" for b in BTN]
gt_btns = df[gt_btn_cols].fillna(False).astype(bool).values
pred_btns = df[pred_btn_cols].fillna(False).astype(bool).values

def btn_names(arr):
    return [BTN[i] for i in range(len(BTN)) if arr[i]]

def frame_desc(i):
    return f"{df.loc[i,'block']} 帧{df.loc[i,'absolute_frame']}(块内{df.loc[i,'frame_in_chunk']})"

cases = {"A": [], "B": [], "C": [], "D": []}

for i in range(len(df)):
    g, p = gt_btns[i], pred_btns[i]
    is_idle = bool(df.loc[i, "idle"])
    jg = float(df.loc[i, "gt_jl_x"]); jp = float(df.loc[i, "pred_jl_x"])
    n_gt = int(g.sum()); n_p = int(p.sum())

    # A: 按键完全命中（含都按下0个键也算一致），非空闲帧
    if (g == p).all() and not is_idle:
        cases["A"].append(i)
    # B: 真值按 >=2 键 且 预测与真值不一致（漏按/多按/换键）
    if n_gt >= 2 and not (g == p).all():
        cases["B"].append(i)
    # C: 摇杆同向：同号 且 幅度都大
    if jg * jp > 0 and abs(jg) > 0.3 and abs(jp) > 0.3:
        cases["C"].append(i)
    # D: 摇杆反向：异号 且 幅度都大
    if jg * jp < 0 and abs(jg) > 0.3 and abs(jp) > 0.3:
        cases["D"].append(i)

names = {"A": "按键完全命中", "B": "组合键按错", "C": "摇杆同向(正确)", "D": "摇杆反向(错误)"}
for k in ["A", "B", "C", "D"]:
    idxs = cases[k]
    print(f"=== {names[k]}：{len(idxs)} 帧 ===")
    # A 类优先展示"命中且有按键"的帧；其余类优先非 idle
    if k == "A":
        idxs = sorted(idxs, key=lambda i: (gt_btns[i].sum() == 0, df.loc[i, "idle"]))
    # 按 block 分组展示（每块最多 3 个候选）
    by_block = {}
    for i in idxs:
        blk = df.loc[i, "block"]
        by_block.setdefault(blk, []).append(i)
    for blk, lst in by_block.items():
        shown = [i for i in lst if not df.loc[i, "idle"]] or lst
        print(f"  [{blk}]")
        for i in shown[:3]:
            g = gt_btns[i]; p = pred_btns[i]
            jg = float(df.loc[i, "gt_jl_x"]); jp = float(df.loc[i, "pred_jl_x"])
            extra = ""
            if k in ("C", "D"):
                extra = f" jl_x: gt={jg:.2f} pred={jp:.2f}"
            print(f"    帧{df.loc[i,'absolute_frame']}(块内{df.loc[i,'frame_in_chunk']})"
                  f" 真值按键={btn_names(g) or '无'} 预测按键={btn_names(p) or '无'}"
                  f" 空闲={df.loc[i,'idle']}{extra}")
    print()
