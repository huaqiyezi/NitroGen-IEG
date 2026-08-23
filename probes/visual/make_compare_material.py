# make_compare_material.py - 生成"模型输出 vs 标注"对比材料（Day5 演示用）
# 输入：raw_predictions.csv + 对应画面帧 PNG
# 输出：对比表 md + 对比图 png（同类型并排、中文标注）-> day4_report/
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # probes/visual -> 仓库根
CSV = os.path.join(ROOT, "shards", "eval", "raw_predictions.csv")
FRAME_DIR = os.path.join(ROOT, "shards", "frames")
OUT_DIR = os.path.join(ROOT, "shards", "out")
BTN = ["dpad_down","dpad_left","dpad_right","dpad_up",
       "left_shoulder","left_thumb","left_trigger",
       "right_shoulder","right_thumb","right_trigger",
       "south","west","east","north","back","start","guide"]
BTN_CN = {
    "dpad_down": "方向键下", "dpad_left": "方向键左", "dpad_right": "方向键右", "dpad_up": "方向键上",
    "left_shoulder": "左肩键", "left_thumb": "左摇杆按下", "left_trigger": "左扳机",
    "right_shoulder": "右肩键", "right_thumb": "右摇杆按下", "right_trigger": "右扳机",
    "south": "A键", "west": "X键", "east": "B键", "north": "Y键",
    "back": "返回键", "start": "开始键", "guide": "导航键",
}

df = pd.read_csv(CSV)
idx = {int(af): i for i, af in enumerate(df["absolute_frame"])}

def btn_names(r, prefix):
    return [b for b in BTN if bool(r[f"{prefix}_{b}"])]

def fmt_btn(lst):
    return "、".join(f"{b}({BTN_CN[b]})" for b in lst) if lst else "无"

def stick_dir(x):
    if x > 0.5:
        return "右推"
    if x < -0.5:
        return "左推"
    return "居中"

# A2 自动选帧：按键 17 维全命中、有按键、且 jl_x 同向，排除 chunk_0003 与其余已用帧
_USED_AF = {3600, 3869, 78630, 78003, 115353, 3709, 78109}

def auto_pick_a2():
    for i in range(len(df)):
        r = df.loc[i]
        af = int(r["absolute_frame"])
        if r["block"] == "test_v946202192_chunk_0003" or af in _USED_AF:
            continue
        g = [bool(r[f"gt_{b}"]) for b in BTN]
        p = [bool(r[f"pred_{b}"]) for b in BTN]
        if g != p or not any(g):
            continue
        gx, py = float(r["gt_jl_x"]), float(r["pred_jl_x"])
        if gx * py > 0:
            return r["block"], af
    return None

_a2 = auto_pick_a2()
assert _a2 is not None, "未找到符合条件的 A2 案例"
_a2_blk, _a2_af = _a2
print(f"[A2 自动选帧] {_a2_blk} 帧{_a2_af}")

# 8 个案例：(类别, 分块, 绝对帧, 说明)，每类 2 个
CASES = [
    ("A 按键完全命中", "test_v946202192_chunk_0003", 3600, "按键命中且摇杆同向，全对示例"),
    ("A 按键完全命中", _a2_blk, _a2_af, "按键命中且摇杆同向"),
    ("B 组合键按错", "test_v946202192_chunk_0003", 3869, "真值按 A+X 两键，预测只按 A，漏按"),
    ("B 组合键按错", "test_v946202192_chunk_0065", 78630, "真值按 RT+A，预测按成 LB，键位错误"),
    ("C 摇杆同向", "test_v946202192_chunk_0065", 78003, "真值预测同为左推，方向一致"),
    ("C 摇杆同向", "test_v946202192_chunk_0096", 115353, "真值预测同为右推，方向一致"),
    ("D 摇杆反向", "test_v946202192_chunk_0003", 3709, "真值左推，预测右推，方向相反"),
    ("D 摇杆反向", "test_v946202192_chunk_0065", 78109, "真值右推，预测左推，方向相反"),
]

# ---- 1) 对比表 Markdown（同样加中文键名） ----
rows = []
for cat, blk, af, note in CASES:
    i = idx[af]
    r = df.loc[i]
    gx, px = float(r["gt_jl_x"]), float(r["pred_jl_x"])
    rows.append({
        "类别": cat, "分块": blk, "绝对帧": af, "帧内号": int(r["frame_in_chunk"]),
        "真值按键": fmt_btn(btn_names(r, "gt")), "预测按键": fmt_btn(btn_names(r, "pred")),
        "真值摇杆": f"{stick_dir(gx)}({gx:+.2f})", "预测摇杆": f"{stick_dir(px)}({px:+.2f})",
        "说明": note,
    })
os.makedirs(OUT_DIR, exist_ok=True)

md = ["# 模型输出 vs 标注 对比案例（Day5 演示材料）",
      "",
      "> 数据来源：Day3 zero-shot 评测（v946202192 三块 600 帧）；画面帧：shards/frames。",
      "> 键位说明：south=A、west=X、east=B、north=Y（Xbox 布局）。摇杆取左摇杆 x 轴。",
      "",
      "| 类别 | 分块 | 绝对帧 | 真值按键 | 预测按键 | 真值摇杆 | 预测摇杆 | 说明 |",
      "|---|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['类别']} | {r['分块'].split('chunk_')[1]} | {r['绝对帧']} | {r['真值按键']} | {r['预测按键']} | {r['真值摇杆']} | {r['预测摇杆']} | {r['说明']} |")
md_path = os.path.join(OUT_DIR, "对比案例表.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")

# ---- 2) 对比图 PNG：同类型并排（每行一类，两案例左右排开） ----
CAT_ORDER = ["A 按键完全命中", "B 组合键按错", "C 摇杆同向", "D 摇杆反向"]
groups = {}
for cat, blk, af, note in CASES:
    groups.setdefault(cat, []).append((blk, af, note))

n_rows = len(CAT_ORDER)
fig, axes = plt.subplots(n_rows, 4, figsize=(17, 4.6 * n_rows))
for row, cat in enumerate(CAT_ORDER):
    pair = groups[cat]
    for col, (blk, af, note) in enumerate(pair):
        chunk_short = blk.split("chunk_")[1]
        # 左：画面
        ax_img = axes[row, col * 2]
        ax_img.imshow(plt.imread(os.path.join(FRAME_DIR, blk, f"{af}.png")))
        ax_img.set_title(f"{cat} · 块{chunk_short} 帧{af}", fontsize=13) if col == 0 \
            else ax_img.set_title(f"块{chunk_short} 帧{af}", fontsize=13)
        ax_img.axis("off")
        # 右：中文说明
        i = idx[af]
        r = df.loc[i]
        gx, px = float(r["gt_jl_x"]), float(r["pred_jl_x"])
        text = (f"真值按键：{fmt_btn(btn_names(r, 'gt'))}\n"
                f"预测按键：{fmt_btn(btn_names(r, 'pred'))}\n"
                f"真值摇杆：{stick_dir(gx)}（{gx:+.2f}）\n"
                f"预测摇杆：{stick_dir(px)}（{px:+.2f}）\n"
                f"说明：{note}")
        ax_txt = axes[row, col * 2 + 1]
        ax_txt.text(0.03, 0.97, text, va="top", ha="left", fontsize=12,
                    transform=ax_txt.transAxes)
        ax_txt.axis("off")
fig.tight_layout()
png_path = os.path.join(OUT_DIR, "model_vs_gt_compare.png")
fig.savefig(png_path, dpi=120)
plt.close(fig)

print(f"对比表已写：{md_path}")
print(f"对比图已写：{png_path}")
print(f"共 {n_rows} 类 × 2 案例")
