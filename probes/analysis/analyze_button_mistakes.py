# analyze_button_mistakes.py - 预测数据补充分析（不改 Day3 主结论，仅补充挖掘）
# 输入：raw_predictions.csv
# 输出：day4_report/按钮漏按多按排行.csv、day4_report/组合键误配分析.md
import os
from collections import Counter, defaultdict
import pandas as pd

CSV = r"D:\Projects\NitroGen-IEG\shards\eval\raw_predictions.csv"
OUT_DIR = r"D:\Projects\NitroGen-IEG\day4_report"
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
gt = df[[f"gt_{b}" for b in BTN]].fillna(False).astype(bool)
pred = df[[f"pred_{b}" for b in BTN]].fillna(False).astype(bool)
n = len(df)

# ---- 1) 漏按/多按排行 ----
rows = []
for b in BTN:
    g = gt[f"gt_{b}"]
    p = pred[f"pred_{b}"]
    hit = int((g & p).sum())
    miss = int((g & ~p).sum())   # 真值按、预测未按
    extra = int((p & ~g).sum())  # 预测按、真值未按
    rows.append({
        "按键": b, "中文名": BTN_CN[b],
        "真值按下": int(g.sum()), "预测按下": int(p.sum()),
        "命中": hit, "漏按": miss, "多按": extra,
        "漏按率": f"{miss/max(int(g.sum()),1)*100:.1f}%",
        "多按率": f"{extra/max(int(p.sum()),1)*100:.1f}%",
    })
rank = pd.DataFrame(rows)
rank_csv = os.path.join(OUT_DIR, "按钮漏按多按排行.csv")
rank.to_csv(rank_csv, index=False, encoding="utf-8-sig")

print("=== 漏按最多的键（真值按了但模型没按） ===")
for _, r in rank.sort_values("漏按", ascending=False).head(5).iterrows():
    print(f"  {r['按键']}({r['中文名']}): 漏按 {r['漏按']} / 真值{r['真值按下']}  ({r['漏按率']})")
print("\n=== 多按最多的键（模型按了但真值没按） ===")
for _, r in rank.sort_values("多按", ascending=False).head(5).iterrows():
    print(f"  {r['按键']}({r['中文名']}): 多按 {r['多按']} / 预测{r['预测按下']}  ({r['多按率']})")

# ---- 2) 组合键误配 ----
# 只看真值按 >=2 键的帧
combo_mask = gt.sum(axis=1) >= 2
combo_df = df[combo_mask]
print(f"\n=== 组合键帧（真值按≥2键）：{len(combo_df)} 帧 ===")

# 每个真值组合的表现
combo_stat = defaultdict(lambda: {"count": 0, "full": 0, "subset": 0, "has_wrong": 0, "none": 0})
for i in combo_df.index:
    gkeys = {b for b in BTN if gt.loc[i, f"gt_{b}"]}
    pkeys = {b for b in BTN if pred.loc[i, f"pred_{b}"]}
    key = tuple(sorted(gkeys))
    s = combo_stat[key]
    s["count"] += 1
    if pkeys == gkeys:
        s["full"] += 1
    elif pkeys and pkeys < gkeys:
        s["subset"] += 1
    elif pkeys & gkeys:
        s["has_wrong"] += 1  # 有命中键但混入了错误键
    else:
        s["none"] += 1

top_combos = sorted(combo_stat.items(), key=lambda kv: -kv[1]["count"])[:8]

md = ["# 组合键误配分析（补充材料，不改 Day3 主结论）",
      "",
      f"> 数据：zero-shot 评测 600 帧（v946202192 三块）；其中真值按 ≥2 键的组合键帧共 {len(combo_df)} 帧。",
      "",
      "## 一、组合键帧的分类口径",
      "",
      "- **完全命中**：预测按键集合与真值完全一致",
      "- **漏按**：预测是真值的子集（漏按了其中至少一键）",
      "- **混入错键**：命中了部分真值键，但还按了真值没有的键",
      "- **完全错误**：预测按键与真值无任何交集",
      "",
      "## 二、常见组合键及模型表现",
      "",
      "| 真值组合 | 帧数 | 完全命中 | 漏按 | 混入错键 | 完全错误 |",
      "|---|---|---|---|---|---|"]

for key, s in top_combos:
    names = "、".join(f"{b}({BTN_CN[b]})" for b in key)
    md.append(f"| {names} | {s['count']} | {s['full']} | {s['subset']} | {s['has_wrong']} | {s['none']} |")

# 组合键总体表现
tot_full = sum(s["full"] for s in combo_stat.values())
tot_sub = sum(s["subset"] for s in combo_stat.values())
tot_wrong = sum(s["has_wrong"] for s in combo_stat.values())
tot_none = sum(s["none"] for s in combo_stat.values())
md += [
    "",
    "## 三、组合键总体表现",
    "",
    f"- 完全命中 {tot_full} 帧（{tot_full/max(len(combo_df),1)*100:.1f}%）",
    f"- 漏按 {tot_sub} 帧（{tot_sub/max(len(combo_df),1)*100:.1f}%）",
    f"- 混入错键 {tot_wrong} 帧（{tot_wrong/max(len(combo_df),1)*100:.1f}%）",
    f"- 完全错误 {tot_none} 帧（{tot_none/max(len(combo_df),1)*100:.1f}%）",
    "",
    "## 四、结论要点（供报告引用）",
    "",
    "> 占位：根据上述数字填写——模型在组合键上的主要错误形态（漏按为主还是混入错键为主）、最常被漏按/多按的键。",
]
md_path = os.path.join(OUT_DIR, "组合键误配分析.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")

print(f"\n漏按/多按排行已写：{rank_csv}")
print(f"组合键误配分析已写：{md_path}")
