# gen_module_diagram.py v5 - 标签紧贴各自连线(独立text定位), 弧度减小
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(13.5, 10.5))
ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.02, 1.02); ax.axis("off")

def box(cx, cy, w, h, title, sub="", fc="#eaf2fb"):
    x = cx - w/2; y = cy - h/2
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.015",
                 fc=fc, ec="#2c3e50", lw=1.6, zorder=3))
    ax.text(cx, cy + h*0.18, title, ha="center", va="center",
            fontsize=13, fontweight="bold", zorder=4)
    if sub:
        ax.text(cx, cy - h*0.24, sub, ha="center", va="center",
                fontsize=8.5, color="#444", zorder=4)

def arrow(p1, p2, rad=0.0, dbl=False):
    style = "<|-|>" if dbl else "-|>"
    ax.annotate("", xy=p2, xytext=p1, zorder=2,
        arrowprops=dict(arrowstyle=style, lw=1.6, color="#555",
                        connectionstyle=f"arc3,rad={rad}"))

def label(x, y, s, ha="center"):
    ax.text(x, y, s, fontsize=9, ha=ha, va="center", color="#b91c1c",
            bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.5), zorder=5)

# ===== 框 =====
box(0.5, 0.95, 0.30, 0.06, "开发者 刘依帆", "配置 / 运行全部模块", fc="#fde9d9")
box(0.5, 0.80, 0.28, 0.08, "① 数据准备", "扫 metadata / 下子集 parquet")
box(0.12, 0.58, 0.23, 0.08, "④ 可视化", "按键频率 / 摇杆分布 / 序列图")
box(0.38, 0.58, 0.21, 0.08, "⑤ LoRA 微调", "rank16 / DiT注意力层")
box(0.74, 0.58, 0.21, 0.08, "③ 评测", "一致率 / 摇杆相关 / 对照表")
box(0.5, 0.38, 0.22, 0.08, "② 推理服务", "serve.py + ng.pt")
box(0.5, 0.16, 0.30, 0.06, "验收", "互评同学 + 老师", fc="#e7f4e4")

# ===== 连线(锚点贴框边) =====
arrow((0.5, 0.92), (0.5, 0.84))                                   # 开发者→①
arrow((0.44, 0.76), (0.15, 0.62), rad=0.10)                       # ①→④
arrow((0.5, 0.76), (0.38, 0.62), rad=0.03)                        # ①→⑤
arrow((0.56, 0.76), (0.71, 0.62), rad=-0.10)                      # ①→③
arrow((0.38, 0.54), (0.49, 0.42), rad=0.03)                       # ⑤→②
arrow((0.71, 0.54), (0.53, 0.42), dbl=True, rad=-0.05)           # ③↔②
arrow((0.12, 0.54), (0.42, 0.19), rad=0.12)                       # ④→验收
arrow((0.74, 0.54), (0.58, 0.19), rad=-0.12)                      # ③→验收

# ===== 标签: 紧贴各自连线中点 =====
label(0.53, 0.88, "配置/运行", ha="left")              # 开发者→① 右侧贴线
label(0.30, 0.71, "统计子集")                          # ①→④ 线上方
label(0.45, 0.71, "训练子集", ha="left")               # ①→⑤ 右侧贴线
label(0.62, 0.71, "测试标签", ha="right")              # ①→③ 左侧贴线
label(0.46, 0.50, "LoRA权重", ha="left")               # ⑤→② 右侧贴线
label(0.60, 0.49, "帧/预测", ha="right")               # ③↔② 左侧贴线
label(0.13, 0.40, "交付")                              # ④→验收 左外侧
label(0.73, 0.40, "交付")                              # ③→验收 右外侧

plt.tight_layout()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # probes/visual -> 仓库根
out1 = os.path.join(ROOT, "shards", "out", "模块图.png")
os.makedirs(os.path.dirname(out1), exist_ok=True)
fig.savefig(out1, dpi=150, bbox_inches="tight")
print("已生成:", out1)
