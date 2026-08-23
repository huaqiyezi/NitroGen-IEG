# n2_stats_v4.py - 选10个空闲率适当的processed分块,重做十条序列图(删旧图+抽+出图)
import os, csv, glob, io, json, tarfile, time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
    if os.path.exists(fp): font_manager.fontManager.addfont(fp)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei","SimHei","DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # probes/analysis -> 仓库根
BASE = os.path.join(ROOT, "shards", "hk_chunks")
OUT = os.path.join(BASE, "n2_stats")
TAR = os.path.join(ROOT, "shards", "actions", "SHARD_0000.tar.gz")

# 空闲率适中的10个processed分块(37%~47%, 避开极端)
WANT = ["v1658382176_chunk_0302","v1658382176_chunk_0413","v1658382176_chunk_0308",
        "v1658382176_chunk_0353","v1658382176_chunk_0356","v1658382176_chunk_0305",
        "v1658382176_chunk_0202","v1658382176_chunk_0307","v1658382176_chunk_0349",
        "v1658382176_chunk_0331"]

# ===== 1. 删旧图/旧CSV =====
os.makedirs(OUT, exist_ok=True)
removed = 0
for f in glob.glob(os.path.join(OUT,"*")):
    os.remove(f); removed += 1
print(f"删除旧产出 {removed} 个")

# ===== 2. 抽10个processed分块 =====
print("抽取10个processed分块...")
t0=time.time()
tf = tarfile.open(TAR, "r|gz")
cur_dir = None; cur_processed = None
saved = set()
for m in tf:
    name = m.name
    d = name.rsplit("/",1)[0] if "/" in name else ""
    base = name.rsplit("/",1)[-1]
    if d != cur_dir:
        cur_dir = d; cur_processed = None
    if base == "actions_processed.parquet":
        cur_processed = tf.extractfile(m).read()
        continue
    if base == "metadata.json":
        try: meta = json.loads(tf.extractfile(m).read())
        except: continue
        if meta.get("game") == "hollow_knight" and cur_processed is not None:
            chunk = os.path.basename(d)
            if chunk in WANT:
                local = os.path.join(BASE, chunk + "_actions_processed.parquet")
                open(local,"wb").write(cur_processed)
                saved.add(chunk)
                if len(saved) >= len(WANT):
                    break
tf.close()
print(f"抽出 {len(saved)} 个, 用时 {time.time()-t0:.0f}s")

# ===== 3. 统计出图 =====
BTN = ["dpad_down","dpad_left","dpad_right","dpad_up","left_shoulder","left_thumb",
       "left_trigger","right_shoulder","right_thumb","right_trigger","south","west",
       "east","north","back","start","guide"]
BTN_CN = {"dpad_down":"十字键下","dpad_left":"十字键左","dpad_right":"十字键右","dpad_up":"十字键上",
    "left_shoulder":"左肩键(LB)","left_thumb":"左摇杆按下","left_trigger":"左扳机(LT)",
    "right_shoulder":"右肩键(RB)","right_thumb":"右摇杆按下","right_trigger":"右扳机(RT)",
    "south":"南键(A)","west":"西键(X)","east":"东键(B)","north":"北键(Y)",
    "back":"返回键","start":"开始键","guide":"主页键"}

FILES = sorted(glob.glob(os.path.join(BASE,"v1658382176_chunk_*_actions_processed.parquet")))
# 只取WANT里的
FILES = [p for p in FILES if os.path.basename(p).replace("_actions_processed.parquet","") in WANT]
print(f"\n出图分块数: {len(FILES)}")

summary = []
for k, path in enumerate(FILES, 1):
    df = pd.read_parquet(path)
    tag = os.path.basename(path).replace("_actions_processed.parquet","")
    df["jl_x"] = df["j_left"].apply(lambda v: v[0]); df["jl_y"] = df["j_left"].apply(lambda v: v[1])
    df["jr_x"] = df["j_right"].apply(lambda v: v[0]); df["jr_y"] = df["j_right"].apply(lambda v: v[1])

    csv_path = os.path.join(OUT, f"{tag}_data.csv")
    with open(csv_path,"w",newline="",encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["帧号"]+[BTN_CN[b] for b in BTN]+["左摇杆x","左摇杆y","右摇杆x","右摇杆y","是否空闲帧"])
        for i in range(len(df)):
            r = df.iloc[i]; pressed = [int(r[b]) for b in BTN]
            w.writerow([i]+pressed+[round(float(r["jl_x"]),4),round(float(r["jl_y"]),4),
                round(float(r["jr_x"]),4),round(float(r["jr_y"]),4),0 if any(pressed) else 1])

    fig = plt.figure(figsize=(13,9))
    gs = fig.add_gridspec(3,1,height_ratios=[2,1,1],hspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(df[BTN].T.values.astype(float),aspect="auto",cmap="Blues",interpolation="nearest")
    ax1.set_yticks(range(len(BTN))); ax1.set_yticklabels([BTN_CN[b] for b in BTN],fontsize=10)
    ax1.set_xlabel("帧号",fontsize=12); ax1.set_title(f"序列{k} {tag} — 按键热力图",fontsize=13,fontweight="bold")
    fig.colorbar(im,ax=ax1,shrink=0.8)
    ax2 = fig.add_subplot(gs[1])
    x=range(len(df))
    ax2.plot(x,df["jl_x"],label="左摇杆x"); ax2.plot(x,df["jl_y"],label="左摇杆y")
    ax2.plot(x,df["jr_x"],label="右摇杆x"); ax2.plot(x,df["jr_y"],label="右摇杆y")
    ax2.axhline(0,color="gray",lw=0.5,ls="--"); ax2.set_ylim(-1.05,1.05)
    ax2.set_xlabel("帧号",fontsize=12); ax2.set_ylabel("摇杆值",fontsize=12)
    ax2.set_title(f"序列{k} {tag} — 摇杆时序",fontsize=12,fontweight="bold")
    ax2.legend(loc="upper right",fontsize=9,ncol=4); ax2.grid(alpha=0.3)
    gs2 = gs[2].subgridspec(1,2,wspace=0.2)
    axL=fig.add_subplot(gs2[0]); axL.scatter(df["jl_x"],df["jl_y"],s=2,alpha=0.4,c="steelblue")
    axL.set_xlim(-1.05,1.05); axL.set_ylim(-1.05,1.05); axL.set_aspect("equal")
    axL.set_title("左摇杆轨迹",fontsize=11); axL.set_xlabel("x"); axL.set_ylabel("y"); axL.grid(alpha=0.3)
    axR=fig.add_subplot(gs2[1]); axR.scatter(df["jr_x"],df["jr_y"],s=2,alpha=0.4,c="darkorange")
    axR.set_xlim(-1.05,1.05); axR.set_ylim(-1.05,1.05); axR.set_aspect("equal")
    axR.set_title("右摇杆轨迹",fontsize=11); axR.set_xlabel("x"); axR.set_ylabel("y"); axR.grid(alpha=0.3)
    fig_path = os.path.join(OUT,f"{tag}.png")
    plt.savefig(fig_path,dpi=110,bbox_inches="tight"); plt.close()

    freq = df[BTN].mean()
    pressed_frames = int(df[BTN].any(axis=1).sum())
    summary.append({"序列":k,"分块":tag,"帧数":len(df),"空闲率":round(1-pressed_frames/len(df),3),
        "最高频键":str(freq.idxmax()),"最高频占比":round(float(freq.max()),3),
        "左摇杆x均值":round(float(df["jl_x"].mean()),3),"左摇杆y均值":round(float(df["jl_y"].mean()),3),
        "右摇杆x均值":round(float(df["jr_x"].mean()),3),"右摇杆y均值":round(float(df["jr_y"].mean()),3)})
    print(f"序列{k} {tag}: 空闲率{summary[-1]['空闲率']} 最高频{summary[-1]['最高频键']}({summary[-1]['最高频占比']})")

summ = os.path.join(OUT,"summary.csv")
with open(summ,"w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
print(f"\n完成: {len(FILES)} 张图 + {len(FILES)} 数据CSV + summary")
