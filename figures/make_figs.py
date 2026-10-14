# Figures for Paper 2. Usage: python make_figs.py <fig_dir> <result_dir>
# Draft figures for Paper 2 (JBHI). Numbers are copied from the audited logs
# (c_coupling/out, audit 2026-10-13); see the D1 protocol for sources.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd

BLUE, ORANGE, INK, INK2, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#898781", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
import sys
OUT = sys.argv[1].rstrip("/") + "/"  # output folder; also holds blind_own.txt
RES = sys.argv[2]  # analysis result folder (step10_censoring.csv)


def save(fig, name):
    fig.savefig(OUT + name + ".pdf", bbox_inches="tight")
    fig.savefig(OUT + name + ".png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 2 coupling
own = [float(x) for x in open(OUT + "blind_own.txt").read().split(",")]
rows = [  # label, eps2, lo, hi, derma-xai?
    ("Derma-XAI train+val, vs truth", 0.387, 0.382, 0.393, True),
    ("Derma-XAI aware test, vs truth", 0.300, 0.280, 0.321, True),
    ("   without annotator 1", 0.211, 0.188, 0.237, True),
    ("Derma-XAI blind test, vs reader's own call", own[0], own[1], own[2], True),
    ("Derma-XAI blind test, vs truth", 0.026, 0.014, 0.043, True),
    ("PH2 (one expert)", 0.206, 0.153, 0.265, False),
    ("Derm7pt (atlas)", 0.138, 0.121, 0.158, False),
    ("EASY (5 blinded experts)", 0.065, 0.049, 0.085, False),
    ("SkinCon DDI, malignant vs not", 0.023, 0.015, 0.034, False),
    ("SkinCon Fitz17k, malignant vs not", 0.008, 0.007, 0.010, False),
]
fig, ax = plt.subplots(figsize=(3.5, 2.7))
for i, (lab, e, lo, hi, dx) in enumerate(rows):
    yy = len(rows) - 1 - i
    c, m = (BLUE, "o") if dx else (INK2, "s")
    ax.plot([lo, hi], [yy, yy], color=c, lw=1.2, solid_capstyle="round")
    ax.plot(e, yy, m, color=c, ms=4.5, mec="white", mew=0.6)
    ax.text(hi + 0.008, yy, f"{e:.3f}", va="center", fontsize=6.5, color=INK2)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows][::-1])
ax.set_xlim(0, 0.45)
ax.set_xlabel("Coupling ε² (label variance explained by the diagnosis)")
ax.axhline(4.5, color=GRID, lw=0.8)
ax.xaxis.grid(True, color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
ax.text(0.45, 9.55, "Derma-XAI: labels explain a diagnosis", ha="right", color=BLUE, fontsize=6.5)
ax.text(0.45, 3.5, "Other label sources", ha="right", color=INK2, fontsize=6.5)
save(fig, "fig2_coupling")

# ---------------------------------------------------------------- Fig. 3 F settings
F = [  # label, model AUROC, null AUROC, excess, lo, hi
    ("F1  Derma-XAI model → aware labels", 0.780, 0.773, 0.007, -0.011, 0.023),
    ("F2  Derma-XAI model → blind labels", 0.663, 0.628, 0.035, 0.022, 0.049),
    ("F3  Derma-XAI model → EASY (blinded)", 0.712, 0.649, 0.063, 0.049, 0.078),
    ("F4  Derm7pt model → Derm7pt labels", 0.869, 0.718, 0.151, 0.135, 0.166),
    ("F5  Derm7pt model → EASY (blinded)", 0.776, 0.639, 0.138, 0.119, 0.158),
]
fig, ax = plt.subplots(figsize=(3.5, 1.9))
for i, (lab, mo, nu, ex, lo, hi) in enumerate(F):
    yy = len(F) - 1 - i
    ax.plot([nu, mo], [yy, yy], color=GRID, lw=2.5, solid_capstyle="round", zorder=1)
    ax.plot(nu, yy, "s", color=MUTED, ms=4.5, mec="white", mew=0.6, zorder=2)
    ax.plot(mo, yy, "o", color=BLUE, ms=4.8, mec="white", mew=0.6, zorder=3)
    ax.text(0.905, yy, f"{ex:+.3f} [{lo:+.3f}, {hi:+.3f}]", va="center", fontsize=6.3, color=INK2)
ax.set_yticks(range(len(F)))
ax.set_yticklabels([f[0] for f in F][::-1])
ax.set_xlim(0.58, 0.9)
ax.set_xlabel("Mean concept AUROC")
ax.axhline(1.5, color=GRID, lw=0.8)
ax.xaxis.grid(True, color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
ax.text(0.905, 4.75, "Excess [95% CI]", fontsize=6.3, color=INK, fontweight="bold")
ax.plot([], [], "o", color=BLUE, ms=4.5, label="Concept model")
ax.plot([], [], "s", color=MUTED, ms=4.5, label="Diagnosis-only null")
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, frameon=False, fontsize=6.5, handletextpad=0.3)
save(fig, "fig3_F")

# ---------------------------------------------------------------- Fig. 4 V and V2
v = pd.read_csv(RES + "/step10_censoring.csv")
v = v[v.scenario == "T"].sort_values("pi")
fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 1.75), sharey=True)
a.plot(v.pi, v.all_model, "-o", color=BLUE, ms=3.5, lw=1.4, mec="white", mew=0.5)
a.plot(v.pi, v.all_null, "-s", color=MUTED, ms=3.5, lw=1.4, mec="white", mew=0.5)
a.set_title("(a) V: censor blinded labels", fontsize=7, loc="left", color=INK)
a.set_xlabel("π applied to EASY labels")
a.set_ylabel("Mean concept AUROC vs EASY")
a.text(1.0, v.all_model.iloc[-1] + 0.006, "model", ha="right", color=BLUE, fontsize=6.3)
a.text(1.0, v.all_null.iloc[-1] - 0.032, "null", ha="right", color=MUTED, fontsize=6.3)
a.text(0.10, 0.618, "excess 0.138 → 0.093\nΔ −0.044 [−0.063, −0.028]", fontsize=5.8, color=INK2,
       va="center")
V2 = [(0.0, 0.773, 0.641), (0.5, 0.763, 0.648), (1.0, 0.748, 0.650)]
b.plot([p for p, _, _ in V2], [m for _, m, _ in V2], "-o", color=BLUE, ms=3.5, lw=1.4, mec="white", mew=0.5)
b.plot([p for p, _, _ in V2], [n for _, _, n in V2], "-s", color=MUTED, ms=3.5, lw=1.4, mec="white", mew=0.5)
b.set_title("(b) V2: train on censored labels", fontsize=7, loc="left", color=INK)
b.set_xlabel("π applied to training labels")
b.text(0.04, 0.715, "excess 0.132 → 0.098\nΔ −0.034 [−0.048, −0.020]", fontsize=5.8, color=INK2,
       va="center")
for ax in (a, b):
    ax.set_xticks([0, 0.5, 1])
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
a.set_ylim(0.6, 0.86)
save(fig, "fig4_V_V2")

# ---------------------------------------------------------------- Fig. 1 schematic
fig, ax = plt.subplots(figsize=(7.1, 1.9))
ax.set_xlim(0, 100)
ax.set_ylim(0, 30)
ax.axis("off")


def box(x, y, w, h, text, fc="#f3f2ef", ec=MUTED, color=INK, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.2, color=color,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(x0, y0, x1, y1, color=MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, color=color, lw=0.9))


box(1, 17, 17, 10, "Labeller holds a\ndiagnosis\n(true or own call)")
box(21, 17, 17, 10, "Explains it: marks\nonly characteristics\nsupporting it")
box(41, 17, 17, 10, "Labels encode the\ndiagnosis (no\nopposite-side marks)")
box(61, 17, 17, 10, "Concept model\ntrained and scored\non these labels")
box(81, 17, 17, 10, "High concept AUROC,\nlargely from the\ndiagnosis", ec=ORANGE)
for x0, x1 in ((18.8, 20.2), (38.8, 40.2), (58.8, 60.2), (78.8, 80.2)):
    arrow(x0, 22, x1, 22)
box(37, 1.5, 24, 9, "Diagnosis-only null:\np\u00b7E[c|mel] + (1\u2212p)\u00b7E[c|nev]", fc="#eaf2fc", ec=BLUE)
box(64.5, 1.5, 14, 9, "Blinded\nreference\n(e.g. EASY)", fc="#eaf2fc", ec=BLUE)
box(82, 1.5, 16, 9, "Excess =\nAUROC \u2212\nnull AUROC", fc="#eaf2fc", ec=BLUE, bold=True)
arrow(61.8, 6, 63.7, 6, BLUE)
arrow(79.3, 6, 81.2, 6, BLUE)
arrow(90, 11.2, 90, 16.4, BLUE)
ax.text(37, 13.2, "Audit (this paper)", fontsize=6.4, color=BLUE, fontweight="bold")
save(fig, "fig1_schematic")
print("done")
