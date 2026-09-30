# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Accuracy and performance figures of the sample from its retained runs (docs/validation) and references.

    python3 tools/plot_results.py            # writes docs/media/accuracy.png and docs/media/performance.png

Accuracy: the dissipation curves of the four grids against the published 512^3 peak, and the maximum relative error of
eps of every validation row against its check tolerance. Performance: device time per step and wall time of a run to
t = 10 per grid and NeuronCore count, with the FFTW-only floor of a 192-core CPU node as the reference line. Everything is
computed from the run CSVs and logs; no price appears in the figures.
"""
import glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
VAL, REF, OUT = (os.path.join(ROOT, d) for d in ("docs/validation", "references", "docs/media"))
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # categorical slots 1 to 4, fixed order
FFTW_FLOOR = {64: 45.3, 128: 78.3, 256: 600.2, 512: 3876.9}   # ms per step, 27 fp32 FFTs, hpc7a.96xlarge, FFTW_MEASURE
TOL = {(64, 1): 1e-6, (128, 1): 1e-6, (128, 2): 4e-9, (256, 2): 4e-9, (256, 8): 4e-9, (512, 8): 2e-7,
       (128, 32): 4e-9, (256, 32): 4e-9, (512, 32): 2e-7}


def load(path):
    a = np.genfromtxt(path, delimiter=",", skip_header=1, filling_values=np.nan)
    return dict(t=a[:, 1], E=a[:, 2], eps=a[:, 4], ms=a[:, 6])


def reference(N, R):
    if R == 1:
        return load(os.path.join(REF, f"oracle_{N}.csv"))
    return load(os.path.join(REF, f"tgv_dist2_{N}_32.csv"))


def runs():
    """One entry per validation directory: instance, N, R, run CSV, wall of the run in seconds."""
    out = []
    for d in sorted(glob.glob(os.path.join(VAL, "*-n*-r*"))):
        m = re.search(r"([a-z0-9.]+)-n(\d+)-r(\d+)$", d)
        inst, N, R = m.group(1), int(m.group(2)), int(m.group(3))
        csv = os.path.join(d, f"tgv_{N}_{R}.csv")
        log = open(os.path.join(d, "run.log"), encoding="utf-8").read()
        wall = float(re.search(r"(?:device_steps_wall_s|wall_s)=([0-9.]+)", log).group(1))
        out.append(dict(inst=inst, N=N, R=R, run=load(csv), wall=wall))
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def accuracy(rs):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200, facecolor=SURFACE)
    style(a); style(b)
    pick = [(64, 1, "inf2.xlarge"), (128, 2, "inf2.xlarge"), (256, 8, "inf2.24xlarge"), (512, 8, "inf2.24xlarge")]
    for i, (N, R, inst) in enumerate(pick):
        r = next(x for x in rs if x["N"] == N and x["R"] == R and x["inst"] == inst)["run"]
        a.plot(r["t"], r["eps"], color=SLOT[i], lw=2, solid_capstyle="round", label=f"{N}^3, {R} core{'s' if R > 1 else ''}")
        ip = int(np.nanargmax(r["eps"]))
        a.plot(r["t"][ip], r["eps"][ip], "o", ms=7, color=SLOT[i], mec=SURFACE, mew=2)
    a.plot(9.0, 0.0128, "s", ms=7, color=INK2, mec=SURFACE, mew=2)
    a.annotate("published 512^3 DNS:\n0.0128 at t = 9", (9.0, 0.0128), xytext=(4.4, 0.0134), color=INK2, fontsize=8,
               arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    a.set_xlabel("t", color=INK2, fontsize=9); a.set_ylabel("dissipation rate  eps = 2 nu Omega", color=INK2, fontsize=9)
    a.set_title("Dissipation of the Re 1600 Taylor-Green vortex, this sample's runs", color=INK, fontsize=10, loc="left")
    a.set_xlim(0, 10.1); a.set_ylim(0, 0.0145)
    a.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    # max relative error of eps per row against the check tolerance
    rows = sorted(rs, key=lambda x: (x["R"], x["N"]))
    labels, errs, tols = [], [], []
    for x in rows:
        ref = reference(x["N"], x["R"]); n = min(len(ref["eps"]), len(x["run"]["eps"]))
        e = np.abs(x["run"]["eps"][:n] - ref["eps"][:n]).max() / np.abs(ref["eps"][:n]).max()
        labels.append(f"{x['N']}^3, {x['R']} core{'s' if x['R'] > 1 else ''}, {x['inst']}")
        errs.append(e); tols.append(TOL[(x["N"], x["R"])])
    y = np.arange(len(rows))
    floor = 1e-10
    b.barh(y, [max(e, floor) for e in errs], color=SLOT[0], height=0.55)
    b.scatter(tols, y, marker="|", s=160, color=INK, zorder=3)
    for yi, e, tol in zip(y, errs, tols):
        if e == 0:
            b.text(tol * 1.6, yi, "identical to 9 digits", va="center", fontsize=7.5, color=INK2)
    b.annotate("check tolerance", (tols[0], y[0]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK)
    b.set_xscale("log"); b.set_xlim(floor, 2e-5); b.set_yticks(y); b.set_yticklabels(labels, fontsize=7.5, color=INK2)
    b.invert_yaxis(); b.grid(True, axis="x", color=GRID, linewidth=0.8); b.grid(False, axis="y")
    b.set_xlabel("max relative error of eps over the run (fp64 oracle for 1 core, 32-rank Trainium1 run otherwise)", color=INK2, fontsize=8)
    b.set_title("Every validation row against its tolerance", color=INK, fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "accuracy.png"), facecolor=SURFACE); plt.close(fig)


def performance(rs):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200, facecolor=SURFACE)
    style(a); style(b)
    cfgs = [("inf2.xlarge", 1, "1 Inferentia2 core"), ("inf2.xlarge", 2, "2 Inferentia2 cores"),
            ("inf2.24xlarge", 8, "8 Inferentia2 cores"), ("trn1.32xlarge", 32, "32 Trainium1 cores")]
    for i, (inst, R, name) in enumerate(cfgs):
        pts = sorted((x for x in rs if x["inst"] == inst and x["R"] == R), key=lambda x: x["N"])
        Ns = [x["N"] for x in pts]
        ms = [float(np.median(x["run"]["ms"][2:])) for x in pts]
        wall = [x["wall"] for x in pts]
        a.plot(Ns, ms, "-o", color=SLOT[i], lw=2, ms=7, mec=SURFACE, mew=2, label=name)
        b.plot(Ns, wall, "-o", color=SLOT[i], lw=2, ms=7, mec=SURFACE, mew=2, label=name)
        w = wall[-1]; txt = f"{w / 60:.0f} min" if w >= 120 else f"{w:.0f} s"
        if Ns[-1] == 512:
            b.annotate(txt, (Ns[-1], w), xytext=(6, -4), textcoords="offset points", fontsize=7.5, color=INK2)
        else:
            b.annotate(txt, (Ns[-1], w), xytext=(-8, 8 if R == 1 else -12), textcoords="offset points", ha="right", fontsize=7.5, color=INK2)
    fl = sorted(FFTW_FLOOR.items())
    a.plot([k for k, _ in fl], [v for _, v in fl], color=INK2, lw=1.5)
    a.annotate("27 FFTs alone, FFTW on a\n192-core hpc7a.96xlarge", (fl[-1][0], fl[-1][1]), xytext=(-8, 6), textcoords="offset points",
               ha="right", fontsize=7.5, color=INK2)
    for ax, lab, title in ((a, "device time per RK3 step, ms (median over the run)", "Time per step"),
                           (b, "wall time of the run to t = 10, s (statistics every step)", "Time of a run")):
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks([64, 128, 256, 512]); ax.set_xticklabels(["64^3", "128^3", "256^3", "512^3"])
        ax.set_xlim(52, 700); ax.set_xlabel("grid", color=INK2, fontsize=9); ax.set_ylabel(lab, color=INK2, fontsize=9)
        ax.set_title(title, color=INK, fontsize=10, loc="left"); ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    a.set_ylim(3, 8000); b.set_ylim(1, 4000)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "performance.png"), facecolor=SURFACE); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rs = runs()
    accuracy(rs); performance(rs)
    print("wrote", os.path.join(OUT, "accuracy.png"), "and", os.path.join(OUT, "performance.png"), f"from {len(rs)} runs")
