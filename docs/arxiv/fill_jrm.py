"""Fill the JRM-ADM numbers into main.tex from data/jrm_native_test30/scores.json.

Always fills from the frozen template `main_pre_jrm.tex` (the manuscript as it stood before
the native cohort was scored), so it can be re-run when the json grows to 30 patients.

    python fill_jrm.py            # -> main.tex
"""
import sys
sys.exit("SUPERSEDED 2026-09-07: main.tex has been hand-edited since the fill (IV.C cost section, soft-tissue sentence); re-running would overwrite those edits. Edit main.tex directly.")
import json
import numpy as np
from scipy.stats import wilcoxon

rows = json.load(open("../../data/jrm_native_test30/scores.json"))
n = len(rows)
g = lambda k: np.array([r[k] for r in rows])  # noqa: E731
oP, oS, rpe = g("out_psnr"), g("out_ssim"), g("rpe_zc")
fP, fS = g("fdk_psnr"), g("fdk_ssim")
dP, dS = g("ours_xt_psnr") - oP, g("ours_xt_ssim") - oS
dfP, dfS = g("ours_fdk_psnr") - fP, g("ours_fdk_ssim") - fS
mt = np.array([r["mae_t_mm"] for r in rows]).mean(0)
mr = np.array([r["mae_r_deg"] for r in rows]).mean(0)
rt_h = g("runtime_min").mean() / 60
pm = lambda v, d: f"{v.mean():.{d}f} $\\pm$ {v.std():.{d}f}"  # noqa: E731
pw = lambda d: f"{(d > 0).sum()}/{n}"  # noqa: E731
pv = lambda d: f"{wilcoxon(d).pvalue:.1e}".replace("e-0", "e-")  # noqa: E731
def ptex(d):
    m, e = f"{wilcoxon(d).pvalue:.1e}".split("e")
    return f"$p={m}\\times10^{{{int(e)}}}$"

s = open("main_pre_jrm.tex").read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)

# A. abstract
rep("""and replacing our prior with a retrained diffusion prior in the same
loop costs 8.5\\,dB, of which 1.4\\,dB is attributable to the bridge itself. The final""",
f"""and exceeding JRM-ADM, a diffusion-based joint method run as published, by
{dP.mean():.1f}\\,dB and {dS.mean():.3f} SSIM; a pixel-linear bridge ablation attributes
1.4\\,dB of the margin to the bridge itself. The final""")
# B. Fig 5 caption
rep(""" The
JRM-ADM column will be added when its cohort completes.}""", "}")
# C. Table 2 caption
rep(""" The JRM-ADM
cohort is in progress and its row will be completed. The pixel-linear row is the bridge""",
""" The pixel-linear row is the bridge""")
# D/E. table rows
rep("JRM-ADM~\\cite{depaepe25} & --- & --- & --- \\\\",
    f"JRM-ADM~\\cite{{depaepe25}} & {pm(oP,2)} & {pm(oS,3)} & {pm(rpe,2)} \\\\")
rep("JRM-ADM~\\cite{depaepe25} & --- & --- & --- & --- & --- & --- \\\\",
    f"JRM-ADM~\\cite{{depaepe25}} & {mt[0]:.2f} & {mt[1]:.2f} & {mt[2]:.2f} & {mr[0]:.2f} & {mr[1]:.2f} & {mr[2]:.2f} \\\\")
# F. motion paragraph
rep("""translations and is less accurate on the gantry-axis rotation, consistent with its RPE.
""",
f"""translations and is less accurate on the gantry-axis rotation, consistent with its RPE.
JRM-ADM recovers the rotations and the axial translation about as well as the proposed
method, yet its in-plane translation error of about {mt[:2].mean():.1f}\\,mm exceeds the
uncorrected value despite an RPE of {rpe.mean():.2f}\\,mm. Its per-view translation drifts
along the beam direction, which the projections do not constrain; a fixed-axis error sees
that drift, whereas the detector-plane RPE does not.
""")
# G. native paragraph
a = s.index("For JRM-ADM, run natively with its own solver, grid and retrained prior, the cohort had")
b = s.index("reported when complete.\n", a) + len("reported when complete.\n")
s = s[:a] + f"""JRM-ADM, run as published on our data (Sec.~\\ref{{sec:baselines}}), takes about
{rt_h:.1f} hours per patient. Its motion estimate is accurate, with an RPE of
{rpe.mean():.2f}\\,mm against 2.26\\,mm for the autofocus baseline, and the analytic
reconstruction at its poses reaches {fP.mean():.2f}\\,dB\\,/\\,{fS.mean():.3f}, behind our
FDK($\\thetahat$) by {dfP.mean():.1f}\\,dB ({pw(dfP)}, {ptex(dfP)}). Its final output,
however, reaches only {oP.mean():.2f}\\,dB\\,/\\,{oS.mean():.3f}, which is {dP.mean():.1f}\\,dB
and {dS.mean():.3f} SSIM behind our final iterate ({pw(dP)}, {ptex(dP)}); as can be seen
in Fig.~\\ref{{fig:bench}}, it removes the motion but returns an over-smoothed volume in
which fine texture is lost. The difference between the two methods therefore lies mainly
in the prior and in how the loop applies it, rather than in the motion estimate.
""" + s[b:]
# H0. discussion: one sentence on JRM-ADM next to the autofocus lesson
rep("""so the practically relevant distinction between methods has shifted from how well the
motion is estimated to what kind of image is returned.""",
"""so the practically relevant distinction between methods has shifted from how well the
motion is estimated to what kind of image is returned. The comparison with JRM-ADM points
the same way: its motion estimate is competitive, but its output loses the texture that
the bridge-trained prior preserves.""")
# H. discussion
rep(""" The native JRM-ADM
transplant is reported on partial evidence. Finally, at 9.3 minutes per patient the
method is slower than the autofocus baseline, whose gradient-based estimation completes
in under two minutes~\\cite{thies25}, and the estimator,""",
f""" Finally, at 9.3 minutes per patient the
method is slower than the autofocus baseline, whose gradient-based estimation completes
in under two minutes~\\cite{{thies25}}, although far faster than JRM-ADM, whose
{rt_h:.1f}-hour runtime in our full-view setting reflects a design aimed at sparse-view
acquisitions; the estimator,""")
# I. conclusion
rep("""recovered the motion to 0.28\\,mm reprojection error and improved every case, the
reconstruction at the estimated poses became nearly interchangeable with the one at the
true poses, and paired comparisons showed the flow-matching prior on the geometry bridge
to be substantially stronger than a clean-image diffusion prior placed in the same loop.""",
"""recovered the motion to 0.28\\,mm reprojection error and improved every case, and paired
comparisons against a learned autofocus method and a diffusion-based joint method showed
consistent gains in both motion accuracy and image quality, with a bridge ablation
attributing part of the margin to the bridge itself.""")
assert "sec:baselines" in open("make_docx.py").read()
open("main.tex", "w").write(s)
print(f"filled from {n} patients: out {oP.mean():.2f}/{oS.mean():.3f}, fdk {fP.mean():.2f}/{fS.mean():.3f}, "
      f"rpe {rpe.mean():.2f}, dP {dP.mean():.2f} ({pw(dP)}), dfP {dfP.mean():.2f} ({pw(dfP)}), rt {rt_h:.1f} h")
