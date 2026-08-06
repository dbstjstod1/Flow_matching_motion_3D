::: {custom-style="Title"}
Blind rigid-motion correction in head cone-beam CT with a flow-matching prior on the geometry bridge
:::

::: {custom-style="Author"}
Author One^a^, Author Two^a^, and Author Three^a,b^
:::

::: {custom-style="Affiliation"}
^a^Department, Institution, Address, City, Country

^b^Department, Institution, Address, City, Country
:::

# ABSTRACT {.unnumbered}

Patient motion during a cone-beam CT (CBCT) head scan corrupts the reconstruction with streaks and blurring, and correcting it is a *blind* inverse problem: neither the image nor the per-view rigid pose is known, and the measurement is bilinear in the two. We propose a blind motion-correction method built on a flow-matching prior trained on a **geometry bridge** — a path of FDK reconstructions of a scan re-simulated with a linearly decaying fraction of the true motion, running from the uncorrected reconstruction to the motion-free one with no endpoint correction. Because FDK is linear in the sinogram, the regression target is available in closed form as the forward projector's exact geometry derivative carried through it. At test time a 50-step predictor–corrector loop alternates a prior step, a 6-DoF-per-view motion fit on the improved image, and a plug-and-play data-consistency step. On 30 held-out CQ500 patients with Akima-spline motion of 10 mm / 10° peak-to-peak it cuts the mean reprojection error from 6.12 mm to 0.28 mm and raises SSIM from 0.546 to 0.978, improving every patient and landing within 0.005 SSIM of the same reconstruction given the ground-truth motion. *[PLACEHOLDER — these figures are the geometry-bridge (version A) results and must be replaced once the motion-correction prior is trained.]*

**Keywords:** cone-beam CT, rigid motion correction, flow matching, generative priors, plug-and-play, inverse problems
