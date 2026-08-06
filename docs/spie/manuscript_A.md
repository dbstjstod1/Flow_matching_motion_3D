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

# 1. INTRODUCTION

Cone-beam CT serves head and neck imaging in image-guided radiotherapy, interventional suites and point-of-care settings. Scan times run to tens of seconds, and head motion in that window violates the static-object assumption behind analytic reconstruction, producing streaking and blurring severe enough to destroy diagnostic value. The skull being rigid, the motion is well modelled by a 6-DoF pose per view — compactly parameterized, but not easily recovered.

The problem is **blind**. Writing the scan as $y = A_{P(\theta)}x$, neither the volume $x$ nor the trajectory $\theta$ is known, and the measurement is bilinear in the pair: correcting the geometry needs a clean image, and reconstructing one needs the correct geometry. The joint objective is non-convex and carries an exact SE(3) gauge — a global rigid pose on both object and orbit leaves the sinogram unchanged — so $\theta$ is identifiable only up to a rigid transform. Classical remedies add hardware or optimize an image-quality surrogate such as an autofocus sharpness criterion;^1^ learning-based work replaces that surrogate with a trained quality metric driving a gradient-based search.^2^ These estimate motion well, but the image returned is still an analytic reconstruction and inherits that operator's limits. A second line uses a generative model of clean CT as a plug-and-play prior inside an iterative solver,^3,4^ but a prior trained on *clean* images is then asked, at every early iteration, to judge heavily streaked images far outside its training distribution.

We take a different route. Instead of a prior on clean images, we train a flow-matching prior^5,6^ on the **geometry bridge**, the family $x_t = \mathrm{FDK}(y,\,P_\mathrm{nom}T(t\theta)) + t\Delta$, whose residual motion decays linearly from the full amount at $t=0$ to zero at $t=1$. Every point on it is a genuine reconstruction of the measured data — exactly the images a blind solver walks through as its estimate converges. We contribute (i) a flow-matching prior on this bridge, whose velocity target is **analytic** — the exact derivative of the backprojector with respect to the projection matrices — needing no finite differencing; (ii) a blind predictor–corrector loop in which prior, motion estimator and data step bootstrap one another; and (iii) an evaluation on 30 held-out patients under an explicitly stated gauge convention.

# 2. METHOD

## 2.1 Forward model

Let $P_\mathrm{nom}$ be the nominal circular cone-beam geometry for $V$ views and $\theta\in\mathbb{R}^{V\times6}$ the per-view rigid pose (axis–angle rotation and translation). View $v$ is acquired at $P_\mathrm{nom}[v]\,T(\theta_v)$, and the scan is $y = A_{P(\theta)}x$. We seek

$$\min_{x,\;\theta}\;\tfrac12\big\|A_{P(\theta)}x - y\big\|_2^2 \;+\; \lambda\,\mathrm{TV}(x) \;+\; \mathcal{R}(x),$$

with $\mathcal{R}$ the learned prior. Two facts govern the design: per-view translation *along the beam axis* is nearly unobservable, so we report the observable component separately rather than a single translation RMSE; and the SE(3) gauge means images must be rigidly aligned before PSNR/SSIM.

## 2.2 A flow-matching prior on the geometry bridge

Flow matching learns a velocity field $v_\phi(x,t)$ transporting one distribution to another along a prescribed path, by regressing $v_\phi(x_t,t)$ onto $dx_t/dt$. We set that path to the geometry bridge of Fig. 1(a): $x_t$ is the FDK reconstruction of the *measured* sinogram at the geometry $P_\mathrm{nom}T(t\theta)$. Because the bridge scales the axis–angle vector, the residual left between the geometry the data was acquired at and the one it is reconstructed at is exactly $(1-t)\theta$ in rotation, and a fraction $(1-t)$ of the true translation in magnitude: the path **sweeps the motion amplitude linearly to zero**, starting at $t=0$ from the uncorrected reconstruction that inference begins with.

![**Figure 1.** (a) The geometry bridge: each panel is an actual FDK reconstruction of the actual measured scan at geometry $P_\mathrm{nom}T(t\theta)$, which leaves exactly $(1-t)$ of the motion, so the path ends on the motion-free static FDK. The velocity target $dx_t/dt$ is the exact derivative of the backprojector with respect to that geometry. (b) One step of the blind inference loop.](/home/mirlab/Desktop/Flow_matching_motion_3D/figs/spie/fig1_method.png){width=5.3in}

Zero residual motion is still not a motion-free image: FDK is an analytic inverse for a circular, equiangular orbit and the moved orbit is not one, so FDK at the *true* motion sits 1–3 dB below the static reconstruction. We therefore anchor the path with a first-order detrend $\Delta = x_\mathrm{static} - \mathrm{FDK}(y,P(\theta))$, which vanishes at $t=0$ and makes $x_1$ **exactly the static FDK**, leaving the geometry term dominant in between.

The decisive property is that the tangent is **closed-form**. Since $\Delta$ is constant in $t$,

$$\frac{dx_t}{dt} \;=\; \frac{\partial}{\partial t}\,\mathrm{FDK}\big(y,\,P_\mathrm{nom}T(t\theta)\big) \;+\; \Delta,$$

and the first term is the exact derivative of the backprojector with respect to its projection matrices. One fused pass accumulates the reconstruction and its directional derivative from the same filtered detector taps, so a training sample costs no extra reconstruction. The loss is $\|v_\phi(x_t,t) - dx_t/dt\|^2$ with $t\sim\mathcal{U}[0,1]$. The velocity field is a 3D U-Net on $32^3$ patches; following Ref. 7 each patch carries four conditioning channels besides its own intensities — a downsampled view of the whole volume and three coordinate maps — supplying global context a $32^3$ receptive field cannot see.

## 2.3 The blind posterior loop

Inference (Fig. 1(b)) starts cold, from $x=\mathrm{FDK}(y,P_\mathrm{nom})$ and $\hat\theta=0$, and takes $N=50$ Euler steps, each a Gauss–Seidel sweep of three moves. **Predict:** the frozen prior moves first, $x_\mathrm{pred} = x + \Delta t\,v_\phi(x,t)$, evaluated patch-wise and blended over two tilings. **Estimate:** the motion is refit on the image the prior just improved, $\hat\theta = \arg\min_\theta\|A_{P_\mathrm{nom}T(\theta)}x_\mathrm{pred} - y\|_2^2$, with Adam on the projector's exact geometry gradient. The poses are not free per-view parameters: they are the output of a multiresolution hash encoding^8^ of the normalized view index followed by a small MLP, so the trajectory is a continuous function of the view and carries no explicit smoothness penalty. Adam's state persists across flow steps, so each step warm-starts from the last. Fitting on $x_\mathrm{pred}$ rather than $x$ matters: the estimator is only as good as the image it is handed. **Correct:** a plug-and-play forward–backward step enforces consistency at $\hat\theta$ — five conjugate-gradient iterations on $\min_z\|A_{\hat\theta}z-y\|^2$ — then a relaxed total-variation move $z\leftarrow z + \kappa(\mathrm{TV}(z)-z)$. A *filtered* solve rather than a plain adjoint step is essential: the adjoint direction is dominated by low frequencies, whereas the payoff from a better $\hat\theta$ lies in streaks and edges. TV, not the learned denoiser, is used here because the data-prox pushes $z$ off the manifold the network knows. The loop returns FDK($\hat\theta$) and $x_t$; we report both.

## 2.4 Implementation

The prior was trained 500k iterations (AdamW, cosine $10^{-4}\!\to\!10^{-6}$, EMA 0.999, fp16, 64 patches per batch). Projection and backprojection use a differentiable modular-beam projector^9^ pinned to its Joseph kernel, so one operator serves simulation, the estimator gradient and the data step. The estimator runs 200 Adam iterations per flow step on 24 random views, warm-started, at half resolution until $t=0.5$. Its encoder is a stock-resolution hash grid (16 levels, base 16, growth 1.5) at learning rate $3\times10^{-3}$: encoder bandwidth and step size act as one knob, and a larger step spends the extra capacity on view-to-view jitter.


# 3. EXPERIMENTS

We use the CQ500 head CT collection,^10^ 343 patients split 150 / 50 / 143 into train / validation / test, at $256^3$ and 1 mm. Projections are simulated on a native $612^3$ grid at 0.42 mm — finer than the reconstruction grid, avoiding an inverse crime — through the standard head-CBCT geometry: SOD 785 mm, SDD 1200 mm, a 700 × 500 panel at 0.64 mm pitch, 360 views over a full turn. Following Ref. 2, per-view motion is a zero-centred Akima spline^11^ through 10 random nodes per DoF; training draws per-DoF amplitudes up to 15 mm / 20° peak-to-peak, evaluation a fixed 10 mm / 10° pattern per patient (all peak-to-peak). Following the 30-patient protocol of Ref. 2 we score 30 patients from our held-out test split, one pattern each. Images are rigidly aligned before PSNR and SSIM because of the SE(3) gauge, and motion accuracy is the reprojection error (RPE) — the mean detector displacement induced by the pose error — quoted under a **zero-mean gauge**, which uses no ground truth and is therefore honestly reportable.

# 4. RESULTS

Table 1 gives the cohort mean ± standard deviation. Uncorrected reconstruction sits at 23.29 dB / 0.546 SSIM; $x_t$ reaches **36.60 dB / 0.978**, a gain of 13.3 dB and 0.432 SSIM, improving **every one of the 30 patients** (Fig. 3(a)). Mean RPE falls from 6.12 mm to **0.28 mm**, a 22-fold reduction (Fig. 3(b)).

| Method | PSNR (dB) | SSIM | RPE (mm) |
|:---|:---:|:---:|:---:|
| Uncorrected FDK | 23.29 ± 0.93 | 0.546 ± 0.035 | 6.12 ± 0.66 |
| Ours — FDK($\hat\theta$) | 31.76 ± 0.80 | 0.753 ± 0.032 | 0.28 ± 0.10 |
| **Ours — $x_t$ (carried PnP state)** | **36.60 ± 1.68** | **0.978 ± 0.008** | **0.28 ± 0.10** |
| *Reference:* FDK at the ground-truth motion | 32.38 ± 1.08 | 0.757 ± 0.033 | 0 |
| *Reference:* motion-free FDK | 33.11 ± 1.50 | 0.832 ± 0.032 | — |

: **Table 1.** Blind rigid-motion correction on 30 held-out CQ500 patients, Akima motion at 10 mm / 10° peak-to-peak. PSNR and SSIM are computed against the ground-truth volume after rigid alignment (mean ± sample s.d.). RPE is quoted under the zero-mean gauge. The last two rows are references, not competing methods.

Two comparisons locate the remaining error. First, FDK($\hat\theta$) reaches 0.753 SSIM against 0.757 for the FDK given the ground-truth motion — a gap of 0.005, so the geometry estimate is essentially exhausted. Second, $x_t$ exceeds not only FDK($\hat\theta$) by 4.8 dB but the motion-free FDK reference by 3.5 dB and 0.146 SSIM: the analytic operator, not residual motion, is the binding constraint, and $x_t$ bypasses it because the same loop also regularizes the artefacts FDK leaves behind. Residual pose error is 0.27° rotation RMSE (from 3.63°) and 0.083 mm in the observable translation (from 1.68 mm); the raw, ungauged convention gives 1.71 ± 0.55 mm, the difference being the unobservable global pose. Inference costs ~9.3 min per patient on one RTX A6000, the estimator taking roughly 80%.


![**Figure 2.** A representative test patient at 10 mm / 10° peak-to-peak motion, brain window (level 40 / width 420 HU); volumes are rigidly aligned to the ground truth before display and scoring. FDK($\hat\theta$) is indistinguishable from the FDK reconstructed with the ground-truth motion, yet both keep artefacts that only $x_t$ removes.](/home/mirlab/Desktop/Flow_matching_motion_3D/figs/spie/fig2_images.png){width=4.3in}

![**Figure 3.** Cohort results over 30 held-out patients. (a) Per-patient SSIM; grey stems join each patient's uncorrected and corrected score. (b) Reprojection error against flow time (thin: individual patients).](/home/mirlab/Desktop/Flow_matching_motion_3D/figs/spie/fig3_quant.png){width=3.8in}

# 5. DISCUSSION AND CONCLUSION

Training the prior on the geometry bridge rather than on clean images removes the distribution mismatch that troubles plug-and-play generative priors in blind problems: at every flow time the network sees exactly the kind of partly-corrupted reconstruction it was trained on, and the closed-form velocity target makes that path essentially free to train on. The motion-estimation half of the problem appears close to solved at this amplitude; what remains is the reconstruction operator itself, and the carried state overtaking even the motion-free analytic reference indicates the right deliverable is the iterate, not an analytic reconstruction.

Limitations bound these claims. The motion is simulated, and clinical validation is the essential next step; the model is rigid-only, suiting the skull but not the mandible or neck; and our evaluation amplitude is twice that of Ref. 2, so a controlled head-to-head benchmark is left to future work. Inference at ~9 min per patient is dominated by the estimator, the obvious target for acceleration.

# REFERENCES {.unnumbered}

1. Sisniega, A., et al., "Motion compensation in extremity cone-beam CT using a penalized image sharpness criterion," *Phys. Med. Biol.* **62**(9), 3712–3734 (2017).
2. Thies, M., et al., "A gradient-based approach to fast and accurate head motion compensation in cone-beam CT," *IEEE Trans. Med. Imaging* (2025).
3. Venkatakrishnan, S. V., Bouman, C. A., and Wohlberg, B., "Plug-and-play priors for model based reconstruction," in *IEEE GlobalSIP*, 945–948 (2013).
4. Chung, H., Lee, S., and Ye, J. C., "Decomposed diffusion sampler for accelerating large-scale inverse problems," in *ICLR* (2024).
5. Lipman, Y., et al., "Flow matching for generative modeling," in *ICLR* (2023).
6. Delbracio, M. and Milanfar, P., "Inversion by direct iteration," *Trans. Mach. Learn. Res.* (2023).
7. Author(s), "Patch-based 3D flow matching with global context conditioning," arXiv:2512.18161 (2025).
8. Müller, T., et al., "Instant neural graphics primitives with a multiresolution hash encoding," *ACM Trans. Graph.* **41**(4), 102 (2022).
9. Kim, H. and Champley, K., "Differentiable forward projector for X-ray computed tomography," arXiv:2307.05801 (2023).
10. Chilamkurthy, S., et al., "Deep learning algorithms for detection of critical findings in head CT scans," *The Lancet* **392**(10162), 2388–2396 (2018).
11. Akima, H., "A new method of interpolation and smooth curve fitting based on local procedures," *J. ACM* **17**(4), 589–602 (1970).
