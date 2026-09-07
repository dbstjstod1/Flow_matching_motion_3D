**arXiv 원고 수정 계획 · 2026-09-07**

검토 대상은 `/home/mirlab/Desktop/Flow_matching_motion_3D/docs/arxiv/main.tex`의 2026-09-07 16:30 버전이다. 초록부터 참고문헌까지 읽고, 개정 이력, 비교·motion 그림, bridge 학습과 inference 코드, JRM-ADM sampler 및 평가 스크립트의 관련 부분을 대조했다. 아래 줄 번호는 이 버전 기준이다. 원고 파일은 수정하지 않았다. 수치는 현재 원고의 보고값이며, 이번 검토에서 실험이나 cohort scoring을 재실행하지 않았다.

**1. 논문의 중심을 이렇게 잡는다**

이 논문은 head CBCT의 blind rigid-motion correction을 다룬다. 알려지지 않은 volume과 per-view 6-DoF motion을 함께 복원하면서, motion amplitude를 줄여 생성한 FDK 영상들로 flow-matching 학습 경로를 구성한다. 그 경로의 tangent를 projector의 geometry derivative로 구하고, 추론에서는 prior prediction, motion fitting, data consistency를 교대로 수행한다.

원고를 관통할 문장은 다음이 적절하다.

> We train a flow-matching prior on FDK reconstructions generated with progressively reduced motion and use it to guide joint image and motion estimation.

이 문장은 무엇을 학습하고 어디에 사용하는지를 한 번에 말한다. `direct`, `continuous`, `robust`, `correction-aligned`를 연달아 붙이지 않아도 차별점이 드러난다.

현재 근거는 세 층으로 나눠 읽어야 한다.

| 근거 | 현재 보고값 | 이 근거로 말할 수 있는 것 |
|---|---|---|
| 전체 방법의 성능 | 30명, noiseless simulated projections, 10 mm / 10° peak-to-peak; final 36.66 dB / 0.978 SSIM; RPE 0.28 mm | 이 평가 조건에서 image와 motion을 함께 개선했다. |
| 비교 방법 대비 성능 | autofocus RPE 2.26 mm, JRM-ADM RPE 0.84 mm; JRM-ADM final 29.64 dB / 0.912 | 같은 환자·motion·sinogram에서 비교한 전체 pipeline의 차이이다. |
| bridge 선택의 효과 | pixel-linear 35.29 dB / 0.966 / 0.42 mm; geometry 36.66 / 0.978 / 0.28 mm | 고정한 architecture·training recipe·inference loop 안에서 geometry bridge의 효과를 지지한다. |

논문의 개념적 기여를 가장 직접적으로 지지하는 것은 세 번째다. JRM-ADM 대비 7 dB의 전체 차이를 bridge 하나의 효과로 읽히게 해서는 안 된다. Motion-free FDK를 넘는 결과는 iterative reconstruction의 이점을 보여주는 보조 결과로 두는 편이 중심 주장과 잘 맞는다.

**2. 현재 문체가 과도하게 다듬어진 인상을 주는 이유**

특정 단어만의 문제가 아니다. 정의와 수치로 이미 전달한 내용을 다시 설득하는 문장이 많이 붙는다.

- **동일한 작동 원리의 반복:** improved image → better motion → better image가 서론, II.B 말미, II.C Estimate, ablation 해석, Discussion에 반복된다. 방법에서 충분히 설명하고, 결과에서는 관찰한 차이를 쓴다.
- **한 문장 안에 원인과 결론을 여러 번 연결:** `so`, `since`, `which in turn`, `therefore`가 이어지면서 방법 설명이 인과관계의 증명처럼 읽힌다. 한 문장은 하나의 주요 주장과 필요한 근거까지만 맡긴다.
- **추상적 장점의 중첩:** `direct and continuous`, `progressive joint refinement`, `bootstrap one another`가 정확한 연산을 대신한다. `predicts`, `fits`, `updates`로 바꾼다.
- **강한 부사와 보증 표현:** `exactly`, `any`, `nearly exact`, `certifies`, `only`가 수학적으로 성립하는 범위를 넘어간다. Bridge 정의의 정확성과 inference의 경험적 성능을 구분한다.
- **의인화·수사적 비유:** `learning-free witness`, `buys a better motion fit`, `inherits the ceiling`은 저자의 해석을 강조한다. 측정 대상과 연산으로 바꾼다.
- **그림·표를 소개하는 상투문:** `Table ... summarizes`, `As can be seen ...`를 매번 쓰지 않는다. 결과를 먼저 말하고 figure/table reference를 붙인다. 다만 독자에게 실제로 필요한 위치 안내는 남긴다.

단순 단문화도 피한다. 정의와 그 정의가 필요한 이유는 한 문장에 자연스럽게 둘 수 있다. 목표는 모든 문장을 짧게 만드는 것이 아니라, 한 문장에 겹쳐 있는 주장과 불필요한 해설을 덜어내는 것이다.

**3. 영어 표현을 다듬기 전에 정리할 주장**

| 우선순위·위치 | 현재 문제 | 수정 방향 |
|---|---|---|
| P0 · L95 | measurement가 volume과 pose에 `bilinearly` 의존한다고 기술 | `linear in the volume but nonlinear in the poses`로 수정. 현재 θ parameterization에 대한 bilinear 설명은 맞지 않는다. |
| P0 · L57–61, L119–130 | diffusion의 각 clean prediction을 stochastic sample로 일반화하고, 이것이 correction을 방해한다고 단정 | 학습 경로의 차이에 집중. JRM-ADM의 로컬 설정은 DDIM η=0이고 noise term의 계수가 0이다. 초기 random state와 매 단계 fresh stochastic perturbation은 구분해야 한다. |
| P0 · L157–160, L244–249, L659–660, L687–689 | 학습 bridge와 inference state가 정확히 일치하고, data step이 bridge로 되돌린다고 서술 | 학습은 reduced-motion re-simulation, 추론은 고정된 y에 대한 pose fitting 및 CG/TV이다. `designed to represent progressive motion correction`, `guides the joint updates`로 제한한다. |
| P0 · L464–480, L668–670 | FDK(θ̂)를 `learning-free`, geometry의 독립적 증명으로 표현 | FDK 자체에는 neural image update가 없지만 θ̂는 learned prior의 영향을 받는다. `evaluates the estimated poses through analytic reconstruction`가 정확하다. |
| P0 · L76–78, L694–695 | motion-free FDK를 넘었으므로 residual motion이 더 이상 limiting factor가 아니라고 결론 | 이 결과만으로 motion의 영향이 사라졌다고 판단할 수 없다. 알려진 motion으로 같은 loop를 수행한 대조가 필요하다. 현재 결과는 `benefit of iterative reconstruction beyond analytic motion compensation`로 해석한다. |
| P1 · L189–196, L266–268 | explicit R이 없는 velocity update를 하나의 objective에 대한 block-coordinate optimization으로 단정 | Data fidelity, TV, learned velocity를 결합한 algorithmic formulation으로 설명. objective를 유지하면 formal motivation임을 밝히고 그 목적함수의 descent·convergence는 주장하지 않는다. |
| P1 · Algorithm 1, L281 | 5회 CG가 exact argmin처럼 보이고 초기값이 빠짐 | `z ← CG₅(Aθ̂, y; z₀=x_pred)`처럼 finite update와 warm start를 명시. 실제 코드는 prior prediction으로 초기화한다. |
| P1 · L581–583 | beam 방향 translation을 projection이 전혀 constrain하지 않는다고 단정 | `weakly constrained`, `RPE is less sensitive to ...`로 수정. 국소적으로 작은 민감도와 정확한 null space는 다른 주장이다. |
| P1 · L585–598 | 전체 JRM-ADM 비교로 차이의 원인이 주로 prior라고 결론 | observation은 residual smoothing과 성능 차이까지. Prior, estimator, solver, output rule이 달라서 원인 분리는 못 했음을 Discussion에서 짧게 구분한다. |
| P1 · L605–607 | prior 단독보다 loop에서 차이가 커진다고 쓰지만 본문에 단독 수치·실험이 없음 | 해당 비교를 삭제하거나 근거를 제시. Feedback 설명을 남기면 plausible interpretation으로 한정한다. |
| P1 · L415–420 | `run as published`, `unchanged`, more measurements라서 문제가 없다는 논리 | `authors' implementation, with the prior retrained on our training split` 정도로 정확히 기술. 문서에 기록된 grid 및 weight adaptation은 짧은 protocol 설명이나 supplement에 연결한다. |
| P1 · Table 3 | Proposed 전체 행 bold이지만 tx·ty는 pixel-linear가 작음 | 최고값 기준이면 column별 bold. 방법 강조 기준이면 label만 bold. 본문에서도 모든 DoF에서 우세한 것처럼 쓰지 않는다. |
| P1 · Table 4, L634–652 | 같은 GPU·조건에서 측정한 표처럼 보이지만 autofocus는 출판값; network calls도 patch batch와 전체 volume 평가를 혼동 | measured/reported runtime 구분. `network calls` 대신 의도한 단위인 `prior/quality evaluations`를 정의. Evaluation count를 FLOPs나 동일 비용의 작업으로 읽히게 하지 않는다. |

DDIM의 deterministic sampling 가능성은 [원 논문](https://arxiv.org/html/2010.02502v4)으로 확인했다. 로컬 JRM 설정은 `refs/jrm-adm/config/adm_jrm.yaml:41`, 실제 σ=η×… 및 noise term은 `src/sampler/diffusion_utils.py:75`에 있다. 따라서 핵심 대비를 “stochastic diffusion 대 deterministic flow”로 두는 것은 피한다.

**4. 절별 수정 순서와 분량**

| 대상 | 수정 계획 | 분량 방향 |
|---|---|---|
| Title | 현재 제목의 `Prior on the Geometry Bridge`를 더 자연스러운 명사구로 정리. 예: `Rigid-Motion Correction in Head Cone-Beam CT via Flow Matching on a Geometry Bridge`. 3D를 강조할 필요가 있으면 기존 표현을 유지한다. | 선택 사항. 제목보다 본문 논리 우선. |
| Abstract | 문제 1문장 → geometry bridge 2문장 → tangent와 inference 2문장 → 조건과 주요 결과 → ablation의 의미. 긴 diffusion 비판과 FDK ceiling 결론은 제외. | 약 200–230 words를 목표로 재작성. |
| Introduction | 기존의 problem → autofocus → generative methods → proposed bridge → contributions 흐름 유지. 첫 문단의 gauge 상세는 Methods로 옮기고, generative 문단은 JRM의 실제 작동과 path 차이까지만 설명. | 현재보다 약 20–30% 축약 목표. |
| II.A | Forward model 및 pose convention을 정확히 정의. Implicit prior를 explicit regularizer처럼 다루는 문장 정리. Gauge의 원리와 scoring convention을 중복하지 않기. | 핵심 정의 유지. |
| II.B | Bridge 정의 → endpoints → analytic tangent → loss → patch conditioning 순서. `same scan`은 actual y를 단계별 수정한다는 오해가 없도록 `same volume, re-simulated at reduced motion`으로 설명. | 마지막 feedback 반복 문단 축약. |
| II.C | Predict / Estimate / Correct 구조 유지. 각 문단은 입력, 수행 연산, 출력에 집중. Algorithm에 finite solver budget과 CG initial state 추가. | 설명 약 15–20% 축약 목표. |
| Experiments | Dataset, simulation, paired evaluation, baseline adaptation을 짧고 재현 가능하게 기술. 평가 대상이 143명 test split 중 30명임을 분명히 한다. | 필수 정보는 줄이지 않음. |
| IV.A | Final image와 recovered motion의 관찰 결과, 두 output의 용도만 설명. Static FDK 비교는 한 번만 해석. | FDK를 반복 변호하는 문장 축약. |
| IV.B | 기존 그림·표 배치는 유지. 본문을 autofocus → JRM-ADM → component-wise motion → bridge ablation의 논리 순으로 정리할 수 있으나 기존 layout 선호를 우선한다. 숫자 재나열보다 차이의 의미를 쓴다. | 비교 방법별 핵심 결과 1문단. |
| IV.C | 9.3 min, estimator 비용, protocol-specific comparison을 남김. Hardware·operator별 비용을 완전히 분해한 것처럼 설명하지 않음. | 상세 iteration arithmetic는 caption 또는 supplement로 이동. |
| Discussion | Bridge ablation이 지지하는 결론 → full-method comparison의 해석 범위 → 남은 한계. Results 수치와 mutual-refinement 설명 반복 제거. | 3개 역할이 분명한 문단. |
| Conclusion | 수행한 일, 관찰한 성과, 결과가 지지하는 의미로 3문장. | 새로운 causal claim과 ceiling 논리 제외. |

이 비율은 편집 목표이며 기계적으로 맞출 규칙은 아니다. 먼저 주장과 중복을 정리한 뒤 자연스러운 문장 길이를 택한다.

**5. 실제 교체할 표현 예시**

아래는 최종 원고에 적용하기 전 문맥에 맞춰 다듬을 후보이다. 의미를 유지하는 축약과, 과도한 주장의 교정을 구분해 적었다.

| 위치·원문 | 수정 후보 | 목적 |
|---|---|---|
| L53 `Correcting it blindly is a coupled problem` | `Blind correction requires joint estimation of the image and per-view poses.` | 문제를 직접 정의. |
| L95 `the measurement depends bilinearly on the pair` | `The forward model is linear in the volume but nonlinear in the poses.` | 수학적 정확성 교정. |
| L132 `We take a different route.` | 삭제하고 `We define a geometry bridge ...`로 다음 문장 시작 | 내용 없는 전환 축소. |
| L142 `makes the refinement direct and continuous, lowers the computational overhead, and improves the robustness` | `The learned velocity updates the image before each motion fit.` | 여러 장점의 선제 주장 대신 실제 동작. |
| L173 `bootstrap one another through progressive joint refinement` | `alternate image updates and motion estimation` | 중복된 추상 표현 축소. |
| L218 `a sequence of genuine FDK reconstructions ...` | `FDK reconstructions simulated at decreasing motion amplitudes` | genuine 같은 가치 표현 제거. |
| L225 `the best image this scanner and this operator can produce of a still patient` | `the motion-free FDK reconstruction at the nominal geometry` | endpoint의 정확한 정의. |
| L246 `moves any partially corrected reconstruction toward a less corrupted one` | `is trained to predict correction directions along the bridge` | 학습 목표와 모든 입력에 대한 보증 구분. |
| L299 `needs no explicit smoothness penalty` | `We parameterize the trajectory with a hash-encoded MLP and use no explicit smoothness penalty.` | 설계 선택을 일반적 필요성 부정으로 표현하지 않음. |
| L311 `which an adjoint step barely touches` | `We use five CG iterations for the data-consistency update.` | 입증되지 않은 비교 수사 삭제. |
| L458 `the recovered motion is nearly exact` | `The final mean RPE was 0.28 mm.` | 데이터가 정확도를 말하게 함. |
| L477 `a learning-free witness of the recovered geometry` | `an analytic reconstruction at the estimated poses` | prior로 추정한 θ̂의 의존성 보존. |
| L582 `which the projections do not constrain` | `to which the projections are less sensitive` | 근사적 취약성과 비관측성 구분. |
| L590 `reaches only 29.64 dB` | `achieved 29.64 dB` | 비교 결과의 평가적 어조 제거. |
| L605–607 `buys a better motion fit, which buys a cleaner data step` | `This gain is consistent with improved image estimates supporting subsequent motion updates.` | 메커니즘을 해석으로 한정. 더 간결하게는 문장 삭제. |
| L659 `defined exactly on the states a blind correction loop traverses` | `trained on reconstructions generated with progressively reduced motion` | training과 inference의 정확한 일치 주장 제거. |
| L670 `the analytic one certifies the geometry` | `The analytic output provides a complementary assessment of the estimated poses.` | 독립적 인증이라는 과장 축소. |
| L694–695 `further progress ... lies with the reconstruction rather than ... motion estimate` | `The gain over motion-free FDK indicates the benefit of iterative reconstruction beyond analytic motion compensation.` | 결과가 실제 지지하는 결론으로 조정. |

`exact`는 금지할 단어가 아니다. 정해진 discrete projector에 대해 유도한 derivative라는 기술적 의미에서는 남긴다. `exact trajectory`, `exact state matching`, `exact recovery`처럼 증명되지 않은 범위로 확장하지 않는다. `closed-form` 역시 path derivative의 analytic expression을 가리키도록 쓰고, 네트워크의 추론해가 closed-form이라는 오해를 막는다.

**6. 원하는 문체를 보여주는 Abstract 초안**

> Blind motion correction in head cone-beam CT requires joint estimation of the image and per-view rigid poses. We propose a 3D flow-matching prior trained on a geometry bridge: FDK reconstructions generated from the same volume as the simulated motion amplitude decreases linearly to zero. The bridge connects the uncorrected reconstruction to its motion-free FDK counterpart, with an analytic velocity target derived from the forward projector's geometry derivative. At inference, a predictor-corrector loop alternates image updates from the frozen prior, motion fitting with a hash-encoded network, and data-consistency updates. We evaluated the method on simulated projections from 30 held-out CQ500 patients with 10 mm / 10° peak-to-peak rigid motion. The final reconstructions achieved 36.66 dB PSNR and 0.978 SSIM, while mean reprojection error decreased from 6.12 to 0.28 mm. Reprojection error was 2.26 mm for learned autofocus and 0.84 mm for JRM-ADM; the proposed reconstructions exceeded JRM-ADM by 7.0 dB and 0.066 SSIM under the same evaluation protocol. Inference took 9.3 minutes per patient on an RTX A6000. Replacing the geometry bridge with a pixel-linear path reduced PSNR by approximately 1.4 dB, supporting the use of a motion-based training path for joint image and motion recovery.

이 초안은 서론의 diffusion critique를 초록에서 반복하지 않고, 방법의 구체적 차이와 이를 뒷받침하는 결과를 바로 제시한다. Simulation 조건을 명시하되 제한사항을 나열하지 않는다. 1.4 dB는 abstract 수준의 반올림이다. 본문의 1.38 dB와 표의 반올림 수치 차이 1.37 dB는 원시 paired result를 기준으로 최종 통일한다.

**7. 그림과 본문을 같이 다듬을 부분**

- Fig. 1의 caption은 학습 경로의 개념도임을 분명히 한다. CG가 매번 orange bridge 위로 정확히 복귀하는 것처럼 보이는 그림과 `converge together as t→1`을 함께 수정한다. Flow time의 끝과 수렴 보증은 구분한다.
- Fig. 1 왼쪽의 `re-noise`를 JRM-ADM의 모든 step에 새 random noise가 들어간다는 뜻으로 설명하지 않는다. Sampler update와 predicted noise component의 역할을 정확히 반영한다.
- Fig. 2는 panel별 정의와 표시 내용만 설명한다. Abstract·Introduction·Methods에 이미 있는 장점 설명은 caption에서 반복하지 않는다.
- Fig. 5에서 JRM-ADM의 smoothing은 관찰할 수 있으나 `restores grey-white matter differentiation`은 정량적·임상적 복원 보증처럼 읽힐 수 있다. `preserved more soft-tissue contrast in the displayed cases` 정도로 범위를 맞춘다. Proposed에도 ground truth 대비 smoothing은 남아 보이므로 완전한 texture recovery를 주장하지 않는다.
- Fig. 6과 Table 3은 RPE 한 숫자로 나타나지 않는 오차를 보여주는 보완 자료로 해석한다. Geometry bridge가 모든 translation component에서 pixel-linear보다 낫다는 서술은 피한다.
- `static FDK`, `FDK at true motion`, `FDK at estimated motion`, `final iterate`를 명확히 구분한다. Static FDK는 nominal motion-free acquisition이고 FDK at true motion은 motion-corrupted measurement를 true geometry로 재구성한 것이다. 둘을 같은 reference처럼 부르지 않는다.

**8. 최종 문장 수정 전에 확인할 근거와 표기**

이 작업은 새로운 실험을 요구하는 계획이 아니라, 현재 근거로 쓸 수 있는 문장을 선택하기 위한 확인 목록이다. 근거가 없으면 문장을 축소하거나 삭제하는 것으로 마무리할 수 있다.

- `preserves every ordering and significance`라는 body-ROI 문장은 최신 native JRM-ADM까지 실제로 포함했는지 확인한다. 개정 이력의 기존 body-ROI audit는 prior-swap W3DM 시기에 작성되었고, 현재 확인한 native scoring script는 measured-FOV scoring이다. 두 결과를 혼용하지 않는다.
- Paired p-value가 어떤 metric에 해당하는지 명시한다. PSNR·SSIM 뒤에 p 하나를 붙여 두 지표 모두에 적용되는 것처럼 쓰지 않는다. 실제 script의 Wilcoxon 검정을 III.B에 간단히 정의한다.
- `improves every case`가 image quality, RPE, baseline 대비 중 무엇인지 문장별로 명확히 한다. Bridge ablation의 27/30과 전체 방법의 30/30을 섞지 않는다.
- LEAP operator와 저자가 구현한 geometry derivative의 역할을 구분한다. 로컬 `fm3d/leap_projector.py:5`는 geometry derivative가 별도 구현이라고 명시한다. 한 문장만 보완해도 기여의 출처가 더 정확해진다.
- Tangent 식의 `dot T`가 6-DoF 전체 directional derivative인지 rotation generator인지 명확히 한다. `P_t=P_nom T((1−t)θ)`, `dot y_t=D_P A(x;P_t)[dot P_t]` 표기를 쓰면 chain rule이 더 읽기 쉽다.
- Inference image는 `x^(k)`, training bridge sample은 `x_t`, final output은 `x^(N)` 또는 `x_final`처럼 구별하는 방안을 검토한다. 현재 표의 `x_t (final iterate)`는 끝난 결과에도 variable time index를 남기고, bridge sample과 혼동시킨다.
- Runtime의 30배 차이를 flow matching의 본질적 속도 이점으로 단정하지 않는다. 현재 측정은 서로 다른 operator·solver·resolution을 포함한 구현 전체의 비교이다.
- `first` novelty claim을 남기려면 별도 관련연구 점검이 필요하다. 이번 검토는 포괄적 novelty search를 수행하지 않았다. 그 문장을 빼도 bridge 정의와 ablation으로 기여를 설명할 수 있다.

**9. 실제 개정 작업의 완료 기준**

1. Abstract와 Introduction을 읽은 독자가 bridge의 정의와 이 논문의 검증 범위를 한 번에 파악한다.
2. Methods의 연산이 Algorithm과 일치하고, finite updates를 exact optimization 또는 convergence guarantee로 표현하지 않는다.
3. Results의 각 문단은 표를 다시 읽는 대신 하나의 관찰을 설명한다. Observation, interpretation, speculation의 문장 강도가 다르다.
4. Discussion은 `bridge의 효과`, `전체 pipeline 비교의 한계`, `실제 데이터로의 확장`을 다루며 서론의 mutual-refinement 논리를 반복하지 않는다.
5. 기존 결과값·환자 split·평가 convention·사용자가 정한 figure layout을 유지한다. 이전 버전의 prior-swap 결과나 철회한 주장을 다시 넣지 않는다.
6. 최종 diff에서 문체 수정과 과학적 의미 수정이 구분된다. 숫자와 reference를 점검하고 PDF·DOCX를 함께 재생성한다. 최신 README에 따라 `main.tex`를 직접 수정하며, 현재 guarded 상태인 `fill_jrm.py`로 옛 template를 덮어쓰지 않는다.

핵심 원칙: 방법은 연산으로 설명하고, 성능은 수치로 보여주고, 해석은 근거가 허용하는 데까지만 쓴다.
