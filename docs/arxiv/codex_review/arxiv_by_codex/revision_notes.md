**Codex 개정본 · 2026-09-07**

원본: `/home/mirlab/Desktop/Flow_matching_motion_3D/docs/arxiv/main.tex`

제목 끝과 파일명에 `_by_codex`를 표시했다. 원본의 TEX, PDF, DOCX, 그림 및 생성 스크립트는 수정하지 않았다. 이 폴더의 그림은 복사본이며 Fig. 1만 문구와 개념 화살표를 조정했다.

- 초록·서론·Discussion·결론을 재작성하고 Methods와 Results를 전반적으로 축약했다.
- 수치표의 값, cohort, split, 비교 방법, 절과 그림의 순서를 유지했다. Table 3은 column minima 기준으로 bold를 조정했다.
- Diffusion sampling의 확률성 일반화, bridge와 inference 경로의 정확한 일치, FDK의 독립적 인증, motion bottleneck 해소 등의 주장을 정리했다.
- Eq. 1은 data fidelity를 정의하도록 바꾸고 implicit prior를 별도로 설명했다. Tangent의 chain rule을 명확히 하고 Algorithm에 finite solver와 CG 초기값을 명시했다.
- 최신 JRM-ADM까지 검증되었는지 불명확한 body-ROI 전체 순위·유의성 주장은 제외했다. Linear ablation의 불명확한 metric별 p-value·win count도 제외하고 평균 차이를 유지했다. 1.38 dB는 초록과 본문에서 약 1.4 dB로 표현했다.
- JRM-ADM의 grid·prior-weight adaptation과 runtime의 measured/reported 구분을 명시했다.
- 이번 개정에서 새 실험은 수행하지 않았다. 원고의 기존 보고값과 관련 로컬 문서·코드를 사용했다.

PDF 생성: `/home/mirlab/anaconda3/envs/texbuild/bin/tectonic main_by_codex.tex`

Word 생성: `/home/mirlab/anaconda3/envs/flow_matching/bin/python make_docx_by_codex.py`

원본 TEX SHA-256: `6c2ef4abc805fb10b82903fb826c63e259d2559f255829f97f0f55fdcf638f3b`
