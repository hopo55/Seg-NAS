# Hyundai 방법론 개선 계획 (2025+ 논문 대비)

## 현재 방법론 핵심 기술 현황

| 구성 요소 | 현재 구현 | 파일 |
|---|---|---|
| 인코더 | DenseNet121 (pretrained, frozen) | `hyundai/nas/supernet_dense.py` |
| 디코더 | 5단 MixedOp/MixedOpWithWidth (skip-add) | `hyundai/nas/supernet_dense.py` |
| 검색 공간 | 7 ops × 3 widths = 21^5 ≈ 4.1M | `hyundai/nas/search_space.py` |
| 최적화 | DARTS bilateral (weight/alpha 교대) | `hyundai/nas/train_supernet.py` |
| 지연 모델 | 실측 LUT (6 HW, IQR outlier removal) | `hyundai/latency/lut_builder.py` |
| Pareto 탐색 | 샘플링 → weight-sharing 평가 → LUT latency → 비지배 정렬 | `hyundai/nas/pareto_search.py` |
| Cross-HW 예측기 | HW-Arch cross-attention + few-shot adaptation | `hyundai/latency/latency_predictor.py` |

---

## 코드 기반 핵심 Gap 발견 (논문 대비 치명적)

### Gap 1: Gumbel-Softmax / Temperature Annealing이 Dead Code

`operations.py`에 `_gumbel_softmax()` 메서드가 정의되어 있지만, **실제 forward에서 plain softmax만 사용**하고 temperature 인자는 `del temperature`로 무시합니다. 이로 인해:
- Supernet 학습 시 "모든 ops의 가중합"을 보지만, OptimizedNetwork에서는 argmax 하나만 사용 → **expectation gap**
- 최신 논문(FBNetV2, OFA, LLM-NAS 등)은 progressive shrinking이나 Gumbel annealing으로 이 gap을 해소

### Gap 2: Pareto 탐색에 Evolutionary 전략 미구현

`pareto_search.py`에 `'evolutionary'` strategy가 docstring에 언급되지만 구현 없음. 현재는 random/alpha-guided/uniform_pareto만 존재하고, **mutation/crossover 기반의 Pareto 개선이 전혀 없음**. RAM-NAS(IROS 2024), LLM-NAS 등은 evolutionary 또는 LLM-guided 진화로 Pareto front 품질을 크게 향상.

### Gap 3: Pareto 품질 지표 부재

현재는 Pareto front를 추출만 하고 **HV(Hypervolume)**, **IGD(Inverted Generational Distance)** 등의 표준 다목적 최적화 지표를 보고하지 않음. Top-tier 논문에서는 필수.

### Gap 4: Width Projection의 LUT 미반영

`MixedOpWithWidth`에서 width < 1.0일 때 1×1 projection conv가 추가되지만, LUT 측정에는 **op 자체 latency만 포함**. 이 추가 latency가 빠져 LUT 정확도가 떨어짐.

### Gap 5: Throughput ≠ 단일 이미지 Latency

100장/16초 = 160ms/image인데, 현재는 **batch=1 단일 이미지 추론 시간만 측정**. 실제 배치 처리, 데이터 로딩, pre/post-processing의 파이프라인 throughput이 반영되지 않음.

---

## 2025+ 논문 대비 개선 제안 (5가지 제약 조건 모두 준수)

### 제약 조건
1. Pretrained Decoder Utilization (디코더 수정 X)
2. Multi-Hardware Latency-Aware Optimization
3. 100-Image / 16-Second Throughput Constraint
4. Measured LUT-Based Latency Modeling (실측 LUT 기반 지연시간 모델링)
5. Accuracy-Latency Multi-Objective NAS

---

### 개선 1: Progressive Alpha Discretization (PAD)

**문제**: Plain softmax → argmax 전환 시 expectation gap  
**제안**: Temperature annealing을 실제로 활성화하여 학습 중 점진적으로 alpha를 이산화

$$\alpha_t = \text{Gumbel-Softmax}(\alpha, \tau_t), \quad \tau_t = \tau_{\max} \cdot (\tau_{\min}/\tau_{\max})^{t/T}$$

- τ_max = 5.0 (초기, 탐색 다양성), τ_min = 0.1 (후기, one-hot 수렴)
- **적용 위치**: `operations.py`의 `forward()`, `get_sampled_flops()`, `get_sampled_latency()`에서 현재 무시되는 temperature를 실제 적용
- **근거**: FBNetV2, ProxylessNAS, SNAS 등에서 검증된 기법이지만, **LUT 기반 multi-hardware + Gumbel annealing 조합은 기존에 없음** → novelty
- **디코더 수정 없음**: op 선택 확률만 변경, 디코더 구조 자체는 고정

### 개선 2: Evolutionary Pareto Refinement (EPR)

**문제**: Random/guided 샘플링만으로는 Pareto front 품질 한계  
**제안**: Pareto front에서 mutation + crossover로 후속 세대 생성 → iterative Pareto 개선

- **Mutation**: Pareto 해의 1-2개 layer에서 op/width 랜덤 변경
- **Crossover**: 두 Pareto 해의 layer를 교차 조합
- **NSGA-II 스타일 non-dominated sorting + crowding distance**로 다양성 보장
- **적용 위치**: `pareto_search.py`의 `discover_pareto_curve()`에 evolutionary loop 추가
- **이미 구현된 미완성 코드**: docstring에 'evolutionary' 언급됨 → 이를 실제 구현
- **근거**: RAM-NAS, LLM-NAS 등이 evolutionary로 HV/IGD 크게 개선. **실측 LUT + evolutionary + multi-hardware Pareto는 새로운 조합**

### 개선 3: Multi-Fidelity Weight-Sharing Evaluation

**문제**: Weight-sharing 평가 → standalone 성능 간 rank correlation 저하  
**제안**: 2단계 평가 파이프라인

1. **Low-fidelity (빠른)**: 현재 weight-sharing 평가 (val subset)로 대량 필터링
2. **High-fidelity (정확)**: Top-k 후보를 **실제 OptimizedNetwork로 추출** 후 짧은 fine-tune (5-10 epoch) → 정밀 평가

- **현재 `pareto_refine_topk`가 이미 존재**하지만, fine-tune 없이 weight-sharing 추출만 수행 → fine-tune 추가가 핵심
- **적용 위치**: `segmentation.py`의 `_evaluate_extracted_subnet()` 확장
- **근거**: Auto-nnU-Net(AutoML 2025), Multi-Fidelity MO-NAS(arXiv 2025)가 multi-fidelity로 비용 대비 정확도 향상
- **비용 제어**: 상위 5-10개만 fine-tune하므로 전체 비용은 supernet 학습의 ~10%

### 개선 4: LUT 정확도 강화 — Projection Latency + Batch Throughput

**문제**: (a) Width projection 1×1 conv의 latency 누락, (b) Batch throughput 미반영  
**제안**:

**(a) Projection-Inclusive LUT**: LUT 빌더에서 width < 1.0인 op 측정 시 projection conv latency를 포함하여 측정
- **적용 위치**: `lut_builder.py`의 `_create_operation()` + `measure_op_latency()`
- Decoder skip-add 후 projection이 필요한 구간도 측정에 포함

**(b) Throughput-Aware LUT**: 100-image batch 단위로 end-to-end throughput 측정, 배치 효과(GPU utilization, memory bandwidth)를 반영한 "throughput LUT" 추가
- `batch_latency_ms = measure_batch_latency(model, batch_size=100)` → 16초 제약 직접 검증
- **적용 위치**: `lut_builder.py`에 `build_throughput_lut()` 메서드 추가

### 개선 5: 표준 다목적 최적화 지표 도입

**문제**: Pareto front 품질을 정량적으로 비교할 수단 부재  
**제안**: HV(Hypervolume), IGD, Pareto front size를 자동 계산 및 리포트

$$HV = \text{Vol}\left(\bigcup_{i=1}^{|P|} [f_i, r]\right)$$

- Reference point r: 가장 나쁜 (accuracy, latency) + margin
- **적용 위치**: `pareto_search.py`에 `compute_hypervolume()`, `compute_igd()` 추가
- **근거**: LLM-NAS가 HV/IGD로 SOTA 주장. 이 지표 없이는 reviewer가 Pareto 품질을 평가 불가

---

## 논문 기여 포지셔닝 (Top-Tier 제출 관점)

| 기여 | vs 기존 논문 | Novelty |
|---|---|---|
| **실측 LUT + Gumbel Annealing + Multi-HW** | FBNet(LUT+Gumbel), LINAS(LUT+multi-HW) 개별 존재 | **3가지 결합은 최초** |
| **Evolutionary Pareto + 실측 LUT** | RAM-NAS(evolutionary+predictor), RF-DETR(Pareto+sampling) | **실측 LUT 기반 evolutionary multi-HW Pareto는 새로움** |
| **Multi-Fidelity 평가 (WS → Fine-tune)** | Auto-nnU-Net(multi-fidelity HPO), MF-MO-NAS(Co-Kriging) | **산업 segmentation + 실측 LUT 환경에서의 적용은 new** |
| **Projection-Inclusive LUT** | 기존 LUT 논문들은 projection latency 미측정 | **측정 정확도 향상, practical contribution** |
| **산업용 ops (EdgeAware, DilatedDWSep)** | 일반 NAS는 도메인 무관 ops만 사용 | **도메인 특화 ops + 실측 기반 NAS는 차별점** |

---

## 검증 계획

| 항목 | 방법 |
|---|---|
| PAD 효과 | Gumbel annealing ON/OFF ablation → mIoU, FLOPs, latency 비교 |
| EPR 효과 | Random-only vs Evolutionary Pareto → **HV, IGD** 비교 |
| Multi-Fidelity 효과 | WS-only vs WS+fine-tune → rank correlation (Kendall τ) 비교 |
| LUT 정확도 | Projection 포함/미포함 LUT → 예측 latency vs 실측 latency MAE/MAPE |
| Throughput 검증 | 100장 배치 처리 시간 ≤ 16초 직접 측정 (6개 하드웨어) |
| 전체 성능 | Pareto front (mIoU vs latency) 비교: Ours vs baselines (UNet, DeepLabV3+, AutoPatch, RealtimeSeg) |

---

## 비교 대상 2025+ 논문 목록

| 논문 | 연도/벤유 | 핵심 기여 | 관련성 |
|---|---|---|---|
| LLM-NAS | arXiv 2025 | LLM-guided HW-NAS, complexity partitioning, zero-cost predictor | NAS + HW + Pareto(HV/IGD) |
| Sim-is-More | arXiv 2025 | Synthetic devices로 meta-training, few-shot HW-NAS | HW-aware NAS, device transfer |
| Auto-nnU-Net | AutoML 2025 | HPO+NAS+HNAS 통합, Regularized PriorBand | Segmentation NAS, multi-fidelity |
| SuperSAM | arXiv 2025 | SAM → supernet (pruning + parameter prioritization) | Supernet 설계, segmentation |
| NAS-LoRA | arXiv 2025 | NAS block + SAM PEFT, stage-wise optimization | Segmentation + NAS 적응 |
| MF-MO-NAS | arXiv 2025 | Multi-fidelity Co-Kriging, continuous encoding | Multi-objective NAS, segmentation |
| On Latency Predictors | MLSys 2024 | 체계적 latency predictor 평가, HW device split | Latency predictor 설계 개선 |
| RAM-NAS | IROS 2024 | Latency surrogate + evolutionary search, edge HW | Multi-objective + edge HW NAS |

---

*Generated: 2026-02-08*
