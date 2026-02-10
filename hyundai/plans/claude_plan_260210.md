# LINAS+ 통합 개선안: Top-Tier Conference/Journal 제출용

## Context

LINAS(Latency-aware Industrial NAS)는 현대자동차 실러 결함 세그멘테이션을 위한 다중 하드웨어 지연시간 인식 NAS 프레임워크이다. 현재 DenseNet121 인코더(고정) + 5층 MixedOp 디코더(탐색), 6종 하드웨어 LUT 기반 지연시간 최적화, Pareto 프론트 탐색을 구현하고 있다.

세 개의 기존 개선 계획서(`claude_plan_260209.md`, `codex_plan_260208.md`, `copilot_plan_260208.md`)에서 총 14개 제안을 분석하여, 중복을 통합하고 8개 독립 개선안으로 재구성하였다. 각 개선안은 5개 불변 제약조건(C1-C5)을 준수한다.

---

## 우선순위 총괄표

| 순위 | ID | 개선안 | N | I | F | D | 총점 | 전제조건 |
|:---:|:---:|--------|:-:|:-:|:-:|:-:|:---:|---------|
| 1 | U1 | Progressive Gumbel-Softmax Discretization | 3 | 4 | 5 | 5 | **17** | 없음 |
| 2 | U2 | LUT 정확도 향상 (Projection + Interaction) | 4 | 5 | 4 | 4 | **17** | 없음 |
| 3 | U3 | 배치 처리량 LUT 및 100-Image 직접 검증 | 3 | 4 | 5 | 5 | **17** | 없음 |
| 4 | U4 | 표준 다목적 최적화 메트릭 (HV/IGD) | 2 | 5 | 5 | 5 | **17** | 없음 |
| 5 | U5 | 진화적 Pareto 정제 + Zero-Cost Proxy | 4 | 4 | 4 | 3 | **15** | U4 |
| 6 | U6 | Preference-Conditioned Multi-HW Supernet | 5 | 5 | 3 | 2 | **15** | U1, U2 |
| 7 | U7 | CaLR + Multi-Fidelity Evaluation | 3 | 4 | 4 | 4 | **15** | U1 |
| 8 | U8 | Latency-Aware Adaptive KD | 4 | 4 | 4 | 3 | **15** | U2 |

> **N**=Novelty, **I**=Impact, **F**=Feasibility, **D**=개발 용이도 (5=최고)

---

## 구현 의존성 그래프

```
Phase 0 (독립, 병렬 가능):
  U1: Gumbel-Softmax 활성화      [operations.py, train_supernet.py]
  U2: LUT 정확도 + 메타데이터 수정  [lut_builder.py, latency_predictor.py, LUT JSONs]
  U3: 처리량 측정                  [utils.py, lut_builder.py]
  U4: HV/IGD 메트릭               [pareto_search.py]

Phase 1 (Phase 0 완료 후):
  U5: 진화적 Pareto 정제           [pareto_search.py]         ← U4
  U7: CaLR + Multi-Fidelity       [train_supernet.py]        ← U1

Phase 2 (Phase 0+1 완료 후):
  U6: Preference-Conditioned       [train_supernet.py, NEW]   ← U1, U2

Phase 3 (최종, NAS 이후):
  U8: Latency-Aware KD             [train_samplenet.py]       ← U2
```

---

## U1: Progressive Gumbel-Softmax Discretization

### 현재 한계
Supernet 학습 시 `torch.softmax`로 모든 연산의 가중 평균을 사용하나, 추론 시 `argmax`로 단일 연산 선택. 이 **기대값-이산화 갭(expectation gap)**이 성능 저하를 유발한다. `_gumbel_softmax()` 메서드가 이미 구현되어 있으나 dead code로 방치되어 있고, `temperature` 파라미터는 `del temperature`로 삭제된다.

> **근거 파일**: `copilot_plan_260208.md` — "Gap 1: Gumbel-Softmax/Temperature Annealing is dead code — `_gumbel_softmax()` defined but plain softmax used in forward, temperature ignored with `del temperature`"
> **코드 위치**: [operations.py](hyundai/utils/operations.py) (L76-78, L109, L173-179, L259)

### 개선 아이디어
기존 Gumbel-Softmax 코드를 활성화하고 지수 온도 감쇠(exponential temperature annealing) 적용:
- **스케줄**: `τ_t = τ_max · (τ_min / τ_max)^(t/T)`, τ_max=5.0 → τ_min=0.1
- 초기 80% 에폭: `hard=False` (탐색), 후반 20%: `hard=True` (수렴)
- `del temperature` 제거 → training loop에서 `temperature` 전달

### 기대 효과
- mIoU +0.5~1.5% (이산화 갭 해소)
- Alpha 엔트로피 단조 감소로 안정적 수렴
- Weight-sharing 랭킹 Kendall's τ: ~0.6 → 0.7+

### 검증 실험
- Ablation: plain softmax vs. Gumbel annealing (mIoU, alpha entropy 비교)
- τ_min ∈ {0.01, 0.1, 0.5}, τ_max ∈ {3.0, 5.0, 10.0} 그리드 서치
- 에폭별 alpha 엔트로피 추이 시각화

### 리스크/대응
| 리스크 | 대응 |
|--------|------|
| 과도한 조기 수렴으로 차선 연산 선택 | Cosine annealing fallback; warmup 기간 τ 고정 |
| Gumbel 노이즈가 latency-aware 학습 불안정화 | Alpha 업데이트(Phase 2)에만 적용, weight 업데이트(Phase 1) 제외 |

### 점수: N3 / I4 / F5 / D5 = **17점**

### 핵심 수정 파일
- [operations.py](hyundai/utils/operations.py) — Gumbel-Softmax 활성화, `del temperature` 제거
- [train_supernet.py](hyundai/nas/train_supernet.py) — temperature 스케줄링 로직 추가
- [supernet_dense.py](hyundai/nas/supernet_dense.py) — forward에 temperature 전달

---

## U2: LUT 정확도 향상 — Projection-Inclusive & Interaction-Aware Latency

### 현재 한계
**(a) Width projection 누락**: `width_mult < 1.0`일 때 채널 복원을 위한 1×1 Conv 지연시간이 LUT에 미포함. LUT는 base 연산만 측정.
**(b) 연산 간 상호작용 무시**: `get_architecture_latency()`가 개별 연산 지연시간을 독립적으로 합산. 실제로는 메모리 대역폭, 캐시 효과 등으로 15~20% 오차 발생.
**(c) Jetson LUT 메타데이터 오류**: `lut_jetsonorin.json`의 `"hardware"` 필드가 `"CPU"`로 잘못 기록.

> **근거 파일**: `claude_plan_260209.md` — "Operation-level independence assumption in LUT lacks validity (memory bandwidth contention, cache effects exist)"
> `copilot_plan_260208.md` — "Gap 4: Width projection 1×1 conv latency missing from LUT measurements"
> `codex_plan_260208.md` — "Jetson LUT metadata mismatch: hardware field labeled as 'CPU'"
> **코드 위치**: [lut_builder.py](hyundai/latency/lut_builder.py) (L104-126), [operations.py](hyundai/utils/operations.py) (L152-153, L292-312), [latency_predictor.py](hyundai/latency/latency_predictor.py) (L64-93)

### 개선 아이디어
**(a) Projection-Inclusive LUT**: `_create_operation()`에서 `width_mult < 1.0`인 경우 연산 + 1×1 Conv를 `nn.Sequential`로 묶어 통합 측정
**(b) Pairwise Interaction LUT**: 인접 레이어 쌍(4쌍)에 대해 `pair_latency - (lut[op_i] + lut[op_{i+1}])`로 상호작용 항 측정 → LUT JSON에 `"interactions"` 필드 추가
```
corrected_lat = Σ(op_latencies) + Σ(pairwise_interactions)
```
**(c) 메타데이터 수정**: `lut_jetsonorin.json`의 `"hardware": "CPU"` → `"JetsonOrin"`

### 기대 효과
- LUT-to-E2E MAPE: ~15% → **< 5%**
- 엣지 디바이스(JetsonOrin/RaspberryPi5/Odroid)에서 더 큰 개선
- 처리량 제약 만족/불만족 아키텍처 분류 정확도 향상

### 검증 실험
- 하드웨어별 50개 랜덤 아키텍처: LUT 예측 vs. 실측 E2E 비교 (MAE, MAPE, Spearman ρ)
- Ablation: (i) 기존 LUT → (ii) +projection → (iii) +interaction → (iv) +projection+interaction
- Pareto 프론트 변화 분석: 보정 전후 프론트 구성 아키텍처 비교

### 리스크/대응
| 리스크 | 대응 |
|--------|------|
| Interaction 측정 비용: O(ops² × widths² × 4 pairs) | 동일 width 조합만 측정 → 588회로 축소 |
| 고속 GPU에서 noise-dominated interaction | `|interaction| > std_threshold` 조건부 적용 |

### 점수: N4 / I5 / F4 / D4 = **17점**

### 핵심 수정 파일
- [lut_builder.py](hyundai/latency/lut_builder.py) — projection 포함 측정, interaction 측정 추가
- [latency_predictor.py](hyundai/latency/latency_predictor.py) — interaction 항 반영
- [operations.py](hyundai/utils/operations.py) — `get_sampled_latency()` 수정
- [lut_jetsonorin.json](hyundai/latency/luts/lut_jetsonorin.json) — 메타데이터 수정

---

## U3: 배치 처리량 LUT 및 100-Image/16-Second 직접 검증

### 현재 한계
처리량 제약은 "100 images / 16 seconds"이나, 현재 `measure_inference_time()`은 `batch_size=1`로 단일 이미지 지연시간만 측정. GPU 활용률, 메모리 대역폭 포화, 커널 런치 오버헤드, 전후처리 파이프라인이 반영되지 않아 `100 × single_latency ≠ batch_throughput`.

> **근거 파일**: `copilot_plan_260208.md` — "Gap 5: Throughput measured as single-image batch=1 latency, ignoring batching, data loading, pre/post-processing pipeline effects"
> `codex_plan_260208.md` — "Throughput constraint not formally specified"
> **코드 위치**: [utils.py](hyundai/utils/utils.py) (L174-220)

### 개선 아이디어
**(a) Throughput-Aware LUT**: `build_throughput_lut()` 추가 — 100 이미지 전체 모델 E2E 시간 측정 (GPU: batch=100 or 메모리 제약 시 sub-batch; Edge: batch=1 × 100 sequential)
**(b) Throughput Evaluator**: `measure_throughput()` — 데이터 로딩→전처리→추론→후처리 전체 파이프라인으로 100 이미지 처리, `total_time ≤ 16.0s` 검증
**(c) 제약 공식화**: 암묵적 160ms/image 가정 → 명시적 `T_h(100 images) ≤ 16 sec` 제약으로 다목적 손실에 통합

### 기대 효과
- 처리량 실현 가능성 판정: ~80% → **>99%** 정확도
- 엣지 디바이스 메모리 제약 아키텍처 정확한 필터링
- 리뷰어 요구사항 충족: images/sec 직접 보고

### 검증 실험
- 하드웨어별 20개 아키텍처: `100 × single_latency` vs. `actual_100_image_throughput` 불일치 통계
- 단일 이미지 통과 but 100 이미지 실패 아키텍처 비율
- 하드웨어별 처리량(images/sec), p50/p95 지연시간 테이블

### 리스크/대응
| 리스크 | 대응 |
|--------|------|
| 엣지 디바이스 100-image 측정 시간 소요 | Pareto-front 아키텍처에만 적용 (탐색 중 미적용) |
| 열 스로틀링으로 변동 처리량 | 열적 정상상태 후 sustained throughput 보고 |

### 점수: N3 / I4 / F5 / D5 = **17점**

### 핵심 수정 파일
- [utils.py](hyundai/utils/utils.py) — `measure_throughput()` 추가
- [lut_builder.py](hyundai/latency/lut_builder.py) — `build_throughput_lut()` 추가

---

## U4: 표준 다목적 최적화 메트릭 — Hypervolume(HV) & IGD

### 현재 한계
Pareto 탐색이 프론트 추출 및 요약 출력만 수행하고, **표준 다목적 최적화 메트릭(HV, IGD)을 전혀 계산하지 않음**. Top-tier 학회(NeurIPS, ICLR)에서 Pareto 품질 주장에 필수적인 정량 지표 부재.

> **근거 파일**: `copilot_plan_260208.md` — "Gap 3: No standard multi-objective metrics reported (Hypervolume HV, Inverted Generational Distance IGD); only raw Pareto front extracted"
> **코드 위치**: [pareto_search.py](hyundai/nas/pareto_search.py) (L434-469, L535-556)

### 개선 아이디어
**(a) Hypervolume(HV)**: `compute_hypervolume(pareto_front, reference_point)` — reference point = `(worst_accuracy - margin, worst_latency + margin)` (하드웨어별). 2D이므로 O(n log n) 정렬 후 계산.
**(b) IGD**: `compute_igd(pareto_front, true_pareto)` — true Pareto = 전체 방법론 합집합에서 추정
**(c) 자동 로깅**: `discover_pareto_curve()` 완료 후 wandb에 HV/IGD 자동 기록
**(d) Delta-HV 추적**: 평가 아키텍처 수 증가에 따른 HV 수렴 곡선

### 기대 효과
- 논문 제출 준비 완료: 리뷰어 기대 정량 지표 충족
- NSGA-II, MOEA/D 등 기존 방법론과 정량 비교 가능
- 후속 개선(U5-U8) 효과를 HV/IGD delta로 측정 가능

### 검증 실험
- 기존 mixed 샘플링 baseline HV 측정
- Random-only vs. alpha-only vs. mixed sampling HV 비교
- 5 seeds 평균 ± 표준편차 통계적 유의성

### 리스크/대응
| 리스크 | 대응 |
|--------|------|
| Reference point 선택이 HV 해석에 영향 | 모든 방법론에 동일 reference point 고정; 민감도 분석 포함 |

### 점수: N2 / I5 / F5 / D5 = **17점**

### 핵심 수정 파일
- [pareto_search.py](hyundai/nas/pareto_search.py) — `compute_hypervolume()`, `compute_igd()` 추가

---

## U5: 진화적 Pareto 정제 + Zero-Cost Proxy

### 현재 한계
`sample_architecture()`의 docstring에 `'evolutionary'` 전략이 명시되어 있으나 **미구현** (`else` 분기에서 `ValueError` 발생). 현재 1000개 샘플 중 100개만 평가하여 4.1M 탐색 공간의 0.025%만 탐색.

> **근거 파일**: `copilot_plan_260208.md` — "Gap 2: Evolutionary Pareto strategy mentioned in docstring but not implemented; only random/alpha-guided/uniform_pareto exist"
> `claude_plan_260209.md` — "I3: Zero-Cost Guided Pareto Sampling — Activation rank calculation for pre-screening without training; 2-stage sampling"
> **코드 위치**: [pareto_search.py](hyundai/nas/pareto_search.py) (L138-183, L434-469)

### 개선 아이디어
**(a) NSGA-II 기반 진화 전략**:
- Mutation: Pareto 프론트 아키텍처의 1~2개 레이어 op/width 변경
- Crossover: 두 Pareto 아키텍처의 레이어 수준 uniform crossover (p=0.5)
- Non-dominated sorting + Crowding distance로 다양성 유지

**(b) Zero-Cost Proxy 통합 (3단계 파이프라인)**:
```
Stage 1: 10K architectures → zero-cost proxy(activation rank) + LUT latency → 500 후보 (< 2분)
Stage 2: 500 후보 → weight-sharing 평가 → 100 평가 완료
Stage 3: 100 평가 → evolutionary refinement (5~10 세대) → 정제된 Pareto 프론트
```

### 기대 효과
- Pareto HV: **+15~25%** (현재 mixed sampling 대비)
- 동일 평가 예산으로 10배 더 많은 아키텍처 스크리닝
- 미탐색 trade-off 영역의 Pareto 프론트 밀도 향상

### 검증 실험
- HV 비교: mixed sampling vs. evolutionary vs. zero-cost+evolutionary
- 수렴 곡선: 평가 수 vs. HV
- Ablation: mutation only / crossover only / mutation+crossover
- 세대별 HV 개선량 (수확체감 분석)

### 리스크/대응
| 리스크 | 대응 |
|--------|------|
| 지역 Pareto 프론트 수렴 | Crowding distance 유지 + 매 세대 랜덤 immigrants 주입 |
| Zero-cost proxy와 실제 정확도 상관관계 부족 | 50개 holdout 아키텍처로 proxy 랭킹 상관관계 사전 검증 |
| SVD 계산 속도 (activation rank) | `torch.svd_lowrank` 사용 또는 레이어 서브셋만 계산 |

### 점수: N4 / I4 / F4 / D3 = **15점**

### 핵심 수정 파일
- [pareto_search.py](hyundai/nas/pareto_search.py) — evolutionary 전략 구현, zero-cost proxy 통합

---

## U6: Preference-Conditioned Multi-Hardware Supernet

### 현재 한계
다중 하드웨어 최적화가 단일 scalarization(`Σ λ·ReLU(lat - target)`)으로 Pareto 프론트의 한 점만 탐색. 다양한 accuracy-latency trade-off 탐색에는 `latency_lambda` 변경 후 재학습 필요 → 6 하드웨어 × N preference = 6N회 학습.

> **근거 파일**: `claude_plan_260209.md` — "I1: Preference-Conditioned Supernet — Hypernetwork generates alpha parameters from hardware embeddings + preference vectors; enables single supernet training for all hardware/trade-off points"
> `codex_plan_260208.md` — "D: introduce preference-conditioned NAS head for diverse accuracy-latency trade-offs"
> **코드 위치**: [train_supernet.py](hyundai/nas/train_supernet.py) (L236-246), [hardware_encoder.py](hyundai/latency/hardware_encoder.py) (L46-120, L146-191)

### 개선 아이디어
**(a) Hardware embedding**: 기존 `HardwareEncoder`/`HardwareEmbedding` 클래스 재활용
**(b) Preference vector**: 2D `[w_accuracy, w_latency]` (w_acc + w_lat = 1)
**(c) Hypernetwork**: 소형 MLP가 `concat(hw_embedding, preference_vector)` → `alpha_op`, `alpha_width` (5 decoder layers) 생성
**(d) 학습**: 매 배치마다 랜덤 `(hardware, preference)` 쌍 샘플 → hypernetwork로 alpha 생성 → task loss + preference-weighted latency penalty 역전파

### 기대 효과
- Pareto HV: **+15~30%** (연속 Pareto 프론트)
- 학습 비용: **6배 감소** (단일 실행으로 모든 하드웨어 커버)
- 미지 하드웨어로 Zero-shot 전이 가능
- 런타임 trade-off 선택 (재학습 불필요)

### 검증 실험
- 6개 독립 학습 Pareto 프론트 vs. 단일 preference-conditioned 실행 비교
- 하드웨어별 HV 보고
- 10개 균등 preference vector 샘플 → 단조적 accuracy-latency trade-off 검증
- Leave-one-hardware-out: 1개 하드웨어 제외 학습 → zero-shot 전이 성능

### 리스크/대응
| 리스크 | 대응 |
|--------|------|
| Hypernetwork 최적화 복잡성 | 2-layer MLP부터 시작, 점진적 용량 증가 |
| Mode collapse (preference 무관 동일 alpha 생성) | Alpha 분포 KL divergence 다양성 정규화 추가 |
| GPU 메모리 증가 | Hypernetwork ~10K params << supernet ~50M params (무시 가능) |

### 점수: N5 / I5 / F3 / D2 = **15점**

### 핵심 수정 파일
- [train_supernet.py](hyundai/nas/train_supernet.py) — hypernetwork 통합, preference 샘플링
- [operations.py](hyundai/utils/operations.py) — 외부 alpha 주입 인터페이스
- [hardware_encoder.py](hyundai/latency/hardware_encoder.py) — 기존 코드 재활용
- NEW: `hyundai/nas/hypernetwork.py` — preference-conditioned alpha 생성기

---

## U7: Complexity-Aware Fair Training + Multi-Fidelity Evaluation

### 현재 한계
**(a) 불공정 학습**: 모든 서브넷이 동일 learning rate와 단일 momentum 버퍼 공유. 저복잡도 서브넷(작은 ops, 좁은 width)이 빠르게 수렴/과적합, 고복잡도 서브넷은 과소학습 → weight-sharing 랭킹 정확도 저하.
**(b) Fine-tuning 부재**: 추출된 서브넷을 weight-sharing 상태 그대로 평가. `pareto_refine_topk` 파라미터가 존재하나 추가 학습 없이 재평가만 수행.

> **근거 파일**: `claude_plan_260209.md` — "I2: Complexity-Aware Fair Training — CaLR, momentum separation for complexity bins; ensures fair supernet training"
> `copilot_plan_260208.md` — "Improvement 3: Multi-Fidelity Weight-Sharing Evaluation — top-k extracted as OptimizedNetwork with 5-10 epoch fine-tune"
> **코드 위치**: [train_supernet.py](hyundai/nas/train_supernet.py) (L151-318), [segmentation.py](hyundai/segmentation.py) (L34-76)

### 개선 아이디어
**(a) CaLR (Complexity-Aware Learning Rate)**:
```python
complexity_ratio = sampled_flops / max_flops  # 0~1
lr_scale = 0.5 + 0.5 * complexity_ratio       # 저복잡도 → 낮은 LR
```
**(b) Multi-Fidelity Evaluation**:
```
Low-fidelity:  Weight-sharing 평가 → 전체 후보 랭킹
High-fidelity: Top-5 후보를 OptimizedNetwork로 추출 → 5~10 에폭 fine-tune → 최종 선택
```

### 기대 효과
- Kendall's τ: ~0.6 → **> 0.8**
- 선택 아키텍처 mIoU: **+1~2%** (더 정확한 랭킹으로 진정한 최적 선택)
- Supernet 학습 안정성 향상

### 검증 실험
- 50개 아키텍처 WS 평가 vs. standalone 학습(30 에폭) → Kendall's τ 비교
- CaLR ablation: uniform LR vs. CaLR → 랭킹 상관관계
- Multi-fidelity ablation: WS-only vs. +5에폭 vs. +10에폭
- 비용 분석: Multi-fidelity 추가 비용 = 5 models × 5 epochs ≈ supernet 학습의 ~10%

### 리스크/대응
| 리스크 | 대응 |
|--------|------|
| CaLR이 소형 서브넷 수렴 저해 | 보수적 스케일링 (0.7 + 0.3 × ratio) |
| Multi-fidelity 추가 연산 비용 | Top-5만 fine-tune; 5 에폭 제한 |

### 점수: N3 / I4 / F4 / D4 = **15점**

### 핵심 수정 파일
- [train_supernet.py](hyundai/nas/train_supernet.py) — CaLR 적용
- [segmentation.py](hyundai/segmentation.py) — multi-fidelity fine-tune 단계 추가
- [supernet_dense.py](hyundai/nas/supernet_dense.py) — `get_sampled_flops()` for complexity ratio

---

## U8: Latency-Aware Adaptive Knowledge Distillation

### 현재 한계
최종 아키텍처 학습이 `CrossEntropyLoss`만 사용. Supernet 내 full-width "teacher"의 지식이 아키텍처 추출 후 폐기됨. 경량 아키텍처(width=0.5, DWSep 연산)는 용량 한계로 정확도 손실이 크나, KD로 보상 가능.

> **근거 파일**: `claude_plan_260209.md` — "I5: Latency-Aware Knowledge Distillation — 3-level KD; adaptive KD strength proportional to latency ratio; expected mIoU +2-5%"
> **코드 위치**: [train_samplenet.py](hyundai/nas/train_samplenet.py) (전체, 특히 `train_opt()`)

### 개선 아이디어
**(a) Teacher 추출**: Supernet에서 max-width 서브넷(width=1.0, 최고 용량 연산) → 추가 학습 불필요
**(b) Multi-level KD loss**:
- **Logit-level**: `KL(student_logits/T, teacher_logits/T) × T²`
- **Feature-level**: 중간 feature MSE (1×1 adapter conv로 채널 정렬)
- **Inter-class similarity**: 클래스별 코사인 유사도 구조 보존 (binary seg → patch-wise similarity)

**(c) Latency-aware adaptive strength (핵심 기여)**:
```python
latency_ratio = student_latency / teacher_latency  # 통상 0.3~1.0
alpha_kd = base_alpha × (1 + (1 - latency_ratio))  # 경량 student → 강한 KD
```

### 기대 효과
- mIoU: **+2~5%** (특히 경량 아키텍처에서)
- Full-width 대비 정확도 갭의 60~80% 회복
- 추론 시 추가 비용 없음 (학습 시에만 적용)

### 검증 실험
- KD ablation: CE-only vs. +logit-KD vs. +logit+feature-KD vs. +전체
- Adaptive strength ablation: 고정 KD 강도 vs. latency-adaptive; 3 latency 구간 비교
- 아키텍처별 분석: KD mIoU 개선 vs. 아키텍처 복잡도(FLOPs/latency) 산점도

### 리스크/대응
| 리스크 | 대응 |
|--------|------|
| Teacher(full-width supernet)가 충분히 최적화되지 않음 | Teacher를 5 에폭 추가 fine-tune 후 사용 |
| Feature-level KD의 메모리 부담 | Gradient checkpointing; 5개 중 2~3개 레이어만 distill |
| Binary segmentation에서 inter-class similarity 적용 한계 | Patch-wise similarity로 대체 |

### 점수: N4 / I4 / F4 / D3 = **15점**

### 핵심 수정 파일
- [train_samplenet.py](hyundai/nas/train_samplenet.py) — KD loss 통합, adaptive strength
- [supernet_dense.py](hyundai/nas/supernet_dense.py) — teacher 추출 (OptimizedNetwork)

---

## 논문 구조 제안

**제목**: *LINAS+: Preference-Conditioned Latency-Aware NAS with Interaction-Aware LUT for Multi-Hardware Industrial Segmentation*

### 스토리 아크
1. **문제 정의**: 6종 이기종 하드웨어에서의 산업용 세그멘테이션
2. **Gap 1**: 기존 LUT 기반 NAS의 softmax-argmax 갭 및 부정확한 지연시간 (U1, U2, U3)
3. **Gap 2**: Pareto 탐색 비효율성 및 정량 메트릭 부재 (U4, U5)
4. **Gap 3**: 단일 scalarization으로 다중 하드웨어 Pareto 프론트 불가 (U6, U7)
5. **Gap 4**: 경량 모델의 KD 없는 과도한 정확도 손실 (U8)
6. **해결**: LINAS+가 4개 갭을 통합 프레임워크로 해결

### 기대 Ablation 테이블

| Config | mIoU | HV | LUT MAPE | 평가 비용 |
|--------|------|----|----------|----------|
| Baseline LINAS | ref | ref | ~15% | 1000 evals |
| +U1 (Gumbel) | +0.5~1.5% | — | — | — |
| +U2 (LUT fix) | — | — | <5% | — |
| +U4+U5 (Evo+Metrics) | — | +15~25% | — | 100 evals |
| +U6 (Preference) | — | +15~30% | — | 1 run/all HW |
| +U7 (CaLR+MF) | +1~2% | — | — | +10% cost |
| +U8 (KD) | +2~5% | — | — | — |
| **Full LINAS+** | **+3~7%** | **+30~50%** | **<5%** | **10× cheaper** |

### 타겟 학회
CVPR 2026, NeurIPS 2026, ICLR 2026, ECCV 2026

---

## Novelty 분석: 기존 연구 대비 차별점

| 조합 | 차별점 | 가장 유사한 기존 연구 |
|------|--------|---------------------|
| Measured LUT + Gumbel Annealing + Multi-HW (U1+U2) | FBNet은 Gumbel+proxy, LINAS는 LUT+multi-HW; 세 가지 통합은 신규 | FBNet, ProxylessNAS |
| Zero-cost + Evolutionary + Measured LUT (U5) | RAM-NAS는 evolutionary+predictor; measured LUT 위에서 zero-cost 사전 필터 신규 | RAM-NAS, NEAR |
| Preference-conditioned + Industrial Ops + Multi-HW LUT (U6) | MODNAS는 proxy latency로 preference conditioning; measured LUT+산업 도메인 특화 연산은 신규 | MODNAS, OFA |
| Latency-ratio adaptive KD strength (U8) | NAS용 KD 존재(SCTNet-NAS); KD 강도를 지연시간 제약 비율에 비례시키는 것은 신규 | SCTNet-NAS, AICSD |

---

## 검증 방법 요약

1. **메트릭**: mIoU, FLOPs, 하드웨어별 latency(ms), throughput(images/sec), HV, IGD, Kendall's τ, LUT MAPE
2. **Ablation**: 누적 추가 방식 (Baseline → +U1 → +U1U2 → ... → Full LINAS+)
3. **베이스라인**: UNet, DeepLabV3+, AutoPatch, RealtimeSeg (기존 `comparison.py` 활용)
4. **통계**: 5 seeds (0, 1, 2, 42, 100), mean ± std 보고
5. **하드웨어**: 6종 전체에서 Pareto 프론트 비교
6. **코드**: wandb 자동 로깅으로 재현성 보장

---

## 한눈에 보는 비유 요약

| ID | 한 줄 비유 | 핵심 효과 |
|:---:|-----------|----------|
| U1 | 연습을 실전처럼 | mIoU +0.5~1.5% |
| U2 | 시간표 오차 줄이기 | LUT 오차 15%→5% |
| U3 | 실제 생산라인 속도 측정 | 처리량 판정 정확도 99%+ |
| U4 | 마라톤 기록 재기 | 논문 제출 필수 지표 |
| U5 | 도서관 똑똑하게 탐색 | Pareto HV +15~25% |
| U6 | 만능 재봉사 1명 | 학습 비용 6배 감소 |
| U7 | 공정한 시험 + 결선 | 랭킹 정확도 0.6→0.8 |
| U8 | 선생님 필기 참고 학습 | mIoU +2~5% |
