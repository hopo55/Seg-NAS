# LINAS Top-tier 제출용 개선안 통합 제안서 (hyundai/plans + hyundai 관련 문서 종합)

## 요약
- 목표: `hyundai/plans`의 3개 `.md`와 `hyundai` 코드/스크립트/LUT를 교차 검증해, 제출 가능한 수준의 실행 가능한 개선안으로 정리.
- 원칙: 5개 고정 제약(C1~C5)을 유지하면서, 코드 기준의 실제 갭을 우선 보완.
- 총점 산식(확정): `총점 = Novelty + Impact + Feasibility + (6 - 개발 난이도)`.

## 분석 범위
- `hyundai/plans/codex_plan_260208.md`
- `hyundai/plans/copilot_plan_260208.md`
- `hyundai/plans/claude_plan_260209.md`
- `hyundai/nas/*.py`, `hyundai/utils/operations.py`, `hyundai/latency/*.py`, `hyundai/segmentation.py`, `hyundai/scripts/*.sh`, `hyundai/latency/luts/*.json`

## 개선안 상세

### 개선안 1. C1 정합형 구조 전환 (Decoder 고정 + 탐색 변수 외부화)
- 현재 한계: 현재 구현은 searchable decoder 중심이며 C1(Pretrained Decoder 고정)과 충돌.
- 개선 아이디어: decoder를 pretrained checkpoint로 로드 후 freeze하고, 탐색 변수를 encoder-side adapter/해상도/배치 정책으로 이동.
- 기대 효과: 제약 위반 리스크 제거, 리뷰어 질문(C1 위배) 선제 차단, 재현성 향상.
- 검증 실험: Baseline vs `freeze_decoder` vs `freeze_decoder + external search` 비교(mIoU, latency, throughput, Pareto HV).
- 리스크/대응: 탐색공간 축소로 정확도 하락 가능 -> adapter block 최소 추가 및 width/reso 축 탐색으로 보상.
- 근거 파일/핵심 문장: `hyundai/plans/codex_plan_260208.md:8` "Pretrained Decoder Utilization (디코더 수정 금지)"; `hyundai/plans/codex_plan_260208.md:23` "현재 구조는 pretrained encoder + searchable decoder"; `hyundai/nas/supernet_dense.py:49` "`deconv1 = MixedOpWithWidth(...)`"; `hyundai/segmentation.py:121` "`weight_params = [param for param in model.parameters() ...]`".
- 점수: Novelty 4/5, Impact 5/5, Feasibility 3/5, 개발 난이도 4/5, 총점 14/20.

### 개선안 2. PAD (Progressive Alpha Discretization)로 expectation gap 축소
- 현재 한계: Gumbel 함수가 사실상 dead code이며 temperature 인자가 무시됨.
- 개선 아이디어: `forward/get_sampled_flops/get_sampled_latency`에 annealed Gumbel-Softmax(초기 고온, 후기 저온) 적용.
- 기대 효과: supernet 기대값과 argmax subnet 간 괴리 감소, rank correlation 개선.
- 검증 실험: PAD on/off ablation, Kendall tau(WS vs standalone), 최종 선택 subnet 성능 차이.
- 리스크/대응: alpha 조기 붕괴 -> entropy regularization + temperature floor(`tau_min`) 도입.
- 근거 파일/핵심 문장: `hyundai/utils/operations.py:76` "`_gumbel_softmax`"; `hyundai/utils/operations.py:109` "`del temperature`"; `hyundai/utils/operations.py:296` "`del temperature`"; `hyundai/plans/copilot_plan_260208.md:21` "plain softmax만 사용하고 temperature 인자는 del temperature로 무시".
- 점수: Novelty 3/5, Impact 4/5, Feasibility 5/5, 개발 난이도 2/5, 총점 16/20.

### 개선안 3. Pareto 탐색 고도화 (Zero-cost prefilter + Evolutionary refinement)
- 현재 한계: sampling 전략에 evolutionary가 문서상 언급되지만 구현 분기 없음; 1000/100 샘플 기반으로 탐색 효율 제한.
- 개선 아이디어: 1단계 zero-cost proxy prefilter, 2단계 NSGA-II 스타일 mutation/crossover refinement를 `discover_pareto_curve()`에 통합.
- 기대 효과: 동일 예산에서 Pareto front 품질(HV/IGD)과 다양성 향상.
- 검증 실험: 기존 mixed vs hybrid(Zero-cost+Evo) 비교(HV, IGD, front size, 탐색 시간).
- 리스크/대응: 계산량 증가 -> prefilter 후보 수 상한 + early-stop 세대 종료 기준.
- 근거 파일/핵심 문장: `hyundai/nas/pareto_search.py:143` "strategy: 'random', 'alpha_guided', or 'evolutionary'"; `hyundai/nas/pareto_search.py:145` "if strategy == 'random'"; `hyundai/nas/pareto_search.py:165` "elif strategy == 'uniform_pareto'"; `hyundai/nas/pareto_search.py:364` "`sample_architectures(num_samples, strategy)`"; `hyundai/scripts/train_linas.sh:73` "`PARETO_SAMPLES=1000`"; `hyundai/plans/copilot_plan_260208.md:27` "mutation/crossover 기반 ... 미구현".
- 점수: Novelty 5/5, Impact 5/5, Feasibility 3/5, 개발 난이도 4/5, 총점 15/20.

### 개선안 4. LUT 정밀도 강화 (Projection + Inter-op interaction + LUT 무결성)
- 현재 한계: width<1.0 projection이 실제 forward엔 존재하지만 LUT 합산엔 반영되지 않고, layer 독립 합산 가정 및 LUT metadata mismatch(Jetson) 이슈 존재.
- 개선 아이디어: projection-inclusive 측정, 인접 layer pair interaction LUT 추가, `filename-hardware` 무결성 validator 도입.
- 기대 효과: LUT 예측과 E2E 측정 간 MAPE 축소, 하드웨어 식별 안정성 향상.
- 검증 실험: 기존 LUT vs 확장 LUT의 하드웨어별 MAPE/MAE, LOHO 환경에서 latency 예측 안정성 비교.
- 리스크/대응: LUT 차원 폭증 -> 인접 pair만 측정, 중요 조합만 sparse 저장.
- 근거 파일/핵심 문장: `hyundai/utils/operations.py:151` "Projection to restore full channels"; `hyundai/utils/operations.py:305` "`latency_lut.get_op_latency(...)`"; `hyundai/latency/lut_builder.py:104` "`_create_operation`"; `hyundai/latency/lut_builder.py:253` "`measure_op_latency`"; `hyundai/latency/latency_predictor.py:86` "`total_latency = 0.0`"; `hyundai/latency/luts/lut_jetsonorin.json:2` "`\"hardware\": \"CPU\"`"; `hyundai/plans/codex_plan_260208.md:26` "`lut_jetsonorin.json`의 `hardware` 값이 `CPU`".
- 점수: Novelty 4/5, Impact 5/5, Feasibility 4/5, 개발 난이도 3/5, 총점 16/20.

### 개선안 5. C3 직접 반영: 100장/16초 Throughput 제약식 및 평가기 추가
- 현재 한계: 현재 보고는 사실상 batch=1 단일 추론 시간 중심이며, 100장 윈도우의 end-to-end throughput 제약이 목적함수/리포트에 직접 연결되지 않음.
- 개선 아이디어: `T_h(100)<=16s`를 하드웨어별 hard/soft constraint로 도입하고, dataloader 포함 wall-clock evaluator 추가.
- 기대 효과: 요구사항 C3를 실측으로 직접 만족 검증 가능, 제출 시 산업 제약 정합성 강화.
- 검증 실험: 하드웨어별 100장 처리 반복 5회 이상(p50/p95), 제약 충족률 및 feasible Pareto coverage 보고.
- 리스크/대응: 시스템 부하로 분산 변동 -> 측정 시 고정 전력모드/스레드/워크로드 고정 및 반복 측정.
- 근거 파일/핵심 문장: `hyundai/plans/codex_plan_260208.md:10` "100-Image / 16-Second Throughput Constraint"; `hyundai/plans/copilot_plan_260208.md:39` "현재는 batch=1 단일 이미지 추론 시간만 측정"; `hyundai/nas/train_samplenet.py:222` "`input_size=(1, 3, args.resize, args.resize)`"; `hyundai/nas/train_samplenet.py:232` "`(batch=1)`".
- 점수: Novelty 4/5, Impact 5/5, Feasibility 4/5, 개발 난이도 3/5, 총점 16/20.

### 개선안 6. Multi-fidelity 후보 검증 (WS 평가 -> 짧은 subnet fine-tune 재정렬)
- 현재 한계: `pareto_refine_topk`가 있지만 현재는 extracted subnet 평가만 수행하고 fine-tune 단계 없음.
- 개선 아이디어: shortlist top-k에 5~10 epoch 저비용 fine-tune 후 재정렬하여 최종 선택.
- 기대 효과: weight-sharing bias 완화, 실제 배포 subnet 성능 예측 정확도 향상.
- 검증 실험: WS-only vs WS+fine-tune에서 Kendall tau, top-1 regret, 최종 mIoU 비교.
- 리스크/대응: 시간 증가 -> top-k 제한, 저해상도/조기종료로 budget cap.
- 근거 파일/핵심 문장: `hyundai/segmentation.py:496` "`pareto_refine_topk`"; `hyundai/segmentation.py:517` "`_evaluate_extracted_subnet(...)`"; `hyundai/segmentation.py:34` "`Evaluate an extracted subnet ... on validation set`"; `hyundai/plans/copilot_plan_260208.md:86` "fine-tune 없이 weight-sharing 추출만 수행".
- 점수: Novelty 4/5, Impact 4/5, Feasibility 3/5, 개발 난이도 4/5, 총점 13/20.

### 개선안 7. Predictor 목적함수/평가체계 고도화 (Hybrid rank loss + HV/IGD + LOHO)
- 현재 한계: predictor 학습은 MSE 단일 손실, Pareto 결과물은 front 저장 중심으로 표준 품질지표(HV/IGD) 자동 리포트 부재.
- 개선 아이디어: predictor에 `MSE + pairwise + listwise` hybrid loss 도입, feasible/global Pareto 분리와 HV/IGD 자동 계산, leave-one-hardware-out 평가 추가.
- 기대 효과: latency predictor 순위 품질 개선, 논문 심사에서 필요한 정량 근거 강화.
- 검증 실험: 손실 조합별 MAE/MAPE/Kendall tau, LOHO generalization, HV/IGD 및 constraint satisfaction rate.
- 리스크/대응: 다중 손실 가중치 튜닝 난이도 -> uncertainty weighting 또는 validation 기반 스케줄링.
- 근거 파일/핵심 문장: `hyundai/latency/latency_predictor.py:551` "`loss = F.mse_loss(...)`"; `hyundai/nas/pareto_search.py:508` "`results = {'all_architectures': ..., 'pareto_fronts': ...}`"; `hyundai/plans/copilot_plan_260208.md:31` "HV, IGD ... 보고하지 않음"; `hyundai/plans/codex_plan_260208.md:58` "ranking-aware predictor loss 도입".
- 점수: Novelty 4/5, Impact 4/5, Feasibility 4/5, 개발 난이도 3/5, 총점 15/20.

### 개선안 8. Latency-aware KD로 최종 subnet 정확도 회복
- 현재 한계: 최종 retrain은 CE 중심이라 경량 subnet 성능 회복 여지가 큼.
- 개선 아이디어: max-capacity teacher 기반 3-level KD + latency ratio 기반 adaptive KD 가중.
- 기대 효과: 타이트한 latency target 영역에서 mIoU 회복, Pareto front 상단 개선.
- 검증 실험: CE-only vs fixed KD vs latency-adaptive KD 비교(평균 mIoU, tail latency 구간 성능).
- 리스크/대응: teacher bias/과규제 -> teacher EMA, KD warmup, 계층별 가중치 상한.
- 근거 파일/핵심 문장: `hyundai/nas/train_samplenet.py:25` "`loss_value = loss(outputs, labels)`"; `hyundai/plans/claude_plan_260209.md:245` "CE loss만으로 학습"; `hyundai/plans/claude_plan_260209.md:283` "`latency_ratio = student_latency / teacher_latency`".
- 점수: Novelty 3/5, Impact 4/5, Feasibility 4/5, 개발 난이도 3/5, 총점 14/20.

## 우선순위 (총점 기준)
- 1순위: 개선안 5 (16점)
- 2순위: 개선안 4 (16점)
- 3순위: 개선안 2 (16점)
- 4순위: 개선안 3 (15점)
- 5순위: 개선안 7 (15점)
- 6순위: 개선안 1 (14점)
- 7순위: 개선안 8 (14점)
- 8순위: 개선안 6 (13점)

## 공개 API/인터페이스 변경안
- `hyundai/utils/argument.py`에 추가:
- `--freeze_decoder`, `--decoder_ckpt`, `--search_scope {decoder_fixed,encoder_adapter}`
- `--tau_max`, `--tau_min`, `--tau_schedule_epochs`
- `--pareto_strategy {mixed,zero_cost,evolutionary,hybrid}`, `--pareto_generations`
- `--throughput_window_images`, `--throughput_window_seconds`, `--throughput_batch_size`
- `--predictor_loss {mse,hybrid_rank}`, `--pareto_metrics {hv,igd}`
- `--pareto_feasible_only`, `--loho_eval`
- `--refine_finetune_epochs`, `--refine_finetune_lr`
- `--kd_mode {none,fixed,latency_adaptive}`, `--kd_teacher {max_subnet,ema}`
- 산출물 파일:
- `throughput_eval.json`
- `pareto_metrics.json`
- `pareto_feasible.json`
- `lut_integrity_report.json`
- `predictor_loho.json`

## 테스트 케이스/시나리오
- PAD on/off: 기대-argmax gap, Kendall tau, 최종 mIoU 비교.
- Pareto 전략 비교: mixed vs hybrid(Zero-cost+Evo), 동일 예산에서 HV/IGD 비교.
- LUT 정밀도: projection/interactions 적용 전후 MAPE/MAE 비교(6 하드웨어).
- Throughput 제약 검증: 하드웨어별 `T_h(100)` 5회 반복, p50/p95 및 충족률.
- Multi-fidelity 효과: WS-only vs WS+fine-tune top-k의 최종 선택 regret.
- Predictor 일반화: LOHO(MAE/MAPE/rank corr) + feasible/global Pareto 동시 리포트.
- KD 효과: CE-only vs fixed KD vs latency-adaptive KD의 mIoU/latency trade-off.

## 가정 및 기본값
- 관련 문서 범위는 `hyundai` 하위 코드/스크립트/LUT JSON을 포함.
- 문헌/벤유 표기는 `plans` 문서의 기술을 기준으로 사용(외부 재검증 없음).
- 우선순위 산식은 사용자 선택값(실행가능성 우선) 적용.
- C1~C5는 하드 제약으로 간주하며, 총점과 별개로 제출 전 필수 준수 게이트로 관리.
