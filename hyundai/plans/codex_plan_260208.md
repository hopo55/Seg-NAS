# Codex Hyundai NAS 개선안 (Top-tier 제출용)

작성일: 2026-02-09  
대상: `hyundai` 방법론
작성: `Codex`

## 0. 필수 고정 조건 (절대 변경 불가)
1. Pretrained Decoder Utilization (디코더 수정 금지)
2. Multi-Hardware Latency-Aware Optimization
3. 100-Image / 16-Second Throughput Constraint
4. Measured LUT-Based Latency Modeling (실측 LUT 기반)
5. Accuracy-Latency Multi-Objective NAS

---

## 1. 현재 코드 기준 핵심 진단

### 강점
- 산업 특화 연산(search space)과 latency-aware NAS가 이미 구현됨.
- LUT + predictor + Pareto 기반 다중 하드웨어 선택 파이프라인이 존재함.

### 즉시 보완 필요
- 현재 구조는 pretrained **encoder** + searchable decoder 형태임.  
  고정 조건(Pretrained Decoder)과 방향이 다름.
- Jetson LUT 메타데이터 불일치 가능성:
  - `hyundai/latency/luts/lut_jetsonorin.json`의 `hardware` 값이 `CPU`
  - predictor 로딩 시 하드웨어 키 매칭 누락 위험

---

## 2. 2025+ Top-tier 대비 핵심 개선 방향

### A. Decoder 고정형 NAS로 구조 전환 (조건 1 준수)
- Decoder를 사전학습 가중치로 로드하고 `freeze`.
- Decoder 내부 layer/op/channel/skip 구조 변경 금지.
- 탐색 변수는 decoder 외부로 제한:
  - encoder width
  - adapter block
  - input resolution
  - inference batch policy

### B. Multi-hardware 제약식 명시화 (조건 2, 3, 5 준수)
- 하드웨어 집합: `A6000, RTX3090, RTX4090, JetsonOrin, RaspberryPi5, Odroid`
- 모든 하드웨어에 대해 제약:
  - `T_h(100 images) <= 16 sec`
- 목적:
  - `maximize mIoU`
  - subject to throughput constraints for all hardware

### C. 실측 LUT 확장 (조건 4 준수)
- LUT 축 확장:
  - `layer × op × width × resolution × batch`
- 학습 중 latency는 FLOPs proxy 금지, LUT 기반으로만 계산.
- 전체 모델 실측(100장 wall-clock)으로 LUT 합산 오차 보정 캘리브레이터 추가.

### D. Multi-objective NAS 고도화 (조건 5)
- feasible Pareto(제약 만족)와 global Pareto(전체) 동시 보고.
- ranking-aware predictor loss 도입:
  - MSE + pairwise rank + listwise rank
- preference-conditioned NAS 헤드 추가:
  - 한 번의 탐색으로 다양한 정확도-지연 타협점 생성

---

## 3. 최신 논문 대비 차별화 포인트

- CVPR 2025 Subnet-aware supernet 학습 관점을 도입해 ranking 안정성 보강
- ICCV 2025 predictor loss 연구 방향 반영 (MSE 단일 손실 한계 해소)
- ICCV 2025 CARL 관점 반영 (하드웨어 분포이동 일반화 평가 추가)
- ICLR 2025 MODNAS 방향 반영 (preference-conditioned multi-objective search)
- ICCV 2025 TRNAS / ICLR 2025 NEAR 방향 반영 (저비용 후보 pruning)

---

## 4. 실험 설계 (논문 제출용)

### 필수 리포트 지표
- mIoU
- hardware별 `T_h(100)` (100장 wall-clock)
- hardware별 img/s
- throughput 제약 충족률
- Pareto hypervolume

### 재현성 설정
- seed 5회 이상 (`0,1,2,3,4`)
- hardware별 반복 실측 5회 이상
- 평균, 표준편차, p50, p95 보고

### 비교군
- 기존 Hyundai baseline
- subnet-aware 학습 버전
- predictor-loss 개선 버전
- preference-conditioned 버전
- 저비용 후보 pruning 버전

---

## 5. 구현 우선순위

### P0 (바로 수정)
1. LUT 무결성 검사 추가 (파일명/JSON hardware 일치)
2. Jetson LUT 메타데이터 정합성 수정
3. 100장/16초 throughput evaluator 추가

### P1 (핵심 성능)
1. Decoder freeze 강제 옵션
2. NAS 탐색 변수의 decoder 외부 제한
3. multi-hardware constrained objective 적용

### P2 (논문 경쟁력)
1. ranking-aware predictor loss
2. preference-conditioned NAS
3. leave-one-hardware-out 일반화 평가

---

## 6. 코드 반영 시 공개 인터페이스(제안)

- `--freeze_decoder true`
- `--decoder_ckpt <path>`
- `--throughput_window_images 100`
- `--throughput_window_seconds 16`
- `--predictor_loss {mse,pairwise,listwise,hybrid}`
- `--preference_conditioned true`

산출물:
- `throughput_eval.json`
- `pareto_feasible.json`
- `predictor_generalization.json`

---

## 7. 결론
- 현재 기반은 산업 적용성 측면에서 강점이 분명함.
- 다만 top-tier 기준에서는
  - decoder 고정 조건 준수 구조
  - throughput 제약의 정식화
  - predictor 일반화/랭킹 품질
  - 다목적 탐색의 표현력
  보강이 필요함.
- 위 개선안은 5개 고정 조건을 유지하면서, 2025+ 최신 연구 흐름과 정합되도록 설계됨.
