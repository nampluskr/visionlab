# 결과 기록

PLAN의 Phase마다 결과 파일을 하나씩 두고, 그 Phase의 모델 노트, 레퍼런스, 이식 메모, 적대적 검증, 이식 결과, 결론을 한 파일에 기록한다. 판정 기준은 category별 Image AUROC·Pixel AUROC 차이가 각각 0.01 이내이며, image metric만 내는 모델(DFKDE, GANomaly)은 Image AUROC만 판정한다. 평가 category는 MVTec `bottle`, VisA `candle`이다.

## 파일 규칙

결과 파일은 다음 규칙으로 만든다.

- 이름은 `NN-<kebab-case>.md`이고 `NN`은 PLAN의 Phase 번호다(예: `00-dataset.md`, `01-stfpm.md`, `02-efficient-ad.md`).
- 파일 첫 문단에 모델 폴더 이름(예: `efficient_ad`)과 원본 경로를 적는다.
- 파일은 해당 Phase를 시작할 때 Claude가 만든다. "구현함"과 "실행으로 검증함"을 구별해 적고, 기준 미달이나 실패도 지우지 않고 원인과 판단을 남긴다.
- 표의 수치에는 근거가 되는 `outputs/`의 run 폴더와 실행 로그 경로를 함께 적는다.

## 모델 Phase 파일 양식

모델 Phase(Phase 1~18)의 결과 파일은 다음 절로 구성한다.

| 절 | 내용 |
|---|---|
| 1. 모델 노트 | 원본 파일, 생성자 시그니처, 모델이 고정하는 `trainer_arguments`, 전처리·정규화, 제공 metric, 외부 의존, 로컬 자산 |
| 2. 레퍼런스 | 실행 일시, run 폴더와 로그 경로, category별 AUROC, 2회 run 차이, 실제 적용된 학습 예산 |
| 3. 이식 메모 | 원본 파일과 이식 파일의 대응, 의존성 제거로 바뀐 줄, 새로 구현한 대체 부품 |
| 4. 적대적 검증 | 회차별(최대 3회) 지적, 심각도, 처리, 남은 위험 |
| 5. 이식 결과 | native Lightning과 순수 PyTorch 추론 각각의 category별 AUROC, 레퍼런스 대비 차이, 판정, run 폴더와 로그 경로 |
| 6. 결론 | 종료 여부(사용자 결정), 미충족 항목, 남은 위험 |

Phase 0(`00-dataset.md`)은 1을 데이터셋 노트로, 2와 5를 데이터 대조 결과(manifest 해시를 데이터 검증 ID로, 파일 목록·라벨·mask·분할·최종 입력의 일치 여부)로 바꾼다.

## 실행 산출물 위치

기계가 만든 근거는 `outputs/`에 둔다. 결과 파일은 이 경로로 근거를 연결한다.

| 대상 | 위치 |
|---|---|
| 레퍼런스 실행 | `outputs/anomalib/<Model>/<DataModule>/<category>/v<N>/` (`anomalib_run.py`가 생성) |
| 이식 결과 실행 (native Lightning) | `outputs/visionlab/<model>/<dataset>_<category>/v<N>/` |
| 순수 PyTorch 가중치·추론 | 같은 run 폴더의 `export/`(가중치와 전후처리 상태), `predict/`(추론 결과와 metric) |
| 데이터 manifest | `outputs/data/<dataset>_<category>_manifest.json` (파일 목록, 라벨, 분할, 해시) |
| agy 실행 로그 | `outputs/logs/<NN>-<step>-<YYMMDD-HHMMSS>.log` |

## 레퍼런스 실행 설정 (모든 Phase 공통)

각 Phase의 레퍼런스는 다음 설정으로 얻는다. 같은 설정으로 2회 실행해 run 간 차이를 함께 기록한다.

| 항목 | 값 |
|---|---|
| 진입점 | `evaluation/anomalib/anomalib_run.py train` (`AnomalibCLI`) |
| 설정 | `default.yaml` + `data/<name>.yaml` + `models/NN_<name>.yaml` (이 순서로 `--config`) |
| 학습 예산 | 기본 `max_epochs=10`, `val_split_mode="from_test"`, `val_split_ratio=0.4`, `seed=42`, batch 8/8, `num_workers=0`, `devices=1`. 모델 yaml의 값과 모델이 고정하는 `trainer_arguments`가 우선한다(dfkde·dfm·padim·patchcore·winclip 1 epoch, fre 220 epoch) |
| 결정적 실행 | `deterministic="warn"`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| 평가 시점 | 학습 종료 시점(마지막 epoch) 모델 |

## 동작 확인 기록: 1 epoch (2026-10-25, 판정용 아님)

`--trainer.max_epochs 1`로 레퍼런스 실행이 끝까지 되는지만 확인했다(실행으로 검증함: 오프라인, seed 42 적용, metric 출력). 대상은 mvtec / bottle이다.

| model | 소요 시간 | image_AUROC | pixel_AUROC | 확인한 것 |
|---|---|---|---|---|
| stfpm | 35초 | 0.9846 | 0.8492 | timm 원래 이름의 로컬 가중치 로딩, `Seed set to 42` |
| fastflow | 28초 | 0.9890 | 0.9546 | backbone 이름 검사 통과 |
| reverse_distillation | 36초 | 0.9649 | 0.9701 | decoder 조회 통과 |
| draem | 11분 54초 | 0.5044 | 0.7002 | yaml의 resize 전용 `pre_processor` 파싱, DTD 로컬 경로 |
| efficient_ad (medium) | 학습 69초 | 0.9583 | 0.7567 | `model_size`·`teacher_file` CLI 덮어쓰기, batch 1 |

- 나머지 13개 모델과 VisA는 아직 실행하지 않았다.
