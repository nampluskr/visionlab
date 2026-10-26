# Phase 4. DFM

모델 폴더 이름은 `dfm`이고 원본은 anomalib 2.0.0의 `anomalib\models\image\dfm\`(`C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\Lib\site-packages\anomalib\models\image\dfm\`)이다. 범위와 완료 조건은 `specs/PLAN.md`의 Phase 4와 "Phase 공통 양식"을 따른다. 판정 기준은 category별 Image·Pixel AUROC 차이 0.01 이내다.

## 상태

다음 표는 단계별 진행 상태다. 수정 반복 한도 3회는 단계별로 센다.

| 단계 | 내용 | 상태 |
|---|---|---|
| 레퍼런스 | `anomalib_run.py train` x 2 (mvtec 1회, visa 1회, 1 epoch) | 실행 완료 (2절) |
| 4-A | 순수 모델 (`models/dfm/model.py`, `components/dimensionality_reduction/` PCA, `tests/04_dfm`의 순수 부분) | 구현, 검증, 실행 완료 (수정 요청 0회) |
| 4-B | Lightning (`DfmLitModel`, configs, PCA buffer 저장·복원, export·Predictor 등록, 비교 스크립트) | 구현, 검증, 실행 완료 (수정 요청 0회, 결과 5절). visa Pixel AUROC는 정규화 정의 기준 통과 (사용자 결정 2026-10-26) |

처리 구성요소(PreProcessor, PostProcessor, Evaluator), 특징 추출기(`TimmFeatureExtractor`), `DynamicBufferMixin`은 앞 Phase의 것을 재사용한다.

## 사용자 결정 (2026-10-26)

다음은 Phase 시작 전에 사용자가 정한 항목이다.

| 항목 | 결정 |
|---|---|
| 학습 에폭 | 모델이 고정한 1 epoch을 따른다(PaDiM과 같은 예외). 레퍼런스 yaml과 native yaml 모두 `max_epochs: 1` |
| FR-5 허용 오차 | `atol=1e-6, rtol=1e-5` (Phase 2, 3과 같음) |
| `score_type` | 원본 `SingleClassGaussian`(`nll` 경로)은 복사해 둔다. 실행 검증은 기본값 `fre`로만 하고 `nll`은 모델 단위 테스트로만 다룬다 |
| PCA 상태 차이 | 레퍼런스와 native의 PCA 상태(`mean`, `num_components`, 부분공간)는 정보로 기록하고 판정은 AUROC 0.01로만 한다 |
| visa Pixel AUROC 판정 (결과 확인 후 추가, 2026-10-26) | 레퍼런스와 같은 정의(정규화·clamp된 map의 AUROC)로 비교한다. native 0.748692 대 레퍼런스 0.748692로 통과다. 레퍼런스 raw 값을 구하기 위한 추가 실행은 하지 않는다 |

## 1. 모델 노트

### 1.1 원본과 생성자

다음 표는 원본 파일과 이식 위치의 대응이다(D-15, D-18).

| 원본 (`anomalib\models\image\dfm\`) | 이식 위치 | 비고 |
|---|---|---|
| `torch_model.py` (`DFMModel`, `SingleClassGaussian`) | `src/visionlab/anomaly_detection/models/dfm/model.py` | import 변경, `pre_trained`는 순수 모델에서 구조만 만드는 방식으로(D-19) |
| `lightning_model.py` (`Dfm`) | `.../models/dfm/lightning_model.py` (`DfmLitModel`) | native `LightningModule`, `MemoryBankMixin` hook을 직접 구현(D-11) |
| `__init__.py` | `.../models/dfm/__init__.py` | `lightning_model`을 import하지 않도록 고침(D-18 예외) |
| `components/dimensionality_reduction/pca.py` (`PCA`) | `.../models/components/dimensionality_reduction/pca.py` | 새로 복사. Phase 5 DFKDE가 재사용한다 |
| `components/base/dynamic_buffer.py` | `.../models/components/base/dynamic_buffer.py` | Phase 3에서 복사한 것을 재사용 |

생성자는 다음과 같다. `DfmLitModel`은 `Dfm`과 인자 이름·순서·기본값이 같아야 한다(FR-6, D-5).

| 클래스 | 시그니처 |
|---|---|
| `Dfm` | `backbone="resnet50"`, `layer="layer3"`, `pre_trained=True`, `pooling_kernel_size=4`, `pca_level=0.97`, `score_type="fre"`, `pre_processor=True`, `post_processor=True`, `evaluator=True`, `visualizer=True` (10개) |
| `DFMModel` | `backbone`, `layer`, `pre_trained=True`, `pooling_kernel_size=4`, `n_comps=0.97`, `score_type="fre"` |

### 1.2 학습과 평가 동작

원본의 동작은 다음과 같다.

- **모델:** `TimmFeatureExtractor`(eval)로 `layer` 하나의 특징을 뽑고, `pooling_kernel_size > 1`이면 `avg_pool2d`를 적용해 `(batch, -1)`로 편다. 학습 모드의 `get_features`는 이 특징 벡터만 반환한다.
- **학습:** `training_step`은 `get_features(...).squeeze()`를 `self.embeddings`에 모으고 더미 loss(0)를 반환한다. `configure_optimizers`는 `None`이다. 학습 가능한 파라미터가 없다.
- **적합(`fit`):** `MemoryBankMixin`의 `on_validation_start`와 `on_train_epoch_end`가 `_is_fitted` buffer가 False일 때 한 번 호출한다. `torch.vstack(self.embeddings)`에 PCA를 맞춘다. `score_type="nll"`이면 변환된 특징에 `SingleClassGaussian`도 맞춘다.
- **PCA:** `fit`은 입력에서 평균을 in-place로 빼고 `double` SVD를 한다. `n_components <= 1`이면 분산 비율이 처음 기준 이상이 되는 인덱스를 성분 수로 쓴다(`+1`을 하지 않는다). 성분 수가 정수이면 그 값을 쓴다. `transform`도 입력에서 평균을 in-place로 뺀다. 원본 동작이므로 고치지 않는다.
- **추론(`fre`):** 특징을 PCA로 투영하고 역투영한 재구성과의 차이를 제곱한다. `anomaly_map`은 채널 합을 입력 크기로 bilinear 업샘플한 것이고 `pred_score`는 샘플별 합이다. `nll`이면 `anomaly_map`이 없다.
- **`trainer_arguments`:** `{"gradient_clip_val": 0, "max_epochs": 1, "num_sanity_val_steps": 0}`이다. 사용자 결정에 따라 native도 1 epoch이다.

### 1.3 외부 의존과 로컬 자산

- **외부 의존:** timm, torchvision이다. kornia, FrEIA, open_clip은 쓰지 않는다. anomalib 의존은 `Batch`, `InferenceBatch`, `AnomalibModule`, `MemoryBankMixin`, `PCA`, `DynamicBufferMixin`, `PreProcessor`, `PostProcessor`, `Evaluator`, `Visualizer`, `LearningType`이다.
- **로컬 가중치:** `paths.toml [backbones]`의 `resnet50`(이미 있어 새 항목은 필요 없다). 레퍼런스는 `local_setup.py`가 timm 이름을 로컬 로더로 바꾼다. visionlab은 `DfmLitModel.__init__`이 `feature_extractor`에 주입한다(D-19).
- **데이터:** MVTec `bottle`, VisA `candle`. 분할은 `from_test`, 0.4, seed 42, batch 8/8, `num_workers=0`이다.
- **학습 예산:** `max_epochs=1`(모델 고정, 위 결정)이다.

### 1.4 이식에서 의도적으로 달라지는 곳

- **pretrained 주입(D-19):** 순수 `DFMModel`은 `pre_trained=False`로 구조만 만든다. `DfmLitModel`이 `resnet50` 로컬 파일을 `feature_extractor`에 불러온다. 생성자 인자 수는 유지한다.
- **`MemoryBankMixin` 대체(D-11):** 공유 기반 클래스를 만들지 않고 `_is_fitted` buffer, `on_validation_start`, `on_train_epoch_end`를 `DfmLitModel`에 직접 쓴다(Phase 3과 같다).
- **`DynamicBufferMixin`:** `PCA`의 `mean`, `singular_vectors`, `num_components`는 빈 텐서로 만들어지므로 `load_state_dict`가 크기를 맞추도록 Phase 3에서 복사한 mixin을 재사용한다. `singular_values`는 buffer가 아니라 state에 들어가지 않는다(원본 그대로).
- **원본 동작 유지:** `PCA.fit`의 성분 수 인덱스(`+1` 없음), `PCA.fit`과 `transform`의 in-place 평균 빼기, `training_step`의 `squeeze()`, `SingleClassGaussian.fit`의 `torch.mean(..., device=...)`는 고치지 않는다.
- **히스토리 비교:** loss가 없고 1 epoch이라 `compare_history.py`는 쓰지 않는다. 대신 레퍼런스 ckpt와 native ckpt의 PCA 통계(`mean`, `num_components`, `singular_vectors`의 부분공간 거리)를 표로 기록한다(판정 아님).
- **FR-5 허용 오차:** `pred_score`와 `anomaly_map`은 `atol=1e-6, rtol=1e-5`, backbone 가중치는 `torch.equal`이다.

## 2. 레퍼런스

학습 예산은 모델이 고정한 1 epoch이다(사용자 결정, 2026-10-26). `models/04_dfm.yaml`에 `trainer.max_epochs: 1`이 있고, 저장된 `lightning_logs/version_0/config.yaml`에도 `max_epochs: 1`이 남았다.

### 2.1 1 epoch 레퍼런스

2026-10-26에 agy가 `anomalib_run.py train --config default.yaml --config data/<data>.yaml --config models/04_dfm.yaml`을 데이터셋마다 1회 실행했다(실행으로 검증함). Claude는 로그 파일에서 값을 직접 확인했다. 두 로그 모두 `Seed set to 42`와 `Trainer.fit stopped: max_epochs=1 reached`가 나오고 `network access is blocked`, `URLError`, `ConnectionError`, `Traceback`은 0건이다. 로그는 PowerShell `Tee-Object` 출력이라 UTF-16이다. 종료 코드는 둘 다 0이고 소요 시간은 mvtec 약 30.8초, visa 약 46.5초다(agy 보고).

agy가 PowerShell에서 실행한 명령은 다음과 같다(Claude가 보낸 지시 원문. 파일은 고치지 않았다).

```powershell
cd D:\projects\visionlab\evaluation\anomalib
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/04_dfm.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\04-ref-mvtec-261026-1.log
& $py anomalib_run.py train --config default.yaml --config data/visa.yaml --config models/04_dfm.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\04-ref-visa-261026-1.log
```

다음 표는 1 epoch 레퍼런스(1회) 결과다.

| 데이터 / category | Image AUROC | Pixel AUROC | 참고 | run 폴더 | 로그 |
|---|---|---|---|---|---|
| mvtec / bottle | 1.00000 (0.99999994) | 0.93763 | image F1 0.9867, pixel F1 0.4891 | `outputs/anomalib/Dfm/LocalMVTecAD/bottle/v0` | `outputs/logs/04-ref-mvtec-261026-1.log` |
| visa / candle | 0.96111 | 0.74869 | image F1 0.9268, pixel F1 0.1258 | `outputs/anomalib/Dfm/LocalVisa/candle/v0` | `outputs/logs/04-ref-visa-261026-1.log` |

실제 적용된 학습 예산은 `max_epochs=1`, `val_split_mode=from_test`, `val_split_ratio=0.4`, `seed=42`, `train_batch_size=8`, `eval_batch_size=8`, `num_workers=0`, `deterministic=warn`, `backbone=resnet50`, `layer=layer3`, `pre_trained=true`, `pooling_kernel_size=4`, `pca_level=0.97`, `score_type=fre`다(저장된 `config.yaml`). 체크포인트는 각 run 폴더의 `weights/lightning/model.ckpt`다.

### 2.2 에폭별 학습 히스토리

DFM은 loss가 없고 학습이 1 epoch이라 비교할 히스토리가 없다. 레퍼런스와 native의 비교는 적합 결과(PCA 통계)로 대신하며 5.4절에서 다룬다.

### 2.3 레퍼런스 확인 사항

- 두 데이터셋 모두 종료 코드 0이고 오류가 없다.
- visa 레퍼런스의 Pixel AUROC 0.74869는 mvtec(0.93763)보다 크게 낮다. 원인은 모델이 아니라 지표 계산 방식이며 5.2절에서 규명했다.

## 3. 이식 메모

### 3.1 원본과 이식 파일의 대응

다음 표는 새로 만든 파일과 원본의 대응이다. 원본과의 차이는 `diff`로 확인했다(구현함, 정적 검증).

| 이식 파일 (`src/visionlab/anomaly_detection/models/`) | 원본 | 원본과 다른 줄 |
|---|---|---|
| `dfm/model.py` | `dfm/torch_model.py` | import 4줄(`InferenceBatch`, `DynamicBufferMixin`, `PCA`, `TimmFeatureExtractor`), `TimmFeatureExtractor(... pre_trained=False)`와 주석 한 줄(D-19) |
| `dfm/__init__.py` | `dfm/__init__.py` | `DFMModel`만 노출(D-18 예외) |
| `dfm/lightning_model.py` (`DfmLitModel`) | `dfm/lightning_model.py`와 `components/base/memory_bank_module.py`의 hook | native `LightningModule`로 새로 작성. 10개 인자 동일, `_is_fitted` buffer, `fit`, `on_validation_start`, `on_train_epoch_end`, `squeeze()`, 더미 loss는 원본 그대로, 로컬 backbone 주입(D-19)과 D-8의 post_processor 갱신 제한 |
| `components/dimensionality_reduction/pca.py` | `components/dimensionality_reduction/pca.py` | `DynamicBufferMixin` import 한 줄 |
| `components/dimensionality_reduction/__init__.py` | 같은 이름 | `PCA`만 노출 |

### 3.2 새로 쓴 코드와 앞 Phase 공유 파일 변경

- **새로 쓴 파일:** `configs/models/dfm.yaml`, `evaluation/anomalib/compare_dfm_stats.py`(정보 기록용), `tests/04_dfm/test_dfm.py`(14개 테스트), `tests/04_dfm/test_dfm_lightning.py`(8개 테스트).
- **공유 파일 변경:** `utils/io.py`(`load_export`가 `dfm` 허용), `inference/predictor.py`(모델 dict에 `dfm`), `evaluation/visionlab/predict.py`(`--model dfm`), `evaluation/anomalib/compare_outputs.py`(`--model dfm`), `tests/common/test_cli.py`, `tests/common/test_dependency.py`. `common/metric.py`는 이번 Phase에서 바꾸지 않았다.
- **새 부품:** `PCA`는 원본 복사다. 새로 구현한 대체 부품은 없다.

### 3.3 이식 중 확인된 사항

- **`SingleClassGaussian.fit`의 TypeError:** 원본이 `torch.mean(dataset, dim=1, device=dataset.device)`를 호출하는데, 설치된 torch의 `mean`에는 `device` 인자가 없다. 따라서 `score_type="nll"`의 적합은 anomalib 원본에서도 `TypeError`로 막힌다. codex의 질문에 "원본 유지"로 답했고(사용자 결정 "nll은 모델 단위 테스트로만"과 같은 방향), `fit`의 TypeError를 명시한 테스트와 합성 buffer로 NLL 계산·복원을 확인하는 테스트를 두었다. 이 경로는 실행 검증 범위 밖이다.
- **BatchNorm 상태:** 가중치는 로컬 `resnet50`이고 학습 중 BN 통계는 갱신되지 않는다(레퍼런스와 같다).

## 4. 적대적 검증

검증은 4-A, 4-B 각각 한 회차씩 했다. 수정 요청은 4-A 0회, 4-B 0회(한도 3회 이내)다.

### 4.1 4-A (순수 모델과 PCA)

| 지적 | 심각도 | 처리 |
|---|---|---|
| 원본과의 `diff`: `model.py`는 import와 D-19의 `pre_trained=False` 한 줄뿐이고 `SingleClassGaussian`, `get_features`, `score`, `forward`가 같음. `pca.py`는 `DynamicBufferMixin` import 한 줄뿐이고 성분 수 인덱스, in-place 평균 빼기, double SVD가 유지됨 | 확인 | 수정 없음 |
| codex 질문: `SingleClassGaussian.fit`의 `device=` 인자가 TypeError를 냄 (3.3절) | 낮음 | 원본 유지로 답함 |

실행으로 검증함: `tests/04_dfm/test_dfm.py` 14개 통과(`04-pytest-a-261026-1.log`).

### 4.2 4-B (Lightning, export, 비교 스크립트)

| 지적 | 심각도 | 처리 |
|---|---|---|
| 정적 검토: `DfmLitModel`의 `fit`, 두 hook의 `_is_fitted` 조건, 더미 loss, `configure_optimizers`, `squeeze()`가 원본과 같음. 로컬 backbone 주입은 `pre_trained`가 True일 때만. `trainer_arguments`가 원본과 같음 | 확인 | 수정 없음 |
| 정적 검토: 공유 파일 변경은 `dfm` 등록 한 줄씩이며 앞 모델 동작을 바꾸지 않음. `compare_outputs.py`의 dfm 분기는 PCA 상태를 복제본으로 비교해 in-place 부수효과가 비교를 오염시키지 않음 | 확인 | 수정 없음 |
| 실행으로 발견: visa Pixel AUROC가 레퍼런스와 0.21 다름 (5.2절). 코드 결함이 아니라 지표 정의 차이로 규명됨 | 높음 (판정 문제) | 수정 요청 없음. 사용자 결정으로 비교 정의를 확정함 (6절) |

## 5. 이식 결과

### 5.1 최종 판정 (1 epoch, 기준 차이 0.01 이내)

native는 `train.py`가 test에서 기록한 값이고 순수 PyTorch는 `export/model.pt`만 불러 `predict.py`로 평가한 값이다. 모두 실행으로 검증했다.

| 데이터 / category | 지표 | 레퍼런스 | native | 순수 PyTorch | 차이(native) | 차이(순수) | 판정 |
|---|---|---:|---:|---:|---:|---:|---|
| mvtec / bottle | Image AUROC | 0.999999940 | 1.000000000 | 1.000000000 | 6e-8 | 6e-8 | 통과 |
| mvtec / bottle | Pixel AUROC | 0.937633514 | 0.937633872 | 0.93763387 | 4e-7 | 4e-7 | 통과 |
| visa / candle | Image AUROC | 0.961111128 | 0.961111069 | 0.96111107 | 6e-8 | 6e-8 | 통과 |
| visa / candle | Pixel AUROC | 0.748692274 | 0.960558414 | 0.96055847 | 0.2119 | 0.2119 | 통과 (정규화 정의 기준, 사용자 결정 2026-10-26) |

visa / candle Pixel AUROC 행은 raw map 기준의 값(native 0.960558)을 그대로 비교하면 0.2119 차이로 기준 0.01을 넘는다. 레퍼런스가 정규화·clamp된 map으로 계산하기 때문이며(5.2절), 사용자 결정에 따라 같은 정의로 비교한 값(native 0.748692, 레퍼런스 0.748692)으로 판정한다. raw 0.960558은 참고로 남긴다.

참고로 지표 정의와 무관한 F1은 두 데이터셋 모두 레퍼런스, native, 순수 PyTorch가 소수 여덟째 자리까지 같다(mvtec image 0.98666668, pixel 0.48912215, visa image 0.92682928, pixel 0.12576771).

run 폴더와 로그는 다음과 같다.

| 대상 | 위치 |
|---|---|
| 레퍼런스 | `outputs/anomalib/Dfm/LocalMVTecAD/bottle/v0`, `outputs/anomalib/Dfm/LocalVisa/candle/v0` |
| native | `outputs/visionlab/dfm/mvtec_bottle/v0`, `outputs/visionlab/dfm/visa_candle/v0` (`run.json`: `max_epochs` 1, `num_sanity_val_steps` 0, `gradient_clip_val` 0.0, 가중치 `D:\backbones\resnet50-0676ba61.pth` sha256 `0676ba61...fb8a`) |
| 순수 PyTorch | 같은 run의 `predict/metrics.json` |
| 로그 | `outputs/logs/04-native-{mvtec,visa}-261026-1.log`, `04-pure-{mvtec,visa}-261026-1.log` |

네 native·순수 로그에 `network access is blocked`, `URLError`, `ConnectionError`, `Traceback`은 0건이다.

### 5.2 visa Pixel AUROC 불일치의 원인

모델 출력은 같고 지표 계산 방식만 다르다는 증거는 다음과 같다(실행으로 검증함).

- **같은 상태의 출력(FR-5):** `compare_outputs.py --model dfm`이 mvtec·visa 실제 샘플 8장의 `pred_score`와 `anomaly_map`에서 최대 절대 차이 0.0을 냈다(두 레퍼런스 체크포인트 각각, PASS 23건, FAIL 0건).
- **F1과 Image AUROC:** 위 5.1절대로 같다. 후처리된 pixel 예측으로 계산하는 pixel F1도 같다.
- **진단(읽기 전용):** `diag_norm_dfm.py`(프로젝트 밖 scratchpad의 스크립트, `04-diag-norm-261026-1.log`)가 native export로 visa 전체 test(120장)를 평가해 raw와 정규화 점수의 Pixel AUROC를 비교했다.

| 데이터 | Pixel AUROC (raw anomaly_map) | Pixel AUROC (normalized_anomaly_map) | 정규화 map에서 0.0 | 정규화 map에서 1.0 |
|---|---:|---:|---:|---:|
| mvtec / bottle | 0.937634 | 0.937634 | 0 / 3,276,800 | 4,062 / 3,276,800 |
| visa / candle | 0.960558 | 0.748692 | 7,694,135 / 7,864,320 (97.8%) | 0 |

정규화된 map의 Pixel AUROC 0.748692는 레퍼런스의 0.748692와 소수 여섯째 자리까지 같다. 즉 anomalib의 test AUROC는 후처리로 [0, 1]에 정규화하고 clamp한 map으로 계산되는데, visa/candle에서는 pixel의 97.8%가 0.0으로 clamp되어 동점이 되고 AUROC가 0.96에서 0.75로 떨어진다. visionlab은 D-8과 Phase 3의 `AUROC`(순위 보존)에 따라 raw map으로 계산하므로 clamp의 영향이 없다. 정규화식과 임계값의 구체적 원인(validation의 min, max, 임계값)은 더 분해하지 않았다.

이 현상은 PaDiM(Phase 3)에서 드러난 "raw 점수와 정규화 점수의 AUROC 차이"의 반대 방향이다. Phase 3에서는 visionlab의 계산이 sigmoid 포화로 낮았고 고쳤다. 이번에는 레퍼런스의 계산이 clamp로 낮다. mvtec에서는 clamp가 4,062 pixel(0.12%)뿐이라 두 값이 같다.

### 5.3 테스트와 비교 스크립트

다음 표는 agy가 실행한 묶음의 결과다(Claude가 로그 파일에서 확인).

| 대상 | 결과 | 로그 |
|---|---|---|
| `tests\04_dfm` | 22 passed | `04-pytest-261026-1.log` |
| `tests\common` | 28 passed | `04-pytest-common-261026-1.log` |
| `tests\01_stfpm` | 32 passed | `04-pytest-stfpm-261026-1.log` |
| `tests\02_efficient_ad` | 22 passed | `04-pytest-efficient-ad-261026-1.log` |
| `tests\03_padim` | 16 passed | `04-pytest-padim-261026-1.log` |
| `compare_outputs.py --model dfm` (mvtec 체크포인트, visa 체크포인트) | PASS 23, FAIL 0 (각각) | `04-compare-outputs-{mvtec,visa}-261026-1.log` |
| `compare_outputs.py --model padim / stfpm / efficient_ad` | PASS 28 / 38 / 46, FAIL 0 | `04-compare-{padim,stfpm,efficient-ad}-261026-1.log` |

`tests\00_dataset`은 이번 변경과 무관해 다시 돌리지 않았다. 14개 묶음 명령의 종료 코드는 모두 0이었다(agy 보고).

실행한 명령은 다음과 같다(Claude가 보낸 지시 원문. agy가 시간 측정 줄을 덧붙여 하나의 스크립트로 순서대로 실행했고, 승인 창의 전문을 Claude가 agy의 transcript에서 읽어 지시와 같음을 확인한 뒤 한 번 승인했다).

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py -m pytest tests\04_dfm 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\04-pytest-261026-1.log
& $py -m pytest tests\common 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\04-pytest-common-261026-1.log
& $py -m pytest tests\01_stfpm 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\04-pytest-stfpm-261026-1.log
& $py -m pytest tests\02_efficient_ad 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\04-pytest-efficient-ad-261026-1.log
& $py -m pytest tests\03_padim 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\04-pytest-padim-261026-1.log
& $py evaluation/visionlab/train.py train --data mvtec --model dfm 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\04-native-mvtec-261026-1.log
& $py evaluation/visionlab/predict.py --model dfm --data mvtec 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\04-pure-mvtec-261026-1.log
& $py evaluation/visionlab/train.py train --data visa --model dfm 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\04-native-visa-261026-1.log
& $py evaluation/visionlab/predict.py --model dfm --data visa 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\04-pure-visa-261026-1.log
& $py evaluation/anomalib/compare_outputs.py --model dfm --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\Dfm\LocalMVTecAD\bottle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\04-compare-outputs-mvtec-261026-1.log
& $py evaluation/anomalib/compare_outputs.py --model dfm --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\Dfm\LocalVisa\candle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\04-compare-outputs-visa-261026-1.log
& $py evaluation/anomalib/compare_outputs.py --model padim --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\Padim\LocalMVTecAD\bottle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\04-compare-padim-261026-1.log
& $py evaluation/anomalib/compare_outputs.py --model stfpm 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\04-compare-stfpm-261026-1.log
& $py evaluation/anomalib/compare_outputs.py --model efficient_ad 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\04-compare-efficient-ad-261026-1.log
```

4-A 단독 pytest는 `& $py -m pytest tests\04_dfm\test_dfm.py`(`04-pytest-a-261026-1.log`), 읽기 전용 진단은 scratchpad의 `diag_norm_dfm.py`(`04-diag-norm-261026-1.log`), 통계 비교는 `compare_dfm_stats.py`(5.4절)다. agy는 파일을 고치지 않았다.

### 5.4 PCA 통계 비교 (정보, 판정 아님)

다음 표는 `compare_dfm_stats.py`의 출력이다(`04-stats-{mvtec,visa}-261026-1.log`, `outputs/compare/dfm_stats_{mvtec,visa}.json`).

| 데이터 | `num_components` (레퍼런스 / native) | `mean` 최대 절대 차이 | `singular_vectors` shape | 부분공간 거리 |
|---|---|---:|---|---:|
| mvtec / bottle | 170 / 170 | 0.0 | [16384, 170] | 3.2e-4 |
| visa / candle | 689 / 689 | 0.0 | [16384, 689] | 1.5e-4 |

성분 수와 평균이 같고, 부분공간 거리가 아주 작다(SVD의 GPU 수치 오차로 보이며 FRE 점수에는 영향이 없다).

## 6. 결론

- **레퍼런스:** 두 데이터셋 모두 확보했다.
- **mvtec / bottle:** Image·Pixel AUROC가 native와 순수 PyTorch 모두 레퍼런스와 4e-7 이내다. 통과.
- **visa / candle Image AUROC:** 6e-8 차이로 통과.
- **visa / candle Pixel AUROC:** raw map 값을 문자 그대로 비교하면 0.2119 차이로 기준 0.01을 넘는다. 사용자 결정(2026-10-26)에 따라 레퍼런스와 같은 정의로 비교해 통과로 판정한다. 원인은 구현이 아니라 지표 정의다. 모델 출력과 학습 결과가 같고(FR-5 최대 차이 0.0, PCA 성분 수와 평균 일치, F1 일치), 레퍼런스의 값 0.748692는 정규화·clamp된 map의 AUROC와 일치하며 visionlab의 raw map AUROC는 0.960558이다(5.2절).
- **FR-5, FR-6, FR-23, FR-24:** 통과(5.3절). NFR 판정(의존 분리, 로컬 가중치)은 `tests/common`과 `tests/04_dfm`이 통과했다.
- **수정 요청:** 4-A 0회, 4-B 0회.
- **사용자 결정(2026-10-26):** visa Pixel AUROC는 레퍼런스와 같은 정의(정규화·clamp된 map의 AUROC)로 비교한다. native 0.748692 대 레퍼런스 0.748692로 통과다. 레퍼런스의 raw 값을 구하기 위한 추가 실행은 하지 않는다. 검토한 다른 후보(D-8의 raw AUROC로 비교하며 레퍼런스 raw 값을 따로 구함, 미충족으로 두고 `specs/`의 기준 정의를 바꿈)는 채택하지 않았다.
- **Phase 종료:** 사용자가 종료를 결정했다(2026-10-26).

### 6.1 `specs/` 갱신이 필요한 항목

`specs/`는 사람이 쓴다. Claude는 고치지 않고 다음을 알린다.

- 앞 Phase에서 이어진 항목: `paths.toml [teachers]`가 D-12, D-15, FR-23에 없는 것, D-22 후보(모델이 고정한 에폭의 예외와 FRE 220 epoch 행의 충돌), kornia 예외, D-8의 raw AUROC가 순위 보존 계산이라는 D-20 보충.
- 이번 Phase 추가: Pixel AUROC 비교 정의. 레퍼런스(anomalib)의 test AUROC는 정규화·clamp된 점수로 계산되어 점수 분포에 따라 크게 낮아질 수 있다(visa/candle 0.96에서 0.75). 판정 기준 "레퍼런스와 0.01 이내"를 어느 정의로 비교할지는 이번 Phase에서 사용자가 레퍼런스와 같은 정의로 정했고(6절), 이를 `specs/`에 반영해야 한다. FRE 같이 점수 규모가 큰 모델(Phase 5 DFKDE, Phase 9 FRE 등)에서 반복될 수 있다.

### 6.2 남은 위험

- 레퍼런스가 1회뿐이라 run 간 편차는 측정하지 못했다(DFM은 무작위 요소가 적어 큰 편차는 없을 것으로 보이나 확인하지 않았다).
- 레퍼런스의 raw map AUROC는 직접 구하지 않았다. FR-5 출력 일치와 정규화 AUROC 일치에서 추론했다.
- `score_type="nll"` 경로는 원본 `fit`이 TypeError라 실행 검증을 하지 못했다.
- `tests/00_dataset`은 다시 돌리지 않았다.
