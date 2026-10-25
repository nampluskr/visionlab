# Phase 3. PaDiM

모델 폴더 이름은 `padim`이고 원본은 anomalib 2.0.0의 `anomalib\models\image\padim\`(`C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\Lib\site-packages\anomalib\models\image\padim\`)이다. 범위와 완료 조건은 `specs/PLAN.md`의 Phase 3과 "Phase 공통 양식"을 따른다. 판정 기준은 category별 Image·Pixel AUROC 차이 0.01 이내다.

## 상태

다음 표는 단계별 진행 상태다. 수정 반복 한도 3회는 단계별로 센다.

| 단계 | 내용 | 상태 |
|---|---|---|
| 레퍼런스 | `anomalib_run.py train` x 2 (mvtec 1회, visa 1회, 1 epoch) | 실행 완료 (2절) |
| 3-A | 순수 모델 (`models/padim/model.py`, `anomaly_map.py`, `components/` 부품, `tests/03_padim`의 순수 부분) | 구현, 검증, 실행 완료 (수정 요청 0회) |
| 3-B | Lightning (`PadimLitModel`, configs, 통계 buffer 저장·복원, export·Predictor 등록, 비교 스크립트) | 구현, 검증, 실행 완료 (수정 요청 1회, 결과 5절) |

처리 구성요소(PreProcessor, PostProcessor, Evaluator)와 특징 추출기(`TimmFeatureExtractor`)는 Phase 1의 것을 재사용한다.

## 사용자 결정 (2026-10-25)

다음은 Phase 시작 전에 사용자가 정한 항목이다.

| 항목 | 결정 |
|---|---|
| 학습 에폭 | 모델이 고정한 1 epoch을 따른다. 10 epoch 고정 규칙의 예외다. 레퍼런스 yaml과 native yaml 모두 `max_epochs: 1` |
| `GaussianBlur2d`의 kornia | kornia(0.8.2)를 설치된 의존으로 허용하고 원본처럼 import한다. 독립 구현하지 않는다 |
| FR-5 허용 오차 | `atol=1e-6, rtol=1e-5` (Phase 2와 같음) |
| `idx` 불일치 | native의 `idx`가 레퍼런스와 달라도 주입하지 않는다. 판정은 AUROC 0.01로만 하고, `idx`와 통계의 차이는 정보로 기록한다 |

## 1. 모델 노트

### 1.1 원본과 생성자

다음 표는 원본 파일과 이식 위치의 대응이다(D-15, D-18).

| 원본 (`anomalib\models\image\padim\`) | 이식 위치 | 비고 |
|---|---|---|
| `torch_model.py` (`PadimModel`) | `src/visionlab/anomaly_detection/models/padim/model.py` | import 변경, `pre_trained`는 순수 모델에서 구조만 만드는 방식으로(D-19) |
| `anomaly_map.py` (`AnomalyMapGenerator`) | `.../models/padim/anomaly_map.py` | `GaussianBlur2d` import 위치만 변경 |
| `lightning_model.py` (`Padim`) | `.../models/padim/lightning_model.py` (`PadimLitModel`) | native `LightningModule`, `MemoryBankMixin` hook을 직접 구현(D-11) |
| `__init__.py` | `.../models/padim/__init__.py` | `lightning_model`을 import하지 않도록 고침(D-18 예외) |
| `components/stats/multi_variate_gaussian.py` | `.../models/components/stats/multi_variate_gaussian.py` | 새로 복사 |
| `components/base/dynamic_buffer.py` | `.../models/components/base/dynamic_buffer.py` | 새로 복사 |
| `components/filters/blur.py` (`GaussianBlur2d`) | `.../models/components/filters/blur.py` | 새로 복사, kornia import 유지 |

생성자는 다음과 같다. `PadimLitModel`은 `Padim`과 인자 이름·순서·기본값이 같아야 한다(FR-6, D-5).

| 클래스 | 시그니처 |
|---|---|
| `Padim` | `backbone="resnet18"`, `layers=["layer1", "layer2", "layer3"]`, `pre_trained=True`, `n_features=None`, `pre_processor=True`, `post_processor=True`, `evaluator=True`, `visualizer=True` (8개) |
| `PadimModel` | `backbone="resnet18"`, `layers=["layer1", "layer2", "layer3"]`, `pre_trained=True`, `n_features=None` |

### 1.2 학습과 평가 동작

원본의 동작은 다음과 같다.

- **모델:** `TimmFeatureExtractor`(eval)로 `layers`의 특징을 뽑고 가장 큰 해상도에 맞춰 nearest로 합친 뒤(`generate_embedding`), `idx` buffer가 가리키는 채널 `n_features`개만 고른다. `n_features`는 `resnet18` 100, `wide_resnet50_2` 550이 기본이고 다른 backbone은 지정하지 않으면 `ValueError`다.
- **`idx`:** 생성자에서 `random.sample(range(n_features_original), n_features)`로 고르는 buffer다. 파이썬 `random`을 쓰므로 `seed_everything(42)` 뒤의 호출 순서에 따라 정해지고, `state_dict`에 들어간다.
- **학습:** `training_step`은 임베딩을 `self.embeddings`에 모으고 더미 loss(0)를 반환한다. `configure_optimizers`는 `None`이다. 학습 가능한 파라미터가 없다.
- **적합(`fit`):** `MemoryBankMixin`의 `on_validation_start`와 `on_train_epoch_end`가 `_is_fitted` buffer가 False일 때 한 번 호출한다. `torch.vstack(self.embeddings)`로 `MultiVariateGaussian`을 맞춰 `mean`(C x HW)과 `inv_covariance`(HW x C x C)를 buffer에 둔다. patch 위치마다 공분산에 `0.01 * I`를 더하고 다시 `1e-5 * I`를 더한 뒤 `torch.linalg.inv`를 쓴다.
- **추론:** 임베딩과 `mean`, `inv_covariance`로 Mahalanobis 거리(`clamp(0).sqrt()`)를 구하고 입력 크기로 bilinear 업샘플, `GaussianBlur2d(sigma=4, kernel 33)`로 smoothing한다. `pred_score = amax(anomaly_map)`이다. 학습 모드의 `forward`는 임베딩을 반환한다.
- **`trainer_arguments`:** `{"max_epochs": 1, "val_check_interval": 1.0, "num_sanity_val_steps": 0}`이다. 사용자 결정에 따라 native도 1 epoch이다.
- **전처리:** 기본 PreProcessor는 `Resize((256, 256), antialias=True)` 뒤에 ImageNet `Normalize`다. visionlab은 Dataset이 resize를 하므로 PreProcessor는 정규화를 1회만 한다(FR-8, D-6).
- **후처리와 평가:** Phase 1과 같다. 판정 metric은 Image AUROC와 Pixel AUROC다.

### 1.3 외부 의존과 로컬 자산

- **외부 의존:** timm, torchvision, kornia(`GaussianBlur2d`의 `get_gaussian_kernel2d`, `normalize_kernel2d`, `_compute_padding`)다. FrEIA, open_clip은 쓰지 않는다. anomalib 의존은 `Batch`, `InferenceBatch`, `Tiler`, `AnomalibModule`, `MemoryBankMixin`, `DynamicBufferMixin`, `PreProcessor`, `PostProcessor`, `Evaluator`, `Visualizer`, `LearningType`이다.
- **로컬 가중치:** `paths.toml [backbones]`의 `resnet18`(Phase 1에서 사용). 레퍼런스는 `local_setup.py`가 timm 이름을 로컬 로더로 바꾼다. visionlab은 `PadimLitModel.__init__`이 `feature_extractor`에 주입한다(D-19).
- **데이터:** MVTec `bottle`, VisA `candle`. 분할은 `from_test`, 0.4, seed 42, batch 8/8, `num_workers=0`이다.
- **학습 예산:** `max_epochs=1`(모델 고정, 위 결정)이다.

### 1.4 이식에서 의도적으로 달라지는 곳

- **pretrained 주입(D-19):** 순수 `PadimModel`은 `pre_trained=False`로 구조만 만든다. `PadimLitModel`이 `resnet18` 로컬 파일을 `feature_extractor`에 불러온다. 생성자 인자 수는 유지한다.
- **`MemoryBankMixin` 대체(D-11):** 공유 기반 클래스를 만들지 않고 `_is_fitted` buffer, `on_validation_start`, `on_train_epoch_end`를 `PadimLitModel`에 직접 쓴다.
- **`DynamicBufferMixin`:** `MultiVariateGaussian`의 `mean`, `inv_covariance`는 빈 텐서로 만들어지므로 `load_state_dict`가 크기를 맞추도록 원본 mixin을 복사한다. 순수 모델의 `state_dict` 로딩도 같다.
- **kornia:** 원본처럼 import한다(사용자 결정). `src/visionlab`의 순수 경로에 kornia가 남는다.
- **통계 buffer 저장·복원:** `_is_fitted`가 checkpoint에 들어가 복원·재개 시 다시 적합하지 않아야 한다. PLAN이 Phase 3에서 처음 세우는 경로다.
- **히스토리 비교:** loss가 없고 1 epoch이라 `compare_history.py`는 쓰지 않는다. 대신 레퍼런스 ckpt와 native ckpt의 `idx`, `mean`, `inv_covariance` 차이를 표로 기록한다(판정 아님).
- **FR-5 허용 오차:** `pred_score`와 `anomaly_map`은 `atol=1e-6, rtol=1e-5`, backbone 가중치는 `torch.equal`이다.

## 2. 레퍼런스

학습 예산은 모델이 고정한 1 epoch이다(사용자 결정, 2026-10-25). `models/03_padim.yaml`에 `trainer.max_epochs: 1`이 이미 있고, 모델의 `trainer_arguments`가 `val_check_interval`과 `num_sanity_val_steps`를 덮어쓴 로그(`Overriding ...`)가 남았다.

### 2.1 1 epoch 레퍼런스

2026-10-25~26에 agy가 `anomalib_run.py train --config default.yaml --config data/<data>.yaml --config models/03_padim.yaml`을 데이터셋마다 1회 실행했다(실행으로 검증함). Claude는 로그 파일에서 값을 직접 확인했다. 두 로그 모두 `Seed set to 42`와 `Trainer.fit stopped: max_epochs=1 reached`가 나오고 `network access is blocked`, `URLError`, `ConnectionError`, `Traceback`은 0건이다. 로그는 PowerShell `Tee-Object` 출력이라 UTF-16이다.

agy가 PowerShell에서 실행한 명령은 다음과 같다(Claude가 보낸 지시 원문. agy가 시간 측정 줄을 덧붙여 순서대로 실행했고 종료 코드는 둘 다 0이다).

```powershell
cd D:\projects\visionlab\evaluation\anomalib
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/03_padim.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\03-ref-mvtec-261025-1.log
& $py anomalib_run.py train --config default.yaml --config data/visa.yaml --config models/03_padim.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\03-ref-visa-261025-1.log
```

다음 표는 1 epoch 레퍼런스(1회) 결과다.

| 데이터 / category | Image AUROC | Pixel AUROC | 참고 | run 폴더 | 로그 |
|---|---|---|---|---|---|
| mvtec / bottle | 0.98684 | 0.98436 | image F1 0.9620, pixel F1 0.7320 | `outputs/anomalib/Padim/LocalMVTecAD/bottle/v0` | `outputs/logs/03-ref-mvtec-261025-1.log` |
| visa / candle | 0.89389 | 0.98299 | image F1 0.8182, pixel F1 0.2082 | `outputs/anomalib/Padim/LocalVisa/candle/v0` | `outputs/logs/03-ref-visa-261025-1.log` |

실제 적용된 학습 예산은 `max_epochs=1`, `val_split_mode=from_test`, `val_split_ratio=0.4`, `seed=42`, `train_batch_size=8`, `eval_batch_size=8`, `num_workers=0`, `deterministic=warn`, `backbone=resnet18`, `layers=[layer1, layer2, layer3]`, `pre_trained=true`, `n_features=null`(기본 100)이다(저장된 `config.yaml`). 체크포인트는 각 run 폴더의 `weights/lightning/model.ckpt`다.

### 2.2 에폭별 학습 히스토리

PaDiM은 loss가 없고 학습이 1 epoch이라 비교할 히스토리가 없다. 레퍼런스와 native의 비교는 적합 결과(`idx`, `mean`, `inv_covariance`)로 대신하며 5절에서 다룬다.

### 2.3 레퍼런스 확인 사항

- 두 데이터셋 모두 종료 코드 0이고 오류가 없다.
- 레퍼런스가 1회뿐이고 `idx`가 `random.sample`로 정해지므로, 레퍼런스 자체의 run 간 편차는 측정하지 못했다.

## 3. 이식 메모

### 3.1 원본과 이식 파일의 대응

다음 표는 새로 만든 파일과 원본의 대응이다. 원본과의 차이는 `diff`로 확인했다(구현함, 정적 검증).

| 이식 파일 (`src/visionlab/anomaly_detection/models/`) | 원본 | 원본과 다른 줄 |
|---|---|---|
| `padim/model.py` | `padim/torch_model.py` | import 4줄(`InferenceBatch`, `MultiVariateGaussian`, `TimmFeatureExtractor`, `dryrun_find_featuremap_dims`), `Tiler` import 경로, `TimmFeatureExtractor(... pre_trained=False)`와 주석 한 줄(D-19) |
| `padim/anomaly_map.py` | `padim/anomaly_map.py` | `GaussianBlur2d` import 한 줄 |
| `padim/__init__.py` | `padim/__init__.py` | `PadimModel`만 노출(D-18 예외) |
| `padim/lightning_model.py` (`PadimLitModel`) | `padim/lightning_model.py`와 `components/base/memory_bank_module.py`의 hook | native `LightningModule`로 새로 작성. 8개 인자 동일, `_is_fitted` buffer, `fit`, `on_validation_start`, `on_train_epoch_end`, 더미 loss는 원본 그대로, 로컬 backbone 주입(D-19)과 D-8의 post_processor 갱신 제한 |
| `components/stats/multi_variate_gaussian.py` | `components/stats/multi_variate_gaussian.py` | `DynamicBufferMixin` import 한 줄 |
| `components/base/dynamic_buffer.py` | `components/base/dynamic_buffer.py` | 없음(바이트 단위로 같음) |
| `components/filters/blur.py` | `components/filters/blur.py` | 없음. kornia import 유지(사용자 결정) |

### 3.2 새로 쓴 코드와 Phase 1·2 공유 파일 변경

- **새로 쓴 파일:** `configs/models/padim.yaml`, `evaluation/anomalib/compare_padim_stats.py`(정보 기록용), `tests/03_padim/test_padim.py`(10개), `tests/03_padim/test_padim_lightning.py`(5개).
- **공유 파일 변경(Phase 1·2 파일):** `utils/io.py`(`load_export`가 `padim` 허용, `write_run_info`가 가중치 경로 없음을 허용), `inference/predictor.py`(모델 dict에 `padim`), `evaluation/visionlab/predict.py`(`--model padim`), `evaluation/anomalib/compare_outputs.py`(`--model padim`, `--reference-checkpoint`), `tests/common/test_cli.py`, `tests/common/test_dependency.py`, 그리고 5.2절의 결함 수정으로 `common/metric.py`와 새 파일 `tests/common/test_auroc.py`.
- **새 부품:** 순수 모델이 쓰는 `MultiVariateGaussian`, `DynamicBufferMixin`, `GaussianBlur2d`는 원본 복사다. 새로 구현한 대체 부품은 없다.

### 3.3 이식 중 확인된 사항

- **`n_features=0`:** 원본은 `n_features or <기본값>`이라 0을 기본값 100으로 처리한다. Claude의 지시문이 "0이면 ValueError"라고 잘못 적었고, codex의 질문으로 확인해 원본 동작을 유지했다(테스트도 원본 동작을 확인한다).
- **BatchNorm 상태:** 두 체크포인트 모두 backbone의 `num_batches_tracked`가 0이다. 학습 중 BN 통계는 갱신되지 않았다(레퍼런스와 같다).

## 4. 적대적 검증

검증은 3-A, 3-B 각각 한 회차씩 했다. 수정 요청은 3-A 0회, 3-B 1회(한도 3회 이내)다.

### 4.1 3-A (순수 모델과 대체 부품)

| 지적 | 심각도 | 처리 |
|---|---|---|
| 원본과의 `diff`: 다른 줄은 import와 D-19의 `pre_trained=False` 한 줄뿐이고 `random.sample`, 공분산 for 루프, `0.01·I`와 `1e-5·I`, kornia import가 유지됨 | 확인 | 수정 없음 |
| `base/dynamic_buffer.py`, `filters/blur.py`가 원본과 바이트 단위로 같음 | 확인 | 수정 없음 |
| Claude의 지시문 오류(`n_features=0`), 3.3절 참고 | 낮음 | codex 질문으로 확인하고 원본 동작 유지 |

실행으로 검증함: `tests/03_padim` 10개 통과(`03-pytest-261026-1.log`), `tests/common` 15개 통과(`03-pytest-common-261026-1.log`).

### 4.2 3-B (Lightning, export, 비교 스크립트)

| 지적 | 심각도 | 처리 |
|---|---|---|
| 정적 검토: `PadimLitModel`의 `fit`, 두 hook의 `_is_fitted` 조건, 더미 loss, `configure_optimizers`가 원본과 같음. 로컬 backbone 주입은 `pre_trained`가 True일 때만 | 확인 | 수정 없음 |
| 정적 검토: 테스트 파일명 `test_lightning.py`가 `tests/01_stfpm`과 같아 pytest 수집과 충돌 | 낮음 | `test_padim_lightning.py`로 정함(codex 질문, Claude 승인) |
| 실행으로 발견: native test Image AUROC가 레퍼런스보다 0.035~0.038 낮음(5.2절). 학습 상태와 출력은 레퍼런스와 비트 단위로 같았음 | 높음 | 수정 요청 1회차: `common/metric.py`의 `AUROC` |

수정 요청 1회차의 원인은 다음과 같다. `AUROC`가 raw 점수를 torchmetrics `BinaryROC`에 그대로 넣었고, torchmetrics는 점수가 [0, 1] 밖이면 sigmoid를 자동 적용한다. PaDiM의 raw 점수(Mahalanobis 거리, 수십 규모)는 float32 sigmoid가 1.0으로 포화해 서로 다른 점수가 동점이 된다. anomalib 레퍼런스는 후처리로 [0, 1]에 정규화한 점수로 test AUROC를 계산하므로 영향이 없었다. 수정은 torchmetrics를 유지하고, compute 시점에 점수를 순서와 동점을 보존하는 [0, 1] 순위 점수(float64)로 바꿔 `BinaryROC`에 넣는 것이다(D-8의 raw AUROC 의도, D-20의 torchmetrics 사용 유지). Phase 1·2의 STFPM, EfficientAD 점수는 범위가 작아 이 결함의 영향이 없었다(5.5절).

남은 위험은 6절에 적었다.

## 5. 이식 결과

### 5.1 최종 판정 (1 epoch, 기준 차이 0.01 이내)

native는 `train.py`가 test에서 기록한 값이고 순수 PyTorch는 `export/model.pt`만 불러 `predict.py`로 평가한 값이다. 수정(4.2절) 뒤 새로 학습한 run(`v1`)의 결과이며 모두 실행으로 검증했다.

| 데이터 / category | 지표 | 레퍼런스 | native | 순수 PyTorch | 차이(native) | 차이(순수) | 판정 |
|---|---|---:|---:|---:|---:|---:|---|
| mvtec / bottle | Image AUROC | 0.986842 | 0.986842 | 0.986842 | 0 | 0 | 통과 |
| mvtec / bottle | Pixel AUROC | 0.984364 | 0.984364 | 0.984364 | 6e-8 | 1e-7 | 통과 |
| visa / candle | Image AUROC | 0.893889 | 0.893889 | 0.893889 | 0 | 0 | 통과 |
| visa / candle | Pixel AUROC | 0.982993 | 0.982993 | 0.982993 | 0 | 1e-8 | 통과 |

run 폴더와 로그는 다음과 같다.

| 대상 | 위치 |
|---|---|
| 레퍼런스 | `outputs/anomalib/Padim/LocalMVTecAD/bottle/v0`, `outputs/anomalib/Padim/LocalVisa/candle/v0` |
| native (최종) | `outputs/visionlab/padim/mvtec_bottle/v1`, `outputs/visionlab/padim/visa_candle/v1` (`run.json`: `max_epochs` 1, `num_sanity_val_steps` 0, `val_check_interval` 1.0, 가중치 `D:\backbones\resnet18-f37072fd.pth` sha256 `f37072fd...07ec`) |
| 순수 PyTorch (최종) | 같은 run의 `predict/metrics.json` |
| 로그 | `outputs/logs/03-native-{mvtec,visa}-261026-2.log`, `03-pure-{mvtec,visa}-261026-2.log` |

두 native 로그에 `network access is blocked`, `URLError`, `ConnectionError`, `Traceback`은 0건이다.

### 5.2 폐기한 첫 실행과 결함 수정

수정 전 첫 실행(`v0`)은 기준 미달이었고 지우지 않고 남긴다. 학습 결과(`idx`, `mean`, `inv_covariance`)는 레퍼런스와 비트 단위로 같았으나(5.3절) test Image AUROC만 낮았다.

| 데이터 / category | 지표 | 레퍼런스 | native `v0` | 차이 | 판정 |
|---|---|---:|---:|---:|---|
| mvtec / bottle | Image AUROC | 0.986842 | 0.951754 | 0.0351 | 기준 초과 |
| mvtec / bottle | Pixel AUROC | 0.984364 | 0.983908 | 0.0005 | 통과 |
| visa / candle | Image AUROC | 0.893889 | 0.855694 | 0.0382 | 기준 초과 |
| visa / candle | Pixel AUROC | 0.982993 | 0.982819 | 0.0002 | 통과 |

같은 export로 raw 점수와 후처리 점수의 AUROC를 비교한 읽기 전용 진단(`03-diag-norm-261026-1.log`)에서 후처리 점수 기준 값(mvtec 0.986842, visa 0.893889)이 레퍼런스와 일치했고 raw 기준 값만 낮았다. 원인과 수정은 4.2절에 있다. 수정 뒤 같은 진단(`03-diag-norm-261026-2.log`)에서 `v0`, `v1` 모두 raw와 후처리 점수 AUROC가 같다.

### 5.3 학습 결과(적합 상태) 비교

PaDiM은 loss가 없고 1 epoch이라 에폭별 히스토리 대신 적합 결과를 비교했다(`03-stats-{mvtec,visa}-261026-1.log`, `03-diag-261026-1.log`, 정보 기록이며 판정 아님). 레퍼런스와 native `v0`의 체크포인트 비교 결과는 다음과 같다. `v1`은 같은 코드와 seed로 학습했고 AUROC가 `v0`의 후처리 점수 기준 값과 같다.

| 항목 | mvtec / bottle | visa / candle |
|---|---|---|
| `idx` 일치 | 100개 모두 같음 | 100개 모두 같음 |
| `mean` 최대 절대 차이 | 0.0 | 0.0 |
| `inv_covariance` 최대 절대 차이 | 0.0 | 0.0 |
| backbone 첫 conv 가중치, BN running 통계 | 같음(차이 0.0) | 같음(차이 0.0) |
| BN `num_batches_tracked` | 둘 다 0 | 둘 다 0 |

`seed_everything(42)` 뒤 생성한 `idx`가 레퍼런스와 같았으므로 `idx` 불일치 시 처리 방침(주입하지 않음)은 쓰이지 않았다.

### 5.4 anomalib 대조 (`compare_outputs.py --model padim`)

실행으로 검증함. 레퍼런스 체크포인트(`--reference-checkpoint`)의 상태를 anomalib `PadimModel`과 visionlab `PadimModel`에 같이 불러와 mvtec, visa 실제 샘플 8장(256x256, ImageNet 정규화)으로 비교했다. 허용 오차는 `atol=1e-6, rtol=1e-5`다.

| 항목 | 결과 |
|---|---|
| FR-6 생성자 시그니처 | 일치 |
| FR-23 로컬 backbone 가중치 | `torch.equal`로 같음(sha256 `f37072fd...07ec`) |
| 같은 seed의 `idx` | 같음 |
| FR-5 `pred_score`, `anomaly_map`, `compute_distance`, `GaussianBlur2d` (mvtec, visa) | 최대 절대 차이 0.0 |
| FR-24와 PostProcessor 대조 | 일치 |
| 요약 | PASS 28건, FAIL 0건 (`03-compare-outputs-{mvtec,visa}-261026-2.log`) |

수정 뒤 STFPM, EfficientAD의 같은 대조도 FAIL 0건이다(PASS 38건, 46건; `03-compare-outputs-stfpm-261026-1.log`, `03-compare-outputs-efficient-ad-261026-1.log`).

### 5.5 Phase 1·2 영향 확인

`common/metric.py`는 모든 Phase가 쓰는 공유 파일이다. 새 `AUROC`로 기존 export를 읽기 전용으로 다시 평가해(`03-diag-norm-261026-2.log`) 결과를 확인했다.

| 모델 | 데이터 | 새 AUROC로 다시 평가한 값 | 기존 기록 |
|---|---|---|---|
| STFPM | mvtec / bottle | Image 0.982456, Pixel 0.964436 | 같음 (`01-stfpm.md`) |
| STFPM | visa / candle | Image 0.792500, Pixel 0.880592 | 같음 |
| EfficientAD (10 epoch, `v1`) | mvtec / bottle | Image 1.000000, Pixel 0.974315 | 같음 (`02-efficient-ad.md`) |
| EfficientAD (10 epoch, `v1`) | visa / candle | Image 0.682500, Pixel 0.891759 | 같음 |

### 5.6 테스트

모두 실행으로 검증했다. 수정 뒤 실행한 결과다.

| 대상 | 결과 | 로그 |
|---|---|---|
| `tests/03_padim` | 16개 통과 | `03-pytest-261026-3.log` |
| `tests/common` | 27개 통과(AUROC 신규 테스트 포함) | `03-pytest-common-261026-3.log` |
| `tests/01_stfpm` (Phase 1 회귀) | 32개 통과 | `03-pytest-stfpm-261026-2.log` |
| `tests/02_efficient_ad` (Phase 2 회귀) | 22개 통과 | `03-pytest-efficient-ad-261026-2.log` |

`tests/00_dataset`은 이번 변경과 무관해 다시 실행하지 않았다.

### 5.7 검증 중점 두 가지

| 중점 | 확인 방법 | 결과 |
|---|---|---|
| 1. anomalib 의존성 삭제, native Lightning만 사용 | `tests/common/test_dependency.py`(import 차단), `tests/03_padim`의 격리 프로세스 테스트, 원본과 `diff` | 순수 경로(`model.py`, `anomaly_map.py`, `components/`)에 anomalib, Lightning import 없음. kornia는 허용된 외부 의존으로 남아 있음 |
| 2. 로컬 pretrained, 네트워크 차단, 레퍼런스 일치 | `train.py`의 `block_network()`, 로그 검색, `run.json`, `compare_outputs.py` | 로컬 `resnet18-f37072fd.pth`만 사용, 네트워크 오류 0건, 레퍼런스와 가중치 `torch.equal` |

### 5.8 실행한 명령

agy가 PowerShell에서 실행한 명령은 다음과 같다(2026-10-26, 작업 폴더 `D:\projects\visionlab`, `$py`는 `C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe`, Claude가 보낸 지시 원문). agy는 명령을 하나의 스크립트로 묶어 Write-Output과 Stopwatch를 덧붙여 순서대로 실행했고, 앞 명령이 실패해도 진행하게 했다. 모두 종료 코드를 확인했다. 아래는 수정 뒤(최종) 묶음이다.

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py -m pytest tests\common 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\03-pytest-common-261026-3.log
& $py -m pytest tests\03_padim 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\03-pytest-261026-3.log
& $py -m pytest tests\01_stfpm 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\03-pytest-stfpm-261026-2.log
& $py -m pytest tests\02_efficient_ad 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\03-pytest-efficient-ad-261026-2.log
& $py evaluation/visionlab/train.py train --data mvtec --model padim 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\03-native-mvtec-261026-2.log
& $py evaluation/visionlab/predict.py --model padim --data mvtec 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\03-pure-mvtec-261026-2.log
& $py evaluation/visionlab/train.py train --data visa --model padim 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\03-native-visa-261026-2.log
& $py evaluation/visionlab/predict.py --model padim --data visa 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\03-pure-visa-261026-2.log
& $py evaluation/anomalib/compare_outputs.py --model padim --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\Padim\LocalMVTecAD\bottle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\03-compare-outputs-mvtec-261026-2.log
& $py evaluation/anomalib/compare_outputs.py --model padim --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\Padim\LocalVisa\candle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\03-compare-outputs-visa-261026-2.log
& $py evaluation/anomalib/compare_outputs.py --model stfpm --samples 8 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\03-compare-outputs-stfpm-261026-1.log
& $py evaluation/anomalib/compare_outputs.py --model efficient_ad --samples 8 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\03-compare-outputs-efficient-ad-261026-1.log
& $py <scratchpad>\diag_norm2.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\03-diag-norm-261026-2.log
```

수정 전(첫) 묶음은 같은 pytest 4개(`03-pytest-261026-2.log`, `03-pytest-common-261026-2.log`, `03-pytest-stfpm-261026-1.log`, `03-pytest-efficient-ad-261026-1.log`), 같은 형식의 native 학습·순수 추론·`compare_outputs.py --model padim`(로그 이름 `-1`, 폐기한 `v0` 실행)이었다. 읽기 전용 진단은 `diag_padim.py`(`03-diag-261026-1.log`), `compare_padim_stats.py`(`03-stats-{mvtec,visa}-261026-1.log`), `diag_norm.py`(`03-diag-norm-261026-1.log`)이고, 진단 스크립트(`diag_*.py`)는 프로젝트 밖 scratchpad에 있으며 agy는 파일을 고치지 않았다. 레퍼런스 실행 명령은 2.1절이다.

## 6. 결론

PLAN의 Phase 3 판정 기준(범주별 Image·Pixel AUROC 차이 0.01 이내)은 mvtec/bottle, visa/candle 모두 native Lightning과 순수 PyTorch에서 충족했다. 차이의 최댓값은 1e-7(mvtec Pixel AUROC, 순수 PyTorch)이다. 적합 상태(`idx`, `mean`, `inv_covariance`)와 같은 상태의 출력(FR-5)이 레퍼런스와 비트 단위로 일치한다. FR-6, FR-22, FR-23, FR-24와 통계 buffer 저장·복원(재적합 없음)은 pytest와 `compare_outputs.py`로 확인했다. 두 검증 중점도 로그와 산출물로 확인했다. 수정 요청은 3-B에서 1회(한도 3회 이내)였고, 실행에서 드러난 `AUROC`의 sigmoid 포화 결함이다.

남은 항목과 위험은 다음과 같다.

- **Phase 종료:** 사용자 결정 대기. 종료가 결정되면 `Phase 03: padim ...` 형식으로 커밋·푸시한다.
- **재현 편차 미측정:** 레퍼런스가 데이터셋마다 1회뿐이다. PaDiM은 `idx`가 `random.sample`이라 seed 순서가 바뀌면 달라질 수 있으나, 이번에는 같았다.
- **공유 `AUROC` 변경:** 모든 Phase에 영향을 주는 변경이다. STFPM, EfficientAD의 기존 export는 새 `AUROC`로 다시 평가해 기록값과 같음을 확인했으나(5.5절) 재학습하지는 않았다. `tests/00_dataset`은 다시 실행하지 않았다.
- **사용자 문서 갱신 필요(`specs/`):**
  - `paths.toml [teachers]`(Phase 2에서 이어진 항목)
  - 1 epoch 고정 모델의 예외(10 epoch 고정 규칙의 예외, 이번 결정). D-22 후보
  - 순수 모델의 외부 의존에 kornia 허용(`GaussianBlur2d`). D-4의 "순수 PyTorch만"과의 관계
  - D-8의 "raw AUROC"가 torchmetrics sigmoid 포화를 피하는 순서 보존 계산이라는 점 (D-20 보충)
  - 에폭 히스토리 허용 오차는 PaDiM에서 해당 없음(미결 유지)
- **기타:** agy가 auto 모드에서 승인 창이 계속 뜨는 경우가 있어 사용자 지시(한 건씩 승인)에 따라 처리했다. 첫 실행(`v0`)의 산출물은 삭제하지 않았다.
