# Phase 1. STFPM

모델 폴더 이름은 `stfpm`이고 원본은 anomalib 2.0.0의 `anomalib\models\image\stfpm\`(`C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\Lib\site-packages\anomalib\models\image\stfpm\`)이다. 범위와 완료 조건은 `specs/PLAN.md`의 Phase 1과 "Phase 공통 양식", 절차는 `docs/workflow/model-port.md`를 따른다. 2~5절은 진행 기록과 실행 결과이고 6절은 현재 결론이다. 1절은 색인이고 판단의 근거는 원본 소스다. 사용자가 2026-10-25에 Phase 1 종료를 결정했고 커밋·푸시했다.

## 상태

다음 표는 단계별 진행 상태다. 구현은 4단계로 나누어 지시하고 검증하며, 수정 반복 한도 3회도 단계별로 센다.

| 단계 | 내용 | 상태 |
|---|---|---|
| 레퍼런스 | `anomalib_run.py train` x 2 (mvtec 1회, visa 1회, 사용자 결정으로 반복 실행 안 함) | 실행 완료 (2절) |
| 1-A | 순수 모델 (`models/stfpm/`, `components/feature_extractors`, `components/tiling/tiler.py`, `test_model.py`) | 구현, 검증, 실행 완료 (수정 요청 0회) |
| 1-B | 순수 처리 구성요소 (`common/`의 pre_processor, post_processor, evaluator, metric, visualizer, learning_type, `test_components.py`) | 구현, 검증, 실행 완료 (수정 요청 0회) |
| 1-C | Lightning (`StfpmLitModel`, lightning_visualizer, trainers, configs, `utils/`, `train.py`) | 구현, 검증, 실행 완료 (수정 요청 2회, 재실행 결과 5절) |
| 1-D | 추론 (`inference/predictor.py`, 가중치 저장·로딩, `predict.py`, `compare_outputs.py`, `compare_history.py`) | 구현, 검증, 실행 완료 (수정 요청 0회) |
| 개정 | 사용자 결정(2026-10-25)으로 metric을 torchmetrics로(D-20, FR-24), 테스트를 pytest로(D-21, NFR-6) 재작성 | 재작성, 검증, 실행 완료 (5.6절) |

## 1. 모델 노트

### 1.1 원본과 생성자

다음 표는 원본 파일과 이식 위치의 대응이다(D-15, D-18).

| 원본 (`anomalib\models\image\stfpm\`) | 이식 위치 (`src/visionlab/anomaly_detection/models/stfpm/`) | 비고 |
|---|---|---|
| `torch_model.py` (`STFPMModel`) | `model.py` | D-19에 따라 teacher 생성 줄만 다름 |
| `anomaly_map.py` (`AnomalyMapGenerator`) | `anomaly_map.py` | 복사 |
| `loss.py` (`STFPMLoss`) | `loss.py` | 복사 |
| `lightning_model.py` (`Stfpm`) | `lightning_model.py` (`StfpmLitModel`) | native `LightningModule`, teacher 주입 추가 |
| `__init__.py` | `__init__.py` | `lightning_model`을 import하지 않도록 고침(D-18 예외) |
| `models/components/feature_extractors/` (`TimmFeatureExtractor`, `dryrun_find_featuremap_dims`) | `models/components/feature_extractors/` | 복사 |
| `anomalib.data.utils.tiler.Tiler` | `models/components/tiling/tiler.py` | 새로 구현(FR-16) |

생성자는 다음과 같다. `StfpmLitModel`은 `Stfpm`과 인자 이름·순서·기본값이 같아야 한다(FR-6, D-5).

| 클래스 | 시그니처 |
|---|---|
| `Stfpm` | `backbone="resnet18"`, `layers=("layer1", "layer2", "layer3")`, `pre_processor=True`, `post_processor=True`, `evaluator=True`, `visualizer=True` |
| `STFPMModel` | `layers`, `backbone="resnet18"` |
| `TimmFeatureExtractor` | `backbone`, `layers`, `pre_trained=True`, `requires_grad=False` |

### 1.2 학습과 평가 동작

원본의 동작은 다음과 같다.

- **모델:** teacher와 student 모두 `TimmFeatureExtractor`(timm `features_only`, `out_indices`는 layer 이름에서 찾음)다. teacher는 pretrained로 만들어 `eval()`과 `requires_grad=False`로 고정하고, student는 `pre_trained=False`, `requires_grad=True`다. teacher를 먼저 만들고 student를 만든다(난수 소비 순서).
- **forward:** 학습 모드는 `(teacher_features, student_features)`를 반환하고, 평가 모드는 `InferenceBatch(pred_score, anomaly_map)`을 반환한다. `pred_score = amax(anomaly_map, (-2, -1))`이다. `tiler`가 있으면 tile, untile을 하고(`tiler`의 기본은 `None`) anomaly map은 입력 해상도다.
- **loss:** `STFPMLoss`는 L2 정규화한 teacher, student 특징의 MSE(`reduction="sum"`)를 `0.5 / (H * W)`로 나눠 layer별로 더한다.
- **anomaly map:** layer별 `0.5 * ||norm(t) - norm(s)||^2`를 bilinear로 입력 크기로 올린 뒤 layer 사이를 원소별로 곱한다.
- **optimizer:** `SGD(student_model.parameters(), lr=0.4, momentum=0.9, dampening=0.0, weight_decay=0.001)`다. 학습 단계는 `train_loss`를 `on_epoch=True`로 기록한다.
- **`trainer_arguments`:** `{"gradient_clip_val": 0, "num_sanity_val_steps": 0}`다. `max_epochs`를 고정하지 않으므로 공유 yaml의 10 epoch를 따른다. `learning_type`은 `ONE_CLASS`다.
- **전처리:** 기본 PreProcessor는 `Resize((256, 256), antialias=True)` 뒤에 ImageNet `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`다. visionlab은 Dataset이 resize를 하므로 PreProcessor는 정규화를 1회만 한다(FR-8, D-6).
- **후처리와 평가:** 기본 PostProcessor는 `PostProcessor()`(`ONE_CLASS`)이고, 기본 Evaluator는 image AUROC, image F1, pixel AUROC, pixel F1이다. 판정 metric은 Image AUROC와 Pixel AUROC 둘이다.

### 1.3 외부 의존과 로컬 자산

- **외부 의존:** timm(`create_model`, `features_only`)과 torchvision의 `create_feature_extractor`만 쓴다. FrEIA, open_clip 등은 쓰지 않는다. anomalib 의존은 `Batch`, `InferenceBatch`, `Tiler`, `PreProcessor`, `PostProcessor`, `Evaluator`, `Visualizer`, `AnomalibModule`, `LearningType`이고 모두 visionlab에서 새로 구현한다(D-18).
- **로컬 가중치:** `D:\backbones\resnet18-f37072fd.pth`이고 `paths.toml`의 `[backbones].resnet18`이 가리킨다. 레퍼런스 실행은 `evaluation/anomalib/local_setup.py`가 timm의 `resnet18` 등록을 이 파일로 바꾼다(D-12).
- **데이터:** MVTec `bottle`, VisA `candle`. 분할은 `from_test`, 0.4, seed 42이고 batch는 8/8, `num_workers=0`이다(Phase 0에서 anomalib과 일치 확인).

### 1.4 이식에서 의도적으로 달라지는 곳 (D-19)

원본과 다르게 구현할 곳은 아래 하나다. 이 밖의 변경은 의존성 제거와 `torch_model.py`의 이름 변경뿐이다.

- **teacher 생성:** 원본 `STFPMModel`은 `TimmFeatureExtractor(backbone, pre_trained=True, ...)`로 teacher를 만든다. 이식본의 `STFPMModel`은 `pre_trained=False`로 구조만 만들고, `StfpmLitModel.__init__`이 `paths.toml`의 `resnet18` 파일을 teacher에 불러온다. 생성자 인자는 늘리지 않는다. 파일이 없으면 `FileNotFoundError`이고 무작위 대체는 없다.
- **구현 시 확인할 것:** timm `features_only` 모델의 state dict 키는 torchvision `resnet18` 파일의 키와 같고 `fc.*`만 없다. 불러올 때 missing key가 비어 있어야 하고 unexpected key는 `fc.*`뿐이어야 한다. 결과 teacher 가중치는 anomalib 레퍼런스(timm 로컬 override)의 teacher와 같아야 한다(FR-23, `evaluation/anomalib/compare_outputs.py`).

## 2. 레퍼런스

2026-10-25에 agy가 `anomalib_run.py train --config default.yaml --config data/<data>.yaml --config models/01_stfpm.yaml`을 실행했다(실행으로 검증함). Claude는 로그 파일에서 값을 직접 확인했다. 사용자가 실행 시간 비용 때문에 데이터셋마다 1회만 실행하기로 결정해(2026-10-25), 계획했던 반복 실행은 하지 않는다.

agy가 PowerShell에서 실행한 명령은 다음과 같다(2026-10-25, 승인 창마다 Claude가 한 번씩 승인). 처음 지시에는 데이터셋마다 2회씩 총 4개 명령이 있었고, 사용자 결정으로 4번째를 중단했다.

```powershell
cd D:\projects\visionlab\evaluation\anomalib
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
# 1) mvtec 1차, 종료 코드 0, run 폴더 v2
& $py anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/01_stfpm.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-ref-mvtec-261025-1.log
# 2) mvtec 2차(참고용, 반복 실행 취소 결정 전에 완료), 종료 코드 0, run 폴더 v3
& $py anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/01_stfpm.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-ref-mvtec-261025-2.log
# 3) visa 1차, 종료 코드 0, run 폴더 v0
& $py anomalib_run.py train --config default.yaml --config data/visa.yaml --config models/01_stfpm.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-ref-visa-261025-1.log
# 4) visa 2차: 시작된 뒤 사용자 결정으로 중단(Claude가 python 프로세스 종료), 미완료 run 폴더 v1
& $py anomalib_run.py train --config default.yaml --config data/visa.yaml --config models/01_stfpm.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-ref-visa-261025-2.log
```

다음 표는 레퍼런스(1회) 결과다.

| 데이터 / category | Image AUROC | Pixel AUROC | 참고 | run 폴더 | 로그 |
|---|---|---|---|---|---|
| mvtec / bottle | 0.98246 | 0.96444 | image F1 0.9250, pixel F1 0.6090 | `outputs/anomalib/Stfpm/LocalMVTecAD/bottle/v2` | `outputs/logs/01-ref-mvtec-261025-1.log` |
| visa / candle | 0.79250 | 0.88059 | image F1 0.7143, pixel F1 0.1957 | `outputs/anomalib/Stfpm/LocalVisa/candle/v0` | `outputs/logs/01-ref-visa-261025-1.log` |

실제 적용된 학습 예산과 확인 사항은 다음과 같다.

- `Seed set to 42`, `max_epochs=10`, `val_split_mode=from_test`, `val_split_ratio=0.4`, 모델이 고정한 `gradient_clip_val`, `num_sanity_val_steps` 덮어쓰기 로그(`Overriding ...`) 확인. 네트워크 차단 오류(`network access is blocked`)는 나오지 않았다. 데이터는 `D:\datasets\mvtec`, VisA는 `D:\datasets\visa_anomalib\visa_pytorch`다.
- mvtec은 중단 지시가 도착하기 전에 같은 설정의 두 번째 실행이 이미 끝났다(`v3`, `01-ref-mvtec-261025-2.log`). Image·Pixel AUROC와 에폭별 train loss가 `v2`와 소수 셋째 자리까지 같았다. 반복 실행은 하지 않기로 했으므로 이는 참고로만 적는다.
- visa의 두 번째 실행은 시작된 뒤 사용자 결정으로 중단했다(Claude가 해당 python 프로세스를 종료). 미완료 폴더 `outputs/anomalib/Stfpm/LocalVisa/candle/v1`(체크포인트만 있고 평가 없음)과 `01-ref-visa-261025-2.log`는 레퍼런스가 아니다. 삭제하지 않고 남겨 두었다.
- mvtec `v0`, `v1`은 이전의 1 epoch 동작 확인 run이며 레퍼런스가 아니다.

다음 표는 에폭별 학습 히스토리(`train_loss_epoch`)다. 값은 콘솔 진행 줄에서 읽어 유효숫자 3자리다. 정밀한 값은 run 폴더의 `lightning_logs/version_0/events.out.tfevents.*`에서 읽을 수 있고, 그 파일에 기록된 태그는 `train_loss_epoch`, `train_loss_step`, `epoch`다(validation 지표의 에폭별 기록은 확인하지 못했다).

| 에폭 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| mvtec / bottle | 6.660 | 2.610 | 1.790 | 1.190 | 0.955 | 0.859 | 0.801 | 0.734 | 0.690 | 0.752 |
| visa / candle | 2.560 | 0.864 | 0.691 | 0.646 | 0.580 | 0.558 | 0.846 | 0.611 | 0.560 | 0.520 |

VisA의 Image AUROC가 0.79로 낮고 train loss가 에폭 6에서 올라갔다(0.558에서 0.846). SGD lr 0.4의 불안정일 수 있으나 원인은 진단하지 않았다. 이식 결과와 비교하는 기준 수치이므로 그대로 둔다.

## 3. 이식 메모

### 1-A 순수 모델

2026-10-25에 codex가 구현했다. 원본 대응은 1.1절의 표와 같고, 복사한 7개 파일을 Claude가 원본과 `diff`로 대조했다. 원본과 다른 줄은 다음 셋뿐이다.

| 파일 | 원본과 다른 줄 |
|---|---|
| `models/stfpm/model.py` | import 3줄(`InferenceBatch`, `TimmFeatureExtractor`, TYPE_CHECKING의 `Tiler`)을 visionlab으로 변경 |
| `models/stfpm/model.py` | teacher 생성 줄의 `pre_trained=True`를 `False`로 변경하고 영어 주석 한 줄(D-19) 추가 |
| `models/stfpm/__init__.py` | `from .lightning_model import Stfpm`을 `from .model import STFPMModel`로 변경, `__all__` 변경 |

`anomaly_map.py`, `loss.py`, `components/feature_extractors/{timm,utils,__init__}.py`는 원본과 바이트 단위로 같다(줄 끝 제외). `components/tiling/tiler.py`의 `Tiler`는 `anomalib.data.utils.tiler`를 읽고 새로 구현했다.

`Tiler`가 원본과 다른 곳은 다음과 같다.

- 타일 추출을 `Tensor.unfold`로 구현했다(원본은 루프). 타일 순서(배치 우선, 행 우선), 겹침 평균, border 제거, 특징 크기 환산은 원본과 같게 확인했다.
- `stride=None`이면 `stride=tile_size`로 처리한다(원본은 속성이 없어 `AttributeError`).
- `validate_size_type`의 길이 2 검사(`ValueError`)를 뺐다. 길이가 다른 sequence는 앞 두 값을 쓴다.
- 원본의 모듈 함수 `compute_new_image_size`, `upscale_image`, `downscale_image`는 따로 만들지 않고 `Tiler` 안에 넣었다.

### 1-B 순수 처리 구성요소

2026-10-25에 codex가 구현했다. anomalib 원본(`post_processing`, `pre_processing`, `metrics`, `__init__.py`의 `LearningType`)은 복사하지 않고 동작을 읽어 새로 썼다(D-18).

| visionlab 파일 | 대응하는 anomalib 원본 | 원본과 다른 곳 |
|---|---|---|
| `common/pre_processor.py` | `PreProcessor` | Callback, resize 제거. 입력 검증과 정규화만 한다 |
| `common/post_processor.py` | `PostProcessor`, `MinMax`, `F1AdaptiveThreshold` | Callback 제거, `update`와 `finalize`로 보정 시점 분리, 정규화식과 임계값 선택(F1 최대, 동점이면 가장 낮은 임계값)은 원본과 같음 |
| `common/metric.py` | `AUROC`, `F1Score` | 개정 후(D-20): torchmetrics `BinaryROC`와 `auc`, `BinaryF1Score`를 CPU에서 호출하고, 계산 불가 시 NaN과 `reason`을 남긴다. 개정 전 첫 구현은 순수 torch였다(5.6절) |
| `common/evaluator.py` | `Evaluator` | val와 test 지표를 분리해 복제, dict 입력, 기본 이름은 anomalib 로그 이름과 같음 |
| `common/visualizer.py` | `ImageVisualizer` | 원본/히트맵 겹침/예측 mask/정답 mask 4패널로 단순화, 저장은 하지 않음 |
| `common/learning_type.py` | `anomalib.LearningType` | 값과 이름이 같음 |

anomalib의 metric은 torchmetrics를 쓴다(`BinaryROC`, `auc`, `BinaryF1Score`, `BinaryPrecisionRecallCurve`). 첫 구현은 순수 PyTorch 평가(FR-20)를 위해 순수 torch로 직접 구현했으나, 사용자가 torchmetrics 사용을 결정해(2026-10-25, D-20, FR-24) 같은 클래스로 다시 구현했다. `PostProcessor`의 F1 임계값도 `BinaryPrecisionRecallCurve(normalization=None)`로 계산한다. anomalib의 `BinaryPrecisionRecallCurve`는 torchmetrics의 sigmoid 정규화를 피하려고 클래스를 상속해 덮어쓰는데, 설치된 torchmetrics 1.8.2는 `normalization=None` 인자로 같은 효과를 낸다.

### 1-C Lightning

2026-10-25에 codex가 구현했다. native `LightningModule`이며 anomalib을 import하지 않는다.

| visionlab 파일 | 하는 일 |
|---|---|
| `models/stfpm/lightning_model.py` (`StfpmLitModel`) | 인자 6개는 `Stfpm`과 같다. `STFPMModel`, `STFPMLoss`를 만들고 `load_teacher_weights`로 teacher에 로컬 가중치를 주입한다(D-19). PostProcessor `update`는 fit의 validation에서만 하고 `finalize`는 validation epoch 끝에서 한다. `on_fit_end`에서 `export/model.pt`를 저장한다 |
| `common/lightning_visualizer.py` | `LitVisualizer` callback. `configure_callbacks`가 visualizer가 있을 때 돌려준다 |
| `trainers/lightning_cli.py` (`LitCLI`) | Lightning `LightningCLI` 기반. `trainer.yaml`의 값과 모델이 고정하는 trainer 인자를 덮어쓴다 |
| `utils/io.py` | `read_backbone_path`, `load_teacher_weights`, `file_sha256`, `block_network`, `create_versioned_dir`, `save_export`, `load_export`, `write_run_info` |
| `configs/{default,trainer}.yaml`, `configs/data/`, `configs/models/stfpm.yaml` | 예산: `max_epochs=10`, `seed=42`, batch 8/8, `num_workers=0`, `devices=1`, `deterministic=warn`, TensorBoard logger |
| `evaluation/visionlab/train.py` | `LitCLI`를 호출하는 실행 진입점. 네트워크를 차단하고 `outputs/visionlab/stfpm/<data>_<category>/vN`에 쓴다 |
| `data/mvtec/lightning_data.py` | `predict_dataloader()` 추가(Phase 0 파일 수정, 사용자 승인 아래 `test_dataloader()` 반환) |

`lightning_trainer.py`와 `utils/config.py`는 필요하지 않아 만들지 않았다.

### 1-D 순수 추론과 비교

2026-10-25에 codex가 구현했다.

| 파일 | 하는 일 |
|---|---|
| `inference/predictor.py` | `Predictor.from_export(path)`, `predict`. Lightning, anomalib을 import하지 않는 순수 PyTorch 추론 |
| `evaluation/visionlab/predict.py` | export(`model.pt`)를 불러 test 전체를 평가하고 image·pixel AUROC, F1을 출력 |
| `evaluation/anomalib/compare_outputs.py` | 같은 가중치와 입력으로 anomalib과 visionlab의 생성자 시그니처(FR-6), teacher state_dict(FR-23), raw 출력(FR-5), 지표와 PostProcessor를 대조 |
| `evaluation/anomalib/compare_history.py` | 두 run 폴더의 TensorBoard event 파일에서 epoch별 `train_loss_epoch`와 test 지표를 읽어 표로 비교 |

## 4. 적대적 검증

이 절의 테스트 명령과 결과(`python.exe <file>.py`, `unittest`)와 torchmetrics 정적 분석은 개정(D-20, D-21) 전 기록이다. 근거로 남기며, 개정 후 pytest 결과와 정적 분석은 5.6절이 대체한다.

### 1-B 순수 처리 구성요소

Claude가 원본 소스와 이식 코드를 대조했다. 중점 항목은 anomalib 의존성(import 0건)이다.

| 회차 | 지적 | 심각도 | 처리 |
|---|---|---|---|
| 1 | `PostProcessor`의 정규화식 `clamp((x - threshold) / (max - min) + 0.5, 0, 1)`, F1 임계값 선택(PR 곡선 F1 최대, eps 1e-10, 첫 최대), 폴백(min/max가 NaN이면 raw 복사, 임계값이 NaN이면 반대 수준 임계값 또는 중점)이 원본과 같음 | 이상 없음 | 해당 없음 |
| 1 | min=max이면 원본과 같이 0으로 나누어 NaN이 나올 수 있다 | Minor | 원본과 같은 동작이므로 수정하지 않음 |
| 1 | 정적 분석 범위에서 anomalib, Lightning, torchmetrics, sklearn import 0건 | 이상 없음 | 해당 없음 |

수정 요청은 0회다. 실행 명령은 다음과 같다(agy, 2026-10-25, 종료 코드 0, 승인 창 1회 승인).

```powershell
cd D:\projects\visionlab
& 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe' 'D:\projects\visionlab\tests\01_stfpm\test_components.py' 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-B-test-261025-1.log
$LASTEXITCODE
```

실행 결과(실행으로 검증함)는 `tests/01_stfpm/test_components.py` 13개 테스트 통과(`outputs/logs/01-B-test-261025-1.log`)다. FR-8(정규화 값), FR-9(보정 상태 분리), FR-10(독립 상태와 NaN 사유), FR-11(파일 이름 충돌 방지) 중 순수 부분과 NFR-1·NFR-2를 확인했다. FR-7~FR-12의 Lightning 연결 부분은 1-C에서 확인한다.

### 1-A 순수 모델

Claude가 원본 소스와 이식 코드를 대조했다(codex의 설명은 근거로 쓰지 않았다). 중점 항목은 anomalib 의존성(import 0건)과 pretrained 읽기 제거(D-19)다.

| 회차 | 지적 | 심각도 | 처리 |
|---|---|---|---|
| 1 | 복사 파일 7개가 원본과 다른 줄이 위 세 곳뿐이고, 모두 의도한 변경(의존성 제거, D-19, `__init__.py` 예외) | 이상 없음 | 해당 없음 |
| 1 | `Tiler`의 타일 순서와 untile 좌표, 겹침 평균, 모드별 복원이 원본 알고리즘과 같음 | 이상 없음 | 해당 없음 |
| 1 | `Tiler.validate_size_type`의 길이 검사 누락 | Minor | 수정하지 않음. STFPM은 정수만 쓴다 |
| 1 | `stride=None` 기본 동작이 원본(`AttributeError`)과 다름 | Minor | 수정하지 않음. 문서의 기본 동작(`stride=tile_size`)과 같게 구현한 의도된 차이 |

수정 요청은 0회다. 남은 위험은 teacher 주입과 학습·평가 동작이 아직 구현 전이라는 점이다(1-C).

실행 명령은 다음과 같다(agy, 2026-10-25, 작업 폴더 `D:\projects\visionlab`).

```powershell
cd D:\projects\visionlab
& 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe' 'D:\projects\visionlab\tests\01_stfpm\test_model.py' 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-A-test-261025-1.log
$LASTEXITCODE
```

실행 결과(실행으로 검증함)는 `tests/01_stfpm/test_model.py` 10개 테스트 통과(`outputs/logs/01-A-test-261025-1.log`, 종료 코드 0)다. FR-4(shape), FR-16(1024 입력 16타일, 1024 해상도 anomaly map), FR-22(생성자 I/O 차단, 네트워크 차단 상태에서 생성과 `state_dict` 복원), NFR-5(헤더), NFR-1·NFR-2(import 차단)를 확인했다.

### 1-C Lightning

Claude가 anomalib 원본 `Stfpm`과 이식 코드를 대조했다(codex의 설명은 근거로 쓰지 않았다). 중점 항목은 anomalib 의존성(import 0건, native Lightning)과 로컬 pretrained 사용(D-19)이다.

| 회차 | 지적 | 심각도 | 처리 |
|---|---|---|---|
| 1 | `load_teacher_weights`가 실제 파일 `D:\backbones\resnet18-f37072fd.pth`에서 실패한다. 이 파일에는 BatchNorm의 `num_batches_tracked` 키가 없는데, 테스트는 timm이 만든 임시 파일(키 있음)만 써서 잡지 못했다 | Major | 수정 요청 1/3. `.num_batches_tracked` 키는 누락을 허용하게 고치고 실제 파일 형식의 회귀 테스트를 추가했다. `test_lightning.py` 7개 통과 |
| 2 | 에폭별 `train_loss_epoch`가 레퍼런스와 일정 비율 어긋난다. `train_loss_step`은 소수점 끝까지 같아 학습 궤적은 같다. mvtec 약 3%, visa 약 0.3% 크다. 마지막 작은 배치(mvtec 1장, visa 4장)의 가중이 원인이다. Lightning이 배치 크기로 가중 평균하고, 레퍼런스는 step마다 같은 비중이다 | Major | 수정 요청 2/3. `self.log("train_loss", ..., batch_size=1)`을 지정해 step마다 같은 비중으로 평균한다. 재실행 결과는 5절 |
| 1 | 정적 분석: `src/` 아래 anomalib, torchmetrics, sklearn import 0건. Lightning import는 `lightning_data.py`, `lightning_model.py`, `lightning_visualizer.py`, `lightning_cli.py` 4개 파일에만 있다 | 이상 없음 | 해당 없음 |
| 1 | teacher 주입이 `StfpmLitModel.__init__`에서만 일어나고 `STFPMModel` 생성자는 파일을 읽지 않는다(D-19). 생성자 인자 6개 불변 | 이상 없음 | 해당 없음 |

수정 요청은 2회다(한도 3회 이내). 첫 실행(수정 2 이전)의 native 결과는 `outputs/visionlab/stfpm/mvtec_bottle/v0`, `visa_candle/v0`이고 5절에서 재실행(`v1`)과 구분해 적는다.

agy가 PowerShell에서 실행한 명령은 다음과 같다(2026-10-25, 승인 창마다 Claude가 한 번씩 승인). 작업 폴더는 `D:\projects\visionlab`이고 `$py`는 `C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe`다.

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
# 단위 테스트 (1차: 수정 요청 전, 2차: 수정 1 뒤, 3차: 수정 2 뒤는 5절)
& $py tests/01_stfpm/test_lightning.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-C-test-lightning-261025-1.log
& $py tests/common/test_cli.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-C-test-cli-261025-1.log
& $py tests/01_stfpm/test_lightning.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-C-test-lightning-261025-2.log
& $py tests/common/test_cli.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-C-test-cli-261025-2.log
# native 학습·평가 첫 실행(수정 2 이전, run 폴더 v0)
& $py evaluation/visionlab/train.py train --data mvtec --model stfpm 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-native-mvtec-261025-1.log
& $py evaluation/visionlab/train.py train --data visa --model stfpm 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-native-visa-261025-1.log
```

### 1-D 순수 추론과 비교

Claude가 `predictor.py`, `predict.py`, 비교 스크립트를 읽고 대조했다. 지적은 없다. 수정 요청은 0회다. 중점 항목은 순수 PyTorch 경로의 import(Lightning, anomalib 0건)와 export만으로 복원되는지다.

agy가 PowerShell에서 실행한 명령은 다음과 같다(2026-10-25, 작업 폴더와 `$py`는 위와 같음).

```powershell
& $py tests/01_stfpm/test_predictor.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-D-test-predictor-261025-1.log
& $py tests/common/test_dependency.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-D-test-dependency-261025-1.log
& $py evaluation/visionlab/predict.py --model stfpm --data mvtec 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-pure-mvtec-261025-1.log
& $py evaluation/visionlab/predict.py --model stfpm --data visa 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-pure-visa-261025-1.log
& $py evaluation/anomalib/compare_outputs.py --model stfpm 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-compare-outputs-261025-1.log
& $py evaluation/anomalib/compare_history.py --reference D:\projects\visionlab\outputs\anomalib\Stfpm\LocalMVTecAD\bottle\v2 --native D:\projects\visionlab\outputs\visionlab\stfpm\mvtec_bottle\v0 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-history-mvtec-261025-1.log
& $py evaluation/anomalib/compare_history.py --reference D:\projects\visionlab\outputs\anomalib\Stfpm\LocalVisa\candle\v0 --native D:\projects\visionlab\outputs\visionlab\stfpm\visa_candle\v0 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-history-visa-261025-1.log
```

`compare_outputs.py`의 결과(`01-compare-outputs-261025-1.log`, 종료 코드 0, 실행으로 검증함)는 다음과 같다.

| 항목 | 판정 | 내용 |
|---|---|---|
| FR-6 생성자 시그니처 | 일치 | `Stfpm`과 `StfpmLitModel`의 인자 이름, 순서, 기본값 |
| FR-23 teacher | 일치 | state_dict 키·값 불일치 0건. 가중치 `D:\backbones\resnet18-f37072fd.pth`, sha256 `f37072fd...07ec` |
| FR-5 raw 출력 | 일치 | pred_score, anomaly_map, 학습 모드 특징의 최대 절대 차이 0.0 (mvtec, visa) |
| 지표와 후처리 | 일치 | AUROC, F1은 torchmetrics와 1e-6 이내, PostProcessor 임계값·min·max 일치, 정규화 차이 0 |

## 5. 이식 결과

수정 요청 2(`batch_size=1`) 뒤에 agy가 native 학습·평가, 순수 PyTorch 추론, 에폭별 비교, 단위 테스트를 다시 실행했다(2026-10-25, 종료 코드는 로그로 확인, 실행으로 검증함). 이 재실행(`v1`)은 개정(D-20, D-21) 전의 이식 결과이고, 개정 후 최종 결과는 5.6절(`v2`)이다. 수정 전 첫 실행(`v0`)은 4절 1-C의 명령 기록과 5.3절에 근거로만 남긴다. 5.1~5.5절의 값과 테스트 개수는 모두 개정 전 기록이다.

agy가 PowerShell에서 실행한 명령은 다음과 같다(승인 창마다 Claude가 명령 텍스트를 확인하고 한 번씩 승인, 영구 허용 없음). 작업 폴더는 `D:\projects\visionlab`이다.

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py evaluation/visionlab/train.py train --data mvtec --model stfpm 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-native-mvtec-261025-2.log
& $py evaluation/visionlab/predict.py --model stfpm --data mvtec 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-pure-mvtec-261025-2.log
& $py evaluation/visionlab/train.py train --data visa --model stfpm 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-native-visa-261025-2.log
& $py evaluation/visionlab/predict.py --model stfpm --data visa 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-pure-visa-261025-2.log
& $py evaluation/anomalib/compare_history.py --reference D:\projects\visionlab\outputs\anomalib\Stfpm\LocalMVTecAD\bottle\v2 --native D:\projects\visionlab\outputs\visionlab\stfpm\mvtec_bottle\v1 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-history-mvtec-261025-2.log
& $py evaluation/anomalib/compare_history.py --reference D:\projects\visionlab\outputs\anomalib\Stfpm\LocalVisa\candle\v0 --native D:\projects\visionlab\outputs\visionlab\stfpm\visa_candle\v1 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-history-visa-261025-2.log
& $py tests/01_stfpm/test_lightning.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-C-test-lightning-261025-3.log
& $py tests/common/test_cli.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-C-test-cli-261025-3.log
```

### 5.1 최종 성능 (레퍼런스, native, 순수 PyTorch)

다음 표는 category별 Image AUROC와 Pixel AUROC다. 판정 기준은 레퍼런스와의 차이 0.01 이내다(PLAN 검증 방식, FR-18, FR-20). native는 `train.py`가 test에서 기록한 값이고 순수 PyTorch는 `model.pt`만 불러 `predict.py`로 평가한 값이다.

| 데이터 / category | metric | 레퍼런스 | native Lightning | 순수 PyTorch | 차이(native) | 차이(순수) | 판정 |
|---|---|---:|---:|---:|---:|---:|---|
| mvtec / bottle | Image AUROC | 0.982456 | 0.982456 | 0.982456 | 0.000000 | 0.000000 | 통과 |
| mvtec / bottle | Pixel AUROC | 0.964436 | 0.964436 | 0.964436 | 0.000000 | 0.000000 | 통과 |
| visa / candle | Image AUROC | 0.792500 | 0.792500 | 0.792500 | 0.000000 | 0.000000 | 통과 |
| visa / candle | Pixel AUROC | 0.880591 | 0.880591 | 0.880591 | 0.000000 | 0.000000 | 통과 |

참고로 F1은 다음과 같다. 세 단계 모두 소수 여섯째 자리까지 같다.

| 데이터 / category | image F1 (레퍼런스 = native = 순수) | pixel F1 (레퍼런스 = native = 순수) |
|---|---:|---:|
| mvtec / bottle | 0.925000 | 0.609043 |
| visa / candle | 0.714286 | 0.195682 |

차이의 최댓값은 약 4e-7(visa Pixel AUROC, native 0.8805914521 대 레퍼런스 0.8805910348)이고 부동소수점 오차 범위다. run 폴더와 로그는 다음과 같다.

| 데이터 | native run 폴더 | 순수 PyTorch 결과 | 로그 |
|---|---|---|---|
| mvtec / bottle | `outputs/visionlab/stfpm/mvtec_bottle/v1` | `v1/predict/metrics.json` | `01-native-mvtec-261025-2.log`, `01-pure-mvtec-261025-2.log` |
| visa / candle | `outputs/visionlab/stfpm/visa_candle/v1` | `v1/predict/metrics.json` | `01-native-visa-261025-2.log`, `01-pure-visa-261025-2.log` |

각 run 폴더에는 `run.json`(설정, teacher 가중치 경로와 sha256), `export/model.pt`(순수 추론용), `weights/lightning`(체크포인트), `lightning_logs`(TensorBoard)가 있다.

### 5.2 에폭별 학습 히스토리 (Claude 검증 중점 2)

다음 표는 에폭별 `train_loss_epoch`다. 사용자가 허용 오차를 정하지 않아 판정 기준에서 제외하고 비교 표로만 기록한다. 값은 `compare_history.py`가 event 파일에서 읽었다(`01-history-*-261025-2.log`). 수정 요청 2 이후 두 데이터셋 모두 모든 에폭에서 상대차가 약 4e-7 이내다.

| 에폭 | mvtec 레퍼런스 | mvtec native | 상대차 | visa 레퍼런스 | visa native | 상대차 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.655728 | 6.655728 | 7e-8 | 2.564517 | 2.564517 | 3e-7 |
| 1 | 2.609133 | 2.609133 | 0 | 0.864440 | 0.864440 | 2e-7 |
| 2 | 1.794155 | 1.794155 | 0 | 0.691374 | 0.691374 | 3e-7 |
| 3 | 1.192001 | 1.192001 | 2e-7 | 0.646018 | 0.646018 | 0 |
| 4 | 0.954929 | 0.954929 | 6e-8 | 0.579970 | 0.579970 | 0 |
| 5 | 0.858509 | 0.858509 | 7e-8 | 0.557586 | 0.557586 | 2e-7 |
| 6 | 0.801398 | 0.801399 | 7e-8 | 0.846130 | 0.846130 | 4e-7 |
| 7 | 0.733786 | 0.733786 | 2e-7 | 0.610965 | 0.610965 | 0 |
| 8 | 0.689619 | 0.689619 | 0 | 0.559539 | 0.559539 | 2e-7 |
| 9 | 0.751645 | 0.751645 | 3e-7 | 0.520411 | 0.520411 | 0 |

visa에서 에폭 6의 loss가 올라가는 현상(0.558 $\to$ 0.846)도 native가 같은 값으로 재현했다. 따라서 이는 이식 오류가 아니라 레퍼런스 자체의 특성이다.

native의 validation 지표(`val_image_AUROC`, `val_pixel_AUROC`)는 에폭마다 기록되며, 레퍼런스 event 파일에는 이 태그가 없어(2절 참고) 비교하지 않았다.

### 5.3 수정 요청 2 전후

다음 표는 `batch_size=1` 수정 전 첫 실행(`v0`, `01-history-*-261025-1.log`)과 수정 후(`v1`)의 `train_loss_epoch` 상대차다. 최종 AUROC는 수정 전에도 이미 같았다. 수정이 바꾼 것은 loss의 에폭 평균 기록뿐이고 학습 궤적은 같다(`train_loss_step`이 소수점 끝까지 같았다).

| 데이터 | 수정 전 상대차 (에폭 0~9) | 수정 후 상대차 |
|---|---|---|
| mvtec / bottle | 약 0.028 ~ 0.031 | 0 ~ 3e-7 |
| visa / candle | 약 0.0003 ~ 0.0036 | 0 ~ 4e-7 |

### 5.4 두 검증 중점의 확인 결과

| 검증 중점 | 확인 방법 | 결과 |
|---|---|---|
| 1. anomalib 의존성 완전 삭제, native Lightning만 사용 | `src/` 전체 정적 검색, `tests/common/test_dependency.py`(import 차단) | `src/visionlab`에 anomalib, torchmetrics, sklearn import 0건. Lightning은 4개 파일에만 있고 순수 PyTorch 경로(`predictor.py`, `model.py` 등)에는 없다. `test_dependency.py` 2개 통과 |
| 2. 로컬 pretrained 사용, 네트워크 차단 | `train.py`의 `block_network()`, 로그 검색, `run.json`의 teacher 가중치 기록, `compare_outputs.py` | 두 native 로그에 `network access is blocked`, `URLError`, `ConnectionError` 0건. `run.json`의 teacher sha256은 `f37072fd...07ec`로 `D:\backbones\resnet18-f37072fd.pth`와 같다. teacher state_dict는 레퍼런스와 키·값이 같다(FR-23) |

### 5.5 테스트와 비교 스크립트 요약

다음 표는 재실행 이후를 포함한 전체 실행 결과다. 모두 실행으로 검증했다.

| 대상 | 결과 | 로그 |
|---|---|---|
| `tests/01_stfpm/test_model.py` | 10개 통과 | `01-A-test-261025-1.log` |
| `tests/01_stfpm/test_components.py` | 13개 통과 | `01-B-test-261025-1.log` |
| `tests/01_stfpm/test_lightning.py` | 7개 통과 (수정 2 뒤 재실행) | `01-C-test-lightning-261025-3.log` |
| `tests/common/test_cli.py` | 5개 통과 (수정 2 뒤 재실행) | `01-C-test-cli-261025-3.log` |
| `tests/01_stfpm/test_predictor.py` | 1개 통과 | `01-D-test-predictor-261025-1.log` |
| `tests/common/test_dependency.py` | 2개 통과 | `01-D-test-dependency-261025-1.log` |
| `evaluation/anomalib/compare_outputs.py` | FR-5, FR-6, FR-23, 지표·후처리 모두 일치, 종료 코드 0 | `01-compare-outputs-261025-1.log` |

`compare_outputs.py`는 수정 2 이전에 실행했고 이후 코드 변경은 `self.log`의 `batch_size` 한 줄뿐이라(모델 출력, 가중치, 지표에 영향 없음) 다시 실행하지 않았다. 개정 후에는 5.6절에서 다시 실행했다.

### 5.6 개정(D-20 torchmetrics, D-21 pytest) 후 결과

사용자가 두 가지를 결정했다(2026-10-25). 첫째 metric은 torchmetrics로 계산한다(D-20, FR-24). 둘째 테스트는 pytest로 작성하고 실행한다(D-21, NFR-6). 모든 모델 Phase에 공통 적용하기로 해서 `specs/DECISIONS.md`(D-20, D-21, D-14, D-15), `specs/SPEC.md`(FR-24, NFR-6), `specs/PLAN.md`(공통 양식), `AGENTS.md`, `docs/workflow/model-port.md`에 반영했다. 같은 날 codex가 코드를 다시 썼다.

다음 표는 codex가 바꾼 것이다.

| 파일 | 변경 |
|---|---|
| `common/metric.py` | AUROC는 `BinaryROC`와 `auc(reorder=True)`, F1은 `BinaryF1Score`로 계산. 입력 검증, NaN과 `reason`(FR-10)은 유지 |
| `common/post_processor.py` | F1 임계값을 `BinaryPrecisionRecallCurve(normalization=None)`와 anomalib과 같은 식(`2PR/(P+R+1e-10)`, `argmax`)으로 계산. 보정 구조와 폴백은 유지 |
| `tests/conftest.py` (신규) | `<root>/src`를 `sys.path`에 추가 |
| `tests/00_dataset/test_data.py`, `tests/01_stfpm/test_{model,components,lightning,predictor}.py`, `tests/common/test_{cli,dependency}.py` | `unittest`를 pytest 함수, `assert`, fixture, `monkeypatch`, `tmp_path`로 변환. import 차단 확인은 subprocess. 변환 전 48개 검사가 53개가 되었다(처리 구성요소 14, 의존성 6) |
| `tests/common/test_dependency.py` | NFR-6(`unittest` import 0건), FR-24(sklearn import 0건, Lightning·anomalib을 막은 자식 프로세스에서 metric 계산 성공), 테스트 basename 유일성 추가 |
| `evaluation/anomalib/compare_outputs.py` | 지표 대조 대상을 torchmetrics 직접 호출에서 anomalib 자체 클래스(`AUROC`, `F1Score`, `F1AdaptiveThreshold`)로 바꿈 |

Claude의 검증은 다음과 같다. `metric.py`, `post_processor.py`를 anomalib `auroc.py`, `f1_score.py`, `precision_recall_curve.py`, `threshold/f1_adaptive_threshold.py`와 대조했다. 호출은 같고, `normalization=None`은 torchmetrics 1.8.2 소스에서 sigmoid 적용을 건너뛰는 인자임을 확인했다. `tests/`에 `unittest` import는 0건(`test_dependency.py`의 검사 문자열 1곳 제외), `anomalib`, `sklearn` import는 0건이다. 지적은 없고 수정 요청은 0회다. 한 가지 Minor는 `tests/common/test_cli.py`가 pytest 안에서 `LightningCLI(args=...)`를 부를 때 `sys.argv`(pytest 인자)가 함께 보인다는 Lightning 경고가 나오는 것이다(동작에는 영향 없음, 수정하지 않음).

agy가 PowerShell에서 실행한 명령은 다음과 같다(2026-10-25, 승인 창마다 Claude가 명령 텍스트를 확인하고 한 번씩 승인, 영구 허용 없음). 작업 폴더는 `D:\projects\visionlab`이다.

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py -m pytest tests\00_dataset 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\00-pytest-261025-1.log
& $py -m pytest tests\01_stfpm 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-pytest-261025-1.log
& $py -m pytest tests\common 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-pytest-common-261025-1.log
& $py evaluation/anomalib/compare_outputs.py --model stfpm --samples 8 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-compare-outputs-261025-2.log
& $py evaluation/visionlab/train.py train --data mvtec --model stfpm 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-native-mvtec-261025-3.log
& $py evaluation/visionlab/predict.py --model stfpm --data mvtec 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-pure-mvtec-261025-3.log
& $py evaluation/visionlab/train.py train --data visa --model stfpm 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-native-visa-261025-3.log
& $py evaluation/visionlab/predict.py --model stfpm --data visa 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\01-pure-visa-261025-3.log
```

다음 표는 pytest 결과다(실행으로 검증함). 모두 실패 0건이다.

| 대상 | 결과 | 로그 |
|---|---|---|
| `tests/00_dataset` | 10개 통과 | `00-pytest-261025-1.log` |
| `tests/01_stfpm` (model 10, components 14, lightning 7, predictor 1) | 32개 통과, 경고 36개(Lightning 일반 경고) | `01-pytest-261025-1.log` |
| `tests/common` (cli 5, dependency 6) | 11개 통과, 경고 48개 | `01-pytest-common-261025-1.log` |
| `compare_outputs.py` | FR-5, FR-6, FR-23, FR-24 모두 PASS, FAIL 0건, 종료 코드 0 | `01-compare-outputs-261025-2.log` |

`compare_outputs.py`의 FR-24 항목은 anomalib `AUROC`, `F1Score`, `F1AdaptiveThreshold`와 값이 같다(AUROC 0.5005772709846497 대 0.5005772709846497, 범위 밖 점수의 AUROC raw 0.5008024573326111 대 같음, F1 0.4991747736930847 대 같음, 임계값과 PostProcessor min·max·정규화 모두 같음). 이 대조에 쓴 임계값 입력은 무작위 점수(AUROC 약 0.5)여서 임계값이 최솟값 -8.0으로 정해지는 약한 사례다. 임계값 선택이 실제 데이터에서 맞는지는 아래 최종 성능의 pixel F1(0.6090)이 레퍼런스와 같다는 점으로 확인한다.

다음 표는 개정 후 최종 성능이다. 판정 기준은 레퍼런스와의 차이 0.01 이내다(FR-18, FR-20).

| 데이터 / category | metric | 레퍼런스 | native Lightning | 순수 PyTorch | 차이(native) | 차이(순수) | 판정 |
|---|---|---:|---:|---:|---:|---:|---|
| mvtec / bottle | Image AUROC | 0.982456 | 0.982456 | 0.982456 | 0.000000 | 0.000000 | 통과 |
| mvtec / bottle | Pixel AUROC | 0.964436 | 0.964436 | 0.964436 | 0.000000 | 0.000000 | 통과 |
| visa / candle | Image AUROC | 0.792500 | 0.792500 | 0.792500 | 0.000000 | 0.000000 | 통과 |
| visa / candle | Pixel AUROC | 0.880591 | 0.880591 | 0.880591 | 0.000000 | 0.000001 | 통과 |

F1은 다음과 같다. 세 단계 모두 소수 여섯째 자리까지 같다.

| 데이터 / category | image F1 | pixel F1 |
|---|---:|---:|
| mvtec / bottle | 0.925000 | 0.609043 |
| visa / candle | 0.714286 | 0.195682 |

mvtec native의 Pixel AUROC는 0.9644361138로 레퍼런스 0.9644361138과 소수 열 자리까지 같다(개정 전 순수 torch 구현은 0.9644360542로 6e-8 달랐다). 차이의 최댓값은 약 5e-7(visa Pixel AUROC, 순수 PyTorch 0.88059151 대 레퍼런스 0.88059103)이다. 두 native 로그에 `network access is blocked`, `URLError`, `ConnectionError`, `Traceback`은 0건이고, `run.json`의 teacher sha256은 `f37072fd...07ec`로 변하지 않았다.

| 데이터 | native run 폴더 | 순수 PyTorch 결과 | 로그 |
|---|---|---|---|
| mvtec / bottle | `outputs/visionlab/stfpm/mvtec_bottle/v2` | `v2/predict/metrics.json` | `01-native-mvtec-261025-3.log`, `01-pure-mvtec-261025-3.log` |
| visa / candle | `outputs/visionlab/stfpm/visa_candle/v2` | `v2/predict/metrics.json` | `01-native-visa-261025-3.log`, `01-pure-visa-261025-3.log` |

개정은 학습 코드(모델, loss, optimizer)를 건드리지 않았으므로 에폭별 `train_loss_epoch` 비교(`compare_history.py`)는 다시 실행하지 않았다. 대신 `v2` native 로그의 에폭 종료 진행 줄에서 읽은 `train_loss_epoch`(유효숫자 3자리)가 2절의 레퍼런스 표와 두 데이터셋 모두 10개 에폭에서 전부 같음을 확인했다(mvtec 6.660, 2.610, 1.790, 1.190, 0.955, 0.859, 0.801, 0.734, 0.690, 0.752, visa 2.560, 0.864, 0.691, 0.646, 0.580, 0.558, 0.846, 0.611, 0.560, 0.520). 소수 7자리 비교는 `v1`(5.2절)까지만 했다.

## 6. 결론

PLAN의 Phase 1 판정 기준(범주별 Image·Pixel AUROC 차이 0.01 이내)은 mvtec/bottle, visa/candle 모두 native Lightning과 순수 PyTorch에서 충족했다. 개정(torchmetrics, pytest) 후 최종 결과(5.6절)에서 차이의 최댓값은 약 5e-7이다. 두 검증 중점(anomalib 의존성 제거, 로컬 pretrained 사용)도 로그와 산출물로 확인했다. 수정 요청은 1-C에서 2회(한도 3회 이내)였고, 사용자 결정에 따른 개정은 수정 요청 횟수에 넣지 않았다.

남은 항목과 위험은 다음과 같다.

- **Phase 종료:** 사용자가 2026-10-25에 종료를 결정했고 `Phase 01: stfpm ...` 커밋을 푸시했다.
- **에폭 히스토리 허용 오차:** 사용자가 정하지 않아 판정에서 제외했다. 개정 전 `v1`은 약 4e-7로, 개정 후 `v2`는 유효숫자 3자리로 일치한다.
- **개정 결정의 공통 적용:** torchmetrics(D-20, FR-24)와 pytest(D-21, NFR-6)는 이후 모든 Phase에 적용한다. Phase 0 테스트도 pytest로 바뀌었다(10개 통과).
- **확인하지 못한 것:** 레퍼런스는 데이터셋마다 1회만 실행해 레퍼런스 자체의 재현 편차는 mvtec의 `v3`(참고 실행, 같은 값) 외에는 측정하지 못했다. 범주는 mvtec `bottle`, visa `candle` 하나씩이다. FR-24의 임계값 대조는 무작위 입력의 약한 사례이고, 실제 데이터의 임계값 검증은 pixel F1 일치로 대신했다. 개정 후 에폭 loss는 유효숫자 3자리까지만 비교했다.- **중단·보존 산출물:** visa 레퍼런스 `v1`(중단), mvtec 레퍼런스 `v0`/`v1`(1 epoch 확인), native `v0`(수정 전 첫 실행), `v1`(개정 전 최종)은 삭제하지 않고 남겼다.
