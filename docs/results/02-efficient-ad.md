# Phase 2. EfficientAD

모델 폴더 이름은 `efficient_ad`이고 원본은 anomalib 2.0.0의 `anomalib\models\image\efficient_ad\`(`C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\Lib\site-packages\anomalib\models\image\efficient_ad\`)이다. 범위와 완료 조건은 `specs/PLAN.md`의 Phase 2와 "Phase 공통 양식", 절차는 `docs/workflow/model-port.md`를 따른다. 1절은 색인이고 판단의 근거는 원본 소스다.

## 상태

다음 표는 단계별 진행 상태다. 수정 반복 한도 3회는 단계별로 센다.

| 단계 | 내용 | 상태 |
|---|---|---|
| 레퍼런스 | `anomalib_run.py train` x 2 (mvtec 1회, visa 1회, 10 epoch) | 실행 완료 (2절) |
| 2-A | 순수 모델 (`models/efficient_ad/model.py`, `tests/02_efficient_ad/test_efficient_ad.py`의 순수 부분) | 구현, 검증, 실행 완료 (수정 요청 0회) |
| 2-B | Lightning (`EfficientAdLitModel`, configs, teacher 주입, export 일반화, Predictor 일반화, 비교 스크립트) | 구현, 검증, 실행 완료 (수정 요청 1회, 결과 5절) |

처리 구성요소(PreProcessor, PostProcessor, Evaluator)는 Phase 1의 것을 재사용하므로 별도 단계를 두지 않는다.

## 1. 모델 노트

### 1.1 원본과 생성자

다음 표는 원본 파일과 이식 위치의 대응이다(D-15, D-18).

| 원본 (`anomalib\models\image\efficient_ad\`) | 이식 위치 (`src/visionlab/anomaly_detection/models/efficient_ad/`) | 비고 |
|---|---|---|
| `torch_model.py` (`EfficientAdModel`, `EfficientAdModelSize`, PDN, AutoEncoder) | `model.py` | import 한 줄(`InferenceBatch`)만 변경 |
| `lightning_model.py` (`EfficientAd`) | `lightning_model.py` (`EfficientAdLitModel`) | native `LightningModule`, teacher 주입 추가 |
| `__init__.py` | `__init__.py` | `lightning_model`을 import하지 않도록 고침(D-18 예외) |

생성자는 다음과 같다. `EfficientAdLitModel`은 `EfficientAd`와 인자 이름·순서·기본값이 같아야 한다(FR-6, D-5).

| 클래스 | 시그니처 |
|---|---|
| `EfficientAd` | `imagenet_dir="./datasets/imagenette"`, `teacher_out_channels=384`, `model_size=EfficientAdModelSize.S`, `lr=0.0001`, `weight_decay=0.00001`, `padding=False`, `pad_maps=True`, `pre_processor=True`, `post_processor=True`, `evaluator=True`, `visualizer=True` (11개) |
| `EfficientAdModel` | `teacher_out_channels`, `model_size=S`, `padding=False`, `pad_maps=True` |

### 1.2 학습과 평가 동작

원본의 동작은 다음과 같다.

- **모델:** teacher와 student는 PDN(small은 conv 4개, medium은 conv 6개)이고 입력에 내부에서 ImageNet 정규화를 적용한다. AutoEncoder(Encoder, Decoder)도 내부에서 정규화한다. `mean_std`와 `quantiles`는 `nn.ParameterDict`라 `state_dict`에 들어간다. teacher의 pretrained 가중치는 생성자가 아니라 `prepare_pretrained_model`이 학습 시작 때 읽는다.
- **학습 시작(`on_train_start`):** `train_batch_size != 1`이면 `ValueError`. pre_processor에 `Normalize`가 있으면 `ValueError`. `prepare_pretrained_model`로 teacher 로딩, `prepare_imagenette_data`로 Imagenette 로더(batch 1, shuffle, `Resize(2x)`, `RandomGrayscale(0.3)`, `CenterCrop`, `ToTensor`) 준비, `mean_std`가 미설정이면 train 전체로 teacher 채널 평균·표준편차를 계산한다.
- **손실:** `loss_st`(상위 0.1% 거리의 평균 + Imagenette penalty), `loss_ae`, `loss_stae`의 합이다. 증강은 밝기·대비·채도 중 하나를 `np.random.default_rng()`로 고르며 seed가 걸리지 않는다. 로그 이름은 `train_st`, `train_ae`, `train_stae`, `train_loss`(모두 `on_epoch=True`)다.
- **optimizer:** `Adam(student + ae, lr, weight_decay)`와 `StepLR(step_size=int(0.95 * num_steps), gamma=0.1)`이다. scheduler를 dict 없이 반환하므로 Lightning 기본 interval(epoch)이 적용된다. 원본 동작을 그대로 따른다.
- **검증(`on_validation_start`):** validation의 정상 이미지로 student, ae 맵의 90%, 99.5% quantile(`qa_st`, `qb_st`, `qa_ae`, `qb_ae`)을 계산해 `quantiles`를 갱신한다. 원본은 단독 validate에서도 무조건 갱신한다. visionlab은 D-8에 따라 fit validation에서만 갱신하고 단독 validate·test·predict는 읽기만 한다.
- **평가:** `anomaly_map = 0.5 * map_st + 0.5 * map_stae`(quantile로 정규화), `pred_score = amax(anomaly_map)`이다.
- **`trainer_arguments`:** `{"num_sanity_val_steps": 0}`이다. 모델이 `max_epochs`를 고정하지 않으므로 공유 yaml의 10 epoch를 따른다(사용자 결정, 2026-10-25: 여러 에폭이 필요한 모델은 모두 10 epoch 고정, 모델 yaml로 덮어쓰지 않는다. 처음 yaml의 20 epoch는 제거했다). `learning_type`은 `ONE_CLASS`다.
- **전처리:** 기본 PreProcessor는 `Resize((256, 256), antialias=True)`만 하고 외부 정규화는 없다(D-6, FR-8). visionlab은 Dataset이 resize를 하므로 PreProcessor는 정규화 없이 입력 검증만 한다.
- **후처리와 평가:** Phase 1과 같다. 판정 metric은 Image AUROC와 Pixel AUROC다.

### 1.3 외부 의존과 로컬 자산

- **외부 의존:** torchvision(`ImageFolder`, `transforms.v2`), numpy, tqdm이다. FrEIA, open_clip, timm은 쓰지 않는다. anomalib 의존은 `Batch`, `InferenceBatch`, `AnomalibModule`, `LearningType`, `PreProcessor`, `PostProcessor`, `Evaluator`, `Visualizer`, `download_and_extract`, `extract_transforms_by_type`이며 다운로드 두 곳(`IMAGENETTE_DOWNLOAD_INFO`, `WEIGHTS_DOWNLOAD_INFO`)은 이식하지 않는다(NFR-3).
- **로컬 가중치:** `D:\backbones\efficientad_pretrained_weights\pretrained_teacher_{small,medium}.pth`다. 레퍼런스는 `LocalEfficientAd`가 yaml의 `teacher_file`로 읽는다. visionlab은 `paths.toml`의 `[teachers]`(`efficient_ad_small`, `efficient_ad_medium`, 키는 `efficient_ad_<model_size>`)에서 `EfficientAdLitModel.__init__`이 읽는다(D-19의 "EfficientAD는 그 Phase에서 주입 경로를 정한다"에 대한 선택, 아래 1.4절).
- **Imagenette:** `D:\datasets\imagenette2\train`(9,469장)이며 yaml의 `imagenet_dir` 인자로 전달한다(D-12).
- **데이터:** MVTec `bottle`, VisA `candle`. 분할은 `from_test`, 0.4, seed 42이고 학습 batch 1, 평가 batch 16, `num_workers=0`이다.
- **학습 예산:** 공유 yaml 그대로 `max_epochs=10`이다. 모델 yaml은 `train_batch_size=1`, `eval_batch_size=16`만 덮어쓴다.

### 1.4 이식에서 의도적으로 달라지는 곳

- **teacher 주입(D-19):** 원본 `EfficientAd`는 `on_train_start`의 `prepare_pretrained_model`에서 파일을 읽는다. 이식본은 `EfficientAdLitModel.__init__`이 `paths.toml [teachers]`의 `efficient_ad_<model_size>` 파일을 `model.teacher`에 `strict` 로딩한다. 생성자 인자는 늘리지 않는다. 파일이 없으면 `FileNotFoundError`다. 순수 `EfficientAdModel` 생성자는 원본부터 파일을 읽지 않는다. 사용자 결정 대기: `[teachers]` 섹션은 DECISIONS D-12·D-19에 아직 반영되지 않았다(DECISIONS는 사람이 쓴다).
- **quantile 갱신 시점(D-8):** 위 1.2절. fit 안에서는 원본과 같고 단독 validate에서만 다르다.
- **Imagenette 다운로드와 teacher 다운로드:** 제거하고 없으면 오류를 낸다.
- **재현성:** 원본 증강이 seed되지 않으므로(`np.random.default_rng()`) 레퍼런스도 run마다 값이 달라진다. 알고리즘은 바꾸지 않는다. 에폭별 loss는 STFPM처럼 일치하지 않을 수 있고, 판정은 AUROC 0.01로만 한다.
- **FR-5 허용 오차:** `pred_score`와 `anomaly_map`은 `atol=1e-6, rtol=1e-5`, teacher state_dict는 `torch.equal`이다.

## 2. 레퍼런스

학습 예산은 10 epoch다(사용자 결정, 2026-10-25: 여러 에폭이 필요한 모델은 모두 10 epoch 고정). 처음 `models/02_efficient_ad.yaml`에는 `max_epochs: 20`이 있었고 그 예산으로 레퍼런스와 native를 먼저 실행했으나, 결정 뒤에 그 yaml 항목을 제거하고 10 epoch로 전부 다시 실행했다. 판정에는 10 epoch 결과만 쓴다. 20 epoch 실행은 2.4절에 폐기된 예산으로 남긴다.

### 2.1 10 epoch 레퍼런스

2026-10-25에 agy가 `anomalib_run.py train --config default.yaml --config data/<data>.yaml --config models/02_efficient_ad.yaml`을 데이터셋마다 1회 실행했다(실행으로 검증함). Claude는 로그 파일과 event 파일에서 값을 직접 확인했다. 두 로그 모두 `Seed set to 42`가 나오고 `network access is blocked`, `URLError`, `ConnectionError`, `Traceback`은 0건이다. 로그는 PowerShell `Tee-Object` 출력이라 UTF-16이다.

agy가 PowerShell에서 실행한 명령은 다음과 같다(Claude가 보낸 지시 원문. agy는 묶음 A와 묶음 B를 서로 다른 백그라운드 명령으로 동시에 실행했고 종료 코드를 출력하는 줄을 덧붙였다. 이 실행부터 agy가 auto 모드여서 승인 창이 거의 뜨지 않았다. 모드 변경은 Claude가 한 것이 아니다).

```powershell
cd D:\projects\visionlab\evaluation\anomalib
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
# 묶음 A 1) mvtec 레퍼런스, run 폴더 v2
& $py anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/02_efficient_ad.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\02-ref-mvtec-261025-2.log
# 묶음 A 2) visa 레퍼런스, run 폴더 v1
& $py anomalib_run.py train --config default.yaml --config data/visa.yaml --config models/02_efficient_ad.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\02-ref-visa-261025-2.log
```

다음 표는 10 epoch 레퍼런스(1회) 결과다.

| 데이터 / category | Image AUROC | Pixel AUROC | 참고 | run 폴더 | 로그 |
|---|---|---|---|---|---|
| mvtec / bottle | 1.00000 | 0.97226 | image F1 0.9867, pixel F1 0.7296 | `outputs/anomalib/LocalEfficientAd/LocalMVTecAD/bottle/v2` | `outputs/logs/02-ref-mvtec-261025-2.log` |
| visa / candle | 0.67861 | 0.89552 | image F1 0.6957, pixel F1 0.1686 | `outputs/anomalib/LocalEfficientAd/LocalVisa/candle/v1` | `outputs/logs/02-ref-visa-261025-2.log` |

실제 적용된 학습 예산은 `max_epochs=10`(공유 yaml), `val_split_mode=from_test`, `val_split_ratio=0.4`, `seed=42`, `train_batch_size=1`, `eval_batch_size=16`, `num_workers=0`, `deterministic=warn`, `model_size=small`이다. 저장된 `config.yaml`의 `teacher_file`은 `D:/backbones/efficientad_pretrained_weights/pretrained_teacher_small.pth`, `imagenet_dir`는 `D:\datasets\imagenette2\train`이다. 모델이 고정하는 `trainer_arguments`는 `num_sanity_val_steps=0`뿐이다.

원본 증강이 seed되지 않으므로(`np.random.default_rng()`) 이 수치는 run마다 달라질 수 있다. 반복 실행은 하지 않기로 했으므로 레퍼런스 자체의 편차는 측정하지 못했다.

### 2.2 에폭별 학습 히스토리 (10 epoch 레퍼런스)

값은 run 폴더의 `lightning_logs/version_0/events.out.tfevents.*`에서 읽었다(소수 넷째 자리). 태그는 `train_st_epoch`, `train_ae_epoch`, `train_stae_epoch`, `train_loss_epoch`다. native와의 비교는 5.2절이다.

| 에폭 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| mvtec / bottle `train_loss_epoch` | 10.6256 | 7.1642 | 5.8357 | 5.0814 | 4.5653 | 4.1868 | 3.9448 | 3.7362 | 3.5367 | 3.4021 |
| visa / candle `train_loss_epoch` | 2.2235 | 0.8142 | 0.6312 | 0.5474 | 0.4557 | 0.3959 | 0.3663 | 0.3348 | 0.3140 | 0.2786 |

### 2.3 레퍼런스 확인 사항

- 두 데이터셋 모두 종료 코드 0이다.
- mvtec 폴더 `v0`(1 epoch 동작 확인, medium)과 `v1`(20 epoch, 2.4절)은 레퍼런스가 아니다. visa `v0`(20 epoch)도 레퍼런스가 아니다.

### 2.4 폐기된 예산: 20 epoch 실행

처음 yaml의 `max_epochs: 20`으로 실행한 결과다. 10 epoch 고정 결정으로 판정에서 제외하고 지우지 않고 남긴다.

| 데이터 / category | 단계 | Image AUROC | Pixel AUROC | run 폴더 | 로그 |
|---|---|---|---|---|---|
| mvtec / bottle | 레퍼런스 | 1.00000 | 0.98392 | `outputs/anomalib/LocalEfficientAd/LocalMVTecAD/bottle/v1` | `02-ref-mvtec-261025-1.log` |
| mvtec / bottle | native | 1.00000 | 0.98465 | `outputs/visionlab/efficient_ad/mvtec_bottle/v0` | `02-native-mvtec-261025-1.log` |
| mvtec / bottle | 순수 PyTorch | 1.00000 | 0.98465 | 같은 run의 `predict/metrics.json` | `02-pure-mvtec-261025-1.log` |
| visa / candle | 레퍼런스 | 0.74222 | 0.92242 | `outputs/anomalib/LocalEfficientAd/LocalVisa/candle/v0` | `02-ref-visa-261025-1.log` |
| visa / candle | native | 중단 | 중단 | `outputs/visionlab/efficient_ad/visa_candle/v0`(export 없음) | `02-native-visa-261025-1.log` |

20 epoch native visa는 학습 중에 사용자 결정(10 epoch 고정)으로 Claude가 해당 python 프로세스(`train.py train --data visa --model efficient_ad`)를 종료했다. mvtec의 20 epoch 에폭 히스토리 비교는 `02-history-mvtec-261025-1.log`에 있고 `train_loss_epoch`의 상대차는 최대 약 1.5%였다. 실행했던 명령은 10 epoch 실행의 `-2` 로그 이름만 `-1`이고 나머지는 같다(20 epoch 실행의 명령은 yaml에 `max_epochs: 20`이 있던 상태에서 실행했다).

## 3. 이식 메모

### 2-A 순수 모델

2026-10-25에 codex가 구현했다. 원본 대응은 1.1절의 표와 같고 Claude가 `diff`로 원본과 대조했다. 원본과 다른 줄은 다음 둘뿐이다.

| 파일 | 원본과 다른 줄 |
|---|---|
| `models/efficient_ad/model.py` | `from anomalib.data import InferenceBatch`를 `from visionlab.anomaly_detection.common.batch import InferenceBatch`로 변경(1줄) |
| `models/efficient_ad/__init__.py` | `from .lightning_model import EfficientAd`를 `from .model import EfficientAdModel, EfficientAdModelSize`로 변경, `__all__` 변경(D-18 예외) |

`choose_random_aug_image`의 `np.random.default_rng()`를 포함해 알고리즘은 바꾸지 않았다. 이 모델의 teacher는 원본 생성자가 파일을 읽지 않으므로 D-19용 변경(`pre_trained=False`)이 필요 없었다.

### 2-B Lightning, export·Predictor 일반화, 비교 스크립트

2026-10-25에 codex가 구현했다.

| visionlab 파일 | 대응 또는 하는 일 |
|---|---|
| `models/efficient_ad/lightning_model.py` (`EfficientAdLitModel`) | 원본 `EfficientAd`의 native 이식. 인자 11개 동일. `prepare_imagenette_data`, `teacher_channel_mean_std`, `map_norm_quantiles`, `on_train_start`, `training_step`, `on_validation_start`, `configure_optimizers`는 원본과 같은 동작이고, `batch.image`를 `(image, target)` 튜플로 바꾼 곳과 다운로드 제거가 다르다 |
| `utils/io.py` | `load_pretrained_teacher(module, key)` 추가(`paths.toml [teachers]`를 읽어 strict 로딩). `save_export`에 `model_name` 인자를 추가하고 `load_export`가 `stfpm`, `efficient_ad`를 받도록 일반화(Phase 1 파일 수정) |
| `inference/predictor.py` | `model_name`으로 `STFPMModel` 또는 `EfficientAdModel`을 만든다(Phase 1 파일 수정, dict 하나) |
| `trainers/lightning_cli.py` | 산출 폴더 이름을 class_path 끝에서 세 번째 요소로 정한다(`stfpm`은 그대로, `efficient_ad`)(Phase 1 파일 수정) |
| `models/stfpm/lightning_model.py` | `save_export` 호출에 `model_name="stfpm"` 추가(Phase 1 파일 수정) |
| `configs/models/efficient_ad.yaml` | 레퍼런스 yaml과 같은 모델 인자, `train_batch_size=1`, `eval_batch_size=16`. `max_epochs`는 덮어쓰지 않고 `trainer.yaml`의 10을 따른다 |
| `evaluation/visionlab/predict.py` | `--model efficient_ad` 추가, 모델 yaml의 `eval_batch_size`를 반영(Phase 1 파일 수정) |
| `evaluation/anomalib/compare_outputs.py` | `--model efficient_ad` 추가(FR-6, FR-23, FR-5, FR-24)(Phase 1 파일 수정) |
| `evaluation/anomalib/compare_history.py` | `--tags`로 여러 태그를 표로 비교(Phase 1 파일 수정) |
| `tests/02_efficient_ad/test_efficient_ad.py`, `tests/common/test_history.py` | 신규. `tests/01_stfpm/test_{lightning,predictor}.py`, `tests/common/test_{cli,dependency}.py`는 새 시그니처와 efficient_ad 범위로 수정 |

`paths.toml`의 `[teachers]` 섹션은 Claude가 직접 추가했다(codex의 쓰기 범위 밖). 원본과 다르게 구현한 곳은 1.4절과 같다.

## 4. 적대적 검증

중점 1(anomalib 의존성 완전 삭제)과 중점 2(로컬 pretrained, 오프라인)를 모든 회차에서 확인했다.

### 2-A 순수 모델

| 회차 | 지적 | 심각도 | 처리 |
|---|---|---|---|
| 1 | `model.py`가 원본과 import 한 줄만 다르고 알고리즘 변경이 없음. `src/visionlab`에 anomalib import 0건(docstring 문구 제외), lightning import 0건 | 이상 없음 | 해당 없음 |
| 1 | `__init__.py`가 `lightning_model`을 import하지 않음(D-18 예외) | 이상 없음 | 해당 없음 |

수정 요청은 0회다.

### 2-B Lightning

Claude가 원본 `lightning_model.py`와 이식본을 hook별로 대조했다(codex의 설명은 근거로 쓰지 않았다).

| 회차 | 지적 | 심각도 | 처리 |
|---|---|---|---|
| 1 | `on_train_start`(batch size 1, Normalize 금지, Imagenette 로더, `mean_std` 미설정일 때만 계산), `training_step`(Imagenette iterator 재생성, 로그 4종), `configure_optimizers`(Adam, StepLR, dict 없이 scheduler 반환), `map_norm_quantiles`(라벨 0만, 한 장씩 `get_maps`)가 원본과 같음. 원본의 seed 없는 증강, epoch interval의 scheduler를 그대로 둠 | 이상 없음 | 해당 없음 |
| 1 | teacher 주입이 `EfficientAdLitModel.__init__`에서만 일어나고 `EfficientAdModel` 생성자는 파일을 읽지 않음. `[teachers]` 누락 시 `FileNotFoundError`. `src/`에 anomalib, sklearn, unittest import 0건 | 이상 없음 | 해당 없음 |
| 1 | `test_native_fit_checkpoint_restore_resume_and_evaluation`이 재개 `fit` 직전에 모델을 `eval()` 상태로 두어 `training_step`에서 `TypeError: cannot unpack non-iterable InferenceBatch object`로 실패. 구현이 아니라 테스트의 모드 문제 | Major | 수정 요청 1/3. 재개 `fit` 직전 `restored.train()` 추가. 재실행 22개 통과 |

수정 요청은 1회다(한도 3회 이내).

agy가 PowerShell에서 실행한 명령은 다음과 같다(2026-10-25, 작업 폴더 `D:\projects\visionlab`, `$py`는 `C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe`). 승인 창은 Claude가 명령 텍스트를 확인하고 한 번씩 승인했다.

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
# 1차 (수정 요청 전). 02_efficient_ad 22개 중 21 통과 1 실패, 나머지 통과
& $py -m pytest tests\02_efficient_ad 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\02-pytest-261025-1.log
& $py -m pytest tests\01_stfpm 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\02-pytest-stfpm-261025-1.log
& $py -m pytest tests\common 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\02-pytest-common-261025-1.log
& $py -m pytest tests\00_dataset 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\02-pytest-dataset-261025-1.log
# 2차 (수정 요청 1 뒤)
& $py -m pytest tests\02_efficient_ad 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\02-pytest-261025-2.log
& $py evaluation/anomalib/compare_outputs.py --model efficient_ad --samples 8 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\02-compare-outputs-261025-1.log
```

다음 표는 실행 결과다(실행으로 검증함). agy가 `outputs\logs`의 쓰기 가능 여부를 확인하려는 임시 파일 쓰기(`test.txt` 생성 후 삭제)를 요청해 한 번 승인했다. 위 명령 목록 밖의 실행이다.

| 대상 | 결과 | 로그 |
|---|---|---|
| `tests/02_efficient_ad` | 1차 21개 통과 1개 실패, 2차 22개 통과 | `02-pytest-261025-1.log`, `02-pytest-261025-2.log` |
| `tests/01_stfpm` (Phase 1 회귀) | 32개 통과 | `02-pytest-stfpm-261025-1.log` |
| `tests/common` | 15개 통과(Phase 1 11개에서 증가) | `02-pytest-common-261025-1.log` |
| `tests/00_dataset` | 10개 통과 | `02-pytest-dataset-261025-1.log` |
| `compare_outputs.py --model efficient_ad` | FAIL 0건. FR-6, FR-23, FR-5(mvtec, visa, 통계 미설정·설정 두 경우 모두), FR-24 일치. 최대 절대 차이 0.0 | `02-compare-outputs-261025-1.log` |

FR-23의 teacher 가중치는 `D:\backbones\efficientad_pretrained_weights\pretrained_teacher_small.pth`(sha256 `a16ded54...ae7d`)이고 레퍼런스 쪽 teacher와 `torch.equal`로 같다.

## 5. 이식 결과 (10 epoch)

agy가 PowerShell에서 실행한 명령은 다음과 같다(2026-10-25, 작업 폴더 `D:\projects\visionlab`, `$py`는 `C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe`). 묶음 B는 레퍼런스 묶음 A와 동시에 실행했다. pytest 3차는 10 epoch 변경 뒤의 확인이다.

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py -m pytest tests\02_efficient_ad 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\02-pytest-261025-3.log
& $py -m pytest tests\common 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\02-pytest-common-261025-2.log
# 묶음 B
& $py evaluation/visionlab/train.py train --data mvtec --model efficient_ad 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\02-native-mvtec-261025-2.log
& $py evaluation/visionlab/predict.py --model efficient_ad --data mvtec 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\02-pure-mvtec-261025-2.log
& $py evaluation/visionlab/train.py train --data visa --model efficient_ad 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\02-native-visa-261025-2.log
& $py evaluation/visionlab/predict.py --model efficient_ad --data visa 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\02-pure-visa-261025-2.log
# 히스토리 비교
& $py evaluation/anomalib/compare_history.py --reference D:\projects\visionlab\outputs\anomalib\LocalEfficientAd\LocalMVTecAD\bottle\v2 --native D:\projects\visionlab\outputs\visionlab\efficient_ad\mvtec_bottle\v1 --tags train_st_epoch train_ae_epoch train_stae_epoch train_loss_epoch 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\02-history-mvtec-261025-2.log
& $py evaluation/anomalib/compare_history.py --reference D:\projects\visionlab\outputs\anomalib\LocalEfficientAd\LocalVisa\candle\v1 --native D:\projects\visionlab\outputs\visionlab\efficient_ad\visa_candle\v1 --tags train_st_epoch train_ae_epoch train_stae_epoch train_loss_epoch 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\02-history-visa-261025-2.log
```

### 5.1 최종 성능 (레퍼런스, native, 순수 PyTorch)

다음 표는 category별 Image AUROC와 Pixel AUROC다. 판정 기준은 레퍼런스와의 차이 0.01 이내다(FR-18, FR-20). 세 단계 모두 10 epoch다. native는 `train.py`가 test에서 기록한 값이고 순수 PyTorch는 `export/model.pt`만 불러 `predict.py`로 평가한 값이다.

| 데이터 / category | metric | 레퍼런스 | native Lightning | 순수 PyTorch | 차이(native) | 차이(순수) | 판정 |
|---|---|---:|---:|---:|---:|---:|---|
| mvtec / bottle | Image AUROC | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 통과 |
| mvtec / bottle | Pixel AUROC | 0.972259 | 0.974315 | 0.974315 | 0.002056 | 0.002056 | 통과 |
| visa / candle | Image AUROC | 0.678611 | 0.682500 | 0.682500 | 0.003889 | 0.003889 | 통과 |
| visa / candle | Pixel AUROC | 0.895520 | 0.891759 | 0.891759 | 0.003761 | 0.003761 | 통과 |

참고로 F1은 다음과 같다. 판정 대상이 아니다. pixel F1은 임계값 선택에 민감해 편차가 크다(visa candle 0.1686 대 0.1169).

| 데이터 / category | image F1 (레퍼런스 / native / 순수) | pixel F1 (레퍼런스 / native / 순수) |
|---|---|---|
| mvtec / bottle | 0.9867 / 0.9867 / 0.9867 | 0.7296 / 0.7507 / 0.7507 |
| visa / candle | 0.6957 / 0.6988 / 0.6988 | 0.1686 / 0.1169 / 0.1169 |

native와 순수 PyTorch의 차이는 Image AUROC 0, Pixel AUROC 약 1e-7이다. run 폴더와 로그는 다음과 같다.

| 데이터 | 레퍼런스 run | native run 폴더 | 순수 PyTorch 결과 | 로그 |
|---|---|---|---|---|
| mvtec / bottle | `outputs/anomalib/LocalEfficientAd/LocalMVTecAD/bottle/v2` | `outputs/visionlab/efficient_ad/mvtec_bottle/v1` | `v1/predict/metrics.json` | `02-native-mvtec-261025-2.log`, `02-pure-mvtec-261025-2.log` |
| visa / candle | `outputs/anomalib/LocalEfficientAd/LocalVisa/candle/v1` | `outputs/visionlab/efficient_ad/visa_candle/v1` | `v1/predict/metrics.json` | `02-native-visa-261025-2.log`, `02-pure-visa-261025-2.log` |

각 native run 폴더에는 `run.json`(설정, teacher 가중치 경로와 sha256), `export/model.pt`, `weights/lightning`, `lightning_logs`가 있다. `run.json`의 `max_epochs`는 10, `train_batch_size`는 1, seed는 42이고 teacher sha256은 두 run 모두 `a16ded54...ae7d`다.

### 5.2 에폭별 학습 히스토리 (Claude 검증 중점 2)

사용자가 허용 오차를 정하지 않아 판정 기준에서 제외하고 비교 표로만 기록한다. 값은 `compare_history.py`가 event 파일에서 읽었다(`02-history-*-261025-2.log`, 소수 넷째 자리). 다음 표는 `train_loss_epoch`다(상대차 = 절댓값 차이 / 레퍼런스).

| 에폭 | mvtec 레퍼런스 | mvtec native | 상대차 | visa 레퍼런스 | visa native | 상대차 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.6256 | 10.6187 | 0.0007 | 2.2235 | 2.2111 | 0.0056 |
| 1 | 7.1642 | 7.1386 | 0.0036 | 0.8142 | 0.7977 | 0.0203 |
| 2 | 5.8357 | 5.8110 | 0.0042 | 0.6312 | 0.6088 | 0.0355 |
| 3 | 5.0814 | 5.0472 | 0.0067 | 0.5474 | 0.5375 | 0.0180 |
| 4 | 4.5653 | 4.5283 | 0.0081 | 0.4557 | 0.4524 | 0.0073 |
| 5 | 4.1868 | 4.1637 | 0.0055 | 0.3959 | 0.3997 | 0.0095 |
| 6 | 3.9448 | 3.9184 | 0.0067 | 0.3663 | 0.3528 | 0.0369 |
| 7 | 3.7362 | 3.7096 | 0.0071 | 0.3348 | 0.3284 | 0.0192 |
| 8 | 3.5367 | 3.5104 | 0.0074 | 0.3140 | 0.3086 | 0.0170 |
| 9 | 3.4021 | 3.3710 | 0.0091 | 0.2786 | 0.2724 | 0.0222 |

다음 표는 태그별 10 에폭 중 최대 상대차다.

| 태그 | mvtec | visa |
|---|---:|---:|
| `train_st_epoch` | 0.0074 | 0.0417 |
| `train_ae_epoch` | 0.0420 | 0.0531 |
| `train_stae_epoch` | 0.0728 | 0.0640 |
| `train_loss_epoch` | 0.0091 | 0.0369 |

STFPM과 달리 값이 소수 일곱째 자리까지 일치하지 않는다. 원본의 밝기·대비·채도 증강이 seed되지 않아(`np.random.default_rng()`) 같은 seed로도 run마다 다르기 때문으로 보인다. 곡선의 모양과 크기는 두 데이터셋 모두 레퍼런스와 같다. 20 epoch 실행(2.4절)의 mvtec 상대차도 최대 약 1.5%로 같은 수준이었다. 이 편차를 직접 측정하려면 레퍼런스 반복 실행이 필요하나 하지 않기로 했으므로, 원인이 증강의 난수라는 것은 원본 소스에서 읽은 추정이고 실측으로 확인하지 않았다.

### 5.3 두 검증 중점의 확인 결과

| 검증 중점 | 확인 방법 | 결과 |
|---|---|---|
| 1. anomalib 의존성 완전 삭제, native Lightning만 사용 | `src/` 정적 검색, `tests/common/test_dependency.py`(import 차단), `tests/02_efficient_ad`의 subprocess 차단 테스트 | `src/visionlab`에 anomalib, sklearn, unittest import 0건(docstring 문구 제외). Lightning import는 `lightning_model.py`, `lightning_data.py`, `lightning_visualizer.py`, `lightning_cli.py`에만 있고 순수 경로(`model.py`, `predictor.py`)에는 없다. pytest 통과 |
| 2. 로컬 pretrained 사용, 네트워크 차단 | `train.py`의 `block_network()`, 로그 검색, `run.json`의 teacher 기록, `compare_outputs.py` | 두 native 로그에 `network access is blocked`, `URLError`, `ConnectionError`, `Traceback` 0건. teacher는 `paths.toml [teachers]`의 `pretrained_teacher_small.pth`(sha256 `a16ded54...ae7d`)이고 레퍼런스 teacher와 `torch.equal`로 같다(FR-23) |

### 5.4 테스트와 비교 스크립트 요약

모두 실행으로 검증했다.

| 대상 | 결과 | 로그 |
|---|---|---|
| `tests/02_efficient_ad` | 22개 통과(10 epoch 변경 뒤 3차) | `02-pytest-261025-3.log` |
| `tests/common` | 15개 통과(10 epoch 변경 뒤) | `02-pytest-common-261025-2.log` |
| `tests/01_stfpm`, `tests/00_dataset` | 32개, 10개 통과(Phase 1·0 회귀, 10 epoch 변경 전 실행) | `02-pytest-stfpm-261025-1.log`, `02-pytest-dataset-261025-1.log` |
| `evaluation/anomalib/compare_outputs.py --model efficient_ad` | FAIL 0건. FR-6, FR-23, FR-5, FR-24 일치, 최대 절대 차이 0.0 (4절) | `02-compare-outputs-261025-1.log` |

10 epoch 변경은 yaml과 테스트 기대값뿐이어서 모델·loss·optimizer 코드는 바뀌지 않았다. `tests/01_stfpm`은 그 변경 뒤에 다시 실행하지 않았다(STFPM 코드와 yaml은 변경 대상이 아니다).

## 6. 결론

PLAN의 Phase 2 판정 기준(범주별 Image·Pixel AUROC 차이 0.01 이내)은 mvtec/bottle, visa/candle 모두 native Lightning과 순수 PyTorch에서 10 epoch 기준으로 충족했다. 차이의 최댓값은 0.0039(visa Image AUROC)다. FR-15(Hook별 상태, 재개 시 teacher·통계 미초기화), FR-5, FR-6, FR-22, FR-23, FR-24는 pytest와 `compare_outputs.py`로 확인했다. 두 검증 중점(anomalib 의존성 제거, 로컬 pretrained 사용)도 로그와 산출물로 확인했다. 수정 요청은 2-B에서 1회(한도 3회 이내)였다.

남은 항목과 위험은 다음과 같다.

- **Phase 종료:** 사용자가 종료를 결정했다(2026-10-25). `Phase 02: efficient_ad ...` 형식으로 커밋·푸시한다. 종료 전에 제안했던 Phase 1·0 회귀 테스트 재실행과 native visa 반복 실행(편차 측정)은 사용자 결정으로 하지 않았다. agy에 보낸 명령은 승인 전에 취소했고 실행된 것이 없다.
- **재현 편차 미측정:** 레퍼런스가 데이터셋마다 1회뿐이고 원본 증강이 seed되지 않아 레퍼런스 자체의 run 간 편차는 측정하지 못했다. native 반복 실행도 하지 않았다. visa Image AUROC가 낮아(약 0.68) 우연히 기준 안에 든 것일 수 있다. 20 epoch 실행은 visa Image AUROC가 0.742로 10 epoch(0.679)와 0.06 차이가 났으나, 예산 차이와 편차를 분리할 수 없는 값이다.
- **에폭 히스토리 허용 오차:** 사용자가 정하지 않아 판정에서 제외했다. 상대차는 mvtec 최대 약 0.9%(`train_loss_epoch`), visa 최대 약 3.7%다.
- **사용자 문서 갱신 필요:** `specs/`는 사람이 고친다. `paths.toml`의 `[teachers]` 섹션은 DECISIONS D-12(73행), D-15 트리(95행), SPEC FR-23(31행)의 "`[backbones]`만" 서술과 어긋나고 D-19(232행)에 결과가 없다. 10 epoch 고정은 PLAN 84행 기본값과는 맞지만 결정으로는 기록되지 않았고, PLAN 121행(FRE 220 epoch)과 충돌한다.
- **Phase 1·다른 Phase 파일 변경:** `io.py`, `predictor.py`, `lightning_cli.py`, `predict.py`, `compare_outputs.py`, `compare_history.py`, `models/stfpm/lightning_model.py`와 일부 STFPM·공통 테스트를 일반화했고, 다른 Phase용 yaml(07, 08, 12)에서 에폭 덮어쓰기를 제거했다.
- **기타:** agy가 중간에 auto 모드로 바뀌었다(제가 바꾼 것이 아님). 지시 밖 실행은 읽기 전용 로그 조회들과 `outputs\logs`에 임시 파일을 쓰고 지우는 확인 1건이다. 중단한 20 epoch 산출물은 삭제하지 않았다.
