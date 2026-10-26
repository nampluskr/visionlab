# Phase 9. FRE

모델 폴더 이름은 `fre`이고 원본은 anomalib 2.0.0의 `anomalib\models\image\fre\`(`C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\Lib\site-packages\anomalib\models\image\fre\`)이다. 범위와 완료 조건은 `specs/PLAN.md`의 Phase 9와 "Phase 공통 양식"을 따른다. 판정 기준은 category별 Image·Pixel AUROC 차이 0.01 이내다.

## 상태

다음 표는 단계별 진행 상태다. 수정 반복 한도 3회는 단계별로 센다.

| 단계 | 내용 | 상태 |
|---|---|---|
| 회귀 실행 | 사용자 승인 회귀 실행 1회 (`tests\common`, `tests\01_stfpm` ~ `08_cfa`) | 실행 완료, 230 passed (아래 절) |
| 중단한 220 epoch 레퍼런스 | 최초 결정(220 epoch, capsule)으로 시작했다가 에폭 정책 변경으로 중단 | 중단 (아래 절) |
| 9-R | 레퍼런스 어댑터 `LocalFre`(`max_epochs` 10)와 `09_fre.yaml` | 구현, 검증, 실행 완료 (수정 요청 0회) |
| 레퍼런스 | `anomalib_run.py train` x 2 (mvtec/bottle 1회, visa/candle 1회, 10 epoch, `LocalFre`) | 실행 완료 (2절) |
| 9-A | 순수 모델 (`models/fre/model.py`, `tests/09_fre`의 순수 부분) | 구현, 검증, 실행 완료 (수정 요청 0회) |
| 9-B | Lightning (`FreLitModel`, Adam 학습, configs, export·Predictor 등록, 비교 스크립트) | 구현, 검증, 실행 완료 (수정 요청 0회, 결과 5절) |

특징 추출기(`TimmFeatureExtractor`)와 처리 구성요소(PreProcessor, PostProcessor, Evaluator)는 앞 Phase의 것을 재사용한다.

## 사용자 결정 (2026-10-26)

다음은 Phase 시작 전에 사용자가 정한 항목이다.

| 항목 | 결정 |
|---|---|
| 에폭 정책 변경 (2026-10-26) | 최초 결정(220 epoch 그대로)을 철회하고 안 B, 10 epoch 통일로 바꾼다. 사용자 요청은 "10 에폭만 실행"이다. 레퍼런스와 native 모두 10 epoch다. 레퍼런스는 `evaluation/anomalib/local_models.py`의 `LocalFre`가 모델이 고정한 `max_epochs: 220`을 10으로 바꾸고, native는 `FreLitModel.trainer_arguments`의 `max_epochs`를 10으로 둔다(원본의 220과 다른 의도된 차이). 레퍼런스와 native가 같은 10 epoch인지 `run.json`과 로그의 실제 `max_epochs`로 확인해 기록한다 |
| 평가 category | 기본값 mvtec/bottle과 visa/candle로 되돌린다. 최초 결정의 capsule은 220 epoch일 때만 해당하는 조건이었다. `specs/PLAN.md`는 고치지 않는다 |
| 학습 시간 관문 | 10 epoch 합계 추정(약 15분)의 1.5배(약 22분)를 넘을 것으로 보이면 계속하기 전에 사용자에게 보고한다. category는 스스로 바꾸지 않는다 |
| 앞 모델 회귀 | 9-A 전에 `tests\common`과 `tests\01_stfpm` ~ `08_cfa`를 한 번 실행한다(사용자 승인, 이번 한 번에 한해 예외). 실패가 나오면 스스로 고치지 않고 멈춰 보고한다 |
| FR-5 허용 오차 | `atol=1e-6, rtol=1e-5` (Phase 2~8과 같음) |
| 에폭별 `train_loss` 히스토리 | 허용 오차 없이 비교표만 기록한다. 10행이므로 요약 없이 전체 표를 본문에 적는다 |
| `specs/` 변경 (최종 보고 뒤, 2026-10-26) | 사용자가 권고를 채택했다. `DECISIONS.md` D-22를 6.1절의 초안대로 고치고 `PLAN.md` 9행 문구를 이번 Phase에 맞게 최소한으로 고친다. 변경은 이 두 곳뿐이고 INTENT와 SPEC은 건드리지 않는다 |
| `test_dependency.py` 전체 실행 (사후, 2026-10-26) | 사용자가 승인했다 |
| 9-B 이후 앞 모델 회귀 (2026-10-26) | 보류한다(사용자 결정). 이번에는 돌리지 않고 영향받을 수 있는 공유 파일은 6.2절의 표에 남긴다 |
| Phase 종료 (2026-10-26) | 사용자가 종료를 결정했다. `Phase 09: fre ...` 커밋으로 푸시한다 |
| 실행 범위 | FRE의 명령만 실행한다(위 회귀 1회만 예외) |
| 판정 | Image·Pixel AUROC를 판정한다. 정규화 점수 기준 값(D-24)과 raw 값을 함께 기록한다 |

## 사용자 승인 회귀 실행 (2026-10-26)

9-A 전에 앞 모델의 단위 테스트를 한 번 실행했다(사용자 승인, 이번 한 번에 한해 예외). 학습, 평가, `compare_outputs`는 포함하지 않았다. Phase 8 직전(Phase 7 직전 142개)의 마지막 전 범위 실행 뒤 처음 하는 실행이다. 실행으로 검증했고 Claude가 로그 파일에서 요약 줄을 확인했다.

| 대상 | 결과 | 로그 |
|---|---|---|
| `tests\common` | 34 passed | `outputs/logs/09-regress-common-261026-1.log` |
| `tests\01_stfpm` | 32 passed | `09-regress-stfpm-261026-1.log` |
| `tests\02_efficient_ad` | 22 passed | `09-regress-efficient-ad-261026-1.log` |
| `tests\03_padim` | 16 passed | `09-regress-padim-261026-1.log` |
| `tests\04_dfm` | 22 passed | `09-regress-dfm-261026-1.log` |
| `tests\05_dfkde` | 20 passed | `09-regress-dfkde-261026-1.log` |
| `tests\06_patchcore` | 24 passed | `09-regress-patchcore-261026-1.log` |
| `tests\07_reverse_distillation` | 25 passed | `09-regress-reverse-distillation-261026-1.log` |
| `tests\08_cfa` | 35 passed | `09-regress-cfa-261026-1.log` |

합계 230개가 통과했고 실패는 없다. 실행한 명령은 다음과 같다(Claude가 보낸 지시 원문. agy가 시간 측정 줄을 덧붙여 하나의 스크립트로 순서대로 실행했고, 승인 창의 전문을 Claude가 agy의 transcript에서 읽어 지시와 같음을 확인한 뒤 한 번 승인했다).

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py -m pytest tests\common 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\09-regress-common-261026-1.log
& $py -m pytest tests\01_stfpm 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\09-regress-stfpm-261026-1.log
& $py -m pytest tests\02_efficient_ad 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\09-regress-efficient-ad-261026-1.log
& $py -m pytest tests\03_padim 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\09-regress-padim-261026-1.log
& $py -m pytest tests\04_dfm 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\09-regress-dfm-261026-1.log
& $py -m pytest tests\05_dfkde 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\09-regress-dfkde-261026-1.log
& $py -m pytest tests\06_patchcore 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\09-regress-patchcore-261026-1.log
& $py -m pytest tests\07_reverse_distillation 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\09-regress-reverse-distillation-261026-1.log
& $py -m pytest tests\08_cfa 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\09-regress-cfa-261026-1.log
```

## 중단한 220 epoch 레퍼런스

최초 결정(220 epoch 그대로, category는 bottle과 capsule)으로 시작한 레퍼런스를 에폭 정책 변경(위 결정 표)으로 중단했다. 지우지 않고 기록한다.

| 항목 | 내용 |
|---|---|
| 시작 | 2026-10-26 17:19, `anomalib_run.py train ... models/09_fre.yaml`(원본 `Fre`, 220 epoch), 로그 `outputs/logs/09-ref-mvtec-261026-1.log` |
| 중단 | 약 17:22, 에폭 18까지(19 에폭) 완료한 시점(220 중 약 9%). 사용자 지시(2026-10-26)에 따른 중단이며 실패가 아니다 |
| 중단 방법 | agy가 스크립트를 실행하던 PowerShell(PID 17148)과 python(PID 2920)만 종료했다. 스크립트를 먼저 종료해 다음 명령(capsule)이 시작되지 않았다. Claude가 학습 python 프로세스 0개와 GPU 947 MiB, 1%를 직접 확인했다 |
| 속도 | 에폭당 학습 약 6초(약 4 it/s, 27 step)였고 검증을 더해 에폭당 약 10초(19 에폭이 약 3분)였다. 220 epoch로 환산하면 bottle 약 35분으로 계획의 추정(33~44분) 범위 안이다 |
| 남은 산출물 | `outputs/logs/09-ref-mvtec-261026-1.log`, `outputs/anomalib/Fre/LocalMVTecAD/bottle/v0`(학습 도중의 체크포인트 `weights/lightning/model.ckpt` 208,213,708 bytes 포함, 에폭 18 시점의 가중치)를 지우지 않았다. 이 run은 레퍼런스가 아니다. 새 10 epoch 레퍼런스는 어댑터 클래스 이름의 폴더 `outputs/anomalib/LocalFre/...`에 있다 |

## 1. 모델 노트

### 1.1 원본과 생성자

다음 표는 원본 파일과 이식 위치의 대응이다(D-15, D-18).

| 원본 (`anomalib\models\image\fre\`) | 이식 위치 | 비고 |
|---|---|---|
| `torch_model.py` (`FREModel`, `TiedAE`) | `src/visionlab/anomaly_detection/models/fre/model.py` | import 변경, `pre_trained`는 순수 모델에서 구조만 만드는 방식으로(D-19) |
| `lightning_model.py` (`Fre`) | `.../models/fre/lightning_model.py` (`FreLitModel`) | native `LightningModule`로 새로 작성 |
| `__init__.py` | `.../models/fre/__init__.py` | `lightning_model`을 import하지 않도록 고침(D-18 예외) |

생성자는 다음과 같다. `FreLitModel`은 `Fre`와 인자 이름·순서·기본값이 같아야 한다(FR-6, D-5).

| 클래스 | 시그니처 |
|---|---|
| `Fre` | `backbone="resnet50"`, `layer="layer3"`, `pre_trained=True`, `pooling_kernel_size=2`, `input_dim=65536`, `latent_dim=220`, `pre_processor=True`, `post_processor=True`, `evaluator=True`, `visualizer=True` (10개) |
| `FREModel` | `backbone`, `layer`, `input_dim=65536`, `latent_dim=220`, `pre_trained=True`, `pooling_kernel_size=4` |
| `TiedAE` | `input_dim`, `latent_dim` |

`FREModel`의 `pooling_kernel_size` 기본값(4)과 `Fre`의 기본값(2)이 다르다. `FreLitModel`은 `Fre`와 같게 둔다.

### 1.2 학습과 평가 동작

원본의 동작은 다음과 같다.

- **모델:** `TimmFeatureExtractor`(eval)로 `layer` 하나의 특징(resnet50 `layer3`, 1024채널)을 뽑고 `avg_pool2d(kernel 2)`한 뒤 `(batch, -1)`로 편다(256x256 입력에서 1024x8x8 = 65536차원). `TiedAE`가 `latent_dim`(220)차원으로 줄였다가 같은 가중치의 전치로 복원한다(`weight`는 `(latent_dim, input_dim)`, `xavier_uniform_`, 편향은 0으로 시작). `get_features`는 `no_grad` 없이 특징을 만들고 `detach()`한다.
- **학습:** `Adam(lr=1e-3)`이 `TiedAE`의 파라미터만 학습한다. 손실은 입력 특징과 복원의 `MSELoss`이고 `train_loss`를 `on_epoch=True`로 로그한다.
- **추론:** 복원 오차의 제곱을 채널 방향으로 합해 `anomaly_map`(`[B,1,h,w]` 뒤 입력 크기로 bilinear 보간)을 만들고, `pred_score`는 위치 방향 합(`[B]`)이다.
- **`trainer_arguments`:** 원본은 `{"gradient_clip_val": 0, "max_epochs": 220, "num_sanity_val_steps": 0}`이다. `anomalib/engine/engine.py`가 이를 설정 위에 덮어쓰므로 어댑터 없이는 레퍼런스가 220 epoch를 돈다. 이번 Phase는 사용자 결정으로 10 epoch를 쓰므로 `LocalFre`가 `max_epochs`만 10으로 바꾸고, `FreLitModel.trainer_arguments`도 `max_epochs`를 10으로 둔다. visionlab의 `LitCLI`는 모델의 `trainer_arguments`를 강제한다.

### 1.3 외부 의존과 로컬 자산

- **외부 의존:** timm, torch뿐이다. 새 외부 라이브러리는 없다. anomalib 의존은 `Batch`, `InferenceBatch`, `AnomalibModule`, `TimmFeatureExtractor`, `PreProcessor`, `PostProcessor`, `Evaluator`, `Visualizer`, `LearningType`이다.
- **로컬 가중치:** `paths.toml [backbones]`의 `resnet50`(이미 있다). `TiedAE`는 무작위 초기화다. visionlab은 `FreLitModel.__init__`이 `feature_extractor`에 주입한다(D-19).
- **데이터:** MVTec `bottle`(학습 209장), VisA `candle`(학습 900장). 분할은 `from_test`, 0.4, seed 42, batch 8/8, `num_workers=0`이다.
- **학습 예산:** `max_epochs=10`(사용자 결정. 원본의 고정값 220을 바꾼 것)이다.

### 1.4 이식에서 의도적으로 달라지는 곳과 유지하는 곳

- **pretrained 주입(D-19):** 순수 `FREModel`은 `pre_trained=False`로 특징 추출기 구조만 만든다. `FreLitModel`이 `resnet50` 로컬 파일을 불러온다.
- **원본과 다르게 둔 에폭(사용자 결정 2026-10-26):** 원본이 고정한 `max_epochs=220`을 10으로 바꾼다. 레퍼런스는 `LocalFre`(`evaluation/anomalib/local_models.py`), native는 `FreLitModel.trainer_arguments`다. 220 epoch에서 얻는 "원본이 의도한 성능"은 이번 Phase에서 확인하지 못한다(6.2절).
- **원본 동작 유지:** `get_features`의 `no_grad` 없는 특징 추출과 `detach()`, 매 호출 `feature_extractor.eval()`, `TiedAE`의 초기화 순서(`weight`, `decoder_bias`, `encoder_bias`), `input_dim` 가정(256x256, `resnet50` `layer3`, `pooling_kernel_size=2`)은 고치지 않는다.
- **`train_loss` 로그의 집계:** 원본은 batch 크기를 추론하지 못해 epoch 평균이 step마다 1회씩 센 평균이 된다. visionlab은 `batch_size=1`을 두어 맞춘다(Phase 7 선례).
- **초기 가중치:** `TiedAE`의 `xavier_uniform_`은 난수를 소비한다. 레퍼런스와 같은 seed에서 같은 순서로 소비해야 하며 seed 42 초기 가중치 비교를 정보로 기록한다.
- **FR-5 허용 오차:** `pred_score`와 `anomaly_map`은 `atol=1e-6, rtol=1e-5`, 특징 추출기 가중치는 `torch.equal`이다.

## 2. 레퍼런스

학습 예산은 사용자 결정(2026-10-26)에 따른 10 epoch다. 원본이 고정한 220 epoch를 `LocalFre`가 10으로 바꾼 것이다(1.2절, 1.4절).

### 2.1 10 epoch 레퍼런스

2026-10-26에 agy가 `anomalib_run.py train --config default.yaml --config data/<data>.yaml --config models/09_fre.yaml`을 데이터셋마다 1회 실행했다(실행으로 검증함). `09_fre.yaml`은 `local_models.LocalFre`를 쓴다. Claude는 로그 파일에서 값을 직접 확인했다. 두 로그 모두 `network access is blocked`, `URLError`, `ConnectionError`, `Traceback`, `out of memory`는 0건이다. 로그는 UTF-16이다. 승인 창의 스크립트 전문은 Claude가 agy의 transcript에서 읽어 지시한 명령과 같음을 확인했다.

agy가 PowerShell에서 실행한 명령은 다음과 같다(Claude가 보낸 지시 원문. 파일은 고치지 않았다).

```powershell
cd D:\projects\visionlab\evaluation\anomalib
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/09_fre.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\09-ref-mvtec-261026-2.log
& $py anomalib_run.py train --config default.yaml --config data/visa.yaml --config models/09_fre.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\09-ref-visa-261026-1.log
```

다음 표는 10 epoch 레퍼런스(1회) 결과다.

| 데이터 / category | Image AUROC | Pixel AUROC | 참고 | run 폴더 | 로그 |
|---|---|---|---|---|---|
| mvtec / bottle | 0.98904 | 0.97401 | image F1 0.9737, pixel F1 0.6396 | `outputs/anomalib/LocalFre/LocalMVTecAD/bottle/v0` | `outputs/logs/09-ref-mvtec-261026-2.log` |
| visa / candle | 0.92528 | 0.97813 | image F1 0.8571, pixel F1 0.1847 | `outputs/anomalib/LocalFre/LocalVisa/candle/v0` | `outputs/logs/09-ref-visa-261026-1.log` |

run 폴더 이름이 `LocalFre`인 것은 어댑터 클래스 이름이다. 적용된 학습 예산은 `max_epochs=10`, `val_split_mode=from_test`, `val_split_ratio=0.4`, `seed=42`, `batch 8/8`, `num_workers=0`, `deterministic=warn`, `backbone=resnet50`, `layer=layer3`, `pre_trained=true`, `pooling_kernel_size=2`, `input_dim=65536`, `latent_dim=220`이다. 체크포인트는 각 run 폴더의 `weights/lightning/model.ckpt`이고 크기는 208,213,708 bytes다.

### 2.2 레퍼런스와 native의 에폭 확인

사용자 지시에 따라 레퍼런스와 native가 같은 10 epoch인지 실제 값으로 확인했다.

| 대상 | 근거 | 값 |
|---|---|---|
| 레퍼런스 (`LocalFre`, bottle, visa) | 로그의 `Trainer.fit stopped: max_epochs=10 reached`, 완료한 가장 큰 에폭 번호 9 | 10 |
| native (bottle, visa) | `run.json`의 `config.trainer.max_epochs`, 로그의 `Trainer.fit stopped: max_epochs=10 reached`, 완료한 가장 큰 에폭 번호 9 | 10 |
| 원본 `Fre` (중단한 run) | 로그의 `Overriding max_epochs from 10 ...` 줄, 10을 넘겨 에폭 18까지 진행함 | 220 (덮어쓰기가 적용됨) |

`LocalFre`를 쓴 레퍼런스 로그에는 `max_epochs` 덮어쓰기 줄이 없다. 어댑터가 반환한 10이 공유 설정의 10과 같기 때문이다.

### 2.3 학습 시간 관문과 실제 학습 시간

사용자 결정에 따라 10 epoch 합계 추정(약 15분)의 1.5배(약 22분)를 넘을 것으로 보이면 보고하기로 했다. 실제로는 파일 시각으로 bottle 레퍼런스가 약 2분(17:27 시작, 체크포인트 17:28), visa 레퍼런스가 약 4~5분(17:28 시작, 체크포인트 17:33)이었다. 합계 약 7분이라 추정 이내였고 관문 기준을 넘지 않아 보고 없이 진행했다.

참고로 중단한 220 epoch run의 속도(에폭당 약 10초, 19 에폭이 약 3분)로 계산하면 bottle 220 epoch가 약 35분, visa(에폭당 약 27초)가 약 100분이다.

### 2.4 에폭별 학습 히스토리

레퍼런스와 native의 비교는 5.3절에 표로 기록한다. 허용 오차는 정하지 않았다.

## 3. 이식 메모

### 3.1 원본과 이식 파일의 대응

다음 표는 새로 만든 파일과 원본의 대응이다. 원본과의 차이는 `difflib`로 확인했다(구현함, 정적 검증).

| 이식 파일 (`src/visionlab/anomaly_detection/models/fre/`) | 원본 | 원본과 다른 줄 |
|---|---|---|
| `model.py` | `torch_model.py` | import 2줄(`InferenceBatch`, `TimmFeatureExtractor`), `TimmFeatureExtractor(... pre_trained=False)`와 주석 한 줄(D-19) |
| `__init__.py` | `__init__.py` | `FREModel`만 노출(D-18 예외) |
| `lightning_model.py` (`FreLitModel`) | `lightning_model.py` | native `LightningModule`로 새로 작성. 10개 인자 동일, `Adam(lr 1e-3)`이 `TiedAE`만 학습, MSE 손실, `train_loss` 로그(`batch_size=1`), 로컬 backbone 주입(D-19), D-8의 post_processor 갱신 제한, `trainer_arguments`의 `max_epochs` 10(원본 220, 사용자 결정) |

### 3.2 새로 쓴 코드와 앞 Phase 공유 파일 변경

- **새로 쓴 파일:** `configs/models/fre.yaml`, `tests/09_fre/test_fre.py`, `tests/09_fre/test_fre_lightning.py`. `tests\09_fre`는 합계 16개 테스트다.
- **레퍼런스 어댑터 변경:** `evaluation/anomalib/local_models.py`에 `LocalFre`를 추가했고(파일 첫 줄 설명도 확장), `evaluation/anomalib/models/09_fre.yaml`의 `class_path`를 `local_models.LocalFre`로 바꾸고 주석 한 줄을 달았다. `LocalFre`는 부모의 `trainer_arguments`를 복사해 `max_epochs`만 10으로 바꾼다.
- **공유 파일 변경:** `utils/io.py`(`load_export`가 `fre` 허용), `inference/predictor.py`(모델 dict에 등록), `evaluation/visionlab/predict.py`(`--model fre`), `evaluation/anomalib/compare_outputs.py`(`--model fre`), `tests/common/test_cli.py`, `tests/common/test_dependency.py`. `common/` 아래 파일은 바꾸지 않았다. 영향 범위는 6.2절의 표에 정리했다.
- **새 부품:** `TiedAE`와 `FREModel`은 원본 복사다. 새로 구현한 대체 부품은 없다.

### 3.3 이식 중 확인된 사항

- **에폭 정책 변경과 중단한 run:** 처음에는 220 epoch로 시작했다가 사용자 지시로 10 epoch로 바꾸고 중단했다. 중단한 run의 기록은 "중단한 220 epoch 레퍼런스" 절에 있다.
- **`LocalFre`의 적용 근거:** `anomalib/engine/engine.py`의 `_setup_trainer()`가 모델의 `trainer_arguments`를 캐시에 반영한 뒤 `Trainer(**self._cache.args)`를 만든다(codex가 `engine.py` 300~311행으로 확인). 실제 적용은 위 2.2절의 로그로 확인했다.
- **`train_loss`의 집계:** Phase 7과 같이 `self.log(..., batch_size=1)`을 처음부터 두었고, native의 에폭별 손실이 레퍼런스와 상대 차이 5.2e-7 이내로 일치했다(5.3절).
- **초기 가중치:** 같은 seed 42로 새로 만든 레퍼런스와 visionlab 순수 모델의 `TiedAE` 초기 가중치(`weight`, `encoder_bias`, `decoder_bias`)가 서로 같다(`INFO`, 정보 기록).
- **가중치 주입 헬퍼:** 계획의 `load_teacher_weights(self.model, ...)` 대신 DFM 선례와 같이 `self.model.feature_extractor`를 넘긴다(`TimmFeatureExtractor`의 `.feature_extractor`를 헬퍼가 다시 벗기므로 구조에 맞다).

## 4. 적대적 검증

검증은 9-A, 9-B 각각 한 회차씩 했다. 수정 요청은 9-A 0회, 9-B 0회(한도 3회 이내)다.

### 4.1 9-A (순수 모델)

| 지적 | 심각도 | 처리 |
|---|---|---|
| 원본과의 `difflib` 대조: `model.py`는 import 2줄과 D-19의 `pre_trained=False` 한 줄뿐이고 `TiedAE`의 초기화 순서, `get_features`, `forward`가 같음 | 확인 | 수정 없음 |

### 4.2 9-B (Lightning, export, 비교 스크립트)

| 지적 | 심각도 | 처리 |
|---|---|---|
| `ReverseDistillationLitModel`과의 `difflib` 대조: 차이가 생성자 인자, `FREModel`과 `load_teacher_weights`, MSE 손실과 `get_features`, Adam, export 인자, `trainer_arguments`의 `max_epochs` 10뿐임. `batch_size=1` 로그, hook, D-8 처리가 같음. 공유 파일 변경은 등록뿐 | 확인 | 수정 없음 |
| 정적 검토는 `LitModel`, yaml, 등록 코드, `LocalFre` 위주로 했다. `compare_outputs.py`의 fre 분기는 줄 단위로 읽지 않고 실행 결과(5.2절)로 확인했다 | 낮음 | 수정 없음 |

남은 위험은 6.2절에 적었다.

## 5. 이식 결과

### 5.1 최종 판정 (10 epoch, Image·Pixel AUROC 차이 0.01 이내)

native는 `train.py`가 test에서 기록한 값이고 순수 PyTorch는 `export/model.pt`만 불러 `predict.py`로 평가한 값이다. 모두 실행으로 검증했다. D-24에 따라 레퍼런스가 보고하는 값과 같은 정의로 비교하며, 순수 PyTorch의 정규화 점수 AUROC도 함께 적는다.

| 데이터 / category | 지표 | 레퍼런스 | native (raw) | 순수 PyTorch (raw) | 순수 PyTorch (정규화) | 차이(native) | 차이(순수 raw) | 판정 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| mvtec / bottle | Image AUROC | 0.989035070 | 0.989035070 | 0.98903507 | 0.98903507 | 0 | 1e-10 | 통과 |
| mvtec / bottle | Pixel AUROC | 0.974012375 | 0.974012375 | 0.97401237 | 0.97401237 | 0 | 5e-9 | 통과 |
| visa / candle | Image AUROC | 0.925277710 | 0.925277710 | 0.92527771 | 0.92527771 | 0 | 1e-9 | 통과 |
| visa / candle | Pixel AUROC | 0.978129923 | 0.978129923 | 0.97812986 | 0.97812998 | 0 | 6e-8 | 통과 |

raw와 정규화 값이 0.01 넘게 다른 곳은 없어 D-24의 원인 기록 조건에는 해당하지 않는다. image F1과 pixel F1도 레퍼런스, native, 순수 PyTorch가 같다(bottle image 0.97368419, pixel 0.63957196. visa image 0.85714287, pixel 0.18472788).

run 폴더와 로그는 다음과 같다.

| 대상 | 위치 |
|---|---|
| 레퍼런스 | `outputs/anomalib/LocalFre/LocalMVTecAD/bottle/v0`, `outputs/anomalib/LocalFre/LocalVisa/candle/v0` |
| native | `outputs/visionlab/fre/mvtec_bottle/v0`, `outputs/visionlab/fre/visa_candle/v0` (`run.json`: `max_epochs` 10, `num_sanity_val_steps` 0, `gradient_clip_val` 0.0, 가중치 `D:\backbones\resnet50-0676ba61.pth` sha256 `0676ba61...fb8a`) |
| 순수 PyTorch | 같은 run의 `predict/metrics.json` |
| 로그 | `outputs/logs/09-native-{mvtec,visa}-261026-1.log`, `09-pure-{mvtec,visa}-261026-1.log` |

네 native·순수 로그에 `network access is blocked`, `URLError`, `ConnectionError`, `Traceback`, `out of memory`는 0건이다.

### 5.2 테스트와 비교 스크립트

다음 표는 agy가 실행한 묶음의 결과다(Claude가 로그 파일에서 확인). 실행 대상은 FRE뿐이다(9-A 전의 회귀 실행은 위의 별도 절).

| 대상 | 결과 | 로그 |
|---|---|---|
| `tests\09_fre` | 16 passed | `09-pytest-261026-1.log` |
| `tests\common\test_cli.py -k fre` | 1 passed (14 deselected) | `09-pytest-cli-261026-1.log` |
| `tests\common\test_dependency.py` (전체 파일) | 6 passed | `09-pytest-dependency-261026-1.log` |
| `compare_outputs.py --model fre` (bottle 체크포인트, visa 체크포인트) | PASS 23, FAIL 0, INFO 3 (각각). FR-6, FR-23(특징 추출기 `torch.equal`), FR-5(`pred_score [8]`과 `anomaly_map [8,1,256,256]` 최대 차이 0.0, mvtec·visa 입력 8장씩), FR-24. INFO는 seed 42 초기 `TiedAE` 가중치 세 개 일치 | `09-compare-outputs-{mvtec,visa}-261026-1.log` |

`test_dependency.py`는 모든 순수 모듈의 import 분리 검사를 한 파일이 하므로 이 모델의 모듈 검증을 위해 전체를 돌렸다(Phase 5, 6, 7에서 사용자가 사후 승인한 것과 같은 사유). 이번 Phase에서의 실행은 사용자가 사후에 승인했다(2026-10-26).

실행한 명령은 다음과 같다(Claude가 보낸 지시 원문. agy가 시간 측정 줄을 덧붙여 하나의 스크립트로 순서대로 실행했고, 승인 창의 전문을 Claude가 agy의 transcript에서 읽어 지시와 같음을 확인한 뒤 한 번 승인했다).

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py -m pytest tests\09_fre 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\09-pytest-261026-1.log
& $py -m pytest tests\common\test_cli.py -k fre 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\09-pytest-cli-261026-1.log
& $py -m pytest tests\common\test_dependency.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\09-pytest-dependency-261026-1.log
& $py evaluation/visionlab/train.py train --data mvtec --model fre 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\09-native-mvtec-261026-1.log
& $py evaluation/visionlab/predict.py --model fre --data mvtec 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\09-pure-mvtec-261026-1.log
& $py evaluation/visionlab/train.py train --data visa --model fre 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\09-native-visa-261026-1.log
& $py evaluation/visionlab/predict.py --model fre --data visa 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\09-pure-visa-261026-1.log
& $py evaluation/anomalib/compare_outputs.py --model fre --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\LocalFre\LocalMVTecAD\bottle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\09-compare-outputs-mvtec-261026-1.log
& $py evaluation/anomalib/compare_outputs.py --model fre --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\LocalFre\LocalVisa\candle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\09-compare-outputs-visa-261026-1.log
& $py evaluation/anomalib/compare_history.py --reference D:\projects\visionlab\outputs\anomalib\LocalFre\LocalMVTecAD\bottle\v0 --native D:\projects\visionlab\outputs\visionlab\fre\mvtec_bottle\v0 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\09-history-mvtec-261026-1.log
& $py evaluation/anomalib/compare_history.py --reference D:\projects\visionlab\outputs\anomalib\LocalFre\LocalVisa\candle\v0 --native D:\projects\visionlab\outputs\visionlab\fre\visa_candle\v0 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\09-history-visa-261026-1.log
```

agy는 파일을 고치지 않았다.

### 5.3 에폭별 학습 히스토리 비교 (허용 오차 없음, 비교표만)

다음 표는 `compare_history.py`의 `train_loss_epoch` 출력이다. 10행이라 전체를 적는다. 판정에 쓰지 않는다.

mvtec / bottle (레퍼런스 `LocalFre/.../bottle/v0`, native `mvtec_bottle/v0`):

| Epoch | 레퍼런스 | native | 절대 차이 | 상대 차이 |
|---:|---:|---:|---:|---:|
| 0 | 0.00658449484 | 0.006584497169 | 2.3e-9 | 3.5e-7 |
| 1 | 0.001324792393 | 0.001324792276 | 1.2e-10 | 8.8e-8 |
| 2 | 0.0005439125234 | 0.0005439125816 | 5.8e-11 | 1.1e-7 |
| 3 | 0.000480586692 | 0.0004805867793 | 8.7e-11 | 1.8e-7 |
| 4 | 0.0004735756083 | 0.000473575521 | 8.7e-11 | 1.8e-7 |
| 5 | 0.0004692216171 | 0.000469221588 | 2.9e-11 | 6.2e-8 |
| 6 | 0.0004557841166 | 0.0004557840293 | 8.7e-11 | 1.9e-7 |
| 7 | 0.0004495603498 | 0.0004495603498 | 0 | 0 |
| 8 | 0.0004532994353 | 0.0004532994644 | 2.9e-11 | 6.4e-8 |
| 9 | 0.0004565693671 | 0.000456569338 | 2.9e-11 | 6.4e-8 |

visa / candle (레퍼런스 `LocalFre/.../candle/v0`, native `visa_candle/v0`):

| Epoch | 레퍼런스 | native | 절대 차이 | 상대 차이 |
|---:|---:|---:|---:|---:|
| 0 | 0.001929469872 | 0.001929469639 | 2.3e-10 | 1.2e-7 |
| 1 | 0.000533955521 | 0.0005339554045 | 1.2e-10 | 2.2e-7 |
| 2 | 0.0004960348015 | 0.0004960347433 | 5.8e-11 | 1.2e-7 |
| 3 | 0.0004602411354 | 0.0004602411937 | 5.8e-11 | 1.3e-7 |
| 4 | 0.0004487090337 | 0.0004487089755 | 5.8e-11 | 1.3e-7 |
| 5 | 0.0004407657252 | 0.000440765667 | 5.8e-11 | 1.3e-7 |
| 6 | 0.0004272656515 | 0.0004272656515 | 0 | 0 |
| 7 | 0.0004118966172 | 0.0004118965007 | 1.2e-10 | 2.8e-7 |
| 8 | 0.0003980706679 | 0.0003980704641 | 2.0e-10 | 5.1e-7 |
| 9 | 0.000384386658 | 0.0003843866871 | 2.9e-11 | 7.6e-8 |

모든 에폭의 상대 차이가 5.2e-7 이하다. 손실 값이 1e-4~1e-2 수준이라 절대 차이는 2.4e-9 이하다. float32 단위 오차 수준이어서 학습 경로(초기 가중치, 데이터 순서, 손실, optimizer)가 레퍼런스와 같다고 볼 수 있다. 에폭별 허용 오차는 사용자가 아직 정하지 않았다.

### 5.4 crop·입력 크기

이 모델은 crop이 없고 입력 크기는 256x256이다. FR-5의 `anomaly_map [8,1,256,256]` 최대 차이가 0.0이다.

## 6. 결론

- **레퍼런스:** bottle, visa 확보(10 epoch, `LocalFre`). 220 epoch 레퍼런스는 사용자 지시로 중단했다.
- **mvtec / bottle:** Image·Pixel AUROC가 native와 순수 PyTorch 모두 레퍼런스와 5e-9 이내다. 통과.
- **visa / candle:** Image AUROC 일치, Pixel AUROC 차이 native 0, 순수 PyTorch 6e-8. 통과. 정규화 점수 기준(D-24)도 같다.
- **FR-5, FR-6, FR-23, FR-24:** 통과(5.2절). `pred_score`와 `anomaly_map` 최대 차이 0.0이다. 에폭별 `train_loss_epoch`는 상대 차이 5.2e-7 이내(5.3절)이다.
- **에폭:** 레퍼런스와 native 모두 실제 10 epoch임을 `run.json`과 로그로 확인했다(2.2절). 원본의 220 epoch 설정과 달라 "원본이 의도한 성능"은 확인하지 못했다(6.2절).
- **테스트:** `tests\09_fre` 16개, `test_cli.py -k fre` 1개, `test_dependency.py` 6개가 통과했다. 9-A 전의 사용자 승인 회귀 실행 230개도 통과했다.
- **수정 요청:** 9-A 0회, 9-B 0회.
- **`specs/` 변경:** 사용자 결정(2026-10-26)에 따라 `DECISIONS.md` D-22와 `PLAN.md` 9행을 고쳤다(6.1절). INTENT와 SPEC은 건드리지 않았다.
- **Phase 종료:** 사용자가 종료를 결정했다(2026-10-26).

### 6.1 `specs/` 갱신 항목 (반영 완료)

`specs/`는 사람이 쓴다. 이번 Phase의 두 항목은 사용자 결정(2026-10-26)에 따라 Claude가 반영했다. 변경은 이 두 곳뿐이다.

- **`specs/DECISIONS.md` D-22:**
  - 선택에 원본이 `trainer_arguments`로 1이 아닌 에폭을 고정하는 모델(FRE의 220 epoch)은 레퍼런스 어댑터(`LocalFre`)가 그 값을 10으로 바꾸고 native의 `trainer_arguments`도 10으로 둔다는 문장을 추가했다(Phase 9, 사용자 결정 2026-10-26).
  - 적용 범위의 "모델이 10이 아닌 여러 에폭을 고정하는 경우(PLAN의 FRE 220 epoch)는 이 결정에 포함되지 않는다. 해당 Phase 시작 전에 사용자가 정한다" 문장을 "레퍼런스와 native 모두 10으로 통일한다. 원본이 의도한 에폭에서의 성능은 확인하지 않는다"로 교체했다.
  - 근거에 FRE의 220 epoch는 알고리즘이 아니라 학습 예산이므로 예외로 정했다는 점과 시간 비교(220 epoch는 레퍼런스와 native 합계 약 4.5시간 예상, 10 epoch는 약 7분)를 추가했다.
  - 배제한 대안에 FRE의 220 epoch 그대로(처음 정한 뒤 철회)와 임의의 N epoch 절충을 추가했고, "남은 한계" 항목(원본이 의도한 수렴 상태의 성능은 확인하지 않음)을 새로 두었다.
- **`specs/PLAN.md` 9행:** "모델이 220 epoch을 고정해 검증 시간이 길다"를 "원본은 220 epoch을 고정하지만 10 epoch로 통일한다(D-22)"로 고쳤다.

### 6.2 남은 위험과 사용자 판단 대상

- **220 epoch 성능 미확인:** 원본이 고정한 220 epoch가 아니라 10 epoch의 레퍼런스와 비교했다. 이식의 정확성(손실, 가중치 경로, 출력 일치)은 확인했지만 원본이 의도한 수렴 상태의 AUROC는 이번 Phase에서 확인하지 못했다.
- 레퍼런스가 category당 1회뿐이다. 다만 native가 AUROC, 에폭별 손실, FR-5 최대 차이까지 일치해 학습 경로의 편차는 작아 보인다.
- `tests\common\test_dependency.py` 전체 실행은 사용자가 사후에 승인했다(2026-10-26).
- 앞 모델 회귀는 9-A 전에 승인된 1회(230개 통과)만 실행했다. 9-B의 공유 파일 변경 이후의 회귀는 보류했다(사용자 결정 2026-10-26). 실행하지 않았고 변경은 모두 기존 항목을 건드리지 않는 등록과 분기 추가다.

| 공유 파일 | 변경 | 영향받을 수 있는 앞 모델 |
|---|---|---|
| `src/visionlab/utils/io.py` | `load_export`가 허용하는 이름에 `fre` 추가 | 모든 모델의 export 로딩 |
| `src/visionlab/inference/predictor.py` | `from_export` 모델 dict에 `fre` 등록 | 모든 모델의 `Predictor.from_export` |
| `evaluation/visionlab/predict.py` | `--model` 선택지에 추가 | 모든 모델의 순수 PyTorch 평가 |
| `evaluation/anomalib/compare_outputs.py` | `--model fre` 분기 | stfpm, efficient_ad, padim, dfm, dfkde, patchcore, reverse_distillation, cfa 비교 |
| `evaluation/anomalib/local_models.py` | `LocalFre` 추가(다른 어댑터는 그대로) | 같은 파일의 `LocalCfa` 등 어댑터 |
| `tests/common/test_cli.py`, `tests/common/test_dependency.py` | fre 테스트와 분류 추가 | 같은 파일의 앞 모델 테스트 |

- 체크포인트가 약 208MB(optimizer 상태 포함)라 `outputs/`의 디스크 사용은 크지 않다(커밋하지 않는다).
