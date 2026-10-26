# Phase 10. SuperSimpleNet

모델 폴더 이름은 `supersimplenet`이고 원본은 anomalib 2.0.0의 `anomalib\models\image\supersimplenet\`(`C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\Lib\site-packages\anomalib\models\image\supersimplenet\`)이다. 범위와 완료 조건은 `specs/PLAN.md`의 Phase 10과 "Phase 공통 양식"을 따른다. 판정 기준은 category별 Image·Pixel AUROC 차이 0.01 이내다.

## 상태

다음 표는 단계별 진행 상태다. 수정 반복 한도 3회는 단계별로 센다.

| 단계 | 내용 | 상태 |
|---|---|---|
| 회귀 실행 | 사용자 승인 회귀 실행 1회 (`tests\common`, `tests\01_stfpm` ~ `09_fre`) | 완료 (247개 통과) |
| 레퍼런스 | `anomalib_run.py train` x 2 (mvtec/bottle 1회, visa/candle 1회, 10 epoch) | 완료 |
| 10-A | 순수 모델 (`models/supersimplenet/model.py`, `anomaly_generator.py`, `loss.py`, `common/perlin.py`의 `generate_perlin_noise`, `tests/10_supersimplenet`의 순수 부분) | 완료 |
| 10-B | Lightning (`SupersimplenetLitModel`, AdamW와 `MultiStepLR` 학습, configs, export·Predictor 등록, 비교 스크립트) | 완료 |

특징 추출기(`TimmFeatureExtractor`), `GaussianBlur2d`, 처리 구성요소(PreProcessor, PostProcessor, Evaluator)는 앞 Phase의 것을 재사용한다.

## 사용자 결정 (2026-10-26)

다음은 Phase 시작 전에 사용자가 정한 항목이다. 사용자가 계획 보고에 "이 계획대로 진행해 주세요"라고 답해, 계획의 권고를 모두 채택한 것으로 처리했다.

| 항목 | 결정 |
|---|---|
| 학습 에폭 | 원본이 `max_epochs`를 고정하지 않으므로(`trainer_arguments`는 `gradient_clip_val`과 `num_sanity_val_steps`뿐) D-22의 첫 문장대로 공유 yaml의 10 epoch를 레퍼런스와 native에 똑같이 쓴다. 예외는 없다. `MultiStepLR`의 milestone은 `max_epochs`에서 `[8, 9]`로 정해진다(gamma 0.4) |
| 평가 category | PLAN대로 mvtec/bottle과 visa/candle이다 |
| 학습 시간 관문 | 계획 추정(합계 약 25분)의 1.5배(약 38분)를 넘을 것으로 보이면 계속하기 전에 사용자에게 보고한다. category는 스스로 바꾸지 않는다 |
| 공통 Perlin 함수의 위치 | `src/visionlab/anomaly_detection/common/perlin.py`에 anomalib `generate_perlin_noise` 하나만 둔다(D-15가 `common/`을 models 밖의 anomalib 부품 대체로 둔 것에 맞춘다). `PerlinAnomalyGenerator`는 Phase 11 범위다. D-15 트리 목록에 이 파일을 적는 일은 사용자 몫이다 |
| 난수가 어긋날 때의 판정 | PaDiM, PatchCore 선례대로 난수 상태를 주입하지 않고 AUROC 0.01로만 판정한다. 합성 이상 일치 검증과 초기 가중치 비교는 기록한다 |
| 앞 모델 회귀 | 10-A 전에 `tests\common`과 `tests\01_stfpm` ~ `09_fre`를 한 번 실행한다(사용자 승인, 이번 한 번에 한해 예외). 실패가 나오면 스스로 고치지 않고 멈춰 보고한다 |
| FR-5 허용 오차 | `atol=1e-6, rtol=1e-5` (Phase 2~9와 같음). 합성 이상·학습 단계 일치 검증도 같은 허용 오차로 판정한다 |
| 에폭별 `train_loss` 히스토리 | 허용 오차 없이 비교표만 기록한다. 10행이므로 전체 표를 본문에 적는다 |
| `specs/` 변경 | 없다. 필요한 항목은 결과 문서의 "specs 갱신 필요"에 적고 최종 보고에서 묻는다 |
| 실행 범위 | SuperSimpleNet의 명령만 실행한다(위 회귀 1회만 예외) |
| 판정 | Image·Pixel AUROC를 판정한다. 정규화 점수 기준 값(D-24)과 raw 값을 함께 기록한다 |
| Phase 종료 (2026-10-26) | 사용자가 종료를 결정했다. `Phase 10: supersimplenet ...` 커밋으로 푸시한다. 6.1절의 `specs/` 갱신과 `test_dependency.py` 전체 실행의 사후 승인은 이 결정에 포함되지 않아 미결로 남는다 |

## 1. 모델 노트

### 1.1 원본과 생성자

다음 표는 원본 파일과 이식 위치의 대응이다(D-15, D-18).

| 원본 | 이식 위치 | 비고 |
|---|---|---|
| `models/image/supersimplenet/torch_model.py` (`SupersimplenetModel`, `UpscalingFeatureExtractor`, `FeatureAdapter`, `SegmentationDetectionModule`, `AnomalyMapGenerator`, `init_weights`) | `src/visionlab/anomaly_detection/models/supersimplenet/model.py` | import 변경, 특징 추출기는 순수 모델에서 구조만 만드는 방식으로(D-19) |
| `models/image/supersimplenet/anomaly_generator.py` (`AnomalyGenerator`) | `.../models/supersimplenet/anomaly_generator.py` | `generate_perlin_noise` import 경로 변경 |
| `models/image/supersimplenet/loss.py` (`SSNLoss`) | `.../models/supersimplenet/loss.py` | 변경 없음 |
| `data/utils/generators/perlin.py` (`generate_perlin_noise`만) | `src/visionlab/anomaly_detection/common/perlin.py` | 함수 하나만 복사. `PerlinAnomalyGenerator`는 Phase 11 |
| `models/image/supersimplenet/lightning_model.py` (`Supersimplenet`) | `.../models/supersimplenet/lightning_model.py` (`SupersimplenetLitModel`) | native `LightningModule`로 새로 작성 |
| `models/image/supersimplenet/__init__.py` | `.../models/supersimplenet/__init__.py` | `lightning_model`을 import하지 않도록 고침(D-18 예외) |

생성자는 다음과 같다. `SupersimplenetLitModel`은 `Supersimplenet`과 인자 이름·순서·기본값이 같아야 한다(FR-6, D-5).

| 클래스 | 시그니처 |
|---|---|
| `Supersimplenet` | `perlin_threshold=0.2`, `backbone="wide_resnet50_2"`, `layers=["layer2", "layer3"]`, `supervised=False`, `pre_processor=True`, `post_processor=True`, `evaluator=True`, `visualizer=True` (8개) |
| `SupersimplenetModel` | `perlin_threshold=0.2`, `backbone="wide_resnet50_2"`, `layers=["layer2", "layer3"]`, `stop_grad=True` |
| `AnomalyGenerator` | `noise_mean`, `noise_std`, `threshold` |
| `SSNLoss` | `truncation_term=0.5` |

`layers`의 기본값이 변경 가능한 리스트다. D-5에 맞춰 같은 기본값을 쓴다. `supervised`는 `Supersimplenet`에서 `stop_grad`(`not supervised`)와 `gradient_clip_val`(`supervised`이면 1, 아니면 0)로 바뀌어 모델에 전달된다.

### 1.2 학습과 평가 동작

원본의 동작은 다음과 같다.

- **특징:** `UpscalingFeatureExtractor`가 `TimmFeatureExtractor`(`no_grad`, 매 호출 `eval()`)로 `layer2`와 `layer3`를 뽑아 `layer2`의 2배 해상도(256 입력에서 64x64)로 bilinear 보간해 채널 방향으로 합치고(1536채널) `AvgPool2d(3, 1, 1)`로 풀링한다. 생성 시 `get_channels_dim`이 `torch.rand(1, 3, 256, 256)`을 한 번 통과시켜 채널 수를 얻는다(난수를 소비한다).
- **학습 모드 `forward`:** 특징을 `FeatureAdapter`(1x1 합성곱, `xavier_normal_`)에 넣고, 마스크를 특징 해상도로 내려 이진화하고, `AnomalyGenerator`가 batch를 2배로 늘려 Perlin 마스크 영역에 가우시안 노이즈(평균 0, 표준편차 0.015)를 더해 합성 이상과 확장된 마스크·라벨을 만든다. `SegmentationDetectionModule`이 `anomaly_map`과 `anomaly_score`를 내고 `(anomaly_map, anomaly_score, masks, labels)`를 반환한다.
- **합성 이상(Perlin):** `generate_perlin_noise`(`device` 기본값 `cuda if available`)로 2의 거듭제곱 크기의 Perlin 잡음을 만들어 batch 크기로 보간하고 임계값(`perlin_threshold` 0.2)으로 이진화한다. 각 샘플은 `torch.rand(1) > 0.5`이면 0 마스크로 바뀐다. 정상 이미지의 마스크와 곱해 노이즈 영역을 정하고 `labels`를 갱신한다.
- **손실:** `SSNLoss` = map의 focal loss(`alpha=-1`, `gamma=4`) + map의 truncated L1(`truncation_term` 0.5) + score의 focal loss.
- **평가 모드 `forward`:** 합성 없이 `adapted`를 `segdec`에 넣고, `anomaly_map`을 입력 크기로 보간해 `GaussianBlur2d`(sigma 4)를 씌운다. `pred_score`는 `anomaly_score`(logit, `squeeze()`)다.
- **optimizer:** `AdamW`(adaptor lr 1e-4, segdec lr 2e-4와 weight_decay 1e-5, adaptor 그룹은 기본 weight_decay 0.01)와 `MultiStepLR`(milestone이 `trainer.max_epochs`의 0.8, 0.9배, gamma 0.4).
- **`trainer_arguments`:** `{"gradient_clip_val": <0 또는 1>, "num_sanity_val_steps": 0}`이며 `max_epochs`는 고정하지 않는다(공유 yaml의 10 epoch).

### 1.3 외부 의존과 로컬 자산

- **외부 의존:** timm, torchvision(`ops.focal_loss.sigmoid_focal_loss`), kornia(`GaussianBlur2d`, D-23)다. scipy, sklearn, einops는 쓰지 않는다. anomalib 의존은 `Batch`, `InferenceBatch`, `AnomalibModule`, `TimmFeatureExtractor`, `GaussianBlur2d`, `generate_perlin_noise`, `PreProcessor`, `PostProcessor`, `Evaluator`, `Visualizer`, `LearningType`이다.
- **로컬 가중치:** `paths.toml [backbones]`의 `wide_resnet50_2`(이미 있다). `Supersimplenet`에는 `pre_trained` 인자가 없고 항상 사전학습 backbone을 쓴다. 원본의 `TimmFeatureExtractor`는 `pre_trained=True`가 기본값이다. visionlab은 순수 모델이 `False`로 만들고 `SupersimplenetLitModel.__init__`이 항상 로컬 가중치를 주입한다(D-19).
- **데이터:** MVTec `bottle`(학습 209장), VisA `candle`(학습 900장). 분할은 `from_test`, 0.4, seed 42, batch 8/8, `num_workers=0`이다. 학습 batch에는 `Dataset`이 정상 이미지에 0 마스크(`int64 [256, 256]`)와 라벨 0을 준다.
- **학습 예산:** 공유 yaml의 `max_epochs=10`이다.

### 1.4 이식에서 의도적으로 달라지는 곳과 유지하는 곳

- **pretrained 주입(D-19):** 순수 `SupersimplenetModel`은 특징 추출기를 `pre_trained=False`로 만든다. `LitModel`이 `wide_resnet50_2` 로컬 파일을 불러온다.
- **Perlin 함수의 위치:** `generate_perlin_noise`를 `common/perlin.py`로 옮긴다. 알고리즘과 `device` 기본값은 그대로 둔다.
- **원본 동작 유지(고치지 않음):** `get_channels_dim`의 `torch.rand` 호출(생성 순서), `AnomalyGenerator`의 난수 소비 순서와 `torch.rand(1).item() > 0.5`, `torch.normal`, `squeeze()`, `stop_grad` 분기, `init_weights`, AdamW 그룹별 설정, `trunc_l1_loss`가 빈 입력에서 `torch.tensor(0)`을 반환하는 것은 고치지 않는다.
- **`train_loss` 로그의 집계:** 원본은 batch 크기를 추론하지 못해 epoch 평균이 step마다 1회씩 센 평균이 된다. visionlab은 `batch_size=1`을 두어 맞춘다(Phase 7 선례).
- **히스토리 비교:** 레퍼런스와 native의 에폭별 `train_loss`를 `compare_history.py`로 표로 기록한다(판정 아님).
- **FR-5 허용 오차:** `pred_score`와 `anomaly_map`은 `atol=1e-6, rtol=1e-5`, backbone 가중치는 `torch.equal`이다. 합성 이상·학습 단계 일치 검증도 같다.

## 2. 레퍼런스

학습 예산은 공유 yaml의 10 epoch다. 원본이 `max_epochs`를 고정하지 않으므로 D-22의 첫 문장대로 레퍼런스와 native가 같은 값을 쓴다(1.2절, 사용자 결정).

### 2.1 10 epoch 레퍼런스

2026-10-26에 agy가 `anomalib_run.py train --config default.yaml --config data/<data>.yaml --config models/10_supersimplenet.yaml`을 데이터셋마다 1회 실행했다(실행으로 검증함). Claude는 로그 파일에서 값을 직접 확인했다. 모든 레퍼런스·native·순수 로그에서 `network access is blocked`, `URLError`, `ConnectionError`, `Traceback`, `out of memory`는 0건이다. 로그는 UTF-16이다. 승인 창의 스크립트 전문은 Claude가 agy의 transcript에서 읽어 지시한 명령과 같음을 확인했다.

agy가 PowerShell에서 실행한 명령은 다음과 같다(Claude가 보낸 지시 원문. 파일은 고치지 않았다).

```powershell
cd D:\projects\visionlab\evaluation\anomalib
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/10_supersimplenet.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\10-ref-mvtec-261026-1.log
& $py anomalib_run.py train --config default.yaml --config data/visa.yaml --config models/10_supersimplenet.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\10-ref-visa-261026-1.log
```

다음 표는 10 epoch 레퍼런스(1회) 결과다.

| 데이터 / category | Image AUROC | Pixel AUROC | 참고 | run 폴더 | 로그 |
|---|---|---|---|---|---|
| mvtec / bottle | 0.71272 | 0.97418 | image F1 0.8095, pixel F1 0.6707 | `outputs/anomalib/Supersimplenet/LocalMVTecAD/bottle/v0` | `outputs/logs/10-ref-mvtec-261026-1.log` |
| visa / candle | 0.96278 | 0.98927 | image F1 0.8772, pixel F1 0.2863 | `outputs/anomalib/Supersimplenet/LocalVisa/candle/v0` | `outputs/logs/10-ref-visa-261026-1.log` |

적용된 학습 예산은 `max_epochs=10`, `val_split_mode=from_test`, `val_split_ratio=0.4`, `seed=42`, `batch 8/8`, `num_workers=0`, `perlin_threshold=0.2`, `backbone=wide_resnet50_2`, `layers=[layer2, layer3]`, `supervised=false`이다. 원본의 `trainer_arguments`가 `gradient_clip_val`을 덮어쓰는 `Overriding gradient_clip_val` 줄이 로그에 있다(`supervised=false`이므로 값은 0). 체크포인트는 각 run 폴더의 `weights/lightning/model.ckpt`이고 크기는 206,039,598 bytes다.

mvtec/bottle의 Image AUROC 0.7127은 낮다. 학습 이미지가 209장이라 10 epoch가 step 270뿐이고, 학습 중 검증 Image AUROC도 0.66~0.83에서 흔들렸다(5.3절). 이 값은 10 epoch 예산에서 레퍼런스가 낸 값이며, 원본이 의도한 학습량에서의 성능은 확인하지 않았다(6.2절).

### 2.2 레퍼런스와 native의 에폭 확인

사용자 지시에 따라 레퍼런스와 native가 같은 10 epoch인지 실제 값으로 확인했다.

| 대상 | 근거 | 값 |
|---|---|---|
| 레퍼런스 (bottle, visa) | 로그의 `Trainer.fit stopped: max_epochs=10 reached`, 완료한 가장 큰 에폭 번호 9 | 10 |
| native (bottle, visa) | `run.json`의 `config.trainer.max_epochs` 10, 로그의 `Trainer.fit stopped: max_epochs=10 reached`, 완료한 가장 큰 에폭 번호 9 | 10 |

### 2.3 학습 시간 관문과 실제 학습 시간

계획의 합계 추정은 약 25분이고 1.5배(약 38분)를 넘으면 보고하기로 했다. 실제로는 agy가 보고한 학습 시간이 bottle 약 175초, visa 약 613초였고 레퍼런스 두 run의 합계가 약 15분이었다. 추정 이내라 보고 없이 진행했다. visa는 에폭당 약 1분(113 batch, 약 2 it/s)이다.

## 3. 이식 메모

### 3.1 원본과 이식 파일의 대응

다음 표는 새로 만든 파일과 원본의 대응이다. 원본과의 차이는 `difflib`로 확인했다(구현함, 정적 검증).

| 이식 파일 | 원본 | 원본과 다른 줄 |
|---|---|---|
| `models/supersimplenet/model.py` | `torch_model.py` | import 3줄(`InferenceBatch`, `GaussianBlur2d`, `TimmFeatureExtractor`)과 `AnomalyGenerator` import 경로, `TimmFeatureExtractor(... pre_trained=False)`와 주석 한 줄(D-19) |
| `models/supersimplenet/anomaly_generator.py` | `anomaly_generator.py` | `generate_perlin_noise` import 한 줄 |
| `models/supersimplenet/loss.py` | `loss.py` | 없음 |
| `common/perlin.py` | `data/utils/generators/perlin.py` | `generate_perlin_noise` 함수 하나만 복사. `PerlinAnomalyGenerator`와 그 import(`Path`, `io`, `IMG_EXTENSIONS`, `v2`, `MultiRandomChoice`)는 제외(Phase 11 범위) |
| `models/supersimplenet/__init__.py` | `__init__.py` | `SupersimplenetModel`만 노출(D-18 예외) |
| `models/supersimplenet/lightning_model.py` (`SupersimplenetLitModel`) | `lightning_model.py` | native `LightningModule`로 새로 작성. 8개 인자 동일, AdamW 두 그룹과 `MultiStepLR`, `train_loss` 로그(`batch_size=1`), 로컬 backbone 항상 주입(D-19), D-8의 post_processor 갱신 제한, `trainer_arguments`에 `max_epochs` 없음 |

### 3.2 새로 쓴 코드와 앞 Phase 공유 파일 변경

- **새로 쓴 파일:** `configs/models/supersimplenet.yaml`, `common/perlin.py`, `tests/10_supersimplenet/test_supersimplenet.py`, `tests/10_supersimplenet/test_supersimplenet_lightning.py`. `tests\10_supersimplenet`은 합계 27개 테스트다(순수 19개, Lightning 8개).
- **공유 파일 변경:** `utils/io.py`(`load_export`가 `supersimplenet` 허용), `inference/predictor.py`(모델 dict에 등록), `evaluation/visionlab/predict.py`(`--model supersimplenet`), `evaluation/anomalib/compare_outputs.py`(`--model supersimplenet` 분기와 검증 항목), `tests/common/test_cli.py`, `tests/common/test_dependency.py`. `common/` 아래 기존 파일은 바꾸지 않았다. 영향 범위는 6.2절의 표에 정리했다.
- **새 부품:** 대체 부품은 없다. `SupersimplenetModel`, `AnomalyGenerator`, `SSNLoss`, `generate_perlin_noise`는 원본 복사다.
- **레퍼런스 어댑터 변경:** 없다. 원본 `Supersimplenet`을 그대로 쓴다.

### 3.3 이식 중 확인된 사항

- **난수 소비 순서:** 원본의 `get_channels_dim`의 `torch.rand`, Perlin의 CUDA 난수, `torch.rand(1).item() > 0.5`를 고치지 않고 그대로 복사했다. 계획 단계에서 우려한 난수 어긋남은 일어나지 않았다. 같은 seed에서 `generate_perlin_noise`(CPU, CUDA)와 `AnomalyGenerator`의 출력이 레퍼런스와 최대 차이 0.0으로 일치했고(5.2절), 학습 forward와 `SSNLoss`도 일치했다.
- **`train_loss`의 집계:** `self.log(..., batch_size=1)`을 처음부터 두었고, native의 에폭별 손실이 레퍼런스와 상대 차이 약 7e-7 이내로 일치했다(5.3절).
- **초기 가중치:** 같은 seed 42로 새로 만든 레퍼런스와 visionlab 순수 모델의 `adaptor`, `segdec` 초기 가중치가 서로 같다(`INFO`, 정보 기록). 이 때문에 난수 주입 없이 학습 경로가 같다.
- **batch 1의 `pred_score`:** 평가 모드에서 원본의 `squeeze()`로 batch 1이면 0차원이 된다. 원본 동작을 고치지 않고 두었고, 이미지 저장 callback에서만 `_SupersimplenetLitVisualizer`가 축을 복원한다(CFA 선례와 같은 방식).
- **가중치 주입 헬퍼:** `load_teacher_weights`에 `self.model.feature_extractor.feature_extractor`(내부 `TimmFeatureExtractor`)를 넘긴다. 헬퍼가 첫 인자의 `.feature_extractor`를 벗기므로 `UpscalingFeatureExtractor`가 한 겹 더 있는 구조에 맞춘 것이다.
- **학습 batch의 마스크:** 학습 batch는 `Dataset`이 정상 이미지에 준 0 마스크(`int64 [B,256,256]`)와 라벨 0을 `training_step`이 `target["mask"]`, `target["label"]`로 읽는다. 모델이 batch를 2배(16)로 늘려 합성 이상을 만든다.

## 4. 적대적 검증

검증은 10-A, 10-B 각각 한 회차씩 했다. 수정 요청은 10-A 0회, 10-B 0회(한도 3회 이내)다.

### 4.1 10-A (순수 모델)

| 지적 | 심각도 | 처리 |
|---|---|---|
| 원본과의 `difflib` 대조: `model.py`는 import 4줄과 D-19의 `pre_trained=False` 한 줄뿐이고 생성 순서, `forward`, `squeeze()`가 같음. `anomaly_generator.py`는 import 한 줄, `loss.py`는 차이 없음. `perlin.py`는 `generate_perlin_noise`만 남고 함수 본문 차이 없음 | 확인 | 수정 없음 |
| `test_dependency.py`의 분류 변경은 순수 모듈 세 파일을 `pure` 그룹에 추가한 2줄뿐이고 sklearn 허용 검사는 그대로임 | 확인 | 수정 없음 |

### 4.2 10-B (Lightning, export, 비교 스크립트)

| 지적 | 심각도 | 처리 |
|---|---|---|
| 원본 `lightning_model.py`와의 대조: AdamW 두 그룹의 lr·weight_decay, `MultiStepLR` milestone과 gamma, `supervised` 분기, `SSNLoss` 호출 인자, `train_loss` 로그가 같음. `CfaLitModel`과 비교해 batch 1 스칼라 처리와 hook, D-8 처리가 같은 방식임. 공유 파일 변경은 등록과 허용 목록 추가뿐 | 확인 | 수정 없음 |
| `compare_outputs.py`의 새 분기는 `git diff`로 읽었고 중간 약 36줄은 전문을 읽지 않았다. 실행 결과(5.2절)로 확인했다 | 낮음 | 수정 없음 |
| `tests/10_supersimplenet/test_supersimplenet_lightning.py`는 테스트 이름과 구조만 확인했고 본문은 줄 단위로 읽지 않았다. 통과 여부는 실행 결과(5.2절)로 판정했다 | 낮음 | 수정 없음 |

남은 위험은 6.2절에 적었다.

## 5. 이식 결과

### 5.1 최종 판정 (10 epoch, Image·Pixel AUROC 차이 0.01 이내)

native는 `train.py`가 test에서 기록한 값이고 순수 PyTorch는 `export/model.pt`만 불러 `predict.py`로 평가한 값이다. 모두 실행으로 검증했다. D-24에 따라 레퍼런스가 보고하는 값과 같은 정의로 비교하며, 순수 PyTorch의 정규화 점수 AUROC도 함께 적는다.

| 데이터 / category | 지표 | 레퍼런스 | native (raw) | 순수 PyTorch (raw) | 순수 PyTorch (정규화) | 차이(native) | 차이(순수 raw) | 판정 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| mvtec / bottle | Image AUROC | 0.712719321 | 0.712719262 | 0.71271926 | 0.71271932 | 6e-8 | 6e-8 | 통과 |
| mvtec / bottle | Pixel AUROC | 0.974180818 | 0.974180877 | 0.97418088 | 0.97418082 | 6e-8 | 6e-8 | 통과 |
| visa / candle | Image AUROC | 0.962777793 | 0.962777793 | 0.96277779 | 0.96277779 | 0 | 0 | 통과 |
| visa / candle | Pixel AUROC | 0.989270449 | 0.989270329 | 0.98927033 | 0.98927045 | 1.2e-7 | 1.2e-7 | 통과 |

raw와 정규화 값이 0.01 넘게 다른 곳은 없어 D-24의 원인 기록 조건에는 해당하지 않는다. image F1과 pixel F1도 레퍼런스, native, 순수 PyTorch가 같다(bottle image 0.80952382, pixel 0.67069179. visa image 0.87719297, pixel 0.28630316).

run 폴더와 로그는 다음과 같다.

| 대상 | 위치 |
|---|---|
| 레퍼런스 | `outputs/anomalib/Supersimplenet/LocalMVTecAD/bottle/v0`, `outputs/anomalib/Supersimplenet/LocalVisa/candle/v0` |
| native | `outputs/visionlab/supersimplenet/mvtec_bottle/v0`, `outputs/visionlab/supersimplenet/visa_candle/v0` (`run.json`: `max_epochs` 10, `num_sanity_val_steps` 0, `gradient_clip_val` 0.0, seed 42, 가중치 `D:\backbones\wide_resnet50_2-95faca4d.pth` sha256 `95faca4d...50b8`, 학습 이미지 bottle 209장, candle 900장) |
| 순수 PyTorch | 같은 run의 `predict/metrics.json`, `export/model.pt`(135,130,810 bytes) |
| 로그 | `outputs/logs/10-native-{mvtec,visa}-261026-1.log`, `10-pure-{mvtec,visa}-261026-1.log` |

### 5.2 테스트와 비교 스크립트

다음 표는 agy가 실행한 묶음의 결과다(Claude가 로그 파일에서 확인). 실행 대상은 SuperSimpleNet뿐이다(10-A 전의 회귀 실행은 위의 별도 절).

| 대상 | 결과 | 로그 |
|---|---|---|
| `tests\10_supersimplenet\test_supersimplenet.py` (10-A 직후 단독 실행) | 19 passed | `10-test-pure-261026-1.log` |
| `tests\10_supersimplenet` | 27 passed | `10-pytest-261026-1.log` |
| `tests\common\test_cli.py -k supersimplenet` | 2 passed (15 deselected) | `10-pytest-cli-261026-1.log` |
| `tests\common\test_dependency.py` (전체 파일) | 6 passed | `10-pytest-dependency-261026-1.log` |
| `compare_outputs.py --model supersimplenet` (bottle 체크포인트, visa 체크포인트) | PASS 40, FAIL 0, INFO 2 (각각). 아래 항목 참조 | `10-compare-outputs-{mvtec,visa}-261026-1.log` |

`compare_outputs`의 PASS 항목은 다음과 같다. 최대 차이는 모두 0.0이다.

- **FR-6, FR-23:** 생성자 시그니처 일치, `wide_resnet50_2` 특징 추출기 state_dict가 레퍼런스와 `torch.equal`(sha256 `95faca4d...50b8`).
- **난수 일치:** `generate_perlin_noise`가 CPU와 CUDA에서 seed 42로 `torch.equal`. `AnomalyGenerator`의 `perturbed`, `masks`, `labels`가 CUDA에서 일치.
- **FR-5 학습 forward:** 레퍼런스 체크포인트 상태에서 mvtec·visa 실제 샘플 8장(256x256)의 `anomaly_map [16,1,64,64]`, `anomaly_score [16]`, `masks [16,1,64,64]`, `labels [16]`, `SSNLoss`가 일치.
- **FR-5 평가 forward:** `pred_score [8]`, `anomaly_map [8,1,256,256]`이 일치.
- **FR-24와 PostProcessor 대조:** 모델과 무관한 기존 항목이 모두 통과.
- **INFO 2개:** seed 42 초기 `adaptor`, `segdec` 가중치가 서로 같다.

`test_dependency.py`는 모든 순수 모듈의 import 분리 검사를 한 파일이 하므로 이 모델의 모듈 검증을 위해 전체를 돌렸다(Phase 5, 6, 7, 9에서 사용자가 사후 승인한 것과 같은 사유). 이번 Phase에서의 실행은 사용자 판단 대상으로 보고한다(6.2절).

실행한 명령은 다음과 같다(Claude가 보낸 지시 원문. agy가 하나의 스크립트로 순서대로 실행했고, 승인 창의 전문을 Claude가 agy의 transcript에서 읽어 지시한 13줄과 모두 같음을 확인한 뒤 한 번 승인했다. 10-A 직후의 단독 pytest 1회는 승인 창 없이 실행되었고 같은 방식으로 로그에서 확인했다).

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py -m pytest tests\10_supersimplenet\test_supersimplenet.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\10-test-pure-261026-1.log
& $py -m pytest tests\10_supersimplenet 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\10-pytest-261026-1.log
& $py -m pytest tests\common\test_cli.py -k supersimplenet 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\10-pytest-cli-261026-1.log
& $py -m pytest tests\common\test_dependency.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\10-pytest-dependency-261026-1.log
& $py evaluation/visionlab/train.py train --data mvtec --model supersimplenet 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\10-native-mvtec-261026-1.log
& $py evaluation/visionlab/predict.py --model supersimplenet --data mvtec 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\10-pure-mvtec-261026-1.log
& $py evaluation/visionlab/train.py train --data visa --model supersimplenet 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\10-native-visa-261026-1.log
& $py evaluation/visionlab/predict.py --model supersimplenet --data visa 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\10-pure-visa-261026-1.log
& $py evaluation/anomalib/compare_outputs.py --model supersimplenet --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\Supersimplenet\LocalMVTecAD\bottle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\10-compare-outputs-mvtec-261026-1.log
& $py evaluation/anomalib/compare_outputs.py --model supersimplenet --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\Supersimplenet\LocalVisa\candle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\10-compare-outputs-visa-261026-1.log
& $py evaluation/anomalib/compare_history.py --reference D:\projects\visionlab\outputs\anomalib\Supersimplenet\LocalMVTecAD\bottle\v0 --native D:\projects\visionlab\outputs\visionlab\supersimplenet\mvtec_bottle\v0 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\10-history-mvtec-261026-1.log
& $py evaluation/anomalib/compare_history.py --reference D:\projects\visionlab\outputs\anomalib\Supersimplenet\LocalVisa\candle\v0 --native D:\projects\visionlab\outputs\visionlab\supersimplenet\visa_candle\v0 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\10-history-visa-261026-1.log
```

agy는 파일을 고치지 않았다. agy가 보고한 종료 코드는 11개 모두 0이다. native 로그에 PowerShell의 `NativeCommandError` 줄이 있는 것은 LightningCLI가 stderr로 낸 경고 메시지를 PowerShell이 감싼 것이고, 학습은 끝까지 진행되어 test 지표가 기록되었다.

### 5.3 에폭별 학습 히스토리 비교 (허용 오차 없음, 비교표만)

다음 표는 `compare_history.py`의 `train_loss_epoch` 출력이다. 10행이라 전체를 적는다. 판정에 쓰지 않는다.

mvtec / bottle (레퍼런스 `Supersimplenet/.../bottle/v0`, native `mvtec_bottle/v0`):

| Epoch | 레퍼런스 | native | 절대 차이 | 상대 차이 |
|---:|---:|---:|---:|---:|
| 0 | 1.582725406 | 1.582725406 | 0 | 0 |
| 1 | 1.095002055 | 1.095002055 | 0 | 0 |
| 2 | 1.047276378 | 1.047276258 | 1.192092896e-07 | 1.138279179e-07 |
| 3 | 1.045373678 | 1.045373678 | 0 | 0 |
| 4 | 1.119049311 | 1.119049072 | 2.384185791e-07 | 2.130545784e-07 |
| 5 | 1.05001545 | 1.05001545 | 0 | 0 |
| 6 | 1.022612572 | 1.022612572 | 0 | 0 |
| 7 | 1.001114368 | 1.001114488 | 1.192092896e-07 | 1.190765944e-07 |
| 8 | 0.9770019054 | 0.9770019054 | 0 | 0 |
| 9 | 0.9357683063 | 0.9357684255 | 1.192092896e-07 | 1.273918862e-07 |

visa / candle (레퍼런스 `Supersimplenet/.../candle/v0`, native `visa_candle/v0`):

| Epoch | 레퍼런스 | native | 절대 차이 | 상대 차이 |
|---:|---:|---:|---:|---:|
| 0 | 1.151362658 | 1.151362896 | 2.384185791e-07 | 2.070751362e-07 |
| 1 | 0.9489269853 | 0.9489268064 | 1.788139343e-07 | 1.884380327e-07 |
| 2 | 0.8162215948 | 0.8162217736 | 1.788139343e-07 | 2.190752309e-07 |
| 3 | 0.6164801121 | 0.6164796948 | 4.172325134e-07 | 6.767980106e-07 |
| 4 | 0.4131273031 | 0.4131273925 | 8.940696716e-08 | 2.164150529e-07 |
| 5 | 0.2269075364 | 0.2269075215 | 1.490116119e-08 | 6.567063144e-08 |
| 6 | 0.1087277383 | 0.1087277532 | 1.490116119e-08 | 1.370502269e-07 |
| 7 | 0.06893075258 | 0.06893078983 | 3.725290298e-08 | 5.404395222e-07 |
| 8 | 0.0369383432 | 0.03693835065 | 7.450580597e-09 | 2.017031613e-07 |
| 9 | 0.02829577588 | 0.02829577588 | 0 | 0 |

모든 에폭의 상대 차이가 6.8e-7 이하다. float32 단위 오차 수준이어서 학습 경로(초기 가중치, 데이터 순서, 합성 이상의 난수, 손실, optimizer)가 레퍼런스와 같다고 볼 수 있다. 에폭별 허용 오차는 사용자가 아직 정하지 않았다.

참고로 native의 학습 중 `val_image_AUROC`는 bottle에서 0.665~0.83, candle에서 0.49~0.97 사이로 크게 변했다. 최종 평가(test)와 별도 split(`from_test`, 0.4)의 값이며 판정에 쓰지 않는다.

### 5.4 crop·입력 크기

이 모델은 crop이 없고 입력 크기는 256x256이다. 특징 지도는 64x64(layer2의 2배 해상도)다. FR-5의 `anomaly_map [8,1,256,256]` 최대 차이가 0.0이다.

## 6. 결론

- **레퍼런스:** bottle, visa 확보(10 epoch, 원본 `Supersimplenet`).
- **mvtec / bottle:** Image·Pixel AUROC가 native와 순수 PyTorch 모두 레퍼런스와 6e-8 이내다. 통과.
- **visa / candle:** Image AUROC 차이 0, Pixel AUROC 차이 1.2e-7 이내. 통과. 정규화 점수 기준(D-24)도 같다.
- **FR-5, FR-6, FR-23, FR-24:** 통과(5.2절). 학습 forward, `SSNLoss`, 평가 forward의 최대 차이가 0.0이고 `generate_perlin_noise`와 `AnomalyGenerator`도 CPU·CUDA 난수까지 일치한다. 에폭별 `train_loss_epoch`는 상대 차이 6.8e-7 이내(5.3절)이다.
- **에폭:** 레퍼런스와 native 모두 실제 10 epoch임을 `run.json`과 로그로 확인했다(2.2절).
- **테스트:** `tests\10_supersimplenet` 27개, `test_cli.py -k supersimplenet` 2개, `test_dependency.py` 6개가 통과했다. 10-A 전의 사용자 승인 회귀 실행 247개도 통과했다.
- **수정 요청:** 10-A 0회, 10-B 0회.
- **`specs/` 변경:** 없다. 갱신 후보는 6.1절에 적었다.
- **Phase 종료:** 사용자가 종료를 결정했다(2026-10-26).

### 6.1 `specs/` 갱신이 필요한 항목

`specs/`는 사람이 쓴다. Claude는 고치지 않고 다음을 알린다. 최종 보고에서 반영 여부를 묻는다.

- **D-15 트리 목록:** `src/visionlab/anomaly_detection/common/perlin.py`(공통 Perlin 생성 함수)와 `models/supersimplenet/`가 목록에 없다. 이 파일을 적는 일은 사용자 몫이다(사용자 결정).
- **FR-4의 batch 1 예외 문구:** 현재 FR-4에는 image-only와 CFA squeeze 예외가 있다. SuperSimpleNet도 평가 모드 `pred_score`가 batch 1이면 0차원이 되는 원본 동작을 그대로 두었으므로(CFA와 같은 성격) 예외 문구에 포함할지 사용자가 정한다. 코드 동작은 이미 이 방식이다.

### 6.2 남은 위험과 사용자 판단 대상

- **10 epoch의 낮은 성능:** mvtec/bottle의 Image AUROC가 0.7127로 낮다. 레퍼런스가 같은 값을 냈으므로 이식 오류는 아니지만, 원본이 의도한 학습량에서의 성능은 확인하지 않았다. 사용자가 정한 공유 10 epoch 예산의 결과다.
- 레퍼런스가 category당 1회뿐이다. native와 순수 PyTorch가 AUROC, 에폭별 손실, 학습 forward 최대 차이까지 일치해 학습 경로의 편차는 작아 보인다.
- 일치가 매우 정확한 것은 seed 42와 난수 소비 순서가 같고 CUDA에서 같은 연산이 쓰였기 때문이다. 다른 GPU나 드라이버에서도 같은 정도로 일치한다고 보장하지 않는다. 판정 기준은 AUROC 0.01이다.
- `tests\common\test_dependency.py` 전체 실행은 사용자 판단 대상이다(Phase 5, 6, 7, 9에서는 사후 승인).
- 앞 모델 회귀는 10-A 전에 승인된 1회(247개 통과)만 실행했다. 10-B의 공유 파일 변경 이후의 회귀는 실행하지 않았다. 변경은 모두 기존 항목을 건드리지 않는 등록과 분기 추가다.

| 공유 파일 | 변경 | 영향받을 수 있는 앞 모델 |
|---|---|---|
| `src/visionlab/utils/io.py` | `load_export`가 허용하는 이름에 `supersimplenet` 추가 | 모든 모델의 export 로딩 |
| `src/visionlab/inference/predictor.py` | `from_export` 모델 dict에 `supersimplenet` 등록 | 모든 모델의 `Predictor.from_export` |
| `evaluation/visionlab/predict.py` | `--model` 선택지에 추가 | 모든 모델의 순수 PyTorch 평가 |
| `evaluation/anomalib/compare_outputs.py` | `--model supersimplenet` 분기와 검증 함수 | stfpm, efficient_ad, padim, dfm, dfkde, patchcore, reverse_distillation, cfa, fre 비교 |
| `tests/common/test_cli.py`, `tests/common/test_dependency.py` | supersimplenet 테스트와 분류 추가 | 같은 파일의 앞 모델 테스트 |

- 체크포인트가 약 206MB(optimizer 상태 포함)라 `outputs/`의 디스크 사용은 크지 않다(커밋하지 않는다).
