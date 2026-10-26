# Phase 7. Reverse Distillation

모델 폴더 이름은 `reverse_distillation`이고 원본은 anomalib 2.0.0의 `anomalib\models\image\reverse_distillation\`(`C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\Lib\site-packages\anomalib\models\image\reverse_distillation\`)이다. 범위와 완료 조건은 `specs/PLAN.md`의 Phase 7과 "Phase 공통 양식"을 따른다. 판정 기준은 category별 Image·Pixel AUROC 차이 0.01 이내다.

## 상태

다음 표는 단계별 진행 상태다. 수정 반복 한도 3회는 단계별로 센다.

| 단계 | 내용 | 상태 |
|---|---|---|
| 회귀 실행 | 사용자 승인 회귀 실행 1회 (`tests\common`, `tests\01_stfpm` ~ `05_dfkde`) | 실행 완료 (아래 표) |
| 레퍼런스 | `anomalib_run.py train` x 2 (mvtec/bottle 1회, visa/candle 1회, 10 epoch) | 실행 완료 (2절) |
| 7-A | 순수 모델 (`models/reverse_distillation/model.py`, `anomaly_map.py`, `loss.py`, `components/`, `tests/07_reverse_distillation`의 순수 부분) | 구현, 검증, 실행 완료 (수정 요청 0회) |
| 7-B | Lightning (`ReverseDistillationLitModel`, Adam 학습, configs, export·Predictor 등록, 비교 스크립트) | 구현, 검증, 실행 완료 (수정 요청 1회, `train_loss` 로그의 `batch_size=1`, 결과 5절) |

처리 구성요소(PreProcessor, PostProcessor, Evaluator), 특징 추출기(`TimmFeatureExtractor`), `GaussianBlur2d`는 앞 Phase의 것을 재사용한다.

## 사용자 결정 (2026-10-26)

다음은 Phase 시작 전에 사용자가 정한 항목이다.

| 항목 | 결정 |
|---|---|
| 평가 category | PLAN대로 mvtec/bottle과 visa/candle이다. capsule 대체는 PatchCore(Phase 6)의 예외이며 이번에는 쓰지 않는다. `specs/PLAN.md`는 고치지 않는다 |
| 학습 에폭 | 원본이 `max_epochs`를 고정하지 않으므로 공유 yaml의 10 epoch를 레퍼런스와 native에 똑같이 쓴다(D-22) |
| FR-5 허용 오차 | `atol=1e-6, rtol=1e-5` (Phase 2~6과 같음) |
| 에폭별 `train_loss` 히스토리 | 허용 오차 없이 비교표만 기록한다. 판정은 최종 AUROC로만 한다 |
| 실행 범위 | Reverse Distillation의 명령만 실행한다. 7-A 전의 회귀 실행 1회만 이번 한 번에 한해 예외로 승인되었고 방침을 바꾸는 것은 아니다 |
| 학습 시간 관문 | visa 레퍼런스의 첫 1~2 에폭 속도로 전체 예상이 약 2시간을 넘으면 계속하기 전에 사용자에게 보고한다. category는 스스로 바꾸지 않는다 |
| 판정 | Image·Pixel AUROC를 판정한다. 정규화 점수 기준 값(D-24)과 raw 값을 함께 기록한다 |
| `test_dependency.py` 전체 실행 (사후, 2026-10-26) | 사용자가 승인했다 |
| 7-B 이후 앞 모델 회귀 (2026-10-26) | 보류한다(사용자 결정). 이번에는 돌리지 않고 영향받을 수 있는 공유 파일은 6.2절의 표에 남긴다 |
| 에폭별 히스토리 허용 오차 (2026-10-26) | 미정을 유지한다. 이번 값(상대 차이 6e-7 이내)은 참고로만 남기고 허용 오차를 정하지 않는다 |
| `specs/` 반영 (2026-10-26) | 새로 반영할 항목이 없으므로 고치지 않는다 |
| Phase 종료 (2026-10-26) | 사용자가 종료를 결정했다. `Phase 07: reverse_distillation ...` 커밋으로 푸시한다 |

## 사용자 승인 회귀 실행 (2026-10-26)

7-A 전에 앞 모델의 단위 테스트를 한 번 실행했다(사용자 승인, 이번 한 번에 한해 예외). 학습, 평가, `compare_outputs`는 포함하지 않았다. 공유 파일(`metric.py`, `pre_processor.py`, `io.py`, `predictor.py`, `predict.py`) 변경 뒤 처음 하는 실행이다. 마지막 회귀 실행은 Phase 4였다. 실행으로 검증했고 Claude가 로그 파일에서 요약 줄을 확인했다.

| 대상 | 결과 | 로그 |
|---|---|---|
| `tests\common` | 30 passed | `outputs/logs/07-regress-common-261026-1.log` |
| `tests\01_stfpm` | 32 passed | `07-regress-stfpm-261026-1.log` |
| `tests\02_efficient_ad` | 22 passed | `07-regress-efficient-ad-261026-1.log` |
| `tests\03_padim` | 16 passed | `07-regress-padim-261026-1.log` |
| `tests\04_dfm` | 22 passed | `07-regress-dfm-261026-1.log` |
| `tests\05_dfkde` | 20 passed | `07-regress-dfkde-261026-1.log` |

합계 142개가 통과했고 실패는 없다. `tests\common`은 Phase 4(28개)보다 2개 늘었다(Phase 5, 6의 dfkde, patchcore 테스트). `tests\06_patchcore`는 대상이 아니었다(Phase 6에서 24개 통과).

실행한 명령은 다음과 같다(Claude가 보낸 지시 원문. agy가 시간 측정 줄을 덧붙여 하나의 스크립트로 순서대로 실행했고, 승인 창의 전문을 Claude가 agy의 transcript에서 읽어 지시와 같음을 확인한 뒤 한 번 승인했다).

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py -m pytest tests\common 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\07-regress-common-261026-1.log
& $py -m pytest tests\01_stfpm 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\07-regress-stfpm-261026-1.log
& $py -m pytest tests\02_efficient_ad 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\07-regress-efficient-ad-261026-1.log
& $py -m pytest tests\03_padim 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\07-regress-padim-261026-1.log
& $py -m pytest tests\04_dfm 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\07-regress-dfm-261026-1.log
& $py -m pytest tests\05_dfkde 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\07-regress-dfkde-261026-1.log
```

## 1. 모델 노트

### 1.1 원본과 생성자

다음 표는 원본 파일과 이식 위치의 대응이다(D-15, D-18).

| 원본 (`anomalib\models\image\reverse_distillation\`) | 이식 위치 | 비고 |
|---|---|---|
| `torch_model.py` (`ReverseDistillationModel`) | `src/visionlab/anomaly_detection/models/reverse_distillation/model.py` | import 변경, `pre_trained`는 순수 모델에서 구조만 만드는 방식으로(D-19) |
| `anomaly_map.py` (`AnomalyMapGenerator`, `AnomalyMapGenerationMode`) | `.../models/reverse_distillation/anomaly_map.py` | `GaussianBlur2d` import 변경, `omegaconf`의 `ListConfig` import 제거 |
| `loss.py` (`ReverseDistillationLoss`) | `.../models/reverse_distillation/loss.py` | 변경 없음 |
| `components/bottleneck.py` (`get_bottleneck_layer`), `components/de_resnet.py` (`get_decoder`) | `.../models/reverse_distillation/components/` | 변경 없음 (torchvision의 `BasicBlock`, `Bottleneck`, `conv1x1`, `conv3x3` 사용) |
| `lightning_model.py` (`ReverseDistillation`) | `.../models/reverse_distillation/lightning_model.py` (`ReverseDistillationLitModel`) | native `LightningModule`로 새로 작성 |
| `__init__.py` | `.../models/reverse_distillation/__init__.py` | `lightning_model`을 import하지 않도록 고침(D-18 예외) |

생성자는 다음과 같다. `ReverseDistillationLitModel`은 `ReverseDistillation`과 인자 이름·순서·기본값이 같아야 한다(FR-6, D-5).

| 클래스 | 시그니처 |
|---|---|
| `ReverseDistillation` | `backbone="wide_resnet50_2"`, `layers=("layer1", "layer2", "layer3")`, `anomaly_map_mode=AnomalyMapGenerationMode.ADD`, `pre_trained=True`, `pre_processor=True`, `post_processor=True`, `evaluator=True`, `visualizer=True` (8개) |
| `ReverseDistillationModel` | `backbone`, `input_size`, `layers`, `anomaly_map_mode`, `pre_trained=True` |

### 1.2 학습과 평가 동작

원본의 동작은 다음과 같다.

- **모델:** encoder는 `TimmFeatureExtractor`(`layers`의 특징, 학습 중에도 매 `forward`에서 `eval()`), 그 특징을 `bottleneck`(OCBE)에 넣고 `decoder`가 층별 특징을 복원한다. 학습 모드의 `forward`는 `(encoder_features, decoder_features)` 리스트를 반환한다.
- **손실:** 층별로 특징을 `(batch, -1)`로 펴 `CosineSimilarity`(dim 1)를 구하고 `1 - 유사도`의 평균을 모두 더한다.
- **학습:** `configure_optimizers`는 `decoder`와 `bottleneck` 파라미터만 `Adam(lr=0.005, betas=(0.5, 0.99))`으로 학습한다(encoder 제외). `training_step`이 `train_loss`(epoch 평균)를 로그에 남긴다.
- **추론:** 층별 `1 - cosine_similarity`를 입력 크기로 bilinear(`align_corners=True`) 보간해 `ADD`(합) 또는 `MULTIPLY`(곱)로 합치고, 매 호출마다 새로 만든 `GaussianBlur2d(sigma 4)`로 smoothing한다. `pred_score`는 `anomaly_map`의 `amax`다.
- **`trainer_arguments`:** `{"gradient_clip_val": 0, "num_sanity_val_steps": 0}`이며 `max_epochs`는 고정하지 않는다(공유 yaml의 10 epoch).
- **입력 크기:** 원본의 `input_size`는 전처리의 크기에서 얻는다. visionlab은 Dataset이 256x256으로 맞추므로 `LitModel`에서 상수 `(256, 256)`으로 둔다(생성자 인자 8개는 유지).

### 1.3 외부 의존과 로컬 자산

- **외부 의존:** timm, torchvision(`resnet.BasicBlock`, `Bottleneck`, `conv1x1`, `conv3x3`), kornia(`GaussianBlur2d`, D-23)다. anomalib 의존은 `Batch`, `InferenceBatch`, `AnomalibModule`, `TimmFeatureExtractor`, `GaussianBlur2d`, `PreProcessor`, `PostProcessor`, `Evaluator`, `Visualizer`, `LearningType`이고, `omegaconf`는 타입 표기에만 쓰여 제거한다.
- **로컬 가중치:** `paths.toml [backbones]`의 `wide_resnet50_2`(이미 있다). encoder에만 불러오고 `bottleneck`과 `decoder`는 무작위 초기화다. visionlab은 `ReverseDistillationLitModel.__init__`이 `encoder`에 주입한다(D-19).
- **데이터:** MVTec `bottle`(학습 209장), VisA `candle`(학습 900장). 분할은 `from_test`, 0.4, seed 42, batch 8/8, `num_workers=0`이다.
- **학습 예산:** 공유 yaml의 `max_epochs=10`이다.

### 1.4 이식에서 의도적으로 달라지는 곳과 유지하는 곳

- **pretrained 주입(D-19):** 순수 모델은 `pre_trained=False`로 encoder 구조만 만든다. `LitModel`이 `wide_resnet50_2` 로컬 파일을 `encoder`에 불러온다.
- **`omegaconf` 제거:** `ListConfig` import는 타입 표기뿐이라 제거한다(D-18의 의존 제거). `image_size`의 list 또는 tuple 처리는 원본 그대로다.
- **초기 가중치 동일성:** `bottleneck`과 `decoder`는 난수로 초기화되므로 레퍼런스와 같은 seed에서 같은 순서로 난수를 소비해야 초기 가중치가 같다. 순수 모델이 `pre_trained=False`로 encoder를 만들어도 timm이 초기화하므로 소비 순서는 같아야 하며(Phase 1 STFPM 선례), 확인용 비교를 정보로 기록한다.
- **원본 동작 유지:** `forward`의 `encoder.eval()` 호출, `tiler=None` 속성, 매 호출 새 `GaussianBlur2d`를 만드는 것, 손실의 `CosineSimilarity()` 사용은 고치지 않는다.
- **히스토리 비교:** 레퍼런스와 native의 에폭별 `train_loss`를 `compare_history.py`로 표로 기록한다(판정 아님, 허용 오차 없음).
- **FR-5 허용 오차:** `pred_score`와 `anomaly_map`은 `atol=1e-6, rtol=1e-5`, encoder 가중치는 `torch.equal`이다.

## 2. 레퍼런스

학습 예산은 공유 yaml의 10 epoch다(D-22). 저장된 `lightning_logs/version_0/config.yaml`에 `max_epochs: 10`이 남았고 로그에 `Epoch 9`까지 학습한 기록이 있다.

### 2.1 10 epoch 레퍼런스

2026-10-26에 agy가 `anomalib_run.py train --config default.yaml --config data/<data>.yaml --config models/07_reverse_distillation.yaml`을 데이터셋마다 1회 실행했다(실행으로 검증함). Claude는 로그 파일에서 값을 직접 확인했다. 두 로그 모두 `network access is blocked`, `URLError`, `ConnectionError`, `Traceback`, `out of memory`는 0건이다. 로그는 PowerShell `Tee-Object` 출력이라 UTF-16이다. 승인 창의 스크립트 전문은 Claude가 agy의 transcript에서 읽어 지시한 명령과 같음을 확인했다.

agy가 PowerShell에서 실행한 명령은 다음과 같다(Claude가 보낸 지시 원문. 파일은 고치지 않았다).

```powershell
cd D:\projects\visionlab\evaluation\anomalib
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/07_reverse_distillation.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\07-ref-mvtec-261026-1.log
& $py anomalib_run.py train --config default.yaml --config data/visa.yaml --config models/07_reverse_distillation.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\07-ref-visa-261026-1.log
```

다음 표는 10 epoch 레퍼런스(1회) 결과다.

| 데이터 / category | Image AUROC | Pixel AUROC | 참고 | run 폴더 | 로그 |
|---|---|---|---|---|---|
| mvtec / bottle | 0.99781 | 0.98395 | image F1 0.9589, pixel F1 0.7298 | `outputs/anomalib/ReverseDistillation/LocalMVTecAD/bottle/v1` | `outputs/logs/07-ref-mvtec-261026-1.log` |
| visa / candle | 0.94194 | 0.98917 | image F1 0.8833, pixel F1 0.3281 | `outputs/anomalib/ReverseDistillation/LocalVisa/candle/v0` | `outputs/logs/07-ref-visa-261026-1.log` |

mvtec/bottle의 `v0`은 2026-10-25에 만든 `max_epochs: 1`의 이전 동작 확인 run이며 레퍼런스가 아니다. 이번 레퍼런스는 `v1`이다. 적용된 학습 예산은 `max_epochs=10`, `val_split_mode=from_test`, `val_split_ratio=0.4`, `seed=42`, `train_batch_size=8`, `eval_batch_size=8`, `num_workers=0`, `deterministic=warn`, `backbone=wide_resnet50_2`, `layers=[layer1, layer2, layer3]`, `pre_trained=true`이다(저장된 `config.yaml`). 체크포인트는 각 run 폴더의 `weights/lightning/model.ckpt`이고 크기는 802,755,588 bytes다.

### 2.2 학습 시간 관문

사용자 결정에 따라 visa 레퍼런스의 첫 1~2 에폭 속도로 전체 예상이 약 2시간을 넘으면 계속하기 전에 보고하기로 했다. 계획 보고에서의 추정은 visa 한 번당 약 40~60분이었다. 실제 속도는 visa 에폭당 약 45~50초(약 2.5 it/s)였고, 파일 시각으로 bottle 레퍼런스가 약 4분(13:55 시작, 체크포인트 13:59), visa 레퍼런스가 약 9분(체크포인트 14:08)이었다. 추정이 약 5배 빗나갔다. 관문 기준(2시간)을 넘지 않아 보고 없이 진행했다. 추정의 근거였던 STFPM(`resnet18`)과의 비용 비율이 과대했다.

### 2.3 에폭별 학습 히스토리

레퍼런스와 native의 비교는 5.3절에 표로 기록한다. 허용 오차는 정하지 않았다.

## 3. 이식 메모

### 3.1 원본과 이식 파일의 대응

다음 표는 새로 만든 파일과 원본의 대응이다. 원본과의 차이는 `difflib`로 확인했다(구현함, 정적 검증).

| 이식 파일 (`src/visionlab/anomaly_detection/models/reverse_distillation/`) | 원본 | 원본과 다른 줄 |
|---|---|---|
| `model.py` | `torch_model.py` | import 2줄(`InferenceBatch`, `TimmFeatureExtractor`), `Tiler` 타입 참조 경로, encoder의 `pre_trained=False`와 주석 한 줄(D-19) |
| `anomaly_map.py` | `anomaly_map.py` | `omegaconf`의 `ListConfig` import와 타입 표기 제거, `GaussianBlur2d` import 한 줄 |
| `loss.py` | `loss.py` | 없음 |
| `components/bottleneck.py`, `components/de_resnet.py` | 같은 이름 | 없음 |
| `__init__.py`, `components/__init__.py` | 같은 이름 | 노출 범위 제한(D-18 예외) |
| `lightning_model.py` (`ReverseDistillationLitModel`) | `lightning_model.py` | native `LightningModule`로 새로 작성. 8개 인자 동일, Adam(lr 0.005, betas 0.5와 0.99)이 decoder와 bottleneck만 학습, `train_loss` 로그, 로컬 encoder 주입(D-19), D-8의 post_processor 갱신 제한, 입력 크기 상수 (256, 256) |

### 3.2 새로 쓴 코드와 앞 Phase 공유 파일 변경

- **새로 쓴 파일:** `configs/models/reverse_distillation.yaml`, `tests/07_reverse_distillation/test_reverse_distillation.py`, `tests/07_reverse_distillation/test_reverse_distillation_lightning.py`. `tests\07_reverse_distillation`은 합계 25개 테스트다.
- **공유 파일 변경:** `utils/io.py`(`load_export`가 `reverse_distillation` 허용), `inference/predictor.py`(모델 dict에 등록, `anomaly_map_mode` 문자열을 Enum으로 복원), `evaluation/visionlab/predict.py`(`--model reverse_distillation`), `evaluation/anomalib/compare_outputs.py`(`--model reverse_distillation`), `tests/common/test_cli.py`, `tests/common/test_dependency.py`. `common/` 아래 파일은 바꾸지 않았다. 영향 범위는 6.2절의 표에 정리했다.
- **새 부품:** 순수 모델의 decoder, bottleneck, 손실은 원본 복사다. 새로 구현한 대체 부품은 없다. `compare_history.py`는 바꾸지 않고 그대로 썼다.

### 3.3 이식 중 확인된 사항

- **`pred_score`의 shape:** 원본은 `anomaly_map [B,1,H,W]`에 `amax(dim=(-2,-1))`를 적용하므로 `pred_score`가 `[B,1]`이다(SPEC FR-4도 `[B,1]`). Claude의 지시문이 `(B,)`로 잘못 적었고, codex의 질문으로 확인해 원본을 유지했다.
- **`get_bottleneck_layer`의 fallback:** 지원하지 않는 backbone 이름에도 `Bottleneck` 구성을 반환하고, 오류를 내는 쪽은 `get_decoder`뿐이다. 원본 동작이며 테스트가 둘을 구분한다.
- **`train_loss` 로그의 집계:** 원본은 batch 크기를 추론하지 못해 epoch 평균이 step마다 1회씩 센 평균이 된다. visionlab의 batch는 튜플이라 Lightning이 첫 텐서 크기로 가중 평균을 내므로 `batch_size=1`을 두어야 한다(STFPM 선례). 7-B에서 빠져 있어 수정 요청 1회로 고쳤다(4.2절).
- **초기 가중치:** 같은 seed 42로 새로 만든 레퍼런스 순수 모델과 visionlab 순수 모델의 bottleneck, decoder 초기 가중치가 서로 같다(`INFO`, 정보 기록). 난수 소비 순서가 같다는 뜻이다.

## 4. 적대적 검증

검증은 7-A, 7-B 각각 한 회차씩 했다. 수정 요청은 7-A 0회, 7-B 1회(한도 3회 이내)다.

### 4.1 7-A (순수 모델과 decoder·bottleneck 부품)

| 지적 | 심각도 | 처리 |
|---|---|---|
| 원본과의 `difflib` 대조: `model.py`는 import와 D-19의 `pre_trained=False` 한 줄뿐이고 `forward`, `encoder.eval()`, tiler 분기, `pred_score` 계산이 같음. `anomaly_map.py`는 `omegaconf` 제거와 import 한 줄뿐. `loss.py`, `bottleneck.py`, `de_resnet.py`는 변경 없음 | 확인 | 수정 없음 |
| codex 질문: `pred_score`가 `[B,1]` (3.3절). Claude의 지시문 오류 | 낮음 | 원본 유지로 답함 |

### 4.2 7-B (Lightning, export, 비교 스크립트)

| 지적 | 심각도 | 처리 |
|---|---|---|
| `StfpmLitModel`과의 `difflib` 대조: 차이가 생성자 인자, 입력 크기 상수, encoder 주입, Adam, export 인자뿐임. `_outputs`, hook, D-8 처리가 같음. 공유 파일 변경은 등록과 Enum 복원 분기뿐 | 확인 | 수정 없음 |
| 정적 대조로 발견: `training_step`의 `self.log("train_loss", ...)`에 STFPM의 `batch_size=1`이 없음. 레퍼런스의 에폭 평균 집계와 어긋남(마지막 batch가 작을 때 가중 평균) | 중간 | 수정 요청 1회차: `batch_size=1`과 주석을 추가하고 테스트에 로그 인자 확인을 더함 |

수정 뒤 native의 에폭별 `train_loss_epoch`가 레퍼런스와 상대 차이 6e-7 이내로 일치했다(5.3절). 수정 전 코드로는 학습을 돌리지 않아 이 수정이 일치에 얼마나 기여했는지는 실행으로 확인하지 않았다. 정적 검토는 `LitModel`, 등록 코드, yaml 위주로 했고 `compare_outputs.py`의 분기는 줄 단위로 읽지 않고 실행 결과(5.2절)로 확인했다.

## 5. 이식 결과

### 5.1 최종 판정 (10 epoch, Image·Pixel AUROC 차이 0.01 이내)

native는 `train.py`가 test에서 기록한 값이고 순수 PyTorch는 `export/model.pt`만 불러 `predict.py`로 평가한 값이다. 모두 실행으로 검증했다. D-24에 따라 레퍼런스가 보고하는 값과 같은 정의로 비교하며, 순수 PyTorch의 정규화 점수 AUROC도 함께 적는다.

| 데이터 / category | 지표 | 레퍼런스 | native (raw) | 순수 PyTorch (raw) | 순수 PyTorch (정규화) | 차이(native) | 차이(순수 raw) | 판정 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| mvtec / bottle | Image AUROC | 0.997807026 | 0.997807026 | 0.99780703 | 0.99780703 | 0 | 4e-9 | 통과 |
| mvtec / bottle | Pixel AUROC | 0.983950615 | 0.983950615 | 0.98395061 | 0.98395061 | 0 | 5e-9 | 통과 |
| visa / candle | Image AUROC | 0.941944361 | 0.941944361 | 0.94194436 | 0.94194436 | 0 | 1e-9 | 통과 |
| visa / candle | Pixel AUROC | 0.989173472 | 0.989173412 | 0.98917329 | 0.98917341 | 6e-8 | 1.8e-7 | 통과 |

raw와 정규화 값이 0.01 넘게 다른 곳은 없어 D-24의 원인 기록 조건에는 해당하지 않는다. image F1도 같다(bottle 0.95890409, visa 0.88333333). pixel F1은 native가 레퍼런스와 같고(bottle 0.72984642, visa 0.32814491), 순수 PyTorch는 bottle 0.72984809로 2e-6 이내이다(판정 대상이 아니다).

run 폴더와 로그는 다음과 같다.

| 대상 | 위치 |
|---|---|
| 레퍼런스 | `outputs/anomalib/ReverseDistillation/LocalMVTecAD/bottle/v1`, `outputs/anomalib/ReverseDistillation/LocalVisa/candle/v0` |
| native | `outputs/visionlab/reverse_distillation/mvtec_bottle/v0`, `outputs/visionlab/reverse_distillation/visa_candle/v0` (`run.json`: `max_epochs` 10, `num_sanity_val_steps` 0, `gradient_clip_val` 0.0, 가중치 `D:\backbones\wide_resnet50_2-95faca4d.pth` sha256 `95faca4d...50b8`) |
| 순수 PyTorch | 같은 run의 `predict/metrics.json` |
| 로그 | `outputs/logs/07-native-{mvtec,visa}-261026-1.log`, `07-pure-{mvtec,visa}-261026-1.log` |

네 native·순수 로그에 `network access is blocked`, `URLError`, `ConnectionError`, `Traceback`, `out of memory`는 0건이다.

### 5.2 테스트와 비교 스크립트

다음 표는 agy가 실행한 묶음의 결과다(Claude가 로그 파일에서 확인). 실행 대상은 Reverse Distillation뿐이다(7-A 전의 회귀 실행은 위의 별도 절).

| 대상 | 결과 | 로그 |
|---|---|---|
| `tests\07_reverse_distillation` | 25 passed | `07-pytest-261026-1.log` |
| `tests\common\test_cli.py -k reverse_distillation` | 2 passed (10 deselected) | `07-pytest-cli-261026-1.log` |
| `tests\common\test_dependency.py` (전체 파일) | 6 passed | `07-pytest-dependency-261026-1.log` |
| `compare_outputs.py --model reverse_distillation` (bottle 체크포인트, visa 체크포인트) | PASS 23, FAIL 0, INFO 2 (각각). FR-6, FR-23(encoder `torch.equal`), FR-5(`pred_score [8,1]`과 `anomaly_map [8,1,256,256]` 최대 차이 0.0, mvtec·visa 입력 8장씩), FR-24. INFO는 seed 42 초기 bottleneck, decoder 가중치 일치 | `07-compare-outputs-{mvtec,visa}-261026-1.log` |
| `compare_history.py` | 5.3절 | `07-history-{mvtec,visa}-261026-1.log` |

`test_dependency.py`는 모든 순수 모듈의 import 분리 검사를 한 파일이 하므로 이 모델의 모듈 검증을 위해 전체를 돌렸다(Phase 5, 6에서 사용자가 사후 승인한 것과 같은 사유). 이번 Phase에서의 실행은 사용자가 사후에 승인했다(2026-10-26).

실행한 명령은 다음과 같다(Claude가 보낸 지시 원문. agy가 시간 측정 줄을 덧붙여 하나의 스크립트로 순서대로 실행했고, 승인 창의 전문을 Claude가 agy의 transcript에서 읽어 지시와 같음을 확인한 뒤 한 번 승인했다).

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py -m pytest tests\07_reverse_distillation 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\07-pytest-261026-1.log
& $py -m pytest tests\common\test_cli.py -k reverse_distillation 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\07-pytest-cli-261026-1.log
& $py -m pytest tests\common\test_dependency.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\07-pytest-dependency-261026-1.log
& $py evaluation/visionlab/train.py train --data mvtec --model reverse_distillation 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\07-native-mvtec-261026-1.log
& $py evaluation/visionlab/predict.py --model reverse_distillation --data mvtec 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\07-pure-mvtec-261026-1.log
& $py evaluation/visionlab/train.py train --data visa --model reverse_distillation 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\07-native-visa-261026-1.log
& $py evaluation/visionlab/predict.py --model reverse_distillation --data visa 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\07-pure-visa-261026-1.log
& $py evaluation/anomalib/compare_outputs.py --model reverse_distillation --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\ReverseDistillation\LocalMVTecAD\bottle\v1\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\07-compare-outputs-mvtec-261026-1.log
& $py evaluation/anomalib/compare_outputs.py --model reverse_distillation --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\ReverseDistillation\LocalVisa\candle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\07-compare-outputs-visa-261026-1.log
& $py evaluation/anomalib/compare_history.py --reference D:\projects\visionlab\outputs\anomalib\ReverseDistillation\LocalMVTecAD\bottle\v1 --native D:\projects\visionlab\outputs\visionlab\reverse_distillation\mvtec_bottle\v0 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\07-history-mvtec-261026-1.log
& $py evaluation/anomalib/compare_history.py --reference D:\projects\visionlab\outputs\anomalib\ReverseDistillation\LocalVisa\candle\v0 --native D:\projects\visionlab\outputs\visionlab\reverse_distillation\visa_candle\v0 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\07-history-visa-261026-1.log
```

agy는 파일을 고치지 않았다.

### 5.3 에폭별 학습 히스토리 비교 (허용 오차 없음, 비교표만)

다음 표는 `compare_history.py`의 `train_loss_epoch` 출력이다. 판정에 쓰지 않는다.

mvtec / bottle (레퍼런스 `bottle/v1`, native `mvtec_bottle/v0`):

| Epoch | 레퍼런스 | native | 절대 차이 | 상대 차이 |
|---:|---:|---:|---:|---:|
| 0 | 0.8874665499 | 0.8874664903 | 6.0e-8 | 6.7e-8 |
| 1 | 0.3506495059 | 0.3506494462 | 6.0e-8 | 1.7e-7 |
| 2 | 0.214470759 | 0.214470759 | 0 | 0 |
| 3 | 0.1738847494 | 0.1738847047 | 4.5e-8 | 2.6e-7 |
| 4 | 0.1486987025 | 0.1486986727 | 3.0e-8 | 2.0e-7 |
| 5 | 0.1631381363 | 0.1631381363 | 0 | 0 |
| 6 | 0.1263706237 | 0.1263706535 | 3.0e-8 | 2.4e-7 |
| 7 | 0.1206816956 | 0.120681718 | 2.2e-8 | 1.9e-7 |
| 8 | 0.1154354811 | 0.1154354736 | 7.5e-9 | 6.5e-8 |
| 9 | 0.1109993309 | 0.1109993532 | 2.2e-8 | 2.0e-7 |

visa / candle (레퍼런스 `candle/v0`, native `visa_candle/v0`):

| Epoch | 레퍼런스 | native | 절대 차이 | 상대 차이 |
|---:|---:|---:|---:|---:|
| 0 | 0.4011861384 | 0.4011861384 | 0 | 0 |
| 1 | 0.1592369378 | 0.1592369974 | 6.0e-8 | 3.7e-7 |
| 2 | 0.133858189 | 0.1338581294 | 6.0e-8 | 4.5e-7 |
| 3 | 0.122242786 | 0.122242786 | 0 | 0 |
| 4 | 0.1137274578 | 0.1137273908 | 6.7e-8 | 5.9e-7 |
| 5 | 0.1074094996 | 0.1074095145 | 1.5e-8 | 1.4e-7 |
| 6 | 0.1006191075 | 0.1006191224 | 1.5e-8 | 1.5e-7 |
| 7 | 0.09570588171 | 0.09570587426 | 7.5e-9 | 7.8e-8 |
| 8 | 0.09015484154 | 0.09015485644 | 1.5e-8 | 1.7e-7 |
| 9 | 0.08577395976 | 0.08577398956 | 3.0e-8 | 3.5e-7 |

모든 에폭의 상대 차이가 6e-7 이하이고 절대 차이가 7e-8 이하다. float32 단위 오차 수준이어서, 학습 경로(초기 가중치, 데이터 순서, 손실, optimizer)가 레퍼런스와 같다고 볼 수 있다. 에폭별 허용 오차는 사용자가 아직 정하지 않았으며, 이 값을 참고용으로 남긴다.

### 5.4 crop·입력 크기

이 모델은 crop이 없고 입력 크기는 256x256이다. FR-5의 `anomaly_map [8,1,256,256]` 최대 차이가 0.0이다.

## 6. 결론

- **레퍼런스:** bottle, visa 확보(10 epoch).
- **mvtec / bottle:** Image·Pixel AUROC가 native와 순수 PyTorch 모두 레퍼런스와 5e-9 이내다. 통과.
- **visa / candle:** Image AUROC 일치, Pixel AUROC 차이 native 6e-8, 순수 PyTorch 1.8e-7. 통과. 정규화 점수 기준(D-24)도 같다.
- **FR-5, FR-6, FR-23, FR-24:** 통과(5.2절). `pred_score`와 `anomaly_map` 최대 차이 0.0이다. 에폭별 `train_loss_epoch`는 상대 차이 6e-7 이내(5.3절)이다.
- **테스트:** `tests\07_reverse_distillation` 25개, `test_cli.py -k reverse_distillation` 2개, `test_dependency.py` 6개가 통과했다. 7-A 전의 사용자 승인 회귀 실행 142개도 통과했다.
- **수정 요청:** 7-A 0회, 7-B 1회(`train_loss` 로그의 `batch_size=1`).
- **Phase 종료:** 사용자가 종료를 결정했다(2026-10-26).

### 6.1 `specs/` 갱신이 필요한 항목

`specs/`는 사람이 쓴다. 이번 Phase에서 새로 반영이 필요한 항목은 없고, 사용자 결정(2026-10-26)으로 `specs/`를 고치지 않았다. 다음은 참고 사항이다.

- **에폭별 히스토리의 허용 오차:** Phase 2부터 미정이며, 사용자 결정(2026-10-26)으로 미정을 유지한다. 이번 Phase의 값(상대 차이 6e-7 이내)은 참고로만 남긴다.
- **입력 크기 상수:** 원본의 `input_size`를 전처리에서 얻는 대신 `LitModel`에서 상수 (256, 256)으로 두었다. Dataset이 256x256으로 맞추는 현재 구성(D-6)에 의존한다. 다른 입력 크기가 필요한 Phase가 생기면 구조를 다시 정해야 한다.

### 6.2 남은 위험과 사용자 판단 대상

- 레퍼런스가 category당 1회뿐이다. 다만 native가 레퍼런스와 수치까지 일치해(AUROC, 에폭별 손실, FR-5 최대 차이 0.0) 학습 경로의 편차는 작아 보인다.
- `tests\common\test_dependency.py` 전체 실행은 사용자가 사후에 승인했다(2026-10-26).
- 앞 모델 회귀는 이번 Phase에서 7-A 전에 승인된 1회(142개 통과)만 실행했다. 7-B의 공유 파일 변경 이후의 회귀는 보류했다(사용자 결정 2026-10-26). 실행하지 않았고 변경은 모두 기존 항목을 건드리지 않는 추가다.

| 공유 파일 | 변경 | 영향받을 수 있는 앞 모델 |
|---|---|---|
| `src/visionlab/utils/io.py` | `load_export`가 허용하는 이름에 `reverse_distillation` 추가 | 모든 모델의 export 로딩 |
| `src/visionlab/inference/predictor.py` | `from_export` 모델 dict에 등록, `anomaly_map_mode` 복원 분기 | 모든 모델의 `Predictor.from_export` |
| `evaluation/visionlab/predict.py` | `--model` 선택지에 추가 | 모든 모델의 순수 PyTorch 평가 |
| `evaluation/anomalib/compare_outputs.py` | `--model reverse_distillation` 분기 | stfpm, efficient_ad, padim, dfm, dfkde, patchcore 비교 |
| `tests/common/test_cli.py`, `tests/common/test_dependency.py` | reverse_distillation 테스트와 분류 추가 | 같은 파일의 앞 모델 테스트 |

- 체크포인트가 약 800MB(optimizer 상태 포함)라 `outputs/`의 디스크 사용이 크다(커밋하지 않는다).
