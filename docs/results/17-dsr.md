# Phase 17. DSR

모델 폴더 이름은 `dsr`이고 원본은 anomalib 2.0.0의 `anomalib\models\image\dsr\`(`C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\Lib\site-packages\anomalib\models\image\dsr\`)이다. 범위와 완료 조건은 `specs/PLAN.md`의 Phase 17과 "Phase 공통 양식"을 따른다. 판정 기준은 category별 Image·Pixel AUROC 차이 0.01 이내다.

## 상태

다음 표는 단계별 진행 상태다. 수정 반복 한도 3회는 단계별로 센다.

| 단계 | 내용 | 상태 |
|---|---|---|
| 사전 확인 | `D:\backbones\vq_model_pretrained_128_4096.pckl`이 `weights_only=True`로 읽히는지 읽기 전용 확인, `paths.toml [teachers]`에 `dsr_vq` 등록 | 완료 |
| 레퍼런스 | `anomalib_run.py train` x 2 (mvtec/bottle 1회, visa/candle 1회, 10 epoch) | 완료 |
| 17-A | 순수 모델 (`models/dsr/{model,anomaly_generator,loss}.py`, `tests/17_dsr`의 순수 부분) | 완료 (수정 요청 0회) |
| 17-B | Lightning (`DsrLitModel`, 두 optimizer 단계 전환, VQ 가중치 주입, configs, export·Predictor 등록, 비교 스크립트) | 완료 (수정 요청 1회) |
| 판정 | Image·Pixel AUROC 차이 0.01 이내 (두 데이터) | 통과 |
| 사후 회귀 | 사용자 승인 회귀 실행(2026-10-28): `tests\common` | 완료 (63 passed) |
| specs 정리 | FR-4 DSR 예외, D-15 트리, D-23 일반화 (사용자 결정) | 완료 |
| Phase 종료 | 사용자 결정과 커밋·푸시 | 사용자 결정(2026-10-28), 커밋·푸시 |

`PerlinAnomalyGenerator`와 `generate_perlin_noise`(`common/perlin.py`, Phase 10~11), `PreProcessor`(정규화 없는 resize 전용 구성, DRAEM 선례), `PostProcessor`, `Evaluator`, 수동 최적화 골격(CFLOW, GANomaly 선례), `load_pretrained_teacher`는 앞 Phase의 것을 재사용한다.

## 사용자 결정 (2026-10-28)

다음은 Phase 시작 전에 사용자가 정한 항목이다. 계획 보고의 제안 1~6을 그대로 채택했다.

| 항목 | 결정 |
|---|---|
| 학습 에폭 | 원본 `trainer_arguments`는 `{"num_sanity_val_steps": 0}`뿐이고 `max_epochs`를 고정하지 않으므로 D-22의 첫 문장대로 공유 10 epoch를 레퍼런스와 native에 똑같이 쓴다. 단계 전환은 epoch 7(`second_phase=7`)이다. 실제 10 epoch임을 `run.json`과 로그의 `max_epochs=10 reached`로 확인해 기록한다 |
| 시간과 category | 그대로 진행한다(CSFlow 선례, candle을 줄이지 않음). 45분 초과 가능성은 계획 보고로 사전 보고가 끝났다. 중간 보고 관문은 예상 상한(약 60분)의 1.5배인 약 90분이다. 넘을 것으로 보이면 계속하기 전에 보고하고 category는 스스로 바꾸지 않는다 |
| VQ 가중치 | `D:\backbones\vq_model_pretrained_128_4096.pckl`을 `paths.toml [teachers]`의 `dsr_vq` 키로 등록한다(한 줄 추가를 승인함). `DsrLitModel`이 생성 직후 `weights_only=True`, `strict`로 주입하고 순수 `DsrModel`에서는 파일을 읽는 로딩 메서드를 제거한다(D-19). `.pckl`이 `weights_only=True`로 읽히는지 17-A 전에 읽기 전용으로 확인하고, 읽히지 않으면 멈추고 보고한다 |
| kornia(D-23) | 코드는 원본처럼 kornia를 import한다(독립 구현은 배제). specs는 지금 고치지 않고 D-23 갱신 초안(`dsr/loss.py` 추가, 가능하면 "학습 전용 모델 패키지 `loss.py`에서 kornia 허용"처럼 일반화한 문구 선택지)을 6.1절에 적고 최종 보고에서 묻는다. kornia 허용 위치를 검사하는 앞 Phase 테스트가 실패하면 한 항목만 고치고 사후 실행 필요를 최종 보고에 올린다 |
| FR-4 예외 | `anomaly_map [B,H,W]`, `pred_score [B]` 예외는 17-A에서 코드로 확정하고 지금은 specs를 고치지 않으며 최종 보고에서 묻는다 |
| D-15 트리 | 지금은 고치지 않는다. 추가 항목(`tests/17_dsr/`, `configs/models/dsr.yaml`, `models/dsr/`)을 최종 보고에서 묻는다. 새 D 번호는 불필요하다 |
| 앞 모델 회귀(01~16) | 보류. 돌리지 않는다. 영향 표는 6.2절에 남긴다. 최종 보고에서 사후 `tests\common` 1회를 묻는다. `test_dependency.py` 전체 실행이 필요하면 사후 승인 항목으로 올린다 |
| 최종 보고 뒤 결정 (2026-10-28) | 결정 1~3을 권고안대로 채택했다. (1) specs 초안을 반영한다(FR-4에 DSR 예외를 기존 예외 문장 형식대로 최소한으로, D-15 트리에 `tests/17_dsr/`, `configs/models/dsr.yaml`, `models/dsr/` 다섯 파일, D-23은 (b)로 일반화해 제목, 선택, 적용 범위를 "학습 전용 모델 패키지의 `loss.py`에서 kornia 허용"으로 고치고 현재 import하는 곳 세 파일을 코드로 확인해 열거한다. 새 D 번호는 만들지 않고 INTENT와 PLAN은 고치지 않는다). (2) 커밋 전 `tests\common` 1회를 사후 실행한다(01~16 모델별 회귀는 계속 보류). (3) Phase 종료를 결정했고 커밋·푸시한다 |
| 시간 관문 (2026-10-28) | 레퍼런스 mvtec 완료 뒤 남은 실행이 약 90분 관문을 넘을 것으로 보여 보고했고, 사용자가 "그대로 계속"을 결정했다. category, epoch, batch를 바꾸지 않고 추가 관문 없이 최종 보고까지 진행한다 |
| 파라미터 수 | 17-A에서 codex가 계층별로 합산해 이 문서에 기록한다(3.3절) |
| codex 세션 | 새 세션으로 시작한다 |
| FR-5 허용 오차 | `atol=1e-6, rtol=1e-5` (Phase 2~16과 같음) |
| 학습 히스토리 | `train_loss_epoch`의 10 epoch 표만 기록한다(허용 오차 없음). 단계 전환(epoch 7)에서 손실 정의가 바뀌는 불연속은 표에 표시한다 |
| 난수가 어긋날 때의 판정 | 난수 상태를 주입하지 않고 AUROC 0.01로만 판정한다. 어긋나면 스스로 처리하지 않고 멈춰 보고한다(반복 실행, 판정 완화, 미달 기록은 사용자가 정한다) |
| 판정 | Image·Pixel AUROC를 판정한다. 정규화 점수 기준 값(D-24)과 raw 값을 함께 기록한다 |
| 실행 범위 | DSR의 명령만 실행한다 |

## 1. 모델 노트

### 1.1 원본과 생성자

다음 표는 원본 파일과 이식 위치의 대응이다(D-15, D-18).

| 원본 | 이식 위치 | 비고 |
|---|---|---|
| `models/image/dsr/torch_model.py` (`DsrModel`과 하위 모듈) | `src/visionlab/anomaly_detection/models/dsr/model.py` | import 변경, 사전학습 로딩 메서드 제거(D-19) |
| `models/image/dsr/anomaly_generator.py` (`DsrAnomalyGenerator`) | `.../models/dsr/anomaly_generator.py` | `generate_perlin_noise`를 visionlab 것으로 |
| `models/image/dsr/loss.py` (`DsrSecondStageLoss`, `DsrThirdStageLoss`) | `.../models/dsr/loss.py` | 변경 없음(kornia `FocalLoss`) |
| `models/image/dsr/lightning_model.py` (`Dsr`) | `.../models/dsr/lightning_model.py` (`DsrLitModel`) | native `LightningModule`로 새로 작성 |
| `models/image/dsr/__init__.py` | `.../models/dsr/__init__.py` | `lightning_model`을 import하지 않도록 고침(D-18 예외) |

생성자는 다음과 같다. `DsrLitModel`은 `Dsr`와 인자 이름·순서·기본값이 같아야 한다(FR-6, D-5).

| 클래스 | 시그니처 |
|---|---|
| `Dsr` | `latent_anomaly_strength=0.2`, `upsampling_train_ratio=0.7`, `pre_processor=True`, `post_processor=True`, `evaluator=True`, `visualizer=True` (6개) |
| `DsrModel` | `latent_anomaly_strength=0.2`, `embedding_dim=128`, `num_embeddings=4096`, `num_hiddens=128`, `num_residual_layers=2`, `num_residual_hiddens=64` (6개) |
| `DsrAnomalyGenerator` | `p_anomalous=0.5` |
| `DsrSecondStageLoss`, `DsrThirdStageLoss` | 인자 없음 |

`Dsr`에는 `input_size`가 없고 모델도 입력 크기 인자를 받지 않는다. 레퍼런스는 모델 yaml의 `pre_processor`(`Resize(256)`만, 정규화 없음)로 256을 맞추고 native의 Dataset도 256을 준다. anomalib 2.0.0은 이 모델의 `configure_transforms`를 호출하지 않아 기본 `PreProcessor`가 ImageNet 정규화를 더하므로(레퍼런스 yaml 주석), 원본의 의도(입력은 `[0, 1]`)를 맞추려고 레퍼런스 yaml이 resize 전용 `pre_processor`를 둔다.

### 1.2 학습과 평가 동작

원본의 동작은 다음과 같다.

- **구조:** 고정된 사전학습 `DiscreteLatentModel`(VQ-VAE-2 형태, 인코더 상하 두 단과 `VectorQuantizer`), 객체별 `ImageReconstructionNetwork`, 두 개의 `SubspaceRestrictionModule`(lo, hi), `AnomalyDetectionModule`(U-Net), `UpsamplingModule`로 이루어진다. `DiscreteLatentModel`의 파라미터는 `requires_grad=False`다.
- **단계 전환:** `configure_optimizers`가 `num_steps = max(trainer.max_steps // len(train_dataloader), trainer.max_epochs)`로 `second_phase = int(num_steps * 0.7)`을 정한다. `max_steps=-1`이면 `num_steps`는 10이고 `second_phase`는 7이다. epoch 0~6은 2단계, epoch 7~9는 3단계다.
- **2단계(epoch < 7):** `DsrAnomalyGenerator.augment_batch`로 마스크를 만들고 모델 forward가 잠재 공간에 이상을 주입한다. `DsrSecondStageLoss`(MSE 세 개와 focal)를 `ph1_opt`(`Adam`, lr 2e-4, 이미지 복원·subspace 둘·이상 탐지 모듈의 파라미터)로 갱신한다.
- **3단계(epoch >= 7):** `PerlinAnomalyGenerator`로 입력에 합성 이상을 넣고 `DsrThirdStageLoss`(focal)를 `ph2_opt`(`Adam`, lr 2e-4, `upsampling_module`)로 갱신한다.
- **스케줄러:** `StepLR(optimizer_d, int(0.8 * second_phase), gamma=0.1)`을 만들지만 수동 최적화라 Lightning이 step하지 않고 원본도 호출하지 않는다. lr은 2e-4로 일정하다.
- **평가 `forward`:** 학습 모드가 아니면 `anomaly_map`과 `pred_score`(`amax` 풀링, batch 1의 0차원은 `unsqueeze`로 보정)를 `InferenceBatch`로 반환한다.
- **로그:** `train_loss`를 `on_epoch=True`로 기록한다. batch가 Batch 객체라 Lightning이 batch 크기를 추론하지 못해 epoch 평균은 step마다 1회씩 센 평균이다(CFLOW, GANomaly와 같음). native는 `batch_size=1`로 맞춘다.
- **`trainer_arguments`:** `{"num_sanity_val_steps": 0}`이며 `max_epochs`는 고정하지 않는다(공유 10 epoch).
- **사전학습 가중치:** `on_train_start`에서 `prepare_pretrained_model()`(없으면 다운로드)과 `torch.load(ckpt)`로 `discrete_latent_model`에 불러온다.

### 1.3 외부 의존과 로컬 자산

- **외부 의존:** torch, torchvision(`v2.RandomAffine`), kornia(`FocalLoss`, `loss.py`에서만. D-23 범위 확인 필요). timm, FrEIA, sklearn은 쓰지 않는다. `anomalib.data.utils`의 다운로드(`DownloadInfo`, `download_and_extract`)는 NFR-3 때문에 제거한다.
- **로컬 가중치:** `D:\backbones\vq_model_pretrained_128_4096.pckl`(15.4MB). `torch.load(weights_only=True)`로 읽혀 `OrderedDict` 키 40개, 모두 float32 텐서, 총 4,029,571개 원소다(`_encoder_t._conv_1.weight`, ..., `upsample_t.bias`). 로그 `outputs/logs/17-pckl-check-261028-1.log`. 레퍼런스는 `local_models.LocalDsr`가 `prepare_pretrained_model`을 가로채 이 파일을 쓴다. native는 `paths.toml [teachers]`의 `dsr_vq`로 주입한다.
- **데이터:** MVTec `bottle`(학습 209장), VisA `candle`(학습 900장). 분할은 `from_test`, 0.4, seed 42, batch 8/8, `num_workers=0`이다.
- **학습 예산:** 공유 yaml의 `max_epochs=10`이다.

### 1.4 이식에서 의도적으로 달라지는 곳과 유지하는 곳

- **pretrained 주입(D-19):** 순수 `DsrModel`에서 `load_pretrained_discrete_model_weights`를 제거하고 `DsrLitModel`이 생성 직후 주입한다(원본은 `on_train_start`). 주입은 난수를 소비하지 않는다.
- **anomalib 의존 제거:** `InferenceBatch`, `generate_perlin_noise`, `PerlinAnomalyGenerator`는 visionlab 것으로 바꾼다. 다운로드 코드는 제거한다.
- **원본 동작 유지(고치지 않음):** 모듈 생성 순서(`state_dict` 키와 난수 소비 순서), 2단계에서 `torch.rand`의 호출 순서, `StepLR`이 step되지 않는 점, 2단계와 3단계의 손실 정의와 optimizer 분리.
- **FR-5 허용 오차:** `pred_score`와 `anomaly_map`은 `atol=1e-6, rtol=1e-5`다.

## 2. 레퍼런스

학습 예산은 공유 `max_epochs=10`이고 데이터 batch는 공유 8/8이다(1.2절). 입력은 모델 yaml의 `pre_processor`(`Resize(256)`만, 정규화 없음)로 맞췄다. 어댑터는 `local_models.LocalDsr`(VQ 가중치를 로컬 파일로 바꿈)다.

### 2.1 레퍼런스 실행

2026-10-28에 agy가 `anomalib_run.py train --config default.yaml --config data/<data>.yaml --config models/17_dsr.yaml`을 데이터셋마다 1회 실행했다(실행으로 검증함). Claude는 로그 파일에서 값을 직접 확인했다. 두 로그 모두 `Traceback`, `out of memory`, `URLError`가 0건이고 `Trainer.fit stopped: max_epochs=10 reached`다. 로그는 UTF-16이다. 두 명령의 승인 창은 지시한 줄과 정확히 같음을 읽고 한 번씩만 승인했다(영구 허용 없음).

agy가 PowerShell에서 실행한 명령은 다음과 같다(Claude가 보낸 지시 원문. 파일은 고치지 않았다).

```powershell
cd D:\projects\visionlab\evaluation\anomalib
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/17_dsr.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\17-ref-mvtec-261028-1.log
& $py anomalib_run.py train --config default.yaml --config data/visa.yaml --config models/17_dsr.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\17-ref-visa-261028-1.log
```

다음 표는 레퍼런스(1회) 결과다.

| 데이터 / category | Image AUROC | Pixel AUROC | 참고 | 에폭당 시간(진행 바) | run 폴더 | 로그 |
|---|---|---|---|---|---|---|
| mvtec / bottle | 0.66666669 | 0.77135885 | image F1 0.8736, pixel F1 0.3410 | 약 37~47초 | `outputs/anomalib/LocalDsr/LocalMVTecAD/bottle/v0` | `outputs/logs/17-ref-mvtec-261028-1.log` |
| visa / candle | 0.55583328 | 0.58133513 | image F1 0.6592, pixel F1 0.0253 | 약 2분 27초~2분 57초 | `outputs/anomalib/LocalDsr/LocalVisa/candle/v0` | `outputs/logs/17-ref-visa-261028-1.log` |

레퍼런스의 `train_loss_epoch`는 2단계(epoch 0~6)에서 감소하다가 3단계(epoch 7부터)에서 손실 정의가 바뀌어 불연속하게 떨어진다(5.3절).

### 2.2 학습 시간

계획의 추정(레퍼런스 candle 에폭당 약 3분)과 비슷했다. 에폭당 시간은 2단계(epoch 0~6)가 3단계(epoch 7~9)보다 길다. 이 모델은 step당 약 1.6초라 candle 한 번에 약 30분이 걸린다.

## 3. 이식 메모

### 3.1 원본과 이식 파일의 대응

다음 표는 새로 만든 파일과 원본의 대응이다. 원본과의 차이는 `difflib`로 확인했다(구현함, 정적 검증).

| 이식 파일 (`src/visionlab/anomaly_detection/models/dsr/`) | 원본 | 원본과 다른 줄 |
|---|---|---|
| `model.py` | `torch_model.py` | `Path` import 제거, `InferenceBatch` import 경로, `load_pretrained_discrete_model_weights` 메서드 제거(D-19, 주석 한 줄) |
| `anomaly_generator.py` | `anomaly_generator.py` | `generate_perlin_noise` import 경로 한 줄 |
| `loss.py` | `loss.py` | 없음(kornia `FocalLoss`) |
| `__init__.py` | `__init__.py` | `DsrModel`만 노출(D-18 예외, Lightning import 제거) |
| `lightning_model.py` (`DsrLitModel`) | `lightning_model.py` | native `LightningModule`로 새로 작성. 6개 인자 동일, `configure_optimizers`의 `second_phase` 계산과 optimizer 순서·`StepLR` 구성 유지, 생성 직후 VQ 주입(D-19), `batch_size=1` 로그, 다운로드 코드 제거 |

### 3.2 새로 쓴 코드와 앞 Phase 공유 파일 변경

- **새로 쓴 파일:** `configs/models/dsr.yaml`, `models/dsr/`(다섯 파일), `tests/17_dsr/test_dsr.py`, `tests/17_dsr/test_dsr_lightning.py`(합계 26개).
- **설정 변경:** `paths.toml [teachers]`에 `dsr_vq` 한 줄(Claude, 사용자 승인). 레퍼런스 yaml과 어댑터는 이 Phase 전에 이미 있었다.
- **공유 파일 변경:** `utils/io.py`(`load_export`가 `dsr` 허용), `inference/predictor.py`(`models` 사전에 `DsrModel` import와 등록 두 줄), `evaluation/visionlab/predict.py`(`--model dsr`), `evaluation/anomalib/compare_outputs.py`(`--model dsr` 분기), `tests/common/test_cli.py`(dsr 테스트 추가), `tests/common/test_dependency.py`(`dsr/model.py`, `anomaly_generator.py`, `loss.py` 분류 두 줄). `common/`와 `components/`는 바꾸지 않았다. 영향 범위는 6.2절의 표에 정리했다.
- **앞 Phase 테스트 변경:** 없다. kornia import 위치를 검사하는 전역 허용 목록 테스트가 없어(`tests/11_draem/test_draem.py`는 kornia를 직접 import할 뿐이다) 한 항목 수정이 필요하지 않았다.
- **새 부품:** 없다. 모두 원본 복사이며 `PerlinAnomalyGenerator`와 `generate_perlin_noise`는 Phase 10~11의 `common/perlin.py`를 재사용했다.

### 3.3 이식 중 확인된 사항

- **파라미터 수(codex가 계층별로 합산, 모델을 만들지 않음):** 기본 구성에서 `discrete_latent_model` 4,021,379, `image_reconstruction_network` 9,418,371, `subspace_restriction_module_lo`와 `hi` 각 7,818,240, `anomaly_detection_module` 5,610,690, `upsampling_module` 5,611,842로 합계 40,298,762개다. VQ에 EMA buffer 8,192개 원소가 따로 있어 `.pckl`의 4,029,571개 원소와 맞는다. 체크포인트는 약 452MB(optimizer 둘 포함), export는 약 161MB다.
- **평가 출력 shape:** `anomaly_map [B,H,W]`, `pred_score [B]`(batch 1에서도 `[1]`)다. 기본 규칙 `[B,1]`, `[B,1,H,W]`와 달라 FR-4 예외가 필요하다(6.1절).
- **입력 크기 제약:** 입력 공간축은 32의 배수여야 다운샘플링·복원 크기가 맞는다. 정사각형 입력은 InstanceNorm의 공간 원소 조건 때문에 최소 64x64다.
- **BatchNorm 없음:** 원본에는 BatchNorm이 없다. 사전학습 backbone과 `TimmFeatureExtractor` dry-run도 없어 Phase 15의 BatchNorm 오염은 해당되지 않는다.
- **`Tensor.cuda()` 고정:** 원본 2단계 `forward`에 `.cuda()`가 5곳 고정되어 있어 CPU에서는 돌지 않는다. 원본을 그대로 보존했고, 순수 테스트와 Lightning 테스트, `compare_outputs`의 CPU 비교에서만 `Tensor.cuda`를 identity로 바꾼다.
- **VQ 가중치 주입:** 원본은 `on_train_start`에서 불러오지만 native는 생성 직후 `load_pretrained_teacher`로 `strict` 주입한다. 난수를 쓰지 않아 같다. 레퍼런스 체크포인트의 `discrete_latent_model` 상태가 로컬 `.pckl`과 키 40개 모두 같다(`compare_outputs`의 INFO). 학습 중 VQ 상태가 변하지 않는다.
- **kornia:** `loss.py`만 `kornia.losses.FocalLoss`를 import한다. `Predictor`가 이미 PaDiM 때문에 kornia를 최상단에서 간접 import하므로 export 로딩 격리 테스트는 kornia 차단을 제외했다.
- **`StepLR`:** 수동 최적화라 step되지 않아 lr은 2e-4로 일정하다. 테스트가 확인한다.

## 4. 적대적 검증

검증은 17-A, 17-B 각각 한 회차씩 했다. 수정 요청은 17-A 0회, 17-B 1회(한도 3회 이내)다.

### 4.1 17-A (순수 모델)

| 지적 | 심각도 | 처리 |
|---|---|---|
| 원본과의 `difflib` 대조: `loss.py`는 차이 없음, `anomaly_generator.py`는 import 한 줄, `model.py`는 `Path` import와 `InferenceBatch` 경로와 로딩 메서드 제거뿐. 모듈 이름, 생성 순서, 난수 호출 순서까지 원본과 같음 | 확인 | 수정 없음 |
| codex의 질문: 원본 2단계 `forward`의 `Tensor.cuda()` 고정 때문에 CPU 테스트가 불가능함. 모델 원본은 보존하고 테스트에서만 `Tensor.cuda`를 identity로 monkeypatch하는 안(옵션 1)으로 답함 | 낮음 | 지시 보완 |

### 4.2 17-B (Lightning, export, 비교 스크립트)

| 지적 | 심각도 | 처리 |
|---|---|---|
| 원본 `lightning_model.py`와의 대조: 생성 순서, `second_phase` 계산(`max(max_steps // len(dl), max_epochs)`), optimizer 파라미터 순서(hi 다음 lo), `StepLR(anneal, 0.1)`, 2단계·3단계 분기, `PerlinAnomalyGenerator()` 호출 방식, `train_loss`(`batch_size=1`)가 같음. 의도한 차이는 VQ 주입 시점과 native 처리·체크포인트·export 연결뿐. `predictor.py`는 import와 등록 두 줄만 추가했고 기존 분기는 그대로임을 `git diff`로 확인 | 확인 | 수정 없음 |
| codex의 질문: `Predictor`가 PaDiM 때문에 kornia를 최상단에서 간접 import하므로 kornia 차단 자식 프로세스에서 DSR export를 로드할 수 없음. 제 지시문의 kornia 차단이 잘못이었고 기존 구조 유지, export 검사에서 kornia 차단 제외(옵션 2)로 답함 | 낮음 | 지시 보완 |
| 1차 `compare_outputs --model dsr`에서 mvtec 체크포인트 비교 PASS 52 FAIL 3, candle 체크포인트 비교 PASS 46 FAIL 9. 원인 (1) 입력 범위 검사 `max <= 1`이 antialias resize 결과 1.000000238을 허용하지 않음(스크립트 결함), (2) 2단계·3단계 학습 forward를 GPU에서 비교하는데 VQ argmin에서 거의 같은 거리의 코드가 GPU 연산 잡음으로 뒤집힘(embedding_bot 최대 0.112 등. GPU 비결정성으로 추정, 평가 forward는 CPU에서 0.0). 모델 코드 문제가 아님 | 중간 | 수정 요청 1회(비교 스크립트만): 입력 범위에 1e-5 허용, 단계 forward 비교를 CPU(`Tensor.cuda` identity)에서 판정하고 GPU 비교는 INFO로 분리. 재실행에서 두 체크포인트 모두 PASS 55, FAIL 0 |
| `tests/17_dsr/test_dsr_lightning.py`와 `compare_outputs.py`의 dsr 분기는 테스트 이름과 함수 구조를 확인했고 본문은 줄 단위로 읽지 않았다(실행 결과로 판정) | 낮음 | 수정 없음 |

## 5. 이식 결과

### 5.1 최종 판정 (10 epoch, Image·Pixel AUROC 차이 0.01 이내)

native는 `train.py`가 test에서 기록한 값이고 순수 PyTorch는 `export/model.pt`만 불러 `predict.py`로 평가한 값이다. 모두 실행으로 검증했다. D-24에 따라 레퍼런스가 보고하는 값과 같은 정의(정규화하고 clamp한 점수)로 비교하며, 순수 PyTorch의 정규화 AUROC를 판정에 쓰고 raw 값은 참고로 적는다.

| 데이터 / category | 지표 | 레퍼런스 | native (raw) | 순수 PyTorch (raw) | 순수 PyTorch (정규화) | 차이(순수 정규화) | 판정 |
|---|---|---:|---:|---:|---:|---:|---|
| mvtec / bottle | Image AUROC | 0.66666669 | 0.84429824 | 0.84429824 | 0.66666669 | 0 | 통과 |
| mvtec / bottle | Pixel AUROC | 0.77135885 | 0.77136463 | 0.77136463 | 0.77135891 | 6e-8 | 통과 |
| visa / candle | Image AUROC | 0.55583328 | 0.59791666 | 0.59833330 | 0.55611110 | 2.8e-4 | 통과 |
| visa / candle | Pixel AUROC | 0.58133513 | 0.58357704 | 0.58357972 | 0.58133793 | 2.8e-6 | 통과 |

Image F1과 Pixel F1도 레퍼런스, native, 순수 PyTorch가 같다(mvtec image 0.8736, pixel 0.3410. visa image 0.6592, pixel 0.0253).

두 정의의 값이 0.01 넘게 다르면 원인을 적어야 한다(D-24). mvtec Image AUROC는 raw 0.84430과 정규화 0.66667이 0.1776, visa Image AUROC는 raw 0.59833과 정규화 0.55611이 0.0422 다르다. 원인은 PostProcessor의 임계값 중심 정규화 뒤 `clamp(0, 1)`에서 점수가 0 또는 1로 잘려 동점이 생기는 것이다(DFM visa, GANomaly와 같은 현상). Pixel AUROC는 mvtec 6e-8, visa 2.2e-3으로 차이가 작다. 레퍼런스의 raw 값은 구하지 않았다(D-24의 남은 한계). 참고로 visa Image AUROC의 native raw(0.59792)와 순수 PyTorch raw(0.59833)가 4.2e-4 다르다. 두 값은 평가 경로가 달라(native `train.py`의 test 단계와 export 기반 `predict.py`) 생긴 작은 차이로 보이며 원인은 확인하지 않았다. 판정에 쓰는 정규화 값과 0.01 기준에는 영향이 없다.

run 폴더와 로그는 다음과 같다.

| 대상 | 위치 |
|---|---|
| 레퍼런스 | `outputs/anomalib/LocalDsr/LocalMVTecAD/bottle/v0`, `outputs/anomalib/LocalDsr/LocalVisa/candle/v0` |
| native | `outputs/visionlab/dsr/mvtec_bottle/v0`, `outputs/visionlab/dsr/visa_candle/v0` (`run.json`: `max_epochs` 10, `num_sanity_val_steps` 0, seed 42, 가중치 `D:\backbones\vq_model_pretrained_128_4096.pckl`, sha256 `5efd4662...`) |
| 순수 PyTorch | 같은 run의 `predict/metrics.json`, `export/model.pt`(161,280,162 bytes) |
| 로그 | `outputs/logs/17-native-{mvtec,visa}-261028-1.log`, `17-pure-{mvtec,visa}-261028-1.log` |

### 5.2 테스트와 비교 스크립트

다음 표는 agy가 실행한 묶음의 결과다(Claude가 로그 파일에서 확인). 실행 대상은 DSR뿐이다(앞 모델 회귀는 보류).

| 대상 | 결과 | 로그 |
|---|---|---|
| `tests\17_dsr` | 26 passed (순수 14, Lightning 12, 46초) | `17-pytest-261028-1.log` |
| `tests\common\test_cli.py -k dsr` | 4 passed (39 deselected) | `17-pytest-cli-261028-1.log` |
| `tests\common\test_dependency.py` (전체 파일) | 6 passed | `17-pytest-dependency-261028-1.log` |
| 사용자 승인 회귀 실행(2026-10-28): `tests\common` | 63 passed (이전 59개에 dsr 4개 추가) | `17-regress-common-261028-1.log` |
| `compare_outputs.py --model dsr` (1차, bottle 체크포인트) | PASS 52, FAIL 3, INFO 2 (스크립트 결함과 GPU 잡음, 4.2절) | `17-compare-outputs-mvtec-261028-1.log` |
| `compare_outputs.py --model dsr` (1차, candle 체크포인트) | PASS 46, FAIL 9, INFO 2 (같은 원인) | `17-compare-outputs-visa-261028-1.log` |
| `compare_outputs.py --model dsr` (수정 후 재실행, bottle 체크포인트) | PASS 55, FAIL 0, INFO 32 | `17-compare-outputs2-mvtec-261028-1.log` |
| `compare_outputs.py --model dsr` (수정 후 재실행, candle 체크포인트) | PASS 55, FAIL 0, INFO 32 | `17-compare-outputs2-visa-261028-1.log` |

수정 후 `compare_outputs`의 PASS 항목은 FR-6(생성자 6개), VQ 주입(로컬 파일과 키·값 모두 일치), FR-5의 mvtec·visa 실제 샘플 8장(256x256, `[0, 1]` 입력) 평가 forward(`anomaly_map [8,256,256]`, `pred_score [8]`)와 CPU에서의 2단계(출력 딕셔너리 전부, 손실)·3단계(`anomaly_map`, 손실) 학습 forward, 증강(`DsrAnomalyGenerator`와 `PerlinAnomalyGenerator`), 구성요소, FR-24와 PostProcessor 대조이고 CPU 판정 32개 비교의 최대 차이는 모두 0.0이다. INFO는 시드 42 초기 상태 일치(equal=True), 레퍼런스 체크포인트의 `discrete_latent_model` 상태와 로컬 파일 일치(equal=True, 키 40개), GPU 비교 30개다. GPU 비교의 최대 차이는 1.1e-3(`anomaly_map`)이고 candle 체크포인트로 mvtec 샘플을 비교한 `embedding_bot`에서 0.112가 한 번 나왔다. VQ argmin의 근접 동점이 GPU 연산의 1e-6 수준 잡음으로 뒤집히는 GPU 비결정성으로 추정하며, CPU에서는 같은 비교가 0.0이다. 학습 step 자체의 비교는 레퍼런스 Lightning 모듈을 돌려야 해서 만들지 않았다(보고서 limitations에 기록: 단계 전환을 포함한 학습 결과는 AUROC와 epoch 손실 표가 대신한다).

`test_dependency.py`는 모든 순수 모듈의 import 분리 검사를 한 파일이 하므로 이번 Phase에서 전체를 한 번 실행했다. 이 전체 실행은 사용자가 사후에 승인했다(2026-10-28).

실행한 명령은 다음과 같다(Claude가 보낸 지시 원문). agy는 첫 명령에서만 `cd`, `$py`, 첫 `pytest`를 한 명령으로 합쳐 승인 창을 띄웠고 Claude가 지시와 같음을 읽고 한 번만 승인했다(영구 허용 없음). 나머지 명령은 승인 창 없이 이어 실행되었다. 이 때문에 개별 명령을 승인 창에서 대조하지 못했고 agy의 기록 파일 대조는 권한 문제로 하지 못했다. 로그 파일 이름, 출력, 산출물 위치가 지시와 일치함은 확인했다. `compare_outputs` 재실행의 첫 명령도 승인 창을 읽고 한 번 승인했다.

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py -m pytest tests\17_dsr 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\17-pytest-261028-1.log
& $py -m pytest tests\common\test_cli.py -k dsr 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\17-pytest-cli-261028-1.log
& $py -m pytest tests\common\test_dependency.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\17-pytest-dependency-261028-1.log
& $py evaluation/visionlab/train.py train --data mvtec --model dsr 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\17-native-mvtec-261028-1.log
& $py evaluation/visionlab/predict.py --model dsr --data mvtec 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\17-pure-mvtec-261028-1.log
& $py evaluation/visionlab/train.py train --data visa --model dsr 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\17-native-visa-261028-1.log
& $py evaluation/visionlab/predict.py --model dsr --data visa 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\17-pure-visa-261028-1.log
& $py evaluation/anomalib/compare_outputs.py --model dsr --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\LocalDsr\LocalMVTecAD\bottle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\17-compare-outputs-mvtec-261028-1.log
& $py evaluation/anomalib/compare_outputs.py --model dsr --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\LocalDsr\LocalVisa\candle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\17-compare-outputs-visa-261028-1.log
& $py evaluation/anomalib/compare_history.py --reference D:\projects\visionlab\outputs\anomalib\LocalDsr\LocalMVTecAD\bottle\v0 --native D:\projects\visionlab\outputs\visionlab\dsr\mvtec_bottle\v0 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\17-history-mvtec-261028-1.log
& $py evaluation/anomalib/compare_history.py --reference D:\projects\visionlab\outputs\anomalib\LocalDsr\LocalVisa\candle\v0 --native D:\projects\visionlab\outputs\visionlab\dsr\visa_candle\v0 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\17-history-visa-261028-1.log
& $py evaluation/anomalib/compare_outputs.py --model dsr --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\LocalDsr\LocalMVTecAD\bottle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\17-compare-outputs2-mvtec-261028-1.log
& $py evaluation/anomalib/compare_outputs.py --model dsr --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\LocalDsr\LocalVisa\candle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\17-compare-outputs2-visa-261028-1.log
```

사후 회귀 명령은 다음과 같다(승인 창 없이 실행되어 로그 이름과 출력으로 확인).

```powershell
& $py -m pytest tests\common 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\17-regress-common-261028-1.log
```

agy는 파일을 고치지 않았다.

### 5.3 에폭별 학습 히스토리 비교 (허용 오차 없음, 비교표만)

다음은 `train_loss_epoch`의 10 epoch 비교이다. 판정에 쓰지 않는다. 상대 차이는 mvtec 최대 3.6e-7, visa 최대 4.6e-7이다. 2단계(epoch 0~6)와 3단계(epoch 7~9)는 손실 정의가 달라(2단계는 MSE 세 개와 focal의 합, 3단계는 focal만) epoch 7에서 값이 크게 떨어지는 불연속이 있다. 이 불연속은 레퍼런스와 native가 같은 위치(`second_phase=7`)에서 보인다.

| Epoch | 단계 | mvtec 레퍼런스 | mvtec native | visa 레퍼런스 | visa native |
|---:|---|---:|---:|---:|---:|
| 0 | 2 | 1.846074 | 1.846074 | 0.336757 | 0.336757 |
| 1 | 2 | 0.323705 | 0.323705 | 0.055728 | 0.055728 |
| 2 | 2 | 0.190911 | 0.190911 | 0.036526 | 0.036526 |
| 3 | 2 | 0.121867 | 0.121867 | 0.028379 | 0.028379 |
| 4 | 2 | 0.086784 | 0.086784 | 0.024712 | 0.024712 |
| 5 | 2 | 0.069038 | 0.069038 | 0.021881 | 0.021881 |
| 6 | 2 | 0.061900 | 0.061900 | 0.024869 | 0.024869 |
| 7 | 3 (불연속) | 0.000655 | 0.000655 | 0.000354 | 0.000354 |
| 8 | 3 | 0.000373 | 0.000373 | 0.000287 | 0.000287 |
| 9 | 3 | 0.000302 | 0.000302 | 0.000271 | 0.000271 |

난수 사용이 많은 합성 이상(Perlin 노이즈, `RandomAffine`, 잠재 이상 강도 `torch.rand`)과 argmin 양자화가 있는데도 손실이 10 epoch 내내 float32 단위 오차 수준으로 일치한다. 초기 가중치, 난수 소비 순서, 데이터 순서, 두 optimizer의 단계 전환 시점(epoch 7)이 같다고 볼 수 있다.

### 5.4 crop·입력 크기

이 모델은 crop이 없고 입력은 256x256, `[0, 1]`(정규화 없음)이다. 평가 출력은 `anomaly_map [B,256,256]`, `pred_score [B]`다.

## 6. 결론

- **레퍼런스:** bottle, visa 확보(10 epoch, 원본 `Dsr`, 로컬 VQ 가중치).
- **mvtec / bottle:** Image AUROC(D-24 정의) 차이 0, Pixel AUROC 차이 6e-8. 통과.
- **visa / candle:** Image AUROC 차이 2.8e-4, Pixel AUROC 차이 2.8e-6. 통과.
- **FR-5, FR-6, FR-24:** 통과(5.2절). 평가 forward의 최대 차이 0.0, CPU 단계 forward 32개 비교의 최대 차이 0.0이다. 시드 42 초기 상태가 같고 레퍼런스 체크포인트의 VQ 상태가 로컬 파일과 같다. FR-23은 VQ 주입 일치로 확인했다.
- **학습 히스토리:** `train_loss_epoch`가 10 epoch 모두 상대 4.6e-7 이하로 일치한다(5.3절).
- **에폭:** 레퍼런스와 native 모두 실제 10 epoch임을 `run.json`과 로그의 `max_epochs=10 reached`로 확인했다. 단계 전환은 epoch 7이다(손실 불연속의 위치로 확인).
- **테스트:** `tests\17_dsr` 26개, `test_cli.py -k dsr` 4개, `test_dependency.py` 6개가 통과했다. 커밋 전 사용자 승인 회귀 실행(2026-10-28)으로 `tests\common` 63개도 통과했다. 01~16 모델별 회귀는 사용자 결정으로 돌리지 않았다.
- **수정 요청:** 17-A 0회, 17-B 1회.
- **`specs/` 변경:** 사용자 결정(2026-10-28)에 따라 `SPEC.md` FR-4에 DSR 예외를 더하고, `DECISIONS.md`의 D-15 트리에 dsr 파일을 추가하고 D-23을 "학습 전용 모델 패키지의 `loss.py`에서 kornia 허용"으로 일반화했다(6.1절). 새 D 번호를 만들지 않았고 INTENT와 PLAN은 고치지 않았다.
- **한계:** 낮은 AUROC(visa Image 0.556, Pixel 0.581)는 이 모델과 공유 10 epoch 예산의 결과로 보이며 원본이 의도한 긴 학습에서의 성능은 확인하지 않았다. 레퍼런스 raw 값은 구하지 않았다.
- **Phase 종료:** 사용자가 종료를 결정했다(2026-10-28). `Phase 17: dsr ...` 커밋으로 푸시한다.

### 6.1 `specs/` 갱신 항목

`specs/`는 사람이 쓴다. 최종 보고에 대한 사용자 결정(2026-10-28)에 따라 아래와 같이 처리했다. INTENT와 PLAN은 고치지 않았고 새 D 번호를 만들지 않았다.

- **FR-4 (반영 완료):** `SPEC.md` FR-4의 CSFlow 문장 뒤에 "DSR은 원본의 평가 `forward`를 유지해 anomaly map이 `[B,H,W]`, score가 `[B]`(batch 1에서도 `[1]`)다"를 더하고 판정 방법 문구에 DSR을 포함했다.
- **D-15 트리 (반영 완료):** `tests/17_dsr/`(두 파일), `configs/models`의 `dsr.yaml`, 모델 패키지 `models/dsr/`(`__init__.py`, `anomaly_generator.py`, `lightning_model.py`, `loss.py`, `model.py`)를 추가했다.
- **D-23 일반화 (반영 완료, 선택지 b):** 제목, 선택, 적용 범위를 "학습 전용 모델 패키지의 `loss.py`에서 kornia 허용"으로 고쳤다. 코드로 확인한 현재 kornia import는 `components/filters/blur.py`, `models/draem/loss.py`, `models/dsr/loss.py` 세 파일이다. 두 `loss.py`는 각 모델의 `lightning_model.py`만 import하고(`draem/lightning_model.py`, `dsr/lightning_model.py`), `Predictor`와 export 경로(각 `model.py`, `__init__.py`, `utils/io.py`)는 import하지 않으며 두 `__init__.py`는 모델 클래스만 노출한다. INTENT와의 긴장 관계 서술은 유지했다.

### 6.2 남은 위험과 사용자 판단 대상

- **GPU 비결정성:** GPU에서의 2단계·3단계 forward 비교는 VQ argmin의 근접 동점이 뒤집혀 일부 항목에서 최대 0.112까지 어긋나는 실행이 있었다(INFO). 학습 자체는 손실이 10 epoch 내내 일치했으나 다른 GPU나 환경에서도 같다고 보장하지 않는다.
- **레퍼런스 raw 미확인:** mvtec Image AUROC의 raw(0.844)와 정규화(0.667)가 크게 달라 정규화 AUROC의 변별력이 낮다. 학습 손실이 10 epoch 내내 일치하고 평가 forward가 같은 상태에서 0.0 차이인 점이 보완 증거다. 레퍼런스 체크포인트의 raw AUROC를 구하는 추가 확인은 하지 않았다(Phase 16에서는 승인되어 일치를 확인했다).
- **원본이 의도한 긴 학습에서의 성능은 확인하지 않았다.** 공유 10 epoch(2단계 7 epoch, 3단계 3 epoch)의 결과만 비교했다.
- **일부 agy 명령의 승인 대조 누락.** 5.2절에 적었다.
- **`test_dsr_lightning.py` 본문은 줄 단위로 읽지 않았다**(실행 결과로 판정). `compare_outputs.py`의 dsr 분기도 구조만 훑었다.
- **앞 Phase 레퍼런스 오염 가능성(후속 확인 과제):** Phase 16에서 정적 grep으로 확인한 대로 `nn.Module` backbone을 공유 `TimmFeatureExtractor`에 넘기는 stock 모델은 CSFlow뿐이다. DSR에는 해당되지 않는다.
- **사후 실행:** 사용자가 승인한 `tests\common\test_dependency.py` 전체 실행은 6개, `tests\common` 전체는 63개가 통과했다(5.2절, 사용자 승인 회귀 실행 2026-10-28). 이번 Phase에서 앞 Phase 테스트 파일은 수정하지 않았다.
- **앞 모델 회귀(01~16) 보류(사용자 결정):** 돌리지 않았다. 변경은 기존 항목을 건드리지 않는 등록과 분기 추가이고 `predictor.py`의 기존 분기는 그대로다.

| 공유 파일 | 변경 | 영향받을 수 있는 앞 모델 |
|---|---|---|
| `src/visionlab/utils/io.py` | `load_export`가 허용하는 이름에 `dsr` 추가 | 모든 모델의 export 로딩 |
| `src/visionlab/inference/predictor.py` | `DsrModel` import와 `models` 사전 등록 두 줄(기존 분기는 그대로) | 모든 모델의 `Predictor.from_export` |
| `evaluation/visionlab/predict.py` | `--model` choices에 `dsr` 추가 | 없음(선택지 추가) |
| `evaluation/anomalib/compare_outputs.py` | `--model dsr` 분기 추가 | 없음(기존 분기 유지) |
| `tests/common/test_cli.py` | dsr 테스트만 추가 | 없음 |
| `tests/common/test_dependency.py` | 순수 모듈 3개 분류 두 줄과 Lightning 허용 경로 한 줄 | 분류 규칙 전체(`tests\common` 63 passed) |
| `paths.toml` | `[teachers]`에 `dsr_vq` 한 줄 | 없음 |
