# Phase 16. GANomaly

모델 폴더 이름은 `ganomaly`이고 원본은 anomalib 2.0.0의 `anomalib\models\image\ganomaly\`(`C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\Lib\site-packages\anomalib\models\image\ganomaly\`)이다. 범위와 완료 조건은 `specs/PLAN.md`의 Phase 16과 "Phase 공통 양식"을 따른다. 판정 기준은 category별 Image AUROC 차이 0.01 이내다. 이 모델은 image metric만 내므로 Pixel AUROC는 판정하지 않는다.

## 상태

다음 표는 단계별 진행 상태다. 수정 반복 한도 3회는 단계별로 센다.

| 단계 | 내용 | 상태 |
|---|---|---|
| 레퍼런스 | `anomalib_run.py train` x 2 (mvtec/bottle 1회, visa/candle 1회, 10 epoch, 데이터 batch 32) | 완료 |
| 16-A | 순수 모델 (`models/ganomaly/{model,loss}.py`, `tests/16_ganomaly`의 순수 부분) | 완료 (수정 요청 0회) |
| 16-B | Lightning (`GanomalyLitModel`, 수동 최적화 두 optimizer, configs, export·Predictor 등록, 비교 스크립트) | 완료 (수정 요청 1회) |
| 판정 | Image AUROC 차이 0.01 이내 (두 데이터) | 통과 |
| 추가 확인 | 사용자 승인 추가 확인(2026-10-28): 레퍼런스 체크포인트의 raw Image AUROC | 완료 (native raw와 같음) |
| 사후 회귀 | 사용자 승인 회귀 실행(2026-10-28): `tests\common` | 완료 (59 passed) |
| specs 정리 | FR-4 `score [B]`, D-15 트리 (사용자 결정) | 완료 |
| Phase 종료 | 사용자 결정과 커밋·푸시 | 사용자 결정(2026-10-28), 커밋·푸시 |

`InferenceBatch`, `PreProcessor`(ImageNet 정규화), `PostProcessor`, `Evaluator`(image 전용 구성, DFKDE 선례), 수동 최적화 골격(CFLOW 선례)은 앞 Phase의 것을 재사용한다.

## 사용자 결정 (2026-10-28)

다음은 Phase 시작 전에 사용자가 정한 항목이다. 계획 보고의 제안 1~6을 그대로 채택했다.

| 항목 | 결정 |
|---|---|
| 학습 에폭 | 원본 `trainer_arguments`는 `{"gradient_clip_val": 0, "num_sanity_val_steps": 0}`이고 `max_epochs`를 고정하지 않으므로 D-22의 첫 문장대로 공유 10 epoch를 레퍼런스와 native에 똑같이 쓴다. 예외는 없다. 실제 10 epoch임을 `run.json`과 로그의 `max_epochs=10 reached`로 확인해 기록한다 |
| 데이터 batch | 32를 유지한다(`Ganomaly`의 `batch_size=32` 생성자 기본값과 레퍼런스 yaml에 맞춤). native 모델 yaml에도 `data:` 덮어쓰기를 둔다(EfficientAD 선례). 공유 데이터 yaml의 8/8과 다르다 |
| `specs/` 변경 | 지금은 없다. FR-4의 `pred_score [B]` 명시 여부와 D-15 트리(`tests/16_ganomaly/`, `configs/models/ganomaly.yaml`, `models/ganomaly/`)는 6.1절에 초안으로 적고 최종 보고에서 묻는다. 새 D 번호는 필요 없다 |
| 앞 Phase 레퍼런스 오염 후속 확인 | 별도 실행은 하지 않는다. 정적 grep으로 확인한 내용을 6.2절에 적는다 |
| 시간·category | 변경 없음. 전체 예상이 약 45분을 넘을 것으로 보이면 계속하기 전에 보고한다. category는 mvtec/bottle과 visa/candle이며 스스로 바꾸지 않는다 |
| 앞 모델 회귀(01~15) | 보류. 돌리지 않는다. 영향 표는 6.2절에 남긴다. 최종 보고에서 사후 `tests\common` 1회를 묻는다. `test_dependency.py` 전체 실행이 필요하면 사후 승인 항목으로 올린다 |
| codex 세션 | 새 세션으로 시작한다 |
| FR-5 허용 오차 | `atol=1e-6, rtol=1e-5` (Phase 2~15와 같음) |
| 학습 히스토리 | `generator_loss_epoch`, `discriminator_loss_epoch`의 10 epoch 표만 기록한다(허용 오차 없음) |
| 난수가 어긋날 때의 판정 | 난수 상태를 주입하지 않고 AUROC 0.01로만 판정한다. 적대적 학습이라 레퍼런스와 native가 0.01 넘게 어긋나면 스스로 처리하지 않고 멈춰 보고한다(반복 실행, 판정 완화, 미달 기록은 사용자가 정한다) |
| 판정 | Image AUROC를 판정한다. 정규화 점수 기준 값(D-24)과 raw 값을 함께 기록한다 |
| 실행 범위 | GANomaly의 명령만 실행한다 |
| 최종 보고 뒤 결정 (2026-10-28) | 결정 1~4를 권고안대로 채택했다. (1) specs 초안을 반영한다(FR-4에 GANomaly `score [B]`를 최소한으로 명시하고 기존 "image 전용 모델은 anomaly map 없이 score만 낸다" 문장을 살린다. D-15 트리에 `tests/16_ganomaly/`, `configs/models/ganomaly.yaml`, `models/ganomaly/` 네 파일을 추가한다. 새 D 번호는 만들지 않고 INTENT와 PLAN은 고치지 않는다). (2) 레퍼런스 체크포인트(mvtec, visa)를 순수 `GanomalyModel`에 불러 raw 순서 보존 AUROC를 native raw와 비교하는 읽기 전용 추가 확인을 한다(codex 스크립트, agy 실행, 학습 없음). 같지 않으면 멈추고 보고한다. (3) 커밋 전 `tests\common` 1회를 사후 실행한다(01~15 모델별 회귀는 계속 보류). (4) Phase 종료를 결정했고 커밋·푸시한다 |

## 1. 모델 노트

### 1.1 원본과 생성자

다음 표는 원본 파일과 이식 위치의 대응이다(D-15, D-18).

| 원본 | 이식 위치 | 비고 |
|---|---|---|
| `models/image/ganomaly/torch_model.py` (`Encoder`, `Decoder`, `Discriminator`, `Generator`, `GanomalyModel`) | `src/visionlab/anomaly_detection/models/ganomaly/model.py` | import 변경, `pad_nextpow2`를 모델 파일 안에 복사(anomalib 의존 제거) |
| `models/image/ganomaly/loss.py` (`GeneratorLoss`, `DiscriminatorLoss`) | `.../models/ganomaly/loss.py` | 변경 없음 |
| `models/image/ganomaly/lightning_model.py` (`Ganomaly`) | `.../models/ganomaly/lightning_model.py` (`GanomalyLitModel`) | native `LightningModule`로 새로 작성 |
| `models/image/ganomaly/__init__.py` | `.../models/ganomaly/__init__.py` | `lightning_model`을 import하지 않도록 고침(D-18 예외) |

생성자는 다음과 같다. `GanomalyLitModel`은 `Ganomaly`와 인자 이름·순서·기본값이 같아야 한다(FR-6, D-5).

| 클래스 | 시그니처 |
|---|---|
| `Ganomaly` | `batch_size=32`, `n_features=64`, `latent_vec_size=100`, `extra_layers=0`, `add_final_conv_layer=True`, `wadv=1`, `wcon=50`, `wenc=1`, `lr=0.0002`, `beta1=0.5`, `beta2=0.999`, `pre_processor=True`, `post_processor=True`, `evaluator=True`, `visualizer=True` (15개) |
| `GanomalyModel` | `input_size`, `num_input_channels`, `n_features`, `latent_vec_size`, `extra_layers=0`, `add_final_conv_layer=True` (6개, 앞 네 개는 기본값이 없다) |
| `GeneratorLoss` | `wadv=1`, `wcon=50`, `wenc=1` |
| `DiscriminatorLoss` | 인자 없음 |

`input_size`는 `Ganomaly`의 생성자 인자가 아니다. anomalib에서는 pre-processor의 Resize 크기(기본 256)에서 얻는다. `GanomalyLitModel`은 `(256, 256)` 상수를 쓴다. `batch_size`는 사용되지 않는 `real_label`, `fake_label` 텐서의 크기에만 쓰이고 데이터 batch를 정하지 않는다.

### 1.2 학습과 평가 동작

원본의 동작은 다음과 같다.

- **구조:** `Generator`는 `encoder1`, `decoder`, `encoder2`로 이루어진 인코더-디코더-인코더이고, `Discriminator`는 `Encoder`를 임시로 만들어 하위 층을 펼친 뒤 마지막 층을 분류기(Sigmoid)로 둔다.
- **학습(수동 최적화):** optimizer 둘(`Adam`, lr 2e-4, betas (0.5, 0.999))을 `[optimizer_d, optimizer_g]`로 반환한다. `training_step`은 한 번의 forward로 `(padded, fake, latent_i, latent_o)`를 얻고, 판별기에 `padded`와 `fake`를 넣어 G 손실을 만든 뒤 `g_opt.zero_grad()`, `manual_backward(g_loss, retain_graph=True)`, `g_opt.step()` 순으로 갱신한다. 이어서 `fake.detach()`로 판별기를 다시 평가해 D 손실을 만들고 `d_opt.zero_grad()`, `manual_backward(d_loss)`, `d_opt.step()`한다.
- **손실:** G 손실 = `MSE(pred_real, pred_fake) * wadv + L1(images, fake) * wcon + SmoothL1(latent_i, latent_o) * wenc`. D 손실 = `(BCE(pred_real, 1) + BCE(pred_fake, 0)) * 0.5`.
- **평가 `forward`:** `pred_score = mean((latent_i - latent_o)^2, dim=1).view(-1)`이다(`[B]`, anomaly map 없음).
- **로그:** `generator_loss`와 `discriminator_loss`를 `on_epoch=True`로 기록한다(`.item()`). batch가 Batch 객체라 Lightning이 batch 크기를 추론하지 못해 epoch 평균은 step마다 1회씩 센 평균이다(CFLOW와 같음). native는 `batch_size=1`로 맞춘다.
- **`trainer_arguments`:** `{"gradient_clip_val": 0, "num_sanity_val_steps": 0}`이며 `max_epochs`는 고정하지 않는다(공유 10 epoch).
- **평가 구성:** `configure_evaluator`가 test 지표로 Image AUROC와 Image F1만 둔다(val 지표 없음).
- **min/max 정규화:** 원본은 `on_validation_batch_end`, `on_test_batch_end`에서 `pred_score`를 running min/max로 정규화한다. Lightning은 callback(`Evaluator`, `PostProcessor`)을 모듈 hook보다 먼저 부르고 `Evaluator`가 batch 객체를 직접 읽으므로 이 정규화는 지표에 영향이 없다고 코드 읽기로 판단했다. 실행 결과(AUROC 일치)로 확인한다.
- **원본의 주의점:** `weights_init(self.generator)`는 모듈 자체에 호출되고 클래스 이름이 `Generator`라 아무것도 하지 않는다(PyTorch 기본 초기화가 남음). `add_final_conv_layer=False`이면 `final_conv_layer` 속성이 없어 `forward`가 `AttributeError`를 낸다.

### 1.3 외부 의존과 로컬 자산

- **외부 의존:** 순수 torch뿐이다. `anomalib.data.utils.image.pad_nextpow2`만 anomalib 의존이라 모델 파일 안에 복사한다. FrEIA, timm, torchvision, kornia, sklearn은 쓰지 않는다.
- **로컬 가중치:** 없다(사전학습 가중치를 쓰지 않는다). `paths.toml`을 바꾸지 않는다.
- **데이터:** MVTec `bottle`(학습 209장), VisA `candle`(학습 900장). 분할은 `from_test`, 0.4, seed 42, batch는 학습·평가 모두 32(이 모델 yaml이 덮어씀), `num_workers=0`이다.
- **학습 예산:** 공유 yaml의 `max_epochs=10`이다.
- **규모(코드 읽기 추정):** 기본 구성(256 입력, `n_features=64`)의 파라미터는 약 1.9억 개다(인코더와 디코더 각 약 4.8천만, 생성기 세 개 약 1.44억, 판별기 약 4.5천만). 모델 상태가 약 0.76GB이고 optimizer 둘을 포함한 체크포인트는 2GB 이상이다. 계획 보고에서 모델이 작다고 한 말은 틀렸다.

### 1.4 이식에서 의도적으로 달라지는 곳과 유지하는 곳

- **anomalib 의존 제거:** `InferenceBatch`는 `visionlab.anomaly_detection.common.batch`, `pad_nextpow2`는 모델 파일 안에 복사.
- **원본 동작 유지(고치지 않음):** 모듈 이름과 생성 순서(`state_dict` 키와 난수 소비 순서를 결정), 효과 없는 `weights_init`, `Discriminator`가 임시 `Encoder`를 만드는 방식, G 손실 backward 뒤 D 손실을 만드는 step 순서, `train_loss`가 아니라 `generator_loss`와 `discriminator_loss`를 `on_epoch`로 기록하는 점, running min/max 정규화 코드.
- **FR-5 허용 오차:** `pred_score`는 `atol=1e-6, rtol=1e-5`다.

## 2. 레퍼런스

학습 예산은 공유 `max_epochs=10`이고 데이터 batch는 32다(1.2절). 입력은 anomalib 기본 pre-processor(`Resize(256)`과 ImageNet 정규화)다. 어댑터와 가중치 등록은 없다(stock `anomalib.models.Ganomaly`).

### 2.1 레퍼런스 실행

2026-10-28에 agy가 `anomalib_run.py train --config default.yaml --config data/<data>.yaml --config models/16_ganomaly.yaml`을 데이터셋마다 1회 실행했다(실행으로 검증함). Claude는 로그 파일에서 값을 직접 확인했다. 두 로그 모두 `Traceback`, `out of memory`, `URLError`가 0건이고 `Trainer.fit stopped: max_epochs=10 reached`다. 로그는 UTF-16이다. visa 명령의 승인 창은 지시한 줄과 정확히 같음을 읽고 한 번만 승인했다(영구 허용 없음). mvtec 명령은 승인 창 없이 실행되어 로그 이름으로만 확인했다.

agy가 PowerShell에서 실행한 명령은 다음과 같다(Claude가 보낸 지시 원문. 파일은 고치지 않았다).

```powershell
cd D:\projects\visionlab\evaluation\anomalib
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/16_ganomaly.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\16-ref-mvtec-261028-1.log
& $py anomalib_run.py train --config default.yaml --config data/visa.yaml --config models/16_ganomaly.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\16-ref-visa-261028-1.log
```

다음 표는 레퍼런스(1회) 결과다. 레퍼런스가 보고하는 Image AUROC는 PostProcessor로 정규화하고 clamp한 점수로 계산한 값이다(D-24).

| 데이터 / category | Image AUROC | Image F1 | 에폭당 시간(진행 바) | run 폴더 | 로그 |
|---|---|---|---|---|---|
| mvtec / bottle | 0.44736841 | 0.86363637 | 약 13초 | `outputs/anomalib/Ganomaly/LocalMVTecAD/bottle/v0` | `outputs/logs/16-ref-mvtec-261028-1.log` |
| visa / candle | 0.75694442 | 0.74666667 | 약 48초 | `outputs/anomalib/Ganomaly/LocalVisa/candle/v0` | `outputs/logs/16-ref-visa-261028-1.log` |

적용된 학습 예산은 `max_epochs=10`, `val_split_mode=from_test`, `val_split_ratio=0.4`, `seed=42`, `batch 32/32`, `num_workers=0`, `n_features=64`, `latent_vec_size=100`, `extra_layers=0`, `wadv=1`, `wcon=50`, `wenc=1`, `lr=0.0002`, `beta=(0.5, 0.999)`다. 체크포인트는 각 run 폴더의 `weights/lightning/model.ckpt`이고 크기는 2,264,521,336 bytes(optimizer 둘 포함)다.

### 2.2 학습 시간과 에폭 확인

| 대상 | 학습 시간 | 근거 |
|---|---|---|
| 레퍼런스 mvtec/bottle | 약 2분(에폭당 약 13초) | `max_epochs=10 reached` |
| 레퍼런스 visa/candle | 약 8분(에폭당 약 48초) | `max_epochs=10 reached` |
| native mvtec/bottle | 약 4분 전체(학습과 평가) | `run.json`의 `max_epochs` 10, `gradient_clip_val` 0.0, `num_sanity_val_steps` 0, seed 42, 데이터 batch 32 |
| native visa/candle | 약 10분 전체 | 같음 |

계획의 추정(agy 실행 15~25분)과 45분 관문을 넘지 않았다. 파일 시각으로 agy 실행 묶음 전체(pytest, native 학습, 추론, 비교, 히스토리)가 약 17분이었다.

### 2.3 레퍼런스와 native의 에폭 확인

| 대상 | 근거 | 값 |
|---|---|---|
| 레퍼런스 (bottle, visa) | 로그의 `Trainer.fit stopped: max_epochs=10 reached` | 10 |
| native (bottle, visa) | `run.json`의 `max_epochs` 10, 로그의 `max_epochs=10 reached` | 10 |

## 3. 이식 메모

### 3.1 원본과 이식 파일의 대응

다음 표는 새로 만든 파일과 원본의 대응이다. 원본과의 차이는 `difflib`로 확인했다(구현함, 정적 검증).

| 이식 파일 (`src/visionlab/anomaly_detection/models/ganomaly/`) | 원본 | 원본과 다른 줄 |
|---|---|---|
| `model.py` | `torch_model.py` | import 2줄(`InferenceBatch`, `torch.nn.functional`)과 `pad_nextpow2` 함수 복사(원본 `anomalib.data.utils.image`, 주석 한 줄) |
| `loss.py` | `loss.py` | 없음 |
| `__init__.py` | `__init__.py` | `GanomalyModel`만 노출(D-18 예외, Lightning import 제거) |
| `lightning_model.py` (`GanomalyLitModel`) | `lightning_model.py` | native `LightningModule`로 새로 작성. 15개 인자 동일, `(256, 256)` 상수, 수동 최적화 순서와 optimizer 둘, `generator_loss`·`discriminator_loss` 로그(`batch_size=1`), running min/max 정규화 코드(지표에 영향 없음), image 전용 `configure_evaluator` |

### 3.2 새로 쓴 코드와 앞 Phase 공유 파일 변경

- **새로 쓴 파일:** `configs/models/ganomaly.yaml`, `models/ganomaly/`(네 파일), `tests/16_ganomaly/test_ganomaly.py`, `tests/16_ganomaly/test_ganomaly_lightning.py`(합계 30개).
- **추가 확인 스크립트:** `evaluation/anomalib/check_ganomaly_raw_auroc.py`(읽기 전용, codex가 작성. 사용자 승인 추가 확인용, 5.5절).
- **설정 변경:** 없다(`paths.toml`, 레퍼런스 yaml은 그대로. 레퍼런스 yaml `16_ganomaly.yaml`은 이 Phase 전에 이미 있었다).
- **공유 파일 변경:** `utils/io.py`(`load_export`가 `ganomaly` 허용), `inference/predictor.py`(일반 `models` 사전에 `GanomalyModel` 등록, ganomaly일 때만 `input_size`를 tuple로 변환), `evaluation/visionlab/predict.py`(`--model ganomaly`), `evaluation/anomalib/compare_outputs.py`(`--model ganomaly` 분기), `tests/common/test_cli.py`(ganomaly 테스트 추가), `tests/common/test_dependency.py`(순수 모듈 2개 분류와 Lightning 허용 경로 한 줄). `common/`와 `components/`는 바꾸지 않았다. 영향 범위는 6.2절의 표에 정리했다.
- **앞 Phase 테스트 변경:** 없다(이 모델은 FrEIA를 쓰지 않는다).
- **새 부품:** `pad_nextpow2`를 모델 파일 안에 복사했다(anomalib 의존 제거).

### 3.3 이식 중 확인된 사항

- **규모:** 기본 구성(256 입력, `n_features=64`)의 파라미터는 188,689,536개다(codex가 계층별 합산, 모델을 만들지 않음). 인코더의 초기 convolution 출력은 131x131, 피라미드는 5단계(65, 32, 16, 8, 4), 마지막 convolution 직전은 `[N, 2048, 4, 4]`이고 잠재 벡터는 `[N, 100, 1, 1]`이다. 모델 상태가 약 0.75GB(export 754,924,606 bytes), 체크포인트가 약 2.26GB다.
- **효과 없는 `weights_init`:** 원본 `weights_init(self.generator)`가 모듈 자체에 호출되고 클래스 이름이 `Generator`라 아무것도 하지 않는다. 기본 PyTorch 초기화가 남는다. 원본 그대로 복사했고 테스트가 이 사실을 확인한다. 같은 seed 42에서 레퍼런스와 visionlab 순수 모델의 초기 `state_dict`가 같다(`compare_outputs`의 INFO).
- **효과 없는 min/max 정규화:** 원본의 `on_*_batch_end`에서의 running min/max 정규화는 Lightning이 callback을 모듈 hook보다 먼저 부르므로 지표에 영향이 없다고 판단했다. native는 evaluator·post_processor 갱신이 끝난 뒤에만 정규화를 두어 같은 순서를 지켰다. 테스트가 `trainer.test`의 `image_AUROC`가 원시 점수로 직접 계산한 AUROC와 같음을 확인한다. 실행에서도 native raw 값과 순수 PyTorch raw 값이 같았다.
- **FR-23:** 해당 없다(사전학습 가중치 없음). 같은 상태 복원(`compare_outputs`의 strict 로딩)으로 대체했다.
- **`add_final_conv_layer=False`:** 원본의 `AttributeError`를 그대로 두었고 테스트가 이를 문서화한다.
- **BatchNorm 오염:** 해당 없다. 사전학습 backbone과 `TimmFeatureExtractor` 생성자 dry-run이 없다(Phase 15의 오염과 무관).

## 4. 적대적 검증

검증은 16-A, 16-B 각각 한 회차씩 했다. 수정 요청은 16-A 0회, 16-B 1회(한도 3회 이내)다.

### 4.1 16-A (순수 모델)

| 지적 | 심각도 | 처리 |
|---|---|---|
| 원본과의 `difflib` 대조: `loss.py`는 차이 없음, `model.py`는 import 2줄과 `pad_nextpow2` 복사뿐. 모듈 이름, 생성 순서, 효과 없는 `weights_init`, 임시 `Encoder` 생성까지 원본과 같음 | 확인 | 수정 없음 |
| 파라미터 수 188,689,536개와 256 입력의 피라미드 단계(5단계, 최종 4x4)를 codex가 계층별 합산으로 확인함. 계획 보고에서 모델이 작다고 한 말은 틀렸다 | 낮음 | 지시와 결과 문서에 정정 |

### 4.2 16-B (Lightning, export, 비교 스크립트)

| 지적 | 심각도 | 처리 |
|---|---|---|
| 원본 `lightning_model.py`와의 대조: 생성 순서, `[optimizer_d, optimizer_g]`, `training_step`의 순서(G 손실 backward에 `retain_graph=True`, `g_opt.step()`, 그다음 `fake.detach()`로 D 손실), `log_dict`(`batch_size=1`), `trainer_arguments`, running min/max 코드가 같음. 의도한 차이는 `(256, 256)` 상수, native 처리·체크포인트·export 연결, image 전용 evaluator뿐. `predictor.py`는 일반 등록과 `input_size` tuple 변환만 추가했고 기존 분기는 그대로임을 `git diff`로 확인 | 확인 | 수정 없음 |
| 1차 실행에서 테스트 6건 실패(`tests\16_ganomaly` 2개, `test_cli.py -k ganomaly` 4개). 원인은 모두 테스트가 `generator.encoder1.input_size`를 단언했으나 원본 `Encoder`가 `input_size`를 속성으로 저장하지 않아 `AttributeError`가 난 것이다. 모델 코드 문제가 아니라 테스트의 가정 오류 | 중간 | 수정 요청 1회: 세 곳의 단언을 `final_conv_layer.in_channels == n_features * 2**5`와 디코더 첫 convT 출력 채널 확인으로 바꿈(256 입력의 피라미드 5단계에서 나옴). 모델 코드는 바꾸지 않음. 재실행에서 30 passed, 4 passed |
| `tests/16_ganomaly/test_ganomaly_lightning.py`와 `compare_outputs.py`의 ganomaly 분기는 테스트 이름과 함수 구조를 확인했고 본문은 줄 단위로 읽지 않았다(실행 결과로 판정) | 낮음 | 수정 없음 |

## 5. 이식 결과

### 5.1 최종 판정 (10 epoch, Image AUROC 차이 0.01 이내)

native는 `train.py`가 test에서 기록한 값이고 순수 PyTorch는 `export/model.pt`만 불러 `predict.py`로 평가한 값이다. 모두 실행으로 검증했다. D-24에 따라 레퍼런스가 보고하는 값과 같은 정의(정규화하고 clamp한 점수)로 비교하며, 순수 PyTorch의 정규화 AUROC를 판정에 쓰고 raw 값은 참고로 적는다. 이 모델은 image metric만 내므로 Pixel AUROC는 판정하지 않는다(pixel 값은 `NaN`, `no_data`).

| 데이터 / category | 지표 | 레퍼런스 | native (raw) | 순수 PyTorch (raw) | 순수 PyTorch (정규화) | 차이(순수 정규화) | 판정 |
|---|---|---:|---:|---:|---:|---:|---|
| mvtec / bottle | Image AUROC | 0.44736841 | 0.20614035 | 0.20614035 | 0.44736841 | 0 | 통과 |
| visa / candle | Image AUROC | 0.75694442 | 0.75444448 | 0.75444448 | 0.75694442 | 0 | 통과 |

Image F1은 레퍼런스, native, 순수 PyTorch가 같다(mvtec 0.86363637, visa 0.74666667).

두 정의의 값이 0.01 넘게 다르면 원인을 적어야 한다(D-24). mvtec은 raw 0.20614와 정규화 0.44737이 0.2412 다르고 visa는 0.75444와 0.75694가 0.0025 다르다. mvtec의 원인은 PostProcessor의 임계값 중심 정규화 뒤 `clamp(0, 1)`에서 대부분의 점수가 0 또는 1로 잘려 동점이 생기기 때문이다(DFM visa와 같은 현상). raw 0.206이 0.5보다 훨씬 작다는 것은 10 epoch 학습된 이 모델의 점수 순위가 거의 뒤집혀(정상 이미지의 점수가 이상 이미지보다 높음) 있다는 뜻이고, clamp 동점이 값을 0.5 쪽으로 끌어올린다. 레퍼런스의 raw 값은 구하지 않았다(D-24의 남은 한계). 정규화 AUROC는 동점 때문에 변별력이 낮아 일치가 학습 결과 일치의 강한 증거는 아니다. 보완 증거는 같은 상태에서의 출력 일치(`compare_outputs`의 최대 차이 0.0)와 Image F1 일치, 학습 손실이 상대 2.2e-7 이하로 일치한 점(5.3절)이다.

run 폴더와 로그는 다음과 같다.

| 대상 | 위치 |
|---|---|
| 레퍼런스 | `outputs/anomalib/Ganomaly/LocalMVTecAD/bottle/v0`, `outputs/anomalib/Ganomaly/LocalVisa/candle/v0` |
| native | `outputs/visionlab/ganomaly/mvtec_bottle/v0`, `outputs/visionlab/ganomaly/visa_candle/v0` (`run.json`: `max_epochs` 10, `gradient_clip_val` 0.0, `num_sanity_val_steps` 0, seed 42, 데이터 batch 32/32) |
| 순수 PyTorch | 같은 run의 `predict/metrics.json`, `export/model.pt`(754,924,606 bytes) |
| 로그 | `outputs/logs/16-native-{mvtec,visa}-261028-1.log`, `16-pure-{mvtec,visa}-261028-1.log` |

### 5.2 테스트와 비교 스크립트

다음 표는 agy가 실행한 묶음의 결과다(Claude가 로그 파일에서 확인). 실행 대상은 GANomaly뿐이다(앞 모델 회귀는 보류).

| 대상 | 결과 | 로그 |
|---|---|---|
| `tests\16_ganomaly` (1차) | 28 passed, 2 failed (테스트의 `input_size` 속성 가정, 4.2절) | `16-pytest-261028-1.log` |
| `tests\common\test_cli.py -k ganomaly` (1차) | 4 failed (같은 원인) | `16-pytest-cli-261028-1.log` |
| `tests\16_ganomaly` (수정 후 재실행) | 30 passed | `16-pytest-261028-2.log` |
| `tests\common\test_cli.py -k ganomaly` (수정 후 재실행) | 4 passed (35 deselected) | `16-pytest-cli-261028-2.log` |
| `tests\common\test_dependency.py` (전체 파일) | 6 passed | `16-pytest-dependency-261028-1.log` |
| 사용자 승인 회귀 실행(2026-10-28): `tests\common` | 59 passed (이전 55개에 ganomaly 4개 추가) | `16-regress-common-261028-1.log` |
| `compare_outputs.py --model ganomaly` (bottle 체크포인트) | PASS 42, FAIL 0, INFO 1 | `16-compare-outputs-mvtec-261028-1.log` |
| `compare_outputs.py --model ganomaly` (candle 체크포인트) | PASS 42, FAIL 0, INFO 1 | `16-compare-outputs-visa-261028-1.log` |

`compare_outputs`의 PASS 항목은 FR-6(생성자 15개), FR-5의 mvtec·visa 실제 샘플 8장(256x256, 한 batch) 평가 forward(`pred_score [8]`)와 학습 forward(`padded`, `fake`, `latent_i`, `latent_o`와 G, D 손실, 허용 오차 `atol=1e-6, rtol=1e-5`), 구성요소(`GeneratorLoss`, `DiscriminatorLoss`, `pad_nextpow2`의 여러 입력 크기), FR-24와 PostProcessor 대조이고 22개 비교의 최대 차이는 모두 0.0이다. INFO 1개는 시드 42 초기 `state_dict`의 일치(equal=True)다. FR-23은 해당 없고 학습 step 비교는 만들지 않았다(레퍼런스 Lightning 모듈을 돌려야 하므로. 수동 최적화 학습 결과는 AUROC와 epoch 손실 표가 대신한다).

`test_dependency.py`는 모든 순수 모듈의 import 분리 검사를 한 파일이 하므로 이번 Phase에서 전체를 한 번 실행했다. 이 전체 실행은 사용자가 사후에 승인했다(2026-10-28).

실행한 명령은 다음과 같다(Claude가 보낸 지시 원문). agy는 첫 명령에서만 `cd`, `$py`, 첫 `pytest`를 한 명령으로 합쳐 승인 창을 띄웠고 Claude가 지시와 같음을 읽고 한 번만 승인했다(영구 허용 없음). 나머지 명령과 수정 후 재실행은 승인 창 없이 이어 실행되었다. 이 때문에 개별 명령을 승인 창에서 대조하지 못했고 agy의 기록 파일 대조는 권한 문제로 하지 못했다. 로그 파일 이름, 출력, 산출물 위치가 지시와 일치함은 확인했다.

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py -m pytest tests\16_ganomaly 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\16-pytest-261028-1.log
& $py -m pytest tests\common\test_cli.py -k ganomaly 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\16-pytest-cli-261028-1.log
& $py -m pytest tests\common\test_dependency.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\16-pytest-dependency-261028-1.log
& $py evaluation/visionlab/train.py train --data mvtec --model ganomaly 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\16-native-mvtec-261028-1.log
& $py evaluation/visionlab/predict.py --model ganomaly --data mvtec 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\16-pure-mvtec-261028-1.log
& $py evaluation/visionlab/train.py train --data visa --model ganomaly 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\16-native-visa-261028-1.log
& $py evaluation/visionlab/predict.py --model ganomaly --data visa 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\16-pure-visa-261028-1.log
& $py evaluation/anomalib/compare_outputs.py --model ganomaly --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\Ganomaly\LocalMVTecAD\bottle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\16-compare-outputs-mvtec-261028-1.log
& $py evaluation/anomalib/compare_outputs.py --model ganomaly --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\Ganomaly\LocalVisa\candle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\16-compare-outputs-visa-261028-1.log
& $py evaluation/anomalib/compare_history.py --reference D:\projects\visionlab\outputs\anomalib\Ganomaly\LocalMVTecAD\bottle\v0 --native D:\projects\visionlab\outputs\visionlab\ganomaly\mvtec_bottle\v0 --tags generator_loss_epoch discriminator_loss_epoch 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\16-history-mvtec-261028-1.log
& $py evaluation/anomalib/compare_history.py --reference D:\projects\visionlab\outputs\anomalib\Ganomaly\LocalVisa\candle\v0 --native D:\projects\visionlab\outputs\visionlab\ganomaly\visa_candle\v0 --tags generator_loss_epoch discriminator_loss_epoch 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\16-history-visa-261028-1.log
& $py -m pytest tests\16_ganomaly 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\16-pytest-261028-2.log
& $py -m pytest tests\common\test_cli.py -k ganomaly 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\16-pytest-cli-261028-2.log
```

agy는 파일을 고치지 않았다.

### 5.3 에폭별 학습 히스토리 비교 (허용 오차 없음, 비교표만)

다음은 `generator_loss_epoch`와 `discriminator_loss_epoch`의 10 epoch 비교이다. 판정에 쓰지 않는다. 상대 차이는 생성기 손실이 mvtec 최대 1.1e-7, visa 최대 1.7e-7이고 판별기 손실이 mvtec 최대 1.8e-7, visa 최대 2.1e-7이다.

| Epoch | mvtec G 레퍼런스 | mvtec G native | mvtec D 레퍼런스 | mvtec D native | visa G 레퍼런스 | visa G native | visa D 레퍼런스 | visa D native |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 63.5554 | 63.5554 | 0.19630 | 0.19630 | 44.8777 | 44.8777 | 0.83537 | 0.83537 |
| 1 | 47.9616 | 47.9616 | 3.4412e-4 | 3.4412e-4 | 32.8313 | 32.8313 | 5.1547e-3 | 5.1547e-3 |
| 2 | 41.7929 | 41.7929 | 3.3230e-4 | 3.3230e-4 | 31.2921 | 31.2921 | 6.9543e-4 | 6.9543e-4 |
| 3 | 39.6061 | 39.6061 | 6.6896 | 6.6896 | 30.4630 | 30.4630 | 6.0034e-5 | 6.0034e-5 |
| 4 | 38.3829 | 38.3829 | 0.14321 | 0.14321 | 29.9763 | 29.9763 | 3.7847e-5 | 3.7847e-5 |
| 5 | 37.8412 | 37.8412 | 5.5212e-4 | 5.5212e-4 | 29.6928 | 29.6928 | 2.9175e-5 | 2.9175e-5 |
| 6 | 37.1295 | 37.1295 | 4.4561e-5 | 4.4561e-5 | 29.4585 | 29.4585 | 1.9897e-5 | 1.9897e-5 |
| 7 | 36.4810 | 36.4810 | 3.3823e-5 | 3.3823e-5 | 29.3123 | 29.3123 | 1.9044e-5 | 1.9044e-5 |
| 8 | 36.1527 | 36.1527 | 2.6475e-5 | 2.6475e-5 | 29.1966 | 29.1966 | 1.2859e-5 | 1.2859e-5 |
| 9 | 35.8583 | 35.8583 | 2.1628e-5 | 2.1628e-5 | 29.0355 | 29.0355 | 1.1371e-5 | 1.1371e-5 |

수동 최적화 두 optimizer의 적대적 학습이 10 epoch 내내 float32 단위 오차 수준으로 일치한다. 초기 가중치, 난수 소비 순서, 데이터 순서, optimizer 상태와 step 순서가 같다고 볼 수 있다. 판별기 손실이 크게 흔들리는 구간(mvtec epoch 3에서 6.69)까지 레퍼런스와 native가 같다.

### 5.4 crop·입력 크기

이 모델은 crop이 없고 입력은 256x256이다. 평가 출력은 `pred_score [B]`이고 anomaly map은 없다.

### 5.5 사용자 승인 추가 확인 (2026-10-28)

레퍼런스가 보고하는 값(mvtec 0.447)이 clamp 동점 아티팩트라는 설명을 확인하기 위해 사용자 승인으로 읽기 전용 확인을 했다. `check_ganomaly_raw_auroc.py`(codex가 작성)가 레퍼런스 체크포인트와 native 체크포인트를 순수 `GanomalyModel`(eval 모드)에 `strict`로 불러, native 평가와 같은 데이터 모듈(`from_test` 분할 0.4, seed 42, 평가 batch 32)의 test 이미지에서 raw 점수와 raw 순서 보존 AUROC(visionlab의 `AUROC`)를 계산한다. 학습 이미지를 읽지 않고 학습도 하지 않으며 파일을 쓰지 않는다. 같은 스크립트가 native 체크포인트로 native 로그의 raw 값을 재현하는지도 함께 확인했다.

| 실행 | test 이미지 | raw Image AUROC | 정상 점수 평균 | 이상 점수 평균 |
|---|---|---:|---:|---:|
| mvtec 레퍼런스 체크포인트 | 50장 (정상 12, 이상 38) | 0.20614035 | 3.7386 | 3.4090 |
| mvtec native 체크포인트 | 50장 | 0.20614035 | 3.7386 | 3.4090 |
| visa 레퍼런스 체크포인트 | 120장 (정상 60, 이상 60) | 0.75444448 | 1.3680 | 2.3438 |
| visa native 체크포인트 | 120장 | 0.75444448 | 1.3680 | 2.3438 |

- **결과:** 레퍼런스 raw 값이 native raw 값과 mvtec, visa 모두 8자리까지 같다(차이 0.00000000). 정상·이상 점수의 평균, 최솟값, 최댓값도 모든 자리에서 같다. 스크립트가 native 로그의 raw 값(mvtec 0.20614035, visa 0.75444448)을 그대로 재현했다.
- **설명이 확인되었는가:** 확인되었다. 레퍼런스와 native는 같은 raw 점수 순위를 가진다. mvtec에서 정상 이미지의 평균 점수(3.74)가 이상 이미지(3.41)보다 높아 raw 순위는 뒤집혀 있고(AUROC 0.206), 레퍼런스가 보고한 0.447은 같은 점수에 임계값 중심 정규화와 `clamp(0, 1)`을 적용해 생긴 동점 때문에 0.5 쪽으로 올라온 값과 일치한다. 레퍼런스 자체의 정규화 AUROC 계산을 따로 재현하지는 않았다. 즉 D-24 정의에 따른 정규화 AUROC 일치(차이 0)에 더해 raw 순위의 일치까지 확인되어, 이식이 레퍼런스와 같은 모델을 만든다는 근거가 강해졌다.
- **명령:** 아래 명령을 agy가 승인 창 없이 실행했다(로그 파일 이름과 출력으로 일치를 확인).

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py evaluation/anomalib/check_ganomaly_raw_auroc.py --run mvtec-ref=mvtec=D:\projects\visionlab\outputs\anomalib\Ganomaly\LocalMVTecAD\bottle\v0\weights\lightning\model.ckpt --run mvtec-native=mvtec=D:\projects\visionlab\outputs\visionlab\ganomaly\mvtec_bottle\v0\weights\lightning\model.ckpt --run visa-ref=visa=D:\projects\visionlab\outputs\anomalib\Ganomaly\LocalVisa\candle\v0\weights\lightning\model.ckpt --run visa-native=visa=D:\projects\visionlab\outputs\visionlab\ganomaly\visa_candle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\16-raw-auroc-261028-1.log
```

`tests\common` 회귀 실행은 다음 명령이다(승인 창 없이 실행됨).

```powershell
& $py -m pytest tests\common 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\16-regress-common-261028-1.log
```

## 6. 결론

- **레퍼런스:** bottle, visa 확보(10 epoch, 데이터 batch 32, 원본 `Ganomaly`).
- **mvtec / bottle:** Image AUROC(D-24 정의) 차이 0. 통과.
- **visa / candle:** Image AUROC(D-24 정의) 차이 0. 통과.
- **FR-5, FR-6, FR-24:** 통과(5.2절). 동일 상태 출력의 최대 차이가 0.0이다. 시드 42 초기 상태가 같다(INFO). FR-23은 해당 없다.
- **학습 히스토리:** `generator_loss_epoch`, `discriminator_loss_epoch`가 10 epoch 모두 상대 2.2e-7 이하로 일치한다(5.3절).
- **에폭:** 레퍼런스와 native 모두 실제 10 epoch임을 `run.json`과 로그로 확인했다(2.3절).
- **테스트:** `tests\16_ganomaly` 30개(수정 후), `test_cli.py -k ganomaly` 4개(수정 후), `test_dependency.py` 6개가 통과했다. 커밋 전 사용자 승인 회귀 실행(2026-10-28)으로 `tests\common` 59개도 통과했다. 01~15 모델별 회귀는 사용자 결정으로 돌리지 않았다.
- **수정 요청:** 16-A 0회, 16-B 1회.
- **`specs/` 변경:** 사용자 결정(2026-10-28)에 따라 `SPEC.md` FR-4에 GANomaly의 `score [B]`를 더하고(기존 image 전용 모델 문장 유지), `DECISIONS.md`의 D-15 트리에 ganomaly 파일을 추가했다(6.1절). 새 D 번호를 만들지 않았고 INTENT와 PLAN은 고치지 않았다.
- **한계:** raw 순위는 mvtec에서 거의 뒤집혀 있어(0.206) 정규화 AUROC가 clamp 동점의 영향을 받는다. 레퍼런스 raw는 추가 확인(5.5절)에서 native raw와 같음을 확인했다. 원본이 의도한 긴 학습(논문 15 epoch, 이전 anomalib 설정 100 epoch)에서의 성능은 확인하지 않았다.
- **Phase 종료:** 사용자가 종료를 결정했다(2026-10-28). `Phase 16: ganomaly ...` 커밋으로 푸시한다.

### 6.1 `specs/` 갱신 항목

`specs/`는 사람이 쓴다. 최종 보고에 대한 사용자 결정(2026-10-28)에 따라 아래와 같이 처리했다. INTENT와 PLAN은 고치지 않았고 새 D 번호를 만들지 않았다.

- **FR-4 (반영 완료):** `SPEC.md` FR-4의 기존 문장 "image metric만 내는 모델(DFKDE, GANomaly)은 anomaly map 없이 score만 낸다"를 살리고 뒤에 "GANomaly의 score는 `[B]`다."를 더했다.
- **D-15 트리 (반영 완료):** `tests/16_ganomaly/`(두 파일), `configs/models`의 `ganomaly.yaml`, 모델 패키지 `models/ganomaly/`(네 파일: `__init__.py`, `lightning_model.py`, `loss.py`, `model.py`)를 추가했다.

### 6.2 남은 위험과 사용자 판단 대상

- **적대적 학습의 재현성은 이번 실행(category당 1회)에서만 확인했다.** 이번에는 손실이 비트에 가깝게 일치했지만 다른 GPU나 환경에서도 같다고 보장하지 않는다.
- **정규화 AUROC의 변별력이 낮다.** mvtec의 raw AUROC는 0.206이고 레퍼런스가 보고하는 정규화 값은 0.447이다. D-24의 정의대로 판정했으나 clamp 동점 때문에 판정 지표가 둔감하다. 사용자 승인 추가 확인(5.5절)에서 레퍼런스 체크포인트의 raw AUROC가 native raw와 8자리까지 같음을 확인해 이 위험을 낮췄다.
- **원본이 의도한 긴 학습에서의 성능은 확인하지 않았다.** 공유 10 epoch의 결과만 비교했다.
- **앞 Phase 레퍼런스에 같은 dry-run 오염이 있었을 가능성(후속 확인 과제).** 정적 grep으로 확인했다. anomalib의 stock 모델 중 `nn.Module` backbone을 공유 `TimmFeatureExtractor`에 넘기는 것은 CSFlow뿐이고 나머지(CFLOW, DFKDE, DFM, FRE, PaDiM, PatchCore, Reverse Distillation, STFPM, SuperSimpleNet)는 backbone 이름(문자열)을 넘기며, 문자열 경로는 `timm.create_model(features_only=True)`를 쓰고 dry-run이 없다. 한계로 CFA의 torchvision 경로(`create_feature_extractor`를 직접 호출)는 정적 확인을 하지 않았다. 별도 실행은 하지 않았다.
- **일부 agy 명령의 승인 대조 누락.** 2.1절과 5.2절에 적었다.
- **`test_ganomaly_lightning.py` 본문은 줄 단위로 읽지 않았다**(실행 결과로 판정). `compare_outputs.py`의 ganomaly 분기도 구조만 훑었다.
- **모델 규모.** 체크포인트 약 2.26GB, export 약 0.75GB다. `outputs/`는 커밋하지 않는다.
- **사후 실행:** 사용자가 승인한 `tests\common\test_dependency.py` 전체 실행은 6개, `tests\common` 전체는 59개가 통과했다(5.2절, 사용자 승인 회귀 실행 2026-10-28). 이번 Phase에서 앞 Phase 테스트 파일은 수정하지 않았다.
- **앞 모델 회귀(01~15) 보류(사용자 결정):** 돌리지 않았다. 변경은 기존 항목을 건드리지 않는 등록과 분기 추가이고 `predictor.py`의 기존 분기는 그대로다.

| 공유 파일 | 변경 | 영향받을 수 있는 앞 모델 |
|---|---|---|
| `src/visionlab/utils/io.py` | `load_export`가 허용하는 이름에 `ganomaly` 추가 | 모든 모델의 export 로딩 |
| `src/visionlab/inference/predictor.py` | `models` 사전에 `GanomalyModel` 등록과 ganomaly일 때만 `input_size` tuple 변환(기존 분기는 그대로) | 모든 모델의 `Predictor.from_export` |
| `evaluation/visionlab/predict.py` | `--model` choices에 `ganomaly` 추가 | 없음(선택지 추가) |
| `evaluation/anomalib/compare_outputs.py` | `--model ganomaly` 분기 추가 | 없음(기존 분기 유지) |
| `tests/common/test_cli.py` | ganomaly 테스트만 추가 | 없음 |
| `tests/common/test_dependency.py` | 순수 모듈 2개 분류와 Lightning 허용 경로 한 줄 | 분류 규칙 전체(`tests\common` 59 passed) |
