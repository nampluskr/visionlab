# Phase 11. DRAEM

모델 폴더 이름은 `draem`이고 원본은 anomalib 2.0.0의 `anomalib\models\image\draem\`(`C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\Lib\site-packages\anomalib\models\image\draem\`)이다. 범위와 완료 조건은 `specs/PLAN.md`의 Phase 11과 "Phase 공통 양식"을 따른다. 판정 기준은 category별 Image·Pixel AUROC 차이 0.01 이내다.

## 상태

다음 표는 단계별 진행 상태다. 수정 반복 한도 3회는 단계별로 센다.

| 단계 | 내용 | 상태 |
|---|---|---|
| 레퍼런스 | `anomalib_run.py train` x 1 (mvtec/bottle 1회, 10 epoch. visa/candle은 제외) | 완료 |
| 11-A | 순수 모델 (`models/draem/model.py`, `loss.py`, `components/layers/sspcab.py`, `common/perlin.py`의 `PerlinAnomalyGenerator`, `common/multi_random_choice.py`, `tests/11_draem`의 순수 부분) | 완료 |
| 회귀 실행 | 사용자 승인 회귀 실행 1회 (`tests\common`, `tests\10_supersimplenet`. `common/perlin.py` 변경 뒤) | 완료 (37개, 27개 통과) |
| 11-B | Lightning (`DraemLitModel`, Adam과 `MultiStepLR` 학습, configs, export·Predictor 등록, 비교 스크립트) | 완료 |

`PreProcessor`(`mean=None`이면 resize 전용), `PostProcessor`, `Evaluator`, `generate_perlin_noise`는 앞 Phase의 것을 재사용한다.

## 사용자 결정 (2026-10-26)

다음은 Phase 시작 전에 사용자가 정한 항목이다. 사용자가 계획 보고에 "이 계획대로 진행해 주세요"라고 답해, 계획의 권고를 모두 채택한 것으로 처리했다.

| 항목 | 결정 |
|---|---|
| 앞 모델 회귀 | 11-A에서 `common/perlin.py`를 고친 뒤 `tests\common`과 `tests\10_supersimplenet`만 한 번 실행한다(사용자 승인, 이번 한 번에 한해 예외). 01~09 전체는 생략한다. 실패가 나오면 스스로 고치지 않고 멈춰 보고한다 |
| 학습 에폭 | 원본이 `max_epochs`를 고정하지 않으므로(`trainer_arguments`는 `gradient_clip_val` 0과 `num_sanity_val_steps` 0뿐) D-22의 첫 문장대로 공유 yaml의 10 epoch를 레퍼런스와 native에 똑같이 쓴다. 예외는 없다. 원본 `MultiStepLR`의 milestone `[400, 600]`은 10 epoch에서 작동하지 않으며 그대로 복사한다 |
| kornia 사용 위치 | `draem/loss.py`가 kornia의 `FocalLoss`, `SSIMLoss`를 쓰는 것을 허용한다. D-23은 사용 위치를 `model.py`와 `components/`로 제한하므로 문구 해석을 넓히는 일이며 반영은 사용자 몫이다. 학습 전용이고 `Predictor`는 `loss`를 import하지 않는다 |
| 새 부품 위치 | `common/perlin.py`에 `PerlinAnomalyGenerator`를 추가하고, `common/multi_random_choice.py`, `models/components/layers/sspcab.py`를 새로 둔다. D-15 트리 목록에 적는 일은 사용자 몫이다 |
| 평가 category (변경, 2026-10-27) | mvtec/bottle만 평가한다. 레퍼런스 mvtec/bottle의 에폭당 시간이 실측 약 6~8분(step당 약 15초, SuperSimpleNet의 약 30배)이라 10 epoch가 약 70분이고, visa/candle(step 113개)은 에폭당 약 29분으로 한 번에 약 5시간이 예상되어 사용자가 visa/candle을 이번 Phase에서 제외했다. 원인(증강인지 모델인지)은 측정하지 않았다. `PLAN.md`의 Phase 공통 양식은 mvtec과 visa를 모두 요구하므로 6.1절의 "specs 갱신 필요"에 적는다 |
| 학습 시간 관문 | 계획 추정(합계 약 25분)의 1.5배(약 38분)를 넘을 것으로 보이면 계속하기 전에 사용자에게 보고한다. category는 스스로 바꾸지 않는다. 실제로 초과가 확인되어 보고했고(2026-10-26, 2026-10-27) 사용자가 위 category 변경을 결정했다 |
| 난수가 어긋날 때의 판정 | PaDiM, PatchCore, SuperSimpleNet 선례대로 난수 상태를 주입하지 않고 AUROC 0.01로만 판정한다. 증강 일치 검증과 초기 가중치 비교는 기록한다 |
| FR-5 허용 오차 | `atol=1e-6, rtol=1e-5` (Phase 2~10과 같음). 증강·학습 단계 일치 검증도 같은 허용 오차로 판정한다 |
| 에폭별 `train_loss` 히스토리 | 허용 오차 없이 비교표만 기록한다. 10행이므로 전체 표를 본문에 적는다 |
| `anomaly_map` 3차원 | 평가 출력이 `[B,H,W]`인 원본 동작을 그대로 둔다. FR-4 예외 문구에 DRAEM을 포함할지는 Phase 끝에 묻는다 |
| `specs/` 변경 | 없다. 필요한 항목은 결과 문서의 "specs 갱신 필요"에 적고 최종 보고에서 묻는다 |
| 실행 범위 | DRAEM의 명령만 실행한다(위 회귀 1회만 예외) |
| 판정 | Image·Pixel AUROC를 판정한다. 정규화 점수 기준 값(D-24)과 raw 값을 함께 기록한다 |
| 최종 보고 뒤 결정 (2026-10-27) | category 예외는 이번 Phase 한정(visa/candle 미평가, 사용자 결정)으로 이 문서에만 기록하고 `PLAN.md`는 고치지 않는다. D-23은 승인하여 kornia 허용 범위를 DRAEM의 `loss.py`까지 넓히도록 `DECISIONS.md`를 고쳤다. D-15 트리와 FR-4 예외(3차원 `anomaly_map`)는 미정으로 둔다. `test_dependency.py` 전체 실행은 사후 승인한다. Phase 종료를 결정했다 |
| codex 세션 | 컨텍스트 부족(7% 남음)으로 새 세션에서 시작한다(계획 8번 권고 채택). 지시문은 자체 완결이다 |

## 1. 모델 노트

### 1.1 원본과 생성자

다음 표는 원본 파일과 이식 위치의 대응이다(D-15, D-18).

| 원본 | 이식 위치 | 비고 |
|---|---|---|
| `models/image/draem/torch_model.py` (`DraemModel`, `ReconstructiveSubNetwork`, `DiscriminativeSubNetwork`, 인코더·디코더) | `src/visionlab/anomaly_detection/models/draem/model.py` | import 변경. 사전학습 가중치 없음 |
| `models/image/draem/loss.py` (`DraemLoss`) | `.../models/draem/loss.py` | kornia 그대로 |
| `models/components/layers/sspcab.py` (`SSPCAB`, `AttentionModule`) | `.../models/components/layers/sspcab.py` | 변경 없음 |
| `data/utils/generators/perlin.py` (`PerlinAnomalyGenerator`) | `.../common/perlin.py` | Phase 10이 만든 파일에 클래스 추가. import 경로 변경 |
| `data/transforms/multi_random_choice.py` (`MultiRandomChoice`) | `.../common/multi_random_choice.py` | 변경 없음 |
| `models/image/draem/lightning_model.py` (`Draem`) | `.../models/draem/lightning_model.py` (`DraemLitModel`) | native `LightningModule`로 새로 작성 |
| `models/image/draem/__init__.py` | `.../models/draem/__init__.py` | `lightning_model`을 import하지 않도록 고침(D-18 예외) |

생성자는 다음과 같다. `DraemLitModel`은 `Draem`과 인자 이름·순서·기본값이 같아야 한다(FR-6, D-5).

| 클래스 | 시그니처 |
|---|---|
| `Draem` | `enable_sspcab=False`, `sspcab_lambda=0.1`, `anomaly_source_path=None`, `beta=(0.1, 1.0)`, `pre_processor=True`, `post_processor=True`, `evaluator=True`, `visualizer=True` (8개) |
| `DraemModel` | `sspcab=False` |
| `PerlinAnomalyGenerator` | `anomaly_source_path=None`, `probability=0.5`, `blend_factor=(0.2, 1.0)`, `rotation_range=(-90, 90)` |
| `DraemLoss` | 인자 없음 |

### 1.2 학습과 평가 동작

원본의 동작은 다음과 같다.

- **구조:** 재구성 망(`ReconstructiveSubNetwork`, base_width 128, 인코더 block1~5와 디코더)이 입력을 복원하고, 판별 망(`DiscriminativeSubNetwork`, 입력 채널 6, 출력 2)이 입력과 복원 결과를 이어 붙여 정상·이상 두 클래스 지도를 낸다. `sspcab=True`이면 인코더 block5가 `SSPCAB`다.
- **학습 `training_step`:** `Draem.__init__`이 `PerlinAnomalyGenerator(anomaly_source_path, blend_factor=beta)`, `DraemModel(sspcab)`, `DraemLoss()` 순으로 만든다(난수 소비 순서). 학습 batch의 `image`에 증강기를 적용해 `(augmented_image, anomaly_mask)`를 얻고, 모델이 `(reconstruction, prediction)`을 내며, 손실은 원본 이미지, 복원 결과, 마스크, 예측으로 계산한다. `enable_sspcab`이면 `sspcab_lambda` * MSE(encoder.mp4 출력, encoder.block5 출력)을 더한다(forward hook으로 활성값을 수집).
- **증강:** 확률(`probability` 0.5)로 샘플마다 Perlin 마스크를 만들고, DTD 이미지(`anomaly_source_path`가 있으면 무작위 한 장, 없으면 Perlin 기반 소스)에 `MultiRandomChoice`(변환 10개 중 3개)를 적용한 뒤 마스크로 자르고 `beta`로 섞는다. 증강은 CPU와 CUDA 난수를 모두 쓴다.
- **손실:** `DraemLoss` = MSE(복원, 입력) + kornia `SSIMLoss`(window 11) * 2 + kornia `FocalLoss`(alpha 1, mean)(예측, 마스크).
- **평가 `forward`:** `softmax(prediction, dim=1)[:, 1, ...]`이 `anomaly_map`([B,H,W] 3차원)이고 `pred_score`는 지도의 `amax`다.
- **optimizer:** `Adam(lr=0.0001)`과 `MultiStepLR(milestones=[400, 600], gamma=0.1)`.
- **`trainer_arguments`:** `{"gradient_clip_val": 0, "num_sanity_val_steps": 0}`이며 `max_epochs`는 고정하지 않는다(공유 yaml의 10 epoch).
- **전처리:** 원본은 `configure_transforms`가 있지만 anomalib 2.0.0은 이를 호출하지 않는다. 레퍼런스 yaml이 resize 전용 `PreProcessor`(Resize 256)를 명시한다. 입력은 정규화 없이 [0, 1]이다.

### 1.3 외부 의존과 로컬 자산

- **외부 의존:** torchvision(`transforms.v2`, `io`, `datasets.folder`), kornia(`FocalLoss`, `SSIMLoss`)다. timm, scipy, sklearn, einops는 쓰지 않는다. anomalib 의존은 `Batch`, `InferenceBatch`, `AnomalibModule`, `SSPCAB`, `PerlinAnomalyGenerator`, `MultiRandomChoice`, `PreProcessor`, `PostProcessor`, `Evaluator`, `Visualizer`, `LearningType`이다.
- **로컬 가중치:** 없다. DRAEM은 사전학습 backbone이 없다(D-19 해당 없음). `DraemLitModel.teacher_weights_path`는 `None`이고 `write_run_info`가 이미 None을 처리한다.
- **DTD:** `D:/datasets/dtd/images`(5,641개). 경로는 yaml에 직접 적는다(D-12). 내려받지 않는다(NFR-3).
- **데이터:** MVTec `bottle`(학습 209장), VisA `candle`(학습 900장). 분할은 `from_test`, 0.4, seed 42, batch 8/8, `num_workers=0`이다.
- **학습 예산:** 공유 yaml의 `max_epochs=10`이다.

### 1.4 이식에서 의도적으로 달라지는 곳과 유지하는 곳

- **사전학습 주입 없음:** DRAEM은 backbone이 없어 `load_teacher_weights`를 쓰지 않는다. FR-23은 해당 없음이고 FR-22(오프라인 생성)로 대신 확인한다.
- **새 부품 위치:** `PerlinAnomalyGenerator`는 `common/perlin.py`, `MultiRandomChoice`는 `common/multi_random_choice.py`, `SSPCAB`는 `components/layers/`다.
- **원본 동작 유지(고치지 않음):** 증강의 난수 소비 순서(생성 시 `RandomSolarize` 임계값, `rglob` 순서, 호출 시 `torch.rand`·`randint`·`multinomial`), `anomaly_map`의 3차원 출력, `MultiStepLR` milestone, `sspcab` 분기, `torch.cat(..., axis=1)`.
- **`train_loss` 로그의 집계:** 원본은 batch 크기를 추론하지 못해 epoch 평균이 step마다 1회씩 센 평균이다. visionlab은 `batch_size=1`을 두어 맞춘다(Phase 7, 10 선례).
- **기본 pre_processor:** 정규화 없는 `PreProcessor()`(resize는 데이터셋이 256으로 맞춘다). export·Predictor 경로에서 `mean=None`이 처음 쓰이므로 테스트로 확인한다.
- **히스토리 비교:** 레퍼런스와 native의 에폭별 `train_loss`를 `compare_history.py`로 표로 기록한다(판정 아님).
- **FR-5 허용 오차:** `pred_score`와 `anomaly_map`은 `atol=1e-6, rtol=1e-5`다. 증강·학습 단계 일치 검증도 같다.

## 2. 레퍼런스

학습 예산은 공유 yaml의 10 epoch다. 원본이 `max_epochs`를 고정하지 않으므로 D-22의 첫 문장대로 레퍼런스와 native가 같은 값을 쓴다(1.2절, 사용자 결정). visa/candle은 사용자 결정(2026-10-27)으로 이번 Phase에서 제외했다(사용자 결정 표).

### 2.1 10 epoch 레퍼런스

2026-10-26에 agy가 `anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/11_draem.yaml`을 1회 실행했다(실행으로 검증함). Claude는 로그 파일에서 값을 직접 확인했다. 레퍼런스·native·순수 로그에서 `network access is blocked`, `URLError`, `ConnectionError`, `Traceback`, `out of memory`는 0건이다. 로그는 UTF-16이다. agy는 지시한 두 명령(mvtec, visa)을 한 명령씩 따로 실행하며 승인을 요청했고, Claude는 승인 창에 표시된 mvtec 명령이 지시한 줄과 같음을 화면에서 확인해 한 번 승인했다. visa 명령은 사용자 결정에 따라 agy에 실행하지 말라고 알렸고 실행되지 않았다.

agy가 PowerShell에서 실행한 명령은 다음과 같다(Claude가 보낸 지시 원문. 파일은 고치지 않았다).

```powershell
cd D:\projects\visionlab\evaluation\anomalib
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/11_draem.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\11-ref-mvtec-261026-1.log
```

다음 표는 10 epoch 레퍼런스(1회) 결과다.

| 데이터 / category | Image AUROC | Pixel AUROC | 참고 | run 폴더 | 로그 |
|---|---|---|---|---|---|
| mvtec / bottle | 0.86404 | 0.79246 | image F1 0.8810, pixel F1 0.4135 | `outputs/anomalib/Draem/LocalMVTecAD/bottle/v1` | `outputs/logs/11-ref-mvtec-261026-1.log` |

적용된 학습 예산은 `max_epochs=10`, `val_split_mode=from_test`, `val_split_ratio=0.4`, `seed=42`, `batch 8/8`, `num_workers=0`, `enable_sspcab=false`, `sspcab_lambda=0.1`, `anomaly_source_path=D:/datasets/dtd/images`, `beta=(0.1, 1.0)`이다. 체크포인트는 run 폴더의 `weights/lightning/model.ckpt`이고 크기는 1,170,046,546 bytes다. 같은 폴더 이름 아래의 `v0`는 2026-10-25에 만든 이전 실행이며 쓰지 않았다.

DRAEM이 10 epoch에서 낸 값은 Image 0.864, Pixel 0.792로 낮다. 학습 이미지가 209장이라 step 270뿐이다. 원본이 의도한 학습량에서의 성능은 확인하지 않았다(6.2절).

### 2.2 레퍼런스와 native의 에폭 확인

사용자 지시에 따라 레퍼런스와 native가 같은 10 epoch인지 실제 값으로 확인했다.

| 대상 | 근거 | 값 |
|---|---|---|
| 레퍼런스 (bottle) | 로그의 `Trainer.fit stopped: max_epochs=10 reached`, 완료한 가장 큰 에폭 번호 9 | 10 |
| native (bottle) | `run.json`의 `config.trainer.max_epochs` 10, 로그의 `Trainer.fit stopped: max_epochs=10 reached`, 완료한 가장 큰 에폭 번호 9 | 10 |

### 2.3 학습 시간 관문과 실제 학습 시간

계획의 합계 추정은 약 25분이고 1.5배(약 38분)를 넘으면 보고하기로 했다. 레퍼런스가 시작하자 에폭당 약 7분(step당 약 15초)이 확인되어 10 epoch가 bottle만 약 70분, visa(step 113개)는 한 번에 약 5시간이 예상되어 사용자에게 보고했고(2026-10-26, 2026-10-27), 사용자가 visa/candle 제외를 결정했다.

실제로는 다음과 같다.

| 구분 | 에폭별 시간 (로그의 진행 바) |
|---|---|
| 레퍼런스 bottle | 0: 6분 59초, 1: 7분 26초, 2: 6분 04초, 3: 6분 26초, 4: 7분 53초, 5: 4분 18초, 6: 3분 34초, 7: 3분 18초, 8: 2분 41초, 9: 2분 41초. 합계 약 54분 (23:43~00:37) |
| native bottle | 약 22분 (로그 시각 07:30~07:52). 에폭당 약 2분 |

레퍼런스의 앞쪽 에폭이 뒤쪽과 native보다 2~3배 느린 원인은 측정하지 않았다. GPU(GTX 1080 Ti)를 다른 프로세스가 쓰고 있었을 가능성이 있으나 확인하지 못했다. 이식 코드의 속도 문제는 아니다(native가 레퍼런스의 뒤쪽 에폭과 같은 속도다).

### 2.4 에폭별 학습 히스토리

레퍼런스와 native의 비교는 5.3절에 표로 기록한다. 허용 오차는 정하지 않았다.

## 3. 이식 메모

### 3.1 원본과 이식 파일의 대응

다음 표는 새로 만든 파일과 원본의 대응이다. 원본과의 차이는 `difflib`로 확인했다(구현함, 정적 검증).

| 이식 파일 | 원본 | 원본과 다른 줄 |
|---|---|---|
| `models/draem/model.py` | `torch_model.py` | import 2줄(`InferenceBatch`, `SSPCAB`) |
| `models/draem/loss.py` | `loss.py` | 없음 (kornia 그대로) |
| `models/components/layers/{__init__,sspcab}.py` | `components/layers/{__init__,sspcab}.py` | docstring 예제의 import 3줄 |
| `common/perlin.py` | `data/utils/generators/perlin.py` | 기존 `generate_perlin_noise`는 Phase 10 상태 그대로. `PerlinAnomalyGenerator` 클래스 전체는 원본과 동일, import 구역에 `Path`, `io`, `IMG_EXTENSIONS`, `v2`, `MultiRandomChoice` 추가 |
| `common/multi_random_choice.py` | `data/transforms/multi_random_choice.py` | 없음 |
| `models/draem/__init__.py` | `__init__.py` | `DraemModel`만 노출(D-18 예외) |
| `models/draem/lightning_model.py` (`DraemLitModel`) | `lightning_model.py` | native `LightningModule`로 새로 작성. 8개 인자 동일, 생성 순서(증강기, 모델, 손실) 유지, `beta` list를 tuple로 변환, Adam과 `MultiStepLR`, SSPCAB hook, `train_loss` 로그(`batch_size=1`), `teacher_weights_path=None`, D-8의 post_processor 갱신 제한, `trainer_arguments`에 `max_epochs` 없음 |

### 3.2 새로 쓴 코드와 앞 Phase 공유 파일 변경

- **새로 쓴 파일:** `configs/models/draem.yaml`, `common/multi_random_choice.py`, `models/components/layers/`, `models/draem/`, `tests/11_draem/test_draem.py`, `tests/11_draem/test_draem_lightning.py`. `tests\11_draem`은 합계 39개 테스트다(순수 28개, Lightning 11개).
- **공유 파일 변경:** `utils/io.py`(`load_export`가 `draem` 허용), `inference/predictor.py`(모델 dict에 등록), `evaluation/visionlab/predict.py`(`--model draem`), `evaluation/anomalib/compare_outputs.py`(`--model draem` 분기), `tests/common/test_cli.py`, `tests/common/test_dependency.py`, `common/perlin.py`(클래스 추가). `common/` 아래 기존 evaluator·metric 등은 바꾸지 않았다. 영향 범위는 6.2절의 표에 정리했다.
- **앞 Phase 테스트 변경:** `tests/10_supersimplenet/test_supersimplenet.py`의 `test_ssn_original_headers_and_perlin_function_only`는 마지막 assert가 `PerlinAnomalyGenerator` 클래스가 없어야 통과하는 조건이라 Phase 11의 클래스 추가와 충돌했다. 이 assert 한 줄을 `def generate_perlin_noise(`가 있음을 확인하는 조건으로 바꾸고 docstring만 고쳤다(변경 2줄). 함수 이름과 라이선스 assert는 그대로다. 이후 `tests\10_supersimplenet` 27개가 통과했다.
- **새 부품:** 대체 부품은 없다. 모두 원본 복사다.
- **레퍼런스 어댑터 변경:** 없다. 원본 `Draem`을 그대로 쓴다.

### 3.3 이식 중 확인된 사항

- **`beta`의 list와 tuple:** yaml의 `beta: [0.1, 1.0]`은 레퍼런스에서는 jsonargparse가 tuple로 바꾸지만 visionlab의 `DraemLitModel`은 type hint가 없어 list로 들어온다. `PerlinAnomalyGenerator`는 tuple일 때만 무작위 beta를 쓰고 그 밖은 0.5 고정값으로 떨어지므로, 그대로 두면 레퍼런스와 조용히 다른 학습이 된다. 사용자 지시 전에 코드 검토에서 발견해 `DraemLitModel`이 list를 tuple로 바꾸게 했고(원본 동작 유지를 위한 변환), 테스트와 `compare_outputs`(jsonargparse로 레퍼런스의 변환까지 재현)가 확인한다. 두 쪽 모두 `(0.1, 1.0)`이다.
- **난수 소비 순서:** 증강기, 모델, 손실 순의 생성과 증강 호출의 CPU·CUDA 난수 순서를 고치지 않았다. 같은 seed 42에서 모델 초기 가중치 전체와 증강기의 `RandomSolarize` 임계값(0.4576)이 레퍼런스와 같고(`INFO`), 노이즈 소스와 DTD 소스 증강 결과의 최대 차이가 0.0이다(5.2절).
- **`train_loss`의 집계:** `self.log(..., batch_size=1)`을 처음부터 두었고, native의 에폭별 손실이 레퍼런스와 상대 차이 1.9e-7 이내로 일치했다(5.3절).
- **사전학습 가중치 없음:** `teacher_weights_path`가 `None`이고 `write_run_info`가 이미 `None`을 처리해 공유 파일 변경이 필요 없었다. native의 `run.json`에서 `teacher_weights`가 `null`이다.
- **resize 전용 전처리:** `PreProcessor()`는 `mean=None`이라 항등이다. export의 `pre_processor.mean`이 `null`로 저장·복원되는 경로를 처음 썼고(순수 추론 결과 5.1절, Lightning 테스트 11개) 이상이 없었다.
- **3차원 `anomaly_map`:** 평가 출력 `[B,H,W]`를 그대로 두었다. 후처리, metric, 순수 추론이 모두 이 형태에서 정상 동작했다. 이미지 저장 callback(`LitVisualizer`)과의 결합은 `visualizer: false`라 이번에 실행하지 않았다.

## 4. 적대적 검증

검증은 11-A, 11-B 각각 한 회차씩 했다. 수정 요청은 11-A 1회(불필요했던 요청, 아래), 11-B 1회(한도 3회 이내)다.

### 4.1 11-A (순수 모델, 증강 부품)

| 지적 | 심각도 | 처리 |
|---|---|---|
| 원본과의 `difflib` 대조: `model.py`는 import 2줄뿐, `loss.py`와 `multi_random_choice.py`는 차이 없음, `sspcab.py`는 docstring import만 다름. `PerlinAnomalyGenerator` 클래스 본문이 원본과 동일 | 확인 | 수정 없음 |
| `common/perlin.py`의 import가 파일 중간에 있다고 지적해 수정을 요청했으나, 실제로는 처음부터 맨 위에 있었다. 검토자(Claude)의 `git diff` 오독이었다 | 정정 | codex가 이미 맞다고 답했고 수정 없음. 요청 1회로 센다 |
| 코드 실행 전 codex가 보고한 충돌: Phase 10 테스트가 클래스 부재를 요구함 | 중간 | 앞 Phase 테스트의 마지막 assert 한 줄만 고침(3.2절). 실행으로 확인 |

### 4.2 11-B (Lightning, export, 비교 스크립트)

| 지적 | 심각도 | 처리 |
|---|---|---|
| 원본 `lightning_model.py`와의 대조: 생성 순서, `training_step`(손실 입력이 증강 전 이미지), SSPCAB hook과 손실 항, `Adam(lr 1e-4)`, `MultiStepLR([400, 600], 0.1)`, `trainer_arguments`가 같음. `beta` 변환은 의도한 차이임. 공유 파일 변경은 등록과 허용 목록 추가뿐 | 확인 | 수정 없음 |
| `compare_outputs`의 `FR-5 mvtec inputs`가 FAIL. 입력 범위 확인 조건이 `images.max() <= 1`이었고 details에 값이 없었다. 재실행 로그의 값은 `max=1.000000238`이었다(float 반올림, 데이터셋의 antialias resize). 모델 비교가 아니라 입력 전제 확인 항목 | 낮음 | 수정 요청 1회: details에 shape·min·max를 남기고 범위에 `DRAEM_ATOL`(1e-6)을 허용. 재실행에서 PASS 40, FAIL 0 |
| `compare_outputs`의 draem 분기는 `git diff`로 전체를 읽었다. `tests/11_draem/test_draem_lightning.py`는 테스트 이름과 구조만 확인했고 본문은 줄 단위로 읽지 않았다(실행 결과로 판정) | 낮음 | 수정 없음 |

남은 위험은 6.2절에 적었다.

## 5. 이식 결과

### 5.1 최종 판정 (10 epoch, Image·Pixel AUROC 차이 0.01 이내)

native는 `train.py`가 test에서 기록한 값이고 순수 PyTorch는 `export/model.pt`만 불러 `predict.py`로 평가한 값이다. 모두 실행으로 검증했다. D-24에 따라 레퍼런스가 보고하는 값과 같은 정의로 비교하며, 순수 PyTorch의 정규화 점수 AUROC도 함께 적는다.

| 데이터 / category | 지표 | 레퍼런스 | native (raw) | 순수 PyTorch (raw) | 순수 PyTorch (정규화) | 차이(native) | 차이(순수 raw) | 판정 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| mvtec / bottle | Image AUROC | 0.864035130 | 0.864035070 | 0.86403507 | 0.86403513 | 6e-8 | 6e-8 | 통과 |
| mvtec / bottle | Pixel AUROC | 0.792461872 | 0.792483270 | 0.79248321 | 0.79246175 | 2.1e-5 | 2.1e-5 | 통과 |

raw와 정규화 값이 0.01 넘게 다른 곳은 없어 D-24의 원인 기록 조건에는 해당하지 않는다. image F1과 pixel F1도 레퍼런스, native, 순수 PyTorch가 같다(image 0.88095236, pixel 0.41354021). visa/candle은 이번 Phase에서 평가하지 않았다.

run 폴더와 로그는 다음과 같다.

| 대상 | 위치 |
|---|---|
| 레퍼런스 | `outputs/anomalib/Draem/LocalMVTecAD/bottle/v1` |
| native | `outputs/visionlab/draem/mvtec_bottle/v0` (`run.json`: `max_epochs` 10, `num_sanity_val_steps` 0, `gradient_clip_val` 0.0, seed 42, `teacher_weights` null, 학습 이미지 209장, `beta` [0.1, 1.0]) |
| 순수 PyTorch | 같은 run의 `predict/metrics.json`, `export/model.pt`(389,929,318 bytes) |
| 로그 | `outputs/logs/11-native-mvtec-261027-1.log`, `11-pure-mvtec-261027-1.log` |

### 5.2 테스트와 비교 스크립트

다음 표는 agy가 실행한 묶음의 결과다(Claude가 로그 파일에서 확인). 실행 대상은 DRAEM뿐이다(11-A 뒤의 회귀 실행은 위의 별도 항목).

| 대상 | 결과 | 로그 |
|---|---|---|
| 회귀: `tests\common` | 37 passed | `11-regress-common-261027-1.log` |
| 회귀: `tests\10_supersimplenet` | 27 passed | `11-regress-ssn-261027-1.log` |
| `tests\11_draem` (11-B 뒤 전체) | 39 passed | `11-pytest-261027-1.log` (11-A 뒤 단독 실행은 `11-pytest-pure-261027-1.log`로 같은 39개) |
| `tests\common\test_cli.py -k draem` | 2 passed (17 deselected) | `11-pytest-cli-261027-1.log` |
| `tests\common\test_dependency.py` (전체 파일) | 6 passed | `11-pytest-dependency-261027-1.log` |
| `compare_outputs.py --model draem` 1차 | PASS 39, FAIL 1(`FR-5 mvtec inputs`), INFO 1 | `11-compare-outputs-mvtec-261027-1.log` |
| `compare_outputs.py --model draem` 재실행(수정 1회차 뒤) | PASS 40, FAIL 0, INFO 1 | `11-compare-outputs-mvtec-261027-2.log` |

`compare_outputs`의 PASS 항목은 다음과 같다. 최대 차이는 모두 0.0이다.

- **FR-6, beta:** 생성자 시그니처 일치, `blend_factor`가 레퍼런스와 native 모두 tuple `(0.1, 1.0)`.
- **증강 일치:** 노이즈 소스와 DTD 소스 각각에서 `augmented_image`와 `anomaly_mask`가 같은 seed로 일치(mvtec 실제 샘플 8장, 256x256, CUDA).
- **FR-5 학습 forward:** 레퍼런스 체크포인트 상태에서 `reconstruction [8,3,256,256]`, `prediction [8,2,256,256]`, `DraemLoss`가 일치(노이즈·DTD 증강 입력 각각).
- **FR-5 평가 forward:** `pred_score [8]`, `anomaly_map [8,256,256]`(3차원)이 일치.
- **SSPCAB:** `sspcab=True` 모델의 학습·평가 forward와 손실이 일치(학습은 하지 않음).
- **FR-24와 PostProcessor 대조:** 모델과 무관한 기존 항목이 모두 통과.
- **INFO 1개:** seed 42 초기 모델 가중치 전체와 `RandomSolarize` 임계값이 서로 같다.
- **FR-23:** 해당 없음(사전학습 가중치 없음). 결과 manifest에 `not_applicable`로 적었다.

`test_dependency.py`는 모든 순수 모듈의 import 분리 검사를 한 파일이 하므로 이 모델의 모듈 검증을 위해 전체를 돌렸다(Phase 5, 6, 7, 9, 10과 같은 사유). 이번 Phase에서의 실행은 사용자가 사후에 승인했다(2026-10-27).

실행한 명령은 다음과 같다(Claude가 보낸 지시 원문). agy가 명령을 한 줄씩 나누어 승인을 요청했고, Claude는 승인 창의 명령이 지시한 줄과 정확히 같을 때만 한 번씩 승인했다(영구 허용 없음). 첫 명령은 직접 확인했고 나머지는 같은 규칙으로 동작하는 확인 스크립트가 처리했으나, pytest의 `-k draem`과 `test_dependency.py` 두 명령은 승인 요청 없이 실행되어 승인 창에서 확인하지 못했다. 이 두 명령의 스크립트 전문은 agy의 transcript에서 대조하려 했으나 읽기 권한이 거부되어 대조하지 못했다. 로그 파일의 pytest 머리글(수집 대상과 개수)이 지시한 명령과 일치함은 확인했다.

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py -m pytest tests\common 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\11-regress-common-261027-1.log
& $py -m pytest tests\10_supersimplenet 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\11-regress-ssn-261027-1.log
& $py -m pytest tests\11_draem 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\11-pytest-pure-261027-1.log
& $py -m pytest tests\11_draem 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\11-pytest-261027-1.log
& $py -m pytest tests\common\test_cli.py -k draem 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\11-pytest-cli-261027-1.log
& $py -m pytest tests\common\test_dependency.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\11-pytest-dependency-261027-1.log
& $py evaluation/anomalib/compare_outputs.py --model draem --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\Draem\LocalMVTecAD\bottle\v1\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\11-compare-outputs-mvtec-261027-1.log
& $py evaluation/visionlab/train.py train --data mvtec --model draem 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\11-native-mvtec-261027-1.log
& $py evaluation/visionlab/predict.py --model draem --data mvtec 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\11-pure-mvtec-261027-1.log
& $py evaluation/anomalib/compare_history.py --reference D:\projects\visionlab\outputs\anomalib\Draem\LocalMVTecAD\bottle\v1 --native D:\projects\visionlab\outputs\visionlab\draem\mvtec_bottle\v0 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\11-history-mvtec-261027-1.log
& $py evaluation/anomalib/compare_outputs.py --model draem --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\Draem\LocalMVTecAD\bottle\v1\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\11-compare-outputs-mvtec-261027-2.log
```

agy는 파일을 고치지 않았다. 보고를 만들려고 로그를 읽는 `Get-Content` 명령을 지시 없이 추가로 실행한 기록이 있다(읽기 전용).

### 5.3 에폭별 학습 히스토리 비교 (허용 오차 없음, 비교표만)

다음 표는 `compare_history.py`의 `train_loss_epoch` 출력이다. 10행이라 전체를 적는다. 판정에 쓰지 않는다.

mvtec / bottle (레퍼런스 `Draem/.../bottle/v1`, native `mvtec_bottle/v0`):

| Epoch | 레퍼런스 | native | 절대 차이 | 상대 차이 |
|---:|---:|---:|---:|---:|
| 0 | 0.5910148025 | 0.5910148025 | 0 | 0 |
| 1 | 0.2812461853 | 0.2812462151 | 2.980232239e-08 | 1.059652502e-07 |
| 2 | 0.200462833 | 0.2004628181 | 1.490116119e-08 | 7.433378531e-08 |
| 3 | 0.1454322338 | 0.1454322189 | 1.490116119e-08 | 1.024612e-07 |
| 4 | 0.1198952869 | 0.1198952645 | 2.235174179e-08 | 1.864271931e-07 |
| 5 | 0.1260309964 | 0.1260309815 | 1.490116119e-08 | 1.182340981e-07 |
| 6 | 0.1082197577 | 0.1082197502 | 7.450580597e-09 | 6.884676844e-08 |
| 7 | 0.1008497253 | 0.1008497179 | 7.450580597e-09 | 7.387804552e-08 |
| 8 | 0.09690821916 | 0.09690821916 | 0 | 0 |
| 9 | 0.1000083089 | 0.1000083089 | 0 | 0 |

모든 에폭의 상대 차이가 1.9e-7 이하다. float32 단위 오차 수준이어서 학습 경로(초기 가중치, 데이터 순서, 증강의 난수, 손실, optimizer)가 레퍼런스와 같다고 볼 수 있다. 에폭별 허용 오차는 사용자가 아직 정하지 않았다. 참고로 native의 학습 중 `val_image_AUROC`는 0.64~0.92, `val_pixel_AUROC`는 0.67~0.89 사이에서 변했다. 별도 split(`from_test`, 0.4)의 값이며 판정에 쓰지 않는다.

### 5.4 crop·입력 크기

이 모델은 crop이 없고 입력 크기는 256x256이다(resize 전용 전처리, 정규화 없음, 입력 [0, 1]). 평가 출력 `anomaly_map`은 `[B,256,256]`이다. FR-5의 `anomaly_map [8,256,256]` 최대 차이가 0.0이다.

## 6. 결론

- **레퍼런스:** bottle 확보(10 epoch, 원본 `Draem`). visa/candle은 사용자 결정으로 제외.
- **mvtec / bottle:** Image AUROC 차이 6e-8, Pixel AUROC 차이 2.1e-5(native와 순수 PyTorch 모두). 통과. 정규화 점수 기준(D-24)도 같다.
- **FR-5, FR-6, FR-24:** 통과(5.2절). 증강, 학습 forward, 손실, 평가 forward, SSPCAB의 최대 차이가 0.0이다. FR-23은 해당 없음. 에폭별 `train_loss_epoch`는 상대 차이 1.9e-7 이내(5.3절)이다.
- **에폭:** 레퍼런스와 native 모두 실제 10 epoch임을 `run.json`과 로그로 확인했다(2.2절).
- **테스트:** `tests\11_draem` 39개, `test_cli.py -k draem` 2개, `test_dependency.py` 6개가 통과했다. 회귀 `tests\common` 37개와 `tests\10_supersimplenet` 27개도 통과했다.
- **수정 요청:** 11-A 1회(불필요했던 요청), 11-B 1회.
- **판정 범위:** mvtec/bottle 한 category다. PLAN의 공통 양식(mvtec과 visa)을 모두 채우지는 못했다(사용자 결정, 6.1절).
- **`specs/` 변경:** 사용자 결정(2026-10-27)에 따라 `DECISIONS.md` D-23만 고쳤다(6.1절). `PLAN.md`, INTENT, SPEC은 건드리지 않았다.
- **Phase 종료:** 사용자가 종료를 결정했다(2026-10-27). `Phase 11: draem ...` 커밋으로 푸시한다.

### 6.1 `specs/` 갱신 항목

`specs/`는 사람이 쓴다. 최종 보고에 대한 사용자 결정(2026-10-27)에 따라 아래와 같이 처리했다.

- **D-23의 kornia 사용 위치 (반영 완료):** Claude가 `DECISIONS.md` D-23을 최소한으로 고쳤다. 코드에서 확인한 kornia import 위치는 `components/filters/blur.py`와 `models/draem/loss.py` 두 파일이다. `loss.py`는 `draem/lightning_model.py`만 import하고 `Predictor`, `draem/model.py`, `utils/io.py`는 import하지 않는다.
  - 제목: "kornia를 순수 모델과 학습 손실의 외부 의존으로 허용한다".
  - 선택: DRAEM의 `DraemLoss`가 원본처럼 kornia의 `FocalLoss`, `SSIMLoss`를 import한다는 문장을 추가(Phase 11, 사용자 결정 2026-10-27).
  - 적용 범위: 사용 위치에 `models/draem/loss.py`를 더하고 현재 import 위치와 `Predictor`가 import하지 않는다는 점을 적었다. "kornia에만 해당하며 다른 라이브러리는 포함하지 않는다"는 문구는 그대로다.
  - 근거와 배제한 대안: DRAEM 손실의 원본 구현과 독립 구현 배제를 추가. INTENT와의 관계 서술은 유지하고 INTENT는 고치지 않았다.
- **PLAN의 category 구성 (미반영, 사용자 결정):** `PLAN.md`의 공통 양식은 mvtec과 visa를 모두 평가하도록 정한다. 이번 Phase는 사용자 결정으로 mvtec/bottle만 평가했다. 이 예외는 이 문서에만 "이번 Phase 한정 예외(visa/candle 미평가, 사용자 결정)"로 기록하고 `PLAN.md`는 고치지 않는다. 시간 근거는 2.3절에 있다.
- **D-15 트리 목록 (미정):** `common/perlin.py`의 `PerlinAnomalyGenerator`, `common/multi_random_choice.py`, `models/components/layers/sspcab.py`, `models/draem/`가 목록에 없다. Phase 10의 `common/perlin.py`, `models/supersimplenet/`도 아직 없다. 이번에는 정하지 않았다.
- **FR-4의 `anomaly_map` 형태 예외 (미정):** 현재 FR-4에는 image-only와 CFA squeeze 예외가 있다. DRAEM의 평가 출력 `anomaly_map [B,H,W]`(3차원)와 SuperSimpleNet의 batch 1 스칼라 점수를 예외 문구에 포함할지는 이번에 정하지 않았다. 코드 동작은 이미 원본 그대로다.

### 6.2 남은 위험과 사용자 판단 대상

- **10 epoch의 낮은 성능:** Image 0.864, Pixel 0.792로 낮다. 레퍼런스도 같은 값이므로 이식 오류는 아니지만 원본이 의도한 학습량에서의 성능은 확인하지 않았다. 원본의 `MultiStepLR` milestone `[400, 600]`은 10 epoch에서 작동하지 않아 검증하지 못했다(단위 테스트로 설정값만 확인).
- **레퍼런스가 1회뿐이다.** 같은 seed에서 native와 순수 PyTorch가 AUROC, 에폭별 손실, 학습 forward 최대 차이까지 일치해 학습 경로의 편차는 작아 보인다. 일치가 매우 정확한 것은 seed 42와 난수 소비 순서가 같고 같은 GPU에서 같은 연산이 쓰였기 때문이다. 다른 GPU나 드라이버에서도 같은 정도로 일치한다고 보장하지 않는다. DRAEM의 디코더는 `Upsample`이 학습 경로에 있어 원칙적으로 역전파가 비결정적일 수 있으나, 이번 실행에서는 어긋남이 나타나지 않았다.
- **visa/candle 미평가:** 사용자 결정이다. visa에서 DRAEM이 어떻게 동작하는지는 확인하지 못했다.
- **레퍼런스 학습 속도:** 앞쪽 에폭이 뒤쪽보다 2~3배 느린 원인을 측정하지 않았다(2.3절). 같은 환경에서 다음 Phase의 시간 추정에 불확실성이 남는다.
- **SSPCAB 학습 미검증:** `enable_sspcab=True`는 forward 일치와 hook·손실 항의 단위 테스트만 확인했고 학습 실행은 하지 않았다(레퍼런스 yaml이 `false`).
- **`LitVisualizer`와 3차원 `anomaly_map`:** `visualizer: false`라 이번에 실행하지 않았다.
- **Lightning 테스트 본문:** `test_draem_lightning.py`의 본문은 줄 단위로 읽지 않았다(실행 결과로 판정). 또한 일부 테스트가 `cuda` 장치를 매개변수로 직접 쓰므로 GPU가 없는 환경에서는 실패할 수 있다.
- **일부 agy 명령의 승인 대조 누락:** 5.2절에 적었다.
- `tests\common\test_dependency.py` 전체 실행은 사용자가 사후에 승인했다(2026-10-27).
- 앞 모델 회귀는 `tests\common`과 `tests\10_supersimplenet`만 실행했다(사용자 승인). 01~09의 테스트는 돌리지 않았다. 11-B의 공유 파일 변경 이후 전체 회귀도 실행하지 않았다. 변경은 모두 기존 항목을 건드리지 않는 등록과 분기 추가이고 `common/perlin.py`에 클래스를 추가했다.

| 공유 파일 | 변경 | 영향받을 수 있는 앞 모델 |
|---|---|---|
| `src/visionlab/anomaly_detection/common/perlin.py` | `PerlinAnomalyGenerator` 클래스와 import 추가. `generate_perlin_noise` 줄은 그대로 | SuperSimpleNet (회귀 27개 통과) |
| `src/visionlab/utils/io.py` | `load_export`가 허용하는 이름에 `draem` 추가 | 모든 모델의 export 로딩 |
| `src/visionlab/inference/predictor.py` | `from_export` 모델 dict에 `draem` 등록 | 모든 모델의 `Predictor.from_export` |
| `evaluation/visionlab/predict.py` | `--model` 선택지에 추가 | 모든 모델의 순수 PyTorch 평가 |
| `evaluation/anomalib/compare_outputs.py` | `--model draem` 분기와 검증 함수, jsonargparse·DRAEM import 추가 | stfpm, efficient_ad, padim, dfm, dfkde, patchcore, reverse_distillation, cfa, fre, supersimplenet 비교 |
| `tests/common/test_cli.py`, `tests/common/test_dependency.py` | draem 테스트와 분류 추가 | 같은 파일의 앞 모델 테스트 (회귀 37개 통과) |
| `tests/10_supersimplenet/test_supersimplenet.py` | 마지막 assert 한 줄 변경 (3.2절) | SuperSimpleNet 테스트 |

- 체크포인트가 약 1.17GB(optimizer 상태 포함)라 `outputs/`의 디스크 사용이 크다(커밋하지 않는다).
