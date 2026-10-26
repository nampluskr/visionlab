# Phase 6. PatchCore

모델 폴더 이름은 `patchcore`이고 원본은 anomalib 2.0.0의 `anomalib\models\image\patchcore\`(`C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\Lib\site-packages\anomalib\models\image\patchcore\`)이다. 범위와 완료 조건은 `specs/PLAN.md`의 Phase 6과 "Phase 공통 양식"을 따른다. 판정 기준은 category별 Image·Pixel AUROC 차이 0.01 이내다.

## 상태

다음 표는 단계별 진행 상태다. 수정 반복 한도 3회는 단계별로 센다.

| 단계 | 내용 | 상태 |
|---|---|---|
| 레퍼런스 | `anomalib_run.py train` x 2 (mvtec/bottle 1회, mvtec/capsule 1회, 1 epoch). visa/candle은 중단(사용자 결정, 2.2절) | 실행 완료 (2절) |
| 6-A | 순수 모델 (`models/patchcore/model.py`, `anomaly_map.py`, `components/sampling/`의 `KCenterGreedy`, `components/dimensionality_reduction/`의 `SparseRandomProjection`, `tests/06_patchcore`의 순수 부분) | 구현, 검증, 실행 완료 (수정 요청 1회, 테스트의 차단 시점) |
| 6-B | Lightning (`PatchcoreLitModel`, `PreProcessor`의 `center_crop`과 mask 정렬, configs, memory bank 저장·복원, export·Predictor 등록, 비교 스크립트) | 구현, 검증, 실행 완료 (수정 요청 0회, 결과 5절) |

처리 구성요소(PostProcessor, Evaluator), 특징 추출기(`TimmFeatureExtractor`), `DynamicBufferMixin`, `GaussianBlur2d`는 앞 Phase의 것을 재사용한다.

## 사용자 결정 (2026-10-26)

다음은 Phase 시작 전과 진행 중에 사용자가 정한 항목이다.

| 항목 | 결정 |
|---|---|
| 학습 에폭 | 모델이 고정한 1 epoch을 따른다(D-22). 레퍼런스 yaml과 native yaml 모두 `max_epochs: 1` |
| CenterCrop 224 | `PreProcessor`에 선택 인자 `center_crop`을 추가한다(기본 `None`, 앞 모델 동작 불변). mask 정렬은 `PatchcoreLitModel._outputs`와 `predict.py`가 같은 crop을 mask에도 적용한다 |
| sklearn | `SparseRandomProjection._sparse_random_matrix` 안에서 `sample_without_replacement`를 지연 import해 추론 환경에는 sklearn이 필요 없게 한다. 사용자 결정으로 `specs/DECISIONS.md`에 D-25로 반영한다(6.1절) |
| coreset 불일치 | PaDiM 선례대로 memory bank를 주입하지 않고 AUROC 0.01로만 판정한다. memory bank 차이는 정보로 기록한다 |
| FR-5 허용 오차 | `atol=1e-6, rtol=1e-5` (Phase 2~5와 같음) |
| 실행 범위 | PatchCore의 명령만 실행한다. 앞 모델의 pytest와 `compare_outputs`는 돌리지 않고, 영향받을 수 있는 공유 파일은 결과 문서에 남긴다 |
| 판정 | Image·Pixel AUROC를 판정한다. 정규화 점수 기준 값(D-24)과 raw 값을 함께 기록한다 |
| `test_dependency.py` 전체 실행 (사후, 2026-10-26) | 사용자가 승인했다 |
| 앞 모델 회귀 (2026-10-26) | 보류한다(사용자 결정). 이번에는 돌리지 않고 영향받을 수 있는 공유 파일은 6.2절의 표에 남긴다 |
| `specs/` 반영 (2026-10-26) | `DECISIONS.md`에 D-25(scikit-learn 지연 import)를 추가하고, `DECISIONS.md` D-6과 `SPEC.md` FR-8에 `PreProcessor`의 선택 인자 `center_crop`을 덧붙인다. `specs/PLAN.md`의 평가 category 문구는 고치지 않는다 |
| Phase 종료 (2026-10-26) | 사용자가 종료를 결정했다. `Phase 06: patchcore ...` 커밋으로 푸시한다 |
| 평가 category 변경 (실행 중 추가, 2026-10-26) | visa/candle은 coreset 선택이 약 4시간 40분 걸려 제외하고 mvtec/capsule로 대체한다. 판정 대상은 mvtec/bottle과 mvtec/capsule이다. 진행 중이던 visa 레퍼런스는 중단했다(2.2절). `specs/PLAN.md`의 category 문구는 고치지 않고 이번 Phase 한정 예외로 이 문서에 기록한다 |

## 1. 모델 노트

### 1.1 원본과 생성자

다음 표는 원본 파일과 이식 위치의 대응이다(D-15, D-18).

| 원본 | 이식 위치 | 비고 |
|---|---|---|
| `models/image/patchcore/torch_model.py` (`PatchcoreModel`) | `src/visionlab/anomaly_detection/models/patchcore/model.py` | import 변경, `pre_trained`는 순수 모델에서 구조만 만드는 방식으로(D-19) |
| `models/image/patchcore/anomaly_map.py` (`AnomalyMapGenerator`) | `.../models/patchcore/anomaly_map.py` | `GaussianBlur2d` import 위치만 변경 |
| `models/image/patchcore/lightning_model.py` (`Patchcore`) | `.../models/patchcore/lightning_model.py` (`PatchcoreLitModel`) | native `LightningModule`, `MemoryBankMixin` hook을 직접 구현(D-11), center crop 전처리 |
| `models/image/patchcore/__init__.py` | `.../models/patchcore/__init__.py` | `lightning_model`을 import하지 않도록 고침(D-18 예외) |
| `components/sampling/k_center_greedy.py` (`KCenterGreedy`) | `.../models/components/sampling/k_center_greedy.py` | 새로 복사 |
| `components/dimensionality_reduction/random_projection.py` (`SparseRandomProjection`, `NotFittedError`) | `.../models/components/dimensionality_reduction/random_projection.py` | 새로 복사. sklearn import를 함수 안으로 옮김 |

생성자는 다음과 같다. `PatchcoreLitModel`은 `Patchcore`와 인자 이름·순서·기본값이 같아야 한다(FR-6, D-5).

| 클래스 | 시그니처 |
|---|---|
| `Patchcore` | `backbone="wide_resnet50_2"`, `layers=("layer2", "layer3")`, `pre_trained=True`, `coreset_sampling_ratio=0.1`, `num_neighbors=9`, `pre_processor=True`, `post_processor=True`, `evaluator=True`, `visualizer=True` (9개) |
| `PatchcoreModel` | `layers`, `backbone="wide_resnet50_2"`, `pre_trained=True`, `num_neighbors=9` |

### 1.2 학습과 평가 동작

원본의 동작은 다음과 같다.

- **전처리:** `configure_pre_processor`가 `Resize((256, 256), antialias=True)`, `CenterCrop((224, 224))`, ImageNet `Normalize`를 쓴다. anomalib의 `PreProcessor`는 이 변환을 이미지와 `gt_mask`에 함께 적용하므로(`batch.image, batch.gt_mask = self.transform(...)`) 레퍼런스의 Pixel AUROC는 224x224로 잘린 영역에서 계산된다.
- **모델:** `TimmFeatureExtractor`(eval)로 `layers`의 특징을 뽑고(`no_grad`), layer마다 `AvgPool2d(3, 1, 1)`로 풀링한다. 첫 layer 해상도에 맞춰 나머지를 bilinear로 보간해 채널 방향으로 합치고(`generate_embedding`) `(B*H*W, C)`로 편다(`reshape_embedding`). 학습 모드의 `forward`는 이 임베딩을 반환한다.
- **학습:** `training_step`은 `self.model(batch.image)`를 `self.embeddings`에 모으고 더미 loss(0)를 반환한다. `configure_optimizers`는 `None`이다. 학습 가능한 파라미터가 없다.
- **적합(`fit`):** `MemoryBankMixin`의 `on_validation_start`와 `on_train_epoch_end`가 `_is_fitted` buffer가 False일 때 한 번 호출한다. `torch.vstack(self.embeddings)`를 `KCenterGreedy`(sparse random projection 후 greedy farthest point)로 `coreset_sampling_ratio`만큼 줄여 `memory_bank`(dynamic buffer)로 둔다.
- **추론:** 임베딩과 `memory_bank`의 유클리드 거리(`clamp_min_`, `sqrt_`)로 patch별 최근접 거리와 위치를 구한다. `pred_score`는 최대 patch에 대해 `num_neighbors` 이웃으로 만든 가중치를 곱한 값이고(`num_neighbors == 1`이면 `amax`), `anomaly_map`은 patch 점수를 입력 크기로 `F.interpolate`(nearest)하고 `GaussianBlur2d(sigma 4, kernel 33)`로 smoothing한 것이다.
- **`trainer_arguments`:** `{"gradient_clip_val": 0, "max_epochs": 1, "num_sanity_val_steps": 0}`이다.

### 1.3 외부 의존과 로컬 자산

- **외부 의존:** timm, torchvision, kornia(`GaussianBlur2d`, D-23), tqdm(`KCenterGreedy`의 진행 표시)이다. sklearn(`sample_without_replacement`)은 `SparseRandomProjection`에서 지연 import한다(D-25). anomalib 의존은 `Batch`, `InferenceBatch`, `AnomalibModule`, `MemoryBankMixin`, `KCenterGreedy`, `SparseRandomProjection`, `GaussianBlur2d`, `TimmFeatureExtractor`, `PreProcessor`, `PostProcessor`, `Evaluator`, `Visualizer`, `LearningType`이다.
- **로컬 가중치:** `paths.toml [backbones]`의 `wide_resnet50_2`(이미 있다). 레퍼런스는 `local_setup.py`가 timm 이름을 로컬 로더로 바꾼다. visionlab은 `PatchcoreLitModel.__init__`이 `feature_extractor`에 주입한다(D-19).
- **데이터:** MVTec `bottle`(학습 209장), VisA `candle`(학습 900장). 분할은 `from_test`, 0.4, seed 42, batch 8/8, `num_workers=0`이다.
- **학습 예산:** `max_epochs=1`(모델 고정)이다.

### 1.4 이식에서 의도적으로 달라지는 곳과 유지하는 곳

- **pretrained 주입(D-19):** 순수 `PatchcoreModel`은 `pre_trained=False`로 구조만 만든다. `PatchcoreLitModel`이 `wide_resnet50_2` 로컬 파일을 `feature_extractor`에 불러온다.
- **`MemoryBankMixin` 대체(D-11):** 공유 기반 클래스를 만들지 않고 `_is_fitted` buffer, `on_validation_start`, `on_train_epoch_end`를 `PatchcoreLitModel`에 직접 쓴다.
- **center crop:** visionlab의 `PreProcessor`는 정규화만 하므로 선택 인자 `center_crop`을 추가한다. `PatchcoreLitModel`은 `PreProcessor(mean, std, center_crop=(224, 224))`를 기본으로 쓰고, 평가 때 `gt_mask`도 같은 crop을 적용한다. `Predictor`와 `predict.py`도 같은 crop을 mask에 적용한다.
- **sklearn 지연 import:** `sample_without_replacement`를 `_sparse_random_matrix` 안에서 import한다. 나머지는 원본 그대로다.
- **원본 동작 유지:** `KCenterGreedy`의 `torch.randint` 첫 index와 `SparseRandomProjection`의 `Binomial` 표본, `compute_anomaly_score`의 softmax 가중치, `euclidean_dist`의 in-place `clamp_min_`와 `sqrt_`, `anomaly_map`의 nearest 보간, `tiler=None` 속성은 고치지 않는다.
- **히스토리 비교:** loss가 없고 1 epoch이라 `compare_history.py`는 쓰지 않는다. 대신 레퍼런스 ckpt와 native ckpt의 `memory_bank`를 정보로 비교한다(shape와 행 일치 정도, 판정 아님).
- **FR-5 허용 오차:** `pred_score`와 `anomaly_map`은 `atol=1e-6, rtol=1e-5`, backbone 가중치는 `torch.equal`이다.

## 2. 레퍼런스

학습 예산은 모델이 고정한 1 epoch이다(D-22). `models/06_patchcore.yaml`에 `trainer.max_epochs: 1`이 있고, 저장된 `lightning_logs/version_0/config.yaml`과 로그에 `max_epochs=1`이 남았다.

### 2.1 1 epoch 레퍼런스

2026-10-26에 agy가 `anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/06_patchcore.yaml`을 category마다 1회 실행했다(실행으로 검증함). Claude는 로그 파일에서 값을 직접 확인했다. 로그에 `network access is blocked`, `URLError`, `ConnectionError`, `Traceback`, `out of memory`는 0건이다. 로그는 PowerShell `Tee-Object` 출력이라 UTF-16이다. capsule 레퍼런스는 약 120초가 걸렸다(agy 보고). 승인 창의 스크립트 전문은 Claude가 agy의 transcript에서 읽어 지시한 명령과 같음을 확인했다.

agy가 PowerShell에서 실행한 명령은 다음과 같다(Claude가 보낸 지시 원문. 파일은 고치지 않았다). capsule은 data yaml의 category를 명령줄 override(`--data.init_args.category capsule`)로 바꿨다. 코드 변경이 없는 방식이다.

```powershell
cd D:\projects\visionlab\evaluation\anomalib
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/06_patchcore.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\06-ref-mvtec-261026-1.log
& $py anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/06_patchcore.yaml --data.init_args.category capsule 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\06-ref-capsule-261026-1.log
```

다음 표는 1 epoch 레퍼런스(1회) 결과다.

| 데이터 / category | Image AUROC | Pixel AUROC | 참고 | run 폴더 | 로그 |
|---|---|---|---|---|---|
| mvtec / bottle | 1.00000 (0.99999994) | 0.98316 | image F1 1.0000, pixel F1 0.7456 | `outputs/anomalib/Patchcore/LocalMVTecAD/bottle/v0` | `outputs/logs/06-ref-mvtec-261026-1.log` |
| mvtec / capsule | 0.96104 | 0.98706 | image F1 0.9618, pixel F1 0.5280 | `outputs/anomalib/Patchcore/LocalMVTecAD/capsule/v0` | `outputs/logs/06-ref-capsule-261026-1.log` |

적용된 학습 예산은 `max_epochs=1`, `val_split_mode=from_test`, `val_split_ratio=0.4`, `seed=42`, `batch 8/8`, `num_workers=0`, `backbone=wide_resnet50_2`, `layers=[layer2, layer3]`, `pre_trained=true`, `coreset_sampling_ratio=0.1`, `num_neighbors=9`이다(`models/06_patchcore.yaml`과 공유 yaml). 체크포인트는 각 run 폴더의 `weights/lightning/model.ckpt`다. memory bank의 크기는 bottle 16,385 x 1536, capsule 17,169 x 1536이다.

### 2.2 중단한 visa/candle 레퍼런스

사용자가 visa/candle을 평가 category에서 제외하고 mvtec/capsule로 대체했다(2026-10-26). 사유는 coreset 선택 시간이다. visa/candle 레퍼런스는 같은 날 11:23에 시작했고 사용자 결정으로 중단했다. 기록은 다음과 같다(지우지 않았다).

| 항목 | 내용 |
|---|---|
| 상태 | 중단(사용자 결정 2026-10-26, 사유: coreset 선택 시간) |
| 진행률 | 사용자가 확인한 시점은 coreset 6,620 / 70,560 이후다. Claude가 중단 직전(약 11:54)에 로그에서 읽은 값은 7,768 / 70,560(11%), 경과 30분 25초, 속도 4.21 it/s, 남은 예상 4시간 8분 51초다 |
| 중단 방법 | agy가 `Get-CimInstance Win32_Process ... | Stop-Process`로 해당 python 프로세스(PID 7820)만 종료했다. Claude가 `anomalib_run|train.py|predict.py|compare_` 프로세스 0개와 GPU 775 MiB, 0%를 직접 확인했다 |
| 남은 산출물 | `outputs/logs/06-ref-visa-261026-1.log`, `outputs/anomalib/Patchcore/LocalVisa/candle/v0`(`lightning_logs`만 있고 `weights`는 없다)를 지우지 않았다 |
| 관찰 | capsule의 coreset 선택은 17,169회를 1분 14초(229.51 it/s)에 마쳤다. visa는 705,600 x 1536 임베딩에서 4.21 it/s였다. 속도 차이는 약 54배이고, 당시 GPU는 10,900 MiB / 11,264 MiB를 99%로 쓰고 있었다. 원인이 GPU 메모리 한도 근처의 동작인지는 확인하지 않았다 |

visa 항목에 해당하는 native 학습, 순수 추론, 비교 실행은 하지 않았다.

### 2.3 mvtec/capsule 데이터 확인

읽기 전용으로 확인했다. `D:\datasets\mvtec\capsule`에 학습 `train\good` 219장, 테스트 132장(good 23, crack 23, faulty_imprint 22, poke 21, scratch 23, squeeze 20)과 `ground_truth`(crack, faulty_imprint, poke, scratch, squeeze)가 있다. bottle은 학습 209장이다.

### 2.4 에폭별 학습 히스토리와 확인 사항

PatchCore는 loss가 없고 학습이 1 epoch이라 비교할 히스토리가 없다. 레퍼런스와 native의 비교는 memory bank로 대신하며 5.4절에서 다룬다. 두 category 모두 종료 코드 0이고 오류가 없다. 레퍼런스가 category당 1회뿐이라 run 간 편차는 측정하지 못했다.

## 3. 이식 메모

### 3.1 원본과 이식 파일의 대응

다음 표는 새로 만든 파일과 원본의 대응이다. 원본과의 차이는 `difflib`로 확인했다(구현함, 정적 검증).

| 이식 파일 (`src/visionlab/anomaly_detection/models/`) | 원본 | 원본과 다른 줄 |
|---|---|---|
| `patchcore/model.py` | `patchcore/torch_model.py` | import 4줄(`InferenceBatch`, `DynamicBufferMixin`, `TimmFeatureExtractor`, `KCenterGreedy`), `Tiler` 타입 참조 경로, `TimmFeatureExtractor(... pre_trained=False)`와 주석 한 줄(D-19) |
| `patchcore/anomaly_map.py` | `patchcore/anomaly_map.py` | `GaussianBlur2d` import 한 줄 |
| `patchcore/__init__.py` | `patchcore/__init__.py` | `PatchcoreModel`만 노출(D-18 예외) |
| `patchcore/lightning_model.py` (`PatchcoreLitModel`) | `patchcore/lightning_model.py`와 `components/base/memory_bank_module.py`의 hook | native `LightningModule`로 새로 작성. 9개 인자 동일, `_is_fitted` buffer, `fit`, `on_validation_start`, `on_train_epoch_end`, 더미 loss는 원본 그대로, 로컬 backbone 주입(D-19), D-8의 post_processor 갱신 제한, center crop 전처리와 mask 정렬 |
| `components/sampling/k_center_greedy.py` | `components/sampling/k_center_greedy.py` | `SparseRandomProjection` import 한 줄 |
| `components/dimensionality_reduction/random_projection.py` | 같은 이름 | `sklearn.utils.random.sample_without_replacement` import를 `_sparse_random_matrix` 안으로 옮기고 주석 한 줄 |

### 3.2 새로 쓴 코드와 앞 Phase 공유 파일 변경

- **새로 쓴 파일:** `configs/models/patchcore.yaml`, `evaluation/anomalib/compare_patchcore_stats.py`(정보 기록용), `tests/06_patchcore/test_patchcore.py`, `tests/06_patchcore/test_patchcore_lightning.py`. `tests\06_patchcore`는 합계 24개 테스트다.
- **공유 파일 변경:** `common/pre_processor.py`(선택 인자 `center_crop`과 `crop` 메서드), `utils/io.py`(export의 `pre_processor`에 `center_crop` 직렬화, 없는 값은 `None`, `patchcore` 등록), `inference/predictor.py`(`center_crop` 복원, `patchcore` 등록), `evaluation/visionlab/predict.py`(`--model patchcore`, mask에 `crop` 적용), `components/dimensionality_reduction/__init__.py`(`SparseRandomProjection` 노출), `evaluation/anomalib/compare_outputs.py`(`--model patchcore`), `tests/common/test_cli.py`, `tests/common/test_dependency.py`. 영향받을 수 있는 앞 모델은 6.2절의 표에 정리했다.
- **새 부품:** `KCenterGreedy`, `SparseRandomProjection`은 원본 복사다. 새로 구현한 대체 부품은 없다.

### 3.3 이식 중 확인된 사항

- **center crop과 mask 정렬:** 레퍼런스의 `PreProcessor`가 `Resize`, `CenterCrop(224)`, `Normalize`를 이미지와 `gt_mask`에 함께 적용하므로 Pixel AUROC가 224x224 영역에서 계산된다. visionlab은 `PreProcessor(center_crop=(224, 224))`로 이미지를 자르고, `PatchcoreLitModel._outputs`와 `predict.py`가 mask에 같은 crop을 적용한다. native와 순수 PyTorch의 Pixel AUROC가 레퍼런스와 같은 값으로 나와 정렬이 맞음을 확인했다(5.5절).
- **sklearn 지연 import:** 추론 경로는 sklearn 없이 import되고 `fit`에서만 필요하다. 테스트로 확인했다(4.1절).
- **coreset 재현:** 같은 seed에서 native memory bank의 모든 행이 레퍼런스와 정확히 같았다(5.4절). 따라서 사용자 결정 3(불일치 시 주입하지 않음)은 적용할 일이 없었다.

## 4. 적대적 검증

검증은 6-A, 6-B 각각 한 회차씩 했다. 수정 요청은 6-A 1회, 6-B 0회(한도 3회 이내)다.

### 4.1 6-A (순수 모델과 coreset 부품)

| 지적 | 심각도 | 처리 |
|---|---|---|
| 원본과의 `difflib` 대조: `model.py`는 import와 D-19의 `pre_trained=False` 한 줄뿐이고 `forward`, `generate_embedding`, `euclidean_dist`, `nearest_neighbors`, `compute_anomaly_score`가 같음. `anomaly_map.py`, `k_center_greedy.py`는 import 한 줄뿐. `random_projection.py`는 sklearn import 위치뿐 | 확인 | 수정 없음 |
| 실행으로 발견: `test_isolated_import_inference_and_fitting_dependency`가 실패(24개 중 1개). 자식 프로세스가 처음부터 sklearn 등을 차단해 timm 안의 `import torch._dynamo`가 `AttributeError: partially initialized module 'torch._dynamo' has no attribute 'config'`로 실패함. 제품 코드가 아니라 테스트의 차단 시점 문제 | 중간 | 수정 요청 1회차: 외부 라이브러리(`torch`, `timm`)를 먼저 정상 import하고 그 뒤에 차단하도록 바꿈. 마지막 검사는 sys.modules의 전후 차이로 sklearn이 새로 import되지 않았음을 봄 |

수정 뒤 `tests\06_patchcore` 24개가 통과했다(`06-pytest-261026-2.log`).

### 4.2 6-B (Lightning, center crop, export, 비교 스크립트)

| 지적 | 심각도 | 처리 |
|---|---|---|
| `git diff` 검토: `PreProcessor`의 `center_crop` 기본값이 `None`이고 `crop`이 항등이라 앞 모델의 출력이 달라지지 않음. `__call__`은 입력 검증 뒤 crop, 그 뒤 정규화 순서(레퍼런스와 같음) | 확인 | 수정 없음 |
| `DfkdeLitModel`과의 `difflib` 대조: 차이가 생성자 인자, 기본 `PreProcessor(center_crop)`, `Evaluator.default`, `_outputs`의 이미지·mask crop, `subsample_embedding`, export 인자뿐임. `_is_fitted` 조건, 더미 loss, hook, D-8 처리가 같음 | 확인 | 수정 없음 |
| `io.py`는 export의 `pre_processor.center_crop`을 읽을 때 `setdefault(..., None)`으로 기존 형식을 허용함. `predictor.py`는 이를 사용함 | 확인 | 수정 없음 |
| 정적 검토는 `LitModel`, `PreProcessor`, `io.py`, `predictor.py`, `predict.py`, yaml 위주로 했다. `compare_outputs.py`의 patchcore 분기는 줄 단위로 읽지 않고 실행 결과(5.3절)로 확인했다 | 낮음 | 수정 없음 |

남은 위험은 6.2절에 적었다.

## 5. 이식 결과

### 5.1 최종 판정 (1 epoch, Image·Pixel AUROC 차이 0.01 이내)

native는 `train.py`가 test에서 기록한 값이고 순수 PyTorch는 `export/model.pt`만 불러 `predict.py`로 평가한 값이다. 모두 실행으로 검증했다. D-24에 따라 레퍼런스가 보고하는 값과 같은 정의로 비교하며, 순수 PyTorch의 정규화 점수 AUROC도 함께 적는다.

| 데이터 / category | 지표 | 레퍼런스 | native (raw) | 순수 PyTorch (raw) | 순수 PyTorch (정규화) | 차이(native) | 차이(순수 raw) | 판정 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| mvtec / bottle | Image AUROC | 0.999999940 | 1.000000000 | 1.00000000 | 0.99999994 | 6e-8 | 6e-8 | 통과 |
| mvtec / bottle | Pixel AUROC | 0.983161569 | 0.983161569 | 0.98316145 | 0.98316157 | 0 | 1.2e-7 | 통과 |
| mvtec / capsule | Image AUROC | 0.961038947 | 0.961038947 | 0.96103895 | 0.96103895 | 0 | 3e-9 | 통과 |
| mvtec / capsule | Pixel AUROC | 0.987055302 | 0.987055540 | 0.98705554 | 0.98705530 | 2.4e-7 | 2.4e-7 | 통과 |

정규화 점수 기준(D-24의 비교 정의)으로도 레퍼런스와의 차이는 1e-8 이하다. raw와 정규화 값이 0.01 넘게 다르지 않으므로 D-24의 원인 기록 조건에는 해당하지 않는다. F1도 같다(image F1: bottle 1.0000, capsule 0.9618. pixel F1: native는 레퍼런스와 같은 0.74560910, 0.52796662이고 순수 PyTorch는 0.74560738, 0.52797133으로 5e-6 이내이다. 후자는 판정 대상이 아니다).

run 폴더와 로그는 다음과 같다.

| 대상 | 위치 |
|---|---|
| 레퍼런스 | `outputs/anomalib/Patchcore/LocalMVTecAD/bottle/v0`, `outputs/anomalib/Patchcore/LocalMVTecAD/capsule/v0` |
| native | `outputs/visionlab/patchcore/mvtec_bottle/v0`, `outputs/visionlab/patchcore/mvtec_capsule/v0` (가중치 `D:\backbones\wide_resnet50_2-95faca4d.pth` sha256 `95faca4d...50b8`) |
| 순수 PyTorch | 같은 run의 `predict/metrics.json` |
| 로그 | `outputs/logs/06-native-{mvtec,capsule}-261026-1.log`, `06-pure-{mvtec,capsule}-261026-1.log` |

네 native·순수 로그에 `network access is blocked`, `URLError`, `ConnectionError`, `Traceback`, `out of memory`는 0건이다.

### 5.2 테스트와 비교 스크립트

다음 표는 agy가 실행한 묶음의 결과다(Claude가 로그 파일에서 확인). 실행 대상은 PatchCore만이고 앞 모델의 회귀 pytest와 `compare_outputs`는 돌리지 않았다.

| 대상 | 결과 | 로그 |
|---|---|---|
| `tests\06_patchcore` (1회차) | 23 passed, 1 failed (4.1절의 import 분리 테스트) | `06-pytest-261026-1.log` |
| `tests\06_patchcore` (수정 뒤 2회차) | 24 passed | `06-pytest-261026-2.log` |
| `tests\common\test_cli.py -k patchcore` | 1 passed (9 deselected) | `06-pytest-cli-261026-1.log` |
| `tests\common\test_dependency.py` (전체 파일) | 6 passed | `06-pytest-dependency-261026-1.log` |
| `compare_outputs.py --model patchcore` (bottle 체크포인트, capsule 체크포인트) | PASS 23, FAIL 0 (각각). FR-6, FR-23, FR-5(`pred_score`와 `anomaly_map` 최대 차이 0.0, mvtec·visa 입력 8장씩, 224x224), FR-24 | `06-compare-outputs-{mvtec,capsule}-261026-1.log` |

`test_dependency.py`는 모든 순수 모듈의 import 분리 검사를 한 파일이 하므로 PatchCore 모듈 검증을 위해 전체를 돌렸다(Phase 5에서 사용자가 승인한 것과 같은 사유). 이번 Phase에서의 실행은 사용자가 사후에 승인했다(2026-10-26). `compare_outputs`의 FR-5 입력은 data yaml의 category(mvtec은 bottle, visa는 candle)로 만든 실제 샘플이라 capsule 체크포인트로도 같은 입력을 쓴다.

실행한 명령은 다음과 같다(Claude가 보낸 지시 원문. agy가 시간 측정 줄을 덧붙여 하나의 스크립트로 순서대로 실행했고, 승인 창의 전문을 Claude가 agy의 transcript에서 읽어 지시와 같음을 확인한 뒤 한 번 승인했다).

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py -m pytest tests\06_patchcore 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\06-pytest-261026-1.log
& $py -m pytest tests\common\test_cli.py -k patchcore 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\06-pytest-cli-261026-1.log
& $py -m pytest tests\common\test_dependency.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\06-pytest-dependency-261026-1.log
& $py evaluation/visionlab/train.py train --data mvtec --model patchcore 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\06-native-mvtec-261026-1.log
& $py evaluation/visionlab/predict.py --model patchcore --data mvtec 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\06-pure-mvtec-261026-1.log
& $py evaluation/visionlab/train.py train --data mvtec --model patchcore --data.init_args.category capsule 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\06-native-capsule-261026-1.log
& $py evaluation/visionlab/predict.py --model patchcore --data mvtec --category capsule 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\06-pure-capsule-261026-1.log
& $py evaluation/anomalib/compare_outputs.py --model patchcore --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\Patchcore\LocalMVTecAD\bottle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\06-compare-outputs-mvtec-261026-1.log
& $py evaluation/anomalib/compare_outputs.py --model patchcore --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\Patchcore\LocalMVTecAD\capsule\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\06-compare-outputs-capsule-261026-1.log
& $py evaluation/anomalib/compare_patchcore_stats.py --reference D:\projects\visionlab\outputs\anomalib\Patchcore\LocalMVTecAD\bottle\v0\weights\lightning\model.ckpt --native D:\projects\visionlab\outputs\visionlab\patchcore\mvtec_bottle\v0\weights\lightning\model.ckpt --out D:\projects\visionlab\outputs\compare\patchcore_stats_mvtec.json 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\06-stats-mvtec-261026-1.log
& $py evaluation/anomalib/compare_patchcore_stats.py --reference D:\projects\visionlab\outputs\anomalib\Patchcore\LocalMVTecAD\capsule\v0\weights\lightning\model.ckpt --native D:\projects\visionlab\outputs\visionlab\patchcore\mvtec_capsule\v0\weights\lightning\model.ckpt --out D:\projects\visionlab\outputs\compare\patchcore_stats_capsule.json 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\06-stats-capsule-261026-1.log
```

수정 뒤 재실행은 `& $py -m pytest tests\06_patchcore 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\06-pytest-261026-2.log` 한 줄이다. visa 레퍼런스 중단 명령은 2.2절에 적었다. agy는 파일을 고치지 않았다.

### 5.3 memory bank 비교 (정보, 판정 아님)

다음 표는 `compare_patchcore_stats.py`의 출력이다(`06-stats-{mvtec,capsule}-261026-1.log`, `outputs/compare/patchcore_stats_{mvtec,capsule}.json`).

| category | memory bank shape (레퍼런스 / native) | 정확히 같은 행의 비율 | 채널 평균 최대 차이 |
|---|---|---:|---:|
| mvtec / bottle | [16385, 1536] / [16385, 1536] | 1.0 | 0.0 |
| mvtec / capsule | [17169, 1536] / [17169, 1536] | 1.0 | 0.0 |

같은 seed에서 native의 memory bank가 레퍼런스와 모든 행이 같다. 따라서 coreset의 난수 의존(`torch.randint`, 투영 행렬, `sample_without_replacement`)은 이번 실행에서 레퍼런스와 같은 순서로 소비되었다.

### 5.4 crop 정렬 확인

사용자 지시에 따라 Pixel AUROC 차이가 크면 crop 정렬을 먼저 의심하기로 했다. 두 category의 Pixel AUROC 차이가 2.4e-7 이하이고 FR-5의 `anomaly_map`이 224x224에서 최대 차이 0.0이므로, 이미지와 mask의 crop 정렬은 레퍼런스와 같다.

## 6. 결론

- **레퍼런스:** bottle, capsule 확보. visa/candle은 중단(2.2절).
- **mvtec / bottle:** Image·Pixel AUROC가 native와 순수 PyTorch 모두 레퍼런스와 1.2e-7 이내다. 통과.
- **mvtec / capsule:** Image·Pixel AUROC가 native와 순수 PyTorch 모두 레퍼런스와 2.4e-7 이내다. 통과. 정규화 점수 기준(D-24)도 같다.
- **FR-5, FR-6, FR-23, FR-24:** 통과(5.2절). `pred_score`와 `anomaly_map` 최대 차이 0.0이다.
- **테스트:** `tests\06_patchcore` 24개(수정 뒤), `test_cli.py -k patchcore` 1개, `test_dependency.py` 6개가 통과했다.
- **수정 요청:** 6-A 1회(테스트의 차단 시점), 6-B 0회.
- **PLAN의 평가 category:** 이번 Phase는 사용자 결정(2026-10-26)으로 mvtec/bottle과 mvtec/capsule을 판정한다. `specs/PLAN.md`는 고치지 않았고, 이 문서에 이번 Phase 한정 예외로 기록한다.
- **Phase 종료:** 사용자가 종료를 결정했다(2026-10-26).

### 6.1 `specs/` 갱신 항목

`specs/`는 사람이 쓴다. 이번 Phase의 항목은 사용자 요청(2026-10-26)에 따라 다음과 같이 반영했다.

- **D-25 추가:** scikit-learn을 coreset 적합 때만 쓰는 지연 외부 의존으로 허용한다. `specs/DECISIONS.md` D-25에 선택, 적용 범위, INTENT와의 관계(D-23과 같은 긴장 관계, INTENT는 고치지 않음), 근거, 배제한 대안을 적었다.
- **D-6과 FR-8:** `PreProcessor`가 선택 인자 `center_crop`(기본 `None`)으로 중앙 crop도 할 수 있고 기존 모델 동작이 바뀌지 않는다는 점을 `DECISIONS.md` D-6과 `SPEC.md` FR-8에 덧붙였다.
- **PLAN의 평가 category 문구:** 고치지 않았다. 이번 Phase 한정 예외로 이 문서에만 기록하며, 시간이 오래 걸리는 모델에서 필요해지면 사용자 결정으로 바꾼다.
- **코드 주석:** `random_projection.py`의 지연 import 주석이 `(D-25 draft)`로 남아 있다. 동작에는 영향이 없다.

### 6.2 남은 위험과 사용자 판단 대상

- 레퍼런스가 category당 1회뿐이라 run 간 편차는 측정하지 못했다.
- visa/candle은 판정하지 못했다. 705,600 x 1536 임베딩에서 coreset 선택 속도가 크게 떨어졌고 GPU 메모리가 한도에 가까웠던 원인은 확인하지 않았다.
- `tests\common\test_dependency.py` 전체 실행은 사용자가 사후에 승인했다(2026-10-26).
- 앞 모델 회귀는 보류했다(사용자 결정 2026-10-26). 돌리지 않았다. 영향받을 수 있는 공유 파일은 다음 표와 같고, 코드 변경은 모두 기본값 `None`이거나 기존 항목을 건드리지 않는 추가다. codex가 영향 가능성을 지적한 기존 테스트는 STFPM의 전처리·export·Predictor 테스트와 Phase 2~5의 export 복원 테스트이며 수정하지 않았고 실행하지도 않았다.

| 공유 파일 | 변경 | 영향받을 수 있는 앞 모델 |
|---|---|---|
| `src/visionlab/anomaly_detection/common/pre_processor.py` | `center_crop`(기본 `None`)과 `crop` 메서드, `__call__`에서 crop 호출 | 모든 모델의 전처리(기본값이면 출력 동일) |
| `src/visionlab/utils/io.py` | export의 `pre_processor`에 `center_crop` 저장, 읽을 때 기본 `None`, `patchcore` 허용 | 모든 모델의 export 저장·로딩 |
| `src/visionlab/inference/predictor.py` | `PreProcessor`에 `center_crop` 전달, `patchcore` 등록 | 모든 모델의 `Predictor.from_export` |
| `evaluation/visionlab/predict.py` | mask에 `crop` 적용, `--model patchcore` | 모든 모델의 순수 PyTorch 평가(crop이 없으면 항등) |
| `src/visionlab/anomaly_detection/models/components/dimensionality_reduction/__init__.py` | `SparseRandomProjection` 노출 추가 | dfm, dfkde(`PCA` import) |
| `evaluation/anomalib/compare_outputs.py` | `--model patchcore` 분기 | stfpm, efficient_ad, padim, dfm, dfkde 비교 |
| `tests/common/test_cli.py`, `tests/common/test_dependency.py` | patchcore 테스트와 분류 추가 | 같은 파일의 앞 모델 테스트 |
