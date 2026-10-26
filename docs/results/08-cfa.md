# Phase 8. CFA

모델 폴더 이름은 `cfa`이고 원본은 anomalib 2.0.0의 `anomalib\models\image\cfa\`(`C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\Lib\site-packages\anomalib\models\image\cfa\`)이다. 범위와 완료 조건은 `specs/PLAN.md`의 Phase 8과 "Phase 공통 양식"을 따른다. 판정 기준은 category별 Image·Pixel AUROC 차이 0.01 이내다.

## 상태

다음 표는 단계별 진행 상태다. 수정 반복 한도 3회는 단계별로 센다.

| 단계 | 내용 | 상태 |
|---|---|---|
| 레퍼런스 | `anomalib_run.py train` x 2 (mvtec/bottle 1회, visa/candle 1회, 10 epoch, `LocalCfa`) | 실행 완료 (2절) |
| 8-A | 순수 모델 (`models/cfa/model.py`, `anomaly_map.py`, `loss.py`, `tests/08_cfa`의 순수 부분) | 구현, 검증, 실행 완료 (수정 요청 0회) |
| 8-B | Lightning (`CfaLitModel`, centroid 초기화, AdamW 학습, configs, export·Predictor 등록, 비교 스크립트) | 구현, 검증, 실행 완료 (수정 요청 0회, 결과 5절) |
| 회귀 실행 | 사용자 승인 회귀 실행 1회 (`tests\common`만, 8-B 뒤) | 실행 완료, 34 passed (5.2절) |

처리 구성요소(PreProcessor, PostProcessor, Evaluator), `DynamicBufferMixin`, `GaussianBlur2d`, `dryrun_find_featuremap_dims`는 앞 Phase의 것을 재사용한다.

## 사용자 결정 (2026-10-26)

다음은 Phase 시작 전에 사용자가 정한 항목이다.

| 항목 | 결정 |
|---|---|
| 평가 category | PLAN대로 mvtec/bottle과 visa/candle이다. `specs/PLAN.md`는 고치지 않는다 |
| 학습 에폭 | 원본이 `max_epochs`를 고정하지 않으므로 공유 yaml의 10 epoch를 레퍼런스와 native에 똑같이 쓴다(D-22) |
| einops | 같은 결과의 torch 연산(`reshape`, `permute`)으로 대체한다. 새 의존과 새 결정을 만들지 않는다. 테스트 파일에서만 einops를 import해 `rearrange`와 `torch.equal`로 동일성을 확인하고 `src/visionlab`에는 einops import가 남지 않게 한다 |
| sklearn `KMeans` | `gamma_c > 1` 분기 안에서 지연 import하고 D-25의 적용 범위를 `KMeans`까지 넓힌다. 이 분기는 기본값(`gamma_c=1`)에서 실행되지 않아 실행 검증이 없다. `tests/common/test_dependency.py`의 sklearn 허용 검사를 이에 맞게 바꾼다 |
| FR-4 예외 | `anomaly_map`이 `[B,H,W]`가 되는 원본 동작을 유지하고 `specs/SPEC.md` FR-4에 예외 문구를 추가한다(image 전용 모델 예외와 같은 형식) |
| 앞 모델 회귀 | 보류한다. 다만 공유 테스트의 판정 규칙이 바뀌므로 8-B 뒤에 `tests\common`만 한 번 실행한다(사용자 승인, 2026-10-26). 01~07은 실행하지 않는다 |
| FR-5 허용 오차 | `atol=1e-6, rtol=1e-5` (Phase 2~7과 같음) |
| 에폭별 히스토리 | 원본이 `train_loss`를 로그하지 않아 레퍼런스에 히스토리가 없으므로 비교는 해당 없음으로 기록한다 |
| `specs/` 변경 | D-25 적용 범위 확장과 FR-4 예외 두 곳뿐이다. 판정·기록 단계에서 이 문서와 함께 반영한다. PLAN은 고치지 않는다 |
| 01~07 모델별 회귀 (최종 보고 뒤, 2026-10-26) | 보류한다(사용자 결정). 돌리지 않고 영향받을 수 있는 공유 파일은 6.2절의 표에 남긴다 |
| 계획에 없던 추가 두 가지 (2026-10-26) | 수용한다(사용자 결정). `_CfaLitVisualizer`(batch 1 시각화 shape 복원)와 `on_train_start`의 `memory_bank` 스칼라 복원이다. 3.3절 참고 |
| `specs/` 추가 변경 (2026-10-26) | 없다. 반영한 두 곳(D-25, FR-4)은 그대로 두고 더 고치지 않는다. PLAN은 건드리지 않는다 |
| Phase 종료 (2026-10-26) | 사용자가 종료를 결정했다. `Phase 08: cfa ...` 커밋으로 푸시한다 |
| 실행 범위 | CFA의 명령만 실행한다(위 회귀 1회만 예외) |
| 학습 시간 관문 | visa 레퍼런스의 첫 1~2 에폭 속도로 전체 예상이 약 2시간을 넘으면 계속하기 전에 사용자에게 보고한다. category는 스스로 바꾸지 않는다 |
| 판정 | Image·Pixel AUROC를 판정한다. 정규화 점수 기준 값(D-24)과 raw 값을 함께 기록한다 |

## 1. 모델 노트

### 1.1 원본과 생성자

다음 표는 원본 파일과 이식 위치의 대응이다(D-15, D-18).

| 원본 (`anomalib\models\image\cfa\`) | 이식 위치 | 비고 |
|---|---|---|
| `torch_model.py` (`CfaModel`, `Descriptor`, `CoordConv2d`, `AddCoords`, `get_return_nodes`, `get_feature_extractor`) | `src/visionlab/anomaly_detection/models/cfa/model.py` | import 변경, einops 제거, `KMeans` 지연 import, 사전학습 가중치를 내려받지 않음(D-19), `initialize_centroid`의 batch 형식 |
| `anomaly_map.py` (`AnomalyMapGenerator`) | `.../models/cfa/anomaly_map.py` | `GaussianBlur2d` import 변경, einops 제거 |
| `loss.py` (`CfaLoss`) | `.../models/cfa/loss.py` | 변경 없음 |
| `lightning_model.py` (`Cfa`) | `.../models/cfa/lightning_model.py` (`CfaLitModel`) | native `LightningModule`로 새로 작성 |
| `__init__.py` | `.../models/cfa/__init__.py` | `lightning_model`을 import하지 않도록 고침(D-18 예외) |

생성자는 다음과 같다. `CfaLitModel`은 `Cfa`와 인자 이름·순서·기본값이 같아야 한다(FR-6, D-5).

| 클래스 | 시그니처 |
|---|---|
| `Cfa` | `backbone="wide_resnet50_2"`, `gamma_c=1`, `gamma_d=1`, `num_nearest_neighbors=3`, `num_hard_negative_features=3`, `radius=1e-5`, `pre_processor=True`, `post_processor=True`, `evaluator=True`, `visualizer=True` (10개) |
| `CfaModel` | `backbone`, `gamma_c`, `gamma_d`, `num_nearest_neighbors`, `num_hard_negative_features`, `radius` (모두 기본값 없음) |

`Cfa`와 `CfaModel`에는 `pre_trained` 인자가 없고 항상 사전학습 backbone을 쓴다. 따라서 `CfaLitModel`은 항상 로컬 가중치를 주입한다.

### 1.2 학습과 평가 동작

원본의 동작은 다음과 같다.

- **모델:** torchvision `wide_resnet50_2`를 `create_feature_extractor`로 `layer1`, `layer2`, `layer3`만 남긴 `GraphModule`이 특징을 만든다(`no_grad`, 매 `forward`에서 `eval()`). `Descriptor`가 layer마다 `avg_pool2d(3, 1, 1)`한 특징을 첫 layer 해상도에 bilinear로 맞춰 합치고(1792채널) `CoordConv2d`(좌표 채널 2개를 붙인 1x1 합성곱)로 변환한다.
- **centroid 초기화:** `on_train_start`가 `trainer.datamodule.train_dataloader()`를 한 번 순회하며 `Descriptor` 출력의 batch 평균을 증분 평균으로 모아 `memory_bank`에 둔다. 이 순회의 이미지는 데이터셋이 크기만 맞춘(Resize) 값이며 `PreProcessor`의 정규화를 거치지 않는다(anomalib은 정규화를 학습 루프의 hook에서 한다). `gamma_c > 1`이면 `KMeans`로 줄인다. `memory_bank`는 `(채널, 위치 수)`로 전치된다.
- **학습:** 학습 가능한 파라미터는 `Descriptor`뿐이다(특징 추출은 `no_grad`). `AdamW(lr=1e-3, weight_decay=5e-4, amsgrad=True)`, 손실 `CfaLoss`(attraction과 repulsion, radius 1e-5), `backward`는 `retain_graph=True`로 재정의된다. `train_loss`를 로그하지 않는다.
- **추론:** `compute_distance`가 `Descriptor` 출력과 `memory_bank`의 제곱 거리를 구하고, `AnomalyMapGenerator`가 최근접 `num_nearest_neighbors`개로 점수를 만들어 입력 크기로 bilinear 보간하고 `GaussianBlur2d`(sigma 4)를 씌운다. `anomaly_map`에는 `.squeeze()`가 적용되어 `[B,H,W]`(batch 1이면 `[H,W]`)이고 `pred_score`는 그 `amax`다.
- **`trainer_arguments`:** `{"gradient_clip_val": 0, "num_sanity_val_steps": 0}`이며 `max_epochs`는 고정하지 않는다(공유 yaml의 10 epoch).

### 1.3 외부 의존과 로컬 자산

- **외부 의존:** torchvision(`models`, `feature_extraction.create_feature_extractor`), kornia(`GaussianBlur2d`, D-23), tqdm이다. einops는 torch 연산으로 대체해 의존에서 뺀다. sklearn `KMeans`는 `gamma_c > 1`일 때만 지연 import한다.
- **로컬 가중치:** `paths.toml [backbones]`의 `wide_resnet50_2`(torchvision 형식, 이미 있다). 레퍼런스는 `local_models.LocalCfa`가 `get_feature_extractor`를 `weights=None`과 `load_state_dict`로 바꾼다. visionlab은 `CfaLitModel.__init__`이 `load_teacher_weights(self.model, backbone)`로 주입한다(D-19). `CfaModel`이 `.feature_extractor`(GraphModule)를 가지므로 기존 헬퍼가 그대로 맞고, 파일에 있는 `layer4`, `fc` 키는 헬퍼가 허용한다.
- **데이터:** MVTec `bottle`(학습 209장), VisA `candle`(학습 900장). 분할은 `from_test`, 0.4, seed 42, batch 8/8, `num_workers=0`이다.
- **학습 예산:** 공유 yaml의 `max_epochs=10`이다.

### 1.4 이식에서 의도적으로 달라지는 곳과 유지하는 곳

- **pretrained 주입(D-19):** 순수 `CfaModel`은 `weights=None`으로 backbone 구조만 만든다(원본의 `pretrained=True`로 내려받는 경로를 쓰지 않는다). `CfaLitModel`이 로컬 파일을 불러온다.
- **einops 제거:** `rearrange` 네 곳을 같은 결과의 torch 연산으로 바꾼다. 테스트에서만 einops로 동일성을 확인한다.
- **`KMeans` 지연 import:** `initialize_centroid`의 `gamma_c > 1` 분기 안으로 옮긴다(D-25 적용 범위 확장). 이 분기는 실행 검증이 없다.
- **`initialize_centroid`의 batch 형식:** 원본은 anomalib `Batch`의 `.image`를 읽고 visionlab의 DataLoader는 `(image, target)` 튜플을 낸다. 이미지만 읽도록 바꾼다. 정규화를 하지 않는 순회는 원본 그대로 둔다.
- **원본 동작 유지:** `forward`의 `squeeze()`, `get_scale`을 `gamma_c`와 무관하게 호출하는 것, `torch.ones(1, requires_grad=True) * radius`로 만든 `radius`, `AddCoords`의 `int32` 좌표 계산, `fit`을 시작할 때마다(재개 포함) centroid를 다시 계산하는 것, `backward(retain_graph=True)`는 고치지 않는다.
- **히스토리 비교:** 해당 없음. 레퍼런스가 `train_loss`를 로그하지 않는다. 원본 동작을 유지하므로 로그를 추가하지 않는다.
- **FR-5 허용 오차:** `pred_score`와 `anomaly_map`은 `atol=1e-6, rtol=1e-5`, backbone 가중치는 `torch.equal`이다.

## 2. 레퍼런스

학습 예산은 공유 yaml의 10 epoch다(D-22). 저장된 `lightning_logs/version_0/config.yaml`에 `max_epochs: 10`이 있고 로그에 `Epoch 9`까지 학습한 기록이 있다.

### 2.1 10 epoch 레퍼런스

2026-10-26에 agy가 `anomalib_run.py train --config default.yaml --config data/<data>.yaml --config models/08_cfa.yaml`을 데이터셋마다 1회 실행했다(실행으로 검증함). `08_cfa.yaml`은 `local_models.LocalCfa`를 쓰며, 이 어댑터가 `get_feature_extractor`를 `weights=None`과 로컬 `load_state_dict`로 바꾼다. Claude는 로그 파일에서 값을 직접 확인했다. 두 로그 모두 `network access is blocked`, `URLError`, `ConnectionError`, `Traceback`, `out of memory`는 0건이다(로그의 `NativeCommandError`는 PowerShell이 stderr의 경고를 표시한 것이다). 로그는 UTF-16이다. 승인 창의 스크립트 전문은 Claude가 agy의 transcript에서 읽어 지시한 명령과 같음을 확인했다.

agy가 PowerShell에서 실행한 명령은 다음과 같다(Claude가 보낸 지시 원문. 파일은 고치지 않았다).

```powershell
cd D:\projects\visionlab\evaluation\anomalib
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/08_cfa.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\08-ref-mvtec-261026-1.log
& $py anomalib_run.py train --config default.yaml --config data/visa.yaml --config models/08_cfa.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\08-ref-visa-261026-1.log
```

다음 표는 10 epoch 레퍼런스(1회) 결과다.

| 데이터 / category | Image AUROC | Pixel AUROC | 참고 | run 폴더 | 로그 |
|---|---|---|---|---|---|
| mvtec / bottle | 1.00000 (0.99999994) | 0.98670 | image F1 1.0000, pixel F1 0.7414 | `outputs/anomalib/LocalCfa/LocalMVTecAD/bottle/v0` | `outputs/logs/08-ref-mvtec-261026-1.log` |
| visa / candle | 0.86250 | 0.99001 | image F1 0.8264, pixel F1 0.2850 | `outputs/anomalib/LocalCfa/LocalVisa/candle/v0` | `outputs/logs/08-ref-visa-261026-1.log` |

run 폴더 이름이 `LocalCfa`인 것은 어댑터 클래스 이름이다. 적용된 학습 예산은 `max_epochs=10`, `val_split_mode=from_test`, `val_split_ratio=0.4`, `seed=42`, `batch 8/8`, `num_workers=0`, `deterministic=warn`, `backbone=wide_resnet50_2`, `gamma_c=1`, `gamma_d=1`, `num_nearest_neighbors=3`, `num_hard_negative_features=3`, `radius=1e-5`이다(저장된 `config.yaml`과 `models/08_cfa.yaml`). 체크포인트는 각 run 폴더의 `weights/lightning/model.ckpt`이고 크기는 193,391,178 bytes다.

### 2.2 학습 시간 관문

사용자 결정에 따라 visa 레퍼런스의 첫 1~2 에폭 속도로 전체 예상이 약 2시간을 넘으면 계속하기 전에 보고하기로 했다. 계획의 추정은 bottle 3~5분, visa 8~12분이었다. 실제 속도는 visa 에폭당 약 40초(약 2.7 it/s)였고, 파일 시각으로 bottle 레퍼런스가 약 4분(16:10 시작, 체크포인트 16:14), visa 레퍼런스가 약 8분(체크포인트 16:22)이었다. 추정과 맞았고 관문 기준을 넘지 않아 보고 없이 진행했다.

### 2.3 에폭별 학습 히스토리

해당 없음. 원본이 `train_loss`를 로그하지 않아 레퍼런스에 히스토리가 없다. 원본 동작을 유지하므로 native도 로그를 추가하지 않았다. 레퍼런스와 native의 비교는 학습된 `Descriptor` 가중치와 `memory_bank`로 대신하며 5.3절에서 다룬다.

## 3. 이식 메모

### 3.1 원본과 이식 파일의 대응

다음 표는 새로 만든 파일과 원본의 대응이다. 원본과의 차이는 `difflib`로 확인했다(구현함, 정적 검증).

| 이식 파일 (`src/visionlab/anomaly_detection/models/cfa/`) | 원본 | 원본과 다른 줄 |
|---|---|---|
| `model.py` | `torch_model.py` | `einops`, `sklearn.cluster.KMeans` 최상단 import 제거, import 3줄(`InferenceBatch`, `DynamicBufferMixin`, `dryrun_find_featuremap_dims`), `get_feature_extractor`의 `weights=None`과 주석(D-19), `initialize_centroid`의 batch 읽기(`data[0]`), `KMeans` 지연 import(`gamma_c > 1` 분기 안), `rearrange` 세 곳을 `permute`와 `reshape`로 대체 |
| `anomaly_map.py` | `anomaly_map.py` | `einops` 제거, `GaussianBlur2d` import 한 줄, `rearrange` 한 곳을 `reshape`와 `permute`로 대체 |
| `loss.py` | `loss.py` | 없음 |
| `__init__.py` | `__init__.py` | `CfaModel`만 노출(D-18 예외) |
| `lightning_model.py` (`CfaLitModel`) | `lightning_model.py` | native `LightningModule`로 새로 작성. 10개 인자 동일, `on_train_start`의 centroid 초기화, `AdamW(lr 1e-3, weight_decay 5e-4, amsgrad)`, `retain_graph=True`의 `backward`, 로컬 backbone 주입(D-19), D-8의 post_processor 갱신 제한 |

### 3.2 새로 쓴 코드와 앞 Phase 공유 파일 변경

- **새로 쓴 파일:** `configs/models/cfa.yaml`, `evaluation/anomalib/compare_cfa_stats.py`(정보 기록용), `tests/08_cfa/test_cfa.py`, `tests/08_cfa/test_cfa_lightning.py`. `tests\08_cfa`는 합계 35개 테스트다.
- **공유 파일 변경:** `utils/io.py`(`load_export`가 `cfa` 허용), `inference/predictor.py`(모델 dict에 등록), `evaluation/visionlab/predict.py`(`--model cfa`), `evaluation/anomalib/compare_outputs.py`(`--model cfa`, `LocalCfa`가 바꾸는 전역 factory를 `finally`로 복원), `tests/common/test_cli.py`, `tests/common/test_dependency.py`. `common/` 아래 파일은 바꾸지 않았다. 영향 범위는 6.2절의 표에 정리했다.
- **`test_dependency.py`의 판정 규칙 변경:** 이전에는 `random_projection.py`의 `SparseRandomProjection._sparse_random_matrix` 안의 `from sklearn.utils.random import sample_without_replacement` 한 곳만 허용했다. 이번에는 `SKLEARN_IMPORTS` 상수(파일, 클래스, 메서드, 모듈, 이름) 두 항목만 허용한다. 하나는 위의 것이고 다른 하나는 `cfa/model.py`의 `CfaModel.initialize_centroid` 안의 `from sklearn.cluster import KMeans`다. 지정한 메서드 안의 단일 이름 `ImportFrom` 하나씩만 허용하고, 최상단, 중첩 함수, 다른 위치, 별칭, 복수 이름, 중복 import는 실패한다. 이를 확인하는 테스트도 추가되었다.
- **새 부품:** 순수 모델은 원본 복사다. 새로 구현한 대체 부품은 없다(einops 대체는 `reshape`와 `permute`뿐이다).

### 3.3 이식 중 확인된 사항

- **centroid는 정규화 전 이미지로 만든다:** anomalib은 `Resize`만 Dataset에 적용하고 정규화는 학습 루프의 `PreProcessor` hook에서 하므로, `on_train_start`가 순회하는 `train_dataloader()`의 이미지는 크기만 256으로 맞춘 정규화 전 값이다. 학습과 평가 때의 특징은 정규화된 이미지에서 나온다. 원본 동작이며 고치지 않았다. visionlab도 같게 `CfaLitModel.on_train_start`에서 `PreProcessor`를 적용하지 않는다. `compare_outputs`의 `raw centroid initialization`(최대 차이 0.0)과 native `memory_bank`의 레퍼런스 일치(5.3절)가 이를 확인한다.
- **두 번째 `fit`과 재개:** 원본 `initialize_centroid`는 `memory_bank`가 이미 `(C,N)`이면 첫 증분 평균에서 shape가 충돌한다(codex 질문). Claude가 권장안을 택해 `CfaLitModel.on_train_start`가 `memory_bank`를 0차원 0 텐서로 되돌린 뒤 원본 함수를 부르도록 했다. 첫 `fit`의 결과는 바뀌지 않고 재개와 두 번째 `fit`이 동작한다. 원본과 다른 점이며 사용자가 수용했다(2026-10-26).
- **batch 1의 시각화(미사용 추가, 사용자 수용 2026-10-26):** 원본 `forward`의 `squeeze()` 때문에 batch가 1장이면 `anomaly_map`이 `[H,W]`, `pred_score`가 0차원이다. 공유 시각화 callback을 위해 `CfaLitModel`이 `LitVisualizer`의 작은 하위 클래스(`_CfaLitVisualizer`)에서 batch 축만 복원한다. 모델, 평가, `Predictor` 출력은 원본 shape를 유지한다. 계획에 없던 추가이며 `visualizer: false`인 이번 실행에는 쓰이지 않았다. 단위 테스트만 거친 미사용 추가다.
- **torchvision 가중치 주입:** `CfaModel`이 `.feature_extractor`(GraphModule)를 가지므로 기존 `load_teacher_weights(self.model, backbone)`가 그대로 맞았다. 파일의 `layer4`, `fc` 키는 헬퍼가 허용한다. FR-23 비교에서 `LocalCfa`의 특징 추출기 state와 `torch.equal`로 일치했다.
- **초기 가중치:** 같은 seed 42로 새로 만든 레퍼런스와 visionlab 순수 모델의 `Descriptor` 초기 가중치가 서로 같다(`INFO`, 정보 기록).

## 4. 적대적 검증

검증은 8-A, 8-B 각각 한 회차씩 했다. 수정 요청은 8-A 0회, 8-B 0회(한도 3회 이내)다.

### 4.1 8-A (순수 모델, anomaly_map, loss)

| 지적 | 심각도 | 처리 |
|---|---|---|
| 원본과의 `difflib` 대조: `model.py`는 import, `weights=None`과 주석, `data[0]`, `KMeans` 지연 import, `rearrange` 세 곳의 대체뿐이고 `forward`, `squeeze()`, `compute_distance`, `Descriptor`, `CoordConv2d`, `AddCoords`가 같음. `anomaly_map.py`는 `einops`, import 한 줄, `rearrange` 한 곳뿐. `loss.py`는 변경 없음 | 확인 | 수정 없음 |

### 4.2 8-B (Lightning, export, 비교 스크립트)

| 지적 | 심각도 | 처리 |
|---|---|---|
| `ReverseDistillationLitModel`과의 `difflib` 대조: 차이가 생성자 인자, `load_teacher_weights(self.model, ...)`, `on_train_start`, `train_loss` 로그 없음(원본 동작), `backward` 재정의, AdamW, export 인자, 시각화 하위 클래스뿐. hook, `_outputs`, D-8 처리가 같음. 공유 파일 변경은 등록뿐 | 확인 | 수정 없음 |
| codex 질문: 두 번째 `fit`과 재개에서 `memory_bank` shape 충돌(3.3절). 원본에도 있는 한계 | 낮음 | 권장안으로 답함(`LitModel`에서 스칼라로 되돌린 뒤 호출) |
| 정적 검토는 `LitModel`, 등록 코드, yaml 위주로 했다. `compare_outputs.py`의 cfa 분기는 줄 단위로 읽지 않고 실행 결과(5.2절)로 확인했다 | 낮음 | 수정 없음 |

남은 위험은 6.2절에 적었다.

## 5. 이식 결과

### 5.1 최종 판정 (10 epoch, Image·Pixel AUROC 차이 0.01 이내)

native는 `train.py`가 test에서 기록한 값이고 순수 PyTorch는 `export/model.pt`만 불러 `predict.py`로 평가한 값이다. 모두 실행으로 검증했다. D-24에 따라 레퍼런스가 보고하는 값과 같은 정의로 비교하며, 순수 PyTorch의 정규화 점수 AUROC도 함께 적는다.

| 데이터 / category | 지표 | 레퍼런스 | native (raw) | 순수 PyTorch (raw) | 순수 PyTorch (정규화) | 차이(native) | 차이(순수 raw) | 판정 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| mvtec / bottle | Image AUROC | 0.999999940 | 1.000000000 | 1.00000000 | 0.99999994 | 6e-8 | 6e-8 | 통과 |
| mvtec / bottle | Pixel AUROC | 0.986701131 | 0.986701012 | 0.98670113 | 0.98670113 | 1.2e-7 | 0 | 통과 |
| visa / candle | Image AUROC | 0.862499952 | 0.862499952 | 0.86249995 | 0.86249995 | 0 | 0 | 통과 |
| visa / candle | Pixel AUROC | 0.990010858 | 0.990010858 | 0.99001074 | 0.99001080 | 0 | 1.2e-7 | 통과 |

raw와 정규화 값이 0.01 넘게 다른 곳은 없어 D-24의 원인 기록 조건에는 해당하지 않는다. F1도 같다(image F1: bottle 1.0000, visa 0.8264. pixel F1: native는 레퍼런스와 같은 0.74137235, 0.28495252이고 순수 PyTorch는 0.74137115, 0.28495252로 2e-6 이내이다. 판정 대상이 아니다).

run 폴더와 로그는 다음과 같다.

| 대상 | 위치 |
|---|---|
| 레퍼런스 | `outputs/anomalib/LocalCfa/LocalMVTecAD/bottle/v0`, `outputs/anomalib/LocalCfa/LocalVisa/candle/v0` |
| native | `outputs/visionlab/cfa/mvtec_bottle/v0`, `outputs/visionlab/cfa/visa_candle/v0` (가중치 `D:\backbones\wide_resnet50_2-95faca4d.pth` sha256 `95faca4d...50b8`) |
| 순수 PyTorch | 같은 run의 `predict/metrics.json` |
| 로그 | `outputs/logs/08-native-{mvtec,visa}-261026-1.log`, `08-pure-{mvtec,visa}-261026-1.log` |

네 native·순수 로그에 `network access is blocked`, `URLError`, `ConnectionError`, `Traceback`, `out of memory`는 0건이다.

### 5.2 테스트와 비교 스크립트

다음 표는 agy가 실행한 묶음의 결과다(Claude가 로그 파일에서 확인). 실행 대상은 CFA뿐이고, 아래 `tests\common` 한 번만 사용자가 승인한 예외다.

| 대상 | 결과 | 로그 |
|---|---|---|
| `tests\08_cfa` | 35 passed | `08-pytest-261026-1.log` |
| `tests\common` (사용자 승인 회귀 실행, 2026-10-26) | 34 passed | `08-regress-common-261026-1.log` |
| `compare_outputs.py --model cfa` (bottle 체크포인트, visa 체크포인트) | PASS 24, FAIL 0, INFO 1 (각각). FR-6, FR-23(특징 추출기 `torch.equal`), FR-5(`pred_score [8]`과 `anomaly_map [8,256,256]` 최대 차이 0.0, 단 visa 체크포인트의 mvtec 입력 `anomaly_map`은 2.4e-7로 허용 오차 이내), raw centroid 초기화 일치(최대 차이 0.0), FR-24. INFO는 seed 42 초기 `Descriptor` 가중치 일치 | `08-compare-outputs-{mvtec,visa}-261026-1.log` |

`tests\common`은 Phase 7 직전의 회귀 실행(30개)보다 4개 늘었다(`test_cli.py`의 cfa 테스트 2개 등). 01~07의 모델별 테스트는 실행하지 않았다. 이 실행은 `test_dependency.py`의 판정 규칙이 바뀌어 사용자가 승인한 것이다.

실행한 명령은 다음과 같다(Claude가 보낸 지시 원문. agy가 시간 측정 줄을 덧붙여 하나의 스크립트로 순서대로 실행했고, 승인 창의 전문을 Claude가 agy의 transcript에서 읽어 지시와 같음을 확인한 뒤 한 번 승인했다).

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py -m pytest tests\08_cfa 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\08-pytest-261026-1.log
& $py -m pytest tests\common 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\08-regress-common-261026-1.log
& $py evaluation/visionlab/train.py train --data mvtec --model cfa 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\08-native-mvtec-261026-1.log
& $py evaluation/visionlab/predict.py --model cfa --data mvtec 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\08-pure-mvtec-261026-1.log
& $py evaluation/visionlab/train.py train --data visa --model cfa 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\08-native-visa-261026-1.log
& $py evaluation/visionlab/predict.py --model cfa --data visa 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\08-pure-visa-261026-1.log
& $py evaluation/anomalib/compare_outputs.py --model cfa --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\LocalCfa\LocalMVTecAD\bottle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\08-compare-outputs-mvtec-261026-1.log
& $py evaluation/anomalib/compare_outputs.py --model cfa --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\LocalCfa\LocalVisa\candle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\08-compare-outputs-visa-261026-1.log
& $py evaluation/anomalib/compare_cfa_stats.py --reference D:\projects\visionlab\outputs\anomalib\LocalCfa\LocalMVTecAD\bottle\v0\weights\lightning\model.ckpt --native D:\projects\visionlab\outputs\visionlab\cfa\mvtec_bottle\v0\weights\lightning\model.ckpt --out D:\projects\visionlab\outputs\compare\cfa_stats_mvtec.json 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\08-stats-mvtec-261026-1.log
& $py evaluation/anomalib/compare_cfa_stats.py --reference D:\projects\visionlab\outputs\anomalib\LocalCfa\LocalVisa\candle\v0\weights\lightning\model.ckpt --native D:\projects\visionlab\outputs\visionlab\cfa\visa_candle\v0\weights\lightning\model.ckpt --out D:\projects\visionlab\outputs\compare\cfa_stats_visa.json 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\08-stats-visa-261026-1.log
```

agy는 파일을 고치지 않았다.

### 5.3 학습된 가중치와 memory bank 비교 (정보, 판정 아님)

다음 표는 `compare_cfa_stats.py`의 출력이다(`08-stats-{mvtec,visa}-261026-1.log`, `outputs/compare/cfa_stats_{mvtec,visa}.json`).

| 텐서 | shape (레퍼런스 / native) | bottle 최대 절대 차이 | visa 최대 절대 차이 |
|---|---|---:|---:|
| `memory_bank` | [1792, 4096] / [1792, 4096] | 0.0 | 0.0 |
| `descriptor.layer.weight` | [1792, 1792, 1, 1] | 0.0 | 0.0 |
| `descriptor.layer.bias` | [1792] | 0.0 | 0.0 |
| `descriptor.layer.conv2d.weight` | [1792, 1794, 1, 1] | 0.0 | 0.0 |
| `descriptor.layer.conv2d.bias` | [1792] | 0.0 | 0.0 |

학습된 `Descriptor`의 모든 가중치와 `memory_bank`가 레퍼런스와 비트 단위로 같다(10 epoch 학습 뒤). 학습 경로(초기 가중치, centroid, 데이터 순서, 손실, optimizer)가 레퍼런스와 같다는 뜻이다.

## 6. 결론

- **레퍼런스:** bottle, visa 확보(10 epoch).
- **mvtec / bottle:** Image·Pixel AUROC가 native와 순수 PyTorch 모두 레퍼런스와 1.2e-7 이내다. 통과.
- **visa / candle:** Image·Pixel AUROC가 native와 순수 PyTorch 모두 레퍼런스와 1.2e-7 이내다. 통과. 정규화 점수 기준(D-24)도 같다.
- **FR-5, FR-6, FR-23, FR-24:** 통과(5.2절). `pred_score`와 `anomaly_map` 최대 차이가 0.0(한 곳은 2.4e-7)이고 centroid 초기화도 일치한다.
- **에폭별 히스토리:** 해당 없음(원본이 `train_loss`를 로그하지 않는다). 대신 학습된 가중치와 `memory_bank`가 비트 단위로 같다.
- **테스트:** `tests\08_cfa` 35개와 사용자 승인 회귀 실행 `tests\common` 34개가 통과했다. 실패는 없다.
- **수정 요청:** 8-A 0회, 8-B 0회.
- **`specs/` 변경:** 사용자 결정에 따라 D-25의 적용 범위를 `KMeans`까지 넓히고 FR-4에 CFA 예외를 추가했다(6.1절). 더 고치지 않는다. PLAN은 고치지 않았다.
- **Phase 종료:** 사용자가 종료를 결정했다(2026-10-26).

### 6.1 `specs/` 갱신 항목

`specs/`는 사람이 쓴다. 이번 Phase의 두 항목은 사용자 결정(2026-10-26)에 따라 Claude가 반영했다. 커밋 전이며 변경 줄은 다음과 같다.

- **D-25 확장(`specs/DECISIONS.md`):** 제목을 "scikit-learn을 적합 때만 쓰는 지연 외부 의존으로 허용한다"로 바꾸고, CFA의 `initialize_centroid`가 `gamma_c > 1`일 때 쓰는 `sklearn.cluster.KMeans`를 같은 방식으로 지연 import한다고 선택에 추가했다. 적용 범위를 `sample_without_replacement`와 `KMeans` 두 곳으로 넓히고, `test_dependency.py`의 허용 목록 상수 검사와 `tests/06_patchcore`, `tests/08_cfa`의 sklearn 차단 테스트를 적었다. `KMeans` 분기는 기본값에서 실행되지 않아 이 분기의 수치는 실행으로 검증하지 않았다는 점을 본문에 명시했다. 근거에 `KMeans`는 기본값에서 쓰이지 않아 import 위치만 옮겼다는 문장을 더했고, 배제한 대안에 `KMeans`의 독립 구현을 더했다.
- **FR-4 예외(`specs/SPEC.md`):** CFA는 원본의 `squeeze()`를 유지해 anomaly map이 `[B,H,W]`(batch 1이면 `[H,W]`), score가 `[B]`(batch 1이면 0차원)라는 문구와 그 shape 확인을 판정 방법에 추가했다(image 전용 모델 예외와 같은 형식).
- **PLAN:** 고치지 않았다.

### 6.2 남은 위험과 사용자 판단 대상

- 레퍼런스가 category당 1회뿐이다. 다만 native가 레퍼런스와 AUROC뿐 아니라 학습된 가중치와 `memory_bank`까지 비트 단위로 같아 학습 경로의 편차는 작아 보인다.
- `gamma_c > 1`의 `KMeans` 분기는 실행으로 검증하지 않았다(기본값에서 실행되지 않음, D-25 본문에 명시).
- `_CfaLitVisualizer`와 batch 1 평가의 shape 복원은 미사용 추가다. 단위 테스트로만 확인했고, 실제 시각화 실행은 하지 않았다(`visualizer: false`). 사용자가 수용했다(2026-10-26).
- 앞 모델 회귀는 `tests\common` 한 번만 실행했다(사용자 승인). 01~07의 모델별 테스트는 보류했다(사용자 결정 2026-10-26). 실행하지 않았다. 변경은 모두 기존 항목을 건드리지 않는 등록과 분기 추가이고, `test_dependency.py`의 규칙 변경은 `tests\common`이 통과했다. 다음 표는 영향받을 수 있는 공유 파일이다.

| 공유 파일 | 변경 | 영향받을 수 있는 앞 모델 |
|---|---|---|
| `src/visionlab/utils/io.py` | `load_export`가 허용하는 이름에 `cfa` 추가 | 모든 모델의 export 로딩 |
| `src/visionlab/inference/predictor.py` | `from_export` 모델 dict에 `cfa` 등록 | 모든 모델의 `Predictor.from_export` |
| `evaluation/visionlab/predict.py` | `--model` 선택지에 추가 | 모든 모델의 순수 PyTorch 평가 |
| `evaluation/anomalib/compare_outputs.py` | `--model cfa` 분기, `LocalCfa`의 전역 factory를 `finally`로 복원 | stfpm, efficient_ad, padim, dfm, dfkde, patchcore, reverse_distillation 비교 |
| `tests/common/test_cli.py`, `tests/common/test_dependency.py` | cfa 테스트, sklearn 허용 판정 규칙 변경(`SKLEARN_IMPORTS` 상수) | 같은 파일의 앞 모델 테스트(`tests\common`에서 통과) |

- 체크포인트가 약 193MB라 `outputs/`의 디스크 사용은 RD보다 작다(커밋하지 않는다).
