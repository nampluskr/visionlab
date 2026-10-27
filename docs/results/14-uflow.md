# Phase 14. UFlow

모델 폴더 이름은 `uflow`이고 원본은 anomalib 2.0.0의 `anomalib\models\image\uflow\`(`C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\Lib\site-packages\anomalib\models\image\uflow\`)이다. 범위와 완료 조건은 `specs/PLAN.md`의 Phase 14와 "Phase 공통 양식"을 따른다. 판정 기준은 category별 Image·Pixel AUROC 차이 0.01 이내이며, 이번 Phase는 레퍼런스와 native 모두 입력 256x256으로 통일한 조건에서 비교한다.

## 상태

다음 표는 단계별 진행 상태다. 수정 반복 한도 3회는 단계별로 센다.

| 단계 | 내용 | 상태 |
|---|---|---|
| 사전 작업 | 레퍼런스 yaml에 256 입력의 `pre_processor` 지정 | 완료 |
| 레퍼런스 | `anomalib_run.py train` x 2 (mvtec/bottle 1회, visa/candle 1회, 10 epoch) | 완료 |
| 14-A | 순수 모델 (`models/uflow/{model,feature_extraction,anomaly_map,loss}.py`, `tests/14_uflow`의 순수 부분) | 완료 (수정 요청 0회) |
| 14-B | Lightning (`UflowLitModel`, Adam과 `LinearLR` 학습, configs, export·Predictor 등록, 비교 스크립트) | 완료 (수정 요청 0회) |
| 판정 | Image·Pixel AUROC 차이 0.01 이내 (두 데이터) | 통과 |
| 사후 실행 | 사용자 승인 사후 실행 (`tests\12_fastflow\test_fastflow.py`, `tests\13_cflow\test_cflow.py`, `tests\common`) | 완료 (모두 통과) |
| specs 정리 | D-27 신설, D-15 트리에 uflow 추가 (사용자 결정) | 완료 |
| Phase 종료 | 사용자 결정과 커밋·푸시 | 사용자 결정, 커밋·푸시 |

`components/flow/all_in_one_block.py`(Phase 12), `TimmFeatureExtractor`, `PreProcessor`(ImageNet 정규화), `PostProcessor`, `Evaluator`, `load_teacher_weights`는 앞 Phase의 것을 재사용한다.

## 사용자 결정 (2026-10-27)

다음은 Phase 시작 전에 사용자가 정한 항목이다. 사용자가 계획 보고의 결정 항목을 분석한 뒤 1번은 C안, 나머지는 제안대로 채택했다.

| 항목 | 결정 |
|---|---|
| 입력 크기 (C안) | 원본은 `Resize((448, 448))`로 입력을 448로 고정하지만 visionlab의 `BaseDataset`은 이미지와 mask를 256으로 고정하고 `PreProcessor`에 resize가 없다(D-6: 공간 변환은 Dataset 몫). 데이터 생성자에 `image_size`를 더하면 D-10이 배제한 대안을 되살리고 `compare_data`의 생성자 시그니처 검사(FR-17)와 공유 코드, 데이터 테스트, specs를 바꿔야 한다. 그래서 레퍼런스와 native 모두 256 입력으로 통일한다. 레퍼런스는 모델 yaml의 `pre_processor`에 `Resize(256) + ImageNet Normalize`를 지정하고(어댑터 없음), native는 `UflowLitModel`이 `(256, 256)` 상수를 쓴다(FastFlow 선례). 원본 알고리즘은 바꾸지 않는다(`UflowModel.input_size`는 생성자 인자). 원본이 의도한 448에서의 성능은 확인하지 않는다 |
| 학습 에폭 | 원본 `trainer_arguments`는 `{"num_sanity_val_steps": 0}`뿐이고 `max_epochs`를 고정하지 않으므로 D-22의 첫 문장대로 공유 10 epoch를 레퍼런스와 native에 똑같이 쓴다. 예외는 없다 |
| `anomaly_map.py`의 외부 import | `scipy.stats`, `scipy.integrate`, `mpmath`, `numpy`는 원본대로 최상단에 둔다(`mpmath`는 torch의 sympy와 함께, `scipy`는 FrEIA 요구로 이미 배포에 필요). `omegaconf.ListConfig`만 제거한다(FastFlow 선례) |
| 히스토리 비교 | 원본이 `loss`를 `on_step`으로만 기록하고 `train_loss_epoch`가 없다. native도 원본처럼 `self.log_dict({"loss": loss}, on_step=True, on_epoch=False)`로 기록하고 `compare_history.py --tags loss`의 step 값을 비교한다. 허용 오차 없이 요약만 기록한다 |
| 백본 | 기본값 `"mcait"`는 원본과 같게 둔다(D-5). `mcait`(`cait_m48_448`, `cait_s24_224`)는 로컬 가중치가 없어 코드만 복사하고 실행·검증하지 않는다(생성 시 `FileNotFoundError`, NFR-3). 실행은 레퍼런스 yaml의 `resnet18`이고 `wide_resnet50_2`는 구조 생성만 확인한다 |
| 앞 모델 회귀(01~13) | 보류. 돌리지 않는다. 영향 표는 6.2절에 남긴다. 최종 보고에서 사후에 `tests\common`만 1회 실행할지 묻는다 |
| 평가 category | mvtec/bottle과 visa/candle이다. 시간이 이전 Phase 합계 추정의 1.5배(약 45분)를 넘을 것으로 보이면 계속하기 전에 보고하고 스스로 바꾸지 않는다 |
| D-26 범위 | 이미 UFlow의 `GraphINN`, `InputNode`, `Node`, `OutputNode`, `fm.Split`을 사용처로 적었다. `fm.IRevNetUpsampling`, `fm.Concat`은 같은 FrEIA의 부품이라 새 결정을 만들지 않는다 |
| codex 세션 | 새 세션으로 시작한다. 지시문은 자체 완결이다 |
| `Predictor` 등록 | `predictor.py`의 fastflow·cflow 분기는 건드리지 않고 `uflow` 분기를 따로 지연 import로 추가한다(D-26) |
| FR-5 허용 오차 | `atol=1e-6, rtol=1e-5` (Phase 2~13과 같음) |
| 난수가 어긋날 때의 판정 | 난수 상태를 주입하지 않고 AUROC 0.01로만 판정한다. 초기 가중치 비교는 기록한다 |
| `specs/` 변경 | 없다. 필요한 항목(입력 256 통일)은 6.1절 "specs 갱신 필요"에 초안으로 적고 최종 보고에서 묻는다 |
| 실행 범위 | UFlow의 명령만 실행한다 |
| `test_dependency.py` 전체 실행 | 필요하면 사후 승인 항목으로 올린다 |
| 판정 | Image·Pixel AUROC를 판정한다. 정규화 점수 기준 값(D-24)과 raw 값을 함께 기록한다 |
| 최종 보고 뒤 결정 (2026-10-27) | 결정 1~3을 제안대로 채택했다. (1) specs 초안을 반영한다(D-27 신설, D-15 트리). FR-4는 기존 문구가 UFlow에 맞아 고치지 않는다. (2) 사후 실행으로 `tests\12_fastflow\test_fastflow.py`, `tests\13_cflow\test_cflow.py`, `tests\common`을 돌린다(`test_dependency.py` 전체 실행도 사후 승인). (3) Phase 종료를 결정했고 커밋·푸시한다 |

## 1. 모델 노트

### 1.1 원본과 생성자

다음 표는 원본 파일과 이식 위치의 대응이다(D-15, D-18).

| 원본 | 이식 위치 | 비고 |
|---|---|---|
| `models/image/uflow/torch_model.py` (`UflowModel`, `AffineCouplingSubnet`) | `src/visionlab/anomaly_detection/models/uflow/model.py` | import 변경. FrEIA는 원본대로 import |
| `models/image/uflow/feature_extraction.py` (`get_feature_extractor`, `LayerNormFeatureExtractor`, `CaitFeatureExtractor`) | `.../models/uflow/feature_extraction.py` | `pre_trained=True` 고정을 `False`로(D-19) |
| `models/image/uflow/anomaly_map.py` (`AnomalyMapGenerator`) | `.../models/uflow/anomaly_map.py` | `omegaconf.ListConfig` 제거 |
| `models/image/uflow/loss.py` (`UFlowLoss`) | `.../models/uflow/loss.py` | 변경 없음 |
| `models/components/flow/all_in_one_block.py` | `.../models/components/flow/all_in_one_block.py` | Phase 12 것을 그대로 재사용 |
| `models/image/uflow/lightning_model.py` (`Uflow`) | `.../models/uflow/lightning_model.py` (`UflowLitModel`) | native `LightningModule`로 새로 작성 |
| `models/image/uflow/__init__.py` | `.../models/uflow/__init__.py` | `lightning_model`을 import하지 않도록 고침(D-18 예외) |

생성자는 다음과 같다. `UflowLitModel`은 `Uflow`와 인자 이름·순서·기본값이 같아야 한다(FR-6, D-5).

| 클래스 | 시그니처 |
|---|---|
| `Uflow` | `backbone="mcait"`, `flow_steps=4`, `affine_clamp=2.0`, `affine_subnet_channels_ratio=1.0`, `permute_soft=False`, `pre_processor=True`, `post_processor=True`, `evaluator=True`, `visualizer=True` (9개) |
| `UflowModel` | `input_size=(448, 448)`, `flow_steps=4`, `backbone="mcait"`, `affine_clamp=2.0`, `affine_subnet_channels_ratio=1.0`, `permute_soft=False` (6개, 인자 순서가 `Uflow`와 다르다) |
| `AffineCouplingSubnet` | `kernel_size`, `subnet_channels_ratio` |
| `AnomalyMapGenerator` | `input_size` |
| `UFlowLoss` | 인자 없음 |

`input_size`는 `Uflow`의 생성자 인자가 아니다. anomalib에서는 pre-processor의 Resize 크기에서 얻는다. 이번 Phase는 `UflowLitModel`이 `(256, 256)` 상수를 쓴다.

### 1.2 학습과 평가 동작

원본의 동작은 다음과 같다.

- **특징 추출:** `resnet18`·`wide_resnet50_2`는 `LayerNormFeatureExtractor`(`TimmFeatureExtractor` 상속, layer1~3, 특징마다 `LayerNorm`)이고 `mcait`는 `CaitFeatureExtractor`(timm `cait_m48_448`과 `cait_s24_224`)다. 파라미터는 모두 고정이다.
- **flow(`build_flow`):** 특징 스케일마다 `InputNode`를 만들고 가장 거친 스케일부터 flow 단계(`AllInOneBlock` `flow_steps`개, 커널이 3x3과 1x1을 번갈아)를 거쳐 `fm.Split`으로 나눈다. 한쪽은 출력이 되고 다른 쪽은 `fm.IRevNetUpsampling`으로 올려 다음 스케일의 입력과 `fm.Concat`한다. 마지막에 `ff.GraphINN`으로 묶는다.
- **학습 `forward`:** 특징을 `encode`에 넣어 잠재 변수 `z`(스케일별 목록)와 log Jacobian `ljd`를 반환한다. 손실 `UFlowLoss` = `mean(sum(0.5 * sum(z_i^2)) - ljd)`.
- **평가 `forward`:** 같은 흐름에서 `AnomalyMapGenerator`가 `z`로 지도를 만들고 `pred_score`는 지도의 `amax`다.
- **optimizer:** `Adam`(lr 1e-3, weight_decay 1e-5, 모델 전체 파라미터)과 `LinearLR`(start 1.0, end 0.4, `total_iters` 25000, epoch마다 진행). 10 epoch에서는 lr이 거의 변하지 않는다.
- **로그:** `loss`를 `on_step=True, on_epoch=False`로 기록한다.
- **`trainer_arguments`:** `{"num_sanity_val_steps": 0}`이며 `max_epochs`와 `gradient_clip_val`을 고정하지 않는다(공유 10 epoch).

### 1.3 외부 의존과 로컬 자산

- **외부 의존:** timm, FrEIA(0.2, `GraphINN`, `InputNode`, `Node`, `OutputNode`, `fm.Split`, `fm.IRevNetUpsampling`, `fm.Concat`, D-26), scipy(`stats`, `integrate`)와 mpmath와 numpy(`anomaly_map.py`), scipy(`AllInOneBlock`)다. `omegaconf`는 type hint용이라 제거한다. kornia, sklearn, einops는 쓰지 않는다.
- **로컬 가중치:** `paths.toml [backbones]`의 `resnet18`(레퍼런스 yaml이 쓰는 백본), `resnet50`, `wide_resnet50_2`가 있다. `cait_m48_448`, `cait_s24_224`는 없다.
- **데이터:** MVTec `bottle`(학습 209장), VisA `candle`(학습 900장). 분할은 `from_test`, 0.4, seed 42, batch 8/8, `num_workers=0`이다.
- **학습 예산:** 공유 yaml의 `max_epochs=10`이다.

### 1.4 이식에서 의도적으로 달라지는 곳과 유지하는 곳

- **입력 256 통일(사용자 결정 C안):** 원본의 448 고정 대신 256을 쓴다. `UflowModel`의 `input_size`는 원본 인자 그대로이고 값만 다르다.
- **FrEIA(D-26):** `GraphINN`과 노드, `fm.Split`, `fm.IRevNetUpsampling`, `fm.Concat`은 원본처럼 FrEIA에서 import한다. `AllInOneBlock`은 Phase 12의 anomalib 복사본을 쓴다.
- **pretrained 주입(D-19):** 순수 모델은 특징 추출기를 `pre_trained=False`로 만든다(`LayerNormFeatureExtractor`는 원본이 `pre_trained=True`를 박아 둔 곳을 고친다). `UflowLitModel`은 `pre_trained` 인자가 없고(원본도 항상 사전학습) 항상 로컬 가중치를 주입한다.
- **원본 동작 유지(고치지 않음):** 노드 이름과 생성 순서(`state_dict` 키를 결정), `AllInOneBlock`의 `torch.randperm`, 매 호출 `feature_extractor.eval()`, `log_dict({"loss": ...})`의 `on_step`만 기록, `mp.dps = 15` 모듈 수준 설정.
- **히스토리 비교:** `train_loss_epoch`가 없어 `loss` step 값을 `compare_history.py --tags loss`로 비교한다(판정 아님).
- **FR-5 허용 오차:** `pred_score`와 `anomaly_map`은 `atol=1e-6, rtol=1e-5`다.

## 2. 레퍼런스

학습 예산은 공유 `max_epochs=10`이다(1.2절). 입력은 모델 yaml의 `pre_processor`(`Resize(256)` 뒤 ImageNet 정규화)로 256x256으로 맞췄고 어댑터는 쓰지 않았다.

### 2.1 10 epoch 레퍼런스

2026-10-27에 agy가 `anomalib_run.py train --config default.yaml --config data/<data>.yaml --config models/14_uflow.yaml`을 데이터셋마다 1회 실행했다(실행으로 검증함). Claude는 로그 파일에서 값을 직접 확인했다. 레퍼런스·native·순수 로그에서 `URLError`, `Traceback`, `out of memory`는 0건이다. 로그는 UTF-16이다. 두 명령의 승인 창은 지시한 줄과 정확히 같을 때만 한 번씩 승인했다(영구 허용 없음).

agy가 PowerShell에서 실행한 명령은 다음과 같다(Claude가 보낸 지시 원문. 파일은 고치지 않았다).

```powershell
cd D:\projects\visionlab\evaluation\anomalib
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/14_uflow.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\14-ref-mvtec-261027-1.log
& $py anomalib_run.py train --config default.yaml --config data/visa.yaml --config models/14_uflow.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\14-ref-visa-261027-1.log
```

다음 표는 10 epoch 레퍼런스(1회) 결과다.

| 데이터 / category | Image AUROC | Pixel AUROC | 참고 | run 폴더 | 로그 |
|---|---|---|---|---|---|
| mvtec / bottle | 0.99999994 | 0.96423113 | image F1 1.0000, pixel F1 0.6060 | `outputs/anomalib/Uflow/LocalMVTecAD/bottle/v0` | `outputs/logs/14-ref-mvtec-261027-1.log` |
| visa / candle | 0.94666666 | 0.97587919 | image F1 0.8644, pixel F1 0.2122 | `outputs/anomalib/Uflow/LocalVisa/candle/v0` | `outputs/logs/14-ref-visa-261027-1.log` |

적용된 학습 예산은 `max_epochs=10`(로그의 `Trainer.fit stopped: max_epochs=10 reached`), `val_split_mode=from_test`, `val_split_ratio=0.4`, `seed=42`, `batch 8/8`, `num_workers=0`, `backbone=resnet18`, `flow_steps=4`, `affine_clamp=2.0`, `affine_subnet_channels_ratio=1.0`, `permute_soft=false`이다. 체크포인트는 각 run 폴더의 `weights/lightning/model.ckpt`다(bottle 43,309,530 bytes).

### 2.2 레퍼런스와 native의 에폭 확인

| 대상 | 근거 | 값 |
|---|---|---|
| 레퍼런스 (bottle, visa) | 로그의 `Trainer.fit stopped: max_epochs=10 reached` | 10 |
| native (bottle, visa) | `run.json`의 `config.trainer.max_epochs` 10, 로그의 `max_epochs=10 reached`, `num_sanity_val_steps` 0 | 10 |

### 2.3 학습 시간

계획의 추정(약 15~25분)과 이전 Phase 합계 추정의 1.5배(약 45분)를 넘지 않아 보고 없이 진행했다. 측정한 구간은 다음과 같다(진행 바와 파일 시각).

| 구간 | 걸린 시간 |
|---|---|
| 레퍼런스 에폭당 | bottle 약 9~13초, candle 약 28~35초 |
| 레퍼런스 두 run (파일 시각 19:21~19:30) | 약 9분 |
| native 에폭당 | bottle 약 9초, candle 약 30초 |
| native 학습과 순수 추론 (19:44~19:52) | 약 8분 |
| agy 실행 묶음 전체 (19:40~19:54) | 약 14분 |
| codex 14-A, 14-B | 약 6분, 약 7분 |

### 2.4 에폭별 학습 히스토리

원본은 `loss`를 step마다만 기록하므로 `train_loss_epoch`가 없다. step 값은 5.3절에서 비교했다.

## 3. 이식 메모

### 3.1 원본과 이식 파일의 대응

다음 표는 새로 만든 파일과 원본의 대응이다. 원본과의 차이는 `difflib`로 확인했다(구현함, 정적 검증).

| 이식 파일 (`src/visionlab/anomaly_detection/models/uflow/`) | 원본 | 원본과 다른 줄 |
|---|---|---|
| `model.py` | `torch_model.py` | import 2줄(`InferenceBatch`, `AllInOneBlock`)뿐 |
| `feature_extraction.py` | `feature_extraction.py` | import 한 줄, `LayerNormFeatureExtractor`의 `pre_trained=True`를 `False`로(주석, D-19), `CaitFeatureExtractor`의 `pretrained=True` 두 곳을 `False`로(주석, D-19) |
| `anomaly_map.py` | `anomaly_map.py` | `omegaconf.ListConfig` import 제거와 type hint 두 곳을 `list`로. `scipy`, `mpmath`, `numpy` import와 `mp.dps = 15`는 원본대로 |
| `loss.py` | `loss.py` | 없음 |
| `__init__.py` | `__init__.py` | `UflowModel`만 노출(D-18 예외, Lightning import 제거) |
| `lightning_model.py` (`UflowLitModel`) | `lightning_model.py` | native `LightningModule`로 새로 작성. 9개 인자 동일, `(256, 256)` 상수 `UFLOW_INPUT_SIZE`, 항상 로컬 backbone 주입(D-19), `training_step`·`configure_optimizers`·`trainer_arguments` 원본 값, D-8의 post_processor 갱신 제한 |

`models/components/flow/all_in_one_block.py`는 Phase 12의 것을 수정 없이 재사용했다.

### 3.2 새로 쓴 코드와 앞 Phase 공유 파일 변경

- **새로 쓴 파일:** `configs/models/uflow.yaml`, `models/uflow/`(여섯 파일), `tests/14_uflow/test_uflow.py`, `tests/14_uflow/test_uflow_lightning.py`. `tests\14_uflow`는 합계 25개 테스트다(순수 16개, Lightning 9개).
- **평가 설정 변경:** `evaluation/anomalib/models/14_uflow.yaml`에 256 입력의 `pre_processor`와 영어 주석 두 줄을 더했다.
- **공유 파일 변경:** `utils/io.py`(`load_export`가 `uflow` 허용), `inference/predictor.py`(`uflow` 분기를 fastflow, cflow 분기와 따로 추가하고 분기 안에서 지연 import, `input_size`를 tuple로 바꿔 키워드로 생성), `evaluation/visionlab/predict.py`(`--model uflow`), `evaluation/anomalib/compare_outputs.py`(`--model uflow` 분기), `tests/common/test_cli.py`(uflow 테스트 추가), `tests/common/test_dependency.py`(순수 모듈 4개 분류와 Lightning 허용 경로 한 줄). `common/`와 `components/flow/`는 바꾸지 않았다. 영향 범위는 6.2절의 표에 정리했다.
- **앞 Phase 테스트 변경:** `tests/12_fastflow/test_fastflow.py`의 `test_fastflow_freia_import_locations`와 `tests/13_cflow/test_cflow.py`의 `test_cflow_source_dependency_locations_and_headers`의 허용 목록에 `uflow/model.py` 한 항목씩을 추가했다. UFlow의 `model.py`가 FrEIA를 import하는 파일이 되어 기존 목록 검사가 실패하게 되기 때문이다. 두 파일은 사용자가 승인한 사후 실행으로 확인했다(5.2절, 각각 17개와 20개 통과).
- **새 부품:** 대체 부품은 없다. 모두 원본 복사다.

### 3.3 이식 중 확인된 사항

- **입력 256:** 순수 `UflowModel`의 `input_size` 기본값은 원본 그대로 `(448, 448)`이고 `UflowLitModel`만 `(256, 256)` 상수로 만든다. 순수 모델의 `Predictor` 분기는 `model_args["input_size"]`(list)를 tuple로 바꿔 쓴다. 테스트가 이를 확인한다.
- **반환 형태:** FrEIA `GraphINN.forward`가 다중 스케일 `z`를 tuple로 돌려주고 원본 `encode`가 그대로 넘긴다. 순수 테스트도 같은 형태(tuple 3개)를 확인한다. 평가 출력은 `anomaly_map [B,1,H,W]`, `pred_score [B,1]`이다.
- **난수 소비 순서:** 같은 seed 42에서 레퍼런스 `UflowModel`과 visionlab 순수 모델의 `state_dict` 중 flow 부분은 모두 같고, 다른 75개 키는 모두 timm backbone(`feature_extractor.feature_extractor.*`)의 초기 가중치다. 원인은 따로 확인하지 않았다(순수 모델이 pretrained를 주입하지 않는 D-19와 레퍼런스 쪽 로컬 가중치 가로채기의 차이로 추정한다). INFO이며 합격 판정이 아니다.
- **`local_setup.py`:** 확장 추출기의 `timm.create_model(..., pretrained=True)` 호출을 로컬 entrypoint로 가로챈다(FR-23이 `torch.equal`로 확인).
- **`TimmFeatureExtractor`의 모드:** `LayerNormFeatureExtractor.forward`가 매 호출 `feature_extractor.eval()`을 부르므로 학습 중에도 backbone은 eval이고 순수/native/레퍼런스 forward가 결정적이다(비교 항목 `train backbone eval`).
- **`mcait` 기본값:** `cait_m48_448`과 `cait_s24_224`의 로컬 가중치가 없다. 생성하면 로컬 매핑 조회에서 `FileNotFoundError`가 난다(codex가 코드로 확인, 실행하지 않음, NFR-3).

## 4. 적대적 검증

검증은 14-A, 14-B 각각 한 회차씩 했다. 수정 요청은 14-A 0회, 14-B 0회(한도 3회 이내)다.

### 4.1 14-A (순수 모델)

| 지적 | 심각도 | 처리 |
|---|---|---|
| 원본과의 `difflib` 대조: `loss.py`는 차이 없음, `model.py`는 import 2줄, `anomaly_map.py`는 `ListConfig` 세 곳, `feature_extraction.py`는 import 한 줄과 `pre_trained`, `pretrained` 세 곳(D-19 주석)뿐. 노드 이름, 생성 순서, 연산, 기본값이 같음 | 확인 | 수정 없음 |
| codex가 코드 작성 중 물은 두 질문: (1) `LayerNormFeatureExtractor` 테스트 범위, (2) `GraphINN` 출력 형태. 각각 원본 동작 보존을 고르게 했다(내부 backbone의 freeze, eval만 검사하고 `LayerNorm`은 학습 가능, `z`는 FrEIA tuple 3개) | 낮음 | 지시 보완 |
| codex가 보고한 충돌: `tests/12_fastflow/test_fastflow.py`와 `tests/13_cflow/test_cflow.py`의 FrEIA import 허용 목록이 UFlow 추가로 실패하게 됨 | 낮음 | 두 파일에 `uflow/model.py` 한 항목씩 추가를 승인(3.2절). 사후 실행으로 확인함(5.2절) |

### 4.2 14-B (Lightning, export, 비교 스크립트)

| 지적 | 심각도 | 처리 |
|---|---|---|
| 원본 `lightning_model.py`와의 대조: 생성 순서(구성요소 뒤 `UflowModel` 키워드 생성 뒤 `UFlowLoss`), `Adam([{"params": self.parameters(), "initial_lr": 1e-3}], lr=1e-3, weight_decay=1e-5)`, `LinearLR(1.0, 0.4, 25000)`, `training_step`의 `log_dict({"loss": loss}, on_step=True, on_epoch=False, prog_bar=False, logger=True)`, `trainer_arguments`가 같음. 의도한 차이는 `(256, 256)` 상수, 생성 뒤 로컬 가중치 주입, native 처리·체크포인트·export 연결뿐. `predictor.py`는 `uflow` 분기만 추가했고 fastflow, cflow 분기는 `git diff`로 그대로임을 확인 | 확인 | 수정 없음 |
| `tests/14_uflow/test_uflow_lightning.py`는 줄 단위로 읽었다(옵티마이저·스케줄러 값, `loss`만 기록, 격리된 자식 프로세스의 export 로딩, FrEIA 지연 import). `compare_outputs.py`의 uflow 분기는 `git diff`로 함수 구조와 비교 항목만 훑었고 본문 전체를 줄 단위로 읽지 않았다(실행 결과로 판정) | 낮음 | 수정 없음 |
| `tests/14_uflow/test_uflow.py`는 모듈 docstring대로 `wide_resnet50_2`와 `mcait`의 생성·구조를 확인하지 않는다(`resnet18`만). 사용자 결정(`wide_resnet50_2` 구조 생성 확인)과 어긋남 | 낮음 | 수정 요청 없이 미검증 항목으로 기록(6.2절). 실행하지 않는 backbone이라 판정에는 영향 없음 |

남은 위험은 6.2절에 적었다.

## 5. 이식 결과

### 5.1 최종 판정 (10 epoch, Image·Pixel AUROC 차이 0.01 이내)

native는 `train.py`가 test에서 기록한 값이고 순수 PyTorch는 `export/model.pt`만 불러 `predict.py`로 평가한 값이다. 모두 실행으로 검증했다. D-24에 따라 레퍼런스가 보고하는 값과 같은 정의로 비교하며, 순수 PyTorch의 정규화 점수 AUROC도 함께 적는다.

| 데이터 / category | 지표 | 레퍼런스 | native (raw) | 순수 PyTorch (raw) | 순수 PyTorch (정규화) | 차이(native raw) | 차이(순수 정규화) | 판정 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| mvtec / bottle | Image AUROC | 0.99999994 | 1.00000000 | 1.00000000 | 0.99999994 | 6e-8 | 0 | 통과 |
| mvtec / bottle | Pixel AUROC | 0.96423113 | 0.96422953 | 0.96422958 | 0.96423107 | 1.6e-6 | 6e-8 | 통과 |
| visa / candle | Image AUROC | 0.94666666 | 0.94666666 | 0.94666666 | 0.94666666 | 0 | 0 | 통과 |
| visa / candle | Pixel AUROC | 0.97587919 | 0.97587913 | 0.97587919 | 0.97587919 | 6e-8 | 0 | 통과 |

image F1과 pixel F1도 레퍼런스, native, 순수 PyTorch가 같다(bottle image 1.0, pixel 0.60602. candle image 0.86441, pixel 0.21224).

run 폴더와 로그는 다음과 같다.

| 대상 | 위치 |
|---|---|
| 레퍼런스 | `outputs/anomalib/Uflow/LocalMVTecAD/bottle/v0`, `outputs/anomalib/Uflow/LocalVisa/candle/v0` |
| native | `outputs/visionlab/uflow/mvtec_bottle/v0`, `outputs/visionlab/uflow/visa_candle/v0` (`run.json`: `max_epochs` 10, `num_sanity_val_steps` 0, seed 42) |
| 순수 PyTorch | 같은 run의 `predict/metrics.json`, `export/model.pt`(bottle 23,938,186 bytes) |
| 로그 | `outputs/logs/14-native-{mvtec,visa}-261027-1.log`, `14-pure-{mvtec,visa}-261027-1.log` |

### 5.2 테스트와 비교 스크립트

다음 표는 agy가 실행한 묶음의 결과다(Claude가 로그 파일에서 확인). 실행 대상은 UFlow뿐이다(앞 모델 회귀는 보류).

| 대상 | 결과 | 로그 |
|---|---|---|
| `tests\14_uflow` | 25 passed (순수 16, Lightning 9, 55초) | `14-pytest-261027-1.log` |
| `tests\common\test_cli.py -k uflow` | 4 passed (27 deselected) | `14-pytest-cli-261027-1.log` |
| `tests\common\test_dependency.py` (전체 파일) | 6 passed (43초) | `14-pytest-dependency-261027-1.log` |
| `compare_outputs.py --model uflow` (bottle 체크포인트) | PASS 39, FAIL 0, INFO 1 | `14-compare-outputs-mvtec-261027-1.log` |
| `compare_outputs.py --model uflow` (candle 체크포인트) | PASS 39, FAIL 0, INFO 1 | `14-compare-outputs-visa-261027-1.log` |
| 사용자 승인 사후 실행(2026-10-27): `tests\12_fastflow\test_fastflow.py` | 17 passed (허용 목록 한 항목 수정 확인) | `14-pytest-fastflow-261027-1.log` |
| 사용자 승인 사후 실행(2026-10-27): `tests\13_cflow\test_cflow.py` | 20 passed (허용 목록 한 항목 수정 확인) | `14-pytest-cflow-261027-1.log` |
| 사용자 승인 회귀 실행(2026-10-27): `tests\common` | 51 passed (이전 47개에 uflow 4개 추가) | `14-regress-common-261027-1.log` |

`compare_outputs`의 PASS 항목은 FR-6(생성자 시그니처 9개), FR-23(`feature_extractor`의 timm 모델 `state_dict` 일치, `D:\backbones\resnet18-f37072fd.pth`), FR-5의 mvtec·visa 실제 샘플 8장(256x256, 한 batch) 평가 forward(`anomaly_map [8,1,256,256]`, `pred_score [8,1]`)와 학습 forward(`z` 세 개, `ljd`, `UFlowLoss`, 허용 오차 `atol=1e-6, rtol=1e-5`)이고 최대 차이는 모두 0.0이다. 그 밖에 학습 중 backbone의 eval 모드, FR-24와 PostProcessor 대조가 있다. INFO 1개는 시드 42 초기 `state_dict`의 일치다(3.3절). 각 실행은 하나의 체크포인트 상태로 mvtec·visa 두 데이터의 샘플을 비교한다.

`test_dependency.py`는 이번 Phase에서 전체를 한 번 실행했고, 이 전체 실행은 사용자가 사후에 승인했다(2026-10-27).

실행한 명령은 다음과 같다(Claude가 보낸 지시 원문). 첫 명령은 agy가 `cd`, `$py`, `pytest` 세 줄을 한 명령으로 합쳐 승인 창을 띄웠고, 창의 내용이 지시한 줄과 같음을 Claude가 직접 읽고 사용자의 요청에 따라 한 번만 승인했다(영구 허용 없음). 나머지 명령은 agy가 하나의 작업으로 묶어 승인 창 없이 이어 실행했다. 이 때문에 개별 명령을 승인 창에서 대조하지 못했고 agy의 기록 파일 대조는 권한 문제로 하지 못했다. 로그 파일 이름, 출력, 산출물 위치가 지시와 일치함은 확인했다. 자동 승인 스크립트는 합쳐진 명령을 알아보지 못해 이 첫 창에서 멈췄다.

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py -m pytest tests\14_uflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\14-pytest-261027-1.log
& $py -m pytest tests\common\test_cli.py -k uflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\14-pytest-cli-261027-1.log
& $py -m pytest tests\common\test_dependency.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\14-pytest-dependency-261027-1.log
& $py evaluation/visionlab/train.py train --data mvtec --model uflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\14-native-mvtec-261027-1.log
& $py evaluation/visionlab/predict.py --model uflow --data mvtec 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\14-pure-mvtec-261027-1.log
& $py evaluation/visionlab/train.py train --data visa --model uflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\14-native-visa-261027-1.log
& $py evaluation/visionlab/predict.py --model uflow --data visa 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\14-pure-visa-261027-1.log
& $py evaluation/anomalib/compare_outputs.py --model uflow --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\Uflow\LocalMVTecAD\bottle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\14-compare-outputs-mvtec-261027-1.log
& $py evaluation/anomalib/compare_outputs.py --model uflow --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\Uflow\LocalVisa\candle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\14-compare-outputs-visa-261027-1.log
& $py evaluation/anomalib/compare_history.py --reference D:\projects\visionlab\outputs\anomalib\Uflow\LocalMVTecAD\bottle\v0 --native D:\projects\visionlab\outputs\visionlab\uflow\mvtec_bottle\v0 --tags loss 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\14-history-mvtec-261027-1.log
& $py evaluation/anomalib/compare_history.py --reference D:\projects\visionlab\outputs\anomalib\Uflow\LocalVisa\candle\v0 --native D:\projects\visionlab\outputs\visionlab\uflow\visa_candle\v0 --tags loss 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\14-history-visa-261027-1.log
```

사후 실행 세 명령은 다음과 같다. agy가 앞 세 줄(`cd`, `$py`, 첫 `pytest`)을 한 명령으로 합친 승인 창을 띄웠고 Claude가 지시와 같음을 읽고 한 번만 승인했다. 나머지 두 명령은 승인 창 없이 이어 실행됐다(영구 허용 없음).

```powershell
& $py -m pytest tests\12_fastflow\test_fastflow.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\14-pytest-fastflow-261027-1.log
& $py -m pytest tests\13_cflow\test_cflow.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\14-pytest-cflow-261027-1.log
& $py -m pytest tests\common 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\14-regress-common-261027-1.log
```

agy는 파일을 고치지 않았다. 11개 종료 코드는 agy의 최종 보고가 오기 전에 로그와 산출물로 판정했다(모든 로그에 Traceback 0건, pytest 요약 줄과 metric 표 확인).

### 5.3 학습 히스토리 비교 (허용 오차 없음, 비교표만)

원본이 `loss`를 step마다만 기록해 `compare_history.py --tags loss`로 step 값을 비교했다. 기본 간격이 50 step이라 다음은 그 간격의 값이다. 모두 판정에 쓰지 않는다.

| 데이터 / category | 비교한 step 수 | 레퍼런스와 native의 차이 | 첫 값과 마지막 값 (레퍼런스) |
|---|---:|---|---|
| mvtec / bottle | 5 (step 49, 99, 149, 199, 249) | 모든 행에서 절대·상대 차이 0 | 466833.5625 에서 -88452.48438 |
| visa / candle | 22 (step 49 ~ 1099) | 모든 행에서 절대·상대 차이 0 | 448739.28125 에서 -633014.25 |

비교표(`compare_history`의 Epoch 열)는 epoch마다 마지막 step 값을 쓰며 bottle은 짝수 번호가 비어 있다. 손실이 레퍼런스와 비트 단위로 같아 학습 경로(난수 소비 순서, 데이터 순서, optimizer 상태)가 같다고 볼 수 있다. native의 `val_image_AUROC`, `val_pixel_AUROC`도 에폭별로 기록되며 레퍼런스에는 검증 기록이 없어 비교하지 않았다.

### 5.4 crop·입력 크기

이 모델은 crop이 없고 입력은 256x256이다(레퍼런스는 `Resize(256)`, native는 Dataset의 256). 특징 지도는 layer1~3의 64x64, 32x32, 16x16이다. 평가 출력은 `anomaly_map [B,1,256,256]`, `pred_score [B,1]`이다. 원본의 448 입력에서의 성능은 확인하지 않았다.

## 6. 결론

- **레퍼런스:** bottle, visa 확보(10 epoch, 원본 `Uflow`, `resnet18`, 256 입력).
- **mvtec / bottle:** Image AUROC 차이 6e-8 이하, Pixel AUROC 차이 1.6e-6 이하(native와 순수 PyTorch). 통과.
- **visa / candle:** Image AUROC 차이 0, Pixel AUROC 차이 6e-8 이하. 통과.
- **FR-5, FR-6, FR-23, FR-24:** 통과(5.2절). 평가 forward와 학습 forward(`z`, `ljd`, 손실)의 최대 차이는 0.0이다. `loss` step 값은 레퍼런스와 같다(5.3절).
- **에폭:** 레퍼런스와 native 모두 실제 10 epoch임을 `run.json`과 로그로 확인했다(2.2절).
- **테스트:** `tests\14_uflow` 25개, `test_cli.py -k uflow` 4개, `test_dependency.py` 6개가 통과했다. 커밋 전 사용자 승인 사후 실행으로 `tests\12_fastflow\test_fastflow.py` 17개, `tests\13_cflow\test_cflow.py` 20개, `tests\common` 51개도 통과했다(2026-10-27). 01~13 모델별 회귀는 사용자 결정으로 돌리지 않았다.
- **수정 요청:** 14-A 0회, 14-B 0회.
- **`specs/` 변경:** 사용자 결정(2026-10-27)에 따라 `DECISIONS.md`에 D-27(입력 크기 256 통일)을 새로 쓰고 D-15 트리에 uflow 파일을 추가했다(6.1절). `SPEC.md`, `PLAN.md`와 INTENT는 고치지 않았다.
- **한계:** 원본이 쓰는 448 입력, 기본 backbone `mcait`, `wide_resnet50_2`에서의 성능은 확인하지 않았다.
- **Phase 종료:** 사용자가 종료를 결정했다(2026-10-27). `Phase 14: uflow ...` 커밋으로 푸시한다.

### 6.1 `specs/` 갱신 항목

`specs/`는 사람이 쓴다. 최종 보고에 대한 사용자 결정(2026-10-27)에 따라 아래와 같이 처리했다. `SPEC.md`, `PLAN.md`와 INTENT는 고치지 않았다.

- **D-27 입력 256 통일 (반영 완료):** `DECISIONS.md`에 D-27 "원본이 입력 크기를 고정한 모델은 Dataset의 256x256으로 통일한다"를 새로 썼다. 선택은 레퍼런스(모델 yaml의 `pre_processor`)와 native(`UflowLitModel`의 `(256, 256)` 상수)를 같은 256 조건으로 비교하고 순수 `UflowModel`의 `input_size` 기본값과 알고리즘은 원본 그대로 둔다는 것이다. 근거는 D-6, D-10, FR-17, 학습 시간과 메모리이고, 배제한 대안은 데이터 생성자의 `image_size`, 증강 훅, 모델 안 보간이며, 남은 한계는 448, `mcait`, `wide_resnet50_2` 미확인이다.
- **D-15 트리 (반영 완료):** `tests/14_uflow/`(두 파일), `configs/models`의 `uflow.yaml`, 모델 패키지 `models/uflow/`(여섯 파일)을 추가했다.
- **FR-4 (변경 없음):** UFlow의 평가 출력 `anomaly_map [B,1,H,W]`, `pred_score [B,1]`이 기존 일반 문구와 같아 고치지 않았다.

### 6.2 남은 위험과 사용자 판단 대상

- **448 입력의 성능을 확인하지 않았다.** 원본은 448을 쓰고 이번 비교는 256에서 했다(사용자 결정 C안). 입력을 키워 학습하면 결과가 달라질 수 있다.
- **기본 backbone `mcait`는 실행하지 않았다.** 로컬 가중치가 없어 생성 시 `FileNotFoundError`가 난다(NFR-3). 코드는 복사만 했고 동작은 검증하지 않았다.
- **`wide_resnet50_2`는 구조 생성도 확인하지 않았다.** 사용자 결정은 구조 생성 확인이었으나 `test_uflow.py`가 `resnet18`만 만든다(4.2절). 필요하면 별도 지시가 필요하다.
- **10 epoch 학습, category당 1회.** 같은 seed에서 손실까지 일치한 것은 seed 42, 난수 소비 순서, 같은 GPU와 연산 때문이다. 다른 환경에서도 같다고 보장하지 않는다.
- **`LinearLR`은 epoch마다 진행해 10 epoch에서 lr이 거의 변하지 않는다.** 원본 동작이며 같다.
- **배포 환경 요건:** 순수 `UflowModel`과 uflow export 로딩에는 FrEIA 0.2와 numpy, scipy, mpmath가 필요하다(D-26). 다른 모델의 `Predictor`는 FrEIA 없이 import되도록 `uflow` 분기 안에서만 지연 import한다.
- **일부 agy 명령의 승인 대조 누락:** 5.2절에 적었다.
- **사후 실행:** 사용자가 승인한 `tests\common\test_dependency.py` 전체 실행은 6개, 허용 목록을 한 항목씩 고친 `test_fastflow.py`는 17개, `test_cflow.py`는 20개, `tests\common` 전체는 51개가 통과했다(5.2절).
- **앞 모델 회귀(01~13) 보류(사용자 결정):** 01~13 모델별 테스트는 돌리지 않았다. 변경은 기존 항목을 건드리지 않는 등록과 분기 추가이고 `predictor.py`의 fastflow, cflow 분기는 그대로다. 사용자가 승인한 사후 `tests\common` 1회 실행은 51개가 통과했다(5.2절).

| 공유 파일 | 변경 | 영향받을 수 있는 앞 모델 |
|---|---|---|
| `src/visionlab/utils/io.py` | `load_export`가 허용하는 이름에 `uflow` 추가 | 모든 모델의 export 로딩 |
| `src/visionlab/inference/predictor.py` | `from_export`에 `uflow` 분기 추가(분기 안 지연 import, 기존 분기는 그대로) | 모든 모델의 `Predictor.from_export` |
| `evaluation/visionlab/predict.py` | `--model` choices에 `uflow` 추가 | 없음(선택지 추가) |
| `evaluation/anomalib/compare_outputs.py` | `--model uflow` 분기 추가 | 없음(기존 분기 유지) |
| `tests/common/test_cli.py` | uflow 테스트만 추가 | 없음 |
| `tests/common/test_dependency.py` | 순수 모듈 4개 분류와 Lightning 허용 경로 한 줄 | 분류 규칙 전체(이번 Phase에 6 passed) |
| `tests/12_fastflow/test_fastflow.py`, `tests/13_cflow/test_cflow.py` | FrEIA import 허용 목록에 한 항목씩 | 두 파일의 해당 테스트(사후 실행 17개, 20개 통과) |
