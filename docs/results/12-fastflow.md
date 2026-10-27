# Phase 12. FastFlow

모델 폴더 이름은 `fastflow`이고 원본은 anomalib 2.0.0의 `anomalib\models\image\fastflow\`(`C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\Lib\site-packages\anomalib\models\image\fastflow\`)이다. 범위와 완료 조건은 `specs/PLAN.md`의 Phase 12와 "Phase 공통 양식"을 따른다. 판정 기준은 category별 Image·Pixel AUROC 차이 0.01 이내다.

## 상태

다음 표는 단계별 진행 상태다. 수정 반복 한도 3회는 단계별로 센다.

| 단계 | 내용 | 상태 |
|---|---|---|
| D-26 | `specs/DECISIONS.md`에 FrEIA 허용 결정 신설 (12-A 시작 전) | 완료 |
| 레퍼런스 | `anomalib_run.py train` x 2 (mvtec/bottle 1회, visa/candle 1회, 10 epoch) | 완료 |
| 12-A | 순수 모델 (`models/fastflow/{model,loss,anomaly_map}.py`, `models/components/flow/all_in_one_block.py`, `tests/12_fastflow`의 순수 부분) | 완료 |
| 12-B | Lightning (`FastflowLitModel`, Adam 학습, configs, export·Predictor 등록, 비교 스크립트) | 완료 |
| specs 정리 | D-15 트리와 FR-4 예외 반영 (12-B 뒤, 사용자 결정) | 완료 |
| 회귀 실행 | 사용자 승인 사후 회귀 실행 1회 (`tests\common`, 2026-10-27, 커밋 전) | 완료 (43개 통과) |

`TimmFeatureExtractor`는 쓰지 않는다(FastFlow는 timm 모델을 직접 만든다). `PreProcessor`(ImageNet 정규화), `PostProcessor`, `Evaluator`, `load_teacher_weights`는 앞 Phase의 것을 재사용한다.

## 사용자 결정 (2026-10-27)

다음은 Phase 시작 전에 사용자가 정한 항목이다. 사용자가 계획 보고의 결정 항목 1~8번을 모두 권고안대로 채택했다.

| 항목 | 결정 |
|---|---|
| FrEIA | FrEIA(0.2)를 import해서 쓰고 독립 구현하지 않는다. `specs/DECISIONS.md`에 D-26을 신설했다(12-A 시작 전, Claude가 초안 작성). 선택, 적용 범위(flow 계열 Phase 12~15), 추론 경로에도 필요(지연 import 불가), 배포 환경에 FrEIA·numpy·scipy 필요, INTENT와의 긴장 관계(INTENT는 고치지 않음)를 적었다 |
| `AllInOneBlock` 출처 | anomalib의 `all_in_one_block.py`를 그대로 복사하고 `InvertibleModule` 등 FrEIA 부품만 FrEIA에서 import한다. FrEIA의 `AllInOneBlock`을 직접 쓰지 않는다(anomalib 복사본은 hard permutation에 `torch.randperm`을 쓰고 FrEIA 원본은 `np.random.permutation`을 써서, 직접 쓰면 레퍼런스와 초기 가중치가 어긋난다). 복사본의 `scipy` import 등 anomalib이 아닌 외부 의존은 원본대로 두고 이 문서에 적는다 |
| PLAN·INTENT 문구 | PLAN 12행과 공통 양식 90행, INTENT 문구는 Phase 종료 시점에 고친다. 지금은 `PLAN.md`와 INTENT를 고치지 않고, 고칠 문구 초안을 6.1절에 적어 최종 보고에서 반영 여부를 묻는다. INTENT는 사용자 요청 없이는 고치지 않는다 |
| D-15 트리와 FR-4 예외 | 이번 Phase에서 함께 정리한다. 12-B가 끝나 파일 구조가 확정된 뒤 `DECISIONS.md` D-15 트리(Phase 10·11·12 신규 파일)와 `SPEC.md` FR-4 예외 문구(3차원 `anomaly_map`, SuperSimpleNet 점수)를 최소한으로 반영하고 diff를 최종 보고에 포함한다 |
| 앞 모델 회귀(01~11) | 보류(사용자 결정 2026-10-27). 돌리지 않는다. 영향 표는 6.2절에 남긴다. 최종 보고에서 사후에 `tests\common`만 1회 실행할지 묻는다 |
| 최종 보고 뒤 결정 (2026-10-27) | PLAN 12행과 공통 양식 90행은 권고대로 최소한으로 고친다(INTENT는 이번에도 고치지 않고 참고 초안만 유지). `test_dependency.py` 전체 실행은 사후 승인한다. 커밋 전에 `tests\common`만 1회 실행하는 사후 회귀를 승인한다(이번 한 번에 한해 예외, 01~11 모델별 테스트는 계속 보류). 실패가 나오면 고치지 않고 멈춰 보고한다. Phase 종료를 결정했다 |
| 평가 category | mvtec/bottle과 visa/candle이다. 시간이 계획의 1.5배(약 30분)를 넘을 것으로 보이면 계속하기 전에 보고하고 스스로 바꾸지 않는다 |
| CaiT·DeiT 백본 | 로컬 가중치(`paths.toml`)가 없어 코드만 원본대로 복사하고 실행·검증하지 않는다. 6.2절 남은 위험에 적는다 |
| 학습 에폭 | 원본이 `max_epochs`를 고정하지 않으므로(`trainer_arguments`는 `gradient_clip_val` 0과 `num_sanity_val_steps` 0뿐) D-22의 첫 문장대로 공유 yaml의 10 epoch를 레퍼런스와 native에 똑같이 쓴다. 예외는 없다 |
| 난수가 어긋날 때의 판정 | 난수 상태를 주입하지 않고 AUROC 0.01로만 판정한다. 초기 가중치 비교는 기록한다 |
| FR-5 허용 오차 | `atol=1e-6, rtol=1e-5` (Phase 2~11과 같음) |
| 에폭별 `train_loss` 히스토리 | 허용 오차 없이 비교표만 기록한다. 10행이므로 전체 표를 본문에 적는다 |
| `specs/` 변경 | `DECISIONS.md`의 D-26 신설과, 12-B 뒤 D-15 트리·FR-4 예외의 최소 반영뿐이다. PLAN과 INTENT는 고치지 않는다 |
| 실행 범위 | FastFlow의 명령만 실행한다 |
| `test_dependency.py` 전체 실행 | 필요하면 사후 승인 항목으로 올린다 |
| 판정 | Image·Pixel AUROC를 판정한다. 정규화 점수 기준 값(D-24)과 raw 값을 함께 기록한다 |
| codex 세션 | 새 세션으로 시작한다. 지시문은 자체 완결이다 |

## 1. 모델 노트

### 1.1 원본과 생성자

다음 표는 원본 파일과 이식 위치의 대응이다(D-15, D-18).

| 원본 | 이식 위치 | 비고 |
|---|---|---|
| `models/image/fastflow/torch_model.py` (`FastflowModel`, `create_fast_flow_block`, `subnet_conv_func`) | `src/visionlab/anomaly_detection/models/fastflow/model.py` | import 변경, timm 백본은 구조만 만드는 방식으로(D-19), FrEIA는 원본대로 import |
| `models/image/fastflow/loss.py` (`FastflowLoss`) | `.../models/fastflow/loss.py` | 변경 없음 |
| `models/image/fastflow/anomaly_map.py` (`AnomalyMapGenerator`) | `.../models/fastflow/anomaly_map.py` | `omegaconf.ListConfig` import 제거 |
| `models/components/flow/all_in_one_block.py` (`AllInOneBlock`) | `.../models/components/flow/{__init__,all_in_one_block}.py` | 그대로 복사. `InvertibleModule`은 FrEIA에서 import, `scipy.stats.special_ortho_group` import는 원본대로 |
| `models/image/fastflow/lightning_model.py` (`Fastflow`) | `.../models/fastflow/lightning_model.py` (`FastflowLitModel`) | native `LightningModule`로 새로 작성 |
| `models/image/fastflow/__init__.py` | `.../models/fastflow/__init__.py` | `lightning_model`을 import하지 않도록 고침(D-18 예외) |

생성자는 다음과 같다. `FastflowLitModel`은 `Fastflow`와 인자 이름·순서·기본값이 같아야 한다(FR-6, D-5).

| 클래스 | 시그니처 |
|---|---|
| `Fastflow` | `backbone="resnet18"`, `pre_trained=True`, `flow_steps=8`, `conv3x3_only=False`, `hidden_ratio=1.0`, `pre_processor=True`, `post_processor=True`, `evaluator=True`, `visualizer=True` (9개) |
| `FastflowModel` | `input_size`, `backbone`, `pre_trained=True`, `flow_steps=8`, `conv3x3_only=False`, `hidden_ratio=1.0` |
| `AnomalyMapGenerator` | `input_size` |
| `FastflowLoss` | 인자 없음 |

`input_size`는 `Fastflow`의 생성자 인자가 아니다. anomalib에서는 pre-processor의 Resize 크기에서 얻는다. visionlab의 `PreProcessor`에는 resize가 없고 Dataset이 256x256으로 맞추므로 `FastflowLitModel`이 `(256, 256)`을 고정으로 쓴다.

### 1.2 학습과 평가 동작

원본의 동작은 다음과 같다.

- **구조:** `timm.create_model(backbone, features_only=True, out_indices=[1, 2, 3])`(resnet18, wide_resnet50_2)이 3개 해상도의 특징을 내고, 각 특징에 `LayerNorm([C, H/scale, W/scale])`을 적용한 뒤 해상도마다 `SequenceINN`(FrEIA)이 flow 블록을 쌓는다. flow 블록은 `flow_steps`개의 `AllInOneBlock`이며 커널이 홀수 단계에서 1x1이고 나머지가 3x3이다(`conv3x3_only=False`). CaiT·DeiT는 단일 특징 지도를 쓴다. 백본 파라미터는 모두 `requires_grad=False`다.
- **학습 `forward`:** 백본을 매 호출 `eval()`로 두고 특징을 얻어 flow 블록에 넣고 `(hidden_variables, log_jacobians)`를 반환한다.
- **평가 `forward`:** 같은 흐름에서 `AnomalyMapGenerator`가 각 hidden variable의 `-exp(-mean(h^2, dim=1) / 2)`를 입력 크기로 bilinear 보간해 평균낸 `[N,1,H,W]` 지도를 만들고, `pred_score`는 지도의 `amax(dim=(-2,-1))`다.
- **손실:** `FastflowLoss` = 해상도별 `mean(0.5 * sum(h^2, dim=(1,2,3)) - log_jacobian)`의 합.
- **optimizer:** `Adam`(lr 1e-3, weight_decay 1e-5)이 모델 파라미터를 학습한다.
- **`trainer_arguments`:** `{"gradient_clip_val": 0, "num_sanity_val_steps": 0}`이며 `max_epochs`는 고정하지 않는다(공유 yaml의 10 epoch).

### 1.3 외부 의존과 로컬 자산

- **외부 의존:** timm, FrEIA(0.2, `SequenceINN`, `InvertibleModule`), scipy(anomalib `AllInOneBlock` 복사본의 최상단 import, `permute_soft=True`일 때만 쓰임), numpy(FrEIA 요구)다. kornia, sklearn, einops는 쓰지 않는다. anomalib 의존은 `InferenceBatch`, `AllInOneBlock`, `AnomalibModule`, `PreProcessor`, `PostProcessor`, `Evaluator`, `Visualizer`, `LearningType`이다. `omegaconf`는 `anomaly_map.py`의 type hint용 import뿐이라 제거한다.
- **로컬 가중치:** `paths.toml [backbones]`의 `resnet18`(레퍼런스 yaml이 쓰는 백본), `wide_resnet50_2`가 있다. `cait_m48_448`, `deit_base_distilled_patch16_384`는 없다. 순수 모델은 `pre_trained=False`로 구조만 만들고(D-19) `FastflowLitModel.__init__`이 `pre_trained=True`일 때 로컬 가중치를 주입한다.
- **데이터:** MVTec `bottle`(학습 209장), VisA `candle`(학습 900장). 분할은 `from_test`, 0.4, seed 42, batch 8/8, `num_workers=0`이다.
- **학습 예산:** 공유 yaml의 `max_epochs=10`이다.

### 1.4 이식에서 의도적으로 달라지는 곳과 유지하는 곳

- **FrEIA(D-26):** `SequenceINN`과 `InvertibleModule`은 원본처럼 FrEIA에서 import한다. 독립 구현하지 않는다. `AllInOneBlock`은 anomalib 복사본을 그대로 쓴다.
- **pretrained 주입(D-19):** 순수 `FastflowModel`은 timm 백본을 `pretrained=False`로 만든다. `LitModel`이 `pre_trained=True`일 때 로컬 가중치를 불러온다.
- **원본 동작 유지(고치지 않음):** 생성 순서(백본, `norms`, flow 블록, 맵 생성기)와 난수 소비 순서, `AllInOneBlock`의 `torch.randperm`, 매 호출 `feature_extractor.eval()`, `pred_score`가 `[N,1]`로 나오는 `amax`, 손실의 `torch.tensor(0.0)` 누적.
- **`train_loss` 로그의 집계:** 원본은 batch 크기를 추론하지 못해 epoch 평균이 step마다 1회씩 센 평균이다. visionlab은 `batch_size=1`을 두어 맞춘다(Phase 7, 10, 11 선례).
- **히스토리 비교:** 레퍼런스와 native의 에폭별 `train_loss`를 `compare_history.py`로 표로 기록한다(판정 아님).
- **FR-5 허용 오차:** `pred_score`와 `anomaly_map`은 `atol=1e-6, rtol=1e-5`다.

## 2. 레퍼런스

학습 예산은 공유 yaml의 10 epoch다. 원본이 `max_epochs`를 고정하지 않으므로 D-22의 첫 문장대로 레퍼런스와 native가 같은 값을 쓴다(1.2절, 사용자 결정).

### 2.1 10 epoch 레퍼런스

2026-10-27에 agy가 `anomalib_run.py train --config default.yaml --config data/<data>.yaml --config models/12_fastflow.yaml`을 데이터셋마다 1회 실행했다(실행으로 검증함). Claude는 로그 파일에서 값을 직접 확인했다. 레퍼런스·native·순수 로그에서 `network access is blocked`, `URLError`, `ConnectionError`, `Traceback`, `out of memory`는 0건이다. 로그는 UTF-16이다. 두 명령의 승인 창을 확인 스크립트가 지시한 줄과 정확히 같을 때만 한 번씩 승인했다(영구 허용 없음).

agy가 PowerShell에서 실행한 명령은 다음과 같다(Claude가 보낸 지시 원문. 파일은 고치지 않았다).

```powershell
cd D:\projects\visionlab\evaluation\anomalib
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/12_fastflow.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\12-ref-mvtec-261027-1.log
& $py anomalib_run.py train --config default.yaml --config data/visa.yaml --config models/12_fastflow.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\12-ref-visa-261027-1.log
```

다음 표는 10 epoch 레퍼런스(1회) 결과다.

| 데이터 / category | Image AUROC | Pixel AUROC | 참고 | run 폴더 | 로그 |
|---|---|---|---|---|---|
| mvtec / bottle | 0.99999994 | 0.96084613 | image F1 1.0000, pixel F1 0.5710 | `outputs/anomalib/Fastflow/LocalMVTecAD/bottle/v1` | `outputs/logs/12-ref-mvtec-261027-1.log` |
| visa / candle | 0.95916659 | 0.97614616 | image F1 0.8381, pixel F1 0.1461 | `outputs/anomalib/Fastflow/LocalVisa/candle/v0` | `outputs/logs/12-ref-visa-261027-1.log` |

적용된 학습 예산은 `max_epochs=10`, `val_split_mode=from_test`, `val_split_ratio=0.4`, `seed=42`, `batch 8/8`, `num_workers=0`, `backbone=resnet18`, `pre_trained=true`, `flow_steps=8`, `conv3x3_only=false`, `hidden_ratio=1.0`이다. 체크포인트는 각 run 폴더의 `weights/lightning/model.ckpt`이고 크기는 59,029,796 bytes다. `Fastflow/LocalMVTecAD/bottle/v0`는 2026-10-25에 만든 이전 실행이며 쓰지 않았다.

### 2.2 레퍼런스와 native의 에폭 확인

사용자 지시에 따라 레퍼런스와 native가 같은 10 epoch인지 실제 값으로 확인했다.

| 대상 | 근거 | 값 |
|---|---|---|
| 레퍼런스 (bottle, visa) | 로그의 `Trainer.fit stopped: max_epochs=10 reached`, 완료한 가장 큰 에폭 번호 9 | 10 |
| native (bottle, visa) | `run.json`의 `config.trainer.max_epochs` 10, 로그의 `Trainer.fit stopped: max_epochs=10 reached`, 완료한 가장 큰 에폭 번호 9 | 10 |

### 2.3 학습 시간 관문과 실제 학습 시간

계획 추정(레퍼런스와 native 합계 약 10~20분)의 1.5배(약 30분)를 넘지 않아 보고 없이 진행했다. 실제 시간은 다음과 같다(로그의 진행 바와 파일 시각).

| 구분 | bottle | visa |
|---|---|---|
| 레퍼런스 에폭당 | 9~11초 | 31~37초 |
| 레퍼런스 학습 전체 | 약 100초 | 약 5분 30초 |
| native 학습 전체 (파일 시각) | 약 2분 30초 | 약 6분 |

합계 약 17분이었다.

### 2.4 에폭별 학습 히스토리

레퍼런스와 native의 비교는 5.3절에 표로 기록한다. 허용 오차는 정하지 않았다.

## 3. 이식 메모

### 3.1 원본과 이식 파일의 대응

다음 표는 새로 만든 파일과 원본의 대응이다. 원본과의 차이는 `difflib`로 확인했다(구현함, 정적 검증).

| 이식 파일 | 원본 | 원본과 다른 줄 |
|---|---|---|
| `models/components/flow/all_in_one_block.py` | `models/components/flow/all_in_one_block.py` | 없음(줄 단위로 같음). FrEIA의 `InvertibleModule`, `scipy.stats.special_ortho_group` import는 원본대로 |
| `models/components/flow/__init__.py` | `__init__.py` | docstring 예제의 import 한 줄 |
| `models/fastflow/model.py` | `torch_model.py` | import 2줄(`InferenceBatch`, `AllInOneBlock`), 두 `timm.create_model` 호출의 `pretrained=pre_trained`를 `pretrained=False`로(D-19 주석). FrEIA와 timm import는 원본대로 |
| `models/fastflow/loss.py` | `loss.py` | 없음 |
| `models/fastflow/anomaly_map.py` | `anomaly_map.py` | `omegaconf.ListConfig` import 제거와 type hint·docstring의 `list` 대체. `tuple` 변환 식은 그대로 |
| `models/fastflow/__init__.py` | `__init__.py` | `FastflowModel`만 노출(D-18 예외) |
| `models/fastflow/lightning_model.py` (`FastflowLitModel`) | `lightning_model.py` | native `LightningModule`로 새로 작성. 9개 인자 동일, 생성 순서 유지, `input_size`는 상수 `(256, 256)`, `Adam(lr 1e-3, wd 1e-5)`, `train_loss` 로그(`batch_size=1`), 로컬 backbone 주입(D-19), D-8의 post_processor 갱신 제한, `trainer_arguments`에 `max_epochs` 없음 |

### 3.2 새로 쓴 코드와 앞 Phase 공유 파일 변경

- **새로 쓴 파일:** `configs/models/fastflow.yaml`, `models/components/flow/`, `models/fastflow/`, `tests/12_fastflow/test_fastflow.py`, `tests/12_fastflow/test_fastflow_lightning.py`. `tests\12_fastflow`는 합계 25개 테스트다(순수 17개, Lightning 8개).
- **공유 파일 변경:** `utils/io.py`(`load_export`가 `fastflow` 허용), `inference/predictor.py`(`fastflow` 분기 안에서만 `FastflowModel`을 지연 import), `evaluation/visionlab/predict.py`(`--model fastflow`), `evaluation/anomalib/compare_outputs.py`(`--model fastflow` 분기), `tests/common/test_cli.py`, `tests/common/test_dependency.py`. `common/` 아래 파일은 바꾸지 않았다. 영향 범위는 6.2절의 표에 정리했다.
- **새 부품:** `AllInOneBlock`을 포함해 모두 원본 복사다. 독립 구현한 부품은 없다.
- **레퍼런스 어댑터 변경:** 없다. 원본 `Fastflow`를 그대로 쓴다.

### 3.3 이식 중 확인된 사항

- **FrEIA 사용(D-26):** `SequenceINN`과 `InvertibleModule`은 FrEIA(0.2)에서 import한다. `AllInOneBlock`은 anomalib 복사본을 그대로 쓴다. 사용자 결정으로 `specs/DECISIONS.md`에 D-26을 12-A 시작 전에 신설했다. 복사본의 `from scipy.stats import special_ortho_group` 최상단 import는 원본대로 두었다(`permute_soft=True`일 때만 쓰이며 FastFlow는 `False`다). FrEIA도 scipy와 numpy를 요구하므로 새로운 의존은 아니다.
- **FrEIA는 추론 경로에도 필요하다:** `fast_flow_blocks`가 FrEIA의 `SequenceINN`이고 평가 `forward`가 이를 호출한다. 순수 모델 테스트가 Lightning과 anomalib을 막은 격리 프로세스에서 순수 `FastflowModel`을 만들고 평가 forward를 실행하는 것(FrEIA 허용)과, FrEIA를 막으면 `model.py` import가 `ModuleNotFoundError`를 내는 것을 확인한다.
- **`Predictor`의 FrEIA 지연 import:** 지시문에 없던 설계를 12-B 지시에 넣었다. `predictor.py`가 `FastflowModel`을 모듈 최상단에서 import하면 FrEIA가 없는 배포 환경에서 다른 모델의 `Predictor`까지 import에 실패한다. 그래서 `model_name == "fastflow"`일 때만 `from_export` 안에서 import한다. Lightning 테스트가 FrEIA를 막은 자식 프로세스에서 `predictor`의 import가 성공하고 fastflow export의 `from_export`만 `ModuleNotFoundError`를 내는 것을 확인한다.
- **초기화 난수:** 같은 seed 42로 새로 만든 레퍼런스와 visionlab 순수 모델의 `state_dict` 전체(flow 블록의 permutation 버퍼, 가중치, `LayerNorm`, timm 백본의 초기 가중치 포함)가 서로 같다(`INFO`). anomalib 복사본을 그대로 쓴 결과다. FrEIA 원본 `AllInOneBlock`을 직접 썼다면 hard permutation의 난수원이 달라졌을 것이다(1.4절).
- **`train_loss`의 집계:** `self.log(..., batch_size=1)`을 처음부터 두었고, native의 에폭별 손실이 레퍼런스와 상대 차이 7.2e-7 이내로 일치했다(5.3절).
- **`input_size`:** 원본은 pre-processor의 Resize에서 얻는다. visionlab은 `PreProcessor`에 resize가 없고 Dataset이 256x256을 주므로 `FASTFLOW_INPUT_SIZE = (256, 256)` 상수를 쓴다. 생성자 인자 9개는 원본과 같다. export의 `model_args`에 `input_size`가 `[256, 256]`으로 들어간다.
- **timm 특징 크기:** 설치된 timm의 resnet18 `out_indices=[1, 2, 3]`은 reduction `[4, 8, 16]`이다. 12-A 지시문에 적은 수치(8, 16, 32)가 틀렸고 codex가 원본에 맞춰 테스트를 고쳤다.
- **가중치 주입 헬퍼:** `load_teacher_weights(self.model, backbone)`가 `self.model.feature_extractor`(timm `FeatureListNet`)에 그대로 맞았다(DFM, FRE 선례). 가중치 파일의 `fc.*` 키는 대상에 없는 최상위 이름이라 허용된다.

## 4. 적대적 검증

검증은 12-A, 12-B 각각 한 회차씩 했다. 수정 요청은 12-A 0회, 12-B 0회(한도 3회 이내)다.

### 4.1 12-A (순수 모델, flow 블록)

| 지적 | 심각도 | 처리 |
|---|---|---|
| 원본과의 `difflib` 대조: `all_in_one_block.py`는 줄 단위로 같음, `loss.py`는 차이 없음, `model.py`는 import 2줄과 `pretrained=False` 두 곳, `anomaly_map.py`는 `ListConfig` 제거뿐. 생성 순서, 난수 소비, `forward`가 같음 | 확인 | 수정 없음 |
| 지시문의 특징 크기 수치가 틀렸음(검토자 오류). codex가 원본 알고리즘을 유지하고 테스트를 `feature_info`에 맞춤 | 낮음 | 수정 없음 |

### 4.2 12-B (Lightning, export, 비교 스크립트)

| 지적 | 심각도 | 처리 |
|---|---|---|
| 원본 `lightning_model.py`와의 대조: 생성 순서, `Adam(params=self.model.parameters(), lr=0.001, weight_decay=0.00001)`, `training_step`, `trainer_arguments`가 같음. 의도한 차이는 `input_size` 상수와 로컬 가중치 주입뿐. 공유 파일 변경은 등록과 허용 목록 추가, `predictor.py`의 분기 안 지연 import | 확인 | 수정 없음 |
| `compare_outputs`의 fastflow 분기는 `git diff`로 전체를 읽었다. `tests/12_fastflow/test_fastflow_lightning.py`는 테스트 구조만 확인했고 본문은 줄 단위로 읽지 않았다(실행 결과로 판정) | 낮음 | 수정 없음 |

남은 위험은 6.2절에 적었다.

## 5. 이식 결과

### 5.1 최종 판정 (10 epoch, Image·Pixel AUROC 차이 0.01 이내)

native는 `train.py`가 test에서 기록한 값이고 순수 PyTorch는 `export/model.pt`만 불러 `predict.py`로 평가한 값이다. 모두 실행으로 검증했다. D-24에 따라 레퍼런스가 보고하는 값과 같은 정의로 비교하며, 순수 PyTorch의 정규화 점수 AUROC도 함께 적는다.

| 데이터 / category | 지표 | 레퍼런스 | native (raw) | 순수 PyTorch (raw) | 순수 PyTorch (정규화) | 차이(native raw) | 차이(순수 정규화) | 판정 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| mvtec / bottle | Image AUROC | 0.99999994 | 0.99999994 | 0.99999994 | 0.99999994 | 0 | 0 | 통과 |
| mvtec / bottle | Pixel AUROC | 0.96084613 | 0.96084607 | 0.96084607 | 0.96084613 | 6e-8 | 0 | 통과 |
| visa / candle | Image AUROC | 0.95916659 | 0.95916665 | 0.95916665 | 0.95916659 | 6e-8 | 0 | 통과 |
| visa / candle | Pixel AUROC | 0.97614616 | 0.97792017 | 0.97792023 | 0.97614622 | 1.8e-3 | 6e-8 | 통과 |

visa Pixel AUROC에서 raw 값(0.97792)과 레퍼런스가 보고하는 정규화 값(0.97615)이 1.8e-3 다르다. D-24의 두 정의 차이이고 0.01 이내여서 원인 기록 조건에는 해당하지 않는다. 같은 정의(정규화 점수)로 비교하면 차이는 6e-8이다. image F1과 pixel F1도 레퍼런스, native, 순수 PyTorch가 같다(bottle image 1.0000, pixel 0.57099. visa image 0.83810, pixel 0.14615).

run 폴더와 로그는 다음과 같다.

| 대상 | 위치 |
|---|---|
| 레퍼런스 | `outputs/anomalib/Fastflow/LocalMVTecAD/bottle/v1`, `outputs/anomalib/Fastflow/LocalVisa/candle/v0` |
| native | `outputs/visionlab/fastflow/mvtec_bottle/v0`, `outputs/visionlab/fastflow/visa_candle/v0` (`run.json`: `max_epochs` 10, `num_sanity_val_steps` 0, `gradient_clip_val` 0.0, seed 42, 가중치 `D:\backbones\resnet18-f37072fd.pth` sha256 `f37072fd...07ec`, 학습 이미지 bottle 209장, candle 900장) |
| 순수 PyTorch | 같은 run의 `predict/metrics.json`, `export/model.pt`(30,773,522 bytes) |
| 로그 | `outputs/logs/12-native-{mvtec,visa}-261027-1.log`, `12-pure-{mvtec,visa}-261027-1.log` |

### 5.2 테스트와 비교 스크립트

다음 표는 agy가 실행한 묶음의 결과다(Claude가 로그 파일에서 확인). 실행 대상은 FastFlow뿐이다(앞 모델 회귀는 보류).

| 대상 | 결과 | 로그 |
|---|---|---|
| `tests\12_fastflow\test_fastflow.py` (12-A 직후 단독 실행) | 17 passed | `12-pytest-pure-261027-1.log` |
| `tests\12_fastflow` (12-B 뒤 전체) | 25 passed (순수 17, Lightning 8) | `12-pytest-261027-1.log` |
| `tests\common\test_cli.py -k fastflow` | 4 passed (19 deselected) | `12-pytest-cli-261027-1.log` |
| `tests\common\test_dependency.py` (전체 파일) | 6 passed | `12-pytest-dependency-261027-1.log` |
| `compare_outputs.py --model fastflow` (bottle 체크포인트) | PASS 41, FAIL 0, INFO 1 | `12-compare-outputs-mvtec-261027-1.log` |
| `compare_outputs.py --model fastflow` (candle 체크포인트) | PASS 41, FAIL 0, INFO 1 | `12-compare-outputs-visa-261027-1.log` |
| 사용자 승인 회귀 실행(2026-10-27): `tests\common` | 43 passed (이전 Phase 37개에 draem 2개, fastflow 4개 추가) | `12-regress-common-261027-1.log` |

`compare_outputs`의 PASS 항목은 FR-6(생성자 시그니처), FR-23(특징 추출기 `torch.equal`), FR-5의 mvtec·visa 실제 샘플 8장(256x256) 평가 forward(`pred_score [8,1]`, `anomaly_map [8,1,256,256]`)와 학습 forward(hidden variable 3개, Jacobian 3개)와 `FastflowLoss` 비교(허용 오차 `atol=1e-6, rtol=1e-5`), FR-24와 PostProcessor 대조다. INFO 1개는 시드 42 초기 `state_dict` 전체의 일치다. 각 실행은 하나의 체크포인트 상태로 mvtec·visa 두 데이터의 샘플을 비교한다.

`test_dependency.py`는 모든 순수 모듈의 import 분리 검사를 한 파일이 하므로 이 모델의 모듈 검증을 위해 전체를 돌렸다. 이번 Phase에서의 실행은 사용자가 사후에 승인했다(2026-10-27).

실행한 명령은 다음과 같다(Claude가 보낸 지시 원문). agy가 명령을 한 줄씩 나누어 승인을 요청한 경우 확인 스크립트가 승인 창의 명령이 지시한 줄과 정확히 같을 때만 한 번씩 승인했다(영구 허용 없음). 앞쪽 pytest 세 명령은 승인 요청 없이 실행되어 승인 창에서 확인하지 못했다. 이 세 명령의 스크립트 전문은 대조하지 못했고, 로그 파일의 pytest 머리글(수집 대상과 개수)이 지시한 명령과 일치함은 확인했다.

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py -m pytest tests\12_fastflow\test_fastflow.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\12-pytest-pure-261027-1.log
& $py -m pytest tests\12_fastflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\12-pytest-261027-1.log
& $py -m pytest tests\common\test_cli.py -k fastflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\12-pytest-cli-261027-1.log
& $py -m pytest tests\common\test_dependency.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\12-pytest-dependency-261027-1.log
& $py evaluation/visionlab/train.py train --data mvtec --model fastflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\12-native-mvtec-261027-1.log
& $py evaluation/visionlab/predict.py --model fastflow --data mvtec 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\12-pure-mvtec-261027-1.log
& $py evaluation/visionlab/train.py train --data visa --model fastflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\12-native-visa-261027-1.log
& $py evaluation/visionlab/predict.py --model fastflow --data visa 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\12-pure-visa-261027-1.log
& $py evaluation/anomalib/compare_outputs.py --model fastflow --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\Fastflow\LocalMVTecAD\bottle\v1\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\12-compare-outputs-mvtec-261027-1.log
& $py evaluation/anomalib/compare_outputs.py --model fastflow --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\Fastflow\LocalVisa\candle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\12-compare-outputs-visa-261027-1.log
& $py evaluation/anomalib/compare_history.py --reference D:\projects\visionlab\outputs\anomalib\Fastflow\LocalMVTecAD\bottle\v1 --native D:\projects\visionlab\outputs\visionlab\fastflow\mvtec_bottle\v0 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\12-history-mvtec-261027-1.log
& $py evaluation/anomalib/compare_history.py --reference D:\projects\visionlab\outputs\anomalib\Fastflow\LocalVisa\candle\v0 --native D:\projects\visionlab\outputs\visionlab\fastflow\visa_candle\v0 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\12-history-visa-261027-1.log
& $py -m pytest tests\common 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\12-regress-common-261027-1.log
```

agy는 파일을 고치지 않았다.

### 5.3 에폭별 학습 히스토리 비교 (허용 오차 없음, 비교표만)

다음 표는 `compare_history.py`의 `train_loss_epoch` 출력이다. 10행이라 전체를 적는다. 판정에 쓰지 않는다. 손실 값이 음수이고 절대값이 크다(Jacobian 항이 크다).

mvtec / bottle (레퍼런스 `Fastflow/.../bottle/v1`, native `mvtec_bottle/v0`):

| Epoch | 레퍼런스 | native | 절대 차이 | 상대 차이 |
|---:|---:|---:|---:|---:|
| 0 | 14816.79785 | 14816.80566 | 0.0078125 | 5.272731719e-07 |
| 1 | -262332.625 | -262332.6562 | 0.03125 | 1.191235745e-07 |
| 2 | -404306.625 | -404306.625 | 0 | 0 |
| 3 | -510518.6562 | -510518.7188 | 0.0625 | 1.224245172e-07 |
| 4 | -582440.1875 | -582440.25 | 0.0625 | 1.073071559e-07 |
| 5 | -641586.375 | -641586.375 | 0 | 0 |
| 6 | -689435.875 | -689435.9375 | 0.0625 | 9.065382622e-08 |
| 7 | -738417.5 | -738417.4375 | 0.0625 | 8.464046424e-08 |
| 8 | -772934.1875 | -772934 | 0.1875 | 2.425821021e-07 |
| 9 | -801652.875 | -801652.75 | 0.125 | 1.559278385e-07 |

visa / candle (레퍼런스 `Fastflow/.../candle/v0`, native `visa_candle/v0`):

| Epoch | 레퍼런스 | native | 절대 차이 | 상대 차이 |
|---:|---:|---:|---:|---:|
| 0 | -284935.9688 | -284935.9062 | 0.0625 | 2.193475267e-07 |
| 1 | -607784.1875 | -607784.4375 | 0.25 | 4.113302141e-07 |
| 2 | -729656.0625 | -729655.9375 | 0.125 | 1.713135906e-07 |
| 3 | -809374.9375 | -809374.9375 | 0 | 0 |
| 4 | -873331.1875 | -873331.8125 | 0.625 | 7.156506134e-07 |
| 5 | -928128 | -928128.3125 | 0.3125 | 3.366992484e-07 |
| 6 | -980164.3125 | -980164.4375 | 0.125 | 1.27529638e-07 |
| 7 | -1024285.625 | -1024285.75 | 0.125 | 1.220362728e-07 |
| 8 | -1079668.5 | -1079668.375 | 0.125 | 1.157762776e-07 |
| 9 | -1130112 | -1130112.25 | 0.25 | 2.212170121e-07 |

모든 에폭의 상대 차이가 7.2e-7 이하다. float32 단위 오차 수준이어서 학습 경로(초기 가중치, 데이터 순서, 손실, optimizer)가 레퍼런스와 같다고 볼 수 있다. 에폭별 허용 오차는 사용자가 아직 정하지 않았다.

### 5.4 crop·입력 크기

이 모델은 crop이 없고 입력 크기는 256x256이다. 평가 출력 `anomaly_map`은 `[B,1,256,256]`, `pred_score`는 `[B,1]`이다. FR-5의 최대 차이는 허용 오차 이내였다(5.2절).

## 6. 결론

- **레퍼런스:** bottle, visa 확보(10 epoch, 원본 `Fastflow`).
- **mvtec / bottle:** Image AUROC 차이 0, Pixel AUROC 차이 6e-8(native와 순수 PyTorch). 통과.
- **visa / candle:** Image AUROC 차이 6e-8, Pixel AUROC는 같은 정의(정규화, D-24)로 6e-8, raw 값은 1.8e-3. 통과.
- **FR-5, FR-6, FR-23, FR-24:** 통과(5.2절). 에폭별 `train_loss_epoch`는 상대 차이 7.2e-7 이내(5.3절)이다.
- **에폭:** 레퍼런스와 native 모두 실제 10 epoch임을 `run.json`과 로그로 확인했다(2.2절).
- **테스트:** `tests\12_fastflow` 25개, `test_cli.py -k fastflow` 4개, `test_dependency.py` 6개가 통과했다. 커밋 전 사용자 승인 사후 회귀 `tests\common` 43개도 통과했다(2026-10-27). 01~11 모델별 회귀는 사용자 결정으로 돌리지 않았다.
- **수정 요청:** 12-A 0회, 12-B 0회.
- **`specs/` 변경:** 사용자 결정(2026-10-27)에 따라 `DECISIONS.md`에 D-26 신설(12-A 전)과 D-15 트리 갱신, `SPEC.md` FR-4 예외 반영, `PLAN.md` 12행과 공통 양식 90행의 최소 수정을 했다(6.1절). INTENT는 고치지 않았다.
- **Phase 종료:** 사용자가 종료를 결정했다(2026-10-27). `Phase 12: fastflow ...` 커밋으로 푸시한다.

### 6.1 `specs/` 갱신 항목

`specs/`는 사람이 쓴다. 사용자 결정(2026-10-27)에 따라 아래와 같이 처리했다.

- **D-26 (반영 완료, 12-A 시작 전):** `DECISIONS.md` 끝에 신설했다. 제목은 "FrEIA를 flow 계열 순수 모델의 외부 의존으로 허용한다"이고 D-23, D-25 형식(선택, 적용 범위, INTENT와의 관계, 근거, 배제한 대안, 남은 한계)을 따랐다. 선택은 FrEIA를 import하고 독립 구현하지 않는다(`AllInOneBlock`은 anomalib 복사본 사용), 적용 범위는 FrEIA에만 해당하며 flow 계열 Phase 12~15, 추론 경로에도 필요해 지연 import 불가, 한계는 배포 환경에 FrEIA·numpy·scipy 필요, INTENT와의 긴장 관계(INTENT는 고치지 않음)다. 원본 코드에서 확인한 FrEIA 사용처(FastFlow와 CFLOW의 `SequenceINN`, UFlow의 `GraphINN`·`InputNode`·`Node`·`OutputNode`·`Split`, CSFlow의 `GraphINN`·`InvertibleModule`)를 적었다.
- **D-15 트리 (반영 완료):** `tests/` 아래 `10_supersimplenet`, `11_draem`, `12_fastflow` 폴더와 파일, `configs/models`의 yaml 이름 세 개, `models/components/`의 `flow/all_in_one_block.py`와 `layers/sspcab.py`, 모델 패키지 `draem/`, `fastflow/`, `supersimplenet/`, `common/`의 `multi_random_choice.py`와 `perlin.py`를 추가했다. Phase 3~9 모델 파일은 이 트리에 원래 없고 이번에도 추가하지 않았다(사용자 지시는 Phase 10~12였다).
- **FR-4 예외 (반영 완료):** `SPEC.md` FR-4에 SuperSimpleNet(원본 `squeeze()` 유지, score `[B]`·batch 1이면 0차원, anomaly map `[B,1,H,W]`)과 DRAEM(`softmax(...)[:, 1]` 유지, anomaly map `[B,H,W]`, score `[B]`)을 더했고 판정 방법 문구도 "CFA, SuperSimpleNet, DRAEM은 원본 동작의 shape를 확인"으로 맞췄다.
- **PLAN 12행 (반영 완료):** 사용자 결정(2026-10-27)에 따라 "공통 flow 블록(AllInOneBlock)과 Jacobian 계산 독립 구현"을 "FrEIA(`SequenceINN`, `InvertibleModule`)를 import해 쓰고 독립 구현하지 않는다(D-26). `AllInOneBlock`은 anomalib 복사본을 공통 flow 블록으로 둔다"로 고쳤다.
- **PLAN 공통 양식 90행 (반영 완료):** "학습에 쓰던 외부 라이브러리(FrEIA, open_clip 등)가 추론 경로에 남지 않는다(NFR-2)..."를 "학습에 쓰던 외부 라이브러리(open_clip 등)가 추론 경로에 남지 않는다(NFR-2)..."로 고치고 "단 FrEIA는 flow 계열(Phase 12~15)의 모델 구조라 D-26에 따라 예외로 추론 경로에도 남는다"를 덧붙였다.
- **INTENT 문구 (미반영, 이번에도 고치지 않음):** INTENT는 사용자 요청 없이 고치지 않는다. 참고용 초안만 적는다. "모델을 배포하는 곳에는 anomalib과 Lightning이 설치되어 있지 않고, PyTorch만 사용할 수 있다"(19행)와 "배포되는 모델은 순수 PyTorch만으로 동작한다"(25행)에 "PyTorch 위의 라이브러리(kornia, FrEIA와 FrEIA가 요구하는 numpy, scipy)를 포함한다"는 취지의 괄호 문구를 더하는 안이다.

### 6.2 남은 위험과 사용자 판단 대상

- **CaiT·DeiT 백본 미검증:** `cait_m48_448`, `deit_base_distilled_patch16_384`는 로컬 가중치가 `paths.toml`에 없고 모델이 커서 코드만 원본대로 복사했고 실행·검증하지 않았다(사용자 결정). `FastflowLitModel`은 이 이름들에서 로컬 가중치 누락 `FileNotFoundError`를 낸다(NFR-3, 무작위 대체 없음).
- **배포 환경 요건:** 순수 `FastflowModel`과 fastflow export 로딩에는 FrEIA 0.2와 FrEIA가 요구하는 numpy, scipy가 필요하다(D-26). 다른 모델의 `Predictor`는 FrEIA 없이 import되도록 지연 import로 분리했다.
- **레퍼런스가 category당 1회뿐이다.** 같은 seed에서 native와 순수 PyTorch가 AUROC, 에폭별 손실, forward 출력까지 일치해 학습 경로의 편차는 작아 보인다. 이 정확한 일치는 seed 42, 난수 소비 순서, 같은 GPU와 연산 때문이다. 다른 GPU나 드라이버에서도 같은 정도라고 보장하지 않는다.
- **10 epoch 예산:** FastFlow는 원래 훨씬 오래 학습하는 모델이다. 이번 값(bottle Image 1.0, visa Image 0.959)이 원본이 의도한 학습량의 성능이라고 보장하지 않는다(공유 예산, 사용자 결정).
- **`test_fastflow_lightning.py` 본문:** 줄 단위로 읽지 않았다(실행 결과로 판정).
- **일부 agy 명령의 승인 대조 누락:** 5.2절에 적었다.
- `tests\common\test_dependency.py` 전체 실행은 사용자가 사후에 승인했다(2026-10-27). 사후 `tests\common` 1회 실행도 승인되어 43개가 통과했다(5.2절).
- **앞 모델 회귀(01~11) 보류(사용자 결정 2026-10-27):** 돌리지 않았다. 변경은 모두 기존 항목을 건드리지 않는 등록과 분기 추가이고, `predictor.py`는 `fastflow` 분기 안에서만 import한다.

| 공유 파일 | 변경 | 영향받을 수 있는 앞 모델 |
|---|---|---|
| `src/visionlab/utils/io.py` | `load_export`가 허용하는 이름에 `fastflow` 추가 | 모든 모델의 export 로딩 |
| `src/visionlab/inference/predictor.py` | `from_export`에 `fastflow` 분기(분기 안 지연 import) | 모든 모델의 `Predictor.from_export` |
| `evaluation/visionlab/predict.py` | `--model` 선택지에 추가 | 모든 모델의 순수 PyTorch 평가 |
| `evaluation/anomalib/compare_outputs.py` | `--model fastflow` 분기와 검증 함수 | 앞 모델 전부의 비교 |
| `tests/common/test_cli.py`, `tests/common/test_dependency.py` | fastflow 테스트와 분류 추가 | 같은 파일의 앞 모델 테스트 |
| `specs/DECISIONS.md`, `specs/SPEC.md` | D-26 신설, D-15 트리, FR-4 예외 | 해당 문서 (코드 영향 없음) |

- 체크포인트가 약 59MB라 `outputs/`의 디스크 사용은 크지 않다(커밋하지 않는다).
