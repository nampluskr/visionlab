# Phase 15. CSFlow

모델 폴더 이름은 `csflow`이고 원본은 anomalib 2.0.0의 `anomalib\models\image\csflow\`(`C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\Lib\site-packages\anomalib\models\image\csflow\`)이다. 범위와 완료 조건은 `specs/PLAN.md`의 Phase 15와 "Phase 공통 양식"을 따른다. 판정 기준은 category별 Image·Pixel AUROC 차이 0.01 이내다.

## 상태

다음 표는 단계별 진행 상태다. 수정 반복 한도 3회는 단계별로 센다.

| 단계 | 내용 | 상태 |
|---|---|---|
| 사전 작업 | `paths.toml [teachers]`에 EfficientNet-B5 가중치 등록 (Claude, 사용자 승인) | 완료 |
| 레퍼런스 | `anomalib_run.py train` x 2 (mvtec/bottle 1회, visa/candle 1회, 10 epoch). visa가 오염으로 퇴화해 사용자 결정(2026-10-28)으로 어댑터를 보정하고 2차 실행 | 완료 (1차 `v0` 오염, 2차 `v1` `eval()` 보정, 3차 `v2` 재보정, 판정은 3차 기준) |
| 15-A | 순수 모델 (`models/csflow/{model,anomaly_map,loss}.py`, `tests/15_csflow`의 순수 부분) | 완료 (수정 요청 0회) |
| 15-B | Lightning (`CsflowLitModel`, Adam 학습, configs, export·Predictor 등록, 비교 스크립트) | 완료 (수정 요청 0회) |
| 진단·보정 | 레퍼런스 체크포인트의 BatchNorm buffer 오염 진단, `LocalCsflow` 보정 2회 (사용자 결정 2026-10-28) | 완료 |
| 판정 | Image·Pixel AUROC 차이 0.01 이내 (두 데이터, 3차 레퍼런스 기준) | 통과 |
| 사후 실행 | 사용자 승인 회귀 실행 (`tests\12_fastflow\test_fastflow.py`, `tests\13_cflow\test_cflow.py`, `tests\14_uflow\test_uflow.py`, `tests\common`, 2026-10-28) | 완료 (모두 통과) |
| specs 정리 | FR-4, D-15 트리, D-26 재확인, PLAN 공통 양식의 레퍼런스 항목 (사용자 결정) | 완료 |
| Phase 종료 | 사용자 결정과 커밋·푸시 | 사용자 결정(2026-10-28), 커밋·푸시 |

`TimmFeatureExtractor`(`nn.Module` backbone 경로), `PreProcessor`(ImageNet 정규화), `PostProcessor`, `Evaluator`, `load_pretrained_teacher`는 앞 Phase의 것을 재사용한다. 이 모델은 `AllInOneBlock`을 쓰지 않고 자체 coupling을 쓴다.

## 사용자 결정 (2026-10-27)

다음은 Phase 시작 전에 사용자가 정한 항목이다. 계획 보고의 제안 1~8을 그대로 채택했다.

| 항목 | 결정 |
|---|---|
| 로컬 가중치 등록 | `paths.toml [teachers]`에 `csflow_efficientnet_b5 = "D:/backbones/efficientnet_b5_lukemelas-1a07897c.pth"`를 Claude가 한 줄 추가한다(D-12는 timm이 아닌 teacher를 `[teachers]`에 둔다). codex는 `paths.toml`을 고치지 않는다. 가중치는 `Lightning` 모델이 생성 뒤 backbone에 `strict` 로딩으로 주입한다(D-19, EfficientAD 선례) |
| 학습 에폭 | 원본 `trainer_arguments`는 `{"gradient_clip_val": 1, "num_sanity_val_steps": 0}`이고 `max_epochs`를 고정하지 않으므로 D-22의 첫 문장대로 공유 10 epoch를 레퍼런스와 native에 똑같이 쓴다. `gradient_clip_val: 1`은 native 모델 yaml에도 둔다. 논문이 쓰는 긴 학습은 확인하지 않는다 |
| 입력 크기 | 256x256 유지. `Csflow`는 `configure_pre_processor`를 덮어쓰지 않아 anomalib 기본 `Resize(256)`과 ImageNet 정규화를 쓰므로 원본 동작과 편차가 없다. 논문의 768 입력은 확인하지 않는다. D-27은 확장하지 않는다 |
| 앞 Phase 테스트 수정 | `tests/12_fastflow`, `tests/13_cflow`, `tests/14_uflow`의 FrEIA import 허용 목록에 `csflow/model.py`를 한 항목씩만 더하는 것을 승인한다. 바뀐 줄은 결과 문서에 적고 사후 실행에 올린다 |
| `specs/` 변경 | 없다. FR-4 출력 shape 문장, D-15 트리, D-26 재확인은 6.1절에 초안으로 적고 최종 보고에서 묻는다. D-26은 `GraphINN`, `InvertibleModule`, `InputNode`, `Node`, `OutputNode`가 이미 범위라 새 결정이 필요 없다 |
| 앞 모델 회귀(01~14) | 보류. 돌리지 않는다. 영향 표는 6.2절에 남긴다. 최종 보고에서 사후에 `tests\common`과 FrEIA 허용 목록을 고친 세 파일을 실행할지 묻는다 |
| 시간 | 추정 15~30분(미측정). 45분을 넘을 것으로 보이면 계속하기 전에 보고한다 |
| codex 세션 | 새 세션으로 시작한다 |
| 평가 category | mvtec/bottle과 visa/candle이다. 스스로 바꾸지 않는다 |
| FR-5 허용 오차 | `atol=1e-6, rtol=1e-5` (Phase 2~14와 같음) |
| 난수가 어긋날 때의 판정 | 난수 상태를 주입하지 않고 AUROC 0.01로만 판정한다 |
| FR-23 비교 범위 | EfficientNet의 BatchNorm running 통계는 비교하지 않고 parameter만 비교한다(레퍼런스는 생성자의 dry-run이 가중치 주입 뒤에 buffer를 건드릴 수 있고 native는 주입이 dry-run 뒤다). 공유 `TimmFeatureExtractor`는 고치지 않는다 |
| 실행 범위 | CSFlow의 명령만 실행한다 |
| 판정 | Image·Pixel AUROC를 판정한다. 정규화 점수 기준 값(D-24)과 raw 값을 함께 기록한다 |
| 진단과 어댑터 보정 (2026-10-28) | visa 레퍼런스 기준 미달 보고에 대해 사용자가 "A로 원인 확인 후 B 진행"을 정했다. A는 두 레퍼런스 체크포인트의 BatchNorm buffer를 가중치 파일과 직접 비교하는 읽기 전용 진단이고, B는 오염이 확인되면 `LocalCsflow`가 backbone을 `eval()`로 넘기게 고쳐 레퍼런스를 다시 실행하는 것이다. "레퍼런스는 데이터셋마다 1회" 규칙의 첫 번째 예외다. 원본 anomalib의 오염 동작과 달라지는 점은 한계로 적는다 |
| 최종 보고 뒤 결정 (2026-10-28) | 재보정 결과 보고에 대해 사용자가 2~4번을 권고안대로 채택했다. (1) specs 초안을 반영한다(FR-4 CSFlow 출력 shape, D-15 트리, D-26 재확인, 레퍼런스 어댑터 보정 기록은 D 번호를 새로 만들지 않고 `PLAN.md` 공통 양식의 레퍼런스 항목에 최소한으로 적는다. INTENT는 고치지 않는다). (2) 사후 승인 실행으로 `tests\12_fastflow\test_fastflow.py`, `tests\13_cflow\test_cflow.py`, `tests\14_uflow\test_uflow.py`, `tests\common`을 각각 1회 실행한다(이번 한 번에 한해 예외이며 01~11 모델별 회귀는 계속 보류). (3) Phase 종료를 결정했고 커밋·푸시한다 |
| 재보정 (2026-10-28) | `eval()` 보정 뒤에도 visa Pixel AUROC 차이가 0.0103으로 근소 미달이어서 사용자가 (c)를 정했다. `LocalCsflow`를 다시 보정해 dry-run을 원본처럼 train 모드로 두어 난수 소비 순서를 맞추고 생성 직후 가중치를 다시 불러 BatchNorm을 복원한다. 레퍼런스를 `15-ref3-*` 로그로 다시 실행한다. 두 번째 예외이며 세 실행(1차 `v0` 오염, 2차 `v1` `eval()` 보정, 3차 `v2` 재보정)을 모두 남기고 판정은 3차 기준이다. specs 반영, 사후 승인 실행, 커밋은 (c)의 판정 뒤에 정한다 |

## 1. 모델 노트

### 1.1 원본과 생성자

다음 표는 원본 파일과 이식 위치의 대응이다(D-15, D-18).

| 원본 | 이식 위치 | 비고 |
|---|---|---|
| `models/image/csflow/torch_model.py` (`CsFlowModel`, `CrossScaleFlow`, `ParallelGlowCouplingLayer`, `ParallelPermute`, `CrossConvolutions`, `CsFlowMultiScaleFeatureExtractor`) | `src/visionlab/anomaly_detection/models/csflow/model.py` | import 변경, `efficientnet_b5(weights=None)`(D-19). FrEIA는 원본대로 import |
| `models/image/csflow/anomaly_map.py` (`AnomalyMapGenerator`, `AnomalyMapMode`) | `.../models/csflow/anomaly_map.py` | import 변경 없음 |
| `models/image/csflow/loss.py` (`CsFlowLoss`) | `.../models/csflow/loss.py` | 변경 없음 |
| `models/image/csflow/lightning_model.py` (`Csflow`) | `.../models/csflow/lightning_model.py` (`CsflowLitModel`) | native `LightningModule`로 새로 작성 |
| `models/image/csflow/__init__.py` | `.../models/csflow/__init__.py` | `lightning_model`을 import하지 않도록 고침(D-18 예외) |

생성자는 다음과 같다. `CsflowLitModel`은 `Csflow`와 인자 이름·순서·기본값이 같아야 한다(FR-6, D-5).

| 클래스 | 시그니처 |
|---|---|
| `Csflow` | `cross_conv_hidden_channels=1024`, `n_coupling_blocks=4`, `clamp=3`, `num_channels=3`, `pre_processor=True`, `post_processor=True`, `evaluator=True`, `visualizer=True` (8개) |
| `CsFlowModel` | `input_size`, `cross_conv_hidden_channels`, `n_coupling_blocks=4`, `clamp=3`, `num_channels=3` (5개, 앞 두 개는 기본값이 없고 `Csflow`와 순서가 다르다) |
| `AnomalyMapGenerator` | `input_dims`, `mode=AnomalyMapMode.ALL` |
| `CsFlowLoss` | 인자 없음 |

`input_size`는 `Csflow`의 생성자 인자가 아니다. anomalib에서는 pre-processor의 Resize 크기(기본 256)에서 얻는다. `CsflowLitModel`은 `(256, 256)` 상수를 쓴다.

### 1.2 학습과 평가 동작

원본의 동작은 다음과 같다.

- **특징 추출:** `torchvision.models.efficientnet_b5`의 `features.6.8`(304채널, stride 32)을 세 스케일(입력, 1/2, 1/4)에 적용한다. 256 입력이면 특징 지도는 8x8, 4x4, 2x2다.
- **flow:** `GraphINN`이 세 `InputNode`를 받아 `ParallelPermute`(seed는 블록 번호)와 `ParallelGlowCouplingLayer`(`CrossConvolutions`로 스케일 사이를 오가며 s, t를 만듦)를 `n_coupling_blocks`번 쌓는다. 커널은 마지막 블록만 5이고 나머지는 3이다. `CrossConvolutions`의 `gamma`가 0으로 초기화되어 학습 시작 시 flow는 항등이다.
- **학습 `forward`:** `(z_dist 목록 3개, jacobians)`를 반환한다. jacobian은 batch와 스케일 전체의 합인 스칼라다. 손실 `CsFlowLoss` = `mean(0.5 * sum(z²) - jacobians) / (flat z 길이)`.
- **평가 `forward`:** `anomaly_map [B,1,H,W]`(스케일별 `mean(z²)`를 보간해 곱함)와 `pred_score [B]`(평탄화한 `z²/2`의 평균).
- **optimizer:** `Adam`(lr 2e-4, eps 1e-4, weight_decay 1e-5, betas (0.5, 0.9), 전체 `self.parameters()`), scheduler 없음.
- **로그:** `train_loss`를 `on_epoch=True`로 기록한다. batch가 Batch 객체라 Lightning이 batch 크기를 추론하지 못해 epoch 평균은 step마다 1회씩 센 평균이다(CFLOW와 같음). native는 `batch_size=1`로 맞춘다.
- **`trainer_arguments`:** `{"gradient_clip_val": 1, "num_sanity_val_steps": 0}`이며 `max_epochs`는 고정하지 않는다(공유 10 epoch).
- **원본의 주의점:** 역방향(`rev=True`)은 오류가 있어(`t02` 등을 `r01`에서 꺼냄) 쓰이지 않는다. `Csflow.__init__`의 `feature_extractor.eval()`은 `fit`이 train 모드로 되돌린다.

### 1.3 외부 의존과 로컬 자산

- **외부 의존:** torchvision(`efficientnet_b5`, `create_feature_extractor`), FrEIA(0.2, `GraphINN`, `InputNode`, `Node`, `OutputNode`, `InvertibleModule`, D-26), numpy(`ParallelPermute`)다. kornia, sklearn, einops, scipy, omegaconf는 쓰지 않는다.
- **로컬 가중치:** `D:\backbones\efficientnet_b5_lukemelas-1a07897c.pth`(torchvision IMAGENET1K_V1, 117MB). 레퍼런스는 `local_models.LocalCsflow`가 `efficientnet_b5`를 가로채 이 파일을 쓴다. native는 `paths.toml [teachers]`의 `csflow_efficientnet_b5`로 주입한다.
- **데이터:** MVTec `bottle`(학습 209장), VisA `candle`(학습 900장). 분할은 `from_test`, 0.4, seed 42, batch 8/8, `num_workers=0`이다.
- **학습 예산:** 공유 yaml의 `max_epochs=10`이다.
- **규모:** 기본 구성의 파라미터는 약 2.75억 개(coupling 블록의 `CrossConvolutions`, 마지막 블록이 5x5)로, 모델 상태가 약 1.1GB, optimizer를 포함한 체크포인트가 3GB 이상이다.

### 1.4 이식에서 의도적으로 달라지는 곳과 유지하는 곳

- **pretrained 주입(D-19):** 순수 모델은 `efficientnet_b5(weights=None)`로 만든다. `CsflowLitModel`은 생성 뒤 `load_pretrained_teacher`로 로컬 가중치를 `strict`하게 주입한다.
- **원본 동작 유지(고치지 않음):** 노드 이름과 생성 순서(`state_dict` 키를 결정), `ParallelPermute`의 seed 기반 순열(`state_dict`에 없는 일반 리스트), 스칼라 jacobian, 역방향의 오류, `gamma` 0 초기화, train 모드에서의 EfficientNet BatchNorm 통계 갱신과 stochastic depth, `train_loss`의 `on_epoch` 집계.
- **FrEIA(D-26):** 원본처럼 import한다.
- **FR-5 허용 오차:** `pred_score`와 `anomaly_map`은 `atol=1e-6, rtol=1e-5`다.

## 2. 레퍼런스

학습 예산은 공유 `max_epochs=10`이다(1.2절). 입력은 anomalib 기본 pre-processor(`Resize(256)`과 ImageNet 정규화)이고 어댑터는 `local_models.LocalCsflow`(가중치 파일만 로컬로 바꿈)다.

### 2.1 1차 레퍼런스 (오염된 실행, 보존)

2026-10-27에 agy가 `anomalib_run.py train --config default.yaml --config data/<data>.yaml --config models/15_csflow.yaml`을 데이터셋마다 1회 실행했다(실행으로 검증함). Claude는 로그 파일에서 값을 직접 확인했다. 두 로그 모두 `Traceback`, `out of memory`, `URLError`가 0건이고 `Trainer.fit stopped: max_epochs=10 reached`다. 로그는 UTF-16이다. 두 명령의 승인 창은 지시한 줄과 정확히 같을 때만 한 번씩 승인했다(영구 허용 없음).

agy가 PowerShell에서 실행한 명령은 다음과 같다(Claude가 보낸 지시 원문. 파일은 고치지 않았다).

```powershell
cd D:\projects\visionlab\evaluation\anomalib
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/15_csflow.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\15-ref-mvtec-261027-1.log
& $py anomalib_run.py train --config default.yaml --config data/visa.yaml --config models/15_csflow.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\15-ref-visa-261027-1.log
```

다음 표는 1차 레퍼런스(1회) 결과다. visa는 퇴화 형태(Image AUROC 1.0인데 F1 0.0, Pixel AUROC 0.56)이고 2.3절의 오염 때문으로 판단했다.

| 데이터 / category | Image AUROC | Pixel AUROC | 참고 | run 폴더 | 로그 |
|---|---|---|---|---|---|
| mvtec / bottle | 0.99780703 | 0.87767863 | image F1 0.9744, pixel F1 0.3421 | `outputs/anomalib/LocalCsflow/LocalMVTecAD/bottle/v0` | `outputs/logs/15-ref-mvtec-261027-1.log` |
| visa / candle | 1.00000000 | 0.55990332 | image F1 0.0, pixel F1 0.0 | `outputs/anomalib/LocalCsflow/LocalVisa/candle/v0` | `outputs/logs/15-ref-visa-261027-1.log` |

### 2.2 학습 시간과 에폭 확인

| 대상 | 에폭당 시간(진행 바) | 근거 |
|---|---|---|
| 레퍼런스 mvtec/bottle (27 batch) | 약 20~23초 | `max_epochs=10 reached` |
| 레퍼런스 visa/candle (113 batch) | 약 76~78초 | `max_epochs=10 reached` |
| native mvtec/bottle | 약 4분 전체 | `run.json`의 `max_epochs` 10, `gradient_clip_val` 1.0, `num_sanity_val_steps` 0, seed 42 |
| native visa/candle | 약 17분 전체 | 같음 |

계획의 추정(15~30분)과 45분 관문을 넘을 것으로 보여 실행 전에 보고했고 사용자가 그대로 계속 진행하도록 결정했다. 이 모델은 파라미터가 약 2.75억 개라 UFlow(에폭당 10초, 30초)보다 느리다. 체크포인트는 3.4GB, export는 1.3GB다.

### 2.3 레퍼런스 BatchNorm buffer 오염 (진단)

visa 레퍼런스가 퇴화했고 학습 손실이 첫 epoch부터 native와 크게 달랐다(5.65 대 10.59). 원인 후보는 공유 `TimmFeatureExtractor`의 생성자 dry-run이다. `nn.Module` backbone 경로에서 train 모드의 `torch.empty` 입력으로 forward를 해 EfficientNet의 BatchNorm running 통계를 오염시키고, 이후 forward는 항상 `eval()`이라 오염이 학습 중 사라지지 않는다. 사용자 결정(2026-10-28)에 따라 `evaluation/anomalib/diagnose_csflow_buffers.py`(읽기 전용, codex가 작성)로 두 체크포인트의 backbone buffer를 가중치 파일과 직접 비교했다(로그 `15-diag-buffers-261028-1.log`).

| 항목 (1차 레퍼런스) | mvtec / bottle | visa / candle |
|---|---|---|
| buffer 총 개수 / 파일과 다른 개수 | 348 / 318 | 348 / 318 |
| `num_batches_tracked` 차이 | 106개가 +1, 10개가 0 | 106개가 +1, 10개가 0 |
| `running_var` 체크포인트/파일 비율 | 중앙값 0.99855, 최소 0.99, 최대 3.553 | 중앙값 0.99913, 최소 0.99, 유한 최대 1.351 |
| `running_mean` 상대 차이 최대 | 49.1 | 2.9e28 (`features.0.1`) |
| NaN, inf 원소 | 0, 0 | 0, `running_var` 48개가 inf |
| 절댓값 1e6 초과 buffer | 0 | `running_mean` 1개, `running_var` 1개 |

두 체크포인트 모두 106개 BatchNorm이 오염되었다. 변하지 않은 10개는 `features.6.8` 뒤쪽 층이라 dry-run이 거치지 않는다. mvtec은 통계가 몇 배 정도 변했고 visa는 첫 BatchNorm의 분산이 inf가 되어 특징이 무너져 퇴화한 것으로 설명된다. 오염 크기는 미초기화 메모리의 내용에 달려 실행마다 다르다. FX 중복 경로(`feature_extractor.feature_extractor.feature_extractor.`)는 두 체크포인트 모두 backbone 경로와 같았다. 이 오염은 anomalib 원본 코드에도 있는 동작이다(원본도 torchvision 가중치를 먼저 로드한 뒤 생성자가 돈다).

진단 명령(Claude가 보낸 지시 원문, agy는 승인 창 없이 실행했다)은 다음과 같다.

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py evaluation/anomalib/diagnose_csflow_buffers.py --checkpoint mvtec=D:\projects\visionlab\outputs\anomalib\LocalCsflow\LocalMVTecAD\bottle\v0\weights\lightning\model.ckpt --checkpoint visa=D:\projects\visionlab\outputs\anomalib\LocalCsflow\LocalVisa\candle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\15-diag-buffers-261028-1.log
```

### 2.4 2차 레퍼런스 (`eval()` 보정, 참고)

사용자 결정(2026-10-28)으로 `evaluation/anomalib/local_models.py`의 `LocalCsflow`가 backbone을 `eval()` 상태로 넘기게 고쳤다(한 줄, `return model.eval()`). codex가 설치된 코드를 읽어 확인한 근거는 `torch.fx`의 일반 `GraphModule`이 `root.training`을 복사하고 `create_feature_extractor`가 호출 당시 상태를 저장하고 복원하므로 dry-run이 eval 상태로 돌아 BatchNorm이 갱신되지 않는다는 것이다. 이는 "레퍼런스는 데이터셋마다 1회만" 규칙의 사용자 승인 첫 번째 예외다. 1차 실행의 로그와 `v0` 산출물은 지우지 않았고 2차는 별도 로그 이름과 `v1` 폴더에 두었다. native와 순수 PyTorch는 그대로 두었다. 이 보정은 오염은 없앴지만 레퍼런스의 난수 소비 순서를 바꿨다(아래).

```powershell
cd D:\projects\visionlab\evaluation\anomalib
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/15_csflow.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\15-ref2-mvtec-261028-1.log
& $py anomalib_run.py train --config default.yaml --config data/visa.yaml --config models/15_csflow.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\15-ref2-visa-261028-1.log
```

두 명령의 승인 창은 지시한 줄과 정확히 같음을 읽고 한 번씩만 승인했다. 두 로그 모두 `Traceback`, `out of memory`가 0건이고 `max_epochs=10 reached`다. 2차 실행은 mvtec이 약 7분, visa가 약 17분 걸렸다.

| 데이터 / category | Image AUROC | Pixel AUROC | 참고 | run 폴더 | 로그 |
|---|---|---|---|---|---|
| mvtec / bottle | 0.99780703 | 0.87871659 | image F1 0.9737, pixel F1 0.3448 | `outputs/anomalib/LocalCsflow/LocalMVTecAD/bottle/v1` | `outputs/logs/15-ref2-mvtec-261028-1.log` |
| visa / candle | 0.92583334 | 0.88551557 | image F1 0.8288, pixel F1 0.0259 | `outputs/anomalib/LocalCsflow/LocalVisa/candle/v1` | `outputs/logs/15-ref2-visa-261028-1.log` |

2차 체크포인트에 진단 스크립트를 다시 돌린 결과(로그 `15-diag2-buffers-261028-1.log`) 두 체크포인트 모두 buffer 348개 중 다른 개수가 0이고 `num_batches_tracked` 차이가 모두 0이며 `running_var` 비율이 1이다. 오염이 사라졌다. visa는 퇴화가 없어졌고 F1이 0이 아니다. 그러나 2차 `compare_outputs`의 INFO "seeded initial CSFlow graph state"가 같음(1차)에서 다름으로 바뀌었다. `eval()` 보정으로 레퍼런스의 dry-run이 stochastic depth 난수를 뽑지 않게 되어 flow 초기화의 난수 순서가 어긋난 것이다. 그 결과 학습 손실이 native와 가깝지만 같지는 않았고(epoch 0 상대 차이 0.0008~0.0014, 이후 최대 0.13) visa Pixel AUROC 차이가 0.01030(native raw 0.87521160 대 레퍼런스 0.88551557)으로 기준 0.01을 근소하게 넘었다. 이 2차 값은 판정에 쓰지 않는다.

| 2차 레퍼런스 `v1` | Image AUROC | Pixel AUROC | 참고 |
|---|---|---|---|
| mvtec / bottle | 0.99780703 | 0.87871659 | image F1 0.9737, pixel F1 0.3448 |
| visa / candle | 0.92583334 | 0.88551557 | image F1 0.8288, pixel F1 0.0259 |

### 2.5 3차 레퍼런스 (재보정, 판정에 쓰는 실행)

사용자 결정(2026-10-28)의 (c)에 따라 `LocalCsflow`를 다시 보정했다(두 번째 예외). 직전의 `.eval()`을 되돌려 모델을 train 모드로 넘기고(원본 anomalib이 weights를 로드한 뒤 train 모드 모델을 넘기는 것과 같음), `super().__init__()`이 끝난 뒤 같은 가중치 파일을 backbone에 `strict`로 다시 불러온다. dry-run이 train 모드로 돌아 난수 소비 순서가 원본과 같고, 재불러오기는 난수를 쓰지 않으며 BatchNorm buffer를 파일 값으로 복원한다. codex가 설치된 코드를 읽어 확인한 사실은 다음과 같다. FX 추출기는 원본 하위 모듈의 텐서를 복제하지 않고 참조하며, `load_state_dict`는 기본값(`assign=False`)에서 기존 텐서에 `copy_`하므로 parameter와 persistent buffer 모두 FX 쪽 공유 값까지 복원된다. 원본 anomalib의 오염 동작과는 달라지는 점이 한계다(오염 크기가 실행마다 달라 원본은 결정적이지 않다).

```powershell
cd D:\projects\visionlab\evaluation\anomalib
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/15_csflow.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\15-ref3-mvtec-261028-1.log
& $py anomalib_run.py train --config default.yaml --config data/visa.yaml --config models/15_csflow.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\15-ref3-visa-261028-1.log
```

이 두 명령은 승인 창 없이 실행되어 승인 창에서 대조하지 못했다. 로그 파일 이름과 출력이 지시와 일치함은 확인했다. 두 로그 모두 `Traceback`, `out of memory`가 0건이고 `max_epochs=10 reached`다. 소요는 mvtec이 약 8분 22초, visa가 약 16분 37초다(agy 보고).

| 3차 레퍼런스 `v2` | Image AUROC | Pixel AUROC | 참고 | run 폴더 | 로그 |
|---|---|---|---|---|---|
| mvtec / bottle | 0.99780697 | 0.87749320 | image F1 0.9870, pixel F1 0.3440 | `outputs/anomalib/LocalCsflow/LocalMVTecAD/bottle/v2` | `outputs/logs/15-ref3-mvtec-261028-1.log` |
| visa / candle | 0.92611110 | 0.87521750 | image F1 0.8214, pixel F1 0.0241 | `outputs/anomalib/LocalCsflow/LocalVisa/candle/v2` | `outputs/logs/15-ref3-visa-261028-1.log` |

3차 체크포인트의 확인 결과(로그 `15-diag3-buffers-261028-1.log`, `15-compare-outputs3-*`)는 다음과 같다.

| 확인 항목 | 결과 |
|---|---|
| backbone BatchNorm buffer 348개 중 파일과 다른 개수 | mvtec 0, visa 0 (`num_batches_tracked` 차이 모두 0, `running_mean`·`running_var` 상대 차이 최대 0, FX 중복 경로 불일치 0) |
| `compare_outputs` INFO "reference backbone buffers after constructor dry-run" | equal=True, 다른 buffer 0 (두 체크포인트) |
| `compare_outputs` INFO "seeded initial CSFlow graph state" | equal=True (시드 42 초기 `graph.*` 상태가 같음으로 복귀, 두 체크포인트) |

세 실행의 요약은 다음과 같다. 판정 기준은 마지막(3차) 레퍼런스다.

| 실행 | 보정 | 폴더 | 로그 | 상태 |
|---|---|---|---|---|
| 1차 | 없음(원본 동작) | `v0` | `15-ref-*-261027-1.log` | BatchNorm 오염, visa 퇴화. 참고용 보존 |
| 2차 | `eval()` | `v1` | `15-ref2-*-261028-1.log` | 오염 없음, 난수 순서가 어긋남. 참고용 보존 |
| 3차 | train 모드 dry-run 후 가중치 복원 | `v2` | `15-ref3-*-261028-1.log` | 오염 없음, 난수 순서 일치. 판정에 사용 |

## 3. 이식 메모

### 3.1 원본과 이식 파일의 대응

다음 표는 새로 만든 파일과 원본의 대응이다. 원본과의 차이는 `difflib`로 확인했다(구현함, 정적 검증).

| 이식 파일 (`src/visionlab/anomaly_detection/models/csflow/`) | 원본 | 원본과 다른 줄 |
|---|---|---|
| `model.py` | `torch_model.py` | import 3줄(`InferenceBatch`, `TimmFeatureExtractor`, `EfficientNet_B5_Weights` 제거), `efficientnet_b5(weights=None)`와 D-19 주석 |
| `anomaly_map.py` | `anomaly_map.py` | 없음 |
| `loss.py` | `loss.py` | 없음 |
| `__init__.py` | `__init__.py` | `CsFlowModel`만 노출(D-18 예외, Lightning import 제거) |
| `lightning_model.py` (`CsflowLitModel`) | `lightning_model.py` | native `LightningModule`로 새로 작성. 8개 인자 동일, `(256, 256)` 상수, `eval()` 호출과 `train_loss`의 `on_epoch` 집계(`batch_size=1`) 유지, 생성 뒤 `load_pretrained_teacher`로 로컬 가중치 주입(D-19), `trainer_arguments`와 optimizer 원본 값 |

`components/flow/all_in_one_block.py`는 쓰지 않았다(자체 coupling).

### 3.2 새로 쓴 코드와 앞 Phase 공유 파일 변경

- **새로 쓴 파일:** `configs/models/csflow.yaml`, `models/csflow/`(다섯 파일), `tests/15_csflow/test_csflow.py`, `tests/15_csflow/test_csflow_lightning.py`(합계 30개), `evaluation/anomalib/diagnose_csflow_buffers.py`(읽기 전용 진단).
- **설정 변경:** `paths.toml [teachers]`에 `csflow_efficientnet_b5` 한 줄(Claude, 사용자 승인). `evaluation/anomalib/local_models.py`의 `LocalCsflow`가 생성 직후 가중치를 backbone에 다시 불러 BatchNorm buffer를 복원(2.5절, 중간에 `eval()` 보정을 거침).
- **공유 파일 변경:** `utils/io.py`(`load_export`가 `csflow` 허용), `inference/predictor.py`(`csflow` 분기를 fastflow, cflow, uflow 분기와 따로 추가하고 분기 안에서 지연 import), `evaluation/visionlab/predict.py`(`--model csflow`), `evaluation/anomalib/compare_outputs.py`(`--model csflow` 분기), `tests/common/test_cli.py`(csflow 테스트 추가), `tests/common/test_dependency.py`(순수 모듈 3개 분류와 Lightning 허용 경로 한 줄). `common/`와 `components/`는 바꾸지 않았다. 영향 범위는 6.2절의 표에 정리했다.
- **앞 Phase 테스트 변경:** `tests/12_fastflow/test_fastflow.py`, `tests/13_cflow/test_cflow.py`, `tests/14_uflow/test_uflow.py`의 FrEIA import 허용 목록에 `csflow/model.py` 한 항목씩을 추가했다. 세 파일은 사용자가 승인한 사후 회귀 실행으로 확인했다(5.2절, 각각 17개, 20개, 16개 통과).

### 3.3 이식 중 확인된 사항

- **입력 256:** `Csflow`의 `input_size`는 pre-processor의 Resize 크기(기본 256)에서 얻으므로 `CsflowLitModel`의 `(256, 256)` 상수는 원본과 같다. 특징 지도는 8x8, 4x4, 2x2다.
- **backbone의 동결:** 공유 `TimmFeatureExtractor`는 `nn.Module` backbone의 파라미터를 `requires_grad_(False)`로 바꾸지 않는다. `requires_grad=False`는 forward에서 `eval()`과 `no_grad()` 경로를 고르는 설정이다. 그래서 backbone 파라미터의 `requires_grad`는 True로 남고 gradient는 만들어지지 않는다. 테스트는 설정값과 gradient 없음을 확인한다.
- **중복 state_dict 키:** 공유 `TimmFeatureExtractor`가 원본 모듈을 `self.backbone`으로 등록하고 FX 추출기와 파라미터를 공유하므로 `state_dict`에 `backbone.*`와 `feature_extractor.*`가 중복된다. 체크포인트와 export가 그만큼 크다.
- **원본 동작 유지:** `ParallelPermute`의 seed 기반 순열(`state_dict`에 없음), 스칼라 jacobian, 역방향의 오류(쓰이지 않아 테스트하지 않음), `gamma` 0 초기화, FrEIA 그래프의 학습 반환이 `z_dist` tuple 3개와 jacobian `[B]`인 점.
- **난수 소비 순서:** 같은 seed 42에서 1차 `compare_outputs`는 레퍼런스와 visionlab 순수 모델의 `graph.*` 초기 상태가 같다고 보고했다(INFO). 2차(`eval()` 보정)는 다르다고 보고했다. 생성자 dry-run이 train 모드이면 EfficientNet의 stochastic depth가 난수를 뽑아 이후 flow 초기화의 난수 순서가 정해지는데, `eval()` 보정으로 레퍼런스의 dry-run이 난수를 뽑지 않게 된 것이 원인으로 보인다. 3차(재보정)에서 train 모드 dry-run으로 되돌리고 가중치를 복원하자 INFO가 같음으로 돌아왔고 학습 손실도 비트에 가깝게 일치했다(5.3절). 이는 추정 원인과 일치하는 결과이다.

## 4. 적대적 검증

검증은 15-A, 15-B 각각 한 회차씩 했다. 수정 요청은 15-A 0회, 15-B 0회(한도 3회 이내)다. 별도로 사용자 결정에 따른 진단 스크립트와 `LocalCsflow` 보정을 codex에 요청했다.

### 4.1 15-A (순수 모델)

| 지적 | 심각도 | 처리 |
|---|---|---|
| 원본과의 `difflib` 대조: `anomaly_map.py`, `loss.py`는 차이 없음, `model.py`는 import 3줄과 `efficientnet_b5(weights=None)` 한 곳뿐. 노드 이름, 생성 순서, 역방향 오류까지 원본과 같음 | 확인 | 수정 없음 |
| codex의 질문 3개: (1) backbone `requires_grad` 검사 방식, (2) 학습 반환 형태, (3) BatchNorm 통계의 비결정성. 제 지시문이 backbone이 모두 `requires_grad` False라고 가정한 것이 틀렸다(공유 부품은 동결하지 않음). 각각 원본 보존(설정값과 gradient 없음, tuple 3개와 `[B]`, 수치 검사 helper에서만 통계 초기화)으로 답함 | 낮음 | 지시 보완 |
| 앞 Phase 테스트 세 개의 FrEIA 허용 목록에 `csflow/model.py` 한 항목씩 추가를 승인 | 낮음 | 사후 실행에 올림(6.2절) |

### 4.2 15-B (Lightning, export, 비교 스크립트)

| 지적 | 심각도 | 처리 |
|---|---|---|
| 원본 `lightning_model.py`와의 대조: 생성 순서, `eval()` 호출, `Adam(self.parameters(), lr=2e-4, eps=1e-4, weight_decay=1e-5, betas=(0.5, 0.9))`, `train_loss`의 `on_epoch` 로그(`batch_size=1`), `trainer_arguments`가 같음. 의도한 차이는 `(256, 256)` 상수와 생성 뒤 로컬 가중치 주입, native 처리·체크포인트·export 연결뿐. `predictor.py`는 `csflow` 분기만 추가했고 기존 분기는 `git diff`로 그대로임을 확인 | 확인 | 수정 없음 |
| `tests/15_csflow/test_csflow_lightning.py`는 테스트 이름과 구조를 확인했고 본문은 줄 단위로 읽지 않았다(실행 결과로 판정). `compare_outputs.py`의 csflow 분기도 `git diff`로 함수 구조만 훑었다 | 낮음 | 수정 없음 |

### 4.3 진단과 보정

| 지적 | 심각도 | 처리 |
|---|---|---|
| 계획 단계에서 BatchNorm 오염을 학습 중 사라질 것으로 봤으나 틀렸다. `TimmFeatureExtractor.forward`가 매 호출 `eval()`로 돌려 오염이 유지된다 | 중간 | 진단으로 확인(2.3절), 어댑터 보정(2.4절) |
| 진단 스크립트는 파일을 쓰지 않고 anomalib을 import하지 않음을 소스로 확인한 뒤 실행을 요청했다 | 확인 | 수정 없음 |
| `eval()` 보정은 오염을 없앴지만 레퍼런스의 난수 소비 순서를 바꿔 초기 flow 가중치가 native와 달라졌다. 2차 `compare_outputs`의 INFO가 이를 드러냈다 | 중간 | 재보정(2.5절). 3차에서 INFO가 같음으로 복귀 |
| 재보정 코드는 `git diff`로 읽었다(`eval()` 제거, `super().__init__()` 뒤 `load_state_dict(strict=True)`). codex가 FX 추출기와의 텐서 공유와 `load_state_dict`의 복사 동작을 코드 위치로 확인했다 | 확인 | 수정 없음 |

## 5. 이식 결과

### 5.1 최종 판정 (10 epoch, 3차 레퍼런스 `v2` 기준, Image·Pixel AUROC 차이 0.01 이내)

native는 `train.py`가 test에서 기록한 값이고 순수 PyTorch는 `export/model.pt`만 불러 `predict.py`로 평가한 값이다. 모두 실행으로 검증했다. D-24에 따라 레퍼런스가 보고하는 값과 같은 정의로 비교하며, 순수 PyTorch의 정규화 점수 AUROC도 함께 적는다.

| 데이터 / category | 지표 | 레퍼런스 (3차) | native (raw) | 순수 PyTorch (raw) | 순수 PyTorch (정규화) | 차이(native raw) | 차이(순수 정규화) | 판정 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| mvtec / bottle | Image AUROC | 0.99780697 | 0.99780703 | 0.99780703 | 0.99780697 | 6e-8 | 0 | 통과 |
| mvtec / bottle | Pixel AUROC | 0.87749320 | 0.87749261 | 0.87749267 | 0.87749332 | 5.9e-7 | 1.2e-7 | 통과 |
| visa / candle | Image AUROC | 0.92611110 | 0.92611110 | 0.92611110 | 0.92611110 | 0 | 0 | 통과 |
| visa / candle | Pixel AUROC | 0.87521750 | 0.87521160 | 0.87521124 | 0.87521732 | 5.9e-6 | 1.8e-7 | 통과 |

image F1과 pixel F1도 레퍼런스(3차), native, 순수 PyTorch가 같다(mvtec image 0.9870, pixel 0.3440. visa image 0.8214, pixel 0.0241).

다음 표는 앞의 두 레퍼런스에 대한 같은 비교로 참고용이며 판정에 쓰지 않는다. 1차는 BatchNorm 오염(2.3절), 2차는 난수 순서 어긋남(2.4절)이 있었다.

| 데이터 / category | 지표 | 1차 `v0` | 차이(native raw) | 2차 `v1` | 차이(native raw) |
|---|---|---:|---:|---:|---:|
| mvtec / bottle | Image AUROC | 0.99780703 | 0 | 0.99780703 | 0 |
| mvtec / bottle | Pixel AUROC | 0.87767863 | 1.9e-4 | 0.87871659 | 1.22e-3 |
| visa / candle | Image AUROC | 1.00000000 | 0.0739 | 0.92583334 | 2.8e-4 |
| visa / candle | Pixel AUROC | 0.55990332 | 0.3153 | 0.88551557 | 0.01030 |

run 폴더와 로그는 다음과 같다.

| 대상 | 위치 |
|---|---|
| 레퍼런스 | 1차 `outputs/anomalib/LocalCsflow/LocalMVTecAD/bottle/v0`·`LocalVisa/candle/v0`, 2차 `.../v1`, 3차(판정) `.../v2` |
| native | `outputs/visionlab/csflow/mvtec_bottle/v0`, `outputs/visionlab/csflow/visa_candle/v0` (`run.json`: `max_epochs` 10, `gradient_clip_val` 1.0, `num_sanity_val_steps` 0, seed 42, 가중치 `D:\backbones\efficientnet_b5_lukemelas-1a07897c.pth`) |
| 순수 PyTorch | 같은 run의 `predict/metrics.json`, `export/model.pt`(mvtec 1,294,458,594 bytes) |
| 로그 | `outputs/logs/15-native-{mvtec,visa}-261027-1.log`, `15-pure-{mvtec,visa}-261027-1.log` |

### 5.2 테스트와 비교 스크립트

다음 표는 agy가 실행한 묶음의 결과다(Claude가 로그 파일에서 확인). 실행 대상은 CSFlow와 사용자 승인 사후 회귀뿐이다(01~11 모델별 회귀는 보류).

| 대상 | 결과 | 로그 |
|---|---|---|
| `tests\15_csflow` | 30 passed (72초) | `15-pytest-261027-1.log` |
| `tests\common\test_cli.py -k csflow` | 4 passed (31 deselected) | `15-pytest-cli-261027-1.log` |
| `tests\common\test_dependency.py` (전체 파일) | 6 passed | `15-pytest-dependency-261027-1.log` |
| `compare_outputs.py --model csflow` (1차 bottle, candle 체크포인트) | PASS 51, FAIL 0, INFO 2 (각각) | `15-compare-outputs-{mvtec,visa}-261027-1.log` |
| `compare_outputs.py --model csflow` (2차 bottle, candle 체크포인트) | PASS 51, FAIL 0, INFO 2 (각각) | `15-compare-outputs2-{mvtec,visa}-261028-1.log` |
| `compare_outputs.py --model csflow` (3차 bottle, candle 체크포인트) | PASS 51, FAIL 0, INFO 2 (각각). INFO 두 개가 모두 equal=True | `15-compare-outputs3-{mvtec,visa}-261028-1.log` |
| 사용자 승인 회귀 실행(2026-10-28): `tests\12_fastflow\test_fastflow.py` | 17 passed (허용 목록 한 항목 수정 확인) | `15-regress-fastflow-261028-1.log` |
| 사용자 승인 회귀 실행(2026-10-28): `tests\13_cflow\test_cflow.py` | 20 passed (허용 목록 한 항목 수정 확인) | `15-regress-cflow-261028-1.log` |
| 사용자 승인 회귀 실행(2026-10-28): `tests\14_uflow\test_uflow.py` | 16 passed (허용 목록 한 항목 수정 확인) | `15-regress-uflow-261028-1.log` |
| 사용자 승인 회귀 실행(2026-10-28): `tests\common` | 55 passed (이전 51개에 csflow 4개 추가) | `15-regress-common-261028-1.log` |
| `diagnose_csflow_buffers.py` (1차, 2차, 3차 체크포인트) | 1차: 318개 다름, 2차: 0개 다름, 3차: 0개 다름 | `15-diag-buffers-261028-1.log`, `15-diag2-buffers-261028-1.log`, `15-diag3-buffers-261028-1.log` |

`compare_outputs`의 PASS 항목은 FR-6(생성자 8개), FR-23(native의 주입 가중치가 파일과 일치, 레퍼런스의 backbone parameter가 파일과 일치), FR-5의 mvtec·visa 실제 샘플 8장(256x256, 한 batch) 평가 forward(`anomaly_map [8,1,256,256]`, `pred_score [8]`)와 학습 forward(`z` 3개, jacobian `[8]`, `CsFlowLoss`, 허용 오차 `atol=1e-6, rtol=1e-5`)이고 최대 차이는 모두 0.0이다. 그 밖에 coupling 층과 `ParallelPermute`, `AnomalyMapGenerator`, `CsFlowLoss`의 구성요소 비교와 FR-24, PostProcessor 대조가 있다. INFO 2개는 "레퍼런스 backbone buffer가 생성자 dry-run 뒤 파일과 같은지"(1차 체크포인트 비교에서는 318개 다름, 2차와 3차에서는 같음)와 "시드 42 초기 `graph.*` 상태의 일치"(1차 같음, 2차 다름, 3차 같음)다.

실행한 명령은 다음과 같다(Claude가 보낸 지시 원문). agy는 첫 명령에서만 `cd`, `$py`, 첫 `pytest`를 한 명령으로 합쳐 승인 창을 띄웠고 Claude가 지시와 같음을 읽고 한 번만 승인했다(영구 허용 없음). 나머지 명령은 승인 창 없이 이어 실행되었다. 진단 명령과 2차 비교 묶음의 뒷 명령도 같은 방식이었다. 이 때문에 개별 명령을 승인 창에서 대조하지 못했고 agy의 기록 파일 대조는 권한 문제로 하지 못했다. 로그 파일 이름, 출력, 산출물 위치가 지시와 일치함은 확인했다.

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py -m pytest tests\15_csflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\15-pytest-261027-1.log
& $py -m pytest tests\common\test_cli.py -k csflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\15-pytest-cli-261027-1.log
& $py -m pytest tests\common\test_dependency.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\15-pytest-dependency-261027-1.log
& $py evaluation/visionlab/train.py train --data mvtec --model csflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\15-native-mvtec-261027-1.log
& $py evaluation/visionlab/predict.py --model csflow --data mvtec 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\15-pure-mvtec-261027-1.log
& $py evaluation/visionlab/train.py train --data visa --model csflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\15-native-visa-261027-1.log
& $py evaluation/visionlab/predict.py --model csflow --data visa 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\15-pure-visa-261027-1.log
& $py evaluation/anomalib/compare_outputs.py --model csflow --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\LocalCsflow\LocalMVTecAD\bottle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\15-compare-outputs-mvtec-261027-1.log
& $py evaluation/anomalib/compare_outputs.py --model csflow --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\LocalCsflow\LocalVisa\candle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\15-compare-outputs-visa-261027-1.log
& $py evaluation/anomalib/compare_history.py --reference D:\projects\visionlab\outputs\anomalib\LocalCsflow\LocalMVTecAD\bottle\v0 --native D:\projects\visionlab\outputs\visionlab\csflow\mvtec_bottle\v0 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\15-history-mvtec-261027-1.log
& $py evaluation/anomalib/compare_history.py --reference D:\projects\visionlab\outputs\anomalib\LocalCsflow\LocalVisa\candle\v0 --native D:\projects\visionlab\outputs\visionlab\csflow\visa_candle\v0 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\15-history-visa-261027-1.log
```

2차와 3차 비교 묶음은 다음과 같다(진단, `compare_outputs` 2회, `compare_history` 2회. 2차는 로그 이름이 `15-diag2-buffers-`, `15-compare-outputs2-`, `15-history2-`이고 참조가 `v1`이며, 3차는 `15-diag3-buffers-`, `15-compare-outputs3-`, `15-history3-`이고 참조가 `v2`다. 날짜는 261028이고 native는 `v0`다. 3차 묶음도 첫 명령의 승인 창만 읽고 한 번 승인했다).

```powershell
& $py evaluation/anomalib/diagnose_csflow_buffers.py --checkpoint mvtec=...LocalCsflow\LocalMVTecAD\bottle\v1\weights\lightning\model.ckpt --checkpoint visa=...LocalCsflow\LocalVisa\candle\v1\weights\lightning\model.ckpt
& $py evaluation/anomalib/compare_outputs.py --model csflow --samples 8 --reference-checkpoint ...LocalCsflow\LocalMVTecAD\bottle\v1\weights\lightning\model.ckpt
& $py evaluation/anomalib/compare_outputs.py --model csflow --samples 8 --reference-checkpoint ...LocalCsflow\LocalVisa\candle\v1\weights\lightning\model.ckpt
& $py evaluation/anomalib/compare_history.py --reference ...LocalCsflow\LocalMVTecAD\bottle\v1 --native ...csflow\mvtec_bottle\v0
& $py evaluation/anomalib/compare_history.py --reference ...LocalCsflow\LocalVisa\candle\v1 --native ...csflow\visa_candle\v0
```

사후 회귀 네 명령은 다음과 같다. agy가 이 묶음을 승인 창 없이 실행해 승인 창에서 대조하지 못했고 로그 파일 이름과 출력으로 일치를 확인했다.

```powershell
& $py -m pytest tests\12_fastflow\test_fastflow.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\15-regress-fastflow-261028-1.log
& $py -m pytest tests\13_cflow\test_cflow.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\15-regress-cflow-261028-1.log
& $py -m pytest tests\14_uflow\test_uflow.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\15-regress-uflow-261028-1.log
& $py -m pytest tests\common 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\15-regress-common-261028-1.log
```

agy는 파일을 고치지 않았다.

### 5.3 에폭별 학습 히스토리 비교 (허용 오차 없음, 비교표만)

다음은 `train_loss_epoch`의 10 epoch 비교이다. 3차 레퍼런스(재보정)와 native를 비교했고 판정에 쓰지 않는다. 절대 차이는 mvtec이 최대 1.9e-6, visa가 최대 2.9e-6이다.

| Epoch | mvtec 레퍼런스 | mvtec native | 상대 차이 | visa 레퍼런스 | visa native | 상대 차이 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 11.98285866 | 11.98285675 | 1.6e-7 | 10.59431934 | 10.5943222 | 2.7e-7 |
| 1 | 10.10266781 | 10.10266685 | 9.4e-8 | 3.979575396 | 3.979575634 | 6.0e-8 |
| 2 | 7.966346264 | 7.96634531 | 1.2e-7 | 0.9679414034 | 0.967941165 | 2.5e-7 |
| 3 | 6.436906338 | 6.436906815 | 7.4e-8 | -0.8717141151 | -0.8717142344 | 1.4e-7 |
| 4 | 5.181756496 | 5.181756496 | 0 | -2.474453926 | -2.474454165 | 9.6e-8 |
| 5 | 4.04635334 | 4.046352863 | 1.2e-7 | -3.236599684 | -3.236599207 | 1.5e-7 |
| 6 | 3.231389523 | 3.231389046 | 1.5e-7 | -4.00963068 | -4.009629726 | 2.4e-7 |
| 7 | 2.636137009 | 2.636137486 | 1.8e-7 | -4.429409027 | -4.429410458 | 3.2e-7 |
| 8 | 2.29812479 | 2.29812479 | 0 | -4.652621269 | -4.652620316 | 2.0e-7 |
| 9 | 1.508589149 | 1.508589149 | 0 | -5.129364014 | -5.129363537 | 9.3e-8 |

손실이 10 epoch 내내 float32 단위 오차 수준(상대 3e-7 이하)으로 일치한다. 초기 flow 가중치, 난수 소비 순서, 데이터 순서, optimizer 상태가 같다고 볼 수 있다. 앞의 두 레퍼런스와 비교하면 1차 visa의 epoch 0 손실은 5.65 대 10.59(상대 0.87)로 크게 달라 오염을 가리킨 첫 정황이었고, 2차는 epoch 0이 0.1% 이내로 같았지만 이후 최대 13%까지 벌어졌다.

### 5.4 crop·입력 크기

이 모델은 crop이 없고 입력은 256x256이다. 평가 출력은 `anomaly_map [B,1,256,256]`, `pred_score [B]`다.

## 6. 결론

- **레퍼런스:** bottle, visa 확보. 세 번 실행했다. 1차는 BatchNorm 오염으로 visa가 퇴화했고(2.3절), 2차(`eval()` 보정)는 난수 순서가 어긋나 visa Pixel AUROC가 0.0103 차이로 근소 미달이었으며(2.4절), 3차(재보정)를 판정에 썼다(2.5절). 앞의 두 실행의 로그와 산출물은 보존했다.
- **mvtec / bottle:** Image AUROC 차이 6e-8 이하, Pixel AUROC 차이 5.9e-7 이하. 통과.
- **visa / candle:** Image AUROC 차이 0, Pixel AUROC 차이 5.9e-6(native raw) 또는 1.8e-7(순수 정규화). 통과.
- **FR-5, FR-6, FR-23, FR-24:** 통과(5.2절). 동일 상태 출력의 최대 차이가 0.0이다. 시드 42 초기 `graph.*` 상태가 같다(INFO).
- **학습 히스토리:** `train_loss_epoch`가 10 epoch 모두 상대 3e-7 이하로 일치한다(5.3절).
- **에폭:** 레퍼런스와 native 모두 실제 10 epoch임을 로그와 `run.json`으로 확인했다(2.2절).
- **테스트:** `tests\15_csflow` 30개, `test_cli.py -k csflow` 4개, `test_dependency.py` 6개가 통과했다. 커밋 전 사용자 승인 사후 실행(2026-10-28)으로 `tests\12_fastflow\test_fastflow.py` 17개, `tests\13_cflow\test_cflow.py` 20개, `tests\14_uflow\test_uflow.py` 16개, `tests\common` 55개도 통과했다. 01~11 모델별 회귀는 사용자 결정으로 돌리지 않았다.
- **수정 요청:** 15-A 0회, 15-B 0회. 사용자 결정으로 진단 스크립트와 어댑터 보정 2회를 추가했다.
- **`specs/` 변경:** 사용자 결정(2026-10-28)에 따라 `SPEC.md` FR-4에 CSFlow 출력 shape 문장을 더하고, `DECISIONS.md`의 D-15 트리에 csflow 파일을 추가하고 D-26에 CSFlow 적용 재확인을 적었으며, `PLAN.md` 공통 양식의 레퍼런스 항목에 어댑터 보정 기록을 더했다(6.1절). INTENT는 고치지 않았다.
- **한계:** 레퍼런스 어댑터가 원본 anomalib의 BatchNorm 오염 동작과 다르다. 논문의 긴 학습과 768 입력에서의 성능은 확인하지 않았다.
- **Phase 종료:** 사용자가 종료를 결정했다(2026-10-28). `Phase 15: csflow ...` 커밋으로 푸시한다.

### 6.1 `specs/` 갱신 항목

`specs/`는 사람이 쓴다. 최종 보고에 대한 사용자 결정(2026-10-28)에 따라 아래와 같이 처리했다. INTENT는 고치지 않았다.

- **FR-4 (반영 완료):** `SPEC.md` FR-4에 "CSFlow는 원본의 평탄화 평균을 유지해 score가 `[B]`이고 anomaly map은 `[B,1,H,W]`다"를 CFLOW 문장 뒤에 더하고 판정 방법 문구를 "CFA, SuperSimpleNet, DRAEM, CFLOW, CSFlow는 원본 동작의 shape를 확인"으로 맞췄다.
- **D-15 트리 (반영 완료):** `tests/15_csflow/`(두 파일), `configs/models`의 `csflow.yaml`, 모델 패키지 `models/csflow/`(다섯 파일)을 추가했다.
- **D-26 재확인 (반영 완료):** "Phase 13~15는 해당 Phase 시작 때 다시 확인한다" 문장에 Phase 15에서 CSFlow의 사용처가 `GraphINN`, `InvertibleModule`, `InputNode`, `Node`, `OutputNode`임을 확인했고 새 결정은 필요 없다는 내용을 더했다.
- **레퍼런스 어댑터 보정 기록 (반영 완료):** D 번호를 새로 만들지 않고 `PLAN.md` 공통 양식의 레퍼런스 항목("레퍼런스는 데이터셋마다 1회만 실행한다...") 뒤에 두 문장을 더했다. 내용은 `nn.Module` backbone을 공유 `TimmFeatureExtractor`에 넘기는 모델에서 생성자 dry-run이 BatchNorm 통계를 오염시킬 수 있고 원본 anomalib 동작이라 결과가 실행마다 달라질 수 있다는 점, 그런 모델의 어댑터는 dry-run을 train 모드로 두어 난수 순서를 맞추고 생성 직후 가중치를 복원한다는 점(Phase 15 예외)이다.

### 6.2 남은 위험과 사용자 판단 대상

- **레퍼런스 보정은 원본 anomalib 동작과 다르다.** 원본은 dry-run 오염을 안고 있어 실행마다 결과가 달라질 수 있다. 3차 레퍼런스는 결정적이고 native와 일치하지만 원본 그대로의 결과가 아니다. 원본 그대로 실행하면 mvtec처럼 거의 맞을 수도, visa처럼 퇴화할 수도 있다.
- **앞 Phase 레퍼런스에 같은 dry-run 오염이 있었을 가능성(후속 확인 과제).** `nn.Module` backbone을 공유 `TimmFeatureExtractor`에 넘기는 모델이 앞 Phase에도 있다면 같은 오염이 있었을 수 있다. 이번에는 조사하지 않았다.
- **논문의 긴 학습과 768 입력에서의 성능은 확인하지 않았다.** 공유 10 epoch, 256 입력의 결과만 비교했다.
- **일부 agy 명령의 승인 대조 누락.** 5.2절과 2.5절에 적었다.
- **`test_csflow_lightning.py` 본문은 줄 단위로 읽지 않았다**(실행 결과로 판정). `compare_outputs.py`의 csflow 분기도 구조만 훑었다.
- **모델 규모.** 체크포인트 약 3.4GB, export 약 1.3GB다. 레퍼런스를 세 번 실행해 `outputs/`가 커졌다. `outputs/`는 커밋하지 않는다.
- **배포 환경 요건:** 순수 `CsFlowModel`과 csflow export 로딩에는 FrEIA 0.2와 numpy가 필요하다(D-26). 다른 모델의 `Predictor`는 FrEIA 없이 import되도록 `csflow` 분기 안에서만 지연 import한다.
- **사후 실행:** 사용자가 승인한 `tests\common\test_dependency.py` 전체 실행은 6개, 허용 목록을 한 항목씩 고친 `test_fastflow.py`는 17개, `test_cflow.py`는 20개, `test_uflow.py`는 16개, `tests\common` 전체는 55개가 통과했다(5.2절, 사용자 승인 회귀 실행 2026-10-28).
- **앞 모델 회귀 보류(사용자 결정):** 01~11 모델별 테스트는 돌리지 않았다. 변경은 기존 항목을 건드리지 않는 등록과 분기 추가이고 `predictor.py`의 기존 분기는 그대로다. 사용자가 승인한 사후 회귀(5.2절)는 모두 통과했다.

| 공유 파일 | 변경 | 영향받을 수 있는 앞 모델 |
|---|---|---|
| `src/visionlab/utils/io.py` | `load_export`가 허용하는 이름에 `csflow` 추가 | 모든 모델의 export 로딩 |
| `src/visionlab/inference/predictor.py` | `from_export`에 `csflow` 분기 추가(분기 안 지연 import, 기존 분기는 그대로) | 모든 모델의 `Predictor.from_export` |
| `evaluation/visionlab/predict.py` | `--model` choices에 `csflow` 추가 | 없음(선택지 추가) |
| `evaluation/anomalib/compare_outputs.py` | `--model csflow` 분기 추가 | 없음(기존 분기 유지) |
| `evaluation/anomalib/local_models.py` | `LocalCsflow`가 dry-run 뒤 가중치를 복원 | 없음(`LocalCsflow`만) |
| `tests/common/test_cli.py` | csflow 테스트만 추가 | 없음 |
| `tests/common/test_dependency.py` | 순수 모듈 3개 분류와 Lightning 허용 경로 한 줄 | 분류 규칙 전체(`tests\common` 55 passed) |
| `tests/12_fastflow/test_fastflow.py`, `tests/13_cflow/test_cflow.py`, `tests/14_uflow/test_uflow.py` | FrEIA import 허용 목록에 한 항목씩 | 세 파일의 해당 테스트(사후 실행 17, 20, 16개 통과) |
