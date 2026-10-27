# Phase 13. CFLOW

모델 폴더 이름은 `cflow`이고 원본은 anomalib 2.0.0의 `anomalib\models\image\cflow\`(`C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\Lib\site-packages\anomalib\models\image\cflow\`)이다. 범위와 완료 조건은 `specs/PLAN.md`의 Phase 13과 "Phase 공통 양식"을 따른다. 판정 기준은 category별 Image·Pixel AUROC 차이 0.01 이내이며, 이번 Phase는 레퍼런스와 native 모두 1 epoch만 학습한 결과로 비교한다.

## 상태

다음 표는 단계별 진행 상태다. 수정 반복 한도 3회는 단계별로 센다.

| 단계 | 내용 | 상태 |
|---|---|---|
| 사전 작업 | 레퍼런스 yaml에 `trainer.max_epochs: 1` 추가, `default.yaml` 주석 수정 | 완료 |
| 레퍼런스 | `anomalib_run.py train` x 2 (mvtec/bottle 1회, visa/candle 1회, 1 epoch) | 완료 |
| 13-A | 순수 모델 (`models/cflow/{model,utils,anomaly_map}.py`, `tests/13_cflow`의 순수 부분) | 완료 |
| 13-B | Lightning (`CflowLitModel`, 수동 최적화, configs, export·Predictor 등록, 비교 스크립트) | 완료 |
| 회귀 실행 | 사용자 승인 사후 실행 (`tests\12_fastflow\test_fastflow.py` 17개, `tests\common` 47개, 2026-10-27, 커밋 전) | 완료 (모두 통과) |
| specs 정리 | D-22 1 epoch 예외, FR-4 예외, D-15 트리 반영 (사용자 결정) | 완료 |

`components/flow/all_in_one_block.py`(Phase 12), `TimmFeatureExtractor`, `PreProcessor`(ImageNet 정규화), `PostProcessor`, `Evaluator`, `load_teacher_weights`는 앞 Phase의 것을 재사용한다.

## 사용자 결정 (2026-10-27)

다음은 Phase 시작 전에 사용자가 정한 항목이다. 사용자가 계획 보고의 결정 항목 1~7번을 모두 권고안대로 채택했다.

| 항목 | 결정 |
|---|---|
| 학습 에폭 (1 epoch 예외) | CFLOW는 학습에 시간이 많이 걸려 레퍼런스와 native 모두 1 epoch만 학습하고, 그 1 epoch 결과의 AUROC만 비교한다(차이 0.01 이내). 10 epoch 이상의 비교는 하지 않는다. 원본 `trainer_arguments`는 `max_epochs`를 고정하지 않으므로(`gradient_clip_val` 0과 `num_sanity_val_steps` 0뿐) 모델 yaml의 `trainer.max_epochs: 1`을 레퍼런스(`13_cflow.yaml`)와 native(`configs/models/cflow.yaml`) 모두에 둔다. `LocalCflow` 어댑터는 만들지 않는다. `evaluation/anomalib/default.yaml`의 주석만 이 예외에 맞게 고친다. 레퍼런스와 native가 실제 같은 1 epoch인지 `run.json`과 로그의 `max_epochs`로 확인해 기록한다 |
| D-22 정합 | 문구 초안을 6.1절에 적고 Phase 끝에 반영 여부를 묻는다. 지금 `specs/`는 고치지 않는다. 초안에는 "사용자가 Phase 시작 전에 정한 모델에 한해 모델 yaml로 에폭을 정한다"를 명시해 D-22의 "모델 yaml로 덮어쓰지 않는다"와 충돌하지 않게 한다 |
| `einops` | CFA와 같이 같은 결과의 torch 연산(`permute`, `reshape`, `repeat`)으로 바꾼다. 테스트에서 `einops`와 `torch.equal`로 동일성을 확인하고 `src/visionlab`에는 `einops` import가 남지 않는다 |
| 평가 category | mvtec/bottle과 visa/candle이다. 전체 예상이 약 45분을 넘을 것으로 보이면 계속하기 전에 보고하고 스스로 바꾸지 않는다 |
| 앞 모델 회귀(01~12) | 보류(사용자 결정 2026-10-27). 돌리지 않는다. 영향 표는 6.2절에 남긴다. 최종 보고에서 사후에 `tests\common`만 1회 실행할지 묻는다(돌리지 않는다) |
| FR-4 예외와 batch 의존 | CFLOW·DRAEM의 `anomaly_map [B,H,W]` 예외와 `anomaly_map`의 batch 의존은 Phase 끝에 정한다. 지금 `specs/`는 고치지 않고 초안과 사용 시 주의를 6.1절, 6.2절에 적어 최종 보고에서 반영 여부를 묻는다 |
| codex 세션 | 새 세션으로 시작한다. 지시문은 자체 완결이다 |
| `Predictor` 등록 | `predictor.py`의 fastflow 분기는 건드리지 않고 `cflow` 분기를 따로 지연 import로 추가한다(D-26) |
| FR-5 허용 오차 | `atol=1e-6, rtol=1e-5` (Phase 2~12와 같음) |
| 에폭별 `train_loss` 히스토리 | 1 epoch이므로 1행 표만 기록한다. 허용 오차 없음 |
| 난수가 어긋날 때의 판정 | 난수 상태를 주입하지 않고 AUROC 0.01로만 판정한다. 초기 가중치 비교는 기록한다 |
| `specs/` 변경 | 없다. 필요한 항목은 6.1절 "specs 갱신 필요"에 초안으로 적고 최종 보고에서 묻는다 |
| 실행 범위 | CFLOW의 명령만 실행한다 |
| `test_dependency.py` 전체 실행 | 필요하면 사후 승인 항목으로 올린다 |
| 판정 | Image·Pixel AUROC를 판정한다. 정규화 점수 기준 값(D-24)과 raw 값을 함께 기록한다 |
| 최종 보고 뒤 결정 (2026-10-27) | 결정 1~5를 권고안대로 채택했다. (1) D-22 1 epoch 예외 문구와 FR-4 예외 문구를 specs에 반영한다. (2) D-15 트리에 cflow 파일을 추가한다. (3) `test_dependency.py` 전체 실행은 사후 승인한다. (4) 수정한 `tests/12_fastflow/test_fastflow.py`를 실행한다. (5) 사후에 `tests\common`을 1회 실행한다(이번 한 번에 한해 예외, 01~12 모델별 테스트는 계속 보류). Phase 종료를 결정했다 |

## 1. 모델 노트

### 1.1 원본과 생성자

다음 표는 원본 파일과 이식 위치의 대응이다(D-15, D-18).

| 원본 | 이식 위치 | 비고 |
|---|---|---|
| `models/image/cflow/torch_model.py` (`CflowModel`) | `src/visionlab/anomaly_detection/models/cflow/model.py` | import 변경, `einops` 제거, `TimmFeatureExtractor`는 구조만 만드는 방식으로(D-19) |
| `models/image/cflow/utils.py` (`cflow_head`, `get_logp`, `positional_encoding_2d`, `subnet_fc`) | `.../models/cflow/utils.py` | `AllInOneBlock` import 경로 변경. FrEIA `SequenceINN`, `numpy`는 원본대로 |
| `models/image/cflow/anomaly_map.py` (`AnomalyMapGenerator`) | `.../models/cflow/anomaly_map.py` | 변경 없음(예상) |
| `models/components/flow/all_in_one_block.py` | `.../models/components/flow/all_in_one_block.py` | Phase 12 것을 그대로 재사용 |
| `models/image/cflow/lightning_model.py` (`Cflow`) | `.../models/cflow/lightning_model.py` (`CflowLitModel`) | native `LightningModule`로 새로 작성(수동 최적화) |
| `models/image/cflow/__init__.py` | `.../models/cflow/__init__.py` | `lightning_model`을 import하지 않도록 고침(D-18 예외) |

생성자는 다음과 같다. `CflowLitModel`은 `Cflow`와 인자 이름·순서·기본값이 같아야 한다(FR-6, D-5).

| 클래스 | 시그니처 |
|---|---|
| `Cflow` | `backbone="wide_resnet50_2"`, `layers=("layer2", "layer3", "layer4")`, `pre_trained=True`, `fiber_batch_size=64`, `decoder="freia-cflow"`, `condition_vector=128`, `coupling_blocks=8`, `clamp_alpha=1.9`, `permute_soft=False`, `lr=0.0001`, `pre_processor=True`, `post_processor=True`, `evaluator=True`, `visualizer=True` (14개) |
| `CflowModel` | `backbone`, `layers`, `pre_trained=True`, `fiber_batch_size=64`, `decoder="freia-cflow"`, `condition_vector=128`, `coupling_blocks=8`, `clamp_alpha=1.9`, `permute_soft=False` (9개, 앞 두 개는 기본값 없음) |
| `AnomalyMapGenerator` | `pool_layers` |

### 1.2 학습과 평가 동작

원본의 동작은 다음과 같다.

- **구조:** `TimmFeatureExtractor`(wide_resnet50_2의 layer2~4, `eval()`, 파라미터 고정)가 특징을 내고, layer마다 `cflow_head`가 만든 조건부 `SequenceINN`(`AllInOneBlock` `coupling_blocks`개, 조건 벡터 `condition_vector`)이 decoder가 된다.
- **학습(수동 최적화):** `training_step`이 `automatic_optimization=False`로 `self.optimizers()`를 직접 쓴다. layer마다 특징 `[B,C,H,W]`와 위치 encoding(`positional_encoding_2d`)을 `(B*H*W, C)`와 `(B*H*W, P)`로 펼치고 `torch.randperm`으로 섞은 뒤 `fiber_batch_size`(64)개씩 잘라, 조각마다 `opt.zero_grad()`, decoder 호출, `get_logp`, `-logsigmoid(log_prob / C)`, `manual_backward(loss.mean())`, `opt.step()`을 한다. `avg_loss`는 조각 손실의 합이고 `train_loss`로 로그한다.
- **평가 `forward`:** 같은 펼침을 섞지 않고 순서대로 decoder에 넣어 layer마다 `log_prob` 분포를 모으고, `AnomalyMapGenerator`가 layer별 분포에서 `exp(분포 - 분포.max())`로 확률을 만들고 입력 크기로 보간해 합친 뒤 `score_map.max() - score_map`을 낸다. 최대값을 batch 전체에서 구하므로 점수가 batch 구성에 의존한다. `anomaly_map`은 `.squeeze(1)` 뒤 `[B,H,W]`이고 `pred_score`는 `amax`로 `[B]`다.
- **optimizer:** decoder 파라미터만 `Adam(lr=0.0001)`이 학습한다(scheduler 없음).
- **`trainer_arguments`:** `{"gradient_clip_val": 0, "num_sanity_val_steps": 0}`이며 `max_epochs`는 고정하지 않는다. 이번 Phase는 모델 yaml로 1 epoch를 정한다(사용자 결정).

### 1.3 외부 의존과 로컬 자산

- **외부 의존:** timm, FrEIA(0.2, `SequenceINN`, D-26), numpy(`get_logp`의 `np.log`, FrEIA 요구), scipy(`AllInOneBlock`의 최상단 import)다. `einops`는 원본이 쓰지만 제거한다. kornia, sklearn은 쓰지 않는다. anomalib 의존은 `InferenceBatch`, `TimmFeatureExtractor`, `AllInOneBlock`, `AnomalibModule`, `PreProcessor`, `PostProcessor`, `Evaluator`, `Visualizer`, `LearningType`이다.
- **로컬 가중치:** `paths.toml [backbones]`의 `resnet18`, `resnet50`, `wide_resnet50_2`가 있다. 기본값(`wide_resnet50_2`, layer2~4)만 실제로 학습·평가하고 다른 백본은 구조 생성만 확인한다.
- **데이터:** MVTec `bottle`(학습 209장), VisA `candle`(학습 900장). 분할은 `from_test`, 0.4, seed 42, batch 8/8, `num_workers=0`이다.
- **학습 예산:** 모델 yaml의 `max_epochs=1`이다.

### 1.4 이식에서 의도적으로 달라지는 곳과 유지하는 곳

- **`einops` 제거(사용자 결정):** `repeat`와 `rearrange` 세 곳(`model.py` 평가 forward, `lightning_model.py` 학습)을 같은 결과의 torch 연산으로 바꾼다.
- **pretrained 주입(D-19):** 순수 `CflowModel`은 `TimmFeatureExtractor`를 `pre_trained=False`로 만든다. `LitModel`이 `pre_trained=True`일 때 로컬 가중치를 주입한다.
- **원본 동작 유지(고치지 않음):** decoder 생성 순서와 난수 소비, 학습의 `torch.randperm` 호출 순서(layer마다 batch마다), 수동 최적화의 step 순서, 평가의 batch 의존 정규화, `anomaly_map` 3차원 출력, `decoder` 인자가 쓰이지 않는 점.
- **`train_loss` 로그의 집계:** 원본은 batch 크기를 추론하지 못해 epoch 평균이 step마다 1회씩 센 평균이다. visionlab은 `batch_size=1`을 두어 맞춘다(Phase 7, 10, 11, 12 선례).
- **히스토리 비교:** 1 epoch이므로 `train_loss_epoch`가 1행뿐이다. 한 점의 일치 확인에 그친다(판정 아님).
- **FR-5 허용 오차:** `pred_score`와 `anomaly_map`은 `atol=1e-6, rtol=1e-5`다.

## 2. 레퍼런스

학습 예산은 모델 yaml의 `trainer.max_epochs=1`이다. 원본이 `max_epochs`를 고정하지 않으므로 어댑터 없이 yaml만으로 정했고(사용자 결정, 1.2절), 레퍼런스와 native 모두 1 epoch다. 10 epoch 이상의 비교는 하지 않는다.

### 2.1 1 epoch 레퍼런스

2026-10-27에 agy가 `anomalib_run.py train --config default.yaml --config data/<data>.yaml --config models/13_cflow.yaml`을 데이터셋마다 1회 실행했다(실행으로 검증함). Claude는 로그 파일에서 값을 직접 확인했다. 레퍼런스·native·순수 로그에서 `network access is blocked`, `URLError`, `ConnectionError`, `Traceback`, `out of memory`는 0건이다. 로그는 UTF-16이다. 두 명령의 승인 창은 지시한 줄과 정확히 같을 때만 한 번씩 승인했다(영구 허용 없음).

agy가 PowerShell에서 실행한 명령은 다음과 같다(Claude가 보낸 지시 원문. 파일은 고치지 않았다).

```powershell
cd D:\projects\visionlab\evaluation\anomalib
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/13_cflow.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\13-ref-mvtec-261027-1.log
& $py anomalib_run.py train --config default.yaml --config data/visa.yaml --config models/13_cflow.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\13-ref-visa-261027-1.log
```

다음 표는 1 epoch 레퍼런스(1회) 결과다.

| 데이터 / category | Image AUROC | Pixel AUROC | 참고 | run 폴더 | 로그 |
|---|---|---|---|---|---|
| mvtec / bottle | 0.99999994 | 0.98360181 | image F1 1.0000, pixel F1 0.7378 | `outputs/anomalib/Cflow/LocalMVTecAD/bottle/v0` | `outputs/logs/13-ref-mvtec-261027-1.log` |
| visa / candle | 0.92694449 | 0.99019313 | image F1 0.8372, pixel F1 0.2941 | `outputs/anomalib/Cflow/LocalVisa/candle/v0` | `outputs/logs/13-ref-visa-261027-1.log` |

적용된 학습 예산은 `max_epochs=1`, `val_split_mode=from_test`, `val_split_ratio=0.4`, `seed=42`, `batch 8/8`, `num_workers=0`, `backbone=wide_resnet50_2`, `layers=[layer2, layer3, layer4]`, `pre_trained=true`, `fiber_batch_size=64`, `decoder=freia-cflow`, `condition_vector=128`, `coupling_blocks=8`, `clamp_alpha=1.9`, `permute_soft=false`, `lr=0.0001`이다. 체크포인트는 각 run 폴더의 `weights/lightning/model.ckpt`이고 크기는 1,600,026,636 bytes다.

### 2.2 레퍼런스와 native의 에폭 확인

사용자 지시에 따라 레퍼런스와 native가 같은 1 epoch인지 실제 값으로 확인했다.

| 대상 | 근거 | 값 |
|---|---|---|
| 레퍼런스 (bottle, visa) | 로그의 `Trainer.fit stopped: max_epochs=1 reached`, 완료한 에폭 번호 0 | 1 |
| native (bottle, visa) | `run.json`의 `config.trainer.max_epochs` 1, 로그의 `Trainer.fit stopped: max_epochs=1 reached`, 완료한 에폭 번호 0 | 1 |

### 2.3 학습 시간 관문과 실제 학습 시간

계획의 추정(레퍼런스와 native 합계 약 25~30분, 범위 12~60분)의 1.5배(약 45분)를 넘지 않아 보고 없이 진행했다. 실제 1 epoch 학습 시간(로그의 진행 바)은 다음과 같다.

| 구분 | bottle (27 batch) | visa (113 batch) |
|---|---|---|
| 레퍼런스 | 2분 13초 | 8분 21초 |
| native | 1분 58초 | 8분 38초 |

파일 시각으로 레퍼런스 두 run이 약 12분(16:38~16:50), native 두 run과 평가가 약 16분(17:09~17:25)이었고 합계 약 28분이었다.

### 2.4 에폭별 학습 히스토리

1 epoch이므로 `train_loss_epoch`는 한 행뿐이다(5.3절). 추이 비교는 불가능하다.

## 3. 이식 메모

### 3.1 원본과 이식 파일의 대응

다음 표는 새로 만든 파일과 원본의 대응이다. 원본과의 차이는 `difflib`로 확인했다(구현함, 정적 검증).

| 이식 파일 (`src/visionlab/anomaly_detection/models/cflow/`) | 원본 | 원본과 다른 줄 |
|---|---|---|
| `model.py` | `torch_model.py` | import 3줄(`InferenceBatch`, `TimmFeatureExtractor`, `einops` 제거), `TimmFeatureExtractor(... pre_trained=False)`와 주석(D-19), `einops` 세 곳(`repeat` 한 곳, `rearrange` 두 곳)을 같은 결과의 torch 연산으로 대체 |
| `utils.py` | `utils.py` | `AllInOneBlock` import 경로 한 줄. FrEIA `SequenceINN`과 `numpy` import는 원본대로 |
| `anomaly_map.py` | `anomaly_map.py` | 없음 |
| `__init__.py` | `__init__.py` | `CflowModel`만 노출(D-18 예외) |
| `lightning_model.py` (`CflowLitModel`) | `lightning_model.py` | native `LightningModule`로 새로 작성. 14개 인자 동일, 수동 최적화 루프와 `torch.randperm` 호출 순서 유지, `einops`를 같은 torch 연산으로 대체, `train_loss` 로그(`batch_size=1`), 로컬 backbone 주입(D-19), D-8의 post_processor 갱신 제한, `trainer_arguments`에 `max_epochs` 없음 |

`models/components/flow/all_in_one_block.py`는 Phase 12의 것을 수정 없이 재사용했다.

### 3.2 새로 쓴 코드와 앞 Phase 공유 파일 변경

- **새로 쓴 파일:** `configs/models/cflow.yaml`, `models/cflow/`, `tests/13_cflow/test_cflow.py`, `tests/13_cflow/test_cflow_lightning.py`. `tests\13_cflow`는 합계 29개 테스트다(순수 20개, Lightning 9개).
- **평가 설정 변경:** `evaluation/anomalib/models/13_cflow.yaml`에 `trainer.max_epochs: 1`과 주석 한 줄, `evaluation/anomalib/default.yaml`의 맨 위 주석을 이번 예외에 맞게 고쳤다(값 `max_epochs: 10`은 그대로).
- **공유 파일 변경:** `utils/io.py`(`load_export`가 `cflow` 허용), `inference/predictor.py`(`cflow` 분기를 fastflow 분기와 따로 추가하고 분기 안에서 지연 import), `evaluation/visionlab/predict.py`(`--model cflow`), `evaluation/anomalib/compare_outputs.py`(`--model cflow` 분기), `tests/common/test_cli.py`, `tests/common/test_dependency.py`. `common/`와 `components/flow/`는 바꾸지 않았다. 영향 범위는 6.2절의 표에 정리했다.
- **앞 Phase 테스트 변경:** `tests/12_fastflow/test_fastflow.py`의 `test_fastflow_freia_import_locations`(FrEIA를 import하는 파일의 허용 목록)에 `cflow/utils.py` 한 항목과 docstring 한 줄을 추가했다. CFLOW가 FrEIA를 import하는 파일이 늘어 기존 목록 검사가 실패하게 되기 때문이다. 이번 Phase에서 이 파일은 실행하지 않았다(6.2절).
- **새 부품:** 대체 부품은 없다. 모두 원본 복사이며 `einops`만 같은 결과의 torch 연산으로 바꿨다.
- **레퍼런스 어댑터 변경:** 없다. 원본 `Cflow`를 그대로 쓴다.

### 3.3 이식 중 확인된 사항

- **1 epoch 설정:** 모델 yaml의 `trainer.max_epochs: 1`이 레퍼런스에서 그대로 적용되었다(로그의 `max_epochs=1 reached`와 에폭 번호 0으로 확인). 해당 yaml 방식은 DFM의 native yaml과 같다.
- **난수 소비 순서:** decoder 생성(`AllInOneBlock`의 `torch.randperm`, MLP 초기화)과 학습 루프의 `torch.randperm`(layer마다 batch마다) 순서를 고치지 않았다. 같은 seed 42에서 레퍼런스와 visionlab 순수 모델의 `state_dict` 전체(encoder 초기 가중치, decoder의 permutation 버퍼와 가중치 포함)가 서로 같다(`INFO`). 학습 결과도 레퍼런스와 일치한다(5절).
- **조건부 flow:** `cflow_head`가 `SequenceINN`과 `AllInOneBlock`을 `cond=0, cond_shape=(condition_vector,)`로 쓴다. Phase 12의 `AllInOneBlock` 복사본이 그대로 맞았고, 조건을 쓰는 호출 `decoder(x, [c])`의 `(z, logdet)`가 레퍼런스와 일치한다(5.2절).
- **`train_loss`의 집계:** `self.log(..., batch_size=1)`을 처음부터 두었고, native의 `train_loss_epoch`가 레퍼런스와 상대 차이 4.1e-7 이내로 일치했다(5.3절).
- **`TimmFeatureExtractor`의 모드:** 원본 `TimmFeatureExtractor`는 `train()`을 재정의하지 않는다. `fit`이 모델을 train 모드로 바꾸면 encoder의 BatchNorm 통계가 갱신될 수 있다. 이는 원본 동작이므로 `training_step`에 `encoder.eval()`을 넣지 않았다. 레퍼런스와 native가 같은 동작이라 AUROC와 손실이 일치했다.
- **평가 지도의 batch 의존:** `AnomalyMapGenerator`가 분포의 `max()`와 점수 지도의 `max()`를 batch 전체에서 구해 정규화하므로 같은 이미지라도 batch 구성에 따라 `anomaly_map`과 `pred_score`가 달라진다. 원본 동작을 그대로 두었고 순수 모델 테스트가 이를 명시한다(`test_cflow_forward_preserves_batch_dependent_scores`).
- **`anomaly_map` 형태:** 원본 docstring은 `[B,1,H,W]`라고 적혀 있으나 코드는 `.squeeze(1)` 뒤 `[B,H,W]`이고 `pred_score`는 `[B]`다. DRAEM과 같다.
- **가중치 주입 헬퍼:** `load_teacher_weights(self.model.encoder, backbone)`가 `encoder`(`TimmFeatureExtractor`) 안의 timm 모델에 그대로 맞았다.

## 4. 적대적 검증

검증은 13-A, 13-B 각각 한 회차씩 했다. 수정 요청은 13-A 1회, 13-B 0회(한도 3회 이내)다.

### 4.1 13-A (순수 모델)

| 지적 | 심각도 | 처리 |
|---|---|---|
| 원본과의 `difflib` 대조: `anomaly_map.py`는 차이 없음, `utils.py`는 import 한 줄, `model.py`는 import 3줄과 `pre_trained=False`, `einops` 세 곳의 torch 연산 대체뿐. 생성 순서, `forward`, `squeeze` 결과가 같음 | 확인 | 수정 없음 |
| agy 실행에서 `tests\13_cflow` 28 passed, 1 failed. `test_cflow_forward_preserves_batch_dependent_scores`가 `torch.allclose(alone.pred_score, torch.zeros(1))`를 기본 허용 오차(atol 1e-8)로 비교했으나 batch 하나에서 정규화한 점수가 정확한 0이 아니라 7.15e-07이었다(bilinear 보간과 float 반올림). 모델 코드 문제가 아니라 테스트의 정확한 0 비교였다 | 낮음 | 수정 요청 1회: 해당 assert와 같은 종류의 비교를 FR-5 허용 오차 `atol=1e-6, rtol=1e-5`로 바꿈. 재실행에서 29 passed. 승인 창의 명령은 Claude가 직접 읽어 지시한 줄과 같음을 확인한 뒤 승인 |
| 코드 실행 전 codex가 보고한 충돌: `tests/12_fastflow/test_fastflow.py`의 FrEIA import 파일 목록 | 낮음 | 목록에 `cflow/utils.py`를 추가(3.2절). Phase 12 테스트를 실행하지 않았음을 6.2절에 기록 |

### 4.2 13-B (Lightning, export, 비교 스크립트)

| 지적 | 심각도 | 처리 |
|---|---|---|
| 원본 `lightning_model.py`와의 대조: 생성 순서, `Adam(decoder 파라미터, lr)`, 수동 최적화 루프(`zero_grad`, `manual_backward`, `step`의 순서와 횟수), `torch.randperm` 호출 위치, `train_loss`, `trainer_arguments`가 같음. 의도한 차이는 `einops` 대체와 로컬 가중치 주입, `batch_size=1`뿐. 공유 파일 변경은 등록과 분기 추가이고 `predictor.py`의 fastflow 분기는 그대로임 | 확인 | 수정 없음 |
| `compare_outputs`의 cflow 분기는 `git diff`로 전체를 읽었다. 학습 루프는 수동 최적화라 단계 비교를 만들지 않았고(보고서 `limitations`에 기록) 1 epoch AUROC와 `train_loss_epoch`로 대신한다. `tests/13_cflow/test_cflow_lightning.py`는 테스트 이름과 구조만 확인했고 본문은 줄 단위로 읽지 않았다(실행 결과로 판정) | 낮음 | 수정 없음 |

남은 위험은 6.2절에 적었다.

## 5. 이식 결과

### 5.1 최종 판정 (1 epoch, Image·Pixel AUROC 차이 0.01 이내)

native는 `train.py`가 test에서 기록한 값이고 순수 PyTorch는 `export/model.pt`만 불러 `predict.py`로 평가한 값이다. 모두 실행으로 검증했다. D-24에 따라 레퍼런스가 보고하는 값과 같은 정의로 비교하며, 순수 PyTorch의 정규화 점수 AUROC도 함께 적는다. 레퍼런스와 native는 모두 1 epoch이고 이 값만 비교 대상이다.

| 데이터 / category | 지표 | 레퍼런스 | native (raw) | 순수 PyTorch (raw) | 순수 PyTorch (정규화) | 차이(native raw) | 차이(순수 정규화) | 판정 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| mvtec / bottle | Image AUROC | 0.99999994 | 1.00000000 | 1.00000000 | 0.99999994 | 6e-8 | 0 | 통과 |
| mvtec / bottle | Pixel AUROC | 0.98360181 | 0.98360181 | 0.98360181 | 0.98360181 | 0 | 0 | 통과 |
| visa / candle | Image AUROC | 0.92694449 | 0.92694443 | 0.92694443 | 0.92694449 | 6e-8 | 0 | 통과 |
| visa / candle | Pixel AUROC | 0.99019313 | 0.99042362 | 0.99042368 | 0.99019325 | 2.3e-4 | 1.2e-7 | 통과 |

visa Pixel AUROC에서 raw 값(0.99042)과 레퍼런스가 보고하는 정규화 값(0.99019)이 2.3e-4 다르다. D-24의 두 정의 차이이고 0.01 이내여서 원인 기록 조건에는 해당하지 않는다. 같은 정의로 비교하면 차이는 1.2e-7이다. image F1과 pixel F1도 레퍼런스, native, 순수 PyTorch가 같다(bottle image 1.0000, pixel 0.73782. visa image 0.83721, pixel 0.29410. 순수 PyTorch bottle pixel F1은 0.73781669로 1.1e-6 차이).

run 폴더와 로그는 다음과 같다.

| 대상 | 위치 |
|---|---|
| 레퍼런스 | `outputs/anomalib/Cflow/LocalMVTecAD/bottle/v0`, `outputs/anomalib/Cflow/LocalVisa/candle/v0` |
| native | `outputs/visionlab/cflow/mvtec_bottle/v0`, `outputs/visionlab/cflow/visa_candle/v0` (`run.json`: `max_epochs` 1, `num_sanity_val_steps` 0, `gradient_clip_val` 0.0, seed 42, 가중치 `D:\backbones\wide_resnet50_2-95faca4d.pth`, 학습 이미지 bottle 209장, candle 900장) |
| 순수 PyTorch | 같은 run의 `predict/metrics.json`, `export/model.pt`(946,657,434 bytes) |
| 로그 | `outputs/logs/13-native-{mvtec,visa}-261027-1.log`, `13-pure-{mvtec,visa}-261027-1.log` |

### 5.2 테스트와 비교 스크립트

다음 표는 agy가 실행한 묶음의 결과다(Claude가 로그 파일에서 확인). 실행 대상은 CFLOW뿐이다(앞 모델 회귀는 보류).

| 대상 | 결과 | 로그 |
|---|---|---|
| `tests\13_cflow` (1차) | 28 passed, 1 failed (테스트의 정확한 0 비교, 4.1절) | `13-pytest-261027-1.log` |
| `tests\13_cflow` (수정 후 재실행) | 29 passed (순수 20, Lightning 9) | `13-pytest-261027-2.log` |
| `tests\common\test_cli.py -k cflow` | 4 passed (23 deselected) | `13-pytest-cli-261027-1.log` |
| `tests\common\test_dependency.py` (전체 파일) | 6 passed | `13-pytest-dependency-261027-1.log` |
| `compare_outputs.py --model cflow` (bottle 체크포인트) | PASS 35, FAIL 0, INFO 1 | `13-compare-outputs-mvtec-261027-1.log` |
| `compare_outputs.py --model cflow` (candle 체크포인트) | PASS 35, FAIL 0, INFO 1 | `13-compare-outputs-visa-261027-1.log` |
| 사용자 승인 사후 실행(2026-10-27): `tests\12_fastflow\test_fastflow.py` | 17 passed (허용 목록 한 항목 수정 확인) | `13-pytest-fastflow-261027-1.log` |
| 사용자 승인 회귀 실행(2026-10-27): `tests\common` | 47 passed (이전 43개에 cflow 4개 추가) | `13-regress-common-261027-1.log` |

`compare_outputs`의 PASS 항목은 FR-6(생성자 시그니처), FR-23(`encoder`의 timm 모델 `state_dict`가 `torch.equal`), FR-5의 mvtec·visa 실제 샘플 8장(256x256, 한 batch) 평가 forward(`pred_score [8]`, `anomaly_map [8,256,256]`, 허용 오차 `atol=1e-6, rtol=1e-5`), decoder 호출 `(z, logdet)`와 `get_logp`와 `positional_encoding_2d`의 구성요소 비교(무작위 입력), FR-24와 PostProcessor 대조다. INFO 1개는 시드 42 초기 `state_dict` 전체의 일치다. 학습 루프는 수동 최적화라 단계 비교를 만들지 않았다. 각 실행은 하나의 체크포인트 상태로 mvtec·visa 두 데이터의 샘플을 비교한다.

`test_dependency.py`는 모든 순수 모듈의 import 분리 검사를 한 파일이 하므로 이 모델의 모듈 검증을 위해 전체를 돌렸다. 이번 Phase에서의 실행은 사용자가 사후에 승인했다(2026-10-27).

실행한 명령은 다음과 같다(Claude가 보낸 지시 원문). agy가 승인 창을 띄운 명령은 창의 명령이 지시한 줄과 정확히 같을 때만 한 번씩 승인했다(영구 허용 없음). 승인 창 없이 실행된 명령(`tests\13_cflow` 1차, `test_dependency.py`, `compare_history.py` 두 개 등)은 승인 창에서 확인하지 못했고, agy의 기록 파일 대조는 권한 문제로 하지 못했다. 로그 파일의 머리글과 출력이 지시한 명령과 일치함은 확인했다.

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py -m pytest tests\13_cflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\13-pytest-261027-1.log
& $py -m pytest tests\common\test_cli.py -k cflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\13-pytest-cli-261027-1.log
& $py -m pytest tests\common\test_dependency.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\13-pytest-dependency-261027-1.log
& $py evaluation/visionlab/train.py train --data mvtec --model cflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\13-native-mvtec-261027-1.log
& $py evaluation/visionlab/predict.py --model cflow --data mvtec 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\13-pure-mvtec-261027-1.log
& $py evaluation/visionlab/train.py train --data visa --model cflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\13-native-visa-261027-1.log
& $py evaluation/visionlab/predict.py --model cflow --data visa 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\13-pure-visa-261027-1.log
& $py evaluation/anomalib/compare_outputs.py --model cflow --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\Cflow\LocalMVTecAD\bottle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\13-compare-outputs-mvtec-261027-1.log
& $py evaluation/anomalib/compare_outputs.py --model cflow --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\Cflow\LocalVisa\candle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\13-compare-outputs-visa-261027-1.log
& $py evaluation/anomalib/compare_history.py --reference D:\projects\visionlab\outputs\anomalib\Cflow\LocalMVTecAD\bottle\v0 --native D:\projects\visionlab\outputs\visionlab\cflow\mvtec_bottle\v0 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\13-history-mvtec-261027-1.log
& $py evaluation/anomalib/compare_history.py --reference D:\projects\visionlab\outputs\anomalib\Cflow\LocalVisa\candle\v0 --native D:\projects\visionlab\outputs\visionlab\cflow\visa_candle\v0 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\13-history-visa-261027-1.log
& $py -m pytest tests\13_cflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\13-pytest-261027-2.log
& $py -m pytest tests\12_fastflow\test_fastflow.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\13-pytest-fastflow-261027-1.log
& $py -m pytest tests\common 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\13-regress-common-261027-1.log
```

agy는 파일을 고치지 않았다.

### 5.3 에폭별 학습 히스토리 비교 (허용 오차 없음, 비교표만)

1 epoch이므로 `compare_history.py`의 `train_loss_epoch`는 한 행뿐이다. 판정에 쓰지 않는다. 값은 fiber batch 조각 손실의 합이라 크다.

| 데이터 / category | Epoch | 레퍼런스 | native | 절대 차이 | 상대 차이 |
|---|---:|---:|---:|---:|---:|
| mvtec / bottle | 0 | 1798.909302 | 1798.90918 | 0.0001220703125 | 6.785795836e-08 |
| visa / candle | 0 | 1185.529297 | 1185.529785 | 0.00048828125 | 4.118677213e-07 |

상대 차이가 4.1e-7 이하다. float32 단위 오차 수준이어서 1 epoch 동안의 학습 경로(수천 번의 수동 optimizer step과 `torch.randperm`)가 레퍼런스와 같다고 볼 수 있다. 에폭 추이는 비교할 수 없다.

### 5.4 crop·입력 크기

이 모델은 crop이 없고 입력 크기는 256x256이다. 특징 지도는 layer2~4의 32x32, 16x16, 8x8이다. 평가 출력 `anomaly_map`은 `[B,256,256]`(3차원), `pred_score`는 `[B]`다. FR-5의 최대 차이는 허용 오차 이내였다(5.2절).

## 6. 결론

- **레퍼런스:** bottle, visa 확보(1 epoch, 원본 `Cflow`).
- **mvtec / bottle:** Image AUROC 차이 6e-8, Pixel AUROC 차이 0(native와 순수 PyTorch). 통과.
- **visa / candle:** Image AUROC 차이 6e-8, Pixel AUROC는 같은 정의(정규화, D-24)로 1.2e-7, raw 값은 2.3e-4. 통과.
- **FR-5, FR-6, FR-23, FR-24:** 통과(5.2절). `train_loss_epoch`는 상대 차이 4.1e-7 이내(5.3절)이다.
- **에폭:** 레퍼런스와 native 모두 실제 1 epoch임을 `run.json`과 로그로 확인했다(2.2절).
- **테스트:** `tests\13_cflow` 29개(재실행), `test_cli.py -k cflow` 4개, `test_dependency.py` 6개가 통과했다. 커밋 전 사용자 승인 사후 실행으로 `tests\12_fastflow\test_fastflow.py` 17개와 `tests\common` 47개도 통과했다(2026-10-27). 01~12 모델별 회귀는 사용자 결정으로 돌리지 않았다.
- **수정 요청:** 13-A 1회, 13-B 0회.
- **`specs/` 변경:** 사용자 결정(2026-10-27)에 따라 `DECISIONS.md`의 D-22에 CFLOW 1 epoch 예외를 더하고 D-15 트리에 cflow 파일을 추가했으며, `SPEC.md` FR-4에 CFLOW 예외를 더했다(6.1절). `PLAN.md`와 INTENT는 고치지 않았다.
- **한계:** 원본이 의도한 에폭에서의 성능은 확인하지 않았다. 1 epoch 결과의 AUROC만 비교했다.
- **Phase 종료:** 사용자가 종료를 결정했다(2026-10-27). `Phase 13: cflow ...` 커밋으로 푸시한다.

### 6.1 `specs/` 갱신 항목

`specs/`는 사람이 쓴다. 최종 보고에 대한 사용자 결정(2026-10-27)에 따라 아래와 같이 처리했다. `PLAN.md`와 INTENT는 고치지 않았다.

- **D-22 1 epoch 예외 (반영 완료):** `DECISIONS.md` D-22의 다섯 항목(선택, 적용 범위, 근거, 배제한 대안, 남은 한계) 끝에 문장을 더했다. 선택은 "원본이 `max_epochs`를 고정하지 않아도 학습 시간이 지나치게 길어 사용자가 Phase 시작 전에 1 epoch로 줄인 모델(CFLOW)은 레퍼런스 yaml과 native yaml의 `trainer.max_epochs: 1`로 에폭을 정하고 1 epoch 결과의 AUROC만 비교한다"이고, 적용 범위는 "사용자가 Phase 시작 전에 정한 모델에 한해 모델 yaml로 에폭을 정한다. 이는 '여러 에폭이 필요한 모델은 모델 yaml로 덮어쓰지 않는다'의 예외이며 일반 규칙이 아니다"다. 근거는 step 수(bottle 약 4,500, candle 약 19,000)와 실제 1 epoch 시간(레퍼런스 bottle 2분 13초, candle 8분 21초)과 10 epoch 환산(측정하지 않음), 배제한 대안은 공유 10 epoch, 남은 한계는 "원본이 의도한 에폭에서의 성능은 확인하지 않았다"다. 제목은 바꾸지 않았다.
- **FR-4 예외 (반영 완료):** `SPEC.md` FR-4에 "CFLOW도 원본의 `.squeeze(1)`를 유지해 anomaly map이 `[B,H,W]`, score가 `[B]`이며, 두 값은 batch 전체의 최댓값으로 정규화되므로 같은 이미지라도 batch 구성에 따라 값이 달라진다"를 DRAEM 문장 뒤에 더하고 판정 방법 문구를 "CFA, SuperSimpleNet, DRAEM, CFLOW는 원본 동작의 shape를 확인"으로 맞췄다.
- **D-15 트리 (반영 완료):** `tests/13_cflow/`(두 파일), `configs/models`의 `cflow.yaml`, 모델 패키지 `models/cflow/`(다섯 파일)을 추가했다.

### 6.2 남은 위험과 사용자 판단 대상

- **원본이 의도한 에폭에서의 성능은 확인하지 않았다.** 1 epoch 학습 결과의 AUROC만 레퍼런스와 비교했다(사용자 결정). 10 epoch 이상에서의 차이는 알 수 없다.
- **학습량이 적어 AUROC가 우연의 영향을 더 받을 수 있다.** 다만 레퍼런스와 native가 AUROC, 손실, 구성요소 출력까지 일치했고 값 자체는 높다(bottle Image 1.0, visa Pixel 0.99).
- **에폭별 히스토리 비교가 1행뿐이다.** 추이를 확인하지 못했다. 검증력은 FR-5 구성요소 일치와 AUROC에 의존한다. 학습 루프(수동 최적화)는 단계별 비교를 만들지 않았다.
- **평가 지도가 batch에 의존한다(사용 시 주의).** `Predictor`나 `predict.py`에서 batch를 다르게 주면 같은 이미지의 `anomaly_map`, `pred_score`, AUROC가 달라질 수 있다. 레퍼런스와 같은 batch 8·같은 순서의 평가에서만 값이 일치한다. 한 장씩 추론하면 batch 하나에서 정규화되어 점수 분포가 달라진다. 원본 동작이라 고치지 않았다.
- **레퍼런스가 category당 1회뿐이다.** 같은 seed에서 일치가 매우 정확한 것은 seed 42, 난수 소비 순서, 같은 GPU와 연산 때문이다. 다른 환경에서도 같다고 보장하지 않는다.
- **`permute_soft=True`는 검증하지 않았다**(scipy를 쓰고 느리다). 기본값(wide_resnet50_2, layer2~4)만 실제로 학습·평가했고 다른 백본(`resnet18`, `resnet50`)은 구조 생성만 확인했다.
- **배포 환경 요건:** 순수 `CflowModel`과 cflow export 로딩에는 FrEIA 0.2와 numpy, scipy가 필요하다(D-26). 다른 모델의 `Predictor`는 FrEIA 없이 import되도록 `cflow` 분기 안에서만 지연 import한다.
- **체크포인트와 export가 크다.** `model.ckpt`가 약 1.6GB, `export/model.pt`가 약 947MB다(encoder 가중치 포함). `outputs/`는 커밋하지 않는다.
- **`test_cflow_lightning.py` 본문:** 줄 단위로 읽지 않았다(실행 결과로 판정).
- **일부 agy 명령의 승인 대조 누락:** 5.2절에 적었다.
- **Phase 12 테스트:** `tests/12_fastflow/test_fastflow.py`의 허용 목록 한 항목 수정을 사용자가 승인한 사후 실행으로 확인했다(17 passed, 5.2절).
- `tests\common\test_dependency.py` 전체 실행은 사용자가 사후에 승인했다(2026-10-27).
- **앞 모델 회귀(01~12) 보류(사용자 결정 2026-10-27):** 돌리지 않았다. 변경은 모두 기존 항목을 건드리지 않는 등록과 분기 추가이고, `predictor.py`의 fastflow 분기는 건드리지 않았다. 사용자가 승인한 사후 `tests\common` 1회 실행은 47개가 통과했다(5.2절).

| 공유 파일 | 변경 | 영향받을 수 있는 앞 모델 |
|---|---|---|
| `src/visionlab/utils/io.py` | `load_export`가 허용하는 이름에 `cflow` 추가 | 모든 모델의 export 로딩 |
| `src/visionlab/inference/predictor.py` | `from_export`에 `cflow` 분기 추가(분기 안 지연 import, fastflow 분기는 그대로) | 모든 모델의 `Predictor.from_export` |
| `evaluation/visionlab/predict.py` | `--model` 선택지에 추가 | 모든 모델의 순수 PyTorch 평가 |
| `evaluation/anomalib/compare_outputs.py` | `--model cflow` 분기와 검증 함수 | 앞 모델 전부의 비교 |
| `evaluation/anomalib/default.yaml` | 맨 위 주석만 수정(값은 그대로) | 없음 |
| `tests/common/test_cli.py`, `tests/common/test_dependency.py` | cflow 테스트와 분류 추가 | 같은 파일의 앞 모델 테스트 |
| `tests/12_fastflow/test_fastflow.py` | FrEIA import 허용 목록에 `cflow/utils.py` 추가(사후 실행 17개 통과) | FastFlow 테스트 |
