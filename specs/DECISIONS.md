> 버전: v0.1 · 작성일: 2026-10-22

# DECISIONS

## D-1. 패키지 이름과 import 경로

- **선택**: 패키지 이름은 `visionlab`이다. 소스는 `src/visionlab/<task>`에 두고 내부 import는 `visionlab.*`를 사용한다. `src`는 import 이름에 포함하지 않는다.
- **근거**: lightning-tutorial과 구분되는 새 패키지로 시작한다. 기존 `src.anomaly_detection.*`를 계속 쓰면 패키지 이름을 도입하는 의미가 약해진다.
- **배제한 대안**: 기존 `src.anomaly_detection.*` import 경로 유지.

## D-2. 첫 대상 task·모델·데이터

- **선택**: 첫 task는 `anomaly_detection`이다. 평가 데이터는 MVTec AD `bottle`, VisA `candle`이다. 이식 대상은 `evaluation/anomalib/models/*.yaml`의 anomalib image 모델 18개이며, STFPM, EfficientAD를 먼저 이식하고 나머지는 PLAN의 순서로 한 Phase에 하나씩 추가한다.
- **근거**: 기존 구현이 있는 두 모델로 구조·입출력·검증 절차를 먼저 확인한다. 이후 순서는 공통 부품 재사용과 검증 변수를 점진적으로 늘리는 기준(통계형, 역전파형, 합성 이상형, flow, 특수 학습)으로 정했다. category를 데이터셋마다 하나로 줄여 모델 수가 늘어난 비교 비용을 줄인다.
- **배제한 대안**: 전체 image 모델을 한 번에 복사. 분류용·미구현 task 폴더를 미리 추가.

## D-3. 이식·비교 기준 버전

- **선택**: 이식과 성능 비교의 기준은 설치된 anomalib 2.0.0이다. 모델별 실제 이식 원본은 작업 시작 시 버전·파일 경로를 고정해 기록한다.
- **근거**: 생성자 인자와 비교 실행을 같은 소스에 맞춘다. 기존 이식 참조 커밋 `baf5e5fe12bc73cd520523046ba1d68279462e69`와 2.0.0을 같다고 간주하지 않는다. 최신 2.7 계열은 PyTorch 2.10 이상을 요구한다.
- **배제한 대안**: 기존 이식 참조 커밋을 기준으로 유지. 최신 main 또는 2.7 계열로 조용히 교체.

## D-4. 순수 모델과 native Lightning의 분리

- **선택**: 모델별 `model.py`는 순수 `nn.Module`, `lightning_model.py`는 native `LightningModule`이다. 데이터 연결은 native `LightningDataModule`, 실행은 native `Trainer`·`LightningCLI`를 사용한다.
- **근거**: 배포되는 모델은 순수 PyTorch만으로 동작해야 한다(INTENT). Wrapper는 Hook·optimizer·metric 상태만 연결하고 알고리즘을 중복 작성하지 않는다.
- **배제한 대안**: anomalib을 상속한 Wrapper. anomalib Engine을 얇게 감싸는 방식. 별도 학습 루프·Engine.

## D-5. anomalib과 동일한 생성자

- **선택**: `XXXLitModel`·`XXXLitData`의 생성자 인자 이름·순서·기본값을 대응하는 anomalib 2.0.0 클래스와 동일하게 맞춘다. PreProcessor·PostProcessor·Evaluator·Visualizer 인자는 원본처럼 `True`·`False`·visionlab 구성요소 객체를 받는다. Enum과 구성요소 타입은 visionlab에서 정의한다.
- **근거**: anomalib 사용자가 같은 방식으로 쓸 수 있어야 한다(INTENT).
- **배제한 대안**: 설정값만 받고 객체 주입을 금지하는 방식(lightning-tutorial v0.3 기획 D05). 원본에 없는 `weights_path`·`teacher_path`·`tiler`·`lr`·`image_size`·`valid_ratio` 인자 추가. anomalib 객체·Enum을 입력으로 받는 호환 계층.

## D-6. Dataset과 PreProcessor의 전처리 분담

- **선택**: Dataset Transform은 RGB 변환·이미지와 mask의 공간 변환·float 변환을 맡는다. PreProcessor는 Wrapper에서 입력 확인과 모델별 정규화를 맡는다. STFPM은 ImageNet 정규화를 1회 적용하고, EfficientAD는 외부 정규화를 하지 않는다. PreProcessor는 선택 인자 `center_crop`(기본 `None`)으로 중앙 crop도 할 수 있다. PatchCore 레퍼런스의 Resize와 CenterCrop을 재현하기 위한 것이며(Phase 6), 값을 주지 않는 기존 모델의 동작은 바뀌지 않는다.
- **근거**: 서로 다른 원본 해상도를 collate 전에 같은 크기로 맞추고, 모델별 정규화는 모델 호출 경로에서 통제한다. 독립 추론도 같은 두 단계를 조합해 이중 resize·정규화를 막는다.
- **배제한 대안**: lightning-tutorial v0.2의 Dataset 내 정규화. anomalib처럼 PreProcessor가 resize·정규화를 함께 하고 Callback을 상속하는 방식.

## D-7. 네 처리 구성요소를 순수 기능으로 구성

- **선택**: PreProcessor·PostProcessor·Evaluator·ImageVisualizer는 Lightning 없이 쓰는 순수 기능으로 `anomaly_detection/common`에 둔다. Wrapper가 전처리·모델·후처리·평가 순서를 제어하고, 시각화 파일 저장만 native Callback(`common/lightning_visualizer.py`)이 맡는다.
- **근거**: 구성요소 간 순서를 Callback 등록 순서에 의존시키지 않는다. 독립 추론과 순수 PyTorch 평가에서 같은 구현을 재사용한다.
- **배제한 대안**: anomalib의 `nn.Module`·`Callback` 다중 상속 구조. 네 개의 독립 처리 Callback.

## D-8. 후처리 보정과 평가 상태의 분리

- **선택**: PostProcessor의 image/pixel min·max와 F1 기반 임계값은 fit validation 전체에서 수집해 epoch 종료에 확정한다. standalone validate·test·predict는 저장 상태를 읽기만 한다. 정규화는 임계값 중심 변환 `clamp((x-threshold)/(max-min)+0.5, 0, 1)`을 사용한다. raw 예측은 보존하고 후처리 결과를 별도로 둔다. validation은 raw AUROC, test는 raw AUROC와 후처리 F1을 기록한다. 여기서 raw AUROC는 점수의 순서를 보존하는 계산이다(D-20).
- **근거**: 같은 validation에서 고른 임계값의 F1을 독립 평가 결과로 내지 않는다. 테스트 정답으로 보정하지 않는다.
- **배제한 대안**: 단순 `(score-min)/(max-min)` 정규화. 테스트 배치나 배치별 범위로 재보정. anomalib 2.0.0 EfficientAD처럼 `on_validation_start`에서 무조건 quantile을 갱신하는 방식.

## D-9. 시각화 기본 활성

- **선택**: `visualizer=True`를 기본값으로 한다. 순수 ImageVisualizer가 이미지를 만들고 native Callback이 test·predict 결과를 실행 폴더 아래 `images`에 저장한다.
- **근거**: 생성자를 anomalib 2.0.0과 동일하게 맞춘다(D-5).
- **배제한 대안**: 시각화 기본 비활성(lightning-tutorial v0.3 기획 D11·D14).

## D-10. DataModule 기본 분할

- **선택**: `XXXLitData`의 분할 기본값은 anomalib 2.0.0과 같은 `test_split_mode="from_dir"`, `val_split_mode="same_as_test"`, `seed=None`이다. 분리된 valid/test가 필요하면 `val_split_mode="from_test"`와 `val_split_ratio`, `seed`를 명시한다. 분할 Enum은 `data/common/split.py`에 순수 Python으로 정의한다.
- **근거**: 생성자 인자와 기본 동작을 anomalib과 맞춘다(D-5).
- **배제한 대안**: 기존 `valid_ratio=0.4`·seed 42·양쪽 라벨 확보 100회 재시도를 기본 정책으로 유지. `image_size`·`valid_ratio` 생성자 인자.

## D-11. 모델 공유 연산의 배치

- **선택**: 모델 내부 공유 연산은 `models/components/<그룹>`에 둔다. 초기 그룹은 `feature_extractors`·`tiling`이다. 출력 형식·metric은 task `common`에 둔다. 별도 `models/common`은 두지 않는다.
- **근거**: 역할별 하위 그룹으로 나눠 위치를 명확히 한다. anomalib 2.0.0의 `models/components` 구조와 대응한다.
- **배제한 대안**: 평면적인 `models/common`에 계속 추가. 광범위한 utils 파일. 실제 필요 전에 빈 그룹 폴더 생성. 공유 Lightning 기반 클래스를 미리 추가.

## D-12. 로컬 경로의 단일 해석

- **선택**: 데이터셋과 가중치 파일의 경로는 실행 설정(yaml)에 절대경로로 직접 적는다(예: `data.init_args.root`, `model.init_args.weights_file`·`teacher_file`). 그 값을 Dataset·특징 추출기·teacher 로더까지 그대로 전달한다. 루트 `paths.toml`에는 `[backbones]`와 `[teachers]`를 둔다. `[backbones]`의 키는 timm 모델 이름이고 값은 로컬 가중치 파일이다. `[teachers]`는 timm 모델이 아닌 pretrained teacher(EfficientAD)를 위한 것으로, 키는 `<model>_<model_size>`(예: `efficient_ad_small`)이고 값은 로컬 teacher 가중치 파일이다(Phase 2에서 추가). timm 백본은 이름으로 조회되므로 실행 시작 시 `[backbones]` 매핑으로 timm의 같은 이름을 로컬 가중치 로더로 바꾼다.
- **근거**: CLI가 저장하는 `config.yaml`에 실제 사용한 경로가 남아 별도 실행 기록 파일 없이도 재현할 수 있다. yaml·CLI의 override가 생성자에 그대로 전달된다. 기존 보조 로더가 내부에서 기본 `paths.toml`을 다시 읽어 CLI 선택과 달라질 수 있었다.
- **배제한 대안**: `paths.toml`을 모든 경로의 단일 원본으로 두고 `Local*` 생성자가 생략된 인자를 채우는 방식(저장된 config에 실제 경로가 남지 않는다). 로더 내부에서 기본 `paths.toml` 재읽기. `local_<name>` 별칭으로 timm에 등록하는 방식(anomalib의 backbone 이름 검사에 걸린다).

## D-13. 데이터 비교를 모델 비교의 선행 조건으로

- **선택**: 모델 성능을 비교하기 전에 Dataset·DataModule의 파일 목록·라벨·mask·split·최종 모델 입력을 비교한다. 통제 비교에서는 실제 파일 목록을 manifest로 고정하고, 데이터 검증 ID를 모델 결과에 연결한다.
- **근거**: anomalib 기본 분할과 기존 분할이 다르고, anomalib PreProcessor가 입력을 다시 바꿀 수 있다.
- **배제한 대안**: 파일 수·seed·분할 비율이 같다는 이유로 동일 Dataset으로 판정. 양쪽에 같은 Dataset을 주입한 결과만으로 원래 구현이 같다고 판정.

## D-14. 개발 환경과 자산 위치

- **선택**: 개발 환경은 `C:\winpython\WPy64-31180_gpu`의 Python 3.11.8, torch 2.5.1+cu121, torchvision 0.20.1, numpy 2.2.5, Lightning 2.5.5, anomalib 2.0.0(openvino 2025.4.1, jsonargparse 4.39.0), torchmetrics 1.8.2, pytest 9.1.1이다. 데이터는 `D:\datasets`, 가중치는 `D:\backbones`를 사용한다. 설치 내역은 `docs/anomalib/anomalib-install-guide.md`를 따른다.
- **근거**: 기존 PyTorch·CUDA를 유지한 채 설치했고 기존 패키지 변경은 0건이다. openvino 2026 이상은 `openvino.runtime`이 없고, jsonargparse 4.40 이상은 anomalib 2.0.0 CLI가 실패한다.
- **배제한 대안**: 원 문서 환경(torch 2.2.1+cu118, numpy 1.26.4)으로 다운그레이드. `E:\datasets`·`E:\backbones` 경로.

## D-15. 코드 배치

- **선택**: 목표 구조는 아래와 같다. 각 Phase는 자기 범위의 파일만 만들고, 아직 필요 없는 폴더·파일은 만들지 않는다.

```text
visionlab/                                      # project root
├── paths.toml                                  # [backbones] timm name -> local weights, [teachers] <model>_<size> -> teacher file (D-12)
├── docs/
│   ├── anomalib/                               # install guide
│   ├── results/                                # one file per Phase: NN-<kebab-case>.md, rules in README.md
│   └── workflow/
│       └── model-port.md                       # per-Phase procedure and agent instructions
├── evaluation/                                 # evaluation code (reference runs, comparisons, ported model evaluation)
│   ├── anomalib/                               # imports anomalib
│   │   ├── data/                               # shared data yaml (mvtec.yaml, visa.yaml)
│   │   ├── models/                             # shared model yaml (NN_<name>.yaml), per-Phase reference
│   │   ├── anomalib_run.py                     # AnomalibCLI entry point, reference runs
│   │   ├── compare_data.py                     # visionlab data vs anomalib data, --data (FR-17, D-13)
│   │   ├── compare_history.py                  # epoch-wise train_loss_epoch vs reference, --reference --native
│   │   ├── compare_outputs.py                  # raw output vs anomalib, --model (FR-5, diagnosis)
│   │   ├── default.yaml                        # shared seed, output root, trainer
│   │   ├── local_data.py                       # LocalMVTecAD, LocalVisa
│   │   ├── local_models.py                     # Local* models that load local assets
│   │   └── local_setup.py                      # offline setup, timm local weights, run folder patch
│   └── visionlab/                              # no anomalib import
│       ├── predict.py                          # pure PyTorch inference and AUROC, --model
│       └── train.py                            # native Lightning train and test, LitCLI
├── notebooks/                                  # Jupyter notebooks
├── outputs/                                    # run outputs, excluded from git
├── specs/                                      # INTENT, DECISIONS, SPEC, PLAN
├── tests/                                      # visionlab pytest tests (FR·NFR checks), run with pytest (D-21)
│   ├── conftest.py                             # adds <root>/src to sys.path
│   ├── 00_dataset/
│   │   └── test_data.py                        # FR-1~3
│   ├── 01_stfpm/
│   │   ├── test_components.py                  # FR-7~11, common parts first built in Phase 1
│   │   ├── test_lightning.py                   # FR-6, FR-12
│   │   ├── test_model.py                       # FR-4, FR-16, NFR-5
│   │   └── test_predictor.py                   # FR-14
│   ├── 02_efficient_ad/
│   │   └── test_efficient_ad.py                # FR-15
│   └── common/                                 # tests that span Phases
│       ├── test_cli.py                         # FR-13
│       └── test_dependency.py                  # NFR-1, NFR-2, run for every Phase
└── src/
    └── visionlab/
        ├── anomaly_detection/
        │   ├── configs/
        │   │   ├── default.yaml                # common seed
        │   │   ├── trainer.yaml                # Trainer, logger, callbacks
        │   │   ├── data/                       # mvtec.yaml, visa.yaml
        │   │   └── models/                     # stfpm.yaml, efficient_ad.yaml
        │   ├── data/
        │   │   ├── common/
        │   │   │   └── split.py                # TestSplitMode, ValSplitMode (D-10)
        │   │   ├── mvtec/
        │   │   │   ├── dataset.py              # MVTecDataset, MVTecTransform
        │   │   │   └── lightning_data.py       # MVTecLitData
        │   │   └── visa/
        │   │       ├── dataset.py              # VisADataset, VisATransform
        │   │       └── lightning_data.py       # VisALitData
        │   ├── models/
        │   │   ├── components/                 # copied from anomalib models/components, only what is used (D-11, D-18)
        │   │   │   ├── feature_extractors/     # copied: TimmFeatureExtractor, feature shape helpers
        │   │   │   └── tiling/
        │   │   │       └── tiler.py            # Tiler, new (replaces anomalib.data.utils.tiler)
        │   │   ├── efficient_ad/               # copied from anomalib models/image/efficient_ad (D-18)
        │   │   │   ├── __init__.py
        │   │   │   ├── lightning_model.py      # EfficientAdLitModel, native LightningModule
        │   │   │   └── model.py                # EfficientAdModel, EfficientAdModelSize (was torch_model.py)
        │   │   └── stfpm/                      # copied from anomalib models/image/stfpm (D-18)
        │   │       ├── __init__.py
        │   │       ├── anomaly_map.py          # AnomalyMapGenerator
        │   │       ├── lightning_model.py      # StfpmLitModel, native LightningModule
        │   │       ├── loss.py                 # STFPMLoss
        │   │       └── model.py                # STFPMModel (was torch_model.py)
        │   ├── common/                         # new, replaces anomalib parts outside models (D-7, D-18)
        │   │   ├── batch.py                    # Batch, InferenceBatch (anomalib.data), Phase 0
        │   │   ├── evaluator.py                # Evaluator (anomalib.metrics)
        │   │   ├── learning_type.py            # LearningType (anomalib)
        │   │   ├── lightning_visualizer.py     # native file-saving Callback
        │   │   ├── metric.py                   # AUROC, F1
        │   │   ├── post_processor.py           # PostProcessor (anomalib.post_processing)
        │   │   ├── pre_processor.py            # PreProcessor (anomalib.pre_processing)
        │   │   └── visualizer.py               # ImageVisualizer
        │   └── trainers/
        │       ├── lightning_trainer.py
        │       └── lightning_cli.py
        ├── inference/
        │   └── predictor.py                    # pure PyTorch Predictor
        └── utils/
            ├── config.py
            └── io.py
```

| 위치 | Lightning | anomalib |
|---|---|---|
| `data/*/dataset.py`, `data/common/`, `models/*/model.py`, `models/*/loss.py`, `models/*/anomaly_map.py` 등 순수 연산 파일, `models/components/`, `common/`(lightning_visualizer 제외), `inference/`, `utils/` | import하지 않음 | import하지 않음 |
| `data/*/lightning_data.py`, `models/*/lightning_model.py`, `common/lightning_visualizer.py`, `trainers/` | native API만 사용 | import하지 않음 |

패키지 `__init__.py`는 Lightning 연결 파일을 자동 import하지 않는다. 원본에서 복사한 모델 패키지의 `__init__.py`도 예외로 고친다(D-18). 모델에 필요 없는 `loss.py`·`metric.py`를 관례만으로 만들지 않는다.

- **근거**: D-1·D-7·D-11에 흩어진 배치를 한 트리로 묶어, Phase마다 무엇을 어디에 만드는지 PLAN이 참조할 수 있게 한다. 파일 위치만으로 Lightning·anomalib 의존 여부가 드러나게 한다.
- **배제한 대안**: `models/common`(D-11). 프로젝트 루트 바로 아래의 Python 파일(평가 스크립트는 `evaluation/`에 둔다). 모델마다 `NN_train_<model>.py`처럼 Phase별 실행 스크립트를 따로 두는 방식(18개 모델이면 36개가 되고 같은 절차가 반복된다). 평가 코드를 `examples/`에 두는 방식(예제와 정식 평가 자료가 섞인다). 평가 코드를 `tests/`에 두는 방식(장시간 학습과 단위 테스트가 섞이고 tests의 anomalib 미사용 원칙과 충돌한다). 테스트 파일 이름에 Phase 번호를 접두사로 붙이는 방식(`test_01_stfpm_model.py`, 한 폴더에 18 Phase x 여러 파일이 섞인다). `src`의 모델 패키지에 번호를 붙이는 방식(import 이름이 식별자여야 하고 구현 순서가 바뀌면 코드 이름이 바뀐다). 모든 task 폴더와 하위 폴더를 미리 생성. 순수 파일과 Lightning 연결을 한 파일에 섞는 방식. `pyproject.toml`·`pip install -e .` 개발 설치. 실행 전 `PYTHONPATH` 환경변수 지정. `-m` 모듈 실행.
`outputs/`는 git에서 제외한다(`.gitignore`). anomalib을 import하는 비교·테스트 코드는 `evaluation/anomalib/`에만 두고 `src/visionlab`에서 참조하지 않는다(D-3·INTENT). `evaluation/anomalib/`에는 레퍼런스 실행 진입점(`anomalib_run.py`는 `AnomalibCLI`), 공유 yaml, 공통 설정 `local_setup.py`(오프라인 설정, timm 이름의 로컬 가중치 연결, Windows 실행 폴더 처리), `local_data.py`의 `LocalMVTecAD`·`LocalVisa`(D-17), `local_models.py`의 로컬 자산 모델, visionlab 데이터·출력을 anomalib과 대조하는 비교 스크립트(FR-17, FR-5)를 둔다.

패키징 설정(`pyproject.toml` 등)과 `pip install -e .` 개발 설치는 사용하지 않는다. 실행 스크립트는 PowerShell에서 WinPython의 전체 경로로 실행하고, 테스트는 같은 파이썬의 pytest로 실행한다(D-21).

```powershell
C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe <file>.py --<option>
C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe -m pytest tests\<NN_model>\<file>.py
```

`evaluation/`의 스크립트는 첫머리에서 `<root>/src`를 `sys.path`에 추가한 뒤 `visionlab`을 import한다. `tests/`는 `tests/conftest.py`가 같은 일을 하므로 테스트 파일에는 `sys.path` 설정을 두지 않는다. visionlab 자체 테스트는 루트 `tests/`에 둔다. `tests/`는 anomalib을 import하지 않는다.

스크립트 이름 규칙은 다음과 같다. Phase 하나에만 쓰는 일회성 스크립트는 `NN_<단계>.py`이고 `NN`은 PLAN의 Phase 번호다(현재 해당 파일은 없다. 데이터 대조는 이후 Phase에서도 다시 쓰므로 `compare_data.py`로 번호 없이 둔다, 사용자 결정 2026-10-25). 여러 Phase에서 쓰는 공용 진입점은 번호 없이 `anomalib_run.py`, `compare_data.py`, `compare_history.py`, `compare_outputs.py`, `train.py`, `predict.py`로 두고 모델은 yaml 또는 `--model` 인자로 고른다. 공용 모듈은 `_` 접두사를 붙이되, `evaluation/anomalib/`의 공용 모듈은 yaml의 `class_path`로 import되므로 `local_` 접두사를 쓴다. 모델별 파일이 Phase마다 늘어나는 곳은 Phase 번호(두 자리)를 붙인다. `tests/`는 `tests/NN_<모델 폴더 이름>/test_*.py`이고 여러 Phase에 걸친 테스트는 `tests/common/`에 둔다. `evaluation/anomalib/models/`는 `NN_<모델 폴더 이름>.yaml`이다. 번호는 PLAN의 대응 표가 유일한 원본이다. `src/visionlab/.../models/<model>/`(Python 패키지 이름)과 `outputs/`(anomalib 출력 규칙)에는 번호를 붙이지 않는다. `tests/`는 `evaluation/`과 달리 anomalib을 import하지 않는다. anomalib을 import하는 스크립트만 `evaluation/anomalib/`에 두고, 나머지 평가 스크립트는 `evaluation/visionlab/`에 둔다. CLI는 `-m` 실행 대신 `evaluation/visionlab/train.py`가 `sys.path`를 설정한 뒤 `trainers/lightning_cli.py`의 `LitCLI`를 호출한다. 실행 결과는 `outputs/`에 저장한다.

## D-16. Phase 순서: 모델별 레퍼런스 확보, 이식, 비교

- **선택**: Phase 0은 모든 모델이 공유하는 데이터셋을 새로 구현하고 anomalib 데이터와 대조한다(D-13). Phase 1부터는 Phase마다 모델 하나를 다룬다. 각 모델 Phase는 그 모델의 레퍼런스 수치를 anomalib 그대로(로컬 데이터·로컬 backbone·anomalib model + Engine, `anomalib_run.py`) 먼저 확보하고, 원본 복사 후 anomalib 의존성을 제거하는 방식(D-18)으로 이식을 끝까지 진행한 뒤, native Lightning 결과와 순수 PyTorch 추론 결과를 각각 레퍼런스와 비교한다. 데이터는 모델 비교 전에 anomalib과 대조한다(D-13).
- **근거**: 원본을 복사해 의존성만 바꾸므로 이식본과 원본을 파일 단위로 대조할 수 있고, 수치가 어긋나면 raw 출력 비교(FR-5)로 원인을 좁힌다. 공통 부품은 Phase 1에서 만들고 이후 Phase는 모델 고유 부분만 추가한다.
- **배제한 대안**: anomalib Engine에 visionlab 부품을 adapter로 하나씩 끼워 단계마다 비교하는 방식(복사 이식에서는 중간 단계의 이득이 작고 adapter 코드가 늘어난다). 여러 모델을 한 Phase에서 동시에 진행하는 방식. `lightning_run.py`(Engine 없는 LightningCLI)로 레퍼런스를 재현하는 별도 단계. 18개 모델의 레퍼런스를 별도 Phase에서 한꺼번에 먼저 확보하는 방식(이식하지 않을 모델까지 실행 비용이 들고, 레퍼런스와 이식의 시점이 벌어진다).

## D-17. VisA anomalib 작업 복사본

- **선택**: `evaluation/anomalib/`의 `LocalVisa`는 anomalib `Visa`를 상속해 `split_root`만 원본 밖 `D:\datasets\visa_anomalib\visa_pytorch`로 바꾸고, anomalib 원래 `apply_cls1_split`을 그대로 실행한다. `root`는 원본 `D:\datasets\visa`이며 읽기만 한다.
- **근거**: anomalib 2.0.0 `Visa.prepare_data()`는 `root/visa_pytorch`에 이미지를 복사하고 mask를 이진화한다. 원본을 바꾸지 않으면서 레퍼런스가 anomalib 기본 처리를 그대로 따르게 한다. lightning-tutorial도 원본 폴더에서 `visa_pytorch` 준비를 실행하지 않기로 했다.
- **배제한 대안**: 원본 `D:\datasets\visa` 안에 `visa_pytorch` 생성. 복사 없이 `split_csv/1cls.csv`를 직접 읽는 연결 코드(레퍼런스가 순수 anomalib 동작이 아니게 됨).

## D-18. 이식 방식: 원본 복사 후 anomalib 의존성 제거

- **선택**: 모델마다 anomalib 2.0.0의 `models/image/<model>/` 파일을 `src/visionlab/anomaly_detection/models/<model>/`로 폴더 이름과 파일 구성 그대로 복사한다. `torch_model.py`만 `model.py`로 이름을 바꾸고 import를 맞춘다. 모델이 쓰는 `models/components/`의 공통 요소는 필요한 것만 같은 상대 위치로 복사한다. 복사본에서 anomalib import를 모두 제거하고, `lightning_model.py`는 native `LightningModule`로 바꾼다. 알고리즘(연산, loss, anomaly map, 학습 hook의 동작)은 바꾸지 않는다.
- **데이터**: 데이터셋(`anomalib.data`의 datamodules·datasets)은 복사 대상이 아니며 Phase 0에서 새로 구현한다.
- **anomalib 의존성 대체**: models 밖의 anomalib 부품(`anomalib.data`의 Batch·InferenceBatch·Tiler, `pre_processing`, `post_processing`, `metrics`, `visualization`, `AnomalibModule`, `LearningType` 등)은 복사하지 않고 visionlab에서 필요한 것만 새로 구현한다(D-7, D-11).
- **클래스 이름**: 순수 모델 클래스는 원본 이름을 유지한다(예: `STFPMModel`, `EfficientAdModel`). LightningModule은 원본 이름에 `LitModel`을 붙인다(예: `Stfpm` -> `StfpmLitModel`, `EfficientAd` -> `EfficientAdLitModel`). 생성자 인자의 이름·순서·기본값은 원본과 같다(D-5).
- **코드 표기**: 복사한 원본 코드는 원본 표기(type hint, docstring, 저작권·SPDX 헤더 등)를 유지한다(NFR-5). AGENTS.md의 Python 코드 규칙은 visionlab에서 새로 작성하는 코드(의존성 대체 부품, 실행·비교 스크립트, 테스트)에만 적용한다. 복사본에서 의존성 제거로 바뀐 줄은 주변 원본 표기에 맞춘다.
- **예외**: 원본 모델 패키지의 `__init__.py`는 `lightning_model`을 import한다(예: `from .lightning_model import Stfpm`). 그대로 두면 순수 모델(`model.py`)을 import할 때도 Lightning이 필요해 순수 PyTorch 배포(INTENT, NFR-2)가 깨지므로, `__init__.py`는 Lightning 모델을 import하지 않도록 고친다(순수 모델만 노출하거나 비운다).
- **근거**: 원본과 이식본을 파일 단위로 나란히 대조할 수 있어, 적대적 검증과 수치 불일치의 원인 추적이 쉽다. 알고리즘을 다시 쓰면서 생기는 차이를 막는다.
- **배제한 대안**: 모델을 visionlab 스타일로 새로 작성하는 방식. `components/` 전체를 한 번에 복사하는 방식. models 밖의 anomalib 부품까지 복사해 의존성을 걷어 내는 방식.

## D-19. pretrained 가중치는 LitModel이 주입한다

- **선택**: 순수 모델(`model.py`)의 생성자는 pretrained 가중치 파일을 읽거나 내려받지 않는다. 생성 시 pretrained로 만들던 부분(STFPM의 teacher 등)은 `pre_trained=False`로 구조만 만든다. native `XXXLitModel.__init__`이 생성자 인자를 늘리지 않고 루트 `paths.toml`의 `[backbones]`에 적힌 로컬 파일을 그 부분에 불러온다. 파일이 없으면 무작위 대체 없이 `FileNotFoundError`를 낸다(NFR-3). 학습하지 않는 teacher도 `state_dict`에 포함되므로 배포 artifact와 checkpoint에 들어간다.
- **적용 범위**: Phase 1 STFPM부터 적용한다. 다른 모델은 그 Phase의 모델 노트에서 생성자가 pretrained를 읽는지 확인하고, 읽으면 같은 방식을 쓴다. EfficientAD teacher처럼 `paths.toml`의 timm 이름 키로 표현되지 않는 자산은 그 Phase에서 주입 경로를 따로 정한다. EfficientAD(Phase 2)는 `paths.toml [teachers]`의 `efficient_ad_<model_size>` 파일을 `EfficientAdLitModel.__init__`이 `strict`로 불러온다(D-12).
- **D-12와의 관계**: `paths.toml`의 `[backbones]` 형식(timm 모델 이름 키, 로컬 가중치 파일 값)은 그대로 쓴다. 실행 시작 시 timm 이름을 로컬 로더로 바꾸는 방식은 `evaluation/anomalib/local_setup.py`(레퍼런스 실행)에만 남기고, visionlab은 쓰지 않는다.
- **D-5, D-18과의 관계**: `XXXLitModel` 생성자 인자의 이름·순서·기본값은 anomalib 2.0.0과 같다(D-5). 원본 알고리즘(연산, loss, anomaly map, 학습 hook)은 바꾸지 않으며, 바뀌는 것은 pretrained 로딩 위치뿐이다. D-18의 "의존성 제거 외 변경 없음"에 대한 의도된 예외이며, 해당 줄은 이식 파일 주석과 결과 파일의 "원본과 다르게 구현한 곳"에 적는다(NFR-5).
- **근거**: 배포 머신에는 학습된 checkpoint만 있고 `D:\backbones`가 없다. 생성자가 pretrained를 읽으면 순수 PyTorch 배포가 불가능하다(INTENT). 생성 후 주입하면 checkpoint 평가·재개에서 불필요한 pretrained 읽기도 없어진다. 주입은 한 줄이고 `state_dict`로 덮어쓰인 결과는 같다.
- **배제한 대안**: `weights_file` 같은 인자를 LitModel 생성자에 추가하는 방식(D-5, FR-6 위반). timm 이름을 실행 시작 시 교체하는 방식(순수 모델 생성자가 파일을 읽어 배포 요구와 충돌). pretrained 파일이 있으면 읽고 없으면 건너뛰는 방식(실행 결과가 파일 유무로 조용히 바뀐다).

## D-20. 평가 metric은 torchmetrics를 사용한다

- **선택**: `common/metric.py`와 PostProcessor의 F1 임계값 계산은 anomalib 2.0.0과 같은 torchmetrics 클래스를 사용한다. AUROC는 `BinaryROC`와 `torchmetrics.utilities.compute.auc`, F1은 `BinaryF1Score`, F1 기반 임계값은 `BinaryPrecisionRecallCurve`다(anomalib `metrics/auroc.py`, `f1_score.py`, `threshold/f1_adaptive_threshold.py`를 읽고 같은 호출을 쓴다). 계산 불가 시 사유를 남기는 동작(FR-10)과 valid·test 상태 분리는 visionlab 객체가 맡고, 계산 자체는 torchmetrics에 맡긴다.
- **적용 범위**: Phase 1 STFPM부터 모든 Phase에 적용한다. 순수 파일(`common/metric.py`)이 torchmetrics를 import하지만 torchmetrics는 Lightning을 import하지 않으므로 NFR-2는 유지된다. torchmetrics는 `src/visionlab`의 허용된 외부 의존이다. 다른 metric 라이브러리(scikit-learn 등)는 쓰지 않는다.
- **AUROC 입력**: raw 점수를 `BinaryROC`에 그대로 넣지 않는다. 값이 [0, 1]을 벗어나면 `BinaryROC`가 sigmoid를 적용해 큰 점수가 포화되고 동점이 생기므로, 점수를 순서(rank)로 바꿔 [0, 1]에 놓은 뒤 넣는다. 동점은 그대로 유지한다. 이렇게 하면 D-8의 raw AUROC가 점수 규모와 무관해진다(Phase 3에서 드러남).
- **근거**: anomalib 레퍼런스가 같은 torchmetrics 연산을 쓰므로 직접 구현보다 수치 차이의 원인이 줄고 검증 범위(동점 처리, 곡선 보간)가 작아진다. 사용자 결정(2026-10-25)이다.
- **배제한 대안**: 순수 torch로 AUROC, F1을 직접 구현하는 방식(Phase 1 첫 구현, 일치 검증 대상이 늘고 유지 비용이 든다). scikit-learn 사용(의존이 늘고 anomalib과 호출이 다르다). raw 점수를 `BinaryROC`에 그대로 넣는 방식(PaDiM 첫 실행에서 test Image AUROC가 레퍼런스보다 0.035~0.038 낮았다).

## D-21. 테스트는 pytest로 작성하고 실행한다

- **선택**: `tests/`의 테스트는 pytest 스타일(함수 `test_*`, `assert`, fixture, `pytest.raises`, `monkeypatch`, `tmp_path`)로 작성하고 `python.exe -m pytest <경로>`로 실행한다. `unittest`는 쓰지 않는다. `tests/conftest.py`가 `<root>/src`를 `sys.path`에 추가하며 설정 파일(`pyproject.toml`, `pytest.ini`)은 두지 않는다. import 차단을 확인하는 테스트(NFR-1, NFR-2, FR-14, FR-19, FR-22)는 같은 프로세스에서 모듈 캐시 때문에 틀릴 수 있으므로 subprocess로 새 인터프리터를 띄워 확인한다.
- **적용 범위**: Phase 0 테스트를 포함해 모든 Phase의 `tests/`에 적용한다. 새 Phase의 테스트도 pytest로 쓴다. `evaluation/`의 실행·비교 스크립트(`train.py`, `predict.py`, `compare_*.py`)는 테스트가 아니므로 pytest 대상이 아니고 D-15의 방식으로 실행한다.
- **D-15와의 관계**: D-15가 배제한 `-m` 모듈 실행은 실행 스크립트(CLI)에 대한 것이고, 테스트에 한해 `-m pytest`를 쓴다.
- **근거**: assert 재작성과 fixture로 테스트가 짧아지고, 실패 출력이 읽기 쉽다. 사용자 결정(2026-10-25)이다.
- **배제한 대안**: `unittest`(Phase 0, Phase 1 첫 구현). 테스트 파일을 `python <file>.py`로 직접 실행하는 방식. `pyproject.toml`·`pytest.ini` 설정 파일(D-15의 무설정 원칙).

## D-22. 학습 에폭은 10 epoch로 고정하고, 1 epoch 적합 모델은 예외로 한다

- **선택**: 여러 에폭이 필요한 모델은 공유 yaml의 `max_epochs=10`을 레퍼런스와 native에 똑같이 쓰고 모델 yaml로 덮어쓰지 않는다(Phase 2 EfficientAD, 사용자 결정 2026-10-25). 원본이 `trainer_arguments`로 `max_epochs=1`을 고정하는 적합형 모델은 그 값을 따르며, 레퍼런스 yaml과 native yaml 모두 `max_epochs: 1`이다(Phase 3 PaDiM 사용자 결정 2026-10-25, Phase 4 DFM 사용자 결정 2026-10-26). 원본이 `trainer_arguments`로 1이 아닌 에폭을 고정하는 모델(FRE의 220 epoch)은 레퍼런스 어댑터(`evaluation/anomalib/local_models.py`의 `LocalFre`)가 그 값을 공유 예산의 10으로 바꾸고, native의 `trainer_arguments`도 10으로 둔다(Phase 9, 사용자 결정 2026-10-26).
- **적용 범위**: 모델이 10이 아닌 여러 에폭을 고정하는 경우는 레퍼런스와 native 모두 10으로 통일한다. 원본이 의도한 에폭에서의 성능은 확인하지 않는다.
- **근거**: 원본이 고정한 값을 바꾸면 알고리즘을 바꾸는 것이다(D-18). 레퍼런스와 native가 같은 학습 예산이어야 FR-18 비교가 성립한다. FRE의 220 epoch는 알고리즘(구조, 손실)이 아니라 학습 예산이므로 사용자가 예외로 정했다. 220 epoch는 측정한 에폭당 속도로 bottle 약 35분, visa 약 100분이 1회당 들고 레퍼런스와 native를 합치면 약 4.5시간이 예상되었다. 10 epoch는 두 category 합계 약 7분이었다(Phase 9).
- **배제한 대안**: 1 epoch 적합 모델에도 10 epoch를 강제하는 방식(원본 동작이 바뀐다). native에서만 에폭을 다르게 주는 방식(레퍼런스와 예산이 어긋난다). EfficientAD 모델 yaml에 20 epoch를 적는 방식(처음 시도했고 결정 뒤 제거했다). FRE에 원본의 220 epoch를 그대로 쓰는 방식(시간이 길다. Phase 9에서 처음 정한 뒤 사용자가 철회했다). 임의의 N epoch로 절충하는 방식(근거가 없다).
- **남은 한계**: FRE는 원본이 고정한 220 epoch가 아니라 10 epoch로 학습한 결과를 비교하므로 원본이 의도한 수렴 상태의 성능은 확인하지 않았다(Phase 9).

## D-23. kornia를 순수 모델과 학습 손실의 외부 의존으로 허용한다

- **선택**: PaDiM의 `GaussianBlur2d`는 원본처럼 kornia(0.8.2)를 import한다. 독립 구현하지 않는다(Phase 3, 사용자 결정 2026-10-25). DRAEM의 `DraemLoss`도 원본처럼 kornia의 `FocalLoss`와 `SSIMLoss`를 import한다(Phase 11, 사용자 결정 2026-10-27).
- **적용 범위**: kornia에만 해당한다. 다른 라이브러리는 이 결정에 포함되지 않으며 필요한 Phase에서 따로 정한다. kornia는 `src/visionlab`에서 순수 모델 쪽(`model.py`와 `components/`)과 DRAEM의 학습 전용 손실(`models/draem/loss.py`)에만 쓰고, 배포 환경에도 설치되어 있어야 한다. 현재 kornia를 import하는 곳은 `components/filters/blur.py`와 `models/draem/loss.py` 두 파일이다. `loss.py`는 `draem/lightning_model.py`만 import하며 `Predictor`와 export 경로(`draem/model.py`, `utils/io.py`)는 import하지 않는다.
- **INTENT와의 관계**: INTENT의 "배포되는 모델은 순수 PyTorch만으로 동작한다"와 긴장 관계에 있다. kornia는 PyTorch 위의 라이브러리이므로 이 결정은 그 범위를 PyTorch 생태계 패키지까지 넓혀 읽는다. INTENT는 고치지 않는다.
- **근거**: 원본 알고리즘 보존(D-18)과 FR-5 일치 검증의 범위를 줄인다. DRAEM의 손실은 원본이 kornia의 `FocalLoss`와 `SSIMLoss`로 계산하며, 학습 전용이라 추론 경로에 남지 않는다.
- **배제한 대안**: `GaussianBlur2d`를 순수 torch로 독립 구현하는 방식(검증 대상이 늘고 원본과 어긋날 위험이 있다). `FocalLoss`와 `SSIMLoss`를 독립 구현하는 방식(같은 이유).

## D-24. 레퍼런스 비교는 레퍼런스가 보고하는 값과 같은 정의로 한다

- **선택**: FR-18, FR-20의 AUROC 비교는 anomalib이 test에서 보고하는 값과 같은 정의로 한다. anomalib의 test AUROC는 PostProcessor로 정규화하고 clamp한 점수로 계산하므로, visionlab도 같은 export의 정규화 점수(D-8의 임계값 중심 변환과 clamp)로 AUROC를 계산해 비교한다. visionlab 자체 지표(D-8, D-20의 raw 순서 보존 AUROC)는 그대로 두고 결과 파일에 참고로 함께 적는다. 정규화 AUROC는 export 기반 `evaluation/visionlab/predict.py`에서 raw와 함께 기록하며, native `train.py` 로그는 raw만 기록한다(Phase 5 사용자 결정 2026-10-26). 두 정의의 값이 0.01 넘게 다르면 그 원인을 결과 파일에 적는다.
- **근거**: Phase 4 DFM의 visa/candle에서 모델 출력은 같았다(FR-5 최대 차이 0.0, F1 일치). 그런데 raw 값 0.9606과 레퍼런스 0.7487이 0.2119 달랐다. anomalib의 정규화 점수에서 pixel의 97.8%가 0.0으로 clamp되어 동점이 생기기 때문이다. 같은 정의로 비교하면 visionlab 0.748692, 레퍼런스 0.748692로 일치한다. 사용자 결정(2026-10-26)이다.
- **남은 한계**: 레퍼런스의 raw 값은 구하지 않았다. 사용자가 추가 실행 없이 Phase를 닫기로 했으므로, raw 정의의 비교는 출력 일치(FR-5)와 정규화 값 일치에서 추론한 것이다. 점수 규모가 큰 모델(Phase 5 DFKDE, Phase 9 FRE 등)에서 같은 차이가 반복될 수 있다.
- **배제한 대안**: raw AUROC를 기준으로 하고 레퍼런스의 raw 값을 따로 구해 비교하는 방식(레퍼런스 평가를 추가로 실행해야 한다). 미충족으로 두고 기준 정의를 바꾸는 방식(차이가 구현이 아니라 지표 정의에서 나오므로 Phase가 막히기만 한다).

## D-25. scikit-learn을 적합 때만 쓰는 지연 외부 의존으로 허용한다

- **선택**: PatchCore의 `SparseRandomProjection`은 원본처럼 `sklearn.utils.random.sample_without_replacement`를 쓴다. 독립 구현하지 않고, 이 import를 `_sparse_random_matrix` 함수 안(밀도가 1이 아닌 분기)으로 옮긴다(Phase 6, 사용자 결정 2026-10-26). CFA의 `initialize_centroid`가 `gamma_c > 1`일 때 쓰는 `sklearn.cluster.KMeans`도 같은 방식으로 그 분기 안에서 지연 import한다(Phase 8, 사용자 결정 2026-10-26).
- **적용 범위**: `sklearn.utils.random.sample_without_replacement`(PatchCore)와 `sklearn.cluster.KMeans`(CFA) 두 곳에만 해당한다. metric 계산에 scikit-learn을 쓰지 않는다는 D-20은 그대로다. 다른 라이브러리는 이 결정에 포함되지 않으며 필요한 Phase에서 따로 정한다. 추론 경로는 scikit-learn 없이 import되고 동작한다. `tests/common/test_dependency.py`가 허용 목록(파일, 클래스, 메서드, 모듈, 이름의 상수)에 있는 두 import 외의 scikit-learn import가 `src/visionlab`에 없음을 검사하고, `tests/06_patchcore`와 `tests/08_cfa`가 scikit-learn을 차단한 자식 프로세스에서 순수 모듈 import와 추론이 성공하고 적합 분기만 `ModuleNotFoundError`를 내는 것을 확인한다. CFA의 `KMeans` 분기(`gamma_c > 1`)는 기본값(`gamma_c=1`)에서 실행되지 않으므로 이 분기의 수치는 실행으로 검증하지 않았다(Phase 8).
- **INTENT와의 관계**: INTENT의 "배포되는 모델은 순수 PyTorch만으로 동작한다"와 긴장 관계에 있다(D-23과 같다). scikit-learn은 PyTorch 위의 라이브러리가 아니므로 이 결정은 그 범위를 적합 시점의 예외로 넓혀 읽는다. 배포되는 추론 경로에는 남지 않는다. INTENT는 고치지 않는다.
- **근거**: 원본 알고리즘 보존(D-18). 독립 구현하면 난수를 소비하는 순서가 달라져 coreset이 레퍼런스와 어긋날 수 있다. 같은 seed에서 native의 memory bank가 레퍼런스와 모든 행이 같았다(Phase 6, bottle 16,385행, capsule 17,169행). CFA의 `KMeans`는 기본값에서 쓰이지 않아 import 위치만 옮겼다.
- **배제한 대안**: numpy나 torch로 `sample_without_replacement`와 `KMeans`를 독립 구현하는 방식(coreset이 레퍼런스와 어긋날 위험이 있다). 모듈 최상단에서 import하는 방식(추론 환경에도 scikit-learn이 필요해진다).
