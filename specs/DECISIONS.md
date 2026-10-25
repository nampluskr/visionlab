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

- **선택**: Dataset Transform은 RGB 변환·이미지와 mask의 공간 변환·float 변환을 맡는다. PreProcessor는 Wrapper에서 입력 확인과 모델별 정규화를 맡는다. STFPM은 ImageNet 정규화를 1회 적용하고, EfficientAD는 외부 정규화를 하지 않는다.
- **근거**: 서로 다른 원본 해상도를 collate 전에 같은 크기로 맞추고, 모델별 정규화는 모델 호출 경로에서 통제한다. 독립 추론도 같은 두 단계를 조합해 이중 resize·정규화를 막는다.
- **배제한 대안**: lightning-tutorial v0.2의 Dataset 내 정규화. anomalib처럼 PreProcessor가 resize·정규화를 함께 하고 Callback을 상속하는 방식.

## D-7. 네 처리 구성요소를 순수 기능으로 구성

- **선택**: PreProcessor·PostProcessor·Evaluator·ImageVisualizer는 Lightning 없이 쓰는 순수 기능으로 `anomaly_detection/common`에 둔다. Wrapper가 전처리·모델·후처리·평가 순서를 제어하고, 시각화 파일 저장만 native Callback(`common/lightning_visualizer.py`)이 맡는다.
- **근거**: 구성요소 간 순서를 Callback 등록 순서에 의존시키지 않는다. 독립 추론과 순수 PyTorch 평가에서 같은 구현을 재사용한다.
- **배제한 대안**: anomalib의 `nn.Module`·`Callback` 다중 상속 구조. 네 개의 독립 처리 Callback.

## D-8. 후처리 보정과 평가 상태의 분리

- **선택**: PostProcessor의 image/pixel min·max와 F1 기반 임계값은 fit validation 전체에서 수집해 epoch 종료에 확정한다. standalone validate·test·predict는 저장 상태를 읽기만 한다. 정규화는 임계값 중심 변환 `clamp((x-threshold)/(max-min)+0.5, 0, 1)`을 사용한다. raw 예측은 보존하고 후처리 결과를 별도로 둔다. validation은 raw AUROC, test는 raw AUROC와 후처리 F1을 기록한다.
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

- **선택**: 데이터셋과 가중치 파일의 경로는 실행 설정(yaml)에 절대경로로 직접 적는다(예: `data.init_args.root`, `model.init_args.weights_file`·`teacher_file`). 그 값을 Dataset·특징 추출기·teacher 로더까지 그대로 전달한다. 루트 `paths.toml`에는 `[backbones]`만 두며, 키는 timm 모델 이름이고 값은 로컬 가중치 파일이다. timm 백본은 이름으로 조회되므로 실행 시작 시 이 매핑으로 timm의 같은 이름을 로컬 가중치 로더로 바꾼다.
- **근거**: CLI가 저장하는 `config.yaml`에 실제 사용한 경로가 남아 별도 실행 기록 파일 없이도 재현할 수 있다. yaml·CLI의 override가 생성자에 그대로 전달된다. 기존 보조 로더가 내부에서 기본 `paths.toml`을 다시 읽어 CLI 선택과 달라질 수 있었다.
- **배제한 대안**: `paths.toml`을 모든 경로의 단일 원본으로 두고 `Local*` 생성자가 생략된 인자를 채우는 방식(저장된 config에 실제 경로가 남지 않는다). 로더 내부에서 기본 `paths.toml` 재읽기. `local_<name>` 별칭으로 timm에 등록하는 방식(anomalib의 backbone 이름 검사에 걸린다).

## D-13. 데이터 비교를 모델 비교의 선행 조건으로

- **선택**: 모델 성능을 비교하기 전에 Dataset·DataModule의 파일 목록·라벨·mask·split·최종 모델 입력을 비교한다. 통제 비교에서는 실제 파일 목록을 manifest로 고정하고, 데이터 검증 ID를 모델 결과에 연결한다.
- **근거**: anomalib 기본 분할과 기존 분할이 다르고, anomalib PreProcessor가 입력을 다시 바꿀 수 있다.
- **배제한 대안**: 파일 수·seed·분할 비율이 같다는 이유로 동일 Dataset으로 판정. 양쪽에 같은 Dataset을 주입한 결과만으로 원래 구현이 같다고 판정.

## D-14. 개발 환경과 자산 위치

- **선택**: 개발 환경은 `C:\winpython\WPy64-31180_gpu`의 Python 3.11.8, torch 2.5.1+cu121, torchvision 0.20.1, numpy 2.2.5, Lightning 2.5.5, anomalib 2.0.0(openvino 2025.4.1, jsonargparse 4.39.0)이다. 데이터는 `D:\datasets`, 가중치는 `D:\backbones`를 사용한다. 설치 내역은 `docs/anomalib/anomalib-install-guide.md`를 따른다.
- **근거**: 기존 PyTorch·CUDA를 유지한 채 설치했고 기존 패키지 변경은 0건이다. openvino 2026 이상은 `openvino.runtime`이 없고, jsonargparse 4.40 이상은 anomalib 2.0.0 CLI가 실패한다.
- **배제한 대안**: 원 문서 환경(torch 2.2.1+cu118, numpy 1.26.4)으로 다운그레이드. `E:\datasets`·`E:\backbones` 경로.

## D-15. 코드 배치

- **선택**: 목표 구조는 아래와 같다. 각 Phase는 자기 범위의 파일만 만들고, 아직 필요 없는 폴더·파일은 만들지 않는다.

```text
visionlab/                                      # project root
├── paths.toml                                  # [backbones] timm name -> local weights file (D-12)
├── docs/
│   ├── anomalib/                               # install guide
│   ├── results/                                # one file per Phase: NN-<kebab-case>.md, rules in README.md
│   └── workflow/
│       └── model-port.md                       # per-Phase procedure and agent instructions
├── evaluation/                                 # evaluation code (reference runs, comparisons, ported model evaluation)
│   ├── anomalib/                               # imports anomalib
│   │   ├── data/                               # shared data yaml (mvtec.yaml, visa.yaml)
│   │   ├── models/                             # shared model yaml (<name>.yaml), per-Phase reference
│   │   ├── 00_compare_data.py                  # Phase 0, visionlab data vs anomalib data (FR-17, D-13)
│   │   ├── anomalib_run.py                     # AnomalibCLI entry point, reference runs
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
├── tests/                                      # visionlab test scripts (FR·NFR checks)
│   ├── test_data.py                            # FR-1~3, Phase 0
│   ├── test_stfpm_model.py                     # FR-4, FR-16, NFR-5
│   ├── test_components.py                      # FR-7~11
│   ├── test_stfpm_lightning.py                 # FR-6, FR-12
│   ├── test_cli.py                             # FR-13
│   ├── test_predictor.py                       # FR-14
│   ├── test_dependency.py                      # NFR-1, NFR-2
│   └── test_efficient_ad.py                    # FR-15
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
- **배제한 대안**: `models/common`(D-11). 프로젝트 루트 바로 아래의 Python 파일(평가 스크립트는 `evaluation/`에 둔다). 모델마다 `NN_train_<model>.py`처럼 Phase별 실행 스크립트를 따로 두는 방식(18개 모델이면 36개가 되고 같은 절차가 반복된다). 평가 코드를 `examples/`에 두는 방식(예제와 정식 평가 자료가 섞인다). 평가 코드를 `tests/`에 두는 방식(장시간 학습과 단위 테스트가 섞이고 tests의 anomalib 미사용 원칙과 충돌한다). 모든 task 폴더와 하위 폴더를 미리 생성. 순수 파일과 Lightning 연결을 한 파일에 섞는 방식. `pyproject.toml`·`pip install -e .` 개발 설치. 실행 전 `PYTHONPATH` 환경변수 지정. `-m` 모듈 실행.
`outputs/`는 git에서 제외한다(`.gitignore`). anomalib을 import하는 비교·테스트 코드는 `evaluation/anomalib/`에만 두고 `src/visionlab`에서 참조하지 않는다(D-3·INTENT). `evaluation/anomalib/`에는 레퍼런스 실행 진입점(`anomalib_run.py`는 `AnomalibCLI`), 공유 yaml, 공통 설정 `local_setup.py`(오프라인 설정, timm 이름의 로컬 가중치 연결, Windows 실행 폴더 처리), `local_data.py`의 `LocalMVTecAD`·`LocalVisa`(D-17), `local_models.py`의 로컬 자산 모델, visionlab 데이터·출력을 anomalib과 대조하는 비교 스크립트(FR-17, FR-5)를 둔다.

패키징 설정(`pyproject.toml` 등)과 `pip install -e .` 개발 설치는 사용하지 않는다. 테스트와 실행은 사용자가 PowerShell에서 WinPython의 전체 경로로 직접 실행한다.

```powershell
C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe <file>.py --<option>
```

`evaluation/`·`tests/`의 스크립트는 첫머리에서 `<root>/src`를 `sys.path`에 추가한 뒤 `visionlab`을 import한다. visionlab 자체 테스트 스크립트는 루트 `tests/`에 둔다. `tests/`는 anomalib을 import하지 않는다.

스크립트 이름 규칙은 다음과 같다. Phase에 묶인 일회성 스크립트(데이터 대조 등)는 `NN_<단계>.py`이고 `NN`은 PLAN의 Phase 번호다. 모델 Phase 전체에서 쓰는 공용 진입점은 번호 없이 `anomalib_run.py`, `compare_outputs.py`, `train.py`, `predict.py`로 두고 모델은 yaml 또는 `--model` 인자로 고른다. 공용 모듈은 `_` 접두사를 붙이되, `evaluation/anomalib/`의 공용 모듈은 yaml의 `class_path`로 import되므로 `local_` 접두사를 쓴다. 테스트는 `tests/test_<대상>.py`다. anomalib을 import하는 스크립트만 `evaluation/anomalib/`에 두고, 나머지 평가 스크립트는 `evaluation/visionlab/`에 둔다. CLI는 `-m` 실행 대신 `evaluation/visionlab/train.py`가 `sys.path`를 설정한 뒤 `trainers/lightning_cli.py`의 `LitCLI`를 호출한다. 실행 결과는 `outputs/`에 저장한다.

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
