# visionlab

## 개요

anomalib 같은 통합 학습·평가·추론 프레임워크를 native Lightning으로 직접 구현하고, 학습한 모델과
가중치를 순수 PyTorch만으로 배포하는 프로젝트다. 배포 환경에는 anomalib과 Lightning이 없고 PyTorch만
쓸 수 있기 때문이다.

첫 대상 task는 anomaly detection(STFPM·EfficientAD, MVTec AD·VisA)이다. 이후 classification,
segmentation, object detection 등 다른 vision task로 넓힌다. 구현 결과는 anomalib 2.0.0을 설치한 환경에서
같은 데이터·학습 예산으로 실행해 Image·Pixel AUROC로 비교·검증한다.

검증은 다음 세 단계로 한다(INTENT, PLAN). 2·3단계는 각각 1단계와 비교하며, 범주별 Image·Pixel AUROC 차이가
각각 0.01 이내여야 한다.

1. anomalib image 모델을 로컬 자산으로 그대로 실행한 레퍼런스 수치(`evaluation/anomalib/`, 각 Phase에서 그 모델만)
2. native Lightning으로 이식한 모델의 학습·평가 수치(PLAN Phase 1~18, Phase마다 모델 하나, 레퍼런스 확보 다음)
3. 2단계 가중치를 순수 PyTorch 모델로 불러온 추론 수치(PLAN Phase 1~18)

결과는 Phase마다 `docs/results/NN-<kebab-case>.md`에 기록한다.

## 요구 환경

개발과 검증에 쓰는 환경은 다음과 같다.

- Windows, WinPython `C:\winpython\WPy64-31180_gpu` (Python 3.11.8, venv 없음)
- torch 2.5.1+cu121, torchvision 0.20.1, Lightning 2.5.5, anomalib 2.0.0
- 로컬 데이터 `D:\datasets`(mvtec, visa), 로컬 가중치 `D:\backbones`

설치 내역과 로컬 경로 연결 방법은 `docs/anomalib/anomalib-install-guide.md`에 있다.

## 사용법

모든 예제는 WinPython 파이썬으로 실행한다.

```powershell
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
```

### anomalib 레퍼런스 실행

PLAN 각 Phase의 첫 단계에서 그 모델의 레퍼런스 수치(mvtec `bottle`, visa `candle`)를 이 실행으로 얻는다.
VisA는 처음 실행할 때 anomalib의 1cls 분할을 `D:\datasets\visa_anomalib\visa_pytorch`로 복사한다. 원본
`D:\datasets\visa`는 바꾸지 않으며, 복사본이 이미 있으면 복사를 건너뛴다(D-17).

`evaluation/anomalib/anomalib_run.py`는 anomalib의 `AnomalibCLI`로 레퍼런스를 얻는 진입점이다. 상수 대신 yaml과 명령행 인자를 받는다.

설정은 `default.yaml`, `data/<name>.yaml`, `models/NN_<name>.yaml`을 `--config`로 차례로 겹쳐 지정하며, 뒤에 준 값이 앞의
값을 덮어쓴다. 데이터와 가중치 경로는 이 yaml에 적혀 있고, 루트 `paths.toml`의 `[backbones]`는 timm이 `resnet18` 같은
모델 이름을 요청할 때 로컬 가중치 파일을 읽게 하는 데에만 쓴다. 오프라인 설정, 실행 폴더 패치, timm 등록은
`local_setup.py`를 거친다.

`evaluation/anomalib`에서 다음과 같이 실행한다.

```powershell
cd evaluation/anomalib
& $py anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/01_stfpm.yaml
```

- `train`은 학습 후 테스트를 한다. `fit`은 학습만, `test`는 테스트만 수행한다.
- 별도 `test`에는 `--ckpt_path`가 필요하다.
- 범주를 바꾸려면 `--data.init_args.category cable`을 덧붙인다.
- 산출물은 `outputs/anomalib/<Model>/<DataModule>/<category>/v<N>/`에 저장되고 `latest`가 가장 최근 버전을 가리킨다.
  모델과 DataModule 이름은 클래스 이름이므로 `Local` 접두사가 있는 클래스는 접두사가 폴더명에 남는다.
- `seed_everything`은 anomalib CLI가 선언만 하고 호출하지 않아 `anomalib_run.py`가 직접 호출한다. `data/*.yaml`의 `seed`는 데이터 분할용이며 `default.yaml`의 값과
  같게 유지한다.
- 모델 yaml의 backbone은 anomalib 원본과 같은 이름(`resnet18` 등)이다. `local_setup.py`가 timm의 해당 이름을 로컬
  가중치 로더로 바꾸며, 이때 timm이 덮어쓰기 경고를 낸다. `02_efficient_ad.yaml`은 `model_size`를 바꾸면 `teacher_file`도
  같이 바꾼다.
- `default_root_dir`는 yaml 최상위 키만 쓴다. `trainer.default_root_dir`를 직접 지정하지 않는다.
- `11_draem.yaml`과 `17_dsr.yaml`은 정규화 없는 resize 전용 `pre_processor`를 직접 지정한다. anomalib 2.0.0은 두 모델의
  `configure_transforms`를 호출하지 않아 지정하지 않으면 ImageNet 정규화가 적용된다.

## 폴더 구조

현재 저장소에 있는 폴더와 파일은 다음과 같다.

```text
visionlab/
├── .claude/
│   └── skills/
│       └── herdr/
│           └── SKILL.md                # project-only herdr skill
├── docs/
│   ├── anomalib/
│   │   └── anomalib-install-guide.md   # install and local path setup
│   ├── results/
│   │   └── README.md                   # result file rules, shared reference settings
│   ├── workflow/
│   │   └── model-port.md               # per-model reference, port and verification procedure
│   └── local-assets-usage.md           # local path wiring candidates
├── evaluation/
│   └── anomalib/                       # per-Phase anomalib reference runs (evaluation/visionlab is added with Phase 1)
│       ├── data/                       # per-dataset configs (mvtec.yaml, visa.yaml)
│       ├── models/                     # per-model configs (NN_<name>.yaml, NN = Phase number)
│       ├── anomalib_run.py             # AnomalibCLI entry point (seed hook)
│       ├── default.yaml                # default seed_everything, output root and trainer
│       ├── local_data.py               # datamodule subclasses for local datasets
│       ├── local_models.py             # model subclasses that load local assets
│       └── local_setup.py              # offline setup, Windows run folder patch, local timm backbones
├── outputs/                            # run outputs (regenerable)
├── specs/
│   ├── DECISIONS.md
│   ├── INTENT.md                       # what and why (SSOT)
│   ├── PLAN.md
│   └── SPEC.md
├── AGENTS.md                           # agent rules (CLAUDE.md points here)
├── CLAUDE.md
├── paths.toml                          # local backbone weight paths (D-12)
└── README.md
```

위 트리는 지금 있는 파일만 보여 준다. 최종 목표 구조(`src/visionlab/`, `tests/`, `notebooks/` 등)는
`specs/DECISIONS.md`의 D-15에 있으며, 각 Phase는 자기 범위의 파일만 만든다.

## 문서

프로젝트 문서는 다음과 같다.

| 문서 | 내용 |
| --- | --- |
| `docs/anomalib/anomalib-install-guide.md` | anomalib 설치와 로컬 데이터·가중치 연결 |
| `docs/results/` | Phase별 결과(모델 노트, 레퍼런스, 이식 메모, 적대적 검증, 이식 결과)와 작성 규칙 |
| `docs/workflow/model-port.md` | 모델별 레퍼런스 확보, 이식, 검증 절차와 에이전트(Claude, codex, agy) 지시 양식 |
| `specs/DECISIONS.md` | 설계 결정과 근거 |
| `specs/INTENT.md` | 무엇을 왜 하는가, 변하지 않는 원칙 (SSOT) |
| `specs/PLAN.md` | Phase별 목적·완료 조건·검증 방식 |
| `specs/SPEC.md` | 기능·비기능 요구사항과 판정 방법 |
