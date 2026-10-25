# 로컬 자산 경로 관리 방법 비교

작성일: 2026-10-23. 검토용 문서다. claude, codex(GPT-6.1-Sol), agy(Gemini 3.8 Flash)의 검토를 취합했다. 결정이 나면 확정된 내용은 사람이 `specs/DECISIONS.md`로 옮긴다.

## 1. 문서의 목적과 읽는 법

### 1.1 세 평가 단계와 비교 범위

visionlab은 다음 세 단계 순서로 수치를 맞춰 가며 검증한다(D-16). 이 문서는 세 단계 모두에서 로컬 데이터셋(`D:\datasets`)과 가중치(`D:\backbones`)를 연결하고 인터넷 다운로드를 막는 방법을 클라이언트 코드로 비교한다.

| 단계 | 쓰는 것 | 쓰지 않는 것 | 해당 Phase |
|---|---|---|---|
| anomalib 평가 | anomalib 2.0.0 DataModule, 모델, Engine | visionlab | Phase 1~3 |
| native Lightning 평가 | visionlab LitData, LitModel, Lightning Trainer | anomalib | Phase 4~5, 7 |
| pure PyTorch 평가 | visionlab Dataset, `nn.Module`, Predictor | anomalib, Lightning | Phase 6 |

모든 단계에 공통으로 걸리는 제약은 다음과 같다.

- 데이터와 가중치는 로컬 파일만 쓴다. 필요한 파일이 없으면 무작위 대체나 다운로드 없이 오류를 낸다(NFR-3).
- anomalib 라이브러리 코드는 수정하지 않는다. 경로는 클라이언트 코드에서 정한다.
- visionlab 생성자는 anomalib 2.0.0과 이름, 순서, 기본값이 같아야 한다(D-5). `weights_path` 같은 인자를 추가하지 않는다.
- 배포 코드는 순수 PyTorch만으로 동작한다(INTENT, NFR-1, NFR-2).

### 1.2 표기와 비교 기준

코드 예시마다 상태를 다음 세 가지로 표기한다.

- **구현**: 현재 저장소에 있는 코드다. `examples/anomalib_old/_common.py`가 여기에 해당한다.
- **제안**: 아직 없는 코드다. visionlab의 `src/` 아래 클래스 이름은 D-15의 목표 구조를 따른 가정이며, 실제 생성자는 구현 시 SPEC으로 확정한다.
- **미검증**: 실행으로 확인하지 않은 동작이다.

후보는 다음 다섯 기준으로 비교한다.

| 기준 | 묻는 것 |
|---|---|
| 추적성 | 코드만 읽고 어떤 파일이 쓰이는지 알 수 있는가 |
| 오프라인 보장 | 다운로드가 일어나지 않음을 구조적으로 막는가 |
| 의존성 경계 | pure PyTorch 단계에 표준 라이브러리 밖 의존성이 생기지 않는가 |
| 재현성 | 세 단계가 같은 경로와 같은 가중치를 쓴다는 것을 기록으로 보일 수 있는가 |
| DECISIONS 정합성 | D-4, D-5, D-12, D-15와 충돌하지 않는가 |

## 2. 공통 자산과 비교 시나리오

### 2.1 로컬 자산 목록

현재 사용하는 로컬 자산은 다음과 같다.

| 이름 | 경로 | 쓰는 곳 |
|---|---|---|
| MVTec AD | `D:\datasets\mvtec` | 15개 범주. `root`는 범주 폴더의 상위 폴더 |
| VisA 원본 | `D:\datasets\visa` | 12개 범주와 `split_csv/1cls.csv`. 읽기 전용 |
| VisA anomalib 작업 복사본 | `D:\datasets\visa_anomalib\visa_pytorch` | anomalib `Visa`의 `split_root` (D-17) |
| Imagenette | `D:\datasets\imagenette2\train` | EfficientAD 학습의 penalty 데이터 |
| ResNet18 | `D:\backbones\resnet18-f37072fd.pth` | STFPM teacher |
| EfficientAD teacher | `D:\backbones\efficientad_pretrained_weights\pretrained_teacher_{small,medium}.pth` | EfficientAD teacher |

### 2.2 원본, 작업 복사본, 캐시, 실행 산출물의 구분

경로의 성격은 네 가지로 나뉘며 다루는 규칙이 다르다.

- **원본**(`D:\datasets\mvtec`, `D:\datasets\visa`, `D:\backbones\*.pth`): 읽기만 한다. 없으면 오류다.
- **작업 복사본**(`visa_pytorch`): 원본에서 다시 만들 수 있다. 첫 실행 전에는 없을 수 있으므로 존재 검사 대상에서 뺀다.
- **캐시**(`TORCH_HOME`, `HF_HOME`): 다운로드가 일어났을 때 저장되는 위치다. 이 프로젝트에서는 비어 있어야 정상이다.
- **실행 산출물**(`outputs/`): checkpoint, 로그, manifest. 다시 만들 수 있다.

### 2.3 비교 시나리오

모든 코드 예시는 기본 사례 하나를 같은 방식으로 보인다. 추가 사례는 기본 사례와 달라지는 부분만 보인다.

- **기본 사례**: MVTec `bottle`, STFPM, ResNet18 teacher.
- **추가 사례 1**: VisA `candle`. 작업 복사본 경로가 하나 더 필요하다.
- **추가 사례 2**: EfficientAD small. teacher 파일과 Imagenette가 필요하다.

### 2.4 학습 재현, checkpoint 평가, 독립 추론의 자산 요구 차이

같은 모델이라도 무엇을 하느냐에 따라 필요한 자산이 다르다. 원칙은 "최소 학습에는 pretrained만, 평가와 추론에는 checkpoint만"이다. checkpoint에는 학습하지 않는 teacher까지 모델 전체가 들어 있기 때문이다. 경로 검사는 지금 하는 작업에 필요한 것만 한다. 원칙상 필요한 자산은 다음과 같다.

| 작업 | 데이터 | pretrained (backbone, teacher) | Imagenette | checkpoint |
|---|---|---|---|---|
| 최소 학습 (처음부터) | 필요 | 필요 | EfficientAD만 필요 | 만듦 |
| 학습 재개 | 필요 | 불필요 | EfficientAD만 필요 | 필요 |
| checkpoint 평가 | 필요 | 불필요 | 불필요 | 필요 |
| 배포 추론 | 입력 이미지만 | 불필요, 없어도 오류가 나면 안 됨 | 불필요 | 배포 artifact 필요 |

anomalib 2.0.0은 생성 코드 때문에 이 원칙에서 벗어나는 경우가 두 가지 있다(설치 소스로 확인).

- **STFPM**: 생성자가 `TimmFeatureExtractor(pre_trained=True)`로 teacher를 만든다(`stfpm/torch_model.py:94`). 그래서 checkpoint 평가에도 생성 단계에서 resnet18 파일이 필요하다. 읽은 값은 직후 checkpoint로 덮어쓰이므로 결과에는 영향이 없다.
- **EfficientAD**: 학습 재개 시에도 `on_train_start`가 teacher 파일과 Imagenette를 다시 읽는다. test와 predict에는 teacher 파일이 필요 없다. standalone validate는 `on_validation_start`에서 quantile을 다시 계산한다.

visionlab(Lightning, pure PyTorch)은 원칙을 그대로 따르도록 설계한다(3.7).

## 3. 설계 축과 후보

### 3.1 다섯 축의 구분

"경로를 어떻게 관리하나"는 서로 다른 다섯 질문이 섞인 것이다. 후보는 이 축들의 조합이다.

| 축 | 질문 | 선택지 |
|---|---|---|
| 경로 저장 | 경로 문자열이 어디에 적혀 있나 | 모듈 상수, 환경변수와 `.env`, TOML, YAML, JSON |
| 설정 표현 | 읽은 값을 무엇으로 들고 다니나 | dict, dataclass, 명시 인자, 전역 상태 |
| 이름 해석 | `"mvtec"` 같은 이름을 클래스로 어떻게 바꾸나 | 직접 import, if 분기, registry, `class_path` |
| 객체 생성 | 누가 생성자를 호출하나 | 스크립트, 조립 함수, factory, CLI |
| 오프라인 강제 | 다운로드를 무엇으로 막나 | 누락 시 오류, 라이브러리 환경변수, 네트워크 차단 |

dataclass는 AGENTS의 type hint 금지와 충돌하고, Hydra는 현재 규모에 비해 설정 계층과 실행 폴더 관리 부담이 커서 후보에서 뺐다.

오프라인 강제 축은 모든 후보에 공통으로 적용한다. 환경변수(`HF_HUB_OFFLINE`)는 Hugging Face 다운로드만 막고 torch.hub와 anomalib의 URL 다운로드는 막지 못한다. 따라서 "로컬 파일을 명시하고 없으면 오류"를 기본으로 두고, 네트워크 차단 실행으로 증명한다(7장).

### 3.2 여섯 후보의 정의

후보는 다음 여섯 가지다. 단계 열의 O는 사용 가능, △는 제한적, X는 사용 불가다.

| # | 후보 | 경로 저장 | 객체 생성 | anomalib | Lightning | PyTorch | 비고 |
|---|---|---|---|---|---|---|---|
| 1 | 모듈 상수와 Local 서브클래스 | `_common.py` 상수 | 직접 생성자 호출 | O | O | O | 현재 구현된 방식 |
| 2 | 환경변수와 `.env` | `.env`, 환경변수 | 직접 생성자 호출 | O | O | △ | `.env` 읽기에 python-dotenv 필요 |
| 3 | `paths.toml`과 명시 인자 주입 | `paths.toml` (D-12) | 진입점에서 한 번 읽어 인자로 전달 | O | O | O | 세 검토자 공통 추천. 표준 라이브러리 `tomllib`만 사용 |
| 4 | YAML과 CLI | YAML | anomalib CLI, LightningCLI의 `class_path` | O | O | X | Phase 5와 겹침 |
| 5 | 설정 파일과 작은 조립 함수 | TOML 또는 YAML | 명시 인자를 받는 `build_*` 함수 | O | O | O | D-4 문구 정리 필요 |
| 6 | 자체 registry와 `create_*` 함수 | TOML 또는 YAML | 이름으로 클래스를 찾아 생성 | O | O | △ | D-4가 배제한 방식 |

후보 1의 PyTorch 단계는 상수와 `_common.py`를 분리해야 쓸 수 있다. `_common.py`는 anomalib을 import하기 때문이다.

### 3.3 커뮤니티 관례

주요 라이브러리의 관례는 하나로 통일되어 있지 않다. 관례마다 이 프로젝트에 주는 시사점이 다르다.

| 라이브러리 | 방식 | 시사점 |
|---|---|---|
| torchvision | 직접 생성자와 이름 기반 `get_model`. `weights=` 선택이 다운로드로 이어짐 | 생성자에 이름만 주면 출처가 숨는다 |
| timm | registry와 `create_model`. `pretrained_cfg_overlay={"file": ...}`로 로컬 파일 지정 가능 | anomalib 단계에서 쓸 수 있는 유일한 로컬 backbone 연결 지점 |
| MMEngine, MMDetection | 전역 registry와 설정의 `type` 키로 조립 | 큰 생태계에 유리, 작은 프로젝트에는 scope와 import 관리가 부담 |
| Detectron2 | registry와 CfgNode, 또는 LazyConfig로 객체를 직접 구성 | 유연하지만 설정 체계를 따로 익혀야 함 |
| Hugging Face | `from_pretrained(로컬 디렉터리)`, `local_files_only=True`, `HF_HUB_OFFLINE` | 오프라인에 가장 친화적. 캐시 설정만으로는 오프라인 정책이 되지 않음 |
| LightningCLI | jsonargparse가 YAML의 `class_path`, `init_args`로 생성자를 호출 | registry 없이 명시적. 모델 내부 다운로드까지는 통제하지 못함 |
| anomalib | LightningCLI 기반 CLI와 클래스별 상대경로 기본값. 경로가 없으면 다운로드 | 기본값에 기대면 안 되고, 다운로드 메서드를 서브클래스로 막아야 함 |

MMEngine과 Detectron2는 일반 구조 설명이며 최신 버전은 확인하지 않았다.

### 3.4 시작 시 적용 순서

어느 후보든 스크립트 시작 시 순서는 같다. 순서가 바뀌면 환경변수가 적용되지 않거나 RNG 소비 순서가 달라질 수 있다.

```text
1. 설정 읽기        paths = load_paths(PATHS_FILE)        (표준 라이브러리만)
2. 환경 적용        os.environ["HF_HUB_OFFLINE"] = "1" 등  (제삼자 import 전)
3. import           anomalib, lightning, timm, visionlab
4. 필요 자산 검사    require_paths(paths, [...])
5. seed             seed_everything(SEED)
6. 객체 생성        DataModule, 모델
7. 가중치 로딩       backbone, teacher, 또는 checkpoint
8. 실행과 기록       fit, test, predict, manifest 저장
```

`HF_HUB_OFFLINE`은 huggingface_hub이 import될 때 읽히므로 2가 3보다 앞서야 한다. `TORCH_HOME`은 다운로드 호출 시점에 읽히므로 순서 제약이 약하다. 7에서 생성 후 가중치를 넣을 때 RNG 소비가 anomalib과 같은지는 미검증이다(5.2).

### 3.5 기록 계약

실행마다 다음을 `outputs/` 아래에 남긴다(NFR-4). 세 단계가 같은 자산을 썼다는 근거가 된다.

- 해석된 경로 전체(`paths` dict를 JSON으로).
- 사용한 가중치 파일의 SHA-256.
- seed, 분할 파일 목록(manifest), checkpoint 식별 정보.

### 3.6 bootstrap 함수 계약

상태: 제안. 세 단계의 평가 스크립트는 모두 다음 여섯 단계로 쓴다. 경로 설정과 다운로드 차단은 1단계 `bootstrap()` 한 곳에 모으고, 2~5단계는 `bootstrap()`이 돌려준 경로만 쓴다.

```text
1. bootstrap()   paths.toml 읽기, 필요 자산 검사, 다운로드 차단, 실행 폴더 생성
2. data          bootstrap 결과의 경로로 Dataset 또는 DataModule 생성
3. model         모델 생성과 가중치 로딩 (backbone, teacher, 또는 checkpoint)
4. trainer       Engine, Trainer, 또는 순수 평가 루프 구성
5. run           학습, 평가, 추론
6. save          해석된 경로, 가중치 해시, seed, metric 저장
```

`bootstrap()`은 후보 3(`paths.toml`)을 기준으로 하며 표준 라이브러리만 쓴다. 그래서 세 단계 모두에서, pure PyTorch 단계에서도 그대로 import할 수 있다. 위치는 `src/visionlab/bootstrap.py`로 결정했고 구현되어 있다. `examples/anomalib_old/_common.py`가 `src`를 `sys.path`에 추가해 import한다(visionlab은 아직 설치하지 않았다). 아래 코드는 구현본과 같은 내용이다.

```python
# src/visionlab/bootstrap.py: load local asset paths, block downloads and record a run
import hashlib
import json
import os
import socket
import tomllib


def block_network(*args, **kwargs):
    """Raise on any outgoing connection attempt."""
    raise RuntimeError("network access is blocked")


def bootstrap(paths_file, required, run_dir, offline=True):
    """Load paths.toml, check required assets, block downloads and create the run folder."""
    with open(paths_file, "rb") as f:
        paths = tomllib.load(f)
    for key in required:
        section, name = key.split(".")
        if not os.path.exists(paths[section][name]):
            raise FileNotFoundError(f"{key}: {paths[section][name]}")
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        socket.socket.connect = block_network
        socket.create_connection = block_network
    os.makedirs(run_dir, exist_ok=True)
    return {"paths": paths, "required": required, "run_dir": run_dir}


def file_sha256(path):
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_run(ctx, seed, metrics):
    """Write resolved paths, required file hashes, seed and metrics to run.json."""
    hashes = {}
    for key in ctx["required"]:
        section, name = key.split(".")
        path = ctx["paths"][section][name]
        if os.path.isfile(path):
            hashes[key] = file_sha256(path)
    record = {"paths": ctx["paths"], "hashes": hashes, "seed": seed, "metrics": metrics}
    with open(os.path.join(ctx["run_dir"], "run.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
```

설계상 판단은 다음과 같다.

- **소켓 차단을 기본으로 둔다**: `HF_HUB_OFFLINE`은 huggingface_hub이 import될 때 읽히므로, 스크립트 상단에서 anomalib이나 timm을 이미 import한 뒤 `bootstrap()`을 부르면 효과가 없다. 소켓 차단은 import 순서와 무관하게 이후의 모든 연결을 막으므로 `bootstrap()`이 `main()` 안에서 호출되어도 보장이 유지된다. 로컬 연결(localhost)도 막히므로 원격 logger를 쓰는 실행에서는 `offline=False`로 끈다. Windows의 DataLoader worker가 소켓 패치 없이 동작하는지는 미검증이다.
- **seed는 `bootstrap()`에 넣지 않는다**: anomalib, Lightning 단계는 Phase 1 레퍼런스와 같은 `seed_everything(SEED, workers=True)`를 써야 수치 비교가 맞다. pure PyTorch 단계는 Lightning이 없으므로 `torch.manual_seed`를 쓴다. 단계마다 seed 함수가 달라서 1단계 직후에 스크립트가 명시적으로 호출한다.
- **checkpoint는 `paths.toml`에 넣지 않는다**: checkpoint는 원본 자산이 아니라 실행 산출물이다(2.2). 스크립트 상단 상수로 두고, 없으면 로딩 시 오류가 난다.
- **`required`는 그 작업에 필요한 키만 적는다**: 학습 재현과 checkpoint 평가의 필요 자산이 다르다(2.4).

### 3.7 pretrained와 checkpoint 로딩 정책

상태: 제안. claude, codex, agy가 공통으로 권고한 정책이다. 생성할 때 pretrained를 기본으로 읽을지, checkpoint 유무에 따라 나눌지를 다룬다.

#### 3.7.1 비용 비교

생성 시 로컬 pretrained를 기본으로 읽는 비용은 다음과 같다. 수치는 실측이 아닌 추정이다.

| 항목 | 추정 |
|---|---|
| 시간 | ResNet18(44.7MB) 수십~수백 ms, teacher small(10.3MB) 10~50ms, medium(30.6MB) 30~150ms. 첫 디스크 접근이나 백신 검사가 끼면 수 초 |
| 메모리 | 읽는 가중치 크기만큼 잠시 늘어남. 최종 모델 크기는 같음 |
| 결과 수치 | checkpoint를 strict로 복원하면 차이 0. 복원 뒤 RNG를 쓰는 후속 처리가 이어지면 별도 확인 |
| 자산 의존성 | 배포 머신에 `D:\backbones`가 없으면 생성 단계에서 실패함 |

시간과 성능 비용은 무시할 수준이다. 실제 비용은 마지막 행의 자산 의존성이다. 배포 머신에는 학습된 checkpoint만 있고 pretrained 파일은 없으므로, 생성자가 pretrained를 읽으면 배포가 불가능하다.

#### 3.7.2 라이브러리 관례

주요 라이브러리는 생성자의 기본을 "불러오지 않음"으로 두고, pretrained는 명시 요청 시에만 불러온다(설치 소스로 확인).

| 라이브러리 | 기본값과 동작 |
|---|---|
| timm | `create_model(pretrained=False)`가 기본. `pretrained=True`와 `checkpoint_path`를 함께 주면 pretrained로 만든 뒤 checkpoint를 다시 불러와 이중 로딩한다(`timm/models/_factory.py:46, 146`) |
| Hugging Face | transformers 5.14.1의 `from_pretrained`는 meta device에서 빈 구조를 만들고 가중치를 바로 채워 중복 초기화를 피한다. `_fast_init`, `low_cpu_mem_usage`는 무시되는 옛 인자다(`modeling_utils.py:3899, 4240`) |
| torchvision | `weights=None`이 기본. 사용자 가중치는 `load_state_dict`로 따로 불러온다 |
| Lightning | `load_from_checkpoint`는 생성자를 다시 호출한 뒤 상태를 불러온다. 생성자 안의 pretrained 로딩을 자동으로 건너뛰지 않는다(`lightning/pytorch/core/saving.py:165`) |

anomalib STFPM처럼 생성자에 `pre_trained=True`를 고정한 것은 예외적인 설계다.

#### 3.7.3 단계별 정책

정책은 단계별로 다르다.

| 단계 | 정책 | 분기 비용 |
|---|---|---|
| anomalib | 원래 동작대로 항상 읽는다. 개발 환경에는 `D:\backbones`가 있고, 레퍼런스와 다른 경로를 만들지 않는다 | 없음 |
| native Lightning | 생성자는 구조만 만든다. 최소 학습에서만 생성 후 주입하고, 평가와 재개는 checkpoint만 쓴다 | 주입 한 줄과 목적 분기 |
| pure PyTorch | pretrained 개념이 없다. 배포 artifact만 strict로 불러온다 | 없음 |

분기 기준은 "checkpoint 파일이 있으면"이 아니라 사용자가 정한 실행 목적(`PURPOSE`)이다. 파일이 우연히 있거나 없을 때 동작이 조용히 바뀌지 않게 하기 위해서다. 목적별로 필요한 파일을 정하고, 없으면 오류로 멈춘다. 다른 출처로 몰래 대체하지 않는다.

#### 3.7.4 배포 요구

배포 머신에서 pretrained 파일 없이 동작하려면 다음이 모두 지켜져야 한다.

- **순수 모델 생성자는 pretrained 파일을 읽거나 내려받지 않는다.** anomalib `STFPMModel`을 그대로 이식하면 `TimmFeatureExtractor(pre_trained=True)`도 따라오고, 배포 머신에서 `timm.create_model(pretrained=True)`가 HF Hub 다운로드를 시도하다 실패한다. 이식할 때 이 부분을 바꾸고, 원본과의 의도된 차이로 이식 파일 주석과 DECISIONS에 기록한다(NFR-5).
- **배포 artifact에 추론에 필요한 상태를 모두 넣는다.** 학습하지 않는 teacher도 포함한다. STFPM은 teacher와 student, 후처리 상태가 필요하다. EfficientAD는 teacher, student, autoencoder, mean/std, quantile, 후처리 상태가 필요하다. 전처리 설정도 함께 넣는다.
- **검증**: `D:\backbones`에 접근할 수 없고 네트워크를 차단한 상태에서 Predictor 생성과 추론이 성공하고, 출력이 Wrapper 경로와 allclose다(6.4).

SPEC에는 아직 이 요구가 없다. 사람이 SPEC에 추가해야 한다(9장).

## 4. anomalib 2.0.0 평가에서의 적용

### 4.1 연결 지점과 제약

anomalib 2.0.0에는 전역 경로 설정이 없다. 경로는 클래스마다 생성자 인자이거나 코드에 박혀 있다.

| 대상 | 기본값 | 생성자로 바꿀 수 있나 |
|---|---|---|
| `MVTecAD(root=...)` | `./datasets/MVTecAD` | 예 |
| `Visa(root=...)` | `./datasets/visa` | 예. 단 `split_root`는 `root/visa_pytorch`로 고정 |
| `EfficientAd(imagenet_dir=...)` | `./datasets/imagenette` | 예 |
| EfficientAD teacher | `./pre_trained/efficientad_pretrained_weights` | 아니오 (메서드 안에 하드코딩) |
| STFPM backbone 가중치 | timm 기본 출처(HF Hub) | 아니오 (`backbone` 이름만 받음) |

### 4.2 다운로드 지점과 라이브러리 미수정 차단

anomalib 코드를 고치지 않고 다운로드를 막는 지점은 다음 다섯 곳이다.

| 다운로드 지점 | 차단 방법 | 구현 위치 |
|---|---|---|
| `MVTecAD.prepare_data` | 서브클래스에서 폴더 존재만 검사하고 없으면 오류 | `LocalMVTecAD` |
| `Visa.prepare_data`의 다운로드 분기 | 서브클래스에서 원본이 없으면 오류, 있으면 원래 `apply_cls1_split` 실행 | `LocalVisa` |
| `EfficientAd.prepare_pretrained_model` | 서브클래스에서 로컬 teacher 파일을 직접 로드 | `LocalEfficientAd` |
| `EfficientAd.prepare_imagenette_data` | 앞서 호출되는 `prepare_pretrained_model`에서 `imagenet_dir` 존재 검사 | `LocalEfficientAd` |
| timm pretrained 가중치 | `register_model`로 `pretrained_cfg_overlay={"file": ...}`인 backbone 이름 등록 | `local_resnet18` |

다섯 곳 모두 `examples/anomalib_old/_common.py`에 구현되어 있다(상태: 구현). STFPM `local_resnet18`의 소켓 차단 forward와 EfficientAD teacher state_dict 복원은 설치 시 확인했다(`anomalib-install-guide.md` 8절). `LocalEfficientAd`의 Imagenette 검사는 미검증이다.

timm 등록과 Local 서브클래스는 어느 후보를 고르든 anomalib 단계에서 필요하다. 후보가 바꾸는 것은 "이 클래스들이 경로를 어디서 받나"뿐이다.

### 4.3 후보별 USAGE

아래 예시는 모두 `examples/anomalib_old/` 안의 스크립트다. 학습과 평가 본문은 다음과 같이 같고, 후보마다 다른 것은 `datamodule`과 `model`을 만드는 부분이다.

```python
engine = Engine(default_root_dir=run_dir, max_epochs=MAX_EPOCHS, devices=1, deterministic="warn")
engine.fit(model=model, datamodule=datamodule)
engine.test(model=model, datamodule=datamodule)
```

#### 4.3.1 후보 1. 모듈 상수와 Local 서브클래스

상태: 구현. 경로는 `_common.py` 상단 상수이고, `_common.py` import 시 환경변수가 적용된다.

```python
# examples/anomalib_old/01_reference_stfpm.py: Phase 1 anomalib STFPM reference metrics
from _common import MVTEC_ROOT, LocalMVTecAD

from anomalib.models import Stfpm

datamodule = LocalMVTecAD(root=MVTEC_ROOT, category="bottle", train_batch_size=8, eval_batch_size=8, num_workers=0)
model = Stfpm(backbone="local_resnet18", visualizer=False)
```

EfficientAD와 VisA는 상수 이름만 달라진다.

```python
from _common import IMAGENETTE_DIR, VISA_ROOT, LocalEfficientAd, LocalVisa

datamodule = LocalVisa(root=VISA_ROOT, category="candle", train_batch_size=1, eval_batch_size=8, num_workers=0)
model = LocalEfficientAd(imagenet_dir=IMAGENETTE_DIR, model_size="small", visualizer=False)
```

장점은 코드가 가장 짧고 이미 검증된 경로라는 것이다. 단점은 경로가 `_common.py`에 박혀 있어 Lightning, PyTorch 단계에서 같은 값을 쓰려면 경로를 한 번 더 적어야 한다는 것이다. 이 이원화가 D-12가 막으려는 상황이다.

#### 4.3.2 후보 2. 환경변수와 `.env`

상태: 제안. 프로젝트 루트의 `.env`에 경로를 적고, `_common.py`가 import 시 읽는다.

```text
# .env
VISIONLAB_MVTEC=D:/datasets/mvtec
VISIONLAB_RESNET18=D:/backbones/resnet18-f37072fd.pth
HF_HUB_OFFLINE=1
```

```python
# examples/anomalib_old/_common.py (excerpt)
from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
MVTEC_ROOT = os.environ["VISIONLAB_MVTEC"]
RESNET18_WEIGHTS = os.environ["VISIONLAB_RESNET18"]
```

스크립트 쪽 코드는 후보 1과 같다. 머신마다 경로가 다를 때 `.env`만 바꾸면 되는 것이 장점이다. 단점은 부모 프로세스의 환경변수가 `.env`보다 우선하거나 뒤섞여 어떤 값이 쓰였는지 코드만 보고 알기 어렵다는 것이다. 모든 값이 문자열이고 구조(섹션)를 표현할 수 없다. anomalib이 이미 python-dotenv를 설치했으므로 이 단계에서는 의존성 문제가 없다.

#### 4.3.3 후보 3. `paths.toml`과 명시 인자 주입

상태: 제안. 프로젝트 루트의 `paths.toml` 하나를 진입점에서 한 번 읽어 모든 생성자에 넘긴다(D-12).

```toml
# paths.toml
[datasets]
mvtec = "D:/datasets/mvtec"
visa = "D:/datasets/visa"
visa_split = "D:/datasets/visa_anomalib/visa_pytorch"
imagenette = "D:/datasets/imagenette2/train"

[backbones]
resnet18 = "D:/backbones/resnet18-f37072fd.pth"
efficientad_small = "D:/backbones/efficientad_pretrained_weights/pretrained_teacher_small.pth"
efficientad_medium = "D:/backbones/efficientad_pretrained_weights/pretrained_teacher_medium.pth"
```

읽기와 검사는 표준 라이브러리만 쓰는 두 함수로 나눈다. 읽기는 부작용이 없고, 검사는 지금 필요한 키만 한다(2.4).

```python
import os
import tomllib


def load_paths(config_file):
    """Read paths.toml into a dict of sections."""
    with open(config_file, "rb") as f:
        return tomllib.load(f)


def require_paths(paths, keys):
    """Raise if any of the given 'section.name' paths is missing."""
    for key in keys:
        section, name = key.split(".")
        path = paths[section][name]
        if not os.path.exists(path):
            raise FileNotFoundError(f"{key}: {path}")
```

anomalib 단계에서는 timm 등록 함수가 경로를 인자로 받아 캡처한다. 전역 상수를 나중에 조회하지 않는다.

```python
# examples/anomalib_old/_common.py (excerpt)
def register_local_resnet18(weights_file):
    """Register 'local_resnet18' in timm with the given local weights file."""
    @register_model
    def local_resnet18(pretrained=False, **kwargs):
        kwargs["pretrained_cfg_overlay"] = {"file": weights_file, "hf_hub_id": None, "url": ""}
        return resnet18(pretrained=pretrained, **kwargs)
```

```python
# examples/anomalib_old/01_reference_stfpm.py
PATHS_FILE = os.path.join(PROJECT_ROOT, "paths.toml")

paths = load_paths(PATHS_FILE)
require_paths(paths, ["datasets.mvtec", "backbones.resnet18"])
register_local_resnet18(paths["backbones"]["resnet18"])
datamodule = LocalMVTecAD(root=paths["datasets"]["mvtec"], category="bottle", train_batch_size=8, eval_batch_size=8, num_workers=0)
model = Stfpm(backbone="local_resnet18", visualizer=False)
```

EfficientAD는 anomalib 생성자가 teacher 경로를 받지 않으므로 `LocalEfficientAd`가 키워드 인자 `teacher_file`을 추가로 받는다. 나머지 인자는 anomalib과 같다.

```python
model = LocalEfficientAd(imagenet_dir=paths["datasets"]["imagenette"], teacher_file=paths["backbones"]["efficientad_small"],
                         model_size="small", visualizer=False)
```

장점은 경로 정의가 한 곳뿐이고, 같은 `paths.toml`을 세 단계가 그대로 읽을 수 있다는 것이다. 단점은 스크립트마다 `bootstrap()` 호출과 `paths` 꺼내기가 몇 줄 늘어난다는 것이다.

#### 4.3.4 후보 4. YAML과 anomalib CLI

상태: 제안, 미검증. anomalib CLI는 LightningCLI 기반이라 YAML의 `class_path`로 생성자를 호출한다.

```yaml
# examples/anomalib_old/configs/stfpm_mvtec.yaml
model:
  class_path: anomalib.models.Stfpm
  init_args:
    backbone: local_resnet18
data:
  class_path: _common.LocalMVTecAD
  init_args:
    root: D:/datasets/mvtec
    category: bottle
    num_workers: 0
```

`local_resnet18` 등록과 `LocalMVTecAD` import는 CLI 프로세스 안에서 먼저 일어나야 한다. `python -m anomalib.cli.cli`로는 `_common.py`가 실행되지 않으므로 진입 스크립트를 따로 둔다.

```python
# examples/anomalib_old/cli.py: anomalib CLI with local assets registered first
import _common  # noqa: F401  registers local_resnet18, sets offline env

from anomalib.cli.cli import AnomalibCLI

if __name__ == "__main__":
    AnomalibCLI()
```

장점은 실행 설정 전체가 파일로 남는다는 것이다. 단점은 경로가 YAML마다 반복되어 D-12의 단일 해석과 충돌한다는 것이다. `_common.LocalMVTecAD`를 `class_path`로 찾으려면 실행 폴더가 `sys.path`에 있어야 한다. 이 후보는 Phase 1의 비교 목적에는 과하다.

#### 4.3.5 후보 5. 설정 파일과 작은 조립 함수

상태: 제안. 후보 3의 `paths.toml`을 쓰고, 이름에서 객체를 만드는 분기를 명시 인자 함수 하나에 모은다. 범용 factory가 아니라 이 비교 스크립트들이 쓰는 조합만 다룬다.

```python
# examples/anomalib_old/_common.py (excerpt)
def build_datamodule(paths, data, category, train_batch_size, eval_batch_size, seed):
    """Build the local anomalib DataModule for 'mvtec' or 'visa'."""
    if data == "mvtec":
        return LocalMVTecAD(root=paths["datasets"]["mvtec"], category=category,
                            train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, num_workers=0,
                            val_split_mode="from_test", val_split_ratio=0.4, seed=seed)
    if data == "visa":
        return LocalVisa(root=paths["datasets"]["visa"], split_root=paths["datasets"]["visa_split"], category=category,
                         train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, num_workers=0,
                         val_split_mode="from_test", val_split_ratio=0.4, seed=seed)
    raise ValueError(data)
```

```python
paths = load_paths(PATHS_FILE)
datamodule = build_datamodule(paths, DATA, CATEGORY, train_batch_size=8, eval_batch_size=8, seed=SEED)
model = Stfpm(backbone="local_resnet18", visualizer=False)
```

장점은 데이터별 분기가 한 곳에 모이고, 함수 시그니처에 필요한 값이 다 드러난다는 것이다. 단점은 조합이 늘면 함수가 커지고, 인자 기본값이 anomalib 기본값과 어긋날 위험이 있다는 것이다.

#### 4.3.6 후보 6. 자체 registry와 `create_*` 함수

상태: 제안. 클래스를 이름으로 등록하고 `create_dataset(name)`, `create_model(name)`으로 만든다.

```python
DATASETS = {"mvtec": LocalMVTecAD, "visa": LocalVisa}
MODELS = {"stfpm": Stfpm, "efficientad": LocalEfficientAd}


def create_dataset(name, root, category, train_batch_size, eval_batch_size):
    """Create a registered DataModule by name."""
    return DATASETS[name](root=root, category=category, train_batch_size=train_batch_size,
                          eval_batch_size=eval_batch_size, num_workers=0)
```

```python
datamodule = create_dataset("mvtec", paths["datasets"]["mvtec"], "bottle", 8, 8)
model = create_model("stfpm", backbone="local_resnet18")
```

두 클래스의 생성자가 서로 달라(`LocalVisa`의 `split_root`, `LocalEfficientAd`의 `imagenet_dir`) 공통 시그니처로 묶으면 `**kwargs`나 분기가 다시 필요해진다. 데이터 2개, 모델 2개에서는 registry의 이득보다 이름 뒤에 경로와 인자가 숨는 손해가 크다. D-4가 배제한 방식이다.

### 4.4 검증

anomalib 단계에서 확인할 항목은 다음과 같다.

- `LocalMVTecAD`, `LocalVisa`가 존재하지 않는 `root`에서 `FileNotFoundError`를 낸다.
- 소켓 차단 상태에서 `Stfpm(backbone="local_resnet18")` 생성과 forward가 성공한다.
- `LocalEfficientAd`가 teacher 파일이나 `imagenet_dir`이 없을 때 학습 시작 전에 멈춘다.
- 분할 목록과 mask가 레퍼런스 실행과 같다(D-13).
- 평가한 모델 상태(마지막 epoch)가 기록과 같다.

### 4.5 bootstrap 기반 전체 스크립트

상태: 구현(미검증). 3.6의 여섯 단계로 쓴 Phase 1 레퍼런스 스크립트이며 `examples/anomalib_old/01_reference_stfpm.py`가 이 구조로 바뀌어 있다. `_common.py`는 경로 상수를 없애고 `PATHS_FILE`, `register_local_resnet18(weights_file)`, Local 서브클래스를 제공한다. 실제 파일은 DATA에 따라 `LocalMVTecAD`와 `LocalVisa`를 나눠 호출한다.

```python
# examples/anomalib_old/01_reference_stfpm.py: Phase 1 anomalib STFPM reference metrics on local assets
import os

from _common import OUTPUTS_DIR, PROJECT_ROOT, LocalMVTecAD, register_local_resnet18

from anomalib.engine import Engine
from anomalib.models import Stfpm
from lightning.pytorch import seed_everything
from visionlab.bootstrap import bootstrap, save_run


PATHS_FILE = os.path.join(PROJECT_ROOT, "paths.toml")
CATEGORY = "bottle"
SEED = 42
MAX_EPOCHS = 10


def main():
    """Train and test anomalib Stfpm on local MVTec with a local ResNet18 teacher."""
    # 1. bootstrap
    run_dir = os.path.join(OUTPUTS_DIR, "anomalib", "01_reference_stfpm", f"mvtec_{CATEGORY}")
    ctx = bootstrap(PATHS_FILE, required=["datasets.mvtec", "backbones.resnet18"], run_dir=run_dir)
    paths = ctx["paths"]
    seed_everything(SEED, workers=True)

    # 2. data
    datamodule = LocalMVTecAD(root=paths["datasets"]["mvtec"], category=CATEGORY,
                              train_batch_size=8, eval_batch_size=8, num_workers=0,
                              val_split_mode="from_test", val_split_ratio=0.4, seed=SEED)

    # 3. model
    register_local_resnet18(paths["backbones"]["resnet18"])
    model = Stfpm(backbone="local_resnet18", visualizer=False)

    # 4. engine
    engine = Engine(default_root_dir=run_dir, max_epochs=MAX_EPOCHS, devices=1, deterministic="warn")

    # 5. run
    engine.fit(model=model, datamodule=datamodule)
    results = engine.test(model=model, datamodule=datamodule)

    # 6. save
    save_run(ctx, seed=SEED, metrics=results[0])


if __name__ == "__main__":
    main()
```

EfficientAD와 VisA는 1~3단계만 달라진다. `LocalEfficientAd`는 4.3.3처럼 teacher 파일을 속성으로 받는다고 가정한다.

```python
ctx = bootstrap(PATHS_FILE, required=["datasets.visa", "datasets.imagenette", "backbones.efficientad_small"], run_dir=run_dir)
paths = ctx["paths"]
datamodule = LocalVisa(root=paths["datasets"]["visa"], split_root=paths["datasets"]["visa_split"], category="candle",
                       train_batch_size=1, eval_batch_size=8, num_workers=0)
model = LocalEfficientAd(imagenet_dir=paths["datasets"]["imagenette"], teacher_file=paths["backbones"]["efficientad_small"],
                         model_size="small", visualizer=False)
```

`datasets.visa_split`는 첫 실행 전에 없을 수 있으므로 `required`에 넣지 않는다(2.2).

## 5. native Lightning 평가에서의 적용

### 5.1 연결 지점과 제약

이 단계에서는 anomalib을 import하지 않는다. `_common.py`도 쓸 수 없다. 연결 지점은 visionlab의 다음 세 곳이다(상태: 제안, D-15).

| 연결 지점 | 경로 인자 | 제약 |
|---|---|---|
| `MVTecLitData`, `VisALitData` | `root` | 생성자가 anomalib `MVTecAD`, `Visa`와 같아야 함(FR-2). 다운로드 코드 자체가 없음 |
| `STFPMLitModel` | 없음(`backbone` 이름만) | 생성자 인자 6개가 anomalib `Stfpm`과 같아야 함(FR-6). `weights_path` 추가 불가(D-5) |
| `EfficientAdLitModel` | `imagenet_dir` | 인자 11개가 anomalib과 같아야 함. teacher 경로 인자 없음 |

visionlab LitData는 처음부터 다운로드 코드를 두지 않으므로, 데이터 쪽 다운로드 차단은 "없으면 오류"만으로 충분하다. 문제는 생성자가 경로를 받지 않는 backbone과 teacher다.

### 5.2 로컬 가중치 주입과 checkpoint 복원 순서

3.7의 정책에 따라 visionlab 생성자는 구조만 만들고 pretrained 파일을 읽지 않는다. D-5를 지키면서 최소 학습에서 로컬 가중치를 넣는 방법은 두 가지다.

- **방법 A. 생성 후 주입**: 진입점에서 `load_backbone_weights(model.model.teacher, file)`로 넣는다. codex가 추천했고 이 문서의 기본 가정이다. 생성자가 파일을 읽지 않으므로 평가와 배포에서 pretrained가 없어도 된다.
- **방법 B. timm 이름 등록**: anomalib 단계처럼 `local_resnet18`을 timm에 등록하고 `backbone="local_resnet18"`을 넘긴다. 시그니처는 그대로지만 순수 모델이 timm registry와 등록 순서에 의존하고, 생성자가 pretrained를 읽으므로 3.7.4의 배포 요구와 충돌한다.

생성 후 주입이 anomalib과 같은 RNG 소비 순서를 만드는지는 미검증이다. anomalib은 timm이 teacher를 무작위 초기화한 뒤 가중치를 덮어쓰므로, visionlab도 teacher와 student를 같은 순서로 만들면 같을 것으로 추정한다.

목적별 순서는 다음과 같다. checkpoint를 쓰는 목적에서는 pretrained를 읽지 않는다.

```text
최소 학습:   생성 -> backbone/teacher 주입 (없으면 오류) -> fit
학습 재개:   생성 -> fit(ckpt_path) 로 복원 (teacher, 통계 재초기화 금지, FR-15)
평가:        생성 -> test(ckpt_path) 로 복원 (pretrained 불필요)
```

D-5는 인자 이름, 순서, 기본값의 동일성만 요구하므로 이 설계와 충돌하지 않는다. 다만 "생성 시 pretrained를 읽는다"는 초기화 동작은 anomalib과 다르다. "시그니처는 같고 초기화 동작은 다르다"를 DECISIONS에 명시할지는 미결이다(9장).

최소 학습에서 주입을 빠뜨리면 무작위 teacher로 학습이 진행된다. 주입 여부를 학습 시작 전에 검사해 막는다.

### 5.3 후보별 USAGE

아래 예시는 `examples/` 바로 아래의 스크립트다(D-15). 실행 본문은 다음과 같이 같다.

```python
trainer = Trainer(default_root_dir=run_dir, max_epochs=MAX_EPOCHS, devices=1, deterministic="warn")
trainer.fit(model, datamodule=datamodule)
trainer.test(model, datamodule=datamodule)
```

#### 5.3.1 후보 1. 모듈 상수와 직접 생성

상태: 제안. `_common.py`를 쓸 수 없으므로 상수를 스크립트 상단에 다시 적는다.

```python
# examples/04_train_stfpm.py: Phase 4 STFPM with visionlab LitData, LitModel and native Trainer
MVTEC_ROOT = "D:/datasets/mvtec"
RESNET18_WEIGHTS = "D:/backbones/resnet18-f37072fd.pth"

datamodule = MVTecLitData(root=MVTEC_ROOT, category="bottle", train_batch_size=8, eval_batch_size=8, num_workers=0)
model = STFPMLitModel(backbone="resnet18")
load_backbone_weights(model, RESNET18_WEIGHTS)
```

경로가 `_common.py`와 이 스크립트 두 곳에 적힌다. 스크립트가 늘면 그만큼 복사된다.

#### 5.3.2 후보 2. 환경변수와 `.env`

상태: 제안. anomalib 단계와 같은 `.env`를 읽는다.

```python
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
datamodule = MVTecLitData(root=os.environ["VISIONLAB_MVTEC"], category="bottle", num_workers=0)
model = STFPMLitModel(backbone="resnet18")
load_backbone_weights(model, os.environ["VISIONLAB_RESNET18"])
```

python-dotenv는 Lightning 환경에는 있지만 visionlab의 의존성으로 새로 생긴다.

#### 5.3.3 후보 3. `paths.toml`과 명시 인자 주입

상태: 제안. anomalib 단계와 같은 파일, 같은 함수를 쓴다. `load_paths`, `require_paths`는 표준 라이브러리만 쓰므로 visionlab 안에 두어도 의존성이 늘지 않는다. 위치는 D-15에 아직 없다(9장).

```python
paths = load_paths(PATHS_FILE)
require_paths(paths, ["datasets.mvtec", "backbones.resnet18"])
datamodule = MVTecLitData(root=paths["datasets"]["mvtec"], category="bottle", train_batch_size=8, eval_batch_size=8, num_workers=0)
model = STFPMLitModel(backbone="resnet18")
load_backbone_weights(model, paths["backbones"]["resnet18"])
```

VisA는 이 단계에서 작업 복사본이 필요 없을 수 있다. visionlab `VisADataset`이 원본과 `split_csv`를 직접 읽는다면 `datasets.visa`만 쓴다. 이 경우 anomalib 단계와 물리 경로가 달라지므로 7.3의 sample ID 대조가 필요하다.

#### 5.3.4 후보 4. YAML과 LightningCLI

상태: 제안. Phase 5의 `examples/cli.py`와 4계층 YAML(FR-13)을 쓴다. 경로를 YAML마다 적지 않도록 LightningCLI의 hook에서 `paths.toml` 값을 주입한다.

```yaml
# src/visionlab/anomaly_detection/configs/data/mvtec.yaml
data:
  class_path: visionlab.anomaly_detection.data.mvtec.lightning_data.MVTecLitData
  init_args:
    category: bottle
    num_workers: 0
```

```python
# examples/cli.py (excerpt)
class LitCLI(LightningCLI):
    """LightningCLI that fills dataset roots from paths.toml."""

    def before_instantiate_classes(self):
        paths = load_paths(PATHS_FILE)
        config = self.config[self.subcommand]
        config.data.init_args.root = paths["datasets"][DATA_KEYS[config.data.class_path]]
```

backbone 가중치는 생성자로 넘길 수 없으므로 `before_fit` 등 hook에서 `load_backbone_weights`를 호출해야 한다. 어느 hook이 적절한지는 미검증이다.

#### 5.3.5 후보 5. 설정 파일과 작은 조립 함수

상태: 제안. anomalib 단계의 `build_datamodule`과 같은 모양이지만 반환하는 클래스가 visionlab LitData다. 두 단계의 조립 함수는 파일이 달라야 한다(`_common.py`는 anomalib을 import하므로).

```python
datamodule = build_lit_datamodule(paths, DATA, CATEGORY, train_batch_size=8, eval_batch_size=8, seed=SEED)
model = build_lit_model(paths, MODEL)  # creates the LitModel and loads the local backbone or teacher
```

#### 5.3.6 후보 6. 자체 registry와 `create_*` 함수

상태: 제안. registry가 visionlab 패키지 안에 생기면 pure PyTorch 단계에도 따라간다.

```python
datamodule = create_dataset("mvtec", root=paths["datasets"]["mvtec"], category="bottle")
model = create_model("stfpm", backbone="resnet18")
load_backbone_weights(model, paths["backbones"]["resnet18"])
```

### 5.4 검증

Lightning 단계에서 확인할 항목은 다음과 같다.

- `inspect.signature`로 LitData, LitModel 생성자가 anomalib과 같다(FR-2, FR-6).
- anomalib import를 차단해도 실행된다(NFR-1).
- 최종 모델 입력이 anomalib 단계와 같다(FR-17).
- `ckpt_path` 복원 후 같은 출력이 나오고 backbone 파일 없이 평가된다(FR-12).
- 범주별 Image, Pixel AUROC가 레퍼런스와 0.01 이내다.

### 5.5 bootstrap 기반 전체 스크립트

상태: 제안. anomalib 단계(4.5)와 1단계, 6단계가 같고, 2~4단계의 클래스만 visionlab으로 바뀐다. backbone 가중치는 5.2의 방법 A(생성 후 주입)를 가정한다.

```python
# examples/04_train_stfpm.py: Phase 4 STFPM with visionlab LitData, LitModel and native Trainer
import os

from lightning.pytorch import Trainer, seed_everything
from visionlab.anomaly_detection.data.mvtec.lightning_data import MVTecLitData
from visionlab.anomaly_detection.models.stfpm.lightning_model import STFPMLitModel
from visionlab.anomaly_detection.models.components.feature_extractors.timm import load_backbone_weights
from visionlab.bootstrap import bootstrap, save_run


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATHS_FILE = os.path.join(PROJECT_ROOT, "paths.toml")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
CATEGORY = "bottle"
SEED = 42
MAX_EPOCHS = 10


def main():
    """Train and test visionlab STFPM on local MVTec with a native Trainer."""
    # 1. bootstrap
    run_dir = os.path.join(OUTPUTS_DIR, "04_train_stfpm", f"mvtec_{CATEGORY}")
    ctx = bootstrap(PATHS_FILE, required=["datasets.mvtec", "backbones.resnet18"], run_dir=run_dir)
    paths = ctx["paths"]
    seed_everything(SEED, workers=True)

    # 2. data
    datamodule = MVTecLitData(root=paths["datasets"]["mvtec"], category=CATEGORY,
                              train_batch_size=8, eval_batch_size=8, num_workers=0,
                              val_split_mode="from_test", val_split_ratio=0.4, seed=SEED)

    # 3. model: structure only, the constructor reads no weight file
    model = STFPMLitModel(backbone="resnet18", visualizer=False)
    load_backbone_weights(model.model.teacher, paths["backbones"]["resnet18"])

    # 4. trainer
    trainer = Trainer(default_root_dir=run_dir, max_epochs=MAX_EPOCHS, devices=1, deterministic="warn")

    # 5. run
    trainer.fit(model, datamodule=datamodule)
    results = trainer.test(model, datamodule=datamodule)

    # 6. save
    save_run(ctx, seed=SEED, metrics=results[0])


if __name__ == "__main__":
    main()
```

최소 학습과 checkpoint 평가를 한 스크립트에서 다루려면 상단 상수 `PURPOSE`로 나눈다. 분기 기준은 checkpoint 파일의 유무가 아니라 목적이다(3.7.3). 평가에서는 pretrained를 `required`에서 빼고 주입하지 않는다.

```python
PURPOSE = "train"  # "train" or "evaluate"

required = ["datasets.mvtec", "backbones.resnet18"] if PURPOSE == "train" else ["datasets.mvtec"]
ctx = bootstrap(PATHS_FILE, required=required, run_dir=run_dir)
model = STFPMLitModel(backbone="resnet18", visualizer=False)
if PURPOSE == "train":
    load_backbone_weights(model.model.teacher, ctx["paths"]["backbones"]["resnet18"])
    trainer.fit(model, datamodule=datamodule)
    results = trainer.test(model, datamodule=datamodule)
elif PURPOSE == "evaluate":
    results = trainer.test(model, datamodule=datamodule, ckpt_path=CHECKPOINT_FILE)
else:
    raise ValueError(PURPOSE)
```

## 6. pure PyTorch 평가에서의 적용

### 6.1 연결 지점과 제약

이 단계는 배포 환경이다. anomalib과 Lightning을 import하지 않고, 표준 라이브러리와 PyTorch(및 torchvision 등 모델이 원래 요구하는 것)만 쓴다. 연결 지점은 다음과 같다(상태: 제안).

| 연결 지점 | 경로 인자 | 비고 |
|---|---|---|
| `MVTecDataset`, `VisADataset` | `root` | 평가 데이터. 독립 추론에는 불필요 |
| `STFPMModel`, `EfficientAdModel` | 없음 | 생성자는 구조만 만들고 pretrained를 읽지 않음. 가중치는 배포 artifact로 복원 |
| Predictor | 배포 artifact 파일 | FR-14 |

배포 머신에는 학습된 checkpoint(배포 artifact)만 있고 `D:\backbones`는 없다. 이 상태에서 오류가 나면 안 된다. 요구 사항은 3.7.4에 정리했다.

### 6.2 의존성 경계

pure PyTorch 단계는 두 사용 형태가 다르다.

- **평가**: 개발 환경에서 테스트 데이터로 AUROC를 낸다. 데이터 경로와 checkpoint가 필요하다.
- **배포 추론**: 다른 머신에서 이미지 한 장씩 추론한다. 배포 artifact와 입력 이미지만 필요하다. pretrained 파일과 데이터셋은 없다.

배포 추론 코드는 설정 파일 형식에 의존하지 않는 것이 좋다. Predictor는 해석된 경로 문자열만 받고, 설정 파일을 읽는 것은 호출하는 스크립트의 몫이다. `tomllib`, `json`, `os.environ`은 표준 라이브러리라 허용되지만, python-dotenv와 PyYAML은 새 의존성이다.

### 6.3 후보별 USAGE

평가 본문은 다음과 같이 같다.

```python
loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
model.eval()
with torch.no_grad():
    outputs = [model(pre_processor(batch["image"])) for batch in loader]
```

#### 6.3.1 후보 1. 모듈 상수와 직접 생성

상태: 제안. 상수를 스크립트에 적는다. 의존성은 늘지 않는다.

```python
# examples/06_predict.py: Phase 6 pure PyTorch evaluation without Lightning or anomalib
MVTEC_ROOT = "D:/datasets/mvtec"
CHECKPOINT_FILE = os.path.join(OUTPUTS_DIR, "stfpm", "mvtec_bottle", "last.pt")

dataset = MVTecDataset(root=MVTEC_ROOT, category="bottle", split="test", transform=MVTecTransform())
model = STFPMModel(backbone="resnet18")
model.load_state_dict(torch.load(CHECKPOINT_FILE, map_location="cpu", weights_only=True))
```

#### 6.3.2 후보 2. 환경변수와 `.env`

상태: 제안. 환경변수만 쓰면 표준 라이브러리로 충분하지만, `.env` 파일을 읽으려면 python-dotenv가 배포 의존성이 된다(△).

```python
dataset = MVTecDataset(root=os.environ["VISIONLAB_MVTEC"], category="bottle", split="test", transform=MVTecTransform())
```

#### 6.3.3 후보 3. `paths.toml` 로딩과 해석된 경로 전달

상태: 제안. 스크립트가 `paths.toml`을 읽고, Dataset과 Predictor에는 문자열만 넘긴다. 배포 머신에 `paths.toml`이 없어도 Predictor는 동작한다.

```python
paths = load_paths(PATHS_FILE)
dataset = MVTecDataset(root=paths["datasets"]["mvtec"], category="bottle", split="test", transform=MVTecTransform())
predictor = Predictor(checkpoint_file=CHECKPOINT_FILE)
```

#### 6.3.4 후보 4. YAML·JSON 읽기와 LightningCLI를 쓸 수 없는 범위

상태: 제안. LightningCLI는 Lightning에 속하므로 이 단계에서 쓸 수 없다(X). Phase 5의 YAML을 다시 쓰려면 PyYAML이 필요하고 `class_path` 해석을 직접 구현해야 한다. 학습 때 저장한 최종 설정을 JSON으로 남겼다면 표준 `json`으로 읽을 수 있다.

```python
with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
    config = json.load(f)
model = STFPMModel(backbone=config["model"]["backbone"])
```

#### 6.3.5 후보 5. 설정 파일과 작은 조립 함수

상태: 제안. Lightning 단계와 별도로, 순수 구성요소만 반환하는 조립 함수를 둔다.

```python
dataset = build_dataset(paths, DATA, CATEGORY, split="test")
model = build_model(MODEL, checkpoint_file=CHECKPOINT_FILE)
```

#### 6.3.6 후보 6. 순수 구성요소만 쓰는 registry와 `create_*` 함수

상태: 제안. registry 자체는 표준 Python이지만, 배포 코드가 등록 순서와 import 부작용에 의존하게 된다(△). 등록이 일어나려면 모든 모델 모듈을 미리 import해야 한다.

```python
model = create_model("stfpm", backbone="resnet18")
model.load_state_dict(torch.load(CHECKPOINT_FILE, map_location="cpu", weights_only=True))
```

### 6.4 검증

pure PyTorch 단계에서 확인할 항목은 다음과 같다.

- anomalib, Lightning import를 차단해도 실행된다(NFR-1, NFR-2).
- checkpoint 파일이 없으면 무작위 가중치로 진행하지 않고 오류를 낸다(NFR-3).
- `D:\backbones`에 접근할 수 없고 네트워크를 차단한 상태에서 Predictor 생성과 추론이 성공한다(3.7.4).
- 배포 artifact를 strict로 불러올 때 누락 키가 없다. teacher 키가 빠진 artifact는 실패한다.
- 같은 입력에서 Lightning 경로와 출력이 allclose다(FR-14).
- 범주별 metric이 Lightning 단계와 같다.

### 6.5 bootstrap 기반 전체 스크립트

상태: 제안. 이 단계에는 Trainer가 없으므로 4단계에서 DataLoader와 처리 구성요소로 평가 루프를 구성한다. 가중치는 학습 산출물인 checkpoint에서 복원하므로 `required`에는 데이터 키만 넣는다. checkpoint 형식(Lightning `.ckpt`에서 순수 state_dict를 꺼낼지, 별도로 export할지)은 Phase 6에서 정한다. 아래는 export한 배포 artifact를 가정한다. artifact에는 teacher를 포함한 모델 전체와 후처리 상태가 들어 있으므로(3.7.4) pretrained 파일이 필요 없다.

```python
# examples/06_predict.py: Phase 6 pure PyTorch evaluation without Lightning or anomalib
import os

import torch
from torch.utils.data import DataLoader
from visionlab.anomaly_detection.common.evaluator import Evaluator
from visionlab.anomaly_detection.common.post_processor import PostProcessor
from visionlab.anomaly_detection.common.pre_processor import PreProcessor
from visionlab.anomaly_detection.data.mvtec.dataset import MVTecDataset, MVTecTransform
from visionlab.anomaly_detection.models.stfpm.model import STFPMModel
from visionlab.bootstrap import bootstrap, save_run


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATHS_FILE = os.path.join(PROJECT_ROOT, "paths.toml")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_FILE = os.path.join(OUTPUTS_DIR, "04_train_stfpm", "mvtec_bottle", "stfpm_state.pt")
CATEGORY = "bottle"
SEED = 42


def main():
    """Evaluate an exported STFPM state on local MVTec with pure PyTorch."""
    # 1. bootstrap
    run_dir = os.path.join(OUTPUTS_DIR, "06_predict", f"mvtec_{CATEGORY}")
    ctx = bootstrap(PATHS_FILE, required=["datasets.mvtec"], run_dir=run_dir)
    paths = ctx["paths"]
    torch.manual_seed(SEED)

    # 2. data
    dataset = MVTecDataset(root=paths["datasets"]["mvtec"], category=CATEGORY, split="test", transform=MVTecTransform())

    # 3. model
    state = torch.load(CHECKPOINT_FILE, map_location="cpu", weights_only=True)
    model = STFPMModel(backbone="resnet18")
    model.load_state_dict(state["model"])
    model.eval()

    # 4. evaluation loop parts
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    pre_processor = PreProcessor()
    post_processor = PostProcessor()
    post_processor.load_state_dict(state["post_processor"])
    evaluator = Evaluator()

    # 5. run
    with torch.no_grad():
        for image, target in loader:
            output = post_processor(model(pre_processor(image)))
            evaluator.update(output, target)
    metrics = evaluator.compute()

    # 6. save
    save_run(ctx, seed=SEED, metrics=metrics)


if __name__ == "__main__":
    main()
```

배포 추론에서는 1단계와 2단계가 빠진다. `bootstrap()`과 `paths.toml` 없이 checkpoint 파일과 입력 이미지만으로 Predictor를 만든다(6.2).

```python
predictor = Predictor(checkpoint_file=CHECKPOINT_FILE)
output = predictor.predict(image)
```

## 7. 세 단계의 오프라인 보장과 데이터 동등성 판정

### 7.1 자산 누락과 다운로드 시도 시 동작

각 단계에서 자산이 없을 때 기대하는 동작은 다음과 같다.

| 상황 | anomalib 단계 | Lightning 단계 | PyTorch 단계 |
|---|---|---|---|
| 데이터 폴더 없음 | Local 서브클래스가 `FileNotFoundError` | LitData 또는 `require_paths`가 오류 | Dataset 또는 `require_paths`가 오류 |
| backbone 파일 없음 | `local_resnet18`이 `FileNotFoundError` (STFPM은 평가에서도) | 최소 학습에서만 `load_backbone_weights`가 오류. 평가, 재개는 정상 | 정상 (읽지 않음) |
| teacher 파일 없음 | `LocalEfficientAd`가 학습과 재개에서 오류. test, predict는 정상 | 최소 학습에서만 오류 | 정상 (읽지 않음) |
| checkpoint 없음 | 평가 시 `ckpt_path` 오류 | 평가, 재개 시 `ckpt_path` 오류 | Predictor가 오류 |

### 7.2 소켓 차단 실행

환경변수는 다운로드를 완전히 막지 못한다. 다운로드가 일어나지 않음은 소켓을 차단한 프로세스에서 실행해 증명한다. 설치 검증(`anomalib-install-guide.md` 8절)과 같은 방법이다.

```python
import socket


def block_network(*args, **kwargs):
    """Raise on any outgoing connection attempt."""
    raise RuntimeError("network access is blocked")


socket.socket.connect = block_network
socket.create_connection = block_network
```

3.6의 `bootstrap(offline=True)`가 같은 차단을 한다. 평가 스크립트에서만 쓰고 배포 추론 코드(Predictor)에는 넣지 않는다.

### 7.3 가중치 해시, manifest, D-13 대조

같은 경로를 썼다는 것만으로 같은 데이터라고 판정하지 않는다(D-13). 세 단계 모두 다음을 기록하고 대조한다.

- **가중치**: 사용한 파일의 SHA-256.
- **데이터**: 파일 목록, 라벨, mask, split을 manifest로 고정한다.
- **VisA**: anomalib 단계는 `visa_pytorch` 복사본을, visionlab은 원본을 읽을 수 있어 물리 경로가 다르다. 원본 기준 sample ID로 대응시킨다.
- **최종 입력**: 모델에 들어가는 텐서를 대조한다(FR-17).

## 8. 비교와 권고

### 8.1 단계 x 후보 적용 가능성

단계별 적용 가능성을 다시 모으면 다음과 같다. 칸의 내용은 그 단계에서 경로를 얻는 방법이다.

| 후보 | anomalib | Lightning | PyTorch |
|---|---|---|---|
| 1. 모듈 상수 | O `_common.py` 상수 | O 스크립트에 상수 복사 | O 스크립트에 상수 복사 |
| 2. 환경변수와 `.env` | O | O (python-dotenv 추가) | △ `.env`는 의존성 추가 |
| 3. `paths.toml` | O | O | O (스크립트만 읽음) |
| 4. YAML과 CLI | O (진입 스크립트 필요) | O (hook으로 주입) | X |
| 5. 조립 함수 | O | O (별도 파일) | O (별도 파일) |
| 6. registry | O | O | △ 배포 코드에 등록 부작용 |

### 8.2 기준별 비교

1.2의 기준으로 비교하면 다음과 같다. 상, 중, 하는 이 프로젝트 기준의 상대 평가다.

| 후보 | 추적성 | 오프라인 보장 | 의존성 경계 | 재현성 | DECISIONS 정합성 |
|---|---|---|---|---|---|
| 1. 모듈 상수 | 상 | 상 | 상 | 하 (경로 복사본이 여러 곳) | 중 (D-12와 어긋남) |
| 2. 환경변수와 `.env` | 하 | 상 | 중 | 중 | 중 |
| 3. `paths.toml` | 상 | 상 | 상 | 상 | 상 (D-12 그대로) |
| 4. YAML과 CLI | 중 | 상 | 하 | 상 | 중 (경로 반복) |
| 5. 조립 함수 | 중 | 상 | 상 | 상 | 중 (D-4 문구 정리 필요) |
| 6. registry | 하 | 상 | 중 | 중 | 하 (D-4 배제) |

오프라인 보장은 모든 후보가 같다. 실제로 다운로드를 막는 것은 Local 서브클래스, 누락 시 오류, 소켓 차단이고 경로 저장 방식이 아니기 때문이다.

### 8.3 권고

세 검토자 모두 후보 3을 기본으로 추천했다.

**결정(2026-10-23, 사용자)**: 후보 3을 채택한다. 경로는 `paths.toml`에 두고, 스크립트는 `bootstrap()`으로 시작한다(3.6). 9장의 나머지 항목은 아직 미결이다. 확정 내용은 사람이 `specs/DECISIONS.md`에 옮긴다.

- `paths.toml` 하나를 진입점에서 한 번 읽고, 세 단계 모두 해석된 문자열을 생성자에 넘긴다.
- 읽기, 자산 검사, 다운로드 차단, 실행 폴더 생성은 `bootstrap()` 하나로 모으고, 평가 스크립트는 3.6의 여섯 단계로 쓴다.
- anomalib 단계의 Local 서브클래스와 timm 등록은 유지하되, 경로를 상수가 아니라 인자로 받게 한다.
- Lightning 단계의 YAML(후보 4)은 Phase 5에서 학습 설정용으로 쓰고, 경로는 `paths.toml`에서 주입한다.
- 반복이 실제로 생기면 후보 5의 작은 조립 함수를 단계별 파일에 둔다. 자체 registry(후보 6)는 두지 않는다.
- 다운로드 차단은 후보와 무관하게 누락 시 오류와 소켓 차단 검증으로 보장한다.
- visionlab 생성자는 pretrained를 읽지 않는다. 최소 학습에서만 생성 후 주입하고, 평가와 배포는 checkpoint만 쓴다(3.7). 배포 머신에 pretrained가 없어도 동작해야 한다.

## 9. 미결 결정과 바뀌는 DECISIONS·SPEC·AGENTS 항목

후보 3을 채택하면 다음을 정해야 한다.

| 미결 사항 | 선택지 | 관련 문서 |
|---|---|---|
| 로컬 backbone 주입 방식 | 생성 후 주입(codex 추천, 문서 기본 가정) 또는 timm 이름 등록. timm 이름 등록은 3.7.4 배포 요구와 충돌. 생성자에 경로 인자 추가는 D-5, INTENT 위반 | D-5, SPEC |
| `bootstrap`, `save_run` 위치 | 결정: `src/visionlab/bootstrap.py`. 구현됨. `src/visionlab/`에 `__init__.py`와 `pyproject.toml`은 아직 없고, 설치 방식은 미결 | D-15 |
| `bootstrap()`의 소켓 차단 기본값 | 기본 켜짐(`offline=True`), 또는 검증 실행에서만 켬 | NFR-3 |
| VisA 작업 복사본의 키 | 결정: `[datasets]` 안의 `visa_split`. 구현됨 | D-12, D-17 |
| `_common.py`의 import 시 부작용 | 현재처럼 유지, 또는 명시 초기화 함수로 이동 | D-15 |
| 적용 시점 | 결정: 지금 적용. `paths.toml`, `bootstrap.py`, `_common.py`, `01_reference_stfpm.py`가 반영됨. 코드 실행 검증은 아직 안 함 | PLAN |
| visionlab 설치 방식 | `pip install -e .`(`pyproject.toml`, `--no-build-isolation --no-deps`) 또는 현재의 `sys.path` 추가 유지 | D-1, D-14 |
| D-4 문구 | "registry, factory framework 배제"를 "범용 framework 배제"로 명확히 할지 | D-4 |
| 초기화 동작 차이의 명시 | "시그니처는 anomalib과 같고, 생성 시 pretrained를 읽지 않는 초기화 동작은 다르다"를 DECISIONS에 적을지 | D-5 |
| 배포 요구 | "순수 모델 생성자는 pretrained를 읽거나 내려받지 않고, 배포 artifact만으로 추론한다"와 그 검증(3.7.4)을 SPEC에 추가 | SPEC, FR-14, NFR-3 |
| 배포 artifact 형식 | Lightning `.ckpt`에서 꺼낼지, 별도 export할지. 포함 항목(teacher, 통계, 후처리, 전처리 설정, 원본 checkpoint 해시) | SPEC, Phase 6 |
| 이식 시 원본 변경 기록 | `TimmFeatureExtractor(pre_trained=True)`를 바꾼 것을 이식 파일 주석과 DECISIONS에 남길 방식 | NFR-5 |
| EfficientAD warm start | 학습한 가중치에서 다시 학습할 때 mean/std와 quantile을 유지할지, 다시 계산할지 | FR-15 |

이 문서를 저장소에 두면 `AGENTS.md`의 구조 표에 `docs/local-assets-usage.md` 행을 추가해야 한다.
