# WinPython anomalib 설치 및 로컬 경로 설정

확인 기준일: 2026-10-22.

지정된 WinPython에 이미 설치된 PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, NumPy 2.2.5와 기존 패키지를 하나도 바꾸지 않고 Lightning 2.5.5와 anomalib 2.0.0을 추가 설치한 기록이다. 로컬 데이터·가중치 연결 방법과 실제 검증 범위를 함께 정리한다.

## 1. 설치 위치와 버전 선택

| 구분 | 경로 |
|---|---|
| WinPython | `C:\winpython\WPy64-31180_gpu` |
| Python 실행 파일 | `C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe` |
| 패키지 설치 폴더 | `C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\Lib\site-packages` |
| anomalib 패키지 | 위 폴더의 `anomalib` |
| CLI 실행 파일 | `C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\Scripts\anomalib.exe` (PATH에 없음) |
| 프로젝트 작업 폴더 | `D:\projects\visionlab` |
| 데이터셋 | `D:\datasets` |
| backbone·teacher 가중치 | `D:\backbones` |

### 버전 선택 근거

| 패키지 | 선택 | 이유 |
|---|---|---|
| anomalib | `2.0.0` | visionlab 이식·비교 기준 버전. 요구 조건 Python ≥3.10, torch ≥2, Lightning ≥2.2, torchmetrics ≥1.3.2를 현재 환경이 충족 |
| lightning / pytorch-lightning | `2.5.5` / `2.5.5` | 고정하지 않으면 pip가 pytorch-lightning 2.6.6을 골라 두 패키지 버전이 어긋남 |
| openvino | `2025.4.1` (`<2026`) | 고정하지 않으면 2026.4.1이 선택됨. 2026.0에서 `openvino.runtime`이 제거되는데 anomalib 2.0.0은 이 모듈을 직접 import함 |
| jsonargparse | `4.39.0` | 아래 3.4절 참고. 4.47.0 이상은 anomalib CLI import 실패, 4.40~4.46은 `--help` 실패 |
| 기존 패키지 전체 | 설치 전 버전 유지 | 설치 전 `pip freeze`를 constraints로 고정 |

scikit-learn, requests는 이미 설치되어 있어 기존 버전(1.7.2, 2.34.2)을 그대로 사용한다. python-dotenv는 새로 설치되었다(1.2.4). pandas는 기존 2.3.3에서 MVTec bottle train 209개가 정상 인식되어 따로 고정하지 않았다.

## 2. 확인한 하드웨어·소프트웨어 사양

| 항목 | 확인 결과 |
|---|---|
| CPU | Intel Core i7-8700K @ 3.70GHz |
| CPU 코어 | 물리 6코어, 논리 12개 |
| OS에서 확인한 RAM | 34,294,554,624 bytes, 약 31.94 GiB |
| Windows | Windows 11 Pro 25H2, build 26200.9711 |
| GPU | NVIDIA GeForce GTX 1080 Ti |
| GPU 메모리 | nvidia-smi 기준 11,264 MiB |
| NVIDIA 드라이버 | 572.70 |
| nvidia-smi CUDA 표시 | 12.8 |
| Python | CPython 3.11.8, 64비트 |
| PyTorch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| PyTorch CUDA 런타임 | 12.1 |
| cuDNN | 90100, 즉 9.1.0 |
| NumPy | 2.2.5 |
| timm | 1.0.22 |
| torchmetrics | 1.8.2 |
| pandas | 2.3.3 |
| opencv-python / headless | 4.12.0.88 / 4.12.0.88 |
| pip | 26.2.1 |
| pip 인덱스 | 별도 설정 없음 (`pip config list` 출력 없음, PyPI 사용) |

`nvidia-smi`의 CUDA 12.8은 드라이버의 지원 수준이고, `torch.version.cuda`의 12.1은 설치된 PyTorch 빌드의 CUDA 런타임이다. 두 숫자가 다른 것은 정상이다. CUDA Toolkit의 별도 설치 여부와 `nvcc` 버전은 확인하지 않았다.

## 3. 설치 방법

아래 명령은 PowerShell에서 실행한다. Python은 항상 지정 실행 파일을 사용한다.

```powershell
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
```

### 3.1 설치 전 확인과 constraints 생성

```powershell
& $py -m pip check
& $py -c "import sys, torch, numpy; print(sys.executable); print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(numpy.__version__)"
nvidia-smi

# Pin every currently installed version; skip editable/VCS entries.
& $py -m pip freeze > freeze-before.txt
Get-Content freeze-before.txt | Where-Object { $_ -match '==' } | Set-Content -Encoding utf8 constraints-before.txt
```

설치 전 `pip check`는 `No broken requirements found.`를 출력했다. `pip freeze`는 98줄이었고, 그중 editable 설치(`speech_transcriber`) 1줄을 제외한 97줄을 constraints로 사용했다. constraints는 설치를 강제하지 않고, 이미 있는 패키지의 버전 변경만 막는다. 새로 추가되는 패키지는 constraints에 없으므로 자유롭게 선택된다.

### 3.2 설치 계획 확인

```powershell
& $py -m pip install --dry-run --no-user -c constraints-before.txt `
    'anomalib[core]==2.0.0' `
    'lightning==2.5.5' 'pytorch-lightning==2.5.5' `
    'jsonargparse[signatures]==4.39.0' 'openvino==2025.4.1' `
    scikit-learn python-dotenv requests
```

`--dry-run`은 설치 계획만 계산하지만 패키지·메타데이터 다운로드는 발생할 수 있다. 출력의 `Would install` 목록에 기존 패키지의 다른 버전이 없는지 확인한다.

### 3.3 실제 설치

```powershell
& $py -m pip install --no-user -c constraints-before.txt `
    'anomalib[core]==2.0.0' `
    'lightning==2.5.5' 'pytorch-lightning==2.5.5' `
    'jsonargparse[signatures]==4.39.0' 'openvino==2025.4.1' `
    scikit-learn python-dotenv requests
```

`--no-user`는 사용자별 `%APPDATA%\Python\Python311\site-packages`로 설치되는 것을 막는다. WinPython 폴더에 쓰기 권한이 필요하다. pip 자체는 업그레이드하지 않았다.

이번 작업은 위 명령을 한 번에 실행한 것이 아니다. 실제 순서는 다음과 같았고, 위 명령은 그 결과를 하나로 합친 재설치용 명령이다. 합친 명령 자체는 실행하지 않았다.

1. `jsonargparse`를 지정하지 않고 `openvino<2026`으로 설치 → openvino 2025.4.1, jsonargparse 4.52.0 설치.
2. anomalib CLI import 실패 확인 → `jsonargparse[signatures]==4.46.0`으로 교체.
3. `--help` 실패 확인 → `jsonargparse[signatures]==4.39.0`으로 교체.

### 3.4 jsonargparse 호환 문제

anomalib 2.0.0의 `anomalib/cli/cli.py`는 `from jsonargparse._actions import _ActionSubCommands`로 비공개 클래스를 직접 import한다.

| jsonargparse | 결과 |
|---|---|
| 4.47.0 이상 (4.52.0 포함) | 클래스가 `_subcommands.ActionSubCommands`로 이동해 CLI import 단계에서 `ImportError` |
| 4.46.0 | import와 설정 파싱(`--print_config`)은 성공, `--help` 출력 시 `AttributeError: 'tuple' object has no attribute 'items'` |
| 4.39.0 | import·`--help`·`fit --help`·`--print_config` 모두 성공 |

4.39.0~4.46.0의 `_actions.py`에 클래스가 있는지는 wheel을 내려받아 소스로 확인했고(4.39.0, 4.40.2, 4.41.0, 4.42.0, 4.44.0, 4.46.0 있음 / 4.47.0, 4.48.0, 4.50.0, 4.51.0 없음), 실제 설치·실행은 4.52.0, 4.46.0, 4.39.0에서만 했다.

Lightning 2.5.5의 `LightningCLI`는 실행 시 `jsonargparse[signatures]>=4.27.7`을 요구하고, `lightning[pytorch-extra]` extra는 `>=4.39.0,<5.0`을 요구한다. 4.39.0은 두 조건을 모두 만족하며 `from lightning.pytorch.cli import LightningCLI` import를 확인했다.

### 3.5 설치 확인과 CLI

```powershell
& $py -m pip check
& $py -c "import anomalib, torch; from anomalib.data import MVTecAD; from anomalib.models import Stfpm, EfficientAd; from anomalib.engine import Engine; print(anomalib.__version__); print(anomalib.__file__); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
& $py -m anomalib.cli.cli --help
```

`Scripts` 폴더가 PATH에 없어서 pip가 `anomalib.exe` 등의 위치 경고를 출력했다. 위처럼 지정 Python에서 `-m anomalib.cli.cli`를 호출한다. anomalib 2.0.0에는 패키지 최상위 `__main__.py`가 없으므로 `python -m anomalib`은 사용할 수 없다. `anomalib.exe` 직접 실행은 이번에 확인하지 않았다.

Matplotlib 캐시 폴더 권한 오류가 있을 때는 실행 전에 쓰기 가능한 폴더를 지정한다. 이번 검증은 이 설정을 적용한 상태에서 실행했다.

```powershell
$env:MPLCONFIGDIR = "$env:TEMP\anomalib-mpl"
```

## 4. 기존 데이터셋 연결

### 4.1 확인한 로컬 데이터

| 데이터 | 경로·구조 |
|---|---|
| MVTec AD | `D:\datasets\mvtec\bottle`, `cable`, `grid` 등 15개 범주 |
| VisA | `D:\datasets\visa\candle` 등 12개 범주와 `split_csv` |
| Imagenette | `D:\datasets\imagenette2\train`, `val` (하위에 `imagenette2` 폴더도 있음) |
| 기타 | CIFAR-10, MNIST, Fashion-MNIST, CelebA, DTD, Oxford Pets 등 |

데이터셋 경로는 각 DataModule의 `root`로 지정한다.

```python
from anomalib.data import MVTecAD

datamodule = MVTecAD(
    root="D:/datasets/mvtec",
    category="bottle",
    train_batch_size=8,
    eval_batch_size=8,
    num_workers=0,
)
```

MVTec의 `root`는 `bottle`의 상위 폴더다. `prepare_data()`는 `root/category`가 있으면 다운로드를 생략하고, 없으면 다운로드를 시도한다. 경로 오타에 주의한다.

anomalib CLI용 YAML에서는 다음과 같이 지정한다. visionlab 자체 CLI의 YAML 계약과는 별개다.

```yaml
data:
  class_path: anomalib.data.MVTecAD
  init_args:
    root: D:/datasets/mvtec
    category: bottle
    num_workers: 0
```

VisA도 해당 클래스의 `root`를 지정하되, 원본 파일 구조와 DataModule의 준비 동작을 먼저 확인한다. 준비 코드가 정리·복사 작업을 수행할 수 있다. 이번 설치에서는 VisA의 준비 함수를 실행하지 않았다.

Windows에서 별도 Python 스크립트로 학습할 때는 실행부를 `if __name__ == "__main__":` 안에 두고, 처음에는 `num_workers=0`으로 시작한다.

## 5. 앞으로 다운로드할 backbone 캐시 위치

visionlab은 로컬 자산만 사용하고 다운로드하지 않는다. 아래 설정은 anomalib 비교 실행에서 의도치 않은 다운로드가 생겼을 때 저장 위치를 `D:\backbones` 아래로 모으기 위한 선택 사항이다. 이번 작업에서는 적용하지 않았다.

### 5.1 현재 PowerShell 세션에 적용

```powershell
$env:TORCH_HOME = 'D:\backbones\torch'
$env:HF_HOME = 'D:\backbones\huggingface'
$env:HF_HUB_CACHE = 'D:\backbones\huggingface\hub'
```

| 사용 주체 | 실제 기본 캐시 |
|---|---|
| Torch Hub·torchvision의 URL 가중치 | `D:\backbones\torch\hub\checkpoints` |
| Hugging Face를 사용하는 timm | `D:\backbones\huggingface\hub` |

Hugging Face 다운로드 자체를 막으려면 실행 전에 `$env:HF_HUB_OFFLINE = '1'`을 지정한다. 캐시가 없으면 오류가 발생한다. 이 변수는 anomalib 데이터 다운로드나 Torch Hub의 URL 다운로드까지 차단하지 않는다.

### 5.2 WinPython 실행 환경에 유지

설정 파일은 `C:\winpython\WPy64-31180_gpu\settings\winpython.ini`이다(존재 확인). WinPython 런처로 실행할 때 적용하려면 기존 `[environment]` 섹션에 다음 항목을 추가한다. 기존 섹션을 중복 생성하지 않는다.

```ini
[environment]
TORCH_HOME = D:\backbones\torch
HF_HOME = D:\backbones\huggingface
HF_HUB_CACHE = D:\backbones\huggingface\hub
```

외부 PowerShell·IDE에서 `python.exe`를 직접 실행하는 경우에는 해당 부모 프로세스에서도 환경변수를 지정해야 한다. 이번 작업에서는 INI나 사용자·시스템 환경변수를 변경하지 않았다.

## 6. 이미 보유한 backbone 파일 재사용

### 6.1 확인한 파일

```text
D:\backbones\resnet18-f37072fd.pth
D:\backbones\resnet34-b627a593.pth
D:\backbones\resnet50-0676ba61.pth
D:\backbones\wide_resnet50_2-95faca4d.pth
D:\backbones\wide_resnet101_2-32ee1156.pth
D:\backbones\wide_resnet50_2.tv_in1k\
D:\backbones\efficientad_pretrained_weights\pretrained_teacher_small.pth
D:\backbones\efficientad_pretrained_weights\pretrained_teacher_medium.pth
```

이 밖에 EfficientNet·MobileNet·VGG·DINOv2·CaiT·DeiT 등의 가중치도 있다. 파일 존재 여부는 모델·학습 방식·state_dict 키가 호환됨을 보장하지 않는다. 환경변수만 설정하면 이러한 파일을 재귀 검색해서 자동 선택하지 않는다.

### 6.2 명시적인 timm 로컬 파일 로딩

anomalib 2.0.0의 `Stfpm`과 공통 `TimmFeatureExtractor`는 `pretrained_cfg_overlay`나 `weights_path`를 생성자 인자로 노출하지 않는다. 로컬 파일을 쓰려면 별도 timm 모델을 등록한다.

```python
from timm.models import register_model
from timm.models.resnet import resnet18
from anomalib.models import Stfpm


@register_model
def local_resnet18(pretrained=False, **kwargs):
    kwargs["pretrained_cfg_overlay"] = {
        "file": "D:/backbones/resnet18-f37072fd.pth",
        "hf_hub_id": None,
        "url": "",
    }
    return resnet18(pretrained=pretrained, **kwargs)


model = Stfpm(backbone="local_resnet18")
```

등록은 모델 생성 전에 같은 Python 프로세스에서 실행해야 한다. teacher는 로컬 pretrained 파일을 로드하고 student는 원래 STFPM 동작대로 무작위 초기화된다. 이 예시는 설치된 패키지 소스를 수정하지 않는다.

## 7. EfficientAD의 teacher·Imagenette 연결

### 7.1 Imagenette

```python
from anomalib.models import EfficientAd

model = EfficientAd(
    imagenet_dir="D:/datasets/imagenette2/train",
    model_size="small",
)
```

anomalib 2.0.0은 `ImageFolder(self.imagenet_dir)`를 직접 호출하므로 클래스 폴더가 있는 `train`을 지정한다. `model_size` 값은 `small` 또는 `medium`이다. 학습용 DataModule의 `train_batch_size`는 1이어야 하며 EfficientAD 내부 정규화와 외부 ImageNet 정규화를 중복 적용하지 않는다.

### 7.2 teacher 파일을 명시하는 subclass

anomalib 2.0.0의 `prepare_pretrained_model()`은 현재 작업 폴더 기준 `./pre_trained/efficientad_pretrained_weights`를 사용하고, 없으면 다운로드를 시도한다. 다운로드 대신 로컬 파일을 직접 로드하려면 해당 메서드만 바꾸는 subclass를 사용한다.

```python
import os
import torch
from anomalib.models import EfficientAd


class LocalEfficientAd(EfficientAd):
    def prepare_pretrained_model(self):
        teacher_path = os.path.join(
            "D:/backbones/efficientad_pretrained_weights",
            f"pretrained_teacher_{self.model_size.value}.pth",
        )
        state_dict = torch.load(teacher_path, map_location="cpu", weights_only=True)
        self.model.teacher.load_state_dict(state_dict)


model = LocalEfficientAd(
    imagenet_dir="D:/datasets/imagenette2/train",
    model_size="small",
)
```

정션(`mklink /J <작업폴더>\pre_trained D:\backbones`)으로 연결하는 방법도 있지만, 작업 폴더마다 정션이 필요하므로 subclass를 우선한다. 이번 작업에서는 정션을 만들지 않았다.

## 8. 실제 설치·검증 기록

2026-10-22 지정 WinPython의 `Lib/site-packages`에 anomalib 2.0.0과 Lightning 2.5.5를 설치했다. 설치 전후 `pip freeze`를 비교한 결과 기존 패키지의 버전 변경·삭제는 없었고, 28개 패키지가 추가됐다(98줄 → 126줄). 설치 중 함께 들어온 jsonargparse는 4.52.0 → 4.46.0 → 4.39.0으로 교체했다(3.4절).

검증 스크립트는 `socket.connect`·`socket.create_connection`을 차단하고 `HF_HUB_OFFLINE=1`을 지정한 프로세스에서 실행했다.

| 확인 항목 | 실제 결과 |
|---|---|
| 패키지 경로 | 지정 WinPython의 `Lib/site-packages/anomalib` |
| 기존 패키지 | 버전 변경·삭제 0건 |
| 의존성 검사 | 최종 `pip check`: `No broken requirements found.` |
| 주요 import | `MVTecAD`, `Stfpm`, `EfficientAd`, `Engine` 성공 |
| LightningCLI | `from lightning.pytorch.cli import LightningCLI` 성공 |
| CUDA | 12.1, `torch.cuda.is_available()` → `True`, GTX 1080 Ti 인식 |
| GPU 연산 | CUDA 텐서 덧셈 결과 `[2.0, 2.0]` 확인 |
| CLI | `-m anomalib.cli.cli --help`, `fit --help` 종료 코드 0 |
| CLI 설정 파싱 | `fit --model Stfpm --data MVTecAD --data.root D:/datasets/mvtec --print_config` 종료 코드 0 |
| MVTec bottle | train 209, valid 83, test 83, 첫 이미지 `(3, 900, 900)` |
| STFPM 로컬 ResNet18 | 6.2절 등록 예시로 생성, GPU eval forward 성공, 256×256 입력에 anomaly map `(1, 1, 256, 256)` |
| EfficientAD | 7.2절 subclass로 small·medium teacher state_dict 복원 성공 |
| Imagenette | `D:\datasets\imagenette2\train` 이미지 9,469개 인식 |

MVTec 검증은 기본 `ValSplitMode.SAME_AS_TEST` 설정이므로 valid와 test가 같은 83개다. 독립된 검증·테스트 분할 성능을 확인한 결과가 아니다. STFPM forward는 무작위 합성 입력 1개로 확인했으며, 학습한 모델의 정확도를 뜻하지 않는다.

실행 중 다음 경고가 출력됐다. 패키지 소스를 수정해 경고를 숨기지는 않았다.

- OpenVINO: `openvino.runtime` 모듈이 deprecated이며 2026.0에서 제거된다는 `DeprecationWarning`. openvino를 2026 이상으로 올리면 anomalib 2.0.0 import가 깨질 수 있다는 근거다.
- timm: `timm.models.layers` import가 deprecated라는 `FutureWarning`.
- matplotlib: pyparsing의 `oneOf` 등 deprecated API에 대한 `PyparsingDeprecationWarning`.

데이터셋과 pretrained 가중치는 다운로드하지 않았다. 데이터·가중치 원본, WinPython INI, 사용자·시스템 환경변수는 변경하지 않았다. 캐시 설정과 정션 생성은 문서의 적용 방법이며 자동 적용한 설정이 아니다. 학습·평가·성능 측정, VisA 준비 함수, OpenVINO export, `anomalib.exe` 직접 실행은 이번 확인 범위에 포함하지 않았다.

### 추가된 패키지의 최종 버전

아래는 이번 설치로 추가된 패키지 목록이다. 기존 패키지를 포함한 전체 환경의 lock 파일은 아니다. 재설치 명령은 주요 호환성 버전을 고정하지만 나머지 간접 의존성은 설치 시점에 다시 결정될 수 있다. 정확한 재현이 필요하면 아래 버전도 constraints로 함께 고정한다.

```text
aiohappyeyeballs==2.7.1
aiohttp==3.14.5
aiosignal==1.4.0
anomalib==2.0.0
attrs==26.1.0
docstring_parser==0.18.0
FrEIA==0.2
frozenlist==1.8.0
ftfy==6.3.1
ImageIO==2.38.1
jsonargparse==4.39.0
lazy-loader==0.6
lightning==2.5.5
multidict==7.1.0
omegaconf==2.4.0
open-clip-torch==2.24.0
openvino==2025.4.1
openvino-telemetry==2025.2.0
propcache==0.5.4
python-dotenv==1.2.4
pytorch-lightning==2.5.5
rich-argparse==1.8.0
scikit-image==0.26.0
sentencepiece==0.2.2
tifffile==2026.3.3
typeshed_client==2.14.0
wcwidth==0.9.3
yarl==1.25.1
```
