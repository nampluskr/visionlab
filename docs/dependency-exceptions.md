# 외부 의존 예외 정리 (초안)

> 상태: INTENT 수정안 A가 반영됨(2026-10-28, `docs/intent-appendix-draft.md`). 부록 A에는 torchmetrics, kornia, scikit-learn, FrEIA(와 numpy, scipy), regex만 올랐다. 2.1절의 timm, torchvision, numpy(단독), tqdm, PIL과 3절에서 확인된 mpmath는 허용 결정 대기다. 이 문서의 5절은 반영 전의 수정안 비교다.

INTENT의 "배포되는 모델은 순수 PyTorch만으로 동작한다"와 실제 코드 사이의 간극을 한곳에 모은 초안이다. `specs/`와 INTENT는 고치지 않았고, 이 문서는 결정을 대신하지 않는다. INTENT 수정안은 5절에 있고 선택은 사용자에게 남아 있다. 작성일은 2026-10-28이고 Phase 1~18이 끝난 커밋 `ddee01d` 기준이다.

## 1. 근거와 한계

적은 내용은 두 곳에서 확인한 사실만이다.

- **specs:** `specs/DECISIONS.md`의 D-4, D-20, D-23, D-25, D-26, D-27, D-28과 `specs/INTENT.md`.
- **코드:** `src/visionlab`의 모든 `.py`를 AST로 읽은 정적 import 스캔. `model.py`에서 시작해 `visionlab` 내부 import를 따라가며 만나는 외부 패키지를 모았고, 함수 안 import는 "지연"으로 구분했다. 스캔 스크립트는 저장소에 두지 않았다.

한계는 다음과 같다. 정적 스캔은 서드파티 패키지가 끌고 오는 간접 의존(예: timm이 요구하는 패키지)을 보지 않는다. 실제 배포 환경에 설치해 import해 본 실험은 하지 않았다. 표의 "필요"는 import 문이 있다는 뜻이고 런타임에 해당 분기가 실행된다는 보증은 아니다.

## 2. 예외 결정 표

다음은 요청받은 다섯 결정이다. "추론 경로 잔존"은 export를 읽는 순수 추론(`Predictor`와 각 `model.py`)에서 해당 의존이 import 또는 실행되는지를 뜻한다.

| 결정 | 대상 | 허용 위치 | 추론 경로 잔존 | 배포 요구 |
|---|---|---|---|---|
| D-23 | kornia 0.8.2 | `components/filters/blur.py`(GaussianBlur2d), `models/draem/loss.py`, `models/dsr/loss.py` | `blur.py`는 남는다(순수 모델 쪽). 두 `loss.py`는 학습 전용이라 남지 않는다 | `blur.py`를 쓰는 모델의 배포 환경에 kornia가 필요하다(3절 표) |
| D-25 | scikit-learn | `components/dimensionality_reduction/random_projection.py`(밀도가 1이 아닌 분기), `models/cfa/model.py`의 `initialize_centroid`(`gamma_c > 1`) | 남지 않는다. 함수 안 지연 import이고 적합 분기에서만 실행된다 | 추론에는 불필요하다 |
| D-26 | FrEIA 0.2(와 numpy, scipy) | FastFlow, CFLOW, UFlow, CSFlow의 모델 파일과 `components/flow/all_in_one_block.py` | 남는다. flow 블록이 모델 구조 자체이고 평가 `forward`가 호출한다 | FrEIA와 numpy, scipy. 스캔에서는 UFlow의 `anomaly_map.py`가 mpmath도 import한다 |
| D-27 | 입력 크기 통일(라이브러리 예외 아님) | 원본이 입력 크기를 고정한 UFlow(448)를 256x256으로 통일. 레퍼런스는 yaml `pre_processor`, native는 `UflowLitModel` 상수 | 해당 없음. 순수 `UflowModel`의 `input_size` 기본값(448)은 원본 그대로다 | 없음 |
| D-28 | open_clip(독립 구현으로 대체), regex | `models/winclip/clip.py`, `tokenizer.py`. `regex`는 `SimpleTokenizer.__init__` 안 지연 import 한 곳 | CLIP 인코더는 순수 torch라 남지만 외부 패키지는 없다. `regex`는 토크나이저를 만들 때만 필요하고 export를 읽는 추론은 `_text_embeddings` 버퍼를 쓰므로 토크나이저를 쓰지 않는다 | 추론에는 외부 패키지가 필요 없다. 텍스트 임베딩을 새로 만드는 `setup(class_name)`에는 regex가 필요하다 |

D-27은 의존 예외가 아니라 입력 조건의 변경이다. 원본 알고리즘 보존 원칙과의 관계 때문에 같은 표에 두었다. D-28의 순수 `encode_image` 앞 resize(입력이 CLIP 이미지 크기와 다를 때만 bicubic)도 같은 종류의 조건 변경이며 D-28에 함께 적혀 있다.

### 2.1 결정 없이 쓰이거나 다른 결정에 속한 의존

위 다섯 결정에 들어 있지 않은 외부 패키지가 추론 경로에 남아 있다. specs에서 아래 항목의 허용 결정을 찾은 것은 torchmetrics(D-20)뿐이다.

| 패키지 | 코드에서 확인한 사용 위치 | specs의 근거 |
|---|---|---|
| torchmetrics | `common/post_processor.py`, `common/metric.py`. `Predictor`가 `PostProcessor`를 import한다 | D-20이 "`src/visionlab`의 허용된 외부 의존"이라고 적는다 |
| torchvision | `common/pre_processor.py`(`center_crop`), 대부분의 모델 파일, `data/*/dataset.py` | 허용 결정을 찾지 못했다. D-28은 winclip 파일에서만 금지한다 |
| timm | `TimmFeatureExtractor`를 쓰는 모델들 | 허용 결정을 찾지 못했다. 가중치 경로는 D-12, D-19가 다룬다 |
| numpy, tqdm | 일부 모델 파일(3절 표) | 허용 결정을 찾지 못했다 |
| PIL | `data/mvtec/dataset.py`, `data/visa/dataset.py` | 허용 결정을 찾지 못했다 |
| lightning | 학습 전용 `lightning_model.py`와 `lightning_visualizer.py` | 순수 파일에서는 import하지 않는다(NFR-2). 학습과 평가에만 필요하다 |

## 3. 모델별 배포 환경 요구

다음은 모델별 순수 모델 경로(`model.py`에서 시작한 정적 스캔)가 import하는 외부 패키지다. torch는 모든 모델에 필요하므로 뺐다. "지연"은 함수 안 import이고 해당 분기에서만 필요하다.

| 모델 | 필요(최상위 import) | 지연 import | 학습 전용 추가 |
|---|---|---|---|
| cfa | kornia, timm, torchvision, tqdm | scikit-learn | |
| cflow | FrEIA, numpy, scipy, timm, torchvision | | |
| csflow | FrEIA, numpy, timm, torchvision | | |
| dfkde | numpy, timm, torchvision | scikit-learn | |
| dfm | numpy, timm, torchvision | scikit-learn | |
| draem | 없음(torch만) | | kornia(`loss.py`) |
| dsr | 없음(torch만) | | kornia(`loss.py`) |
| efficient_ad | numpy, torchvision | | |
| fastflow | FrEIA, scipy, timm | | |
| fre | timm, torchvision | | |
| ganomaly | 없음(torch만) | | |
| padim | kornia, timm, torchvision | | |
| patchcore | kornia, numpy, timm, torchvision, tqdm | scikit-learn | |
| reverse_distillation | kornia, timm, torchvision | | |
| stfpm | timm, torchvision | | |
| supersimplenet | kornia, timm, torchvision | | |
| uflow | FrEIA, mpmath, numpy, scipy, timm, torchvision | | |
| winclip | 없음(torch만) | regex(토크나이저 생성 시) | |

표는 모델 파일만 본 것이어서 실제 배포 요구는 아래 공통 항목이 더해진다.

- **전처리와 후처리:** `Predictor`가 쓰는 `PreProcessor`는 torchvision을, `PostProcessor`는 torchmetrics를 import한다. 따라서 export를 `Predictor`로 읽는 배포에서는 표에서 "torch만"인 draem, dsr, ganomaly, winclip도 torchvision과 torchmetrics가 필요하다.
- **Predictor 모듈 import:** `inference/predictor.py`는 stfpm, efficient_ad, padim, dfm, dfkde, patchcore, reverse_distillation, cfa, fre, supersimplenet, draem, ganomaly, dsr, winclip의 `model.py`를 모듈 최상단에서 import한다. 그래서 어느 모델 하나만 배포해도 `Predictor` 모듈을 import하려면 kornia, numpy, timm, torchmetrics, torchvision, tqdm이 설치되어 있어야 한다(스캔 결과). FastFlow, CFLOW, UFlow, CSFlow는 함수 안에서 지연 import하므로 FrEIA, scipy, mpmath는 해당 모델의 export를 읽을 때만 필요하다.
- **Dataset:** `MVTecDataset`, `VisADataset`은 PIL과 torchvision을 import한다.
- **학습과 평가:** 모든 `XXXLitModel`은 lightning과 torchmetrics가 필요하다.

## 4. INTENT 문구와의 불일치

다음은 INTENT의 문장과 어긋나는 예외 및 정도다. 정도는 추론 경로에 남는지와 영향받는 모델 수로 나눴다(큼: 추론 경로에 남고 여러 모델에 영향, 중: 추론 경로에 남으나 일부 모델, 작음: 추론 경로에 남지 않거나 조건부).

| INTENT 문장 | 어긋나는 예외 | 정도 |
|---|---|---|
| "배포되는 모델은 순수 PyTorch만으로 동작한다"(변하지 않는 원칙) | kornia(D-23), FrEIA와 numpy, scipy, mpmath(D-26)가 추론 경로에 남는다 | 큼: kornia는 5개 모델, FrEIA는 4개 모델 |
| 같은 문장 | timm, torchvision, numpy, tqdm, torchmetrics가 추론 경로에 남는다. torchmetrics만 D-20이 허용하고 나머지는 허용 결정을 찾지 못했다 | 큼: 대부분의 모델과 `Predictor` |
| 같은 문장 | scikit-learn(D-25), regex(D-28) | 작음: 지연 import이고 추론에서 실행되지 않는다 |
| "모델을 배포하는 곳에는 anomalib과 Lightning이 설치되어 있지 않고, PyTorch만 사용할 수 있다"(왜 하는가) | 위의 모든 패키지가 배포 환경에 설치되어 있어야 한다는 요구가 생긴다. "PyTorch만 사용할 수 있다"는 전제와 가장 크게 충돌한다 | 큼 |
| "순수 PyTorch만으로 dataset, transform, dataloader, model, inference를 수행할 수 있는 모델과 가중치를 배포한다"(무엇을 하는가) | Dataset이 PIL과 torchvision을 import한다. 전처리는 torchvision을, 후처리는 torchmetrics를 import한다 | 중: 모델 외 구성요소 |
| "원본(anomalib)의 알고리즘을 보존한다" | D-27(UFlow 입력 448을 256으로 통일), D-28(순수 `encode_image` 앞의 조건부 resize와 open_clip 독립 구현) | 작음: 알고리즘은 같고 입력 조건과 구현 방식이 다르며, 수치 비교로 확인했다 |
| "anomalib 사용자가 같은 방식으로 쓸 수 있게, 클래스와 생성자 인자의 이름·순서·기본값을 anomalib과 맞춘다" | WinCLIP의 `apply_transform=True`와 `few_shot_source`는 `NotImplementedError`다. 인자 이름·순서·기본값은 같다 | 작음: 시그니처는 일치하고 일부 값만 미지원 |
| "로컬에 저장된 데이터셋과 모델 백본 가중치를 사용하고, 인터넷에서 다운로드 받지 않는다" | 어긋나는 예외 없음. D-28이 open_clip의 `pretrained` 다운로드 경로를 없앴다 | 없음 |

specs는 이 긴장을 각 결정의 "INTENT와의 관계"에 이미 적고 있다. D-23, D-25, D-26, D-28은 INTENT를 고치지 않고 "PyTorch 생태계 패키지까지 넓혀 읽는다" 또는 "적합 시점의 예외로 읽는다"고 적었다. 그러나 2.1절의 패키지들은 그런 문장조차 없다.

## 5. INTENT 수정안

INTENT는 사용자의 요청이 있을 때만 고친다. 아래 세 안은 선택지이며 어느 것도 적용하지 않았다.

### 5.1 안 A: 원칙은 유지하고 예외를 부록으로 둔다 (추천 후보)

"변하지 않는 원칙"의 첫 항목은 그대로 두고 "예외는 별도 부록의 목록으로 관리하며 목록에 없는 외부 패키지는 쓰지 않는다"는 한 문장만 INTENT에 더한다. 부록에는 2절과 3절의 표를 옮긴다. 부록의 위치는 `specs/`의 새 문서 또는 `DECISIONS.md`의 절이다.

- **영향:** INTENT 변경이 한 문장으로 가장 작다. 원칙의 의도(배포 환경을 좁게 유지)가 남고 현실과의 간극이 부록에 명시된다. D-23, D-25, D-26, D-28의 "INTENT와의 관계" 문단은 부록을 가리키도록 줄일 수 있다. `tests/common/test_dependency.py`의 허용 목록을 부록과 대조하는 검사로 키우기 쉽다.
- **약점:** 원칙 문장 자체는 여전히 사실과 다르다(예외를 읽어야 맞다). 2.1절의 패키지(timm, torchvision 등)는 부록에 새로 허용 근거를 정해야 한다.

### 5.2 안 B: 원칙의 범위를 PyTorch 생태계 패키지까지 넓힌다

원칙을 "배포되는 모델은 PyTorch와 PyTorch 위에 만든 패키지(예: kornia, timm, torchvision, torchmetrics, FrEIA)만으로 동작한다"로 바꾼다. "왜 하는가"의 "PyTorch만 사용할 수 있다"도 같은 방향으로 고친다.

- **영향:** kornia, FrEIA, timm, torchvision, torchmetrics의 긴장이 한 번에 해소된다. 반면 scikit-learn, numpy, scipy, mpmath, tqdm, PIL, regex는 PyTorch 위의 패키지가 아니어서 여전히 예외가 필요하다. "PyTorch 생태계"의 경계가 모호해 새 패키지를 들일 때마다 해석 논쟁이 생긴다.
- **약점:** 원칙이 느슨해져 "순수"의 의미가 약해지고, 배포 환경이 PyTorch뿐이라는 전제(왜 하는가)와 더 멀어진다.

### 5.3 안 C: 모델별 배포 요구 목록을 신설한다

원칙을 "각 모델의 배포 요구 패키지는 모델별 목록으로 고정하고 목록 밖의 패키지는 추론 경로에서 쓰지 않는다"로 바꾸고, 3절의 표를 목록의 초안으로 쓴다. "왜 하는가"는 "배포 환경에 목록의 패키지만 설치하면 된다"로 고친다.

- **영향:** 검증 가능한 형태가 된다(import 스캔과 목록을 대조). 모델 단위로 요구가 달라지는 현실(draem, dsr, ganomaly, winclip은 모델 파일에 외부 패키지가 없고, uflow는 가장 많다)을 그대로 담는다. 모델을 추가할 때마다 목록을 갱신해야 하는 유지 비용이 생긴다.
- **약점:** INTENT가 구현 세부(패키지 목록)에 가까워진다. `Predictor`가 모든 모델을 최상단에서 import하는 구조(3절)를 그대로 두면 목록이 모델별로 의미가 없어지므로, 모델별 지연 import로 바꾸는 코드 변경이 함께 필요하다. 이 변경은 이 문서의 범위 밖이다.

### 5.4 세 안의 비교

다음은 세 안을 영향 범위로 비교한 것이다.

| 비교 | 안 A | 안 B | 안 C |
|---|---|---|---|
| INTENT 변경 크기 | 한 문장 | 두 문장 | 원칙과 이유 문장 |
| 현재 예외가 모두 설명되는가 | 예(부록 목록으로) | 일부만 | 예(모델별 목록으로) |
| 새 패키지를 들일 때 | 부록에 추가 | 해석이 필요 | 목록에 추가 |
| 추가 코드 변경 | 없음(검사 확장은 선택) | 없음 | `Predictor` 지연 import 등이 필요할 수 있음 |
| `AGENTS.md`의 "배포 모델은 순수 PyTorch만으로 동작한다 (INTENT, D-4)" 문구 | 부록 참조로 수정 가능 | 범위 문구 수정 | 목록 참조로 수정 |

## 6. 사용자 결정 항목

다음은 이 초안에서 사용자 결정이 필요한 항목이다.

- INTENT 수정안 선택(5절의 안 A, B, C 또는 수정 없음).
- 2.1절의 timm, torchvision, numpy, tqdm, PIL에 대해 허용 결정을 새로 만들지 여부.
- 이 문서의 위치와 형식(지금은 `docs/`, 안 A를 택하면 부록으로 `specs/`에 옮길지).
