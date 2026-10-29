# 정리안 후속 초안

`docs/dependency-exceptions.md`와 `docs/intent-appendix-draft.md`에 이어, 안 A 반영(2026-10-28) 뒤 남은 일을 읽기 전용으로 조사한 초안이다. 코드, `specs/`, INTENT, `AGENTS.md`, README는 수정하지 않았고, 이 문서의 제안은 모두 사용자 승인 전이다. 작성일은 2026-10-29이고 기준은 커밋 `286151e`다. 내용은 `src/visionlab`, `evaluation/anomalib`, 설치된 anomalib 2.0.0 소스, 설치 환경의 패키지 메타데이터, `specs/`에서 확인한 사실만 적었다. 정적 import 스캔은 AST로 모든 `.py`를 읽어 얻었고 스크립트는 저장소에 두지 않았다.

상태: 결정 반영됨(2026-10-29). 3절의 (b)안과 "왜 하는가" 후보 W1이 채택되어 3.3절과 3.4절의 문구가 `specs/INTENT.md`, `specs/SPEC.md`(NFR-7, 부록 A), `specs/DECISIONS.md`(D-23, D-25, D-26, D-28), `AGENTS.md`, `README.md`에 들어갔다. 3.3절의 "무엇을 하는가" 첫 문장, 검증 3단계(15행) 문구, "순수 PyTorch" 용어 정리는 승인 전이라 반영하지 않았다.

전제 정정(2026-10-29, 사용자): 배포 환경에서 제외되는 것은 Lightning과 anomalib뿐이다. 그 밖의 라이브러리(kornia, FrEIA, torchmetrics, scikit-learn, regex, timm, torchvision, numpy, PIL, tqdm, scipy 등)는 배포 환경에 포함될 수 있다. 같은 날 먼저 전달된 "배포 환경에는 PyTorch, torchvision, timm만 설치 가능하다"는 전제는 잘못 전달된 것이며, 이 문서에는 반영하지 않았다.

## 1. 전제가 기존 문서에 주는 영향

### 1.1 표현이 어긋나는 곳

이 전제에서 timm, torchvision, kornia 등은 INTENT 원칙의 예외가 아니라 배포 환경에 설치할 요구 패키지다. 현재 문서의 표현과 어긋나는 곳은 다음과 같다.

| 위치 | 현재 표현 | 이 전제에서의 문제 |
|---|---|---|
| `specs/INTENT.md` 원칙 첫 항목 | "배포되는 모델은 순수 PyTorch만으로 동작한다." | 사실과 다르다. `Predictor` 경로까지 보면 18개 모델 중 PyTorch만으로 도는 모델이 없다(모든 모델이 `PreProcessor`의 torchvision과 `PostProcessor`의 torchmetrics를 import한다) |
| `specs/INTENT.md` 원칙 두 번째 항목(안 A로 추가) | "위 원칙의 예외는 추론 경로에서 쓸 수 있는 외부 패키지를 `SPEC`의 허용 목록에 근거와 함께 적은 것에 한하며, 목록에 없는 외부 패키지는 추론 경로에서 쓰지 않는다." | 목록 관리 방식은 이 전제에서도 쓸 수 있다. 다만 목록이 사실상 추론 경로의 모든 외부 패키지(약 10개)가 되어 "예외"라는 틀이 맞지 않는다 |
| `specs/INTENT.md` "왜 하는가" | "모델을 배포하는 곳에는 anomalib과 Lightning이 설치되어 있지 않고, PyTorch만 사용할 수 있다." | 사실과 다르다. 제외되는 것은 anomalib과 Lightning뿐이다 |
| `specs/INTENT.md` "무엇을 하는가" | "순수 PyTorch만으로 dataset, transform, dataloader, model, inference를 수행할 수 있는 모델과 가중치를 배포한다." | 같은 문제. Dataset은 PIL과 torchvision을 import한다 |
| `specs/SPEC.md` NFR-7과 부록 A | "torch와 부록 A의 허용 목록에 있는 외부 패키지만 import한다." 부록 A에는 torchmetrics, kornia, scikit-learn, FrEIA(와 numpy, scipy), regex만 등재 | 문자 그대로 검사하면 현재 코드가 위반이다. 추론 경로가 import하는 timm, torchvision, tqdm, mpmath가 목록에 없고, numpy는 FrEIA 행에 묶여 있어 flow가 아닌 모델(EfficientAD 등)의 numpy 사용이 목록 밖으로 읽힌다. 부록 A는 "허용 목록"이라는 이름이다 |
| `specs/DECISIONS.md` D-23, D-25, D-26, D-28의 "INTENT와의 관계" | "INTENT ... 원칙의 예외다. ... `SPEC` 부록 A에 올린다" | 이 전제에서는 예외가 아니라 배포 요구 패키지다. D-23의 "배포 환경에도 설치되어 있어야 한다"와 D-26의 "배포 환경 요건이 INTENT의 'PyTorch만'보다 넓어진다"는 이 전제와 일치한다 |
| `AGENTS.md` 111행 | "배포 모델은 순수 PyTorch만으로 동작한다. 예외는 `specs/SPEC.md` 부록 A의 허용 목록에 있는 외부 패키지뿐이다 (INTENT, D-4, NFR-7)." | 같은 "예외" 틀 |
| `README.md` 요구 환경 | "순수 PyTorch 추론(`Predictor`)에 필요한 외부 패키지는 모델마다 다르다 ... " | 사실과 어긋나지 않는다(중립 표현) |
| `docs/dependency-exceptions.md`, `docs/intent-appendix-draft.md` | 제목과 본문이 "예외"와 "허용 결정 대기" 틀 | 이 전제에서는 4절의 정리가 필요하다 |

"순수 PyTorch"라는 용어는 `specs/INTENT.md` 3곳, `specs/SPEC.md` 3곳, `specs/DECISIONS.md` 9곳, `specs/PLAN.md` 7곳, `AGENTS.md` 4곳, `README.md` 4곳에 나온다(문자열 `순수 PyTorch` 출현 수). 문구를 고치면 이 용어를 "Lightning과 anomalib 없는" 뜻의 이름으로 그대로 쓸지 정해야 한다.

### 1.2 실제로 지켜야 하는 제약과 `test_dependency.py`의 보장 범위

실제 제약은 "추론 경로에 Lightning과 anomalib을 import하지 않는다"다. `specs/INTENT.md`의 "다루지 않는 것"(실행 시 anomalib을 import하거나 상속하지 않는다), `specs/SPEC.md` NFR-1(anomalib 없이 import·실행), NFR-2(순수 Dataset·모델·처리 구성요소·Predictor는 Lightning 없이 동작)가 이를 정한다. `tests/common/test_dependency.py`가 이를 보장하는 범위와 빠진 곳은 다음과 같다.

| 구분 | 내용 |
|---|---|
| 정적 검사 | `test_no_reference_imports_in_source`가 `src/visionlab`의 모든 `.py`에서 `anomalib`, `open_clip`, `ftfy` import를 금지하고, `sklearn`과 `regex`는 지정한 파일·함수의 지연 import만 허용하며, `lightning`과 `pytorch_lightning`은 `LIGHTNING_FILES`에 적은 파일에서만 허용한다. winclip 폴더는 `timm`, `torchvision`, `kornia`도 금지한다. 함수 안 import도 AST로 잡는다 |
| 동적 검사 | `test_isolated_import_groups`가 새 프로세스에서 `anomalib`(항상), `lightning`, `pytorch_lightning`(pure 그룹)을 차단하고 분류된 모든 모듈을 import하며 `evaluation/visionlab/predict.py`도 import한 뒤 `sys.modules`에 금지 패키지가 없는지 확인한다. 분류되지 않은 새 파일은 실패한다 |
| 모델별 추론 검사 | 01~18 모든 모델 폴더의 테스트가 `pytorch_lightning`을 차단한 자식 프로세스를 쓴다(문자열 검색 기준). export를 `Predictor`로 읽어 추론하는 경로는 이 격리 테스트가 맡는다 |
| 빠진 곳 | (1) 동적 import(`importlib`, `__import__`)는 정적 검사가 못 본다. (2) 차단 이름은 세 개뿐이라 `lightning_fabric`, `lightning_utilities`는 차단하지 않는다(torchmetrics가 `lightning_utilities`를 요구하므로 차단 대상이 아니다). (3) `test_dependency.py` 자체는 import만 검사하고 forward는 각 모델 테스트에 의존한다. (4) `open_clip`, `ftfy`는 정적 검사만 한다. (5) 이 테스트는 외부 패키지의 허용 목록(NFR-7)을 검사하지 않는다 |

따라서 이 전제의 실제 제약(Lightning·anomalib 비의존)은 현재 테스트로 보장된다. NFR-7이 말하는 "허용 목록 대조"는 아직 코드로 검사되지 않는다.

## 2. 모델별 배포 요구

### 2.1 모델별 추론 경로 외부 패키지

다음은 18개 모델 각각을 export로 읽어 추론하는 `Predictor` 경로의 외부 패키지다. torch는 모두 필요하므로 뺐다. "필요"는 모델 `model.py`가 끌어오는 import 폐쇄에 `PreProcessor`(torchvision)와 `PostProcessor`(torchmetrics)를 더한 값이다. "지연"은 함수 안 import이고 해당 분기에서만 필요하다. "학습 전용"은 `Predictor`가 import하지 않는 파일에서만 쓰인다. 기존 `docs/dependency-exceptions.md` 3절의 표는 모델 파일만 본 값이어서 이 표에서 `PreProcessor`와 `PostProcessor` 몫(torchvision, torchmetrics)을 더해 갱신했다.

| 모델 | 필요 | 지연 | 학습 전용 |
|---|---|---|---|
| stfpm | timm, torchvision, torchmetrics | | |
| efficient_ad | numpy, torchvision, torchmetrics | | tqdm |
| padim | kornia, timm, torchvision, torchmetrics | | |
| dfm | numpy, timm, torchvision, torchmetrics | scikit-learn | |
| dfkde | numpy, timm, torchvision, torchmetrics | scikit-learn | |
| patchcore | kornia, numpy, timm, torchvision, torchmetrics, tqdm | scikit-learn | |
| reverse_distillation | kornia, timm, torchvision, torchmetrics | | |
| cfa | kornia, timm, torchvision, torchmetrics, tqdm | scikit-learn | |
| fre | timm, torchvision, torchmetrics | | |
| supersimplenet | kornia, timm, torchvision, torchmetrics | | |
| draem | torchvision, torchmetrics | | kornia |
| ganomaly | torchvision, torchmetrics | | |
| dsr | torchvision, torchmetrics | | kornia |
| winclip | torchvision, torchmetrics | regex(토크나이저 생성) | |
| fastflow | FrEIA, scipy, timm, torchvision, torchmetrics | | |
| cflow | FrEIA, numpy, scipy, timm, torchvision, torchmetrics | | |
| uflow | FrEIA, mpmath, numpy, scipy, timm, torchvision, torchmetrics | | |
| csflow | FrEIA, numpy, timm, torchvision, torchmetrics | | |

표의 한계는 다음과 같다. 정적 스캔이라 서드파티의 간접 의존은 이 표에 없다(아래 2.2절에 설치 메타데이터로 따로 적었다). `Predictor`가 실제로 이 import를 모두 실행하는지는 모델별 export 테스트와 `predict.py` 실행으로만 확인된다. efficient_ad의 tqdm은 학습 전용 파일만 쓴다. 이 표 이외의 PIL은 Dataset과 시각화 파일만 쓰며 `Predictor`는 import하지 않는다.

### 2.2 합집합 requirements 후보

다음은 18개 모델 전체를 배포할 때의 추론 경로 requirements 후보다. 버전은 이 개발 환경(`C:\winpython\WPy64-31180_gpu`)의 `importlib.metadata`로 확인한 설치 버전이며 실제 설치 실험은 하지 않았다. 배포 환경에서 맞는 버전 범위는 확인하지 않았다.

| 패키지 | 확인한 설치 버전 | 필요한 모델 | 비고 |
|---|---|---|---|
| torch | 2.5.1+cu121 | 전부 | |
| torchvision | 0.20.1+cu121 | 전부(`PreProcessor`) | Pillow와 numpy를 요구한다 |
| torchmetrics | 1.8.2 | 전부(`PostProcessor`) | numpy, packaging, lightning-utilities를 요구한다. Lightning 본체가 아니다 |
| timm | 1.0.22 | 13개 모델(backbone, import 기준) | huggingface-hub, PyYAML, safetensors를 요구한다 |
| numpy | 2.2.5 | efficient_ad, dfm, dfkde, patchcore, cflow, uflow, csflow | torchvision이 요구한다 |
| kornia | 0.8.2 | padim, patchcore, reverse_distillation, cfa, supersimplenet | kornia_rs를 요구한다. draem·dsr은 학습 전용 |
| FrEIA | 0.2 | fastflow, cflow, uflow, csflow | numpy와 scipy를 요구한다 |
| scipy | 1.15.3 | flow 4개(FrEIA, `all_in_one_block.py`, uflow `anomaly_map.py`) | torchvision과 timm의 요구 사항에는 없다. FrEIA 또는 scikit-learn을 통해 들어온다 |
| tqdm | 4.70.0 | patchcore, cfa(import) | huggingface-hub가 요구한다. 추론 런타임에는 쓰이지 않는다(최상단 import만) |
| mpmath | 1.3.0 | uflow(import) | torch가 sympy를 거쳐 요구한다. 추론 런타임에는 쓰이지 않는다(`binomial_test` 선택 기능) |
| scikit-learn | 1.7.2 | dfm, dfkde, patchcore, cfa(지연, 적합 때만) | numpy, scipy, joblib, threadpoolctl을 요구한다 |
| regex | 2026.7.19 | winclip(지연, 토크나이저 생성 때만) | 요구 사항이 없다 |

이 환경에서 PIL(Pillow 11.3.0)은 torchvision이 요구하며 `Predictor`는 import하지 않는다. 이 합집합 후보는 Lightning과 anomalib을 포함하지 않는다.

### 2.3 Predictor 최상단 import 선택의 효과

`inference/predictor.py`가 14개 모델의 `model.py`를 최상단에서 import하므로 한 모델만 배포해도 지금은 kornia, numpy, timm, torchmetrics, torchvision, tqdm 여섯 패키지가 필요하다(flow 모델은 FrEIA, scipy, mpmath를 더한다). 이 전제에서 이는 필수 수정이 아니라 요구를 줄이는 선택이다. 모델별 지연 import로 바꾸면 모델당 필요한 외부 패키지 수가 patchcore(6개 그대로)를 뺀 17개 모델에서 줄고, 줄어드는 폭은 최대 6개에서 2개(draem, ganomaly, dsr, winclip: torchvision, torchmetrics)까지다. 자세한 모델별 표와 변경 범위, 리스크는 5.2절에 이전 조사로 남겼다. 리팩토링이라 반대 벤더 적대적 검증이 필수다.

## 3. INTENT·SPEC 문구 개정 선택지

### 3.1 안 A가 이 전제에서도 맞는가

안 A(원칙 유지, 예외 목록을 부록으로)는 "목록에 없는 패키지는 쓰지 않는다"는 통제 방식으로는 이 전제에서도 유효하다. 그러나 "예외"라는 틀은 맞지 않는다. 추론 경로가 쓰는 외부 패키지가 약 10개이고 `Predictor` 경로까지 보면 PyTorch만으로 도는 모델이 하나도 없어서, 원칙 문장("순수 PyTorch만으로 동작한다")을 두고 목록을 예외로 부르면 원칙을 만족하는 모델이 없는 상태가 된다. 안 A는 문구를 조금만 바꾸는 장점이 있고 목록 통제는 그대로 쓸 수 있지만 원칙 문장 자체의 사실 불일치가 남는다.

### 3.2 선택지 (a): 현행 유지

원칙과 안 A로 더한 문장을 그대로 두고 부록 A에 timm, torchvision, numpy(단독), PIL(`data/`와 시각화), tqdm, mpmath도 등재한다.

| 항목 | 내용 |
|---|---|
| INTENT 변경 | "왜 하는가"의 한 문장만 고친다. 후보: "모델을 배포하는 곳에는 anomalib과 Lightning이 설치되어 있지 않고, 설치할 수 있는 패키지는 PyTorch와 `SPEC`의 허용 목록에 적은 것뿐이다." 삽입 위치는 "왜 하는가" 첫 문장 |
| SPEC | 부록 A에 6개 패키지 행 추가. NFR-7 유지 |
| DECISIONS | D-23, D-25, D-26, D-28의 "INTENT와의 관계" 유지 |
| AGENTS.md, README | 변경 없음 |
| 테스트 | `test_dependency.py`에 허용 목록 대조 추가(선택) |
| 단점 | 원칙 첫 문장이 사실과 다르다. 목록이 거의 전체를 "예외"로 담는다 |

### 3.3 선택지 (b): 원칙을 Lightning·anomalib 비의존으로 바꾸고 부록 A를 요구 목록으로 관리

| 항목 | 내용 |
|---|---|
| INTENT 원칙 첫 항목 | 현재 "배포되는 모델은 순수 PyTorch만으로 동작한다." 대신 "배포되는 모델은 Lightning과 anomalib 없이 동작한다." |
| INTENT 원칙 두 번째 항목 | 안 A로 더한 문장 대신 "배포 환경에 필요한 외부 패키지는 `SPEC`의 요구 목록에 근거와 함께 적은 것에 한하며, 목록에 없는 외부 패키지는 추론 경로에서 쓰지 않는다." |
| INTENT "무엇을 하는가" 첫 문장 | 후보: "Lightning과 anomalib 없이 dataset, transform, dataloader, model, inference를 수행할 수 있는 모델과 가중치를 배포한다." |
| INTENT 검증 3단계(15행) | "순수 PyTorch 모델로 불러와 추론하고" 대신 "Lightning 없이 PyTorch 모델로 불러와 추론하고"(선택). 용어를 그대로 두면 1.1절 끝의 용어 정의가 필요하다 |
| SPEC | NFR-7을 "추론 경로가 import하는 외부 패키지는 부록 A(배포 요구 목록)에 있어야 한다"로 바꾸고 부록 A의 제목과 설명에서 "예외·허용"을 "배포 요구"로 고친다. 부록 A에 timm, torchvision, numpy, PIL, tqdm, mpmath 등재 |
| DECISIONS | D-23, D-25, D-26, D-28의 "INTENT와의 관계"를 "예외다"에서 "`SPEC` 부록 A의 요구 목록에 올린다"로 고친다. 근거 문장에 "배포 환경에서 제외되는 것은 Lightning과 anomalib뿐이다"를 한 번 적는다 |
| AGENTS.md | 111행: "배포 모델은 Lightning과 anomalib 없이 동작한다. 추론 경로의 외부 패키지는 `specs/SPEC.md` 부록 A의 요구 목록에 있는 것뿐이다 (INTENT, D-4, NFR-7)." |
| README | 요구 환경 문단은 그대로 맞다. 링크만 확인 |
| 테스트 | 현재 `test_dependency.py`가 이미 이 원칙(Lightning·anomalib 비의존)을 보장한다. 요구 목록 대조는 선택 확장 |
| 문서 | `docs/dependency-exceptions.md`, `docs/intent-appendix-draft.md`의 "예외" 틀 정리 |
| 단점 | SSOT인 INTENT의 원칙 문장이 바뀐다. "순수 PyTorch"를 쓰는 다른 문서(1.1절 끝의 출현 수)와의 용어 정리가 따른다 |

### 3.4 "왜 하는가" 문장 수정 후보

현재 문장은 "모델을 배포하는 곳에는 anomalib과 Lightning이 설치되어 있지 않고, PyTorch만 사용할 수 있다."다. 후보는 다음과 같다.

| 후보 | 문장 | 어울리는 선택지 |
|---|---|---|
| W1 | "모델을 배포하는 곳에는 anomalib과 Lightning이 설치되어 있지 않다. 그 밖에 모델이 필요로 하는 패키지는 배포 환경에 설치하며, 그 목록은 `SPEC`이 관리한다." | (b) |
| W2 | "모델을 배포하는 곳에는 anomalib과 Lightning이 설치되어 있지 않고, PyTorch와 `SPEC`의 요구 목록에 적은 패키지만 설치한다." | (b) |
| W3 | "모델을 배포하는 곳에는 anomalib과 Lightning이 설치되어 있지 않고, 설치할 수 있는 패키지는 PyTorch와 `SPEC`의 허용 목록에 적은 것뿐이다." | (a) |

W1이 가장 사실에 가깝다. 제외 대상(anomalib, Lightning)을 먼저 말하고 나머지는 목록에 맡긴다.

### 3.5 비교와 추천

| 비교 | (a) 현행 유지 | (b) Lightning·anomalib 비의존 + 요구 목록 |
|---|---|---|
| 원칙 문장과 사실의 일치 | 원칙은 여전히 사실과 다르다. 목록이 예외를 설명 | 일치한다 |
| INTENT 변경량 | 1곳 | 3~4곳(원칙 두 항목, 무엇을 하는가, 왜 하는가, 선택으로 검증 3단계) |
| 실제 제약과 테스트의 일치 | 제약(Lightning·anomalib)과 원칙 문장이 다르다 | 원칙이 테스트가 보장하는 제약과 같다 |
| 부록 A | 허용 목록(예외) | 배포 요구 목록 |
| 다른 문서 | 변경 적음 | DECISIONS 네 문단, AGENTS.md, 두 초안 문서 |

추천은 (b)다. 이 전제에서 원칙은 "Lightning과 anomalib 비의존"이고 이미 테스트가 이를 보장한다. (a)는 INTENT 변경이 작지만 원칙 첫 문장이 어떤 모델에도 맞지 않고 "예외" 목록이 추론 경로의 외부 패키지 전체가 된다. (b)에서도 INTENT 수정은 사용자 요청이 있을 때만 하며 이 문서는 후보만 제시한다.

## 4. 이전 D-29 초안의 정리

이전 초안(timm·torchvision·numpy는 등재, PIL은 `data/` 한정, tqdm·mpmath는 지연 import로 추론 경로에서 제거하는 D-29 등급안)에서 이 전제로 불필요해진 것은 다음이다. 배포 환경에 설치 가능한 패키지를 묻는 질문과 그 답에 따른 분기표(torch만, torch+torchvision+timm 등)는 전제가 정해져 사라졌다. PIL을 `data/`로만 한정할지와 시각화 두 파일이 위반인지에 대한 논의도 PIL이 설치 가능하므로 의미가 없다. tqdm과 mpmath의 지연 import 수정은 "추론 경로에서 반드시 빼야 하는 변경"에서 "요구 패키지를 줄이는 선택"으로 바뀌었다. torchvision의 허용 위치를 파일 단위로 구분하는 일, D-29를 "예외 허용" 결정으로 쓰는 일, `test_dependency.py`의 파일 제한 규칙(PIL)도 불필요하다. 남는 결정은 두 가지다. 부록 A에 timm, torchvision, numpy, PIL, tqdm, mpmath를 등재하는 것(근거는 5.1절의 사용처 표)과 3절의 (a), (b) 선택이다. (b)를 택하면 새 D 번호를 만드는 대신 "배포 환경에서 제외되는 것은 Lightning과 anomalib뿐이다"를 D-23, D-25, D-26, D-28의 관계 문단과 부록 A 설명에 적는 것으로 충분하다. 별도 결정 블럭이 필요하면 D-29 한 개("배포 환경에는 Lightning과 anomalib을 제외한 외부 패키지를 설치할 수 있다. 추론 경로의 외부 패키지는 `SPEC` 부록 A의 요구 목록으로 관리한다")로 쓸 수 있다.

## 5. 참고: 이전 조사

다음 두 절은 이전 전제에서 쓴 조사이며, 사용처 사실은 그대로 유효하다. 설치 가능 범위에 관한 판단과 등급안은 4절로 대체됐다.

### 5.1 사용처 전수 조사 (이전 조사)

`src/visionlab`에서 해당 패키지를 import하는 파일과 쓰는 함수는 다음과 같다. "추론 경로"는 `Predictor`나 export 모델이 import 또는 실행하는지를 뜻하고, "import만"은 모듈 최상단 import는 추론 경로에 있으나 쓰는 함수는 학습·적합 전용인 경우다. 경로 판단은 `Predictor`의 최상단 import와 모델별 `model.py` 의존을 따라간 결과이고, 함수가 실제로 불리는지는 소스를 읽어 확인했다.

| 패키지 | 파일 | 쓰는 위치 | 성격 |
|---|---|---|---|
| timm | `models/components/feature_extractors/timm.py` | `TimmFeatureExtractor.__init__`, `_map_layer_to_idx` | 추론 경로(backbone 생성). STFPM, PaDiM, DFM, DFKDE, PatchCore, Reverse Distillation, FRE, SuperSimpleNet, CFLOW가 사용 |
| timm | `models/fastflow/model.py` | `FastflowModel.__init__`, `forward` | 추론 경로(`Predictor`가 지연 import) |
| timm | `models/uflow/feature_extraction.py` | `CaitFeatureExtractor.__init__` | 추론 경로(`Predictor`가 지연 import) |
| torchvision | `common/pre_processor.py` | `PreProcessor.crop`의 `center_crop` | 추론 경로(`center_crop`이 지정된 경우에만 실행, import는 항상) |
| torchvision | `models/components/feature_extractors/timm.py` | `TimmFeatureExtractor.__init__`의 `create_feature_extractor` | 추론 경로(nn.Module backbone 분기) |
| torchvision | `models/cfa/model.py` | `get_feature_extractor`(`torchvision.models`, `create_feature_extractor`) | 추론 경로 |
| torchvision | `models/csflow/model.py` | `CsFlowMultiScaleFeatureExtractor.__init__`의 `efficientnet_b5` | 추론 경로(`Predictor`가 지연 import) |
| torchvision | `models/reverse_distillation/components/bottleneck.py`, `de_resnet.py` | `BasicBlock`, `Bottleneck`, `conv1x1`, `conv3x3` | 추론 경로(구조) |
| torchvision | `models/components/tiling/tiler.py` | `Tiler.tile`의 `RandomCrop` | import만(타일링 옵션) |
| torchvision | `models/efficient_ad/model.py` | `EfficientAdModel.choose_random_aug_image`의 `transforms` | import만(학습 증강) |
| torchvision | `common/perlin.py`, `common/multi_random_choice.py` | Perlin 합성 이상, 무작위 변환 선택 | import만(학습 증강, 일부 모델 import 경로에 있음) |
| torchvision | `models/dsr/anomaly_generator.py`, `models/supersimplenet/loss.py`, `models/efficient_ad/lightning_model.py` | 학습용 생성기, 손실, 데이터 준비 | 학습 전용 |
| torchvision | `data/common/base_dataset.py` | `BaseDataset.__getitem__`의 resize | 데이터 로딩 |
| numpy | `models/csflow/model.py` | `ParallelPermute.get_random_perm`(`np.random.default_rng(seed).permutation`) | 추론 경로(모델 생성 때 순열을 다시 만든다. `state_dict`에 없음) |
| numpy | `models/cflow/utils.py` | `get_logp`(`np.log`, `np.sqrt`, `np.pi`) | 추론 경로(`Predictor`가 지연 import) |
| numpy | `models/uflow/anomaly_map.py` | `AnomalyMapGenerator.binomial_test`의 `np` | import만(`binomial_test`와 `compute_anomaly_mask`는 `forward`와 `src`, `tests`, `evaluation` 어디서도 불리지 않는다) |
| numpy | `models/components/dimensionality_reduction/random_projection.py` | `SparseRandomProjection._sparse_random_matrix`, `_johnson_lindenstrauss_min_dim` | import만(PatchCore 적합) |
| numpy | `models/efficient_ad/model.py` | `choose_random_aug_image` | import만(학습 증강) |
| numpy | `common/visualizer.py` | `ImageVisualizer.visualize`, `_array`, `_heatmap`, `_plane` | 시각화(`Predictor`는 import하지 않음) |
| tqdm | `models/cfa/model.py` | `CfaModel.initialize_centroid` | import만(적합, 최상단 import가 `Predictor` 경로에 있음) |
| tqdm | `models/components/sampling/k_center_greedy.py` | `KCenterGreedy.select_coreset_idxs` | import만(PatchCore 적합, 최상단 import가 `Predictor` 경로에 있음) |
| tqdm | `models/efficient_ad/lightning_model.py` | `map_norm_quantiles`, `teacher_channel_mean_std` | 학습 전용 |
| PIL | `data/common/base_dataset.py` | `BaseDataset.__getitem__` | 데이터 로딩 |
| PIL | `common/visualizer.py`, `common/lightning_visualizer.py` | `ImageVisualizer.visualize`, `LitVisualizer._save` | 시각화(데이터 로딩이 아님) |
| mpmath | `models/uflow/anomaly_map.py` | 최상단 `from mpmath import binomial, mp`와 `mp.dps = 15`. 실제 사용은 `binomial_test`의 high precision 분기뿐 | import만(위 numpy 행과 같은 이유로 실행되지 않음) |


### 5.2 Predictor 최상단 import 상세 (이전 조사, 선택 사항)

#### 5.2.1 현재 구조

`inference/predictor.py`는 모듈 최상단에서 stfpm, efficient_ad, padim, dfm, dfkde, patchcore, reverse_distillation, cfa, fre, supersimplenet, draem, ganomaly, dsr, winclip 14개 모델의 `model.py`를 import한다. 추가로 `reverse_distillation.anomaly_map`(`AnomalyMapGenerationMode`, kornia 의존)과 `components.classification`(`FeatureScalingMethod`, numpy 의존)도 최상단에서 import한다. FastFlow, CFLOW, UFlow, CSFlow는 `from_export` 안에서 모델 이름이 맞을 때만 지연 import한다. `PreProcessor`와 `PostProcessor`도 최상단 import이고 각각 torchvision과 torchmetrics를 import한다.

#### 5.2.2 모델별 요구 변화

다음은 모델별 배포 요구가 어떻게 달라지는지 정적 스캔으로 계산한 표다. "현재"는 `Predictor` 모듈을 import하는 데 필요한 것(최상단 import가 끌어오는 패키지 합)에 그 모델의 지연 import 몫을 더한 값이다. "1단계"는 `Predictor`의 14개 모델 import만 모델별 지연 import로 바꾼 경우이고 `PreProcessor`와 `PostProcessor`의 최상단 import는 그대로다. "2단계"는 `PreProcessor`(`center_crop`)와 `PostProcessor`(`torchmetrics`)도 지연 import로 바꾼 경우다. torch는 모두 필요하므로 뺐다.

| 모델 | 현재 | 1단계 (Predictor 지연) | 2단계 (+ Pre/PostProcessor 지연) |
|---|---|---|---|
| stfpm | kornia, numpy, timm, torchmetrics, torchvision, tqdm | timm, torchmetrics, torchvision | timm, torchvision |
| efficient_ad | 위와 같음 | numpy, torchmetrics, torchvision | numpy, torchvision |
| padim | 위와 같음 | kornia, timm, torchmetrics, torchvision | kornia, timm, torchvision |
| dfm | 위와 같음 (+ scikit-learn 지연) | numpy, timm, torchmetrics, torchvision (+ scikit-learn 지연) | numpy, timm, torchvision |
| dfkde | 위와 같음 (+ scikit-learn 지연) | 위 dfm과 같음 | 위 dfm과 같음 |
| patchcore | 위와 같음 (+ scikit-learn 지연) | kornia, numpy, timm, torchmetrics, torchvision, tqdm | kornia, numpy, timm, torchvision, tqdm |
| reverse_distillation | 위와 같음 | kornia, timm, torchmetrics, torchvision | kornia, timm, torchvision |
| cfa | 위와 같음 (+ scikit-learn 지연) | kornia, timm, torchmetrics, torchvision, tqdm | kornia, timm, torchvision, tqdm |
| fre | 위와 같음 | timm, torchmetrics, torchvision | timm, torchvision |
| supersimplenet | 위와 같음 | kornia, timm, torchmetrics, torchvision | kornia, timm, torchvision |
| draem | 위와 같음 | torchmetrics, torchvision | torch만 |
| ganomaly | 위와 같음 | torchmetrics, torchvision | torch만 |
| dsr | 위와 같음 | torchmetrics, torchvision | torch만 |
| winclip | 위와 같음 (+ regex 지연) | torchmetrics, torchvision (+ regex 지연) | torch만 (+ regex 지연) |
| fastflow | 위와 같음 + FrEIA, scipy | FrEIA, scipy, timm, torchmetrics, torchvision | FrEIA, scipy, timm |
| cflow | 위와 같음 + FrEIA, scipy | FrEIA, numpy, scipy, timm, torchmetrics, torchvision | FrEIA, numpy, scipy, timm, torchvision |
| uflow | 위와 같음 + FrEIA, mpmath, scipy | FrEIA, mpmath, numpy, scipy, timm, torchmetrics, torchvision | FrEIA, mpmath, numpy, scipy, timm, torchvision |
| csflow | 위와 같음 + FrEIA | FrEIA, numpy, timm, torchmetrics, torchvision | FrEIA, numpy, timm, torchvision |

표의 한계는 다음과 같다. "현재" 열의 "위와 같음"은 `Predictor` 최상단 import 합(kornia, numpy, timm, torchmetrics, torchvision, tqdm)이다. 1단계와 2단계는 모델 `model.py`의 import 폐쇄만 본 값이고 `TimmFeatureExtractor` 등 공통 부품의 최상단 import가 끌어오는 패키지는 포함했다. 서드파티의 간접 의존은 보지 않았고, 실제 import해 본 실험은 하지 않았다. 2단계의 "torch만"은 모델 파일과 `Predictor`/`PreProcessor`/`PostProcessor` 경로의 정적 import 기준이다. 5.1절 표의 tqdm, mpmath 지연 import 수정을 함께 하면 patchcore와 cfa의 tqdm, uflow의 mpmath가 빠진다.

#### 5.2.3 변경 범위

변경은 파일과 함수 수준에서 다음과 같다. 코드는 수정하지 않았다.

| 파일 | 함수 | 변경 |
|---|---|---|
| `inference/predictor.py` | 모듈 최상단 import, `Predictor.from_export` | 14개 모델 import와 `AnomalyMapGenerationMode`, `FeatureScalingMethod` import를 제거하고 `from_export`가 모델 이름에 따라 분기 안에서 import한다(FastFlow, CFLOW, UFlow, CSFlow가 이미 쓰는 방식). `models` 사전 구성을 이름에서 클래스로 가는 지연 로더로 바꾼다 |
| `common/pre_processor.py` | 모듈 최상단 import, `PreProcessor.crop` | (2단계) `center_crop` import를 `crop` 안으로 옮긴다. 또는 텐서 슬라이싱으로 대체하되 반올림 규칙을 `center_crop`과 같게 맞춘다 |
| `common/post_processor.py`, `common/metric.py` | 최상단 import, `_f1_threshold`, `PostProcessor.update` | (2단계) torchmetrics import를 `_f1_threshold`와 보정용 `_binary_inputs`가 쓰이는 `update` 안으로 옮긴다. 추론의 `__call__`과 `load_state_dict`는 torchmetrics를 쓰지 않는다 |
| `models/cfa/model.py`, `components/sampling/k_center_greedy.py`, `models/uflow/anomaly_map.py` | `initialize_centroid`, `select_coreset_idxs`, `binomial_test` | (선택, 5.1절 표의 tqdm, mpmath 행) tqdm, mpmath import를 함수 안으로 옮긴다. `mp.dps = 15`는 high precision 분기 안으로 옮긴다 |

#### 5.2.4 리스크

- **전 모델 영향:** `Predictor`는 18개 모델의 export 로딩 경로다. 지연 import 분기를 하나라도 잘못 쓰면 해당 모델의 `Predictor.from_export`가 깨진다. 각 모델 테스트(`tests/NN_*/`)의 export 로딩 테스트와 `tests/common/test_dependency.py`, 모델별 `evaluation/visionlab/predict.py` 실행으로만 잡힌다. 전체 pytest 회귀(549개)에 더해 모델별 `predict.py`와 export 호환 확인이 필요하다.
- **import 부작용:** UFlow의 `mp.dps = 15`처럼 모듈 로드 시 실행되는 문장을 옮기면 동작이 바뀔 수 있다. 서로 다른 모델의 최상단 import 순서에 기대는 코드가 있는지는 확인하지 못했다.
- **import 차단 테스트:** 여러 테스트가 자식 프로세스에서 일부 패키지를 차단하고 `Predictor`를 import한다. `Predictor` import 범위가 줄면 기존 테스트의 기대(예: FrEIA가 `sys.modules`에 있어야 한다는 단언)가 바뀔 수 있다. 15_csflow 테스트의 `assert "FrEIA" in sys.modules`가 한 예다.
- **CPU 비교 정확성:** 공통 파일(`pre_processor.py`)의 `center_crop`을 슬라이싱으로 대체하면 수치가 달라질 수 있다. 지연 import만 쓰면 위험이 작다.
- **검증 규정:** 이는 리팩토링이므로 `AGENTS.md`와 워크스페이스 규정에 따라 반대 벤더 적대적 검증이 필수다. 구현은 codex가, 검증은 Claude가 하고(`project-workflow/docs/ADVERSARIAL-REVIEW.md` 3절의 리팩토링 규정), 이 프로젝트에서는 결과를 `docs/results/`의 문서에 남긴다. 최대 3회 수정 요청 한도도 적용된다.

## 6. LICENSE와 출처 표기 점검

### 6.1 현황

루트에 `LICENSE` 파일이 없다. `src/visionlab`의 `.py` 126개 중 `SPDX-License-Identifier: Apache-2.0` 줄이 있는 파일은 105개이고(그중 7개는 `SPDX-License-Identifier: MIT` 줄도 함께 둔다), SPDX는 없고 저작권 줄만 있는 파일이 2개(`models/winclip/clip.py`, `tokenizer.py`: open_clip의 MIT 저작권 표기), 둘 다 없는 파일이 19개다. `evaluation/`의 `.py` 16개와 `tests/`의 44개는 모두 표기가 없다(새로 작성한 코드).

표기가 없는 19개 파일의 성격은 다음과 같다. 코드와 `specs/DECISIONS.md` D-15, D-18의 설명에 따라 분류했으며 어느 파일이 anomalib 코드의 파생물인지에 대한 법적 판단은 하지 않았다.

| 성격 | 파일 | 개수 |
|---|---|---|
| anomalib의 해당 부품 동작을 새로 구현한 공통 부품 | `common/evaluator.py`, `post_processor.py`, `pre_processor.py`, `visualizer.py`, `batch.py`, `learning_type.py`, `metric.py`, `data/common/split.py`, `models/components/tiling/tiler.py` | 9 |
| native Lightning으로 새로 쓴 모델 파일 | `models/stfpm/lightning_model.py`(다른 17개 모델의 `lightning_model.py`는 Intel 저작권과 SPDX가 있다) | 1 |
| visionlab 고유 구현 | `common/lightning_visualizer.py`, `data/common/base_dataset.py`, `data/mvtec/dataset.py`, `lightning_data.py`, `data/visa/dataset.py`, `lightning_data.py`, `trainers/lightning_cli.py`, `inference/predictor.py`, `utils/io.py` | 9 |

대표 예는 `models/stfpm/lightning_model.py`다. Phase 1 파일이며 헤더가 없다. 원본 `stfpm/lightning_model.py`를 바탕으로 했는지는 확인하지 못했다. 이후 Phase의 같은 종류 파일(예: `models/draem/lightning_model.py`)은 `# Copyright (C) 2022-2025 Intel Corporation`과 `# SPDX-License-Identifier: Apache-2.0`을 둔다. 이는 Phase 1과 이후의 표기 관행 차이로 보이며 원인은 확인하지 못했다. 공통 부품 9개는 `AGENTS.md`와 D-18이 "새로 구현한다(복사 아님)"고 한 파일이다. `specs/SPEC.md` NFR-5는 "이식한 파일은 원본 저작권·SPDX 표기를 유지한다"이며 새로 구현한 파일에는 표기를 요구하지 않는다.

복사본의 변경 표시는 따로 없다. Apache-2.0 파일은 의존성 제거로 바뀐 줄이 있지만 "변경했다"는 표기를 두지 않았다(D-18: 바뀐 줄은 주변 원본 표기에 맞춘다).

### 6.2 서드파티 자산과 표기 위치

복사해 온 서드파티 자산의 표기 위치는 다음과 같다.

| 자산 | 출처와 라이선스 | 표기 위치 |
|---|---|---|
| `bpe_simple_vocab_16e6.txt.gz` | OpenAI CLIP이 만든 어휘, open_clip 2.24.0 배포본에서 바이트 복사 | `models/winclip/LICENSE`(open_clip 저작권과 OpenAI 2021 저작권, MIT 문구), `tokenizer.py` 머리 주석 |
| `models/winclip/clip.py`, `tokenizer.py` | open_clip(MIT)을 참고해 독립 구현한 새 코드 | 두 파일의 첫 줄 뒤 저작권 주석과 "derived from open_clip (MIT license)" |
| 상류의 anomalib 복사 파일 | anomalib 2.0.0, Apache-2.0(저작권 Intel Corporation) | 각 파일의 `# Copyright (C) 2022-2025 Intel Corporation`과 `# SPDX-License-Identifier: Apache-2.0` |
| MIT 표기를 가진 파일 7개 | `components/flow/all_in_one_block.py`, `csflow/model.py`, `draem/model.py`, `reverse_distillation/anomaly_map.py`, `components/bottleneck.py`, `de_resnet.py`, `supersimplenet/model.py` | 원본(anomalib)이 둔 `SPDX-License-Identifier: MIT`를 그대로 유지 |
| anomalib 라이선스 전문 | 설치된 anomalib의 `dist-info/LICENSE`(Apache-2.0)가 있고 `NOTICE` 파일은 없다 | 저장소에는 라이선스 전문이 없다 |

### 6.3 두 안의 영향 비교

라이선스 선택은 사용자 결정이다. 아래는 법률 자문이 아니며 일반적인 영향만 비교한 것이다. 정확한 판단은 법률 전문가의 확인이 필요하다.

| 항목 | 안 A: Apache-2.0 + NOTICE | 안 B: MIT |
|---|---|---|
| 이식 코드와의 관계 | 이식 코드의 원본 라이선스(Apache-2.0)와 같아 한 라이선스로 맞는다 | 이식한 Apache-2.0 파일은 일반적으로 MIT로 바꾸기 어렵다. 프로젝트는 MIT이고 이식 파일은 Apache-2.0으로 남는 혼합 구조가 된다 |
| 의무 | 라이선스 전문 제공, 원본 저작권 표기 유지, 바꾼 파일의 변경 표시, 원본에 `NOTICE`가 있으면 그 내용 유지(anomalib에는 없음) | MIT 문구와 저작권 표기를 유지. 이식 파일은 Apache-2.0 의무가 그대로 남는다 |
| 외부 코드 | open_clip과 OpenAI(MIT) 고지를 `NOTICE`에 모아 적는다 | 같은 고지를 `LICENSE`나 별도 파일에 적는다 |
| 다른 사용자 입장 | 특허 허여 조항이 있어 기업 사용에 관행적으로 선호된다 | 단순하고 허용적이다 |
| 이 프로젝트에서 필요한 작업 | 루트에 `LICENSE`(Apache-2.0)와 `NOTICE` 추가, 변경 표시 방식 결정, 표기 없는 파일 19개의 처리 결정 | 루트에 `LICENSE`(MIT) 추가, 이식 파일의 Apache-2.0 유지를 문서화, 같은 19개의 처리 결정 |

공통으로 정해야 할 것은 표기가 없는 19개 파일에 헤더를 더할지, 변경 표시를 어떻게 할지다. 이 판단도 법률 자문이 아니다.

## 7. 앞 Phase 레퍼런스 오염 정적 점검

### 7.1 점검 기준

Phase 15에서 발견한 문제는 공유 `TimmFeatureExtractor`가 nn.Module backbone을 받는 분기에서 `dryrun_find_featuremap_dims`가 train 모드의 `torch.empty` 입력으로 forward를 돌려 pretrained BatchNorm의 running 통계를 오염시킨 것이다. anomalib 코드에서 이 dry-run을 부르는 곳은 `TimmFeatureExtractor`의 nn.Module 분기(CSFlow만 이 분기를 쓴다), `cfa/torch_model.py`의 `get_scale`, `padim/torch_model.py`의 `_deduce_dims`(정의만 있고 호출되지 않는다)뿐이다. 문자열 backbone 경로(`timm.create_model`)에는 dry-run이 없다.

### 7.2 모델별 점검 표

다음은 `evaluation/anomalib/`의 레퍼런스 어댑터(`local_models.py`, `local_setup.py`)와 anomalib 소스를 읽은 결과다. 실행은 하지 않았다. 판정 "없음"은 코드상 오염 경로가 없다는 뜻이고 실행으로 확인한 것은 아니다.

| 모델 | BatchNorm | dry-run 경로 | 오염 가능성 | 근거 |
|---|---|---|---|---|
| 01 stfpm | 있음(resnet18 teacher, student) | 없음(문자열 backbone) | 없음 | teacher는 `.eval()`, `TimmFeatureExtractor.forward`가 `requires_grad`가 False면 매번 `eval()`과 `no_grad`로 돌린다. student는 의도된 학습 |
| 02 efficient_ad | 없음(PDN에 BatchNorm 없음) | 없음 | 없음 | `efficient_ad/torch_model.py`에 BatchNorm이 없다 |
| 03 padim | 있음 | `_deduce_dims`가 정의만 있고 호출되지 않음 | 없음 | 호출 지점이 없고 backbone이 `.eval()`로 생성 |
| 04 dfm | 있음 | 없음 | 없음 | 생성 시 `.eval()`, forward에서도 `.eval()` |
| 05 dfkde | 있음 | 없음 | 없음 | 생성 시 `.eval()`, forward에서도 `.eval()` |
| 06 patchcore | 있음 | 없음 | 없음 | 생성 시 `.eval()`, forward에서 `TimmFeatureExtractor`가 eval 고정 |
| 07 reverse_distillation | 있음(encoder, 학습되는 decoder) | 없음 | 없음 | forward 앞에서 `self.encoder.eval()`. decoder의 BatchNorm은 pretrained가 아니다 |
| 08 cfa | 있음(wide_resnet50_2, torchvision) | 있음: `initialize_centroid` 뒤의 `get_scale`이 `torch.empty` 입력으로 forward한다 | 없음(조건부, 실행 미확인) | `get_feature_extractor`가 `.eval()`로 만든다. anomalib `Cfa.trainer_arguments`가 `num_sanity_val_steps = 0`을 정하므로 sanity 검증이 없어 `on_validation_model_train`(전체 `model.train()`)이 `on_train_start`보다 먼저 불리지 않는다. 그래서 `initialize_centroid`(`on_train_start`)와 `get_scale`이 eval 상태의 extractor로 돈다. 이후 forward는 `self.feature_extractor.eval()`을 먼저 부른다 |
| 09 fre | 있음 | 없음 | 없음 | 생성 시 `.eval()`, forward에서도 `.eval()` |
| 10 supersimplenet | 있음(backbone) | 없음 | 없음 | forward 두 곳에서 `self.feature_extractor.eval()` |
| 11 draem | 있음(학습하는 네트워크) | 없음 | 해당 없음 | pretrained BatchNorm이 없다 |
| 12 fastflow | 있음(resnet18) | 없음(`timm.create_model` 직접) | 없음 | forward 앞에서 `self.feature_extractor.eval()`, `requires_grad = False` |
| 13 cflow | 있음(wide_resnet50_2) | 없음 | 없음 | 생성 시 `.eval()`, forward 앞에서 `self.encoder.eval()` |
| 14 uflow | 있음(resnet18) | 없음 | 없음 | `LayerNormFeatureExtractor`가 문자열 backbone이고 `.eval()`을 붙이며, `TimmFeatureExtractor.forward`가 eval을 고정한다 |
| 15 csflow | 있음(EfficientNet-B5, nn.Module backbone) | 있음(`TimmFeatureExtractor`의 nn.Module 분기) | 있음(확인됨) | Phase 15에서 실측으로 확인했고 `LocalCsflow`가 `eval()`을 넘기게 고쳤다(2차 레퍼런스) |
| 16 ganomaly | 있음(학습하는 네트워크) | 없음 | 해당 없음 | pretrained BatchNorm이 없다 |
| 17 dsr | 없음(`dsr/torch_model.py`에 BatchNorm 없음) | 없음 | 없음 | 해당 BatchNorm이 없다 |
| 18 winclip | 없음(LayerNorm) | 없음 | 없음 | BatchNorm이 없다. 다른 문제(forward hook 누적)는 Phase 18에서 발견·처리했다 |

### 7.3 다시 돌릴 후보

정적 점검만으로 "다시 돌릴 후보"로 표시할 모델은 없다. 오염 경로가 확인된 모델은 CSFlow뿐이고 이미 2차 레퍼런스로 처리했다. CFA는 `num_sanity_val_steps = 0`이라는 코드 근거로 "없음"으로 판정했으나 실행으로 확인한 것은 아니다. 가장 가벼운 확인은 Phase 15에서 쓴 `diagnose_csflow_buffers.py`와 같은 방식으로 기존 CFA 레퍼런스 체크포인트의 backbone buffer를 가중치 파일과 비교하는 것이며, 레퍼런스 재실행은 필요하지 않다. 이 확인은 사용자 결정을 기다린다.

한계는 다음과 같다. 점검은 anomalib 소스와 어댑터를 읽은 것이고 Lightning 훅 순서에 대한 판단은 설치된 Lightning 2.5.5의 `core/hooks.py`(`on_validation_model_train`)와 `Cfa.trainer_arguments`를 근거로 한다. 이 판단이 틀릴 수 있는 지점은 Lightning이 fit 시작 때 모델을 train 모드로 바꾸는 코드가 있는 경우인데, `lightning/pytorch/loops`와 `trainer`에서 그런 호출을 찾지 못했다.

## 8. 사용자 결정 항목

다음은 이 초안에서 사용자 결정이 필요한 항목이다.

- INTENT·SPEC 문구 개정 선택지 (a) 현행 유지 또는 (b) Lightning·anomalib 비의존으로 원칙을 바꾸고 부록 A를 요구 목록으로 관리(3절, 추천 (b)). "왜 하는가" 문장 후보 W1 ~ W3 중 선택.
- 부록 A에 timm, torchvision, numpy, PIL, tqdm, mpmath를 등재하는 것(4절, 선택지와 무관하게 필요).
- 용어 "순수 PyTorch"를 그대로 쓸지, Lightning·anomalib 비의존 뜻으로 정의하거나 바꿀지(1.1절).
- 배포 requirements 후보(2.2절)의 버전 범위 확정과 실제 설치 확인 여부.
- `Predictor` 모델별 지연 import를 요구 감소를 위해 진행할지(2.3절, 선택 사항, 반대 벤더 적대적 검증 필수)와 tqdm·mpmath 지연 import 수정 여부.
- 라이선스 안 A 또는 안 B 선택과 표기 없는 19개 파일의 처리(6절).
- CFA 레퍼런스 체크포인트의 buffer 비교 확인 여부(7.3절).
