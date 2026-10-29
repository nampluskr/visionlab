# INTENT 안 A 적용 초안 (예외 부록, 안 A 시점의 기록)

> 상태: 개정됨(2026-10-29). 2026-10-28에 반영한 안 A의 문구는 (b)안으로 교체됐다(`docs/cleanup-proposals-draft.md` 3.3, 3.4절). `specs/INTENT.md`의 원칙 두 항목과 "왜 하는가"는 새 문구이고, `specs/SPEC.md`의 NFR-7과 부록 A는 "배포 요구 목록"이며, D-23, D-25, D-26, D-28과 `AGENTS.md`도 고쳤다. 아래 본문은 안 A 시점의 초안(예외 부록 틀)이며 현재 문구와 다르다.

사용자가 INTENT 수정안 중 안 A(원칙은 유지하고 예외를 부록으로 둔다)를 채택했다(`docs/dependency-exceptions.md` 5.1절). 이 문서는 적용 전 초안이다. INTENT, `specs/`, `AGENTS.md`는 아직 고치지 않았고, 사용자가 문구를 확인한 뒤에 고친다. 작성일은 2026-10-28이다.

## 1. INTENT에 더할 문장

INTENT의 "변하지 않는 원칙" 첫 항목 바로 뒤에 항목 하나를 더한다. 현재 INTENT의 해당 부분은 다음과 같다.

```text
## 변하지 않는 원칙

- 배포되는 모델은 순수 PyTorch만으로 동작한다.
- 원본(anomalib)의 알고리즘을 보존한다.
```

제안하는 문구는 다음과 같다. 패키지 이름은 INTENT에 쓰지 않는다. 프로젝트 문서 규칙(`project-workflow/docs/DOC-SCHEMA.md`)이 어떤 라이브러리를 쓸지는 INTENT가 아니라 `DECISIONS`와 `SPEC`이 정한다고 하기 때문이다.

```diff
 - 배포되는 모델은 순수 PyTorch만으로 동작한다.
+- 위 원칙의 예외는 추론 경로에서 쓸 수 있는 외부 패키지를 `SPEC`의 허용 목록에 근거와 함께 적은 것에 한하며, 목록에 없는 외부 패키지는 추론 경로에서 쓰지 않는다.
 - 원본(anomalib)의 알고리즘을 보존한다.
```

문구에 대해 정할 것이 두 가지 있다.

- **"추론 경로"의 정의:** 문구는 "추론 경로"를 정의하지 않는다. 정의는 `SPEC`의 NFR 행에 둔다(2절). 정의를 INTENT에 넣으면 구현 세부가 INTENT로 올라온다.
- **더 짧은 대안:** "예외는 `SPEC`의 허용 목록에 적은 외부 패키지뿐이다." 이 문구는 짧지만 "추론 경로" 한정이 빠져 학습 전용 의존(lightning, kornia의 손실)도 목록 대상으로 읽힐 수 있다.

### 1.1 이 문구로 남는 불일치

문구를 더해도 INTENT의 다른 문장은 그대로여서 다음 긴장이 남는다. 이 초안은 이를 고치지 않는다.

| INTENT 문장 | 남는 문제 |
|---|---|
| "모델을 배포하는 곳에는 anomalib과 Lightning이 설치되어 있지 않고, PyTorch만 사용할 수 있다."(왜 하는가) | 배포 환경에 허용 목록의 패키지가 설치되어 있어야 한다는 요구가 이 문장과 충돌한다. 필요하면 "PyTorch와 허용 목록의 패키지만 사용할 수 있다"로 따로 고친다 |
| "순수 PyTorch만으로 dataset, transform, dataloader, model, inference를 수행할 수 있는 모델과 가중치를 배포한다."(무엇을 하는가) | Dataset이 PIL과 torchvision을 import한다. 이 문장도 허용 목록 기준으로 읽어야 한다 |

## 2. 예외 부록의 위치

부록(허용 목록)을 둘 위치는 세 후보가 있다.

| 후보 | 장점 | 단점 |
|---|---|---|
| `specs/SPEC.md`에 NFR 행과 부록 A | 허용 목록은 "만족해야 할 것"이라 `SPEC`의 성격과 맞고(스키마상 `SPEC`은 요구와 판정 방법을 담는다), 판정 방법에 `tests/common/test_dependency.py`를 적을 수 있다. 결정 번호(D-23 등)는 근거로 남는다 | `SPEC`에 부록이라는 새 구조가 생긴다. 표가 길면 SPEC 본문이 길어진다 |
| `specs/DECISIONS.md` 끝에 부록 | 예외 결정(D-23, D-25, D-26, D-28)이 이미 있는 문서라 한곳에서 읽는다 | 스키마상 `DECISIONS`의 블럭은 선택, 근거, 배제한 대안을 가져야 하는데 허용 목록 표는 결정 블럭이 아니라 요구에 가깝다. 번호 없는 부록은 기존 형식과 다르다 |
| `specs/`의 새 문서(예: `DEPENDENCIES.md`) | 표를 길게 쓸 수 있고 다른 문서를 건드리지 않는다 | 스키마가 정한 문서 목록 밖의 문서가 늘고, `AGENTS.md`의 문서 규칙과 `specs/` 읽기 목록을 함께 고쳐야 한다 |

추천은 `specs/SPEC.md`의 NFR 행 하나와 부록 A다. 이유는 두 가지다. 첫째, 허용 목록은 "배포되는 모델이 만족해야 할 제약"이어서 요구 문서의 성격과 맞고 `tests/common/test_dependency.py`가 이미 판정 수단으로 쓰이고 있다. 둘째, 기존 D 번호는 그대로 두어 결정과 요구를 분리할 수 있다. 부록 A의 초안은 3절에 있다.

## 3. 부록에 옮길 내용

`docs/dependency-exceptions.md`의 2절(예외 결정 표)과 3절(모델별 요구)에서 허용 목록으로 옮길 내용은 다음과 같다. 새 NFR 행의 초안은 다음과 같다.

```diff
 | NFR-6 | `tests/`의 테스트는 pytest로 작성하고 실행한다. ... |
+| NFR-7 | 추론 경로(각 `model.py`, `models/components/`, 순수 `common/` 파일, `inference/`)는 torch와 부록 A의 허용 목록에 있는 외부 패키지만 import한다. 지연 import로 허용한 곳은 부록 A에 적은 파일과 함수뿐이다 | `tests/common/test_dependency.py`가 허용 목록 밖의 import를 차단한다 |
```

부록 A 허용 목록의 초안은 다음과 같다. 허용 근거가 있는 항목만 올리고, 결정이 없는 항목은 "결정 필요"로 따로 둔다.

| 패키지 | 허용 근거 | 허용 위치 | 추론 경로 |
|---|---|---|---|
| torchmetrics | D-20 | `common/metric.py`, `common/post_processor.py` | 남음 |
| kornia | D-23 | `models/components/filters/blur.py`(추론), `models/draem/loss.py`, `models/dsr/loss.py`(학습 전용) | `blur.py`만 남음 |
| scikit-learn | D-25 | `components/dimensionality_reduction/random_projection.py`, `models/cfa/model.py`의 `initialize_centroid`(지연) | 남지 않음 |
| FrEIA(와 numpy, scipy) | D-26 | FastFlow, CFLOW, UFlow, CSFlow 모델 파일과 `components/flow/all_in_one_block.py` | 남음 |
| regex | D-28 | `models/winclip/tokenizer.py`의 `SimpleTokenizer.__init__`(지연) | 남지 않음 |

결정 필요 항목은 다음과 같다. 사용자가 결정하기 전에는 목록에 올리지 않는다.

- 허용 결정이 없는 패키지: timm, torchvision, numpy, tqdm, PIL, mpmath(`docs/dependency-exceptions.md` 2.1절과 3절)
- 모델별 배포 요구는 부록 A가 아니라 `docs/dependency-exceptions.md` 3절 표를 유지하거나 부록 B로 옮길지

## 4. 다른 문서의 변경 목록

안 A를 적용하면 다음 문서를 고친다. 아래는 diff 형태의 초안이며 실제 수정은 사용자가 문구를 확인한 뒤에 한다.

### 4.1 `specs/DECISIONS.md`의 "INTENT와의 관계" 문단

네 결정(D-23, D-25, D-26, D-28)의 문단이 "긴장 관계" 서술과 "PyTorch 생태계까지 넓혀 읽는다" 식의 해석을 하고 있다. 안 A에서는 INTENT가 예외를 허용 목록으로 인정하므로 해석 대신 목록 등재로 바꾼다.

```diff
 ## D-23
-- **INTENT와의 관계**: INTENT의 "배포되는 모델은 순수 PyTorch만으로 동작한다"와 긴장 관계에 있다. kornia는 PyTorch 위의 라이브러리이므로 이 결정은 그 범위를 PyTorch 생태계 패키지까지 넓혀 읽는다. INTENT는 고치지 않는다.
+- **INTENT와의 관계**: INTENT의 "배포되는 모델은 순수 PyTorch만으로 동작한다"의 예외다. kornia는 `SPEC`의 NFR-7과 부록 A의 허용 목록에 이 결정을 근거로 올린다.
```

```diff
 ## D-25
-- **INTENT와의 관계**: INTENT의 "배포되는 모델은 순수 PyTorch만으로 동작한다"와 긴장 관계에 있다(D-23과 같다). scikit-learn은 PyTorch 위의 라이브러리가 아니므로 이 결정은 그 범위를 적합 시점의 예외로 넓혀 읽는다. 배포되는 추론 경로에는 남지 않는다. INTENT는 고치지 않는다.
+- **INTENT와의 관계**: INTENT의 "배포되는 모델은 순수 PyTorch만으로 동작한다"의 예외다(D-23과 같다). scikit-learn은 `SPEC` 부록 A에 적합 시점의 지연 import로만 올리며 배포되는 추론 경로에는 남지 않는다.
```

```diff
 ## D-26
-- **INTENT와의 관계**: INTENT의 "배포되는 모델은 순수 PyTorch만으로 동작한다", "PyTorch만 사용할 수 있다"와 긴장 관계에 있다(D-23과 같다). FrEIA는 PyTorch 위에 만든 라이브러리이므로 이 결정은 그 범위를 PyTorch 생태계 패키지까지 넓혀 읽는다. numpy와 scipy는 FrEIA의 요구 사항으로 따라오는 간접 의존이다. INTENT는 고치지 않는다.
+- **INTENT와의 관계**: INTENT의 "배포되는 모델은 순수 PyTorch만으로 동작한다"의 예외다(D-23과 같다). FrEIA는 `SPEC` 부록 A에 올린다. numpy와 scipy는 FrEIA의 요구 사항으로 따라오는 간접 의존이며 같은 행에 적는다. "PyTorch만 사용할 수 있다"(왜 하는가)와의 관계는 INTENT에서 별도로 정한다.
```

```diff
 ## D-28
-- **INTENT와의 관계**: INTENT의 "배포되는 모델은 순수 PyTorch만으로 동작한다"와 긴장 관계에 있다(D-25와 같다). `regex`는 PyTorch 위의 라이브러리가 아니며, 이 결정은 그 범위를 토크나이저 생성 시점의 예외로 넓혀 읽는다. 텍스트 임베딩은 export의 `_text_embeddings` 버퍼에 들어 있어 export를 읽는 순수 추론 경로(Predictor)는 토크나이저를 쓰지 않는다. INTENT는 고치지 않는다.
+- **INTENT와의 관계**: INTENT의 "배포되는 모델은 순수 PyTorch만으로 동작한다"의 예외다(D-25와 같다). `regex`는 `SPEC` 부록 A에 토크나이저 생성 시점의 지연 import로만 올린다. 텍스트 임베딩은 export의 `_text_embeddings` 버퍼에 들어 있어 export를 읽는 순수 추론 경로(Predictor)는 토크나이저를 쓰지 않는다.
```

각 문단의 "INTENT는 고치지 않는다." 문장은 INTENT가 이미 고쳐진 뒤에는 사실과 다르므로 지운다. D-23의 "적용 범위"에 적힌 "배포 환경에도 설치되어 있어야 한다"와 D-26의 "남은 한계"는 그대로 둔다.

### 4.2 `AGENTS.md`의 D-4 문구

```diff
-- 배포 모델은 순수 PyTorch만으로 동작한다 (INTENT, D-4).
+- 배포 모델은 순수 PyTorch만으로 동작한다. 예외는 `specs/SPEC.md` 부록 A의 허용 목록에 있는 외부 패키지뿐이다 (INTENT, D-4, NFR-7).
```

`AGENTS.md`의 같은 절에 있는 다른 문장은 바꾸지 않는다.

### 4.3 `tests/common/test_dependency.py`

현재 이 테스트는 `anomalib`, `open_clip`, `ftfy`, `sklearn`, `regex`, `lightning`만 규칙으로 검사한다(예: `regex`와 `sklearn`은 지정한 파일과 함수의 지연 import만 허용). NFR-7을 판정 수단으로 쓰려면 허용 목록에 없는 외부 패키지(timm, torchvision 등)를 포함한 검사로 확장해야 한다. 이는 코드 변경이라 이 초안의 범위 밖이며, 부록 A가 확정된 뒤 codex에 요청한다.

### 4.4 `docs/dependency-exceptions.md`와 README

`docs/dependency-exceptions.md`의 5절은 INTENT 수정안 비교였으므로, 안 A가 적용되면 "적용됨"으로 표시하거나 부록으로 옮긴 뒤 이 문서를 가리키도록 줄인다. README의 "요구 환경"에 적은 `docs/dependency-exceptions.md` 안내는, 부록이 `specs/`에 확정되면 그 위치로 바꾼다. README에는 안 A가 적용되기 전에는 예외 부록을 확정된 것처럼 쓰지 않았다.

## 5. 적용 순서 제안

사용자가 문구를 확인한 뒤의 순서는 다음과 같다. 각 단계는 사용자 요청이 있을 때만 한다.

1. INTENT에 1절의 문장을 더한다(사용자가 직접 또는 요청으로).
2. `specs/SPEC.md`에 NFR-7과 부록 A를 더한다.
3. `specs/DECISIONS.md`의 네 문단을 4.1절대로 고친다.
4. `AGENTS.md`의 문구를 4.2절대로 고친다.
5. 결정 필요 항목(timm, torchvision 등)의 허용 결정을 만들고 부록 A에 올린다.
6. `tests/common/test_dependency.py`를 4.3절대로 확장한다(codex 요청, agy 실행).
