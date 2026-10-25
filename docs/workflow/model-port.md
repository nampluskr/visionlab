# 모델 이식 절차

PLAN의 Phase 하나(모델 하나)를 진행하는 절차다. 역할과 권한은 `AGENTS.md`의 "에이전트 역할", 완료 기준은 `specs/PLAN.md`의 "Phase 공통 양식"과 "검증 방식"을 따른다. 이 문서는 절차와 지시 양식만 정하며, 요구와 기준을 새로 만들지 않는다.

## 1. 역할과 흐름

한 Phase는 다음 순서로 진행한다. Claude가 모든 단계를 조율하고, 사용자에게는 Claude만 보고한다.

| 단계 | 작업 | 담당 | 다음 단계로 넘기는 것 |
| --- | --- | --- | --- |
| 1 | 레퍼런스 실행 | Claude가 명령을 보내고 agy가 실행 | 콘솔 출력, run 폴더 |
| 2 | 레퍼런스 진단·기록 | Claude | 결과 파일의 1·2절 |
| 3 | 이식 구현 | Claude가 지시하고 codex가 구현 | 변경 파일 목록, 이식 메모 |
| 4 | 적대적 검증, 수정 요청 | Claude (최대 3회) | 검증 기록 |
| 5 | 이식 결과 실행 | Claude가 명령을 보내고 agy가 실행 | 콘솔 출력, run 폴더 |
| 6 | 수치 판정·기록 | Claude (기준 미달이면 3으로, 최대 3회) | 결과 파일의 5절 |
| 7 | 보고 | Claude가 보고하고 사용자가 종료 결정 | |
| 8 | 커밋·푸시 | Claude (종료 결정 후) | 커밋, 원격 반영 |

단계 사이에는 파일과 출력만 넘긴다. codex와 agy에는 이전 대화의 해석 대신 원본 소스, `specs/`, 이 문서, 산출물 경로를 준다.

## 2. Phase 0 (데이터셋)

Phase 0은 모델이 아니라 데이터셋을 다룬다. 1절의 순서를 따르되 다음이 다르다.

- 레퍼런스 실행(1~2단계) 대신 anomalib 데이터(`LocalMVTecAD`, `LocalVisa`)와의 대조가 기준이다.
- codex는 데이터셋을 새로 구현한다(원본 복사 아님, D-18). 만들 파일과 완료 조건은 `specs/PLAN.md`의 Phase 0을 따른다.
- 지시와 검증은 0-A(순수 Dataset: `tests/00_dataset/test_data.py`)와 0-B(LitData와 `evaluation/anomalib/compare_data.py`)로 나눠 각각 진행한다. 수정 반복 한도(3회)도 단계별로 센다.
- agy는 단계별로 해당 테스트와 대조 스크립트를 실행한다.
- Claude는 파일 목록, 라벨, mask, 분할, 최종 입력 텐서의 일치를 판정한다(FR-17).

## 3. 모델 Phase 시작 준비 (Claude)

Phase를 시작할 때 Claude가 다음을 준비한다.

- 결과 파일 `docs/results/NN-<kebab-case>.md`를 `docs/results/README.md`의 양식으로 만들고, 1절(모델 노트)을 채운다. 원본 소스 경로(`anomalib/models/image/<name>/`), 생성자 시그니처, 모델이 고정하는 `trainer_arguments`, 전처리·정규화, 제공 metric(image만인지), 외부 의존(FrEIA, open_clip 등), 로컬 자산 경로를 적는다. 노트는 색인이며, 판단의 근거는 원본 소스다.

## 4. 레퍼런스 실행과 기록

### 4.1 실행 명령 (Claude가 agy에 보냄)

`NN`은 PLAN의 Phase 번호, `<name>`은 모델 폴더 이름이다(예: `01_stfpm`). `evaluation\anomalib`에서 데이터셋마다 1회 실행한다(실행 시간 비용 때문에 반복 실행은 하지 않는다. 사용자 결정, 2026-10-25). run 폴더는 `v<N>`으로 남는다.

```powershell
cd D:\projects\visionlab\evaluation\anomalib
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/NN_<name>.yaml
& $py anomalib_run.py train --config default.yaml --config data/visa.yaml --config models/NN_<name>.yaml
```

### 4.2 진단 (Claude)

Claude는 agy가 돌려준 출력과 run 폴더로 다음을 확인한다.

- 종료 상태와 오류, 네트워크 차단 오류(`network access is blocked`)가 없는지
- `Seed set to 42`, 실제 적용된 `max_epochs`와 `Overriding ...` 로그(모델 `trainer_arguments`)
- 저장된 `lightning_logs/version_*/config.yaml`의 경로·예산이 의도와 같은지
- 에폭별 학습 히스토리(`train_loss_epoch` 등)가 로그에 남았는지

run이 비정상이면(오류, 중단, 예산 불일치) 레퍼런스로 채택하지 않고 원인을 기록한 뒤 사용자에게 보고한다. 반복 실행이 없으므로 run 간 재현성은 별도로 확인하지 않는다.

### 4.3 기록

결과 파일의 2절(레퍼런스)에 다음을 적는다.

- 실행 일시, run 폴더(`outputs/anomalib/<Model>/<DataModule>/<category>/v<N>`)
- category별 Image·Pixel AUROC(image metric만 내는 모델은 Image만)
- 에폭별 학습 히스토리(에폭마다의 `train_loss_epoch` 등)와 그 출처 파일
- agy가 실행한 명령 전체(코드 블록, 7.2절 참고)
- 실제 적용된 학습 예산
- 실패나 이상이 있으면 지우지 않고 원인과 판단

## 5. 이식 구현 (codex)

### 5.1 지시 양식 (Claude가 codex에 보냄)

```text
[Phase N 이식 지시: <model>]
목표: anomalib 2.0.0 <model>을 native Lightning과 순수 PyTorch로 이식한다.
읽을 것: specs/INTENT.md, specs/SPEC.md, specs/PLAN.md(Phase N, 공통 양식), specs/DECISIONS.md(D-4, D-5, D-15, D-20, D-21),
  AGENTS.md(Python 코드 규칙), docs/workflow/model-port.md, docs/results/NN-<kebab-case>.md(1·2절),
  원본 C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\Lib\site-packages\anomalib\models\image\<name>\
쓸 수 있는 곳: evaluation/, src/, tests/ (테스트는 tests/NN_<model>/)
금지: anomalib import(src/visionlab), 코드 실행, specs/·docs/ 수정, 원본 알고리즘 변경
만들 것: <Phase N의 파일 목록 (D-15 배치)>
돌려줄 것: 변경 파일 목록, 원본 파일·함수와의 대응표, 원본과 다르게 구현한 곳과 이유, 미구현 항목
```

### 5.2 구현 절차 (D-18)

이식은 원본을 복사한 뒤 anomalib 의존성만 걷어 내는 순서로 한다.

1. **복사:** `anomalib/models/image/<model>/`의 파일을 `src/visionlab/anomaly_detection/models/<model>/`로 폴더 이름과 파일 구성 그대로 복사한다(예: `stfpm/`, `efficient_ad/`).
2. **이름 변경:** `torch_model.py`만 `model.py`로 바꾸고, 이를 참조하는 import를 맞춘다. 패키지 `__init__.py`는 Lightning 모델(`lightning_model`)을 import하지 않도록 고친다(D-18 예외, NFR-2).
3. **공통 요소:** 모델이 import하는 `anomalib/models/components/`의 요소만 `models/components/`의 같은 상대 위치로 복사한다. 앞 Phase에서 이미 복사한 요소는 재사용한다.
4. **의존성 제거:** 복사본에서 anomalib import를 모두 제거한다. models 밖의 anomalib 부품(Batch·InferenceBatch·Tiler, PreProcessor, PostProcessor, Evaluator·metric, Visualizer, `AnomalibModule`, `LearningType` 등)은 복사하지 않고 visionlab에서 필요한 것만 새로 구현한다. 앞 Phase에서 구현한 대체 부품은 재사용한다.
5. **native Lightning:** `lightning_model.py`는 native `LightningModule`로 바꾼다.

### 5.3 구현 규칙

- 원본 알고리즘(연산, loss, anomaly map, 학습 hook의 동작)은 바꾸지 않는다(INTENT).
- 클래스 이름: 순수 모델은 원본 이름을 유지하고(예: `STFPMModel`), LightningModule은 원본 이름에 `LitModel`을 붙인다(예: `StfpmLitModel`, `EfficientAdLitModel`). 생성자 인자의 이름·순서·기본값은 원본과 같다(D-5).
- 순수 파일(`model.py`, `loss.py`, `anomaly_map.py` 등, `models/components/`)은 Lightning·anomalib을 import하지 않는다(D-4, D-15). `lightning_model.py`는 native Lightning API만 쓴다.
- 코드 표기: 복사한 원본 코드는 원본 표기(type hint, docstring, 저작권·SPDX 헤더)를 유지하고(NFR-5), 의존성 제거로 바뀐 줄은 주변 원본 표기에 맞춘다. AGENTS.md의 Python 코드 규칙은 visionlab에서 새로 작성하는 코드(대체 부품, 스크립트, 테스트)에만 적용한다.
- 학습에 쓰던 외부 라이브러리(FrEIA, open_clip 등)는 추론 경로에서 독립 구현한다.
- metric(AUROC, F1, F1 임계값)은 anomalib과 같은 torchmetrics 클래스로 계산한다. 직접 구현하지 않는다(D-20, FR-24).
- 테스트는 pytest로 쓴다(함수 `test_*`, `assert`, fixture). `unittest`를 쓰지 않고 테스트 파일에 `sys.path` 설정을 두지 않는다(`tests/conftest.py`). import 차단 테스트는 subprocess로 새 인터프리터를 띄워 확인한다(D-21, NFR-6).
- 이식 메모에 원본 파일과 이식 파일의 대응, 의존성 제거로 바뀐 줄, 새로 구현한 대체 부품을 적는다.

## 6. 적대적 검증 (Claude)

Claude는 이식 결과가 원본과 같게 동작한다는 가정을 깨는 쪽으로 검증한다. 근거는 원본 소스와 이식 코드이며, codex의 설명은 근거로 쓰지 않는다.

가장 중점적으로 보는 것은 다음 두 가지다(사용자 결정, 2026-10-25). 아래 6.1 표의 나머지 항목보다 먼저, 모든 회차에서 확인한다.

1. **anomalib 의존성의 완전한 삭제:** `src/visionlab` 전체에서 anomalib import, 상속, anomalib 객체·Enum 사용이 하나도 없어야 한다. native Lightning만으로 구현되고(`XXXLitModel`), 순수 파일에는 Lightning도 없어야 한다. 소스 grep과 import 차단 테스트로 확인한다.
2. **로컬 pretrained 가중치 사용:** 외부 인터넷 연결을 차단한 상태로 학습·평가를 실행하고, 로컬 가중치(`paths.toml`)만 쓰는지(없으면 `FileNotFoundError`, 무작위 대체 없음) 확인한다. anomalib 레퍼런스와 codex가 작성한 코드의 에폭별 학습 히스토리(`train_loss_epoch` 등)와 최종 성능값(Image·Pixel AUROC)을 비교한다. 히스토리의 수치 허용 오차는 사용자가 정할 때까지 비교 표로 기록하고, 어긋남이 크면 원인을 진단한다.

### 6.1 공격 항목

검증할 항목은 다음과 같다.

| 항목 | 확인 내용 |
| --- | --- |
| anomalib 의존성(중점 1) | `src/visionlab`의 anomalib import·상속·객체 사용 0건, native `LightningModule`·`LightningDataModule`·`Trainer`만 사용, 순수 파일의 Lightning import 0건 |
| 로컬 가중치와 오프라인(중점 2) | 네트워크 차단 실행 성공, pretrained 로딩 경로가 `paths.toml`의 로컬 파일뿐, 누락 시 오류, 불러온 가중치가 레퍼런스와 같음(FR-23) |
| 에폭별 히스토리(중점 2) | 레퍼런스와 이식 결과의 에폭별 `train_loss_epoch` 등 비교, 최종 Image·Pixel AUROC 비교 |
| 원본 대비 변경 | 원본 파일과 복사본을 줄 단위로 대조해, 의존성 제거와 `model.py` 이름 변경 외의 변경이 있는지 |
| 연산 동등성 | 원본 forward·loss·score·anomaly map 계산과 다른 곳, 누락된 단계, 연산 순서 |
| 대체 부품 | 새로 구현한 Batch·PreProcessor·PostProcessor·Evaluator·Visualizer·Tiler가 원본 anomalib 부품과 같은 동작을 하는지 |
| 학습 흐름 | hook(`on_train_start`, `on_validation_epoch_end` 등)과 optimizer·scheduler, 모델이 고정하는 `trainer_arguments` |
| 상태 | buffer·통계·threshold·normalization 상태의 생성, 저장, checkpoint 복원 |
| 전처리 | 정규화 위치와 횟수, resize, 입력 범위 |
| 난수 | seed 소비 순서, 데이터 분할, 무작위 선택 |
| 생성자 | 인자 이름·순서·기본값(D-5) |
| 의존성 | `src/visionlab`의 anomalib import, 순수 파일의 Lightning import, 추론 경로의 외부 라이브러리 |
| 자산 | 다운로드 경로, 로컬 가중치 연결 |
| 범위 | 요청하지 않은 파일·추상화, D-15 배치 위반 |

### 6.2 지적과 수정 요청

지적은 Critical / Major / Minor와 원본·이식 파일의 위치, 재현 조건으로 적는다. Critical과 Major는 수정을 요청하고, Minor는 수정 여부와 근거를 기록한다. 수정 요청은 최대 3회이며, 넘기면 남은 지적과 위험을 사용자에게 보고한다.

```text
[Phase N 수정 요청 k/3: <model>]
지적: <심각도> <원본 위치> vs <이식 위치>: <차이와 영향>
요청: <수정할 내용>
제약: 5.1과 같다
돌려줄 것: 변경 파일 목록, 지적별 처리 내용
```

### 6.3 기록

결과 파일의 4절(적대적 검증)에 회차별 지적, 처리, 남은 위험을 남긴다. 이 프로젝트는 워크스페이스 `ADVERSARIAL-REVIEW.md`의 `docs/reviews/A{n}.md` 대신 Phase 결과 파일에 기록한다. codex의 이식 메모는 Claude가 검증한 뒤 3절에 기록한다.

## 7. 이식 결과 실행 (agy)

Claude가 `evaluation/visionlab/train.py`(native Lightning 학습·평가, LitCLI, yaml로 모델 선택)와 `evaluation/visionlab/predict.py`(순수 PyTorch 추론·평가, `--model` 인자)로 명령을 만들어 agy에 보낸다. 두 스크립트는 모델마다 만들지 않고 공용으로 쓰며, 모델별 차이는 모델 yaml과 LitModel에서 처리한다. native Lightning 학습·평가와 순수 PyTorch 추론을 category마다 실행한다. 학습 예산은 레퍼런스와 같다.

단위 테스트는 같은 방식으로 pytest 명령을 보낸다(D-21).

```powershell
& $py -m pytest tests\NN_<name>\test_<file>.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\<NN>-<step>-<YYMMDD>-<n>.log
```

### 7.1 agy 지시 양식

```text
[Phase N 실행: <model> <reference|native|pure>]
작업 폴더: D:\projects\visionlab
명령: 아래 명령만 순서대로 그대로 실행한다. 명령을 고치거나 파일을 수정하지 않는다.
로그: 각 명령의 출력을 `2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\<NN>-<step>-<YYMMDD-HHMMSS>.log`로 함께 저장한다(명령 목록에 Claude가 붙여 보낸다).
  <명령 목록>
돌려줄 것: 명령별 종료 코드, 소요 시간, 마지막 metric 표, 오류가 있으면 traceback 전문, 생성된 run 폴더 경로, 로그 파일 경로
```

### 7.2 실행 명령의 기록

Claude는 agy에 보낸 명령을 그대로 결과 파일(`docs/results/NN-*.md`)의 해당 절에 코드 블록으로 적는다. 레퍼런스 실행은 2절, 단계별 테스트와 이식 결과 실행은 그 실행을 다루는 절(3~5절)에 적는다. 적는 내용은 다음과 같다.

- 작업 폴더와 파이썬 전체 경로, 인자, `Tee-Object`로 저장한 로그 경로를 명령 그대로 적는다. 실행 일시와 종료 코드도 함께 적는다.
- 중단하거나 실패한 명령, 사용자 결정으로 취소한 명령도 지우지 않고 그 사실을 적는다.
- agy의 자기 보고가 아니라 Claude가 보낸 명령 원문을 적는다. agy가 지시와 다른 명령을 실행했다면 실제 실행된 명령과 차이를 적는다.

## 8. 수치 판정 (Claude)

Claude는 agy의 결과로 다음을 판정한다.

- native Lightning의 category별 Image·Pixel AUROC와 레퍼런스의 차이가 각각 0.01 이내인가
- 순수 PyTorch 추론의 category별 Image·Pixel AUROC와 레퍼런스의 차이가 각각 0.01 이내인가
- image metric만 내는 모델은 Image AUROC만 판정한다

판정과 함께 레퍼런스와 native Lightning 학습의 에폭별 학습 히스토리(`train_loss_epoch` 등)를 에폭 순서로 나란히 표에 적는다. 이 비교의 허용 오차는 사용자가 정하기 전까지 합격 조건이 아니며, 어긋남이 크면 원인을 진단한다.

기준 미달이면 원인을 진단해 codex에 수정을 요청하고 단계 3으로 돌아간다(최대 3회). 진단에는 같은 state·입력에서의 raw 출력 비교(FR-5)와 native·순수 출력 비교를 쓸 수 있다. 결과는 결과 파일의 5절에 비교 표로 남기며, 기준 미달 결과도 지우지 않고 원인과 판단을 함께 적는다.

## 9. 보고

Claude는 Phase 끝에 사용자에게 다음을 보고한다. 사용자가 Phase 종료를 결정한다.

- 레퍼런스와 이식 결과의 비교 표, 판정
- 적대적 검증의 회차와 남은 지적
- 수정 반복 횟수, 미충족 항목과 남은 위험

## 10. 커밋·푸시

사용자가 Phase 종료를 결정하면 Claude가 그 Phase의 변경을 커밋하고 `origin`에 푸시한다. 커밋 메시지는 `Phase NN: <모델 폴더 이름> <요약>` 형식이다. 커밋에는 코드, yaml, 문서(`docs/results/NN-*.md` 포함)를 담고 `outputs/`는 담지 않는다. Phase 중간에는 사용자가 요청할 때만 커밋·푸시하며, force push는 사용자 확인 없이 하지 않는다.
