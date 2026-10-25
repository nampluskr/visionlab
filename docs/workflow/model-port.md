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

단계 사이에는 파일과 출력만 넘긴다. codex와 agy에는 이전 대화의 해석 대신 원본 소스, `specs/`, 이 문서, 산출물 경로를 준다.

## 2. Phase 0 (데이터셋)

Phase 0은 모델이 아니라 데이터셋을 다룬다. 1절의 순서를 따르되 다음이 다르다.

- 레퍼런스 실행(1~2단계) 대신 anomalib 데이터(`LocalMVTecAD`, `LocalVisa`)와의 대조가 기준이다.
- codex는 데이터셋을 새로 구현한다(원본 복사 아님, D-18). 만들 파일과 완료 조건은 `specs/PLAN.md`의 Phase 0을 따른다.
- agy는 `evaluation/anomalib/00_compare_data.py`와 `tests/test_data.py`를 실행한다.
- Claude는 파일 목록, 라벨, mask, 분할, 최종 입력 텐서의 일치를 판정한다(FR-17).

## 3. 모델 Phase 시작 준비 (Claude)

Phase를 시작할 때 Claude가 다음을 준비한다.

- 결과 파일 `docs/results/NN-<kebab-case>.md`를 `docs/results/README.md`의 양식으로 만들고, 1절(모델 노트)을 채운다. 원본 소스 경로(`anomalib/models/image/<name>/`), 생성자 시그니처, 모델이 고정하는 `trainer_arguments`, 전처리·정규화, 제공 metric(image만인지), 외부 의존(FrEIA, open_clip 등), 로컬 자산 경로를 적는다. 노트는 색인이며, 판단의 근거는 원본 소스다.

## 4. 레퍼런스 실행과 기록

### 4.1 실행 명령 (Claude가 agy에 보냄)

`evaluation\anomalib`에서 category마다 2회 실행한다. 같은 설정의 두 번째 실행은 `v<N+1>` 폴더로 따로 남는다.

```powershell
cd D:\projects\visionlab\evaluation\anomalib
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/<name>.yaml
& $py anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/<name>.yaml
& $py anomalib_run.py train --config default.yaml --config data/visa.yaml --config models/<name>.yaml
& $py anomalib_run.py train --config default.yaml --config data/visa.yaml --config models/<name>.yaml
```

### 4.2 진단 (Claude)

Claude는 agy가 돌려준 출력과 run 폴더로 다음을 확인한다.

- 종료 상태와 오류, 네트워크 차단 오류(`network access is blocked`)가 없는지
- `Seed set to 42`, 실제 적용된 `max_epochs`와 `Overriding ...` 로그(모델 `trainer_arguments`)
- 저장된 `lightning_logs/version_*/config.yaml`의 경로·예산이 의도와 같은지
- 두 run의 AUROC 차이

두 run이 일치하지 않으면 레퍼런스로 채택하지 않고 원인(비결정 연산 등)을 기록한 뒤 사용자에게 보고한다.

### 4.3 기록

결과 파일의 2절(레퍼런스)에 다음을 적는다.

- 실행 일시, run 폴더(`outputs/anomalib/<Model>/<DataModule>/<category>/v<N>`)
- category별 Image·Pixel AUROC(image metric만 내는 모델은 Image만), 두 run의 차이
- 실제 적용된 학습 예산
- 실패나 이상이 있으면 지우지 않고 원인과 판단

## 5. 이식 구현 (codex)

### 5.1 지시 양식 (Claude가 codex에 보냄)

```text
[Phase N 이식 지시: <model>]
목표: anomalib 2.0.0 <model>을 native Lightning과 순수 PyTorch로 이식한다.
읽을 것: specs/INTENT.md, specs/SPEC.md, specs/PLAN.md(Phase N, 공통 양식), specs/DECISIONS.md(D-4, D-5, D-15),
  AGENTS.md(Python 코드 규칙), docs/workflow/model-port.md, docs/results/NN-<kebab-case>.md(1·2절),
  원본 C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\Lib\site-packages\anomalib\models\image\<name>\
쓸 수 있는 곳: evaluation/, src/, tests/
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
- 이식 메모에 원본 파일과 이식 파일의 대응, 의존성 제거로 바뀐 줄, 새로 구현한 대체 부품을 적는다.

## 6. 적대적 검증 (Claude)

Claude는 이식 결과가 원본과 같게 동작한다는 가정을 깨는 쪽으로 검증한다. 근거는 원본 소스와 이식 코드이며, codex의 설명은 근거로 쓰지 않는다.

### 6.1 공격 항목

검증할 항목은 다음과 같다.

| 항목 | 확인 내용 |
| --- | --- |
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

### 7.1 agy 지시 양식

```text
[Phase N 실행: <model> <reference|native|pure>]
작업 폴더: D:\projects\visionlab
명령: 아래 명령만 순서대로 그대로 실행한다. 명령을 고치거나 파일을 수정하지 않는다.
로그: 각 명령의 출력을 `2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\<NN>-<step>-<YYMMDD-HHMMSS>.log`로 함께 저장한다(명령 목록에 Claude가 붙여 보낸다).
  <명령 목록>
돌려줄 것: 명령별 종료 코드, 소요 시간, 마지막 metric 표, 오류가 있으면 traceback 전문, 생성된 run 폴더 경로, 로그 파일 경로
```

## 8. 수치 판정 (Claude)

Claude는 agy의 결과로 다음을 판정한다.

- native Lightning의 category별 Image·Pixel AUROC와 레퍼런스의 차이가 각각 0.01 이내인가
- 순수 PyTorch 추론의 category별 Image·Pixel AUROC와 레퍼런스의 차이가 각각 0.01 이내인가
- image metric만 내는 모델은 Image AUROC만 판정한다

기준 미달이면 원인을 진단해 codex에 수정을 요청하고 단계 3으로 돌아간다(최대 3회). 진단에는 같은 state·입력에서의 raw 출력 비교(FR-5)와 native·순수 출력 비교를 쓸 수 있다. 결과는 결과 파일의 5절에 비교 표로 남기며, 기준 미달 결과도 지우지 않고 원인과 판단을 함께 적는다.

## 9. 보고

Claude는 Phase 끝에 사용자에게 다음을 보고한다. 사용자가 Phase 종료를 결정한다.

- 레퍼런스와 이식 결과의 비교 표, 판정
- 적대적 검증의 회차와 남은 지적
- 수정 반복 횟수, 미충족 항목과 남은 위험
