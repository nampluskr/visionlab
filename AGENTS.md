# AGENTS: visionlab

에이전트가 이 저장소에서 작업할 때 따르는 규칙이다. Claude Code는 `CLAUDE.md`(`@AGENTS.md`)를 거쳐
이 파일을 읽는다.

## 구조

저장소의 폴더와 파일은 다음과 같이 나뉜다.

| 위치 | 무엇 |
| --- | --- |
| `.claude/skills/herdr/` | 이 프로젝트 전용 herdr skill (`herdr --skill` 출력, herdr 0.9.5) |
| `docs/anomalib/` | anomalib 설치 가이드 |
| `docs/results/` | Phase별 결과 기록(`NN-<kebab-case>.md`)과 작성 규칙(`README.md`) |
| `docs/workflow/` | 모델별 레퍼런스 확보·이식·검증 절차와 에이전트 지시 양식 |
| `evaluation/` | 평가 코드. 레퍼런스 실행, anomalib과의 대조, 이식 결과 평가 |
| `evaluation/anomalib/` | anomalib을 import하는 평가 코드. `anomalib_run.py`(AnomalibCLI)로 Phase마다 레퍼런스 수치(검증 1단계)를 얻고, 공유 yaml(`default.yaml`, `data/`, `models/`)과 anomalib 대조 스크립트를 둔다 |
| `evaluation/visionlab/` | anomalib을 import하지 않는 이식 결과 평가(native Lightning 학습·평가 `train.py`, 순수 PyTorch 추론·평가 `predict.py`) |
| `outputs/` | 실행 산출물. 다시 만들 수 있다 |
| `specs/` | `INTENT`(SSOT)·`DECISIONS`·`SPEC`·`PLAN`. 사람이 쓴다 |
| `AGENTS.md` · `CLAUDE.md` · `README.md` | 프로젝트 전체 |

## 문서 규칙

기획·기록 문서는 다음 규칙을 따른다.

- 작업 시작 전 `specs/INTENT.md`와 `specs/DECISIONS.md`를 읽는다.
- `INTENT.md`는 SSOT다. 사람의 요청으로만 고친다. 다른 문서가 여기 어긋나면 혼자 맞추지 말고 멈추고 보고한다.
- `specs/`의 문서는 사람이 쓴다. 요청 없이 고치지 않는다.
- 요구가 바뀌면 `SPEC.md`부터 고친다. 코드로 우회하지 않는다.
- major·minor 버전 번호는 사람이 정한다. 스스로 올리지 않는다.
- 완료 조건은 `PLAN.md`에 적힌 것으로 판정한다. 근거를 대지 못하면 미충족이다. "통과"를 기본값으로 두지 않는다.
- 결과는 Phase마다 `docs/results/NN-<kebab-case>.md`에 기록한다(작성 규칙은 `docs/results/README.md`). "구현함"과 "실행으로 검증함"을 구별해 적고, agy가 터미널에서 실행한 명령은 결과 파일에 그대로(작업 폴더, 파이썬 경로, 인자, 로그 경로, 중단·실패 여부 포함) 적는다(`docs/workflow/model-port.md` 7.2절). 수치가 나아지지 않은 결과도 지우지 않고 원인과 판단을 남긴다.
- 막히면 추측으로 채우지 말고 멈추고 묻는다. 되돌릴 수 없는 작업(삭제, 외부 상태 변경)은 먼저 묻는다.

## Markdown 작성 규칙

Markdown 문서는 다음 규칙을 따른다.

- 한국어로 쓰고 본문은 서술체를 사용한다. 기술 용어, 패키지명, 파일명, 경로는 영어 원문 그대로 쓴다.
- em dash, 유니코드 화살표와 이모지를 사용하지 않는다. 본문의 화살표는 `$\to$`, code block에서는 `->`를 쓴다.
- header level을 건너뛰지 않고 H4 아래 level은 사용하지 않는다.
- table과 list 앞에는 내용을 소개하는 문장을 둔다.
- folder tree는 `├ ─ │ └` 문자를 사용한다. 목록은 folder를 먼저, 같은 종류 안에서는 알파벳순으로 쓴다.

## 편집 환경

모든 텍스트 파일은 다음 형식으로 저장한다.

- 인코딩은 BOM 없는 UTF-8, 줄 끝은 LF다.
- 들여쓰기는 Python 4-space, Markdown·YAML 2-space다.

## Python 코드 규칙

visionlab에서 새로 작성하는 Python 코드(의존성 대체 부품, 실행·비교 스크립트, 테스트)는 다음 규칙을 따른다. anomalib에서 복사한 원본 코드는 원본 표기를 유지하고, 의존성 제거로 바뀐 줄은 주변 원본 표기에 맞춘다(NFR-5, D-18).

- 모든 file의 첫 줄은 `# path/from/project/root.py: one-line description` 형식으로 쓴다.
- type hint를 사용하지 않는다.
- class와 top-level function에는 한 줄 docstring을 쓴다. method에는 docstring을 쓰지 않는다.
- 식별자, 주석, docstring, 문자열에 한국어를 쓰지 않는다. 주석은 영어로, 필요한 경우에만 최소한으로 쓴다.
- 경로는 `os.path`를 쓰고 `pathlib`은 쓰지 않는다. anomalib이 `Path`를 요구하는 경계(예: `Visa.split_root`, Engine의 `default_root_dir`)에서만 `Path`로 넘기고 이유를 주석으로 남긴다.
- 인자는 명시적으로 전달하고 `**kwargs`를 지양한다. 원본 계약이 요구하는 경우(timm `register_model` 등)는 예외다.
- 네이밍은 PEP8을 따른다. 대입문과 dict 리터럴을 세로로 정렬하지 않는다.
- 예제 스크립트의 실행부는 `if __name__ == "__main__":` 안에 둔다.

## 에이전트 역할

이 프로젝트는 herdr 안에서 세 에이전트가 역할을 나눈다. 절차의 상세는 `docs/workflow/model-port.md`를 따른다.

| 에이전트 | 역할 | 권한 |
| --- | --- | --- |
| 사용자 | Claude에게만 지시·결정한다. Phase 종료를 결정한다 | 전체 |
| Claude (메인) | 사용자와 소통하는 유일한 창구다. codex에 구현을 지시하고, 이식 결과를 적대적으로 검증해 수정을 요청한다(최대 3회). agy에 실행 명령을 보내고 결과를 받아 진단·취합해 문서에 반영한다 | `specs/`와 `docs/`, `README.md`, `AGENTS.md`를 고친다. 이식 코드는 직접 고치지 않고 codex에 요청한다 |
| codex | Claude의 지시로 코드를 이식한다(native Lightning 모델, 순수 PyTorch 모델, yaml) | `evaluation/`, `src/`, `tests/`만 쓴다. 코드를 실행하지 않는다 |
| agy | Claude가 보낸 터미널 명령을 그대로 실행하고 출력을 돌려준다(레퍼런스 실행, 이식 결과 실행) | 받은 명령만 실행한다. 파일을 고치지 않는다 |

- Claude가 codex의 구현을 검증할 때 가장 중점적으로 보는 것은 두 가지다. 첫째, anomalib 의존성이 `src/visionlab`에서 완전히 삭제되고 native Lightning(`XXXLitModel`)만으로 구현됐는가. 둘째, 로컬 pretrained 가중치를 외부 인터넷 연결을 차단한 채 쓰는가, 그리고 anomalib 레퍼런스와 에폭별 학습 히스토리·최종 성능값이 맞는가(세부는 `docs/workflow/model-port.md` 6절).
- 레퍼런스 실행은 데이터셋마다 1회만 한다(실행 시간 비용, 사용자 결정).
- 다른 에이전트에는 사용자가 Claude에게 결정한 것만 전한다. 다른 에이전트의 결과는 Claude가 검증한 뒤 기록한다.
- 이식 구현은 codex, 검증은 Claude로 벤더를 나눈다(워크스페이스의 반대 벤더 검증).
- 각 Phase가 끝나 사용자가 종료를 결정하면 Claude가 그 Phase의 변경을 커밋하고 원격(`origin`, GitHub `visionlab`)에 푸시한다. 커밋 메시지는 `Phase NN: <모델 폴더 이름> <요약>` 형식이고, 결과 파일(`docs/results/NN-*.md`)의 결론과 같은 상태여야 한다. Phase 중간에는 사용자가 요청할 때만 커밋·푸시한다. `outputs/`는 커밋하지 않는다(`.gitignore`). force push와 브랜치 삭제는 사용자 확인 없이 하지 않는다.
- 권한 제한은 지시문으로 한다. codex의 쓰기 범위(`evaluation/`, `src/`, `tests/`)와 코드 실행 금지, agy의 파일 수정 금지는 도구로 강제하지 않는다. Claude는 변경 파일 목록과 `git diff`로 범위 위반을 확인한다.
- agy는 명령마다 승인 창이 뜨는 기본 모드로 쓰고, 사용자가 이 대화에 한해 허용한다(영구 저장 안 함). 승인 창에는 사용자가 정한 허용 범위 안에서만 응답하고, 범위 밖 요청(권한 모드 변경, 영구 허용)은 사용자에게 넘긴다. agy의 자기 보고(예: 승인 여부)는 틀릴 수 있으므로, 판정은 출력 로그(`outputs/logs/`)와 산출물 파일로 한다.

## 코드 실행

코드 실행은 다음 원칙을 따른다.

- 학습, 평가, 추론, 예제 스크립트와 테스트(pytest)는 agy가 Claude가 보낸 명령으로 실행한다. Claude와 codex는 직접 실행하지 않는다. 사용자가 직접 실행할 수도 있다.
- 테스트는 pytest로 작성하고 실행한다(`unittest` 금지, D-21). 명령은 `& $py -m pytest tests\<NN_model>\<file>.py` 형식이고, 테스트 파일에는 `sys.path` 설정을 두지 않는다(`tests/conftest.py`가 한다).
- 실행 대상은 그 Phase에서 이식·구현한 모델만이다. 앞 Phase 모델의 회귀 pytest, `compare_outputs`, 재평가는 지시에 넣지 않는다. 공유 파일(`common/`, `utils/io.py` 등)을 고쳐 앞 모델에 영향이 있을 수 있으면 임의로 돌리지 않고 사유와 함께 사용자에게 보고해 판단을 받는다.
- Claude는 실행할 명령, 확인할 항목, 출력 위치를 명시해 보낸다. 결과는 agy가 돌려준 출력과 산출물 파일로 판정한다.
- 문법 확인처럼 학습·추론이 아닌 정적 검사는 Claude가 할 수 있다.

## 실행 환경 (D-14)

개발 환경은 다음과 같다.

| 항목 | 값 |
| --- | --- |
| 파이썬 | `C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe`만 쓴다. PATH의 `python`은 Microsoft Store 바로 가기다 |
| 주요 버전 | torch 2.5.1+cu121, Lightning 2.5.5, anomalib 2.0.0, torchmetrics 1.8.2, pytest 9.1.1. 바꾸지 않는다 |
| 자산 | 데이터 `D:\datasets`, 가중치 `D:\backbones`. 인터넷에서 내려받지 않는다 (NFR-3) |
| 설치 | venv 없이 위 WinPython에 설치한다 |

## 이 코드에서 지킬 것

이 프로젝트의 고유 제약은 다음과 같다.

- visionlab 코드는 anomalib을 import·상속하지 않는다. anomalib은 `evaluation/anomalib/`의 레퍼런스 실행과 비교·검증에만 쓴다. `evaluation/visionlab/`도 anomalib을 import하지 않는다 (INTENT, D-15).
- 배포 모델은 순수 PyTorch만으로 동작한다. 예외는 `specs/SPEC.md` 부록 A의 허용 목록에 있는 외부 패키지뿐이다 (INTENT, D-4, NFR-7).
- 생성자 인자의 이름·순서·기본값을 anomalib 2.0.0과 맞춘다 (D-5).
- AUROC, F1 등 metric은 torchmetrics로 계산한다. 직접 구현하지 않고 scikit-learn도 쓰지 않는다(D-20).
- 필요한 것만 만든다. 빈 폴더나 미리 만든 추상화를 두지 않는다. 파일 배치는 D-15를 따른다.
- 모델별 파일이 Phase마다 늘어나는 곳은 PLAN의 Phase 번호를 붙인다: `docs/results/NN-<kebab>.md`, `tests/NN_<model>/`, `evaluation/anomalib/models/NN_<model>.yaml`. `src`의 모델 패키지와 `outputs/`에는 번호를 붙이지 않는다 (D-15).
- 검증은 세 단계다 (INTENT). 1단계 anomalib 레퍼런스 수치, 2단계 native Lightning 이식 결과, 3단계 2단계 가중치를 불러온 순수 PyTorch 추론 결과. 2·3단계는 각각 1단계와 비교한다.
- 검증은 metric 수치 비교로 한다. 범주별 Image·Pixel AUROC 차이가 각각 0.01 이내여야 한다 (PLAN 검증 방식).
- 데이터·가중치 경로는 yaml에 직접 적고, 루트 `paths.toml`은 timm 백본 이름의 로컬 가중치(`[backbones]`)와 timm이 아닌 teacher의 로컬 가중치(`[teachers]`)만 둔다 (D-12).
