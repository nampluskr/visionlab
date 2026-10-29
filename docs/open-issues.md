# 미진 사항과 마감 점검

작성일은 2026-10-29이고 마감 선언 시점의 갱신본이다(기준 커밋 `ff1db68`에 이어지는 문서 커밋). 이 문서는 프로젝트 마감을 앞두고 끝내지 못했거나 알려진 한계인 항목을 한곳에 모은 것이며, 결정이 필요한 항목은 결정을 대신하지 않고 후보만 적었다.

## 마감 선언

다음은 이 프로젝트의 마감 선언이다. `[가정]` 표시는 사용자가 따로 정하지 않아 권고안으로 채운 항목이며 사용자가 바꿀 수 있다.

| 항목 | 내용 |
|---|---|
| 버전 | v0.1 `[가정]` |
| 날짜 | 2026-10-29 |
| 검토 루프 종료 | 사용자 판단으로 종료(에이전트가 판정하지 않음). 미진 사항은 이 문서에 남긴다 |
| 마감 절차 변형 | `PROGRESS.md`는 새로 만들지 않고 이 문서로 대신한다. `docs/history/` 복사는 하지 않고 태그와 `specs/`, `docs/results/` 기록을 유지한다. `DECISIONS`의 승격은 `AGENTS.md`로 하지 않고 미진 사항(항목 1, 2)에 남긴다 |
| 태그 | 주석 태그 `v0.1` `[가정]` |
| 마감 전 확인 | 전체 pytest 회귀 1회: 20개 폴더 551 passed, 0 failed(`docs/results/final-regression.md` "마감 전 재실행") |
| 저장소 위생 | `.serena/` ignore, 루트 `LICENSE`(Apache-2.0)와 `NOTICE` 추가. 저작권자는 `[저작권자 확인 필요]`로 비워 두었다 `[가정]` |
| 남은 미진 사항 | 34건 중 해소 2건을 뺀 32건: 사용자 결정 필요 8, 구현 필요 2, 확인 필요 6, 알려진 한계로 수용 16(2.1절) |

## 프로젝트 요약

visionlab은 anomalib 2.0.0의 이상 탐지 image 모델 18개를 native Lightning과 Lightning 없이 도는 PyTorch 모델로 이식하고, 같은 데이터·학습 예산으로 실행한 anomalib 레퍼런스와 category별 Image·Pixel AUROC 차이 0.01 이내로 비교·검증한 프로젝트다. 데이터셋 Phase 0과 모델 Phase 1~18(STFPM, EfficientAD, PaDiM, DFM, DFKDE, PatchCore, Reverse Distillation, CFA, FRE, SuperSimpleNet, DRAEM, FastFlow, CFLOW, UFlow, CSFlow, GANomaly, DSR, WinCLIP)을 모두 마쳤고 각 Phase는 사용자가 종료를 결정해 `Phase NN: ...` 커밋으로 푸시했다(커밋 19개, git tag 없음). 각 Phase 결과 문서는 native Lightning과 export를 읽은 순수 PyTorch 추론이 레퍼런스의 판정 기준을 충족했다고 기록한다(예외: Phase 6은 visa/candle 대신 mvtec/capsule, Phase 11은 mvtec/bottle만, 에폭은 Phase 9가 220 대신 10, Phase 13이 1). 2026-10-28 전체 pytest 회귀는 549 passed, 2 failed였고 실패한 csflow 두 건은 원인(검증 점수 동점으로 정규화가 0/0)을 진단해 테스트 비교를 `equal_nan=True`로 고쳤다. 마감 전 2026-10-29에 전체 회귀를 다시 실행해 551 passed, 0 failed를 한 번에 측정했다. 문서는 `specs/`(INTENT, DECISIONS, SPEC, PLAN), `docs/results/NN-<kebab-case>.md`(Phase별 기록 19개)와 `docs/results/final-regression.md`(전체 회귀와 csflow 진단), `docs/workflow/model-port.md`(이식 절차), `docs/anomalib/`(설치 가이드), `docs/dependency-exceptions.md`·`docs/intent-appendix-draft.md`·`docs/cleanup-proposals-draft.md`(의존 정리 초안), `README.md`, `AGENTS.md`에 있다.

## 1. 마감 전 상태 점검

### 1.1 마감 조건 4개

`project-workflow/docs/VERSIONING.md` 3절의 조건 네 개를 이 프로젝트에 대조한 결과는 다음과 같다. 이 프로젝트는 `specs/`를 쓰고 `PROGRESS.md`, `docs/current/`, `docs/history/`, `BRIEF.md`, `backlog.json`이 없으며 `git tag`도 없다.

| # | 조건 | 충족 여부 | 근거 |
|---|---|---|---|
| 1 | `PLAN.md`의 전 Phase 완료 | 충족 | `backlog`를 쓰지 않으므로 Phase 완료 조건으로 판정한다. `docs/results/00~18` 19개 문서가 모두 "6. 결론"에 판정과 Phase 종료 결정을 기록하고, `Phase NN:` 커밋이 19개(Phase 0~18) 있다 |
| 2 | 각 Phase 검증 게이트 통과 | 충족(예외 있음) | 각 결과 문서의 판정(Image·Pixel AUROC 0.01 이내, FR-5·FR-6·FR-23·FR-24, pytest)과 적대적 검증 기록. 단 PLAN 공통 양식을 다 채우지 못한 Phase가 있다: Phase 6(visa/candle 미판정), Phase 11(visa 미평가), 에폭 예외 Phase 9, 13. 이 예외는 사용자 결정으로 결과 문서에만 기록했고 `PLAN.md`는 고치지 않았다 |
| 3 | 검토 루프 종료 | 충족(사용자 판단) | 사용자가 종료를 판단했다(2026-10-29). 이 문서의 미진 항목이 검토 루프의 재료였다 |
| 4 | `PROGRESS.md`에 빠진 항목이 없다 | 적용 불가 | `PROGRESS.md`가 없다. task 수와 기록 수를 대조할 대상이 없고, 대응하는 기록은 `docs/results/NN-*.md`와 git 이력이다. 조건 4를 어떻게 읽을지는 사람이 정해야 한다 |

### 1.2 마감 절차 10단계 대조

VERSIONING.md 4절의 10단계를 이 프로젝트에 대조했다. 버전 번호는 사람이 정하므로 제안하지 않는다.

| # | 단계 | 구분 | 이 프로젝트에서 |
|---|---|---|---|
| 1 | 마감 조건 4개 확인 | 적용 가능 | 1.1절의 표 |
| 2 | 마감 선언·버전 번호 확정(사람) | 적용 가능 | 번호와 선언 시점은 사용자 결정 |
| 3 | `PROGRESS.md` 마무리(계획 대비, 계획 외 개선, 남긴 것, 다음 버전으로) | 변경 필요 | `PROGRESS.md`가 없다. 이 문서의 요약과 아래 표가 "남긴 것"과 "다음 버전으로"에 해당한다. `PROGRESS.md`를 새로 만들지, 이 문서로 대신할지 결정 필요 |
| 4 | 승격 판정(`DECISIONS.md` 항목 분류, 사람 승인) | 변경 필요 | D-1~D-28을 "앞으로 지켜야 할 제약"과 "근거"로 나눈다. 제약의 승격 대상은 `CLAUDE.md`/`.claude/rules/`인데 이 프로젝트의 `CLAUDE.md`는 `@AGENTS.md` 한 줄이고 `.claude/rules/`가 없다. 승격 위치는 `AGENTS.md`가 된다 |
| 5 | `README.md` 갱신(사람 승인) | 적용 가능 | README는 이미 갱신했다(실제 동작 기준). 마감 때 요구 환경·사용법·알려진 한계를 재확인 |
| 6 | `docs/history/vX.Y/` 생성 | 변경 필요 | `docs/history/`가 없다. 만들 경우 `docs/current/`가 없으므로 복사 원본이 `specs/`가 된다 |
| 7 | `docs/current/` 문서를 `INTENT.md` 빼고 복사 | 변경 필요 | `docs/current/`가 없고 `specs/`가 그 역할이다. 복사 대상은 `specs/`의 DECISIONS, SPEC, PLAN이며 BRIEF, PROGRESS, backlog는 없다. `docs/results/`를 포함할지 결정 필요. INTENT 제외 규칙(SSOT는 한 벌)은 그대로 적용 가능 |
| 8 | 복사본과 원본 대조 | 적용 가능 | 6~7단계를 하면 해시·diff 대조 |
| 9 | 커밋·`git tag vX.Y`·push | 적용 가능 | tag가 아직 없다. tag 이름과 방식(경량/주석)은 사용자 결정 |
| 10 | 보고 | 적용 가능 | |

적용 불가로 볼 부분은 조건 4(`PROGRESS.md`)와 `docs/current/` 상주를 전제로 한 `VERSIONING.md` 9절의 마감 검사 목록(`docs/history/vX.Y/` 존재, `docs/current/`의 INTENT 상주, `PROGRESS.md` 요약 블럭)다. 이 프로젝트는 `INIT.md`의 문서 구조(`docs/current/`)로 착수한 프로젝트가 아니므로 이 검사 목록을 그대로 적용하면 항상 걸린다. 구조를 맞출지 이 프로젝트용 마감 절차로 바꿀지는 사용자 결정이다.

## 2. 미진 사항

다음은 이 프로젝트에서 끝내지 못했거나 알려진 한계인 항목이다. 출처는 `docs/results/*.md`, `docs/cleanup-proposals-draft.md`, `docs/dependency-exceptions.md`, `docs/intent-appendix-draft.md`, 그리고 이 대화에서 보고한 항목이다. 분류는 "사용자 결정 필요", "구현 필요", "확인 필요", "알려진 한계로 수용", "해소" 중 하나다. Phase별 "확인하지 못한 것"은 중복을 합쳐 반영했고, Phase 1~18의 `6.2 남은 위험` 항목 중 이미 해소된 것(앞 모델 회귀 보류는 2026-10-28 전체 회귀로, `specs/` 갱신 항목은 각 Phase의 반영으로, `tests/00_dataset` 재실행은 전체 회귀 10 passed로, csflow 간헐 실패는 `equal_nan`으로)은 표에서 뺐다.

| 번호 | 항목 | 현재 상태(사실) | 영향 | 해소 방법 후보 | 사용자 결정 필요 | 분류 | 출처 |
|---|---|---|---|---|---|---|---|
| 1 | INTENT 밖의 "순수 PyTorch" 용어 | INTENT의 5행과 15행은 승인에 따라 고쳤다(INTENT에는 "순수 PyTorch"가 0곳이고 읽어서 INTENT 내부 불일치가 없음을 확인했다). 나머지 SPEC 3, DECISIONS 5, PLAN 7, AGENTS.md 3, README 4곳(합계 22곳, `docs` 초안과 `docs/results`는 제외)은 일괄 치환하지 않았다 | 다른 문서의 "순수 PyTorch"가 INTENT의 "Lightning과 anomalib 없이"와 같은 뜻인지 독자가 따로 해석해야 한다. README 4행("순수 PyTorch만으로 배포")과 6행의 새 문장이 한 단락에서 어긋날 소지가 있다 | (1) 일괄 치환, (2) "순수 PyTorch" 정의 문장 추가, (3) 그대로 둔다 | 예 | 사용자 결정 필요 | `docs/cleanup-proposals-draft.md` 1.1, 3.3절 |
| 2 | 낡은 인용 문구 | `DECISIONS.md` D-26 "남은 한계"(386행)가 INTENT의 "PyTorch만"을 인용한다. D-4 근거(26행)와 D-19 근거(329행)는 "순수 PyTorch만으로"를 쓴다. `SPEC.md` NFR-7 판정 칸은 "허용 목록 밖의 import를 차단한다"로 옛 용어이고 현재 테스트는 목록 대조를 하지 않는다 | 읽는 사람이 현재 INTENT와 어긋난 문장을 만난다. NFR-7 판정 칸은 아직 코드로 충족되지 않는 문장이다 | 문구만 수정(결정은 사용자). NFR-7 판정 칸은 5번 항목의 테스트 확장 때 함께 정리 | 예 | 사용자 결정 필요 | 이 대화의 (b)안 반영 보고 |
| 3 | requirements 후보의 버전 범위와 설치 확인 | 추론 경로 합집합 후보(torch 2.5.1+cu121, torchvision 0.20.1, torchmetrics 1.8.2, timm 1.0.22, numpy 2.2.5, kornia 0.8.2, FrEIA 0.2, scipy 1.15.3, tqdm 4.70.0, mpmath 1.3.0, 지연 scikit-learn 1.7.2, regex 2026.7.19)는 이 개발 환경의 설치 버전이며 배포 환경에서 설치해 본 적이 없다. 버전 범위는 정해지지 않았다. `SPEC.md` 부록 A에 확인 버전 표만 있다 | 배포 환경에서 `Predictor`가 실제로 import되고 동작하는지 확인되지 않았다(INTENT의 배포 목적에 직결) | 깨끗한 환경(Lightning·anomalib 없음)에 후보를 설치하고 18개 모델의 export 로딩·추론을 실행해 확인. 범위는 확인 뒤 requirements 파일로 확정 | 예(버전 범위), 확인은 agy 실행 | 확인 필요 | `SPEC.md` 부록 A, `docs/cleanup-proposals-draft.md` 2.2절 |
| 4 | Predictor 모델별 지연 import | `inference/predictor.py`가 14개 모델 `model.py`를 최상단에서 import한다(FastFlow, CFLOW, UFlow, CSFlow는 지연). 한 모델만 배포해도 6개 패키지(kornia, numpy, timm, torchmetrics, torchvision, tqdm)가 필요하다. 이 전제에서는 필수 수정이 아니라 요구를 줄이는 선택이다 | 배포 요구가 모델 수에 관계없이 큰 합집합이 된다. 지연 import 시 patchcore를 뺀 17개 모델의 요구가 줄고 최대 6개에서 2개까지 줄어든다 | 1단계: `Predictor` 모델 import 지연. 2단계: `PreProcessor`(`center_crop`)와 `PostProcessor`(torchmetrics) 지연. 리팩토링이라 codex 구현, Claude 적대적 검증, 전 모델 export 회귀 필요 | 예 | 구현 필요 | `docs/cleanup-proposals-draft.md` 2.3, 5.2절 |
| 5 | `test_dependency.py`의 빈 곳 | (1) 동적 import(`importlib`, `__import__`)는 정적 검사가 못 본다. (2) 차단 이름이 `anomalib`, `lightning`, `pytorch_lightning`뿐이라 `lightning_fabric`, `lightning_utilities`는 차단하지 않는다(후자는 torchmetrics가 요구하며 Lightning 본체가 아니다). (3) NFR-7(부록 A 대조)을 검사하는 코드가 없다. 실제 제약(Lightning·anomalib 비의존)은 현재 테스트가 보장한다 | 부록 A에 없는 새 외부 패키지가 추론 경로에 들어와도 테스트가 잡지 못한다 | 외부 패키지 허용 집합 대조 추가, 지연 import 전용 튜플(tqdm, mpmath 등), 필요하면 `lightning_fabric` 차단. codex 구현 | 예(범위) | 구현 필요 | `docs/cleanup-proposals-draft.md` 1.2, 5.1절 |
| 6 | 라이선스 선택과 저작권자 확인 | 안 A(Apache-2.0 + NOTICE)를 가정해 루트에 `LICENSE`(Apache-2.0 전문)와 `NOTICE`를 추가했다. 저작권자는 `[저작권자 확인 필요]`로 비워 두었고 `LICENSE` 부록의 저작권 줄은 표준 자리표시자(`[yyyy] [name of copyright owner]`)다. 안 A는 사용자가 따로 정하지 않은 권고안이다. 변경 표시 의무 이행 방식은 정하지 않았다 | 저작권자 미정이라 라이선스 고지가 완성되지 않았다. 안 B(MIT)로 바꾸려면 파일을 다시 써야 한다 | 사용자가 안 A 확정 여부와 저작권자를 정한다. 법률 자문이 아니므로 전문가 확인 권장 | 예 | 사용자 결정 필요 | `docs/cleanup-proposals-draft.md` 6.3절 |
| 7 | 출처 표기 없는 `.py` 19개 | `src/visionlab`의 `.py` 126개 중 SPDX Apache-2.0 표기 105개, 저작권 줄만 2개(winclip `clip.py`, `tokenizer.py`), 표기 없음 19개(공통 부품 9, Phase 1의 `models/stfpm/lightning_model.py` 1, visionlab 고유 9). `evaluation/` 16개와 `tests/` 44개도 표기 없음. 복사본의 변경 표시도 따로 없다. 사용자 지시(2026-10-29)로 헤더 추가는 하지 않기로 했다 | 새 구현인지 anomalib 파생인지 경계가 불분명한 파일(예: `stfpm/lightning_model.py`)이 있다. Apache-2.0의 변경 표시 의무와 관련될 수 있다. `NOTICE`가 변경 표시를 파일별로 하지 않았음을 밝힌다 | 필요해지면 파일별 출처 분류 후 헤더 추가 | 아니오 | 알려진 한계로 수용 | `docs/cleanup-proposals-draft.md` 6.1, 6.2절 |
| 8 | CFA 레퍼런스 BatchNorm 오염 확인 | 정적 점검상 CFA는 `Cfa.trainer_arguments`가 `num_sanity_val_steps = 0`이라 `initialize_centroid`와 `get_scale`의 forward가 eval 상태로 돌아 오염이 없다고 판정했다(조건부, 실행 미확인). 오염이 확인된 모델은 CSFlow 하나이고 처리했다 | CFA 레퍼런스가 오염됐다면 native와의 비교 기준이 흔들린다(native가 같은 동작을 재현했다면 영향 없을 수 있다) | 기존 CFA 레퍼런스 체크포인트의 backbone buffer를 가중치 파일과 비교하는 읽기 전용 진단(`diagnose_csflow_buffers.py` 방식, 재실행 불필요) | 예(진단 실행) | 확인 필요 | `docs/cleanup-proposals-draft.md` 7.2, 7.3절 |
| 9 | WinCLIP few-shot 미실행 | `k_shot > 0` 경로(순수 모델의 `_collect_visual_embeddings`, `_compute_few_shot_scores`와 Lit의 `collect_reference_images`)는 이식했으나 실행하지 않았다. `few_shot_source`와 `apply_transform=True`는 `NotImplementedError`다 | few-shot 사용 시 동작 보증이 없다. `few_shot_source`를 지정하는 anomalib 사용 방식은 지원하지 않는다 | few-shot 실행과 AUROC 비교 Phase 추가, `few_shot_source`용 이미지 로더 구현 | 예(범위) | 알려진 한계로 수용 | `docs/results/18-winclip.md` 6.2절 |
| 10 | 전체 회귀 합계 551 확정 | 해소: 2026-10-29 마감 전 전체 회귀를 20개 폴더 모두 새로 실행해 551 passed, 0 failed를 한 번에 측정했다. 정정 합계와 같다 | 해소됨. 다만 csflow fixture는 시드가 없어 항목 11이 남는다 | 없음 | 아니오 | 해소 | `docs/results/final-regression.md` 정정 합계 |
| 11 | csflow 테스트 fixture의 비결정성 | `fitted_csflow`가 시드 없이 모델을 만들고 임의 초기화 EfficientNet-B5 특징으로 학습해, 검증 점수가 정확히 같아지면 `PostProcessor` 정규화가 0/0이 된다(시드 0, 2에서 재현, 시드 없는 전체 실행 5회 중 3회). `equal_nan=True`로 완화했고 NaN이 양쪽에서 같은 위치일 때만 통과한다 | 점수가 동점인 실행은 정규화 값 검증이 약해진다. 두 경로가 똑같이 NaN인 버그는 이 세 비교에서 잡지 못한다(raw 값은 유한성 검사로 보호) | fixture 시드 고정(B안), 점수가 구분되는 fixture 설계, `PostProcessor`의 `max == min` 처리(전 모델 영향, 별도 결정) | 예 | 알려진 한계로 수용 | `docs/results/final-regression.md` csflow 진단 |
| 12 | `core.autocrlf true`와 `.gitattributes` 없음 | 전역 `core.autocrlf`가 `true`이고 저장소에 `.gitattributes`가 없다. 편집 규칙은 LF(AGENTS.md)인데 git 명령마다 "LF will be replaced by CRLF" 경고가 나온다 | 작업 트리는 LF이지만 체크아웃 환경에 따라 줄 끝이 CRLF로 바뀔 수 있고 diff 노이즈가 생긴다. 지금까지 경고는 그대로 두었다 | `.gitattributes`에 `* text=auto eol=lf` 등 명시 | 예 | 사용자 결정 필요 | 이 대화(커밋 때 경고) |
| 13 | 초안 문서 정리 | (b)안 반영과 문서는 모두 커밋했다. `docs/dependency-exceptions.md`, `docs/intent-appendix-draft.md`, `docs/cleanup-proposals-draft.md`는 안 A 시점의 "예외" 틀을 역사 기록으로 보존한 초안이며 파일명에 `draft`가 남아 있다 | 초안 문서가 현재 문구와 달라 독자를 혼동시킬 수 있다 | 초안 문서의 보관·삭제·`docs/history`로의 이동 결정 | 예 | 사용자 결정 필요 | 이 대화 |
| 14 | .serena/ untracked | 해소: `.gitignore`에 `.serena/`를 추가했다(커밋 `ff1db68`) | 해소됨 | 없음 | 아니오 | 해소 | 이 대화(커밋 절차) |
| 15 | 시각화 실행 미검증 | 18개 모델 yaml이 모두 `visualizer: false`다. `ImageVisualizer`, `LitVisualizer`를 참조하는 테스트는 `tests/01_stfpm` 두 파일뿐이고(grep), D-9의 기본값(True)으로 실제 학습·평가를 돌린 기록이 없다. CFA의 `_CfaLitVisualizer`와 DRAEM의 3차원 `anomaly_map` 시각화는 단위 테스트 또는 미실행이다 | 기본 설정으로 쓰면 시각화 경로가 처음 실행되며 모델별로 실패할 수 있다 | 모델별 `visualizer: true` 스모크 실행(소규모) 또는 기본값 재검토 | 예 | 확인 필요 | `docs/results/08-cfa.md`, `11-draem.md` 6.2절, `configs/models/*.yaml` |
| 16 | Phase 6 PatchCore visa/candle 미판정 | visa/candle의 coreset 선택이 약 4시간 40분 걸리고(705,600 x 1536 임베딩, GPU 메모리 한도 근접, 원인 미확인) 사용자 결정으로 평가 category를 mvtec/bottle과 mvtec/capsule로 바꿨다. `PLAN.md`는 고치지 않고 결과 문서에만 예외로 기록했다 | PatchCore는 VisA에서 검증되지 않았다. 3단계 검증의 데이터셋 범위가 이 모델만 다르다 | visa 판정 실행(시간·메모리 개선 필요) 또는 PLAN에 예외 명시 | 예 | 사용자 결정 필요 | `docs/results/06-patchcore.md` 2.2절, 6.2절 |
| 17 | Phase 11 DRAEM은 mvtec/bottle만 | visa/candle은 사용자 결정으로 제외했다. 레퍼런스 앞쪽 에폭이 뒤쪽보다 2~3배 느린 원인은 측정하지 않았다. SSPCAB 학습(`enable_sspcab=True`)은 forward 일치와 단위 테스트만 확인했고 학습 실행은 하지 않았다. `MultiStepLR` milestone `[400, 600]`은 10 epoch에서 작동하지 않아 검증하지 못했다. 일부 테스트가 `cuda`를 매개변수로 직접 써서 GPU 없는 환경에서 실패할 수 있다 | DRAEM은 VisA에서 검증되지 않았고 SSPCAB·scheduler 경로는 실행 검증이 없다 | visa 판정 실행, SSPCAB 학습 스모크, 테스트의 장치 처리 수정 | 예 | 사용자 결정 필요 | `docs/results/11-draem.md` 6.1, 6.2절 |
| 18 | Phase 9 FRE 10 epoch 통일 | 원본이 고정한 220 epoch가 아니라 공유 10 epoch(레퍼런스와 native 모두)로 비교했다. 220 epoch 레퍼런스는 사용자 지시로 중단했다(D-22에 기록) | 원본이 의도한 수렴 상태의 AUROC는 확인하지 못했다 | 220 epoch 레퍼런스·native 비교 실행(레퍼런스와 native 합계 약 4.5시간 예상) | 예 | 알려진 한계로 수용 | `docs/results/09-fre.md` 6.2절, D-22 |
| 19 | Phase 13 CFLOW 1 epoch | 학습 시간 때문에 사용자가 Phase 시작 전에 1 epoch로 줄였다(D-22). 에폭 히스토리 비교가 1행뿐이다. `permute_soft=True`와 다른 백본(`resnet18`, `resnet50`)은 구조 생성만 확인했다. 평가 지도가 batch 구성에 의존한다(원본 동작) | 10 epoch 이상에서의 차이는 알 수 없다. 한 장씩 추론하면 점수 분포가 달라진다 | 에폭 확대 비교, `Predictor`에 batch 의존 주의 문서화 | 예 | 알려진 한계로 수용 | `docs/results/13-cflow.md` 6.2절 |
| 20 | 공유 10 epoch 예산의 한계 | 원본이 의도한 학습량에서의 성능을 확인하지 못했다. 10 epoch 결과의 절대 성능이 낮은 모델이 있다(SuperSimpleNet bottle Image 0.7127, DRAEM Image 0.864/Pixel 0.792, EfficientAD visa Image 약 0.68, GANomaly mvtec 정규화 Image 0.447). 레퍼런스와 같은 값이라 이식 오류는 아니라고 판단했다 | 이식의 정확성은 확인됐으나 "원본 성능 재현"은 10 epoch 조건에 한정된다 | 대표 모델 몇 개에 대해 원본 권장 epoch로 비교 | 예 | 알려진 한계로 수용 | `docs/results/02, 10, 11, 14, 15, 16, 17` 6.2절 |
| 21 | Phase 12 FastFlow CaiT·DeiT 미검증 | `cait_m48_448`, `deit_base_distilled_patch16_384`는 로컬 가중치가 `paths.toml`에 없고 모델이 커서 코드만 복사했다. `FastflowLitModel`은 이 이름에서 `FileNotFoundError`를 낸다(NFR-3) | resnet18 외 backbone의 동작은 검증이 없다 | 가중치 확보 후 실행, 또는 지원 backbone을 문서에 명시 | 예 | 알려진 한계로 수용 | `docs/results/12-fastflow.md` 6.2절 |
| 22 | Phase 14 UFlow 입력·백본 | 원본 입력 448을 256으로 통일했다(D-27). 기본 backbone `mcait`는 로컬 가중치가 없어 실행하지 않았고, `wide_resnet50_2`는 구조 생성도 확인하지 않았다(`test_uflow.py`는 `resnet18`만 만든다) | 448 입력과 기본 backbone의 성능·동작은 검증이 없다 | `wide_resnet50_2` 구조 생성 테스트 추가, 448 입력 비교 | 예 | 알려진 한계로 수용 | `docs/results/14-uflow.md` 6.2절 |
| 23 | Phase 15 CSFlow 레퍼런스 보정 | 원본 anomalib 코드가 BatchNorm dry-run 오염을 안고 있어 레퍼런스 어댑터(`LocalCsflow`)가 backbone을 `eval()`로 넘기게 고쳤다(사용자 승인 예외). 3차 레퍼런스는 결정적이고 native와 일치하지만 원본 그대로의 결과가 아니다. 논문의 긴 학습과 768 입력의 성능은 확인하지 않았다 | 원본을 그대로 실행하면 mvtec처럼 거의 맞거나 visa처럼 퇴화할 수 있다 | 원본 그대로의 레퍼런스를 여러 번 실행해 편차 기록(시간 비용) | 예 | 알려진 한계로 수용 | `docs/results/15-csflow.md` 2.3, 2.4, 6.2절 |
| 24 | Phase 18 WinCLIP 해상도·export | Pixel 평가가 레퍼런스는 240, native·순수는 256이고(차이 2.7e-4 이하), 입력은 `Resize(256)` 다음 `Resize(240, bicubic)`라 원본의 직접 240 resize와 다르다(D-27, D-28). 레퍼런스 쪽 `Engine.export(TORCH)`의 forward hook pickle 문제는 이 Phase에서 쓰지 않아 남아 있다. export `model.pt`가 795.0MB다 | 해상도·resize 차이는 현재 수치에서는 판정에 영향이 없다 | 필요하면 240 직접 resize 비교, 레퍼런스 어댑터 export 훅 수정 | 아니오 | 알려진 한계로 수용 | `docs/results/18-winclip.md` 6.2절 |
| 25 | 레퍼런스가 category당 1회, run 간 편차 미측정 | 모든 Phase의 레퍼런스는 데이터셋마다 1회만 실행했다(사용자 결정). Phase 2는 원본 증강이 seed되지 않아 편차를 분리할 수 없었다. 일부 Phase는 native가 손실·가중치까지 비트에 가깝게 일치해 편차가 작아 보인다고만 적었다 | 레퍼런스 자체의 재현 편차를 모른다 | 대표 모델 레퍼런스 반복 실행 | 예 | 알려진 한계로 수용 | `docs/results/01~17` 6절(공통) |
| 26 | category 범위와 데이터 한정 | 모델마다 mvtec `bottle`과 visa `candle` 하나씩(Phase 6은 `bottle`·`capsule`, Phase 11은 `bottle`만)으로 판정했다. Phase 0의 입력 비교는 subset 앞쪽 8개뿐이고 test·val 정상 이미지 입력은 비교하지 않았으며, anomalib 2.0.0과 현재 로컬 데이터에 한정된다 | 다른 category·데이터셋에서의 동등성은 확인하지 않았다 | category 확대 실행(시간 비용 큼) | 예 | 알려진 한계로 수용 | `docs/results/00-dataset.md` 4절, 6절 |
| 27 | GPU·드라이버 의존 일치 | native와 레퍼런스가 비트에 가깝게 일치하는 것은 seed 42, 난수 소비 순서, 같은 GPU와 연산 덕분이다. DSR은 GPU forward 비교에서 VQ argmin 근접 동점이 뒤집혀 일부 항목이 최대 0.112까지 어긋났다(INFO). 다른 GPU·드라이버에서도 같다는 보장이 없다 | 환경이 바뀌면 일치 정도가 달라질 수 있다(판정 기준 0.01은 AUROC 기준) | 다른 환경에서 재실행 | 아니오 | 알려진 한계로 수용 | `docs/results/10~17` 6.2절 |
| 28 | 정규화 AUROC의 변별력 | D-24에 따라 레퍼런스의 정규화(clamp 포함) 점수로 비교한다. GANomaly mvtec은 raw 0.206, 정규화 0.447이고 DSR mvtec Image는 raw 0.844, 정규화 0.667이다. GANomaly는 레퍼런스 체크포인트의 raw AUROC가 native raw와 8자리까지 같음을 확인했으나 DSR과 DFM은 레퍼런스 raw를 구하지 않았다 | clamp 동점으로 판정 지표가 둔감한 모델이 있다 | DSR·DFM 레퍼런스 체크포인트 raw AUROC 확인 | 예 | 확인 필요 | `docs/results/16-ganomaly.md` 5.5절, `17-dsr.md`, `04-dfm.md` 6.2절 |
| 29 | 에폭 히스토리 허용 오차 미정 | 에폭별 `train_loss_epoch` 비교의 허용 오차를 사용자가 정하지 않아 판정에서 제외하고 표로만 기록했다(Phase 1, 2, 3 등). 실제 상대 차이는 대부분 1e-6 이하, EfficientAD는 mvtec 최대 약 0.9%, visa 최대 약 3.7%였다 | 히스토리 일치는 합격 조건이 아니다 | 허용 오차 결정(예: 1e-4 상대) | 예 | 사용자 결정 필요 | `docs/results/01, 02, 03` 6절, `docs/workflow/model-port.md` 8절 |
| 30 | Phase 1~5 개별 확인 미비 항목 | Phase 1: FR-24 임계값 대조가 무작위 입력의 약한 사례(pixel F1 일치로 대신)이고 에폭 loss는 유효숫자 3자리 비교. Phase 2: EfficientAD visa Image AUROC가 낮아(약 0.68) 우연히 기준 안일 수 있고 20 epoch(0.742)와의 차이를 분리하지 못했다. Phase 3: 공유 `AUROC` 변경 뒤 기존 export는 재평가만 하고 재학습하지 않았다. Phase 4: DFM `score_type="nll"`은 원본 `fit`이 TypeError라 검증하지 못했고 레퍼런스 raw map AUROC는 구하지 않았다. Phase 5: DFKDE의 KDE float32 포화 발생 여부를 측정하지 않았고, `max_training_points` 초과(`random.sample`) 경로는 실행되지 않았으며, 다중 layer 경로는 단위 테스트만, mvtec `singular_vectors` 부분공간 거리 3.6e-2는 원인을 분해하지 않았다 | 각 모델의 부가 경로는 실행 검증이 없다 | 항목별 소규모 확인 실행 | 예 | 확인 필요 | `docs/results/01~05` 6절 |
| 31 | Phase 8 CFA 부가 경로 | `gamma_c > 1`의 `KMeans` 분기는 기본값에서 실행되지 않아 수치를 검증하지 않았다(D-25). `_CfaLitVisualizer`와 batch 1 평가의 shape 복원은 단위 테스트만 했다 | 비기본 설정의 동작은 보증이 없다 | `gamma_c > 1` 실행 비교 | 아니오 | 알려진 한계로 수용 | `docs/results/08-cfa.md` 6.2절 |
| 32 | Lightning 테스트와 비교 스크립트의 미열람 부분 | Phase 11~17의 `test_*_lightning.py` 본문은 줄 단위로 읽지 않고 실행 결과로만 판정했다. `compare_outputs.py`의 draem, ganomaly, csflow, dsr 분기는 구조만 훑었다 | 테스트가 통과해도 검증 범위가 의도와 다를 수 있다 | 해당 테스트·분기의 정독 검토 | 아니오 | 확인 필요 | `docs/results/11~17` 6.2절 |
| 33 | agy 승인 교차 확인 불가 | agy는 묶음마다 첫 승인 창만 띄우고 이후 명령은 창 없이 실행해 Claude가 명령 일치를 교차 확인하지 못한 구간이 있다. 마감 전 회귀도 같았다. 일부 묶음은 agy가 명령을 `powershell -Command`로 감싸려 해서 승인하지 않고 다시 보냈다. 판정은 출력 로그와 산출물로 했다 | agy의 자기 보고는 틀릴 수 있어 로그에만 의존한다 | 절차 문서화(승인 창 정책) | 아니오 | 알려진 한계로 수용 | `docs/results/11~18` 5.2절, `final-regression.md` |
| 34 | 대용량 산출물 | `outputs/`의 체크포인트·export가 크다(예: CSFlow 체크포인트 약 3.4GB, export 약 1.3GB, CFLOW 약 1.6GB/947MB, GANomaly 2.26GB). 커밋 대상은 아니며 다시 만들 수 있다 | 디스크 사용이 크다 | 정리 기준 결정(보존할 레퍼런스 run) | 아니오 | 알려진 한계로 수용 | `docs/results/13, 15, 16` 6.2절 |

### 2.1 분류별 건수

미진 사항은 모두 34건이며 이 중 마감 전에 2건(항목 10, 14)이 해소되어 남은 것은 32건이다. 분류별 건수는 다음과 같다.

| 분류 | 건수 |
|---|---|
| 사용자 결정 필요 | 8 |
| 구현 필요 | 2 |
| 확인 필요 | 6 |
| 알려진 한계로 수용 | 16 |
| 해소 | 2 |

### 2.2 마감 후 우선순위 3개 제안

다음 세 가지를 제안한다. 순서는 위험과 프로젝트 목적(배포)에 대한 직결성으로 정했다.

1. **라이선스 확정(항목 6):** `LICENSE`와 `NOTICE`는 안 A 가정으로 추가했으나 저작권자가 비어 있다. 외부에 영향이 있는 유일한 항목이라 가장 먼저 정한다. 법률 전문가 확인을 권한다.
2. **용어 일관성(항목 1, 2):** INTENT 내부는 정리했으나 다른 문서의 "순수 PyTorch" 22곳과 specs의 낡은 인용이 남아 있다. 결정은 사용자에게 있고 구현 비용은 문서 수정뿐이다.
3. **배포 요구의 실제 확인(항목 3, 5, 4):** 프로젝트의 목적은 Lightning·anomalib 없는 배포인데 requirements 후보를 깨끗한 환경에 설치해 18개 모델 export 추론을 확인한 적이 없고, 부록 A 대조(NFR-7)를 검사하는 코드도 없다. 확인 결과에 따라 `Predictor` 지연 import(항목 4)의 필요성이 정해진다.

나머지 "확인 필요" 항목 중 비용이 작은 것은 CFA 버퍼 비교(항목 8)다.

### 2.3 이 문서에서 확인하지 못한 것

- Phase 1~18 결과 문서의 "6.2 남은 위험"과 "6. 결론"을 읽어 항목을 모았다. 각 문서의 다른 절(예: 4절 적대적 검증, 5절 실행 기록)에만 있는 한계는 빠졌을 수 있다.
- "순수 PyTorch" 출현 수는 문자열 검색 결과이고 문맥(용어/인용)은 구분하지 않았다.
- 지시서의 "약 30곳"은 specs, AGENTS.md, README 합계 24곳과 다르다. 차이의 원인(docs 초안 포함 여부 등)은 확인하지 않았다.
- 레포 공개 여부는 `gh repo view`로 확인했다(`visibility: PUBLIC`).
- 모든 판정 수치는 각 Phase 결과 문서의 값이며 이 문서 작성 중 다시 계산하지 않았다.
