# Phase 0. 데이터셋

MVTec AD와 VisA 데이터셋을 visionlab에서 새로 구현하고 anomalib 2.0.0 데이터와 대조한다. 범위와 완료 조건은 `specs/PLAN.md`의 Phase 0, 절차는 `docs/workflow/model-port.md`를 따른다. 이 파일은 Phase 0을 시작할 때 만든 뼈대이며, 아래 2~6절은 구현과 대조가 끝나는 대로 채운다.

## 상태

| 단계 | 내용 | 상태 |
|---|---|---|
| 0-A | 순수 Dataset (`batch.py`, `split.py`, `dataset.py` x2, `test_data.py`) | 구현, 검증, 실행 완료 (수정 요청 0회) |
| 0-B | LitData x2, `compare_data.py`(실행 당시 이름 `00_compare_data.py`), anomalib 대조 | 구현, 검증, 실행 완료 (수정 요청 1회) |

## 1. 데이터셋 노트

| 항목 | MVTec AD | VisA |
|---|---|---|
| 평가 category | `bottle` | `candle` |
| 로컬 위치 | `D:\datasets\mvtec\bottle` | 원본 `D:\datasets\visa`, anomalib 작업 복사본 `D:\datasets\visa_anomalib\visa_pytorch` (D-17) |
| anomalib 대응 클래스 | `anomalib.data.MVTecAD` | `anomalib.data.Visa` |
| 레퍼런스에서 쓰는 클래스 | `LocalMVTecAD` (`evaluation/anomalib/local_data.py`) | `LocalVisa` (같은 파일, `split_root`만 복사본으로 바꿈) |
| 폴더 구조 | `train/good`, `test/<good 또는 결함>`, `ground_truth/<결함>` | `visa_pytorch/<category>/{train,test}/{good,bad}`, `ground_truth/bad` |
| 파일 수 (목록 확인, 이미지 내용 미검증) | train/good 209, test/good 20, 이상 63, 이상 mask 63 | train/good 900, test/good 100, test/bad 100, mask 100 |
| 읽는 곳 | `root/<category>` | `root/visa_pytorch/<category>` |

레퍼런스 분할 설정은 `test_split_mode="from_dir"`, `val_split_mode="from_test"`, `val_split_ratio=0.4`, `seed=42`, batch 8/8이다. 위 파일 수는 2026-10-25에 codex가 목록으로 확인한 값이며, 0-B에서 `00_compare_data.py`가 다시 센 값(두 폴더 모두)과 일치했다. 이상 test 이미지는 MVTec `broken_large` 20, `broken_small` 22, `contamination` 21이고 VisA는 `bad` 100이다.

## 2. anomalib 대조 결과 (데이터 검증 ID)

2026-10-25에 agy가 `00_compare_data.py`를 category별로 실행했다(실행으로 검증함, 종료 코드 0). 이 스크립트는 같은 날 사용자 결정으로 `compare_data.py`로 이름이 바뀌었다(D-15의 이름 규칙 통일). 아래 명령과 지시문은 실행 당시의 이름을 그대로 남기고, 이름 변경 뒤 다시 실행한 결과는 이 절 끝에 적는다. Claude는 manifest 파일의 sha256과 로그의 PASS, FAIL 줄을 직접 다시 확인했다. 데이터 검증 ID는 manifest 파일의 sha256이다.

다음 표는 category별 대조 결과다. 판정 항목 187개가 모두 PASS이고 FAIL은 0개다.

| data / category | 데이터 검증 ID (manifest 해시) | 파일 목록 | 라벨 | mask | 분할 | 최종 입력 | 시그니처 | 로그 |
|---|---|---|---|---|---|---|---|---|
| mvtec / bottle | `f4d5782042cfb5c5111474280d654849304820814a92b71207fce1c390c10ea7` | 일치 | 일치 | 일치 | 일치 | 일치 | 일치 | `outputs/logs/00-0B-mvtec-261025-1.log` |
| visa / candle | `b3de65b8daee9a9a8787406550f6c8e804822c0611fe82f43e69023c188dcb48` | 일치 | 일치 | 일치 | 일치 | 일치 | 일치 | `outputs/logs/00-0B-visa-261025-1.log` |

대조 범위와 한계는 다음과 같다.

- manifest 위치는 `outputs/data/mvtec_bottle_manifest.json`, `outputs/data/visa_candle_manifest.json`이다. 두 로그는 UTF-16으로 저장돼 있다(PowerShell `Tee-Object`).
- 파일 목록, 라벨, mask 경로는 train, val, test 전체를 순서까지 비교했다. 경로는 `os.path.normpath` 후에 비교했다(anomalib은 `/`와 `\`가 섞인 경로를 쓴다).
- 분할 크기는 MVTec이 train 209, val 33, test 50이고 VisA가 train 900, val 80, test 120이다. 양쪽 모두 anomalib과 같다.
- 최종 입력 텐서는 subset마다 앞쪽 8개만 비교했다(image는 float32, `allclose` rtol 1e-5, atol 1e-6, mask는 값 `equal`). test와 val의 앞쪽 8개는 모두 이상 이미지이므로, 정상 이미지의 빈 mask 경로는 train 앞쪽 8개로만 확인했다.
- 알려진 차이로, anomalib mask는 `torch.bool`이고 visionlab mask는 FR-1에 따라 `torch.int64`다. 이 차이는 판정에서 제외하고 manifest에 기록했다(양쪽을 int64로 바꾼 값이 모두 일치했다).
- 생성자 시그니처는 `MVTecAD`, `Visa`와 `inspect.signature`의 이름, 순서, 종류, 기본값 repr이 일치했다. 증강 인자는 둘 다 받고 `None`만 허용한다. annotation은 프로젝트 규칙(type hint 금지)에 따라 비교에서 제외했다.
- VisA는 anomalib이 `D:\datasets\visa_anomalib\visa_pytorch`에 만든 작업 복사본(D-17)을 두 쪽이 같이 읽는다. visionlab 쪽 root는 `D:\datasets\visa_anomalib`이다.

### 실행 명령

agy가 PowerShell에서 실행한 명령이다(2026-10-25, 작업 폴더 `D:\projects\visionlab`, 종료 코드 모두 0). 로그는 `outputs/logs/`에 UTF-16으로 저장됐다. 명령마다 승인 창이 떴고 Claude가 한 번씩 승인했다.

0-A 순수 Dataset 테스트(`00-0A-test-261025-1.log`, 리팩토링 뒤 회귀는 `-2.log`)는 다음과 같다.

```powershell
New-Item -ItemType Directory -Force D:\projects\visionlab\outputs\logs | Out-Null
cd D:\projects\visionlab
& 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe' 'D:\projects\visionlab\tests\00_dataset\test_data.py' 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\00-0A-test-261025-1.log
$LASTEXITCODE
```

0-B anomalib 대조(`00-0B-mvtec-261025-1.log`, `00-0B-visa-261025-1.log`, 회귀는 `-2.log`)는 다음과 같다.

```powershell
cd D:\projects\visionlab
& 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe' 'D:\projects\visionlab\evaluation\anomalib\00_compare_data.py' --data mvtec 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\00-0B-mvtec-261025-1.log
$LASTEXITCODE
& 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe' 'D:\projects\visionlab\evaluation\anomalib\00_compare_data.py' --data visa 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\00-0B-visa-261025-1.log
$LASTEXITCODE
```

`BaseDataset` 리팩토링 뒤 회귀 실행은 위 세 명령을 로그 이름만 `-2.log`로 바꿔 한 번에 실행했다(테스트 1개, 대조 2개 순서).

스크립트 이름을 `00_compare_data.py`에서 `compare_data.py`로 바꾼 뒤(사용자 결정, 2026-10-25, 내용은 첫 줄 주석의 경로 한 줄만 다름을 커밋된 원본과 `diff`로 확인) 같은 대조를 다시 실행했다(agy, 승인 창마다 Claude가 명령을 확인하고 한 번씩 승인). 결과는 두 데이터셋 모두 PASS 186줄, FAIL 0줄이고 manifest의 sha256이 위에 기록한 데이터 검증 ID와 같다(실행으로 검증함).

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py evaluation/anomalib/compare_data.py --data mvtec 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\00-0B-mvtec-261025-3.log
& $py evaluation/anomalib/compare_data.py --data visa 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\00-0B-visa-261025-3.log
```

## 3. 구현 메모

다음 표는 0-A, 0-B 파일과 anomalib 원본의 대응이다. 데이터 코드는 복사하지 않고 새로 썼다(D-18).

| visionlab 파일 | 대응하는 anomalib 원본 |
|---|---|
| `common/batch.py` | `ImageBatch` 필드 이름, `InferenceBatch` 출력 필드 |
| `data/common/split.py` | `utils/split.py`의 Enum, `random_split`, `split_by_label`, `datamodules/base/image.py`의 분할 순서 |
| `data/mvtec/dataset.py` | `make_mvtec_ad_dataset`, `MVTecADDataset`, `read_image`, `read_mask` |
| `data/visa/dataset.py` | `VisaDataset`(`MVTecDataset`을 상속, 확장자와 `visa_pytorch` 경로만 다름) |
| `data/mvtec/lightning_data.py`, `data/visa/lightning_data.py` | `MVTecAD`, `Visa` 데이터모듈의 생성자, `setup`, dataloader |
| `evaluation/anomalib/compare_data.py` | 대조 스크립트(새로 작성, 처음 이름은 `00_compare_data.py`) |

의도적으로 다르게 한 곳은 다음과 같다.

- Dataset은 `(image, target)`을 반환하고, `target`은 label, mask, sample_id, image_path, mask_path를 가진다. 256x256 resize(image bilinear antialias, mask nearest)는 Dataset이 하고 정규화는 하지 않는다(D-6).
- pandas 대신 dict 목록을 쓴다. 부분집합과 병합 결과는 anomalib과 같이 이미지 경로순으로 정렬한다.
- `random_split`은 label별 `Generator` 재생성, 반올림 방식, `seed=0`일 때 전역 RNG 사용까지 원본 동작을 유지한다. 원본에서 0으로 나누게 되는 1개 샘플 분할은 `ValueError`로 처리한다.
- `FROM_TRAIN`, `SYNTHETIC` 등 Phase 0 범위 밖 모드는 Enum 이름은 두되 실행하면 `ValueError`를 낸다.
- `setup()`은 한 번 만든 분할을 이후 호출에서도 유지한다. dataloader의 collate는 기본 collate다(anomalib은 `ImageBatch.collate`).
- 실제 파일 수 대조와 레퍼런스 데이터 로딩은 `evaluation/anomalib/`에서만 anomalib을 import한다. `src/visionlab`은 anomalib을 import하지 않고, 순수 Dataset은 Lightning도 import하지 않는다.

### 리팩토링: BaseDataset 분리

사용자 요청으로 `VisADataset`이 `MVTecDataset`을 상속하던 구조를 바꿨다(codex 구현, Claude 검증). 공통 구현(`make_samples`, `__init__`, `__len__`, `__getitem__`, `subsample`, `__add__`)을 `data/common/base_dataset.py`의 `BaseDataset`으로 옮기고, `MVTecDataset`과 `VisADataset`은 각각 `BaseDataset`을 상속해 `extensions`와 생성자 기본값, root 해석만 가진다. `make_mvtec_samples`는 MVTec 전용이 아니므로 `make_samples`로 이름을 바꿨다. 옮긴 코드는 원본과 줄 단위로 같음을 Claude가 확인했고, `tests/00_dataset/test_data.py`에 상속 관계 검사 하나가 추가됐다.

회귀 실행(2026-10-25, 실행으로 검증함)은 테스트 10개 통과(`outputs/logs/00-0A-test-261025-2.log`), 대조 두 category 모두 FAIL 0이다(`00-0B-mvtec-261025-2.log`, `00-0B-visa-261025-2.log`). 두 manifest를 다시 만들었고 sha256이 2절의 데이터 검증 ID와 같다.

## 4. 적대적 검증

Claude가 anomalib 원본 소스와 이식 코드를 줄 단위로 대조했다. codex의 설명은 근거로 쓰지 않았다.

다음 표는 단계별 회차와 지적이다.

| 단계 | 회차 | 지적 | 심각도 | 처리 |
|---|---|---|---|---|
| 0-A | 검증 1 | `random_split`의 label별 Generator, 반올림, 정렬, 병합 순서, `from_test` 결과 | 이상 없음 | 해당 없음 |
| 0-A | 검증 1 | 경로 구분자가 anomalib과 다르다(`os.path.join`의 `\`, anomalib은 `/` 혼용). 정렬 순서가 달라질 수 있는 경우는 한 label이 다른 label의 접두어일 때뿐이고 `bottle`, `candle`에는 없다 | Minor | 수정하지 않음. 0-B 스크립트가 `normpath`로 비교한다 |
| 0-A | 검증 1 | `os.path.abspath`를 root에 적용하고, 샘플 경로의 깊이를 정확히 3으로 제한한다(anomalib은 `parts[-3:]`) | Minor | 수정하지 않음. 지원하는 폴더 구조에서는 결과가 같다 |
| 0-A | 검증 1 | `src/visionlab`에 `__init__.py`가 없다(namespace package) | Minor | 수정하지 않음. 패키징은 Phase 1에서 `pyproject`와 함께 정한다 |
| 0-B | 검증 1 | `compare_tensor`가 mask dtype(bool vs int64)을 합격 조건에 넣어 항상 FAIL이 된다 | Major | 수정 요청 1/3: mask는 shape과 값 일치로 판정하고 dtype은 known_difference로 기록 |
| 0-B | 검증 1 | 이미 bool로 바뀐 mask에 transform을 다시 적용했다(anomalib 실제 경로와 다름) | Minor | 수정 요청 1/3에 포함: reference `augmentations`에 transform을 지정해 `reference[index]`를 그대로 사용 |
| 0-B | 검증 2 | 수정 확인(코드 대조) | 이상 없음 | 해당 없음 |

수정 요청 횟수는 0-A가 0회, 0-B가 1회(최대 3회)이다. 남은 위험은 다음과 같다.

- 최종 입력 텐서 비교가 subset의 앞쪽 8개뿐이다. 전체 이미지의 픽셀 일치는 확인하지 않았다.
- test, val의 정상 이미지 입력은 직접 비교하지 않았다(train의 정상 이미지와 같은 코드 경로).
- 이 대조는 anomalib 2.0.0과 현재 로컬 데이터에 대한 결과다. 다른 category에서는 다시 실행해야 한다.

## 5. 판정

다음 표는 PLAN의 Phase 0 완료 조건별 판정이다.

| 요구 | 근거 | 판정 |
|---|---|---|
| FR-1 (항목 계약) | `tests/00_dataset/test_data.py` 9개 테스트 통과(`outputs/logs/00-0A-test-261025-1.log`, 실행으로 검증함). 실제 bottle, candle의 앞쪽 8개 샘플도 anomalib과 일치 | 충족 |
| FR-3 (분할 재현) | 같은 seed 42에서 새 인스턴스도 같은 목록, valid와 test가 겹치지 않음, 미구현 모드 예외(테스트). 실제 데이터의 `from_test` 분할이 anomalib과 같은 파일 목록(대조) | 충족 |
| FR-2 (시그니처) | `inspect.signature` 일치(대조 `*.constructor` PASS) | 충족 |
| FR-17 (anomalib 대조) | 파일 목록, 라벨, mask 경로, 분할, 최종 입력이 두 category에서 일치. 데이터 검증 ID는 2절 | 충족(앞쪽 8개 한계 있음) |
| NFR-1, NFR-2 | 순수 Dataset은 Lightning과 anomalib을 import하지 않고 테스트에서 import를 차단한 채 통과 | 충족 |
| NFR-3 | 다운로드와 복사본 생성 없음, 폴더가 없으면 `FileNotFoundError`(테스트) | 충족 |

사용자 결정(2026-10-25, D-21, NFR-6)으로 `tests/00_dataset/test_data.py`를 `unittest`에서 pytest로 바꿨다. 위 표의 9개 통과 기록은 변환 전이고, BaseDataset 리팩토링 검사 1개가 합쳐진 변환 후 10개 테스트가 통과했다. 명령은 다음과 같다(agy, 2026-10-25, 종료 코드는 로그로 확인, 실행으로 검증함).

```powershell
cd D:\projects\visionlab
& 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe' -m pytest tests\00_dataset 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\00-pytest-261025-1.log
```

결과는 `10 passed in 5.23s`(`outputs/logs/00-pytest-261025-1.log`)다.

지시 과정에서 남길 점 두 가지를 적는다. agy는 0-A 실행 보고에서 "승인 프롬프트 없이 실행됐다"고 했으나 실제로는 승인 창이 떴고 Claude가 명령마다 한 번씩만 승인했다(0-A 1회, 0-B 2회, 영구 허용 없음). agy의 자기 보고는 근거로 쓰지 않았다. 0-B 지시문의 금지 항목 첫 줄에 "anomalib·Lightning 외"라는 어색한 문구가 들어갔으나, 같은 줄의 규칙(`src/`는 anomalib import 금지)과 결과 파일로 의도대로 구현됐다.

## 6. 결론

판정은 6개 요구 모두 충족이다. 사용자가 2026-10-25에 Phase 0 종료를 결정했다. 미충족 항목은 없고, 남은 위험은 4절에 적은 세 가지다(최종 입력 비교가 subset 앞쪽 8개뿐, test·val 정상 이미지 입력 미비교, anomalib 2.0.0과 현재 로컬 데이터 한정).

## 부록. codex 구현 지시문

새 세션이 그대로 codex에 보낸다. 지시 양식은 `docs/workflow/model-port.md`의 "이식 구현 (codex)"를 따른다.

### 0-A. 순수 Dataset

```text
[Phase 0-A 구현 지시: 순수 Dataset]
목표: MVTec AD와 VisA의 순수 PyTorch Dataset과 분할 로직을 visionlab에서 새로 구현한다. Lightning과 anomalib을 import하지 않는다.
읽을 것: AGENTS.md(에이전트 역할, 코드 실행, Python 코드 규칙), specs/INTENT.md, specs/SPEC.md(FR-1, FR-3, NFR-1~3), specs/PLAN.md(Phase 0), specs/DECISIONS.md(D-4, D-5, D-6, D-10, D-13, D-15, D-17), docs/workflow/model-port.md, docs/results/00-dataset.md(1절),
  참고용 원본(읽기만): C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\Lib\site-packages\anomalib\data\ 의 datamodules\image\mvtecad.py, visa.py, datamodules\base\image.py, datasets\image\mvtecad.py, visa.py, utils\split.py, utils\label.py, utils\path.py
쓸 수 있는 곳: src/visionlab/anomaly_detection/ (common/, data/), tests/00_dataset/
금지: anomalib·Lightning import, 코드 실행(문법 확인 포함, 실행은 agy가 한다), 원본 anomalib 설치 파일 수정, specs/·docs/ 수정, 다운로드 구현, 요청하지 않은 파일·추상화
만들 것:
  - common/batch.py: Batch, InferenceBatch (순수 dataclass, 필드는 anomalib ImageBatch와 이름을 맞춘다: image, gt_label, gt_mask, image_path, mask_path, sample_id 등 필요한 최소)
  - data/common/split.py: TestSplitMode(none, from_dir, synthetic), ValSplitMode(none, same_as_test, from_test, synthetic)의 값과 이름을 anomalib과 같게, 분할 함수. 구현하지 않는 모드는 명시적 오류. from_test 분할은 같은 seed에서 anomalib과 같은 파일 목록이어야 하므로 원본 알고리즘(난수 사용 방식 포함)을 따른다
  - data/mvtec/dataset.py MVTecDataset, data/visa/dataset.py VisADataset: (image, target) 또는 Batch 반환(FR-1: image float32 [3,H,W] [0,1], label 0/1, mask int64 [H,W], sample_id, image_path). 폴더 구조와 샘플 정렬 순서는 anomalib과 같게. Transform은 RGB 변환, 이미지와 mask의 resize, float 변환만(D-6). resize 크기와 보간 방식은 anomalib 기본 PreProcessor와 같게(256x256, 이미지 bilinear+antialias, mask nearest). VisA는 root/visa_pytorch/<category>를 읽고 없으면 FileNotFoundError
  - tests/00_dataset/test_data.py: 임시 폴더에 작은 가짜 MVTec/VisA 폴더 구조를 만들어 FR-1(shape, dtype, 범위, 필드), FR-3(from_test에서 valid와 test가 겹치지 않음, 같은 seed에서 새 인스턴스도 같은 목록, 미구현 모드 예외)을 검사한다. anomalib·Lightning import 금지. 스크립트로 직접 실행 가능하게(WinPython 전체 경로)
증강 인자는 Phase 0에서 받지 않거나 None만 허용한다(값이 있으면 명시적 오류).
돌려줄 것: 변경 파일 목록, 원본 파일·함수와의 대응표, 원본과 다르게 구현한 곳과 이유, 확신하지 못한 부분과 agy가 실행해 확인해야 할 항목
```

### 0-B. LitData와 anomalib 대조

```text
[Phase 0-B 구현 지시: LitData와 anomalib 대조]
목표: 0-A의 Dataset을 감싸는 native LightningDataModule 두 개와, anomalib 데이터와의 대조 스크립트를 만든다.
읽을 것: 0-A와 같다 + 0-A에서 만든 src/visionlab/anomaly_detection/ 파일, evaluation/anomalib/local_data.py, evaluation/anomalib/anomalib_run.py, evaluation/anomalib/data/*.yaml
쓸 수 있는 곳: src/visionlab/anomaly_detection/data/, evaluation/anomalib/
금지: 0-A와 같다. 단 evaluation/anomalib/ 안의 코드는 anomalib을 import해도 된다. src/는 import 금지
만들 것:
  - data/mvtec/lightning_data.py MVTecLitData, data/visa/lightning_data.py VisALitData: native LightningDataModule. 생성자 인자 이름·순서·기본값이 anomalib 2.0.0 MVTecAD, Visa와 같다(FR-2). prepare_data는 로컬 폴더 존재만 확인하고 다운로드하지 않는다(없으면 FileNotFoundError). 증강 인자가 None이 아니면 명시적 오류. name, category 속성은 anomalib과 같은 의미로 둔다
  - evaluation/anomalib/00_compare_data.py: 첫머리에서 <root>/src를 sys.path에 추가한 뒤 visionlab을 import한다. 인자 --data mvtec|visa, --category. 같은 설정(from_test, 0.4, seed 42, batch 8/8, num_workers 0)으로 anomalib LocalMVTecAD/LocalVisa(local_data.py)와 visionlab LitData를 만들어 비교한다: (1) 생성자 시그니처(inspect.signature, 인자 이름·순서·기본값), (2) train/val/test 파일 목록과 순서, 라벨, mask 경로, (3) 각 subset의 앞쪽 N개(기본 8) 샘플의 최종 입력 텐서 allclose와 mask 일치, (4) 결과를 outputs/data/<data>_<category>_manifest.json로 저장하고 그 sha256을 데이터 검증 ID로 출력한다. 불일치는 항목별로 출력하고 종료 코드를 0이 아니게 한다. src/visionlab에서는 anomalib을 import하지 않는다
돌려줄 것: 0-A와 같다 + 비교 항목과 manifest 필드 설명
```
