# Phase 0. 데이터셋

MVTec AD와 VisA 데이터셋을 visionlab에서 새로 구현하고 anomalib 2.0.0 데이터와 대조한다. 범위와 완료 조건은 `specs/PLAN.md`의 Phase 0, 절차는 `docs/workflow/model-port.md`를 따른다. 이 파일은 Phase 0을 시작할 때 만든 뼈대이며, 아래 2~6절은 구현과 대조가 끝나는 대로 채운다.

## 상태

| 단계 | 내용 | 상태 |
|---|---|---|
| 0-A | 순수 Dataset (`batch.py`, `split.py`, `dataset.py` x2, `test_data.py`) | 시작 전 |
| 0-B | LitData x2, `00_compare_data.py`, anomalib 대조 | 시작 전 |

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

레퍼런스 분할 설정은 `test_split_mode="from_dir"`, `val_split_mode="from_test"`, `val_split_ratio=0.4`, `seed=42`, batch 8/8이다. 위 파일 수는 2026-10-25에 codex가 목록으로 확인한 값이며, 0-B에서 `00_compare_data.py`가 다시 계산해 이 표를 갱신한다.

## 2. anomalib 대조 결과 (데이터 검증 ID)

아직 없다. 0-B가 끝나면 category별로 다음을 적는다.

| data / category | 데이터 검증 ID (manifest 해시) | 파일 목록 | 라벨 | mask | 분할 | 최종 입력 | 시그니처 | 로그 |
|---|---|---|---|---|---|---|---|---|

## 3. 구현 메모

아직 없다. 0-A, 0-B가 끝나면 원본(anomalib) 동작과 구현의 대응, 의도적으로 다르게 한 곳을 적는다.

## 4. 적대적 검증

아직 없다. 단계별 회차(최대 3회)마다 지적, 심각도, 처리, 남은 위험을 적는다.

## 5. 판정

아직 없다.

## 6. 결론

아직 없다. 사용자가 Phase 종료를 결정하면 미충족 항목과 남은 위험을 적고 커밋·푸시한다.

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
