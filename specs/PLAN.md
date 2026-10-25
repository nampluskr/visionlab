# PLAN

Phase 0에서 모든 모델이 공유하는 데이터셋을 구현하고 anomalib과 대조한다. Phase 1부터는 Phase마다 모델 하나를 다룬다. 각 모델 Phase는 그 모델의 anomalib 레퍼런스 수치를 먼저 확보하고, 코드를 native Lightning으로 이식한 뒤, 이식 결과를 레퍼런스와 비교한다(D-16). 파일 위치는 D-15를 따르고, 각 Phase는 자기 범위의 파일만 만든다.

- **평가 category**: MVTec `bottle`, VisA `candle`.
- **산출 위치**: 실행 산출물은 `outputs/`, 결과는 Phase마다 `docs/results/NN-<kebab-case>.md`(규칙과 산출물 위치는 `docs/results/README.md`).
- **레퍼런스 실행**: `evaluation/anomalib/anomalib_run.py`(`AnomalibCLI`)와 공유 yaml(`default.yaml`, `data/*.yaml`, `models/NN_<name>.yaml`).

각 모델 Phase 안에서 INTENT의 검증 세 단계를 다음 순서로 수행한다.

| 순서 | 검증 단계 | 내용 |
| --- | --- | --- |
| 1 | 레퍼런스 확보 | 그 모델을 anomalib 그대로 로컬 자산으로 실행한 수치 |
| 2 | native Lightning 이식 | 이식한 모델의 학습·평가 수치를 1과 비교 |
| 3 | 순수 PyTorch 추론 | 2의 가중치를 순수 PyTorch로 불러온 추론 수치를 1과 비교 |

에이전트 역할과 한 모델 Phase의 진행 순서는 다음과 같다. Phase 0은 아래 표의 순서를 따르되 레퍼런스 실행 대신 anomalib 데이터 대조를 한다. 상세 절차와 지시 양식은 `docs/workflow/model-port.md`를 따른다.

| 순서 | 작업 | 담당 |
| --- | --- | --- |
| 1 | 레퍼런스 실행 명령 전달, 실행 | Claude가 명령을 보내고 agy가 터미널에서 실행 |
| 2 | 레퍼런스 진단·기록 | Claude |
| 3 | 이식 구현 | Claude가 지시하고 codex가 구현 |
| 4 | 이식 결과 적대적 검증, 수정 요청 | Claude. 수정 반복은 최대 3회 |
| 5 | 이식 결과 실행(native Lightning 학습·평가, 순수 PyTorch 추론) | Claude가 명령을 보내고 agy가 실행 |
| 6 | 수치 판정·기록 | Claude. 기준 미달이면 원인을 진단해 3으로 돌아간다(최대 3회) |
| 7 | 보고, Phase 종료 결정 | Claude가 보고하고 사용자가 결정 |
| 8 | 커밋·푸시 | Claude. 종료가 결정되면 그 Phase의 변경을 커밋하고 `origin`에 푸시 |

## Phase 번호와 파일 이름

Phase 번호는 두 자리이고 구현 순서다. 모델별 파일이 쌓이는 곳(`docs/results/`, `tests/`, `evaluation/anomalib/models/`)은 이 번호를 붙여 `ls`만으로 순서와 진행 상태를 보이게 한다. 번호와 모델의 대응은 아래 표가 유일한 원본이며, 순서를 바꾸면 번호가 붙은 파일 이름을 함께 바꾼다. `src/`의 모델 패키지와 `outputs/`에는 번호를 붙이지 않는다(D-15).

| NN | 모델 폴더 이름 | NN | 모델 폴더 이름 |
| --- | --- | --- | --- |
| 00 | dataset | 10 | supersimplenet |
| 01 | stfpm | 11 | draem |
| 02 | efficient_ad | 12 | fastflow |
| 03 | padim | 13 | cflow |
| 04 | dfm | 14 | uflow |
| 05 | dfkde | 15 | csflow |
| 06 | patchcore | 16 | ganomaly |
| 07 | reverse_distillation | 17 | dsr |
| 08 | cfa | 18 | winclip |
| 09 | fre | | |

| 위치 | 이름 |
| --- | --- |
| `docs/results/` | `NN-<kebab-case>.md` (예: `02-efficient-ad.md`) |
| `evaluation/anomalib/models/` | `NN_<모델 폴더 이름>.yaml` (예: `02_efficient_ad.yaml`) |
| `tests/` | `NN_<모델 폴더 이름>/test_*.py`. 여러 Phase에 걸친 테스트는 `tests/common/` |

## Phase 0. 데이터셋

- **목적**: 모든 모델이 공유하는 MVTec·VisA 데이터셋을 visionlab에서 새로 구현하고, 모델 비교 전에 anomalib 데이터와 같은지 대조한다(D-13). anomalib 데이터 코드는 복사하지 않는다(D-18의 복사 대상은 models뿐이다).
- **만드는 것**: `data/common/split.py`, `data/mvtec/`·`data/visa/`의 `dataset.py`(`MVTecDataset`, `VisADataset`)와 `lightning_data.py`(`MVTecLitData`, `VisALitData`), `common/batch.py`(Batch, InferenceBatch), 대조 스크립트 `evaluation/anomalib/00_compare_data.py`, `tests/00_dataset/test_data.py`.
- **대응 요구 ID**: FR-1, FR-2, FR-3, FR-17, NFR-1, NFR-3
- **진행 순서**: Claude가 지시하고 codex가 구현한다. Claude가 적대적으로 검증해 수정을 요청한다(최대 3회). agy가 대조 스크립트와 테스트를 실행하고, Claude가 판정·기록한다.
- **완료 조건**:
  - FR-1·FR-2·FR-3의 판정 방법이 통과한다.
  - mvtec `bottle`, visa `candle`에서 파일 목록, 라벨, mask, `from_test` 분할(seed 42), 최종 입력 텐서가 anomalib(`LocalMVTecAD`, `LocalVisa`)과 일치한다(FR-17). 결과를 `docs/results/00-dataset.md`에 데이터 검증 ID(manifest 해시)와 함께 기록한다.
  - `src/visionlab`이 anomalib을 import하지 않는다(NFR-1). Dataset은 Lightning 없이 동작한다.

## Phase 공통 양식

Phase 1~18의 각 모델 Phase에 적용한다. 각 Phase는 모델 하나를 다룬다. 앞 Phase에서 만든 공통 부품(데이터, 특징 추출기, 처리 구성요소, 추론기)을 재사용하고, 그 모델에 필요한 것만 추가한다.

- **대응 요구 ID**: FR-4, FR-5, FR-6, FR-18, FR-19, FR-20, FR-21, NFR-3, NFR-4, NFR-5와 모델별 요구.
- **완료 조건**:
  - 레퍼런스: `anomalib_run.py train`과 `models/NN_<name>.yaml`로 category별 Image·Pixel AUROC와 실제 적용된 학습 예산을 그 Phase의 결과 파일에 기록한다(FR-21). 같은 설정으로 2회 실행한 run 간 차이도 기록한다.
  - 레퍼런스 학습 예산은 공유 yaml을 따른다: 기본 `max_epochs=10`, `val_split_mode="from_test"`, `val_split_ratio=0.4`, `seed=42`(데이터 분할·`seed_everything`), batch 8/8(모델 yaml의 override 우선), `num_workers=0`, `devices=1`, `deterministic="warn"`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. 모델이 고정하는 `trainer_arguments`가 우선한다. 이식 결과도 같은 예산을 쓴다.
  - 순수 모델의 평가 출력 shape(FR-4)과 생성자 인자(FR-6)가 판정 방법을 통과한다.
  - 같은 state·입력에서 순수 모델의 raw 출력이 anomalib 원본과 일치한다(FR-5). 허용 오차는 Phase 시작 전에 확정한다.
  - native Lightning 학습·평가의 category별 Image·Pixel AUROC가 그 Phase의 레퍼런스와 각각 0.01 이내다(FR-18).
  - 학습 가중치를 순수 PyTorch 모델로 불러와 추론한 category별 Image·Pixel AUROC가 그 Phase의 레퍼런스와 각각 0.01 이내다(FR-19, FR-20).
  - image metric만 내는 모델(DFKDE, GANomaly)은 Image AUROC만 기록·판정한다.
  - 학습에 쓰던 외부 라이브러리(FrEIA, open_clip 등)가 추론 경로에 남지 않는다(NFR-2). 독립 구현 비용은 해당 Phase 범위에 포함한다.
  - NFR-3·NFR-4의 판정 방법이 통과한다.

## Phase 1. STFPM

- **목적**: 첫 모델로 이식 기반을 만든다. STFPM 레퍼런스를 확보한 뒤 아래 범위를 이식하고, 이식이 끝난 결과를 레퍼런스와 비교한다(D-16, D-18). Phase 0의 데이터셋을 사용하며, 이 Phase에서 만든 공통 부품은 이후 Phase가 재사용한다.
  - 순수 모델: 원본 복사본의 `STFPMModel`·`STFPMLoss`·`AnomalyMapGenerator`와 필요한 `models/components/`(FR-4, FR-16, NFR-5).
  - LightningModule·Trainer: `StfpmLitModel`, 새로 구현한 `common/` 대체 부품, native Trainer(FR-6~FR-12).
  - CLI·설정: LightningCLI와 YAML(FR-13).
  - 순수 PyTorch 추론: 가중치 저장·로딩과 `inference/predictor.py`(FR-14, FR-19, FR-20, NFR-1, NFR-2).
- **완료 조건**: 공통 양식과 위 요구 ID의 판정이 모두 통과한다. FR-5(raw 출력 비교)는 수치가 어긋날 때의 진단에 쓴다.

## Phase 2. EfficientAD

- **목적**: teacher 통계, Imagenette penalty(`D:\datasets\imagenette2\train`), valid quantile 보정, 학습 batch 1을 이식한다. teacher는 로컬 가중치 파일을 사용한다.
- **추가 요구 ID**: FR-15
- **완료 조건**: 공통 양식과 FR-15의 판정 방법이 통과한다.

## Phase 3~18. 나머지 모델

각 Phase는 공통 양식대로 그 모델의 레퍼런스 확보, 이식, 비교 순으로 진행한다. 순서는 공통 부품 재사용과 검증 변수를 점진적으로 늘리는 기준으로 정했다. 통계형, 역전파형, 합성 이상형, flow 계열, 특수 학습 모델 순이다. 
| Phase | 모델 | 학습 방식 | 이 위치인 이유와 추가 범위 |
| --- | --- | --- | --- |
| 3 | PaDiM | 1 epoch 통계 | 특징 추출기 재사용. 통계 buffer 저장·복원 경로를 처음 세운다 |
| 4 | DFM | 1 epoch 통계 | PCA 도입 |
| 5 | DFKDE | 1 epoch 통계 | DFM의 PCA 재사용. image metric만 판정 |
| 6 | PatchCore | 1 epoch memory bank | coreset 샘플링, kNN |
| 7 | Reverse Distillation | 역전파 | STFPM의 고정 encoder 확장, decoder 추가 |
| 8 | CFA | centroid 초기화 + 역전파 | torchvision backbone, memory bank 경험 활용 |
| 9 | FRE | 역전파 | 모델이 220 epoch을 고정해 검증 시간이 길다 |
| 10 | SuperSimpleNet | 합성 특징 이상 | 공통 Perlin 생성 함수 도입 |
| 11 | DRAEM | 합성 이미지 이상 | Perlin 재사용, DTD augmentation, resize 전용 전처리 |
| 12 | FastFlow | flow | 공통 flow 블록(AllInOneBlock)과 Jacobian 계산 독립 구현 |
| 13 | CFLOW | 조건부 flow | flow 블록 재사용, 위치 encoding, 수동 최적화 |
| 14 | UFlow | 다중 스케일 flow | flow 블록과 graph 연결 확장 |
| 15 | CSFlow | cross-scale flow | 별도 coupling 구현, EfficientNet-B5 로컬 가중치 |
| 16 | GANomaly | GAN, 수동 최적화 | 적대적 학습 변동성. image metric만 판정 |
| 17 | DSR | VQ 2단계 학습 | 사전학습 VQ 가중치, 단계 전환, 두 optimizer, resize 전용 전처리 |
| 18 | WinCLIP | zero-shot | open_clip 없이 CLIP 독립 구현. 학습 없이 validate 후 test |

## 검증 방식

완료 기준은 수치다. anomalib의 성능이 그대로 재현되는지를 metric 수치로 평가하며, visionlab의 결과가 anomalib과 동일 수준이어야 한다. 이식 코드는 구현자(codex)의 반대 벤더인 Claude가 적대적으로 검증하고 수정을 요청한다. 적대적 검증은 수치 판정 전에 원본과 다른 동작을 찾아 고치는 단계이며, 완료 여부는 아래 수치 기준으로만 판정한다.

수정 반복은 다음 두 경우 각각 최대 3회다. 한도를 넘기면 남은 문제와 위험을 사용자에게 보고하고 판단을 요청한다.

- 적대적 검증에서 찾은 문제의 수정 요청
- 수치 기준 미달에 따른 원인 진단과 수정 요청

| 필드 | 내용 |
|---|---|
| 비교 대상 | 각 Phase에서 그 모델로 확보한 anomalib 2.0.0 레퍼런스 수치 |
| 비교하는 결과 | native Lightning 학습·평가 수치(검증 2단계), 순수 PyTorch 추론 수치(검증 3단계) |
| 비교 조건 | 레퍼런스와 같은 데이터·자산·학습 예산 |
| 비교 metric | Image AUROC, Pixel AUROC (image metric만 내는 모델은 Image AUROC) |
| 동일 수준의 기준 | category별 \|visionlab - anomalib\| ≤ 0.01 (metric 각각) |

FR-18은 각 Phase의 native Lightning 판정으로, FR-20은 각 Phase의 순수 PyTorch 추론 판정으로 충족한다.
