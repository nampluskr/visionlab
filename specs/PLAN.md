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
| 5 | 이식 결과 실행(native Lightning 학습·평가, 순수 PyTorch 추론). 그 Phase의 모델만 대상으로 한다 | Claude가 명령을 보내고 agy가 실행 |
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
- **대응 요구 ID**: FR-1, FR-2, FR-3, FR-17, NFR-1, NFR-3
- **진행 순서**: Claude가 지시하고 codex가 구현한다. Claude가 적대적으로 검증해 수정을 요청한다(최대 3회). agy가 대조 스크립트와 테스트를 실행하고, Claude가 판정·기록한다. 지시와 검증은 아래 두 단계로 나눠 각각 진행한다.
- **범위 결정**:
  - 생성자 인자는 anomalib 2.0.0 `MVTecAD`, `Visa`와 이름·순서·기본값이 같다(D-5). 증강 인자(`train_augmentations`, `val_augmentations`, `test_augmentations`, `augmentations`)는 받되 Phase 0에서는 `None`만 지원하고, 값이 있으면 명시적 오류를 낸다.
  - 분할은 `test_split_mode="from_dir"`(기본)와 `val_split_mode`의 `same_as_test`(기본)·`from_test`를 구현한다. 그 밖의 모드는 명시적 오류를 낸다(X-3). `from_test`의 분할 결과는 같은 seed에서 anomalib과 같은 파일 목록이어야 한다.
  - VisA는 anomalib `Visa`와 같이 `root/visa_pytorch`를 읽는다. 복사본 생성(D-17)은 하지 않으며 없으면 `FileNotFoundError`를 낸다. 다운로드는 구현하지 않는다(NFR-3).
  - Dataset Transform은 RGB 변환, 이미지와 mask의 resize, float 변환만 한다. 정규화는 PreProcessor 몫이다(D-6).
  - `src/visionlab/bootstrap.py`는 현행 D-12와 맞지 않고 참조가 없어 삭제했다. 파일 해시와 오프라인 보호는 필요한 Phase에서 `utils/io.py`로 새로 만든다.

### 단계 0-A. 순수 Dataset

- **만드는 것**: `common/batch.py`(Batch, InferenceBatch), `data/common/split.py`(TestSplitMode, ValSplitMode와 분할 함수), `data/mvtec/dataset.py`(`MVTecDataset`), `data/visa/dataset.py`(`VisADataset`), `tests/00_dataset/test_data.py`.
- **완료 조건**: FR-1과 FR-3의 판정 방법이 통과한다. Lightning과 anomalib을 import하지 않고 동작한다(NFR-1, NFR-2). 테스트는 anomalib을 import하지 않는다.

### 단계 0-B. LitData와 anomalib 대조

- **만드는 것**: `data/mvtec/lightning_data.py`(`MVTecLitData`), `data/visa/lightning_data.py`(`VisALitData`), `evaluation/anomalib/compare_data.py`.
- **완료 조건**:
  - FR-2: 두 LitData의 생성자 시그니처(인자 이름·순서·기본값)가 anomalib `MVTecAD`, `Visa`와 `inspect.signature`로 일치한다. 이 비교는 anomalib을 import하므로 `compare_data.py`에서 한다.
  - FR-17: mvtec `bottle`, visa `candle`에서 파일 목록, 라벨, mask, `from_test` 분할(0.4, seed 42), 최종 입력 텐서가 anomalib(`LocalMVTecAD`, `LocalVisa`)과 일치한다. 결과를 `docs/results/00-dataset.md`에 데이터 검증 ID(manifest 해시)와 함께 기록한다.

## Phase 공통 양식

Phase 1~18의 각 모델 Phase에 적용한다. 각 Phase는 모델 하나를 다룬다. 앞 Phase에서 만든 공통 부품(데이터, 특징 추출기, 처리 구성요소, 추론기)을 재사용하고, 그 모델에 필요한 것만 추가한다.

- **대응 요구 ID**: FR-4, FR-5, FR-6, FR-18, FR-19, FR-20, FR-21, FR-22, FR-23, FR-24, NFR-3, NFR-4, NFR-5, NFR-6와 모델별 요구. FR-22, FR-23은 생성자가 pretrained를 읽는 모델에 적용한다(D-19). FR-24, NFR-6은 모든 Phase에 적용한다(D-20, D-21).
- **완료 조건**:
  - 레퍼런스: `anomalib_run.py train`과 `models/NN_<name>.yaml`로 category별 Image·Pixel AUROC와 실제 적용된 학습 예산을 그 Phase의 결과 파일에 기록한다(FR-21). 레퍼런스는 데이터셋마다 1회만 실행한다(실행 시간 비용 때문에 2회 반복을 하지 않는다). 에폭별 학습 히스토리(`train_loss_epoch` 등 run 폴더의 로그에 남은 값)도 함께 기록한다.
  - 레퍼런스 학습 예산은 공유 yaml을 따른다: 기본 `max_epochs=10`, `val_split_mode="from_test"`, `val_split_ratio=0.4`, `seed=42`(데이터 분할·`seed_everything`), batch 8/8(모델 yaml의 override 우선), `num_workers=0`, `devices=1`, `deterministic="warn"`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. 모델이 고정하는 `trainer_arguments`가 우선한다. 이식 결과도 같은 예산을 쓴다(D-22).
  - 순수 모델의 평가 출력 shape(FR-4)과 생성자 인자(FR-6)가 판정 방법을 통과한다.
  - 같은 state·입력에서 순수 모델의 raw 출력이 anomalib 원본과 일치한다(FR-5). 허용 오차는 Phase 시작 전에 확정한다.
  - native Lightning 학습·평가의 category별 Image·Pixel AUROC가 그 Phase의 레퍼런스와 각각 0.01 이내다(FR-18). 비교는 레퍼런스가 보고하는 값과 같은 정의로 한다(D-24).
  - 학습 가중치를 순수 PyTorch 모델로 불러와 추론한 category별 Image·Pixel AUROC가 그 Phase의 레퍼런스와 각각 0.01 이내다(FR-19, FR-20). 비교 정의는 위와 같다(D-24).
  - image metric만 내는 모델(DFKDE, GANomaly)은 Image AUROC만 기록·판정한다.
  - 학습에 쓰던 외부 라이브러리(FrEIA, open_clip 등)가 추론 경로에 남지 않는다(NFR-2). 독립 구현 비용은 해당 Phase 범위에 포함한다.
  - 생성자가 pretrained를 읽는 모델은 FR-22, FR-23의 판정 방법이 통과한다(D-19).
  - metric은 torchmetrics로 계산하고(FR-24, D-20), 그 Phase의 테스트는 pytest로 작성해 `python.exe -m pytest`로 실행한다(NFR-6, D-21).
  - NFR-3·NFR-4의 판정 방법이 통과한다.

## Phase 1. STFPM

- **목적**: 첫 모델로 이식 기반을 만든다. STFPM 레퍼런스를 확보한 뒤 아래 범위를 이식하고, 이식이 끝난 결과를 레퍼런스와 비교한다(D-16, D-18). Phase 0의 데이터셋을 사용하며, 이 Phase에서 만든 공통 부품은 이후 Phase가 재사용한다.
  - 순수 모델: 원본 복사본의 `STFPMModel`·`STFPMLoss`·`AnomalyMapGenerator`와 필요한 `models/components/`(FR-4, FR-16, NFR-5).
  - LightningModule·Trainer: `StfpmLitModel`, 새로 구현한 `common/` 대체 부품, native Trainer(FR-6~FR-12). teacher의 로컬 pretrained는 `STFPMModel`이 `pre_trained=False`로 구조만 만들고 `StfpmLitModel.__init__`이 `paths.toml`의 `resnet18` 파일을 불러온다(FR-22, FR-23, D-19).
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
| 9 | FRE | 역전파 | 원본은 220 epoch을 고정하지만 10 epoch로 통일한다(D-22) |
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

Claude의 검증은 다음 두 가지를 가장 중점적으로 본다.

1. anomalib 의존성의 완전한 삭제. `src/visionlab`은 anomalib을 import하거나 상속하지 않고 native Lightning만으로 구현한다(`XXXLitModel`).
2. 로컬 pretrained 가중치 사용. 외부 인터넷 연결을 차단한 상태에서 실행하고, anomalib 레퍼런스와 codex가 작성한 코드의 에폭별 학습 히스토리와 최종 성능값을 비교한다. 에폭별 히스토리의 수치 허용 오차는 아직 정하지 않았다(사용자가 정한다). 그때까지는 비교 표로 기록하고 어긋남이 크면 원인을 보고한다. 합격 판정은 아래 최종 성능 기준으로만 한다.

수정 반복은 다음 두 경우 각각 최대 3회다. 한도를 넘기면 남은 문제와 위험을 사용자에게 보고하고 판단을 요청한다.

- 적대적 검증에서 찾은 문제의 수정 요청
- 수치 기준 미달에 따른 원인 진단과 수정 요청

| 필드 | 내용 |
|---|---|
| 비교 대상 | 각 Phase에서 그 모델로 확보한 anomalib 2.0.0 레퍼런스 수치 |
| 비교하는 결과 | native Lightning 학습·평가 수치(검증 2단계), 순수 PyTorch 추론 수치(검증 3단계). 2단계는 에폭별 학습 히스토리도 함께 비교한다 |
| 비교 조건 | 레퍼런스와 같은 데이터·자산·학습 예산 |
| 비교 metric | Image AUROC, Pixel AUROC (image metric만 내는 모델은 Image AUROC) |
| 동일 수준의 기준 | category별 \|visionlab - anomalib\| ≤ 0.01 (metric 각각) |

FR-18은 각 Phase의 native Lightning 판정으로, FR-20은 각 Phase의 순수 PyTorch 추론 판정으로 충족한다.
