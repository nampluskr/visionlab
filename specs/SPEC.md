> 버전: v0.1 · 작성일: 2026-10-22

# SPEC

## 1. 기능 요구사항

| ID | 내용 | 판정 방법 |
|---|---|---|
| FR-1 | 순수 `MVTecDataset`·`VisADataset`은 `(image, target)`을 반환한다. image는 float32 `[3,H,W]`·`[0,1]`, 평가 target은 label 0/1, mask int64 `[H,W]`, `sample_id`, `image_path`를 가진다 | 정상·이상 샘플의 shape·dtype·범위·필드 검사 테스트 통과 |
| FR-2 | `MVTecLitData`·`VisALitData`는 native `LightningDataModule`이며 생성자 인자 이름·순서·기본값이 anomalib 2.0.0의 대응 클래스와 같다 | `inspect.signature` 비교 일치. anomalib을 import하는 비교이므로 `evaluation/anomalib/compare_data.py`에서 수행 |
| FR-3 | 분할 기본값은 `test_split_mode="from_dir"`, `val_split_mode="same_as_test"`다. `from_test` 명시 시 valid와 test가 겹치지 않고, 고정 seed에서 새 인스턴스도 같은 목록을 만든다. 미구현 모드는 명시적 오류를 낸다 | 파일 목록 교집합·재생성 비교, 미구현 모드 예외 테스트 |
| FR-4 | 이식한 순수 모델(예: `STFPMModel`·`EfficientAdModel`)의 평가 출력은 score `[B,1]`, anomaly map `[B,1,H,W]`다. image metric만 내는 모델(DFKDE, GANomaly)은 anomaly map 없이 score만 낸다. GANomaly의 score는 `[B]`다. CFA는 원본의 `squeeze()`를 유지해 anomaly map이 `[B,H,W]`(batch 1이면 `[H,W]`), score가 `[B]`(batch 1이면 0차원)다. SuperSimpleNet은 원본의 `squeeze()`를 유지해 score가 `[B]`(batch 1이면 0차원)이고 anomaly map은 `[B,1,H,W]`다. DRAEM은 원본의 `softmax(...)[:, 1]`을 유지해 anomaly map이 `[B,H,W]`, score가 `[B]`다. CFLOW도 원본의 `.squeeze(1)`를 유지해 anomaly map이 `[B,H,W]`, score가 `[B]`이며, 두 값은 batch 전체의 최댓값으로 정규화되므로 같은 이미지라도 batch 구성에 따라 값이 달라진다. CSFlow는 원본의 평탄화 평균을 유지해 score가 `[B]`이고 anomaly map은 `[B,1,H,W]`다. DSR은 원본의 평가 `forward`를 유지해 anomaly map이 `[B,H,W]`, score가 `[B]`(batch 1에서도 `[1]`)다. WinCLIP도 원본의 평가 `forward`를 유지해 anomaly map이 `[B,H,W]`, score가 `[B]`다 | forward shape 테스트. image 전용 모델은 anomaly map이 없음을 확인. CFA, SuperSimpleNet, DRAEM, CFLOW, CSFlow, DSR, WinCLIP은 원본 동작의 shape를 확인 |
| FR-5 | 같은 state·입력에서 순수 모델의 raw 출력이 anomalib 원본과 일치한다 | score·map의 allclose 결과 기록. 허용 오차는 실험 전에 확정 |
| FR-6 | 이식한 LightningModule(예: `StfpmLitModel` 인자 6개, `EfficientAdLitModel` 인자 11개, D-18)의 생성자 인자 이름·순서·기본값이 anomalib 2.0.0과 같다 | `inspect.signature` 비교 일치 |
| FR-7 | PreProcessor·PostProcessor·Evaluator·Visualizer 인자는 `True`·`False`·visionlab 객체를 받고, `None`·잘못된 타입은 오류를 낸다 | 네 경우 각각의 테스트 |
| FR-8 | PreProcessor는 STFPM에 ImageNet 정규화를 1회 적용하고 EfficientAD에는 외부 정규화를 적용하지 않는다. 선택 인자 `center_crop`(기본 `None`, PatchCore용)으로 중앙 crop도 할 수 있으며 기존 모델 동작은 바뀌지 않는다. train·valid·test·predict에서 같다 | 단계별 최종 forward 입력값 비교 |
| FR-9 | PostProcessor 보정 상태는 fit validation에서만 확정된다. standalone validate·test·predict 전후 상태가 같고, 후처리 전후 raw tensor가 바뀌지 않는다 | 상태 전후 비교, raw 불변 테스트 |
| FR-10 | Evaluator는 valid에서 raw Image·Pixel AUROC, test에서 raw AUROC와 후처리 F1을 계산한다. valid/test 상태가 분리되고, 계산 불가 시 0 대신 사유를 남긴다 | 두 epoch 연속 실행, 배치 분할 변경 시 지표 일치, 단일 클래스 사례 테스트 |
| FR-11 | Visualizer는 기본 활성이며 test·predict 결과를 PNG로 저장한다. sample ID별 경로로 파일 충돌이 없다 | 생성 파일 수 = 샘플 수, 같은 basename 다른 범주 구분 |
| FR-12 | native fit·validate·test·predict, `ckpt_path` 복원, 학습 재개가 동작한다. checkpoint에 같은 epoch의 후처리 상태가 포함된다 | 단계별 실행 성공, 복원 후 같은 출력 |
| FR-13 | LightningCLI는 4계층 YAML(default·data·models·trainer)과 `--data`·`--model` 선택을 지원하고, 알 수 없는 인자는 오류를 낸다 | CLI fit·test 실행, 오류 사례 테스트 |
| FR-14 | 독립 Predictor는 순수 모델·PreProcessor·PostProcessor로 추론하며 Wrapper 경로와 같은 출력을 낸다. pretrained 가중치 파일 없이 생성된다 | Lightning import 차단 상태 실행, 출력 allclose. `D:\backbones`에 접근할 수 없는 상태에서 Predictor 생성과 추론 성공 |
| FR-15 | EfficientAD는 teacher 통계, Imagenette penalty, valid quantile 보정, 학습 batch 1로 동작한다 | Hook별 상태 확인, 재개 시 teacher·통계 미초기화 확인 |
| FR-16 | STFPM의 선택적 Tiler는 1024 입력·256 타일에서 이미지당 16타일을 만들고 anomaly map은 1024 해상도다 | shape 테스트 |
| FR-17 | Dataset·DataModule의 파일·라벨·mask·split·최종 입력을 anomalib과 대조한다 | 데이터 검증 ID와 결과 기록 |
| FR-18 | 같은 manifest·자산·학습 예산으로 MVTec·VisA 범주별 AUROC를 anomalib과 비교한다 | 범주별 Image AUROC·Pixel AUROC 차이가 각각 0.01 이내. 비교는 레퍼런스가 보고하는 값과 같은 정의로 하며, 정규화 AUROC는 export 기반 `predict.py`에서 raw와 함께 기록한다(D-24). 비교 결과 표 기록 |
| FR-19 | native Lightning 학습 결과에서 순수 PyTorch 모델 가중치(`state_dict`, 학습하지 않는 teacher 포함)와 추론에 필요한 PreProcessor·PostProcessor 상태를 저장하고, Lightning·anomalib 없이 순수 PyTorch 모델로 불러온다 | Lightning·anomalib import 차단 상태에서 로딩 성공, 저장 전후 같은 입력의 출력 allclose, 저장 파일에 teacher 키 포함 |
| FR-20 | FR-19로 불러온 순수 PyTorch 모델의 추론으로 MVTec·VisA 범주별 AUROC를 계산해 anomalib 레퍼런스와 비교한다 | 범주별 Image AUROC·Pixel AUROC 차이가 각각 0.01 이내. 비교는 레퍼런스가 보고하는 값과 같은 정의로 한다(D-24). 비교 결과 표 기록 |
| FR-21 | `evaluation/anomalib/`은 각 Phase의 anomalib 2.0.0 image 모델(`models/*.yaml` 18개 중 하나)을 로컬 데이터·가중치만으로 실행해 MVTec·VisA 평가 category의 레퍼런스 수치를 얻는다 | 네트워크 차단 실행 성공, 모델·범주별 Image·Pixel AUROC와 학습 예산 기록 |
| FR-22 | 순수 모델 생성자는 pretrained 파일을 읽거나 내려받지 않는다. 학습하지 않는 부분(STFPM teacher 등)의 가중치도 `state_dict`에 포함된다 | `paths.toml`과 `D:\backbones`에 접근할 수 없고 네트워크를 차단한 상태에서 순수 모델 생성과 `state_dict` 로딩이 성공하는 테스트. 생성자 실행 중 파일 열기 호출이 없음을 확인 |
| FR-23 | `XXXLitModel`은 생성자 인자를 늘리지 않고 생성 시 `paths.toml [backbones]`(timm 모델이 아닌 teacher는 `[teachers]`)의 로컬 파일을 pretrained 부분에 불러온다. 파일이 없으면 `FileNotFoundError`를 내고 무작위 대체를 하지 않는다 | `inspect.signature`가 FR-6과 같음. 파일 누락 시 예외 테스트. 불러온 teacher 가중치가 anomalib 레퍼런스의 teacher 가중치와 같음(`torch.equal` 또는 allclose, 허용 오차는 실험 전에 확정, anomalib을 import하므로 `evaluation/anomalib/`에서 수행) |
| FR-24 | AUROC, F1, F1 기반 임계값은 anomalib 2.0.0과 같은 torchmetrics 클래스(`BinaryROC`와 `auc`, `BinaryF1Score`, `BinaryPrecisionRecallCurve`)로 계산한다(D-20). 순수 파일이 이를 import해도 Lightning 없이 동작한다 | `src/visionlab`에서 torchmetrics 외의 metric 구현(sklearn 등) import 0건. 같은 입력에서 anomalib metric과 값이 같음(`evaluation/anomalib/`에서 대조). Lightning import 차단 상태에서 metric 계산 성공 |

## 2. 비기능 요구사항

| ID | 내용 | 판정 방법 |
|---|---|---|
| NFR-1 | visionlab은 anomalib 없이 import·실행된다 | anomalib import 차단 테스트 |
| NFR-2 | 순수 Dataset·모델·처리 구성요소·Predictor는 Lightning 없이 동작한다 | Lightning import 차단 테스트 |
| NFR-3 | 로컬 자산만 사용한다. 필요한 자산이 없으면 무작위 대체 없이 오류를 낸다 | 네트워크 차단 실행, 누락 자산 테스트 |
| NFR-4 | 실행마다 최종 설정·분할 목록·가중치 해시·seed·checkpoint 식별 정보를 저장한다 | 실행 산출물에 해당 파일 존재 |
| NFR-5 | 이식한 파일은 원본 저작권·SPDX 표기를 유지한다 | 이식 파일 헤더 grep |
| NFR-6 | `tests/`의 테스트는 pytest로 작성하고 실행한다. `unittest`를 쓰지 않는다(D-21) | `tests/`의 `unittest` import 0건, `python.exe -m pytest tests\...` 실행 성공. import 차단 테스트는 subprocess로 확인 |
| NFR-7 | 추론 경로(각 `model.py`, `models/components/`, 순수 `common/` 파일, `inference/`)는 torch와 부록 A의 허용 목록에 있는 외부 패키지만 import한다. 지연 import로 허용한 곳은 부록 A에 적은 파일과 함수뿐이다 | `tests/common/test_dependency.py`가 허용 목록 밖의 import를 차단한다. 현재 이 테스트는 anomalib, open_clip, ftfy, scikit-learn, regex, lightning만 검사하며, 허용 목록 전체를 검사하도록 넓히는 일은 부록 A 확정 뒤 별도로 한다 |

## 3. 제약사항

- **C-1**: D-14의 환경을 사용한다. WinPython Python 3.11.8, torch 2.5.1+cu121, Lightning 2.5.5, anomalib 2.0.0, 데이터 `D:\datasets`, 가중치 `D:\backbones`.
- **C-2**: EfficientAD의 지원 기준은 256×256 입력, 학습 batch 1, 단일 장치, gradient accumulation 1이다.
- **C-3**: Windows 예제는 `num_workers=0`으로 시작하고, EfficientAD는 `num_sanity_val_steps=0`을 사용한다.

## 4. 미구현 대상

- **X-1**: classification 등 anomaly_detection 외의 task.
- **X-2**: `evaluation/anomalib/models/*.yaml` 18개 밖의 anomalib 모델. 18개 모델은 PLAN의 Phase 순서로 하나씩 이식한다. FR-4·FR-5·FR-6·FR-18·FR-19·FR-20은 각 Phase의 모델에 같은 방식으로 적용한다.
- **X-3**: synthetic 분할 모드. 지정 시 오류만 보고한다.
- **X-4**: sensitivity, image/pixel threshold matching, export용 변환, 고급 시각화 옵션.
- **X-5**: 결과를 원본 파일 해상도로 되돌리는 기능.
- **X-6**: 기존 checkpoint 키·옛 import와의 자동 호환.
- **X-7**: TensorBoard·W&B 로거.
- **X-8**: 분산 실행 검증.

## 부록 A. 추론 경로 외부 패키지 허용 목록

NFR-7이 가리키는 허용 목록이다. 각 항목의 근거는 `DECISIONS.md`의 해당 결정이며, 사용자가 허용을 결정한 패키지만 올린다(INTENT 2026-10-28 승인). 다음은 허용 목록이다.

| 패키지 | 허용 근거 | 허용 위치 | 추론 경로 |
|---|---|---|---|
| torchmetrics | D-20 | `common/metric.py`, `common/post_processor.py` | 남음 |
| kornia | D-23 | `models/components/filters/blur.py`(추론), `models/draem/loss.py`, `models/dsr/loss.py`(학습 전용) | `blur.py`만 남음 |
| scikit-learn | D-25 | `models/components/dimensionality_reduction/random_projection.py`, `models/cfa/model.py`의 `initialize_centroid`(함수 안 지연 import) | 남지 않음 |
| FrEIA(와 numpy, scipy) | D-26 | FastFlow, CFLOW, UFlow, CSFlow 모델 파일과 `models/components/flow/all_in_one_block.py` | 남음 |
| regex | D-28 | `models/winclip/tokenizer.py`의 `SimpleTokenizer.__init__`(지연 import) | 남지 않음 |

timm, torchvision, numpy(FrEIA의 요구 사항이 아닌 단독 사용), tqdm, PIL, mpmath는 코드에서 추론 경로에 import되지만 허용 결정이 없어 이 목록에 올리지 않았다. 허용 여부는 사용자 결정을 기다린다(`docs/dependency-exceptions.md` 2.1절).
