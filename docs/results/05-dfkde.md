# Phase 5. DFKDE

모델 폴더 이름은 `dfkde`이고 원본은 anomalib 2.0.0의 `anomalib\models\image\dfkde\`(`C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\Lib\site-packages\anomalib\models\image\dfkde\`)이다. 범위와 완료 조건은 `specs/PLAN.md`의 Phase 5와 "Phase 공통 양식"을 따른다. image metric만 내는 모델이라 category별 Image AUROC 차이 0.01 이내만 판정한다(Pixel AUROC는 없다).

## 상태

다음 표는 단계별 진행 상태다. 수정 반복 한도 3회는 단계별로 센다.

| 단계 | 내용 | 상태 |
|---|---|---|
| 레퍼런스 | `anomalib_run.py train` x 2 (mvtec 1회, visa 1회, 1 epoch) | 실행 완료 (2절) |
| 5-0 | `evaluation/visionlab/predict.py`가 정규화 점수 AUROC를 raw와 함께 출력 (D-24) | 구현, 검증, 실행 완료 (수정 요청 0회) |
| 5-A | 순수 모델 (`models/dfkde/model.py`, `components/stats/kde.py`, `components/classification/`, `tests/05_dfkde`의 순수 부분) | 구현, 검증, 실행 완료 (수정 요청 0회) |
| 5-B | Lightning (`DfkdeLitModel`, image 전용 Evaluator, configs, PCA·KDE buffer 저장·복원, export·Predictor 등록, 비교 스크립트) | 구현, 검증, 실행 완료 (수정 요청 0회, 결과 5절) |

처리 구성요소(PreProcessor, PostProcessor), 특징 추출기(`TimmFeatureExtractor`), `DynamicBufferMixin`, Phase 4의 `PCA`는 앞 Phase의 것을 재사용한다.

## 사용자 결정 (2026-10-26)

다음은 Phase 시작 전에 사용자가 정한 항목이다.

| 항목 | 결정 |
|---|---|
| 학습 에폭 | 모델이 고정한 1 epoch을 따른다(PaDiM, DFM과 같은 예외, D-22). 레퍼런스 yaml과 native yaml 모두 `max_epochs: 1` |
| FR-5 허용 오차 | `atol=1e-6, rtol=1e-5` (Phase 2~4와 같음) |
| D-24 구현 | `evaluation/visionlab/predict.py`가 export의 정규화 점수 AUROC를 raw 순위 보존 AUROC와 함께 출력한다. native 로그와 `Evaluator`는 바꾸지 않는다 |
| FR-4 | image 전용 모델(DFKDE, GANomaly)은 anomaly map 없이 score만 낸다. `specs/SPEC.md`의 FR-4에 반영했다 |
| 실행 범위 | 새로 구현한 모델(dfkde)의 명령만 실행한다. 앞 모델의 pytest와 `compare_outputs`는 돌리지 않는다 |
| 판정 | Image AUROC만 판정한다. 정규화 점수 기준 값(D-24)과 raw 값을 함께 기록한다 |
| D-24 범위 (결과 확인 후 추가, 2026-10-26) | export 한정으로 확정한다. 정규화 AUROC는 export 기반 `predict.py`에서 raw와 함께 기록하고, native `train.py` 로그는 raw만 기록한다. `specs/DECISIONS.md` D-24와 `specs/SPEC.md` FR-18에 반영하며 Phase 5 커밋에 포함한다 |
| `test_dependency.py` 전체 실행 (사후, 2026-10-26) | 사용자가 승인했다 |
| 앞 모델 회귀 (2026-10-26) | 보류한다(사용자 결정). 이번에는 돌리지 않는다 |
| image 전용 Evaluator의 val `image_AUROC` (2026-10-26) | 원본과의 차이로 결과 문서에 기록만 한다. `specs/`는 고치지 않는다 |
| Phase 종료 (2026-10-26) | 사용자가 종료를 결정했다. 방침 문서 3곳은 별도 Setup 커밋으로, Phase 5 변경은 `Phase 05: dfkde ...` 커밋으로 푸시한다 |

## 1. 모델 노트

### 1.1 원본과 생성자

다음 표는 원본 파일과 이식 위치의 대응이다(D-15, D-18).

| 원본 | 이식 위치 | 비고 |
|---|---|---|
| `models/image/dfkde/torch_model.py` (`DfkdeModel`) | `src/visionlab/anomaly_detection/models/dfkde/model.py` | import 변경, `pre_trained`는 순수 모델에서 구조만 만드는 방식으로(D-19) |
| `models/image/dfkde/lightning_model.py` (`Dfkde`) | `.../models/dfkde/lightning_model.py` (`DfkdeLitModel`) | native `LightningModule`, `MemoryBankMixin` hook을 직접 구현(D-11), image 전용 `Evaluator` |
| `models/image/dfkde/__init__.py` | `.../models/dfkde/__init__.py` | `lightning_model`을 import하지 않도록 고침(D-18 예외) |
| `components/classification/kde_classifier.py` (`KDEClassifier`, `FeatureScalingMethod`) | `.../models/components/classification/kde_classifier.py` | 새로 복사 |
| `components/stats/kde.py` (`GaussianKDE`) | `.../models/components/stats/kde.py` | 새로 복사 |
| `components/dimensionality_reduction/pca.py` (`PCA`) | Phase 4에서 복사한 것을 재사용 | `fit_transform`을 처음 쓴다 |

생성자는 다음과 같다. `DfkdeLitModel`은 `Dfkde`와 인자 이름·순서·기본값이 같아야 한다(FR-6, D-5).

| 클래스 | 시그니처 |
|---|---|
| `Dfkde` | `backbone="resnet18"`, `layers=("layer4",)`, `pre_trained=True`, `n_pca_components=16`, `feature_scaling_method=FeatureScalingMethod.SCALE`, `max_training_points=40000`, `pre_processor=True`, `post_processor=True`, `evaluator=True`, `visualizer=True` (10개) |
| `DfkdeModel` | `backbone`, `layers`, `pre_trained=True`, `n_pca_components=16`, `feature_scaling_method=FeatureScalingMethod.SCALE`, `max_training_points=40000` |

### 1.2 학습과 평가 동작

원본의 동작은 다음과 같다.

- **모델:** `TimmFeatureExtractor`(eval)로 `layers`의 특징을 뽑고 layer마다 `adaptive_avg_pool2d((1, 1))` 뒤 편 벡터를 `torch.cat(...)`(dim 0)으로 합친다. 기본 설정은 layer가 `layer4` 하나라 512차원 특징이다. 학습 모드의 `forward`는 이 특징을 반환하고, 평가 모드는 `KDEClassifier`의 점수를 `InferenceBatch(pred_score=...)`로 반환한다(`anomaly_map` 없음).
- **학습:** `training_step`은 `self.model(batch.image)`(학습 모드 특징)를 `self.embeddings`에 모으고 더미 loss(0)를 반환한다. `configure_optimizers`는 `None`이다. 학습 가능한 파라미터가 없다.
- **적합(`fit`):** `MemoryBankMixin`의 `on_validation_start`와 `on_train_epoch_end`가 `_is_fitted` buffer가 False일 때 한 번 호출한다. `torch.vstack(self.embeddings)`로 `KDEClassifier.fit`을 호출한다. 학습 점수가 `n_pca_components`보다 적으면 모델을 만들지 않고 `False`를 반환한다. `max_training_points`를 넘으면 `random.sample`로 고른다(이번 데이터에서는 해당 없음).
- **분류기:** `PCA.fit_transform`(in-place 평균 빼기, full SVD, `num_components`는 정수)으로 16성분을 만들고, `scale`이면 최대 길이로 나눈다. `GaussianKDE.fit`이 bandwidth 변환(Cholesky)과 `norm`을 만든다. 점수는 `exp` 합의 평균에 `1e-300`을 더한 값의 `log`를 `1 / (1 + exp(0.05 * (score - 12)))`로 변환한 것이다.
- **`trainer_arguments`:** `{"gradient_clip_val": 0, "max_epochs": 1, "num_sanity_val_steps": 0}`이다.
- **평가:** `configure_evaluator`가 test metric으로 `image_AUROC`, `image_F1Score`만 둔다. 판정 metric은 Image AUROC다.

### 1.3 외부 의존과 로컬 자산

- **외부 의존:** timm, torchvision이다. anomalib 의존은 `Batch`, `InferenceBatch`, `AnomalibModule`, `MemoryBankMixin`, `KDEClassifier`, `GaussianKDE`, `PCA`, `TimmFeatureExtractor`, `PreProcessor`, `PostProcessor`, `Evaluator`, `Visualizer`, `LearningType`이다.
- **로컬 가중치:** `paths.toml [backbones]`의 `resnet18`(Phase 1, 3에서 사용). 레퍼런스는 `local_setup.py`가 timm 이름을 로컬 로더로 바꾼다. visionlab은 `DfkdeLitModel.__init__`이 `feature_extractor`에 주입한다(D-19).
- **데이터:** MVTec `bottle`, VisA `candle`. 분할은 `from_test`, 0.4, seed 42, batch 8/8, `num_workers=0`이다.
- **학습 예산:** `max_epochs=1`(모델 고정)이다.

### 1.4 이식에서 의도적으로 달라지는 곳과 유지하는 곳

- **pretrained 주입(D-19):** 순수 `DfkdeModel`은 `pre_trained=False`로 구조만 만든다. `DfkdeLitModel`이 `resnet18` 로컬 파일을 `feature_extractor`에 불러온다.
- **`MemoryBankMixin` 대체(D-11):** 공유 기반 클래스를 만들지 않고 `_is_fitted` buffer, `on_validation_start`, `on_train_epoch_end`를 `DfkdeLitModel`에 직접 쓴다(Phase 3, 4와 같다).
- **image 전용 Evaluator:** `Evaluator.default`는 pixel 지표를 포함하므로 `DfkdeLitModel`이 image 전용 `Evaluator`를 만든다. test는 `image_AUROC`, `image_F1Score`만 둔다(anomalib의 `configure_evaluator`와 같다).
- **원본 동작 유지:** `get_features`의 `torch.cat(dim 0)`, `+1e-300`(float32에서 사라져 효과가 없다), `PCA.fit_transform`의 in-place 평균 빼기와 full SVD, `GaussianKDE.cov`의 in-place 평균 빼기, 표본별 KDE 루프는 고치지 않는다.
- **점수 포화 가능성:** 위 `1e-300`이 float32에서 0이라 KDE 값이 underflow하면 `log`가 -inf가 되고 점수가 1.0이 된다. 레퍼런스와 같이 동점이 생길 수 있다. D-24에 따라 정규화 점수 AUROC도 함께 기록한다.
- **히스토리 비교:** loss가 없고 1 epoch이라 `compare_history.py`는 쓰지 않는다. 대신 레퍼런스 ckpt와 native ckpt의 PCA·KDE 통계를 표로 기록한다(판정 아님).
- **FR-5 허용 오차:** `pred_score`는 `atol=1e-6, rtol=1e-5`, backbone 가중치는 `torch.equal`이다.

## 2. 레퍼런스

학습 예산은 모델이 고정한 1 epoch이다(D-22). `models/05_dfkde.yaml`에 `trainer.max_epochs: 1`이 있고, 두 로그 모두 `Trainer.fit stopped: max_epochs=1 reached`가 나온다.

### 2.1 1 epoch 레퍼런스

2026-10-26에 agy가 `anomalib_run.py train --config default.yaml --config data/<data>.yaml --config models/05_dfkde.yaml`을 데이터셋마다 1회 실행했다(실행으로 검증함). Claude는 로그 파일에서 값을 직접 확인했다. 두 로그 모두 `network access is blocked`, `URLError`, `ConnectionError`, `Traceback`은 0건이다. 로그는 PowerShell `Tee-Object` 출력이라 UTF-16이다. 종료 코드는 둘 다 0이다(agy 보고). 승인 창의 스크립트 전문은 Claude가 agy의 transcript에서 읽어 지시한 명령과 같음을 확인했다(agy가 시간 측정 줄을 덧붙였다).

agy가 PowerShell에서 실행한 명령은 다음과 같다(Claude가 보낸 지시 원문. 파일은 고치지 않았다).

```powershell
cd D:\projects\visionlab\evaluation\anomalib
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/05_dfkde.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\05-ref-mvtec-261026-1.log
& $py anomalib_run.py train --config default.yaml --config data/visa.yaml --config models/05_dfkde.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\05-ref-visa-261026-1.log
```

다음 표는 1 epoch 레퍼런스(1회) 결과다. image 전용 모델이라 Pixel 지표는 없다.

| 데이터 / category | Image AUROC | 참고 | run 폴더 | 로그 |
|---|---|---|---|---|
| mvtec / bottle | 0.93860 | image F1 0.8810 | `outputs/anomalib/Dfkde/LocalMVTecAD/bottle/v0` | `outputs/logs/05-ref-mvtec-261026-1.log` |
| visa / candle | 0.83417 | image F1 0.6923 | `outputs/anomalib/Dfkde/LocalVisa/candle/v0` | `outputs/logs/05-ref-visa-261026-1.log` |

적용된 학습 예산은 `max_epochs=1`, `val_split_mode=from_test`, `val_split_ratio=0.4`, `seed=42`, `batch 8/8`, `num_workers=0`, `backbone=resnet18`, `layers=[layer4]`, `n_pca_components=16`, `max_training_points=40000`이다(`models/05_dfkde.yaml`과 공유 yaml). 체크포인트는 각 run 폴더의 `weights/lightning/model.ckpt`다.

### 2.2 에폭별 학습 히스토리

DFKDE는 loss가 없고 학습이 1 epoch이라 비교할 히스토리가 없다. 레퍼런스와 native의 비교는 적합 결과(PCA·KDE 통계)로 대신하며 5.4절에서 다룬다.

### 2.3 레퍼런스 확인 사항

- 두 데이터셋 모두 종료 코드 0이고 오류가 없다.
- 레퍼런스가 1회뿐이라 run 간 편차는 측정하지 못했다.

## 3. 이식 메모

### 3.1 원본과 이식 파일의 대응

다음 표는 새로 만든 파일과 원본의 대응이다. 원본과의 차이는 `difflib`로 확인했다(구현함, 정적 검증).

| 이식 파일 (`src/visionlab/anomaly_detection/models/`) | 원본 | 원본과 다른 줄 |
|---|---|---|
| `dfkde/model.py` | `dfkde/torch_model.py` | import 3줄(`InferenceBatch`, `TimmFeatureExtractor`, `KDEClassifier`·`FeatureScalingMethod`), `TimmFeatureExtractor(... pre_trained=False)`와 주석 한 줄(D-19) |
| `dfkde/__init__.py` | `dfkde/__init__.py` | `DfkdeModel`만 노출(D-18 예외) |
| `dfkde/lightning_model.py` (`DfkdeLitModel`) | `dfkde/lightning_model.py`와 `components/base/memory_bank_module.py`의 hook | native `LightningModule`로 새로 작성. 10개 인자 동일, `_is_fitted` buffer, `fit`(`classifier.fit` 반환값 무시), `on_validation_start`, `on_train_epoch_end`, 더미 loss는 원본 그대로, 로컬 backbone 주입(D-19), D-8의 post_processor 갱신 제한, image 전용 Evaluator |
| `components/classification/kde_classifier.py` | `components/classification/kde_classifier.py` | `PCA`, `GaussianKDE` import 두 줄 |
| `components/stats/kde.py` | `components/stats/kde.py` | `DynamicBufferMixin` import 한 줄 |

### 3.2 새로 쓴 코드와 앞 Phase 공유 파일 변경

- **새로 쓴 파일:** `configs/models/dfkde.yaml`, `evaluation/anomalib/compare_dfkde_stats.py`(정보 기록용), `tests/05_dfkde/test_dfkde.py`(11개 함수), `tests/05_dfkde/test_dfkde_lightning.py`(5개 함수). `tests\05_dfkde`는 합계 20개 테스트다.
- **공유 파일 변경:** `evaluation/visionlab/predict.py`(0단계: 정규화 점수 AUROC 두 행과 `--model dfkde`), `components/stats/__init__.py`(`GaussianKDE` 노출 추가), `utils/io.py`와 `inference/predictor.py`(`dfkde` 등록, `feature_scaling_method` 문자열을 Enum으로 복원), `evaluation/anomalib/compare_outputs.py`(`--model dfkde`), `tests/common/test_cli.py`, `tests/common/test_dependency.py`.
- **문서 변경:** `specs/SPEC.md` FR-4에 image 전용 모델의 예외 문구를 사용자 결정으로 추가했다. 실행 범위 방침을 `AGENTS.md`, `docs/workflow/model-port.md`, `specs/PLAN.md`에 반영한 것은 Phase 5 변경과 별개다.
- **새 부품:** `GaussianKDE`, `KDEClassifier`, `FeatureScalingMethod`는 원본 복사다. 새로 구현한 대체 부품은 없다.

### 3.3 이식 중 확인된 사항

- **image 전용 Evaluator:** `Evaluator.default`는 pixel 지표를 포함하므로 `DfkdeLitModel`이 image 전용 Evaluator를 만든다. 원본 `configure_evaluator`는 test metric만 두지만, D-8의 validation raw AUROC를 따라 val `image_AUROC`를 추가했다(코드에 한 줄 주석).
- **Enum 직렬화:** export의 `model_args`에는 `feature_scaling_method`를 문자열 값으로 넣고 `Predictor`가 `FeatureScalingMethod`로 복원한다.
- **학습 모드 forward:** `training_step`은 원본처럼 `self.model(batch.image)`(학습 모드 특징)를 `embeddings`에 넣는다. `DfmLitModel`과 달리 `get_features`를 직접 부르지 않고 `squeeze()`도 없다(원본이 그렇다).

## 4. 적대적 검증

검증은 5-0, 5-A, 5-B 각각 한 회차씩 했다. 수정 요청은 5-0 0회, 5-A 0회, 5-B 0회(한도 3회 이내)다.

### 4.1 5-0 (predict.py의 정규화 점수 AUROC)

| 지적 | 심각도 | 처리 |
|---|---|---|
| `git diff`: 기존 네 지표의 이름·순서·값이 그대로이고 `image_AUROC_normalized`, `pixel_AUROC_normalized` 두 행과 영어 주석 한 줄만 추가됨. 입력이 없으면 NaN과 `no_data`로 출력됨 | 확인 | 수정 없음 |

### 4.2 5-A (순수 모델과 KDE·분류기 부품)

| 지적 | 심각도 | 처리 |
|---|---|---|
| 원본과의 `difflib` 대조: `model.py`는 import와 D-19의 `pre_trained=False` 한 줄뿐이고 `get_features`의 `cat(dim 0)`, `forward`, 학습·평가 분기가 같음. `kde_classifier.py`는 import 두 줄, `kde.py`는 import 한 줄뿐이고 `1e-300`, 표본별 루프, in-place 연산, `random.sample`이 유지됨 | 확인 | 수정 없음 |

### 4.3 5-B (Lightning, export, 비교 스크립트)

| 지적 | 심각도 | 처리 |
|---|---|---|
| `DfmLitModel`과의 `difflib` 대조: 차이가 생성자 인자, image 전용 Evaluator, `training_step`의 `self(image)`, `classifier.fit`, export 인자뿐임. `_is_fitted` 조건, 더미 loss, hook, D-8 처리가 같음. `Predictor` 등록은 `feature_scaling_method` 복원 외 기존 동작을 바꾸지 않음 | 확인 | 수정 없음 |
| 정적 검토는 LitModel, yaml, 등록 코드 위주로 했다. `compare_outputs.py`의 dfkde 분기는 코드를 줄 단위로 읽지 않고 실행 결과(5.3절)로 확인했다 | 낮음 | 수정 없음 |

남은 위험은 6절에 적었다.

## 5. 이식 결과

### 5.1 최종 판정 (1 epoch, Image AUROC 차이 0.01 이내)

native는 `train.py`가 test에서 기록한 값이고 순수 PyTorch는 `export/model.pt`만 불러 `predict.py`로 평가한 값이다. 모두 실행으로 검증했다. D-24에 따라 레퍼런스가 보고하는 값(정규화 점수 기준)과 같은 정의로 비교하며, visionlab의 raw 순위 보존 AUROC도 함께 적는다.

| 데이터 / category | 레퍼런스 | native (raw) | 순수 PyTorch (raw) | 순수 PyTorch (정규화) | 차이(native) | 차이(순수, 정규화) | 판정 |
|---|---:|---:|---:|---:|---:|---:|---|
| mvtec / bottle | 0.938596487 | 0.938596487 | 0.93859649 | 0.93859649 | 0 | 0 | 통과 |
| visa / candle | 0.834166646 | 0.834166646 | 0.83416665 | 0.83416665 | 0 | 0 | 통과 |

raw 값과 정규화 값이 두 데이터셋 모두 같으므로 D-24의 "두 정의가 0.01 넘게 다르면 원인을 적는다" 조건에는 해당하지 않는다. image F1도 레퍼런스, native, 순수 PyTorch가 같다(mvtec 0.88095236, visa 0.69230771). Pixel 지표는 이 모델에 없어 `predict.py`가 NaN(`no_data`)로 출력한다.

run 폴더와 로그는 다음과 같다.

| 대상 | 위치 |
|---|---|
| 레퍼런스 | `outputs/anomalib/Dfkde/LocalMVTecAD/bottle/v0`, `outputs/anomalib/Dfkde/LocalVisa/candle/v0` |
| native | `outputs/visionlab/dfkde/mvtec_bottle/v0`, `outputs/visionlab/dfkde/visa_candle/v0` (가중치 `D:\backbones\resnet18-f37072fd.pth` sha256 `f37072fd...07ec`) |
| 순수 PyTorch | 같은 run의 `predict/metrics.json` |
| 로그 | `outputs/logs/05-native-{mvtec,visa}-261026-1.log`, `05-pure-{mvtec,visa}-261026-1.log` |

네 native·순수 로그에 `network access is blocked`, `URLError`, `ConnectionError`, `Traceback`은 0건이다.

### 5.2 테스트와 비교 스크립트

다음 표는 agy가 실행한 묶음의 결과다(Claude가 로그 파일에서 확인). 실행 대상은 DFKDE만이고 앞 모델의 회귀 pytest와 `compare_outputs`는 돌리지 않았다.

| 대상 | 결과 | 로그 |
|---|---|---|
| `tests\05_dfkde` | 20 passed | `05-pytest-261026-1.log` |
| `tests\common\test_cli.py -k dfkde` | 1 passed (8 deselected) | `05-pytest-cli-261026-1.log` |
| `tests\common\test_dependency.py` (전체 파일) | 6 passed | `05-pytest-dependency-261026-1.log` |
| `compare_outputs.py --model dfkde` (mvtec 체크포인트, visa 체크포인트) | PASS 23, FAIL 0 (각각). FR-6, FR-23, FR-5(`pred_score` 최대 차이 0.0, `anomaly_map`이 두 쪽 모두 None), FR-24 | `05-compare-outputs-{mvtec,visa}-261026-1.log` |

`test_dependency.py`는 파일 하나가 모든 순수 모듈의 import 분리(NFR-1, NFR-2)를 한 번에 검사하는 구조여서 DFKDE 모듈 검증을 위해 전체를 돌렸다. 학습이나 앞 모델 실행은 없고 import 검사만 한다. 공유 테스트 파일의 전체 실행이라 사용자 판단 대상으로 보고했고, 사용자가 사후에 승인했다(2026-10-26).

실행한 명령은 다음과 같다(Claude가 보낸 지시 원문. agy가 시간 측정 줄을 덧붙여 하나의 스크립트로 순서대로 실행했고, 승인 창의 전문을 Claude가 agy의 transcript에서 읽어 지시와 같음을 확인한 뒤 한 번 승인했다).

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py -m pytest tests\05_dfkde 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\05-pytest-261026-1.log
& $py -m pytest tests\common\test_cli.py -k dfkde 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\05-pytest-cli-261026-1.log
& $py -m pytest tests\common\test_dependency.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\05-pytest-dependency-261026-1.log
& $py evaluation/visionlab/train.py train --data mvtec --model dfkde 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\05-native-mvtec-261026-1.log
& $py evaluation/visionlab/predict.py --model dfkde --data mvtec 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\05-pure-mvtec-261026-1.log
& $py evaluation/visionlab/train.py train --data visa --model dfkde 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\05-native-visa-261026-1.log
& $py evaluation/visionlab/predict.py --model dfkde --data visa 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\05-pure-visa-261026-1.log
& $py evaluation/anomalib/compare_outputs.py --model dfkde --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\Dfkde\LocalMVTecAD\bottle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\05-compare-outputs-mvtec-261026-1.log
& $py evaluation/anomalib/compare_outputs.py --model dfkde --samples 8 --reference-checkpoint D:\projects\visionlab\outputs\anomalib\Dfkde\LocalVisa\candle\v0\weights\lightning\model.ckpt 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\05-compare-outputs-visa-261026-1.log
& $py evaluation/anomalib/compare_dfkde_stats.py --reference D:\projects\visionlab\outputs\anomalib\Dfkde\LocalMVTecAD\bottle\v0\weights\lightning\model.ckpt --native D:\projects\visionlab\outputs\visionlab\dfkde\mvtec_bottle\v0\weights\lightning\model.ckpt --out D:\projects\visionlab\outputs\compare\dfkde_stats_mvtec.json 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\05-stats-mvtec-261026-1.log
& $py evaluation/anomalib/compare_dfkde_stats.py --reference D:\projects\visionlab\outputs\anomalib\Dfkde\LocalVisa\candle\v0\weights\lightning\model.ckpt --native D:\projects\visionlab\outputs\visionlab\dfkde\visa_candle\v0\weights\lightning\model.ckpt --out D:\projects\visionlab\outputs\compare\dfkde_stats_visa.json 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\05-stats-visa-261026-1.log
```

레퍼런스 실행은 2.1절이다. agy는 파일을 고치지 않았다.

### 5.3 PCA·KDE 통계 비교 (정보, 판정 아님)

다음 표는 `compare_dfkde_stats.py`의 출력이다(`05-stats-{mvtec,visa}-261026-1.log`, `outputs/compare/dfkde_stats_{mvtec,visa}.json`).

| 데이터 | 성분 수 (레퍼런스 / native) | `pca_model.mean` | `max_length` | `kde_model.norm` | `singular_vectors` 부분공간 거리 | `kde_model.dataset` shape |
|---|---|---:|---:|---:|---:|---|
| mvtec / bottle | 16 / 16 | 최대 차이 0.0 | 0.0 | 0.0 | 3.6e-2 | [209, 16] 일치 |
| visa / candle | 16 / 16 | 최대 차이 0.0 | 0.0 | 0.0 | 0.0 | [900, 16] 일치 |

`bw_transform`과 `dataset`은 부호에 의존하므로 shape만 비교했다. mvtec의 부분공간 거리 3.6e-2는 다른 값(평균, 최대 길이, `norm`, 점수)이 모두 일치하는 것과 어긋나 보이지만 원인을 더 분해하지 않았다. 점수와 AUROC에는 영향이 없었다.

## 6. 결론

- **레퍼런스:** 두 데이터셋 모두 확보했다.
- **Image AUROC:** mvtec/bottle 0.938596, visa/candle 0.834167로 레퍼런스, native, 순수 PyTorch가 소수 여덟째 자리까지 같다. 통과. 정규화 점수 기준 값도 같다(D-24).
- **FR-5, FR-6, FR-23, FR-24:** 통과(5.2절). `pred_score` 최대 차이 0.0, `anomaly_map`은 두 쪽 모두 None이다.
- **테스트:** `tests\05_dfkde` 20개, `test_cli.py -k dfkde` 1개, `test_dependency.py` 6개가 통과했다.
- **수정 요청:** 5-0 0회, 5-A 0회, 5-B 0회.
- **앞 모델 회귀:** 보류(사용자 결정 2026-10-26). 돌리지 않았다. 영향받을 수 있는 공유 파일은 6.2절에 적었다.
- **D-24 범위:** export 한정으로 확정(사용자 결정 2026-10-26). 정규화 AUROC는 export 기반 `predict.py`에서 raw와 함께 기록하고 native `train.py` 로그는 raw만 기록한다.
- **Phase 종료:** 사용자가 종료를 결정했다(2026-10-26).

### 6.1 `specs/` 갱신이 필요한 항목

`specs/`는 사람이 쓴다. Claude는 고치지 않고 다음을 알린다.

- **FR-4 문구:** 사용자 결정으로 `specs/SPEC.md`의 FR-4에 image 전용 예외를 이미 반영했다(미커밋).
- **D-24 문구:** D-24는 "visionlab도 같은 export의 정규화 점수로 AUROC를 계산해 비교한다"고 적었고, 이번 Phase에서 `predict.py`(export 기반 순수 PyTorch 평가)에 구현했다. native `train.py`의 test 로그는 raw 순위 보존 AUROC만 기록한다. 사용자가 export 한정으로 확정했고(2026-10-26), `specs/DECISIONS.md` D-24와 `specs/SPEC.md` FR-18에 이 범위를 반영했다(Phase 5 커밋에 포함).
- **D-8 문구:** image 전용 모델의 Evaluator는 원본과 달리 val `image_AUROC`를 추가했다. D-8의 validation raw AUROC를 따른 것이며, 사용자 결정(2026-10-26)에 따라 원본과의 차이로 이 문서에만 기록하고 `specs/`는 고치지 않았다.

### 6.2 남은 위험과 사용자 판단 대상

- 레퍼런스가 1회뿐이라 run 간 편차는 측정하지 못했다.
- KDE 점수의 float32 포화(`1e-300`이 효과 없음)가 실제로 일어나는지는 직접 측정하지 않았다. raw와 정규화 AUROC가 같아서 이 데이터에서는 판정에 영향이 없었다.
- `max_training_points` 초과 시 `random.sample` 경로는 학습 점수(209장, 900장)가 40000보다 적어 실행되지 않았다. 다중 layer 경로(`cat dim 0`)는 단위 테스트로만 확인했다.
- mvtec의 `singular_vectors` 부분공간 거리 3.6e-2는 원인을 분해하지 않았다(5.3절).
- `tests\common\test_dependency.py` 전체 실행은 사용자가 사후에 승인했다(2026-10-26, 5.2절).
- 앞 모델 회귀는 보류했다(사용자 결정 2026-10-26). `tests\common`의 나머지, 앞 모델의 pytest와 `compare_outputs`는 돌리지 않았고, 아래 공유 파일 변경이 앞 모델에 미치는 영향은 실행으로 확인하지 않았다. 변경은 모두 기존 항목을 건드리지 않는 추가다. 다음 표는 영향받을 수 있는 공유 파일이다.

| 공유 파일 | 변경 | 영향받을 수 있는 앞 모델 |
|---|---|---|
| `evaluation/visionlab/predict.py` | 지표 두 행(`*_AUROC_normalized`) 추가, `--model dfkde` | 모든 모델의 순수 PyTorch 평가 출력 형식(기존 지표의 이름·순서·값은 그대로) |
| `src/visionlab/anomaly_detection/models/components/stats/__init__.py` | `GaussianKDE` 노출 추가 | padim(`MultiVariateGaussian` import) |
| `src/visionlab/utils/io.py` | `load_export`가 허용하는 이름에 `dfkde` 추가 | 모든 모델의 export 로딩 |
| `src/visionlab/inference/predictor.py` | `from_export` 모델 dict에 `dfkde`, `feature_scaling_method` 복원 분기 | 모든 모델의 `Predictor.from_export` |
| `evaluation/anomalib/compare_outputs.py` | `--model dfkde` 분기 | stfpm, efficient_ad, padim, dfm 비교 |
| `tests/common/test_cli.py`, `tests/common/test_dependency.py` | dfkde 테스트와 분류 추가 | 같은 파일의 앞 모델 테스트 |
