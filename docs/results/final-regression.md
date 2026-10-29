# 전체 회귀 01~18 (pytest)

Phase 18 커밋(`ddee01d`) 뒤에 사용자가 승인한 전체 회귀 실행(2026-10-28)의 기록이다. 학습, 평가, `compare_outputs`는 돌리지 않고 `tests\` 아래 폴더별 pytest만 실행했다. 폴더 순서대로 agy가 한 명령씩 실행했다(GPU와 메모리 경합 방지). 결과 문서 `docs/results/NN-*.md`와 달리 특정 Phase의 결론이 아니라 전 Phase의 테스트 상태를 적는다.

## 실행 명령

다음은 agy에 보낸 명령이다. 첫 승인 창은 `cd`, `$py`, 첫 명령이 한 창에 합쳐져 main이 지시한 명령과 일치함을 직접 읽고 한 번 승인했다. 나머지 명령은 승인 창 없이 실행되어 교차 확인할 수 없다. agy는 시킨 명령 외에 로그 요약용 `powershell -Command`를 스스로 실행했다(읽기 전용으로 보이며 이 문서의 수치는 로그 파일에서 main이 직접 읽었다).

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py -m pytest tests\00_dataset 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-00_dataset-261028-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\01_stfpm 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-01_stfpm-261028-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\02_efficient_ad 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-02_efficient_ad-261028-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\03_padim 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-03_padim-261028-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\04_dfm 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-04_dfm-261028-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\05_dfkde 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-05_dfkde-261028-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\06_patchcore 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-06_patchcore-261028-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\07_reverse_distillation 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-07_reverse_distillation-261028-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\08_cfa 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-08_cfa-261028-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\09_fre 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-09_fre-261028-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\10_supersimplenet 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-10_supersimplenet-261028-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\11_draem 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-11_draem-261028-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\12_fastflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-12_fastflow-261028-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\13_cflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-13_cflow-261028-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\14_uflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-14_uflow-261028-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\15_csflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-15_csflow-261028-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\16_ganomaly 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-16_ganomaly-261028-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\17_dsr 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-17_dsr-261028-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\18_winclip 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-18_winclip-261028-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\common 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-common-261028-1.log; "exit=$LASTEXITCODE"
```

첫 시도에서 main이 지시 파일을 잘못 지정해 Phase 2 때의 오래된 명령 목록(native 학습 실행 포함)을 agy에 보냈다. 승인 창에서 첫 명령이 의도와 다름을 확인하고 승인하지 않고 취소했으며, 아무 명령도 실행되지 않았다(해당 로그 파일이 생성되지 않았음을 확인). 이후 올바른 지시 파일로 다시 보냈다.

## 결과

다음은 폴더별 결과다. 소요 시간은 pytest가 보고한 초 단위 값이다. `00_dataset`은 Phase 5 이후 처음 다시 실행한 것이다.

| 폴더 | passed | failed | 소요(초) | 로그 |
|---|---|---|---|---|
| `tests\00_dataset` | 10 | 0 | 5.2 | `final-regress-00_dataset-261028-1.log` |
| `tests\01_stfpm` | 32 | 0 | 68.4 | `final-regress-01_stfpm-261028-1.log` |
| `tests\02_efficient_ad` | 22 | 0 | 44.8 | `final-regress-02_efficient_ad-261028-1.log` |
| `tests\03_padim` | 16 | 0 | 32.3 | `final-regress-03_padim-261028-1.log` |
| `tests\04_dfm` | 22 | 0 | 32.5 | `final-regress-04_dfm-261028-1.log` |
| `tests\05_dfkde` | 20 | 0 | 32.2 | `final-regress-05_dfkde-261028-1.log` |
| `tests\06_patchcore` | 24 | 0 | 34.6 | `final-regress-06_patchcore-261028-1.log` |
| `tests\07_reverse_distillation` | 25 | 0 | 37.2 | `final-regress-07_reverse_distillation-261028-1.log` |
| `tests\08_cfa` | 35 | 0 | 34.6 | `final-regress-08_cfa-261028-1.log` |
| `tests\09_fre` | 16 | 0 | 32.5 | `final-regress-09_fre-261028-1.log` |
| `tests\10_supersimplenet` | 27 | 0 | 39.3 | `final-regress-10_supersimplenet-261028-1.log` |
| `tests\11_draem` | 39 | 0 | 55.4 | `final-regress-11_draem-261028-1.log` |
| `tests\12_fastflow` | 25 | 0 | 51.2 | `final-regress-12_fastflow-261028-1.log` |
| `tests\13_cflow` | 29 | 0 | 54.5 | `final-regress-13_cflow-261028-1.log` |
| `tests\14_uflow` | 25 | 0 | 51.4 | `final-regress-14_uflow-261028-1.log` |
| `tests\15_csflow` | 28 | 2 | 73.1 | `final-regress-15_csflow-261028-1.log` |
| `tests\16_ganomaly` | 30 | 0 | 25.7 | `final-regress-16_ganomaly-261028-1.log` |
| `tests\17_dsr` | 26 | 0 | 44.3 | `final-regress-17_dsr-261028-1.log` |
| `tests\18_winclip` | 31 | 0 | 39.8 | `final-regress-18_winclip-261028-1.log` |
| `tests\common` | 67 | 0 | 83.1 | `final-regress-common-261028-1.log` |
| 합계 | 549 | 2 | 872.0 | |

실패는 다음 항목이다.

- `tests/15_csflow/test_csflow_lightning.py::test_csflow_native_checkpoint_and_frozen_evaluation`
- `tests/15_csflow/test_csflow_lightning.py::test_csflow_native_isolated_export`

## 실패 분석

`tests\15_csflow`의 두 실패는 모두 `AssertionError: Tensor-likes are not close!`이며 `Greatest absolute difference: nan`이다. 첫째(`test_csflow_native_checkpoint_and_frozen_evaluation`)는 학습한 모델과 체크포인트에서 복원한 모델의 `predict_step` 출력 비교에서, 둘째(`test_csflow_native_isolated_export`)는 export를 읽은 자식 프로세스의 출력 비교에서 난다. 두 비교 모두 한쪽 또는 양쪽 출력이 NaN일 때 `assert_close`가 실패한다. 같은 폴더는 Phase 15에서 30 passed(72초)였고 이번에는 28 passed, 2 failed(73초)다. 원인은 확정하지 못했고 후보는 다음과 같다. 수정은 하지 않았다.

- **시드 없는 학습 fixture(첫 실행 때의 유력 후보, 재실행 뒤 약해짐):** `tests\15_csflow\test_csflow_lightning.py`의 `fitted_csflow`는 `make_csflow_model()`로 모델을 만들고 2 epoch 학습하는데, 이 fixture 앞에 `torch.manual_seed`가 없다. 시드 설정(`torch.manual_seed(23)`)은 `csflow_assets`의 임의 EfficientNet-B5 가중치 저장에만 `fork_rng` 안에서 쓰인다(코드로 확인). 전역 난수 상태는 프로세스 시작마다 달라 flow 초기화와 학습 경로가 실행마다 다르고, 학습이 NaN으로 발산하면 두 비교가 모두 실패한다. 임의 초기화한 EfficientNet-B5 특징으로 학습하는 설정이라 발산이 가능하다. 이 설명이 맞다면 Phase 15의 30 passed는 우연한 통과였고 이번 실패는 코드 회귀가 아니라 테스트의 비결정성이다. 확정하지는 못했다.
- **Phase 15 이후 공유 코드 변경:** Phase 15 커밋 이후 `src` 중 csflow와 관련 있을 수 있는 변경은 `components/base/__init__.py`, `trainers/lightning_cli.py`, `inference/predictor.py`, `utils/io.py`의 추가 수정뿐이다. 모두 winclip 분기 추가이고 csflow 코드 경로를 바꾸지 않는다. 첫째 테스트는 Predictor를 쓰지 않는다.
- **실행 순서와 환경:** 이번에는 앞선 폴더(00~14)의 테스트 뒤에 순서대로 실행되었다. 폴더 단독 실행과 달리 GPU 상태나 파일 캐시가 달랐을 수 있다.

확인하려면 같은 폴더를 한 번 더 실행해 재현성을 보는 것이 가장 싸다. 이 재실행은 사용자가 승인했고 아래 재실행 절에 기록했다.

## 재실행 (15_csflow)

사용자 승인(2026-10-28)으로 `tests\15_csflow`만 한 번 더 실행했다. 코드와 테스트는 수정하지 않았다. 명령은 다음과 같다.

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py -m pytest tests\15_csflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-15_csflow-261028-2.log; "exit=$LASTEXITCODE"
```

agy가 처음에는 이 명령을 `powershell -ExecutionPolicy Bypass -Command "..."`로 감싸고 `"exit=$LASTEXITCODE"`를 `Write-Host`로 바꿔 실행하려 해서, 지시한 명령과 정확히 일치하지 않아 승인하지 않고 취소한 뒤 "감싸지 말고 줄 그대로 실행"을 명시해 다시 보냈다. 두 번째 시도는 승인 창 없이 실행되어 교차 확인할 수 없다.

결과는 28 passed, 2 failed(73.7초)로 첫 실행과 같은 두 항목이 같은 방식으로 실패했다(재현). 로그는 `final-regress-15_csflow-261028-2.log`다.

- `tests/15_csflow/test_csflow_lightning.py::test_csflow_native_checkpoint_and_frozen_evaluation`: `Mismatched elements: 1 / 1`, `Greatest absolute difference: nan`. `OUTPUT_NAMES`의 첫 항목은 `pred_score`이고 batch가 1개(`pred_score.shape == (1,)`를 통과한 뒤 비교)라 이 값이 NaN이다.
- `tests/15_csflow/test_csflow_lightning.py::test_csflow_native_isolated_export`: `Mismatched elements: 2 / 2`, 같은 NaN. 자식 프로세스의 export 읽기 경로다.

다음은 읽기 전용으로 좁힌 사실이다. 코드는 실행하지 않았고 로그와 소스만 읽었다.

| 확인 | 결과 |
|---|---|
| 재현 여부 | 2회 실행에서 같은 2건이 같은 오류로 실패했다. 간헐 발산이라면 두 번 연속 같은 결과일 확률이 낮아, 시드 없는 fixture가 원인이라는 앞의 가설은 약해졌다(배제는 못 함) |
| `fitted_csflow`의 시드 | `make_csflow_model()`과 `trainer.fit` 앞에 `torch.manual_seed`가 없다. 데이터는 `torch.Generator().manual_seed(17)`로 고정이다. `manual_seed(23)`은 `csflow_assets`의 임의 EfficientNet-B5 가중치 저장(`fork_rng` 안)에만 있다. 따라서 flow 가중치 초기화는 프로세스마다 다르다 |
| 학습 중 NaN 여부 | 학습은 NaN이 아니다. `fitted_csflow`의 `on_before_optimizer_step`이 매 step `torch.isfinite(norm)`을 단언하고, 같은 실행의 `test_csflow_native_learning_epoch_mean_and_clipping`(epoch 평균 손실 `approx` 비교)이 통과했다. NaN은 평가 모드 출력(`pred_score`)에서 처음 나타난다 |
| 실패 위치 | 첫째는 학습한 모델과 체크포인트 복원 모델의 `predict_step` 비교(`test_csflow_lightning.py` 272행)에서, 둘째는 export를 읽은 자식 프로세스의 비교(341행)에서 난다. 두 경로가 모두 NaN이면 비교가 실패한다. 어느 쪽이 NaN인지는 로그만으로 알 수 없다 |
| 코드 변경 이력 | Phase 15 커밋(`4627f7a`) 이후 `models/csflow`, `models/components`(`base/buffer_list.py` 추가만), `common`, `tests/15_csflow`, `tests/12_fastflow`, `tests/conftest.py`의 변경은 없다(`git diff`로 확인). Phase 15 당시 같은 폴더는 30 passed였다 |

NaN이 처음 생기는 연산(예: 임의 초기화한 EfficientNet-B5의 평가 모드 특징, flow의 `exp` 또는 로그 계산, 점수 집계)은 코드를 실행하지 않고는 특정하지 못했다. 이를 특정하려면 평가 모드 중간 텐서의 유한성을 출력하는 일회성 진단 실행이 필요하며, 이는 사용자 결정을 기다린다. 수정(예: fixture에 시드 고정)도 사용자 결정이다.

## csflow 진단

사용자 승인(2026-10-29)으로 `tests\15_csflow`의 NaN 실패 원인을 읽기 전용으로 진단했다. `src/`, `tests/`, `specs/`는 수정하지 않았다. 진단은 pytest 플러그인(`diag_csflow_plugin.py`)이 `fitted_csflow` fixture의 결과를 바깥에서 읽는 방식이며(forward hook과 상태 복사만 사용, 소스와 테스트 무수정), 실행기(`diag_csflow_run.py`)가 같은 폴더를 pytest로 돌린다. 두 스크립트는 scratchpad에 있고 저장소에는 두지 않았다. 산출 JSON은 `outputs/diag/csflow_diag_<태그>.json`이다(`outputs/`는 커밋 대상이 아니다).

### 결론

NaN은 모델이나 학습에서 생기지 않는다. 최초 NaN 지점은 `PostProcessor._normalize`(`common/post_processor.py`)의 `(prediction - threshold) / (maximum - minimum) + 0.5`이고, fixture가 만든 모델이 검증 이미지 3장에 정확히 같은 `pred_score`를 내서 `image_min == image_max`이면 분모가 0, 분자도 0이라 `0 / 0 = NaN`이 된다. 세 점수가 float32 눈금 한 칸(ulp) 이상 다르면 `normalized_pred_score`가 유한해 테스트가 통과한다. 따라서 이 실패는 BatchNorm, 환경, 코드 회귀가 아니라 퇴화한 테스트 fixture의 점수 동점에 달린 시드·부동소수점 의존이다.

### 근거

다음은 진단 순서별 결과다.

| 단계 | 확인 | 결과 |
|---|---|---|
| 1 | 학습 직후 파라미터와 buffer의 NaN/inf | 없다(`nonfinite_state_entries` 비어 있음). 모든 실행에서 같다 |
| 1 | BatchNorm buffer | 116개 모듈 모두 `running_var` 1.0, `running_mean` 절댓값 최대 0.0, `num_batches_tracked` 0, train 모드인 BatchNorm 0개. backbone이 학습 동안 `eval()`로 고정되어 오염도 갱신도 없다. 따라서 Phase 15의 BatchNorm dry-run 오염과는 다른 현상이다 |
| 1 | 학습 모드 forward | `z_dist`, `jacobians` 모두 유한(`z_dist` 범위 약 -3e-4 ~ 2.5e-4). 학습 손실도 유한하며 값은 1e-6 규모다 |
| 2 | 평가 모드 forward의 모듈별 출력(hook) | 모든 모듈의 출력이 유한하다. `first_nonfinite_outputs`가 모든 batch에서 비어 있고 `anomaly_map`도 유한(약 1e-25 규모)이다 |
| 2 | 후처리 | raw `pred_score`는 유한하다(약 2e-9). `PostProcessor.__call__`의 유한성 검사를 통과한 뒤 `_normalize`에서 `normalized_pred_score`가 NaN이 된다. 실패한 실행에서는 `max_minus_min` = 0.0, `score_minus_threshold` = 0.0이다 |
| 3 | 시드 영향 | 아래 표 |
| 4 | 환경 | 아래 절 |

fixture가 퇴화한 이유는 코드로 확인한 사실과 추론으로 나뉜다. 확인한 사실은 `csflow_assets`가 `efficientnet_b5(weights=None)`의 임의 초기화 가중치를 backbone에 쓰고, 그 특징으로 학습한 결과 `z_dist`가 1e-4 규모로 작으며 검증 이미지 사이 점수 차이가 float32 해상도(약 1e-9의 상대 1e-7)에 가깝다는 것이다. 임의 초기화 backbone의 특징이 입력에 거의 무관해진다는 점은 추론이며 특징 텐서를 직접 비교해 확인하지 않았다.

### 시드와 반복 결과

다음은 진단 중 검증 이미지 3장(batch 2와 1)의 raw `pred_score` 동점 여부와 테스트 결과다. 시드 지정은 `fitted_csflow` 시작 직전에 `torch.manual_seed`를 호출하는 방식이고, `none`은 시드 지정 없이 폴더 전체를 돌린 것, `-k`는 `frozen_evaluation` 테스트만 돌린 것이다.

| 실행 | 시드 | 3장의 raw `pred_score` | `image_min == image_max` | `normalized_pred_score` | 결과 |
|---|---|---|---|---|---|
| n1 (전체) | 없음 | 모두 같다 | 예 | NaN | 2 failed |
| n2 (전체) | 없음 | 마지막 하나만 1 ulp 다르다 | 아니오 | 0.5, 0.5, 1.0 | 30 passed |
| s0 (-k) | 0 | 모두 같다 | 예 | NaN | failed |
| s1 (-k) | 1 | 첫 번째만 1 ulp 다르다 | 아니오 | 1.0, 0.5, 0.5 | passed |
| s2 (-k) | 2 | 모두 같다 | 예 | NaN | failed |
| s3 (-k) | 3 | 마지막 하나만 1 ulp 다르다 | 아니오 | 0.5, 0.5, 0.0 | passed |
| s4 (-k) | 4 | 마지막 하나만 1 ulp 다르다 | 아니오 | 0.5, 0.5, 1.0 | passed |

시드 0, 1, 2의 결과(실패, 통과, 실패)는 첫 번째 진단 실행(플러그인 오류로 JSON은 없지만 pytest 결과는 유효)과 두 번째에서 같았다. 시드를 고정하면 같은 결과가 재현된다. 시드 없는 폴더 전체 실행은 지금까지 5회 중 3회 실패했다(전체 회귀, 재실행, n1이 실패이고 첫 진단 실행의 `none`과 n2가 통과). 시드 없는 `-k` 단독 실행(`kone`) 1회는 통과했다. 원래 실패가 두 번 연속 나온 것은 이 비율(약 절반)과 모순되지 않는다.

### 환경 (4단계)

진단 실행 시점의 버전은 다음과 같다(JSON의 `versions`와 `site-packages`). Python 3.11.8, torch 2.5.1+cu121, torchvision 0.20.1+cu121, numpy 2.2.5, kornia 0.8.2, FrEIA 0.2(`freia-0.2.dist-info`), Lightning 2.5.5, torchmetrics 1.8.2, scipy 1.15.3, CPU 스레드 6. `specs/DECISIONS.md` D-14가 정한 고정 버전과 같다. `docs/results/15-csflow.md`와 Phase 15 로그에는 패키지 버전 목록이 따로 없어 Phase 15 시점의 실제 버전을 로그에서 대조하지는 못했다. D-14는 버전을 바꾸지 않는다고 하고 그 뒤 설치 변경 기록이 없으므로 환경 차이로 보이지 않는다. 앞 폴더 뒤에 순서대로 돌린 영향은 단독 실행(`-k`, 폴더 단독)에서도 같은 실패가 나므로 원인이 아니다.

### 수정 제안 (적용하지 않음)

다음은 근거와 함께 한 제안이다. 실제 수정은 사용자 결정을 기다린다.

| 후보 | 내용 | 장점 | 단점 |
|---|---|---|---|
| A (테스트, 권장) | `test_csflow_lightning.py`의 `OUTPUT_NAMES` 비교(`assert_close`) 세 곳에 `equal_nan=True`를 주고, 점수 동점일 때 `normalized_pred_score`가 NaN이어도 복원·export 경로가 학습 모델과 같은 값을 내는지를 비교하게 한다. 같은 파일 303행의 후처리 상태 비교는 이미 `equal_nan=True`다 | 테스트의 목적(복원과 export가 원본과 같은 출력)에 맞고 이식 코드와 공유 코드를 건드리지 않는다 | NaN 동점 케이스를 "같다"로 인정하게 된다 |
| B (테스트) | `fitted_csflow` 시작에 `torch.manual_seed`를 넣고 점수가 구분되는 시드(예: 1, 3, 4)를 고른다 | 간단하다 | 구분 여부가 부동소수점 한 칸에 달려 플랫폼과 버전이 바뀌면 다시 깨질 수 있다 |
| C (공유 코드) | `PostProcessor._normalize`에서 `maximum == minimum`일 때 NaN 대신 0.5를 반환한다 | 퇴화한 점수에서도 유한한 정규화 값을 준다 | 공유 코드 변경이라 모든 모델에 영향이 있고 D-8의 정규화 정의와 anomalib 원본(같은 나눗셈)과도 다르다. 별도 결정과 전 모델 회귀가 필요하다 |

권장은 A다. B는 A와 함께 쓸 수 있다. 수정은 `tests/`이므로 codex에 요청하고, 영향은 `tests\15_csflow`의 재실행(시드 없이 여러 번)으로 확인한다.

### 진단 명령

첫 번째 묶음(플러그인의 `deepcopy` 오류로 JSON은 만들지 못했으나 pytest 결과는 유효, 로그 `csflow-diag-<태그>-261029-1.log`)과 두 번째 묶음(수정 후, `-2.log`)을 agy에 보냈다. 두 묶음 모두 첫 승인 창은 `cd`, `$py`, 첫 명령이 한 창에 합쳐져 main이 첫 줄들이 보낸 것과 같음을 직접 읽고 한 번 승인했고, 나머지 줄은 창 없이 실행되어 교차 확인할 수 없다.

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py C:\Users\nampl\AppData\Local\Temp\claude\D--projects-visionlab\7fe17b22-8a30-4269-a4a9-6c6b304d9d6a\scratchpad\diag_csflow_run.py n1 none 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\csflow-diag-n1-261029-2.log; "exit=$LASTEXITCODE"
& $py C:\Users\nampl\AppData\Local\Temp\claude\D--projects-visionlab\7fe17b22-8a30-4269-a4a9-6c6b304d9d6a\scratchpad\diag_csflow_run.py n2 none 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\csflow-diag-n2-261029-2.log; "exit=$LASTEXITCODE"
& $py C:\Users\nampl\AppData\Local\Temp\claude\D--projects-visionlab\7fe17b22-8a30-4269-a4a9-6c6b304d9d6a\scratchpad\diag_csflow_run.py s0 0 -k frozen_evaluation 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\csflow-diag-s0-261029-2.log; "exit=$LASTEXITCODE"
& $py C:\Users\nampl\AppData\Local\Temp\claude\D--projects-visionlab\7fe17b22-8a30-4269-a4a9-6c6b304d9d6a\scratchpad\diag_csflow_run.py s1 1 -k frozen_evaluation 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\csflow-diag-s1-261029-2.log; "exit=$LASTEXITCODE"
& $py C:\Users\nampl\AppData\Local\Temp\claude\D--projects-visionlab\7fe17b22-8a30-4269-a4a9-6c6b304d9d6a\scratchpad\diag_csflow_run.py s2 2 -k frozen_evaluation 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\csflow-diag-s2-261029-2.log; "exit=$LASTEXITCODE"
& $py C:\Users\nampl\AppData\Local\Temp\claude\D--projects-visionlab\7fe17b22-8a30-4269-a4a9-6c6b304d9d6a\scratchpad\diag_csflow_run.py s3 3 -k frozen_evaluation 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\csflow-diag-s3-261029-2.log; "exit=$LASTEXITCODE"
& $py C:\Users\nampl\AppData\Local\Temp\claude\D--projects-visionlab\7fe17b22-8a30-4269-a4a9-6c6b304d9d6a\scratchpad\diag_csflow_run.py s4 4 -k frozen_evaluation 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\csflow-diag-s4-261029-2.log; "exit=$LASTEXITCODE"
```

첫 묶음은 같은 형식에서 태그와 시드만 `none none`, `kone none -k frozen_evaluation`, `s0 0 -k ...`, `s1 1 -k ...`, `s2 2 -k ...`였다.

### 확정하지 못한 것

- 임의 초기화 EfficientNet-B5가 입력에 무관한 특징을 내는지는 특징 텐서를 직접 비교하지 않아 추론으로만 적었다.
- Phase 15 시점의 패키지 버전은 기록이 없어 대조하지 못했다. D-14의 고정 버전과 현재 버전이 같다는 것까지만 확인했다.
- Phase 15에서 이 테스트가 통과한 것은 위 확률로 설명되지만, 당시 실행의 점수 값은 남아 있지 않아 직접 확인하지는 못했다.

## csflow 테스트 수정과 재검증

사용자 승인(2026-10-29)으로 csflow 진단의 수정 후보 A(테스트 비교에 `equal_nan=True`)를 적용했다. 수정은 codex가 했고 `tests\15_csflow\test_csflow_lightning.py` 한 파일만 바뀌었다(4줄 추가, 3줄 변경).

| 위치 | 변경 |
|---|---|
| 18행 | 영어 주석 한 줄 추가: 검증 점수가 모두 같으면 `PostProcessor` 정규화가 0/0이 되어 NaN이 같은 위치에 나온다는 설명과 이 문서를 가리키는 참조. 프로젝트 규칙(`AGENTS.md`: 주석은 영어)에 따라 지시받은 한국어 문구를 영어로 옮겼다 |
| 273행 | `test_csflow_native_checkpoint_and_frozen_evaluation`의 `OUTPUT_NAMES` 비교에 `equal_nan=True` |
| 302행 | 같은 테스트의 `observed["test"]` 비교에 `equal_nan=True` |
| 342행 | `consume_csflow_export`(export 읽기 자식 프로세스)의 `OUTPUT_NAMES` 비교에 `equal_nan=True` |

허용 오차(`rtol=1e-5`, `atol=1e-6`), fixture 시드, `("pred_score", "anomaly_map")` 비교와 303행(이미 `equal_nan=True`)은 바꾸지 않았다. `src/`와 다른 테스트 폴더는 수정하지 않았다.

### equal_nan 성질 확인

설치된 torch 소스를 읽어 확인했다. `torch.testing.assert_close(..., equal_nan=True)`는 `_compare_regular_values_close`(`torch/testing/_comparison.py` 1020행)에서 `torch.isclose(actual, expected, rtol, atol, equal_nan=True)`를 부르고 모든 원소가 일치해야 통과한다. `torch.isclose`는 같은 위치에서 양쪽이 모두 NaN이면 일치로 보고 한쪽만 NaN이면 불일치로 본다. 그래서 학습 모델, 체크포인트 복원 모델, export 읽기 경로의 NaN 위치가 다르면 여전히 실패한다. NaN이 아닌 원소는 같은 허용 오차로 비교한다.

### 완화의 한계

이 변경은 비교 조건을 완화한다. NaN 외의 불일치는 가려지지 않는다. 다만 비교 대상이 모두 같은 위치에서 NaN이면 통과하므로, 두 경로에서 똑같이 생기는 NaN(예: 두 경로 모두의 버그)은 이 세 곳에서 더 이상 잡히지 않는다. 완화된 항목 중 raw `pred_score`와 `anomaly_map`은 `PostProcessor.__call__`이 유한성을 검사해 NaN이면 `ValueError`를 내므로 NaN으로 통과할 수 없다. 남는 경우는 `normalized_pred_score`와 `normalized_anomaly_map`이 양쪽 모두 NaN인 때뿐이다. 앞 진단에서 이 NaN의 원인은 점수 동점으로 밝혀졌다.

### 재실행 결과

수정 뒤 `tests\15_csflow`만 3회 연속 실행했다. 시드 없는 fixture라 한 번의 통과로는 부족하다는 사용자 지시에 따른 것이다. 명령은 다음과 같다(agy 종료 코드는 각 로그 기준, 이번 묶음은 승인 창 없이 실행되어 명령 일치를 교차 확인할 수 없다).

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py -m pytest tests\15_csflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\csflow-equalnan-261029-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\15_csflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\csflow-equalnan-261029-2.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\15_csflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\csflow-equalnan-261029-3.log; "exit=$LASTEXITCODE"
```

| 회차 | 결과 | 소요(초) | 로그 |
|---|---|---|---|
| 1 | 30 passed | 76.2 | `csflow-equalnan-261029-1.log` |
| 2 | 30 passed | 74.1 | `csflow-equalnan-261029-2.log` |
| 3 | 30 passed | 73.8 | `csflow-equalnan-261029-3.log` |

세 회 모두 통과했다. 다만 이 세 회에서 검증 점수가 실제로 동점이 되어 `equal_nan`이 쓰였는지는 로그로 알 수 없다. 수정 전 시드 없는 전체 실행의 통과 비율(5회 중 2회)을 쓰면 수정 없이 세 번 연속 통과할 확률은 약 6%이므로 통과는 수정의 효과를 뒷받침하지만 결정적 증거는 아니다.

### 정정한 전체 합계

위 "결과" 표는 2026-10-28 전체 회귀의 실제 기록이므로 고치지 않았다. 수정 뒤 `15_csflow` 폴더는 30 passed(3회 모두)이므로, 다른 19개 폴더의 통과 수(2026-10-28 실행)와 합치면 다음과 같다. 전체 회귀를 다시 돌린 것이 아니라 csflow 폴더만 다시 실행해 합친 값이다.

| 항목 | passed | failed |
|---|---|---|
| 2026-10-28 전체 회귀 기록 | 549 | 2 |
| 정정 합계 (15_csflow를 30 passed로 대체) | 551 | 0 |

## 마감 전 재실행

프로젝트 마감 지시(2026-10-29)에 따라 전체 회귀를 한 번 더 실행했다. 앞의 정정 합계 551 passed가 폴더 재실행을 합친 값이었으므로, 이번에는 20개 폴더를 모두 새로 실행해 한 번에 측정한 값을 얻는 것이 목적이다. 학습, 평가, `compare_outputs`는 돌리지 않았고 pytest만 실행했다. 코드와 테스트는 수정하지 않았다(실행 시점 HEAD는 `ff1db68`이며 앞서 INTENT 문구, `LICENSE`, `NOTICE`, `.gitignore`만 바뀌었다).

### 재실행 명령

다음은 agy에 보낸 명령이다. 첫 승인 창은 `cd`, `$py`, 첫 명령이 한 창에 합쳐져 main이 지시한 명령과 일치함을 직접 읽고 한 번 승인했다(영구 허용은 선택하지 않았다). 나머지 명령은 승인 창 없이 실행되어 교차 확인할 수 없다. agy는 시킨 명령 외에 로그 확인용 명령을 추가로 한 번 실행했으나 승인 창이 뜨지 않았고 읽기 전용으로 보인다(이 문서의 수치는 로그 파일에서 main이 직접 읽었다).

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py -m pytest tests\00_dataset 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-00_dataset-261029-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\01_stfpm 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-01_stfpm-261029-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\02_efficient_ad 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-02_efficient_ad-261029-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\03_padim 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-03_padim-261029-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\04_dfm 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-04_dfm-261029-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\05_dfkde 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-05_dfkde-261029-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\06_patchcore 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-06_patchcore-261029-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\07_reverse_distillation 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-07_reverse_distillation-261029-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\08_cfa 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-08_cfa-261029-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\09_fre 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-09_fre-261029-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\10_supersimplenet 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-10_supersimplenet-261029-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\11_draem 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-11_draem-261029-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\12_fastflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-12_fastflow-261029-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\13_cflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-13_cflow-261029-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\14_uflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-14_uflow-261029-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\15_csflow 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-15_csflow-261029-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\16_ganomaly 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-16_ganomaly-261029-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\17_dsr 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-17_dsr-261029-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\18_winclip 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-18_winclip-261029-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\common 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\final-regress-common-261029-1.log; "exit=$LASTEXITCODE"
```

### 재실행 결과

다음은 폴더별 결과다. 소요 시간은 pytest가 보고한 초 단위 값이다. agy의 터미널 출력에서 `exit=0`이 확인되었고 모든 로그의 마지막 줄이 passed였다.

| 폴더 | passed | failed | 소요(초) | 로그 |
|---|---|---|---|---|
| `tests\00_dataset` | 10 | 0 | 5.94 | `final-regress-00_dataset-261029-1.log` |
| `tests\01_stfpm` | 32 | 0 | 66.9 | `final-regress-01_stfpm-261029-1.log` |
| `tests\02_efficient_ad` | 22 | 0 | 44.61 | `final-regress-02_efficient_ad-261029-1.log` |
| `tests\03_padim` | 16 | 0 | 32.26 | `final-regress-03_padim-261029-1.log` |
| `tests\04_dfm` | 22 | 0 | 32.48 | `final-regress-04_dfm-261029-1.log` |
| `tests\05_dfkde` | 20 | 0 | 32.41 | `final-regress-05_dfkde-261029-1.log` |
| `tests\06_patchcore` | 24 | 0 | 34.61 | `final-regress-06_patchcore-261029-1.log` |
| `tests\07_reverse_distillation` | 25 | 0 | 37.09 | `final-regress-07_reverse_distillation-261029-1.log` |
| `tests\08_cfa` | 35 | 0 | 34.54 | `final-regress-08_cfa-261029-1.log` |
| `tests\09_fre` | 16 | 0 | 32.45 | `final-regress-09_fre-261029-1.log` |
| `tests\10_supersimplenet` | 27 | 0 | 39.35 | `final-regress-10_supersimplenet-261029-1.log` |
| `tests\11_draem` | 39 | 0 | 54.81 | `final-regress-11_draem-261029-1.log` |
| `tests\12_fastflow` | 25 | 0 | 51.52 | `final-regress-12_fastflow-261029-1.log` |
| `tests\13_cflow` | 29 | 0 | 54.89 | `final-regress-13_cflow-261029-1.log` |
| `tests\14_uflow` | 25 | 0 | 51.89 | `final-regress-14_uflow-261029-1.log` |
| `tests\15_csflow` | 30 | 0 | 74.08 | `final-regress-15_csflow-261029-1.log` |
| `tests\16_ganomaly` | 30 | 0 | 25.76 | `final-regress-16_ganomaly-261029-1.log` |
| `tests\17_dsr` | 26 | 0 | 44.39 | `final-regress-17_dsr-261029-1.log` |
| `tests\18_winclip` | 31 | 0 | 42.14 | `final-regress-18_winclip-261029-1.log` |
| `tests\common` | 67 | 0 | 84.14 | `final-regress-common-261029-1.log` |
| 합계 | 551 | 0 | 876.3 | |

### 판정

실패 0이고 합계 551 passed가 실측과 일치한다. 앞의 정정 합계(549 passed에 `15_csflow` 폴더 재실행 30 passed를 합친 551)가 한 번에 측정한 값과 같음을 확인했다. `15_csflow`는 시드 없는 fixture 때문에 통과 여부가 실행마다 달라질 수 있었으나 이번 실행도 30 passed(74.08초)였다. 이 한 번의 통과가 fixture의 비결정성이 사라졌음을 뜻하지는 않는다(`equal_nan=True`로 NaN이 같은 위치일 때 통과하도록 완화한 상태다).
