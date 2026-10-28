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
