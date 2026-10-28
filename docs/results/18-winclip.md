# Phase 18. WinCLIP

모델 폴더 이름은 `winclip`이고 원본은 anomalib 2.0.0의 `anomalib\models\image\winclip\`(`C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\Lib\site-packages\anomalib\models\image\winclip\`)이다. 마지막 Phase다. 범위와 완료 조건은 `specs/PLAN.md`의 Phase 18과 "Phase 공통 양식"을 따른다. 판정 기준은 category별 Image·Pixel AUROC 차이 0.01 이내다.

이 문서는 Phase 18 시작 전에 사용자 결정과 계획을 저장하고(2026-10-28) 실행하면서 결과를 채운 기록이다.

## 상태

다음 표는 단계별 진행 상태다. 수정 반복 한도 3회는 단계별로 센다.

| 단계 | 내용 | 상태 |
|---|---|---|
| 사전 확인 | CLIP `.bin`이 `weights_only=True`로 읽히는 순수 state dict인지 읽기 전용 확인, `paths.toml [teachers]`에 `winclip_clip` 등록, `bpe_simple_vocab_16e6.txt.gz` 라이선스 확인 | 완료 (2.1절) |
| 레퍼런스 | `anomalib_run.py train` x 2 (mvtec/bottle 1회, visa/candle 1회, zero-shot이라 validate 뒤 test). 레퍼런스 yaml의 `pre_processor`를 `Resize(256)` 다음 `Resize(240, bicubic)`와 CLIP 정규화로 수정 | 완료 (2.2절, 첫 실행은 체크포인트 저장 오류로 실패해 어댑터 수정 뒤 재실행) |
| 18-A | 독립 CLIP (`VisionTransformer`, `TextTransformer`, BPE 토크나이저, 어휘 파일 복사) 와 `tests/18_winclip`의 CLIP 부분 | 구현 완료, `test_clip.py` 11 passed (3절) |
| 18-B | `WinClipModel`(window 점수, 프롬프트 앙상블), `WinClipLitModel`, configs, `train.py`의 zero-shot 분기, export·Predictor 등록, 비교 스크립트 | 구현·실행 완료 (3절, 5절) |
| 앞 모델 확인 | `train.py` 분기가 앞 모델에 영향이 없는지 STFPM 하나로 가벼운 확인(agy) | 완료 (5.4절) |
| 판정·기록 | 5절 판정, 최종 보고 | 완료 |
| specs 반영 | 최종 보고 뒤 사용자 결정 1, 2 (6.1절) | 완료 (2026-10-28) |
| 사후 회귀, Phase 종료 | `tests\common` 사후 실행(67 passed), 커밋·푸시 | 완료 (2026-10-28) |

`InferenceBatch`, `PostProcessor`, `Evaluator`, `LearningType`(ZERO_SHOT, FEW_SHOT는 이미 있음), `load_pretrained_teacher`, `DynamicBufferMixin`은 앞 Phase의 것을 재사용한다. `BufferListMixin`은 visionlab에 아직 없어 anomalib `components/base/buffer_list.py`에서 복사한다(D-18). 계획 초안은 이것도 이미 있다고 잘못 적었다.

## 사용자 결정 (2026-10-28)

다음은 Phase 시작 전에 사용자가 정한 항목이다. 계획 보고의 결정 항목 1~7을 아래와 같이 채택했다.

| 번호 | 항목 | 결정 |
|---|---|---|
| 1 | open_clip 처리 | 선택지 B(독립 구현). `VisionTransformer`, `TextTransformer`, BPE 토크나이저를 순수 torch로 새로 쓰고 모듈 이름을 open_clip과 같게 해 로컬 `.bin`을 `strict`로 불러온다. 설치된 open_clip은 비교용으로만 쓰고 `src/visionlab`에는 import를 남기지 않는다. 같은 가중치로 CPU에서 open_clip과 출력을 비교(FR-5)한다 |
| 2 | 토크나이저 | BPE 어휘 `bpe_simple_vocab_16e6.txt.gz`(1.3MB)를 저장소에 복사한다(라이선스 확인 내용을 결과 문서에 기록). `regex`는 토크나이저 한 곳에서 지연 import로 좁게 허용한다(D-25 방식). `ftfy`는 고정 ASCII 프롬프트라 생략한다. specs는 아직 고치지 않고 D 번호 초안을 결과 문서에 적어 최종 보고에서 묻는다 |
| 3 | 입력 크기 | 레퍼런스 yaml을 `Resize(256)` 다음 `Resize(240, bicubic)`와 CLIP 정규화로 맞춘다. native는 `WinClipLitModel` 안에서 bicubic resize와 CLIP 정규화를 하며 `anomaly_map`을 256으로 보간한다(`common/` 수정 없음). pixel AUROC가 0.01을 넘으면 멈추고 보고한다 |
| 4 | zero-shot 흐름 | `evaluation/visionlab/train.py`에 validate 뒤 test 분기를 추가하고 export를 validate 끝으로 옮긴다. export에 CLIP 가중치(약 0.8GB)를 포함한다. 추가로 앞 모델 하나(예: STFPM)로 `train.py train`이 그대로 동작하는지 가벼운 확인을 agy로 요청한다 |
| 5 | FR-4 예외와 D-15 트리 | `anomaly_map [B,H,W]`, `pred_score [B]` 예외와 D-15 트리 추가는 지금 specs를 고치지 않고 최종 보고에서 묻는다 |
| 6 | 시간과 category | 변경 없이 mvtec/bottle + visa/candle. 시간 관문은 두지 않되 예상 밖의 지연이 있으면 보고한다 |
| 7 | 마지막 Phase 종료 뒤 종합 정리 | (a) INTENT 문구와 D-23·D-25·D-26·D-27 예외의 정리 초안 문서는 작성하되 INTENT는 사용자 요청 시에만 고친다. (b) 전체 회귀 01~18을 Phase 18 종료 뒤에 실행한다(승인). (c) README 갱신은 Phase 18 종료 뒤 따로 정한다 |

보조 조건은 다음과 같다.

| 항목 | 내용 |
|---|---|
| `.bin` 사전 확인 | 18-A 전에 `.bin`이 `weights_only=True`로 읽히는 순수 state dict인지 읽기 전용으로 먼저 확인하고, 안 읽히면 멈추고 보고한다 |
| `paths.toml` | `[teachers]`에 `winclip_clip` 키 한 줄 추가를 승인한다 |
| 규모 확정 | 독립 구현 규모(약 500줄 추정)와 파라미터 수는 18-A에서 codex가 합산해 확정하고 이 문서에 기록한다 |
| FR-5 허용 오차 | `atol=1e-6, rtol=1e-5` |
| 에폭 히스토리 | 해당 없음(학습이 없다) |
| 앞 모델 회귀(01~17) | 보류. 마지막 사후 실행은 Phase 17 커밋 전 `tests\common` 63개다. 전체 회귀는 결정 7(b)대로 Phase 18 종료 뒤에 한다 |
| 진행 규칙 | 사용자 결정이 필요한 일은 멈추고 `w2:p2`로 보고한다. 판정·기록까지 마치면 커밋하지 않고 최종 보고한다. agy 승인은 지시한 명령과 정확히 일치할 때만 main이 직접 한 번씩 한다. codex는 새 세션으로 시작한다 |

### 최종 보고 뒤 사용자 결정 (2026-10-28)

다음은 최종 보고에 대한 사용자 답이다.

| 번호 | 항목 | 결정 |
|---|---|---|
| 1 | resize 위치 | 해석 승인. 사용자 결정 3의 "`WinClipLitModel` 안 resize"를 순수 `WinClipModel.encode_image` 맨 앞으로 옮긴 것을 승인했다. 근거는 `common/`을 고치지 않고 export한 Predictor도 같은 resize를 하게 하는 것이며, open_clip 대조 결과는 3.2절과 5.3절에 있다 |
| 2 | specs 반영 | 6.1절 초안대로 반영한다. `SPEC.md` FR-4 예외, `DECISIONS.md`의 D-15 트리, 새 D 번호(D-28), D-8 zero-shot 확장 문구다. INTENT와 PLAN은 고치지 않는다 |
| 3 | 사후 회귀 | 커밋 전에 agy로 `tests\common`을 1회 실행한다(승인). 실패가 나오면 멈추고 보고한다. 전체 회귀 01~18은 커밋 뒤 따로 실행한다 |
| 4 | Phase 종료 | 3이 통과하면 종료하고 `Phase 18: winclip ...` 커밋으로 푸시한다. 그 뒤 멈추고 전체 회귀, 종합 정리 초안 문서, README 갱신은 다음 지시를 기다린다 |

## 1. 모델 노트

### 1.1 원본과 생성자

다음 표는 원본 파일과 이식 위치의 대응 계획이다(D-15, D-18).

| 원본 | 이식 위치 | 비고 |
|---|---|---|
| `models/image/winclip/torch_model.py` (`WinClipModel`) | `src/visionlab/anomaly_detection/models/winclip/model.py` | `open_clip` import 제거, 독립 CLIP 사용, `InferenceBatch`와 buffer 믹스인 경로 변경 |
| `models/image/winclip/utils.py` (`cosine_similarity`, `class_scores`, `harmonic_aggregation`, `visual_association_score`, `make_masks`) | `.../models/winclip/utils.py` | 순수 torch, 변경 없음 예상 |
| `models/image/winclip/prompting.py` (`create_prompt_ensemble`) | `.../models/winclip/prompting.py` | 변경 없음 |
| open_clip의 `VisionTransformer`, `TextTransformer`, 토크나이저 | `.../models/winclip/clip.py`(가칭), `tokenizer.py`(가칭) | 독립 구현, 모듈 이름을 open_clip과 같게 함 |
| `models/image/winclip/lightning_model.py` (`WinClip`) | `.../models/winclip/lightning_model.py` (`WinClipLitModel`) | native `LightningModule`로 새로 작성 |
| `models/image/winclip/__init__.py` | `.../models/winclip/__init__.py` | `lightning_model`을 import하지 않도록 고침(D-18 예외) |

생성자는 다음과 같다. `WinClipLitModel`은 `WinClip`과 인자 이름·순서·기본값이 같아야 한다(FR-6, D-5).

| 클래스 | 시그니처 |
|---|---|
| `WinClip` | `class_name=None`, `k_shot=0`, `scales=(2, 3)`, `few_shot_source=None`, `pre_processor=True`, `post_processor=True`, `evaluator=True`, `visualizer=True` (8개) |
| `WinClipModel` | `class_name=None`, `reference_images=None`, `scales=(2, 3)`, `apply_transform=False` (4개) |

### 1.2 학습과 평가 동작

원본의 동작은 다음과 같다.

- **이미지 인코더:** `ViT-B-16-plus-240`(12층, 폭 896, patch 16, 입력 240, 격자 15x15, 임베딩 640). `output_tokens=True`로 패치 토큰을 얻고, `patch_dropout`에 forward hook을 걸어 특징 맵을 가로챈다.
- **텍스트 인코더:** 12층, 폭 640, 10 head, 문맥 77, 어휘 49408.
- **점수:** scale 2, 3의 sliding window(`make_masks`)마다 선택한 토큰을 `ln_pre`, `transformer`, `ln_post`, `_global_pool`, `proj`에 다시 통과시켜 window 임베딩을 얻고 조화평균(`harmonic_aggregation`)으로 모은다. 텍스트는 프롬프트 앙상블(상태 11개 x 템플릿 21개)의 평균 임베딩 둘(normal, anomalous)이다. 온도 0.07, softmax 뒤 anomalous 확률을 쓴다. 난수는 없다.
- **few-shot:** `k_shot > 0`이면 참조 이미지의 패치·window 임베딩과의 거리로 점수를 섞는다. 기본값은 0(zero-shot)이고 이번 Phase는 zero-shot만 실행한다.
- **학습 없음:** `trainer_arguments`는 `{}`, `configure_optimizers`는 `None`, `learning_type`은 `k_shot`이 0이면 ZERO_SHOT이다. anomalib `Engine.train`은 zero-shot이면 `fit` 대신 `validate`(정규화·임계값 통계)를 한 뒤 `test`를 한다. 클래스 이름은 `datamodule.category`("bottle", "candle")에서 얻는다.
- **전처리:** `Resize(240, bicubic, antialias)` 뒤 CLIP 정규화(mean 0.48145466, 0.4578275, 0.40821073 / std 0.26862954, 0.26130258, 0.27577711). 모델은 `apply_transform=False`로 자체 transform을 건너뛴다.
- **출력:** `pred_score [B]`, `anomaly_map [B,240,240]`(`squeeze(1)`)다.
- **state_dict:** 원본은 FITTING·VALIDATING 중 `model.clip`을 `state_dict`에서 빼고 `load_state_dict`에서 복원한다. `_text_embeddings`는 persistent buffer이고 `masks`는 비영속이다.

### 1.3 외부 의존과 로컬 자산

- **외부 의존:** 원본은 `open_clip`, `torchvision`(`ToPILImage`)을 쓴다. 독립 구현에서는 순수 torch이고 토크나이저만 `regex`를 지연 import한다. `open_clip`, `timm`, `torchvision`, `ftfy`는 `src/visionlab`에 import하지 않는다.
- **로컬 가중치:** `D:\backbones\vit_base_patch16_plus_clip_240.laion400m_e31\open_clip_pytorch_model.bin`(795MB). 레퍼런스는 `local_models.LocalWinClip`이 `PRETRAINED` 상수를 이 경로로 바꾼다. native는 `paths.toml [teachers]`의 `winclip_clip`으로 `strict` 주입한다(D-19).
- **토크나이저 어휘:** 설치된 open_clip 패키지 안의 `bpe_simple_vocab_16e6.txt.gz`(1.3MB)를 저장소에 복사한다. 라이선스 확인 결과는 사전 확인 단계에서 이 문서에 기록한다.
- **데이터:** MVTec `bottle`(val 33, test 50 추정), VisA `candle`(val 80, test 120 추정). 분할은 `from_test`, 0.4, seed 42, batch 8/8이다.

### 1.4 이식에서 의도적으로 달라지는 곳과 유지하는 곳

- **open_clip 대체:** 독립 구현으로 바꾼다. 모듈 이름과 가중치 키를 open_clip과 같게 하고 같은 가중치로 CPU에서 출력(이미지·window·텍스트 임베딩과 최종 점수)을 비교한다.
- **전처리와 평가 해상도:** 레퍼런스는 `Resize(256)` 다음 `Resize(240, bicubic)`와 CLIP 정규화(결정 3). native는 `WinClipLitModel`이 bicubic resize와 CLIP 정규화를 하고 `anomaly_map`을 256으로 bilinear 보간해 mask와 맞춘다.
- **state_dict 규칙:** 원본의 `model.clip` 제외 규칙은 이식하지 않고 export에 CLIP 가중치를 모두 넣는다(결정 4).
- **유지:** 프롬프트 문구, 온도, window 크기·조화평균, 점수 식, few-shot 코드(기본값 0만 실행).
- **FR-5 허용 오차:** `pred_score`와 `anomaly_map`은 `atol=1e-6, rtol=1e-5`다.

## 2. 레퍼런스

### 2.1 사전 확인

다음은 18-A 전에 한 읽기 전용 확인과 자산 준비다. 명령은 agy가 실행했다(2026-10-28). 첫 명령의 승인 창은 `cd`, `$py`, 첫 명령이 한 창에 합쳐져 있어 main이 직접 읽고 한 번 승인했다.

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py -c "import torch; s = torch.load(r'D:\backbones\vit_base_patch16_plus_clip_240.laion400m_e31\open_clip_pytorch_model.bin', map_location='cpu', weights_only=True); print(type(s).__name__, len(s)); ks = list(s.keys()); print(ks[:8]); print(ks[-8:]); print(sum(v.numel() for v in s.values() if isinstance(v, torch.Tensor)), {str(v.dtype) for v in s.values() if isinstance(v, torch.Tensor)}); print(all(isinstance(v, torch.Tensor) for v in s.values())); print(sorted({k.split('.')[0] for k in ks}))" 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\18-bin-check-261028-1.log; "exit=$LASTEXITCODE"
& $py -c "import regex; print(regex.__version__)" 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\18-regex-check-261028-1.log; "exit=$LASTEXITCODE"
```

두 명령의 종료 코드는 0이다. 승인 창은 첫 명령에서만 떴고 두 번째 명령은 창 없이 실행되어 교차 확인할 수 없다. 다음은 확인 결과다.

| 항목 | 결과 |
|---|---|
| `.bin` 형식 | `weights_only=True`로 읽히는 순수 `OrderedDict`, 키 302개, 모두 `torch.Tensor`, 모두 float32, 총 208,377,345개 원소 |
| 최상위 키 접두사 | `ln_final`, `logit_scale`, `positional_embedding`, `text_projection`, `token_embedding`, `transformer`, `visual` |
| 판정 | 멈춤 조건에 해당하지 않아 18-A로 진행했다 |
| `regex` | 2026.7.19, import 성공 |
| `paths.toml [teachers]` | `winclip_clip = "D:/backbones/vit_base_patch16_plus_clip_240.laion400m_e31/open_clip_pytorch_model.bin"` 한 줄 추가(사용자 승인, main이 직접 편집) |
| 어휘 파일 | 설치된 open_clip 2.24.0의 `bpe_simple_vocab_16e6.txt.gz`, 1,356,917바이트, SHA-256 `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a`. 복사본과 바이트가 같다(`test_clip_bpe_copy_bytes`) |
| 라이선스 확인 | open_clip 2.24.0 `dist-info/LICENSE`는 MIT 문구이고(저작권 Gabriel Ilharco 외), `open_clip/tokenizer.py` 머리말은 토크나이저가 OpenAI CLIP에서 복사되었고 원래 MIT 라이선스(Copyright (c) 2021 OpenAI)라고 적는다. 패키지 `METADATA`의 License 필드는 `UNKNOWN`이고 분류자는 Apache로 적혀 있어 서로 어긋나므로 LICENSE 파일과 tokenizer 머리말을 근거로 삼았다. `src/visionlab/anomaly_detection/models/winclip/LICENSE`에 두 저작권 표기와 MIT 문구를 어휘 파일 옆에 두었다 |

### 2.2 레퍼런스 실행

레퍼런스 어댑터는 `evaluation/anomalib/local_models.py`의 `LocalWinClip`이고, 모델 yaml은 `evaluation/anomalib/models/18_winclip.yaml`이다. main이 사용자 결정 3에 따라 yaml에 `pre_processor`(`Resize(256)` 다음 `Resize(240, bicubic, antialias)` 다음 CLIP 정규화)를 추가했다. 원본의 기본 전처리는 입력을 곧바로 240으로 줄이지만 이번 Phase는 visionlab Dataset 입력 크기(D-27)에 맞춘다. 이 yaml은 codex의 쓰기 범위(`evaluation/`)에 속하지만 사용자가 정한 입력 크기 변경이라 main이 직접 고쳤다.

첫 실행은 두 데이터 모두 `validate` 끝에서 실패했다. 지시한 명령은 다음과 같다. 종료 코드는 1이고, 로그는 `18-ref-mvtec-261028-1.log`, `18-ref-visa-261028-1.log`이며, run 폴더 `LocalWinClip/LocalMVTecAD/bottle/v0`와 `LocalWinClip/LocalVisa/candle/v0`는 비어 있어 채택하지 않는다.

```powershell
cd D:\projects\visionlab\evaluation\anomalib
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/18_winclip.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\18-ref-mvtec-261028-1.log; "exit=$LASTEXITCODE"
& $py anomalib_run.py train --config default.yaml --config data/visa.yaml --config models/18_winclip.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\18-ref-visa-261028-1.log; "exit=$LASTEXITCODE"
```

실패 원인은 다음과 같다. anomalib의 zero-shot `Engine.train`은 `validate`를 하고, anomalib의 `ModelCheckpoint` 서브클래스(`anomalib/callbacks/checkpoint.py`의 `_should_skip_saving_checkpoint`)는 fit이 아니어도 저장하므로 validate 끝에서 체크포인트를 저장한다. 이때 `WinClipModel.encode_image`가 호출마다 `clip.visual.patch_dropout`에 걸고 제거하지 않은 forward hook(지역 함수 `get_feature_map.<locals>.hook`)이 pickle되지 않아 `AttributeError: Can't pickle local object 'WinClipModel.encode_image.<locals>.get_feature_map.<locals>.hook'`가 났다. 점수 계산과는 무관한 저장 경로의 문제다.

수정 요청 1/3(종류: 레퍼런스 어댑터)으로 codex가 `LocalWinClip.on_save_checkpoint`에서 `self.model.clip.visual.patch_dropout._forward_hooks.clear()`를 한 뒤 `super().on_save_checkpoint(checkpoint)`를 부르게 했다. hook은 출력에 영향이 없는 기록용이라 비워도 이후 계산이 같다. 같은 pickle 문제가 `Engine.export(TORCH)`(`torch.save({"model": self})`)에도 남지만 이 Phase는 export를 쓰지 않는다. 두 번째 실행의 명령은 다음과 같다(18-A 테스트 두 개와 한 묶음으로 보냈다).

```powershell
cd D:\projects\visionlab\evaluation\anomalib
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/18_winclip.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\18-ref-mvtec-261028-2.log; "exit=$LASTEXITCODE"
& $py anomalib_run.py train --config default.yaml --config data/visa.yaml --config models/18_winclip.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\18-ref-visa-261028-2.log; "exit=$LASTEXITCODE"
```

두 번째 실행은 정상 종료했다(agy 보고 기준 종료 코드 0, 로그에 `Seed set to 42`, category를 데이터 모듈에서 얻음). validate는 mvtec 5 배치, test는 mvtec 7 배치와 visa 15 배치다. 채택한 run 폴더는 `outputs/anomalib/LocalWinClip/LocalMVTecAD/bottle/v1`, `outputs/anomalib/LocalWinClip/LocalVisa/candle/v1`이다. 학습이 없어 에폭 히스토리는 없다. 다음은 레퍼런스 test 결과다.

| 데이터 | category | Image AUROC | Pixel AUROC | Image F1 | Pixel F1 | test 시간 |
|---|---|---|---|---|---|---|
| mvtec | bottle | 0.98904 | 0.85437 | 0.95890 | 0.50472 | 25.1초 |
| visa | candle | 0.96528 | 0.87387 | 0.92308 | 0.06828 | 54.7초 |

수치는 `18-ref-mvtec-261028-2.log`, `18-ref-visa-261028-2.log`의 metric 표에서 읽었다. 이 AUROC는 D-24의 정의대로 레퍼런스의 정규화 점수 기준이다.

## 3. 이식 메모

### 3.1 18-A: 독립 CLIP과 토크나이저

codex가 open_clip 없이 순수 torch로 CLIP과 토크나이저를 구현했다. 모듈과 파라미터 이름은 open_clip과 같아 로컬 `.bin`을 `strict`로 불러온다. 다음은 파일 대응이다.

| open_clip | visionlab | 비고 |
|---|---|---|
| `model.py`의 `CLIP` | `models/winclip/clip.py`의 `CLIP` | 텍스트 쪽은 open_clip처럼 최상위 속성(`token_embedding`, `positional_embedding`, `transformer`, `ln_final`, `text_projection`, `logit_scale`, 비영속 `attn_mask`). `encode_image`, `encode_text`만 구현 |
| `transformer.py`의 `VisionTransformer` | 같은 이름 | `conv1`, `class_embedding`, `positional_embedding`, `patch_dropout`(Identity), `ln_pre`, `transformer`, `ln_post`, `proj`, `_global_pool`, `output_tokens`, `grid_size`, `image_size` |
| `Transformer`, `ResidualAttentionBlock`, `LayerNorm` | 같은 이름 | fp32 경로, `nn.MultiheadAttention`, `nn.GELU`, mlp_ratio 4.0 |
| `tokenizer.py`의 `SimpleTokenizer`, `tokenize` | `models/winclip/tokenizer.py` | `regex`는 `SimpleTokenizer.__init__` 안에서 지연 import, `ftfy.fix_text`는 생략(고정 ASCII 프롬프트) |
| `bpe_simple_vocab_16e6.txt.gz` | `models/winclip/bpe_simple_vocab_16e6.txt.gz`, `LICENSE` | 바이트 단위 복사, MIT 표기 파일을 옆에 둠 |

구현하지 않은 분기는 attentional pooling, CoCa, HF·ResNet·timm 타워, 0이 아닌 patch dropout, QuickGELU, 저정밀 cast, LayerScale, 대체 pooling과 mask, checkpointing과 lock, 토크나이저의 추가 special token과 대체 clean 방식, random·syntax reduction이다. 이 모델(ViT-B-16-plus-240)이 쓰는 경로만 있다.

독립 구현 규모는 `clip.py` 165줄, `tokenizer.py` 144줄로 합계 309줄(주석과 빈 줄 포함)이다. 계획의 약 500줄 추정보다 작다. 파라미터는 합계 208,377,345개다(이미지 인코더 117,213,824개, 텍스트 인코더와 `logit_scale` 91,163,521개). 계획의 약 2.1억(이미지 1.2억, 텍스트 0.9억)과 맞는다. 기본값은 `quick_gelu=False`, `nn.GELU`, 양쪽 mlp_ratio 4.0, 이미지 head 14개(head_width 64), 텍스트 head 10개, `init_logit_scale=log(1/0.07)`, patch dropout 0, 이미지 pooling `tok`, 텍스트 pooling `argmax`, `final_ln_after_pool=False`, LayerNorm epsilon 1e-5다.

### 3.2 18-B: 순수 모델, Lightning 모델, 실행 흐름

원본 파일에서 `utils.py`, `prompting.py`는 변경이 0줄이고(`dl.py` 줄 대조), `model.py`(원본 `torch_model.py`)는 의존성 제거와 아래 항목만 다르다(대조 결과 변경 63줄). 다음은 원본 대응이다.

| 원본 | 이식 위치 | 비고 |
|---|---|---|
| `torch_model.py`의 `WinClipModel` | `models/winclip/model.py` | open_clip과 `PRETRAINED`, torchvision 제거, 독립 `CLIP`과 tokenizer 사용 |
| `utils.py`, `prompting.py` | 같은 이름 | 변경 없음 |
| `components/base/buffer_list.py`의 `BufferListMixin` | `models/components/base/buffer_list.py` | anomalib import 제거 복사(변경 0줄), `base/__init__.py`에 export 추가 |
| `lightning_model.py`의 `WinClip` | `models/winclip/lightning_model.py`의 `WinClipLitModel` | native `LightningModule`, 인자 8개 동일 |
| (없음) | `configs/models/winclip.yaml` | `class_name`, `k_shot`, `scales`, `few_shot_source`, `visualizer` |
| `Engine.train`의 zero-shot 분기 | `trainers/lightning_cli.py`의 `LitCLI._run_subcommand`와 `evaluation/visionlab/train.py` | `train` 명령일 때 `learning_type`이 ZERO_SHOT 또는 FEW_SHOT이면 `fit` 대신 `validate`, 그 뒤 기존 `test` |

원본과 다르게 구현한 곳은 다음과 같다.

- **resize 위치 (사용자 승인 2026-10-28):** 레퍼런스는 입력이 `Resize(256)` 다음 `Resize(240, bicubic, antialias)` 다음 CLIP 정규화다. `common/`을 고치지 않는 조건에서 export한 순수 모델(Predictor)도 같은 resize를 해야 하므로, 사용자 결정 3의 "`WinClipLitModel` 안에서 resize"를 `WinClipModel.encode_image` 맨 앞으로 옮겼다. 입력 공간 크기가 `clip.visual.image_size`와 다를 때만 `interpolate(bicubic, antialias=True)`를 하고, `forward`의 마지막 보간은 원본 코드대로 원래 입력 크기(256)로 한다. 240 입력이면 원본과 같다. 채널별 정규화(아핀)와 보간(가중치 합 1)이 교환되므로 정규화 뒤 resize는 resize 뒤 정규화와 수학적으로 같다. Lit의 `pre_processor`는 `PreProcessor(mean, std)`(CLIP 값)다. 사용자가 이 위치를 승인했다(2026-10-28, 근거: `common/` 무수정, export한 Predictor도 같은 resize). open_clip 대조는 `compare_outputs`에서 같은 240 입력의 출력이 레퍼런스(open_clip)와 최대 차이 0.0이었고, `interpolate`와 torchvision `Resize`의 240 변환도 최대 차이 0.0이었으며, 256 입력 경로는 `pred_score` 3.6e-7 이하, `anomaly_map` 5.4e-4 이하(보간 대상 크기 차이)였다(5.3절).
- **hook 제거:** 원본 `encode_image`가 매 호출마다 `patch_dropout`에 걸고 제거하지 않는 forward hook을 `try/finally`에서 `handle.remove()`한다. 출력은 같다. 원본의 hook 누적이 레퍼런스 체크포인트 pickle 오류의 원인이었다(2.2절).
- **`apply_transform=True`:** 인자 이름·순서·기본값은 그대로 두고 `True`이면 `NotImplementedError`다(open_clip 이미지 transform과 `ToPILImage`는 옮기지 않음). `transform` 프로퍼티와 `_transform`은 없다.
- **few-shot:** 순수 모델의 `k_shot > 0` 경로(`_collect_visual_embeddings`, `_compute_few_shot_scores`)는 옮겼으나 이번 Phase에서 실행하지 않았다. Lit의 `few_shot_source`가 주어지면 `NotImplementedError`다(PredictDataset 미이식). `collect_reference_images`는 학습 dataloader의 `(image, target)` 튜플에서 모은다.
- **state_dict:** 원본의 FITTING·VALIDATING 중 `model.clip` 제외 규칙은 옮기지 않았다. export에 CLIP 가중치 전체(795.0MB)와 `_text_embeddings`가 들어간다.
- **D-8의 zero-shot 확장:** 후처리 통계는 fit validation에서만 갱신하던 규칙을 이 모델에서는 "FITTING 중 validation 또는 ZERO_SHOT·FEW_SHOT 모델의 VALIDATING 단계"로 넓혔다(원본 anomalib은 zero-shot에서 validate 하나로 통계를 만든다). export는 `on_validation_end`에서 `VALIDATING`일 때 한다.
- **텍스트 임베딩:** 원본 `_collect_text_embeddings`에는 정규화가 없고 프롬프트별 `encode_text` 결과를 상태 그룹별로 평균해 이어 붙인다. 원본을 유지했다(이 지시에 정규화를 적은 것은 main의 오기였고 codex가 질문해 바로잡았다).
- **구성 지점:** 단위 테스트가 작은 CLIP을 쓰도록 `model.py`가 `CLIP`을 모듈 이름으로 만든다(`self.clip = CLIP()`). 테스트가 이 이름을 monkeypatch한다.

앞 모델에 영향이 있을 수 있는 공유 파일은 다음과 같다. 앞 모델 회귀(01~17)는 돌리지 않았다.

| 파일 | 변경 | 영향 판단 |
|---|---|---|
| `evaluation/visionlab/train.py` | `LitCLI(args=args, train_and_test=train_and_test)` 전달, 머리말 주석 | `train_and_test` 인자 추가만 |
| `src/visionlab/anomaly_detection/trainers/lightning_cli.py` | `__init__`의 `train_and_test` 인자, `_run_subcommand`의 zero-shot 분기, `LearningType` import | ONE_CLASS와 직접 지정한 `fit`, `validate`, `test`, `predict`는 부모 경로 그대로. 단위 테스트가 모든 `LearningType` x 명령 조합의 dispatch를 확인. STFPM 실행으로 fit 경로 확인(5.4절) |
| `src/visionlab/inference/predictor.py` | `WinClipModel` import와 `models["winclip"]` 한 줄 | 기존 분기 변경 없음 |
| `src/visionlab/utils/io.py`, `evaluation/visionlab/predict.py`, `evaluation/anomalib/compare_outputs.py` | 허용 이름 `winclip` 추가, `compare_outputs`에 winclip 분기 | 기존 동작 유지 |
| `src/visionlab/anomaly_detection/models/components/base/__init__.py` | `BufferListMixin` import와 `__all__` | 추가만 |
| `tests/common/test_cli.py`, `tests/common/test_dependency.py` | winclip 테스트 추가, regex 허용 목록과 분류 | 앞 모델 테스트 본문 변경 없음 |
| `evaluation/anomalib/local_models.py` | `LocalWinClip.on_save_checkpoint` | winclip 전용 |
| `paths.toml` | `winclip_clip` 한 줄 | 추가만 |

codex의 구현 보고에 따른 B단계 코드 줄 수는 `model.py` 470줄(원본 복사 부분 포함), `lightning_model.py` 194줄이다. 테스트는 `tests/18_winclip/test_clip.py` 343줄, `test_winclip.py` 299줄, `test_winclip_lightning.py` 348줄이다.

## 4. 적대적 검증

검증 중점 두 가지의 결과는 다음과 같다.

| 중점 | 확인 | 결과 |
|---|---|---|
| anomalib 의존성의 완전한 삭제 | `src/visionlab/anomaly_detection/models/winclip/*.py`에서 `anomalib`, `open_clip`, `timm`, `torchvision`, `kornia`, `ftfy`를 grep하면 원본 docstring 예시 문구(`>>> from anomalib...`)와 `clip.py`의 "open_clip fp32" 설명 문구뿐이고 import는 없다. `regex`는 `tokenizer.py`의 `SimpleTokenizer.__init__` 한 곳이다. `tests/common/test_dependency.py`(6 passed)가 허용 목록 밖의 `regex` import와 `open_clip`, `ftfy` import, winclip 파일의 `timm`, `torchvision` import를 차단한다. 순수 파일에 Lightning import가 없고 `WinClipLitModel`은 native `LightningModule`이다 | 충족 |
| 로컬 pretrained 가중치 | 순수 `CLIP`은 파일을 읽지 않는다(`test_clip_no_file_loading_methods_or_forbidden_imports`). 주입은 `WinClipLitModel.__init__`의 `load_pretrained_teacher(self.model.clip, "winclip_clip")`(`weights_only=True`, `strict=True`, 누락 시 `FileNotFoundError`)뿐이다. native 실행은 `block_network()` 아래에서 성공했다. `compare_outputs`에서 native CLIP가 로컬 파일과, 레퍼런스(open_clip) CLIP와 키 302개, 값 `torch.equal`로 같았다(FR-23) | 충족 |

회차별 기록은 다음과 같다. 18-A와 18-B의 첫 구현물에 Critical은 없었다.

| 회차 | 지적 | 심각도 | 처리 |
|---|---|---|---|
| 레퍼런스 | 레퍼런스 실행이 `validate` 끝에서 hook pickle 오류로 실패(2.2절) | Major | 수정 요청 1/3(레퍼런스 어댑터): `LocalWinClip.on_save_checkpoint`에서 hook 비움. 재실행 정상 |
| 18-A | `tokenizer.py`를 open_clip과 줄 단위로 대조: `bytes_to_unicode`, `get_pairs`, `bpe`, `encode`, `__call__`의 잘림 규칙, special token 처리가 같고 `ftfy`만 생략. `clip.py`의 `VisionTransformer.forward`는 `ln_post` 뒤 풀링(`final_ln_after_pool=False`), 텍스트는 `argmax` 풀링과 인과 mask가 open_clip과 같다 | 없음 | 수정 없음. `test_clip.py` 11 passed(open_clip과 같은 가중치로 CPU 출력, 토크나이저 462개 문장 정수 일치) |
| 18-B 질문 1 | codex 질문: 텍스트 임베딩 정규화 여부 | 해당 없음 | 원본에 정규화가 없어 원본 평균 유지(지시의 오기 정정) |
| 18-B 질문 2 | codex 질문: Predictor가 기존 모델 import로 torchvision, timm, kornia를 함께 불러 새 프로세스 torchvision 차단 테스트를 충족할 수 없음 | Minor | 기존 모델 import를 건드리지 않고 등록만 추가(codex 선택지 2, main 결정). winclip 파일에 torchvision이 없음은 정적 검사와, 공유 의존성 초기화 뒤 차단하는 격리 테스트로 확인. 새 프로세스에서의 완전한 torchvision 차단은 보장하지 않는다(기존 PostProcessor의 torchmetrics도 torchvision을 간접 import) |
| 18-B | `test_winclip_native_signature_teacher_and_contract`의 `PreProcessor.mean` dtype 비교 실패(float64 대 float32), `components/base/__init__.py`의 `__all__`에 `BufferListMixin` 누락 | Major(테스트 실패), Minor | 수정 요청 1/3(18-B): 테스트의 기대 dtype을 float64로, `__all__`과 import 순서 정리. 재실행 `tests\18_winclip` 31 passed |
| 18-B | `train.py` zero-shot 분기: `LitCLI._run_subcommand` 재정의를 읽고 ONE_CLASS와 직접 지정 서브커맨드가 `super()._run_subcommand`를 타는지 확인, `ckpt_path` 전달과 `record_run` 위치 확인 | 없음 | 수정 없음. STFPM 실행으로 확인 |
| 18-B | `compare_outputs.py` winclip 분기: 레퍼런스 hook 누적을 중간에 비우는지, PASS 판정이 같은 240 입력에서만 이루어지고 256 경로는 INFO인지 확인 | 없음 | 수정 없음 |

수정 요청은 레퍼런스 어댑터 1회, 18-B 1회로 한도(3회)를 넘지 않았다. 18-A는 0회다. codex가 18-B 중 모델 용량 오류로 한 번 멈춰 이어서 하라고 지시했다. 다음은 남은 지적이다.

- 새 프로세스 torchvision 차단 한계(위 표). 이 Phase의 코드가 아니라 기존 Predictor와 torchmetrics의 import 구조다.
- `LocalWinClip.on_save_checkpoint`는 체크포인트 저장 경로만 고친다. `Engine.export(TORCH)`의 같은 pickle 문제는 이 Phase에서 쓰지 않아 남는다.

## 5. 이식 결과

### 5.1 단위 테스트와 비교 스크립트 실행

다음은 18-A 직후 agy에 보낸 명령이다(2026-10-28, 레퍼런스 두 번째 실행과 한 묶음). 종료 코드는 모두 0이다.

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py -m pytest tests\18_winclip\test_clip.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\18-pytest-clip-261028-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\common\test_dependency.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\18-pytest-dependency-261028-1.log; "exit=$LASTEXITCODE"
cd D:\projects\visionlab\evaluation\anomalib
& $py anomalib_run.py train --config default.yaml --config data/mvtec.yaml --config models/18_winclip.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\18-ref-mvtec-261028-2.log; "exit=$LASTEXITCODE"
& $py anomalib_run.py train --config default.yaml --config data/visa.yaml --config models/18_winclip.yaml 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\18-ref-visa-261028-2.log; "exit=$LASTEXITCODE"
```

다음은 18-B 이후 agy에 보낸 명령이다(2026-10-28). 첫 승인 창은 `cd`, `$py`, 첫 명령이 한 창에 합쳐져 main이 직접 읽고 한 번 승인했고, 나머지 명령은 창 없이 실행되어 교차 확인할 수 없다. agy는 시킨 명령 사이에 로그를 읽는 `Get-Content` 명령을 스스로 실행했다(읽기 전용). 종료 코드는 `18-pytest-261028-1.log`의 1 failed를 제외하고 모두 0이다.

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py -m pytest tests\18_winclip 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\18-pytest-261028-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\common\test_cli.py -k winclip 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\18-pytest-cli-261028-1.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\common\test_dependency.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\18-pytest-dependency-261028-2.log; "exit=$LASTEXITCODE"
& $py evaluation/visionlab/train.py train --data mvtec --model winclip 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\18-native-mvtec-261028-1.log; "exit=$LASTEXITCODE"
& $py evaluation/visionlab/predict.py --model winclip --data mvtec 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\18-pure-mvtec-261028-1.log; "exit=$LASTEXITCODE"
& $py evaluation/visionlab/train.py train --data visa --model winclip 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\18-native-visa-261028-1.log; "exit=$LASTEXITCODE"
& $py evaluation/visionlab/predict.py --model winclip --data visa 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\18-pure-visa-261028-1.log; "exit=$LASTEXITCODE"
& $py evaluation/anomalib/compare_outputs.py --model winclip --samples 8 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\18-compare-outputs-261028-1.log; "exit=$LASTEXITCODE"
& $py evaluation/visionlab/train.py train --data mvtec --model stfpm --trainer.max_epochs 1 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\18-stfpm-check-261028-1.log; "exit=$LASTEXITCODE"
```

다음은 수정 요청 뒤 재실행한 명령이다(같은 방식으로 첫 창만 직접 승인). 종료 코드는 모두 0이다.

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py -m pytest tests\18_winclip 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\18-pytest-261028-2.log; "exit=$LASTEXITCODE"
& $py -m pytest tests\common\test_dependency.py 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\18-pytest-dependency-261028-3.log; "exit=$LASTEXITCODE"
```

다음은 pytest 결과 요약이다.

| 대상 | 결과 | 로그 |
|---|---|---|
| `tests\18_winclip\test_clip.py` | 11 passed | `18-pytest-clip-261028-1.log` |
| `tests\18_winclip` 전체 (1차) | 30 passed, 1 failed (`test_winclip_native_signature_teacher_and_contract`, dtype 가정) | `18-pytest-261028-1.log` |
| `tests\18_winclip` 전체 (수정 후) | 31 passed (`test_clip` 11, `test_winclip` 10, `test_winclip_lightning` 10) | `18-pytest-261028-2.log` |
| `tests\common\test_cli.py -k winclip` | 4 passed | `18-pytest-cli-261028-1.log` |
| `tests\common\test_dependency.py` | 6 passed (3회 실행) | `18-pytest-dependency-261028-1.log`, `-2.log`, `-3.log` |
| 사용자 승인 회귀 실행(2026-10-28): `tests\common` | 67 passed (이전 63개에 winclip의 `test_cli` 4개 추가), 실패 0 | `18-regress-common-261028-1.log` |

### 5.2 native Lightning과 순수 PyTorch 판정

학습이 없어 에폭 히스토리 비교는 해당 없다. 판정은 D-24 정의에 따른다. 레퍼런스 값은 정규화 점수의 AUROC이고, 순수 `predict.py`는 raw와 `*_normalized`를 기록하며 native `train.py` 로그는 raw만 기록한다. native 값의 정규화 정의는 같은 모델의 export를 읽는 순수 추론의 `*_normalized`로 본다. 다음은 Image AUROC 비교다.

| category | 레퍼런스 | native (raw) | 차이 | 순수 (raw) | 순수 (normalized) | 차이 (normalized) | 판정 |
|---|---|---|---|---|---|---|---|
| bottle | 0.989035 | 0.989035 | 6e-8 | 0.989035 | 0.989035 | 1e-8 이하 | 충족 |
| candle | 0.965278 | 0.965278 | 6e-8 | 0.965278 | 0.965278 | 1e-8 이하 | 충족 |

다음은 Pixel AUROC 비교다.

| category | 레퍼런스 | native (raw) | 차이 | 순수 (raw) | 순수 (normalized) | 차이 (normalized) | 판정 |
|---|---|---|---|---|---|---|---|
| bottle | 0.854374 | 0.854486 | 1.1e-4 | 0.854486 | 0.854486 | 1.1e-4 | 충족 |
| candle | 0.873874 | 0.880492 | 6.6e-3 | 0.880493 | 0.873607 | 2.7e-4 | 충족 |

candle Pixel AUROC는 raw(0.8805)와 normalized(0.8736)가 0.0069 달라 기준 0.01 안이지만 차이가 있다. 이는 D-24가 적은 정규화 점수의 clamp 동점 때문이다. 레퍼런스 정의(normalized)로 비교한 2.7e-4가 판정값이고, native raw 값도 레퍼런스와의 차이 6.6e-3으로 0.01 안이다. 레퍼런스는 Pixel AUROC를 mask 240x240으로, native와 순수는 256x256으로 평가한다. 해상도 차이로 인한 Pixel AUROC 차이는 bottle 1.1e-4, candle 2.7e-4(normalized)로 0.01에 못 미친다. 다음은 F1 값이다.

| category | 지표 | 레퍼런스 | native | 순수 |
|---|---|---|---|---|
| bottle | Image F1 | 0.958904 | 0.958904 | 0.958904 |
| bottle | Pixel F1 | 0.504720 | 0.498383 | 0.498383 |
| candle | Image F1 | 0.923077 | 0.923077 | 0.923077 |
| candle | Pixel F1 | 0.068276 | 0.070140 | 0.070140 |

native val 값은 bottle `val_image_AUROC` 0.9700, `val_pixel_AUROC` 0.8490, candle 0.9606, 0.8739이다(`18-native-mvtec-261028-1.log`, `18-native-visa-261028-1.log`). 실행 폴더는 `outputs/visionlab/winclip/mvtec_bottle/v0`, `outputs/visionlab/winclip/visa_candle/v0`이며 각 `export/model.pt`는 795.0MB다.

### 5.3 compare_outputs (CPU, FR-5)

`compare_outputs.py --model winclip --samples 8`은 PASS 32개, FAIL 0개다(`outputs/compare/winclip_outputs.json`, `18-compare-outputs-261028-1.log`). 주요 결과는 다음과 같다.

| 항목 | 결과 |
|---|---|
| FR-6 생성자 | anomalib `WinClip`과 `WinClipLitModel` 시그니처 일치(8개) |
| FR-23 | native CLIP와 로컬 `.bin`이 키 302개 모두 `torch.equal`, 레퍼런스(open_clip)와 native 두 데이터 모두 `torch.equal` |
| 텍스트 임베딩 | bottle, candle 모두 shape [2, 640], 레퍼런스와 최대 차이 0.0 |
| 같은 240 입력 `pred_score` [8], `anomaly_map` [8,240,240] | 두 데이터 모두 최대 차이 0.0 (허용 `atol=1e-6, rtol=1e-5`) |
| INFO: 이미지 임베딩, scale 2·3 window 임베딩 | 최대 차이 0.0 |
| INFO: `interpolate`와 torchvision `Resize`의 240 변환 | 최대 차이 0.0 |
| INFO: native 256 경로 `pred_score` | bottle 3.6e-7, candle 2.7e-7 |
| INFO: native 256 `anomaly_map`과 레퍼런스 240 map을 256으로 보간한 값 | bottle 5.4e-4, candle 4.4e-4 (보간 대상 크기가 달라 생기는 작은 차이) |

compare_outputs의 FR-5 비교는 CPU에서 했다(GPU 비교는 이 Phase에서 하지 않았다). AUROC 판정은 GPU에서 실행한 native와 순수 결과이고, 같은 240 입력의 CPU 출력은 레퍼런스와 비트 단위로 같았다.

사후 회귀 명령은 다음과 같다(2026-10-28, agy 종료 코드 0, 승인 창 없이 실행되어 로그 이름과 출력으로 확인).

```powershell
cd D:\projects\visionlab
$py = 'C:\winpython\WPy64-31180_gpu\python-3.11.8.amd64\python.exe'
& $py -m pytest tests\common 2>&1 | Tee-Object -FilePath D:\projects\visionlab\outputs\logs\18-regress-common-261028-1.log; "exit=$LASTEXITCODE"
```

### 5.4 앞 모델 영향 확인 (STFPM)

`train.py train --data mvtec --model stfpm --trainer.max_epochs 1`(1 epoch)이 정상 종료했다(`18-stfpm-check-261028-1.log`). `Overriding gradient_clip_val`, `num_sanity_val_steps` 로그 뒤 test 결과는 image_AUROC 0.9846, pixel_AUROC 0.8492였다. ONE_CLASS 모델의 `fit`, `test` 경로가 새 분기와 무관하게 동작함을 확인한 것이며 STFPM 10 epoch 값과 비교한 것은 아니다. `tests\common` 전체는 사용자 승인 회귀 실행으로 67개가 통과했다(5.1절). 전체 회귀 01~18은 커밋 뒤에 따로 한다(사용자 결정).

## 6. 결론

Phase 18의 네 판정값(bottle과 candle의 Image·Pixel AUROC)이 모두 기준 0.01 안이다. native Lightning(`WinClipLitModel`)과 export를 읽은 순수 PyTorch가 레퍼런스와 같은 결과를 냈다.

| 항목 | 상태 |
|---|---|
| 판정 기준(Image·Pixel AUROC 차이 0.01 이내) | 충족. Image 차이 최대 6e-8, Pixel 차이 최대 normalized 기준 2.7e-4(native raw 6.6e-3) |
| 순수 모델 동등성(FR-5) | CPU에서 같은 240 입력의 `pred_score`, `anomaly_map`이 레퍼런스와 최대 차이 0.0 |
| 의존성 | `src/visionlab`에 anomalib, open_clip, timm, torchvision, kornia, ftfy import 없음. `regex`는 토크나이저 지연 import 한 곳 |
| 가중치 | `paths.toml`의 로컬 `.bin`만 사용, `strict` 주입, 레퍼런스와 같음(FR-23) |
| 테스트 | `tests\18_winclip` 31, `test_cli -k winclip` 4, `test_dependency` 6, 사용자 승인 회귀 `tests\common` 67 모두 통과 |
| 수정 요청 | 레퍼런스 어댑터 1회, 18-B 1회 (한도 3회) |
| 소요 시간 | 시작 약 15:20, 실행 종료 약 16:35로 약 75분. 계획 추정(agy 실행 15~25분, 레퍼런스와 native 각각 수 분)과 같은 규모. 가장 긴 구간은 codex 18-B 구현(약 15분)이었다 |
| Phase 종료 | 사용자가 종료를 결정했다(2026-10-28). `Phase 18: winclip ...` 커밋으로 푸시한다 |

### 6.1 `specs/` 갱신 (반영 완료)

`specs/`는 사람이 쓴다. 최종 보고에 대한 사용자 결정(2026-10-28)에 따라 아래와 같이 반영했다. INTENT와 PLAN은 고치지 않았다.

- **FR-4 (반영 완료):** `SPEC.md` FR-4의 DSR 문장 뒤에 "WinCLIP도 원본의 평가 `forward`를 유지해 anomaly map이 `[B,H,W]`, score가 `[B]`다"를 더하고 판정 방법 문구에 WinCLIP을 포함했다.
- **D-15 트리 (반영 완료):** `tests/18_winclip/`(세 파일), `configs/models`의 `winclip.yaml`, 모델 패키지 `models/winclip/`(`__init__.py`, `bpe_simple_vocab_16e6.txt.gz`, `clip.py`, `LICENSE`, `lightning_model.py`, `model.py`, `prompting.py`, `tokenizer.py`, `utils.py`), `models/components/base/`(`buffer_list.py`, `dynamic_buffer.py`)를 추가했다. 트리에서 uflow가 마지막 항목이 아니게 되어 가지 문자를 맞췄다.
- **D-28 (새 결정, 반영 완료):** 제목은 "CLIP은 독립 구현하고 토크나이저의 regex를 지연 의존으로 허용한다"이고 D-25 형식(선택, 적용 범위, INTENT와의 관계, 근거, 배제한 대안, 남은 한계)이다. open_clip 독립 구현, `regex` 지연 import 한 곳, `ftfy` 미사용, 어휘 파일과 `LICENSE` 복사, 256 입력의 resize를 순수 `encode_image` 맨 앞에 두는 것(D-27이 배제한 "모델 안 보간"과 달리 240 입력에서는 원본과 같음)을 적었다.
- **D-8 확장 (반영 완료):** D-8의 선택 문장에 "학습이 없는 zero-shot·few-shot 모델(WinCLIP)은 validate 단계가 fit validation의 역할을 하므로 그 VALIDATING 단계에서 통계를 만들고 확정한다. test·predict는 읽기만 한다"를 더했다.

### 6.2 남은 위험과 사용자 판단 대상

- **pixel 평가 해상도 차이:** 레퍼런스는 240, native와 순수는 256에서 평가한다. 이번 결과에서는 Pixel AUROC 차이가 2.7e-4 이하(normalized)였고 0.01에 못 미쳤다.
- **두 단계 resize:** 레퍼런스를 `Resize(256)` 다음 `Resize(240, bicubic)`로 맞춘 것은 원본의 직접 240 resize와 다르다(D-27, 한계로 기록).
- **few-shot 미검증:** `k_shot > 0` 경로(순수 모델과 `collect_reference_images`)는 이식했으나 실행하지 않았고, `few_shot_source`와 `apply_transform=True`는 `NotImplementedError`다.
- **export 크기:** `model.pt`가 795.0MB다(CLIP 가중치 포함).
- **Predictor의 torchvision:** 기존 Predictor와 torchmetrics의 import 구조 때문에 새 프로세스 torchvision 차단 테스트는 보장하지 않는다(4절).
- **`Engine.export(TORCH)`:** 레퍼런스 쪽의 같은 hook pickle 문제는 이 Phase에서 쓰지 않아 남는다.
- **앞 모델 회귀:** 01~17 모델별 회귀는 돌리지 않았다. `tests\common`은 커밋 전에 67개 통과했고, 전체 회귀 01~18은 커밋 뒤에 따로 실행한다. 공유 파일(`train.py`, `lightning_cli.py`, `predictor.py`, `io.py`, `compare_outputs.py`, `components/base`)을 바꿨으므로 사용자 결정 7(b)대로 Phase 종료 뒤 전체 회귀 01~18과 `tests\common` 사후 실행을 한다.
- **agy 승인 교차 확인:** 묶음마다 첫 승인 창만 떴고 이후 명령은 창 없이 실행되어 교차 확인할 수 없다.
- **val/test 개수:** 로그에서 mvtec validate 5 배치, test 7 배치, visa test 15 배치를 확인했다(계획의 추정과 같은 규모).
