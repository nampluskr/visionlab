# INTENT

## 무엇을 하는가

순수 PyTorch만으로 dataset, transform, dataloader, model, inference를 수행할 수 있는 모델과 가중치를 배포한다.

anomalib처럼 이상 탐지에 적용된 통합 학습·평가·추론 프레임워크를 native Lightning으로 구현한다.

우선 다루는 task는 anomaly detection이다. 이후 classification, segmentation, object detection, GAN, autoencoder, VAE 등 다른 vision task로 넓힌다.

구현한 프레임워크는 anomalib을 설치한 환경에서 제대로 동작하는지, 모델 성능이 나오는지 검증한다. 검증은 다음 세 단계로 한다.

1. anomalib의 image 이상 탐지 모델을 로컬 자산으로 그대로 실행해 레퍼런스 성능 수치를 확인한다.
2. anomalib image 모델을 native Lightning만으로 이식·수정하고, 학습·평가 수치를 1단계와 비교한다.
3. 2단계에서 학습한 가중치를 순수 PyTorch 모델로 불러와 추론하고, 그 수치를 1단계와 비교한다.

## 왜 하는가

모델을 배포하는 곳에는 anomalib과 Lightning이 설치되어 있지 않고, PyTorch만 사용할 수 있다.

anomalib 같은 통합 학습·평가·추론 프레임워크는 이상 탐지에 이미 적용되어 있다. 이것을 native Lightning으로 직접 구현하고, 이상 탐지에서 다른 vision task로까지 넓히고 싶다.

## 변하지 않는 원칙

- 배포되는 모델은 순수 PyTorch만으로 동작한다.
- 위 원칙의 예외는 추론 경로에서 쓸 수 있는 외부 패키지를 `SPEC`의 허용 목록에 근거와 함께 적은 것에 한하며, 목록에 없는 외부 패키지는 추론 경로에서 쓰지 않는다.
- 원본(anomalib)의 알고리즘을 보존한다.
- anomalib 사용자가 같은 방식으로 쓸 수 있게, 클래스와 생성자 인자의 이름·순서·기본값을 anomalib과 맞춘다.
- 구현했다는 것과 실행으로 검증했다는 것을 구별한다.
- 필요한 것만 만든다. 빈 폴더나 미리 만든 추상화는 두지 않는다.
- 로컬에 저장된 데이터셋과 모델 백본 가중치를 사용하고, 인터넷에서 다운로드 받지 않는다.

## 다루지 않는 것

- 실행 시 anomalib을 import하거나 상속하지 않는다. Engine, 기반 클래스, 전용 Batch도 가져오지 않는다. anomalib은 별도로 설치해 비교·검증하는 용도로만 쓴다.
