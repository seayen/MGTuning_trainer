# MGTuning Trainer

MGTuning Trainer는 **뮤직젠에 프롬프트 튜닝을 적용한 코드**입니다.

## 설치 방법

```bash
# Clone the repository
git clone https://github.com/seayen/MGTuning_trainer.git

# Navigate to the project directory
cd MGTuning_trainer

# Install dependencies
pip install -r requirements.txt

# Note: wandb 및 torch는 각 환경에 맞게 설치해주세요.
# 예: 
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# pip install wandb
```

## 사용 방법


## 환경 설정
trainer.py의 main 함수에서 기본적인 프롬프트 벡터 길이와 간단한 하이퍼파라미터를 수정

자세한 설정 또는 wandb 로그 설정은 train 함수의 trainer 설정에서 진행

데이터셋 생성 주의사항
토크나이저로 벡터화한 입력 텍스트 앞에 해당 프롬프트 길이만큼의 빈 공간(0으로 채워진 공간)을 미리 생성
훈련 실행
```bash
python Musicgen_prompt_tuning_trainer.py
```
학습 결과는 로그 및 저장된 모델 파일에서 확인 가능


## 프로젝트 구조

```plaintext
MGTuning_trainer/
├── Datasets/                  # 데이터셋
├── Musicgen_prompt_tuning_trainer.py # 뮤직젠 프롬프트 튜닝 훈련 스크립트
├── testGenerate.py           # 프롬프트 벡터를 이용한 테스트 스크립트 (단일 샘플 생성)
├── testGenerate_multi.py     # 테스트 스크립트 (다중 샘플 생성)
├── vectors/                  # 프롬프트 벡터 관련 파일 저장 경로
│   ├── load
│       ├── prompt_vector.pt          # 미리 학습된 프롬프트 벡터를 가져와서 이어서 학습시키는 경로
│   ├── prompt_vector.pt               # 프롬프트 벡터가 저장되는 경로
│   └── run
│       ├── prompt_vector_step_000.pt      # 학습 중간에 저장되는 벡터 경로
└── README.md                 # 프로젝트 설명 파일
```

