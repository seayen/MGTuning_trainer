import torch
import torch.nn as nn
from transformers import MusicgenForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_from_disk
from transformers import TrainerCallback
import os


class SavePromptVectorCallback(TrainerCallback):
    def __init__(self, save_dir, prompt_vector, save_steps=100):
        self.save_dir = save_dir
        self.prompt_vector = prompt_vector
        self.save_steps = save_steps

        # 저장 경로가 없으면 생성
        os.makedirs(save_dir, exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        # 특정 스텝마다 프롬프트 벡터 저장
        if state.global_step % self.save_steps == 0:
            save_path = os.path.join(self.save_dir, f"prompt_vector_step_{state.global_step}.pt")
            torch.save(self.prompt_vector, save_path)
            print(f"Saved prompt vector at step {state.global_step} to {save_path}")


class MusicGenPromptTuner:
    def __init__(self, model_name='facebook/musicgen-small', prompt_length=10, hidden_size=768, prompt_vector=None):
        # 모델 불러오기
        self.musicgen_model = MusicgenForConditionalGeneration.from_pretrained(model_name)
        self.musicgen_model.config.decoder.decoder_start_token_id = 0 # 필수 설정으로, 디코더의 시작 토큰을 어떤 것으로 할 지 설정함

        # 모든 가중치 고정 (인코더, 디코더, 오디오 인코더)

        for param in self.musicgen_model.text_encoder.parameters():
            param.requires_grad = False
        for param in self.musicgen_model.decoder.parameters():
            param.requires_grad = False
        for param in self.musicgen_model.audio_encoder.parameters():
            param.requires_grad = False


        # 추가할 프롬프트 벡터 설정
        self.prompt_length = prompt_length
        self.hidden_size = hidden_size
        self.prompt_vector = prompt_vector


    def forward_hook(self, module, input):
        # input은 튜플 형태로 전달됨, 첫 번째 요소가 실제 입력 텐서
        original_input = input[0]
        batch_size, _, hidden_size = original_input.shape

        # 프롬프트 길이 계산
        prompt_length = self.prompt_length

        # 학습 가능한 프롬프트 벡터를 배치 크기에 맞게 확장
        expanded_prompt = self.prompt_vector.to(original_input.device).expand(batch_size, prompt_length, hidden_size)

        # 빈 공간(프롬프트 부분)을 학습 가능한 프롬프트 벡터로 대체
        original_input[:, :prompt_length, :] = expanded_prompt

        # 출력 상태를 그대로 반환
        return (original_input,)

    def train(self, dataset_path, model_dir= r"./tuned_model/musicgen_prompt_tuned" , vector_dir = r"../tuned_model/vectors", learning_rate=5e-5, batch_size=3, num_epochs=3):
        """
        prompt-tuning 이 적용된 뮤직젠 모델을 train하는 함수
        Args:
            dataset_path (str): 데이터 셋 준비 함수로 생성된 데이터 셋 경로
            vector_dir (str): 훈련 결과 모델이 저장되는 경로
            model_dir : 저장될 모델 경로 (저장 안함)
            learning_rate : 옵티마이져에서 사용되는 학습률 (클수록 학습 불안정 속도 증가 | 작을 수록 학습 안정 속도 감소)
            batch_size : 한번에 학습에 처리하는 데이터 샘플의 갯수
            num_epochs : 데이터 셋을 몇번 반복하여 학습할지
        Returns:
        """
        # 데이터셋 로드
        dataset_dict = load_from_disk(dataset_path)
        train_dataset = dataset_dict['train']
        eval_dataset = dataset_dict.get('eval', None)

        # prompt_vector 생성 (입력된 프롬프트 벡터가 없을 경우)
        if self.prompt_vector is None:
            self.prompt_vector = nn.Parameter(torch.randn(1, self.prompt_length, self.hidden_size) * 0.01, requires_grad=True)
            print("trainer : new prompt vector is created")
        else :
            print("trainer : prompt vector is loaded")

        # forward hook 을 인코더의 첫 번째 레이어에 등록
        self.musicgen_model.text_encoder.encoder.block[0].register_forward_pre_hook(self.forward_hook)

        # 옵티마이저 설정 - 프롬프트 벡터만 학습하도록
        optimizer = torch.optim.Adam([self.prompt_vector], lr=learning_rate)

        # 중간에 프롬프트 벡터를 저장
        save_prompt_callback = SavePromptVectorCallback(
            save_dir=vector_dir + "/run",  # 프롬프트 벡터 저장 디렉토리
            prompt_vector=self.prompt_vector,
            save_steps=5  # 2 스텝마다 저장
        )

        # Trainer 설정
        training_args = TrainingArguments(
            output_dir=model_dir,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=5,
            logging_steps=5,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,
            num_train_epochs=num_epochs,
            save_strategy="no",
            report_to="wandb"
        )

        trainer = Trainer(
            model=self.musicgen_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            optimizers=(optimizer, None),
            callbacks = [save_prompt_callback]
        )

        # 학습 시작
        trainer.train()

        # 모델 저장
        # self.musicgen_model.save_pretrained(model_dir)
        # 학습된 프롬프트 벡터 저장
        torch.save(self.prompt_vector, f"{vector_dir}/prompt_vector.pt")

# 진행
if __name__ == "__main__":
    # 프롬프트 벡터 불러오기
    prompt_vector_path = r'./vectors/load/prompt_vector.pt'
    if os.path.exists(prompt_vector_path):
        prompt_vector = torch.load(prompt_vector_path)
        print("loader : load prompt vector")
    else :
        prompt_vector = None

    # MusicGenPromptTuner 클래스 초기화
    musicgen_prompt_tuner = MusicGenPromptTuner(prompt_length=30, prompt_vector=prompt_vector)

    # 생성된 데이터 셋 경로 (dataset 객체 정보 파일)
    dataset_path = r"./Datasets"

    # 생성될 모델 및 프롬프트 벡터 경로 (모델은 저장하지 않음)
    model_path = r"./tuned_model"
    vector_path = r'./vectors'

    # CUDA 사용
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    musicgen_prompt_tuner.musicgen_model.to(device)

    # 학습 시작
    musicgen_prompt_tuner.train(dataset_path=dataset_path, model_dir=model_path, vector_dir = vector_path, learning_rate=1e-5, batch_size=16, num_epochs=200)


