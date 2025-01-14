from transformers import MusicgenForConditionalGeneration, T5Tokenizer
import scipy
import torch
import os

def load_tokenizer():
    return T5Tokenizer.from_pretrained('t5-base')

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    return device

def generate_original_audio(model_path, inputs, device, output_path):
    print(f"\nGenerating music with the original model for {output_path}...")
    model = MusicgenForConditionalGeneration.from_pretrained(model_path).to(device)
    audio = model.generate(input_ids=inputs.to(device), do_sample=True, guidance_scale=1, max_new_tokens=1536)
    sampling_rate = model.config.audio_encoder.sampling_rate
    scipy.io.wavfile.write(output_path, rate=sampling_rate, data=audio[0, 0].cpu().numpy())
    print(f"Original model output saved to {output_path}")

def generate_tuned_audio(model_path, inputs, prompt_vector_path, prompt_length, device, output_path):
    print(f"\nGenerating music with the tuned model for {output_path}...")
    model = MusicgenForConditionalGeneration.from_pretrained(model_path).to(device)

    # Load prompt vector
    prompt_vector = torch.load(prompt_vector_path, map_location=device)

    # Forward hook
    def forward_hook(module, input):
        original_input = input[0]
        batch_size, _, hidden_size = original_input.shape
        expanded_prompt = prompt_vector.expand(batch_size, prompt_length, hidden_size).to(device)
        original_input[:, :prompt_length, :] = expanded_prompt
        return (original_input,)

    model.text_encoder.encoder.block[0].register_forward_pre_hook(forward_hook)

    # Prepare inputs with prompt space
    batch_size, _ = inputs.shape
    prompt_space = torch.zeros((batch_size, prompt_length), dtype=inputs.dtype).to(device)
    final_inputs = torch.cat([prompt_space, inputs.to(device)], dim=1)

    # Generate audio
    audio = model.generate(input_ids=final_inputs, do_sample=True, guidance_scale=3, max_new_tokens=1536)
    sampling_rate = model.config.audio_encoder.sampling_rate
    scipy.io.wavfile.write(output_path, rate=sampling_rate, data=audio[0, 0].cpu().numpy())
    print(f"Tuned model output saved to {output_path}")

def sanitize_filename(prompt):
    return "_".join(prompt.split()[:3])

def generate_music_batch(prompts, ori_model_path, prompt_vector_path, output_dir):
    tokenizer = load_tokenizer()
    device = setup_device()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for prompt in prompts:
        print(f"\nProcessing prompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").input_ids
        sanitized_name = sanitize_filename(prompt)
        output_ori = os.path.join(output_dir, f"{sanitized_name}_ori.wav")
        output_tuned = os.path.join(output_dir, f"{sanitized_name}_tuned.wav")
        output_txt = os.path.join(output_dir, f"{sanitized_name}.txt")

        # Save prompt text
        with open(output_txt, "w") as txt_file:
            txt_file.write(prompt)

        # Generate and save audio
        generate_original_audio(ori_model_path, inputs, device, output_ori)

        prompt_length = 30  # Adjust prompt length as needed
        generate_tuned_audio(ori_model_path, inputs, prompt_vector_path, prompt_length, device, output_tuned)

# Example usage
if __name__ == "__main__":
    prompts = [
        # 악기 기반 프롬프트
        "Create a song featuring the Gayageum in a serene atmosphere.",
        "Design a melody centered on the deep tones of the Geomungo.",
        "Compose a vibrant tune highlighting the plucking of the Bipa.",
        "Write a traditional piece emphasizing the Cheolhyeongeum.",
        "Generate music with the ancient elegance of the Geum.",
    ]


    ori_model_path = 'facebook/musicgen-small'
    prompt_vector_path = './tuned_vector/prompt_vector.pt'
    output_dir = './results/'

    generate_music_batch(prompts, ori_model_path, prompt_vector_path, output_dir)
