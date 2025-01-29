import os
import random
import re
import torch
import soundfile as sf
import librosa
import time
from PIL import Image
from kokoro import KPipeline
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    MoonshineForConditionalGeneration,
    Wav2Vec2Processor,
)
from transformers.image_utils import load_image
from torch.cuda.amp import autocast  # For mixed precision on CUDA (will be a no-op on CPU)
import torch.quantization  # We use PyTorch's built-in dynamic quantization


class VoiceToVoicePipeline:
    def __init__(
        self,
        lang_code='a',
        voice='af_bella',
        images_folder='images',
        vision_model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
        moonshine_model_name="UsefulSensors/moonshine-tiny",
        device=None,
        voice_gen=True,
        max_audio_files=None,
        # By default, reduce max_new_tokens to 32 to speed up CPU inference
        max_new_tokens=32
    ):
        """
        Initialize the VoiceToVoicePipeline.

        Args:
            lang_code (str): Language code for KPipeline ('a' for American English, 'b' for British English).
            voice (str): Voice type for KPipeline.
            images_folder (str): Path to the folder containing images.
            vision_model_name (str): Hugging Face model name for vision processing.
            moonshine_model_name (str): Hugging Face model name for speech-to-text.
            device (str): Device to run models on ('cpu' or 'cuda'). If None, automatically detects.
            voice_gen (bool): Flag to enable or disable speech generation.
            max_audio_files (int or None): Maximum number of audio files to generate per response.
            max_new_tokens (int): Maximum number of tokens to generate for text/vision model responses.
        """
        self.lang_code = lang_code
        self.voice = voice
        self.images_folder = images_folder
        self.voice_gen = voice_gen
        self.max_audio_files = max_audio_files
        self.max_new_tokens = max_new_tokens

        # Initialize KPipeline for Text-to-Speech
        start_time = time.perf_counter()
        self.pipeline = KPipeline(lang_code=self.lang_code)
        self.pipeline_init_time = time.perf_counter() - start_time

        # Detect device (CPU or CUDA). We'll assume CPU usage if CUDA isn't available.
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize Vision Model (No bitsandbytes, just standard CPU or CUDA usage)
        start_time = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(vision_model_name)

        # Load model in FP32, then dynamically quantize linear layers for CPU
        self.vision_model = AutoModelForVision2Seq.from_pretrained(
            vision_model_name
        ).to(self.device)
        self.vision_model.eval()

        # Dynamic quantization for CPU inference (works on nn.Linear layers)
        # If device is 'cpu', this helps reduce memory usage and speeds up matmul ops.
        if self.device == "cpu":
            self.vision_model = torch.quantization.quantize_dynamic(
                self.vision_model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )

        self.vision_model_init_time = time.perf_counter() - start_time

        # Initialize Moonshine for Speech-to-Text (unquantized)
        start_time = time.perf_counter()
        self.moonshine_model = MoonshineForConditionalGeneration.from_pretrained(moonshine_model_name).to(self.device)
        self.moonshine_model.eval()  # Set to evaluation mode
        self.moonshine_processor = Wav2Vec2Processor.from_pretrained(moonshine_model_name)
        self.moonshine_init_time = time.perf_counter() - start_time

        # Define vision-related patterns
        self.vision_patterns = [
            r"\bwhat am i looking at\b",
            r"\bwhat does this image do\b",
            r"\btell me about this picture\b",
            r"\bdescribe this image\b",
            r"\bwhat is in this image\b"
        ]
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.vision_patterns]

        # Dictionary to store timing information
        self.timing_info = {}

    def transcribe_audio(self, audio_path):
        """
        Transcribe audio to text using Moonshine.
        """
        start_time = time.perf_counter()
        audio_array, sampling_rate = sf.read(audio_path)

        # Convert to mono if necessary
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        # Resample to 16000 Hz if necessary using librosa
        if sampling_rate != 16000:
            resample_start = time.perf_counter()
            audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000, res_type='kaiser_fast')
            sampling_rate = 16000
            self.timing_info['resample_time'] = time.perf_counter() - resample_start
        else:
            self.timing_info['resample_time'] = 0.0

        # Prepare inputs for Moonshine
        prepare_start = time.perf_counter()
        with torch.no_grad():
            inputs = self.moonshine_processor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt"
            ).input_values.to(self.device)
        self.timing_info['prepare_transcription_time'] = time.perf_counter() - prepare_start

        # Generate transcription (greedy decoding)
        generate_start = time.perf_counter()
        with torch.no_grad():
            generated_ids = self.moonshine_model.generate(
                inputs,
                max_length=self.max_new_tokens,  # limited for STT decode
                num_beams=1,       # Greedy decoding
                do_sample=False    # Disable sampling
            )
            transcription = self.moonshine_processor.decode(generated_ids[0], skip_special_tokens=True)
        self.timing_info['transcription_time'] = time.perf_counter() - generate_start

        total_time = time.perf_counter() - start_time
        self.timing_info['total_transcription_time'] = total_time

        print(f"Transcription: {transcription}")
        return transcription

    def is_vision_query(self, text):
        """
        Check if the transcribed text matches any vision-related patterns.
        """
        start_time = time.perf_counter()
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                self.timing_info['match_time'] = time.perf_counter() - start_time
                print(f"Matched pattern: {pattern.pattern}")
                return True
        self.timing_info['match_time'] = time.perf_counter() - start_time
        return False

    def select_random_image(self):
        """
        Select a random image from the images folder.
        """
        start_time = time.perf_counter()
        if not os.path.exists(self.images_folder):
            raise FileNotFoundError(f"Images folder '{self.images_folder}' does not exist.")

        images = [file for file in os.listdir(self.images_folder)
                  if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        if not images:
            raise FileNotFoundError(f"No images found in folder '{self.images_folder}'.")

        selected_image = random.choice(images)
        image_path = os.path.join(self.images_folder, selected_image)
        print(f"Selected image: {image_path}")
        self.timing_info['select_image_time'] = time.perf_counter() - start_time
        return image_path

    def process_image(self, image_path):
        """
        Process the image using the (quantized) vision model to generate a textual description.
        """
        start_time = time.perf_counter()
        image = load_image(image_path)

        # Resize image to reduce compute
        resize_start = time.perf_counter()
        image = image.resize((128, 128))
        self.timing_info['resize_time'] = time.perf_counter() - resize_start

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe the content of this image."}
                ]
            },
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        prepare_start = time.perf_counter()
        with torch.no_grad():
            inputs = self.processor(
                text=prompt,
                images=[image],
                return_tensors="pt"
            ).to(self.device)
        self.timing_info['prepare_image_time'] = time.perf_counter() - prepare_start

        # Generate description with greedy decoding (mixed precision if on CUDA)
        generate_start = time.perf_counter()
        if self.device == "cuda":
            with autocast():
                generated_ids = self.vision_model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=1,
                    do_sample=False
                )
        else:
            with torch.no_grad():
                generated_ids = self.vision_model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=1,
                    do_sample=False
                )
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        self.timing_info['image_generation_time'] = time.perf_counter() - generate_start

        description = generated_texts[0].strip()
        self.timing_info['process_image_time'] = time.perf_counter() - start_time

        print(f"Generated description: {description}")
        return description

    def process_text(self, text):
        """
        Process general text query using the (quantized) vision model to generate a response.
        """

        # Optionally truncate if your text is extremely long
        text = text[:512]

        start_time = time.perf_counter()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text}
                ]
            },
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        prepare_start = time.perf_counter()
        with torch.no_grad():
            inputs = self.processor(
                text=prompt,
                return_tensors="pt"
            ).to(self.device)
        self.timing_info['prepare_text_time'] = time.perf_counter() - prepare_start

        # Generate response with greedy decoding
        generate_start = time.perf_counter()
        if self.device == "cuda":
            with autocast():
                generated_ids = self.vision_model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=1,
                    do_sample=False
                )
        else:
            with torch.no_grad():
                generated_ids = self.vision_model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=1,
                    do_sample=False
                )
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        self.timing_info['text_generation_time'] = time.perf_counter() - generate_start

        response = generated_texts[0].strip()
        self.timing_info['process_text_time'] = time.perf_counter() - start_time

        print(f"Generated response: {response}")
        return response

    def generate_speech(self, text, output_path):
        """
        Generate speech from text using KPipeline and save as WAV files.
        """
        if not self.voice_gen:
            print("Voice generation is disabled. Skipping speech generation.")
            return

        start_time = time.perf_counter()
        generator = self.pipeline(
            text,
            voice=self.voice,
            speed=1,
            split_pattern=r'\n+'
        )
        generate_start = time.perf_counter()
        audio_files_generated = 0

        for i, (gs, ps, audio) in enumerate(generator):
            if self.max_audio_files is not None and audio_files_generated >= self.max_audio_files:
                print(f"Reached maximum of {self.max_audio_files} audio files. Stopping speech generation.")
                break
            print(f"Processing segment {i}: {gs}")
            sf.write(f"{output_path}_{i}.wav", audio, 24000)  # Save each segment
            audio_files_generated += 1

        self.timing_info['speech_generation_time'] = time.perf_counter() - generate_start
        self.timing_info['speech_total_time'] = time.perf_counter() - start_time

        if self.voice_gen:
            print(f"Speech generated and saved to {output_path}_0.wav and subsequent segments up to {self.max_audio_files}.")

    def handle_audio_input(self, audio_path, output_audio_path="output"):
        """
        Handle the entire pipeline from audio input to audio output.
        """
        overall_start = time.perf_counter()

        # 1. Transcribe the user audio
        t_start = time.perf_counter()
        transcription = self.transcribe_audio(audio_path)
        self.timing_info['handle_transcription_time'] = time.perf_counter() - t_start

        # 2. Check if it's a vision-related query
        v_start = time.perf_counter()
        is_vision = self.is_vision_query(transcription)
        self.timing_info['handle_vision_query_time'] = time.perf_counter() - v_start

        if is_vision:
            # 3. Select a random image
            si_start = time.perf_counter()
            image_path = self.select_random_image()
            self.timing_info['handle_select_image_time'] = time.perf_counter() - si_start

            # 4. Process the image
            pi_start = time.perf_counter()
            description = self.process_image(image_path)
            self.timing_info['handle_process_image_time'] = time.perf_counter() - pi_start

            # 5. Generate speech from the description
            gs_start = time.perf_counter()
            self.generate_speech(description, output_audio_path)
            self.timing_info['handle_generate_speech_time'] = time.perf_counter() - gs_start
        else:
            # Process as general text
            pt_start = time.perf_counter()
            response = self.process_text(transcription)
            self.timing_info['handle_process_text_time'] = time.perf_counter() - pt_start

            # Generate speech from the response
            self.generate_speech(response, output_audio_path)

        self.timing_info['overall_time'] = time.perf_counter() - overall_start

    def print_timing_info(self):
        """
        Print the collected timing information.
        """
        print("\n--- Timing Information ---")
        for key, value in self.timing_info.items():
            print(f"{key}: {value:.4f} seconds")
        print("--------------------------\n")


if __name__ == "__main__":
    import argparse
    import cProfile
    import pstats

    parser = argparse.ArgumentParser(description="Voice to Voice Pipeline (CPU Dynamic Quant)")
    parser.add_argument("--input_audio", type=str, required=True, help="Path to the input audio file.")
    parser.add_argument("--output_audio", type=str, default="output", help="Base path for the output audio file.")
    parser.add_argument("--images_folder", type=str, default="images", help="Path to the images folder.")
    parser.add_argument("--voice_gen", action='store_true', help="Enable speech generation.")
    parser.add_argument("--no_voice_gen", action='store_false', dest='voice_gen', help="Disable speech generation.")
    parser.set_defaults(voice_gen=True)
    parser.add_argument("--max_audio_files", type=int, default=None,
                        help="Maximum number of audio files to generate per response.")
    parser.add_argument("--max_new_tokens", type=int, default=32,
                        help="Maximum number of tokens to generate for responses.")

    args = parser.parse_args()

    def main():
        pipeline = VoiceToVoicePipeline(
            lang_code='a',
            voice='af_bella',
            images_folder=args.images_folder,
            vision_model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
            moonshine_model_name="UsefulSensors/moonshine-tiny",
            device=None,  # Auto-detect CPU or GPU
            voice_gen=args.voice_gen,
            max_audio_files=args.max_audio_files,
            max_new_tokens=args.max_new_tokens
        )
        pipeline.handle_audio_input(args.input_audio, args.output_audio)
        pipeline.print_timing_info()

    # cProfile instrumentation
    with cProfile.Profile() as pr:
        main()
    stats = pstats.Stats(pr).sort_stats("tottime")
    stats.print_stats(20)
