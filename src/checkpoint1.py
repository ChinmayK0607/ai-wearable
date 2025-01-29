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
    Wav2Vec2Processor
)
from transformers.image_utils import load_image
from torch.cuda.amp import autocast  # For mixed precision

# Import quantization to dynamically quantize the vision model
import torch.quantization


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
        max_new_tokens=100  # Further reduced
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
            max_audio_files (int or None): Maximum number of audio files to generate per response. If None, no limit.
            max_new_tokens (int): Maximum number of tokens to generate for responses.
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

        # Initialize Hugging Face Vision Model
        start_time = time.perf_counter()
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(vision_model_name)
        self.vision_model = AutoModelForVision2Seq.from_pretrained(
            vision_model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            _attn_implementation="flash_attention_2" if self.device == "cuda" else "eager",
        ).to(self.device)
        self.vision_model.eval()  # Set to evaluation mode

        # ---- DYNAMIC QUANTIZATION FOR THE VISION MODEL ----
        # We apply dynamic quantization to reduce CPU memory usage and CPU load.
        # (Only affects linear layers by default.)
        self.vision_model = torch.quantization.quantize_dynamic(
            self.vision_model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        # --------------------------------------------------

        self.vision_model_init_time = time.perf_counter() - start_time

        # Initialize Moonshine for Speech-to-Text (NO QUANTIZATION APPLIED HERE)
        start_time = time.perf_counter()
        self.moonshine_model = MoonshineForConditionalGeneration.from_pretrained(moonshine_model_name).to(self.device)
        self.moonshine_model.eval()  # Set to evaluation mode
        self.moonshine_processor = Wav2Vec2Processor.from_pretrained(moonshine_model_name)
        self.moonshine_init_time = time.perf_counter() - start_time

        # Define vision-related patterns for matching
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

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            str: Transcribed text.
        """
        start_time = time.perf_counter()
        audio_array, sampling_rate = sf.read(audio_path)

        # Convert to mono if necessary
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        # Resample to 16000 Hz if necessary using librosa with 'kaiser_fast'
        if sampling_rate != 16000:
            resample_start = time.perf_counter()
            audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000, res_type='kaiser_fast')
            sampling_rate = 16000
            resample_time = time.perf_counter() - resample_start
            self.timing_info['resample_time'] = resample_time
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
        prepare_time = time.perf_counter() - prepare_start
        self.timing_info['prepare_transcription_time'] = prepare_time

        # Generate transcription with greedy decoding
        generate_start = time.perf_counter()
        with torch.no_grad():
            generated_ids = self.moonshine_model.generate(
                inputs,
                max_length=self.max_new_tokens,
                num_beams=1,       # Greedy decoding
                do_sample=False    # Disable sampling
            )
            transcription = self.moonshine_processor.decode(generated_ids[0], skip_special_tokens=True)
        generate_time = time.perf_counter() - generate_start
        self.timing_info['transcription_time'] = generate_time

        total_time = time.perf_counter() - start_time
        self.timing_info['total_transcription_time'] = total_time

        print(f"Transcription: {transcription}")
        return transcription

    def is_vision_query(self, text):
        """
        Check if the transcribed text matches any vision-related patterns.

        Args:
            text (str): Transcribed text.

        Returns:
            bool: True if it's a vision-related query, False otherwise.
        """
        start_time = time.perf_counter()
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                match_time = time.perf_counter() - start_time
                self.timing_info['match_time'] = match_time
                print(f"Matched pattern: {pattern.pattern}")
                return True
        self.timing_info['match_time'] = time.perf_counter() - start_time
        return False

    def select_random_image(self):
        """
        Select a random image from the images folder.

        Returns:
            str: Path to the selected image.
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
        select_time = time.perf_counter() - start_time
        self.timing_info['select_image_time'] = select_time
        return image_path

    def process_image(self, image_path):
        """
        Process the image using the vision model to generate a textual description.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Generated description of the image.
        """
        start_time = time.perf_counter()
        image = load_image(image_path)

        # Further resize image to reduce processing time
        resize_start = time.perf_counter()
        image = image.resize((128, 128))  # Further reduced size
        resize_time = time.perf_counter() - resize_start
        self.timing_info['resize_time'] = resize_time

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
            )
            inputs = inputs.to(self.device)
        prepare_time = time.perf_counter() - prepare_start
        self.timing_info['prepare_image_time'] = prepare_time

        # Generate description with greedy decoding and mixed precision if possible
        generate_start = time.perf_counter()
        if self.device == "cuda":
            with autocast():
                generated_ids = self.vision_model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=1,      # Greedy decoding
                    do_sample=False   # Disable sampling
                )
        else:
            with torch.no_grad():
                generated_ids = self.vision_model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=1,
                    do_sample=False
                )
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        generate_time = time.perf_counter() - generate_start
        self.timing_info['image_generation_time'] = generate_time

        description = generated_texts[0].strip()
        process_time = time.perf_counter() - start_time
        self.timing_info['process_image_time'] = process_time

        print(f"Generated description: {description}")
        return description

    def process_text(self, text):
        """
        Process the text query using the vision model to generate a response.

        Args:
            text (str): Text query.

        Returns:
            str: Generated response from the vision model.
        """
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
            )
            inputs = inputs.to(self.device)
        prepare_time = time.perf_counter() - prepare_start
        self.timing_info['prepare_text_time'] = prepare_time

        # Generate response with greedy decoding and mixed precision if possible
        generate_start = time.perf_counter()
        if self.device == "cuda":
            with autocast():
                generated_ids = self.vision_model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=1,      # Greedy decoding
                    do_sample=False   # Disable sampling
                )
        else:
            with torch.no_grad():
                generated_ids = self.vision_model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=1,
                    do_sample=False
                )
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        generate_time = time.perf_counter() - generate_start
        self.timing_info['text_generation_time'] = generate_time

        response = generated_texts[0].strip()
        process_time = time.perf_counter() - start_time
        self.timing_info['process_text_time'] = process_time

        print(f"Generated response: {response}")
        return response

    def generate_speech(self, text, output_path):
        """
        Generate speech from text using KPipeline and save as WAV files.

        Args:
            text (str): Text to convert to speech.
            output_path (str): Base path to save the generated audio files.
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

        generate_time = time.perf_counter() - generate_start
        self.timing_info['speech_generation_time'] = generate_time

        total_time = time.perf_counter() - start_time
        self.timing_info['speech_total_time'] = total_time

        if self.voice_gen:
            print(f"Speech generated and saved to {output_path}_0.wav and subsequent segments up to {self.max_audio_files}.")
        return

    def handle_audio_input(self, audio_path, output_audio_path="output"):
        """
        Handle the entire pipeline from audio input to audio output.

        Args:
            audio_path (str): Path to the input audio file.
            output_audio_path (str): Base path for the output audio file.
        """
        overall_start = time.perf_counter()

        # Step 1: Transcribe the user audio
        transcription_start = time.perf_counter()
        transcription = self.transcribe_audio(audio_path)
        transcription_time = time.perf_counter() - transcription_start
        self.timing_info['handle_transcription_time'] = transcription_time

        # Step 2: Check if it's a vision-related query
        vision_query_start = time.perf_counter()
        is_vision = self.is_vision_query(transcription)
        vision_query_time = time.perf_counter() - vision_query_start
        self.timing_info['handle_vision_query_time'] = vision_query_time

        if is_vision:
            # Step 3: Select a random image
            select_image_start = time.perf_counter()
            image_path = self.select_random_image()
            select_image_time = time.perf_counter() - select_image_start
            self.timing_info['handle_select_image_time'] = select_image_time

            # Step 4: Process the image with the vision model
            process_image_start = time.perf_counter()
            description = self.process_image(image_path)
            process_image_time = time.perf_counter() - process_image_start
            self.timing_info['handle_process_image_time'] = process_image_time

            # Step 5: Generate speech from the description
            generate_speech_start = time.perf_counter()
            self.generate_speech(description, output_audio_path)
            generate_speech_time = time.perf_counter() - generate_speech_start
            self.timing_info['handle_generate_speech_time'] = generate_speech_time
        else:
            # Process as a general text query using SmolVLM
            process_text_start = time.perf_counter()
            response = self.process_text(transcription)
            process_text_time = time.perf_counter() - process_text_start
            self.timing_info['handle_process_text_time'] = process_text_time

            # Generate speech from the response
            self.generate_speech(response, output_audio_path)

        overall_time = time.perf_counter() - overall_start
        self.timing_info['overall_time'] = overall_time

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

    parser = argparse.ArgumentParser(description="Voice to Voice Pipeline with Timing and Control Flags")
    parser.add_argument("--input_audio", type=str, required=True, help="Path to the input audio file.")
    parser.add_argument("--output_audio", type=str, default="output", help="Base path for the output audio file.")
    parser.add_argument("--images_folder", type=str, default="images", help="Path to the images folder.")
    parser.add_argument("--voice_gen", action='store_true', help="Enable speech generation.")
    parser.add_argument("--no_voice_gen", action='store_false', dest='voice_gen', help="Disable speech generation.")
    parser.set_defaults(voice_gen=True)
    parser.add_argument("--max_audio_files", type=int, default=None,
                        help="Maximum number of audio files to generate per response.")
    args = parser.parse_args()

    def main():
        # Initialize the pipeline
        pipeline = VoiceToVoicePipeline(
            lang_code='a',
            voice='af_bella',
            images_folder=args.images_folder,
            vision_model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
            moonshine_model_name="UsefulSensors/moonshine-tiny",
            device=None,  # Automatically detects 'cuda' or 'cpu'
            voice_gen=args.voice_gen,
            max_audio_files=args.max_audio_files
        )

        # Handle the audio input
        pipeline.handle_audio_input(args.input_audio, args.output_audio)

        # Print timing information
        pipeline.print_timing_info()

    # ---- cProfile PROFILING ----
    with cProfile.Profile() as pr:
        main()
    stats = pstats.Stats(pr).sort_stats("tottime")
    # Print top 20 functions sorted by total time
    stats.print_stats(20)
