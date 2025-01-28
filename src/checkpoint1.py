import os
import random
import re
import torch
import soundfile as sf
import librosa
from PIL import Image
from kokoro import KPipeline
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from transformers import MoonshineForConditionalGeneration, Wav2Vec2Processor

class VoiceToVoicePipeline:
    def __init__(
        self,
        lang_code='a',
        voice='af_bella',
        images_folder='images',
        vision_model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
        moonshine_model_name="UsefulSensors/moonshine-tiny",
        device=None
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
        """
        self.lang_code = lang_code
        self.voice = voice
        self.images_folder = images_folder

        # Initialize KPipeline for Text-to-Speech
        self.pipeline = KPipeline(lang_code=self.lang_code)

        # Initialize Hugging Face Vision Model
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(vision_model_name)
        self.vision_model = AutoModelForVision2Seq.from_pretrained(
            vision_model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            _attn_implementation="flash_attention_2" if self.device == "cuda" else "eager",
        ).to(self.device)

        # Initialize Moonshine for Speech-to-Text
        self.moonshine_model = MoonshineForConditionalGeneration.from_pretrained(moonshine_model_name).to(self.device)
        self.moonshine_processor = Wav2Vec2Processor.from_pretrained(moonshine_model_name)

        # Define vision-related patterns for matching
        self.vision_patterns = [
            r"\bwhat am i looking at\b",
            r"\bwhat does this image do\b",
            r"\btell me about this picture\b",
            r"\bdescribe this image\b",
            r"\bwhat is in this image\b"
        ]
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.vision_patterns]

    def transcribe_audio(self, audio_path):
        """
        Transcribe audio to text using Moonshine.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            str: Transcribed text.
        """
        audio_array, sampling_rate = sf.read(audio_path)

        # Convert to mono if necessary
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        # Resample to 16000 Hz if necessary
        if sampling_rate != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
            sampling_rate = 16000

        # Prepare inputs for Moonshine
        inputs = self.moonshine_processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        ).input_values.to(self.device)

        # Generate transcription
        generated_ids = self.moonshine_model.generate(inputs, max_length=500)
        transcription = self.moonshine_processor.decode(generated_ids[0], skip_special_tokens=True)
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
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                print(f"Matched pattern: {pattern.pattern}")
                return True
        return False

    def select_random_image(self):
        """
        Select a random image from the images folder.

        Returns:
            str: Path to the selected image.
        """
        if not os.path.exists(self.images_folder):
            raise FileNotFoundError(f"Images folder '{self.images_folder}' does not exist.")

        images = [file for file in os.listdir(self.images_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        if not images:
            raise FileNotFoundError(f"No images found in folder '{self.images_folder}'.")

        selected_image = random.choice(images)
        image_path = os.path.join(self.images_folder, selected_image)
        print(f"Selected image: {image_path}")
        return image_path

    def process_image(self, image_path):
        """
        Process the image using the vision model to generate a textual description.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Generated description of the image.
        """
        image = load_image(image_path)
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
        inputs = self.processor(
            text=prompt,
            images=[image],
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        generated_ids = self.vision_model.generate(**inputs, max_new_tokens=500)
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        description = generated_texts[0].strip()
        print(f"Generated description: {description}")
        return description

    def generate_speech(self, text, output_path):
        """
        Generate speech from text using KPipeline and save as a WAV file.

        Args:
            text (str): Text to convert to speech.
            output_path (str): Path to save the generated audio file.
        """
        generator = self.pipeline(
            text, 
            voice=self.voice,
            speed=1, 
            split_pattern=r'\n+'
        )
        full_audio = b""
        for i, (gs, ps, audio) in enumerate(generator):
            print(f"Processing segment {i}: {gs}")
            sf.write(f"{output_path}_{i}.wav", audio, 24000)  # Save each segment

        # Optionally, concatenate all audio segments into one file
        # This requires handling byte concatenation properly, which may need additional processing
        # For simplicity, only individual segments are saved here

        print(f"Speech generated and saved to {output_path}_0.wav and subsequent segments if any.")

    def handle_audio_input(self, audio_path, output_audio_path="output"):
        """
        Handle the entire pipeline from audio input to audio output.

        Args:
            audio_path (str): Path to the input audio file.
            output_audio_path (str): Base path for the output audio file.
        """
        # Step 1: Transcribe the user audio
        transcription = self.transcribe_audio(audio_path)

        # Step 2: Check if it's a vision-related query
        if self.is_vision_query(transcription):
            # Step 3: Select a random image
            image_path = self.select_random_image()

            # Step 4: Process the image with the vision model
            description = self.process_image(image_path)

            # Step 5: Generate speech from the description
            self.generate_speech(description, output_audio_path)
        else:
            print("The transcribed text is not a vision-related query. No action taken.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Voice to Voice Pipeline")
    parser.add_argument("--input_audio", type=str, required=True, help="Path to the input audio file.")
    parser.add_argument("--output_audio", type=str, default="output", help="Base path for the output audio file.")
    parser.add_argument("--images_folder", type=str, default="images", help="Path to the images folder.")
    args = parser.parse_args()

    # Initialize the pipeline
    pipeline = VoiceToVoicePipeline(
        lang_code='a',
        voice='af_bella',
        images_folder=args.images_folder,
        vision_model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
        moonshine_model_name="UsefulSensors/moonshine-tiny",
        device=None  # Automatically detects 'cuda' or 'cpu'
    )

    # Handle the audio input
    pipeline.handle_audio_input(args.input_audio, args.output_audio)
