import os
import torch
from transformers import pipeline, is_flash_attn_2_available
from accelerate import Accelerator


class TranscriptionService:
   def __init__(self, accelerator: Accelerator):
        self.accelerator = accelerator

        # Get the model name from the environment variable
        model_name = os.getenv("MODEL_NAME", "openai/whisper-large-v3")
        attn_implementation = (
            "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
        )
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Initialize the pipeline once at startup
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device=self.device,
            model_kwargs={"attn_implementation": attn_implementation},
        )
        self.pipe = self.accelerator.prepare(self.pipe)

   def transcribe_file(self, file_path, task, language, chunk_length_s, batch_size, timestamp):
        generate_kwargs = {"task": task, "language": language}
        ts = "word" if timestamp == "word" else True

        outputs = self.pipe(
            file_path,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            generate_kwargs=generate_kwargs,
            return_timestamps=ts,
        )
        return outputs

    def transcribe_stream(
        self, audio_buffer, task, language, chunk_length_s, batch_size, timestamp
    ):
        generate_kwargs = {"task": task, "language": language}
        ts = "word" if timestamp == "word" else True

        outputs = self.pipe(
            audio_buffer,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            generate_kwargs=generate_kwargs,
            return_timestamps=ts,
        )
        return outputs
