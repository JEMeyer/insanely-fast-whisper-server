import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import logging
import os
import time
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


@app.middleware("http")
async def log_duration(request: Request, call_next):
    start_time = time.time()

    # process request
    response = await call_next(request)

    # calculate duration
    duration = time.time() - start_time
    logger.info(f"Request to {request.url.path} took {duration:.2f} seconds")

    return response


# Initialize the pipeline
# Assuming 1 gpu (will always be 0 since it'll be dockerized)
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    torch_dtype=torch.float16,
    device="cuda:0",
    model_kwargs={"attn_implementation": "flash_attention_2"}
    if is_flash_attn_2_available()
    else {"attn_implementation": "sdpa"},
)

logger.info("GPU initialized")


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # Save the file to a temporary location
        with open("temp_audio.wav", "wb") as temp_file:
            temp_file.write(contents)

        # Use the pipeline to transcribe the audio file
        outputs = pipe(
            "temp_audio.wav",
            chunk_length_s=30,
            batch_size=24,
            return_timestamps=True,
        )

        # Remove the temporary file
        os.remove("temp_audio.wav")

        return JSONResponse(content=outputs)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/transcribe/stream")
async def transcribe_stream(request: Request):
    try:
        # Use an in-memory buffer to collect streamed data
        audio_buffer = io.BytesIO()

        # Define a generator to yield transcriptions incrementally
        async def transcribe_generator():
            async for chunk in request.stream():
                audio_buffer.write(chunk)

                # Write the current buffer to a temporary file
                audio_buffer.seek(0)
                with open("temp_audio_stream.wav", "wb") as temp_file:
                    temp_file.write(audio_buffer.read())

                # Use the pipeline to transcribe the current audio buffer
                outputs = pipe(
                    "temp_audio_stream.wav",
                    chunk_length_s=30,
                    batch_size=24,
                    return_timestamps=True,
                )

                # Remove the temporary file
                os.remove("temp_audio_stream.wav")

                yield JSONResponse(content=outputs)

        return StreamingResponse(transcribe_generator(), media_type="application/json")

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
