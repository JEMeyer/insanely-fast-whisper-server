import io
import logging
import time
from fastapi import FastAPI, Request, UploadFile, File, Body
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from transcription_service import TranscriptionService
from utils import save_temp_file, remove_temp_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

transcription_service = TranscriptionService()


@app.middleware("http")
async def log_duration(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"Request to {request.url.path} took {duration:.2f} seconds")
    return response


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Body("openai/whisper-large-v3"),
    task: str = Body("transcribe"),
    language: str = Body(None),
    chunk_length_s: int = Body(30),
    batch_size: int = Body(24),
    timestamp: str = Body("word"),
):
    try:
        contents = await file.read()
        temp_file_path = save_temp_file(contents, "temp_audio.wav")

        outputs = transcription_service.transcribe_file(
            temp_file_path,
            model,
            task,
            language,
            chunk_length_s,
            batch_size,
            timestamp,
        )

        remove_temp_file(temp_file_path)
        return JSONResponse(content=outputs)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/transcribe/stream")
async def transcribe_stream(
    request: Request,
    model: str = Body("openai/whisper-large-v3"),
    task: str = Body("transcribe"),
    language: str = Body(None),
    chunk_length_s: int = Body(30),
    batch_size: int = Body(24),
    timestamp: str = Body("word"),
):
    try:
        audio_buffer = io.BytesIO()

        async def transcribe_generator():
            async for chunk in request.stream():
                audio_buffer.write(chunk)
                audio_buffer.seek(0)

                outputs = transcription_service.transcribe_stream(
                    audio_buffer,
                    model,
                    task,
                    language,
                    chunk_length_s,
                    batch_size,
                    timestamp,
                )
                yield JSONResponse(content=outputs)

        return StreamingResponse(transcribe_generator(), media_type="application/json")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
