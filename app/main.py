from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import PlainTextResponse
import tempfile
import logging
import subprocess
import os
import datetime
from datetime import datetime as dt

# from pydantic import BaseModel
# добавим triton
from tone.triton_pipeline import TritonCTCPipeline
from tone import StreamingCTCPipeline, read_audio, read_example_audio
from tone.decoder import DecoderType

from split import delete_files
from split import split as split_func

from pathlib import Path

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

# параметры Whisper ASR Webservice, которые здесь не существуют. для этой модели вообще не нужны параметры
# class QueryParams(BaseModel):
    # encode: bool = True
    # task: str = "transcribe"
    # language: str = "ru"
    # initial_prompt: str = None
    # vad_filter: bool = False
    # word_timestamps: bool = False
    # output: str = "json"

def log_datetime():
    logger.info(f"{dt.now().date()} at {dt.now().time().hour}:{dt.now().time().minute}")

@app.on_event("startup")
def load_model():
    global pipeline
    logger.info("Loading ASR pipeline")
    # try:
    #     logger.info("Trying to load on CUDA..")
    #     pipeline = StreamingCTCPipeline.from_hugging_face(device="cuda")
    #     logger.info("Loaded on CUDA")
    # except Exception as e:
    #     logger.warning(f"Failed loading on CUDA: {e}. Trying to load default..")
    #     pipeline = StreamingCTCPipeline.from_hugging_face()


    # pipeline = StreamingCTCPipeline.from_hugging_face(decoder_type = DecoderType.GREEDY)
    # pipeline = StreamingCTCPipeline.from_hugging_face() #BeamSearch
    # logger.info("ASR pipeline loaded")

    try:
        logger.info("Trying to load Triton GPU pipeline...")
        pipeline = TritonCTCPipeline.from_triton(
            triton_url="0.0.0.0:8001",  # или ваш URL Triton сервера
            model_name="streaming_acoustic",
            decoder_type=DecoderType.BEAM_SEARCH
        )
        logger.info("Triton GPU pipeline loaded successfully!")

        # Проверка здоровья Triton сервера
        if pipeline.health_check():
            logger.info("Triton server is healthy")
        else:
            logger.warning("Triton server health check failed")

    except Exception as e:
        logger.error(f"Failed to load Triton GPU pipeline: {e}")
        logger.info("Falling back to CPU pipeline...")
        # Fallback на CPU версию
        pipeline = StreamingCTCPipeline.from_hugging_face()
        logger.info("CPU pipeline loaded as fallback")

    log_datetime()


@app.get("/")
def root():
    return {"hello":"world"}


@app.get("/asr/test")
def test_asr():
    # audio = read_example_audio() # or read_audio("your_audio.flac")
    audio = read_audio(r"../recordings/rec.mp3")

    # pipeline = StreamingCTCPipeline.from_hugging_face()

    segments = pipeline.forward_offline(audio)  # run offline recognition

    end_return = {"segments" : segments}

    logger.info("Test ASR completed")

    return end_return


@app.post("/asr")
async def process(
        audio_file: UploadFile = File(..., description="audio file"),
        split: bool = True,
        txt_output: bool = True
):
    logger.info("-------------------------")
    log_datetime()
    logger.info(f"Received file: {audio_file.filename}")
    splitting = str("Split into 2 channels? " + ("Yes" if split else "No"))
    logger.info(splitting)
    if not split:
        logger.info("Received a request, saving to temp file")
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.filename.split('.')[-1]}") as tmp_file:
            content = await audio_file.read()
            tmp_file.write(content)

            tmp_file.flush()
            tmp_file.seek(0)

            audio = read_audio(tmp_file.name)

        logger.info(f"filename: {tmp_file.name}")
        try:
            os.unlink(tmp_file.name)
            logger.info(f"Deleted temp file at: {tmp_file.name}")
        except Exception as e:
            logger.warning(f"Couldn't delete temp file: {e}")

        logger.info("Running recognition")
        segments = pipeline.forward_offline(audio)  # run offline recognition

        logger.info("Recognition completed!")
        end_return = {"segments": segments}

        for seg in segments:
            print(" - ", str(getattr(seg, "text", "")))

        return end_return

    else:
        logger.warning("Please don't use 'split' parameter.")
        return {"Please don't use 'split' parameter"}
        # disable split for now
        logger.info("Received a request, saving to temp file")

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.filename.split('.')[-1]}") as tmp_file:
            content = await audio_file.read()
            tmp_file.write(content)
            tmp_file.flush()
            tmp_filepath = tmp_file.name

        logger.info(f"Temp file created: {tmp_filepath}")

        format = split_func(tmp_filepath)

        try:
            os.unlink(tmp_filepath)
            logger.info(f"Deleted temp file at: {tmp_filepath}")
        except Exception as e:
            logger.warning(f"Couldn't delete file: {e}")


        left_audio = read_audio(f"left.{format}")
        right_audio = read_audio(f"right.{format}")

        logger.info("Running recognition (left channel)")
        segments_left = pipeline.forward_offline(left_audio)
        logger.info("Running recognition (right channel)")
        segments_right = pipeline.forward_offline(right_audio)

        delete_files([f"left.{format}", f"right.{format}"])

        logger.info("Recognition completed!")
        dialog = {"segments left": segments_left, "segments right": segments_right}

        end_return = dialog["segments left"] + dialog["segments right"]

        log_datetime()
        print_dialog(dialog)
        if txt_output:
            end_filepath = output_dialog(dialog, audio_file.filename)
            end_text = """"""
            try:
                with open(end_filepath, "r", encoding="utf-8") as file:
                    end_text = file.read()
                return PlainTextResponse(end_text)
            except Exception as e:
                logger.warning(f"Failed to open file: {e}")
                logger.info("Returning JSON")
                return end_return

        return end_return


def print_dialog(segments):
    segments_left = segments["segments left"]
    segments_right = segments["segments right"]
    seg_timestamps = []
    for seg in segments_left:
        seg_timestamps.append(
            (
                getattr(seg, "start_time", 0),
                f"--1:  {getattr(seg, 'text', '')}"
            )
        )
    for seg in segments_right:
        seg_timestamps.append(
            (
                getattr(seg, "start_time", 0),
                f"--2:  {getattr(seg, 'text', '')}"
            )
        )
    seg_timestamps.sort(key=lambda x: x[0])

    for time, text in seg_timestamps:
        print(f"        ({time}s)")
        print(text)

def output_dialog(segments, filename):
    segments_left = segments["segments left"]
    segments_right = segments["segments right"]
    seg_timestamps = []
    for seg in segments_left:
        text = getattr(seg, "text", "")
        delim = "=" * (len(text) % 100)
        seg_timestamps.append(
            (
                getattr(seg, "start_time", 0),
                f"--1:  {delim}\n      {text}\n----  {delim}"
            )
        )
    for seg in segments_right:
        text = getattr(seg, "text", "")
        delim = "=" * (len(text) if len(text) <= 100 else 100)
        margin = ' ' * 30
        seg_timestamps.append(
            (
                getattr(seg, "start_time", 0),
                f"{margin}--2:  {delim}\n      {margin}{text}\n{margin}----  {delim}"
            )
        )
    seg_timestamps.sort(key=lambda x: x[0])

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{'.'.join(filename.split('.')[:-1])}.txt"
    print(output_path)

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(f"{filename}\n")
        for time, text in seg_timestamps:
            file.write(f" [{time}s]\n")
            file.write(f"{text}\n")

    return output_path