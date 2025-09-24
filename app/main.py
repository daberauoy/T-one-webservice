from fastapi import FastAPI, HTTPException, UploadFile, File
import tempfile
# from pydantic import BaseModel
from tone import StreamingCTCPipeline, read_audio, read_example_audio

app = FastAPI()


# параметры Whisper ASR Webservice, которые здесь не существуют. для этой модели вообще не нужны параметры
# class QueryParams(BaseModel):
    # encode: bool = True
    # task: str = "transcribe"
    # language: str = "ru"
    # initial_prompt: str = None
    # vad_filter: bool = False
    # word_timestamps: bool = False
    # output: str = "json"


@app.get("/")
def root():
    return {"hello":"world"}


@app.get("/asr/test")
def test_asr():
    # audio = read_example_audio() # or read_audio("your_audio.flac")
    audio = read_audio(r"../recordings/rec.mp3")

    pipeline = StreamingCTCPipeline.from_hugging_face()

    segments = pipeline.forward_offline(audio)  # run offline recognition

    end_return = {"segments" : segments}

    print('it is working')

    return end_return


@app.post("/asr")
async def process(
        audio_file: UploadFile = File(..., description="audio file")
):
    with tempfile.NamedTemporaryFile() as tmp_file:
        content = await audio_file.read()
        tmp_file.write(content)

        tmp_file.flush()
        tmp_file.seek(0)

        audio = read_audio(tmp_file.name)

    pipeline = StreamingCTCPipeline.from_hugging_face()

    segments = pipeline.forward_offline(audio)  # run offline recognition

    end_return = {"segments": segments}

    print('Done')

    return end_return
