FROM python:3.10 AS build

ARG POETRY_VERSION=2.1.1

RUN set -ex \
    && curl -sSL https://install.python-poetry.org -o /get-poetry.py \
    && POETRY_HOME=/opt/poetry python /get-poetry.py

ENV PATH=/opt/poetry/bin:$PATH

COPY pyproject.toml poetry.lock README.md /workspace/
COPY tone /workspace/tone
COPY app /workspace/app

WORKDIR /workspace

RUN set -ex \
    && python -m venv /venv --without-pip \
    && . /venv/bin/activate \
    && poetry install --only main -E demo \
    # Reinstall main package in non-editable mode
    && poetry build -f wheel \
    && pip --python /venv/bin/python install dist/*.whl --no-deps --force-reinstall --no-compile \
    && rm -rf ~/.cache

ENV PATH=/venv/bin:$PATH

# RUN set -ex \
#     && python -m tone download /models

FROM python:3.10-slim

# я поверю дипсику и поменяю workdir

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY --from=build /venv /venv
COPY --from=build /workspace/app ./app
# COPY --from=build /models /models
WORKDIR /app

ENV PATH=/venv/bin:$PATH
# Set env variable LOAD_FROM_FOLDER to load model from a local folder instead of downloading from HuggingFace
# ENV LOAD_FROM_FOLDER=/models

RUN useradd -s /bin/bash python

USER python

STOPSIGNAL SIGINT

# CMD ["pwd"]
# RUN cd home
# CMD ["ls home"]
ENTRYPOINT ["uvicorn", "--host", "0.0.0.0", "--port", "8080", "--no-access-log", "main:app"]
# ENTRYPOINT ["uvicorn", "--host", "0.0.0.0", "--port", "8080", "--no-access-log", "tone.demo.website:app"]
# ENTRYPOINT ["python", "main.py"]