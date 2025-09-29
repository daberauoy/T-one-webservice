# tone/onnx_wrapper_triton.py
from __future__ import annotations
from typing import Optional
import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from typing_extensions import Self, TypeAlias

class StreamingCTCModel:
    InputType: TypeAlias = np.ndarray
    OutputType: TypeAlias = np.ndarray
    StateType: TypeAlias = np.ndarray

    SAMPLE_RATE = 8000
    MEAN_TIME_BIAS = 0.33
    AUDIO_CHUNK_SAMPLES = 2400
    FRAME_SIZE = 0.03
    STATE_SIZE = 219729

    def __init__(self, client: InferenceServerClient, model_name: str):
        self.client = client
        self.model_name = model_name

    @classmethod
    def from_hugging_face(cls, *, triton_url: str = "localhost:8000", model_name: str = "streaming_acoustic") -> Self:
        client = InferenceServerClient(url=triton_url)
        return cls(client, model_name)

    @classmethod
    def from_local(cls, model_path, *, triton_url: str = "localhost:8000", model_name: str = "streaming_acoustic") -> Self:
        # model_path not used for Triton but keep signature compatible
        client = InferenceServerClient(url=triton_url)
        return cls(client, model_name)

    def forward(self, audio_chunk: np.ndarray, state: Optional[np.ndarray] = None):
        """
        audio_chunk: shape (B, 2400, 1), dtype np.int32
        state: shape (B, STATE_SIZE), dtype np.float16 or None
        returns: (logits, next_state)
        """
        if audio_chunk.dtype != np.int32:
            raise ValueError("audio_chunk must be np.int32")
        batch_size = audio_chunk.shape[0]

        if state is None:
            state = np.zeros((batch_size, self.STATE_SIZE), dtype=np.float16)
        if state.dtype != np.float16:
            state = state.astype(np.float16)

        # Prepare Triton inputs
        inp_signal = InferInput("signal", audio_chunk.shape, "INT32")
        inp_signal.set_data_from_numpy(audio_chunk)

        inp_state = InferInput("state", state.shape, "FP16")
        inp_state.set_data_from_numpy(state)

        outputs = [InferRequestedOutput("logits"), InferRequestedOutput("next_state")]

        # Blocking call (triton client is synchronous)
        result = self.client.infer(model_name=self.model_name, inputs=[inp_signal, inp_state], outputs=outputs)

        logits = result.as_numpy("logits")        # shape [B, T, V]
        next_state = result.as_numpy("next_state")  # shape [B, STATE_SIZE]

        return logits, next_state
