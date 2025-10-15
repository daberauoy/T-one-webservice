# triton_client.py
import tritonclient.grpc as grpcclient
import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional


class TritonCTCModel:
    """Triton client for CTC acoustic model inference on GPU."""

    SAMPLE_RATE = 8000
    AUDIO_CHUNK_SAMPLES = 2400
    STATE_SIZE = 219729

    def __init__(self, url: str = "localhost:8001", model_name: str = "streaming_acoustic"):
        self.client = grpcclient.InferenceServerClient(url=url)
        self.model_name = model_name

    def forward(self,
                audio_chunk: npt.NDArray[np.int32],
                state: Optional[npt.NDArray[np.float16]] = None
                ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float16]]:  # ⬅️ измените тип возврата
        """Run inference via Triton Server."""

        batch_size = audio_chunk.shape[0]

        # Prepare inputs
        inputs = [
            grpcclient.InferInput("signal", audio_chunk.shape, "INT32"),
        ]
        inputs[0].set_data_from_numpy(audio_chunk)

        # Handle state - используем правильное имя входа
        if state is None:
            state = np.zeros((batch_size, self.STATE_SIZE), dtype=np.float16)

        inputs.append(grpcclient.InferInput("state", state.shape, "FP16"))
        inputs[1].set_data_from_numpy(state)

        # Prepare outputs - используем правильные имена выходов
        outputs = [
            grpcclient.InferRequestedOutput("logprobs"),  # ⬅️ правильное имя
            grpcclient.InferRequestedOutput("state_next"),  # ⬅️ правильное имя
        ]

        # Send request
        response = self.client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs,
        )

        # Extract results - правильные имена
        logprobs = response.as_numpy("logprobs")  # FP32
        new_state = response.as_numpy("state_next")  # FP16

        return logprobs, new_state

    def health_check(self) -> bool:
        """Check if Triton server is healthy."""
        try:
            return self.client.is_server_live()
        except:
            return False

    @classmethod
    def from_triton(cls, url: str, model_name: str) -> "TritonCTCModel":
        """Factory method to create Triton client."""
        return cls(url, model_name)