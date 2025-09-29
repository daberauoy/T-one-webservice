"""Module with wrapper for pretrained CTC acoustic model using Triton Inference Server."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import numpy.typing as npt
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
from typing_extensions import Self, TypeAlias


class TritonCTCModel:
    """Wrapper for a pretrained CTC acoustic model, running with Triton Inference Server.

    This class handles inference with a CTC-based acoustic model via Triton server
    and supports batched streaming inputs. It provides methods to connect to Triton
    server and exposes a forward method to compute log-probabilities from audio chunks.
    """

    InputType: TypeAlias = npt.NDArray[np.int32]
    OutputType: TypeAlias = npt.NDArray[np.float16]
    StateType: TypeAlias = npt.NDArray[np.float16]

    SAMPLE_RATE = 8000
    MEAN_TIME_BIAS = 0.33  # in seconds
    AUDIO_CHUNK_SAMPLES = 2400  # in audio samples
    FRAME_SIZE = 0.03  # in seconds
    STATE_SIZE = 219729

    _client: httpclient.InferenceServerClient
    _model_name: str
    _model_version: str

    def __init__(
            self,
            client: httpclient.InferenceServerClient,
            model_name: str = "ctc_acoustic_model",
            model_version: str = "1"
    ) -> None:
        """Create instance of TritonCTCModel.

        Args:
            client: Triton HTTP client instance
            model_name: Name of the model in Triton model repository
            model_version: Version of the model to use
        """
        self._client = client
        self._model_name = model_name
        self._model_version = model_version

    @classmethod
    def from_server(
            cls,
            url: str = "localhost:8000",
            model_name: str = "ctc_acoustic_model",
            model_version: str = "1",
            verbose: bool = False
    ) -> Self:
        """Initialize the model by connecting to Triton Inference Server.

        Args:
            url: Triton server URL (host:port)
            model_name: Name of the model in Triton model repository
            model_version: Version of the model to use
            verbose: Whether to enable verbose logging

        Returns:
            Self: An instance of TritonCTCModel ready for inference.

        Raises:
            Exception: If connection to Triton server fails or model is not ready
        """
        client = httpclient.InferenceServerClient(url=url, verbose=verbose)

        # Check if server is ready
        if not client.is_server_ready():
            raise RuntimeError("Triton server is not ready")

        # Check if model is ready
        if not client.is_model_ready(model_name, model_version):
            raise RuntimeError(f"Model {model_name} version {model_version} is not ready")

        return cls(client, model_name, model_version)

    def forward(
            self,
            audio_chunk: InputType,
            state: StateType | None = None
    ) -> tuple[OutputType, StateType]:
        """Run the CTC acoustic model on a single audio chunk via Triton server.

        Converts raw audio to frame-level log-probabilities using Triton Inference Server.
        Maintains model state for streaming.

        Args:
            audio_chunk: A batch or single audio chunk to process. Shape: (B, AUDIO_CHUNK_SAMPLES, 1)
            state: Previous state, or None to initialize. Shape: (B, STATE_SIZE)

        Returns:
            Tuple containing:
                - OutputType: Model log-probabilities for each frame
                - StateType: Updated state to pass into the next call

        Raises:
            TypeError: If input types are incorrect
            ValueError: If input shapes or values are invalid
            Exception: If Triton inference fails
        """
        # Input validation
        if not isinstance(audio_chunk, np.ndarray):
            raise TypeError(f"Incorrect 'audio_chunk' type: expected np.ndarray, but got {type(audio_chunk)}")

        if audio_chunk.shape[1:] != (self.AUDIO_CHUNK_SAMPLES, 1):
            raise ValueError(
                f"Shape of 'audio_chunk' must be (B, {self.AUDIO_CHUNK_SAMPLES}, 1), but got {audio_chunk.shape}",
            )

        if audio_chunk.dtype != np.int32:
            raise ValueError(f"Incorrect dtype of 'audio_chunk': expected np.int32, but got {audio_chunk.dtype}")

        if audio_chunk.min() < -32768 or audio_chunk.max() > 32767:
            raise ValueError(
                "Samples in 'audio_chunk' must be in range [-32768; 32767], "
                f"but it is in range [{audio_chunk.min()}; {audio_chunk.max()}]",
            )

        batch_size = audio_chunk.shape[0]

        # Initialize state if not provided
        if state is None:
            state = np.zeros((batch_size, self.STATE_SIZE), dtype=np.float16)

        if not isinstance(state, np.ndarray):
            raise TypeError(f"Incorrect 'state' type: expected np.ndarray or None, but got {type(state)}")

        if state.shape != (batch_size, self.STATE_SIZE):
            raise ValueError(f"Shape of 'state' must be ({batch_size}, {self.STATE_SIZE}), but got {state.shape}")

        if state.dtype != np.float16:
            raise ValueError(f"Incorrect dtype of 'state': expected np.float16, but got {state.dtype}")

        # Prepare inputs for Triton
        inputs = [
            httpclient.InferInput(
                "signal",
                audio_chunk.shape,
                np_to_triton_dtype(audio_chunk.dtype)
            ),
            httpclient.InferInput(
                "state",
                state.shape,
                np_to_triton_dtype(state.dtype)
            )
        ]

        inputs[0].set_data_from_numpy(audio_chunk)
        inputs[1].set_data_from_numpy(state)

        # Perform inference
        try:
            response = self._client.infer(
                model_name=self._model_name,
                model_version=self._model_version,
                inputs=inputs
            )

            # Extract outputs (assuming model returns logits and updated state)
            # You may need to adjust output names based on your model configuration
            log_probs = response.as_numpy("log_probs")  # Adjust output name as needed
            updated_state = response.as_numpy("output_state")  # Adjust output name as needed

            return log_probs, updated_state

        except Exception as e:
            raise RuntimeError(f"Triton inference failed: {e}")

    def get_model_config(self) -> dict:
        """Get the model configuration from Triton server.

        Returns:
            dict: Model configuration
        """
        return self._client.get_model_config(self._model_name, self._model_version)

    def get_server_metadata(self) -> dict:
        """Get Triton server metadata.

        Returns:
            dict: Server metadata
        """
        return self._client.get_server_metadata()

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def model_version(self) -> str:
        """Get the model version."""
        return self._model_version