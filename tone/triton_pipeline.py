# triton_pipeline.py
"""Module with Triton GPU implementation of T-one pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Optional, List
import logging

import numpy as np
import numpy.typing as npt
from typing_extensions import Self, TypeAlias

from tone.decoder import BeamSearchCTCDecoder, DecoderType, GreedyCTCDecoder
from tone.logprob_splitter import StreamingLogprobSplitter

if TYPE_CHECKING:
    from tone.triton_client import TritonCTCModel

logger = logging.getLogger(__name__)


@dataclass
class TextPhrase:
    """Dataclass for the phrase from ASR pipeline."""
    text: str
    start_time: float  # in seconds
    end_time: float  # in seconds


class TritonCTCPipeline:
    """A streaming ASR pipeline for CTC-based models with Triton GPU inference."""

    # Model parameters (same as original)
    SAMPLE_RATE = 8000
    PADDING: int = 2400  # 300ms * 8KHz
    CHUNK_SIZE: int = 2400
    FRAME_SIZE = 0.03  # in seconds
    MEAN_TIME_BIAS = 0.33  # in seconds

    InputType: TypeAlias = npt.NDArray[np.int32]
    OutputType: TypeAlias = List[TextPhrase]
    StateType: TypeAlias = Tuple[npt.NDArray[np.float16], StreamingLogprobSplitter.StateType]

    @classmethod
    def from_triton(
            cls,
            triton_url: str = "localhost:8001",
            model_name: str = "streaming_acoustic",
            *,
            decoder_type: DecoderType = DecoderType.BEAM_SEARCH,
            local_model_dir: str | Path | None = None
    ) -> Self:
        """Creates a pipeline instance using Triton Server for GPU inference."""
        from tone.triton_client import TritonCTCModel

        # Initialize Triton client
        model = TritonCTCModel(triton_url, model_name)
        logger.info(f"Initialized Triton client for model {model_name} at {triton_url}")

        # Initialize logprob splitter and decoder
        logprob_splitter = StreamingLogprobSplitter()

        if decoder_type == DecoderType.GREEDY:
            decoder = GreedyCTCDecoder()
        elif decoder_type == DecoderType.BEAM_SEARCH:
            if local_model_dir:
                decoder = BeamSearchCTCDecoder.from_local(Path(local_model_dir) / "kenlm.bin")
            else:
                decoder = BeamSearchCTCDecoder.from_hugging_face()
        else:
            raise ValueError("Unknown decoder type")

        return cls(model, logprob_splitter, decoder)

    def __init__(
            self,
            model: "TritonCTCModel",
            logprob_splitter: StreamingLogprobSplitter,
            decoder: GreedyCTCDecoder | BeamSearchCTCDecoder,
    ) -> None:
        """Create TritonCTCPipeline instance."""
        self.model = model
        self.logprob_splitter = logprob_splitter
        self.decoder = decoder

    def _adapt_logprobs_for_splitter(self, logprobs: np.ndarray) -> np.ndarray:
        """Adapt logprobs from Triton to match expected format for splitter.

        Triton returns FP32 logprobs with shape [batch, 10, 35], but the splitter
        expects FP32 with shape [time_frames, 35].
        """
        # УБРАЛИ конвертацию в FP16 - splitter ожидает FP32!
        # logprobs уже FP32 из Triton, оставляем как есть

        # Remove batch dimension if present
        if logprobs.ndim == 3 and logprobs.shape[0] == 1:
            logprobs = logprobs[0]  # shape: [10, 35]

        return logprobs

    def forward(
            self,
            audio_chunk: InputType,
            state: Optional[StateType] = None,
            *,
            is_last: bool = False,
    ) -> Tuple[OutputType, StateType]:
        """Perform online (streaming) CTC decoding on a 300 ms audio chunk with GPU."""
        if not isinstance(audio_chunk, np.ndarray):
            raise TypeError(f"Incorrect 'audio_chunk' type: expected np.ndarray, but got {type(audio_chunk)}")
        if audio_chunk.shape != (self.CHUNK_SIZE,):
            raise ValueError(f"Shape of 'audio_chunk' must be ({self.CHUNK_SIZE},), but got {audio_chunk.shape}")

        frame_size, time_bias = self.FRAME_SIZE, self.MEAN_TIME_BIAS

        # Extract states
        model_state = state[0] if state is not None else None
        logprob_state = state[1] if state is not None else None

        # Reshape audio for Triton (batch_size=1, samples, channels=1)
        audio_reshaped = audio_chunk[None, :, None].astype(np.int32)

        # GPU inference via Triton
        logprobs, model_state_next = self.model.forward(audio_reshaped, model_state)

        # Adapt logprobs for the splitter (теперь оставляем FP32)
        logprobs_adapted = self._adapt_logprobs_for_splitter(logprobs)

        # Rest of the pipeline
        logprob_phrases, logprob_state_next = self.logprob_splitter.forward(
            logprobs_adapted, logprob_state, is_last=is_last
        )

        phrases: List[TextPhrase] = []
        for logprob_phrase in logprob_phrases:
            text = self.decoder.forward(logprob_phrase.logprobs)
            start_time = max(
                0,
                round(
                    logprob_phrase.start_frame * frame_size - time_bias - self.PADDING / self.SAMPLE_RATE,
                    2,
                ),
            )
            end_time = max(
                start_time,
                round(
                    logprob_phrase.end_frame * frame_size - time_bias - self.PADDING / self.SAMPLE_RATE,
                    2,
                ),
            )
            phrases.append(
                TextPhrase(
                    text=text,
                    start_time=start_time,
                    end_time=end_time,
                ),
            )

        return (phrases, (model_state_next, logprob_state_next))

    def forward_offline(self, audio: InputType) -> OutputType:
        """Performs offline CTC decoding on a complete audio segment with GPU."""
        if not isinstance(audio, np.ndarray):
            raise TypeError(f"Incorrect 'audio' type: expected np.ndarray, but got {type(audio)}")
        if audio.ndim != 1:
            raise ValueError(f"Shape of 'audio' must be (L,), but got {audio.shape}")

        # Apply padding
        audio = np.pad(audio, (self.PADDING, self.PADDING))
        audio = np.pad(audio, (0, -len(audio) % self.CHUNK_SIZE))
        audio_chunks = np.split(audio, len(audio) // self.CHUNK_SIZE)

        outputs: List[TextPhrase] = []
        state: Optional[StateType] = None

        for i, audio_chunk in enumerate(audio_chunks):
            output, state = self.forward(audio_chunk, state, is_last=i == len(audio_chunks) - 1)
            outputs.extend(output)

        return outputs

    def finalize(self, state: Optional[StateType]) -> Tuple[OutputType, StateType]:
        """Finalize the pipeline by sending an empty chunk."""
        audio_chunk = np.zeros((self.CHUNK_SIZE,), dtype=np.int32)
        return self.forward(audio_chunk, state, is_last=True)

    def health_check(self) -> bool:
        """Check if Triton server is healthy."""
        try:
            return self.model.health_check()
        except Exception as e:
            logger.error(f"Triton health check failed: {e}")
            return False