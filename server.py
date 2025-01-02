import asyncio
import logging
import time
import re
import queue
import threading
import numpy as np
import torch

from faster_whisper import WhisperModel
from fastapi import FastAPI, Request
import uvicorn
from sse_starlette.sse import EventSourceResponse
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

# Constants
INT16_MAX = 32767
DEFAULT_CHUNK_SIZE = 512  # For reference (512 samples at 16 kHz)
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_PARTIAL_INTERVAL = 1.0
DEFAULT_SILENCE_DURATION = 2.0
DEFAULT_VAD_THRESHOLD = 0.75

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more detailed logs
    format="[%(levelname)s] %(asctime)s - %(threadName)s - %(message)s",
)

# Hide debug-level logs from faster_whisper
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

class ProductionReadyTranscriber:
    def __init__(
        self,
        transcription_model: str = "base",
        realtime_transcription_model: str = "tiny",
        vad_model_repo: str = "snakers4/silero-vad",
        vad_threshold: float = DEFAULT_VAD_THRESHOLD,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        silence_duration: float = DEFAULT_SILENCE_DURATION,
        partial_transcription_interval: float = DEFAULT_PARTIAL_INTERVAL,
        pre_speech_length: float = 0.5,
    ):
        logging.info("Initializing ProductionReadyTranscriber with parameters:")
        logging.info(f"  transcription_model: {transcription_model}")
        logging.info(f"  realtime_transcription_model: {realtime_transcription_model}")
        logging.info(f"  vad_model_repo: {vad_model_repo}")
        logging.info(f"  vad_threshold: {vad_threshold}")
        logging.info(f"  sample_rate: {sample_rate}")
        logging.info(f"  silence_duration: {silence_duration}")
        logging.info(
            f"  partial_transcription_interval: {partial_transcription_interval}"
        )
        logging.info(f"  pre_speech_length: {pre_speech_length}")

        # Basic config
        self.sample_rate = sample_rate
        self.silence_duration = silence_duration
        self.partial_transcription_interval = partial_transcription_interval

        # Silero VAD
        try:
            self.vad_model, _ = torch.hub.load(
                repo_or_dir=vad_model_repo, model="silero_vad", force_reload=False
            )
            logging.info("Silero VAD model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load Silero VAD model: {e}")
            raise

        self.vad_threshold = vad_threshold

        # Whisper
        try:
            self.transcription_model = WhisperModel(
                model_size_or_path=transcription_model,
                device="cpu",
            )
            logging.info(
                f"Whisper transcription model '{transcription_model}' loaded successfully."
            )
        except Exception as e:
            logging.error(
                f"Failed to load Whisper transcription model '{transcription_model}': {e}"
            )
            raise

        try:
            self.realtime_transcription_model = WhisperModel(
                model_size_or_path=realtime_transcription_model,
                device="cpu",
            )
            logging.info(
                f"Whisper realtime transcription model '{realtime_transcription_model}' loaded successfully."
            )
        except Exception as e:
            logging.error(
                f"Failed to load Whisper realtime transcription model '{realtime_transcription_model}': {e}"
            )
            raise

        # Threading/event handling
        self.shutdown_event = threading.Event()
        self._processing_thread = None

        # Audio state
        self.is_talking = False
        self.silence_start_time: Optional[float] = None
        self.buffered_audio = np.array([], dtype=np.float32)
        self._last_partial_time = 0.0

        # Pre-speech buffer
        self.pre_speech_length = pre_speech_length
        self.max_pre_speech_samples = int(pre_speech_length * sample_rate)
        self.pre_speech_buffer = np.array([], dtype=np.float32)

        # Queues
        self.audio_queue = queue.Queue(maxsize=100)
        # We'll store SSE "events" (partial or final) in this queue
        self.events_queue = queue.Queue()

        # Buffer to accumulate raw bytes for exactly 512-sample frames
        self.vad_buffer = b""

        # Additional config for dynamic partial interval
        self.min_partial_interval = 0.5  # Minimum partial interval
        self.max_partial_interval = 3.0  # Maximum partial interval
        # If partial transcription takes >= 80% of the current interval, enlarge
        self._enlarge_threshold = 0.8
        # If partial transcription takes <= 50% of the current interval, shrink
        self._shrink_threshold = 0.5
        # Scaling factors
        self._enlarge_factor = 1.2
        self._shrink_factor = 0.9

    def start(self):
        logging.info("Starting transcriber processing thread...")
        self.shutdown_event.clear()

        self._processing_thread = threading.Thread(
            target=self._processing_worker, name="ProcessingThread", daemon=True
        )
        self._processing_thread.start()
        logging.debug("Transcriber processing thread started.")

    def stop(self):
        logging.info("Stopping ProductionReadyTranscriber...")
        self.shutdown_event.set()

        # Let processing thread stop gracefully
        if self._processing_thread:
            logging.debug("Waiting for processing thread to terminate...")
            self._processing_thread.join(timeout=3)
            if self._processing_thread.is_alive():
                logging.warning("Processing thread did not terminate within timeout.")
            else:
                logging.debug("Processing thread terminated.")
        else:
            logging.warning("Processing thread was not running.")

        logging.info("ProductionReadyTranscriber stopped.")

    def feed_audio_chunk(self, audio_bytes: bytes):
        """
        Called by our FastAPI endpoint whenever the client sends
        another chunk of audio data.

        audio_bytes must be 16-bit, single-channel PCM at `self.sample_rate`.
        """

        # 1) Accumulate partial data into self.vad_buffer
        self.vad_buffer += audio_bytes

        # 2) For 16kHz, 512 samples => 512 * 2 = 1024 bytes
        CHUNK_SIZE_BYTES = 512 * 2

        # 3) While we have at least one full frame:
        while len(self.vad_buffer) >= CHUNK_SIZE_BYTES:
            # Extract exactly 1024 bytes for one 512-sample frame
            frame = self.vad_buffer[:CHUNK_SIZE_BYTES]
            self.vad_buffer = self.vad_buffer[CHUNK_SIZE_BYTES:]

            # 4) Enqueue this exact frame into the audio processing queue
            try:
                self.audio_queue.put(frame, timeout=0.2)
                logging.debug(
                    f"Audio chunk (512 samples) enqueued. Queue size: {self.audio_queue.qsize()}"
                )
            except queue.Full:
                logging.warning("Audio queue is full. Dropping chunk.")
                break  # Or continue, depending on desired behavior

    def get_events_generator(self):
        """
        A generator that yields transcription events from self.events_queue
        in SSE format. This allows us to stream partial/final updates
        to the client in real-time.
        """
        logging.debug("Starting events generator.")
        while True:
            # If the transcriber is stopped, break
            if self.shutdown_event.is_set():
                logging.debug("Shutdown event set. Exiting events generator.")
                break

            try:
                event = self.events_queue.get(timeout=0.5)
                logging.debug(f"Yielding event: {event}")
                # event = {"type": "partial"/"final", "text": "..."}
                yield {
                    "event": event["type"],
                    "data": event["text"],
                }
            except queue.Empty:
                # No events in the queue, just continue
                continue

    def _processing_worker(self):
        logging.info("Processing thread running.")
        try:
            self.vad_model.reset_states()
            logging.debug("VAD model states reset.")
        except Exception as e:
            logging.error(f"Failed to reset VAD model states: {e}")

        while not self.shutdown_event.is_set():
            try:
                # Here, each item in the queue is exactly 1024 bytes => 512 samples
                data = self.audio_queue.get(timeout=0.1)
                logging.debug(
                    f"Retrieved audio chunk from queue. Remaining queue size: {self.audio_queue.qsize()}"
                )
            except queue.Empty:
                continue

            try:
                # Convert int16 -> float32
                chunk_i16 = np.frombuffer(data, dtype=np.int16)
                # chunk_i16.shape should be (512,)
                chunk_f32 = chunk_i16.astype(np.float32) / INT16_MAX

                # Pre-speech buffering
                self._update_pre_speech_buffer(chunk_f32)

                # Check speech probability
                vad_prob = self._run_silero_vad(chunk_f32)
                is_speech = vad_prob > self.vad_threshold

                logging.debug(
                    f"vad_prob: {vad_prob:.4f}, is_speech: {is_speech}, is_talking: {self.is_talking}"
                )

                if is_speech:
                    # If user starts/resumes talking
                    if not self.is_talking:
                        self.is_talking = True
                        self.silence_start_time = None
                        # Start with pre_speech
                        self.buffered_audio = np.concatenate(
                            [self.pre_speech_buffer, chunk_f32]
                        )
                        logging.info("[User started talking]")
                    else:
                        # Already talking, accumulate
                        self.buffered_audio = np.concatenate(
                            [self.buffered_audio, chunk_f32]
                        )
                        logging.debug(
                            f"Accumulated buffered_audio length: {len(self.buffered_audio)} samples"
                        )

                        # If we had a silence_start_time, reset it
                        if self.silence_start_time is not None:
                            silence_time = time.time() - self.silence_start_time
                            if silence_time >= 1.5:
                                logging.info(
                                    f"[User paused for {silence_time:.2f}s but resumed. Resetting silence_start_time.]"
                                )
                            self.silence_start_time = None

                    # Periodic partial transcription
                    current_time = time.time()
                    if (
                        current_time - self._last_partial_time
                    ) > self.partial_transcription_interval:
                        self._last_partial_time = current_time
                        start_t = time.time()

                        partial_text = self._transcribe_audio(
                            self.buffered_audio, self.realtime_transcription_model
                        )
                        end_t = time.time()
                        processing_time = end_t - start_t
                        logging.debug(
                            f"Partial transcription took {processing_time:.2f}s"
                        )
                        self._adjust_partial_interval(processing_time)

                        if partial_text:
                            # Send partial update through SSE
                            self.events_queue.put(
                                {"type": "partial", "text": partial_text}
                            )
                            logging.debug(
                                f"Partial transcription enqueued: {partial_text}"
                            )

                else:
                    # No speech in this chunk
                    if self.is_talking:
                        # Start counting silence if not started
                        if self.silence_start_time is None:
                            self.silence_start_time = time.time()
                            logging.debug("Silence detected. Starting silence timer.")

                        silence_time = time.time() - self.silence_start_time
                        logging.debug(
                            f"silence_time: {silence_time:.2f}s, is_done: {silence_time >= self.silence_duration}"
                        )

                        # Finalize after enough silence
                        if silence_time >= self.silence_duration:
                            final_text = self._transcribe_audio(
                                self.buffered_audio, self.transcription_model
                            )
                            if final_text:
                                # Send final update through SSE
                                self.events_queue.put(
                                    {"type": "final", "text": final_text}
                                )
                                logging.debug(
                                    f"Final transcription enqueued: {final_text}"
                                )

                            self.is_talking = False
                            self.buffered_audio = np.array([], dtype=np.float32)
                            self.silence_start_time = None
                            logging.info("[User stopped talking]")

            except Exception as e:
                logging.error(f"Error in processing worker: {e}", exc_info=True)

        logging.info("Processing thread stopped.")

    def _update_pre_speech_buffer(self, chunk_f32: np.ndarray):
        try:
            new_buffer = np.concatenate([self.pre_speech_buffer, chunk_f32])
            if len(new_buffer) > self.max_pre_speech_samples:
                excess = len(new_buffer) - self.max_pre_speech_samples
                new_buffer = new_buffer[excess:]
                logging.debug(
                    f"Pre-speech buffer exceeded max. Truncated by {excess} samples."
                )
            self.pre_speech_buffer = new_buffer
            logging.debug(
                f"Pre-speech buffer updated. Current length: {len(self.pre_speech_buffer)} samples."
            )
        except Exception as e:
            logging.error(f"Failed to update pre-speech buffer: {e}")

    def _run_silero_vad(self, audio_f32: np.ndarray) -> float:
        try:
            with torch.inference_mode():
                # Now we pass exactly 512 samples to the model
                prob = self.vad_model(
                    torch.from_numpy(audio_f32), self.sample_rate
                ).item()
            logging.debug(f"VAD probability: {prob:.4f}")
            return prob
        except Exception as e:
            logging.error(f"Failed to run Silero VAD: {e}")
            return 0.0  # Assume no speech on error

    def _transcribe_audio(
        self, audio_f32: np.ndarray, whisper_model: WhisperModel
    ) -> str:
        # We skip very short audio
        if len(audio_f32) < 2000:
            logging.debug("Audio too short for transcription. Skipping.")
            return ""

        try:
            segments, _info = whisper_model.transcribe(
                audio_f32, beam_size=1, language="en"
            )
            text = " ".join(seg.text for seg in segments).strip()
            text = re.sub(r"\s+", " ", text)
            logging.debug(f"Transcription result: {text}")
            return text
        except Exception as e:
            logging.error(f"Failed to transcribe audio: {e}")
            return ""

    def _adjust_partial_interval(self, processing_time: float):
        original_interval = self.partial_transcription_interval
        if (
            processing_time
            >= self.partial_transcription_interval * self._enlarge_threshold
        ):
            self.partial_transcription_interval *= self._enlarge_factor
            logging.info(
                f"Increasing partial interval to {self.partial_transcription_interval:.2f}s "
                f"(was {original_interval:.2f}s)"
            )
        elif (
            processing_time
            <= self.partial_transcription_interval * self._shrink_threshold
        ):
            self.partial_transcription_interval *= self._shrink_factor
            logging.info(
                f"Decreasing partial interval to {self.partial_transcription_interval:.2f}s "
                f"(was {original_interval:.2f}s)"
            )

        # Clamp
        if self.partial_transcription_interval < self.min_partial_interval:
            self.partial_transcription_interval = self.min_partial_interval
            logging.debug(
                f"Partial interval clamped to minimum: {self.partial_transcription_interval:.2f}s"
            )
        elif self.partial_transcription_interval > self.max_partial_interval:
            self.partial_transcription_interval = self.max_partial_interval
            logging.debug(
                f"Partial interval clamped to maximum: {self.partial_transcription_interval:.2f}s"
            )

        logging.debug(
            f"Partial transcription interval adjusted to {self.partial_transcription_interval:.2f}s"
        )


# ---------------------------------
# FastAPI App Setup & Routes
# ---------------------------------

app = FastAPI(title="Xeno STT API")

# CORS configuration
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Specifies the list of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Create the single instance for demonstration
transcriber = ProductionReadyTranscriber(
    pre_speech_length=0.3,  # store 300ms of pre-speech
)


@app.on_event("startup")
def on_startup():
    logging.info("FastAPI application starting up.")
    # Start the processing thread
    transcriber.start()


@app.on_event("shutdown")
def on_shutdown():
    logging.info("FastAPI application shutting down.")
    # Gracefully stop
    transcriber.stop()

@app.post("/in")
async def stream_audio(request: Request):
    """
    Endpoint to receive raw 16-bit PCM audio from the client.
    The client can repeatedly POST chunks to this endpoint.
    """
    try:
        audio_bytes = await request.body()
        logging.debug(f"Received audio chunk of size {len(audio_bytes)} bytes.")
        if not audio_bytes:
            logging.warning("No data received in /in endpoint.")
            return {"status": "no data received"}

        transcriber.feed_audio_chunk(audio_bytes)
        return {"status": "ok", "length": len(audio_bytes)}
    except Exception as e:
        logging.error(f"Error in /in endpoint: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

@app.get("/out")
async def stream_transcription():
    """
    SSE endpoint. The frontend can connect to this endpoint
    and will receive partial/final transcripts as they occur.

    Each SSE event has:
      - event: either "partial" or "final"
      - data: the text
    """
    logging.info("Client connected to /out SSE endpoint.")

    async def sse_event_generator():
        while not transcriber.shutdown_event.is_set():
            try:
                # Attempt to get an event without blocking.
                event = transcriber.events_queue.get_nowait()
                yield {
                    "event": event["type"],
                    "data": event["text"],
                }
            except queue.Empty:
                # If no event is in the queue, yield no data and allow
                # other tasks to run (sleep briefly).
                await asyncio.sleep(0.1)
    
    return EventSourceResponse(sse_event_generator())


if __name__ == "__main__":
    logging.info("Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
