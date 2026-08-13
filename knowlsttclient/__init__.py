from .streaming_client import StreamingClient
from .offline_client import (
    OfflineClient, OfflineResult, OfflineTranscriptionError, Segment, Word, pcm_from_wav,
)

# StreamingClient  — streaming STT over WebSocket (wss://voice.knowl.io/stt/v1)
# OfflineClient    — one-shot STT over HTTPS   (https://voice.knowl.io/stt/v1/recognize)
__all__ = [
    "StreamingClient",
    "OfflineClient", "OfflineResult", "OfflineTranscriptionError",
    "Segment", "Word", "pcm_from_wav",
]
