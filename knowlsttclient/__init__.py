from .triton_client import StreamingClient, TritonTranscriptionClient

# StreamingClient is the public name; TritonTranscriptionClient is a deprecated
# alias kept for backwards compatibility (to be removed once all callers migrate).
__all__ = ["StreamingClient", "TritonTranscriptionClient"]
