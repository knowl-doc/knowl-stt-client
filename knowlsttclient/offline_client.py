"""Offline (batch) speech-to-text client.

One-shot transcription of a whole clip over HTTPS — the counterpart to the
streaming ``StreamingClient``. You send the entire audio as raw PCM in a single
POST and get back the full transcript with segment/word timestamps.

    from knowlsttclient import OfflineClient

    client = OfflineClient(api_key="knowl_sk_...")
    result = client.recognize(pcm_bytes, sample_rate=8000)   # 8000 (default) or 16000
    print(result.transcript)
    for seg in result.segments:
        print(seg.start, seg.end, seg.text)

Audio must be the SAME format the streaming endpoint accepts: **mono, 16-bit
signed little-endian PCM** at 8 kHz (default) or 16 kHz — no container/header.
For a WAV file, ``pcm_from_wav()`` extracts the bytes for you; other formats
(mp3, …) should be converted to mono PCM first (e.g. with ffmpeg).
"""
from __future__ import annotations

import wave
from dataclasses import dataclass, field
from typing import List, Optional

import requests

DEFAULT_URL = "https://voice.knowl.io/stt/v1/recognize"


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Word:
    start: float
    end: float
    word: str


@dataclass
class OfflineResult:
    transcript: str
    segments: List[Segment] = field(default_factory=list)
    words: List[Word] = field(default_factory=list)
    duration_sec: float = 0.0
    raw: dict = field(default_factory=dict)   # the full server JSON


class OfflineTranscriptionError(RuntimeError):
    """Raised when the server rejects the request or returns an error."""

    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(f"HTTP {status}: {message}")


class OfflineClient:
    """Client for the offline recognize endpoint (``https://voice.knowl.io/stt/v1/recognize``)."""

    def __init__(self, api_key: str, server_url: str = DEFAULT_URL, timeout: float = 120.0):
        """
        Args:
            api_key: API key issued by Knowl (sent as ``Authorization: Bearer <key>``).
            server_url: recognize endpoint (default: the public offline endpoint).
            timeout: per-request timeout in seconds.
        """
        self.api_key = api_key
        self.server_url = server_url
        self.timeout = timeout

    def recognize(self, pcm: bytes, sample_rate: int = 8000,
                  language: Optional[str] = None) -> OfflineResult:
        """Transcribe a whole clip in one call.

        Args:
            pcm: mono, 16-bit signed little-endian PCM bytes (no header).
            sample_rate: 8000 (default) or 16000 — must match the PCM.
            language: optional language-code override (server default otherwise).

        Returns:
            OfflineResult with ``transcript``, ``segments``, ``words``, ``duration_sec``.

        Raises:
            OfflineTranscriptionError: on a non-2xx response.
            ValueError: if the PCM is empty or not 16-bit aligned.
        """
        if not pcm:
            raise ValueError("pcm is empty")
        if len(pcm) % 2 != 0:
            raise ValueError("pcm length must be even (16-bit samples)")
        params = {"sample_rate": sample_rate}
        if language:
            params["language"] = language
        try:
            resp = requests.post(
                self.server_url, params=params, data=pcm,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/octet-stream"},
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise OfflineTranscriptionError(0, str(e)) from e

        if not resp.ok:
            msg = resp.text
            try:
                msg = resp.json().get("error", msg)
            except ValueError:
                pass
            raise OfflineTranscriptionError(resp.status_code, msg)

        data = resp.json()
        return OfflineResult(
            transcript=data.get("transcript", ""),
            segments=[Segment(s.get("start", 0.0), s.get("end", 0.0), s.get("text", ""))
                      for s in data.get("segments", [])],
            words=[Word(w.get("start", 0.0), w.get("end", 0.0), w.get("word", ""))
                   for w in data.get("words", [])],
            duration_sec=data.get("duration_sec", 0.0),
            raw=data,
        )


def pcm_from_wav(path: str) -> "tuple[bytes, int]":
    """Read a WAV file and return (pcm_bytes, sample_rate).

    Validates it's mono 16-bit PCM (what the endpoint requires) and raises
    ValueError otherwise. Convert other formats/layouts to mono 16-bit first.
    """
    with wave.open(path, "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(f"expected mono, got {wf.getnchannels()} channels")
        if wf.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit PCM, got {wf.getsampwidth() * 8}-bit")
        return wf.readframes(wf.getnframes()), wf.getframerate()
