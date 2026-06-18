"""
Client library for connecting to Triton Inference Server WebSocket for real-time audio transcription.

This library is designed for applications that have their own WebSocket connections to VOIP services
and receive 8k mono telephony packets. They can use this library to transcribe audio in real-time.

Example usage:
    import asyncio
    from triton_client import TritonTranscriptionClient
    
    async def on_transcript(transcript: str):
        print(f"Transcript: {transcript}")
    
    async def main():
        client = TritonTranscriptionClient(
            server_url="ws://localhost:8765",
            on_transcript=on_transcript
        )
        
        await client.connect()
        
        # Send audio data (8k mono 16-bit PCM)
        audio_data = b"..."  # Your audio bytes
        await client.send_audio(audio_data)
        
        # Keep connection alive and process transcripts
        await client.wait_for_transcripts()
    
    asyncio.run(main())
"""

import asyncio
import json
import logging
from typing import Optional, Callable, Awaitable
import websockets
from websockets import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
from knowlsttclient.proto import TranscriptionEvent
from knowlsttclient.proto import ControlMessage
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("triton_client")


class TritonTranscriptionClient:
    """
    Client for connecting to Triton Inference Server WebSocket for real-time audio transcription.
    
    The client expects:
    - Audio format: Mono 16-bit signed PCM (little-endian), 8 kHz sample rate
    - Audio packets: Typically 20ms chunks (160 samples = 320 bytes)
    
    The server will:
    - Accumulate audio and run VAD (Voice Activity Detection)
    - Send transcripts as text messages when speech is detected (minimum 1 second of audio)
    """
    
    def __init__(
        self,
        server_url: str = "ws://localhost:8765",
        on_transcript: Optional[Callable[[str], Awaitable[None]]] = None,
        on_vad_event: Optional[Callable[[str], Awaitable[None]]] = None,
        on_error: Optional[Callable[[Exception], Awaitable[None]]] = None,
        on_close: Optional[Callable[[], Awaitable[None]]] = None,
        auto_reconnect: bool = True,
        reconnect_delay: float = 5.0,
        stream_id: Optional[str] = None,
        stop_history_ms: Optional[int] = None,
        endpointing: Optional[dict] = None,
        api_key: Optional[str] = None,
        sample_rate: int = 8000,
    ):
        """
        Initialize the Triton Transcription Client.

        Args:
            server_url: WebSocket URL of the Triton inference server (default: ws://localhost:8765)
            on_transcript: Async callback function called when a transcript is received.
                          Signature: async def on_transcript(transcript: str) -> None
            on_error: Async callback function called when an error occurs.
                     Signature: async def on_error(error: Exception) -> None
            on_close: Async callback function called when the connection is closed.
                     Signature: async def on_close() -> None
            auto_reconnect: Whether to automatically reconnect on connection loss (default: True)
            reconnect_delay: Delay in seconds before attempting to reconnect (default: 5.0)
            stream_id: Optional stream ID to identify the stream.
            stop_history_ms: Optional per-call endpointing duration — trailing
                silence (ms) before a segment is finalized. Lower = faster finals
                (lower latency) at the risk of cutting on natural pauses. Sweet
                spot for telephony is ~300-500 ms. None = use the server default
                (~800 ms baked into the model). Only honoured by the Riva backend.
            endpointing: Optional dict for advanced endpointing knobs, any of:
                stop_history_ms, stop_history_eou_ms, start_history_ms (ints, ms),
                stop_threshold, stop_threshold_eou, start_threshold (floats, 0..1).
                ``stop_history_ms`` (the kwarg) takes precedence over this dict.
            api_key: Optional API key for the gated external endpoint
                (e.g. wss://voice.knowl.io/stt/v1). Sent as
                ``Authorization: Bearer <key>`` on the WebSocket handshake.
                Leave unset for the internal endpoint (no auth).
            sample_rate: Input PCM sample rate in Hz (default 8000). 8000 is
                upsampled to 16 kHz server-side; 16000 is sent to the model
                as-is (no upsampling). Audio must be mono 16-bit PCM at this rate.
        """
        self.server_url = server_url
        self.on_transcript = on_transcript
        self.on_vad_event = on_vad_event
        self.on_error = on_error
        self.on_close = on_close
        self.auto_reconnect = auto_reconnect
        self.reconnect_delay = reconnect_delay
        self.stream_id = stream_id if stream_id else str(uuid.uuid4())
        self.api_key = api_key
        # Input PCM rate. 8000 (default) is upsampled to 16 kHz server-side;
        # 16000 is sent to the model as-is (no upsampling). Sent in METADATA.
        self.sample_rate = sample_rate
        # Assemble per-call endpointing overrides (sent in the METADATA message).
        self.endpointing: dict = dict(endpointing or {})
        if stop_history_ms is not None:
            self.endpointing["stop_history_ms"] = stop_history_ms
        self.websocket: Optional[WebSocketClientProtocol] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._connected = False
        self._should_reconnect = True
        
    def _build_metadata(self) -> str:
        """Build the METADATA payload. Plain stream_id when there are no
        endpointing overrides (back-compatible); a JSON object otherwise."""
        meta = dict(self.endpointing)
        if self.sample_rate != 8000:
            meta["sample_rate"] = self.sample_rate
        if not meta:
            return self.stream_id
        return json.dumps({"stream_id": self.stream_id, **meta})

    async def connect(self) -> None:
        """
        Connect to the Triton inference server WebSocket.

        Raises:
            Exception: If connection fails
        """
        try:
            logger.info(f"Connecting to {self.server_url}...")
            self.websocket = await websockets.connect(
                self.server_url,
                additional_headers=(
                    {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
                ),
                max_size=None,
                max_queue=None
            )
            self._connected = True
            logger.info("Connected to Triton inference server")
            await self.websocket.send(ControlMessage(
                type=ControlMessage.MessageType.METADATA,
                metadata=self._build_metadata(),
            ).SerializeToString())
            
            # Start receiving transcripts
            self._receive_task = asyncio.create_task(self._receive_loop())
            
        except Exception as e:
            self._connected = False
            logger.error(f"Failed to connect to server: {e}")
            if self.on_error:
                await self.on_error(e)
            raise
    
    async def _receive_loop(self) -> None:
        """Internal loop to receive transcripts from the server."""
        try:
            while self._connected and self.websocket:
                try:
                    message = await self.websocket.recv()
                    transcription_event = TranscriptionEvent()
                    # TODO: Handle bad message
                    transcription_event.ParseFromString(message)
                    match transcription_event.type:
                        case TranscriptionEvent.EventType.TRANSCRIPT:
                            if self.on_transcript:
                                try:
                                    await self.on_transcript(transcription_event)
                                except Exception as e:
                                    logger.error(f"Error in on_transcript callback: {e}")
                                    if self.on_error:
                                        await self.on_error(e)
                        case TranscriptionEvent.EventType.VAD_SPEECH_START | TranscriptionEvent.EventType.VAD_SPEECH_END:
                            if self.on_vad_event:
                                try:
                                    await self.on_vad_event(transcription_event)
                                except Exception as e:
                                    logger.error(f"Error in on_vad_event callback: {e}")
                                    if self.on_error:
                                        await self.on_error(e)
                        case TranscriptionEvent.EventType.CLOSED:
                            logger.info("Server closed connection")
                            self._connected = False
                            if self.on_close:
                                await self.on_close()
                            break
                        case _:
                            logger.warning("Received unexpected transcription event type")
                        
                except ConnectionClosedOK:
                    logger.info("Server closed connection cleanly")
                    self._connected = False
                    break
                except ConnectionClosedError as e:
                    logger.warning(f"Server connection error: {e}")
                    self._connected = False
                    break
                except Exception as e:
                    logger.error(f"Error receiving message: {e}")
                    if self.on_error:
                        await self.on_error(e)
                    self._connected = False
                    break
                    
        except asyncio.CancelledError:
            logger.debug("Receive loop cancelled")
        except Exception as e:
            logger.error(f"Unexpected error in receive loop: {e}")
            if self.on_error:
                await self.on_error(e)
        finally:
            self._connected = False
            if self.on_close:
                await self.on_close()
    
    async def send_audio(self, audio_data: bytes) -> None:
        """
        Send audio data to the server for transcription.
        
        Args:
            audio_data: Raw 8 kHz mono 16-bit PCM audio bytes (little-endian)
            
        Raises:
            RuntimeError: If not connected to the server
            Exception: If sending fails
        """
        if not self._connected or not self.websocket:
            raise RuntimeError("Not connected to server. Call connect() first.")
        
        try:
            await self.websocket.send(ControlMessage(type=ControlMessage.MessageType.MEDIA, data=audio_data).SerializeToString())
            logger.debug(f"Sent {len(audio_data)} bytes of audio data")
        except ConnectionClosedOK:
            logger.warning("Connection closed while sending audio")
            self._connected = False
            if self.auto_reconnect and self._should_reconnect:
                await self._reconnect()
            raise
        except ConnectionClosedError as e:
            logger.warning(f"Connection error while sending audio: {e}")
            self._connected = False
            if self.auto_reconnect and self._should_reconnect:
                await self._reconnect()
            raise
        except Exception as e:
            logger.error(f"Error sending audio: {e}")
            if self.on_error:
                await self.on_error(e)
            raise
    
    async def send_close_message(self) -> None:
        """
        Send a close message to the server.

        Raises:
            RuntimeError: If not connected to the server
            Exception: If sending fails
        """
        if not self._connected or not self.websocket:
            raise RuntimeError("Not connected to server. Call connect() first.")
        try:
            await self.websocket.send(ControlMessage(type=ControlMessage.MessageType.CLOSE).SerializeToString())
            logger.debug(f"Sent close message")
        except Exception as e:
            logger.error(f"Error sending close message: {e}")
            if self.on_error:
                await self.on_error(e)
            raise

    async def _reconnect(self) -> None:
        """Attempt to reconnect to the server."""
        logger.info(f"Attempting to reconnect in {self.reconnect_delay} seconds...")
        await asyncio.sleep(self.reconnect_delay)
        
        try:
            await self.connect()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            if self.auto_reconnect and self._should_reconnect:
                # Schedule another reconnection attempt
                asyncio.create_task(self._reconnect())
    
    async def wait_for_transcripts(self) -> None:
        """
        Wait for transcripts to be received. This will block until the connection is closed.
        Useful for keeping the connection alive while processing transcripts.
        """
        if self._receive_task:
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
    
    def is_connected(self) -> bool:
        """Check if currently connected to the server."""
        return self._connected and self.websocket is not None
    
    async def disconnect(self) -> None:
        """Disconnect from the server and clean up resources."""
        logger.info("Disconnecting from server...")
        self._should_reconnect = False
        self._connected = False
        
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.warning(f"Error closing websocket: {e}")
        
        logger.info("Disconnected from server")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

