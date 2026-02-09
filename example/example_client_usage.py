"""
Example usage of the Triton Transcription Client library.
"""

import asyncio
import logging
import wave
from pathlib import Path
from knowlsttclient import TritonTranscriptionClient
from knowlsttclient.proto import TranscriptionEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_audio_file(file_path: str, chunk_size_bytes: int = 320) -> list[bytes]:
    """
    Load a WAV file and convert it to chunks of PCM bytes.
    
    Args:
        file_path: Path to the WAV file
        chunk_size_bytes: Size of each audio chunk in bytes (default: 320 for 20ms at 8kHz)
    
    Returns:
        List of audio chunks as bytes
    """
    audio_chunks = []
    
    with wave.open(file_path, 'rb') as wav_file:
        # Verify audio format
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        
        logger.info(f"Audio file info: {channels} channel(s), {sample_width} bytes/sample, {sample_rate} Hz")
        
        if channels != 1:
            logger.warning(f"Expected mono audio, got {channels} channels. Using first channel only.")
        
        if sample_width != 2:
            raise ValueError(f"Expected 16-bit audio (2 bytes/sample), got {sample_width} bytes/sample")
        
        if sample_rate != 8000:
            logger.warning(f"Expected 8kHz sample rate, got {sample_rate} Hz")
        
        # Read all frames
        frames = wav_file.readframes(wav_file.getnframes())
        
        # Split into chunks
        for i in range(0, len(frames), chunk_size_bytes):
            chunk = frames[i:i + chunk_size_bytes]
            if len(chunk) == chunk_size_bytes:
                audio_chunks.append(chunk)
            elif len(chunk) > 0:
                # Pad last chunk if needed (or just send it as-is)
                audio_chunks.append(chunk)
    
    logger.info(f"Loaded {len(audio_chunks)} audio chunks from {file_path}")
    return audio_chunks


async def example_with_callback():
    """Example using callback function for transcripts."""
    
    async def on_transcript(transcription_event: TranscriptionEvent):
        """Handle received transcript."""
        logger.info(f"📝 Transcript: {transcription_event.transcript}")
        logger.info(
            f"📝 Transcript details: start_time={transcription_event.start_time}, "
            f"end_time={transcription_event.end_time}, is_final={transcription_event.is_final}, "
            f"speech_start_time={transcription_event.speech_start_time}, "
            f"speech_end_time={transcription_event.speech_end_time}, "
            f"vad_speech_start_time={transcription_event.vad_speech_start_time}"
        )
        # Here you can process the transcript, send it to your application, etc.
    
    async def on_error(error: Exception):
        """Handle errors."""
        logger.error(f"❌ Error: {error}")
    
    # Create client with callbacks
    client = TritonTranscriptionClient(
        server_url="ws://localhost:8080",
        on_transcript=on_transcript,
        on_error=on_error,
        auto_reconnect=True
    )
    
    try:
        await client.connect()
        logger.info("✅ Connected to server")
        
        # Load real audio file
        audio_file_path = Path(__file__).parent.parent / "sample-wav" / "audio_8k.wav"
        if not audio_file_path.exists():
            logger.error(f"Audio file not found: {audio_file_path}")
            return
        
        audio_chunks = load_audio_file(str(audio_file_path))
        logger.info(f"Loaded {len(audio_chunks)} audio chunks from {audio_file_path}")
        
        # Send audio chunks simulating real-time streaming
        # In real usage, you would get these from your VOIP WebSocket connection
        # Audio format: 8 kHz mono 16-bit PCM (little-endian)
        # Typical packet size: 20ms = 160 samples = 320 bytes
        
        for chunk in audio_chunks:
            await client.send_audio(chunk)
            await asyncio.sleep(0.01)  # Simulate 20ms packet interval
        logger.info("Sent all audio chunks")
        # Send close message
        await client.send_close_message()
        logger.info("Sent close message")
        
        # Keep connection alive to receive transcripts
        await client.wait_for_transcripts()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await client.disconnect()



if __name__ == "__main__":
    # Run the callback example
    asyncio.run(example_with_callback())

