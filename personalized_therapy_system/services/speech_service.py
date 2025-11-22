"""
Module: voice_chat_interface.py

Description:
    This module enables voice-based interaction with an AI assistant using
    speech-to-text (Whisper) and text-to-speech (Kokoro). It includes:
    
    - Real-time VAD-based audio recording
    - Transcription using FasterWhisper
    - TTS synthesis using Kokoro
    - Optional CLI loop for hands-free back-and-forth interaction

Created: 2025-06-24
Last Modified: 2025-07-08
"""

import sounddevice as sd
import numpy as np
from io import BytesIO
import threading
import collections
import asyncio
import re
import unicodedata
import soundfile as sf
import base64

from faster_whisper import WhisperModel
from kokoro import KPipeline
import webrtcvad

# --------------------------
# Model Initialization
# --------------------------
stt_model = WhisperModel("base.en", compute_type="int8", device='auto')  # Fast + lightweight model
pipeline = KPipeline(lang_code='a')  # Kokoro TTS pipeline initialization

# --------------------------
# Audio Configuration
# --------------------------
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION = 30  # milliseconds
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
MAX_SILENCE_DURATION = 1.0  # seconds


def record_audio_vad(sample_rate=SAMPLE_RATE):
    """
    Capture audio from the microphone using WebRTC VAD.
    Automatically stops recording after silence is detected.

    Returns:
        np.ndarray: Recorded audio samples.
    """
    print("\n Listening... Speak now (auto-stop after silence)")
    vad = webrtcvad.Vad(1)  # Sensitivity: 0 (aggressive) to 3 (lenient)
    buffer = []
    ring_buffer = collections.deque(maxlen=int(MAX_SILENCE_DURATION * 1000 / FRAME_DURATION))
    silence_counter = 0
    recording_done = threading.Event()

    def callback(indata, frames, time_info, status):
        nonlocal silence_counter
        audio_bytes = indata.tobytes()
        if vad.is_speech(audio_bytes, sample_rate):
            ring_buffer.clear()
            silence_counter = 0
            buffer.append(indata.copy())
        else:
            ring_buffer.append(indata.copy())
            silence_counter += 1
            if silence_counter > ring_buffer.maxlen:
                recording_done.set()

    with sd.InputStream(samplerate=sample_rate, channels=CHANNELS, dtype='int16',
                        blocksize=FRAME_SIZE, callback=callback):
        while not recording_done.is_set():
            sd.sleep(int(FRAME_DURATION))

    samples = np.concatenate(buffer + list(ring_buffer), axis=0)
    print("Stopped recording after silence.\n")
    return samples


def clean_text(text: str) -> str:
    """
    Normalize and clean text for TTS input.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned string with only valid characters.
    """
    text = unicodedata.normalize("NFKD", text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-zA-Z0-9.,!?'\s]", '', text)
    return text.strip()


async def transcribe_audio(audio_data) -> str:
    """
    Transcribe audio to text using Whisper.

    Args:
        audio_data (bytes or np.ndarray): Input audio.

    Returns:
        str: Transcribed string.
    """
    if isinstance(audio_data, (bytes, bytearray)):
        with BytesIO(audio_data) as buf:
            audio_np, samplerate = sf.read(buf, dtype='int16')
        if samplerate != SAMPLE_RATE:
            raise ValueError(f"Unsupported sample rate: {samplerate}")
        audio_data = audio_np

    with BytesIO() as wav_io:
        sf.write(wav_io, audio_data.astype(np.int16), SAMPLE_RATE, format="WAV")
        wav_io.seek(0)
        segments, _ = stt_model.transcribe(wav_io, language="en")
    full_text = " ".join(seg.text.strip() for seg in segments)
    return full_text


async def tts_to_base64(response_text: str) -> str:
    """
    Generate speech from text and return as base64-encoded WAV.

    Args:
        response_text (str): Input text.

    Returns:
        str: Base64 string of WAV audio.
    """
    text = clean_text(response_text)
    if not text:
        return ""

    audio_chunks = []
    generator = pipeline(text, voice='af_heart')
    for _, _, chunk in generator:
        if chunk is not None:
            audio_chunks.append(chunk)

    if not audio_chunks:
        return ""

    full_audio = np.concatenate(audio_chunks)

    with BytesIO() as buf:
        sf.write(buf, full_audio, 24000, format="WAV")
        wav_bytes = buf.getvalue()
    return base64.b64encode(wav_bytes).decode('utf-8')


async def synthesize_speech(response_text: str) -> None:
    """
    Generate speech from text and play it out loud (CLI use).

    Args:
        response_text (str): Text to speak.
    """
    text = clean_text(response_text)
    if not text:
        return
    generator = pipeline(text, voice='af_heart')
    audio_out = None
    for _, _, chunk in generator:
        audio_out = chunk
    if audio_out is not None:
        sd.play(audio_out, samplerate=24000)
        sd.wait()


async def mock_llm_response(text: str) -> str:
    """
    Simulated LLM response for testing.

    Args:
        text (str): User input text.

    Returns:
        str: Dummy AI response.
    """
    await asyncio.sleep(0.5)
    return f"I'm here for you. You said: '{text}'"


async def start_conversation_loop():
    """
    CLI loop to record, transcribe, respond, and play audio back.
    """
    print(" Voice AI is active. Press Ctrl+C to exit.\n")
    while True:
        try:
            audio_data = await asyncio.to_thread(record_audio_vad)
            text_input = await transcribe_audio(audio_data)
            if not text_input:
                print("Didn't catch anything. Try again.\n")
                continue

            print(f"You said: {text_input}")
            response = await mock_llm_response(text_input)
            print(f"Response: {response}")
            await synthesize_speech(response)
        except KeyboardInterrupt:
            print("\nConversation ended by user.")
            break
        except Exception as e:
            print(f"Error during loop: {e}")
            break
