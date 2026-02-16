"""
Text-to-Speech client for generating narration audio.
Uses OpenAI TTS API (tts-1 or tts-1-hd models).
Includes audio processing for consistency across segments.
"""
import os
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Consistent TTS settings for all narration segments
DEFAULT_VOICE = "nova"  # Professional, clear voice suitable for academic content
DEFAULT_MODEL = "tts-1"  # Fast generation
DEFAULT_SPEED = 1.0  # Natural speaking speed (consistent across all segments)


def generate_speech(
    text: str,
    output_path: Path,
    voice: str = "alloy",
    model: str = "tts-1",
    speed: float = 1.0,
) -> Path:
    """
    Generate speech audio from text using OpenAI TTS API.
    
    Args:
        text: Text to convert to speech
        output_path: Path where the MP3 file will be saved
        voice: Voice to use ("alloy", "echo", "fable", "onyx", "nova", "shimmer")
        model: Model to use ("tts-1" for faster, "tts-1-hd" for higher quality)
        speed: Speed multiplier (0.25 to 4.0, default 1.0)
    
    Returns:
        Path to the generated audio file
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in .env")
    
    client = OpenAI(api_key=api_key)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Validate voice
    valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    if voice not in valid_voices:
        raise ValueError(f"Invalid voice '{voice}'. Must be one of: {valid_voices}")
    
    # Validate speed
    if not 0.25 <= speed <= 4.0:
        raise ValueError(f"Speed must be between 0.25 and 4.0, got {speed}")
    
    # Validate model
    if model not in ["tts-1", "tts-1-hd"]:
        raise ValueError(f"Invalid model '{model}'. Must be 'tts-1' or 'tts-1-hd'")
    
    print(f"[TTS] Generating speech: {len(text)} chars -> {output_path.name}")
    print(f"   Voice: {voice}, Model: {model}, Speed: {speed}x")
    
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        speed=speed,
    )
    
    # Save to file
    response.stream_to_file(str(output_path))
    
    print(f"[OK] Speech generated: {output_path}")
    return output_path


def normalize_audio_levels(
    audio_path: Path,
    output_path: Optional[Path] = None,
    target_lufs: float = -16.0,
) -> Path:
    """
    Normalize audio levels using EBU R128 loudness normalization.
    Ensures consistent volume levels across all segments.
    
    Args:
        audio_path: Path to input audio file
        output_path: Optional output path (defaults to overwriting input)
        target_lufs: Target integrated loudness in LUFS (default -16.0, standard for streaming)
    
    Returns:
        Path to the normalized audio file
    """
    from video_actions import _ffmpeg_exe
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    ffmpeg = _ffmpeg_exe()
    out_path = output_path or audio_path
    
    # Use loudnorm filter for EBU R128 normalization
    # This ensures consistent perceived loudness across segments
    cmd = [
        ffmpeg,
        "-y",
        "-i", audio_path.as_posix(),
        "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
        "-ar", "44100",  # Standard sample rate
        out_path.as_posix(),
    ]
    
    import subprocess
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    return out_path


def adjust_audio_duration(
    audio_path: Path,
    target_duration_s: float,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Adjust audio duration to match target duration using ffmpeg.
    If audio is shorter, it will be padded with silence.
    If audio is longer, it will be trimmed.
    
    Args:
        audio_path: Path to input audio file
        target_duration_s: Target duration in seconds
        output_path: Optional output path (defaults to overwriting input)
    
    Returns:
        Path to the adjusted audio file
    """
    from video_actions import _ffmpeg_exe
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    ffmpeg = _ffmpeg_exe()
    out_path = output_path or audio_path
    
    # Use atrim/asetpts to trim, and apad to pad with silence
    cmd = [
        ffmpeg,
        "-y",
        "-i", audio_path.as_posix(),
        "-af", f"atrim=0:{target_duration_s},asetpts=PTS-STARTPTS,apad=pad_dur={target_duration_s}",
        "-t", str(target_duration_s),
        out_path.as_posix(),
    ]
    
    import subprocess
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    return out_path


def process_audio_for_consistency(
    audio_path: Path,
    target_duration_s: float,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Process audio for consistency: normalize levels and adjust duration.
    This ensures all narration segments have consistent volume and timing.
    
    Args:
        audio_path: Path to input audio file
        target_duration_s: Target duration in seconds
        output_path: Optional output path (defaults to overwriting input)
    
    Returns:
        Path to the processed audio file
    """
    # First normalize audio levels
    temp_normalized = audio_path.parent / f"{audio_path.stem}_normalized{audio_path.suffix}"
    normalize_audio_levels(audio_path, temp_normalized)
    
    # Then adjust duration
    out_path = output_path or audio_path
    adjust_audio_duration(temp_normalized, target_duration_s, out_path)
    
    # Cleanup temp file
    if temp_normalized.exists() and temp_normalized != out_path:
        temp_normalized.unlink()
    
    return out_path


def combine_audio_with_crossfade(
    audio_files: List[Path],
    output_path: Path,
    crossfade_duration_s: float = 0.25,
) -> Path:
    """
    Combine multiple audio files with smooth crossfades between segments.
    This creates seamless transitions and prevents abrupt volume changes.
    
    Args:
        audio_files: List of audio file paths to combine
        output_path: Path for the combined output file
        crossfade_duration_s: Duration of crossfade between segments (default 0.25s)
    
    Returns:
        Path to the combined audio file
    """
    from video_actions import _ffmpeg_exe
    import subprocess
    
    if not audio_files:
        raise ValueError("No audio files provided")
    
    if len(audio_files) == 1:
        # No need for crossfading with a single file
        import shutil
        shutil.copy2(audio_files[0], output_path)
        return output_path
    
    ffmpeg = _ffmpeg_exe()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get durations for all files
    durations = []
    for audio_file in audio_files:
        probe_cmd = [
            ffmpeg,
            "-i", audio_file.as_posix(),
            "-show_entries", "format=duration",
            "-v", "quiet",
            "-of", "csv=p=0",
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        durations.append(float(result.stdout.strip()))
    
    # Build complex filter for crossfading
    # Strategy: Apply fade-out to end of each segment (except last) and fade-in to start (except first)
    filter_parts = []
    inputs = []
    
    for i, audio_file in enumerate(audio_files):
        inputs.extend(["-i", audio_file.as_posix()])
        
        duration = durations[i]
        fade_start = max(0, duration - crossfade_duration_s)
        
        if i == 0:
            # First segment: fade in at start, fade out at end
            filter_parts.append(
                f"[{i}]afade=t=in:st=0:d={crossfade_duration_s},"
                f"afade=t=out:st={fade_start}:d={crossfade_duration_s}[a{i}]"
            )
        elif i == len(audio_files) - 1:
            # Last segment: fade in at start (to blend with previous), fade out at end
            filter_parts.append(
                f"[{i}]afade=t=in:st=0:d={crossfade_duration_s},"
                f"afade=t=out:st={fade_start}:d={crossfade_duration_s}[a{i}]"
            )
        else:
            # Middle segments: fade in at start, fade out at end
            filter_parts.append(
                f"[{i}]afade=t=in:st=0:d={crossfade_duration_s},"
                f"afade=t=out:st={fade_start}:d={crossfade_duration_s}[a{i}]"
            )
    
    # Concatenate all segments
    concat_inputs = "".join([f"[a{i}]" for i in range(len(audio_files))])
    filter_parts.append(f"{concat_inputs}concat=n={len(audio_files)}:v=0:a=1[out]")
    
    filter_complex = ";".join(filter_parts)
    
    cmd = [
        ffmpeg,
        "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:a", "libmp3lame",
        "-b:a", "128k",
        "-ar", "44100",  # Consistent sample rate
        output_path.as_posix(),
    ]
    
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Apply final normalization to ensure consistent levels across entire combined audio
    temp_normalized = output_path.parent / f"{output_path.stem}_normalized{output_path.suffix}"
    normalize_audio_levels(output_path, temp_normalized)
    
    # Replace original with normalized version
    import shutil
    shutil.move(temp_normalized, output_path)
    
    return output_path
