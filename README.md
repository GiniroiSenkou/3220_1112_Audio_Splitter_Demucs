# Demucs Stem Splitter

A GUI app that splits audio tracks into stems (vocals, drums, bass, other) using [Demucs](https://github.com/facebookresearch/demucs) and converts them to your preferred format.

## Features

- **Drag & drop** or browse to select audio files (MP3, WAV, FLAC, AAC, M4A)
- **Multiple Demucs models** — pick the best one for your use case
- **Output format selector** — FLAC, WAV, MP3 320kbps, or OGG Vorbis
- **Background processing** — GUI stays responsive with a live progress bar
- **Batch queue** — process multiple files with per-file status tracking
- **Cancel** — stop processing at any time
- **GPU / CPU toggle** — use CUDA acceleration if available
- **Dark / Light theme** — toggle with one click, preference is saved
- **Presets** — save and load your favorite model + format combinations
- **Preview / Playback** — listen to split stems directly in the app
- **Open output folder** — quick button to jump to your stems

## Setup

### 1. Install system dependencies

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg gstreamer1.0-plugins-good
```

**Fedora:**
```bash
sudo dnf install ffmpeg gstreamer1-plugins-good
```

**macOS:**
```bash
brew install ffmpeg
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your paths

```bash
cp config.example.json config.json
```

Edit `config.json` with your preferred directories and settings:
```json
{
    "default_input_dir": "/path/to/your/music",
    "default_output_dir": "/path/to/your/stems",
    "theme": "dark",
    "use_gpu": true,
    "output_format": "flac",
    "presets": {}
}
```

This file is gitignored so your personal paths stay private.

### 4. Run

```bash
python main.py
```

## Models

| Model | Best for |
|-------|----------|
| **htdemucs** | Balanced — pop, EDM, most music |
| **mdx_extra_q** | Vocal clarity — lyric-heavy tracks & acapellas |
| **mdx_extra** | Instrumentals — songs with no vocals |
| **mdx** | Fast but lower quality — quick tests |

## Presets

Save your favorite settings (model + format + GPU) as named presets from within the app. Presets are stored in your `config.json`.
