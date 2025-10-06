[简体中文](README_CN.md) | English

# 🎥 Universal Video Watermark Remover & Enhancer

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/zelinewang/Video-WatermarkRemover-Enhancer?style=social)](https://github.com/zelinewang/Video-WatermarkRemover-Enhancer/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/zelinewang/Video-WatermarkRemover-Enhancer?style=social)](https://github.com/zelinewang/Video-WatermarkRemover-Enhancer/network/members)
[![GitHub issues](https://img.shields.io/github/issues/zelinewang/Video-WatermarkRemover-Enhancer)](https://github.com/zelinewang/Video-WatermarkRemover-Enhancer/issues)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zelinewang/Video-WatermarkRemover-Enhancer/blob/master/KLing_Watermark_Remover_Colab.ipynb)

**Professional AI-powered watermark removal for videos from KLing, Sora, and other sources!** 🚀

![Demo](demo.webp)

</div>

## ✨ Key Features

🎯 **Universal Watermark Removal**
- **Fixed-position watermarks** (KLing, standard overlays)
- **Dynamic watermarks** (Sora, moving logos) with template matching
- **Multi-scale detection** for varying watermark sizes
- Lossless quality with smooth, natural edges
- Batch processing support for maximum efficiency

🎨 **AI Video Enhancement**
- Super-resolution technology powered by Real-ESRGAN
- Smart optimization of brightness, contrast, and clarity
- Special facial detail enhancement with GFPGAN
- GPU acceleration for 10-50x faster processing

🎵 **Audio Preservation**
- **NEW**: Keep original audio tracks during processing
- Extract, merge, and sync audio streams
- Support for multiple audio formats (AAC, MP3, etc.)

⚡ **Efficient & Convenient**
- Simple command-line operation
- Google Colab support with free GPU (T4)
- Customizable processing parameters
- Visualization tools for debugging

## 🎬 Supported Video Sources

| Platform | Watermark Type | Detection Method | Accuracy |
|----------|---------------|------------------|----------|
| **KLing** | Fixed position | Preset coordinates | 100% |
| **Sora** | Dynamic position | Template matching | 85-95% |
| **Custom** | Any type | Manual configuration | Varies |

**Can process any video with visible watermarks!**

## 🔧 Installation

### Method 1: Local Installation (CPU/GPU)

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/zelinewang/Video-WatermarkRemover-Enhancer.git
cd Video-WatermarkRemover-Enhancer

# Create conda environment
conda create -n watermark-remover python=3.10
conda activate watermark-remover

# Install dependencies
pip install -r requirements.txt
```

### Method 2: Google Colab (Recommended) 🌟

**Free GPU (T4) access - 10-50x faster than CPU!**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zelinewang/Video-WatermarkRemover-Enhancer/blob/master/KLing_Watermark_Remover_Colab.ipynb)

Click the badge above or visit:
```
https://colab.research.google.com/github/zelinewang/Video-WatermarkRemover-Enhancer/blob/master/KLing_Watermark_Remover_Colab.ipynb
```

**Processing time comparison:**
- CPU: 60-90 minutes for 300 frames
- Colab T4 GPU: 5-10 minutes ⚡

## 🛠️ Configuration

The `config.yaml` file defines parameters for watermark removal and video enhancement.

### Watermark Removal Settings

```yaml
watermark:
  position: [556, 1233, 701, 1267]  # [x1, y1, x2, y2] for fixed watermarks
  ckpt_p: "./weights/sttn.pth"       # STTN model path
  mask_expand: 30                     # Mask expansion in pixels
  neighbor_stride: 10                 # Temporal neighborhood stride
```

**For dynamic watermarks (Sora):** See [Dynamic Watermark Detection Guide](USAGE_EXAMPLES.md#advanced-dynamic-watermark-removal-sora)

### Video Enhancement Settings

```yaml
enhance:
  RealESRGAN_model_path: "./weights/RealESRGAN_x2plus.pth"  # Super-resolution
  GFPGANer_model_path: "./weights/GFPGANv1.4.pth"           # Face enhancement
```

### Download Model Weights

**Required for watermark removal:**
- [sttn.pth](https://drive.google.com/file/d/1ZAMV8547wmZylKRt5qR_tC5VlosXD4Wv/view?usp=sharing) (~200MB)

**Optional for video enhancement:**
- [RealESRGAN_x2plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth)
- [GFPGANv1.4.pth](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth)

Place downloaded models in the `./weights/` directory.

## 🚀 Quick Start

### Basic Usage

```bash
# Remove watermark with audio preservation (recommended)
python main.py --input video.mp4 --remove-watermark --keep-audio

# Remove watermark only (no audio)
python main.py --input video.mp4 --remove-watermark

# Remove watermark + enhance video
python main.py --input video.mp4 --remove-watermark --enhance-video --keep-audio

# Process entire folder
python main.py --input videos_folder/ --remove-watermark --keep-audio
```

### Advanced: Dynamic Watermark Removal (Sora Videos)

**Step 1: Extract watermark template**

```bash
# Extract a frame from your video
ffmpeg -i sora_video.mp4 -ss 00:00:01 -vframes 1 frame.png

# Crop the watermark using an image editor (GIMP, Photoshop, etc.)
# Save as: sora_watermark.png
```

**Step 2: Use the dynamic removal script**

Create `remove_dynamic_watermark.py` (full script in [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)):

```bash
python remove_dynamic_watermark.py \
    --input sora_video.mp4 \
    --template sora_watermark.png \
    --threshold 0.8 \
    --multi-scale \
    --keep-audio \
    --visualize
```

**Parameters:**
- `--threshold`: Detection threshold (0.7-0.9, higher = stricter)
- `--multi-scale`: Handle varying watermark sizes
- `--visualize`: Save detection results to `./debug_detection/`
- `--keep-audio`: Preserve original audio

## 📖 Command-Line Options

```bash
python main.py [OPTIONS]

Options:
  --input PATH             Input video file or directory (required)
  --remove-watermark       Enable watermark removal
  --enhance-video          Enable AI video enhancement
  --keep-audio             Preserve audio from original video
  --no-audio               Explicitly remove audio
  
Examples:
  # KLing video with audio
  python main.py --input kling_video.mp4 --remove-watermark --keep-audio
  
  # Sora video (dynamic watermark - see advanced usage)
  python remove_dynamic_watermark.py --input sora.mp4 --template sora.png
  
  # Batch processing
  python main.py --input ./videos --remove-watermark --keep-audio
```

## 📁 Project Structure

```
Video-WatermarkRemover-Enhancer/
├── 📄 main.py                          # Main entry point
├── 📄 config.yaml                      # Configuration file
├── 📄 requirements.txt                 # Python dependencies
│
├── 📁 modules/                         # Core modules
│   ├── erase.py                       # Watermark removal (fixed)
│   ├── enhance.py                     # Video enhancement
│   └── sttn.py                        # STTN model wrapper
│
├── 📁 utils/                          # Utility functions
│   ├── video_utils.py                 # Video I/O with audio support
│   ├── audio_utils.py                 # Audio processing toolkit
│   ├── watermark_detection.py         # Dynamic watermark detection
│   ├── image_utils.py                 # Image processing
│   └── logging_utils.py               # Logging utilities
│
├── 📁 STTN/                           # Submodule: Video inpainting
├── 📁 Real-ESRGAN/                    # Submodule: Super-resolution
├── 📁 weights/                        # Model weights (download separately)
│
└── 📄 Documentation
    ├── README.md                      # This file
    ├── README_CN.md                   # Chinese version
    ├── WATERMARK_REMOVAL_ANALYSIS.md  # Technical deep dive
    ├── USAGE_EXAMPLES.md              # Detailed usage guide
    ├── COLAB_SETUP_GUIDE.md           # Google Colab instructions
    └── KLing_Watermark_Remover_Colab.ipynb  # Colab notebook
```

## 🔬 How It Works

### Fixed Watermark Removal (KLing)

1. **Extract Frames**: Decompose video into PNG sequence
2. **Create Mask**: Generate mask based on fixed coordinates
3. **STTN Inpainting**: Use Spatio-Temporal Transformer Network to analyze neighboring frames and intelligently fill the masked region
4. **Reconstruct Video**: Reassemble frames with optional audio preservation

### Dynamic Watermark Removal (Sora)

1. **Template Matching**: Detect watermark position in each frame using OpenCV
2. **Multi-Scale Detection**: Handle size variations
3. **Position Smoothing**: Reduce jitter with moving average
4. **Interpolation**: Fill gaps for missed detections
5. **STTN Inpainting**: Apply time-space aware reconstruction
6. **Audio Merge**: Preserve original soundtrack

**Technical Details:** See [WATERMARK_REMOVAL_ANALYSIS.md](WATERMARK_REMOVAL_ANALYSIS.md)

## 📊 Performance

| Setup | GPU | 300 Frames | Speed |
|-------|-----|-----------|-------|
| Local CPU | None | ~60-90 min | Baseline |
| Local GPU (RTX 3080) | Yes | ~8-12 min | **7x faster** |
| Colab T4 GPU | Yes (Free!) | ~5-10 min | **10x faster** |

**Recommendation:** Use Google Colab for fastest processing with free GPU access!

## 🆕 What's New

### Version 2.0 (Current)

✅ **Audio Preservation**
- Keep original audio during watermark removal
- Audio extraction, merging, and syncing tools
- Support for AAC, MP3, and other formats

✅ **Dynamic Watermark Detection**
- Template matching for moving watermarks (Sora)
- Multi-scale detection for size variations
- Position smoothing and interpolation
- Visualization tools for debugging

✅ **Google Colab Support**
- One-click GPU acceleration
- Fixed dependency conflicts
- Direct GitHub integration

✅ **Enhanced Documentation**
- Technical analysis guide
- Comprehensive usage examples
- Troubleshooting FAQ

## 🤝 References & Credits

- [STTN](https://github.com/researchmm/STTN) - Spatio-Temporal Transformer Network for video inpainting
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - Practical image/video super-resolution
- [GFPGAN](https://github.com/TencentARC/GFPGAN) - Practical face restoration algorithm
- Original project: [chenwr727/KLing-Video-WatermarkRemover-Enhancer](https://github.com/chenwr727/KLing-Video-WatermarkRemover-Enhancer)

## 📚 Documentation

- **[Technical Analysis](WATERMARK_REMOVAL_ANALYSIS.md)** - Deep dive into algorithms and architecture
- **[Usage Examples](USAGE_EXAMPLES.md)** - Comprehensive guide with code samples
- **[Colab Guide](COLAB_SETUP_GUIDE.md)** - Google Colab setup and troubleshooting
- **[Modifications Summary](PROJECT_MODIFICATIONS_SUMMARY.md)** - Changelog and improvements

## 🐛 Troubleshooting

### Common Issues

**Q: Audio and video out of sync?**
```python
from utils.audio_utils import sync_audio_video
sync_audio_video("video.mp4", "audio.aac", "output.mp4", audio_delay=0.5)
```

**Q: Watermark detection not working?**
- Lower threshold: `--threshold 0.6`
- Try multi-scale: `--multi-scale`
- Enable visualization: `--visualize`
- Check template quality

**Q: Processing too slow?**
- Use Google Colab with GPU (10-50x faster)
- Reduce video resolution
- Lower frame rate

**More solutions:** See [USAGE_EXAMPLES.md - FAQ](USAGE_EXAMPLES.md#common-issues)

## 🌟 Support This Project

If this project helps you, please:
- ⭐ **Star this repository**
- 🍴 **Fork and contribute**
- 🐛 **Report issues**
- 📝 **Share your results**

## 📄 License

This project is open source. Please check individual submodules for their licenses.

## 🔗 Links

- **GitHub Repository**: https://github.com/zelinewang/Video-WatermarkRemover-Enhancer
- **Google Colab**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zelinewang/Video-WatermarkRemover-Enhancer/blob/master/KLing_Watermark_Remover_Colab.ipynb)
- **Original Project**: https://github.com/chenwr727/KLing-Video-WatermarkRemover-Enhancer

---

**Made with ❤️ for the AI video community**

*Works with KLing, Sora, and any video with visible watermarks!*
