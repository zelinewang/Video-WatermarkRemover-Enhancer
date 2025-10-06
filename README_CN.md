简体中文 | [English](README.md)

# 🎥 通用视频水印移除与增强工具

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/zelinewang/Video-WatermarkRemover-Enhancer?style=social)](https://github.com/zelinewang/Video-WatermarkRemover-Enhancer/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/zelinewang/Video-WatermarkRemover-Enhancer?style=social)](https://github.com/zelinewang/Video-WatermarkRemover-Enhancer/network/members)
[![GitHub issues](https://img.shields.io/github/issues/zelinewang/Video-WatermarkRemover-Enhancer)](https://github.com/zelinewang/Video-WatermarkRemover-Enhancer/issues)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zelinewang/Video-WatermarkRemover-Enhancer/blob/master/KLing_Watermark_Remover_Colab.ipynb)

**支持 KLing、Sora 等多种视频源的专业 AI 水印移除工具！** 🚀

![Demo](demo.webp)

</div>

## ✨ 核心功能

🎯 **通用水印移除**
- **固定位置水印**（KLing、标准覆盖层）
- **动态位置水印**（Sora、移动 Logo）支持模板匹配
- **多尺度检测** 适应不同大小的水印
- 无损画质，边缘自然平滑
- 支持批量处理

🎨 **AI 视频增强**
- Real-ESRGAN 提供的超分辨率技术
- 智能优化亮度、对比度和清晰度
- GFPGAN 人脸细节增强
- GPU 加速，速度提升 10-50 倍

🎵 **音频保留**
- **新功能**：处理时保留原始音轨
- 提取、合并、同步音频流
- 支持多种音频格式（AAC、MP3 等）

⚡ **高效便捷**
- 简单的命令行操作
- Google Colab 支持，免费 GPU (T4)
- 可自定义处理参数
- 调试可视化工具

## 🎬 支持的视频源

| 平台 | 水印类型 | 检测方法 | 准确率 |
|------|---------|---------|--------|
| **KLing** | 固定位置 | 预设坐标 | 100% |
| **Sora** | 动态位置 | 模板匹配 | 85-95% |
| **自定义** | 任意类型 | 手动配置 | 因情况而异 |

**可处理任何带有可见水印的视频！**

## 🔧 安装

### 方法 1：本地安装（CPU/GPU）

```bash
# 克隆仓库（包含子模块）
git clone --recursive https://github.com/zelinewang/Video-WatermarkRemover-Enhancer.git
cd Video-WatermarkRemover-Enhancer

# 创建 conda 环境
conda create -n watermark-remover python=3.10
conda activate watermark-remover

# 安装依赖
pip install -r requirements.txt
```

### 方法 2：Google Colab（推荐）🌟

**免费 GPU (T4) 访问 - 比 CPU 快 10-50 倍！**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zelinewang/Video-WatermarkRemover-Enhancer/blob/master/KLing_Watermark_Remover_Colab.ipynb)

点击上方徽章或访问：
```
https://colab.research.google.com/github/zelinewang/Video-WatermarkRemover-Enhancer/blob/master/KLing_Watermark_Remover_Colab.ipynb
```

**处理时间对比：**
- CPU：300 帧需要 60-90 分钟
- Colab T4 GPU：5-10 分钟 ⚡

## 🛠️ 配置

`config.yaml` 文件定义了水印移除和视频增强的参数。

### 水印移除设置

```yaml
watermark:
  position: [556, 1233, 701, 1267]  # 固定水印的 [x1, y1, x2, y2] 坐标
  ckpt_p: "./weights/sttn.pth"       # STTN 模型路径
  mask_expand: 30                     # 掩码扩展像素数
  neighbor_stride: 10                 # 时间邻域步长
```

**动态水印（Sora）：** 查看 [动态水印检测指南](USAGE_EXAMPLES.md#场景-2移除-sora-动态水印)

### 视频增强设置

```yaml
enhance:
  RealESRGAN_model_path: "./weights/RealESRGAN_x2plus.pth"  # 超分辨率
  GFPGANer_model_path: "./weights/GFPGANv1.4.pth"           # 人脸增强
```

### 下载模型权重

**水印移除必需：**
- [sttn.pth](https://drive.google.com/file/d/1ZAMV8547wmZylKRt5qR_tC5VlosXD4Wv/view?usp=sharing) (~200MB)

**视频增强可选：**
- [RealESRGAN_x2plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth)
- [GFPGANv1.4.pth](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth)

将下载的模型放入 `./weights/` 目录。

## 🚀 快速开始

### 基础用法

```bash
# 移除水印并保留音频（推荐）
python main.py --input video.mp4 --remove-watermark --keep-audio

# 仅移除水印（不保留音频）
python main.py --input video.mp4 --remove-watermark

# 移除水印 + 视频增强
python main.py --input video.mp4 --remove-watermark --enhance-video --keep-audio

# 处理整个文件夹
python main.py --input videos_folder/ --remove-watermark --keep-audio
```

### 高级用法：动态水印移除（Sora 视频）

**步骤 1：提取水印模板**

```bash
# 从视频中提取一帧
ffmpeg -i sora_video.mp4 -ss 00:00:01 -vframes 1 frame.png

# 使用图像编辑器（GIMP、Photoshop 等）裁剪水印
# 保存为：sora_watermark.png
```

**步骤 2：使用动态移除脚本**

创建 `remove_dynamic_watermark.py`（完整脚本在 [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)）：

```bash
python remove_dynamic_watermark.py \
    --input sora_video.mp4 \
    --template sora_watermark.png \
    --threshold 0.8 \
    --multi-scale \
    --keep-audio \
    --visualize
```

**参数说明：**
- `--threshold`：检测阈值（0.7-0.9，越高越严格）
- `--multi-scale`：处理不同大小的水印
- `--visualize`：保存检测结果到 `./debug_detection/`
- `--keep-audio`：保留原始音频

## 📖 命令行选项

```bash
python main.py [选项]

选项：
  --input PATH             输入视频文件或目录（必需）
  --remove-watermark       启用水印移除
  --enhance-video          启用 AI 视频增强
  --keep-audio             保留原视频音频
  --no-audio               显式移除音频
  
示例：
  # KLing 视频保留音频
  python main.py --input kling_video.mp4 --remove-watermark --keep-audio
  
  # Sora 视频（动态水印 - 参见高级用法）
  python remove_dynamic_watermark.py --input sora.mp4 --template sora.png
  
  # 批量处理
  python main.py --input ./videos --remove-watermark --keep-audio
```

## 📁 项目结构

```
Video-WatermarkRemover-Enhancer/
├── 📄 main.py                          # 主程序入口
├── 📄 config.yaml                      # 配置文件
├── 📄 requirements.txt                 # Python 依赖
│
├── 📁 modules/                         # 核心模块
│   ├── erase.py                       # 水印移除（固定）
│   ├── enhance.py                     # 视频增强
│   └── sttn.py                        # STTN 模型封装
│
├── 📁 utils/                          # 工具函数
│   ├── video_utils.py                 # 视频 I/O（支持音频）
│   ├── audio_utils.py                 # 音频处理工具包
│   ├── watermark_detection.py         # 动态水印检测
│   ├── image_utils.py                 # 图像处理
│   └── logging_utils.py               # 日志工具
│
├── 📁 STTN/                           # 子模块：视频修复
├── 📁 Real-ESRGAN/                    # 子模块：超分辨率
├── 📁 weights/                        # 模型权重（需单独下载）
│
└── 📄 文档
    ├── README.md                      # 英文说明
    ├── README_CN.md                   # 本文件
    ├── WATERMARK_REMOVAL_ANALYSIS.md  # 技术深入分析
    ├── USAGE_EXAMPLES.md              # 详细使用指南
    ├── COLAB_SETUP_GUIDE.md           # Google Colab 说明
    └── KLing_Watermark_Remover_Colab.ipynb  # Colab 笔记本
```

## 🔬 工作原理

### 固定水印移除（KLing）

1. **提取帧**：将视频分解为 PNG 序列
2. **创建掩码**：根据固定坐标生成掩码
3. **STTN 修复**：使用时空转换网络分析相邻帧，智能填充掩码区域
4. **重建视频**：重新组合帧，可选保留音频

### 动态水印移除（Sora）

1. **模板匹配**：使用 OpenCV 检测每帧中的水印位置
2. **多尺度检测**：处理大小变化
3. **位置平滑**：使用移动平均减少抖动
4. **插值**：填补漏检的帧
5. **STTN 修复**：应用时空感知重建
6. **音频合并**：保留原始音轨

**技术细节：** 见 [WATERMARK_REMOVAL_ANALYSIS.md](WATERMARK_REMOVAL_ANALYSIS.md)

## 📊 性能

| 设置 | GPU | 300 帧 | 速度 |
|------|-----|--------|------|
| 本地 CPU | 无 | ~60-90 分钟 | 基准 |
| 本地 GPU (RTX 3080) | 有 | ~8-12 分钟 | **快 7 倍** |
| Colab T4 GPU | 有（免费！） | ~5-10 分钟 | **快 10 倍** |

**推荐：** 使用 Google Colab 免费 GPU 获得最快处理速度！

## 🆕 更新日志

### 版本 2.0（当前）

✅ **音频保留**
- 水印移除时保留原始音频
- 音频提取、合并、同步工具
- 支持 AAC、MP3 等格式

✅ **动态水印检测**
- 移动水印的模板匹配（Sora）
- 多尺度检测处理大小变化
- 位置平滑和插值
- 调试可视化工具

✅ **Google Colab 支持**
- 一键 GPU 加速
- 修复依赖冲突
- 直接 GitHub 集成

✅ **增强文档**
- 技术分析指南
- 全面的使用示例
- 故障排除 FAQ

## 🤝 参考与致谢

- [STTN](https://github.com/researchmm/STTN) - 视频修复的时空转换网络
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - 实用图像/视频超分辨率
- [GFPGAN](https://github.com/TencentARC/GFPGAN) - 实用人脸修复算法
- 原始项目：[chenwr727/KLing-Video-WatermarkRemover-Enhancer](https://github.com/chenwr727/KLing-Video-WatermarkRemover-Enhancer)

## 📚 文档

- **[技术分析](WATERMARK_REMOVAL_ANALYSIS.md)** - 算法和架构深入解析
- **[使用示例](USAGE_EXAMPLES.md)** - 代码示例综合指南
- **[Colab 指南](COLAB_SETUP_GUIDE.md)** - Google Colab 设置和故障排除
- **[修改总结](PROJECT_MODIFICATIONS_SUMMARY.md)** - 更新日志和改进

## 🐛 故障排除

### 常见问题

**问：音视频不同步？**
```python
from utils.audio_utils import sync_audio_video
sync_audio_video("video.mp4", "audio.aac", "output.mp4", audio_delay=0.5)
```

**问：水印检测不工作？**
- 降低阈值：`--threshold 0.6`
- 尝试多尺度：`--multi-scale`
- 启用可视化：`--visualize`
- 检查模板质量

**问：处理太慢？**
- 使用 Google Colab 的 GPU（快 10-50 倍）
- 降低视频分辨率
- 降低帧率

**更多解决方案：** 见 [USAGE_EXAMPLES.md - FAQ](USAGE_EXAMPLES.md#常见问题)

## 🌟 支持本项目

如果这个项目对你有帮助，请：
- ⭐ **给这个仓库加星**
- 🍴 **Fork 并贡献代码**
- 🐛 **报告问题**
- 📝 **分享你的结果**

## 📄 许可证

本项目开源。各子模块请查看其各自的许可证。

## 🔗 链接

- **GitHub 仓库**：https://github.com/zelinewang/Video-WatermarkRemover-Enhancer
- **Google Colab**：[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zelinewang/Video-WatermarkRemover-Enhancer/blob/master/KLing_Watermark_Remover_Colab.ipynb)
- **原始项目**：https://github.com/chenwr727/KLing-Video-WatermarkRemover-Enhancer

---

**为 AI 视频社区用 ❤️ 制作**

*适用于 KLing、Sora 以及任何带有可见水印的视频！*
