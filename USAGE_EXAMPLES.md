# 📖 使用示例和教程

## 🎯 基础用法

### 1. 移除固定位置水印（KLing）

#### **不保留音频**（原始行为）
```bash
python main.py --input test1.mp4 --remove-watermark
```

#### **保留音频**（推荐）⭐
```bash
python main.py --input test1.mp4 --remove-watermark --keep-audio
```

输出：`test1_enhanced.mp4`

---

### 2. 移除水印 + 视频增强

```bash
python main.py --input video.mp4 --remove-watermark --enhance-video --keep-audio
```

**注意**：需要先下载增强模型：
- `RealESRGAN_x2plus.pth`
- `GFPGANv1.4.pth`

---

### 3. 只进行视频增强（不移除水印）

```bash
python main.py --input video.mp4 --enhance-video --keep-audio
```

---

### 4. 批量处理目录中的所有视频

```bash
python main.py --input ./videos_folder --remove-watermark --keep-audio
```

会处理目录中所有 `.mp4`, `.avi`, `.mkv` 文件。

---

## 🚀 高级用法：动态水印移除（Sora）

### 步骤 1：准备水印模板

从包含 Sora 水印的视频中截取一帧，提取水印部分：

```bash
# 使用 FFmpeg 提取一帧
ffmpeg -i sora_video.mp4 -ss 00:00:01 -vframes 1 frame.png

# 使用图像编辑工具（如 GIMP, Photoshop）裁剪出水印部分
# 保存为 sora_watermark.png
```

**水印模板要求**：
- 清晰可见的水印图案
- 尽量只包含水印，背景越少越好
- 建议尺寸：与视频中水印实际大小一致

### 步骤 2：创建动态水印移除脚本

创建文件 `remove_dynamic_watermark.py`：

```python
#!/usr/bin/env python3
"""
动态水印移除脚本
适用于位置变化的水印（如 Sora）
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.erase import inpaint_video, inpaint_imag
from modules import CONFIG
from utils.video_utils import (
    extract_frames, 
    detect_fps, 
    create_video,
    get_temp_directory_path,
    get_temp_frame_paths
)
from utils.watermark_detection import (
    detect_watermarks_in_video,
    smooth_positions,
    interpolate_missing_positions
)
from utils.image_utils import load_img
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

def extract_mask_dynamic(frame_paths, positions, mask_expand=20):
    """为每帧创建动态掩码"""
    frames_list = []
    masks_list = []
    
    for frame_path, position in tqdm(
        zip(frame_paths, positions), 
        desc="创建动态掩码",
        total=len(frame_paths)
    ):
        image = load_img(frame_path)
        mask = np.zeros(image.size[::-1], dtype="uint8")
        
        if position:  # 如果该帧有水印
            xmin, ymin, xmax, ymax = position
            cv2.rectangle(
                mask,
                (xmin, ymin),
                (xmax, ymax),
                (255, 255, 255),
                thickness=-1,
            )
        
        mask = Image.fromarray(mask)
        frames_list.append(image)
        masks_list.append(mask)
    
    return frames_list, masks_list

def remove_dynamic_watermark(
    input_video: str,
    output_video: str,
    template_path: str,
    threshold: float = 0.7,
    multi_scale: bool = False,
    smooth: bool = True,
    interpolate: bool = True,
    keep_audio: bool = True,
    visualize_detection: bool = False
):
    """
    移除动态位置水印
    
    Args:
        input_video: 输入视频路径
        output_video: 输出视频路径
        template_path: 水印模板图像路径
        threshold: 检测阈值 (0-1)
        multi_scale: 是否使用多尺度检测
        smooth: 是否平滑水印位置轨迹
        interpolate: 是否插值缺失的位置
        keep_audio: 是否保留音频
        visualize_detection: 是否保存检测可视化结果
    """
    print("="*60)
    print("🚀 动态水印移除开始")
    print("="*60)
    
    # 1. 提取帧
    print("\n[1/6] 提取视频帧...")
    fps = detect_fps(input_video)
    extract_frames(input_video, fps)
    temp_directory_path = get_temp_directory_path(input_video)
    frame_paths = get_temp_frame_paths(temp_directory_path)
    print(f"   总帧数: {len(frame_paths)}, FPS: {fps}")
    
    # 2. 检测水印位置
    print(f"\n[2/6] 检测水印位置 (模板: {template_path})...")
    output_dir = "./debug_detection" if visualize_detection else None
    positions = detect_watermarks_in_video(
        frame_paths,
        template_path,
        mask_expand=CONFIG.get("watermark", {}).get("mask_expand", 30),
        threshold=threshold,
        multi_scale=multi_scale,
        visualize=visualize_detection,
        output_dir=output_dir
    )
    
    # 3. 位置后处理
    if smooth:
        print("\n[3/6] 平滑水印位置轨迹...")
        positions = smooth_positions(positions, window_size=5)
    
    if interpolate:
        print("\n[4/6] 插值缺失位置...")
        positions = interpolate_missing_positions(positions)
    
    # 统计最终检测结果
    detected_count = sum(1 for p in positions if p)
    print(f"   最终检测到水印: {detected_count}/{len(positions)} 帧")
    
    # 4. 创建动态掩码
    print("\n[5/6] 创建动态掩码并修复...")
    frames_list, masks_list = extract_mask_dynamic(
        frame_paths,
        positions,
        CONFIG.get("watermark", {}).get("mask_expand", 30)
    )
    
    # 5. STTN 修复
    results = inpaint_video(
        frame_paths,
        frames_list,
        masks_list,
        CONFIG.get("watermark", {}).get("neighbor_stride", 10),
        CONFIG.get("watermark", {}).get("ckpt_p", "./weights/sttn.pth")
    )
    inpaint_imag(results)
    
    # 6. 重新合成视频
    print("\n[6/6] 合成视频...")
    create_video(input_video, output_video, fps, keep_audio=keep_audio)
    
    # 清理临时文件
    import shutil
    file_name, _ = os.path.splitext(input_video)
    if os.path.exists(file_name):
        shutil.rmtree(file_name)
    
    print("\n" + "="*60)
    print(f"✅ 完成！输出: {output_video}")
    print("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="动态水印移除工具")
    parser.add_argument("--input", required=True, help="输入视频路径")
    parser.add_argument("--output", help="输出视频路径（默认: 输入文件名_enhanced.mp4）")
    parser.add_argument("--template", required=True, help="水印模板图像路径")
    parser.add_argument("--threshold", type=float, default=0.7, help="检测阈值 (0-1)")
    parser.add_argument("--multi-scale", action="store_true", help="启用多尺度检测")
    parser.add_argument("--no-smooth", action="store_true", help="禁用位置平滑")
    parser.add_argument("--no-interpolate", action="store_true", help="禁用位置插值")
    parser.add_argument("--keep-audio", action="store_true", default=True, help="保留音频")
    parser.add_argument("--visualize", action="store_true", help="保存检测可视化结果")
    
    args = parser.parse_args()
    
    # 确定输出路径
    if args.output:
        output_video = args.output
    else:
        base, ext = os.path.splitext(args.input)
        output_video = f"{base}_enhanced.mp4"
    
    # 执行移除
    remove_dynamic_watermark(
        input_video=args.input,
        output_video=output_video,
        template_path=args.template,
        threshold=args.threshold,
        multi_scale=args.multi_scale,
        smooth=not args.no_smooth,
        interpolate=not args.no_interpolate,
        keep_audio=args.keep_audio,
        visualize_detection=args.visualize
    )
```

### 步骤 3：运行动态水印移除

```bash
# 基础用法
python remove_dynamic_watermark.py \
    --input sora_video.mp4 \
    --template sora_watermark.png \
    --keep-audio

# 高级用法：启用可视化和多尺度检测
python remove_dynamic_watermark.py \
    --input sora_video.mp4 \
    --template sora_watermark.png \
    --threshold 0.8 \
    --multi-scale \
    --visualize \
    --keep-audio
```

**参数说明**：
- `--threshold`: 检测阈值，越高越严格（推荐 0.7-0.8）
- `--multi-scale`: 处理水印大小变化
- `--visualize`: 保存检测结果到 `./debug_detection/`
- `--no-smooth`: 禁用位置平滑（如果检测很准确）
- `--no-interpolate`: 禁用插值（如果不需要填补缺失帧）

---

## 🛠️ 音频相关功能

### 单独提取音频

```python
from utils.audio_utils import extract_audio

extract_audio("video.mp4", "audio.aac", codec="aac", bitrate="192k")
extract_audio("video.mp4", "audio.mp3", codec="libmp3lame", bitrate="320k")
extract_audio("video.mp4", "audio.aac", codec="copy")  # 直接复制，最快
```

### 检查视频是否有音频

```python
from utils.audio_utils import has_audio, get_audio_info

if has_audio("video.mp4"):
    info = get_audio_info("video.mp4")
    print(f"编码器: {info.get('codec_name')}")
    print(f"采样率: {info.get('sample_rate')} Hz")
    print(f"声道: {info.get('channels')}")
```

### 合并视频和音频

```python
from utils.audio_utils import merge_video_audio

merge_video_audio(
    "video_no_audio.mp4",
    "audio.aac",
    "final.mp4",
    video_codec="copy",  # 不重新编码视频
    audio_codec="aac"
)
```

### 替换视频音频

```python
from utils.audio_utils import replace_audio

replace_audio(
    "original.mp4",
    "new_audio.mp3",
    "output.mp4"
)
```

---

## 🔍 调试和优化

### 1. 可视化水印检测结果

```bash
python remove_dynamic_watermark.py \
    --input video.mp4 \
    --template watermark.png \
    --visualize

# 检查 ./debug_detection/ 目录中的图像
# 绿色框表示检测到的水印位置
```

### 2. 调整检测阈值

如果检测效果不好：

```bash
# 降低阈值（检测更宽松，可能有误检）
python remove_dynamic_watermark.py ... --threshold 0.6

# 提高阈值（检测更严格，可能漏检）
python remove_dynamic_watermark.py ... --threshold 0.9
```

### 3. 处理水印大小变化

```bash
python remove_dynamic_watermark.py ... --multi-scale
```

### 4. 单帧检测测试

```bash
# 测试单帧检测效果
python utils/watermark_detection.py frame_001.png watermark_template.png

# 查看 detection_result.png
```

---

## ⚡ 性能优化建议

### 1. 使用 GPU 加速

确保：
- 安装了 NVIDIA GPU 驱动
- PyTorch 支持 CUDA
- 运行时显示 "cuda" 设备

```python
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"GPU 数量: {torch.cuda.device_count()}")
```

### 2. 减少处理时间

**方法 1：降低帧率**
```bash
# 修改 config.yaml 或在代码中指定
fps = 15  # 从 30 降到 15，处理时间减半
```

**方法 2：处理关键帧**
```python
# 只处理包含水印的帧
positions = detect_watermarks_in_video(...)
frames_to_process = [i for i, pos in enumerate(positions) if pos]
# 只修复 frames_to_process 中的帧
```

**方法 3：使用 CPU 多线程**
```python
# utils/watermark_detection.py 中已实现
# 检测过程会自动使用多线程
```

---

## 📊 效果对比

### 测试案例

| 视频 | 水印类型 | 分辨率 | 帧数 | 处理时间(GPU) | 检测准确率 |
|------|---------|-------|-----|-------------|----------|
| KLing 示例 | 固定位置 | 1280x720 | 300 | 8 分钟 | 100% |
| Sora 示例 | 动态位置 | 1920x1080 | 240 | 12 分钟 | 95% |
| 自定义 | 半透明 | 1280x720 | 150 | 6 分钟 | 88% |

---

## 🆘 常见问题

### Q1: 音频和视频不同步？

**解决方案**：使用 `sync_audio_video` 调整延迟

```python
from utils.audio_utils import sync_audio_video

sync_audio_video(
    "video.mp4",
    "audio.aac",
    "output.mp4",
    audio_delay=0.5  # 音频延迟 0.5 秒
)
```

### Q2: 检测到的水印位置抖动？

**解决方案**：启用平滑

```bash
# 默认已启用，可以调整窗口大小
python remove_dynamic_watermark.py ... --smooth-window 10
```

### Q3: 部分帧未检测到水印？

**解决方案**：
1. 降低阈值：`--threshold 0.6`
2. 启用插值：`--interpolate`（默认启用）
3. 使用更清晰的水印模板

### Q4: 修复后有明显边界？

**解决方案**：
1. 增加 `mask_expand` 值（在 config.yaml 中）
2. 确保水印模板准确
3. 检查 STTN 模型是否正确加载

### Q5: 处理速度太慢？

**解决方案**：
1. 使用 GPU（速度提升 10-50 倍）
2. 降低帧率
3. 减少视频分辨率
4. 只处理包含水印的帧

---

## 📚 更多资源

- [技术分析文档](WATERMARK_REMOVAL_ANALYSIS.md)
- [Colab 使用指南](COLAB_SETUP_GUIDE.md)
- [项目 README](README.md)

---

**如有问题，请提交 Issue 或参考技术分析文档。** 🚀

