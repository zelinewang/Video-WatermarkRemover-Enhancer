# 🎯 Google Colab 安装指南

## 问题说明

Google Colab 环境已更新到更新的 Python 和包版本，导致原始的依赖配置不兼容。

### 主要冲突：
1. ❌ PyTorch 2.0.1 在新 Colab 中不可用（最低 2.2.0）
2. ❌ Colab 预装的 OpenCV 需要 numpy >= 2.0
3. ❌ 旧版 basicsr 1.4.2 需要 numpy < 2.0

## ✅ 解决方案

已更新 `KLing_Watermark_Remover_Colab.ipynb`，使用以下策略：

### 新的依赖安装策略：
1. ✅ 使用 Colab 预装的 **PyTorch 2.x**（已包含 CUDA 支持）
2. ✅ 使用 Colab 预装的 **numpy 2.x**
3. ✅ 从源码安装 **最新版 basicsr**（兼容 numpy 2.x）
4. ✅ 安装 gfpgan 和 realesrgan

### 测试过的环境：
- Python: 3.10+
- PyTorch: 2.2.0+
- CUDA: 12.x
- numpy: 2.0+

## 🚀 快速使用

1. **上传 notebook 到 Colab**
   - 打开 https://colab.research.google.com/
   - 上传 `KLing_Watermark_Remover_Colab.ipynb`

2. **选择 GPU 运行时**
   - 代码执行程序 → 更改运行时类型 → T4 GPU

3. **按顺序运行单元格**
   - Step 1: 安装依赖（约 3-5 分钟）
   - Step 2: 下载模型（约 1-2 分钟）
   - Step 3: 上传视频
   - Step 4: 处理视频（5-15 分钟）
   - Step 5: 下载结果

## 🔧 如果仍然遇到问题

### 方案 A：完全重置 Colab 环境
```python
# 在 Colab 第一个单元格运行
!pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
!pip install opencv-python
```

### 方案 B：使用兼容性检查
```python
# 检查关键包版本
import sys
import torch
import numpy as np

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"NumPy: {np.__version__}")
```

## 📊 性能对比

| 环境 | 处理 300 帧视频 | GPU 占用 |
|------|----------------|----------|
| CPU 服务器 | ~60-90 分钟 | N/A |
| Colab T4 GPU | ~5-10 分钟 | ~6-8GB |
| Colab A100 GPU | ~2-5 分钟 | ~8-10GB |

## ⚠️ Colab 使用注意事项

1. **免费版限制**：
   - 连续使用时长：12 小时
   - 空闲超时：90 分钟
   - GPU 配额限制（每日/每周）

2. **建议**：
   - 处理完一个视频立即下载
   - 不要长时间占用 GPU
   - 处理完清理临时文件

3. **如果 GPU 配额用完**：
   - 等待 24 小时重置
   - 或升级到 Colab Pro（$9.99/月）
   - 或使用本地 GPU 服务器

## 🆘 常见错误

### Error: "No matching distribution found for torch==2.0.1"
**原因**: Colab 不再提供旧版本 PyTorch  
**解决**: 使用更新的 notebook（已修复）

### Error: "numpy 1.26.4 is incompatible"
**原因**: numpy 版本冲突  
**解决**: 使用从源码安装的 basicsr（已修复）

### Error: "CUDA out of memory"
**原因**: GPU 内存不足  
**解决**: 
1. 重启运行时释放内存
2. 处理较短的视频
3. 降低视频分辨率

## 📞 获取帮助

- 项目 Issues: https://github.com/chenwr727/KLing-Video-WatermarkRemover-Enhancer/issues
- Colab 文档: https://colab.research.google.com/notebooks/welcome.ipynb
