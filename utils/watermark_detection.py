"""
动态水印检测工具模块

支持多种水印检测方法：
1. 模板匹配（Template Matching）
2. 深度学习（YOLO/其他）
3. 频域分析（Frequency Analysis）
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm
from PIL import Image


def detect_watermark_template(
    frame: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.7,
    method: int = cv2.TM_CCOEFF_NORMED
) -> Optional[Tuple[int, int, int, int]]:
    """
    使用模板匹配检测水印位置
    
    适用于：水印图案固定，只是位置在不同帧中变化
    
    Args:
        frame: 视频帧 (BGR 格式)
        template: 水印模板图像 (BGR 格式)
        threshold: 匹配阈值，范围 [0, 1]，越高越严格
        method: OpenCV 模板匹配方法
            - cv2.TM_CCOEFF_NORMED: 归一化相关系数（推荐）
            - cv2.TM_CCORR_NORMED: 归一化互相关
            - cv2.TM_SQDIFF_NORMED: 归一化平方差
    
    Returns:
        (x1, y1, x2, y2): 水印边界框坐标，如果未检测到则返回 None
    
    Example:
        >>> frame = cv2.imread("frame_001.png")
        >>> template = cv2.imread("sora_watermark.png")
        >>> bbox = detect_watermark_template(frame, template, threshold=0.8)
        >>> if bbox:
        >>>     x1, y1, x2, y2 = bbox
        >>>     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    """
    # 转换为灰度图像（提高匹配效率和鲁棒性）
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    h, w = gray_template.shape
    
    # 检查模板是否比帧小
    if h > gray_frame.shape[0] or w > gray_frame.shape[1]:
        print(f"警告: 模板尺寸 ({w}x{h}) 大于帧尺寸 ({gray_frame.shape[1]}x{gray_frame.shape[0]})")
        return None
    
    # 执行模板匹配
    result = cv2.matchTemplate(gray_frame, gray_template, method)
    
    # 获取最佳匹配位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # 对于 TM_SQDIFF 和 TM_SQDIFF_NORMED，最小值表示最佳匹配
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        match_val = 1 - min_val  # 反转以便使用相同的阈值逻辑
        top_left = min_loc
    else:
        match_val = max_val
        top_left = max_loc
    
    # 检查匹配度是否超过阈值
    if match_val >= threshold:
        x1, y1 = top_left
        x2, y2 = x1 + w, y1 + h
        return (x1, y1, x2, y2)
    else:
        return None


def detect_watermark_multi_scale(
    frame: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.7,
    scales: List[float] = None
) -> Optional[Tuple[int, int, int, int]]:
    """
    多尺度模板匹配（处理水印大小变化的情况）
    
    Args:
        frame: 视频帧
        template: 水印模板
        threshold: 匹配阈值
        scales: 尺度列表，例如 [0.8, 0.9, 1.0, 1.1, 1.2]
    
    Returns:
        最佳匹配的边界框
    """
    if scales is None:
        scales = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    best_match = None
    best_val = -1
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    for scale in scales:
        # 缩放模板
        h, w = template.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        if new_h <= 0 or new_w <= 0:
            continue
        if new_h > gray_frame.shape[0] or new_w > gray_frame.shape[1]:
            continue
            
        scaled_template = cv2.resize(template, (new_w, new_h))
        gray_template = cv2.cvtColor(scaled_template, cv2.COLOR_BGR2GRAY)
        
        # 模板匹配
        result = cv2.matchTemplate(gray_frame, gray_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # 更新最佳匹配
        if max_val > best_val:
            best_val = max_val
            x1, y1 = max_loc
            x2, y2 = x1 + new_w, y1 + new_h
            best_match = (x1, y1, x2, y2)
    
    # 检查最佳匹配是否达到阈值
    if best_val >= threshold:
        return best_match
    else:
        return None


def detect_watermarks_in_video(
    frame_paths: List[str],
    template_path: str,
    mask_expand: int = 30,
    threshold: float = 0.7,
    multi_scale: bool = False,
    visualize: bool = False,
    output_dir: str = None
) -> List[Optional[Tuple[int, int, int, int]]]:
    """
    为整个视频的每一帧检测水印位置
    
    Args:
        frame_paths: 所有帧的文件路径列表
        template_path: 水印模板图像路径
        mask_expand: 掩码扩展像素数
        threshold: 匹配阈值
        multi_scale: 是否使用多尺度匹配
        visualize: 是否保存可视化结果
        output_dir: 可视化结果保存目录
    
    Returns:
        每一帧的水印位置列表，如果某帧未检测到水印则为 None
    
    Example:
        >>> frame_paths = ["frame_0001.png", "frame_0002.png", ...]
        >>> positions = detect_watermarks_in_video(
        ...     frame_paths, 
        ...     "sora_watermark.png",
        ...     threshold=0.8,
        ...     visualize=True,
        ...     output_dir="./debug_masks"
        ... )
        >>> print(f"检测到水印的帧数: {sum(1 for p in positions if p)}/{len(positions)}")
    """
    # 加载模板
    template = cv2.imread(template_path)
    if template is None:
        raise ValueError(f"无法加载模板图像: {template_path}")
    
    print(f"水印模板尺寸: {template.shape[1]}x{template.shape[0]}")
    
    positions = []
    detection_count = 0
    
    # 如果需要可视化，创建输出目录
    if visualize and output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    for idx, frame_path in enumerate(tqdm(frame_paths, desc="检测水印位置")):
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"警告: 无法读取帧 {frame_path}")
            positions.append(None)
            continue
        
        # 检测水印
        if multi_scale:
            position = detect_watermark_multi_scale(frame, template, threshold)
        else:
            position = detect_watermark_template(frame, template, threshold)
        
        if position:
            # 扩展掩码区域
            x1, y1, x2, y2 = position
            h, w = frame.shape[:2]
            
            x1 = max(0, x1 - mask_expand)
            y1 = max(0, y1 - mask_expand)
            x2 = min(w, x2 + mask_expand)
            y2 = min(h, y2 + mask_expand)
            
            position = (x1, y1, x2, y2)
            detection_count += 1
            
            # 可视化
            if visualize and output_dir:
                vis_frame = frame.copy()
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    vis_frame, 
                    f"Frame {idx+1}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
                output_path = f"{output_dir}/detection_{idx+1:04d}.png"
                cv2.imwrite(output_path, vis_frame)
        
        positions.append(position)
    
    # 统计信息
    print(f"\n检测统计:")
    print(f"  总帧数: {len(positions)}")
    print(f"  检测到水印: {detection_count} 帧 ({detection_count/len(positions)*100:.1f}%)")
    print(f"  未检测到: {len(positions) - detection_count} 帧")
    
    return positions


def smooth_positions(
    positions: List[Optional[Tuple[int, int, int, int]]],
    window_size: int = 5
) -> List[Optional[Tuple[int, int, int, int]]]:
    """
    平滑水印位置轨迹（处理检测抖动）
    
    使用移动平均滤波器平滑水印位置，减少单帧检测误差
    
    Args:
        positions: 原始检测位置列表
        window_size: 移动平均窗口大小
    
    Returns:
        平滑后的位置列表
    """
    smoothed = []
    
    for i in range(len(positions)):
        # 收集窗口内的有效位置
        window_positions = []
        for j in range(max(0, i - window_size // 2), 
                      min(len(positions), i + window_size // 2 + 1)):
            if positions[j] is not None:
                window_positions.append(positions[j])
        
        # 如果窗口内有位置，计算平均值
        if window_positions:
            avg_x1 = int(np.mean([p[0] for p in window_positions]))
            avg_y1 = int(np.mean([p[1] for p in window_positions]))
            avg_x2 = int(np.mean([p[2] for p in window_positions]))
            avg_y2 = int(np.mean([p[3] for p in window_positions]))
            smoothed.append((avg_x1, avg_y1, avg_x2, avg_y2))
        else:
            smoothed.append(None)
    
    return smoothed


def interpolate_missing_positions(
    positions: List[Optional[Tuple[int, int, int, int]]]
) -> List[Optional[Tuple[int, int, int, int]]]:
    """
    插值缺失的水印位置
    
    如果某些帧未检测到水印，但前后帧都有，则进行线性插值
    
    Args:
        positions: 原始位置列表（可能包含 None）
    
    Returns:
        插值后的位置列表
    """
    interpolated = positions.copy()
    
    for i in range(len(positions)):
        if positions[i] is None:
            # 查找前一个有效位置
            prev_idx = None
            for j in range(i - 1, -1, -1):
                if positions[j] is not None:
                    prev_idx = j
                    break
            
            # 查找后一个有效位置
            next_idx = None
            for j in range(i + 1, len(positions)):
                if positions[j] is not None:
                    next_idx = j
                    break
            
            # 如果前后都有，进行线性插值
            if prev_idx is not None and next_idx is not None:
                prev_pos = positions[prev_idx]
                next_pos = positions[next_idx]
                
                # 计算插值权重
                total_gap = next_idx - prev_idx
                weight_next = (i - prev_idx) / total_gap
                weight_prev = 1 - weight_next
                
                # 插值计算
                x1 = int(prev_pos[0] * weight_prev + next_pos[0] * weight_next)
                y1 = int(prev_pos[1] * weight_prev + next_pos[1] * weight_next)
                x2 = int(prev_pos[2] * weight_prev + next_pos[2] * weight_next)
                y2 = int(prev_pos[3] * weight_prev + next_pos[3] * weight_next)
                
                interpolated[i] = (x1, y1, x2, y2)
    
    return interpolated


if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python watermark_detection.py <frame_image> <template_image>")
        sys.exit(1)
    
    frame_path = sys.argv[1]
    template_path = sys.argv[2]
    
    frame = cv2.imread(frame_path)
    template = cv2.imread(template_path)
    
    # 单帧检测测试
    position = detect_watermark_template(frame, template, threshold=0.7)
    
    if position:
        x1, y1, x2, y2 = position
        print(f"检测到水印: ({x1}, {y1}, {x2}, {y2})")
        
        # 可视化
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite("detection_result.png", frame)
        print("结果已保存到 detection_result.png")
    else:
        print("未检测到水印")

