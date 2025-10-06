"""
音频处理工具模块

提供视频音频提取、合并等功能
"""

import os
import subprocess
from typing import List, Optional


def run_ffmpeg(args: List[str], verbose: bool = False) -> bool:
    """
    执行 FFmpeg 命令
    
    Args:
        args: FFmpeg 参数列表
        verbose: 是否显示详细输出
    
    Returns:
        执行是否成功
    """
    commands = ["ffmpeg", "-hide_banner"]
    
    if not verbose:
        commands.extend(["-loglevel", "error"])
    
    commands.extend(args)
    
    try:
        if verbose:
            subprocess.check_call(commands)
        else:
            subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg 错误: {e}")
        if verbose:
            print(e.output)
        return False
    except Exception as e:
        print(f"执行错误: {str(e)}")
        return False


def has_audio(video_path: str) -> bool:
    """
    检查视频是否包含音频流
    
    Args:
        video_path: 视频文件路径
    
    Returns:
        是否包含音频
    """
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT).decode().strip()
        return output == "audio"
    except:
        return False


def get_audio_info(video_path: str) -> dict:
    """
    获取视频的音频信息
    
    Args:
        video_path: 视频文件路径
    
    Returns:
        音频信息字典
    """
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_rate,channels,bit_rate",
        "-of", "json",
        video_path
    ]
    
    try:
        import json
        output = subprocess.check_output(command).decode()
        data = json.loads(output)
        
        if "streams" in data and len(data["streams"]) > 0:
            return data["streams"][0]
        else:
            return {}
    except:
        return {}


def extract_audio(
    video_path: str, 
    audio_path: str,
    codec: str = "aac",
    bitrate: str = "192k",
    verbose: bool = False
) -> bool:
    """
    从视频中提取音频
    
    Args:
        video_path: 源视频路径
        audio_path: 输出音频路径
        codec: 音频编码器（aac, mp3, copy 等）
        bitrate: 音频比特率（如果重新编码）
        verbose: 是否显示详细信息
    
    Returns:
        是否成功
    
    Example:
        >>> extract_audio("video.mp4", "audio.aac")
        >>> extract_audio("video.mp4", "audio.mp3", codec="libmp3lame", bitrate="320k")
        >>> extract_audio("video.mp4", "audio.aac", codec="copy")  # 直接复制，不重新编码
    """
    # 检查视频是否有音频
    if not has_audio(video_path):
        print(f"警告: 视频 {video_path} 不包含音频流")
        return False
    
    commands = [
        "-i", video_path,
        "-vn",  # 不处理视频流
    ]
    
    if codec == "copy":
        commands.extend(["-acodec", "copy"])
    else:
        commands.extend([
            "-acodec", codec,
            "-b:a", bitrate
        ])
    
    commands.extend(["-y", audio_path])
    
    if verbose:
        print(f"提取音频: {video_path} -> {audio_path}")
        print(f"编码器: {codec}, 比特率: {bitrate}")
    
    return run_ffmpeg(commands, verbose=verbose)


def merge_video_audio(
    video_path: str,
    audio_path: str,
    output_path: str,
    video_codec: str = "copy",
    audio_codec: str = "aac",
    verbose: bool = False
) -> bool:
    """
    合并视频和音频文件
    
    Args:
        video_path: 视频文件路径
        audio_path: 音频文件路径
        output_path: 输出文件路径
        video_codec: 视频编码器（copy 或 libx264 等）
        audio_codec: 音频编码器（copy 或 aac 等）
        verbose: 是否显示详细信息
    
    Returns:
        是否成功
    
    Example:
        >>> merge_video_audio("video_no_audio.mp4", "audio.aac", "final.mp4")
        >>> merge_video_audio("video.mp4", "audio.mp3", "output.mp4", audio_codec="copy")
    """
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        return False
    
    if not os.path.exists(audio_path):
        print(f"错误: 音频文件不存在: {audio_path}")
        return False
    
    commands = [
        "-i", video_path,
        "-i", audio_path,
        "-c:v", video_codec,
        "-c:a", audio_codec,
        "-map", "0:v:0",  # 使用第一个输入的视频流
        "-map", "1:a:0",  # 使用第二个输入的音频流
        "-shortest",      # 以较短的流为准
        "-y", output_path
    ]
    
    if verbose:
        print(f"合并视频和音频:")
        print(f"  视频: {video_path}")
        print(f"  音频: {audio_path}")
        print(f"  输出: {output_path}")
    
    return run_ffmpeg(commands, verbose=verbose)


def replace_audio(
    video_path: str,
    new_audio_path: str,
    output_path: str,
    video_codec: str = "copy",
    audio_codec: str = "aac",
    verbose: bool = False
) -> bool:
    """
    替换视频的音频轨道
    
    Args:
        video_path: 原视频路径
        new_audio_path: 新音频路径
        output_path: 输出视频路径
        video_codec: 视频编码器
        audio_codec: 音频编码器
        verbose: 是否显示详细信息
    
    Returns:
        是否成功
    """
    return merge_video_audio(
        video_path, 
        new_audio_path, 
        output_path,
        video_codec=video_codec,
        audio_codec=audio_codec,
        verbose=verbose
    )


def add_audio_to_silent_video(
    video_path: str,
    audio_path: str,
    output_path: str,
    loop_audio: bool = False,
    fade_in: float = 0.0,
    fade_out: float = 0.0,
    verbose: bool = False
) -> bool:
    """
    为无音频的视频添加音频
    
    Args:
        video_path: 无音频的视频路径
        audio_path: 音频文件路径
        output_path: 输出视频路径
        loop_audio: 如果音频短于视频，是否循环音频
        fade_in: 音频淡入时长（秒）
        fade_out: 音频淡出时长（秒）
        verbose: 是否显示详细信息
    
    Returns:
        是否成功
    """
    commands = [
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
    ]
    
    # 构建音频滤镜
    audio_filters = []
    
    if loop_audio:
        audio_filters.append("aloop=loop=-1:size=2e+09")
    
    if fade_in > 0:
        audio_filters.append(f"afade=t=in:d={fade_in}")
    
    if fade_out > 0:
        audio_filters.append(f"afade=t=out:st={fade_out}")
    
    if audio_filters:
        commands.extend(["-af", ",".join(audio_filters)])
    
    commands.extend([
        "-c:a", "aac",
        "-b:a", "192k",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        "-y", output_path
    ])
    
    if verbose:
        print(f"添加音频到视频:")
        print(f"  视频: {video_path}")
        print(f"  音频: {audio_path}")
        print(f"  循环: {loop_audio}")
        print(f"  淡入: {fade_in}s, 淡出: {fade_out}s")
    
    return run_ffmpeg(commands, verbose=verbose)


def adjust_audio_volume(
    input_path: str,
    output_path: str,
    volume: float = 1.0,
    verbose: bool = False
) -> bool:
    """
    调整音频音量
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        volume: 音量倍数（1.0 = 原音量，0.5 = 减半，2.0 = 加倍）
        verbose: 是否显示详细信息
    
    Returns:
        是否成功
    """
    commands = [
        "-i", input_path,
        "-af", f"volume={volume}",
        "-c:v", "copy",
        "-y", output_path
    ]
    
    if verbose:
        print(f"调整音量: {volume}x")
    
    return run_ffmpeg(commands, verbose=verbose)


def sync_audio_video(
    video_path: str,
    audio_path: str,
    output_path: str,
    audio_delay: float = 0.0,
    verbose: bool = False
) -> bool:
    """
    同步音频和视频（处理音视频不同步问题）
    
    Args:
        video_path: 视频文件路径
        audio_path: 音频文件路径
        output_path: 输出文件路径
        audio_delay: 音频延迟（秒），正数延迟，负数提前
        verbose: 是否显示详细信息
    
    Returns:
        是否成功
    """
    commands = [
        "-i", video_path,
        "-itsoffset", str(audio_delay),
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        "-y", output_path
    ]
    
    if verbose:
        print(f"同步音视频:")
        print(f"  音频延迟: {audio_delay}s")
    
    return run_ffmpeg(commands, verbose=verbose)


if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  检查音频: python audio_utils.py check <video.mp4>")
        print("  提取音频: python audio_utils.py extract <video.mp4> <audio.aac>")
        print("  合并音视频: python audio_utils.py merge <video.mp4> <audio.aac> <output.mp4>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "check":
        if len(sys.argv) < 3:
            print("需要提供视频文件路径")
            sys.exit(1)
        
        video_path = sys.argv[2]
        
        if has_audio(video_path):
            print(f"✅ 视频包含音频流")
            info = get_audio_info(video_path)
            if info:
                print(f"  编码器: {info.get('codec_name', 'unknown')}")
                print(f("  采样率: {info.get('sample_rate', 'unknown')} Hz")
                print(f"  声道: {info.get('channels', 'unknown')}")
                print(f"  比特率: {info.get('bit_rate', 'unknown')} bps")
        else:
            print(f"❌ 视频不包含音频流")
    
    elif command == "extract":
        if len(sys.argv) < 4:
            print("需要提供: <video.mp4> <audio.aac>")
            sys.exit(1)
        
        video_path = sys.argv[2]
        audio_path = sys.argv[3]
        
        if extract_audio(video_path, audio_path, verbose=True):
            print("✅ 音频提取成功")
        else:
            print("❌ 音频提取失败")
    
    elif command == "merge":
        if len(sys.argv) < 5:
            print("需要提供: <video.mp4> <audio.aac> <output.mp4>")
            sys.exit(1)
        
        video_path = sys.argv[2]
        audio_path = sys.argv[3]
        output_path = sys.argv[4]
        
        if merge_video_audio(video_path, audio_path, output_path, verbose=True):
            print("✅ 音视频合并成功")
        else:
            print("❌ 音视频合并失败")
    
    else:
        print(f"未知命令: {command}")
        sys.exit(1)

