import glob
import os
import subprocess
from typing import List

TEMP_VIDEO_FILE = "tmp.mp4"
TEMP_FRAME_FORMAT = "png"


def run_ffmpeg(args: List[str]) -> bool:
    commands = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    commands.extend(args)
    try:
        subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True
    except Exception as e:
        print(str(e))
        pass
    return False


def detect_fps(target_path: str) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        target_path,
    ]
    output = subprocess.check_output(command).decode().strip().split("/")
    try:
        numerator, denominator = map(int, output)
        return numerator / denominator
    except Exception:
        pass
    return 30


def extract_frames(
    target_path: str, fps: float = 30, temp_frame_quality: int = 1
) -> bool:
    temp_directory_path = get_temp_directory_path(target_path)
    commands = [
        "-hwaccel",
        "auto",
        "-i",
        target_path,
        "-q:v",
        str(temp_frame_quality),
        "-pix_fmt",
        "rgb24",
        "-vf",
        "fps=" + str(fps),
        os.path.join(temp_directory_path, "%04d." + TEMP_FRAME_FORMAT),
    ]
    return run_ffmpeg(commands)


def create_video(
    target_path: str,
    output_path: str,
    fps: float = 30,
    output_video_encoder: str = "libx264",
    keep_audio: bool = False,
) -> bool:
    """
    从帧序列创建视频
    
    Args:
        target_path: 原始视频路径（用于提取音频）
        output_path: 输出视频路径
        fps: 帧率
        output_video_encoder: 视频编码器
        keep_audio: 是否保留原视频的音频
    
    Returns:
        是否成功
    """
    temp_directory_path = get_temp_directory_path(target_path)

    if keep_audio and has_audio_stream(target_path):
        # 方案：先创建无音频视频，再与原音频合并
        temp_video_path = output_path.replace(".mp4", "_temp_no_audio.mp4")
        temp_audio_path = output_path.replace(".mp4", "_temp_audio.aac")
        
        # 1. 从帧创建无音频视频
        commands = [
            "-hwaccel", "auto",
            "-r", str(fps),
            "-i", os.path.join(temp_directory_path, "%04d." + TEMP_FRAME_FORMAT),
            "-c:v", output_video_encoder,
            "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-y", temp_video_path
        ]
        
        if not run_ffmpeg(commands):
            return False
        
        # 2. 提取原视频音频
        audio_commands = [
            "-i", target_path,
            "-vn",
            "-acodec", "copy",
            "-y", temp_audio_path
        ]
        
        if not run_ffmpeg(audio_commands):
            # 音频提取失败，返回无音频视频
            if os.path.exists(temp_video_path):
                os.rename(temp_video_path, output_path)
            return True
        
        # 3. 合并视频和音频
        merge_commands = [
            "-i", temp_video_path,
            "-i", temp_audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            "-y", output_path
        ]
        
        success = run_ffmpeg(merge_commands)
        
        # 4. 清理临时文件
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        return success
    else:
        # 不保留音频或原视频无音频
        commands = [
            "-hwaccel", "auto",
            "-r", str(fps),
            "-i", os.path.join(temp_directory_path, "%04d." + TEMP_FRAME_FORMAT),
            "-c:v", output_video_encoder,
            "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-y", output_path
        ]
        
        return run_ffmpeg(commands)


def has_audio_stream(video_path: str) -> bool:
    """
    检查视频是否包含音频流
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


def get_temp_frame_paths(
    temp_directory_path: str, temp_frame_format: str = TEMP_FRAME_FORMAT
) -> List[str]:
    temp_frame_paths = glob.glob(
        (os.path.join(glob.escape(temp_directory_path), "*." + temp_frame_format))
    )
    temp_frame_paths.sort()
    return temp_frame_paths


def get_temp_directory_path(target_path: str) -> str:
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    target_directory_path = os.path.dirname(target_path)
    temp_directory_path = os.path.join(target_directory_path, target_name)
    os.makedirs(temp_directory_path, exist_ok=True)
    return temp_directory_path
