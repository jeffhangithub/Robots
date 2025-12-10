import os
import subprocess
import tempfile
import json
from pathlib import Path

def get_video_info(video_path):
    """获取视频的分辨率和帧率信息"""
    cmd = [
        'ffmpeg', '-i', video_path,
        '-hide_banner', '-loglevel', 'error',
        '-print_format', 'json', '-show_streams'
    ]
    try:
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True
        )
        info = json.loads(result.stdout)
        # 查找视频流
        for stream in info['streams']:
            if stream['codec_type'] == 'video':
                width = int(stream['width'])
                height = int(stream['height'])
                # 尝试获取实际帧率
                r_frame_rate = stream.get('r_frame_rate', '30/1')
                fps = eval(r_frame_rate) if '/' in r_frame_rate else float(r_frame_rate)
                return {'width': width, 'height': height, 'fps': fps}
        return {'width': 640, 'height': 480, 'fps': 30}  # 默认值
    except:
        return {'width': 640, 'height': 480, 'fps': 30}

def merge_videos(video1_path, video2_path, output_path, target_resolution=None, target_fps=None):
    """
    合并两个分辨率和帧数不一致的视频
    
    Args:
        video1_path: 第一个视频路径
        video2_path: 第二个视频路径
        output_path: 合并后的视频路径
        target_resolution: 目标分辨率 (width, height)，None表示自动选择
        target_fps: 目标帧率，None表示自动选择
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 临时文件路径
    temp_dir = tempfile.mkdtemp()
    video1_fixed = os.path.join(temp_dir, "video1_fixed.mp4")
    video2_fixed = os.path.join(temp_dir, "video2_fixed.mp4")
    input_list = os.path.join(temp_dir, "input.txt")
    
    try:
        # 获取视频信息
        info1 = get_video_info(video1_path)
        info2 = get_video_info(video2_path)
        print(f"视频1信息: 分辨率 {info1['width']}x{info1['height']}, 帧率 {info1['fps']}fps")
        print(f"视频2信息: 分辨率 {info2['width']}x{info2['height']}, 帧率 {info2['fps']}fps")
        
        # 确定目标分辨率和帧率
        if target_resolution is None:
            # 选择较大的分辨率作为目标，或使用默认值
            target_width = max(info1['width'], info2['width'], 1280)
            target_height = max(info1['height'], info2['height'], 720)
            target_resolution = (target_width, target_height)
        
        if target_fps is None:
            # 选择常见帧率或两者的平均值
            target_fps = min(info1['fps'], info2['fps'], 60)
            target_fps = max(target_fps, 24)  # 不低于24fps
            target_fps = round(target_fps)
        
        print(f"目标参数: 分辨率 {target_resolution[0]}x{target_resolution[1]}, 帧率 {target_fps}fps")
        
        # 步骤1: 统一分辨率和帧率
        print(f"正在处理视频1...")
        process_video(video1_path, video1_fixed, target_resolution, target_fps)
        
        print(f"正在处理视频2...")
        process_video(video2_path, video2_fixed, target_resolution, target_fps)
        
        # 步骤2: 创建输入列表文件
        with open(input_list, 'w') as f:
            f.write(f"file '{video1_fixed}'\n")
            f.write(f"file '{video2_fixed}'\n")
        
        # 步骤3: 合并视频
        print("正在合并视频...")
        merge_videos_with_ffmpeg(input_list, output_path)
        
        print(f"视频合并完成，保存至: {output_path}")
    except Exception as e:
        print(f"合并过程中出错: {e}")
    finally:
        # 清理临时文件
        print("清理临时文件...")
        try:
            os.remove(video1_fixed)
            os.remove(video2_fixed)
            os.remove(input_list)
            os.rmdir(temp_dir)
        except:
            pass

def process_video(input_path, output_path, resolution, fps):
    """统一视频的分辨率和帧率"""
    width, height = resolution
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2',
        '-r', str(fps),
        '-c:v', 'libx264', '-crf', '23',
        '-c:a', 'aac', '-b:a', '128k',
        output_path
    ]
    run_ffmpeg_command(cmd)

def merge_videos_with_ffmpeg(input_list, output_path):
    """使用FFmpeg合并视频"""
    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat', '-safe', '0',
        '-i', input_list,
        '-c', 'copy',
        output_path
    ]
    run_ffmpeg_command(cmd)

def run_ffmpeg_command(cmd):
    """执行FFmpeg命令并处理输出"""
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # 实时输出FFmpeg进度
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg命令执行失败: {' '.join(cmd)}")
    except FileNotFoundError:
        raise FileNotFoundError("未找到FFmpeg，请先安装FFmpeg并添加到系统路径")
    except Exception as e:
        raise Exception(f"执行命令时出错: {' '.join(cmd)}\n{str(e)}")