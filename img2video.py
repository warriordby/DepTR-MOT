import os
import cv2
import numpy as np
from PIL import Image
import argparse

# 尝试导入自然排序库，若未安装则提示
try:
    from natsort import natsorted
except ImportError:
    print("错误：缺少 'natsort' 库，请先安装：pip install natsort")
    exit(1)


def images_to_video(img_dir, output_path, fps=30):
    """
    将文件夹中的图片合成视频
    :param img_dir: 图片所在文件夹路径
    :param output_path: 输出视频路径（如"output.mp4"）
    :param fps: 视频帧率，默认30
    """
    # 1. 筛选并获取文件夹内的所有图片文件
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']  # 支持的图片格式
    img_files = []
    
    for file in os.listdir(img_dir):
        # 获取文件扩展名并转为小写
        file_ext = os.path.splitext(file)[1].lower()
        if file_ext in supported_extensions:
            # 拼接完整路径
            img_path = os.path.join(img_dir, file)
            img_files.append(img_path)
    
    # 检查是否有图片
    if not img_files:
        print(f"错误：在文件夹 '{img_dir}' 中未找到任何图片文件（支持格式：{supported_extensions}）")
        return
    
    # 按文件名自然排序（确保"2.jpg"在"10.jpg"前）
    img_files = natsorted(img_files)
    print(f"找到 {len(img_files)} 张图片，将按以下顺序合成：")
    for i, path in enumerate(img_files[:5]):  # 只显示前5张，避免过多输出
        print(f"  {i+1}. {os.path.basename(path)}")
    if len(img_files) > 5:
        print(f"  ... 省略剩余 {len(img_files)-5} 张图片")
    
    # 2. 确定统一的图片尺寸（以第一张图片为准）
    try:
        with Image.open(img_files[0]) as first_img:
            # 转换为RGB（处理透明通道）
            first_img_rgb = first_img.convert('RGB')
            # 获取尺寸（宽，高）
            target_width, target_height = first_img_rgb.size
            print(f"\n统一图片尺寸为：宽 {target_width}px × 高 {target_height}px")
    except Exception as e:
        print(f"错误：读取第一张图片时失败：{e}")
        return
    
    # 3. 创建视频写入对象
    # 定义编码器（mp4格式推荐使用mp4v，兼容性较好）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 初始化VideoWriter（输出路径、编码器、帧率、尺寸）
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
    if not video_writer.isOpened():
        print(f"错误：无法创建视频文件 '{output_path}'，请检查路径是否合法或编码器是否支持")
        return
    
    # 4. 遍历图片并写入视频
    print("\n开始合成视频...")
    for idx, img_path in enumerate(img_files):
        try:
            # 用PIL读取图片（支持更多格式）
            with Image.open(img_path) as img:
                # 转换为RGB（处理透明通道和灰度图）
                img_rgb = img.convert('RGB')
                # 调整尺寸（按目标尺寸缩放，使用LANCZOS算法保持清晰度）
                img_resized = img_rgb.resize((target_width, target_height), Image.Resampling.LANCZOS)
                # 转换为OpenCV格式（PIL是RGB，OpenCV需要BGR）
                frame = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
                # 写入视频
                video_writer.write(frame)
            
            # 显示进度
            if (idx + 1) % 10 == 0 or (idx + 1) == len(img_files):
                print(f"进度：{idx+1}/{len(img_files)} 张图片已处理")
        
        except Exception as e:
            print(f"警告：处理图片 '{os.path.basename(img_path)}' 时出错，已跳过：{e}")
            continue
    
    # 5. 释放资源
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"\n视频合成完成！输出路径：{os.path.abspath(output_path)}")


if __name__ == "__main__":
    # 命令行参数配置
    parser = argparse.ArgumentParser(description="将文件夹中的图片批量合成视频")
    parser.add_argument("--img_dir", required=True, help="图片所在的文件夹路径（必填），例如：./my_images")
    parser.add_argument("--output", required=True, help="输出视频的路径（必填），例如：./output.mp4")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率（可选），默认30帧/秒")
    args = parser.parse_args()

    # 调用合成函数
    images_to_video(
        img_dir=args.img_dir,
        output_path=args.output,
        fps=args.fps
    )