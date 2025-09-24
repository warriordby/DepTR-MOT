import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image, ImageDraw
import numpy as np
import random
import os
from sam2.build_sam import build_sam2
# 更改脚本中的gt_file_path和root_path为对应场景数据路径
# 会为每个实例保存图片到对应instance_i文件夹
# 使用img2video脚本可实现将文件夹内图片转换成视频
# 脚本读取

def read_txt_file(file_path):
    data_dict = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # 去除行尾的换行符并按逗号分割数据
                values = line.strip().split(',')
                frame = int(values[0])
                target = int(values[1])
                coordinates = [int(val) for val in values[2:6]]

                if frame not in data_dict:
                    data_dict[frame] = {}

                data_dict[frame][target] = coordinates

    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到。")
    except Exception as e:
        print(f"错误: 发生了一个未知错误: {e}")

    return data_dict

def convert_coordinates(data_dict):
    xyxy_dict = {}
    cxcy_dict = {}
    for frame, targets in data_dict.items():
        xyxy_dict[frame] = {}
        cxcy_dict[frame] = {}
        cxcy_list=[]
        for target, coord in targets.items():
            x, y, w, h = coord
            # 转换为 xyxy 格式
            xyxy = [x, y, x + w, y + h]
            xyxy_dict[frame][target] = xyxy
            # 转换为 cxcy 格式
            cx = x + w // 2
            cy = y + h // 2
            cxcy = [cx, cy] #, w, h]
            cxcy_list.append(cxcy)
        cxcy_dict[frame] = np.array(cxcy_list)

    return xyxy_dict, cxcy_dict
from natsort import natsorted
from torchvision import transforms
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
#本脚本对数据集自带的边界框进行处理边界框为xy wh格式
root_path='/root/autodl-tmp/DanceTrack/train'
out_dir='./out_put_for_train'
video_list=natsorted([i for i in os.listdir(root_path) if 'dancetrack' in i])

for scne in video_list:
    
    gt_file_path = os.path.join(root_path,scne,'gt/gt.txt')  # 请替换为实际的文件路径
    result = read_txt_file(gt_file_path)
    xyxy_result, cxcy_result = convert_coordinates(result)
    
    # 加载模型
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    img_root_path = os.path.join(root_path, scne,'img1')
    
    img_list = natsorted([f for f in os.listdir(img_root_path) if f.lower().endswith('.jpg')])
    print('len(img_list)',len(img_list))
    transform = transforms.Compose([
            transforms.ToTensor(),  # [H,W,C] (0-255 PIL) -> [C,H,W] (0-1 Tensor)
            # 如果模型有特定均值方差，可以加 normalize
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225]),
        ])
    for i, img in enumerate(img_list):
        boxes = xyxy_result[i + 1]
        img_path = os.path.join(img_root_path, img)
        for j, box in boxes.items():
            instance_subdir = os.path.join(out_dir, scne,f'instance_{j}')
            if not os.path.exists(instance_subdir):
                os.makedirs(instance_subdir, exist_ok=True)
            output_filename = os.path.join(instance_subdir, f'{os.path.splitext(img)[0]}.png')
            # if os.path.exists(output_filename):
            #     continue
            # try:
                # 打开图像并设置到预测器
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                image = Image.open(img_path)
                # img_tensor = transform(image).unsqueeze(0)
                predictor.set_image(image)
                masks, scores, logits = predictor.predict(
                    box=box,
                    multimask_output=False
                )
                # 对单个 box 的掩码计算分数
                sorted_ind = np.argsort(scores)[::-1]
                masks = masks[sorted_ind]
                scores = scores[sorted_ind]
                logits = logits[sorted_ind]
                best_mask = masks[0]  # 选取分数最高的掩码

                # 获取图像的高度和宽度
                H, W = best_mask.shape

                # 创建一个全零的 RGB 图像数组
                colored_mask_image = np.zeros((H, W, 3), dtype=np.uint8)

                # 为最终叠加的 mask 生成随机颜色
                color = [0, 255, 0]
                # 根据 mask 的位置填充颜色
                colored_mask_image[best_mask > 0] = color

                # 创建 PIL 图像对象
                colored_mask_pil = Image.fromarray(colored_mask_image)
                # 创建绘图对象
                draw = ImageDraw.Draw(colored_mask_pil)
                # 绘制红色矩形框
                x1, y1, x2, y2 = box
                draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=2)
                
                
                colored_mask_pil.save(output_filename)
                print(f"保存图像: {output_filename}")
    
            # except Exception as e:
            #     print(f"处理图像 {img} 的实例 {j} 时出错: {e}")
    
