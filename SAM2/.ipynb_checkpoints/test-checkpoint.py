import torch
import numpy as np
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 加载模型
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# 读取图像
image = plt.imread("test.jpg")
predictor.set_image(image)

# 定义一组框 (xyxy 格式: x_min, y_min, x_max, y_max)
input_boxes = np.array([
    [100, 50, 200, 150],   # 框1
    [250, 80, 350, 180],   # 框2
    [400, 120, 500, 220],  # 框3
])

# 并行生成所有框对应的掩码
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,    # 一次传入所有框
        multimask_output=False,  # 每个框只输出一个掩码
    )

print(f"共生成 {len(masks)} 个掩码，每个 shape={masks[0].shape}")

# 可视化结果
plt.figure(figsize=(10, 10))
plt.imshow(image)
for i, mask in enumerate(masks):
    plt.contour(mask, colors=[np.random.rand(3,)], linewidths=2)
plt.axis("off")
plt.show()

# 保存第一张 mask 作为可视化
mask = masks[0].astype(np.uint8) * 255
cv2.imwrite("output_mask.png", mask)
