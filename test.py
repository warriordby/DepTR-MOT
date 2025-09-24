import torch

# 原始权重路径
ckpt_path = "/root/autodl-tmp/trackwithdepth/output/dfine_hgnetv2_l_custom/last.pth"
# 新保存路径
new_ckpt_path = "/root/autodl-tmp/trackwithdepth/output/dfine_hgnetv2_l_custom/last_filtered.pth"

# 加载权重
checkpoint = torch.load(ckpt_path, map_location="cpu")

# 假设权重在 checkpoint['model']
state_dict = checkpoint['model']

# 过滤掉所有以 depth_pred 开头的权重
filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("depth_pred")}

# 更新 checkpoint
checkpoint['model'] = filtered_state_dict

# 保存新权重文件
torch.save(checkpoint, new_ckpt_path)

print(f"已删除 depth_pred 权重并保存到 {new_ckpt_path}")
