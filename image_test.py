from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
# 读取图像
image_path = '/media/ssd8T/wyw/Data/NTIRE2025/LSDIR/gt/0059769.png'
image = Image.open(image_path)

# 将图像转换为 PyTorch 张量
image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0  # 转换为 [C, H, W] 并归一化到 [0, 1]
image_tensor = image_tensor.unsqueeze(0)  # 添加 batch 维度 [1, C, H, W]

# 获取原始图像的分辨率
_, _, original_height, original_width = image_tensor.shape

# 四倍下采样（缩小）
downsampled_height = original_height // 4
downsampled_width = original_width // 4
downsampled_image = F.interpolate(image_tensor, size=(downsampled_height, downsampled_width), mode='bicubic', align_corners=False)

# 上采样回原始分辨率（放大）
upsampled_image = F.interpolate(downsampled_image, size=(original_height, original_width), mode='bicubic', align_corners=False)

# 将张量转换回图像
upsampled_image = upsampled_image.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()  # 转换为 [H, W, C] 并限制到 [0, 1]
upsampled_image = (upsampled_image * 255).astype('uint8')  # 转换为 8 位图像
upsampled_image = Image.fromarray(upsampled_image)  # 转换为 PIL 图像

# 保存上采样后的图像
output_path = '0059769_upsampled.png'
upsampled_image.save(output_path)

print(f"处理后的图像已保存到: {output_path}")