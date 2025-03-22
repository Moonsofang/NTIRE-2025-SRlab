import torch
from safetensors import safe_open

import torch

# 加载 PyTorch 模型权重
weights = torch.load('/media/ssd8T/wyw/Pretrained/pisa-sr/pisa_sr.pkl')
# print(len(weights['state_dict_unet'].keys()))
# print(weights['unet_lora_encoder_modules_pix'])
# print(weights['unet_lora_decoder_modules_pix'])
# print(weights['unet_lora_decoder_modules_pix'])

with open('weights_names.txt', 'w') as f:
    for key in weights['unet_lora_others_modules_pix']:
        f.write(key + '\n')

print("权重名称已保存到 'weights_names.txt' 文件中。")

# 加载 .safetensors 文件
# model_path_1 = "/media/ssd8T/wyw/Checkpoints/SeeSR/sam/checkpoint-20000/unet/diffusion_pytorch_model.safetensors"
# model_path_2 = "/media/ssd8T/wyw/Checkpoints/SeeSR/sam/checkpoint-10000/unet/diffusion_pytorch_model.safetensors"
# output_txt_path = "key_value_shapes.txt"
# model_path_1 = '/media/ssd8T/wyw/Checkpoints/SeeSR/ram_local/checkpoint-1/unet/diffusion_pytorch_model.safetensors'
# with safe_open(model_path_1, framework="pt", device="cpu") as f1:
#     with open(output_txt_path, "w") as txt_file:
#         # 遍历所有 key
#         for key in f1.keys():
#             tensor1 = f1.get_tensor(key)
#             shape = tensor1.shape
#             txt_file.write(f"{key}: {shape}\n")
                    
# 打开 .safetensors 文件
# with safe_open(model_path_1, framework="pt", device="cpu") as f1:
#     with safe_open(model_path_2, framework="pt", device="cpu") as f2:
#         with open(output_txt_path, "w") as txt_file:
#             # 遍历所有 key
#             for key in f1.keys():
#                 tensor1 = f1.get_tensor(key)
#                 tensor2 = f2.get_tensor(key)
#                 if not torch.equal(tensor1,tensor2):
#                     shape = tensor1.shape
#                     # 将 key 和 shape 写入 txt 文件
#                     txt_file.write(f"{key}: {shape}\n")

# print(f"Key and value shapes have been saved to {output_txt_path}")