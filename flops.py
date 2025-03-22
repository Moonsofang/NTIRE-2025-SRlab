# import torch
# from fvcore.nn import FlopCountAnalysis
# from models.unet_2d_condition import UNet2DConditionModel

# # 初始化 UNet（替换为你的模型）
# unet = UNet2DConditionModel.from_pretrained('/media/ssd8T/wyw/Checkpoints/SeeSR/sam/checkpoint-90000', subfolder="unet").cuda()
# unet.eval()  # 进入评估模式，避免 Dropout 等随机行为

# # 生成符合输入形状的张量
# latent_model_input = torch.randn(1, 4, 96, 96).cuda()
# t = torch.tensor([1.0]).cuda()  # timestep 需要是 Tensor
# prompt_embeds = torch.randn(1, 77, 1024).cuda()
# ram_encoder_hidden_states = torch.randn(1, 4096, 512).cuda()

# # 确保 `down_block_res_samples` 形状符合 `forward` 方法
# down_block_res_samples = (
#     torch.randn(1, 320, 96, 96).cuda(),
#     torch.randn(1, 320, 96, 96).cuda(),
#     torch.randn(1, 320, 96, 96).cuda(),
#     torch.randn(1, 320, 48, 48).cuda(),
#     torch.randn(1, 640, 48, 48).cuda(),
#     torch.randn(1, 640, 48, 48).cuda(),
#     torch.randn(1, 640, 24, 24).cuda(),
#     torch.randn(1, 1280, 24, 24).cuda(),
#     torch.randn(1, 1280, 24, 24).cuda(),
#     torch.randn(1, 1280, 12, 12).cuda(),
#     torch.randn(1, 1280, 12, 12).cuda(),
#     torch.randn(1, 1280, 12, 12).cuda()
# )
# mid_block_res_sample = torch.randn(1, 1280, 12, 12).cuda()

# # 封装 U-Net，确保 `down_block_res_samples` 作为 Tuple 传递
# class WrappedUNet(torch.nn.Module):
#     def __init__(self, unet, t, prompt_embeds, down_block_res_samples,
#                  mid_block_res_sample, ram_encoder_hidden_states):
#         super().__init__()
#         self.unet = unet
#         self.t = t
#         self.prompt_embeds = prompt_embeds
#         self.down_block_res_samples = down_block_res_samples
#         self.mid_block_res_sample = mid_block_res_sample
#         self.ram_encoder_hidden_states = ram_encoder_hidden_states

#     def forward(self, latent_model_input):
#         return self.unet(
#             sample=latent_model_input,
#             timestep=self.t,
#             encoder_hidden_states=self.prompt_embeds,
#             cross_attention_kwargs=None,
#             added_cond_kwargs=None,
#             down_block_additional_residuals=self.down_block_res_samples,  # 确保是 Tuple
#             mid_block_additional_residual=self.mid_block_res_sample,
#             encoder_attention_mask=None,
#             return_dict=False,
#             image_encoder_hidden_states=self.ram_encoder_hidden_states
#         )[0]  # 仅返回 FLOPs 计算需要的部分

# # 计算 FLOPs
# wrapped_unet = WrappedUNet(unet, t, prompt_embeds, down_block_res_samples, mid_block_res_sample, ram_encoder_hidden_states)
# flops = FlopCountAnalysis(wrapped_unet, latent_model_input)

# # 输出结果
# print(f"U-Net FLOPs: {flops.total() / 1e9:.2f} GFLOPs")


# import torch
# from fvcore.nn import FlopCountAnalysis
# from diffusers import AutoencoderKL

# # 1. 加载 VAE 模型
# vae = AutoencoderKL.from_pretrained('/media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base', subfolder="vae").eval()

# # 2. 创建一个 VAE 编码器的封装类
# class VAEEncoderWrapper(torch.nn.Module):
#     def __init__(self, vae):
#         super().__init__()
#         self.encoder = vae.encoder  # 只取 VAE 的 encoder
#         self.quant_conv = vae.quant_conv  # 量化层（Stable Diffusion 里的 VAE 有）

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.quant_conv(x)  # 确保跟原始 VAE 计算流程一致
#         return x

# # 3. 实例化封装的 VAE Encoder
# vae_encoder = VAEEncoderWrapper(vae).eval()

# # 4. 生成测试输入
# image = torch.randn(1, 3, 1920, 1080)  # Batch size = 1

# # 5. 计算 FLOPs
# flops = FlopCountAnalysis(vae_encoder, (image * 2 - 1,))
# print(f"VAE Encoder FLOPs: {flops.total() / 1e9:.2f} GFLOPs")


# import torch
# from fvcore.nn import FlopCountAnalysis
# from diffusers import AutoencoderKL

# # 1. 加载 VAE 模型
# vae = AutoencoderKL.from_pretrained('/media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base', subfolder="vae").eval()

# # 2. 封装 VAE 解码器
# class VAEDecoderWrapper(torch.nn.Module):
#     def __init__(self, vae):
#         super().__init__()
#         self.post_quant_conv = vae.post_quant_conv  # 量化层
#         self.decoder = vae.decoder  # 解码器

#     def forward(self, x):
#         x = self.post_quant_conv(x)  # 先过量化卷积
#         x = self.decoder(x)  # VAE 解码
#         return x

# # 3. 实例化封装的 VAE Decoder
# vae_decoder = VAEDecoderWrapper(vae).eval()

# # 4. 生成测试输入
# latents = torch.randn(1, 4, 135, 240)  # VAE latent space size

# # 5. 计算 FLOPs
# flops = FlopCountAnalysis(vae_decoder, (latents,))
# print(f"VAE Decoder FLOPs: {flops.total() / 1e9:.2f} GFLOPs")


import torch
from transformers import CLIPTextModel, CLIPTokenizer
from fvcore.nn import FlopCountAnalysis

# 1. 加载 CLIP 文本编码器
text_encoder = CLIPTextModel.from_pretrained('/media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base', subfolder="text_encoder").eval()

# 2. 加载 tokenizer 并创建输入
tokenizer = CLIPTokenizer.from_pretrained('/media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base', subfolder="tokenizer")
text = ["A photo of a cat", "A painting of a sunset"]  # 示例文本
text_input = tokenizer(text, padding="max_length", max_length=77, return_tensors="pt")

# 3. 获取 text_input_ids（输入 token IDs）
text_input_ids = text_input["input_ids"]

# 4. 计算 FLOPs
flops = FlopCountAnalysis(text_encoder, (text_input_ids,))
print(f"CLIP Text Encoder FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
