import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.models.autoencoders import AutoencoderKL
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
#3.93 GB
transformer = SD3Transformer2DModel.from_pretrained("/media/ssd8T/wyw/Pretrained/stable-diffusion-3-medium-diffusers", subfolder = 'transformer', torch_dtype=torch.float16)
transformer = transformer.to('cuda')

#0.24GB
text_encoder = CLIPTextModelWithProjection.from_pretrained("/media/ssd8T/wyw/Pretrained/stable-diffusion-3-medium-diffusers", subfolder = 'text_encoder', torch_dtype=torch.float16)
text_encoder = text_encoder.to('cuda')

#1.30GB
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained("/media/ssd8T/wyw/Pretrained/stable-diffusion-3-medium-diffusers", subfolder = 'text_encoder_2', torch_dtype=torch.float16)
text_encoder_2 = text_encoder_2.to('cuda')

#10.746 GB
text_encoder_3 = T5EncoderModel.from_pretrained("/media/ssd8T/wyw/Pretrained/stable-diffusion-3-medium-diffusers", subfolder = 'text_encoder_3', torch_dtype=torch.float16)
text_encoder_3 = text_encoder_3.to('cpu')

# 0.16GB
vae = AutoencoderKL.from_pretrained("/media/ssd8T/wyw/Pretrained/stable-diffusion-3-medium-diffusers", subfolder = 'vae', torch_dtype=torch.float16)
vae = vae.to('cuda')

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("/media/ssd8T/wyw/Pretrained/stable-diffusion-3-medium-diffusers", subfolder = 'scheduler', torch_dtype=torch.float16)

tokenizer = CLIPTokenizer.from_pretrained("/media/ssd8T/wyw/Pretrained/stable-diffusion-3-medium-diffusers", subfolder = 'tokenizer', torch_dtype=torch.float16)

tokenizer_2 = CLIPTokenizer.from_pretrained("/media/ssd8T/wyw/Pretrained/stable-diffusion-3-medium-diffusers", subfolder = 'tokenizer_2', torch_dtype=torch.float16)

tokenizer_3 = T5TokenizerFast.from_pretrained("/media/ssd8T/wyw/Pretrained/stable-diffusion-3-medium-diffusers", subfolder = 'tokenizer_3', torch_dtype=torch.float16)

pipe = StableDiffusion3Pipeline(transformer, scheduler, vae, text_encoder, tokenizer, text_encoder_2, tokenizer_2, text_encoder_3, tokenizer_3)
# pipe = pipe.to("cuda")
print(torch.cuda.memory_allocated('cuda')/1024**3)
image = pipe(
    "A cat holding a sign that says hello world",
    negative_prompt="",
    num_inference_steps=28,
    guidance_scale=7.0,
).images[0]

output_path = "generated_image.png" 
image.save(output_path)
