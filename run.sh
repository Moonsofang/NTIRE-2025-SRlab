## seesr_sam test
CUDA_VISIBLE_DEVICES='4' python test_seesr_sam.py \
--pretrained_model_path /media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base \
--prompt '' \
--seesr_model_path /media/ssd8T/wyw/Checkpoints/SeeSR/sam/checkpoint-50000 \
--ram_ft_path /media/ssd8T/ly/SeeSR/preset/models/DAPE.pth \
--image_path /media/ssd8T/ly/SeeSR/preset/datasets/test_datasets/synthetic \
--output_dir /media/ssd8T/wyw/Data/NTIRE2025/SeeSR/sam_50000 \
--start_point lr \
--num_inference_steps 50 \
--guidance_scale 4.5 \
--upscale 4 \
--process_size 512

CUDA_VISIBLE_DEVICES='6' python test_seesr_sam.py \
--pretrained_model_path /media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base \
--prompt '' \
--seesr_model_path /media/ssd8T/wyw/Checkpoints/SeeSR/sam/checkpoint-60000 \
--ram_ft_path /media/ssd8T/ly/SeeSR/preset/models/DAPE.pth \
--image_path /media/ssd8T/ly/SeeSR/preset/datasets/test_datasets/wild \
--output_dir /media/ssd8T/wyw/Data/NTIRE2025/SeeSR/sam_60000/wild \
--start_point lr \
--num_inference_steps 50 \
--guidance_scale 5.5 \
--upscale 1 \
--process_size 512

## seesr_sam train
CUDA_VISIBLE_DEVICES="4" accelerate launch --main_process_port 29501 train_seesr_sam.py \
--pretrained_model_name_or_path="/media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base" \
--controlnet_model_name_or_path="/media/ssd8T/ly/SeeSR/preset/models/seesr" \
--unet_model_name_or_path="/media/ssd8T/ly/SeeSR/preset/models/seesr" \
--output_dir '/media/ssd8T/wyw/Checkpoints/seesr' \
--root_folders '/media/ssd8T/ly/SeeSR/preset/datasets/train_datasets/training_for_seesr_1' \
--ram_ft_path '/media/ssd8T/ly/SeeSR/preset/models/DAPE.pth' \
--enable_xformers_memory_efficient_attention \
--mixed_precision="fp16" \
--resolution=512 \
--use_8bit_adam \
--learning_rate=5e-5 \
--train_batch_size=2 \
--gradient_accumulation_steps=2 \
--null_text_ratio=0.5 \
--dataloader_num_workers=0 \
--checkpointing_steps=1000

#### Train seesr+sam
CUDA_VISIBLE_DEVICES="4,5,6," accelerate launch --main_process_port 29502 train_seesr_sam.py \
--pretrained_model_name_or_path="/media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base" \
--controlnet_model_name_or_path="/media/ssd8T/ly/SeeSR/preset/models/seesr" \
--unet_model_name_or_path="/media/ssd8T/ly/SeeSR/preset/models/seesr" \
--output_dir '/media/ssd8T/wyw/Checkpoints/SeeSR/sam' \
--root_folders '/media/ssd8T/wyw/Data/NTIRE2025/LSDIR,/media/ssd8T/wyw/Data/NTIRE2025/NTIRE_syn,/media/ssd8T/wyw/Data/NTIRE2025/NTIRE_syn_deg' \
--ram_ft_path '/media/ssd8T/ly/SeeSR/preset/models/DAPE.pth' \
--enable_xformers_memory_efficient_attention \
--mixed_precision="fp16" \
--resolution=512 \
--use_8bit_adam \
--learning_rate=5e-5 \
--train_batch_size=1 \
--gradient_accumulation_steps=2 \
--null_text_ratio=0.5 \
--dataloader_num_workers=4 \
--checkpointing_steps=10000 \
--trainable_modules image_attentions

### Train seesr
CUDA_VISIBLE_DEVICES="4,5,6" accelerate launch --main_process_port 29501 train_seesr.py \
--pretrained_model_name_or_path="/media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base" \
--controlnet_model_name_or_path="/media/ssd8T/ly/SeeSR/preset/models/seesr" \
--unet_model_name_or_path="/media/ssd8T/ly/SeeSR/preset/models/seesr" \
--output_dir '/media/ssd8T/wyw/Checkpoints/SeeSR/ram' \
--root_folders '/media/ssd8T/wyw/Data/NTIRE2025/LSDIR,/media/ssd8T/wyw/Data/NTIRE2025/NTIRE_syn,/media/ssd8T/wyw/Data/NTIRE2025/NTIRE_syn_deg' \
--ram_ft_path '/media/ssd8T/ly/SeeSR/preset/models/DAPE.pth' \
--enable_xformers_memory_efficient_attention \
--mixed_precision="fp16" \
--resolution=512 \
--use_8bit_adam \
--learning_rate=2e-5 \
--train_batch_size=2 \
--gradient_accumulation_steps=2 \
--null_text_ratio=0.5 \
--dataloader_num_workers=0 \
--checkpointing_steps=10000 \
--trainable_modules image_attentions

CUDA_VISIBLE_DEVICES='4' python test_seesr.py \
--pretrained_model_path /media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base \
--prompt '' \
--seesr_model_path /media/ssd8T/wyw/Checkpoints/SeeSR/ram/checkpoint-80000 \
--ram_ft_path /media/ssd8T/ly/SeeSR/preset/models/DAPE.pth \
--image_path /media/ssd8T/ly/SeeSR/preset/datasets/test_datasets/wild \
--output_dir /media/ssd8T/wyw/Data/NTIRE2025/SeeSR/ram-80000 \
--start_point lr \
--num_inference_steps 50 \
--guidance_scale 5.5 \
--upscale 1 \
--process_size 512

CUDA_VISIBLE_DEVICES='7' python test_seesr.py \
--pretrained_model_path /media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base \
--prompt '' \
--seesr_model_path /media/ssd8T/wyw/Checkpoints/SeeSR/ram/checkpoint-80000 \
--ram_ft_path /media/ssd8T/ly/SeeSR/preset/models/DAPE.pth \
--image_path /media/ssd8T/ly/SeeSR/preset/datasets/test_datasets/synthetic \
--output_dir /media/ssd8T/wyw/Data/NTIRE2025/SeeSR/ram-80000 \
--start_point lr \
--num_inference_steps 50 \
--guidance_scale 5.5 \
--upscale 4 \
--process_size 512


### local
CUDA_VISIBLE_DEVICES="4,5,6" accelerate launch --main_process_port 29501 train_seesr_local.py \
--pretrained_model_name_or_path="/media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base" \
--controlnet_model_name_or_path="/media/ssd8T/ly/SeeSR/preset/models/seesr" \
--unet_model_name_or_path="/media/ssd8T/ly/SeeSR/preset/models/seesr" \
--output_dir '/media/ssd8T/wyw/Checkpoints/SeeSR/ram_local' \
--root_folders '/media/ssd8T/wyw/Data/NTIRE2025/LSDIR,/media/ssd8T/wyw/Data/NTIRE2025/NTIRE_syn,/media/ssd8T/wyw/Data/NTIRE2025/NTIRE_syn_deg' \
--ram_ft_path '/media/ssd8T/ly/SeeSR/preset/models/DAPE.pth' \
--enable_xformers_memory_efficient_attention \
--mixed_precision="fp16" \
--resolution=512 \
--use_8bit_adam \
--learning_rate=5e-5 \
--train_batch_size=1 \
--gradient_accumulation_steps=2 \
--null_text_ratio=0.5 \
--dataloader_num_workers=0 \
--checkpointing_steps=10000 \
--trainable_modules image_attentions local


CUDA_VISIBLE_DEVICES='4' python test_seesr_local.py \
--pretrained_model_path /media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base \
--prompt '' \
--seesr_model_path /media/ssd8T/wyw/Checkpoints/SeeSR/ram_local/checkpoint-110000 \
--ram_ft_path /media/ssd8T/ly/SeeSR/preset/models/DAPE.pth \
--image_path /media/ssd8T/ly/SeeSR/preset/datasets/test_datasets/synthetic \
--output_dir /media/ssd8T/wyw/Data/NTIRE2025/SeeSR/ram_local-110000/synthetic \
--start_point lr \
--num_inference_steps 50 \
--guidance_scale 5.5 \
--upscale 4 \
--process_size 512

CUDA_VISIBLE_DEVICES='7' python test_seesr_local.py \
--pretrained_model_path /media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base \
--prompt '' \
--seesr_model_path /media/ssd8T/wyw/Checkpoints/SeeSR/ram_local/checkpoint-50000 \
--ram_ft_path /media/ssd8T/ly/SeeSR/preset/models/DAPE.pth \
--image_path /media/ssd8T/ly/SeeSR/preset/datasets/test_datasets/wild \
--output_dir /media/ssd8T/wyw/Data/NTIRE2025/SeeSR/ram_local-10000 \
--start_point lr \
--num_inference_steps 50 \
--guidance_scale 5.5 \
--upscale 1 \
--process_size 512

### sam_local
CUDA_VISIBLE_DEVICES="" accelerate launch train_seesr_local.py \
--pretrained_model_name_or_path="/media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base" \
--controlnet_model_name_or_path="/media/ssd8T/wyw/Checkpoints/SeeSR/sam/checkpoint-10000" \
--unet_model_name_or_path="/media/ssd8T/wyw/Checkpoints/SeeSR/sam/checkpoint-10000" \
--output_dir '/media/ssd8T/wyw/Checkpoints/SeeSR/sam_local' \
--root_folders '/media/ssd8T/wyw/Data/NTIRE2025/LSDIR,/media/ssd8T/wyw/Data/NTIRE2025/NTIRE_syn,/media/ssd8T/wyw/Data/NTIRE2025/NTIRE_syn_deg' \
--ram_ft_path '/media/ssd8T/ly/SeeSR/preset/models/DAPE.pth' \
--enable_xformers_memory_efficient_attention \
--mixed_precision="fp16" \
--resolution=512 \
--use_8bit_adam \
--learning_rate=5e-5 \
--train_batch_size=1 \
--gradient_accumulation_steps=2 \
--null_text_ratio=0.5 \
--dataloader_num_workers=0 \
--checkpointing_steps=10000 \
--trainable_modules image_attentions local

CUDA_VISIBLE_DEVICES='6' python test_seesr_sam_local.py \
--pretrained_model_path /media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base \
--prompt '' \
--seesr_model_path /media/ssd8T/wyw/Checkpoints/SeeSR/sam/checkpoint-10000 \
--ram_ft_path /media/ssd8T/ly/SeeSR/preset/models/DAPE.pth \
--image_path /media/ssd8T/NTIRE/data/wild_val \
--output_dir /media/ssd8T/wyw/Data/NTIRE2025/SeeSR/sam_local \
--start_point lr \
--num_inference_steps 50 \
--guidance_scale 5.5 \
--upscale 1 \
--process_size 512

## data
CUDA_VISIBLE_DEVICES="5" python /media/ssd8T/wyw/code/SeeSR/utils_data/make_paired_data.py \
--gt_path /media/ssd8T/DATA/LSDIR/0001000 /media/ssd8T/DATA/LSDIR/0002000 /media/ssd8T/DATA/LSDIR/0003000 /media/ssd8T/DATA/LSDIR/0004000 /media/ssd8T/DATA/LSDIR/0005000 /media/ssd8T/DATA/LSDIR/0006000 /media/ssd8T/DATA/LSDIR/0007000 /media/ssd8T/DATA/LSDIR/0008000 /media/ssd8T/DATA/LSDIR/0009000 /media/ssd8T/DATA/LSDIR/0010000 /media/ssd8T/DATA/LSDIR/0011000 /media/ssd8T/DATA/LSDIR/0012000 /media/ssd8T/DATA/LSDIR/0013000 /media/ssd8T/DATA/LSDIR/0014000 /media/ssd8T/DATA/LSDIR/0015000 /media/ssd8T/DATA/LSDIR/0016000 /media/ssd8T/DATA/LSDIR/0017000 /media/ssd8T/DATA/LSDIR/0018000 /media/ssd8T/DATA/LSDIR/0019000 /media/ssd8T/DATA/LSDIR/0020000 /media/ssd8T/DATA/LSDIR/0021000 /media/ssd8T/DATA/LSDIR/0022000 /media/ssd8T/DATA/LSDIR/0023000 /media/ssd8T/DATA/LSDIR/0024000 /media/ssd8T/DATA/LSDIR/0025000 /media/ssd8T/DATA/LSDIR/0026000 /media/ssd8T/DATA/LSDIR/0027000 /media/ssd8T/DATA/LSDIR/0028000 /media/ssd8T/DATA/LSDIR/0029000 /media/ssd8T/DATA/LSDIR/0030000 /media/ssd8T/DATA/LSDIR/0031000 /media/ssd8T/DATA/LSDIR/0032000 /media/ssd8T/DATA/LSDIR/0033000 /media/ssd8T/DATA/LSDIR/0034000 /media/ssd8T/DATA/LSDIR/0035000 /media/ssd8T/DATA/LSDIR/0036000 /media/ssd8T/DATA/LSDIR/0037000 /media/ssd8T/DATA/LSDIR/0038000 /media/ssd8T/DATA/LSDIR/0039000 /media/ssd8T/DATA/LSDIR/0040000 /media/ssd8T/DATA/LSDIR/0041000 /media/ssd8T/DATA/LSDIR/0042000 /media/ssd8T/DATA/LSDIR/0043000 /media/ssd8T/DATA/LSDIR/0044000 /media/ssd8T/DATA/LSDIR/0045000 /media/ssd8T/DATA/LSDIR/0046000 /media/ssd8T/DATA/LSDIR/0047000 /media/ssd8T/DATA/LSDIR/0048000 /media/ssd8T/DATA/LSDIR/0049000 /media/ssd8T/DATA/LSDIR/0050000 /media/ssd8T/DATA/LSDIR/0051000 /media/ssd8T/DATA/LSDIR/0052000 /media/ssd8T/DATA/LSDIR/0053000 /media/ssd8T/DATA/LSDIR/0054000 /media/ssd8T/DATA/LSDIR/0055000 /media/ssd8T/DATA/LSDIR/0056000 /media/ssd8T/DATA/LSDIR/0057000 /media/ssd8T/DATA/LSDIR/0058000 /media/ssd8T/DATA/LSDIR/0059000 /media/ssd8T/DATA/LSDIR/0060000 /media/ssd8T/DATA/LSDIR/0061000 /media/ssd8T/DATA/LSDIR/0062000 /media/ssd8T/DATA/LSDIR/0063000 /media/ssd8T/DATA/LSDIR/0064000 /media/ssd8T/DATA/LSDIR/0065000 /media/ssd8T/DATA/LSDIR/0066000 /media/ssd8T/DATA/LSDIR/0067000 /media/ssd8T/DATA/LSDIR/0068000 /media/ssd8T/DATA/LSDIR/0069000 /media/ssd8T/DATA/LSDIR/0070000 /media/ssd8T/DATA/LSDIR/0071000 /media/ssd8T/DATA/LSDIR/0072000 /media/ssd8T/DATA/LSDIR/0073000 /media/ssd8T/DATA/LSDIR/0074000 /media/ssd8T/DATA/LSDIR/0075000 /media/ssd8T/DATA/LSDIR/0076000 /media/ssd8T/DATA/LSDIR/0077000 /media/ssd8T/DATA/LSDIR/0078000 /media/ssd8T/DATA/LSDIR/0079000 /media/ssd8T/DATA/LSDIR/0080000 /media/ssd8T/DATA/LSDIR/0081000 /media/ssd8T/DATA/LSDIR/0082000 /media/ssd8T/DATA/LSDIR/0083000 /media/ssd8T/DATA/LSDIR/0084000 /media/ssd8T/DATA/LSDIR/0085000 \
--save_dir /media/ssd8T/wyw/Data/NTIRE2025/LSDIR \
--epoch 1


CUDA_VISIBLE_DEVICES="6" python /media/ssd8T/wyw/code/SeeSR/utils_data/make_paired_data.py \
--gt_path /media/ssd8T/ly/SeeSR/preset/datasets/train_datasets/train_data/synthetic_train/HR \
--save_dir /media/ssd8T/wyw/Data/NTIRE2025/NTIRE_syn_deg \
--epoch 1

CUDA_VISIBLE_DEVICES="7" python /media/ssd8T/wyw/code/SeeSR/utils_data/crop_paired_data.py \
--save_dir /media/ssd8T/wyw/Data/NTIRE2025/validation
CUDA_VISIBLE_DEVICES="7" python /media/ssd8T/wyw/code/SeeSR/utils_data/make_tags.py \
--root_path /media/ssd8T/wyw/Data/NTIRE2025/validation

CUDA_VISIBLE_DEVICES="5" python /media/ssd8T/wyw/code/SeeSR/utils_data/make_tags.py \
--root_path /media/ssd8T/wyw/Data/NTIRE2025/LSDIR

CUDA_VISIBLE_DEVICES="5" python /media/ssd8T/wyw/code/SeeSR/utils/metrics.py --image_dir /media/ssd8T/wyw/Data/NTIRE2025/SeeSR_test/sam_90000_sam/synthetic_gs1/sample00

/media/ssd8T/wyw/Data/NTIRE2025/SeeSR/ram_local-10000/sample00

/media/ssd8T/wyw/Data/NTIRE2025/SeeSR/ram_local-50000/wild
/media/ssd8T/wyw/Data/NTIRE2025/SeeSR/ram_local-50000/wild_noise/sample00
/media/ssd8T/wyw/Data/NTIRE2025/SeeSR/ram_local-60000/wild/sample00
/media/ssd8T/wyw/Data/NTIRE2025/SeeSR/ram_local-110000/wild

/media/ssd8T/wyw/Data/NTIRE2025/SeeSR/ram-10000/wild
/media/ssd8T/wyw/Data/NTIRE2025/SeeSR/ram-10000-llava/wild
/media/ssd8T/wyw/Data/NTIRE2025/SeeSR/ram-80000/wild

/media/ssd8T/wyw/Data/NTIRE2025/SeeSR/sam_60000/wild/sample00

/media/ssd8T/ly/SeeSR/preset/datasets/output_seesr/wild/sample00

## test ram new dataset
CUDA_VISIBLE_DEVICES='7' python test_seesr.py \
--pretrained_model_path /media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base \
--prompt '' \
--seesr_model_path /media/ssd8T/ly/SeeSR/preset/models/seesr \
--ram_ft_path /media/ssd8T/ly/SeeSR/preset/models/DAPE.pth \
--image_path /media/ssd8T/NTIRE/data/wild_test \
--output_dir /media/ssd8T/wyw/Data/NTIRE2025/SeeSR_test/ram/wild_noise \
--start_point noise \
--num_inference_steps 50 \
--guidance_scale 5.5 \
--upscale 1 \
--process_size 512
CUDA_VISIBLE_DEVICES='7' python test_seesr.py \
--pretrained_model_path /media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base \
--prompt '' \
--seesr_model_path /media/ssd8T/ly/SeeSR/preset/models/seesr \
--ram_ft_path /media/ssd8T/ly/SeeSR/preset/models/DAPE.pth \
--image_path /media/ssd8T/NTIRE/data/wild_test \
--output_dir /media/ssd8T/wyw/Data/NTIRE2025/SeeSR_test/ram/wild \
--start_point lr \
--num_inference_steps 50 \
--guidance_scale 5.5 \
--upscale 1 \
--process_size 512

CUDA_VISIBLE_DEVICES='5' python test_seesr.py \
--pretrained_model_path /media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base \
--prompt '' \
--seesr_model_path /media/ssd8T/wyw/Checkpoints/SeeSR/sam/checkpoint-90000 \
--ram_ft_path /media/ssd8T/ly/SeeSR/preset/models/DAPE.pth \
--image_path /media/ssd8T/NTIRE/data/wild_test \
--output_dir /media/ssd8T/wyw/Data/NTIRE2025/SeeSR_test/sam_90000/wild_noise \
--start_point noise \
--num_inference_steps 50 \
--guidance_scale 5.5 \
--upscale 1 \
--process_size 512

CUDA_VISIBLE_DEVICES='7' python test_seesr_sam.py \
--pretrained_model_path /media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base \
--prompt '' \
--seesr_model_path /media/ssd8T/wyw/Checkpoints/SeeSR/sam/checkpoint-10000 \
--ram_ft_path /media/ssd8T/ly/SeeSR/preset/models/DAPE.pth \
--image_path /media/ssd8T/NTIRE/data/wild_test \
--output_dir /media/ssd8T/wyw/Data/NTIRE2025/SeeSR_test/sam_10000_sam/wild_noise \
--start_point noise \
--num_inference_steps 50 \
--guidance_scale 5.5 \
--upscale 1 \
--process_size 512


CUDA_VISIBLE_DEVICES='5' python test_seesr.py \
--pretrained_model_path /media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base \
--prompt '' \
--seesr_model_path /media/ssd8T/ly/SeeSR/preset/models/seesr \
--ram_ft_path /media/ssd8T/ly/SeeSR/preset/models/DAPE.pth \
--image_path /media/ssd8T/NTIRE/data/synthetic_test \
--output_dir /media/ssd8T/wyw/Data/NTIRE2025/SeeSR_test/ram/synthetic_gs1 \
--start_point lr \
--num_inference_steps 50 \
--guidance_scale 1 \
--upscale 4 \
--process_size 512

#wyw2
CUDA_VISIBLE_DEVICES='7' python test_seesr.py \
--pretrained_model_path /media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base \
--prompt '' \
--seesr_model_path /media/ssd8T/wyw/Checkpoints/SeeSR/sam/checkpoint-10000 \
--ram_ft_path /media/ssd8T/ly/SeeSR/preset/models/DAPE.pth \
--image_path /media/ssd8T/NTIRE/data/synthetic_test \
--output_dir /media/ssd8T/wyw/Data/NTIRE2025/SeeSR_test/sam_10000/synthetic_gs1 \
--start_point lr \
--num_inference_steps 50 \
--guidance_scale 1 \
--upscale 4 \
--process_size 512

CUDA_VISIBLE_DEVICES='7' python test_seesr.py \
--pretrained_model_path /media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base \
--prompt '' \
--seesr_model_path /media/ssd8T/wyw/Checkpoints/SeeSR/sam/checkpoint-10000 \
--ram_ft_path /media/ssd8T/ly/SeeSR/preset/models/DAPE.pth \
--image_path /media/ssd8T/NTIRE/data/synthetic_test \
--output_dir /media/ssd8T/wyw/Data/NTIRE2025/SeeSR_test/sam_10000/synthetic_gs15 \
--start_point lr \
--num_inference_steps 50 \
--guidance_scale 1.5 \
--upscale 4 \
--process_size 512

CUDA_VISIBLE_DEVICES='7' python test_seesr.py \
--pretrained_model_path /media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base \
--prompt '' \
--seesr_model_path /media/ssd8T/wyw/Checkpoints/SeeSR/sam/checkpoint-10000 \
--ram_ft_path /media/ssd8T/ly/SeeSR/preset/models/DAPE.pth \
--image_path /media/ssd8T/NTIRE/data/synthetic_test \
--output_dir /media/ssd8T/wyw/Data/NTIRE2025/SeeSR_test/sam_10000/synthetic_gs25 \
--start_point lr \
--num_inference_steps 50 \
--guidance_scale 2.5 \
--upscale 4 \
--process_size 512

#wyw3
CUDA_VISIBLE_DEVICES='2' python test_seesr.py \
--pretrained_model_path /media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base \
--prompt '' \
--seesr_model_path /media/ssd8T/wyw/Checkpoints/SeeSR/sam/checkpoint-10000 \
--ram_ft_path /media/ssd8T/ly/SeeSR/preset/models/DAPE.pth \
--image_path /media/ssd8T/NTIRE/data/wild_test \
--output_dir /media/ssd8T/wyw/Data/NTIRE2025/SeeSR_test/sam_10000/wild_noise_gs65 \
--start_point noise \
--num_inference_steps 50 \
--guidance_scale 6.5 \
--upscale 1 \
--process_size 512

CUDA_VISIBLE_DEVICES='2' python test_seesr.py \
--pretrained_model_path /media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base \
--prompt '' \
--seesr_model_path /media/ssd8T/wyw/Checkpoints/SeeSR/sam/checkpoint-10000 \
--ram_ft_path /media/ssd8T/ly/SeeSR/preset/models/DAPE.pth \
--image_path /media/ssd8T/NTIRE/data/wild_test \
--output_dir /media/ssd8T/wyw/Data/NTIRE2025/SeeSR_test/sam_10000/wild_noise_gs75 \
--start_point noise \
--num_inference_steps 50 \
--guidance_scale 7.5 \
--upscale 1 \
--process_size 512

CUDA_VISIBLE_DEVICES='2' python test_seesr.py \
--pretrained_model_path /media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base \
--prompt '' \
--seesr_model_path /media/ssd8T/wyw/Checkpoints/SeeSR/sam/checkpoint-10000 \
--ram_ft_path /media/ssd8T/ly/SeeSR/preset/models/DAPE.pth \
--image_path /media/ssd8T/NTIRE/data/wild_test \
--output_dir /media/ssd8T/wyw/Data/NTIRE2025/SeeSR_test/sam_10000/wild_noise_gs85 \
--start_point noise \
--num_inference_steps 50 \
--guidance_scale 8.5 \
--upscale 1 \
--process_size 512

CUDA_VISIBLE_DEVICES='7' python test_seesr.py \
--pretrained_model_path /media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base \
--prompt '' \
--seesr_model_path /media/ssd8T/wyw/Checkpoints/SeeSR/sam/checkpoint-10000 \
--ram_ft_path /media/ssd8T/ly/SeeSR/preset/models/DAPE.pth \
--image_path /media/ssd8T/NTIRE/data/synthetic_test \
--output_dir /media/ssd8T/wyw/Data/NTIRE2025/SeeSR_test/sam_10000/synthetic_gs15 \
--start_point lr \
--num_inference_steps 50 \
--guidance_scale 1.5 \
--upscale 4 \
--process_size 512

CUDA_VISIBLE_DEVICES='6' python test_seesr.py \
--pretrained_model_path /media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base \
--prompt '' \
--seesr_model_path /media/ssd8T/wyw/Checkpoints/SeeSR/sam/checkpoint-10000 \
--ram_ft_path /media/ssd8T/ly/SeeSR/preset/models/DAPE.pth \
--image_path /media/ssd8T/NTIRE/data/wild_test \
--output_dir /media/ssd8T/wyw/Data/NTIRE2025/SeeSR_test/sam_10000/wild_noise_gs9 \
--start_point noise \
--num_inference_steps 50 \
--guidance_scale 9 \
--upscale 1 \
--process_size 512


CUDA_VISIBLE_DEVICES='6' python test_seesr_sam.py \
--pretrained_model_path /media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base \
--prompt 'ultura-detailed, ultra-realistic' \
--seesr_model_path /media/ssd8T/wyw/Checkpoints/SeeSR/sam/checkpoint-90000 \
--ram_ft_path /media/ssd8T/ly/SeeSR/preset/models/DAPE.pth \
--image_path /media/ssd8T/NTIRE/data/wild_test \
--output_dir /media/ssd8T/wyw/Data/NTIRE2025/SeeSR_test/sam_90000_sam/wild_noise_gs9_pp \
--start_point noise \
--num_inference_steps 50 \
--guidance_scale 9 \
--upscale 1 \
--process_size 512

CUDA_VISIBLE_DEVICES='6' python test_seesr_sam.py \
--pretrained_model_path /media/ssd8T/ly/SeeSR/preset/models/stable-diffusion-2-base \
--prompt '' \
--seesr_model_path /media/ssd8T/wyw/Checkpoints/SeeSR/sam/checkpoint-90000 \
--ram_ft_path /media/ssd8T/ly/SeeSR/preset/models/DAPE.pth \
--image_path /media/ssd8T/NTIRE/data/synthetic_test \
--output_dir /media/ssd8T/wyw/Data/NTIRE2025/SeeSR_test/sam_90000_sam/synthetic_gs05 \
--start_point lr \
--num_inference_steps 50 \
--guidance_scale 0.5 \
--upscale 4 \
--process_size 512
