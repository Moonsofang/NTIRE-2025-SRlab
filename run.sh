# for wild dataset
python ./test_seesr_sam.py \
--pretrained_model_path ./preset/models/stable-diffusion-2-base \
--prompt '' \
--seesr_model_path ./preset/models/checkpoint-90000 \
--ram_ft_path ./preset/models/DAPE.pth \
--image_path ./preset/datasets/test_datasets/wild \
--output_dir your_output_dir_path/wild \
--start_point noise \
--num_inference_steps 50 \
--guidance_scale 8.5 \
--added_prompt "clean, high-resolution, 8k, ultra-detailed, ultra-realistic" \
--upscale 1 \
--process_size 512

# for synthetic dataset
python ./test_seesr_sam.py \
--pretrained_model_path ./preset/models/stable-diffusion-2-base \
--prompt '' \
--seesr_model_path ./preset/models/checkpoint-90000 \
--ram_ft_path ./preset/models/DAPE.pth \
--image_path ./preset/datasets/test_datasets/synthetic \
--output_dir your_output_dir_path/synthetic \
--start_point noise \
--num_inference_steps 50 \
--guidance_scale 0.9 \
--upscale 4 \
--process_size 512
