## ⚙️ Dependencies and Installation
```
## git clone this repository
git clone https://github.com/Moonsofang/NTIRE-2025-SRlab
cd SRlab

# create an environment with python >= 3.8
conda create -n srlab python=3.8
conda activate srlab
pip install -r requirements.txt

# or you can directly install the environment by following instruct
conda env create -f srlab.yml
conda activate srlab
```

## 🚀 Quick Inference
#### Step 1: Download the pretrained models
- Download the pretrained SD-2-base models from [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-base)
- Download the checkpoint, sam2.1_hiera_tiny, ram_swin_large and DAPE models from [GoogleDrive](https://drive.google.com/drive/folders/1Ce0D8R99t-fDQfACLc8SGvf3gzdMnTwT?usp=sharing).

You can put the models into `preset/models`.

#### Step 2: Prepare testing data
You can put the testing images in the `preset/datasets/test_datasets`.

#### Step 3: Running testing command
```
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
```
More details are [here](asserts/hyp.md)

## 🌈 Train 

Will release soon.

## ❤️ Acknowledgments
This project is based on [diffusers](https://github.com/huggingface/diffusers) and [SeeSR](https://github.com/cswry/SeeSR). Some codes are brought from [PASD](https://github.com/yangxy/PASD), [RAM](https://github.com/xinyu1205/recognize-anything) and [SAM2](https://github.com/facebookresearch/sam2)). Thanks for their awesome works. We also pay tribute to the pioneering work of [StableSR](https://github.com/IceClear/StableSR).

## 📧 Contact
If you have any questions, please feel free to contact: `ly5825761@gmail.com`

## 🎫 License
This project and related weights are released under the [Apache 2.0 license](LICENSE).


<details>
<summary>statistics</summary>

![visitors](https://visitor-badge.laobi.icu/badge?page_id=cswry/SeeSR)

</details>
