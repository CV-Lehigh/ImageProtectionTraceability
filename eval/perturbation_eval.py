import os
import json
from tqdm import tqdm
from PIL import Image

import torch
from diffusers import StableDiffusionImg2ImgPipeline

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Stable Diffusion generation.')
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--model_id_or_path', type=str, default="runwayml/stable-diffusion-v1-5", help='model_id_or_path')
    parser.add_argument('--STRENGTH', type=float, default=0.5, help='STRENGTH')
    parser.add_argument('--GUIDANCE', type=float, default=7.5, help='GUIDANCE')
    parser.add_argument('--NUM_STEPS', type=int, default=50, help='NUM_STEPS')
    parser.add_argument('--ori_dataset_path', type=str, default="/data/qit220/Dataset/transform_watermark_perturbtion/mist", help='ori_dataset_path')
    parser.add_argument('--ori_save_path', type=str, default="/data/qit220/Dataset/transform_watermark_perturbtion/ita_mist", help='ori_save_path')
    return parser.parse_args()

def main(args):
    
    prompt = "change to style to Picasso"

    device = args.device

    model_id_or_path = args.model_id_or_path
    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id_or_path,
        safety_checker=None,
        torch_dtype=torch.float16,
    )
    pipe_img2img = pipe_img2img.to(device)

    SEED = 9222
    STRENGTH = args.STRENGTH
    GUIDANCE = args.GUIDANCE
    NUM_STEPS = args.NUM_STEPS
    ori_dataset_path = args.ori_dataset_path   #os.listdir(args.ori_dataset_path)

    if any(os.path.isdir(os.path.join(ori_dataset_path, f)) for f in os.listdir(ori_dataset_path)):
        image_files = []
        for subfolder in sorted(os.listdir(ori_dataset_path)):
            subfolder_path = os.path.join(ori_dataset_path, subfolder)
            if os.path.isdir(subfolder_path):
                image_files.extend([os.path.join(subfolder, f) for f in sorted(os.listdir(subfolder_path))
                                if f.lower().endswith(('.jpg', '.png'))])
    else:
        image_files = [f for f in sorted(os.listdir(ori_dataset_path)) 
                    if f.lower().endswith(('.jpg', '.png'))]

    ori_save_path = args.ori_save_path
    os.makedirs(ori_save_path, exist_ok=True)
    for image_file in image_files:
        os.makedirs(os.path.join(ori_save_path, image_file.rsplit("/", 1)[0]), exist_ok=True)

    for img_name in tqdm(image_files, colour="green", desc="images", leave=True):
        ori_image = Image.open(os.path.join(args.ori_dataset_path, img_name)).resize((512, 512), resample=Image.BICUBIC)
        with torch.autocast('cuda'):
            # if not os.path.exists(os.path.join(ori_save_path, img_name.rsplit("/", 1)[0])):
            #     os.makedirs(os.path.join(ori_save_path, img_name.rsplit("/", 1)[0]), exist_ok=True)
            torch.manual_seed(SEED)
            image_nat = pipe_img2img(prompt=prompt, image=ori_image, strength=STRENGTH, guidance_scale=GUIDANCE, num_inference_steps=NUM_STEPS).images[0]
            image_nat.save(os.path.join(ori_save_path, img_name))

if __name__ == '__main__':
    args = get_args()
    main(args)