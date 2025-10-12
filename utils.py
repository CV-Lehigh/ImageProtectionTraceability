import os
import ast
import json
import random
from tqdm import tqdm
import google.generativeai as genai
import cv2
from PIL import Image
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from super_image import EdsrModel, ImageLoader

# API_KEY = "YOUR_API_KEY"   
# genai.configure(api_key=API_KEY)
device = "cuda:3"

class Transformations:
    def __init__(self, image_dir) -> None:
        self.image_path = image_dir
        if any(os.path.isdir(os.path.join(image_dir, f)) for f in os.listdir(image_dir)):
            self.image_files = []
            for subfolder in sorted(os.listdir(image_dir)):
                subfolder_path = os.path.join(image_dir, subfolder)
                if os.path.isdir(subfolder_path):
                    self.image_files.extend([os.path.join(subfolder, f) for f in sorted(os.listdir(subfolder_path))
                                    if f.lower().endswith(('.jpg', '.png'))])
        else:
            self.image_files = [f for f in sorted(os.listdir(image_dir)) 
                        if f.lower().endswith(('.jpg', '.png'))]
            
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True, safety_checker=None
        )
        self.pipe = self.pipe.to(device)
        self.gemini_model = genai.GenerativeModel("gemini-2.5-pro")
        self.model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module
        self.model = self.model.to(device)

        with open("cleaned_object_list.json", "r", encoding="utf-8") as f:
            self.object_list = json.load(f)

    def apply(self, method_name, save_path):
        if not hasattr(self, method_name):
            raise ValueError(f"Transformation '{method_name}' not found")
        
        method = getattr(self, method_name)  
        method(save_path)                     

    def crop_and_resize(self, save_path):
        for image in tqdm(self.image_files, desc="Cropping and resizing"):
            if os.path.exists(os.path.join(save_path, image)):
                continue
            img = cv2.imread(os.path.join(self.image_path, image))
            h, w = img.shape[:2]
            cropped_img = img[64:h-64, 64:w-64]
            resized_img = cv2.resize(cropped_img, (512, 512), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(save_path, image), resized_img)
    
    def gaus_blur(self, save_path):
        for image in tqdm(self.image_files, desc="Gaussian blur"):
            if os.path.exists(os.path.join(save_path, image)):
                continue
            img = cv2.imread(os.path.join(self.image_path, image))
            blurred_img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.05)
            cv2.imwrite(os.path.join(save_path, image), blurred_img)
        
    def jpeg_compress(self, save_path):
        for image in tqdm(self.image_files, desc="JPEG compress"):
            if os.path.exists(os.path.join(save_path, image)):
                continue
            img = cv2.imread(os.path.join(self.image_path, image))
            cv2.imwrite(os.path.join(save_path, image), img, [cv2.IMWRITE_JPEG_QUALITY, 75])

    def horizontal_flip(self, save_path):
        for image in tqdm(self.image_files, desc="Horizontal flip"):
            if os.path.exists(os.path.join(save_path, image)):
                continue
            img = cv2.imread(os.path.join(self.image_path, image))
            flipped_img = cv2.flip(img, 1)
            cv2.imwrite(os.path.join(save_path, image), flipped_img)

    def brightness_contrast(self, save_path):
        for image in tqdm(self.image_files, desc="Brightness contrast"):
            if os.path.exists(os.path.join(save_path, image)):
                continue
            img = cv2.imread(os.path.join(self.image_path, image))
            brightness = 10.0
            contrast = 2.0
            adjusted_img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
            cv2.imwrite(os.path.join(save_path, image), adjusted_img)
    
    def super_resolution(self, save_path):
        
        for image in tqdm(self.image_files, desc="Super resolution"):
            if os.path.exists(os.path.join(save_path, image)):
                continue
            img = Image.open(os.path.join(self.image_path, image))
            inputs = ImageLoader.load_image(img)
            inputs = inputs.to(device)
            self.model.to(device)
            outputs = self.model(inputs)
            ImageLoader.save_image(outputs, os.path.join(save_path, image))

    def distortion(self, save_path):
            
        def barrel_distortion(img, k=-0.3):
            import numpy as np
            h, w = img.shape[:2]
     
            map_x = np.zeros((h, w), np.float32)
            map_y = np.zeros((h, w), np.float32)
      
            cx, cy = w / 2, h / 2
            max_radius = np.sqrt(cx**2 + cy**2)
            
            for y in range(h):
                for x in range(w):
                    dx = (x - cx) / max_radius
                    dy = (y - cy) / max_radius
                    r = np.sqrt(dx*dx + dy*dy)
      
                    factor = 1 + k * (r**2)
                    
                    new_x = cx + dx * factor * max_radius
                    new_y = cy + dy * factor * max_radius
                    
                    map_x[y, x] = new_x
                    map_y[y, x] = new_y
            
      
            distorted = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            return distorted

        for image in tqdm(self.image_files, desc="Distortion"):
            if os.path.exists(os.path.join(save_path, image)):
                continue
            img = cv2.imread(os.path.join(self.image_path, image))
            distorted_img = barrel_distortion(img, k=-0.5)
            cv2.imwrite(os.path.join(save_path, image), distorted_img)

    def realistic(self, save_path):
        torch.manual_seed(9222)
        self.pipe.enable_model_cpu_offload()
        # self.pipe.enable_xformers_memory_efficient_attention()
        for image in tqdm(self.image_files, desc="Realistic editing"):
            if os.path.exists(os.path.join(save_path, image)):
                continue
            img = Image.open(os.path.join(self.image_path, image))
            prompt = "Transform the image into a realistic photograph."     #"Make the image more realistic."
            edited_img = self.pipe(prompt, image=img).images[0]
            edited_img.save(os.path.join(save_path, image))
    
    def style(self, save_path):
        torch.manual_seed(9222)
        self.pipe.enable_model_cpu_offload()
        # self.pipe.enable_xformers_memory_efficient_attention()
        for image in tqdm(self.image_files, desc="Style transfer"):
            if os.path.exists(os.path.join(save_path, image)):
                continue
            img = Image.open(os.path.join(self.image_path, image))
            prompt = "Change the style to Ghibli style."
            edited_img = self.pipe(prompt, image=img).images[0]
            edited_img.save(os.path.join(save_path, image))

    def semantic(self, save_path):
        torch.manual_seed(9222)
        self.pipe.enable_model_cpu_offload()
        for image in tqdm(self.image_files, desc="Semantic transformation"):
            if os.path.exists(os.path.join(save_path, image)):
                continue
            img = Image.open(os.path.join(self.image_path, image))
            object = random.choice(self.object_list[image])
            prompt = f"Remove {object} from the image"
            edited_img = self.pipe(prompt, image=img).images[0]
            edited_img.save(os.path.join(save_path, f"{object}_{image}"))
        # prompt = "What objects are in the image? Return a concise list, for example: ['woman', 'earring']."
        # torch.manual_seed(9222)
        # self.pipe.enable_model_cpu_offload()
        # # self.pipe.enable_xformers_memory_efficient_attention()
        # for image in tqdm(self.image_files, desc="Semantic transformation"):
        #     if os.path.exists(os.path.join(save_path, image)):
        #         continue
        #     img = Image.open(os.path.join(self.image_path, image))
        #     try:
        #         response = self.gemini_model.generate_content([prompt, img])
        #         response_string = response.text
        #         if response_string.strip().startswith("```json"):
        #             response_string = response_string.strip()[7:].strip()
        #         if response_string.strip().endswith("```"):
        #             response_string = response_string.strip()[:-3].strip()
        #         object_list = ast.literal_eval(response_string)
        #         editing_prompt = f"Remove {random.choice(object_list)} from the image"
        #         print(f"Editing prompt: {editing_prompt}; Object list: {object_list}")
        #         edited_img = self.pipe(editing_prompt, image=img).images[0]
        #         edited_img.save(os.path.join(save_path, image))
        #         print(f"Saved {image}")
        #         print("--------------------------------")
        #     except Exception as e:
        #         print(f"Error: {e}")
        #         print(f"Skipping {image}")
        #         print("--------------------------------")









