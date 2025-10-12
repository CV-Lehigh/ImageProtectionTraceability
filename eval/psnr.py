import numpy as np
import os
from PIL import Image
from tqdm import tqdm

def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))

if __name__ == "__main__":
    path = "/data/qit220/Dataset/transform_watermark_perturbtion/ori"
    path2 = "/data/qit220/Dataset/transform_watermark_perturbtion/transformation/5level/mist"
    values = []
    # for img_name in tqdm(os.listdir(path)):
    #     img1 = Image.open(os.path.join(path, img_name))
    #     img2 = Image.open(os.path.join(path2, img_name)).resize((512, 512))
    #     psnr = calculate_psnr(img1, img2)
    #     values.append(psnr)
    # print(f"PSNR: {np.mean(values)}, length: {len(values)}")

    for folder_name in tqdm(os.listdir(path2)):
        for img_name in os.listdir(os.path.join(path2, folder_name)):
            try:
                if "semantic" in folder_name:
                    img1 = Image.open(os.path.join(path, img_name.split("_",1)[1]))
                    img2 = Image.open(os.path.join(path2, folder_name, img_name)).resize((512, 512))
                else:
                    img1 = Image.open(os.path.join(path, img_name))
                    img2 = Image.open(os.path.join(path2, folder_name, img_name)).resize((512, 512))

                psnr = calculate_psnr(img1, img2)
                values.append(psnr)
            except Exception as e:
                print(f"Error: {os.path.join(path2, folder_name, img_name)}")
                print(e)
                continue
    print(f"PSNR: {np.mean(values)}, length: {len(values)}")