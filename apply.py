from utils import *
from tqdm import tqdm
import numpy as np

path = "/data/qit220/Dataset/transform_watermark_perturbtion/wm-anything"
first_save_dir = "/data/qit220/Dataset/transform_watermark_perturbtion/transformation/1level/wm-anything"
second_save_dir = "/data/qit220/Dataset/transform_watermark_perturbtion/transformation/2level/wm-anything"
forth_save_dir = "/data/qit220/Dataset/transform_watermark_perturbtion/transformation/4level/wm-anything"
fifth_save_dir = "/data/qit220/Dataset/transform_watermark_perturbtion/transformation/5level/wm-anything"

# 1-level
trans_list = ["semantic", "style", "realistic", "distortion", "super_resolution", "brightness_contrast", "horizontal_flip", "jpeg_compress", "gaus_blur", "crop_and_resize"]
for i in tqdm(trans_list, desc="1level"):
    img_path = path
    trans = Transformations(img_path)
    save_path = os.path.join(first_save_dir, i)
    os.makedirs(save_path, exist_ok=True)
    trans.apply(i, save_path)

#2-level
for i in tqdm(trans_list, desc="1level"):
    img_path = os.path.join(first_save_dir, i)
    trans = Transformations(img_path)
    for j in tqdm([x for x in trans_list if x != i], desc=f"2level"):
        save_path = os.path.join(second_save_dir, f"{i}_{j}")
        os.makedirs(save_path, exist_ok=True)
        if len(os.listdir(save_path))!=140:
            print("--------------------------------")
            print(f"{i}_{j} save to {save_path}")
            print("--------------------------------")  
            trans.apply(j, save_path)
        else:
            print(f"{i}_{j} already exists")

#3-level
thrid_level_list = []
for i in tqdm(trans_list, desc="1level"):
    temp = []
    for j in tqdm([x for x in trans_list if x != i], desc=f"2level"):
        temp.append(f"{i}-{j}")
    selected = random.sample(temp, 5)
    for selected_item in selected:
        third_item = random.choice([x for x in trans_list if x not in selected_item])
        thrid_level_list.append(f"{selected_item}-{third_item}")

np.save(os.path.join('./output', "thrid_level_list.npy"), thrid_level_list)

thrid_level_list = np.load(os.path.join('./output', "thrid_level_list.npy"))
thrid_level_list = [str(x) for x in thrid_level_list]
thrid_save_dir = "/data/qit220/Dataset/transform_watermark_perturbtion/transformation/3level/wm-anything"
for i in tqdm(thrid_level_list, desc="3level"):
    img_path = os.path.join(second_save_dir, i.rsplit("-", 1)[0].replace("-", "_"))
    trans = Transformations(img_path)
    save_path = os.path.join(thrid_save_dir, i)
    os.makedirs(save_path, exist_ok=True)
    if len(os.listdir(save_path))!=140:
        print("--------------------------------")
        print(f"{i} save to {save_path}")
        print("--------------------------------")  
        trans.apply(i.rsplit("-", 1)[1], save_path)
    else:
        print(f"{i} already exists")

#4-level
forth_level_list = []
for i in tqdm(trans_list):
    first_list = [x for x in thrid_level_list if x.split("-", 1)[0] == i]
    less_list = random.sample(first_list, 3)

    for j in tqdm(less_list, desc=f"4level"):
        used_methods = j.split("-")
        forth_item = random.choice([x for x in trans_list if x not in used_methods])

        forth_level_list.append(f"{str(j)}-{forth_item}")

np.save(os.path.join('./output', "forth_level_list.npy"), forth_level_list)
forth_level_list = np.load(os.path.join('./output', "forth_level_list.npy"))
for i in tqdm(forth_level_list, desc="4level"):
    img_path = os.path.join(thrid_save_dir, i.rsplit("-", 1)[0])
    trans = Transformations(img_path)
    save_path = os.path.join(forth_save_dir, i)
    os.makedirs(save_path, exist_ok=True)
    if len(os.listdir(save_path))!=140:
        print("--------------------------------")
        print(f"{i} save to {save_path}")
        print("--------------------------------")  
        trans.apply(i.rsplit("-", 1)[1], save_path)
    else:
        print(f"{i} already exists")



#5-level
fifth_level_list = []
for i in tqdm(trans_list):
    first_list = [x for x in forth_level_list if x.split("-", 1)[0] == i]
    less_list = random.sample(first_list, 2)
    for j in tqdm(less_list, desc=f"5level"):
        fifth_item = random.choice([x for x in trans_list if x not in less_list])
        fifth_level_list.append(f"{str(j)}-{fifth_item}")

np.save(os.path.join('./output', "fifth_level_list.npy"), fifth_level_list)

fifth_level_list = np.load(os.path.join('./output', "fifth_level_list.npy"))
for i in tqdm(fifth_level_list, desc="5level"):
    img_path = os.path.join(forth_save_dir, i.rsplit("-", 1)[0])
    trans = Transformations(img_path)
    save_path = os.path.join(fifth_save_dir, i)
    os.makedirs(save_path, exist_ok=True)
    if len(os.listdir(save_path))!=140:
        print("--------------------------------")
        print(f"{i} save to {save_path}")
        print("--------------------------------")  
        trans.apply(i.rsplit("-", 1)[1], save_path)
    else:
        print(f"{i} already exists")