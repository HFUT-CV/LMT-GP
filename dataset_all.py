import os.path
import time

import torch
import torch.utils.data as data
from PIL import Image, ImageEnhance, ImageOps
import random
from random import randrange
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def select_k_items(lst, k):
    if k > len(lst):
        raise ValueError("k cannot be greater than the length of the list.")
    if k == len(lst):
        return lst
    # random.seed(time.time())
    indices = random.sample(range(len(lst)), k)
    return [lst[i] for i in indices]

def adjust_brightness_contrast(image_path, brightness_factor, contrast_factor):

    image = Image.open(image_path).convert("RGB")

    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)

    return image
class TrainLabeled(data.Dataset):
    def __init__(self, dataroot, phase, finesize):
        super().__init__()
        self.phase = phase
        self.root = dataroot
        self.fineSize = finesize

        self.dir_A = os.path.join(self.root, self.phase + '/low')
        self.dir_B = os.path.join(self.root, self.phase + '/high')


        # image path
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))

    def __getitem__(self, index):
        input_name = self.A_paths[index]
        ps = self.fineSize
        inp_img = Image.open(self.A_paths[index]).convert("RGB")
        tar_img = Image.open(self.B_paths[index]).convert("RGB")

        w, h = tar_img.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]

        # Data Augmentations
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))



        return inp_img, tar_img, input_name



    def __len__(self):
        return len(self.A_paths)


class TrainUnlabeled(data.Dataset):
    def __init__(self, dataroot, phase, finesize, k):
        super().__init__()
        self.phase = phase
        self.root = dataroot
        self.fineSize = finesize

        self.dir_A = os.path.join(self.root, self.phase + '/low')



        # image path
        self.A_paths = select_k_items(sorted(make_dataset(self.dir_A)), k)



    def __getitem__(self, index):

        input_name = self.A_paths[index]
        ps = self.fineSize
        # brightness_factor = random.uniform(0.2, 0.5)
        # contrast_factor = random.uniform(0.6, 0.7)
        # # noise_stddev = random.uniform(0.6, 1)
        # darkened_image = adjust_brightness_contrast(input_name, brightness_factor, contrast_factor)

        inp_img = Image.open(self.A_paths[index]).convert("RGB")
        # inp_img = data_aug(inp_img)

        w, h = inp_img.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')


        inp_img = TF.to_tensor(inp_img)


        hh, ww = inp_img.shape[1], inp_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]

        # Data Augmentations
        if aug == 1:
            inp_img = inp_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))

        return inp_img, input_name

    def __len__(self):
        return len(self.A_paths)


class ValLabeled(data.Dataset):
    def __init__(self, dataroot, phase, finesize):
        super().__init__()
        self.phase = phase
        self.root = dataroot
        self.fineSize = finesize
        self.mul = 8
        self.dir_A = os.path.join(self.root, self.phase + '/low')
        self.dir_B = os.path.join(self.root, self.phase + '/high')

        # image path
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))



    def __getitem__(self, index):

        inp_img = Image.open(self.A_paths[index]).convert("RGB")
        tar_img = Image.open(self.B_paths[index]).convert("RGB")

        w, h = inp_img.size
        # h, w = inp_img.shape[2], inp_img.shape[3]
        H, W = ((h + self.mul) // self.mul) * self.mul, ((w + self.mul) // self.mul) * self.mul
        padh = H - h if h % self.mul != 0 else 0
        padw = W - w if w % self.mul != 0 else 0
        inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
        # tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')
        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)


        return inp_img, tar_img, h, w, self.A_paths[index].split('/')[-1], self.B_paths[index]



    def __len__(self):
        return len(self.A_paths)


class TestData(data.Dataset):
    def __init__(self, dataroot):
        super().__init__()
        self.root = dataroot
        self.mul = 8


        self.dir_A = os.path.join(self.root)
        # self.dir_C = os.path.join(self.root + '/high')


        # image path
        self.A_paths = sorted(make_dataset(self.dir_A))
        # self.C_paths = sorted(make_dataset(self.dir_C))

        # transform
        # self.transform = ToTensor()  # [0,1]
        # self.transform = transforms.Compose([ transforms.Resize(size=(2048,2048)),
        #     ToTensor()])  # [0,1]
    def __getitem__(self, index):
        inp_img = Image.open(self.A_paths[index]).convert("RGB")
        # tar_img = Image.open(self.C_paths[index]).convert("RGB")

        w, h = inp_img.size
        # h, w = inp_img.shape[2], inp_img.shape[3]
        H, W = ((h + self.mul) // self.mul) * self.mul, ((w + self.mul) // self.mul) * self.mul
        padh = H - h if h % self.mul != 0 else 0
        padw = W - w if w % self.mul != 0 else 0
        inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
        # tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')
        inp_img = TF.to_tensor(inp_img)
        # tar_img = TF.to_tensor(tar_img)


        return inp_img,  h, w


    def __len__(self):
        return len(self.A_paths)


def data_aug(images):
    kernel_size = int(random.random() * 4.95)
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    blurring_image = transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
    color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25)
    strong_aug = images
    if random.random() < 0.8:
        strong_aug = color_jitter(strong_aug)
    strong_aug = transforms.RandomGrayscale(p=0.2)(strong_aug)
    if random.random() < 0.5:
        strong_aug = blurring_image(strong_aug)
    return strong_aug


