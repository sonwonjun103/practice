import torchvision
import random
from utils.config import CFG
import numpy as np
import cv2

degree1 = random.randint(0, 45)
degree2 = random.randint(45,90)
degree3 = random.randint(90,135)
degree4 = random.randint(135,180)
print(f"degree : {degree1} {degree2} {degree3} {degree4}")

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((CFG['IMAGE_SIZE'], CFG['IMAGE_SIZE'])),
    #torchvision.transforms.AugMix(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomRotation(degrees=degree1),
    torchvision.transforms.RandomRotation(degrees=degree2),
    torchvision.transforms.RandomRotation(degrees=degree3),
    torchvision.transforms.RandomRotation(degrees=degree4),
    torchvision.transforms.RandomVerticalFlip(p=0.3),
    torchvision.transforms.RandomAdjustSharpness(sharpness_factor=3),  
])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((CFG['IMAGE_SIZE'], CFG['IMAGE_SIZE'])),
    torchvision.transforms.ToTensor(),
])

def gray_to_rgb(img):
    copy_img=img.copy()

    min=np.min(copy_img)
    max=np.max(copy_img)

    copy_img1=copy_img - min
    copy_img=copy_img1 / np.max(copy_img1)

    copy_img*=2**8-1
    copy_img=copy_img.astype(np.uint8)

    copy_img=np.expand_dims(copy_img, axis=-1)
    copy_img=cv2.cvtColor(copy_img, cv2.COLOR_GRAY2BGR)

    return copy_img