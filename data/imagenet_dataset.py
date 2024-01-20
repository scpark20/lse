import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

class ImagenetDataset(Dataset):
    def __init__(self, root_dir, img_size):
        self.root_dir = root_dir
        self.transform = train_transforms = transforms.Compose([
                            transforms.RandomResizedCrop(img_size),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()
                        ])
        self.images = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)
        
        # 이미지 채널이 1개일 경우, 3채널 이미지로 변환
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 여기서는 모든 이미지에 대해 레이블을 0으로 설정합니다. 필요에 따라 수정할 수 있습니다.
        return image, 0