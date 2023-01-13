import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import glob
import PIL.Image
import os
from utility import get_resistance_value


class RFDataset(torch.utils.data.Dataset):
    
    def __init__(self, directory):
        self.directory = directory
        self.image_paths = glob.glob(os.path.join(self.directory, '*.png'))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        image = PIL.Image.open(image_path)
        width, height = image.size
        res_value = int(get_resistance_value(image_path)) + 7
        
        # image = transforms.functional.resize(image, (640, 480))
        image = transforms.functional.to_tensor(image)
        image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        return image, F.one_hot(torch.tensor(res_value), num_classes=15)
    
