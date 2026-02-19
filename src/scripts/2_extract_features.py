import os 
import torch 
import torch.nn as nn 
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset 
from PIL import Image 
from tqdm import tqdm
import h5py


# path configuration 
IMAGE_DIR = ""
OUTPUT_DIR = ""
OUTPUT_FILENAME = ""


# configuration model 
BATCH_SIZE = 64 





# DATASET loader 
class ImageDataset(Dataset):
    def __init__(self, img_dir):
        # get the list of image names and sorted for consistency 
        self.img_dir = img_dir
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.img_names.sort()

        # normalization the image according to ImageNet params 
        self.transform = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.ToTensor(), # convert image (0-255) to Tensor (0-1)
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # get the item, do: convert RGB, transform 
        img_name = self.img_names[index]
        img_path = os.path.join(self.img_dir, img_name)

        # convert to RGB cause ResNet require 
        image = Image.open(img_path).convert("RGB")

        # transform 
        if self.transform:
            image = self.transform(image)

        return image, img_name

        
        

# Feature Extractor (ResNet101)
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        # download ResNet101 Default weight 
        resnet = models.resnet101(weight=models.ResNet101_Weights.DEFAULT)

        # remove 2 last layers (Avepooling, FC)
        layers = list(resnet.children())[:-2]

        self.resnet = nn.Sequential(*layers)

        # freeze params 
        for param in self.resnet.parameters():
            param.requires_grad = False
            
    
    def forward(self, X):
        # X (batch, channel, height, witdh)
        # X (batch, 3, 448, 448)

        out = self.resnet(X) # this return out (batch, 2048, 14, 14)
        # (2048 feature channels, 14, 14) 14x14 spatial dimensions

        # permute the order for later 
        
        out = out.permute(0, 2, 3, 1)

        return out 
    
    


# Main 

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # create output dir 
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    
    
    # Init Dataset and DataLoader 
    dataset = ImageDataset(IMAGE_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, 
                        shuffle=False, num_workers=8)

    
    # Init Model 
    model = FeatureExtractor().to(device)
    model.eval() # switch model to eval mode to turn off dropout, batch norm 
    
    
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    N = len(dataset)

    # open file HDF5 to write 
    with h5py.File(output_path, 'w') as f:
        # create empty dataset 
        # chunks = True for optimize fragment (for random access)
        feat_dset = f.create_dataset('features', (N , 14, 14, 2048), dtype='float32', chunks=True)

        
        # hdf5 orginal does not support python string, so we must define it 
        dt = h5py.string_dtype(encoding='utf-8')

        id_dset = f.create_dataset('ids', (N, ), dtype=dt)

        start_idx = 0 # pointer 
        
        with torch.no_grad(): # turn off grad, save 50% vram 
            for imgs, names in tqdm(loader): # ta-qad-dum (move on) for progress bar in terminal
                imgs = imgs.to(device)

                # forward pass 
                feats = model(imgs)

                # move to cpu (ram) to write to mem 
                feats_np = feats.cpu().numpy()

                batch_size_curr = imgs.size(0) # 0 mean batch_size
                # why don't use BATCH_SIZE -> cause last batch_size ussually odd
                end_idx = start_idx + batch_size_curr
                
                # write to file 
                feat_dset[start_idx : end_idx] = feats_np
                id_dset[start_idx: end_idx] = names 
                
                start_idx = end_idx # this make sure, every loop, it will increase batch_size len
                
                
                
main()

            