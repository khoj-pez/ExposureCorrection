import torch
from torch.utils.data import Dataset, DataLoader
import cv2

from PIL import Image
import numpy as np
from tqdm import tqdm

from models.mlp import FCNet
from models.exp import FCNetExposure

def adjust_exposure(image, exposure):
    image = image * exposure
    image = torch.clamp(image, 0, 1)
    return image

def get_coords(res, normalize = False):
    x = y = torch.arange(res)
    xx, yy = torch.meshgrid(x, y)
    coords = torch.stack([xx, yy], dim=-1)
    if normalize:
        coords = coords / (res - 1)
    return coords

def get_psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr

class ImageDataset(Dataset):
    def __init__(self, image_path, image_size, device = torch.device('cpu'), exposure = 1.0):
        self.image = Image.open(image_path).resize(image_size)
        self.rgb_vals = torch.from_numpy(np.array(self.image)).reshape(-1, 3).to(device)
        self.rgb_vals = self.rgb_vals.float() / 255
        self.adjusted_rgb_vals = self.rgb_vals.clone()
        self.coords = get_coords(image_size[0], normalize=True).reshape(-1, 2).to(device)
        self.exposure = torch.full((len(self.rgb_vals), 1), exposure, device=device, dtype=torch.float32)

    def adjust_exposure(self, exposure):
        self.adjusted_rgb_vals = adjust_exposure(self.rgb_vals, exposure)
        self.exposure = torch.full((len(self.rgb_vals), 1), exposure, device=self.exposure.device, dtype=torch.float32)

    def __len__(self):
        return len(self.rgb_vals)
    def __getitem__(self, idx):
        return self.coords[idx], self.adjusted_rgb_vals[idx], self.rgb_vals[idx], self.exposure[idx]

class Trainer:
    def __init__(self, image_path, image_size, model_type = 'mlp', use_pe = True, device = torch.device('cpu'), exposure = 2):
        self.dataset = ImageDataset(image_path, image_size, device, exposure)
        self.dataloader = DataLoader(self.dataset, batch_size=4096, shuffle=True)
        self.pred = None
        self.stop = False
        self.psnr = 0

        self.dataset.adjust_exposure(exposure)


        if model_type == 'mlp':
           self.model = FCNet().to(device)
           self.load_model('mlp_model_weights.pth')
        elif model_type == 'mlp_exposure':
            self.model = FCNetExposure().to(device)
            self.load_model('mlp_exp_model_weights.pth')
        else:
            pass

        lr = 1e-3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

        self.nepochs = 200

    def run(self):
        pbar = tqdm(range(self.nepochs))
        for epoch in pbar:
            if self.stop:
                break
            self.model.train()
            for coords, adjusted_rgb_vals, rgb_vals, exposure in self.dataloader:
                self.optimizer.zero_grad()
                if isinstance(self.model, FCNetExposure):
                    pred = self.model(coords, exposure)
                    loss = self.criterion(pred, rgb_vals)
                else:
                    pred = self.model(coords)
                    loss = self.criterion(pred, adjusted_rgb_vals)
                # loss = self.criterion(pred, rgb_vals)
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                coords = self.dataset.coords
                exposure = self.dataset.exposure
                if isinstance(self.model, FCNetExposure):
                    pred = self.model(coords, exposure)
                else:
                    pred = self.model(coords)
                gt = self.dataset.rgb_vals
                psnr = get_psnr(pred, gt)
                self.psnr = psnr
            self.pred = pred.cpu().numpy().reshape(*self.dataset.image.size[::-1], 3)
            self.pred = (self.pred * 255).astype(np.uint8)
        if isinstance(self.model, FCNet):
            self.save_model('mlp_model_weights.pth')
        if isinstance(self.model, FCNetExposure):
            self.save_model('mlp_exp_model_weights.pth')


    def save_model(self, filename):
        print("Saving model parameters to ", filename)
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename):
        try:
            self.model.load_state_dict(torch.load(filename))
            self.model.eval()
            print("Loaded model parameters from ", filename)
        except FileNotFoundError:
            print("No saved model parameters found for", filename, "starting from scratch.")

    def get_num_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)