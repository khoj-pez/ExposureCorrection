import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import random
import os

from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm

from models.mlp import FCNet
from models.pe import PE
from models.exp import FCNetExposure, ExposureNet

def adjust_exposure(image, exposure):
    image = image * exposure
    image = torch.clamp(image, 0, 1)
    return image

# def adjust_exposure_np(image, exposure):
#     correction_factor = 1.0 / exposure
#     image = image * correction_factor
#     image = torch.clamp(image, 0, 1)
#     return image

def adjust_exposure_np(image, exposure_matrix):
    # Convert tensors to numpy for convenience
    image_np = image.numpy()
    exposure_np = exposure_matrix.numpy()

    # Prepare output image
    adjusted_image = np.zeros_like(image_np)

    # For each pixel in the image, apply the transformation matrix
    for i in range(image_np.shape[0]):
        for j in range(image_np.shape[1]):
            pixel = image_np[i, j]
            adjusted_pixel = np.dot(exposure_np, pixel)
            adjusted_image[i, j] = adjusted_pixel

    # Clamp values between 0 and 1
    adjusted_image = np.clip(adjusted_image, 0, 1)

    return torch.tensor(adjusted_image, dtype=torch.float32)

def regularization_term(predicted_matrix, alpha=1.0):
    identity_matrix = torch.eye(3, device=predicted_matrix.device)
    penalty = (predicted_matrix - identity_matrix) ** 2
    return alpha * penalty.sum()

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

def split_image(image, patch_size):
    patches = []
    height, width = image.shape[:2]
    for i in range(0, height - patch_size + 1, patch_size // 2):
        for j in range(0, width - patch_size + 1, patch_size // 2):
            patches.append(image[i:i + patch_size, j:j + patch_size])
    return patches

def split_and_save_image(image_path, patch_size, output_dir, overlap=64):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the image and resize
    img = Image.open(image_path).resize((256, 256))
    
    # Convert to numpy array for slicing
    img_np = np.array(img)

    stride = patch_size - overlap

    for i in range(0, img_np.shape[0] - patch_size + 1, stride):
        for j in range(0, img_np.shape[1] - patch_size + 1, stride):
            patch = img_np[i:i+patch_size, j:j+patch_size]
            
            # Saving the patch to the specified output directory
            patch_name = f"patch_{i}_{j}.jpg"
            patch_path = os.path.join(output_dir, patch_name)
            Image.fromarray(patch).save(patch_path)
            
def get_overlapping_patches(img_np, patch_size, overlap=0):
    patches = []
    stride = patch_size - overlap
    for i in range(0, img_np.shape[0] - patch_size + 1, stride):
        for j in range(0, img_np.shape[1] - patch_size + 1, stride):
            patch = img_np[i:i+patch_size, j:j+patch_size]
            patches.append(patch)

    return patches


class ImageDataset(Dataset):
    def __init__(self, image_path, patch_size, device=torch.device('cpu')):
        self.image = Image.open(image_path).resize((256, 256))
        self.gt_image = np.array(self.image)
        self.patches = get_overlapping_patches(np.array(self.image), patch_size, 64)
        self.exposures = torch.tensor([random.uniform(1.2, 1.5) for _ in self.patches]).unsqueeze(-1).to(device)
        self.original_image = np.array(self.image)

        self.patches = [adjust_exposure(torch.from_numpy(patch).float().to(device)/255, exposure) for patch, exposure in zip(self.patches, self.exposures)]
        print(len(self.patches))


    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx], self.get_patch_exposure(idx)

    def get_patch_exposure(self, idx):
        return self.exposures[idx]

class Trainer:
    def __init__(self, image_path, patch_size, use_pe = True, device = torch.device('cpu')):
        self.dataset = ImageDataset(image_path, patch_size, device)
        
        self.visualization_patches, self.visualization_targets = next(iter(DataLoader(self.dataset, batch_size=4096, shuffle=False)))

        self.dataloader = DataLoader(self.dataset, batch_size=4096, shuffle=True)

        self.model = ExposureNet().to(device)

        lr = 1e-3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

        self.nepochs = 200

    def run(self):
        pbar = tqdm(range(self.nepochs))
        for epoch in pbar:
            self.model.train()
            epoch_loss = 0
            for patches, target_exposures in self.dataloader:
                target_exposures = target_exposures.unsqueeze(-1).expand(-1, 3, 3)
                self.optimizer.zero_grad()
                pred_exposures = self.model(patches)
                loss = self.criterion(pred_exposures, target_exposures)

                reg_loss = loss + regularization_term(pred_exposures)
                reg_loss.backward()
                # loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(self.dataloader)

            self.model.eval()
            with torch.no_grad():
                patches = self.visualization_patches
                target_exposures = self.visualization_targets
                pred_exposures = self.model(patches)
                target_exposures = target_exposures.unsqueeze(-1).expand(-1, 3, 3)
                # target_exposures = target_exposures.expand(-1, 3)
                val_loss = self.criterion(pred_exposures, target_exposures)

            reconstructed_image = np.zeros((256, 256, 3), dtype=np.float32)
            weight_map = np.zeros((256, 256, 3), dtype=np.float32)
            stride = 128 - 64  # assuming 50% overlap and patch size of 128

            row_patches = (256 - patch_size) // stride + 1
            col_patches = (256 - patch_size) // stride + 1

            for row in range(row_patches):
                for col in range(col_patches):
                    idx = row * col_patches + col
                    patch = patches[idx].cpu()
                    exposure = pred_exposures[idx].cpu()
                    adjusted_patch = adjust_exposure_np(patch, exposure)
                    
                    i = row * stride
                    j = col * stride

                    reconstructed_image[i:i+patch_size, j:j+patch_size] += adjusted_patch.numpy() * 255
                    weight_map[i:i+patch_size, j:j+patch_size] += 1

            # Normalize by weight_map
            reconstructed_image /= weight_map
            reconstructed_image = np.clip(reconstructed_image, 0, 255)  # Ensure values are within [0, 255]
            reconstructed_image = reconstructed_image.astype(np.uint8)
            concatenated_image = np.hstack((self.dataset.gt_image, reconstructed_image))
            self.visualize(concatenated_image, f'Epoch: {epoch}')

    def get_num_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def visualize(self, image, text):
        save_image = np.ones((300, 512, 3), dtype=np.uint8) * 255
        img_start = (300 - 256)
        # save_image[img_start:img_start + 256, :, :] = image
        save_image[img_start:img_start + image.shape[0], :image.shape[1], :] = image

        save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
        position = (100, 20)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        color = (255, 0, 0)
        thickness = 2

        cv2.putText(save_image, text, position, font, scale, color, thickness)
        cv2.imshow('image', save_image)
        cv2.waitKey(1)

if __name__ == '__main__':
    image_path = 'image4.jpg'
    patch_size = 128
    device = torch.device('cpu')
    
    trainer = Trainer(image_path, patch_size, device)
    print('# params: {}'.format(trainer.get_num_params()))
    # split_and_save_image("image.jpg", 128, "output_patches")
    trainer.run()

