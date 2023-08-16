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

def adjust_exposure_np(image, exposure):
    correction_factor = 1.0 / exposure
    image = image * correction_factor
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

def split_image(image, patch_size):
    patches = []
    height, width = image.shape[:2]
    for i in range(0, height - patch_size + 1, patch_size // 2):
        for j in range(0, width - patch_size + 1, patch_size // 2):
            patches.append(image[i:i + patch_size, j:j + patch_size])
    return patches

# def split_and_save_image(image_path, patch_size, output_dir):    
#     # Create the output directory if it doesn't exist
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     # Open the image
#     img = Image.open(image_path).resize((256, 256))
    
#     # Extract name and extension for output naming
#     name, ext = os.path.splitext(os.path.basename(image_path))
    
#     # Dimensions of the image
#     w, h = img.size
    
#     # Iterate through the image, grabbing patches
#     for i in range(0, h, patch_size):
#         for j in range(0, w, patch_size):
#             # Check if the patches fit into the image. If not, skip to next iteration.
#             if i + patch_size > h or j + patch_size > w:
#                 continue
            
#             # Define the bounding box for each patch
#             box = (j, i, j + patch_size, i + patch_size)
#             patch = img.crop(box)
            
#             # Save the patch
#             patch_filename = os.path.join(output_dir, f"{name}_{i}_{j}{ext}")
#             patch.save(patch_filename)

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

        # output_directory = "output_patches"
        # if not os.path.exists(output_directory):
        #     os.makedirs(output_directory)

        for epoch in pbar:
            self.model.train()
            epoch_loss = 0
            for patches, target_exposures in self.dataloader:
                self.optimizer.zero_grad()
                pred_exposures = self.model(patches)
                loss = self.criterion(pred_exposures, target_exposures)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(self.dataloader)

            self.model.eval()
            with torch.no_grad():
                # Compute the validation loss
                # patches, target_exposures = next(iter(self.dataloader))  # Use one batch from the data loader
                patches = self.visualization_patches
                target_exposures = self.visualization_targets
                pred_exposures = self.model(patches)
                val_loss = self.criterion(pred_exposures, target_exposures)

            # patches, target_exposures = next(iter(self.dataloader))
            
            # Saving the patches
            # for idx, patch in enumerate(patches):
            #     patch_np = patch.cpu().numpy()
            #     patch_img = Image.fromarray((patch_np * 255).astype(np.uint8))

                # # Draw index on the patch
                # draw = ImageDraw.Draw(patch_img)
                # draw.text((50,50), str(idx), fill="red")  # Adjust position and color as needed

                # patch_img.save(os.path.join(output_directory, f'patch_{epoch}_{idx}.jpg'))
            
            # reconstructed_image = np.zeros((256, 256, 3), dtype=np.uint8)  # assuming 256x256 image size
            # stride = 128  # assuming 50% overlap and patch size of 128
            # for i in range(0, 256 - 128 + 1, stride):
            #     for j in range(0, 256 - 128 + 1, stride):                    
            #         idx = i // stride * (256 // stride) + j // stride
            #         patch = patches[idx].cpu()
            #         exposure = pred_exposures[idx].cpu()
            #         adjusted_patch = adjust_exposure_np(patch, exposure)
            #         reconstructed_image[i:i+128, j:j+128] = (adjusted_patch * 255).numpy().astype(np.uint8)

            # cumulative_image = np.zeros((256, 256, 3), dtype=np.float32)
            # count_image = np.zeros((256, 256, 3), dtype=np.float32)

            # stride = 128  # assuming 50% overlap and patch size of 128
            # for i in range(0, 256 - 128 + 1, stride):
            #     for j in range(0, 256 - 128 + 1, stride):
            #         idx = i // stride * (256 // stride) + j // stride
            #         patch = patches[idx].cpu()
            #         exposure = pred_exposures[idx].cpu()
            #         adjusted_patch = adjust_exposure_np(patch, exposure)
                    
            #         # Add the adjusted patch values to the cumulative image
            #         cumulative_image[i:i+128, j:j+128] += (adjusted_patch * 255).numpy().astype(np.float32)
                    
            #         # Increment the count image
            #         count_image[i:i+128, j:j+128] += 1
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


            # for row in range(row_patches):
            #     for col in range(col_patches):
            #         i, j = row * stride, col * stride
            #         idx = row * col_patches + col
            #         patch = patches[idx].cpu()
            #         exposure = pred_exposures[idx].cpu()
            #         adjusted_patch = adjust_exposure_np(patch, exposure)
                    
            #         reconstructed_image[i:i+patch_size, j:j+patch_size] = adjusted_patch.numpy() * 255

            # reconstructed_image = reconstructed_image.astype(np.uint8)



            # final_img = Image.fromarray(reconstructed_image)
            # final_img.save(os.path.join(output_directory, f'final_reconstructed_{epoch}.jpg'))
            
            # patches, target_exposures = next(iter(self.dataloader))
            
            # # We will collect all the original patches here and then join them into one image
            # original_patches = []
            
            # for patch, target_exposure in zip(patches, target_exposures):
            #     # No exposure adjustment, just collect the original patches
            #     original_patches.append(patch.cpu().numpy().transpose((1, 2, 0)))
            
            # # Now, join the original patches into one image
            # # Assume that the patches form a square grid, and that each patch is a square
            # patches_per_dim = int(np.sqrt(len(original_patches)))
            # patch_size = original_patches[0].shape[0]  # The size of a patch along one dimension
            # reconstructed_image = np.zeros((patches_per_dim*patch_size, patches_per_dim*patch_size, 3))
            
            # for i in range(patches_per_dim):
            #     for j in range(patches_per_dim):
            #         reconstructed_image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :] = original_patches[i*patches_per_dim + j]

            # # Visualize the reconstructed image

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



        # def correct_and_reconstruct(self, image, exposure):
        # # Correct the exposure
        # corrected_image = adjust_exposure(image, exposure)

        # # Reconstruct the image
        # patches = split_image(corrected_image, self.dataset.patch_size)
        # patches = torch.stack(patches, dim=0)
        # pred_exposures = self.model(patches)
        # pred_exposures = pred_exposures.squeeze(-1)
        # pred_exposures = pred_exposures.view(1, -1)
        # pred_exposures = torch.clamp(pred_exposures, 0.5, 1.5)
        # pred_exposures = pred_exposures.squeeze(0)
        # pred_exposures = pred_exposures.cpu().numpy()

        # # Reconstruct the image
        # patches = split_image(corrected_image, self.dataset.patch_size)
        # patches = [adjust_exposure(patch, exposure) for patch, exposure in zip(patches, pred_exposures)]
        # reconstructed_image = np.zeros_like(image)
        # reconstructed_image = np.array(reconstructed_image)
        # reconstructed_image = reconstructed_image.astype(np.float32)
        # reconstructed_image = cv2.cvtColor(reconstructed_image, cv2.COLOR_RGB2BGR)
        # height, width = image.shape[:2]
        # for i in range(0, height - self.dataset.patch_size + 1, self.dataset.patch_size // 2):
        #     for j in range(0, width - self.dataset.patch_size + 1, self.dataset.patch_size // 2):
        #         reconstructed_image[i:i + self.dataset.patch_size, j:j + self.dataset.patch_size] += patches.pop(0)
        # reconstructed_image = reconstructed_image / 4
        # reconstructed_image = np.clip(reconstructed_image, 0, 1)
        # reconstructed_image = (reconstructed_image * 255).astype(np.uint8)
        # reconstructed_image = cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2RGB)

        # return reconstructed_image

