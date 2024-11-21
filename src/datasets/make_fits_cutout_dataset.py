# Import necessary libraries
import os
import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from cata2data import CataData
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from logging import getLogger

# Initialize logger and seed
_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
logger = getLogger()

# File paths
image_path = '/content/drive/MyDrive/im_18k4as.deeper.DI.int.restored.fits'
catalogue_path = '/content/drive/MyDrive/ijepa_logs/catalogue.txt'

# Step 1: Open the FITS file and determine the RA/DEC range
with fits.open(image_path) as hdul:
    wcs = WCS(hdul[0].header, naxis=2)
    image_shape = hdul[0].data.shape[-2:]  # Get the spatial dimensions
    print("Image dimensions (height, width):", image_shape)

    bottom_left = SkyCoord.from_pixel(0, 0, wcs=wcs)
    top_right = SkyCoord.from_pixel(image_shape[1] - 1, image_shape[0] - 1, wcs=wcs)
    ra_min, dec_min = bottom_left.ra.deg, bottom_left.dec.deg
    ra_max, dec_max = top_right.ra.deg, top_right.dec.deg

print(f"RA range: {ra_min} to {ra_max}")
print(f"DEC range: {dec_min} to {dec_max}")

# Step 2: Generate Random RA/DEC and Create a Catalogue
num_cutouts = 5
ra_values = np.random.uniform(ra_min, ra_max, num_cutouts)
dec_values = np.random.uniform(dec_min, dec_max, num_cutouts)
df = pd.DataFrame({"RA_host": ra_values, "DEC_host": dec_values, "COSMOS": np.arange(1, num_cutouts + 1)})

# Save catalogue to file
with open(catalogue_path, 'w') as f:
    f.write("# RA_host DEC_host COSMOS\n")
    df.to_csv(f, sep=' ', index=False, header=False)

print("Catalogue saved to:", catalogue_path)

# Step 3: Dataset Class for FITS Cutouts
class FitsCutoutDataset(Dataset):
    def __init__(self, catalogue_path, image_path, cutout_size=224, transform=None, output_folder="./cutouts"):
        self.cata_data = CataData(
            catalogue_paths=[catalogue_path],
            image_paths=[image_path],
            cutout_shape=cutout_size,
            field_names=['COSMOS'],
            catalogue_kwargs={'format': 'commented_header', 'delimiter': ' '}
        )
        self.transform = transform
        self.saved_cutouts = 0
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def __len__(self):
        return len(self.cata_data)

    def __getitem__(self, idx):
        cutout, metadata = self.cata_data[idx]
        if self.transform:
            cutout = self.transform(cutout)

        # Save and display the first few cutouts
        if self.saved_cutouts < 5:
            cutout_image = Image.fromarray(cutout, mode='F')
            output_path = os.path.join(self.output_folder, f"cutout_{self.saved_cutouts}.png")
            cutout_image.save(output_path)
            print(f"Saved grayscale cutout to {output_path}")
            plt.imshow(cutout, cmap='gray')
            plt.title(f"Cutout {self.saved_cutouts}")
            plt.show()
            self.saved_cutouts += 1
        
        return cutout

# Step 4: Transformation Pipeline
def create_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

# Step 5: DataLoader for Cutouts
def make_fits_cutout_dataset(
    transform,
    batch_size=4,
    pin_mem=True,
    num_workers=2,
    cutout_size=224
):
    dataset = FitsCutoutDataset(
        catalogue_path=catalogue_path,
        image_path=image_path,
        cutout_size=cutout_size,
        transform=transform
    )
    logger.info('FITS cutout dataset created')

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_mem,
        num_workers=num_workers
    )
    logger.info('FITS cutout data loader created')
    return dataset, data_loader

# Step 6: Main Execution
if __name__ == "__main__":
    # Create transformation pipeline
    transform = create_transform()

    # Create dataset and dataloader
    dataset, dataloader = make_fits_cutout_dataset(transform=transform)

    # Loop through the DataLoader
    print("Displaying first few cutouts:")
    for batch_idx, cutouts in enumerate(dataloader):
        if batch_idx == 1:  # Display only the first batch
            break

