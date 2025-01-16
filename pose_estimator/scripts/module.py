import json
import math
import torch
from torch import nn, optim
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pytorch_lightning as pl
import random
import numpy as np

# Function to compute rotation matrix
def get_R(pitch, yaw, roll):
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    Rx = [[1, 0, 0], [0, cp, -sp], [0, sp, cp]]
    Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
    Rz = [[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]

    def matmul(A, B):
        return [[sum(a*b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]

    R = matmul(Rz, matmul(Ry, Rx))
    return R

# Utility functions
def normalize_vector(v):
    return v / (v.norm(dim=1, keepdim=True) + 1e-8)

def cross_product(u, v):
    return torch.cross(u, v, dim=1)

def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:, :3]
    y_raw = poses[:, 3:6]
    x = normalize_vector(x_raw)
    z = normalize_vector(cross_product(x, y_raw))
    y = cross_product(z, x)
    x, y, z = x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)
    return torch.cat((x, y, z), dim=2)

# Custom loss function
class GeodesicLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, m1, m2):
        m = torch.bmm(m1, m2.transpose(1, 2))
        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2        
        theta = torch.acos(torch.clamp(cos, -1 + self.eps, 1 - self.eps))
        return torch.mean(theta)

# Dataset class
class JSONDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path) as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        path = Path(item['image_path']).name
        img = Image.open(f"mito_datasets/{path}").convert('RGB')
        e = item['euler']
        pitch, yaw, roll = float(e['x']), float(e['y']), float(e['z'])
        R = torch.tensor(get_R(pitch, yaw, roll), dtype=torch.float)
        if self.transform:
            img = self.transform(img)
        return img, R

# PyTorch Lightning Module
class SixDRepNetModule(pl.LightningModule):
    def __init__(self, backbone_name='RepVGG-B1g2', backbone_file='', deploy=False, pretrained=False, lr=1e-4):
        super().__init__()
        from scripts.model import SixDRepNet  # Ensure model is imported here
        self.model = SixDRepNet(backbone_name=backbone_name, backbone_file=backbone_file, deploy=deploy, pretrained=pretrained)
        self.criterion = GeodesicLoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, Rs = batch
        preds = self(imgs)
        loss = self.criterion(Rs, preds)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, Rs = batch
        preds = self(imgs)
        loss = self.criterion(Rs, preds)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

# PyTorch Lightning DataModule
class MitoDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=32, num_workers=4):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def fix_seed(self):
        torch.random.initial_seed()  
        torch.cuda.manual_seed_all(42)
        random.seed(42)
        np.random.seed(42)
    
    def setup(self, stage=None):
        self.fix_seed()
        dataset = JSONDataset(self.data_path, transform=self.transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
