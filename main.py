# deepfake_detection_hybrid.ipynb or main.py

# =======================
# 1. Imports and Setup
# =======================
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.io import read_image
from PIL import Image
from tqdm import tqdm
import copy

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =======================
# 2. Data Preparation
# =======================

# Custom Dataset for DFDC (assuming images in 'data/real' and 'data/fake')
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for label, subdir in enumerate(['real', 'fake']):
            class_dir = os.path.join(root_dir, subdir)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.samples.append((os.path.join(class_dir, fname), label))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Data augmentation and preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1),
    transforms.ToTensor(),
])

# Load dataset
dataset = DeepfakeDataset('data', transform=transform)
train_len = int(0.33 * len(dataset))
val_len = int(0.33 * len(dataset))
test_len = len(dataset) - train_len - val_len
train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=2)

# =======================
# 3. Model Components
# =======================

# --- CNN Feature Extractor ---
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),
        )
    def forward(self, x):
        return self.conv_layers(x)  # Output: [batch, 128, 32, 32]

# --- Vision Transformer Blocks ---
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=128, patch_size=16, emb_dim=128, img_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)  # [B, emb_dim, n_patches_h, n_patches_w]
        x = x.flatten(2)  # [B, emb_dim, n_patches]
        x = x.transpose(1, 2)  # [B, n_patches, emb_dim]
        return x

class ViTBlock(nn.Module):
    def __init__(self, emb_dim=128, num_heads=8, mlp_dim=256, dropout=0.3):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        x2 = self.ln1(x)
        attn_out, _ = self.attn(x2, x2, x2)
        x = x + attn_out
        x2 = self.ln2(x)
        x = x + self.mlp(x2)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, emb_dim=128, num_heads=8, mlp_dim=256, num_layers=6, dropout=0.3, n_patches=4*4):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, emb_dim))
        self.blocks = nn.ModuleList([
            ViTBlock(emb_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(emb_dim)
    def forward(self, x):
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        return x[:, 0]  # Use CLS token (first patch) for classification

# --- Full Hybrid Model ---
class HybridDeepfakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.patch_embed = PatchEmbedding(in_channels=128, patch_size=8, emb_dim=128, img_size=32)
        self.vit = VisionTransformer(emb_dim=128, num_heads=8, mlp_dim=256, num_layers=6, dropout=0.3, n_patches=16)
        self.fc = nn.Linear(128, 2)
    def forward(self, x):
        x = self.cnn(x)  # [B, 128, 32, 32]
        x = self.patch_embed(x)  # [B, 16, 128]
        x = self.vit(x)  # [B, 128]
        out = self.fc(x)  # [B, 2]
        return out

# =======================
# 4. Self-Distillation Training Loop
# =======================

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """KL divergence between student and teacher outputs (softened by temperature)."""
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)

def train(model, teacher, train_loader, optimizer, epoch, distill_weight=0.3, temperature=2.0):
    model.train()
    total_loss, total_correct = 0, 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        ce_loss = F.cross_entropy(outputs, labels)
        if teacher is not None:
            with torch.no_grad():
                teacher_outputs = teacher(imgs)
            d_loss = distillation_loss(outputs, teacher_outputs, temperature)
            loss = (1 - distill_weight) * ce_loss + distill_weight * d_loss
        else:
            loss = ce_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
    avg_loss = total_loss / len(train_loader.dataset)
    avg_acc = total_correct / len(train_loader.dataset)
    return avg_loss, avg_acc

def evaluate(model, loader):
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    avg_acc = total_correct / len(loader.dataset)
    return avg_loss, avg_acc

# =======================
# 5. Training Script
# =======================

model = HybridDeepfakeModel().to(device)
teacher = None
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
best_val_loss = float('inf')
patience, patience_counter = 5, 0
num_epochs = 20
distill_weight = 0.3
temperature = 2.0
teacher_update_freq = 5

for epoch in range(1, num_epochs + 1):
    # Train
    train_loss, train_acc = train(model, teacher, train_loader, optimizer, epoch, distill_weight, temperature)
    # Validate
    val_loss, val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Acc {train_acc:.4f} | Val Loss {val_loss:.4f}, Acc {val_acc:.4f}")
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping!")
            break
    # Update teacher
    if epoch % teacher_update_freq == 0:
        teacher = copy.deepcopy(model)
        for param in teacher.parameters():
            param.requires_grad = False

# =======================
# 6. Evaluation on Test Set
# =======================

model.load_state_dict(torch.load('best_model.pth'))
test_loss, test_acc = evaluate(model, test_loader)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Optional: Classification report, confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))
print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
