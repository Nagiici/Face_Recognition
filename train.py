import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
from tqdm import tqdm  

# 自定义数据集类
class FERDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []  # 存储 (img_path, label)
        self.class_mapping = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'happy': 3,
            'normal': 4,
            'sad': 5,
            'surprised': 6
        }
        for label in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label)
            if os.path.isdir(label_dir):
                for img_name in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_name)
                    self.samples.append((img_path, self.class_mapping[label.lower()]))
                    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    # 加载数据集
    train_dataset = FERDataset(data_dir='./Training', transform=transform)
    val_dataset   = FERDataset(data_dir='./PublicTest', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 加载模型
    num_classes = 7
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(next(model.parameters()).device)

    # 定义交叉熵损失函数和Adam优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        # 使用 tqdm 显示训练进度
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += torch.sum(preds == labels.data)
            total_samples += images.size(0)

            # 更新进度条显示当前的批次loss
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / total_samples
        epoch_acc = total_correct.double() / total_samples
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

        # 验证过程
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_samples = 0
        with torch.no_grad():
            progress_bar_val = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", leave=False)
            for images, labels in progress_bar_val:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                val_samples += images.size(0)
                progress_bar_val.set_postfix(loss=f"{loss.item():.4f}")
        
        val_loss /= val_samples
        val_acc = val_correct.double() / val_samples
        print(f'Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')

    # 保存模型权重
    torch.save(model.state_dict(), 'vit_fer_model.pth')
