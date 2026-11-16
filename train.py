import paddle
import paddle.nn as nn
from paddle.io import Dataset, DataLoader, WeightedRandomSampler, BatchSampler
from paddle.vision.models import mobilenet_v3_small
from paddle.vision.datasets import DatasetFolder
from paddle.vision.transforms import Compose, Resize, Transpose, Normalize, RandomRotation, RandomHorizontalFlip, \
    ColorJitter
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from PIL import Image

# --- 全局配置 ---
DATA_DIR = 'data/train'
OUTPUT_DIR = 'output'
MODEL_NAME = 'best_model.pdparams'
NUM_CLASSES = 54
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# 确保输出目录存在
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# --- 1. 模型定义 (MobileNetV3 Small) ---
class HerbalModel(nn.Layer):
    def __init__(self, num_classes):
        super(HerbalModel, self).__init__()
        # 加载预训练的 MobileNetV3 Small
        # PaddlePaddle 3.0.0b1 版本兼容
        self.backbone = mobilenet_v3_small(pretrained=True)

        # 替换分类器最后一层
        last_in_features = self.backbone.classifier[-1].weight.shape[1]
        self.backbone.classifier[-1] = nn.Linear(last_in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# --- 2. 自定义数据集  ---
class CustomDataset(Dataset):
    def __init__(self, samples, class_to_idx, transform=None):
        super(CustomDataset, self).__init__()
        self.samples = samples  # (path, target) pairs
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        # 使用 PIL 加载图像，解决可能出现的加载问题
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, np.int64(target)

    def __len__(self):
        return len(self.samples)


# --- 3. 数据集划分与类别映射保存 (解决映射缺失问题) ---
def get_datasets(data_dir, val_split=0.1):
    # 定义训练和验证集的预处理
    train_transforms = Compose([
        Resize((224, 224)),
        RandomRotation(15),
        RandomHorizontalFlip(),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        Transpose(),  # HWC -> CHW
        Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    ])

    val_transforms = Compose([
        Resize((224, 224)),
        Transpose(),
        Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    ])

    # 基础加载：利用 DatasetFolder 获取所有样本和类别映射
    full_dataset_base = DatasetFolder(data_dir, transform=None)
    all_samples = full_dataset_base.samples
    num_total = len(all_samples)
    indices = list(range(num_total))
    np.random.shuffle(indices)

    # 划分索引
    split = int(np.floor(val_split * num_total))
    val_indices, train_indices = indices[:split], indices[split:]

    # 获取划分后的样本
    train_samples = [all_samples[i] for i in train_indices]
    val_samples = [all_samples[i] for i in val_indices]

    # 创建应用了不同 transforms 的数据集实例
    class_to_idx = full_dataset_base.class_to_idx
    train_dataset = CustomDataset(train_samples, class_to_idx, transform=train_transforms)
    val_dataset = CustomDataset(val_samples, class_to_idx, transform=val_transforms)

    # 保存类别映射：索引 -> 药名
    idx_to_class = {str(v): k for k, v in class_to_idx.items()}
    MAP_PATH = os.path.join(OUTPUT_DIR, "idx_to_class.json")
    with open(MAP_PATH, 'w', encoding='utf-8') as f:
        json.dump(idx_to_class, f, ensure_ascii=False, indent=4)
    print(f"✅ 类别映射已成功保存到 {MAP_PATH}")

    # 计算训练集的采样权重 (解决类别不均衡)
    train_targets = [sample[1] for sample in train_samples]
    class_counts = np.bincount(train_targets)
    # 权重与样本数量成反比，防止除以0
    class_weights = 1. / np.where(class_counts > 0, class_counts, 1e-10)
    sample_weights = [class_weights[t] for t in train_targets]
    sample_weights_array = np.array(sample_weights, dtype='float32')

    # 创建加权采样器
    train_sampler = WeightedRandomSampler(
        weights=sample_weights_array,
        num_samples=len(sample_weights),
        replacement=True  # 允许重复采样
    )

    return train_dataset, val_dataset, train_sampler


# --- 4. 绘图功能 ---
def plot_metrics(history, output_dir):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.style.use('seaborn-v0_8-whitegrid')

    # 损失曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue', marker='o', markersize=3)
    plt.plot(epochs, history['val_loss'], label='Validation Loss', color='red', marker='x', markersize=3)
    plt.title('Loss over Epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy', color='green', marker='o', markersize=3)
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy', color='purple', marker='x', markersize=3)
    plt.title('Accuracy over Epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=10)

    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plot_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(plot_path)
    print(f"📈 训练指标图已保存到 {plot_path}")
    plt.close()


# --- 5. 训练主函数 ---
def train():
    # 强制使用 GPU
    device = 'gpu' if paddle.device.is_compiled_with_cuda() else 'cpu'
    paddle.set_device(device)
    print(f"使用设备: {paddle.device.get_device()}")

    # 1. 获取数据和采样器
    train_dataset, val_dataset, train_sampler = get_datasets(DATA_DIR)

    # 使用 BatchSampler 封装 WeightedRandomSampler
    train_batch_sampler = BatchSampler(
        sampler=train_sampler,
        batch_size=BATCH_SIZE,
        drop_last=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,  # 使用 batch_sampler
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    # 2. 模型、损失函数、优化器
    model = HerbalModel(NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()

    # 余弦退火学习率调度器
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=LEARNING_RATE,
        T_max=len(train_loader) * EPOCHS
    )

    # AdamW 优化器
    optimizer = paddle.optimizer.AdamW(
        learning_rate=scheduler,
        parameters=model.parameters(),
        weight_decay=1e-4  # 添加权重衰减
    )

    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # 3. 训练循环
    for epoch in range(1, EPOCHS + 1):
        # --- 训练阶段 ---
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_id, data in enumerate(train_loader()):
            inputs, labels = data
            logits = model(inputs)
            loss = criterion(logits, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            scheduler.step()  # 学习率调度

            # 统计 - 修复 IndexError
            total_train_loss += loss.numpy().item()  # 使用 .item() 提取标量值
            preds = paddle.argmax(logits, axis=1)
            correct_train += paddle.sum(preds == labels).numpy().item()
            total_train += labels.shape[0]

            if (batch_id + 1) % 50 == 0:
                # 修复 IndexError
                print(
                    f"Epoch [{epoch}/{EPOCHS}], Step [{batch_id + 1}/{len(train_loader)}], LR: {scheduler.get_lr():.6f}, Loss: {loss.numpy().item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = correct_train / total_train * 100

        # --- 验证阶段 ---
        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with paddle.no_grad():
            for data in val_loader():
                inputs, labels = data
                logits = model(inputs)
                loss = criterion(logits, labels)

                # 修复 IndexError
                total_val_loss += loss.numpy().item()  # 使用 .item() 提取标量值
                preds = paddle.argmax(logits, axis=1)
                correct_val += paddle.sum(preds == labels).numpy().item()
                total_val += labels.shape[0]

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct_val / total_val * 100

        print(f"\n--- Epoch {epoch} Summary ---")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Acc: {val_acc:.2f}%")

        # 4. 记录历史和保存模型
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(OUTPUT_DIR, MODEL_NAME)
            paddle.save(model.state_dict(), save_path)
            print(f"🏆 验证准确率提升至 {best_acc:.2f}%, 模型已保存到 {save_path}")

    # 5. 训练结束，绘制图像
    plot_metrics(history, OUTPUT_DIR)
    print("\n🎉 训练完成!")


if __name__ == '__main__':
    train()