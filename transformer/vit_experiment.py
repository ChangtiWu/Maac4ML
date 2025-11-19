import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import json
import os
from datetime import datetime


# 精度控制层 - 保留小数点后n位
class PrecisionControl(nn.Module):
    def __init__(self, decimal_places=None):
        super(PrecisionControl, self).__init__()
        self.decimal_places = decimal_places

    def forward(self, x):
        if self.decimal_places is None:
            return x
        # 四舍五入到指定小数位
        scale = 10 ** self.decimal_places
        return torch.round(x * scale) / scale


# Patch Embedding层
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256,
                 precision_decimal=None):
        super(PatchEmbedding, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding卷积层（相当于ViT的第一层）- 线性投影，无激活函数
        self.projection = nn.Conv2d(in_channels, embed_dim,
                                    kernel_size=patch_size, stride=patch_size)

        # 第一层的精度控制
        self.precision_control = PrecisionControl(precision_decimal) if precision_decimal is not None else None

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.projection(x)  # [B, embed_dim, H/P, W/P]

        # 应用精度控制（如果有）
        if self.precision_control is not None:
            x = self.precision_control(x)

        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]

        return x


# Multi-Head Self-Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# MLP Block
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, dropout=0.1):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Transformer Encoder Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# Vision Transformer
class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=100,
                 embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.0, dropout=0.1,
                 precision_decimal=None):
        super(VisionTransformer, self).__init__()

        # Patch Embedding (第一层 - 只有精度控制，无激活函数)
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim,
                                         precision_decimal)

        num_patches = self.patch_embed.num_patches

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize patch embedding
        nn.init.xavier_uniform_(self.patch_embed.projection.weight)
        if self.patch_embed.projection.bias is not None:
            nn.init.zeros_(self.patch_embed.projection.bias)

        # Initialize positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, embed_dim]

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classification
        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)

        return x


def ViT_Small(precision_decimal=None, num_classes=100):
    """Small ViT for CIFAR-100"""
    return VisionTransformer(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        precision_decimal=precision_decimal
    )


# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total

    return train_loss, train_acc


# 测试函数
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total

    return test_loss, test_acc


# 运行实验
def run_experiment(experiment_name, experiment_id, precision_decimal,
                   log_file, checkpoint_dir, num_epochs=50, batch_size=256, learning_rate=0.001):
    print(f"\n{'='*60}")
    print(f"开始实验: {experiment_name}")
    print(f"精度: {precision_decimal if precision_decimal else '无限制'}")
    print(f"{'='*60}\n")

    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")

    # CIFAR-100数据集路径
    data_root = os.path.expanduser('~/projects/MABIPFE/dataset')

    # 数据预处理 (CIFAR-100的归一化参数)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # 加载CIFAR-100数据集
    print("正在加载CIFAR-100数据集...")
    trainset = torchvision.datasets.CIFAR100(root=data_root, train=True,
                                            download=False, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                            shuffle=True, num_workers=32)

    testset = torchvision.datasets.CIFAR100(root=data_root, train=False,
                                           download=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size,
                           shuffle=False, num_workers=32)
    print("数据集加载完成\n")

    # 创建模型 (CIFAR-100有100个类别)
    model = ViT_Small(precision_decimal=precision_decimal,
                     num_classes=100).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 训练循环
    best_acc = 0
    last_test_acc = 0
    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc = train(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = test(model, testloader, criterion, device)
        scheduler.step()

        epoch_time = time.time() - start_time

        # 更新最佳准确率（用于日志记录）
        if test_acc > best_acc:
            best_acc = test_acc

        # 记录最后一个epoch的测试准确率
        last_test_acc = test_acc

        # 创建日志记录
        log_entry = {
            'experiment_id': experiment_id,
            'experiment_name': experiment_name,
            'first_activation': 'none',  # ViT第一层无激活函数
            'precision_decimal': precision_decimal,
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'test_loss': float(test_loss),
            'test_acc': float(test_acc),
            'best_acc': float(best_acc),
            'epoch_time': float(epoch_time),
            'timestamp': datetime.now().isoformat()
        }

        # 写入日志文件
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        print(f'Epoch [{epoch+1}/{num_epochs}] | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
              f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | '
              f'Time: {epoch_time:.2f}s')

    # 保存最后一个epoch的模型
    checkpoint_path = os.path.join(checkpoint_dir, f'{experiment_id}_last.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'last_acc': last_test_acc,
        'best_acc': best_acc,
        'experiment_id': experiment_id,
        'experiment_name': experiment_name,
        'first_activation': 'none',  # ViT第一层无激活函数
        'precision_decimal': precision_decimal,
    }, checkpoint_path)

    print(f"\n实验完成! 最后epoch测试准确率: {last_test_acc:.2f}% | 最佳测试准确率: {best_acc:.2f}%")
    print(f"最后模型已保存到: {checkpoint_path}")

    return last_test_acc


# 主函数
def main():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 创建日志目录和checkpoint目录
    log_dir = './log'
    checkpoint_dir = '/data0/users/wct/projects/MABIPFE/checkpoints/vit'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 创建带时间戳的日志文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'vit_experiment_{timestamp}.jsonl')

    print(f"日志文件: {log_file}")
    print(f"模型保存目录: {checkpoint_dir}")

    num_epochs = 50  # ViT通常需要更多epochs

    # 定义3组对照实验（只控制精度，无激活函数变化）
    experiments = [
        {
            'id': 'exp1',
            'name': '实验1: 原始无精度丢失',
            'precision_decimal': None
        },
        {
            'id': 'exp2',
            'name': '实验2: 保留3位小数',
            'precision_decimal': 3
        },
        {
            'id': 'exp3',
            'name': '实验3: 保留6位小数',
            'precision_decimal': 6
        }
    ]

    best_accs = []

    # 运行所有实验
    for exp in experiments:
        best_acc = run_experiment(
            exp['name'],
            exp['id'],
            exp['precision_decimal'],
            log_file,
            checkpoint_dir,
            num_epochs=num_epochs
        )
        best_accs.append((exp['name'], best_acc))

    # 打印总结
    print("\n" + "="*60)
    print("实验总结")
    print("="*60)
    for name, best_acc in best_accs:
        print(f"{name}: 最佳准确率 = {best_acc:.2f}%")
    print("="*60)
    print(f"\n实验数据已保存到: {log_file}")


if __name__ == '__main__':
    main()
