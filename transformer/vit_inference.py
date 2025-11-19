import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from datetime import datetime
import json


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


# 测试函数
def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = 100. * correct / total
    return test_acc


# 加载模型权重
def load_model_checkpoint(checkpoint_path, inference_precision, device):
    """
    加载训练好的模型权重
    """
    # 创建模型（使用推理时的精度配置）
    model = ViT_Small(
        precision_decimal=inference_precision,
        num_classes=100
    ).to(device)

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"已加载模型: {checkpoint_path}")
    print(f"  训练时配置: 精度={checkpoint.get('precision_decimal')}")
    print(f"  训练最后epoch准确率: {checkpoint.get('last_acc', 'N/A'):.2f}%")

    return model, checkpoint


# 运行推理测试
def run_inference_test(test_name, checkpoint_path, inference_precision,
                       test_loader, device):
    """
    运行单个推理测试
    """
    print(f"\n{'='*70}")
    print(f"测试: {test_name}")
    print(f"模型权重: {os.path.basename(checkpoint_path)}")
    print(f"推理配置: 精度={inference_precision}")
    print(f"{'='*70}")

    # 加载模型
    model, checkpoint = load_model_checkpoint(
        checkpoint_path,
        inference_precision,
        device
    )

    # 测试
    test_acc = test_model(model, test_loader, device)

    print(f"\n测试集准确率: {test_acc:.2f}%")

    return {
        'test_name': test_name,
        'checkpoint_path': checkpoint_path,
        'inference_precision': inference_precision,
        'training_precision': checkpoint.get('precision_decimal'),
        'training_last_acc': float(checkpoint.get('last_acc', 0)),
        'inference_acc': float(test_acc),
        'timestamp': datetime.now().isoformat()
    }


def main():
    print("="*70)
    print("ViT 模型推理测试程序")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 数据集路径
    data_root = os.path.expanduser('~/projects/MABIPFE/dataset')
    checkpoint_dir = '/data0/users/wct/projects/MABIPFE/checkpoints/vit'

    # 数据预处理 (CIFAR-100)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # 加载CIFAR-100测试集
    print("\n正在加载CIFAR-100测试集...")
    testset = torchvision.datasets.CIFAR100(
        root=data_root,
        train=False,
        download=False,
        transform=transform_test
    )
    testloader = DataLoader(
        testset,
        batch_size=256,
        shuffle=False,
        num_workers=32
    )
    print("测试集加载完成\n")

    # 定义推理测试配置
    inference_tests = [
        {
            'name': '测试1: 实验1模型 - 原始精度推理',
            'checkpoint': os.path.join(checkpoint_dir, 'exp1_last.pth'),
            'precision': None,
            'description': '使用实验1(原始无精度丢失)训练的模型，以原始精度进行推理'
        },
        {
            'name': '测试2: 实验1模型 - 第一层保留6位小数推理',
            'checkpoint': os.path.join(checkpoint_dir, 'exp1_last.pth'),
            'precision': 6,
            'description': '使用实验1训练的模型，在第一层后保留6位小数进行推理'
        },
        {
            'name': '测试3: 实验2模型 - 保留3位小数推理',
            'checkpoint': os.path.join(checkpoint_dir, 'exp2_last.pth'),
            'precision': 3,
            'description': '使用实验2(保留3位小数)训练的模型，保持训练配置推理'
        },
        {
            'name': '测试4: 实验3模型 - 保留6位小数推理',
            'checkpoint': os.path.join(checkpoint_dir, 'exp3_last.pth'),
            'precision': 6,
            'description': '使用实验3(保留6位小数)训练的模型，保持训练配置推理'
        }
    ]

    # 运行所有推理测试
    results = []
    for test_config in inference_tests:
        result = run_inference_test(
            test_config['name'],
            test_config['checkpoint'],
            test_config['precision'],
            testloader,
            device
        )
        results.append(result)

    # 打印总结
    print("\n" + "="*70)
    print("推理测试总结")
    print("="*70)
    for result in results:
        print(f"\n{result['test_name']}")
        print(f"  训练配置: 精度={result['training_precision']}")
        print(f"  推理配置: 精度={result['inference_precision']}")
        print(f"  训练最后epoch准确率: {result['training_last_acc']:.2f}%")
        print(f"  推理准确率: {result['inference_acc']:.2f}%")

        # 计算准确率差异
        acc_diff = result['inference_acc'] - result['training_last_acc']
        print(f"  准确率差异: {acc_diff:+.2f}%")

    # 保存结果到JSON文件
    result_dir = './log'
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(result_dir, f'vit_inference_{timestamp}.json')

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n推理结果已保存到: {result_file}")
    print("="*70)


if __name__ == '__main__':
    main()
