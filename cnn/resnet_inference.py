import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from datetime import datetime
import json


# 自定义激活函数：y^2
class SquareActivation(nn.Module):
    def __init__(self):
        super(SquareActivation, self).__init__()

    def forward(self, x):
        return x ** 2


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


# ResNet基本块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu2(out)

        return out


# ResNet模型
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,
                 first_activation='relu', precision_decimal=None):
        super(ResNet, self).__init__()

        self.in_channels = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 第一层卷积的激活函数（可自定义）
        if first_activation == 'relu':
            self.activation1 = nn.ReLU(inplace=True)
        elif first_activation == 'square':
            self.activation1 = SquareActivation()
        else:
            self.activation1 = nn.ReLU(inplace=True)

        # 第一层卷积的精度控制
        self.precision_control = PrecisionControl(precision_decimal) if precision_decimal is not None else None

        # ResNet层
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        # 第一个block可能需要下采样
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion

        # 后续blocks
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 第一层卷积 + BN + 自定义激活函数 + 精度控制
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)

        # 应用精度控制（如果有）
        if self.precision_control is not None:
            x = self.precision_control(x)

        # 后续ResNet层（都使用标准ReLU）
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def ResNet18(first_activation='relu', precision_decimal=None, num_classes=100):
    return ResNet(BasicBlock, [2, 2, 2, 2],
                  num_classes=num_classes,
                  first_activation=first_activation,
                  precision_decimal=precision_decimal)


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
def load_model_checkpoint(checkpoint_path, first_activation, precision_decimal, device):
    """
    加载训练好的模型权重
    """
    # 创建模型
    model = ResNet18(
        first_activation=first_activation,
        precision_decimal=precision_decimal,
        num_classes=100
    ).to(device)

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"已加载模型: {checkpoint_path}")
    print(f"  训练时配置: 激活函数={checkpoint.get('first_activation')}, "
          f"精度={checkpoint.get('precision_decimal')}")
    print(f"  训练最后epoch准确率: {checkpoint.get('last_acc', 'N/A'):.2f}%")

    return model, checkpoint


# 运行推理测试
def run_inference_test(test_name, checkpoint_path, inference_activation,
                       inference_precision, test_loader, device):
    """
    运行单个推理测试
    """
    print(f"\n{'='*70}")
    print(f"测试: {test_name}")
    print(f"模型权重: {os.path.basename(checkpoint_path)}")
    print(f"推理配置: 激活函数={inference_activation}, 精度={inference_precision}")
    print(f"{'='*70}")

    # 加载模型
    model, checkpoint = load_model_checkpoint(
        checkpoint_path,
        inference_activation,
        inference_precision,
        device
    )

    # 测试
    test_acc = test_model(model, test_loader, device)

    print(f"\n测试集准确率: {test_acc:.2f}%")

    return {
        'test_name': test_name,
        'checkpoint_path': checkpoint_path,
        'inference_activation': inference_activation,
        'inference_precision': inference_precision,
        'training_activation': checkpoint.get('first_activation'),
        'training_precision': checkpoint.get('precision_decimal'),
        'training_last_acc': float(checkpoint.get('last_acc', 0)),
        'inference_acc': float(test_acc),
        'timestamp': datetime.now().isoformat()
    }


def main():
    print("="*70)
    print("ResNet18 模型推理测试程序")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 数据集路径
    data_root = os.path.expanduser('~/projects/MABIPFE/dataset')
    checkpoint_dir = '/data0/users/wct/projects/MABIPFE/checkpoints/resnet'

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
            'activation': 'relu',
            'precision': None,
            'description': '使用实验1(原始无精度丢失+ReLU)训练的模型，以原始精度进行推理'
        },
        {
            'name': '测试2: 实验1模型 - 第一层卷积保留6位小数推理',
            'checkpoint': os.path.join(checkpoint_dir, 'exp1_last.pth'),
            'activation': 'relu',
            'precision': 6,
            'description': '使用实验1训练的模型，在第一层卷积后保留6位小数进行推理'
        },
        {
            'name': '测试3: 实验2模型 - 保留3位小数+y^2激活推理',
            'checkpoint': os.path.join(checkpoint_dir, 'exp2_last.pth'),
            'activation': 'square',
            'precision': 3,
            'description': '使用实验2(保留3位小数+y^2激活)训练的模型，保持训练配置推理'
        },
        {
            'name': '测试4: 实验3模型 - 保留6位小数+ReLU推理',
            'checkpoint': os.path.join(checkpoint_dir, 'exp3_last.pth'),
            'activation': 'relu',
            'precision': 6,
            'description': '使用实验3(保留6位小数+ReLU)训练的模型，保持训练配置推理'
        }
    ]

    # 运行所有推理测试
    results = []
    for test_config in inference_tests:
        result = run_inference_test(
            test_config['name'],
            test_config['checkpoint'],
            test_config['activation'],
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
        print(f"  训练配置: 激活={result['training_activation']}, 精度={result['training_precision']}")
        print(f"  推理配置: 激活={result['inference_activation']}, 精度={result['inference_precision']}")
        print(f"  训练最后epoch准确率: {result['training_last_acc']:.2f}%")
        print(f"  推理准确率: {result['inference_acc']:.2f}%")

        # 计算准确率差异
        acc_diff = result['inference_acc'] - result['training_last_acc']
        print(f"  准确率差异: {acc_diff:+.2f}%")

    # 保存结果到JSON文件
    result_dir = './log'
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(result_dir, f'resnet_inference_{timestamp}.json')

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n推理结果已保存到: {result_file}")
    print("="*70)


if __name__ == '__main__':
    main()
