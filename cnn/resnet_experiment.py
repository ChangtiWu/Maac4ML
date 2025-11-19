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
def run_experiment(experiment_name, experiment_id, first_activation, precision_decimal,
                   log_file, checkpoint_dir, num_epochs=20, batch_size=256, learning_rate=0.01):
    print(f"\n{'='*60}")
    print(f"开始实验: {experiment_name}")
    print(f"激活函数: {first_activation}, 精度: {precision_decimal if precision_decimal else '无限制'}")
    print(f"{'='*60}\n")

    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
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
    model = ResNet18(first_activation=first_activation,
                    precision_decimal=precision_decimal,
                    num_classes=100).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                         momentum=0.9, weight_decay=5e-4)
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
            'first_activation': first_activation,
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
        'first_activation': first_activation,
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
    checkpoint_dir = '/data0/users/wct/projects/MABIPFE/checkpoints/resnet'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 创建带时间戳的日志文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'resnet_experiment_{timestamp}.jsonl')

    print(f"日志文件: {log_file}")
    print(f"模型保存目录: {checkpoint_dir}")

    num_epochs = 20  # 可以调整训练轮数

    # 定义3组对照实验
    experiments = [
        {
            'id': 'exp1',
            'name': '实验1: 原始无精度丢失 + ReLU',
            'first_activation': 'relu',
            'precision_decimal': None
        },
        {
            'id': 'exp2',
            'name': '实验2: 保留3位小数 + y^2激活',
            'first_activation': 'square',
            'precision_decimal': 3
        },
        {
            'id': 'exp3',
            'name': '实验3: 保留6位小数 + ReLU',
            'first_activation': 'relu',
            'precision_decimal': 6
        }
    ]

    best_accs = []

    # 运行所有实验
    for exp in experiments:
        best_acc = run_experiment(
            exp['name'],
            exp['id'],
            exp['first_activation'],
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
