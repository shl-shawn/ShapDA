import torch.nn as nn
from torch.autograd import Function
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# Define the ResNet1D model
class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes)
            )

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet1D(nn.Module):
    def __init__(self, block, layers, num_classes=512):
        super(ResNet1D, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension: (batch_size, 1, 3579)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    

class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * self.expansion)
            )

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


def ResNet1D_18():
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2])

def ResNet1D_50():
    return ResNet1D(Bottleneck1D, [3, 4, 6, 3])

def ResNet1D_101():
    return ResNet1D(Bottleneck1D, [3, 4, 23, 3])

def ResNet1D_152():
    return ResNet1D(Bottleneck1D, [3, 8, 36, 3])



class Transformer1D(nn.Module):
    def __init__(self, input_dim=3579, d_model=512, nhead=8, num_encoder_layers=2, dim_feedforward=2048, dropout=0.5):
        super(Transformer1D, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.positional_encoding = nn.Parameter(torch.zeros(1, input_dim, d_model))
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = x.unsqueeze(2)  # Add a channel dimension: (batch_size, input_dim, 1)
        x = x.expand(-1, -1, self.d_model)  # Expand last dimension to d_model: (batch_size, input_dim, d_model)
        x = x + self.positional_encoding  # Add positional encoding
        x = self.transformer_encoder(x)  # Apply transformer encoder
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)  # Final fully connected layer
        return x


def ResNet1D_18():
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2])

def ResNet1D_34():
    return ResNet1D(BasicBlock1D, [3, 4, 6, 3])

def ResNet1D_50():
    return ResNet1D(Bottleneck1D, [3, 4, 6, 3])

def ResNet1D_101():
    return ResNet1D(Bottleneck1D, [3, 4, 23, 3])

def ResNet1D_152():
    return ResNet1D(Bottleneck1D, [3, 8, 36, 3])




class RegressionBranch(nn.Module):
    def __init__(self, input_dim, substance):
        super(RegressionBranch, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 1)  # Output a single value for regression
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Dropout layer with 10% drop probability
        
        self.substance = substance

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')


    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Adding dropout to avoid overfitting
        x = self.fc2(x)  # No activation function for the last layer as it is regression
        
        if self.substance == 'glucose':
            x = torch.clamp(x, min=0, max=150)  # Clamp output to range [0, 150] for glucose
        else:
            x = torch.clamp(x, min=0, max=100)  # Clamp output to range [0, 100] for lacticide

        return x
    

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def forward(self, x, alpha):
        return GradientReversalFunction.apply(x, alpha)


class Discriminator(nn.Module):
    def __init__(self, input_dim=512):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Output for two domains: source and target
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Dropout layer with 10% drop probability
        self.grl = GradientReversalLayer()

        # Initialize weights
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')


    def forward(self, x, alpha):
        x = self.grl(x, alpha)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class DANN(nn.Module):
    def __init__(self, backbone, substance):
        super(DANN, self).__init__()

        if backbone == 'resnet18':
            self.feature_extractor = ResNet1D_18()
            self._input_dim = 512
        elif backbone == 'resnet34':
            self.feature_extractor = ResNet1D_34()
            self._input_dim = 512
        elif backbone == 'resnet50':
            self.feature_extractor = ResNet1D_50()
            self._input_dim = 2048
        elif backbone == 'resnet101':
            self.feature_extractor = ResNet1D_101()
            self._input_dim = 1024
        elif backbone == 'resnet152':
            self.feature_extractor = ResNet1D_152()
            self._input_dim = 2048
        elif backbone == 'transformer':
            self.feature_extractor = Transformer1D()
            self._input_dim = 512
        else:
            raise AssertionError(f'{backbone} is not supported')

        self.regression_predictor = RegressionBranch(input_dim=self._input_dim, substance=substance)
        self.domain_classifier = Discriminator(input_dim=self._input_dim)
        self.alpha_domain = nn.Parameter(torch.tensor(5.0), requires_grad=True)

    def forward(self, x, alpha=1.0, task='regression'):
        features = self.feature_extractor(x)
        if task == 'regression':
            output = self.regression_predictor(features)
        elif task == 'domain':
            output = self.domain_classifier(features, alpha)
        return output