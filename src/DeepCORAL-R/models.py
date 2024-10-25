import torch
import torch.nn as nn
from torch.autograd import Function, Variable

CUDA = True if torch.cuda.is_available() else False


'''
MODELS
'''


def CORAL(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss #/(4*d*d)

    return loss


# def CORAL(source, target):

#     d = source.size(1)  # dim vector

#     source_c = compute_covariance(source)
#     target_c = compute_covariance(target)

#     loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

#     loss = loss / (4 * d * d)
#     return loss


# def compute_covariance(input_data):
#     """
#     Compute Covariance matrix of the input data
#     """
#     n = input_data.size(0)  # batch_size

#     # Check if using gpu or cpu
#     if input_data.is_cuda:
#         device = torch.device('cuda')
#     else:
#         device = torch.device('cpu')

#     id_row = torch.ones(n).resize(1, n).to(device=device)
#     sum_column = torch.mm(id_row, input_data)
#     mean_column = torch.div(sum_column, n)
#     term_mul_2 = torch.mm(mean_column.t(), mean_column)
#     d_t_d = torch.mm(input_data.t(), input_data)
#     c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)

#     return c


class DeepCORAL_1D(nn.Module):
    def __init__(self, backbone, substance):
        super(DeepCORAL_1D, self).__init__()

        if backbone == 'resnet101':
            self.feature_extractor = ResNet1D_101()
            self._input_dim = 1024
            self.sharedNet = ResNet1D_101()

        self.regression_predictor = RegressionBranch(input_dim=self._input_dim, substance=substance)

    def forward(self, source, target):
        source_feature = self.sharedNet(source)
        source_out = self.regression_predictor(source_feature)

        target_feature = self.sharedNet(target)
        target_out = self.regression_predictor(target_feature)
        
        return source_feature, target_feature, source_out, target_out


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x



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
        # self.fc = nn.Linear(512, num_classes)

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
    expansion = 2

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
    


def ResNet1D_101():
    return ResNet1D(Bottleneck1D, [3, 4, 23, 3])
