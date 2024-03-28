import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class EEGNet(nn.Module):
    def __init__(self, n_classes, channels, sampling_rate, dropout_rate, kernLength=64, F1=8, D=2, F2=None):
        super(EEGNet, self).__init__()
        if F2 is None:
            F2 = F1*D
        self.firstConv = nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.depthwiseConv = nn.Conv2d(F1, F2, (channels, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.activation = nn.ELU()
        self.avgPool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(p=dropout_rate)

        self.separableConv = nn.Conv2d(F2, F2, (1, 16), padding=(0, 16 // 2), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.avgPool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(p=dropout_rate)

        self.flatten = nn.Flatten()
        self.dense = nn.Linear(F2 * (sampling_rate // 32), n_classes)

    def forward(self, x):
        x = self.firstConv(x)
        x = self.batchnorm1(x)

        x = self.depthwiseConv(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.avgPool1(x)
        x = self.dropout1(x)

        x = self.separableConv(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.avgPool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.dense(x)
        return x

class TCNBlock(nn.Module):
    def __init__(self, input_dimension, depth, kernel_size, filters, dropout, activation='relu'):
        super(TCNBlock, self).__init__()
        self.depth = depth
        self.dropout = dropout
        self.filters = filters
        self.kernel_size = kernel_size

        self.initial_conv = nn.Conv1d(input_dimension, filters, kernel_size=1, padding='same')

        # Initial block
        self.conv1 = nn.Conv1d(filters, filters, kernel_size, padding=(kernel_size-1), dilation=1)
        self.bn1 = nn.BatchNorm1d(filters)
        self.conv2 = nn.Conv1d(filters, filters, kernel_size, padding=(kernel_size-1), dilation=1)
        self.bn2 = nn.BatchNorm1d(filters)

        # Subsequent blocks with increasing dilation
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(filters, filters, kernel_size, padding=(kernel_size-1) * 2 ** (i + 1), dilation=2 ** (i + 1)),
                nn.BatchNorm1d(filters),
                getattr(nn, activation.capitalize())(),
                nn.Dropout(dropout),
                nn.Conv1d(filters, filters, kernel_size, padding=(kernel_size-1) * 2 ** (i + 1), dilation=2 ** (i + 1)),
                nn.BatchNorm1d(filters),
            )
            for i in range(depth-1)
        ])

        self.activation = getattr(nn, activation.capitalize())()

    def forward(self, x):
        # Match dimensions if necessary
        if x.size(1) != self.filters:
            x = self.initial_conv(x)

        # Initial block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = F.dropout(out, self.dropout, training=self.training)
        res = x + out  # Residual connection

        # Subsequent blocks
        for block in self.blocks:
            out = block(res)
            out = self.activation(out)
            out = F.dropout(out, self.dropout, training=self.training)
            res = res + out  # Residual connection

        return self.activation(res)

class EEGTCNet(nn.Module):
    def __init__(self, n_classes, channels, sampling_rate, layers=2, kernel_s=4, filt=12, dropout=0.3, activation='elu', F1=8, D=2, kernLength=32, dropout_rate=0.2):
        super(EEGTCNet, self).__init__()
        # Assuming EEGNet and TCN_block are defined elsewhere with PyTorch
        self.permute = nn.Permute(2, 1, 0)  # Adjust dimension order
        self.EEGNet_sep = EEGNet(n_classes=n_classes, F1=F1, sampling_rate=sampling_rate, channels=channels, kernLength=kernLength, D=D, dropout_rate=dropout_rate)
        self.TCN_block = TCNBlock(input_dimension=F1*D, depth=layers, kernel_size=kernel_s, filters=filt, dropout=dropout, activation=activation)
        self.dense = nn.Linear(filt, n_classes)  # Assuming the output of TCN_block has 'filt' features
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.permute(x)
        x = self.EEGNet_sep(x)
        x = x[:, :, -1, :]  # Selecting the last feature map as done in Lambda(lambda x: x[:,:,-1,:])(EEGNet_sep)
        x = self.TCN_block(x)
        x = x[:, -1, :]  # Selecting the last timestep as done in Lambda(lambda x: x[:,-1,:])(outs)
        x = self.dense(x)
        x = self.softmax(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, channel, reduction_ratio):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MBEEG_SENet(nn.Module):
    def __init__(self, nb_classes, channels, sampling_rate, D=2):
        super(MBEEG_SENet, self).__init__()
        self.EEGNet_sep1 = EEGNet(n_classes=nb_classes, sampling_rate=sampling_rate, F1=4, kernLength=16, D=D, channels=channels, dropout_rate=0)
        self.EEGNet_sep2 = EEGNet(n_classes=nb_classes, sampling_rate=sampling_rate, F1=8, kernLength=32, D=D, channels=channels, dropout_rate=0.1)
        self.EEGNet_sep3 = EEGNet(n_classes=nb_classes, sampling_rate=sampling_rate, F1=16, kernLength=64, D=D, channels=channels, dropout_rate=0.2)

        self.SE1 = SEBlock(channel=channels, reduction_ratio=4)  # Assuming channel count matches here
        self.SE2 = SEBlock(channel=channels, reduction_ratio=4)
        self.SE3 = SEBlock(channel=channels, reduction_ratio=2)

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(channels * 3, nb_classes)  # Adjust the size according to your SEBlock output
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Assuming x is of shape (batch_size, 1, channels, sampling_rate)
        branch1 = self.EEGNet_sep1(x)
        branch2 = self.EEGNet_sep2(x)
        branch3 = self.EEGNet_sep3(x)

        branch1 = self.SE1(branch1)
        branch2 = self.SE2(branch2)
        branch3 = self.SE3(branch3)

        branch1 = self.flatten(branch1)
        branch2 = self.flatten(branch2)
        branch3 = self.flatten(branch3)

        concatenated = torch.cat((branch1, branch2, branch3), dim=1)  # Concatenate along the feature dimension

        out = self.dense1(concatenated)
        out = self.softmax(out)
        return out

class EEGNeX_8_32(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs):
        super(EEGNeX_8_32, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, (1, 32), padding='same', bias=False)
        self.ln1 = nn.LayerNorm([8, n_features, n_timesteps])
        self.conv2 = nn.Conv2d(8, 32, (1, 32), padding='same', bias=False)
        self.ln2 = nn.LayerNorm([32, n_features, n_timesteps])

        self.depthwiseConv = nn.Conv2d(32, 64, (n_features, 1), groups=32, bias=False)
        self.ln3 = nn.LayerNorm([64, 1, n_timesteps])
        self.pooling1 = nn.AvgPool2d((1, 4), padding='same')
        self.dropout1 = nn.Dropout(0.5)

        self.conv3 = nn.Conv2d(64, 32, (1, 16), padding='same', bias=False, dilation=(1, 2))
        self.ln4 = nn.LayerNorm([32, 1, n_timesteps//4])
        self.conv4 = nn.Conv2d(32, 8, (1, 16), padding='same', bias=False, dilation=(1, 4))
        self.ln5 = nn.LayerNorm([8, 1, n_timesteps//4])
        self.dropout2 = nn.Dropout(0.5)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * n_timesteps // 4, n_outputs)

    def forward(self, x):
        x = F.elu(self.ln1(self.conv1(x)))
        x = F.elu(self.ln2(self.conv2(x)))

        x = F.elu(self.ln3(self.depthwiseConv(x)))
        x = self.dropout1(self.pooling1(x))

        x = F.elu(self.ln4(self.conv3(x)))
        x = F.elu(self.ln5(self.conv4(x)))
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class EEGNetClassifier(nn.Module):
    def __init__(self, n_classes, channels, sampling_rate, dropout_rate, F1=8, D=2, kernLength=64):
        super(EEGNetClassifier, self).__init__()
        self.eegnet = EEGNet(n_classes=n_classes, channels=channels, sampling_rate=sampling_rate, dropout_rate=dropout_rate, kernLength=kernLength, F1=F1, D=D)
        # Assuming EEGNet's final layer outputs a tensor with shape (batch_size, features)
        # where `features` depends on EEGNet's architecture and the input dimensions.
        self.flatten = nn.Flatten()
        # The number of features output by EEGNet needs to be known to define the following layer.
        # For this example, let's assume `eegnet_features` is the calculated or known value.
        eegnet_features = self._get_eegnet_output_features(channels, sampling_rate, F1, D, kernLength)
        self.dense = nn.Linear(eegnet_features, n_classes)
        self.softmax = nn.Softmax(dim=1)  # Use softmax in the inference phase only.

    def _get_eegnet_output_features(self, channels, sampling_rate, F1, D, kernLength):
        # This method should return the number of output features from EEGNet before the classifier layer.
        # The calculation depends on the architecture of EEGNet.
        # Placeholder value:
        return 100  # This should be replaced with the correct calculation.

    def forward(self, x):
        x = self.eegnet(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x  # Return logits for use with nn.CrossEntropyLoss during training


class DeepConvNet(nn.Module):
    def __init__(self, nb_classes, channels, sampling_rate, dropout_rate):
        super(DeepConvNet, self).__init__()
        self.block1_conv1 = nn.Conv2d(1, 25, (1, 10), padding='same')
        self.block1_conv2 = nn.Conv2d(25, 25, (channels, 1))
        self.block1_bn = nn.BatchNorm2d(25)
        self.block1_pool = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.block1_dropout = nn.Dropout(dropout_rate)

        self.block2_conv = nn.Conv2d(25, 50, (1, 10), padding='same')
        self.block2_bn = nn.BatchNorm2d(50)
        self.block2_pool = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.block2_dropout = nn.Dropout(dropout_rate)

        self.block3_conv = nn.Conv2d(50, 100, (1, 10), padding='same')
        self.block3_bn = nn.BatchNorm2d(100)
        self.block3_pool = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.block3_dropout = nn.Dropout(dropout_rate)

        self.block4_conv = nn.Conv2d(100, 200, (1, 10), padding='same')
        self.block4_bn = nn.BatchNorm2d(200)
        self.block4_pool = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.block4_dropout = nn.Dropout(dropout_rate)

        self.flatten = nn.Flatten()
        # Calculate the number of features after flattening
        out_features = self._get_conv_output((1, channels, sampling_rate))
        self.dense = nn.Linear(out_features, nb_classes)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.rand(1, *shape)
            output = self._forward_features(input)
            return int(torch.numel(output) / output.shape[0])

    def _forward_features(self, x):
        x = self.block1_conv1(x)
        x = self.block1_conv2(x)
        x = F.elu(self.block1_bn(x))
        x = self.block1_pool(x)
        x = self.block1_dropout(x)

        x = self.block2_conv(x)
        x = F.elu(self.block2_bn(x))
        x = self.block2_pool(x)
        x = self.block2_dropout(x)

        x = self.block3_conv(x)
        x = F.elu(self.block3_bn(x))
        x = self.block3_pool(x)
        x = self.block3_dropout(x)

        x = self.block4_conv(x)
        x = F.elu(self.block4_bn(x))
        x = self.block4_pool(x)
        x = self.block4_dropout(x)

        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.flatten(x)
        x = self.dense(x)
        return F.log_softmax(x, dim=1)


class SquareActivation(nn.Module):
    def forward(self, x):
        return torch.pow(x, 2)

class LogActivation(nn.Module):
    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-7))  # Clamp values to avoid log(0)

class ShallowConvNet(nn.Module):
    def __init__(self, nb_classes, channels, sampling_rate, droupout_rate=0.5):
        super(ShallowConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 40, (1, 25), padding='same', bias=False)
        # Note: Conv2d's weight norm can be applied via torch.nn.utils.weight_norm or manually in forward
        self.conv2 = nn.Conv2d(40, 40, (channels, 1), bias=False)
        self.batchnorm = nn.BatchNorm2d(40)
        self.square = SquareActivation()
        self.avgpool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.log = LogActivation()
        self.dropout = nn.Dropout(droupout_rate)
        self.flatten = nn.Flatten()
        # Determining the number of flattened features for the dense layer
        self.dense = nn.Linear(self._get_dense_in_features(sampling_rate), nb_classes)

    def _get_dense_in_features(self, sampling_rate):
        # Calculate the size here based on the network architecture and input dimensions
        # Placeholder calculation; replace with actual size
        return 40 * ((sampling_rate - 75) // 15 + 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.square(x)
        x = self.avgpool(x)
        x = self.log(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = F.softmax(x, dim=1)
        return x
