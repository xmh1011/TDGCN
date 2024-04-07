import math
import os
import torch
import config
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

_, os.environ['CUDA_VISIBLE_DEVICES'] = config.set_config()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class PowerLayer(nn.Module):
    """
    The power layer: calculates the log-transformed power of the data
    """

    def __init__(self, dim, length, step):
        super(PowerLayer, self).__init__()
        self.dim = dim
        self.pooling = nn.AvgPool2d(kernel_size=(1, length), stride=(1, step))

    def forward(self, x):
        return torch.log(self.pooling(x.pow(2)))


class TDGCN(nn.Module):
    """
    提供的代码实现了一个动态图卷积神经网络（DGCNN）的架构，其中包含以下几个关键组件和概念：

    1. **时间学习层（Temporal Learners）**：通过`temporal_learner`函数实现，它使用一维卷积层（`nn.Conv2d`）和自定义的`PowerLayer`来处理时间序列数据。这种结构适用于从时间序列中提取特征，如EEG信号。

    2. **动态图卷积层（Dynamic Graph Convolution Layers）**：通过`DynamicGraphConvolution`类实现，它扩展了基本的图卷积层`GraphConvolution`，并通过计算特征的相似度动态地更新邻接矩阵。这使得网络能够根据当前数据的特点调整图结构。

    3. **局部过滤（Local Filtering）**：使用局部过滤器权重和偏置（`self.local_filter_weight`和`self.local_filter_bias`）对提取的特征进行处理。这可以被看作是对特征进行加权和调整，以增强网络的表示能力。

    4. **聚合功能（Aggregator）**：`Aggregator`类根据提供的脑区索引（`self.idx`）将特征聚合到不同的脑区。这是处理脑信号数据时，按脑区组织和整合特征的一种方式。

    5. **全局邻接矩阵（Global Adjacency Matrix）**：`self.global_adj`作为模型参数参与训练，它与动态计算的相似度矩阵结合，共同决定了图的结构。这种设计使得模型能够学习到数据中的全局连接模式。

    ### 具体实现原理
    - **时间学习层**捕获了数据中的时间依赖性特征。
    - **动态图卷积层**根据数据的实时特性动态调整图结构，使网络能够适应数据中的变化模式。
    - **局部过滤和聚合**则是对特征进行空间上的整合和增强，使模型能够更好地理解数据中的空间信息。
    - **全局邻接矩阵**提供了一种学习数据中固有连接模式的机制。

    ### 实现注意点
    - **维度匹配**：在执行矩阵乘法和特征转换时，确保各个张量的维度匹配是关键，这通常涉及到对张量形状的调整。
    - **动态邻接矩阵的计算**：通过特征相似度动态更新邻接矩阵是DGCNN的核心，它要求模型能够基于当前的输入特征调整图结构。
    - **训练稳定性**：动态更新图结构可能会导致训练过程中的不稳定性，因此可能需要细心地调整学习率、正则化等超参数。

    整体而言，这个DGCNN实现框架通过结合时间序列处理、动态图更新和空间特征整合的方式，为处理如EEG这类复杂数据提供了一种有效的方法。然而，要充分发挥这个模型的潜力，可能需要根据具体的数据和任务进行一些调整和优化。
    """

    def temporal_learner(self, in_chan, out_chan, kernel, pool, pool_step_rate):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=(1, 1)),
            PowerLayer(dim=-1, length=pool, step=int(pool_step_rate * pool))
        )

    def __init__(self, num_classes, input_size, sampling_rate, num_T,
                 out_graph, dropout_rate, pool, pool_step_rate, idx_graph):
        super(TDGCN, self).__init__()

        self.num_T = num_T
        self.out_graph = out_graph
        self.dropout_rate = dropout_rate
        self.window = [0.5, 0.25, 0.125, 0.0625]
        self.pool = pool
        self.pool_step_rate = pool_step_rate
        self.idx = idx_graph
        self.channel = input_size[1]
        self.brain_area = len(self.idx)
        ###################
        # 多头注意力相关参数
        self.model_dim = round(num_T/2)
        self.num_heads = 8
        self.window_size = 100
        self.stride = 20
        ###################
        hidden_features = input_size[2]

        # by setting the convolutional kernel being (1,lenght) and the strids being 1, we can use conv2d to
        # achieve the 1d convolution operation.
        self.Tception1 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[0] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception2 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[1] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception3 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[2] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception4 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[3] * sampling_rate)),
                                               self.pool, pool_step_rate)
        # Batch normalization layers
        self.bn_t = nn.BatchNorm2d(num_T)
        self.bn_s = nn.BatchNorm2d(num_T)
        self.OneXOneConv = nn.Sequential(
            nn.Conv2d(num_T, num_T, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.AvgPool2d((1, 2))
        )
        #######################################
        self.feature_integrator = FeatureIntegrator(in_channels=32, out_channels=self.model_dim)
        self.sliding_window_processor = SlidingWindowProcessor(model_dim=self.model_dim, num_heads=self.num_heads,
                                                               window_size=self.window_size, stride=self.stride)
        #######################################
        # diag(W) to assign a weight to each local areas
        size = self.get_size_temporal(input_size)
        # 表示局部滤波器的权重。它被定义为一个形状为(self.channel, size[-1])的浮点型张量，并设置为需要梯度计算（requires_grad=True）
        self.local_filter_weight = nn.Parameter(torch.FloatTensor(self.channel, size[-1]),
                                                requires_grad=True)
        # 用来对local_filter_weight进行初始化，采用的是Xavier均匀分布初始化方法
        nn.init.xavier_uniform_(self.local_filter_weight)
        # 表示局部滤波器的偏置。它被定义为一个形状为(1, self.channel, 1)的浮点型张量，初始值为全零，并设置为需要梯度计算
        self.local_filter_bias = nn.Parameter(torch.zeros((1, self.channel, 1), dtype=torch.float32),
                                              requires_grad=True)
        # aggregate function
        self.aggregate = Aggregator(self.idx)

        # Dynamic Graph Convolution Layers
        # self.dynamic_gcn = DynamicGraphConvolution(size[-1], out_graph)
        self.dynamic_gcn = StackedDynamicGraphConvolution(size[-1], hidden_features, out_graph, num_layers=3)
        # 表示全局邻接矩阵。它被定义为浮点型张量，并设置为需要梯度计算（requires_grad=True）
        self.global_adj = nn.Parameter(torch.FloatTensor(self.brain_area, self.brain_area), requires_grad=True)
        # 根据给定的张量的形状和分布进行参数初始化。用来对global_adj进行初始化，采用的是Xavier均匀分布初始化方法。
        nn.init.xavier_uniform_(self.global_adj)
        # to be used after local graph embedding
        self.bn = nn.BatchNorm1d(self.brain_area)
        self.bn_ = nn.BatchNorm1d(self.brain_area)

        # Fully connected layer for classification
        self.fc = nn.Sequential(  # 组合神经网络模块
            nn.Dropout(p=dropout_rate),
            nn.Linear(int(self.brain_area * out_graph), num_classes)
        )

    def get_size_temporal(self, input_size):
        # input_size: frequency x channel x data point
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        z = self.Tception1(data)
        out = z
        z = self.Tception2(data)
        out = torch.cat((out, z), dim=-1)
        z = self.Tception3(data)
        out = torch.cat((out, z), dim=-1)
        # z = self.Tception4(data)
        # out = torch.cat((out, z), dim=-1)
        # out = self.bn_t(out)
        # out = self.OneXOneConv(out)
        # out = self.bn_s(out)
        # out = out.permute(0, 2, 1, 3)
        #######################################
        out = self.feature_integrator(out)  # 特征整合和降维
        out = self.sliding_window_processor(out)  # 滑动窗口处理
        #######################################
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        size = out.size()
        return size

    # 定义局部滤波器的前向传播函数
    def local_filter_fun(self, x, w):
        w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)
        x = F.relu(torch.mul(x, w) - self.local_filter_bias)
        return x

    def forward(self, x):
        # Temporal convolution
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        # out = torch.cat((out, y), dim=-1)
        # y = self.Tception4(x)
        out = torch.cat((out, y), dim=-1)
        # out = self.bn_t(out)
        # out = self.OneXOneConv(out)
        # out = self.bn_s(out)
        # out = out.permute(0, 2, 1, 3)
        ##############################
        out = self.feature_integrator(out)  # 特征整合和降维
        out = self.sliding_window_processor(out)  # 滑动窗口处理
        ##############################
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        out = self.local_filter_fun(out, self.local_filter_weight)
        out = self.aggregate.forward(out)
        out = self.bn(out)
        out = self.dynamic_gcn(out)
        out = self.bn_(out)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out


class GraphConvolution(nn.Module):
    """
    Simple GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Correctly initialize weights using an empty tensor
        self.weight = Parameter(torch.empty((in_features, out_features)).to(DEVICE))
        torch.nn.init.xavier_uniform_(self.weight, gain=1.414)

        if bias:
            # Bias is correctly initialized to zero
            self.bias = Parameter(torch.zeros(out_features).to(DEVICE))
        else:
            # Properly handle the absence of bias
            self.register_parameter('bias', None)

    def forward(self, x, adj):
        # 假设 x 形状为 [batch_size, num_channels, height, width]
        batch_size, num_channels, height, width = x.size()

        # 调整 x 的形状以匹配权重维度
        x = x.view(batch_size * num_channels, -1)  # 形状变为 [batch_size*num_channels, height*width]

        # 应用权重
        support = torch.matmul(x, self.weight)  # 形状变为 [batch_size*num_channels, out_features]

        # 重塑 support 以匹配 adj 的期望形状
        support = support.view(batch_size, num_channels, -1)  # 形状变为 [batch_size, num_channels, out_features]

        # 这里假设 adj 已经是 [batch_size, num_channels, num_channels] 形状
        # 应用 adj 到 support 上
        output = torch.bmm(adj, support)  # 形状变为 [batch_size, num_channels, out_features]

        # 将 output 重塑回合适的形状以进行后续层的处理
        output = output.view(batch_size * num_channels, -1)  # 形状变为 [batch_size*num_channels, out_features]

        if self.bias is not None:
            output += self.bias

        # 可能需要进一步调整 output 的形状，以适配网络中后续层的期望输入
        return output


class Aggregator():

    def __init__(self, idx_area):
        # chan_in_area: a list of the number of channels within each area
        self.chan_in_area = idx_area
        self.idx = self.get_idx(idx_area)
        self.area = len(idx_area)

    def forward(self, x):
        # x: batch x channel x data
        data = []
        for i, area in enumerate(range(self.area)):
            if i < self.area - 1:
                data.append(self.aggr_fun(x[:, self.idx[i]:self.idx[i + 1], :], dim=1))
            else:
                data.append(self.aggr_fun(x[:, self.idx[i]:, :], dim=1))
        return torch.stack(data, dim=1)

    def get_idx(self, chan_in_area):
        idx = [0] + chan_in_area
        idx_ = [0]
        for i in idx:
            idx_.append(idx_[-1] + i)
        return idx_[1:]

    def aggr_fun(self, x, dim):
        # return torch.max(x, dim=dim).values
        return torch.mean(x, dim=dim)


class DynamicGraphConvolution(GraphConvolution):
    """
    Dynamic Graph Convolution Layer.
    Extends the GraphConvolution layer with a dynamic adjacency matrix based on feature similarity.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(DynamicGraphConvolution, self).__init__(in_features, out_features, bias)

    def forward(self, x, adj=None):
        if adj is None:
            # Compute adjacency matrix dynamically based on feature similarity, for example:
            adj = self.normalize_adjacency_matrix(x)

        output = torch.matmul(x, self.weight)
        if self.bias is not None:
            output += self.bias
        output = F.relu(torch.matmul(adj, output))
        return output

    def compute_similarity(self, x):
        # x: b, node, feature
        x_ = x.permute(0, 2, 1)
        s = torch.bmm(x, x_)
        return s

    def normalize_adjacency_matrix(self, x):
        """
        x：输入的特征矩阵，大小为(b, node, feature)，其中b为批次大小，node为节点数目，feature为每个节点的特征向量维度。
        self_loop：一个布尔值，表示是否在邻接矩阵中加入自环（自己到自己的连接）。
        """
        # x: b, node, feature
        # 利用模型中的self_similarity方法计算输入特征矩阵x的自相似度矩阵。结果为一个大小为(b, n, n)的张量，其中n为节点数目
        adj = self.compute_similarity(x)  # b, n, n
        num_nodes = adj.shape[-1]
        adj = adj + torch.eye(num_nodes).to(DEVICE)
        rowsum = torch.sum(adj, dim=-1)
        # 创建一个与rowsum大小相同的全零张量mask，并将rowsum中和为0的位置置为1。这一步是为了处理邻接矩阵中存在度为0的节点，避免除以0的错误
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        # 将mask添加到rowsum中，实现对邻接矩阵的修正。避免除以0的错误，并保证每个节点的度至少为1
        rowsum += mask
        # 计算度矩阵的逆平方根
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        # 将逆平方根得到的张量转换为对角矩阵
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        # 通过矩阵乘法和广播机制，将度矩阵的逆平方根与邻接矩阵相乘，得到归一化后的邻接矩阵
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj


class StackedDynamicGraphConvolution(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers=3, bias=True):
        super(StackedDynamicGraphConvolution, self).__init__()
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(DynamicGraphConvolution(in_features, hidden_features, bias=bias))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(DynamicGraphConvolution(hidden_features, hidden_features, bias=bias))
        # Last layer
        self.layers.append(DynamicGraphConvolution(hidden_features, out_features, bias=bias))

    def forward(self, x, adj=None):
        for layer in self.layers:
            x = layer(x, adj)
        return x


class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x)))


class FeatureIntegrator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=64, stride=64):
        super(FeatureIntegrator, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        # 假设输入x的形状为 (batch_size, feature_dim, channels, length)
        batch_size, feature_dim, channels, length = x.size()

        # 你想将feature和length维度相结合
        # 首先，将x变形为 (batch_size, channels, feature_dim * length)
        x = x.reshape(batch_size, channels, feature_dim * length)

        # 然后，应用1D卷积
        x = self.conv(x)  # 卷积后的形状为 (batch_size, out_channels, new_length)

        return x


class SlidingWindowProcessor(nn.Module):
    def __init__(self, model_dim, num_heads, window_size, stride):
        super(SlidingWindowProcessor, self).__init__()
        self.window_size = window_size
        self.stride = stride
        self.layer_norm1 = nn.LayerNorm([window_size, model_dim])
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.tcn_block = TemporalConvBlock(in_channels=model_dim, out_channels=32)
        # 定义融合层，使用1D卷积以保留32通道的结构，卷积核大小和步长可以根据需要调整
        self.fusion_conv = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        batch_size, _, length = x.shape
        # 使用列表收集所有窗口的输出
        window_outputs = []

        for window_start in range(0, length - self.window_size + 1, self.stride):
            window_end = window_start + self.window_size
            window = x[:, :, window_start:window_end]
            window = window.permute(0, 2, 1)

            window = self.layer_norm1(window)
            attn_output, _ = self.multi_head_attention(window, window, window)
            attn_output = self.layer_norm2(attn_output + window)
            tcn_input = attn_output.permute(0, 2, 1)
            tcn_output = self.tcn_block(tcn_input)

            window_outputs.append(tcn_output)

        # 将所有窗口的输出沿着时间维度堆叠起来，形成一个新的维度
        stacked_outputs = torch.stack(window_outputs, dim=2)
        # 重新排列维度以匹配卷积层的输入要求
        stacked_outputs = stacked_outputs.permute(0, 3, 1, 2).reshape(batch_size, 32, -1)
        # 通过融合层整合所有窗口的输出
        fused_output = self.fusion_conv(stacked_outputs)

        return fused_output


# class SlidingWindowProcessor(nn.Module):
#     def __init__(self, model_dim, num_heads, window_size, stride):
#         super(SlidingWindowProcessor, self).__init__()
#         self.window_size = window_size
#         self.stride = stride
#         self.layer_norm1 = nn.LayerNorm([window_size, model_dim])
#         self.multi_head_attention = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, batch_first=True)
#         self.layer_norm2 = nn.LayerNorm(model_dim)
#         self.tcn_block = TemporalConvBlock(in_channels=model_dim, out_channels=model_dim)
#
#     def forward(self, x):
#         outputs = []
#
#         for window_start in range(0, x.size(2) - self.window_size + 1, self.stride):
#             window_end = window_start + self.window_size
#             window = x[:, :, window_start:window_end]  # 提取窗口
#             window = window.permute(0, 2, 1)  # 调整形状以匹配多头注意力的输入
#
#             window = self.layer_norm1(window)
#             attn_output, _ = self.multi_head_attention(window, window, window)
#             attn_output = self.layer_norm2(attn_output + window)  # 残差连接
#             tcn_input = attn_output.permute(0, 2, 1)
#             tcn_output = self.tcn_block(tcn_input)
#             outputs.append(tcn_output.permute(0, 2, 1))
#
#         output_tensor = torch.stack(outputs, dim=1)
#         return output_tensor
