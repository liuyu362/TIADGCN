from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DyT(nn.Module):
    def __init__(self,c,init_a=0.5):
        super(DyT, self).__init__()
        self.a = nn.Parameter(torch.ones(1)*init_a)
        self.r = nn.Parameter(torch.ones(c))
        self.b = nn.Parameter(torch.zeros(c))
    def forward(self, x):
        #x b c n t
        x = x.squeeze(3)
        x = x.permute(0,2,1)

        x  = torch.tanh(self.a*x)
        x = self.r*x + self.b

        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        return x

class GLU(nn.Module):
    def __init__(self, features,  dropout=0.1):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv2d(features, features, (1, 1))
        self.conv2 = nn.Conv2d(features, features, (1, 1))
        self.conv3 = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        return out



class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()

        self.time = time
        # temporal embeddings
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        #x:b c n t
        day_emb = x[..., 1]
        time_day = self.time_day[
            (day_emb[:, -1, :] * self.time).type(torch.LongTensor)
        ]
        time_day = time_day.transpose(1, 2).unsqueeze(-1)
        week_emb = x[..., 2]
        time_week = self.time_week[
            (week_emb[:, -1, :]).type(torch.LongTensor)
        ]
        time_week = time_week.transpose(1, 2).unsqueeze(-1)

        tem_emb = time_day + time_week
        return tem_emb


class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, :, :, ::2]

    def odd(self, x):
        return x[:, :, :, 1::2]

    def forward(self, x):
        return (self.even(x), self.odd(x))

class IDGCN(nn.Module):
    def __init__(
            self,
            device,
            channels=64,
            splitting=True,
            num_nodes=170,
            dropout=0.1
    ):
        super(IDGCN, self).__init__()
        device = device
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.splitting = splitting
        self.split = Splitting()
        Conv1 = []
        Conv2 = []
        Conv3 = []
        Conv4 = []
        pad_l = 2
        pad_r = 0


        k1 = 3

        Conv1 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels, kernel_size=(1, k1)),
            nn.Tanh(),
        ]
        Conv2 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)), 
            nn.Conv2d(channels, channels, kernel_size=(1, k1)),
            nn.Tanh(),
        ]
        Conv4 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels, kernel_size=(1, k1)),
            nn.Tanh(),
        ]
        Conv3 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels, kernel_size=(1, k1)),
            nn.Tanh(),
        ]

        self.conv1 = nn.Sequential(*Conv1)
        self.conv2 = nn.Sequential(*Conv2)
        self.conv3 = nn.Sequential(*Conv3)
        self.conv4 = nn.Sequential(*Conv4)

    def forward(self, x):
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        x1 = self.conv1(x_even)
        d = x_odd.mul(torch.exp(x1))

        x2 = self.conv2(x_odd)
        c = x_even.mul(torch.exp(x2))

        x3 = self.conv3(c)
        x_odd_update = d + x3

        x4 = self.conv4(d)
        x_even_update = c + x4

        return (x_odd_update, x_even_update)


class IDGCN_Tree(nn.Module):
    def __init__(
            self, device, channels=64, num_nodes=170, dropout=0.1
    ):
        super().__init__()

        self.IDGCN1 = IDGCN(
            device=device,
            splitting=True,
            channels=channels,
            num_nodes=num_nodes,
            dropout=dropout
        )

    def concat(self, odd, even):
        odd = odd.permute(3, 1, 2, 0)
        even = even.permute(3, 1, 2, 0)
        len = even.shape[0]
        _ = []
        for i in range(len):
            _.append(odd[i].unsqueeze(0))
            _.append(even[i].unsqueeze(0))
        return torch.cat(_, 0).permute(3, 1, 2, 0)

    def forward(self, x):
        x_odd_update1,x_even_update1 = self.IDGCN1(x)
        concat0 = self.concat(x_odd_update1,x_even_update1) + x
        output = concat0
        return output



class TConv(nn.Module):
    def __init__(self, features=128, layer=4, length=12, dropout=0.1):
        super(TConv, self).__init__()

        layers = []
        kernel_size2 = int(length / layer + 1)
        for i in range(layer):
            self.conv = nn.Conv2d(features, features, (1, kernel_size2))
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            layers += [nn.Sequential(self.conv, self.relu, self.dropout)]
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        x1 = nn.functional.pad(x, (1, 0, 0, 0))
        x = self.tcn(x1)+ x[..., -1].unsqueeze(-1)
        return x


class Conv(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(features, features, (1, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class DGCN(nn.Module):
    def __init__(self, device, network_channel, num_nodes, seq_length=1, dropout=0.1):
        super(DGCN, self).__init__()
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.q = Conv(network_channel)
        self.v = Conv(network_channel)
        self.concat = Conv(network_channel)

        self.memory = nn.Parameter(torch.randn(network_channel, num_nodes))
        nn.init.xavier_uniform_(self.memory)
        self.fc = (nn.Sequential(
            nn.Linear(2, 8),  # 增加隐藏层
            nn.ReLU(),
            nn.Linear(8, 1)
        ))

    def forward(self, input, adj_list=None):
        query, value = self.q(input), self.v(input)

        adj_dyn_1 = torch.softmax(
            F.relu(
                torch.einsum("bcnt, cm->bnm", query, self.memory).contiguous()
                / math.sqrt(query.shape[1])
            ),
            -1,
        )
        adj_dyn_2 = torch.softmax(
            F.relu(
                torch.einsum("bcn, bcm->bnm", query.sum(-1), query.sum(-1)).contiguous()
                / math.sqrt(query.shape[1])
            ),
            -1,
        )
        adj_f = torch.cat([(adj_dyn_1).unsqueeze(-1)] + [(adj_dyn_2).unsqueeze(-1)], dim=-1)
        adj_f = torch.softmax(self.fc(adj_f).squeeze(), -1)
        # adj_f =adj_dyn_2

        topk_values, topk_indices = torch.topk(adj_f, k=int(adj_f.shape[1] * 0.8), dim=-1)
        mask = torch.zeros_like(adj_f)
        mask.scatter_(-1, topk_indices, 1)
        adj_f = adj_f * mask

        # 图卷积部分
        batch_size, c, n, t = value.size()

        # 调整value形状并进行图卷积
        value_reshaped = value.permute(0, 2, 1, 3).contiguous().view(batch_size, n, -1)  # [batch, n, c*t]
        aggregated = torch.bmm(adj_f, value_reshaped)  # 邻接矩阵传播
        aggregated = aggregated.view(batch_size, n, c, t).permute(0, 2, 1, 3).contiguous()  # 恢复形状
        # 最终卷积变换
        x = self.concat(aggregated)

        return x

class Encoder(nn.Module):
    def __init__(self, device, network_channel,  num_nodes, seq_length=1, dropout=0.1):
        "Take in model size and number of heads."
        super(Encoder, self).__init__()
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.network_channel = network_channel
        self.dgcn= DGCN(
            device, network_channel, num_nodes, seq_length=seq_length
        )

        self.dyt1 = DyT(network_channel,0.6 )
        self.dyt2 = DyT(network_channel, 0.6 )

        self.dropout1 = nn.Dropout(p=dropout)
        self.glu1 = GLU(network_channel)
        self.glu2 = GLU(network_channel)
        self.dropout2 = nn.Dropout(p=dropout)

        self.weight = nn.Parameter(torch.ones(network_channel, num_nodes, seq_length))


    def forward(self, input, adj_list=None):
        # 64 64 170 12
        x = self.glu1(input) + input
        x = self.dyt1(x)
        x1 = self.dgcn(x)
        x = x + x1
        x = self.dropout1(x)
        x = self.glu2(x) + x
        x = x * self.weight +x
        x = self.dyt2(x)
        x = self.dropout2(x)
        return x

class TIADGCN(nn.Module):
    def __init__(
        self,
        device,
        input_dim=3,
        channels=64,
        num_nodes=170,
        input_len=12,
        output_len=12,
        dropout=0.1,
    ):
        super().__init__()

        # attributes
        self.device = device
        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        if num_nodes == 170 or num_nodes == 307 or num_nodes == 358  or num_nodes == 883 or num_nodes == 207 or num_nodes == 325:
            time = 288

        self.Temb = TemporalEmbedding(time, channels)
        #self.fc = nn.Linear(12,1)
        self.tconv = TConv(channels, layer=4, length=self.input_len)
        self.tree1 = IDGCN_Tree(
            device=device,
            channels=channels,
            num_nodes=self.num_nodes,
            dropout=dropout,
        )

        self.start_conv = nn.Conv2d(self.input_dim, channels, kernel_size=(1, 1))

        self.network_channel = 2*channels

        self.SpatialBlock = Encoder(
            device,
            network_channel=self.network_channel,

            num_nodes=num_nodes,
            seq_length=1,
            dropout=dropout,
        )

        self.fc_st = nn.Conv2d(
            self.network_channel, self.network_channel, kernel_size=(1, 1)
        )

        self.regression_layer = nn.Conv2d(
            self.network_channel, self.output_len, kernel_size=(1, 1)
        )

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])


    def forward(self, history_data):

        input_data = history_data

        history_data = history_data.permute(0, 3, 2, 1)

        input_data = self.start_conv(input_data)

        input_data = self.tree1(input_data)
        #
        #
        #
        # batch_size, height, width, channels = input_data.shape
        # input_data = input_data.reshape(-1, channels)  # 将 x 展平为 [64*128*170, 12]
        # input_data = self.fc(input_data)  # 应用全连接层，输出形状为 [64*128*170, 1]
        # # 将 x 重新塑形为 [64, 128, 170, 1]
        # input_data = input_data.reshape(batch_size, height, width, 1)


        input_data = self.tconv(input_data)

        tem_emb = self.Temb(history_data)

        data_st = torch.cat([input_data] + [tem_emb], dim=1)

        # data_st = input_data

        data_st = self.SpatialBlock(data_st) + self.fc_st(data_st)

        prediction = self.regression_layer(data_st)

        return prediction

