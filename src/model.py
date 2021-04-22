import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.drp = nn.Dropout2d(0.2)
        self.fc_drp = nn.Dropout(0.2)
        self.pool1 = nn.MaxPool2d(2, stride=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.pool3 = nn.MaxPool2d(2, stride=3)

        self.conv1 = nn.Conv2d(
            1, 24, 5, dilation=1, stride=1
        )  # inchannel , outchannel , kernel size ..
        self.conv1_bn = nn.BatchNorm2d(24)

        self.conv2 = nn.Conv2d(24, 32, 7, dilation=1, stride=1)
        self.conv2_bn = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 9, dilation=1, stride=2)
        self.conv3_bn = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, 11, dilation=1, stride=2)
        self.conv4_bn = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 3 * 12, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.conv1_bn(self.pool1(F.relu(self.conv1(x))))
        x = self.drp(x)

        x = self.conv2_bn(self.pool1(F.relu(self.conv2(x))))
        x = self.drp(x)

        x = self.conv3_bn(self.pool2(F.relu(self.conv3(x))))
        x = self.drp(x)

        x = self.conv4_bn(self.pool3(F.relu(self.conv4(x))))
        x = self.drp(x)

        # import pdb ; pdb.set_trace()
        x = x.view(-1, 64 * 3 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc_drp(x)

        x = self.fc2(x)

        return x


class DilatedCausalConv1d(nn.Module):
    def __init__(self, hyperparams: dict, dilation_factor: int, in_channels: int):
        super().__init__()

        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight.data)
                init.zeros_(m.bias.data)

        self.dilation_factor = dilation_factor
        self.padding = (hyperparams["kernel_size"] - 1) * self.dilation_factor
        self.dilated_causal_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hyperparams["nb_filters"],
            kernel_size=hyperparams["kernel_size"],
            padding=self.padding,
            dilation=dilation_factor,
        )

        self.dilated_causal_conv.apply(weights_init)

        self.skip_connection = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hyperparams["nb_filters"],
            kernel_size=1,
            padding=0,
        )

        ## .apply works recursively on nn.Module, and recursively applies same initilization to each layer .
        ## Catch is, Each DialetedCausaConvolution has a Skip conncection !
        self.skip_connection.apply(weights_init)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):

        x1 = self.leaky_relu(self.dilated_causal_conv(x))

        ## since we are shifting a 1*1 kernel, we don't need any padding to maintain the input shape
        x2 = self.skip_connection(x)

        ## remove k-1 * dialation_Factor values from the end
        x1 = x1[:, :, : -self.padding]

        ## residual connection
        return x1 + x2


import torch.nn.init as init


class WaveNet(nn.Module):
    def __init__(self, hyperparams: dict, in_channels: int):
        super().__init__()

        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight.data)
                init.zeros_(m.bias.data)

        self.dilation_factors = [
            2 ** i for i in range(0, hyperparams["nb_layers"])
        ]  ## ( 2^0 , 2^1 , 2^2 , 2^3 )

        self.in_channels = [in_channels] + [
            hyperparams["nb_filters"] for _ in range(hyperparams["nb_layers"])
        ]  ## [1 , 8 ,8 ,8 ,8  ]

        self.dilated_causal_convs = nn.ModuleList(
            [
                DilatedCausalConv1d(
                    hyperparams, self.dilation_factors[i], self.in_channels[i]
                )
                for i in range(hyperparams["nb_layers"])
            ]
        )

        for dilated_causal_conv in self.dilated_causal_convs:
            dilated_causal_conv.apply(weights_init)

        ## output layer is : just 8 channel input, and one channel output
        ## kernel size 1 implaying : only single weight is slide across sequence
        ## across different channels, per_channel , we sum the results
        self.output_layer = nn.Conv1d(
            in_channels=self.in_channels[-1], out_channels=1, kernel_size=1
        )

        self.dense = nn.Linear(512, 64).apply(weights_init)
        self.bn_1 = nn.BatchNorm1d(64)
        self.final_future = nn.Linear(64, 1).apply(weights_init)

        self.output_layer.apply(weights_init)

        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):

        for dilated_causal_conv in self.dilated_causal_convs:
            x = dilated_causal_conv(x)

        print("Result of Dialated Conv:  ", x.shape)

        x = self.leaky_relu(self.output_layer(x))
        # import pdb
        # pdb.set_trace()

        x = F.leaky_relu(self.bn_1(self.dense(x.view(-1, 8 * 352800))))
        x = F.leaky_relu(self.final_future(x))

        return x
