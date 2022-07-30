from .convs import *


class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        self.config = config
        self.layers = self.layer_creator()

    def layer_creator(self):
        base = nn.ModuleList()

        for layer in self.config:

            if layer[0] == 'conv':
                base.append(
                    Conv(layer[1][0], layer[1][1], layer[1][2], layer[1][3], use_ac=layer[2][0], use_bn=layer[2][1])
                )
            elif layer[0] == 'residual':
                base.append(
                    ResidualBlock(layer[1][0], layer[2])
                )
            elif layer[0] == 'up':
                base.append(
                    nn.Upsample(scale_factor=layer[1])
                )
            elif layer[0] == 'linear':
                base.append(
                    nn.Sequential(
                        nn.Linear(layer[1][0], layer[1][1]).to(DEVICE),
                    )
                )
            elif layer[0] == 'max_pool':
                base.append(
                    nn.MaxPool2d(layer[1][0], layer[1][1])
                )
            elif layer[0] == 'middle_flow':
                for _ in range(layer[2]):
                    base.append(
                        MiddleFlow(layer[1][0])
                    )
            elif layer[0] == 'conv_pool':
                base.append(
                    ConvPool(layer[1][0], layer[1][1])
                )
            elif layer[0] == 'conv_route':
                base.append(
                    ConvRoute(layer[1][0], layer[1][1], layer[1][2])
                )
            elif layer[0] == 'conv_pool_cm':
                base.append(
                    ConvPoolCM(layer[1][0], layer[1][1], layer[1][2], layer[1][3], )
                )

        return base

    def forward(self, x, debug: bool = False, predicting: bool = False):
        route = []
        i = 0
        for layer in self.layers:

            if debug:
                print(f'{x.shape} Before layer')

            if isinstance(layer, Conv):
                x = layer(x)
            if isinstance(layer, ConvRoute):
                route.append(layer(x))
                i += 1
            if isinstance(layer, ConvPool):
                x = layer(x)
                if i > 0:
                    x = torch.cat((x, route[-1]), dim=1)

                    route.pop(-1)

            if isinstance(layer, MiddleFlow):
                x = layer(x)
                i = 0
                route = []

            if isinstance(layer, ConvPoolCM):
                x = layer(x)

                x = torch.cat((x, route[-1]), dim=1)
                route.pop(-1)
            if isinstance(layer, nn.Sequential):
                x = x.view(1, -1)
                x = layer(x)

            if debug:
                print(f'{x.shape} after layer \n ----------------')

        return x