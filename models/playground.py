import torch
import torch.nn as nn
from typing import List


class Playground(nn.Module):
    def __init__(self, layer_dims: List[int]):
        super().__init__()
        layers = []
        layer_count = len(layer_dims)
        self.layers = []
        for i in range(1, layer_count):
            layers.append(
                torch.nn.Linear(layer_dims[i - 1], layer_dims[i], True)
            )
            if (i < layer_count-1):
                layers.append(torch.nn.ReLU())
                self.layers.append(torch.nn.Sequential(*layers))
            else:
                layers.append(torch.nn.Softmax(dim=1))

            #self.layers.append(torch.nn.Sequential(*layers))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def forward_features(self, x):
        return [layer(x) for layer in self.layers]

    def forward_param_features(self, x):
        return self.forward_features(x)


def test():
    net = Playground(layer_dims=[2, 3, 2])
    print(net)
    x = torch.randn(100, 2)
    y = net(x)
    print(y.size())

    for i, layer in enumerate(net.forward_features(x)):
        print('layer {} has size {}'.format(i, layer.shape))


if __name__ == '__main__':
    test()
