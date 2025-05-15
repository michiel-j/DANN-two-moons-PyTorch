import torch


def init_weights(layer):
    """
    Init weights for layers w.r.t. the original paper.
    Copied from https://github.com/mashaan14/DANN-toy
    """
    layer_name = layer.__class__.__name__
    
    if layer_name.find('Linear') != -1:
        # torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        torch.nn.init.xavier_uniform_(layer.weight) # This init is used in the ADAPTS example (TensorFlow default)
        layer.bias.data.fill_(0.01) # Init with small value such that ReLU is activated
    return


class Encoder(torch.nn.Module):
    """
    encoder for DAtorch.nn.
    """
    def __init__(self):
        super(Encoder, self).__init__()

        # Copied from https://adapt-python.github.io/adapt/examples/Two_moons.html#Network
        self.linear1 = torch.nn.Linear(2, 10, dtype=torch.float32)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(10, 10, dtype=torch.float32)
        self.relu2 = torch.nn.ReLU()
        
        self.apply(init_weights)
        return

    def forward(self, input):
        out = self.relu1(self.linear1(input))
        out = self.relu2(self.linear2(out))
        return out


class Classifier(torch.nn.Module):
    """
    classifier for DAtorch.nn.
    """
    def __init__(self):
        super(Classifier, self).__init__()

        # Copied from https://adapt-python.github.io/adapt/examples/Two_moons.html#Network
        self.linear1 = torch.nn.Linear(10, 1, dtype=torch.float32) # Copied from https://adapt-python.github.io/adapt/examples/Two_moons.html#Network
        self.sigmoid = torch.nn.Sigmoid()
        self.apply(init_weights)
        return

    def forward(self, input):
        out = self.sigmoid(self.linear1(input))
        return out


class Discriminator(torch.nn.Module):
    """
    Discriminator model for source domain.
    """

    def __init__(self):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        # Copied from https://adapt-python.github.io/adapt/examples/Two_moons.html#Network
        self.linear1 = torch.nn.Linear(10, 10, dtype=torch.float32)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(10, 10, dtype=torch.float32)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(10, 1, dtype=torch.float32)
        self.apply(init_weights)
        return


    def forward(self, input):
        """Forward the discriminator."""
        out = self.linear1(input)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out)
        return out
