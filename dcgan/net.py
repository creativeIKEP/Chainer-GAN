import chainer
import chainer.functions as F
import chainer.links as L


class Generator(chainer.Chain):
    def __init__(self, n_hidden=300):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(n_hidden, 4*4*384)
            self.dc0 = L.Deconvolution2D(384, 192, 4, 2, 1)
            self.dc1 = L.Deconvolution2D(192, 96, 4, 2, 1)
            self.dc2 = L.Deconvolution2D(96, 48, 4, 2, 1)
            self.dc3 = L.Deconvolution2D(48, 24, 4, 2, 1)
            self.dc4 = L.Deconvolution2D(24, 12, 4, 2, 1)
            self.dc5 = L.Deconvolution2D(12, 6, 4, 2, 1)
            self.dc6 = L.Deconvolution2D(6, 3, 3, 1, 1)
            self.bn_l0 = L.BatchNormalization(4*4*384)
            self.bn0 = L.BatchNormalization(192)
            self.bn1 = L.BatchNormalization(96)
            self.bn2 = L.BatchNormalization(48)
            self.bn3 = L.BatchNormalization(24)
            self.bn4 = L.BatchNormalization(12)
            self.bn5 = L.BatchNormalization(6)


    def forward(self, z, batch_size):
        h = F.relu(self.bn_l0(self.l0(z)))
        h = F.reshape(h, (batch_size, 384, 4, 4))
        h = F.relu(self.bn0(self.dc0(h)))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        h = F.relu(self.bn4(self.dc4(h)))
        h = F.relu(self.bn5(self.dc5(h)))
        x = F.sigmoid(self.dc6(h))
        return x


class Discriminator(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(3, 6, 4, 2, 1)
            self.c1 = L.Convolution2D(6, 12, 4, 2, 1)
            self.c2 = L.Convolution2D(12, 24, 4, 2, 1)
            self.c3 = L.Convolution2D(24, 48, 4, 2, 1)
            self.c4 = L.Convolution2D(48, 96, 4, 2, 1)
            self.c5 = L.Convolution2D(96, 192, 4, 2, 1)
            self.l6 = L.Linear(192*4*4, 1)
            self.bn1 = L.BatchNormalization(12)
            self.bn2 = L.BatchNormalization(24)
            self.bn3 = L.BatchNormalization(48)
            self.bn4 = L.BatchNormalization(96)
            self.bn5 = L.BatchNormalization(192)


    def forward(self, x):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(self.bn2(self.c2(h)))
        h = F.leaky_relu(self.bn3(self.c3(h)))
        h = F.leaky_relu(self.bn4(self.c4(h)))
        h = F.leaky_relu(self.bn5(self.c5(h)))
        return self.l6(h)
