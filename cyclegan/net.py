import chainer
import chainer.functions as F
import chainer.links as L


class Generator(chainer.Chain):
    """U-Net Generator"""
    def __init__(self):
        w = chainer.initializers.Normal(0.02)
        super().__init__()
        with self.init_scope():
            """down sampling"""
            self.d0 = L.Convolution2D(3, 8, 4, 2, 1, initialW=w)
            self.d1 = L.Convolution2D(8, 16, 4, 2, 1, initialW=w)
            self.d2 = L.Convolution2D(16, 32, 4, 2, 1, initialW=w)
            self.d3 = L.Convolution2D(32, 64, 4, 2, 1, initialW=w)
            self.d_bn0 = L.BatchNormalization(8)
            self.d_bn1 = L.BatchNormalization(16)
            self.d_bn2 = L.BatchNormalization(32)
            self.d_bn3 = L.BatchNormalization(64)

            """up sampling"""
            self.u0 = L.Deconvolution2D(64, 32, 4, 2, 1, initialW=w)
            self.u1 = L.Deconvolution2D(32*2, 16, 4, 2, 1, initialW=w)
            self.u2 = L.Deconvolution2D(16*2, 8, 4, 2, 1, initialW=w)
            self.u3 = L.Deconvolution2D(8*2, 3, 4, 2, 1, initialW=w)
            self.u_bn0 = L.BatchNormalization(32)
            self.u_bn1 = L.BatchNormalization(16)
            self.u_bn2 = L.BatchNormalization(8)


    def forward(self, x):
        """down sampling"""
        d0_out = F.leaky_relu(self.d_bn0(self.d0(x)))
        d1_out = F.leaky_relu(self.d_bn1(self.d1(d0_out)))
        d2_out = F.leaky_relu(self.d_bn2(self.d2(d1_out)))
        d3_out = F.leaky_relu(self.d_bn3(self.d3(d2_out)))

        """up sampling"""
        u0_out = F.leaky_relu(self.u_bn0(self.u0(d3_out)))
        u1_in = F.concat((u0_out, d2_out), axis=1)
        u1_out = F.leaky_relu(self.u_bn1(self.u1(u1_in)))
        u2_in = F.concat((u1_out, d1_out), axis=1)
        u2_out = F.leaky_relu(self.u_bn2(self.u2(u2_in)))
        u3_in = F.concat((u2_out, d0_out), axis=1)
        y = self.u3(u3_in)
        return y


class Discriminator(chainer.Chain):
    def __init__(self):
        w = chainer.initializers.Normal(0.02)
        super().__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(3, 8, 4, 2, 1, initialW=w)
            self.c1 = L.Convolution2D(8, 16, 4, 2, 1, initialW=w)
            self.c2 = L.Convolution2D(16, 32, 4, 2, 1, initialW=w)
            self.c3 = L.Convolution2D(32, 64, 4, 2, 1, initialW=w)
            self.c4 = L.Convolution2D(64, 1, 4, 2, 1, initialW=w)
            self.l1 = L.Linear(1*8*8, 2, initialW=w)
            self.bn1 = L.BatchNormalization(16)
            self.bn2 = L.BatchNormalization(32)
            self.bn3 = L.BatchNormalization(64)


    def forward(self, x):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(self.bn2(self.c2(h)))
        h = F.leaky_relu(self.bn3(self.c3(h)))
        return self.l1(self.c4(h))
