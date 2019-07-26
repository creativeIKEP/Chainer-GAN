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
            self.d0 = L.Convolution2D(3, 32, 4, 2, 1, initialW=w)
            self.d1 = L.Convolution2D(32, 64, 4, 2, 1, initialW=w)
            self.d2 = L.Convolution2D(64, 128, 4, 2, 1, initialW=w)
            self.d3 = L.Convolution2D(128, 256, 4, 2, 1, initialW=w)
            self.d_bn0 = L.BatchNormalization(32)
            self.d_bn1 = L.BatchNormalization(64)
            self.d_bn2 = L.BatchNormalization(128)
            self.d_bn3 = L.BatchNormalization(256)

            """up sampling"""
            self.u0 = L.Deconvolution2D(256, 128, 4, 2, 1, initialW=w)
            self.u1 = L.Deconvolution2D(128*2, 64, 4, 2, 1, initialW=w)
            self.u2 = L.Deconvolution2D(64*2, 32, 4, 2, 1, initialW=w)
            self.u3 = L.Deconvolution2D(32*2, 3, 4, 2, 1, initialW=w)
            self.u_bn0 = L.BatchNormalization(128)
            self.u_bn1 = L.BatchNormalization(64)
            self.u_bn2 = L.BatchNormalization(32)


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
            self.c0 = L.Convolution2D(3, 64, 4, 2, 1, initialW=w)
            self.c1 = L.Convolution2D(64, 128, 4, 2, 1, initialW=w)
            self.c2 = L.Convolution2D(128, 256, 4, 2, 1, initialW=w)
            self.c3 = L.Convolution2D(256, 512, 4, 2, 1, initialW=w)
            self.c4 = L.Convolution2D(512, 1, 4, 2, 1, initialW=w)
            self.bn1 = L.BatchNormalization(128)
            self.bn2 = L.BatchNormalization(256)
            self.bn3 = L.BatchNormalization(512)


    def forward(self, x):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(self.bn2(self.c2(h)))
        h = F.leaky_relu(self.bn3(self.c3(h)))
        return self.c4(h)
