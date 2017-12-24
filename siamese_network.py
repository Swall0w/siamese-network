import chainer
import chainer.functions as F
import chainer.links as L
from contrastive import contrastive
from chainer import reporter


class SiameseNetwork(chainer.Chain):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 20, ksize=5, stride=2)
            self.conv2 = L.Convolution2D(20, 50, ksize=5, stride=2)
            self.fc3 = L.Linear(None, 500)
            self.fc4 = L.Linear(None, 10)
            self.fc5 = L.Linear(None, 2)

    def forward_once(self, x_data):
        x = chainer.Variable(x_data)
        h = F.max_pooling_2d(self.conv1(x), ksize=2, stride=2)
        h = F.max_pooling_2d(self.conv2(h), ksize=2, stride=2)
        h = F.relu(self.fc3(h))
        h = self.fc4(h)
        h = self.fc5(h)
        return h

    def __call__(self, x0, x1, label):
        y0 = self.forward_once(x0)
        y1 = self.forward_once(x1)
        label = chainer.Variable(label)
        loss = contrastive(y0, y1, label)
        reporter.report({'loss': loss}, self)
       return loss
