import chainer
import argparse
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from contrastive import contrastive
import numpy as np
from chainer import reporter

def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int)
    parser.add_argument('--epoch', '-e', default=200, type=int)
    parser.add_argument('--batch', '-b', default=128, type=int)
    parser.add_argument('--out', '-o', default='result', type=str)

    return parser.parse_args()

class SiameseUpdater(training.StandardUpdater):
    def update_core(self):
        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)
        optimizer = self._optimizers['main']

        x0_batch, y0_batch = in_arrays
        x1_batch = x0_batch[::-1]
        y1_batch = y0_batch[::-1]
        label = np.array(y0_batch == y1_batch, dtype=np.int32)
        optimizer.update(optimizer.target, x0_batch, x1_batch, label)


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


def main():
    args = arg()
    model = SiameseNetwork()
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train, test = chainer.datasets.get_mnist(ndim=3)

    train_iter = chainer.iterators.SerialIterator(train, args.batch)
    test_iter = chainer.iterators.SerialIterator(test, args.batch,
                                                 repeat=False, shuffle=False)

    updater = SiameseUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
#    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()


if __name__ == '__main__':
    main()
