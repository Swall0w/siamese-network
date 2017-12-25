import chainer
import argparse
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from contrastive import contrastive
import numpy as np
from chainer import reporter
from siamese_network import SiameseNetwork
import os
import os.path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int)
    parser.add_argument('--epoch', '-e', default=50, type=int)
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
        xp = optimizer.target.xp
        label = xp.array(y0_batch == y1_batch, dtype=np.int32)
        optimizer.update(optimizer.target, x0_batch, x1_batch, label)


def plot_testdata(model, data, batch, dst='pict'):
    @training.make_extension()
    def plot_image(trainer):
        if not os.path.exists(dst):
            os.makedirs(dst)
        c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
             '#ff00ff', '#990000', '#999900', '#009900', '#009999']
        xp = model.xp
        N = len(data)
        result = xp.empty((N, 2))
        label = xp.empty((N))
        with chainer.using_config('train', False):
            for i in range(0, N, batch):
                x_batch = [dat[0] for dat in data[i: i+batch]]
                y_batch = [dat[1] for dat in data[i: i+batch]]
                x_batch = xp.asarray(x_batch, dtype=xp.float32)
                y_batch = xp.asarray(y_batch, dtype=xp.int32)
                y = model.forward_once(x_batch)
                result[i: i+batch] = y.data
                label[i: i+batch] = y_batch

        for i in range(10):
            feat = result[np.where(label == i)]
            plt.plot(feat[:, 0], feat[:, 1], '.', c=c[i])
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        plt.xlim([-2,2])
        plt.ylim([-2,2])
        plt.savefig('{}/result_{}.png'.format(dst, trainer.updater.epoch))
        plt.clf()
    return plot_image


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
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'elapsed_time', 'main/loss']))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(plot_testdata(model, test, args.batch), trigger=(1, 'epoch'))

    trainer.run()


if __name__ == '__main__':
    main()
