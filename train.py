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

    trainer.run()


if __name__ == '__main__':
    main()
