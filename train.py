import chainer
import argparse
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int)
    parser.add_argument('--epoch', '-e', default=200, type=int)
    parser.add_argument('--batch', '-b', default=128, type=int)
    parser.add_argument('--out', '-o', default='result', type=str)

    return parser.parse_args()


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
        return contrastive(y0, y1, label)


def main():
    args = arg()
    model = SiameseNetwork()
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train, test = chainer.datasets.get_mnist()

    train_iter = chainer.iterators.SerialIterator(train, args.batch)
    test_iter = chainer.iterators.SerialIterator(test, args.batch,
                                                 repeat=False, shuffle=False)

    updater = training.updater.StandardUpdater(
        train_iter, optimizer, device=args.gpu
    )
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()


if __name__ == '__main__':
    main()
