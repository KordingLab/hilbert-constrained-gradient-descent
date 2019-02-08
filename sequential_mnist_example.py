import argparse
import torch.nn.functional as F
from torch import optim

from model import LSTMBaseline
from utils import sequential_MNIST, train, test
from hcgd import HCAdam, HCGD



parser = argparse.ArgumentParser(description='Recurrent Unit Baselines')

parser.add_argument('--batch_size', help='batch size of network', type=int, default=64)
parser.add_argument('--epochs', help='number of epochs', type=int, default=20)
parser.add_argument('--hidden_layer_size', help='size of the hidden layer in the LSTM', type=int, default=128)
parser.add_argument('--no-gpu', help='dont use gpu for training', action='store_true')
parser.add_argument('--learning-rate', help='the learning rate', type=float, default=0.01)
parser.add_argument('--inner-learning-rate', help='the inner learning rate for HC methods', type=float, default=0.01)
parser.add_argument('--n-corrections', help='number of iterations in the inner loop for HC methods',
                                    type=int, default=1)
parser.add_argument('--function-correction-lambda',
                    help='hyperparameter for HC methods, controlling strength of functional regularization',
                    type=float, default=0.5)
parser.add_argument('--gradient_clipping_value', help='the gradient clipping value', type=int, default=1)
parser.add_argument('--log-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before logging training status, if wanting train error')
parser.add_argument('--opt', type=str, default='sgd',  help='which optimizer to use',
                    choices=('hcadam', 'adam', 'sgd', 'hcgd', 'rmsprop'))

args = parser.parse_args()

args.gpu = not args.no_gpu

if __name__ == '__main__':

    # how many pixels to read at one time. =1 is the pure sequential MNIST task
    NUM_PIXELS = 1

    training_data, testing_data = sequential_MNIST(args.batch_size, NUM_PIXELS, gpu=args.gpu)

    val_loader, _ = sequential_MNIST(args.batch_size, NUM_PIXELS, gpu=args.gpu)

    lr = args.learning_rate
    fcl = args.function_correction_lambda
    ilr = args.inner_learning_rate
    ncorr = args.n_corrections

    model = LSTMBaseline(NUM_PIXELS, args.batch_size, hidden_dim=args.hidden_layer_size, num_layers=1)

    if args.gpu:
        model.cuda()

    criterion = F.nll_loss

    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif args.opt == 'sgdw':
        optimizer = HCGD(model.parameters(),
                         lr, momentum=0.9,weight_decay=1e-4, #normal SGD params
                         fcn_change_limiter=fcl, inner_lr=ilr, n_corrections=ncorr,  # HC params
                         )
    elif args.opt == 'hcadam':
        optimizer = HCAdam(model.parameters(), lr,
                           fcn_change_limiter=fcl, inner_lr=ilr, n_corrections=ncorr,  # HC params
                           betas=(0.9, 0.999), eps=1e-08, amsgrad=False,  # adam params
                           weight_decay=0, clip_correction_grad=0, )


    test_accuracy = []
    train_acc = []
    for epoch in range(1, args.epochs):
        tr, te = train(model, training_data, val_loader, testing_data, criterion, args, optimizer)
        if len(te)>0:
            test_error_to_print = te[-1]
        else:
            test_loss, acc = test(model, testing_data, criterion, args, 10000)
            test_error_to_print = acc
        print('Epoch: {}, test accuracy {}'.format(epoch, test_error_to_print))

        # store all the accuracies in a list. These can be used to create figures.
        test_accuracy += te
        train_acc += tr