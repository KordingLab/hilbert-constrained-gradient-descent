import torch
from torchvision import datasets, transforms
from torch.autograd import Variable



def sequential_MNIST(batch_size, input_size, gpu=True, dataset_folder='./data'):
    """Create the sequential MNIST dataset, and return the test and train dataloaders.
    Uses a new random seed every time it's called.

    Arguments:    batch_size: tha batch sz
    =========     input_size: in the sequential task, how many pixels to read in at once.
                                1 is the traditional task
                                28 is one row at a time
                                784 would be one image at a time
                  gpu: move the create dataloaders with pinned memory?
                  dataself_folder: path to the mnist dataset, or where it should go once downloaded

    """

    kwargs = {'num_workers': 0, 'pin_memory': True} if gpu else {}

    permute_mask = torch.randperm(784)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1, input_size)),
        transforms.Lambda(lambda x: x[permute_mask])])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(dataset_folder, train=True, download=True,
                       transform=transform),
        batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(dataset_folder, train=False, transform=transform),
        batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)

    return (train_loader, test_loader)


def train(model, training_data, val_loader, testing_data, criterion, args, optimizer):
    model.train()
    test_accuracy = []
    train_accuracy = []
    correct = 0

    for current_batch, ((data, target), (val_data, _)) in enumerate(zip(training_data, val_loader)):

        if args.gpu:
            data, target, val_data = data.cuda(), target.cuda(), val_data.cuda()
        data, target, val_data = Variable(data), Variable(target), Variable(val_data)

        model.zero_grad()
        model.hidden = model.init_hidden()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), args.gradient_clipping_value)

        # this is the key to using HC methods. We need to create a function that evaluates the model on the validation
        # batch, and then give this function to the step() call for the optimizer instance.
        def validation_eval():
            model.hidden = model.init_hidden()
            return torch.exp(model(val_data))

        if args.opt in ['hcadam','hcgd']:
            optimizer.step(validation_eval=validation_eval)
        else:
            optimizer.step()

        # logging
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum() if (current_batch > 0) else 0
        if (args.log_interval > 0) and (current_batch > 0) and (current_batch % args.log_interval == 0):
            acc = 100. * correct / (args.batch_size * args.log_interval)
            train_accuracy.append(acc)
            correct = 0

            tl, te = test(model, testing_data, criterion, args, n_examples=1000)

            print('Batch {}: Train accuracy ({:.5f}%)\tLoss: {:.6f} Test accuracy ({:.5f}%)\tLoss: {:.6f}'.format(
                current_batch,
                acc, loss.data[0], te, tl))

            test_accuracy.append(te)

    return train_accuracy, test_accuracy

def test(model, testing_data, criterion, args, n_examples):
    """
    Run through the testing data and return the test loss and accuracy.
    Only go through the first n_examples
    """

    model.eval()

    total = 0

    test_loss = 0
    correct = 0
    i = 0

    for data, target in testing_data:
        total += target.size(0)

        if args.gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        output = model(data)
        test_loss += criterion(output, target).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).sum()
        i += args.batch_size

        if i > n_examples:
            break

    test_loss /= i
    acc = 100. * float(correct) / i

    model.train()

    return test_loss, acc