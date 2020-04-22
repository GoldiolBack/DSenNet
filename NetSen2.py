from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self, feature_size=3, kernel_size=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, feature_size, kernel_size, stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size, 1, 1)
        self.rBlock = ResBlock(feature_size, kernel_size)

    def forward(self, sentinel, num_layers=6):
        input20 = sentinel[0:4]
        input10 = sentinel[5:9]
        x = sentinel
        x = self.conv1(x)
        x = F.relu(x)
        for i in range(num_layers):
            x = self.rBlock(x)
        x = self.conv2(x)
        x += input20
        return x


class ResBlock(nn.Module):
    def __init__(self, channels=3, kernel_size=3):
        super(ResBlock, self).__init__()
        self.conv3 = nn.Conv2d(3, 3, kernel_size, 1, 1)

    def forward(self, x, scale=0.1):
        tmp = self.conv3(x)
        tmp = F.relu(tmp)
        tmp = self.conv3(tmp)
        tmp = tmp * scale
        tmp += x
        return tmp


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch NetSen2')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

#    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    example_traindata = torch.rand([10, 2, 3, 32, 32], device=device)
    example_testdata = torch.rand([2, 3, 32, 32], device=device)
    train_loader = torch.utils.data.DataLoader(example_traindata,
        batch_size=2, shuffle=True)
    test_loader = torch.utils.data.DataLoader(example_testdata,
        batch_size=args.test_batch_size, shuffle=True)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "DSen2.pt")


if __name__ == '__main__':
    main()
