import torch
from torchvision import datasets, transforms

def server():
    pass

def client():
    pass

def train(model, train_loader, epoch, learning_rate):
    pass

if __name__ == '__main__':
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_set = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_set)
    test_loader = torch.utils.data.DataLoader(test_set)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Model()
    model.to(device)



