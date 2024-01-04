import numpy as np
from torchvision import datasets, transforms


def iid(dataloader):
    data_set = [data for index, data in enumerate(dataloader)]
    user = []
    for index in range(100):
        user.append(data_set[index * 600: index * 600 + 600])
    return user


def noniid(dataloader):
    data_set = [data for index, data in enumerate(dataloader)]
    data_set.sort(key=lambda x: x[1])
    shard = [i for i in range(300)]
    user = []
    for _ in range(100):
        select_two = set(np.random.choice(shard, 2, replace=False))
        shard = list(set(shard) - select_two)
        two_data = []
        for index in select_two:
            two_data += data_set[index * 600: index * 600 + 600]
        user.append(two_data)
    return user


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor()
                                   ]))
    client_data = noniid(dataset_train)
