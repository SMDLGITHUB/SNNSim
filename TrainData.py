import torch
from torchvision import datasets, transforms

def loaders(args):
    transform = transforms.Compose([
        transforms.RandomChoice([transforms.AutoAugment(),transforms.RandAugment(),transforms.TrivialAugmentWide(),]),
        transforms.ToTensor(),
        transforms.RandomChoice([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomAffine(30),transforms.RandomErasing(), transforms.RandomResizedCrop(32),]),
    ])

    transform2 = transforms.Compose([
        transforms.ToTensor(),
    ])


    if args.dataset.lower() == 'mnist':
        trainset = datasets.MNIST(root='data', train=True, transform=transform2, download=True)
        testset = datasets.MNIST(root='data', train=False, transform=transform2, download=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=0)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=True, num_workers=0)
        image_size, n_image_channel, n_output = 28, 1, 10

    elif args.dataset.lower() == 'cifar10':
        trainset = datasets.CIFAR10(root='cifar-10-python', train=True, transform=transform, download=True)
        testset = datasets.CIFAR10(root='cifar-10-python', train=False, transform=transform2, download=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=8)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=True, num_workers=8)
        image_size, n_image_channel, n_output = 32, 3, 10

    else:
        raise Exception("[ERROR] The dataset " + str(args.dataset) + " is not supported!")

    return trainloader, testloader, image_size, n_image_channel, n_output