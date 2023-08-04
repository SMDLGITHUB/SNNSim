import torch
import torchvision
import torchvision.datasets as dsets
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import random
import os
import argparse

from Networks import MLP
from Networks import Mobile
from Networks import VGG6
from Networks import Mixer
from TrainData import loaders

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

####################Synapse Device Characteristics#######################
def GetCellArea(name):
    if name == 'flash':
        Cell = 6

    return Cell

###################################Learning##############################################
def train(args, model, device, trainloader, optimizer, criterion, scheduler):
    model.train()
    total_loss, total_num, correct = 0, 0, 0
    for x, y in trainloader:
        batch_size = x.shape[0]
        x, y = x.to(device), y.to(device)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1, keepdim=True)

        correct += pred.eq(y.view_as(pred)).sum().item()
        total_loss += loss.data.cpu().numpy().item() * batch_size
        total_num += batch_size

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc


def test(args, model, device, testloader, criterion):
    model.eval()
    total_loss, total_num, correct = 0, 0, 0
    with torch.no_grad():

        if args.train == False:
            # Calculation of Area per Block
            CellArea = GetCellArea(args.Device)
            if args.model == 'MLP':
                Synapse_Area = 784 * 32 * CellArea * args.Tech * args.Tech * 1e-12 * 2
                Synapse_Area += 32 * 10 * CellArea * args.Tech * args.Tech * 1e-12 * 2
                Neuron_area = 1335 * args.Tech * args.Tech * 1e-12
                Neuron_area += ((args.Capmem + args.Cappulse) * 10e-9)*0.8 / (8.85e-12* 2.5 * 3.9) * 1000000
                Neuron_area *= (32+10)
            # if args.Method == 'Offchip':
            #     print('Synapse Area  = {:.5f} mm^2 | Neuron Area = {:.5f} mm^2 '
            #           .format(Synapse_Area, Neuron_area))
            if args.model == 'VGG6':
                Synapse_Area = 9 * 3 * 32 * 32 * 32 * 2 * CellArea * args.Tech * args.Tech * 1e-12
                Synapse_Area +=  9 * 32 * 32 * 32 * 32 * 2 * CellArea * args.Tech * args.Tech * 1e-12
                Synapse_Area += 9 * 32 * 64 * 16 * 16 * 2 *CellArea * args.Tech * args.Tech * 1e-12
                Synapse_Area += 9 * 64 * 64 * 16 * 16 * 2 *CellArea * args.Tech * args.Tech * 1e-12
                Synapse_Area += 4096 * 512 * 2* CellArea * args.Tech * args.Tech * 1e-12
                Synapse_Area += 512 * 10 * 2 * CellArea * args.Tech * args.Tech * 1e-12

                Neuron_area = 1335 * args.Tech * args.Tech * 1e-12
                Neuron_area += ((args.Capmem + args.Cappulse) * 3.2e-9) / (8.85e-12 * 6 * 3.9) * 1000000
                Neuron_area *= ((32*32*32) + (32*32*32) + (64*16*16)+ (64*16*16)+(512)+(10))
            if args.model == 'Mixer':
                Synapse_Area = 784 * 256 * CellArea * args.Tech * args.Tech * 1e-12 * 2
                Synapse_Area += 256 * 10 * CellArea * args.Tech * args.Tech * 1e-12 * 2
                Neuron_area = 1.0
            if args.model == 'Mobile':
                Synapse_Area = 9 * 3 * 32 * 32 * 32 * 2 * CellArea * args.Tech * args.Tech * 1e-12
                Synapse_Area +=  1 * 32 * 32 * 32 * 32 * 2 * CellArea * args.Tech * args.Tech * 1e-12
                Synapse_Area += 9 * 32 * 64 * 16 * 16 * 2 *CellArea * args.Tech * args.Tech * 1e-12
                Synapse_Area += 1 * 64 * 64 * 16 * 16 * 2 *CellArea * args.Tech * args.Tech * 1e-12
                Synapse_Area += 4096 * 512 * 2 * CellArea * args.Tech * args.Tech * 1e-12
                Synapse_Area += 512 * 10 * 2*  CellArea * args.Tech * args.Tech * 1e-12

                Neuron_area = 1335 * args.Tech * args.Tech * 1e-12
                Neuron_area += ((args.Capmem + args.Cappulse) * 3.2e-9) / (8.85e-12 * 6 * 3.9) * 1000000
                Neuron_area *= ((32*32*32) + (32*32*32) + (64*16*16)+ (64*16*16)+(512)+(10))
            TotalArea = Synapse_Area + Neuron_area
            print('Synapse Area = {} mm^2 ' . format(Synapse_Area))
            print('Neuron Area = {} mm^2 ' . format(Neuron_area))
            print('Total Area = {} mm^2 ' . format(TotalArea))
        Time = 0
        for i, (x, y) in enumerate(testloader):
            batch = x.shape[0]
            x, y = x.to(device), y.to(device)
            if len(x.shape) == 3:
                x = x.unsqueeze(1)

            if args.train == True:
                out = model(x)
                loss = criterion(out, y)
                pred = out.argmax(dim=1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()
                total_loss += loss.data.cpu().numpy().item() * batch
            if args.train == False:
                out, time = model(x)
                for i in range(0, batch):
                    if out[i] == y[i]:
                        correct += 1
                Time += time.sum()
            total_num += batch


        print('Average timesteps = {} steps' . format(Time*args.TimeResol/total_num))
        if args.train == False:
            if (i+1)* args.batch >= 10000:
                print('Total Synapse Power = {} W'.format(model.SynpaseEnergy/(args.TimeSteps*args.TimeResol*total_num)))
                print('Total Neuron Power = {} W' . format(model.NeuronEnergy/(args.TimeSteps*args.TimeResol*total_num)))
                #print('Total Latency = {} s' . format(model.Latency))

        acc = 100. * correct / total_num
        final_loss = total_loss / total_num
        if args.train == True:
            if correct > 5000:
                print('saved')
                torch.save(model.state_dict(), 'SNNSimMLP_32.pth.tar')
    return final_loss, acc

#####################################################################################

def main(args):
    # Device Seting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Dataset
    trainloader, testloader, image_size, n_image_channel, n_output = loaders(args)

    # Model, Optimizer, and Criterion
    if args.model == 'VGG6':
        model = VGG6(3, 32, 512, args).to(device)
    if args.model == 'Mobile':
        model = Mobile(3, 32, 512, args).to(device)
    if args.model == 'MLP':
        model = MLP(784, 128, args).to(device)
    if args.model == 'Mixer':
        model = Mixer(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50, 80, 110, 140, 210, 300, 500, 700, 1000, 1300,
                                                        1600, 1900, 2200, 2500, 2800, 3100, 3300], gamma=0.5)
    # Train Proper
    if args.train == True:
        for epoch in range(args.n_epochs):
            tr_loss, tr_acc = train(args, model, device, trainloader, optimizer, criterion, scheduler)
            te_loss, te_acc = test(args, model, device, testloader, criterion)
            print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | Test: loss={:.3f}, acc={:5.1f}%'.format(
                epoch + 1, tr_loss, tr_acc, te_loss, te_acc))
    if args.train == False:
        if args.model == 'MLP':
            model.load_state_dict(torch.load('SNNSimMLP_128.pth.tar'))
            model.fc1.fc.weight = nn.Parameter(model.fc1.fc.weight * torch.clip(torch.normal(mean = 1, std = 0, size = model.fc1.fc.weight.size()), 0, 100).to(device))
            model.fc2.fc.weight = nn.Parameter(model.fc2.fc.weight * torch.clip(torch.normal(mean = 1, std = 0, size = model.fc2.fc.weight.size()), 0, 100).to(device))
            te_loss, te_acc = test(args, model, device, testloader, criterion)
            print('acc={:5.2f}%'.format(te_acc))
        if args.model == 'Mixer':
            model.load_state_dict(torch.load('SNNSimMixer_0331.pth.tar'))
            te_loss, te_acc = test(args, model, device, testloader, criterion)
            print('acc={:5.2f}%'.format(te_acc))
        if args.model == 'VGG6':
            model.load_state_dict(torch.load('SNNSimVGG6_0330.pth.tar'))
            te_loss, te_acc = test(args, model, device, testloader, criterion)
            print('acc={:5.2f}%'.format(te_acc))
        if args.model == 'Mobile':
            model.load_state_dict(torch.load('SNNSimMobile_0401.pth.tar'))
            te_loss, te_acc = test(args, model, device, testloader, criterion)
            print('acc={:5.2f}%'.format(te_acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NeuroTorch arguments')
    parser.add_argument('--Device', type=str, default='flash', metavar='DEVICE')
    parser.add_argument('--Neuron', type=str, default='TypeA', metavar='NEURON') # TypeA, TypeB
    parser.add_argument('--Method', type=str, default='Offchip', metavar='METHOD') # Onchip or Offchip
    parser.add_argument('--TimeResol', type=float, default=200e-9, metavar='TimeResol')
    parser.add_argument('--TimeSteps', type=int, default=250, metavar='TIMESTEPS')
    parser.add_argument('--Tech', type=int, default=22, metavar='TECHNODE') #500 45 22
    parser.add_argument('--Oncurrent', type=float, default=1e-6, metavar='ONCURRENT')
    parser.add_argument('--Offcurrent', type=float, default= 0, metavar='OFFCURRENT')
    parser.add_argument('--SpikeWidth', type=float, default= 1e-7, metavar='SPIKEWIDTH')
    parser.add_argument('--Capmem', type=float, default= 0.2e-12, metavar='CAPMEM')
    parser.add_argument('--Cappulse', type=float, default= 0.1e-12, metavar='CAPPULSE')

    #################################### Learning Condition #################################
    parser.add_argument('--model', type=str, default='VGG6', metavar='MODEL') # MLP, VGG6, Mixer, Mobile
    parser.add_argument('--seed', type=int, default=776, metavar='SEED')
    parser.add_argument('--dataset', type=str, default='cifar10', metavar='DATASET') # mnist, cifar10
    parser.add_argument('--batch', type=int, default=250, metavar='BATCH') # divisor of 10000
    parser.add_argument('--n_epochs', type=int, default=50, metavar='N_EPOCHS')
    parser.add_argument('--lr', type=float, default= 0.3e-3, metavar='LEARNING_RATE')
    parser.add_argument('--gpu', type=str, default='0', metavar='GPU')
    parser.add_argument('--train', type=bool, default= False, metavar='TRAIN')

    args = parser.parse_args()
    print('Arguments =')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))

    main(args)