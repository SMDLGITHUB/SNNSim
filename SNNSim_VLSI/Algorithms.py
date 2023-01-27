from __future__ import print_function
import torch as torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import DeviCe
import NeuronA
import NeuronB
import NeuronC

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def MLP(Device, Neuron, Tech, Encoding, Timesteps, OffCurrent, ReadNoise, SPIKELOSS,Timeperstep):
    with torch.no_grad():
        Weight1 = torch.zeros(256, 784)
        Weight2 = torch.zeros(10, 256)
        if Encoding == 'Rate':
            np_load1 = np.load('weightsMLPFeFET1.npy')
            np_load2 = np.load('weightsMLPFeFET2.npy')
            Weight1 = torch.from_numpy(np_load1).to(device)
            Weight2 = torch.from_numpy(np_load2).to(device)
            Weight1 = Weight1.reshape(256, 784)
            Weight2 = Weight2.reshape(10, 256)
            Weight1 = Weight1.float()
            Weight2 = Weight2.float()
        if Encoding == 'Temp':
            np_load1 = np.load('weights_best1.npy')
            np_load2 = np.load('weights_best2.npy')
            Weight1 = torch.from_numpy(np_load1).to(device)
            Weight2 = torch.from_numpy(np_load2).to(device)
            Weight1 = Weight1.reshape(256, 784)
            Weight2 = Weight2.reshape(10, 256)
            Weight1 = Weight1.float()
            Weight2 = Weight2.float()
            Weight1 = Weight1/93.5380
            Weight2 = Weight2/93.5380

        Batch = 10000
        Corr = 0

        if Neuron == 'A':
            Vini = NeuronA.Vini
            VTh = NeuronA.VTh
            Capmem = NeuronA.Capmem
            Vsub = NeuronA.Vsub
            SpWid = NeuronA.SpWid
        if Neuron == 'B':
            Vini = NeuronB.Vini
            VTh = NeuronB.VTh
            Capmem = NeuronB.Capmem
            Vsub = NeuronB.Vsub
            SpWid = NeuronB.Spwid
        if Neuron == 'C':
            Vini = NeuronC.Vini
            VTh = NeuronC.VTh
            Capmem = NeuronC.Capmem
            Vsub = NeuronC.Vsub
            SpWid = NeuronC.Spwid
        if Tech == 45:
            Vdd = 1
        if Tech == 22:
            Vdd = 0.8

        GetAns = torch.zeros(Batch, 2).to(device)
        LayerSynapseEnergy = torch.zeros(2).to(device)
        LayerNeuronEnergy = torch.zeros(2).to(device)
        Vmem1 = torch.ones(Batch, 256).to(device) * Vini # Start the membrane potential at Vini
        Vmem2 = torch.ones(Batch, 10).to(device) * Vini # Start the membrane potential at Vini
        ISit = torch.zeros(Batch, 10).to(device)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        testset = torchvision.datasets.MNIST(root='MNIST', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=Batch, shuffle=False, num_workers=0)
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            images = images / images.max()
            images = images.view(-1, 784)
            if Encoding == 'Rate':
                for i in range(Timesteps):
                    InputSpike = torch.floor(images + torch.rand(Batch, 784).to(device))
                    ##################1st Layer####################################
                    DeviCe.UpdateEnergy(InputSpike[0:100, :], Weight1, Device, Vdd, SpWid) # Calculate Synapse energy(for 100 data(memory issue))
                    Current = torch.einsum('ab, cb -> ac', InputSpike, DeviCe.Current(Weight1, Device, Vdd, ReadNoise, OffCurrent)) * 0.025
                    NEXTSPIKE = (((Vmem1 * Capmem) + (Current * SpWid)) / (Capmem * VTh))
                    NEXTSPIKE = torch.where(NEXTSPIKE > 0, NEXTSPIKE, torch.zeros(Batch, 256).to(device))
                    NEXTSPIKE = torch.floor(NEXTSPIKE) # Convert NEXT SPIKE into natural number
                    Vmem1 = Vmem1 + (Current * SpWid) / Capmem # Update membrane voltage considering current sum
                    Vmem1 = torch.where(Vmem1 < 0, torch.zeros(Batch, 256).to(device), Vmem1) # Vmem can not go below 0
                    NEXTSPIKE = torch.where(NEXTSPIKE != 0, torch.ones(Batch, 256).to(device), NEXTSPIKE) # Spike fires only once in each time step. If you want higher rate of the neuron reduce the time window per timestep
                    Vmem1 = torch.where(NEXTSPIKE != 0, Vmem1 - Vsub, Vmem1) # Subtract membrane voltage if the neuron fires
                    Vmem1 = torch.where(Vmem1 > VTh, torch.ones(Batch, 256).to(device) * (VTh - 0.01), Vmem1) # if the neuron still exceeds a threshold even after the subtraction, set the Vmem below the threshold
                    if SPIKELOSS != 0: # Consider Stochastic spike loss in the circuits
                        SPKL = torch.rand(Batch, 256).to(device)
                        SPKL = torch.where(SPKL > SPIKELOSS, torch.ones(Batch, 256).to(device), torch.zeros(Batch, 256).to(device))
                        NEXTSPIKE = torch.mul(NEXTSPIKE, SPKL)
                    if Neuron == 'A': # Calculate Neuron energy(for 100 data(memory issue))
                        NeuronA.UpdateEnergy(NEXTSPIKE[0:100, :], Vmem1[0:100, :], Timeperstep)
                        LayerNeuronEnergy[0] = LayerNeuronEnergy[0] + NeuronA.Energy
                        NeuronA.RestartEnergy()
                    if Neuron == 'B': # Calculate Neuron energy(for 100 data(memory issue))
                        NeuronB.UpdateEnergy(NEXTSPIKE[0:100, :], Vmem1[0:100, :], Timeperstep)
                        LayerNeuronEnergy[0] = LayerNeuronEnergy[0] + NeuronB.Energy
                        NeuronB.RestartEnergy()
                    if Neuron == 'C': # Calculate Neuron energy(for 100 data(memory issue))
                        NeuronC.UpdateEnergy(NEXTSPIKE[0:100, :], Vmem1[0:100, :], Timeperstep)
                        LayerNeuronEnergy[0] = LayerNeuronEnergy[0] + NeuronC.Energy
                        NeuronC.RestartEnergy()
                    LayerSynapseEnergy[0] += DeviCe.Energy
                    DeviCe.RestartEnergy()

                    ##################2nd Layer####################################
                    DeviCe.UpdateEnergy(NEXTSPIKE[0:100, :], Weight2, Device, Vdd, SpWid) # Calculate Synapse energy(for 100 data(memory issue))
                    Current = torch.einsum('ab, cb -> ac', NEXTSPIKE, DeviCe.Current(Weight2, Device, Vdd, ReadNoise, OffCurrent)) * 0.025
                    NEXTSPIKE2 = (((Vmem2 * Capmem) + (Current * SpWid)) / (Capmem * VTh))
                    NEXTSPIKE2 = torch.where(NEXTSPIKE2 > 0, NEXTSPIKE2, torch.zeros(Batch, 10).to(device))
                    NEXTSPIKE2 = torch.floor(NEXTSPIKE2)
                    Vmem2 = Vmem2 + (Current * SpWid) / Capmem
                    Vmem2 = torch.where(Vmem2 < 0, torch.zeros(Batch, 10).to(device), Vmem2)
                    NEXTSPIKE2 = torch.where(NEXTSPIKE2 != 0, torch.ones(Batch, 10).to(device), NEXTSPIKE2)
                    Vmem1 = torch.where(Vmem1 > VTh, torch.ones(Batch, 256).to(device) * (VTh - 0.01), Vmem1)
                    if SPIKELOSS != 0: # Consider Stochastic spike loss in the circuits
                        SPKL2 = torch.rand(Batch, 10).to(device)
                        SPKL2 = torch.where(SPKL2 > SPIKELOSS, torch.ones(Batch, 10).to(device), torch.zeros(Batch, 10).to(device))
                        NEXTSPIKE2 = torch.mul(NEXTSPIKE2, SPKL2)
                    ISit += NEXTSPIKE2 + Vmem2 / VTh
                    Vmem2 = torch.where(NEXTSPIKE2 != 0, Vmem2 - Vsub, Vmem2)
                    Vmem2 = torch.where(Vmem2 > VTh, torch.ones(Batch, 10).to(device) * (VTh - 0.01), Vmem2)
                    if Neuron == 'A': # Calculate Neuron energy(for 100 data(memory issue))
                        NeuronA.UpdateEnergy(NEXTSPIKE2[0:100, :], Vmem2[0:100, :], Timeperstep)
                        LayerNeuronEnergy[1] = LayerNeuronEnergy[1] + NeuronA.Energy
                        NeuronA.RestartEnergy()
                    if Neuron == 'B': # Calculate Neuron energy(for 100 data(memory issue))
                        NeuronB.UpdateEnergy(NEXTSPIKE2[0:100, :], Vmem2[0:100, :], Timeperstep)
                        LayerNeuronEnergy[1] = LayerNeuronEnergy[1] + NeuronB.Energy
                        NeuronB.RestartEnergy()
                    if Neuron == 'C': # Calculate Neuron energy(for 100 data(memory issue))
                        NeuronC.UpdateEnergy(NEXTSPIKE2[0:100, :], Vmem2[0:100, :], Timeperstep)
                        LayerNeuronEnergy[1] = LayerNeuronEnergy[1] + NeuronC.Energy
                        NeuronC.RestartEnergy()
                    LayerSynapseEnergy[1] += DeviCe.Energy
                    DeviCe.RestartEnergy()

                    # Get Answer from the output neuron response
                    X = torch.ones(Batch).to(device) * torch.argmax(ISit, dim=1) + 1
                    GetAns[:, 0] += torch.mul(torch.where((ISit.max(dim=1).values >= 1) | (i == (Timesteps - 1)), X,torch.zeros(Batch).to(device)),torch.where(GetAns[:, 1] == 0, torch.ones(Batch).to(device),torch.zeros(Batch).to(device)))
                    GetAns[:, 1] += torch.where((ISit.max(dim=1).values >= 1) | (i == (Timesteps - 1)),torch.ones(Batch).to(device) * (i + 1),torch.zeros(Batch).to(device)) * torch.where(GetAns[:, 1] == 0,torch.ones(Batch).to(device),torch.zeros(Batch).to(device))
                    ISit = ISit.floor()

            if Encoding == 'Temp':
                Active1 = torch.ones(Batch, 256).to(device)
                Active2 = torch.ones(Batch, 10).to(device)
                for i in range(Timesteps):
                    InputSpike = torch.where((images > (Timesteps - i - 1) * 1.0 / Timesteps) & (images <= (Timesteps - i) * 1.0 / Timesteps), torch.ones(Batch, 784).to(device), torch.zeros(Batch, 784).to(device))
                    ##################1st Layer####################################
                    DeviCe.UpdateEnergy(InputSpike[0:100, :], Weight1, Device, Vdd, SpWid) # Calculate Synapse energy(for 100 data(memory issue))
                    Current = torch.einsum('ab, cb -> ac', InputSpike, DeviCe.Current(Weight1, Device, Vdd, ReadNoise, OffCurrent)) * Active1 * 0.025
                    NEXTSPIKE = (((Vmem1 * Capmem) + (Current * SpWid)) / (Capmem * VTh))
                    NEXTSPIKE = torch.where(NEXTSPIKE > 0, NEXTSPIKE, torch.zeros(Batch, 256).to(device))
                    NEXTSPIKE = torch.floor(NEXTSPIKE)
                    Active1 = torch.where(NEXTSPIKE > 0, torch.zeros(Batch, 256).to(device), Active1)
                    Vmem1 = Vmem1 + (Current * SpWid) / Capmem
                    Vmem1 = torch.where(Vmem1 < 0, torch.zeros(Batch, 256).to(device), Vmem1)
                    NEXTSPIKE = torch.where(NEXTSPIKE != 0, torch.ones(Batch, 256).to(device), NEXTSPIKE)
                    Vmem1 = torch.where(NEXTSPIKE != 0, Vmem1 - Vsub, Vmem1)
                    Vmem1 = torch.where(Vmem1 > VTh, torch.ones(Batch, 256).to(device) * (VTh - 0.01), Vmem1)
                    if SPIKELOSS != 0: # Consider Stochastic spike loss in the circuits
                        SPKL = torch.rand(Batch, 256).to(device)
                        SPKL = torch.where(SPKL > SPIKELOSS, torch.ones(Batch, 256).to(device),torch.zeros(Batch, 256).to(device))
                        NEXTSPIKE = torch.mul(NEXTSPIKE, SPKL)
                    if Neuron == 'A': # Calculate Neuron energy(for 100 data(memory issue))
                        NeuronA.UpdateEnergy(NEXTSPIKE[0:100, :], Vmem1[0:100, :], Timeperstep)
                        LayerNeuronEnergy[0] = LayerNeuronEnergy[0] + NeuronA.Energy
                        NeuronA.RestartEnergy()
                    if Neuron == 'B': # Calculate Neuron energy(for 100 data(memory issue))
                        NeuronB.UpdateEnergy(NEXTSPIKE[0:100, :], Vmem1[0:100, :], Timeperstep)
                        LayerNeuronEnergy[0] = LayerNeuronEnergy[0] + NeuronB.Energy
                        NeuronB.RestartEnergy()
                    if Neuron == 'C': # Calculate Neuron energy(for 100 data(memory issue))
                        NeuronC.UpdateEnergy(NEXTSPIKE[0:100, :], Vmem1[0:100, :], Timeperstep)
                        LayerNeuronEnergy[0] = LayerNeuronEnergy[0] + NeuronC.Energy
                        NeuronC.RestartEnergy()
                    LayerSynapseEnergy[0] += DeviCe.Energy
                    DeviCe.RestartEnergy()

                    ##################2nd Layer####################################
                    DeviCe.UpdateEnergy(NEXTSPIKE[0:100, :], Weight2, Device, Vdd, SpWid) # Calculate Synapse energy(for 100 data(memory issue))
                    Current = torch.einsum('ab, cb -> ac', NEXTSPIKE, DeviCe.Current(Weight2, Device, Vdd, ReadNoise, OffCurrent)) * Active2 * 0.025
                    NEXTSPIKE2 = (((Vmem2 * Capmem) + (Current * SpWid)) / (Capmem * VTh))
                    NEXTSPIKE2 = torch.where(NEXTSPIKE2 > 0, NEXTSPIKE2, torch.zeros(Batch, 10).to(device))
                    NEXTSPIKE2 = torch.floor(NEXTSPIKE2)
                    Active2 = torch.where(NEXTSPIKE2 > 0, torch.zeros(Batch, 10).to(device), Active2)
                    Vmem2 = Vmem2 + (Current * SpWid) / Capmem
                    Vmem2 = torch.where(Vmem2 < 0, torch.zeros(Batch, 10).to(device), Vmem2)
                    NEXTSPIKE2 = torch.where(NEXTSPIKE2 != 0, torch.ones(Batch, 10).to(device), NEXTSPIKE2)
                    if SPIKELOSS != 0: # Consider Stochastic spike loss in the circuits
                        SPKL2 = torch.rand(Batch, 10).to(device)
                        SPKL2 = torch.where(SPKL2 > SPIKELOSS, torch.ones(Batch, 10).to(device), torch.zeros(Batch, 10).to(device))
                        NEXTSPIKE2 = torch.mul(NEXTSPIKE2, SPKL2)
                    ISit += NEXTSPIKE2 + Vmem2 / VTh
                    Vmem2 = torch.where(NEXTSPIKE2 != 0, Vmem2 - Vsub, Vmem2)
                    Vmem2 = torch.where(Vmem2 > VTh, torch.ones(Batch, 10).to(device) * (VTh - 0.01), Vmem2)
                    if Neuron == 'A': # Calculate Neuron energy(for 100 data(memory issue))
                        NeuronA.UpdateEnergy(NEXTSPIKE2[0:100, :], Vmem2[0:100, :], Timeperstep)
                        LayerNeuronEnergy[1] = LayerNeuronEnergy[1] + NeuronA.Energy
                        NeuronA.RestartEnergy()
                    if Neuron == 'B': # Calculate Neuron energy(for 100 data(memory issue))
                        NeuronB.UpdateEnergy(NEXTSPIKE2[0:100, :], Vmem2[0:100, :], Timeperstep)
                        LayerNeuronEnergy[1] = LayerNeuronEnergy[1] + NeuronB.Energy
                        NeuronB.RestartEnergy()
                    if Neuron == 'C': # Calculate Neuron energy(for 100 data(memory issue))
                        NeuronC.UpdateEnergy(NEXTSPIKE2[0:100, :], Vmem2[0:100, :], Timeperstep)
                        LayerNeuronEnergy[1] = LayerNeuronEnergy[1] + NeuronC.Energy
                        NeuronC.RestartEnergy()
                    LayerSynapseEnergy[1] += DeviCe.Energy
                    DeviCe.RestartEnergy()

                    # Get Answer from the output neuron response
                    X = torch.ones(Batch).to(device) * torch.argmax(ISit, dim = 1) + 1
                    Y = torch.ones(Batch).to(device) * torch.argmax(Vmem2, dim = 1) + 1
                    GetAns[:, 0] += torch.mul(torch.where((ISit.max(dim = 1).values >= 1), X, torch.zeros(Batch).to(device)), torch.where(GetAns[:,1] == 0, torch.ones(Batch).to(device), torch.zeros(Batch).to(device)))
                    GetAns[:, 0] += torch.mul(torch.where(torch.ones(Batch).to(device)* i == (Timesteps - 1), Y, torch.zeros(Batch).to(device)), torch.where(GetAns[:,1] == 0, torch.ones(Batch).to(device), torch.zeros(Batch).to(device)))
                    GetAns[:, 1] += torch.where((ISit.max(dim = 1).values >= 1) | (i == (Timesteps-1)), torch.ones(Batch).to(device) * (i + 1), torch.zeros(Batch).to(device)) * torch.where(GetAns[:,1] == 0, torch.ones(Batch).to(device), torch.zeros(Batch).to(device))
                    ISit = ISit.floor()

            for i in range(0, Batch):
                if (GetAns[i, 0]-1) == labels[i]:
                    Corr += 1

    #Print SNN performance metrics
    Accuracy = Corr * 1.0 / 10000
    Delay = GetAns[:, 1].mean()
    print('Accuracy : %.2f %%' %(Accuracy*100)) # Accuracy
    print('Delay : %f us' %(Delay.item()*Timeperstep*1000000)) # Latency for single inference (1000000 for us)
    print('1st Synapse Power : %f uW' %(LayerSynapseEnergy[0].item()/(100.0*Timeperstep*Timesteps)*1000000)) # 1st Synapse Power (for single data) (1000000 for uW)
    print('2nd Synapse Power : %f uW' %(LayerSynapseEnergy[1].item()/(100.0*Timeperstep*Timesteps)*1000000)) # 2nd Synapse Power (for single data) (1000000 for uW)
    print('1st Neuron Power : %f uW' %(LayerNeuronEnergy[0].item()/(100.0*Timeperstep*Timesteps)*1000000)) # 1st Neuron Power (for single data) (1000000 for uW)
    print('2nd Neuron Power : %f uW' %(LayerNeuronEnergy[1].item()/(100.0*Timeperstep*Timesteps)*1000000)) # 2nd Neuron Power (for single data) (1000000 for uW)
    print('1st Synapse Area : %f mm^2' %(DeviCe.Area(784, 256, Tech, Device)) ) # 1st Synapse Area
    print('2nd Synapse Area : %f mm^2' %(DeviCe.Area(256, 10, Tech, Device))) # 2nd Synapse Area
    if Neuron == 'A':
        print('1st Neuron Area : %f mm^2' %(NeuronA.Area(256, Tech, Encoding))) # 1st Neuron Area
        print('2nd Neuron Area : %f mm^2' %(NeuronA.Area(10, Tech, Encoding))) # 2nd Neuron Area
    if Neuron == 'B':
        print('1st Neuron Area : %f mm^2' %(NeuronB.Area(256, Tech, Encoding))) # 1st Neuron Area
        print('2nd Neuron Area : %f mm^2' %(NeuronB.Area(10, Tech, Encoding))) # 2nd Neuron Area
    if Neuron == 'C':
        print('1st Neuron Area : %f mm^2' %(NeuronC.Area(256, Tech, Encoding))) # 1st Neuron Area
        print('2nd Neuron Area : %f mm^2' %(NeuronC.Area(10, Tech, Encoding))) # 2nd Neuron Area

