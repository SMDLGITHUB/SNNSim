import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Current(Weight, Device, Vdd, ReadNoise, OnOff):
    if Device == 'Flash':
        Ron = 1000000
        On_Off = 0.0001

        if ReadNoise == True: # Read noise of the device is applied here
            readnoise = torch.empty(Weight.size()).normal_(mean=0, std=0.1).to(device)
            Weight *= (1 + readnoise)
        if OnOff == True: # On_Off ratio of the device is considered here
            if type(Weight) != int:
                Weight = torch.where(torch.abs(Weight) < On_Off, Weight*0, Weight )
                Weight = torch.where(Weight > On_Off ,Weight- On_Off, Weight)
                Weight = torch.where(Weight < On_Off*(-1), Weight+ On_Off, Weight)
        Curr = (Vdd / Ron) * Weight
        return Curr

    if Device == 'FeFET' :
        Ron = 33000
        On_Off = 0.0125
        if ReadNoise == True:
            readnoise = torch.empty(Weight.size()).normal_(mean=0, std=0.1).to(device)
            Weight *= (1 + readnoise)
        if OnOff == True:
            if type(Weight) != int:
                Weight = torch.where(torch.abs(Weight) < On_Off, Weight*0, Weight )
                Weight = torch.where(Weight > On_Off ,Weight- On_Off, Weight)
                Weight = torch.where(Weight < On_Off*(-1), Weight+ On_Off, Weight)
        Curr = (Vdd / Ron) * Weight
        return Curr

    if Device == 'ReRAM' :
        Ron = 17000
        On_Off = 0.226
        if ReadNoise == True:
            readnoise = torch.empty(Weight.size()).normal_(mean=0, std=0.1).to(device)
            Weight *= (1 + readnoise)
        if OnOff == True:
            if type(Weight) != int:
                Weight = torch.where(torch.abs(Weight) < On_Off, Weight*0, Weight )
                Weight = torch.where(Weight > On_Off ,Weight- On_Off, Weight)
                Weight = torch.where(Weight < On_Off*(-1), Weight+ On_Off, Weight)
        Curr = (Vdd / Ron) * Weight
        return Curr

Scale = 0.025 #Currents are scaled down for 1/40 size with current mirrors
Energy = torch.zeros(1).to(device) #global parameter to store energy

def UpdateEnergy(NUMinp, weight, Device, Vdd, time):
    global Energy
    global Scale
    if Device == 'ReRAM' :
        On_Off = 0.226
    if Device == 'FeFET' :
        On_Off = 0.0125
    if Device == 'Flash':
        On_Off = 0.0001
    # For Fully Connected Layers
    ONE = torch.ones(weight.size()).to(device)
    weightassist = torch.where(torch.abs(weight) <On_Off, ONE *On_Off * 2, torch.abs(weight)+On_Off ) # OffCurrent flows in both arrays for weight '0'
    Energyassist = torch.einsum('ab, cb -> a', NUMinp, Current(weightassist, Device, Vdd, False, False)) * time * Vdd * (1 + Scale)
    Energy = torch.sum(Energyassist)

def RestartEnergy():
    global Energy
    Energy *= 0 #Restart Energy

def Area(a, b, Tech, Device):
    #Tech
    if Device == 'ReRAM':
        Size = 12 # 12F^2 for 1T1R
    if Device == 'FeFET':
        Size = 6 # 6F^2 for AND-Type FeFET
    if Device == 'Flash':
        Size = 6 # 6F^2 for AND-Type Flash

    return a * b * 2 * Size * Tech * Tech * 1e-12 # (mm^2)2 arrays for (positive, negative) weights