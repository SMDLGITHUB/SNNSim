import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

Capmem = 0.0034e-12 #45nm Rate ReRAM(0.6e-12) FeFET(0.4e-12) Flash(0.04e-12) Temp ReRAM(X) FeFET(0.08e-12) Flash(0.0034e-12)
                   #22nm Rate ReRAM(1.2e-12) FeFET(0.8e-12) Flash(0.04e-12) Temp ReRAM(X) FeFET(0.11e-12) Flash(0.00435e-12)
Cappulse = 0.4e-12 # Capacitance of Pulse capacitor of neuron circuits
Spwid = 10e-9 #Voltage Spike Width

Energy = torch.zeros(1).to(device)

Is = torch.zeros(100).to(device)
Tech = 45


if Tech == 45:
    Vdd = 1
    VTh = 0.22 #Threshold of Neuron circuit
    Vini = 0.14 # initial voltage of Neuron circuit
    Vsub = 0.08 # Amount of voltage subtraction When a spike fires
    #Shoot-through Current according to membrane voltage
    Is[0] = 3.48e-10
    Is[1] = 9.94e-8
    Is[2] = 1.15e-7
    Is[3] = 1.35e-7
    Is[4] = 1.57e-7
    Is[5] = 1.84e-7
    Is[6] = 2.19e-7
    Is[7] = 2.53e-7
    Is[8] = 3.02e-7
    Is[9] = 3.56e-7
    Is[10] = 4.21e-7
    Is[11] = 5.03e-7
    Is[12] = 5.79e-7
    Is[13] = 6.98e-7
    Is[14] = 8.21e-7
    Is[15] = 9.75e-7
    Is[16] = 1.17e-6
    Is[17] = 1.36e-6
    Is[18] = 1.70e-6
    Is[19] = 2.08e-6
    Is[20] = 2.91e-6
    Is[21] = 4.12e-6
    Is[22] = 5.42e-6

if Tech == 22:
    Vdd = 0.8
    VTh = 0.11
    Vini = 0.06
    Vsub = 0.043
    Is[0] = 1.95e-9
    Is[1] = 7.99e-8
    Is[2] = 8.89e-8
    Is[3] = 1.01e-7
    Is[4] = 1.15e-7
    Is[5] = 1.30e-7
    Is[6] = 1.54e-7
    Is[7] = 1.81e-7
    Is[8] = 2.24e-7
    Is[9] = 2.91e-7
    Is[10] = 3.58e-7
    Is[11] = 4.49e-7

Area = 0

def UpdateEnergy(SPIKENUM, Vmem, Time):
    global Energy
    Vmemint = Vmem * 100
    Vmemint = Vmemint.long()
    Energy += torch.sum(Cappulse * Vdd * Vdd * SPIKENUM * 0.6)
    Energy += torch.sum(Is[Vmemint] * Vdd * Time)

def RestartEnergy():

    global Energy
    Energy *= 0

def Area(a, Tech, method):
    Area = 0.0
    if method == 'Rate':
        # Current Mirrors
        Area += (20e-6 +2e-6) * Tech * 1500e-6 * 11 * 2
        Area += (1e-6 +2e-6) * Tech * 1500e-6 * 7 * 2
        Area += (1e-6 +2e-6) * Tech * 1500e-6 * 7 * 2

        #Neuron
        Area += (1e-6 +2e-6) * Tech * Tech *1e-6 * 7 * 5
        Area += (2e-6 +2e-6) * Tech * Tech * 1e-6 * 7 * 5
        Area += (2e-6 +2e-6) * Tech * Tech * 1e-6 * 7 * 2

        #Capacitor
        Area += ((Capmem + Cappulse) * 4e-9) / (8.85e-12 * 6 * 3.9) * 1000000

        #Neuron Numbers
        Area *= a
        return Area

    elif method == 'Temp':
        # Current Mirrors
        Area += (20e-6 +2e-6) * Tech * 1500e-6 * 11 * 2
        Area += (1e-6 +2e-6) * Tech * 1500e-6 * 7 * 2
        Area += (1e-6 +2e-6) * Tech * 1500e-6 * 7 * 2

        #Neuron
        Area += (1e-6 +2e-6) * Tech * Tech *1e-6 * 7 * 5
        Area += (2e-6 +2e-6) * Tech * Tech * 1e-6 * 7 * 5
        Area += (1e-6 +2e-6) * Tech * Tech * 1e-6 * 7 * 2

        #Capacitor
        Area += ((Capmem + Cappulse) * 4e-9) / (8.85e-12 * 3.9 * 6) * 1000000

        #SRAM
        Area += 1.0e-6 * 1e-6 * Tech * Tech * 100 # SRAM

        #Neuron Numbers
        Area *= a
        return Area
