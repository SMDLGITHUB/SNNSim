import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

Capmem = 0.00094e-12 # 45nm Rate ReRAM(0.6e-12) FeFET(0.25e-12) Flash(0.02e-12), Temporal ReRAM(X) FeFET(0.024e-12) Flash(0.00094e-12)
                   # 22nm Rate ReRAM(0.3e-12) FeFET(0.2e-12) Flash(0.02e-12), Temporal ReRAM(X) FeFET(0.026e-12) Flash(0.00106e-12)
Cappulse = 0.1e-12 # Capacitance of Pulse capacitor of neuron circuits
SpWid = 10e-9 #Voltage Spike Width
Tech = 45

Energy = torch.zeros(1).to(device)
Is = torch.zeros(100).to(device)

if Tech == 45:
    Vdd = 1
    VTh = 0.54 #Threshold of Neuron circuit
    Vini = 0.25 # initial voltage of Neuron circuit
    Vsub = 0.29 # Amount of voltage subtraction When a spike fires
    #Shoot-through Current according to membrane voltage
    Is[0] = 7.66e-10
    Is[1] = 3.52e-9
    Is[2] = 4.34e-9
    Is[3] = 5.14e-9
    Is[4] = 6.76e-9
    Is[5] = 8.82e-9
    Is[6] = 1.09e-8
    Is[7] = 1.32e-8
    Is[8] = 1.53e-8
    Is[9] = 2.06e-8
    Is[10] = 2.79e-8
    Is[11] = 3.52e-8
    Is[12] = 4.36e-8
    Is[13] = 5.09e-8
    Is[14] = 6.85e-8
    Is[15] = 9.23e-8
    Is[16] = 1.16e-7
    Is[17] = 1.43e-7
    Is[18] = 1.67e-7
    Is[19] = 2.20e-7
    Is[20] = 2.91e-7
    Is[21] = 3.61e-7
    Is[22] = 4.41e-7
    Is[23] = 5.11e-7
    Is[24] = 6.52e-7
    Is[25] = 8.30e-7
    Is[26] = 1.04e-6
    Is[27] = 1.22E-6
    Is[28] = 1.39e-6
    Is[29] = 1.71e-6
    Is[30] = 2.08e-6
    Is[31] = 2.46e-6
    Is[32] = 2.89e-6
    Is[33] = 3.27e-6
    Is[34] = 3.84e-6
    Is[35] = 4.46e-6
    Is[36] = 5.18e-6
    Is[37] = 5.80e-6
    Is[38] = 6.43e-6
    Is[39] = 7.26e-6
    Is[40] = 8.09e-6
    Is[41] = 9.04e-6
    Is[42] = 9.88e-6
    Is[43] = 1.07e-5
    Is[44] = 1.17e-5
    Is[45] = 1.27e-5
    Is[46] = 1.37e-5
    Is[47] = 1.47e-5
    Is[48] = 1.57e-5
    Is[49] = 1.67e-5
    Is[50] = 1.78e-5
    Is[51] = 1.89e-5
    Is[52] = 2.00e-5
    Is[53] = 2.12e-5
    Is[54] = 2.25e-5

if Tech == 22:
    Vdd = 0.8
    VTh = 0.435
    Vini = 0.23
    Vsub = 0.2
    #Shoot-through Current according to membrane voltage
    Is[0] = 2.10e-9
    Is[1] = 9.21e-9
    Is[2] = 1.05e-8
    Is[3] = 1.23e-8
    Is[4] = 1.63e-8
    Is[5] = 2.03e-8
    Is[6] = 2.33e-8
    Is[7] = 2.73e-8
    Is[8] = 4.43e-8
    Is[9] = 5.88e-8
    Is[10] = 7.80e-8
    Is[11] = 9.73e-8
    Is[12] = 1.12e-7
    Is[13] = 1.31e-7
    Is[14] = 1.50e-7
    Is[15] = 1.65e-7
    Is[16] = 1.88e-7
    Is[17] = 2.82e-7
    Is[18] = 3.77e-7
    Is[19] = 4.47e-7
    Is[20] = 5.42e-7
    Is[21] = 6.36e-7
    Is[22] = 7.07e-7
    Is[23] = 8.02e-7
    Is[24] = 8.96e-7
    Is[25] = 9.97e-7
    Is[26] = 1.22e-6
    Is[27] = 1.44E-6
    Is[28] = 1.66e-6
    Is[29] = 1.83e-6
    Is[30] = 2.05e-6
    Is[31] = 2.30e-6
    Is[32] = 2.53e-6
    Is[33] = 2.85e-6
    Is[34] = 3.16e-6
    Is[35] = 3.48e-6
    Is[36] = 3.71e-6
    Is[37] = 4.07e-6
    Is[38] = 4.44e-6
    Is[39] = 4.72e-6
    Is[40] = 5.09e-6
    Is[41] = 5.46e-6
    Is[42] = 5.83e-6
    Is[43] = 6.10e-6

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
        Area += (1e-6 +2e-6) * Tech * Tech * 1e-6 * 7 * 5

        #Capacitor
        Area += ((Capmem + Cappulse) * 4e-9) / (8.85e-12 * 6 * 3.9) * 1000000

        #SRAM
        Area += 1e-6 * 1e-6 * Tech * Tech * 100 # SRAM (assume ~100F^2)

        #Neuron Numbers
        Area *= a
        return Area
