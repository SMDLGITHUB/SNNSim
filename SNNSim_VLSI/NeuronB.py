import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

                  #45nm Rate ReRAM(0.4e-12) FeFET(0.2e-12) Flash(0.04e-12), Temporal ReRAM(X) FeFET(0.0325e-12) Flash(0.00132e-12)
Capmem = 0.035e-12 #22nm Rate ReRAM(0.5e12) FeFET(0.3e-12) Flash(0.04e-12), Temporal ReRAM(X) FeFET(0.035e-12) Flash(0.00151e-12)
Spwid = 10e-9 #Voltage Spike Width

Energy = torch.zeros(1).to(device)

Is = torch.zeros(100).to(device)
Tech = 45

if Tech == 45:
    Vdd = 1
    VTh = 0.685  #Threshold of Neuron circuit
    Vini = 0.48 # initial voltage of Neuron circuit
    Vsub = 0.2 # Amount of voltage subtraction When a spike fires
    #Shoot-through Current according to membrane voltage
    Is[0] = 6.09e-11
    Is[1] = 9.54e-10
    Is[2] = 1.32e-9
    Is[3] = 1.68e-9
    Is[4] = 2.13e-9
    Is[5] = 3.16e-9
    Is[6] = 4.18e-9
    Is[7] = 5.36e-9
    Is[8] = 6.38e-9
    Is[9] = 7.71e-9
    Is[10] = 1.13e-8
    Is[11] = 1.49e-8
    Is[12] = 1.90e-8
    Is[13] = 2.26e-8
    Is[14] = 2.72e-8
    Is[15] = 3.89e-8
    Is[16] = 5.06e-8
    Is[17] = 6.40e-8
    Is[18] = 7.57e-8
    Is[19] = 9.04e-8
    Is[20] = 1.25e-7
    Is[21] = 1.60e-7
    Is[22] = 1.99e-7
    Is[23] = 2.34e-7
    Is[24] = 2.77e-7
    Is[25] = 3.68e-7
    Is[26] = 4.58e-7
    Is[27] = 5.62E-7
    Is[28] = 6.52e-7
    Is[29] = 7.62e-7
    Is[30] = 9.60e-7
    Is[31] = 1.16e-6
    Is[32] = 1.39e-6
    Is[33] = 1.58e-6
    Is[34] = 1.82e-6
    Is[35] = 2.16e-6
    Is[36] = 2.56e-6
    Is[37] = 2.90e-6
    Is[38] = 3.25e-6
    Is[39] = 3.65e-6
    Is[40] = 4.12e-6
    Is[41] = 4.66e-6
    Is[42] = 5.14e-6
    Is[43] = 5.61e-6
    Is[44] = 6.15e-6
    Is[45] = 6.70e-6
    Is[46] = 7.32e-6
    Is[47] = 7.86e-6
    Is[48] = 8.48e-6
    Is[49] = 9.03e-6
    Is[50] = 9.61e-6
    Is[51] = 1.03e-5
    Is[52] = 1.09e-5
    Is[53] = 1.15e-5
    Is[54] = 1.21e-5
    Is[55] = 1.28e-5
    Is[56] = 1.34e-5
    Is[57] = 1.40e-5
    Is[58] = 1.47e-5
    Is[59] = 1.53e-5
    Is[60] = 1.60e-5
    Is[61] = 1.66e-5
    Is[62] = 1.73e-5
    Is[63] = 1.79e-5
    Is[64] = 1.85e-5
    Is[65] = 1.94e-5
    Is[66] = 2.02e-5
    Is[67] = 2.10e-5
    Is[68] = 2.18e-5

if Tech == 22:
    Vdd = 0.8
    VTh = 0.514
    Vini = 0.37
    Vsub = 0.14
    Is[0] = 2.77e-10
    Is[1] = 2.61e-9
    Is[2] = 3.37e-9
    Is[3] = 4.22e-9
    Is[4] = 5.74e-9
    Is[5] = 7.43e-9
    Is[6] = 9.30e-9
    Is[7] = 1.10e-8
    Is[8] = 1.48e-8
    Is[9] = 1.92e-8
    Is[10] = 2.40e-8
    Is[11] = 2.83e-8
    Is[12] = 3.78e-8
    Is[13] = 4.84e-8
    Is[14] = 6.02e-8
    Is[15] = 7.08e-8
    Is[16] = 9.25e-8
    Is[17] = 1.17e-7
    Is[18] = 1.43e-7
    Is[19] = 1.68e-7
    Is[20] = 2.13e-7
    Is[21] = 2.62e-7
    Is[22] = 3.18e-7
    Is[23] = 3.67e-7
    Is[24] = 4.50e-7
    Is[25] = 5.39e-7
    Is[26] = 6.38e-7
    Is[27] = 7.28E-7
    Is[28] = 8.57e-7
    Is[29] = 9.94e-7
    Is[30] = 1.15e-6
    Is[31] = 1.28e-6
    Is[32] = 1.46e-6
    Is[33] = 1.66e-6
    Is[34] = 1.84e-6
    Is[35] = 2.03e-6
    Is[36] = 2.24e-6
    Is[37] = 2.48e-6
    Is[38] = 2.70e-6
    Is[39] = 2.92e-6
    Is[40] = 3.17e-6
    Is[41] = 3.44e-6
    Is[42] = 3.68e-6
    Is[43] = 3.93e-6
    Is[44] = 4.22e-6
    Is[45] = 4.48e-6
    Is[46] = 4.74e-6
    Is[47] = 5.04e-6
    Is[48] = 5.31e-6
    Is[49] = 5.59e-6
    Is[50] = 5.89e-6
    Is[51] = 6.17e-6

def UpdateEnergy(SPIKENUM, Vmem, Time):
    global Energy
    Vmemint = Vmem * 100
    Vmemint = Vmemint.long()
    Energy += torch.sum(Is[Vmemint] * Vdd * Time)


def RestartEnergy():
    global Energy
    Energy *= 0

def Area(a, Tech, method):
    Area = 0.0
    if method == 'Rate':
        # Current Mirrors
        Area += (20e-6 +2e-6)* Tech * 1500e-6 * 11 * 2
        Area += (1e-6 +2e-6)* Tech * 1500e-6 * 7 * 2
        Area += (1e-6 +2e-6)* Tech * 1500e-6 * 7 * 2

        #Neuron
        Area += (1e-6 +2e-6) * 1e-6 *Tech * Tech * 7 * 3 # NMOS
        Area += (2e-6 +2e-6) * 1e-6 *Tech * Tech * 7 * 2 # PMOS
        Area += (3e-6 +2e-6) * 1e-6 *Tech * Tech * 7 * 1 # PMOS

        #Capacitor
        Area += (Capmem * 4e-9) / (8.85e-12 * 6 * 3.9) * 1000000

        #Neuron Numbers
        Area *= a
        return Area

    elif method == 'Temp':
        # Current Mirrors
        Area += (20e-6 +2e-6) * Tech * 1500e-6 * 11 * 2
        Area += (1e-6 +2e-6) * Tech * 1500e-6 * 7 * 2
        Area += (1e-6 +2e-6) * Tech * 1500e-6 * 7 * 2

        #Neuron
        Area += (1e-6 +2e-6) * 1e-6 *Tech * Tech * 7 * 3 # NMOS
        Area += (2e-6 +2e-6) * 1e-6 *Tech * Tech * 7 * 2 # PMOS
        Area += (3e-6 +2e-6) * 1e-6 *Tech * Tech * 7 * 1 # PMOS

        #Capacitor
        Area += (Capmem * 4e-9) / (8.85e-12 * 6 * 3.9) * 1000000

        #SRAM
        Area += 1.0e-6 * 1e-6 * Tech * Tech * 100 # SRAM

        #Neuron Numbers
        Area *= a
        return Area
