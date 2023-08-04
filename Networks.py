import torch
import torch.nn as nn
import math
device = 'cuda' if torch.cuda.is_available() else 'cpu'


##############Functions Reflecting Hardware System####################

class SNNLinear(nn.Module):
    def __init__(self, insize, outsize, const, arg):
        super().__init__()
        self.fc = nn.Linear(insize, outsize, bias = False)
        self.relu = nn.ReLU()
        self.SynapseNum = insize * outsize * 2
        self.OnCurr = arg.Oncurrent
        self.OffCurr = arg.Offcurrent
        self.SpWid = arg.SpikeWidth
        self.Neuron = arg.Neuron
        self.time = arg.TimeResol
        self.Cappulse = 1e-13
        self.Capmem = arg.Capmem
        self.mode = arg.train
        self.const = const
        if self.Neuron == 'TypeA':
            if arg.Tech == 500:
                self.Vdd = 2.5
                self.Vth = 1.19
                self.Vini = 0.2
                self.Vsub = 1  # 0.29
                self.Is = torch.zeros(121)
                self.Is[0] = 5.51e-11
                self.Is[1] = 5.51e-11
                self.Is[2] = 5.51e-11
                self.Is[3] = 5.51e-11
                self.Is[4] = 5.51e-11
                self.Is[5] = 5.51e-11
                self.Is[6] = 5.51e-11
                self.Is[7] = 5.51e-11
                self.Is[8] = 5.51e-11
                self.Is[9] = 5.51e-11
                self.Is[11] = 5.51e-11
                self.Is[12] = 5.51e-11
                self.Is[13] = 5.51e-11
                self.Is[14] = 5.51e-11
                self.Is[15] = 5.51e-11
                self.Is[16] = 5.51e-11
                self.Is[17] = 5.51e-11
                self.Is[18] = 5.51e-11
                self.Is[19] = 5.51e-11
                self.Is[20] = 5.51e-11
                self.Is[21] = 5.51e-11
                self.Is[22] = 5.51e-11
                self.Is[23] = 5.51e-11
                self.Is[24] = 5.51e-11
                self.Is[25] = 1.01e-9
                self.Is[26] = 3.58e-9
                self.Is[27] = 7.06E-9
                self.Is[28] = 1.14e-8
                self.Is[29] = 1.70e-8
                self.Is[30] = 2.24e-8
                self.Is[31] = 2.90e-8
                self.Is[32] = 3.57e-8
                self.Is[33] = 4.28e-8
                self.Is[34] = 5.40e-8
                self.Is[35] = 6.51e-8
                self.Is[36] = 7.41e-8
                self.Is[37] = 8.53e-8
                self.Is[38] = 9.79e-8
                self.Is[39] = 1.14e-7
                self.Is[40] = 1.29e-7
                self.Is[41] = 1.42e-7
                self.Is[42] = 1.58e-7
                self.Is[43] = 1.79e-7
                self.Is[44] = 2.01e-7
                self.Is[45] = 2.19e-7
                self.Is[46] = 2.41e-7
                self.Is[47] = 2.63e-7
                self.Is[48] = 2.82e-7
                self.Is[49] = 3.12e-7
                self.Is[50] = 3.37e-7
                self.Is[51] = 3.67e-7
                self.Is[52] = 3.97e-7
                self.Is[53] = 4.22e-7
                self.Is[54] = 4.88e-7
                self.Is[55] = 5.86e-7
                self.Is[56] = 6.29e-7
                self.Is[57] = 7.72e-7
                self.Is[58] = 8.07e-7
                self.Is[59] = 9.50e-7
                self.Is[60] = 1.24e-6
                self.Is[61] = 1.28e-6
                self.Is[62] = 1.36e-6
                self.Is[63] = 1.44e-6
                self.Is[64] = 1.58e-6
                self.Is[65] = 1.64e-6
                self.Is[66] = 1.69e-6
                self.Is[67] = 1.83e-6
                self.Is[68] = 1.98e-6
                self.Is[69] = 2.14e-6
                self.Is[70] = 2.21e-6
                self.Is[71] = 2.28e-6
                self.Is[72] = 2.36e-6
                self.Is[73] = 2.44e-6
                self.Is[74] = 2.50e-6
                self.Is[75] = 2.58e-6
                self.Is[76] = 2.66e-6
                self.Is[77] = 2.93e-6
                self.Is[78] = 3.29e-6
                self.Is[79] = 3.31e-6
                self.Is[80] = 3.39e-6
                self.Is[81] = 3.52e-6
                self.Is[82] = 3.63e-6
                self.Is[83] = 3.74e-6
                self.Is[84] = 3.83e-6
                self.Is[85] = 3.94e-6
                self.Is[86] = 4.06e-6
                self.Is[87] = 4.20e-6
                self.Is[88] = 4.32e-6
                self.Is[89] = 4.48e-6
                self.Is[90] = 4.64e-6
                self.Is[91] = 4.77e-6
                self.Is[92] = 4.93e-6
                self.Is[93] = 5.09e-6
                self.Is[94] = 5.25e-6
                self.Is[95] = 5.38e-6
                self.Is[96] = 5.61e-6
                self.Is[97] = 5.84e-6
                self.Is[98] = 6.07e-6
                self.Is[99] = 6.26e-6
                self.Is[100] = 6.49e-6
                self.Is[101] = 6.72e-6
                self.Is[102] = 6.96e-6
                self.Is[103] = 7.14e-6 + 9.0e-8
                self.Is[104] = 7.43e-6 + 1.52e-7
                self.Is[105] = 7.75e-6 + 2.25e-7
                self.Is[106] = 8.03e-6 + 3.33e-7
                self.Is[107] = 8.32e-6 + 4.62e-7
                self.Is[108] = 8.5e-6 + 5.08e-7
                self.Is[109] = 8.7e-6 + 7.11e-7
                self.Is[110] = 8.92e-6 + 9.48e-7
                self.Is[111] = 9.31e-6 + 1.11e-6
                self.Is[112] = 9.62e-6 + 1.61e-6
                self.Is[113] = 9.94e-6 + 1.81e-6
                self.Is[114] = 10.2e-6 + 2.44e-6
                self.Is[115] = 10.30e-6 + 2.67e-6
                self.Is[116] = 10.5e-6 + 3.65e-6
                self.Is[117] = 10.8e-6 + 4.21e-6
                self.Is[118] = 11.3e-6 + 6.31e-6
                self.Is[119] = 12.5e-6 + 13.3e-6
                self.Is[120] = 12.5e-6 + 13.3e-6
                
            if arg.Tech == 45:
                self.Vdd = 1
                self.Vth = 0.54
                self.Vini = 0.25
                self.Vsub = 0.29  # 0.29
                self.Is = torch.zeros(55)
                self.Is[0] = 7.66e-10
                self.Is[1] = 3.52e-9
                self.Is[2] = 4.34e-9
                self.Is[3] = 5.14e-9
                self.Is[4] = 6.76e-9
                self.Is[5] = 8.82e-9
                self.Is[6] = 1.09e-8
                self.Is[7] = 1.32e-8
                self.Is[8] = 1.53e-8
                self.Is[9] = 2.06e-8
                self.Is[10] = 2.79e-8
                self.Is[11] = 3.52e-8
                self.Is[12] = 4.36e-8
                self.Is[13] = 5.09e-8
                self.Is[14] = 6.85e-8
                self.Is[15] = 9.23e-8
                self.Is[16] = 1.16e-7
                self.Is[17] = 1.43e-7
                self.Is[18] = 1.67e-7
                self.Is[19] = 2.20e-7
                self.Is[20] = 2.91e-7
                self.Is[21] = 3.61e-7
                self.Is[22] = 4.41e-7
                self.Is[23] = 5.11e-7
                self.Is[24] = 6.52e-7
                self.Is[25] = 8.30e-7
                self.Is[26] = 1.04e-6
                self.Is[27] = 1.22E-6
                self.Is[28] = 1.39e-6
                self.Is[29] = 1.71e-6
                self.Is[30] = 2.08e-6
                self.Is[31] = 2.46e-6
                self.Is[32] = 2.89e-6
                self.Is[33] = 3.27e-6
                self.Is[34] = 3.84e-6
                self.Is[35] = 4.46e-6
                self.Is[36] = 5.18e-6
                self.Is[37] = 5.80e-6
                self.Is[38] = 6.43e-6
                self.Is[39] = 7.26e-6
                self.Is[40] = 8.09e-6
                self.Is[41] = 9.04e-6
                self.Is[42] = 9.88e-6
                self.Is[43] = 1.07e-5
                self.Is[44] = 1.17e-5
                self.Is[45] = 1.27e-5
                self.Is[46] = 1.37e-5
                self.Is[47] = 1.47e-5
                self.Is[48] = 1.57e-5
                self.Is[49] = 1.67e-5
                self.Is[50] = 1.78e-5
                self.Is[51] = 1.89e-5
                self.Is[52] = 2.00e-5
                self.Is[53] = 2.12e-5
                self.Is[54] = 2.25e-5

            if arg.Tech == 22:
                self.Vdd = 0.8
                self.Vsub = 0.2
                self.Vth = 0.435
                self.Vini = 0.23
                self.Is = torch.zeros(44)
                self.Is[0] = 2.10e-9
                self.Is[1] = 9.21e-9
                self.Is[2] = 1.05e-8
                self.Is[3] = 1.23e-8
                self.Is[4] = 1.63e-8
                self.Is[5] = 2.03e-8
                self.Is[6] = 2.33e-8
                self.Is[7] = 2.73e-8
                self.Is[8] = 4.43e-8
                self.Is[9] = 5.88e-8
                self.Is[10] = 7.80e-8
                self.Is[11] = 9.73e-8
                self.Is[12] = 1.12e-7
                self.Is[13] = 1.31e-7
                self.Is[14] = 1.50e-7
                self.Is[15] = 1.65e-7
                self.Is[16] = 1.88e-7
                self.Is[17] = 2.82e-7
                self.Is[18] = 3.77e-7
                self.Is[19] = 4.47e-7
                self.Is[20] = 5.42e-7
                self.Is[21] = 6.36e-7
                self.Is[22] = 7.07e-7
                self.Is[23] = 8.02e-7
                self.Is[24] = 8.96e-7
                self.Is[25] = 9.97e-7
                self.Is[26] = 1.22e-6
                self.Is[27] = 1.44E-6
                self.Is[28] = 1.66e-6
                self.Is[29] = 1.83e-6
                self.Is[30] = 2.05e-6
                self.Is[31] = 2.30e-6
                self.Is[32] = 2.53e-6
                self.Is[33] = 2.85e-6
                self.Is[34] = 3.16e-6
                self.Is[35] = 3.48e-6
                self.Is[36] = 3.71e-6
                self.Is[37] = 4.07e-6
                self.Is[38] = 4.44e-6
                self.Is[39] = 4.72e-6
                self.Is[40] = 5.09e-6
                self.Is[41] = 5.46e-6
                self.Is[42] = 5.83e-6
                self.Is[43] = 6.10e-6
            
    def forward(self, input, Vmem):
        if self.mode == True:
            if not hasattr(self.fc.weight,'org'):
                self.fc.weight.org=self.fc.weight.data.clone()
            self.fc.weight.data.clamp_(-1, 1)
            out = self.fc(input)
            out = self.relu(out)
            return out

        if self.mode == False:
            Vmem *= self.Vini
            Current = self.fc(input) * self.OnCurr * self.const
            weightassist = self.fc.weight
            self.fc.weight = nn.Parameter(torch.abs(self.fc.weight))
            Energyassist = self.fc(input)* self.Vdd * self.OnCurr* self.SpWid * (1+self.const)
            SysE = torch.sum(Energyassist)
            self.fc.weight = nn.Parameter(weightassist)
            print(torch.where(Current > 0, Current, Current*0).sum()/torch.count_nonzero(torch.where(Current > 0, Current, Current*0)))
            print(math.sqrt(torch.numel(Current)) * torch.std(torch.where(Current > 0, Current, torch.where(Current > 0, Current, Current*0).sum()/torch.count_nonzero(torch.where(Current > 0, Current, Current*0))))/math.sqrt(torch.count_nonzero(torch.where(Current > 0, Current, Current*0))))
            NEXTSPIKE = (((Vmem * self.Capmem) + (Current * self.SpWid)) / (self.Capmem * self.Vth))
            NEXTSPIKE = torch.where(NEXTSPIKE > 0, NEXTSPIKE, NEXTSPIKE*0)
            #print(NEXTSPIKE.max())
            NEXTSPIKE = torch.floor(NEXTSPIKE)
            Vmem = Vmem + (Current * self.SpWid) / self.Capmem
            Vmem = torch.where(Vmem < 0, Vmem*0, Vmem)
            NEXTSPIKE = torch.where(NEXTSPIKE != 0, NEXTSPIKE/NEXTSPIKE, NEXTSPIKE)
            Vmem = torch.where(NEXTSPIKE != 0, Vmem - self.Vsub, Vmem)
            Vmem = torch.where(Vmem > self.Vth, Vmem/Vmem * (self.Vth - 0.01), Vmem)
            if self.Neuron == 'TypeA':
                Vmemint = Vmem * 100
                Vmemint = Vmemint.long()
                NeuE = torch.sum(self.Cappulse * self.Vdd * self.Vdd * NEXTSPIKE * 0.6)
                NeuE += torch.sum(self.Is[Vmemint] * self.Vdd * self.time)

            if self.Neuron == 'TypeB':
                Vmem = 1
            if self.Neuron == 'TypeC':
                Vmem = 1
            Vmem = Vmem / self.Vini
            return NEXTSPIKE, SysE, NeuE, Vmem

class SNNConv(nn.Module):
    def __init__(self, inchan, outchan, kern, const, arg):
        super().__init__()
        self.con = nn.Conv2d(inchan, outchan, kern, padding = int((kern - 1)/2), bias = False)
        self.relu = nn.ReLU()
        self.SynapseNum = kern * kern * outchan * inchan * 2
        self.OnCurr = arg.Oncurrent
        self.OffCurr = arg.Offcurrent
        self.SpWid = arg.SpikeWidth
        self.Neuron = arg.Neuron
        self.time = arg.TimeResol
        self.Cappulse = 1e-13
        self.Capmem = arg.Capmem
        self.const = const
        self.mode = arg.train
        if self.Neuron == 'TypeA':
            if arg.Tech == 45:
                self.Vdd = 1
                self.Vth = 0.54
                self.Vini = 0.25
                self.Vsub = 0.29  # 0.29
                self.Is = torch.zeros(55)
                self.Is[0] = 7.66e-10
                self.Is[1] = 3.52e-9
                self.Is[2] = 4.34e-9
                self.Is[3] = 5.14e-9
                self.Is[4] = 6.76e-9
                self.Is[5] = 8.82e-9
                self.Is[6] = 1.09e-8
                self.Is[7] = 1.32e-8
                self.Is[8] = 1.53e-8
                self.Is[9] = 2.06e-8
                self.Is[10] = 2.79e-8
                self.Is[11] = 3.52e-8
                self.Is[12] = 4.36e-8
                self.Is[13] = 5.09e-8
                self.Is[14] = 6.85e-8
                self.Is[15] = 9.23e-8
                self.Is[16] = 1.16e-7
                self.Is[17] = 1.43e-7
                self.Is[18] = 1.67e-7
                self.Is[19] = 2.20e-7
                self.Is[20] = 2.91e-7
                self.Is[21] = 3.61e-7
                self.Is[22] = 4.41e-7
                self.Is[23] = 5.11e-7
                self.Is[24] = 6.52e-7
                self.Is[25] = 8.30e-7
                self.Is[26] = 1.04e-6
                self.Is[27] = 1.22E-6
                self.Is[28] = 1.39e-6
                self.Is[29] = 1.71e-6
                self.Is[30] = 2.08e-6
                self.Is[31] = 2.46e-6
                self.Is[32] = 2.89e-6
                self.Is[33] = 3.27e-6
                self.Is[34] = 3.84e-6
                self.Is[35] = 4.46e-6
                self.Is[36] = 5.18e-6
                self.Is[37] = 5.80e-6
                self.Is[38] = 6.43e-6
                self.Is[39] = 7.26e-6
                self.Is[40] = 8.09e-6
                self.Is[41] = 9.04e-6
                self.Is[42] = 9.88e-6
                self.Is[43] = 1.07e-5
                self.Is[44] = 1.17e-5
                self.Is[45] = 1.27e-5
                self.Is[46] = 1.37e-5
                self.Is[47] = 1.47e-5
                self.Is[48] = 1.57e-5
                self.Is[49] = 1.67e-5
                self.Is[50] = 1.78e-5
                self.Is[51] = 1.89e-5
                self.Is[52] = 2.00e-5
                self.Is[53] = 2.12e-5
                self.Is[54] = 2.25e-5

            if arg.Tech == 22:
                self.Vdd = 0.8
                self.Vsub = 0.2
                self.Vth = 0.435
                self.Vini = 0.23
                self.Is = torch.zeros(44)
                self.Is[0] = 2.10e-9
                self.Is[1] = 9.21e-9
                self.Is[2] = 1.05e-8
                self.Is[3] = 1.23e-8
                self.Is[4] = 1.63e-8
                self.Is[5] = 2.03e-8
                self.Is[6] = 2.33e-8
                self.Is[7] = 2.73e-8
                self.Is[8] = 4.43e-8
                self.Is[9] = 5.88e-8
                self.Is[10] = 7.80e-8
                self.Is[11] = 9.73e-8
                self.Is[12] = 1.12e-7
                self.Is[13] = 1.31e-7
                self.Is[14] = 1.50e-7
                self.Is[15] = 1.65e-7
                self.Is[16] = 1.88e-7
                self.Is[17] = 2.82e-7
                self.Is[18] = 3.77e-7
                self.Is[19] = 4.47e-7
                self.Is[20] = 5.42e-7
                self.Is[21] = 6.36e-7
                self.Is[22] = 7.07e-7
                self.Is[23] = 8.02e-7
                self.Is[24] = 8.96e-7
                self.Is[25] = 9.97e-7
                self.Is[26] = 1.22e-6
                self.Is[27] = 1.44E-6
                self.Is[28] = 1.66e-6
                self.Is[29] = 1.83e-6
                self.Is[30] = 2.05e-6
                self.Is[31] = 2.30e-6
                self.Is[32] = 2.53e-6
                self.Is[33] = 2.85e-6
                self.Is[34] = 3.16e-6
                self.Is[35] = 3.48e-6
                self.Is[36] = 3.71e-6
                self.Is[37] = 4.07e-6
                self.Is[38] = 4.44e-6
                self.Is[39] = 4.72e-6
                self.Is[40] = 5.09e-6
                self.Is[41] = 5.46e-6
                self.Is[42] = 5.83e-6
                self.Is[43] = 6.10e-6

    def forward(self, input, Vmem):
        if self.mode == True:
            if not hasattr(self.con.weight,'org'):
                self.con.weight.org=self.con.weight.data.clone()
            self.con.weight.data.clamp_(-1, 1)
            out = self.con(input)
            out = self.relu(out)
            return out
        if self.mode == False:
            Vmem *= self.Vini
            Current = self.con(input) * self.OnCurr * self.const
            weightassist = self.con.weight
            self.con.weight = nn.Parameter(torch.abs(self.con.weight))
            Energyassist = self.con(input)* self.Vdd * self.OnCurr* self.SpWid * (1+self.const)
            SysE = torch.sum(Energyassist)
            self.con.weight = nn.Parameter(weightassist)
            print(torch.where(Current > 0, Current, Current*0).sum()/torch.count_nonzero(torch.where(Current > 0, Current, Current*0)))
            print(math.sqrt(torch.numel(Current)) * torch.std(torch.where(Current > 0, Current, torch.where(Current > 0, Current, Current*0).sum()/torch.count_nonzero(torch.where(Current > 0, Current, Current*0))))/math.sqrt(torch.count_nonzero(torch.where(Current > 0, Current, Current*0))))
            NEXTSPIKE = (((Vmem * self.Capmem) + (Current * self.SpWid)) / (self.Capmem * self.Vth))
            NEXTSPIKE = torch.where(NEXTSPIKE > 0, NEXTSPIKE, NEXTSPIKE*0)
            #print(NEXTSPIKE.max())
            NEXTSPIKE = torch.floor(NEXTSPIKE)
            Vmem = Vmem + (Current * self.SpWid) / self.Capmem
            Vmem = torch.where(Vmem < 0, Vmem*0, Vmem)
            NEXTSPIKE = torch.where(NEXTSPIKE != 0, NEXTSPIKE/NEXTSPIKE, NEXTSPIKE)
            Vmem = torch.where(NEXTSPIKE != 0, Vmem - self.Vsub, Vmem)
            Vmem = torch.where(Vmem > self.Vth, Vmem/Vmem * (self.Vth - 0.01), Vmem)
            if self.Neuron == 'TypeA':
                Vmemint = Vmem * 100
                Vmemint = Vmemint.long()
                NeuE = torch.sum(self.Cappulse * self.Vdd * self.Vdd * NEXTSPIKE * 0.6)
                NeuE += torch.sum(self.Is[Vmemint] * self.Vdd * self.time)
            if self.Neuron == 'TypeB':
                Vmem = 1
            if self.Neuron == 'TypeC':
                Vmem = 1
            Vmem = Vmem / self.Vini
            return NEXTSPIKE, SysE, NeuE, Vmem


###############################################################################################


#########################Define your Network using above functions#############################

class MLP(nn.Module):
    def __init__(self, input, Hidden, args):
        super().__init__()
        self.fc1 = SNNLinear(input, Hidden, 0.1, arg = args)
        self.fc2 = SNNLinear(Hidden, 10, 0.1, arg = args)
        self.Vmem1 = torch.ones(args.batch, Hidden).to(device)
        self.Vmem2 = torch.ones(args.batch, 10).to(device)
        self.SynpaseEnergy = 0
        self.NeuronEnergy = 0
        self.mode = args.train
        self.Timestep = args.TimeSteps
        self.Batch = args.batch
        self.GetAns = torch.zeros(args.batch, 2).to(device)
        self.Isit = torch.zeros(args.batch, 10).to(device)
    def forward(self, x):
        if self.mode == True:
            x = x.view(-1, self.num_flat_features(x))
            x = self.fc1(x, self.Vmem1)
            x = self.fc2(x, self.Vmem2)
            return nn.functional.log_softmax(x, dim=1)

        if self.mode == False:
            self.Vmem1 *= 0
            self.Vmem1 += 1
            self.Vmem2 *= 0
            self.Vmem2 += 1
            self.Isit *= 0
            self.GetAns *= 0
            for i in range(self.Timestep):
                out = x.view(-1, self.num_flat_features(x))
                out = torch.floor(out + torch.rand(out.size()).to(device)).to(device)
                out, SynapseE, NeuronE, self.Vmem1 = self.fc1(out, self.Vmem1)
                self.SynpaseEnergy += SynapseE
                self.NeuronEnergy += NeuronE
                out, SynapseE, NeuronE, self.Vmem2 = self.fc2(out, self.Vmem2)
                self.SynpaseEnergy += SynapseE
                self.NeuronEnergy += NeuronE
                self.Isit += out + (self.Vmem2 * 0.01 / self.Timestep )
                X = torch.ones(self.Batch).to(device) * torch.argmax(self.Isit, dim=1) + 1
                self.GetAns[:, 0] += torch.mul(torch.where((self.Isit.max(dim=1).values >= 1) | (i == (self.Timestep - 1)),
                X,torch.zeros(self.Batch).to(device)),torch.where(self.GetAns[:, 1] == 0,
                torch.ones(self.Batch).to(device), torch.zeros(self.Batch).to(device)))
                self.GetAns[:, 1] += torch.where((self.Isit.max(dim=1).values >= 1) | (i == (self.Timestep - 1)),
                    torch.ones(self.Batch).to(device) * (i + 1),torch.zeros(self.Batch).to(device)) \
                    * torch.where(self.GetAns[:, 1] == 0,torch.ones(self.Batch).to(device)
                    , torch.zeros(self.Batch).to(device))
            return (self.GetAns[:, 0]-1, self.GetAns[:, 1])

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
##################################################################################
class VGG6(nn.Module):
    def __init__(self, kernel, channel, FCLayer, args):
        super().__init__()
        self.conv1 = SNNConv(3, channel, kernel, 0.1, arg = args)
        self.conv2 = SNNConv(channel, channel, kernel, 0.12, arg = args)
        self.conv3 = SNNConv(channel, channel*2, kernel, 0.5, arg = args)
        self.conv4 = SNNConv(channel*2, channel*2, kernel, 0.4, arg = args)
        self.pool = nn.AvgPool2d(2, 2)
        self.Drop = nn.Dropout2d(p=0.2)
        self.Dropfc = nn.Dropout(p=0.2)
        self.fc1 = SNNLinear(channel*2*8*8, FCLayer, 0.2, arg = args)
        self.fc2 = SNNLinear(FCLayer, 10, 0.18, arg = args)
        self.SynpaseEnergy = 0
        self.NeuronEnergy = 0
        self.Vmem1 = torch.ones(args.batch, channel, 32, 32).to(device)
        self.Vmem2 = torch.ones(args.batch, channel, 32, 32).to(device)
        self.Vmem3 = torch.ones(args.batch, channel * 2, 16, 16).to(device)
        self.Vmem4 = torch.ones(args.batch, channel * 2, 16, 16).to(device)
        self.Vmem5 = torch.ones(args.batch, FCLayer).to(device)
        self.Vmem6 = torch.ones(args.batch, 10).to(device)
        self.mode = args.train
        self.Timestep = args.TimeSteps
        self.Batch = args.batch
        self.GetAns = torch.zeros(args.batch, 2).to(device)
        self.Isit = torch.zeros(args.batch, 10).to(device)
    def forward(self, x):
        if self.mode == True:
            x = self.conv1(x, self.Vmem1)
            x = self.Drop(x)
            x = self.conv2(x, self.Vmem2)
            x = self.Drop(x)
            x = self.conv3(self.pool(x), self.Vmem3)
            x = self.Drop(x)
            x = self.conv4(x, self.Vmem4)
            x = self.Drop(x)
            x = self.pool(x).view(-1, self.num_flat_features(self.pool(x)))
            x = self.fc1(x, self.Vmem5)
            x = self.Drop(x)
            x = self.fc2(x, self.Vmem6)
            return nn.functional.log_softmax(x, dim = 1)
        if self.mode == False:
            self.Vmem1 *= 0
            self.Vmem1 += 1
            self.Vmem2 *= 0
            self.Vmem2 += 1
            self.Vmem3 *= 0
            self.Vmem3 += 1
            self.Vmem4 *= 0
            self.Vmem4 += 1
            self.Vmem5 *= 0
            self.Vmem5 += 1
            self.Vmem6 *= 0
            self.Vmem6 += 1
            self.Isit *= 0
            self.GetAns *= 0
            for i in range(self.Timestep):
                out = torch.floor(x + torch.rand(x.size()).to(device)).to(device)
                out, SynapseE, NeuronE, self.Vmem1  = self.conv1(out, self.Vmem1)
                self.SynpaseEnergy += SynapseE
                self.NeuronEnergy += NeuronE
                out, SynapseE, NeuronE, self.Vmem2  = self.conv2(out, self.Vmem2)
                self.SynpaseEnergy += SynapseE
                self.NeuronEnergy += NeuronE
                out, SynapseE, NeuronE, self.Vmem3  = self.conv3(self.pool(out), self.Vmem3)
                self.SynpaseEnergy += SynapseE
                self.NeuronEnergy += NeuronE
                out, SynapseE, NeuronE, self.Vmem4  = self.conv4(out, self.Vmem4)
                self.SynpaseEnergy += SynapseE
                self.NeuronEnergy += NeuronE
                out = self.pool(out).view(-1, self.num_flat_features(self.pool(out)))
                out, SynapseE, NeuronE, self.Vmem5 = self.fc1(out, self.Vmem5)
                self.SynpaseEnergy += SynapseE
                self.NeuronEnergy += NeuronE
                out, SynapseE, NeuronE, self.Vmem6 = self.fc2(out, self.Vmem6)
                self.SynpaseEnergy += SynapseE
                self.NeuronEnergy += NeuronE
                self.Isit += out + (self.Vmem6 * 0.01 / self.Timestep )
                X = torch.ones(self.Batch).to(device) * torch.argmax(self.Isit, dim=1) + 1
                self.GetAns[:, 0] += torch.mul(torch.where((self.Isit.max(dim=1).values >= 40) | (i == (self.Timestep - 1)),
                X,torch.zeros(self.Batch).to(device)),torch.where(self.GetAns[:, 1] == 0,
                torch.ones(self.Batch).to(device), torch.zeros(self.Batch).to(device)))
                self.GetAns[:, 1] += torch.where((self.Isit.max(dim=1).values >= 40) | (i == (self.Timestep - 1)),
                    torch.ones(self.Batch).to(device) * (i + 1),torch.zeros(self.Batch).to(device)) \
                    * torch.where(self.GetAns[:, 1] == 0,torch.ones(self.Batch).to(device)
                    , torch.zeros(self.Batch).to(device))
            return (self.GetAns[:, 0] - 1, self.GetAns[:, 1])

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

################################################################################################
class Mobile(nn.Module):
    def __init__(self, kernel, channel, FCLayer, args):
        super().__init__()
        self.conv1 = SNNConv(3, channel, kernel,0.1, arg = args)
        self.conv2 = SNNConv(channel, channel, 1,0.2, arg = args)
        self.conv3 = SNNConv(channel, channel*2, kernel,0.35, arg = args)
        self.conv4 = SNNConv(channel*2, channel*2, 1,0.4, arg = args)
        self.pool = nn.AvgPool2d(2, 2)
        self.Drop = nn.Dropout2d(p=0.1)
        self.Dropfc = nn.Dropout(p=0.5)
        self.fc1 = SNNLinear(channel*2*8*8, FCLayer,0.35, arg = args)
        self.fc2 = SNNLinear(FCLayer, 10,0.25, arg = args)
        self.SynpaseEnergy = 0
        self.NeuronEnergy = 0
        self.Vmem1 = torch.ones(args.batch, channel, 32, 32).to(device)
        self.Vmem2 = torch.ones(args.batch, channel, 32, 32).to(device)
        self.Vmem3 = torch.ones(args.batch, channel * 2, 16, 16).to(device)
        self.Vmem4 = torch.ones(args.batch, channel * 2, 16, 16).to(device)
        self.Vmem5 = torch.ones(args.batch, FCLayer).to(device)
        self.Vmem6 = torch.ones(args.batch, 10).to(device)
        self.mode = args.train
        self.Timestep = args.TimeSteps
        self.Batch = args.batch
        self.GetAns = torch.zeros(args.batch, 2).to(device)
        self.Isit = torch.zeros(args.batch, 10).to(device)
    def forward(self, x):
        if self.mode == True:
            x = self.conv1(x, self.Vmem1)
            x = self.Drop(x)
            x = self.conv2(x, self.Vmem2)
            x = self.Drop(x)
            x = self.conv3(self.pool(x), self.Vmem3)
            x = self.Drop(x)
            x = self.conv4(x, self.Vmem4)
            x = self.Drop(x)
            x = self.pool(x).view(-1, self.num_flat_features(self.pool(x)))
            x = self.fc1(x, self.Vmem5)
            x = self.Dropfc(x)
            x = self.fc2(x, self.Vmem6)
            return nn.functional.log_softmax(x, dim = 1)
        if self.mode == False:
            self.Vmem1 *= 0
            self.Vmem1 += 1
            self.Vmem2 *= 0
            self.Vmem2 += 1
            self.Vmem3 *= 0
            self.Vmem3 += 1
            self.Vmem4 *= 0
            self.Vmem4 += 1
            self.Vmem5 *= 0
            self.Vmem5 += 1
            self.Vmem6 *= 0
            self.Vmem6 += 1
            self.Isit *= 0
            self.GetAns *= 0
            for i in range(self.Timestep):
                out = torch.floor(x + torch.rand(x.size()).to(device)).to(device)
                out, SynapseE, NeuronE, self.Vmem1  = self.conv1(out, self.Vmem1)
                self.SynpaseEnergy += SynapseE
                self.NeuronEnergy += NeuronE
                out, SynapseE, NeuronE, self.Vmem2  = self.conv2(out, self.Vmem2)
                self.SynpaseEnergy += SynapseE
                self.NeuronEnergy += NeuronE
                out, SynapseE, NeuronE, self.Vmem3  = self.conv3(self.pool(out), self.Vmem3)
                self.SynpaseEnergy += SynapseE
                self.NeuronEnergy += NeuronE
                out, SynapseE, NeuronE, self.Vmem4  = self.conv4(out, self.Vmem4)
                self.SynpaseEnergy += SynapseE
                self.NeuronEnergy += NeuronE
                out = self.pool(out).view(-1, self.num_flat_features(self.pool(out)))
                out, SynapseE, NeuronE, self.Vmem5 = self.fc1(out, self.Vmem5)
                self.SynpaseEnergy += SynapseE
                self.NeuronEnergy += NeuronE
                out, SynapseE, NeuronE, self.Vmem6 = self.fc2(out, self.Vmem6)
                self.SynpaseEnergy += SynapseE
                self.NeuronEnergy += NeuronE
                self.Isit += out + (self.Vmem6 * 0.01 / self.Timestep)
                X = torch.ones(self.Batch).to(device) * torch.argmax(self.Isit, dim=1) + 1
                self.GetAns[:, 0] += torch.mul(
                    torch.where((self.Isit.max(dim=1).values >= 40) | (i == (self.Timestep - 1)),
                                X, torch.zeros(self.Batch).to(device)), torch.where(self.GetAns[:, 1] == 0,
                                                                                    torch.ones(self.Batch).to(device),
                                                                                    torch.zeros(self.Batch).to(device)))
                self.GetAns[:, 1] += torch.where((self.Isit.max(dim=1).values >= 40) | (i == (self.Timestep - 1)),
                                                 torch.ones(self.Batch).to(device) * (i + 1),
                                                 torch.zeros(self.Batch).to(device)) \
                                     * torch.where(self.GetAns[:, 1] == 0, torch.ones(self.Batch).to(device)
                                                   , torch.zeros(self.Batch).to(device))
            return (self.GetAns[:, 0] - 1, self.GetAns[:, 1])

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

###############################################################################################
class Mixer(nn.Module):
    def __init__(self, args):
        super(Mixer, self).__init__()
        self.con1 = SNNConv(3, 32, 3,0.1, args)
        self.con2 = SNNConv(32, 64, 3,0.1, args)
        self.mlp1 = SNNLinear(256, 512,0.125, args)
        self.mlp2 = SNNLinear(16, 512,0.17, args)
        self.mlp3 = SNNLinear(512, 16,0.17, args)
        #self.mlp3 = SNNLinear(100, 16,0.17, args)
        #self.mlp3 = SNNLinear(100, 16,0.17, args)
        self.mlp4 = SNNLinear(512, 10,3, args)
        self.pool = nn.AvgPool2d(2, 2)
        self.Drop = nn.Dropout2d(p= 0.29)
        self.Image = ImageToPatches(patch_size=2)
        self.Vmem = 0
        self.mode = args.train
        self.SynpaseEnergy = 0
        self.NeuronEnergy = 0
        self.Vmem1 = torch.ones(args.batch, 32, 32, 32).to(device)
        self.Vmem2 = torch.ones(args.batch, 32, 32, 32).to(device)
        self.Vmem3 = torch.ones(args.batch, 16, 100).to(device)
        self.Vmem4 = torch.ones(args.batch, 100, 100).to(device)
        self.Vmem5 = torch.ones(args.batch, 100, 16).to(device)
        self.Vmem6 = torch.ones(args.batch, 10).to(device)
        self.Timestep = args.TimeSteps
        self.Batch = args.batch
        self.GetAns = torch.zeros(args.batch, 2).to(device)
        self.Isit = torch.zeros(args.batch, 10).to(device)

    def forward(self, x):
        if self.mode == True:
            x = self.con1(x, self.Vmem)
            x = self.pool(x)
            x = self.con2(x, self.Vmem)
            x = self.pool(x)
            x = self.Drop(x)
            x = self.Image(x)
            x = self.mlp1(x, self.Vmem).permute(0, 2, 1).permute(0, 2, 1)
            x = self.Drop(x)
            x2 = x.permute(0, 2, 1)
            x2 = self.mlp2(x2, self.Vmem).permute(0, 2, 1).permute(0, 2, 1)
            x2 = self.Drop(x2)
            x2 = self.mlp3(x2, self.Vmem).permute(0, 2, 1).permute(0, 2, 1)
            x2 = self.Drop(x2)
            x3 = x2.permute(0, 2, 1)  # + x
            x3 = x3.mean(dim=1)
            x = self.mlp4(x3, self.Vmem)
            return nn.functional.log_softmax(x, dim = 1)
        if self.mode == False:
            self.Vmem1 *= 0
            self.Vmem1 += 1
            self.Vmem2 *= 0
            self.Vmem2 += 1
            self.Vmem3 *= 0
            self.Vmem3 += 1
            self.Vmem4 *= 0
            self.Vmem4 += 1
            self.Vmem5 *= 0
            self.Vmem5 += 1
            self.Vmem6 *= 0
            self.Vmem6 += 1
            self.Isit *= 0
            self.GetAns *= 0
            for i in range(self.Timestep):
                out = torch.floor(x + torch.rand(x.size()).to(device)).to(device)
                out, SynapseE, NeuronE, self.Vmem1  = self.con1(out, self.Vmem1)
                self.SynpaseEnergy += SynapseE
                self.NeuronEnergy += NeuronE
                out, SynapseE, NeuronE, self.Vmem2  = self.con2(out, self.Vmem2)
                self.SynpaseEnergy += SynapseE
                self.NeuronEnergy += NeuronE
                out = self.pool(out)
                out = self.Image(out)
                out, SynapseE, NeuronE, self.Vmem3  = self.mlp1(out, self.Vmem3)
                out = out.permute(0, 2, 1).permute(0, 2, 1)
                self.SynpaseEnergy += SynapseE
                self.NeuronEnergy += NeuronE
                out = out.permute(0, 2, 1)
                out, SynapseE, NeuronE, self.Vmem4  = self.mlp2(out, self.Vmem4)
                out = out.permute(0, 2, 1).permute(0, 2, 1)
                self.SynpaseEnergy += SynapseE
                self.NeuronEnergy += NeuronE
                out, SynapseE, NeuronE, self.Vmem5  = self.mlp3(out, self.Vmem5)
                out = out.permute(0, 2, 1).permute(0, 2, 1)
                self.SynpaseEnergy += SynapseE
                self.NeuronEnergy += NeuronE
                out = out.permute(0, 2, 1)
                if i > 2000:
                    out = out.mean(dim=1)
                    out, SynapseE, NeuronE, self.Vmem6 = self.mlp4(out, self.Vmem6)
                    self.Isit += out + (self.Vmem5 * 0.0001 / self.Timestep )
                X = torch.ones(self.Batch).to(device) * torch.argmax(self.Isit, dim=1) + 1
                self.GetAns[:, 0] += torch.mul(torch.where((self.Isit.max(dim=1).values >= 240) | (i == (self.Timestep - 1)),
                X,torch.zeros(self.Batch).to(device)),torch.where(self.GetAns[:, 1] == 0,
                torch.ones(self.Batch).to(device), torch.zeros(self.Batch).to(device)))
                self.GetAns[:, 1] += torch.where((self.Isit.max(dim=1).values >= 240) | (i == (self.Timestep - 1)),
                    torch.ones(self.Batch).to(device) * (i + 1),torch.zeros(self.Batch).to(device)) \
                    * torch.where(self.GetAns[:, 1] == 0,torch.ones(self.Batch).to(device)
                    , torch.zeros(self.Batch).to(device))
            return (self.GetAns[:, 0] - 1, self.GetAns[:, 1])


class ImageToPatches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.P = patch_size

    def forward(self, x):
        P = self.P
        B, C, H, W = x.shape  # [B,C,H,W]                 4D Image
        x = x.reshape(B, C, H // P, P, W // P, P)  # [B,C, H//P, P, W//P, P]   6D Patches
        x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H//P, W//P, C, P, P]  6D Swap Axes
        x = x.reshape(B, H // P * W // P, C * P * P)  # [B, H//P * W//P, C*P*P]   3D Patches
        # [B, n_tokens, n_pixels]
        return x