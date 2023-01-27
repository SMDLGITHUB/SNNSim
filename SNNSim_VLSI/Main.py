import Algorithms

#Setting (Set SNN options via Main.py)
Device = 'Flash' # Flash, ReRAM, FeFET
Network = 'MLP' #'MLP'(784-256-10), other networks like CNNs are not open to public yet
Neuron = 'C' # A type(ours), B type(Single ended Schmitt trigger with ATB), C type( Schmitt trigger), User
Tech = 45 # 45nm, 22nm Tech nodes are supported (Should change Neuron*.py Tech parameter too)
Encoding = 'Temp' # Rate, Temp
Timesteps = 128 # Set the Maximum timestep to operate the SNN
Timeperstep = 200e-9 # Set the time window for each timestep (200e-9 ==> 200ns for each time step)

#Accuracy effect
OffCurrent = True # Set to True to consider Off-Current of the synaptic devices (Set specifics in DeviCe.py)
ReadNoise = False # Set to True to consider ReadNoise of the synaptic devices (Set specifics in DeviCe.py)
SPIKELOSS = 0 # Set the probability to consider stochastic spike loss in the circuits

# Check Capacitance of Membrane capacitor of Neuron in Neuron*.py
# Check whether Tech of Neuron*py match Tech of Main.py

def GetAccuracyEnergyDelay():
    if Network == 'MLP':
        Algorithms.MLP(Device, Neuron, Tech, Encoding, Timesteps, OffCurrent, ReadNoise, SPIKELOSS,Timeperstep)

GetAccuracyEnergyDelay()