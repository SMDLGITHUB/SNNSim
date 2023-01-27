# SNNSim

This is a simulator for Analog SNN System using analog synaptic devices and analog neuron circuits.

Now, 2-layer fully connected SNN is available with SNNSim. 

(CNN Version is not open to public yet.)

You can get your system performance by setting own options in Main.py, Neuron.py, and DeviCe.py.

You set general options like time steps, time length per time step, neuron type, encoding scheme, synaptic devices, and technology nodes.
(In the case of Tech node, you have to change the parameter in neuron.py, too.)

You can add or change the synaptic devices, CMOS neurons, and neural network operation schemes of your own, referring to our code.

Run Main.py to get performance metrics of analog SNNs.
