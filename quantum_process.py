import numpy as np 
import itertools
from noise_channels import global_depolarizing_channel, amplitude_damping, global_x_rotation, fermion_noise
import random
import copy
import cmath,math
from sample_matchgates import random_FGUclifford, I, X, Y

def list_to_array(lst):
    return np.array(lst)

def quantum_process(rho, U, n, p, noisechannel):
    '''Takes as input an initial state, a unitary to be applied, the size of the system 'n' and
    a probability associated with a specific noise channel, the noise channel and an angle associated to the X rotation (in radians), 
    and measures in the computational basis, outputting a bitstring b of 0s and 1s'''

    rho = U @ (rho @ U.conj().T)
    

    ##apply noise channel based on input
    if noisechannel == 'depolarizing':

        rho = global_depolarizing_channel(n, rho, p) 

    elif noisechannel == 'amplitude damping':

        rho = amplitude_damping(n, rho, p)
    
    elif noisechannel == 'bit flip':

        rho = X_bit_flip(n, rho, p)
    
    elif noisechannel == 'X rotation':

        rho = global_x_rotation(n, rho, p)
    elif noisechannel == 'Fermion':
        rho = fermion_noise(n, rho, p)#p=vector_permutation.
    else: 
        rho = rho

   # print('before measure it, rho=',rho)
    
    ##measure in the computational basis
    probabilities = np.abs(np.diag(rho))
    probabilities = probabilities /np.sum(probabilities)

    
    bitstrings = list(itertools.product([0, 1], repeat=n))
    
    outputs = random.choices(bitstrings, weights = probabilities)[0]

    barray = list_to_array(outputs)
    
    return barray


def noise_channel_ope(rho, n, p, noisechannel):
    '''Takes as input an initial state, the size of the system 'n' and
    a probability associated with a specific noise channel, the noise channel and an angle associated to the X rotation (in radians), 
    return noisechannel(rho)'''

    ##apply noise channel based on input
    if noisechannel == 'depolarizing':

        rho = global_depolarizing_channel(n, rho, p) 

    elif noisechannel == 'amplitude damping':

        rho = amplitude_damping(n, rho, p)
    
    elif noisechannel == 'bit flip':

        rho = X_bit_flip(n, rho, p)
    
    elif noisechannel == 'X rotation':

        rho = global_x_rotation(n, rho, p)
    elif noisechannel == 'Fermion':
        rho = fermion_noise(n, rho, p)
    
    else: 
        rho = rho

   # print('before measure it, rho=',rho)
    return rho


##Example of how to run:
#n=2
#print('n=',n)
### initial state for callibration procedure
#all_zeros = np.zeros((2**n,2**n),dtype= "complex128")
#all_zeros[0,0] = 0.5
#all_zeros[0,1] = 0.5
#all_zeros[1,0] = 0.5
#all_zeros[1,1] = 0.5
#state = copy.copy(all_zeros)
#print('state=',state)
#noise_channel = "depolarizing"

#p = 0
#
#Q, U = random_FGUclifford(n, ret_unitary=True)
#U= np.array([[ 3/5, cmath.exp(1.j*math.pi/4)*4/5, 0.0000000e+00-0.j, 0.0000000e+00-0.j],
# [4/5, 3/5, 0.0000000e+00-0.j, 0.0000000e+00-0.j],
# [ 0.0000000e+00-0.j, 0.0000000e+00-0.j, 0.0000000e+00-0.j, -1.2246468e-16-1.j],
# [ 0.0000000e+00-0.j, 0.0000000e+00-0.j, 1.2246468e-16-1.j, 0.0000000e+00-0.j]]) 

#Q= np.array([[-1., -0.,  0., -0.],
# [-0., -1.,  0., -0.],
# [-0., -0.,  1., -0.],
# [-0., -0.,  0., -1.]])
#U = U.conj().T
#print('type U',type(U))


#print('U=',U, 'Q=',Q, 'p=',p)

#b = quantum_process(state, U, n, p, False, noise_channel)
#
#print('b array for pure state:', b)
