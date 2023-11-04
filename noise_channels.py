import numpy as np 
import random
from sample_matchgates import X, I, gen_matchgate_sigma



##Global depolarizing noise channel
def global_depolarizing_channel(n, rho, p):

    max_mix_state= np.identity(2**n, dtype = 'complex128') / 2**n

    dep_state = (1-p)*rho + p* np.trace(rho) * max_mix_state
    
    return dep_state


##Global amplitude damping channel
def amplitude_damping_ori(n, density_matrix, gamma):

    I = np.identity(2, dtype='complex128')

    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype='complex128')

    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype='complex128')

    for j in range(n-1):
        K0 = np.kron(K0, I)
        K1 = np.kron(K1, I) 

    ampstate = K0 @ density_matrix @ K0.conj().T + K1 @ density_matrix @ K1.conj().T

    return ampstate

def num_to_qubit(n,b):
    #convert n-bit number b into its computational basis rep.
    temp = 1
    vec_b = np.zeros(2**n)
    vec_b[num_b,num_b] = 1
    return vec_b

def amplitude_damping_notcorrect(n, rho, errors):
    # for errors:n by n, generate E_uv, and return sum_uv E_uv rho E_uv^dagger +E_0 rho E_0
    d = 2**n
    E = np.zeros((d,d,d,d),dtype='float')
    pro = np.zeros(d,dtype='float')
    Emax = np.zeros((d,d),dtype = 'float')
    for u in range(d):
        for v in range(d):
            if v == u:
                continue
            E[v][u][u][v]=np.sqrt(errors[u][v])
            pro[v] += errors[v][u]
    for v in range(d):
        Emax[v][v] = np.sqrt(1 - pro[v]) 
    rho_new = Emax @ rho @ np.conj(Emax).T
    for u in range(d):
        for v in range(d):
            rho_new += E[:][:][u][v] @ rho @ np.conj(E[:][:][u][v]).T
    
  #  print('Emax',Emax)
    return rho_new

def amplitude_damping(n, rho, errors):
    # for errors:n by n, generate E_uv, and return sum_uv E_uv rho E_uv^dagger +E_0 rho E_0
    d = 2**n
    pro = np.zeros(d,dtype='float')
    rho_new = np.zeros((d,d), dtype='complex128')
    for u in range(d):
        for v in range(d):
            if v == u:
                continue
            pro[u] += errors[u][v]        

    for u in range(d):
        for v in range(d):
            if u==v:
                continue
            rho_new[v][v] += errors[u][v]*rho[u][u]
    for u in range(d):
        rho_new[u][u] += (1-pro[u]) * rho[u][u]
    return rho_new


##X bit-flip on all qubits with probability p
"""def X_bit_flip(n, density_matrix, p):

    single_bit_flip = (1-p) * I + p * X
    bit_flip = np.copy(single_bit_flip)
    for j in range(n-1):
        bit_flip = np.kron(bit_flip, single_bit_flip)

    flipped_state = bit_flip @ density_matrix @ bit_flip.conj().T

    return flipped_state"""

##X-rotation on all qubits with angle theta and probability p
def global_x_rotation(n, density_matrix, theta):
    ### perform Rx(theta)

    Identity = np.identity(2**n, dtype='complex128')

    single_rotation = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype='complex128')
    
    rotation = np.copy(single_rotation)

    for i in range(n-1):
        rotation = np.kron(rotation, single_rotation)
    
    rotated_state = rotation @ density_matrix @ rotation.conj().T

    return rotated_state

def fermion_noise(n, rho, perm_arr):
    
    Q,U = gen_matchgate_sigma(n, perm_arr,ret_unitary = True)
    
    rho_new = U @ rho @ np.conj(U).T
    return rho_new