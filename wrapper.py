import numpy as np
import numpy.linalg as linalg
import itertools
import scipy.special
import matplotlib.pyplot as plt

from sample_matchgates import random_FGUclifford, I, X, Y
from quantum_process import quantum_process,noise_channel_ope
from fermionic_states import pure_initial, gaussian_density_matrix, fermionic_operators, generate_covariance

Z = np.array([[1, 0],
              [0, -1]], dtype='complex128')

def subsets(n,k):
    """returns all subsets of {1,...,n} of cardinality k"""
    
    return list(itertools.combinations(np.arange(1,n+1), k))
  
  
def ind(S):
    """given a list of integers >=1, S, returns a list with all entries shifted by -1"""
    
    return [(s-1) for s in S]


def HS(X,Y):
    """Hilbert-Schmidt inner product"""
    
    return np.trace(np.matmul(np.conj(X).T,Y))


def matching_sites(b,S):
    """given binary array b and index set S, outputs the number of entries of b with index in S that are 1"""
    
    if len(S) == 0:
        return 0
    
    else: 
        return np.sum(b[S])

def matching_gaussian(mu, S):
    
    if len(S) == 0:
        return 0
    
    else: 
        return np.prod(mu[S])
    
def matching_slater(n,S, tau):
    
    s_res = np.zeros(n)
    s_res[0] = 1
    for j in range(tau):
        s_res[S[j]] = 1
  #  print('set converted',s_res)
    #convert s_res into a number
    num_b = 0
    temp = 1
    for j in range(n):
        num_b = num_b + temp * s_res[n-1-j]
        temp *= 2
    return int(num_b)

def matching_short_slater(n,S,tau):
###return the number representation S for n-bitstring, where the length of S is tau, e.g.S = [1,2], n = 3, numb_b = 110=3
    s_res = np.zeros(n)
    for j in range(tau):
        s_res[S[j]-1] = 1
   # print('set converted',s_res)
    #convert s_res into a number
    num_b = 0
    temp = 1
    for j in range(n):
        num_b = num_b + temp * s_res[n-1-j]
        temp *= 2
    return int(num_b)

def gaussian_state(n):
    """output Gaussian state"""

    purestate = pure_initial(n)
    covar = generate_covariance(n)
    c, a = fermionic_operators(n)
    gaussianstate = gaussian_density_matrix(covar, c, a)
    
    return gaussianstate

def run_calibration(n, noise_channel, p, no_samples, no_trials):
# callibration procedure
   # print("callibration procedure")
    
    f_arr = []
    
    # initial state for callibration procedure
    all_zeros = np.zeros((2**n,2**n),dtype= "complex128")
    all_zeros[0,0] = 1
    
    for k in range(0,n+1):
        # fix callibration parameter f_2k
        
        # median of means
        estimates = np.zeros(no_trials)
        
        for trial in range(no_trials):
            #print('t',trial)
            est = 0
            
            for sample in range(no_samples):
            
                Q, U = random_FGUclifford(n, ret_unitary=True)
                
                b = quantum_process(all_zeros, U, n, p, noise_channel)
                
               # print('b',b)
                
                est += get_f_2k(k,n,Q,b)

            est /= no_samples
            estimates[trial] = est
         
        f_arr.append(np.median(estimates))
     #   print("f_" + str(2*k),'=',estimates)
    
    return f_arr

def run_estimation(n,p,noise_channel, f_arr, state, Q0, S0, no_samples_arr,ideal_val):
# estimation procedure
    """
    n: the number of qubits
    p: noise strength
    """
    no_samples = no_samples_arr[0]
    no_trials = no_samples_arr[1]
    no_err_trials = no_samples_arr[2]
    
   # print("estimation procedure")
    
    err = 0  
    for err_trial in range(no_err_trials):
        estimates = np.zeros(no_trials, dtype='complex128')
        for trial in range(no_trials):
            est = 0
            for sample in range(no_samples):
                Q, U = random_FGUclifford(n, ret_unitary=True)
                b = quantum_process(state, U, n, p, noise_channel)
                est += estimate_gammaS(n,f_arr,Q,b,Q0, S0)
            #est += estimate_gammaS_U(n,f_arr,U,b,U0,S0)   
            est /= no_samples
            estimates[trial] = est
        est_f = np.mean(estimates)
        temp = np.abs(est_f-ideal_val)**2
        err += temp
    err = np.sqrt(err/no_err_trials)
    return err

def run_estimation_gaussian(n,p,noise_channel, f_arr, state, Qg, mu_g, no_samples_arr,ideal_val):
# estimation procedure
    """
    n: the number of qubits
    p: noise strength
    """
    no_samples = no_samples_arr[0]
    no_trials = no_samples_arr[1]
    no_err_trials = no_samples_arr[2]
    
   # print("estimation procedure")
    err = 0  
    for err_trial in range(no_err_trials):
        estimates = np.zeros(no_trials, dtype='complex128')
        for trial in range(no_trials):
            est = 0
            for sample in range(no_samples):
                Q, U = random_FGUclifford(n, ret_unitary=True)
                b = quantum_process(state, U, n, p, noise_channel)
                est += estimate_GaussianState(n,f_arr,Q,b,Qg,mu_g)#(n,f_arr,Q,b,Q0, S0)
                #est += estimate_gammaS_U(n,f_arr,U,b,U0,S0)   
            est /= no_samples
            estimates[trial] = est
        est_f = np.mean(estimates)
        temp = np.abs(est_f-ideal_val)**2
        err += temp
    err = np.sqrt(err/no_err_trials)   
    return err

def run_estimation_slater(n,p,noise_channel, f_arr, state, tau, V, no_samples_arr,ideal_val):
# estimation procedure
    """
    n: the number of qubits
    p: noise strength
    """
    no_samples = no_samples_arr[0]
    no_trials = no_samples_arr[1]
    no_err_trials = no_samples_arr[2]
   
   # print("estimation procedure")
    
    err = 0  
    for err_trial in range(no_err_trials):
        estimates = np.zeros(no_trials, dtype='complex128')
        for trial in range(no_trials):
            est = 0
            for sample in range(no_samples):
                Q, U = random_FGUclifford(n, ret_unitary=True)
                b = quantum_process(state, U, n, p, noise_channel)
                est += estimate_Slater(n,f_arr,Q,b,tau,V)
            est /= no_samples
            estimates[trial] = est
           # print(err_trial,trial,est*2)
        est_f = np.mean(estimates) * 2
        temp = np.abs(est_f-ideal_val)**2
        err += temp
    err = np.sqrt(err/no_err_trials)   
    return err



def get_f_2k(k,n,Q,b):
    """returns single-round estimator for f_2k, given measurement outcome b (an array) and matchgate Q"""
    sets = subsets(n,k)
    estimate = 0
    for S in sets:
        temp=estimate
        S_complete = np.array([(2*i-1,2*i) for i in S]).flatten()
        for Sp in sets:
            Sp_complete = np.array([(2*i-1,2*i) for i in Sp]).flatten()
            m = matching_sites(b, ind(Sp))
            estimate += (-1)**(m)*np.linalg.det(Q[np.ix_(ind(Sp_complete), ind(S_complete))])
    estimate *= 1/(scipy.special.binom(n,k))
    #print(estimate)
    return estimate

"""
def get_f_2k_new(k,n,U,b):
    sets = subsets(n, k)
    est = 0
    all_zeros = np.zeros((2**n,2**n),dtype= "complex128")
    all_zeros[0,0] = 1
    num_b = 0
    temp = 1
    for j in range(n):
        num_b = num_b + temp * b[n-1-j]
        temp = temp*2
        
    mat_b = np.zeros((2**n,2**n),dtype= "complex128")
    mat_b[num_b,num_b] = 1 
    for S in sets:
        NS = np.array([(2*i-1,2*i) for i in S]).flatten()
        gamma_S = majorana_op(n,NS)
        gamma_S_new = np.matmul(U,np.conj(gamma_S).T)
        gamma_S_new = np.matmul(gamma_S_new, np.conj(U).T)
        temp = np.trace(np.matmul(all_zeros, gamma_S)) * np.trace(np.matmul(gamma_S_new, mat_b))
        est = est + temp
    est = est /(scipy.special.binom(n,k))
    
    return est"""

def estimate_gammaS_U(n,f_arr,U,b,U0,S0):
    """returns single-round estimator for tr(gamma_S rho) by U calculation"""
    k = len(S0)
    if k%2 == 1:
        return 0
    k=k//2
    gamma_S0 = majorana_op(n,S0)
    sets = subsets(2*n, 2*k)
    
    # array rep for b
    num_b = 0
    temp = 1
    for j in range(n):
        num_b = num_b + temp * b[n-1-j]
        temp = temp*2
    mat_b = np.zeros((2**n,2**n),dtype= "complex128")
    mat_b[num_b,num_b] = 1 
    
    res = 0
    for S in sets:
        gamma_S = majorana_op(n,S)
        temp= np.trace(np.conj(U0).T @ gamma_S0 @ U0 @ gamma_S)
        temp *= np.trace(U @ np.conj(gamma_S).T @ np.conj(U).T @ mat_b)
        res += temp
    """res = np.trace(U @ np.conj(gamma_S0).T @ np.conj(U).T @ mat_b)
    res = res/f_arr[k]/2**n"""
    return res
    
def estimate_gammaS(n,f_arr,Q,b,Q0, S0):
    """returns single-round estimator for tr(gamma_S rho), given 
    the number of qubits n, 
    classical shadow (Q,b), 
    measurement gamma_S (Q0,S0)
    and array of calibration parameters f_arr"""
    
    if len(S0)%2==1:
    
        # parity preserved
        return 0
        
    k = len(S0)//2
    sets = subsets(n,k)
    sets0 = subsets(2*n,2*k)
    estimate = 0
    
    if f_arr[k] == 0:
        print('f_',k,'=0',S0)
        return 0
    
    for S in sets0:
        
        temp_1 = np.linalg.det(Q0[np.ix_(ind(S0),ind(S))])
        if temp_1 == 0:
            continue
        temp_2 = 0
        for Sp in sets:
            Sp_complete = np.array([(2*i-1,2*i) for i in Sp]).flatten()
            m = matching_sites(b, ind(Sp))
            temp_2 += (-1)**(m)*np.linalg.det(Q[np.ix_(ind(Sp_complete),ind(S))])
           
        estimate += temp_1 * temp_2
   # print('b=',b,'Q=',Q,'temp_1=',temp_1,'temp_2=',temp_2)
   # print('before coef', estimate, 'after coef', 1.j**k*(estimate)/f_arr[k])
    
    estimate = 1.j**k*(estimate)/f_arr[k] 
    return estimate



def estimate_GaussianState(n,f_arr,Q,b,Q_g,mu_g):
    """returns single-round estimator for tr(rho_g rho), given classical shadow (Q,b) and array of calibration parameters f_arr, where rho_g = (mu_g, UQ_g)
    """

    estimate = 0
    
    est = 1
    for k in range(1,n+1):
        temp = 0
        sets_l = subsets(2*n,2*k)
        sets = subsets(n,k)
        for S_l in sets_l:
            temp_b = 0
            temp_g = 0
            for Sp in sets:
                Sp_complete = np.array([(2*i-1,2*i) for i in Sp]).flatten()
                msb = matching_sites(b, ind(Sp))
                msg = matching_gaussian(mu_g,ind(Sp))
                temp_b += (-1)**msb * np.linalg.det(Q[np.ix_(ind(Sp_complete),ind(S_l))])
                temp_g += msg * np.linalg.det(Q_g[np.ix_(ind(Sp_complete),ind(S_l))])
            temp += temp_b * temp_g
       # print(k,'):',temp)
        est += temp/f_arr[k]
    
    return est/2**n



def estimate_Slater(n,f_arr,Q,b,tau,V):
    """returns single-round estimator for <phi_tau|psi> between tau-slater determinant and pure state psi, where phi_tau = sum_S det(V^*|[tau],S) a_S ket{0}, 
    dimension V:n-1 by n-1 Hermian matrix.
    tau: 1,...,n-1.(n0=n-1)
    given classical shadow (Q,b) and array of calibration parameters f_arr.
    """

    est = 0
    n0 = n - 1
    Stau = np.array([j+1 for j in range(tau)])
    for k in range(n+1):
        temp = 0
        sets_l = subsets(2*n,2*k)
        sets_tau = subsets(n0,tau)
        sets_S = subsets(n,k)
        for S_l in sets_l:
            temp_b = 0
            gammaS = majorana_op(n,S_l)
            #output tr(ket(1)ket(phi) bra(0)bra(0^n)gamma_S_l)
            b_tau = matching_slater(n,Stau, tau)
          #  print('val_gamma:',b_tau,gammaS[0][b_tau])
            for Sp in sets_S:
                Sp_complete = np.array([(2*i-1,2*i) for i in Sp]).flatten()
                msb = matching_sites(b, ind(Sp))
                #temp_b += (-1)**msb * np.linalg.det(Q[np.ix_(ind(Sp_complete),ind(S_l))])
                temp_b += (1-2*(msb%2)) * np.linalg.det(Q[np.ix_(ind(Sp_complete),ind(S_l))])
           #     print(S_l,Sp,'):tempb:',temp_b)
           # temp_b *= (-1.j)**k
            temp_b *= ((-1.j)**(k%4))
            temp_tau = 0
            for Sq in sets_tau:
                b_tau = matching_slater(n,Sq, tau)#a number
                temp_tau += np.linalg.det(np.conj(V)[np.ix_(ind(Stau),ind(Sq))]) * gammaS[0][b_tau]
          ###      print(S_l,Sq,'): gamma(0,',b_tau,')=',gammaS[0][b_tau],'det val',np.linalg.det(np.conj(V)[np.ix_(ind(Stau),ind(Sq))]), 'temptau:',temp_tau)
            temp += temp_b * temp_tau
        est += temp/f_arr[k]
    ###    print(k, ')est',est,'=',temp,'/',f_arr[k])
    
    return est/2**n

def trace_classical_b_gamma(n,S, Q, b):
    k = len(S)//2
    sets = subsets(n,k)
    res = 0
    for Sp in sets:
        Sp_complete = np.array([(2*i-1,2*i) for i in Sp]).flatten()
        msb = matching_sites(b, ind(Sp))
        temp = (-1)**msb
        temp *= np.linalg.det(Q[np.ix_(ind(Sp_complete),ind(S))])
        res += temp
       # print(res)
    return res * (-1.j)**k
        

def gen_H(n,rho, tau,V):
    # the size of V equals n by n.
    H = np.zeros((2**n,2**n),dtype='complex128')
    sets = subsets(n-1,tau)
    Stau = np.array([j+1 for j in range(tau)])
    for Sq in sets:
        b_tau = matching_slater(n,Sq, tau)#a number
        H[b_tau][0] += np.linalg.det(np.conj(V)[np.ix_(ind(Stau),ind(Sq))])
    return H

def estimate_Slater_new(n, f_arr,Q,b,tau,H):
    
    res = 0
    for k in range(n+1):
        sets = subsets(2*n,2*k)
        temp = 0
        for S in sets:
            gammaS = majorana_op(n,S)
            res1 = np.trace(H @ gammaS)
            res2 = trace_classical_b_gamma(n,S, Q, b)
            temp += res1 * res2
        res += temp/f_arr[k]
     ###   print(k, ')res',res,'=',temp,'/',f_arr[k])
    return res/2**n

def Verify_Slater(n,psi, tau,V):
    # the size of V equals n by n.
    sets = subsets(n,tau)
    res = 0
    Stau = np.array([j+1 for j in range(tau)])
    for S in sets:
        bs = matching_short_slater(n,S,tau)
    #    print('S:',S, 'phi',bs,'=',psi[bs], np.linalg.det(np.conj(V)[np.ix_(ind(Stau), ind(S))]))
        res += np.linalg.det(np.conj(V)[np.ix_(ind(Stau), ind(S))]) * np.conj(psi[bs])
    return res

def Verify_Slater_new(n,rho, tau,V):
    # the size of V equals n by n.
    res = 0
    sets = subsets(n-1,tau)
    Stau = np.array([j+1 for j in range(tau)])
    for Sq in sets:
        b_tau = matching_slater(n,Sq, tau)#a number
        res += np.linalg.det(np.conj(V)[np.ix_(ind(Stau),ind(Sq))]) * rho[0][b_tau]
    
    return res*2

   
def true_val(n,state,S): #return true value
    
    O = majorana_op(n,S)
    
    return HS(state, O)


def majorana_op(n,S):#return gamma_S
    O = np.identity(2**n, dtype='complex128')
    
    if len(S) == 0:
        return O
     
    for s in S:
        
        if (s+1)//2 -1 == 0:
            
            if s ==1:
                op = X
                
            if s ==2:
                op = Y
            
            for qubit in range(1,n):
                op = np.kron(op, I)
        
        else:
            
            op = Z
            for qubit in range(1,(s+1)//2-1):
                op = np.kron(op, Z)
            
            if s%2==0:
                op = np.kron(op, Y)
            if s%2==1:
                op = np.kron(op, X)
            for qubit in range((s+1)//2,n):
                op = np.kron(op, I)
            
#        print('s',s,'op',op,'l_op',len(op),'l_O',len(O))
        O = O @ op
    
    return O

def to_binary(x,n): # convert x to n bit bitstring, e.g., (3,4)--> 0011
    temp = bin(x)[2:]
    x = ['0' for j in range(n)]
    start = n - len(temp)
    x[start:]=temp[:]
    for j in range(n):
        x[j] = int(x[j])
    return x

def double_set(S):
    DS = [0 for l in range(2*len(S))]
    k = 0
    for element in S:
        DS[2*k] = 2 * element-1
        DS[2*k+1] = 2 * element
        k = k + 1
    return DS

    
def ideal_f_2k_calculation(n,k, p, noisechannel):
# return the ideal f_2k val
    S_set = subsets(n,k)
    l = len(S_set)
    val = 0
    for x in range(2**n):
        for S in S_set:
            x_arr = to_binary(x,n)
            temp = 0
 #           print('x_arr',x_arr)
            for cur_j in S:
                temp = temp + x_arr[int(cur_j-1)]
            temp = (-1)**temp
            rho_mat = [[0 for j in range(2**n)] for k in range(2**n)]
            rho_mat[x][x] = 1
            D_S = double_set(S)
 #           print('x',x,'S',S,'DS',D_S)
 #           print('rho',rho_mat,'O',majorana_op(n,D_S))
 #           print('sign',temp,'trace',np.trace(np.matmul(rho_mat,majorana_op(n,D_S))))
            noisy_ope = noise_channel_ope(majorana_op(n,D_S), n, p, noisechannel)
            temp = temp * np.trace(np.matmul(rho_mat, noisy_ope))
            val = val + temp
    val = val * (-1.j)**k/(scipy.special.binom(2*n,2*k))/2**n
    return val

def Verify_gammaS_U(n,state,U0,S0):
    
    gamma_S0 = majorana_op(n,S0)
   # print(gamma_S0,'\nU0 dagger:',U0.conj().T,'transformed:',U0.conj().T @ gamma_S0 @ U0)
    gamma_after = np.matmul(np.conj(U0).T,gamma_S0)
    gamma_after = np.matmul(gamma_after,U0)
    gamma_after = np.matmul(gamma_after,state)
    res = np.trace(gamma_after)
    return res

def Verify_gammaS_Q(n,state,Q,S0):
    k = len(S0)
    S_set = subsets(2*n,k)
    res = 0
    for S in S_set:
        gamma_S = majorana_op(n,S)
        res += np.linalg.det(Q[np.ix_(ind(S0),ind(S))]) * np.trace(gamma_S @ state)
    return res

def Verify_GaussianState(n,rho,U_g,mu_g):
    # return tr(rho U_g^dagger prod_j(I-i mu_gj gamma_2j-1 gamma_2j) U_g)
    gaus = I + mu_g[0]*Z
    for j in range(1,n):
        gaus = np.kron(gaus, I+mu_g[j]*Z)
    overlap = np.conj(U_g).T @ gaus @ U_g @ rho
    return np.trace(overlap/2**n)


