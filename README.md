# Error-mitigated-fermionic-classical-shadow
Numerics for paper: arXiv:2310.12726

# wrapper.py usage:

the main function is "run_experiment", which combines callibration and estimation procedures 
(individual subtasks are outsourced to the other python scripts):

## calibration process

"run_calibration(n, noise_channel, p, no_samples, no_trials)" takes the following inputs

### n
(integer) the number of fermionic modes (or qubits, if one considers the Jordan-Wigner mapping)

### noise_channel
(string) specifies which noise channel should be used

available are: 

"depolarizing"

"amplitude damping"

"X rotation"

"Fermion"


### p
The error parameters defined in arXiv:2310.12726


### no_samples
(int) the number of samples $N_c$.

### no_trials
(int) the number of trials $K_c$.

## outputs

### (array of floats) f_arr: callibration parameters (medians of means)

for $n$ modes, f_arr will contain $n+1$ callibration parameters (all $f_{2k}$ for $k$ between $0$ and $n$) 


## estimation process
"run_estimation(n,p,noise_channel, f_arr, state, Q0, S0, no_samples_arr,ideal_val)" takes the following inputs

### n
(integer) the number of fermionic modes (or qubits, if one considers the Jordan-Wigner mapping)

### p
The error parameters defined in arXiv:2310.12726

example: for depolarizing noise, $p=1$ corresponds to fully depolarizing noise and $p=0$ corresponds to no noise.


### noise_channel
(string) specifies which noise channel should be used

available are: 

"depolarizing"

"amplitude damping"

"X rotation"

"Fermion"

### f_arr
(numpy array) an n+1 array to denote the estimated ${f_2k}$ from calibration process.

### state
(numpy array) a 2^n times 2^n density matrix with respect to which one wants to estimate expectation values

### Q0, S0
(ordered list of integers) specifies the observable $O = U_Q^\dagger \gamma_S U_Q$ for which one wants to estimate $\text{tr}(O \rho)$

example: for $S=[1,2]$, $\gamma_S=\gamma_1 \gamma_2$

All entries of $S$ must be smaller than $2n$ (total number of majorana operators)
                  



### no_samples_arr = [no_samples_est,no_trials_est,no_err_trials]
(integers) to obtain estimates $\text{tr}(O \rho)$, the median of means estimator is used.
this means that for every quantity that needs to be estimated, no_trials_est batches of no_samples_est single-shot estimates are obtained and the median of means is computed.
no_err_trials: the number of repetitions to get the error.

### ideal_val
expectation value, $tr(\rho O)$

## outputs

### (float) errors for the estimation with median of means estimate for tr$(O \rho)$
    

### 'sample_matchgates.py' file is provided by Janek Denzler.


