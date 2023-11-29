from numba import cuda, void, float32  # for parallel gpu computations
import numpy as np  # for random samples and array management
import time  # for timing
import math  # for sinus


eps = 0.1
step = 1e-1             # 2e-2
a_arr = np.around(np.arange(0, 2+step, step), 3)
b_arr = np.around(np.arange(0, 2+step, step), 3)
pars_arr = np.array([[i, j] for j in b_arr for i in a_arr], dtype = 'float32')
t0 = 0
t_end = 2e2
dt = 2e-2
time = np.array([t0, t_end, dt], dtype = 'float32')
x_arr = np.zeros((len(pars_arr), int((t_end-t0)/dt), 2), dtype = 'float32')
x_arr[:, 0, :] = np.ones(2, dtype = 'float32') * 1e-5

@cuda.jit(void(float32[::1], float32[::1], float32[::1,:]))
def solve_ode(time, pars, x0=np.array([[]])):
    """
    Solve 2DoF ode on gpu, given the initial conditions. The result will be
    stored in the input array.


    Parameters
    ----------
    x : np.array
        Contains initial conditions for the simulations.
        The elements are arranged in pairs:
        [x1_sim1, x2_sim1, x1_sim2, x2_sim2, ...]
    time : np.array
        Three-element list with time details.

    Returns
    -------
    None.

    """
    # time variables
    t = time[0]
    t_end = time[1]
    dt = time[2]
    a = pars[0][0]
    b = pars[0][1]
    k = pars[1]
    dx = x0[1,:]
    # index of thread on GPU
    pos = cuda.grid(1)
    # mappping index to access every
    # second element of the array
    pos = pos * 2

    # condidion to avoid threads
    # accessing indices out of array
    if pos < pars.size:
        # execute until the time reaches t_end
        for i in range(t, t_end-dt, dt):
            dx[0] = (x0[i, 0]*(1-x0[i, 0]*x0[i, 0]/3)-x0[i, 1])/0.1
            dx[1] = x0[i, 0] - b*x0[i, 1] + a   # 1
            
            x0[i+1,:] = x0[i,:] + dx * dt


t = np.array([t0, t_end, dt], dtype="float32")

# generate random initial condiotions
init_states = pars_arr

# manage nr of threads (threads)
threads_per_block = 32
blocks_per_grid = (init_states.size + (threads_per_block - 1)) \
        // threads_per_block

d_init_states = cuda.to_device(init_states)

d_init_states_opt = cuda.to_device(init_states)

d_t = cuda.to_device(t)

# start timer
start = time.perf_counter()

# start parallel simulations
solve_ode[blocks_per_grid, threads_per_block](d_t, d_init_states)
cuda.synchronize()

# measure time elapsed
end = time.perf_counter()

# Copy results back to host
original_result = d_init_states.copy_to_host()



print(f"The result was computed in {end-start} s")