import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sinc
from scipy import signal as sig
import time

#----------------------------------------------------------------------------
# reference:
# http://phaseportrait.blogspot.com/2008/06/sinc-interpolation-in-matlab.html
#----------------------------------------------------------------------------
def sinc_interp(sig, Ts, up_sampling_factor = 10, plot=False, window=False):
    Ts_up = Ts / up_sampling_factor
    n_up = up_sampling_factor * sig.size
    t_up = np.arange(0, n_up, 1) * Ts_up
    x_t = np.zeros(n_up)
    for i in range(0, x_t.size):
        temp = 0
        for j in range(0, sig.size):
            temp = temp + sig[j] * sinc((i * Ts_up - j * Ts) / Ts)
        x_t[i] = temp
    return x_t, t_up

def sinc_interp_vector1(sig, Ts, up_sampling_factor = 10, plot=False, window=False):
    Ts_up = Ts / up_sampling_factor
    n_up = up_sampling_factor * sig.size
    t_up = np.arange(0, n_up, 1) * Ts_up
    x_t = np.zeros(n_up)
    sinc_val = np.zeros((n_up, sig.size))
    for i in range(0, sig.size):
        sinc_val[:, i] = sinc(t_up/Ts-i)            # calculate and store the sinc function
    for i in range(0, x_t.size):
        x_t[i] = np.dot(sinc_val[i, :], sig)        # vector multiplication
    return x_t, t_up

def sinc_interp_vector11(sig, m_prime_TR, TR, window=False, debug=False):
    sig_interp = np.zeros(sig.size, dtype=complex)
    sinc_val = np.zeros((sig.size, sig.size))
    m_TR_val = np.arange(-sig.size/2, sig.size/2)*TR
    # m_TR_val = np.arange(0, sig.size) * TR
    for i, m_TR in enumerate(m_TR_val):
        sinc_val[:, i] = sinc((m_prime_TR - m_TR)/TR)            # calculate and store the sinc
        if (i%100)==0 and debug==True:
            plt.plot(m_TR_val, sinc_val[:, i])
            plt.plot(m_TR_val, abs(sig))
            plt.ylim(-1, 1)
            plt.show()
            print("debug " + str(i))
    for i in range(0, sig.size):
        sig_interp[i] = np.dot(sinc_val[i, :], sig)        # vector multiplication
    return sig_interp, m_TR_val

def sinc_interp_vector2(sig, Ts, up_sampling_factor = 10, plot=False, window=False):
    n = np.arange(0, sig.size, 1)
    Ts_up = Ts / up_sampling_factor
    n_up = up_sampling_factor * sig.size
    t_up = np.arange(0, n_up, 1) * Ts_up
    x_t = np.zeros(n_up)
    for i in range(0, x_t.size):
        sinc_val = sinc(t_up[i]/Ts - n)       # calculate one row of sinc function
        x_t[i] = np.dot(sinc_val, sig)        # vector multiplication
    return x_t, t_up


def sinc_interp_convolve(sig, Ts, up_sampling_factor = 10, plot=False, window=False):
    Ts_up = Ts / up_sampling_factor
    n_up = up_sampling_factor * sig.size
    t_up = np.arange(0, n_up, 1) * Ts_up  # new time scale
    t_sinc = np.arange(-n_up/2, n_up/2, 1) * Ts_up    # move sinc function to the center
    sig_up = np.zeros(n_up)
    sig_up[0:-1: up_sampling_factor] = sig  # [first index, last index, step]; insert zeros in between original sample
    x_t = np.convolve(sig_up, sinc(t_sinc / Ts), "same")    # convolution
    return x_t, t_up