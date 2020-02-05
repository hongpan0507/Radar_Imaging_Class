import numpy as np
from scipy.constants import c, pi
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------------------------------------------
def LFM(BW, F_s, T_p, plot = False):
    T_s = 1 / F_s  # ADC sampling period
    nSample = int(T_p * F_s)  # number of samples from ADC
    chp_rt = BW / T_p  # chirp rate
    t = np.arange(0, nSample, 1) * T_s  # (start, stop, step)
    phase = -2 * pi * BW / 2 * t + pi * chp_rt * t**2
    freq = -BW / 2 + chp_rt * t
    x_t = np.cos(phase)  # un-windowed signal

    # win_len = int(t.size*0.2)
    # window = sig.hamming(win_len)
    # pad_w = int((t.size - win_len)/2)
    # window = np.pad(window, (pad_w, pad_w), 'constant', constant_values=0)      # pad the size of the window to be the same as t
    #
    # x_t = x_t*window    # windowed signal
    #+++ FFT +++ and look the output
    if plot:
        fig = plt.figure()
        plt.suptitle("LFM" + ", Chirp Rate = " + str(chp_rt) + "Hz/s, BW = " + str(BW) + "Hz, PW = " + str(T_p) + "s")

        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(t, x_t)
        ax1.set_xlabel("Time (s)")
        plt.title("Waveform")

        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(t, phase)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Phase (Radian)")
        plt.title("Phase")

        ax3 = plt.subplot(3, 1, 3)
        ax3.plot(t, freq)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Freq (Hz)")
        plt.title("Frequency")

        plt.subplots_adjust(hspace=0.5)
        plt.show()

    return x_t, t
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
#   x_t = sampled waveform, t = time sample
#   tau = propagation delay, t_i = initial sample
#   F_s = sampling frequency
#   att = attenuation, F_D = doppler frequency shift
def prop_delay(x_t, t, tau, t_i, F_s, att=1, F_D=0):
    # FFT
    x_f = np.fft.fft(x_t)  # frequency content of signal x
    x_f = np.fft.fftshift(x_f)  # re-align frequency content
    freq = np.fft.fftfreq(t.shape[-1], 1/F_s)  # set up frequency axis
    freq = np.fft.fftshift(freq)  # re-align frequency axis

    # frequency phase shift or propagation time delay
    x_f_s = x_f * np.exp(-1j * 2 * pi * freq * (tau - t_i))

    # inverse FFT
    x_f_s = np.fft.ifftshift(x_f_s)  # inverse FFTshift
    x_t_s = np.fft.ifft(x_f_s)  # inverse FFT

    # Doppler frequency shift
    x_t_s = x_t_s * att * np.exp(1j * 2 * pi * F_D * t)

    return x_t_s
# ----------------------------------------------------------------------------------------------------------------------
