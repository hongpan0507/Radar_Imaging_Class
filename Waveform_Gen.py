import numpy as np
from scipy.constants import c, pi
import matplotlib.pyplot as plt
from matplotlib import ticker as mtick

# ----------------------------------------------------------------------------------------------------------------------
def matched_filter(x_t_tx, x_t_rx, TD = True):
    h_t = np.conjugate(np.flip(x_t_tx))  # matched filter
    if TD:  # use convolution in time domain
        FMFO = np.convolve(x_t_rx, h_t, 'full')  # full matched filter output; length = size(h_t) + size(x_t_rx) - 1
        MFO = FMFO[(x_t_tx.size-1):FMFO.size]   # remove the delay caused by the matched filter
    # else:   # use multiplication in frequency domain
    #     x_f_tx = np.fft.fft(x_t_tx)     # frequency content of signal x
    #     x_f_tx = np.fft.fftshift(x_f_tx)    # re-align frequency content
    #     x_f_rx = np.fft.fft(x_t_rx)
    #     x_f_rx = np.fft.fftshift(x_f_rx)  # re-align frequency content
    return MFO
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
#   BW = Bandwidth
#   F_s = sampling frequency
#   T_p = Pulse width
def LFM(BW, F_s, T_p, plot = False):
    T_s = 1 / F_s  # ADC sampling period
    nSample = int(T_p * F_s)  # number of samples from ADC
    chp_rt = BW / T_p  # chirp rate
    t = np.arange(0, nSample, 1) * T_s  # (start, stop, step)
    phase = -2 * pi * BW / 2 * t + pi * chp_rt * t**2
    # phase = -2 * pi * BW / 2 * t + pi * chp_rt * np.sqrt(t)   # for testing only
    freq = np.gradient(phase, t)    # derivative; frequency = change of phase over change of time
    # freq = -BW / 2 + chp_rt * t       # for testing only
    # x_t = np.cos(phase)  # un-windowed signal
    x_t = np.exp(1j*phase)  # un-windowed signal
    # x_f = np.fft.fft(x_t)  # frequency content of signal x
    # x_f = np.fft.fftshift(x_f)  # re-align frequency content
    # freq = np.fft.fftfreq(t.shape[-1], 1 / F_s)  # set up frequency axis
    # freq = np.fft.fftshift(freq)  # re-align frequency axis

    if plot:
        fig = plt.figure()
        plt.suptitle("LFM" + ", Chirp Rate = " + str(chp_rt) + "Hz/s, BW = " + str(BW) + "Hz, PW = " + str(T_p) + "s")
        sub_plt_row = 4
        sub_plt_col = 1
        sub_plt_i = 0
        sub_plt_i += 1
        ax1 = plt.subplot(sub_plt_row, sub_plt_col, sub_plt_i)
        ax1.plot(t, x_t.real, color='r')
        ax1.plot(t, x_t.imag, color='b')
        ax1.set_xlabel("Time (s)")
        ax1.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        plt.title("Waveform")

        sub_plt_i += 1
        ax1 = plt.subplot(sub_plt_row, sub_plt_col, sub_plt_i)
        ax1.plot(t, phase)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Phase (Radian)")
        ax1.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        plt.title("Phase")

        sub_plt_i += 1
        ax1 = plt.subplot(sub_plt_row, sub_plt_col, sub_plt_i)
        ax1.plot(t, freq)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Freq (Hz)")
        ax1.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        plt.title("Frequency")
        
        # sub_plt_i += 1
        # ax1 = plt.subplot(sub_plt_row, sub_plt_col, sub_plt_i)
        # ax1.plot(freq, np.abs(x_f))
        # ax1.set_xlabel("Time (s)")
        # ax1.set_ylabel("Freq (Hz)")
        # plt.title("Frequency")

        plt.subplots_adjust(hspace=0.5)
        plt.show()

    return x_t, t
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
#   x_t = sampled waveform, t = time sample
#   tau = propagation delay, t_i = initial sample
#   F_s = sampling frequency
#   att = attenuation, F_D = doppler frequency shift
def propagation(x_t, t, tau, t_i, F_s, att=1, F_D=0, F_0=0, TX_amp=1):
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

    # Doppler frequency shift + 2-way free space loss
    x_t_s = x_t_s * TX_amp * att * np.exp(1j*2*pi*F_D*t) * np.exp(-1j*2*pi*(F_0+F_D)*tau)

    return x_t_s
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
#   x_t = sampled waveform, t = time sample
#   tau = propagation delay, t_i = initial sample
#   F_s = sampling frequency
#   att = attenuation, F_D = doppler frequency shift
def prop_delay(x_t, t, tau, t_i, F_s, att=1, F_D=0, F_0=0):
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


# ----------------------------------------------------------------------------------------------------------------------
# zero pad numpy array such that the size of array is the power of 2
# x_t, t = numpy array
# T_s = ADC sampling period
def zero_pad(x_t, t, T_s):
    x_t_len = x_t.size
    for n in range(0, x_t_len):
        p_o_2 = 2**n    # power of 2
        if p_o_2 >= x_t_len:
            pad_size = p_o_2 - x_t_len
            break
    x_t = np.concatenate((x_t, np.zeros(pad_size)))
    t_pad = np.arange(t.size, t.size + pad_size) * T_s
    t = np.concatenate((t, t_pad))
    return x_t, t, pad_size
# ----------------------------------------------------------------------------------------------------------------------
