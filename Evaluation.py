import math
import numpy as np
import scipy.signal as signal

'''
NMSE Calculation
'''


def NMSE(error, out_ignore_first_mem):

    num = np.mean(np.square(np.abs(error)))
    den = np.mean(np.square(np.abs(out_ignore_first_mem)))

    nmse = 10 * math.log((num / den), 10)

    print('NMSE = ', nmse)


'''
ACEPR Calculation
'''


def ACEPR_cal(error, inp_ignore_first_mem):
    Z = error + inp_ignore_first_mem
    Z = Z.reshape(len(Z), )

    fs = 983.04e6
    nfft = len(Z)
    window = signal.hann(len(Z), False)

    freq, Pin = signal.periodogram(Z, fs, window, nfft)

    Pin = np.fft.fftshift(Pin)
    Freq = np.fft.fftshift(freq)

    P0 = 0
    P1 = 0
    P2 = 0

    for i in range(len(Freq)):
        if Freq[i] > - 100e6 and Freq[i] < 100e6:
            P0 += Pin[i]

        elif Freq[i] > - 200e6 and Freq[i] < -100e6:
            P1 += Pin[i]

        elif Freq[i] > 100e6 and Freq[i] < 200e6:
            P2 += Pin[i]

    ACEPR_left = 10 * math.log((P1 / P0), 10)
    ACEPR_right = 10 * math.log((P2 / P0), 10)
    ACEPR = max(ACEPR_left, ACEPR_right)
    print('ACEPR = ', ACEPR)