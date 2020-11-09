"""
The software in this repository, or a version of it thereof, was used for the following research paper:

  Spectral Processing of COVID-19 Time-Series Data
  https://arxiv.org/abs/2008.08039

The software produced plots for this paper, and it also implemented theory contained in this paper. When reproducing either this software or the plots created by this software, please attribute or cite either this paper or a future peer-reviewed version of this paper. See the licenses for additional details.

---

Plot frequency and phase response for severn numerical differentiation
methods.
"""

import numpy as np
from scipy.signal import freqz
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import common as c


def get_first_diff():
    """Sprectral response of first difference"""
    b = np.array([1, -1])
    a = np.array([1])
    fs = 1
    M = 512
    f, H = freqz(b, a, fs=fs, worN=M)
    return f, H

def get_cent_diff():
    """Sprectral response of 8th order accurate central derivative."""
    b = np.array([1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280])
    a = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])
    fs = 1
    M = 512
    f, H = freqz(b, a, fs=fs, worN=M)
    return f, H

def get_fft_deriv():
    """Sprectral response of FFT derivative."""
    M = 512
    N = 2*M
    freq_l = np.arange(N//2 + N%2) / N
    freq_r = (np.arange(N//2) - N//2 ) / N
    freq = np.concatenate((freq_l, freq_r))
    coef = 2j*np.pi*freq
    return freq[:M], coef[:M]


def main(close=True, show=True, save=False, file_names=("Fig12.pdf", "Fig13.pdf",)):
    """Plot frequency and phase response of derivative algorithms."""

    print(f"\nRunning module \"{__name__}\" ...")

    f, H_cd = get_cent_diff()
    H_cda = np.abs(H_cd)
    H_cd_ph = np.angle(H_cd)

    f, H_fd = get_first_diff()
    H_fda = np.abs(H_fd)
    H_fd_ph = np.angle(H_fd)

    f, H_fft = get_fft_deriv()
    H_ffta = np.abs(H_fft)
    H_fft_ph = np.angle(H_fft)

    def y_label(x, p):
        ad = {1/2: r"$\pi/2$", 0: r"$0$", -1/2: r"$-\pi/2$"}
        try:
            s = ad[x/np.pi]
        except KeyError:
            s = r"%.1f$\pi$" % (x/np.pi)
        return s


    num = c.next_fig_num(close)
    fig2, (ax2) = plt.subplots(1, 1, figsize=(5, 2), num=num)
    ax2.plot(f, H_fft_ph, label="frequency domain", linewidth=1.25)
    ax2.plot(f, H_fd_ph, label="first difference", linewidth=1.25, linestyle=(0,(7,1)))
    ax2.plot(f, H_cd_ph, label="central difference", linewidth=1.25, linestyle=(0,(3,1)))
    ax2.set(xlabel='frequency [1/day]', ylabel='phase angle')
    ax2.legend(loc='center left',  prop={'size': 9}, handlelength=2, bbox_to_anchor=(0.0, 0.425))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(np.pi/2))
    ax2.get_yaxis().set_major_formatter(ticker.FuncFormatter(y_label))
    ax2.axis([0.0, 0.5, -np.pi*0.5*1.2, np.pi*0.5*1.2])
    plt.tight_layout()


    num += 1
    fig1, (ax1) = plt.subplots(1, 1, figsize=(5, 3), num=num)
    ax1.plot(f, H_ffta, label="frequency domain", linewidth=1.25)
    ax1.plot(f, H_fda, label="first difference", linewidth=1.25, linestyle=(0,(7,1)))
    ax1.plot(f, H_cda, label="central difference", linewidth=1.25, linestyle=(0,(3,1)))
    ax1.set(xlabel='frequency [1/day]', ylabel='magnitude [1]')
    ax1.legend(loc='upper left',  prop={'size': 9}, handlelength=2)
    ax1.axis([0.0, 0.5, 0, np.pi])
    plt.tight_layout()

    if show:
        plt.show(block=False)

    if save:
        c.save_fig(fig2, file_names[0])
        c.save_fig(fig1, file_names[1])


if __name__ == "__main__":
    c.cmd_line_invocation(c.sys.argv, main)

