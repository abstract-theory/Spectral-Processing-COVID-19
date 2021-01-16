"""
The software in this repository, or a version of it thereof, was used for the following research paper:

  Spectral Processing of COVID-19 Time-Series Data
  https://arxiv.org/abs/2008.08039

The software produced plots for this paper, and it also implemented theory contained in this paper. When reproducing either this software or the plots created by this software, please attribute or cite either this paper or a future peer-reviewed version of this paper. See the licenses for additional details.

---

1) Plot frequency and phase response for seven-day moving average.
2) Plot frequency response for elliptic filters.
"""

import numpy as np
from scipy.signal import freqz
from scipy.signal import ellip, ellipord
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import common as c


def get_ellip_l1():
    """Elliptical filter frequency response."""
    wp = 1/9.0
    ws = 1/8.0
    gpass = 0.01
    gstop = 40
    N, Wn = ellipord(wp, ws, gpass, gstop, fs=1)
    b, a = ellip(N, gpass, gstop, Wn, 'low', fs=1)
    fs = 1
    M = 512
    f, H = freqz(b, a, fs=fs, worN=M)
    return f, H

def get_ellip_l2():
    """Elliptical filter frequency response."""
    wp = 1/21.0
    ws = 1/19.0
    gpass = 0.01
    gstop = 40
    N, Wn = ellipord(wp, ws, gpass, gstop, fs=1)
    b, a = ellip(N, gpass, gstop, Wn, 'low', fs=1)
    fs = 1
    M = 512
    f, H = freqz(b, a, fs=fs, worN=M)
    return f, H


def get_ellip_h1():
    """Elliptical filter frequency response."""
    wp = 1/7.0
    ws = 1/8.0
    gpass = 0.01
    gstop = 40
    N, Wn = ellipord(wp, ws, gpass, gstop, fs=1)
    b, a = ellip(N, gpass, gstop, Wn, 'high', fs=1)
    fs = 1
    M = 512
    f, H = freqz(b, a, fs=fs, worN=M)
    return f, H


def get_ellip_bp1():
    """Elliptical filter frequency response."""
    wp = [1/8.0, 1/6.0]
    ws = [1/9.0, 1/5.0]
    gpass = 0.01
    gstop = 40
    N, Wn = ellipord(wp, ws, gpass, gstop, fs = 1)
    b, a = ellip(N, gpass, gstop, Wn, 'bandpass', fs = 1)
    fs = 1
    M = 512
    f, H = freqz(b, a, fs=fs, worN=M)
    return f, H


def get_ellip_bp2():
    """Elliptical filter frequency response."""
    wp = [1/19.0, 1/9.0]
    ws = [1/21.0, 1/8.0]
    gpass = 0.01
    gstop = 40
    N, Wn = ellipord(wp, ws, gpass, gstop, fs = 1)
    b, a = ellip(N, gpass, gstop, Wn, 'bandpass', fs = 1)
    fs = 1
    M = 512
    f, H = freqz(b, a, fs=fs, worN=M)
    return f, H


def get_7day_ave():
    """Frequency response of 7-day moving average.

    returns
        f: frequency
        H: frequency response
    """
    b = np.array([1, 1, 1, 1, 1, 1, 1]) * (1/7)
    a = np.array([0, 0, 0, 1, 0, 0, 0]) # time-centered
    fs = 1
    M = 512
    f, H = freqz(b, a, fs=fs, worN=M)
    return f, H


def main(close=True, show=True, save=False, file_names=("Fig4.pdf", "Fig5.pdf", "FigUnused_FR.pdf",)):
    """Filter response plots used in the paper."""

    print(f"\nRunning module \"{__name__}\" ...")

    f, H = get_ellip_l1()
    H_lp1_x2 = H*H
    H_El_lp1_x2_db = 20*np.log10(np.abs(H_lp1_x2))

    f, H = get_ellip_l2()
    H_lp2_x2 = H*H
    H_El_lp2_x2_db = 20*np.log10(np.abs(H_lp2_x2))

    f, H = get_ellip_h1()
    H_hp1_x2 = H*H
    H_El_hp1_x2_db = 20*np.log10(np.abs(H_hp1_x2))

    f, H = get_ellip_bp1()
    H_bp1_x2 = H*H
    H_El_bp1_x2_db = 20*np.log10(np.abs(H_bp1_x2))

    f, H = get_ellip_bp2()
    H_bp2_x2 = H*H
    H_El_bp2_x2_db = 20*np.log10(np.abs(H_bp2_x2))

    f, H_ma = get_7day_ave()
    H_ma_db = 20*np.log10(np.abs(H_ma))
    ma_angles = np.unwrap(np.angle(H_ma)) # phase response


    def y_label(x, p):
        if x/np.pi == 0:
            s = r"       0"
        elif x/np.pi == 1:
            s = r"$\pi$"
        elif x/np.pi == -1:
            s = r"-$\pi$"
        else:
            s = r"%.1f$\pi$" % (x/np.pi)
        return s

    num = c.next_fig_num(close)

    fStyle = (0,(1,1))

    num = c.next_fig_num(close)
    fig1, (ax1) = plt.subplots(1, 1, figsize=(5, 3), num=num)
    ax1.plot(np.array([1/7, 1/7]), np.array([-999, 999]), '#bbbbbb', linewidth=1.0, linestyle=fStyle)
    ax1.plot(np.array([2/7, 2/7]), np.array([-999, 999]), '#bbbbbb', linewidth=1.0, linestyle=fStyle)
    ax1.plot(np.array([3/7, 3/7]), np.array([-999, 999]), '#bbbbbb', linewidth=1.0, linestyle=fStyle)
    ax1.plot(f, H_ma_db, label="moving average", linewidth=1.25)
    ax1.plot(f, H_El_lp2_x2_db, label="elliptic low-pass #2", linewidth=1.25, linestyle=(0,(7,1)))
    ax1.plot(f, H_El_hp1_x2_db, label="elliptic high-pass #1", linewidth=1.25, linestyle=(0,(3,1)))
    ax1.set(xlabel='frequency [1/day]', ylabel='magnitude [dB]')
    ax1.legend(loc='lower right',  prop={'size': 9}, handlelength=2)
    ax1.axis([0.0, 0.5, -120, +5])
    plt.tight_layout()


    num += 1
    fig2, (ax2) = plt.subplots(1, 1, figsize=(5, 2), num=num)
    ax2.plot(np.array([1/7, 1/7]), np.array([-999, 999]), '#bbbbbb', linewidth=1.0, linestyle=fStyle)
    ax2.plot(np.array([2/7, 2/7]), np.array([-999, 999]), '#bbbbbb', linewidth=1.0, linestyle=fStyle)
    ax2.plot(np.array([3/7, 3/7]), np.array([-999, 999]), '#bbbbbb', linewidth=1.0, linestyle=fStyle)
    ax2.plot(f, ma_angles, label="moving average", linewidth=1.25)
    ax2.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 1))
    ax2.get_yaxis().set_major_formatter(ticker.FuncFormatter(y_label))
    ax2.set(xlabel='frequency [1/day]', ylabel='phase angle')
    ax2.legend(loc='upper center', prop={'size': 9}, handlelength=2)
    ax2.axis([0.0, 0.5, -np.pi*1.2, np.pi*1.2])
    plt.tight_layout()


    num += 1
    fig3, (ax3) = plt.subplots(1, 1, figsize=(5, 3), num=num)
    ax3.plot(np.array([1/7, 1/7]), np.array([-999, 999]), '#bbbbbb', linewidth=1.0, linestyle=fStyle)
    ax3.plot(np.array([2/7, 2/7]), np.array([-999, 999]), '#bbbbbb', linewidth=1.0, linestyle=fStyle)
    ax3.plot(np.array([3/7, 3/7]), np.array([-999, 999]), '#bbbbbb', linewidth=1.0, linestyle=fStyle)
    ax3.plot(f, H_El_lp2_x2_db, label="elliptic low-pass #1", linewidth=1.25, linestyle=(0,()))
    ax3.plot(f, H_El_bp1_x2_db, label="elliptic band-pass #1", linewidth=1.25, linestyle=(0,(7,1)))
    ax3.plot(f, H_El_bp2_x2_db, label="elliptic band-pass #2", linewidth=1.25, linestyle=(0,(3,1)))
    ax3.legend(loc='lower right',  prop={'size': 9}, handlelength=2)
    ax3.set(xlabel='frequency [1/day]', ylabel='magnitude [dB]')
    ax3.set(title="Elliptic filter frequency responses", xlabel='frequency [1/day]', ylabel='magnitude [1]')
    ax3.axis([0.0, 0.5, -50, +5])
    plt.tight_layout()


    if show:
        plt.show(block=False)

    if save:
        c.save_fig(fig1, file_names[0])
        c.save_fig(fig2, file_names[1])
        c.save_fig(fig3, file_names[2])


if __name__ == "__main__":
    c.cmd_line_invocation(main)
