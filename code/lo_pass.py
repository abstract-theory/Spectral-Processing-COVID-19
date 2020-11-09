"""
The software in this repository, or a version of it thereof, was used for the following research paper:

  Spectral Processing of COVID-19 Time-Series Data
  https://arxiv.org/abs/2008.08039

The software produced plots for this paper, and it also implemented theory contained in this paper. When reproducing either this software or the plots created by this software, please attribute or cite either this paper or a future peer-reviewed version of this paper. See the licenses for additional details.

---

Low-pass filter Johns Hopkins COVID-19 data.
"""

import numpy as np
from scipy.signal import ellip, ellipord, lfilter, freqz
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import common as c


def seven_day_ave_manual(x):
    """Manually computer moving average filter. """
    x2 = np.zeros(len(x) - 6)
    for n in range(len(x2)):
        x2[n] = np.sum(x[n:n+7]) / 7
    return x2


def ellip_bf(x, f0, f1):
    """backwards/forwards elliptic filter"""
    gpass = 0.01
    gstop = 40
    N, Wn = ellipord(f0, f1, gpass, gstop, fs=1)
    b, a = ellip(N, gpass, gstop, Wn, 'low', fs=1)
    x2 = lfilter(b, a, x[::-1])
    x3 = lfilter(b, a, x2[::-1])
    return x3


def ellip_spec(f0, f1, M):
    """Compute full spectrum of backwards/forwards low-pass elliptic
    filter."""
    gpass = 0.01
    gstop = 40
    N, Wn = ellipord(f0, f1, gpass, gstop, fs=1)
    b, a = ellip(N, gpass, gstop, Wn, 'low', fs=1)
    f, H = freqz(b, a, fs=1, whole=True, worN=M)
    return np.abs(H*H)


def main(close=True, show=True, save=False, file_names=("Fig6.pdf", "Fig7.pdf", "Fig8.pdf",)):
    """Plots to be used in the paper."""

    print(f"\nRunning module \"{__name__}\" ...")


    deaths = c.get_data("deaths", "US")
    deaths_padded, pad_sz = c.extrapolate(deaths)

    m_ave = seven_day_ave_manual(deaths_padded)[pad_sz-3:-(pad_sz-3)] # moving average

    wp, ws = 1/9, 1/8
    y_el_1 = ellip_bf(deaths_padded, wp, ws)[pad_sz:-pad_sz] # elliptic
    H = ellip_spec(wp, ws, 1024)
    y_fft_1 = c.apply_spectrum(deaths_padded, H)[pad_sz:-pad_sz] # FFT

    wp, ws = 1/21, 1/19
    y_el_2 = ellip_bf(deaths_padded, wp, ws)[pad_sz:-pad_sz] # elliptic
    H = ellip_spec(wp, ws, 1024)
    y_fft_2 = c.apply_spectrum(deaths_padded, H)[pad_sz:-pad_sz] # FFT


    start = 100
    print("plots #1, #2, and #3 begin on date: %s" % c.ind2datetime(start))

    y_lables = lambda x, p: "%.1fk" % (x/1000)
    x = np.arange(0, max(deaths[start:].shape), 1)
    dz = np.zeros(len(x))

    num = c.next_fig_num(close)
    fig1, (ax1) = plt.subplots(1, 1, figsize=(5, 3), num=num)
    ax1.fill_between(x, deaths[start:], dz, alpha=0.1, color="#000000", label="daily deaths")
    ax1.plot(x, m_ave[start:], linewidth=1.25, label="moving average")
    ax1.set(xlabel='time [days]', ylabel='daily new deaths')
    ax1.get_yaxis().set_major_formatter(ticker.FuncFormatter(y_lables))
    ax1.grid(True, alpha=0.2)
    ax1.axis([min(x), max(x), 0, 2500])
    ax1.legend(loc='upper right', prop={'size': 9}, handlelength=2)
    plt.tight_layout()

    y_lables = lambda x, p: "%.1fk" % (x/1000)
    x = np.arange(0, max(deaths[start:].shape), 1)
    dz = np.zeros(len(x))

    num += 1
    fig2, (ax2) = plt.subplots(1, 1, figsize=(5, 3), num=num)
    ax2.fill_between(x, deaths[start:], dz, alpha=0.1, color="#000000", label="daily deaths")
    ax2.plot(x, y_el_1[start:], linewidth=1.25, label="elliptic low-pass #1", linestyle=(0,(2,1)))
    ax2.plot(x, y_fft_1[start:], linewidth=1.25, label="FFT low-pass #1", linestyle=(0,(9,9)))
    ax2.set(xlabel='time [days]', ylabel='daily new deaths')
    ax2.get_yaxis().set_major_formatter(ticker.FuncFormatter(y_lables))
    ax2.grid(True, alpha=0.2)
    ax2.axis([min(x), max(x), 0, 2500])
    ax2.legend(loc='upper right', prop={'size': 9}, handlelength=2)
    plt.tight_layout()

    y_lables = lambda x, p: "%.1fk" % (x/1000)
    x = np.arange(0, max(deaths[start:].shape), 1)
    dz = np.zeros(len(x))

    num += 1
    fig4, (ax4) = plt.subplots(1, 1, figsize=(5, 3), num=num)
    ax4.fill_between(x, deaths[start:], dz, alpha=0.1, color="#000000", label="daily deaths")
    ax4.plot(x, y_el_2[start:], linewidth=1.25, label="elliptic low-pass #2", linestyle=(0,(2,1)))
    ax4.plot(x, y_fft_2[start:], linewidth=1.25, label="FFT low-pass #2", linestyle=(0,(9,9)))
    ax4.set(xlabel='time [days]', ylabel='daily new deaths')
    ax4.get_yaxis().set_major_formatter(ticker.FuncFormatter(y_lables))
    ax4.grid(True, alpha=0.2)
    ax4.axis([min(x), max(x), 0, 2500])
    ax4.legend(loc='upper right', prop={'size': 9}, handlelength=2)
    plt.tight_layout()

    if show:
        plt.show(block=False)
    if save:
        c.save_fig(fig1, file_names[0])
        c.save_fig(fig2, file_names[1])
        c.save_fig(fig4, file_names[2])


if __name__ == "__main__":
    c.cmd_line_invocation(c.sys.argv, main)

