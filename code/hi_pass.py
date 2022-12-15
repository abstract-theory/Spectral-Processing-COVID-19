"""
The software in this repository, or a version of it thereof, was used for the following research paper:

  Spectral Processing of COVID-19 Time-Series Data
  https://arxiv.org/abs/2008.08039

The software produced plots for this paper, and it also implemented theory contained in this paper. When reproducing either this software or the plots created by this software, please attribute or cite either this paper or a future peer-reviewed version of this paper. See the licenses for additional details.

---

High-pass filter Johns Hopkins COVID-19 data.
"""

import numpy as np
from scipy.signal import ellip, ellipord, lfilter, freqz
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import common as c


def ellip_bf(x, f0, f1):
    """backwards/forwards elliptic filter"""
    gpass = 0.01
    gstop = 40
    N, Wn = ellipord(f0, f1, gpass, gstop, fs=1)
    b, a = ellip(N, gpass, gstop, Wn, 'high', fs=1)
    x2 = lfilter(b, a, x[::-1])
    x3 = lfilter(b, a, x2[::-1])
    return x3


def ellip_spec(f0, f1, M):
    """Compute full spectrum of backwards/forwards high-pass elliptic
    filter."""
    gpass = 0.01
    gstop = 40
    N, Wn = ellipord(f0, f1, gpass, gstop, fs=1)
    b, a = ellip(N, gpass, gstop, Wn, 'high', fs=1)
    f, H = freqz(b, a, fs=1, whole=True, worN=M)
    return np.abs(H*H)


def main(close=True, show=True, save=False, file_names=("Fig9.pdf",)):
    """Plots to be used in the paper."""

    print(f"\nRunning module \"{__name__}\" ...")

    cases = c.get_data("cases", "US")
    cases_padded, pad_sz = c.extrapolate(cases)

    fft_sz = 2
    max_l = max(1024, len(cases_padded))
    while fft_sz < max_l:
        fft_sz *= 2

    f0, f1 = 1/7.0, 1/8.0
    y_el_1 = ellip_bf(cases_padded, f0, f1)[pad_sz:-pad_sz]
    H = ellip_spec(f0, f1, fft_sz)
    y_fft_1 = c.apply_spectrum(cases_padded, H)[pad_sz:-pad_sz]

    start = 40
    print("plot #1 begins on date: %s" % c.ind2datetime(start))
    y_label = lambda x, p: "%.0fk" % (x/1000)
    x = np.arange(0, max(cases[start:].shape), 1)
    dz = np.zeros(len(x))

    num = c.next_fig_num(close)
    fig, (ax3) = plt.subplots(1, 1, figsize=(5, 3.294), num=num)
    ax3.fill_between(x, cases[start:], dz, alpha=0.1, color="#000000", label="daily cases")
    ax3.plot(x, y_el_1[start:], linewidth=1.25, label="elliptic high-pass #1", linestyle=(0,(2,1)))
    ax3.plot(x, y_fft_1[start:], linewidth=1.25, label="FFT high-pass #1", linestyle=(0,(9,9)))
    ax3.set(xlabel='time [days]', ylabel='daily new cases')
    ax3.get_yaxis().set_major_formatter(ticker.FuncFormatter(y_label))
    ax3.grid(True, alpha=0.2)
    ax3.axis([min(x), max(x), -150000, 800000])
    ax3.legend(loc='upper left', prop={'size': 9}, handlelength=2)
    plt.tight_layout()

    if show:
        plt.show(block=False)
    if save:
        c.save_fig(fig, file_names[0])


if __name__ == "__main__":
    c.cmd_line_invocation(main)
