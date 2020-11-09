"""
The software in this repository, or a version of it thereof, was used for the following research paper:

  Spectral Processing of COVID-19 Time-Series Data
  https://arxiv.org/abs/2008.08039

The software produced plots for this paper, and it also implemented theory contained in this paper. When reproducing either this software or the plots created by this software, please attribute or cite either this paper or a future peer-reviewed version of this paper. See the licenses for additional details.

---

Band-pass filter Johns Hopkins COVID-19 data.
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
    b, a = ellip(N, gpass, gstop, Wn, 'bandpass', fs=1)
    x2 = lfilter(b, a, x[::-1])
    x3 = lfilter(b, a, x2[::-1])
    return x3


def ellip_spec(f0, f1, M):
    """Compute full spectrum of backwards/forwards band-pass elliptic
    filter."""
    gpass = 0.01
    gstop = 40
    N, Wn = ellipord(f0, f1, gpass, gstop, fs=1)
    b, a = ellip(N, gpass, gstop, Wn, 'bandpass', fs=1)
    _f, H = freqz(b, a, fs=1, whole=True, worN=M)
    return np.abs(H*H)


def main(close=True, show=True, save=False, file_names=("Fig10.pdf", "Fig11.pdf", "FigUnused_BP.pdf",)):
    """Plots to be used in the paper."""

    print(f"\nRunning module \"{__name__}\" ...")

    cases = c.get_data("cases", "US")
    deaths = c.get_data("deaths", "US")

    # Extrapolation
    cases_padded, pad_sz = c.extrapolate(cases)
    deaths_padded, pad_sz = c.extrapolate(deaths)

    numPts = 1024

    # first plot
    wp, ws = [1/8.0, 1/6.0], [1/9.0, 1/5.0]
    y_el_1 = ellip_bf(cases_padded, wp, ws)[pad_sz:-pad_sz] # elliptic
    H_1 = ellip_spec(wp, ws, numPts)
    y_fft_1 = c.apply_spectrum(cases_padded, H_1)[pad_sz:-pad_sz] # FFT

    start1 = 40
    print("plot #1 begins on date: %s" % c.ind2datetime(start1))
    y1_lables = lambda x, p: "%.0fk" % (x/1000)
    x1 = np.arange(0, max(cases[start1:].shape), 1)
    dz = np.zeros(len(x1))

    num = c.next_fig_num(close)
    fig1, (ax1) = plt.subplots(1, 1, figsize=(5, 3.294), num=num)
    ax1.fill_between(x1, cases[start1:], dz, alpha=0.1, color="#000000", label="daily deaths")
    ax1.plot(x1, y_el_1[start1:], linewidth=1.25, label="elliptic band-pass #1", linestyle=(0,(2,1)))
    ax1.plot(x1, y_fft_1[start1:], linewidth=1.25, label="FFT band-pass #1", linestyle=(0,(9,9)))
    ax1.get_yaxis().set_major_formatter(ticker.FuncFormatter(y1_lables))
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc='upper left', prop={'size': 9}, handlelength=2)
    ax1.set(xlabel='time [days]')
    ax1.set(ylabel='daily new cases')
    ax1.axis([0, max(y_fft_1[start1:].shape)-1, -15000, 80000])
    plt.tight_layout()

    # second plot
    wp, ws = [1/19, 1/9.0], [1/21.0, 1/8.0]
    y_el_2 = ellip_bf(deaths_padded, wp, ws)[pad_sz:-pad_sz] # elliptic
    H_2 = ellip_spec(wp, ws, numPts)
    y_fft_2 = c.apply_spectrum(deaths_padded, H_2)[pad_sz:-pad_sz]

    start3 = 100
    print("plot #2 begins on date: %s" % c.ind2datetime(start3))
    y3_lables = lambda x, p: "%.1fk" % (x/1000)
    x3 = np.arange(0, max(deaths[start3:].shape), 1)
    dz = np.zeros(len(x3))

    num += 1
    fig3, (ax3) = plt.subplots(1, 1, figsize=(5, 3.3), num=num)
    ax3.fill_between(x3, deaths[start3:], dz, alpha=0.1, color="#000000", label="daily deaths")
    ax3.plot(x3, y_el_2[start3:], linewidth=1.25, label="elliptic band-pass #2", linestyle=(0,(2,1)))
    ax3.plot(x3, y_fft_2[start3:], linewidth=1.25, label="FFT band-pass #2", linestyle=(0,(9,9)))
    ax3.get_yaxis().set_major_formatter(ticker.FuncFormatter(y3_lables))
    ax3.grid(True, alpha=0.2)
    ax3.legend(loc='upper right', prop={'size': 9}, handlelength=2)
    ax3.set(xlabel='time [days]')
    ax3.set(ylabel='daily new deaths')
    ax3.axis([0, max(y_fft_2[start3:].shape)-1, -250, 2500])
    plt.tight_layout()

    # third plot - To check if second plot contains any energy leaking
    # in from bandwidth associated with the seven-day oscillation.
    f = np.arange(0, numPts) / numPts
    d_dt_deaths = c.deriv_fft(deaths)
    Z = np.abs(np.fft.fft(d_dt_deaths, n=int(numPts))/numPts)
    norm_val = np.max(Z[int(numPts*0.1+0.5):int(numPts*0.5+0.5)]) # normalize to max value above 0.1 and 0.5
    Z_deaths = Z/norm_val
    dz = np.zeros(len(f))

    num += 1
    sbStyle = (0,(1,1))
    pbStyle = (0,(2,1))
    fig4, (ax4) = plt.subplots(1, 1, figsize=(6, 3.3), num=num)
    ax4.plot(np.array([wp[0]]*2), np.array([-999, 999]), '#666666', linewidth=1.25, linestyle=pbStyle, label="\"pass-band\"")
    ax4.plot(np.array([wp[1]]*2), np.array([-999, 999]), '#666666', linewidth=1.25, linestyle=pbStyle)
    ax4.plot(np.array([ws[0]]*2), np.array([-999, 999]), '#bbbbbb', linewidth=1.25, linestyle=sbStyle, label="\"stop-band\"")
    ax4.plot(np.array([ws[1]]*2), np.array([-999, 999]), '#bbbbbb', linewidth=1.25, linestyle=sbStyle)
    ax4.fill_between(f, Z_deaths, dz, label="full spectrum", color="#555500", alpha=0.1)
    ax4.plot(f, Z_deaths*np.abs(H_2), label="plotted spectrum", linewidth=1.25, linestyle=(0,()))
    ax4.set(title="The spectrum plotted for \"Band-Pass #2\"", xlabel='frequency [1/day]', ylabel='magnitude [1]')
    ax4.title.set_size(10)
    ax4.axis([0, 0.5, -0.0, 1.05])
    ax4.legend(loc='upper right', prop={'size': 9}, handlelength=2, framealpha=0.92)
    plt.tight_layout()

    if show:
        plt.show(block=False)

    if save:
        c.save_fig(fig1, file_names[0])
        c.save_fig(fig3, file_names[1])
        c.save_fig(fig4, file_names[2])


if __name__ == "__main__":
    c.cmd_line_invocation(c.sys.argv, main)

