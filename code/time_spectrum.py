"""
The software in this repository, or a version of it thereof, was used for the following research paper:

  Spectral Processing of COVID-19 Time-Series Data
  https://arxiv.org/abs/2008.08039

The software produced plots for this paper, and it also implemented theory contained in this paper. When reproducing either this software or the plots created by this software, please attribute or cite either this paper or a future peer-reviewed version of this paper. See the licenses for additional details.

---

Plot the time-spectrum for Brazil.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import common as c


def main(close=True, show=True, save=False, rt="deaths", region="global", country="Brazil", state=None, file_names=("Fig2.pdf",)):
    """time segment FFT analysis.

    keyword args
        close: close plots (boolean)
        show: show plots (boolean)
        save: save plots (boolean)
        rt: record type ("deaths" or "cases")
        region: global or US
        country: country name
        state: name of US State

    returns
        None
    """

    print(f"\nRunning module \"{__name__}\" ...")

    start = 35
    data = c.get_data(rt, region, country=country, state=state)
    data = data[start:]
    data_d_dt = c.deriv_fft(data)

    winSz = 25

    print("time-spectrum start date: %s" % (c.ind2datetime(start)))
    print("initial window center: %s" % (c.ind2datetime(start+winSz/2)))
    print("final window center: %s" % (c.ind2datetime(start+len(data)-1-winSz/2)))
    print("sliding window size: %d" % (winSz))

    N = 1024 # % pts in FFT
    numSamps = len(data)
    numHops = numSamps-winSz # 1 sample hops (sliding window)

    # set frequency range used for plotting
    minFreq = 0.0
    maxFreq = 0.5
    minFreq_ind = int(N*minFreq+0.5)
    maxFreq_ind = int(N*maxFreq+0.5)

    # set frequency range used for normnalization
    normCutMinFreq = 0.1
    normCutMaxFreq = 0.5
    normCutMinFreq_ind = int(N*normCutMinFreq+0.5)
    normCutMaxFreq_ind = int(N*normCutMaxFreq+0.5)


    Z = np.zeros([numHops, maxFreq_ind-minFreq_ind])
    Z_hanning = np.zeros([numHops, maxFreq_ind-minFreq_ind])
    Z_d_dt = np.zeros([numHops, maxFreq_ind-minFreq_ind])

    b_start = 0
    b_stop = winSz

    for n in range(numHops):
        xr =  data[b_start:b_stop]
        X = np.fft.fft(xr, n=N)/N
        X = np.abs( X[:N//2] )
        Z[n] = X[minFreq_ind:maxFreq_ind] / np.max(X[normCutMinFreq_ind:normCutMaxFreq_ind])

        xr =  data[b_start:b_stop]
        X = np.fft.fft(xr*np.hanning(winSz), n=N)/N
        X = np.abs( X[:N//2] )
        Z_hanning[n] = X[minFreq_ind:maxFreq_ind] / np.max(X[normCutMinFreq_ind:normCutMaxFreq_ind])

        xr =  data_d_dt[b_start:b_stop]
        X = np.fft.fft(xr, n=N)/N
        X = np.abs( X[:N//2] )
        Z_d_dt[n] = X[minFreq_ind:maxFreq_ind] / np.max(X[normCutMinFreq_ind:normCutMaxFreq_ind])

        b_start += 1
        b_stop  += 1

    # clip values that exceed unity.
    Z = np.clip(Z, 0, 1)
    Z_hanning = np.clip(Z_hanning, 0, 1)
    Z_d_dt = np.clip(Z_d_dt, 0, 1)


    # interp = None # smallest size
    interp = 'sinc'
    colors = cm.Blues_r
    # colors = cm.magma # largest size, percetual color map
    # colors = cm.gray # smallest size


    xMin = winSz*0.5
    xMax = winSz*0.5 + Z.shape[0]

    num = c.next_fig_num(close)
    fig = plt.figure(figsize=(10.25, 2.3), num=num)

    # --------------------------------------------------------------
    ax0 = plt.subplot2grid((1, 3), (0, 0))
    ax0.imshow(np.flipud(Z.transpose(1,0)), interpolation=interp, cmap=colors, extent=(xMin, xMax, minFreq, maxFreq), aspect='auto')
    ax0.set(xlabel='time [days]', title="Square Window", ylabel='frequency [1/day]')
    ax0.title.set_size(10)
    ax0.yaxis.set_major_locator(plt.MultipleLocator(0.1))

    # --------------------------------------------------------------
    ax1 = plt.subplot2grid((1, 3), (0, 1))
    ax1.imshow(np.flipud(Z_hanning.transpose(1,0)), interpolation=interp, cmap=colors, extent=(xMin, xMax, minFreq, maxFreq), aspect='auto')
    ax1.set(xlabel='time [days]', title="Hanning Window")
    ax1.title.set_size(10)
    ax1.set_yticklabels('')
    ax1.tick_params(axis=u'y', which=u'both',length=0)

    # --------------------------------------------------------------
    ax2 = plt.subplot2grid((1, 3), (0, 2))
    ax2.imshow(np.flipud(Z_d_dt.transpose(1,0)), interpolation=interp, cmap=colors, extent=(xMin, xMax, minFreq, maxFreq), aspect='auto')
    ax2.set(xlabel='time [days]', title="Square Window of Derivative")
    ax2.title.set_size(10)
    ax2.set_yticklabels('')
    ax2.tick_params(axis=u'y', which=u'both',length=0)

    plt.tight_layout()

    if show:
        plt.show(block=False)

    if save:
        c.save_fig(fig, file_names[0])


if __name__ == "__main__":
    c.cmd_line_invocation(c.sys.argv, main)

