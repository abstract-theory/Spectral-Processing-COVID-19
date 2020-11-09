"""
The software in this repository, or a version of it thereof, was used for the following research paper:

  Spectral Processing of COVID-19 Time-Series Data
  https://arxiv.org/abs/2008.08039

The software produced plots for this paper, and it also implemented theory contained in this paper. When reproducing either this software or the plots created by this software, please attribute or cite either this paper or a future peer-reviewed version of this paper. See the licenses for additional details.

---

Compute spectra of deaths and cases for certain geographic areas.
"""

import numpy as np
import matplotlib.pyplot as plt

import common as c


def process_data(data, start, end):
    """Find derivative and compute spectrum"""
    numPts = 1024
    d_dt_data = c.deriv_fft(data)[start:end]
    plotSz = int(numPts/2+0.5)
    win_func = np.hanning(len(d_dt_data))
    f = np.arange(0, plotSz) * 0.5 / plotSz
    Z = np.fft.fft(win_func*d_dt_data, n=int(numPts))/numPts
    Zap = np.abs(Z[:plotSz])
    norm_val = np.max(Zap[int(numPts*0.1+0.5):int(numPts*0.475+0.5)]) # normalize to max value between 0.1 and 0.475
    Zapn = Zap/norm_val
    return f, Zapn


def main(close=True, show=True, save=False, file_names=("Fig1.pdf",)):
    """Create and save frequency spectrum plot for time series (deaths
    and cases)."""

    print(f"\nRunning module \"{__name__}\" ...")

    # some variables
    # countries = ["United Kingdom", "Brazil", "US", "Mexico", "South Africa", "Russia"]
    # countries = ["United Kingdom", "Brazil", "US", "Mexico", "Argentina", "Kenya"]
    countries = ["United Kingdom", "Brazil", "US", "Mexico", "Argentina", "Switzerland"]

    start = 65
    end = c.num_days()
    print("total days: %d, date range: %s, %s" % (end-start, c.ind2datetime(start), c.ind2datetime(end)))
    print("end-start/7 = %f, (end-start)/3.5 = %f" % ((end-start)/7.0, (end-start)/3.5))

    # Create the data
    p_data = dict()
    for country in countries:
        cases = c.get_data("cases", "global", country=country)
        deaths = c.get_data("deaths", "global", country=country)
        f, Z_cases = process_data(cases, start, end)
        f, Z_deaths = process_data(deaths, start, end)
        p_data[country] = {"cases": Z_cases, "deaths": Z_deaths, "f": f}

    # Plot the data
    axis = [0, 0.5, -0.0, 1.05]
    num = c.next_fig_num(close)
    fig = plt.figure(figsize=(10.25, 3), num=num)
    fStyle = (0,(1,1))
    for row in range(2):
        for col in range(3):
            country = countries[col + row*3]
            sub_p_data = p_data[country]
            Z_cases = sub_p_data["cases"]
            Z_deaths = sub_p_data["deaths"]
            f = sub_p_data["f"]

            ax0 = plt.subplot2grid((2, 3), (row, col), colspan=1)
            ax0.plot(np.array([1/7, 1/7]), np.array([-999, 999]), '#bbbbbb', linewidth=1.0, linestyle=fStyle)
            ax0.plot(np.array([2/7, 2/7]), np.array([-999, 999]), '#bbbbbb', linewidth=1.0, linestyle=fStyle)
            ax0.plot(np.array([3/7, 3/7]), np.array([-999, 999]), '#bbbbbb', linewidth=1.0, linestyle=fStyle)
            ax0.plot(f, Z_cases, label="cases", linewidth=1.25)
            ax0.plot(f, Z_deaths, label="deaths", linewidth=1.25, linestyle=(0,(7,1)))
            ax0.set(title=country)
            ax0.title.set_size(10)
            ax0.get_xaxis().set_tick_params(direction='in')
            ax0.get_yaxis().set_tick_params(direction='in')
            ax0.axis(axis)

            if row == 1:
                ax0.set(xlabel='frequency [1/day]')
            else:
                ax0.set_xticklabels('')

            if col == 0:
                ax0.set(ylabel='magnitude')
            else:
                ax0.set_yticklabels('')

            if (row, col) == (0, 2):
                ax0.legend(bbox_to_anchor=(1.05, 1.3), loc='upper right', prop={'size': 9}, handlelength=2, framealpha=0.92)

    plt.tight_layout()

    if show:
        plt.show(block=False)

    if save:
        c.save_fig(fig, file_names[0])


if __name__ == "__main__":
    c.cmd_line_invocation(c.sys.argv, main)


