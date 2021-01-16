"""
The software in this repository, or a version of it thereof, was used for the following research paper:

  Spectral Processing of COVID-19 Time-Series Data
  https://arxiv.org/abs/2008.08039

The software produced plots for this paper, and it also implemented theory contained in this paper. When reproducing either this software or the plots created by this software, please attribute or cite either this paper or a future peer-reviewed version of this paper. See the licenses for additional details.

---

Resynthesize the oscillations for cases and deaths for several
geographics areas.
"""

from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt

import common as c

def make_plot(phf, x, countries, close=False, show=True, save=False, file_name=None):
    """Create plot."""

    days = ["S", "M", "T", "W", "T", "F", "S"]
    xticks = np.arange(7.5, 21.5, 1.0) # xticks at 12-noon
    labels = [days[int(n)%7] for n in xticks]

    num = c.next_fig_num(close)
    fig = plt.figure(figsize=(10.25, 2), num=num)

    for n_co in range(len(countries)):
        co = countries[n_co]
        ax0 = plt.subplot2grid((1, 5), (0, n_co))

        ax0.plot(x, phf['cases'][co], linewidth=1.25, label="cases", linestyle=(0,()))
        ax0.plot(x, phf['deaths'][co], linewidth=1.25, label="deaths", linestyle=(0,(6,1)))
        for n in range(7, 21, 2):
            ax0.fill_between([n, n+1], [-2, -2], [2, 2], alpha=1.0, color="#eeeeee")

        ax0.axis([7, 21, -1.5, 1.2])
        ax0.set_xticks(xticks)
        ax0.set_xticklabels(labels, fontsize=6)
        ax0.tick_params(axis="x",direction="in", pad=-10)
        ax0.set_yticklabels('')
        ax0.tick_params(axis=u'both', which=u'both', length=0)
        ax0.set(xlabel='day of the week', title=co)
        ax0.title.set_size(10)

        if n_co == 4:
            leg = ax0.legend(bbox_to_anchor=(1.11, 0.5), loc='upper right', prop={'size': 9}, handlelength=1.5, framealpha=0.92)
            leg.set_in_layout(False)

    plt.tight_layout()

    if show:
        plt.show(block=False)
    if save:
        c.save_fig(fig, file_name)


def text_summary(phf, t_ref, xs):
    """Text summary of results"""
    print("    %015s | %08s | %13s | %13s |" % ("country", "type", "min", "max"))
    print("    " + "-" * (15+8+13*2+4*3))
    for rt in phf:
        for country in phf[rt]:
            days_to_min = xs[0, np.argmin(phf[rt][country])]
            days_to_max = xs[0, np.argmax(phf[rt][country])]
            day_min = t_ref + timedelta(days=days_to_min)
            day_max = t_ref + timedelta(days=days_to_max)
            print("    %015s | %08s | %13s | %13s |" % (country, rt, day_min.strftime("%a, %H:%M:%S"), day_max.strftime("%a, %H:%M:%S")))
    print()


def main(N=3, close=True, show=True, save=False, file_names=("Fig3.pdf",)):
    """Reconstruct the waveform with 1, 2, or 3 sinusoids. Then search
    for min and max, provided that the harmonics sufficiently big.

    args
        N (int): The number of harmonics to consider (1, 2, or 3).
        close (bool): True if existing plots will be closed.
        show (bool): True if the plot will be displayed on screen.
        save (bool): True if the plot will be saved to file.

    returns
        None
    """

    print(f"\nRunning module \"{__name__}\" ...")

    # some input variables
    start = 70
    end = c.num_days() - 5 # allowing five days for revisions to be made to the repo
    fft_pts = 2**19
    print("Analysis date range: %s - %s" % (c.ind2datetime(start), c.ind2datetime(end)))
    print("Analysis window size: %d" % (end-start))
    print("FFT size: 2^%d = %d points" % (np.log2(fft_pts), fft_pts))

    # The frequencies we are interested in.
    f = np.array([[1/7, 2/7, 3/7]])[:,:N]

    # List of countries to be included in the plots and print out.
    countries = ["US", "Mexico", "Argentina", "United Kingdom", "Brazil"]

    # The datetime at which the integrated derivative begins (time is 12:00 noon).
    t_ref = c.ind2datetime(start) # for FFT derivative plus symbolic integration
    print("Reference time for the integrated derivative: %s" % t_ref)


    num_days = 21 # number of days to synthesize
    step_sz = 20 / (24*60*60) #  step size for synthesis (20 seconds)
    xs = np.array([np.arange(0, num_days, step_sz)]) # time-axis for calculating sinusoids
    x = xs[0] + c.days2decimals(t_ref) # time-axis for plotting sinusoids
    print("Step size for resynthesis: %1.4f seconds" % (step_sz*24*60*60))

    print()

    # This will be the main data object of interest.
    phf = {"cases": dict(), "deaths": dict()}

    for rt in phf:
        for country in countries:
            data = c.get_data(rt, "global", country=country, state=None)

            d_dt_data = c.deriv_fft(data)[start:end] # FFT derivative
            Z = np.fft.fft(d_dt_data, n=int(fft_pts))/fft_pts # spectrum of derivative

            H_f = Z[(fft_pts*f+0.5).astype(np.int)] # freqs. of interest
            theta = np.angle(H_f) # phase angles
            m = np.abs(H_f) # magnitudes
            y = (m/(2*np.pi*f)) * np.sin(2*np.pi*xs.T*f + theta) # Compute integrated sinusoids
            y_sum = np.sum(y, 1) # Sum sinusoids

            y_sum = y_sum - (np.max(y_sum)+np.min(y_sum))/2 # align vertically for plot
            y_sum = y_sum/max(np.abs(y_sum)) # normalize for plot

            phf[rt][country] = y_sum # Collect the data

    # Text summary of results
    text_summary(phf, t_ref, xs)

    # Plot
    if show or save:
        make_plot(phf, x, countries, close=close, show=show, save=save, file_name=file_names[0])


if __name__ == "__main__":
    c.cmd_line_invocation(main)

