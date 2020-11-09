"""
Licenses
========

This repository is subject to the two licenses described below.

************************************************************************

License #1:
-----------

Attribution
    The software in this repository, or a version of it thereof, was used for the following research paper:

        Spectral Processing of COVID-19 Time-Series Data
        https://arxiv.org/abs/2008.08039

    The software produced plots for this paper, and it also implemented theory contained in this paper. When reproducing either this software or the plots created by this software, please attribute or cite either this paper or a future peer-reviewed version of this paper.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files, to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, and to permit persons to whom the software is furnished to do so, subject to the following conditions:

    A) All copies of this software, in whole or in part, and all products that incorporate this software, in whole or in part, in any format (including source code, byte code, binary, compiled, transpiled, translated, ported, and modified) shall include the following item(s):

        1. The above attribution.
        2. This license in its entirety.

    B) All services that incorporate this software, in whole or in part, shall communicate the above attribution to the recipient of those services. If some aspect of the service can be relicensed to a third party, it will be considered a product and subject to the terms described in section A.

The granted permissions described above may be made more restrictive by adding amendments to the end of this license.

THE SOURCE CODE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOURCE CODE OR THE USE OR OTHER DEALINGS IN THE SOURCE CODE.

************************************************************************

License #2:
-----------

The data files (the CSV files) contained in this repository were produced by

    COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University
    https://github.com/CSSEGISandData/COVID-19

This data is subject to the terms of use as set forth by the entity named above (CC BY 4.0).

"""

import os
import sys
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# path to CSV files in Johns Hopkins git repo:
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATA_PATH = os.path.join(BASE_DIR, "data", "latest")
DATA_PATH = os.path.join(BASE_DIR, "data", "20201007")


# data file names
FN_D_US = "time_series_covid19_deaths_US.csv"
FN_C_US = "time_series_covid19_confirmed_US.csv"
FN_D_G = "time_series_covid19_deaths_global.csv"
FN_C_G = "time_series_covid19_confirmed_global.csv"


# path to save figures in
FIG_PATH = os.path.join(BASE_DIR, 'figures')


# load all data into memory only once
try:
    d_us = pd.read_csv(os.path.join(DATA_PATH, FN_D_US))
    c_us = pd.read_csv(os.path.join(DATA_PATH, FN_C_US))
    d_g = pd.read_csv(os.path.join(DATA_PATH, FN_D_G))
    c_g = pd.read_csv(os.path.join(DATA_PATH, FN_C_G))
except FileNotFoundError:
    print("\nERROR: Could not read data file.")
    print("Does the variable DATA_PATH in \"common.py\" point to Johns Hopkins' git repo?")
    exit()


# group all dataframes into an object calld "data"
data = {"deaths": {"US": d_us, "global": d_g}, "cases": {"US": c_us, "global": c_g}}


def get_data(record_type, region, country=None, state=None, admin2=None):
    """Returns new daily count of cases/deaths for input parameters.

    args
        record_type: "deaths" or "cases"
        region: "global" or "US"
        state: state name
        admin2: locality
        country: county name

    returns
        daily count as a numpy array
    """
    df = data[record_type][region]
    if region == "global":
        if country:
            df = df[df['Country/Region'] == country]
        if state:
            df = df[df['Province/State'] == state]
    else:
        if state:
            df = df[df['Province_State'] == state]
        if admin2:
            df = df[df['Admin2'] == admin2]

    cumulative = df.loc[:, '1/22/20':].sum(axis=0) # sum over each date starting with earliet date (1/22/20).
    daily = np.array(cumulative.diff()[1:]) # find first difference, and convert to numpy array.
    return daily


def date_range(df):
    """Abtain the earliest and latest dates contained in the dataframe columns."""
    min_date = datetime.strptime("9999", '%Y')
    max_date = datetime.strptime("0001", '%Y')
    for col in df.columns:
        try:
            # try to parse dates, JHU date format is '%m/%d/%y'
            date = datetime.strptime(col, '%m/%d/%y')
            if date < min_date:
                min_date = date
            if date > max_date:
                max_date = date
        except ValueError:
            continue
    return min_date, max_date


def ind2datetime(N):
    """Return date string for Nth day from epoch.

    args
        N: index to the day

    returns
        new_date: datetime of Nth data point.

    Notes: 1 day is added to the "first" time. This is an artifact of the
    diff() operation. 12 hours is also added. This is because the data
    point is best associated with the time of 12 noon.
    """
    first, _last = date_range(c_g)
    new_date = first + timedelta(days=N) + timedelta(days=1) + timedelta(hours=12)
    return new_date

def num_days():
    """Number of days in first difference"""
    first, last = date_range(c_g)
    N =  (last - first).days
    return N


def days2decimals(dt):
    """Convert datetime to float in the range of [0.0, 7.0)."""
    return dt.isoweekday() + dt.hour/24 + dt.minute/(24*60) + dt.second/(24*60*60) + dt.microsecond/(24*60*60*1000)


def save_fig(fig, file_name):
    """Save plot"""
    file_path_name = os.path.join(FIG_PATH, file_name)
    print(f"saving plot as \"{file_path_name}\"")
    metadata = {"Title": "Spectral Processing of COVID-19 Time-Series Data", "Author": "McGovern S.", "Subject": "COVID-19 data analysis"}
    fig.savefig(file_path_name, metadata=metadata)


def next_fig_num(close):
    """Returns figure number for plots.

    args
        close: True if open plots should be closed.
    """
    if close:
        plt.close('all')
    try:
        num = plt.get_fignums()[-1]+1
    except IndexError:
        num = 1
    return num


def extrapolate(t_series, weeks=4):
    """Pad data with linearly extrapolated numbers.

    args
        t_series: numpy array to be extrapolated.
        weeks: number of weeks to be extraplated.

    returns
        t_series: extrapolated data
        pad_sz: the total number of days that are extrapolated at each
        end of the time-series.
    """
    for _ in range(weeks):
        pad_left = 2*t_series[:7] - t_series[7:2*7]
        pad_right = 2*t_series[-7:] - t_series[-2*7:-7]
        t_series = np.concatenate((pad_left, t_series, pad_right))
    t_series[t_series < 0] = 0
    pad_sz = weeks * 7
    return t_series, pad_sz


def deriv_fft(y):
    """Estimate derivative using FFT."""
    N = len(y)
    l = np.arange(N) - N/2 + (N%2)*0.5
    coef = 2j*np.pi*(l/N)
    Y = np.fft.fftshift(np.fft.fft(y))
    yp = np.fft.ifft(np.fft.ifftshift(Y*coef))
    return np.real(yp)

def apply_spectrum(x, H):
    """Apply a computed spectrum to series data.

    args
        x: series data
        H: computed spectrum

    returns
        y: modified time-series"""
    LX = len(H)
    X = np.fft.fft(x, LX)
    yb_padded = np.real(np.fft.ifft(H*X, LX))
    y = yb_padded[:len(x)]
    return y

def cmd_line_invocation(argv, func):
    """Helper function for running modules independently.

    Display all plots:
        python3 file_name.py

    Save plots:
        python3 file_name.py -s

    Save plots and do not display them:
        python3 file_name.py -s -h
    """

    argv = sys.argv
    kw = {}
    for n in range(len(argv)):
        if argv[n] == "-s":
            kw['save'] = True
        elif argv[n] == "-h":
            kw['show'] = False

    func(**kw)

    print()
    if 'show' not in kw:
        input("Press ENTER to close plots.")
        print("Add the \"-h\" option to hide plots.")
    if 'save' not in kw:
        print("Add the \"-s\" option to write plots to files.")
