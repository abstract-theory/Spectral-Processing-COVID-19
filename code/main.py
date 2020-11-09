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

import numpy as np
import matplotlib.pyplot as plt

import common as c

import spectrum as s
import time_spectrum as ts
import filt_response as fr
import deriv_response as dr
import lo_pass as lp
import hi_pass as hp
import band_pass as bp
import resynth as rs


def main(close=True, show=True, save=False):
    """Generate all plots.

    args
        close: close any open plots.
        show: display new plots on screen
        save: write new plots to a file

    returns
        None
    """

    print(f"\nRunning module \"{__name__}\" ...")

    N = c.num_days()
    print("There are a total of %d days" % (N))
    print("The earleiest date is %s" % (c.ind2datetime(0)))
    print("The latest date is %s" % (c.ind2datetime(N-1)))

    if close:
        _num = c.next_fig_num(close)
        close = False

    s.main(close=close, show=show, save=save)
    ts.main(close=close, show=show, save=save)
    rs.main(close=close, show=show, save=save)
    fr.main(close=close, show=show, save=save)
    lp.main(close=close, show=show, save=save)
    hp.main(close=close, show=show, save=save)
    bp.main(close=close, show=show, save=save)
    dr.main(close=close, show=show, save=save)


if __name__ == "__main__":
    """Generate all plots.

    Display all plots:
        python3 main.py

    Save plots:
        python3 main.py -s

    Save plots and do not display them:
        python3 main.py -s -h
    """
    c.cmd_line_invocation(c.sys.argv, main)
