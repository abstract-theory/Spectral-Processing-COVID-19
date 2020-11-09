"""
The software in this repository, or a version of it thereof, was used for the following research paper:

  Spectral Processing of COVID-19 Time-Series Data
  https://arxiv.org/abs/2008.08039

The software produced plots for this paper, and it also implemented theory contained in this paper. When reproducing either this software or the plots created by this software, please attribute or cite either this paper or a future peer-reviewed version of this paper. See the licenses for additional details.

---

Calculate a brick wall spectrum.
"""

import numpy as np

def brick_wall_spec(f_lo, f_hi, N):
    """Brick wall band-pass spectrum. Set f_lo = 0 for low pass filter and
    f_hi = 1/2 for high pass filter.

    This function should be easily implemented when described with
    standard mathematics notatation.

    args
        f_lo: lowest frequency in pass-band.
        f_hi: highest frequency in pass-band.
        N: number of points in output

    returns
        H: output window.
    """
    H = np.zeros(N, dtype=np.float64)
    for k in range(N):
        if   N*(f_lo)   <= k <= N*(f_hi):
            H[k] = 1
        elif N*(1-f_hi) <= k <= N*(1-f_lo):
            H[k] = 1
    return H

