====================================================
Spectral Processing of COVID-19 Time-Series Data
====================================================

This code was used for the below research paper to both produce plots and implement theory.

    | **Spectral Processing of COVID-19 Time-Series Data**
    | https://arxiv.org/abs/2008.08039

-----------------------------------------------------

Generating The Plots
********************

To generate all 15 plots and display them on your monitor, run the script ``main.py``. For example,

.. code-block:: sh

    python3 main.py

If there are display problems, instead save the plots by running,

.. code-block:: sh

    python3 main.py -sq

The plots will be written to the ``figures`` directory in your local copy of this repository. The display problems are related to Matplotlib's back-end, and, more specifically, the ``False`` in the ``plt.show(block=False)`` statements. There are various approaches for resolving this issue.


**Command Line Options:**

    | ``-s``    Write the plots to PDF files.
    | ``-q``    Do not display the plots on screen.

The script ``main.py`` calls eight other sub-modules to generate the plots. Alternatively, sub-modules can be called independently using the same method as that for ``main.py``. The sub-modules are listed below.

    | spectrum.py
    | time_spectrum.py
    | resynth.py
    | filt_response.py
    | lo_pass.py
    | hi_pass.py
    | band_pass.py
    | deriv_response.py


-----------------------------------------------------

Updating Data
*************
To use more recent data, download Johns Hokins' COVID-19 data repository. Then set the variable "DATA_PATH" in the file "common.py" to point to the directory in the repo that contains the following files:

    | time_series_covid19_deaths_US.csv
    | time_series_covid19_confirmed_US.csv
    | time_series_covid19_deaths_global.csv
    | time_series_covid19_confirmed_global.csv


-----------------------------------------------------

Requirements
*******************
- Python 3
- pandas
- matplotlib
- scipy
- numpy

-----------------------------------------------------

Credits
*******
This repository contains data files (the CSV files) that were produced by

        | **COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University**
        | https://github.com/CSSEGISandData/COVID-19

See the license file for additional details.
