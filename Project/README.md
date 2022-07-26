# Installation

## Virtual environment
This code was run using a `miniconda` virtual environment, available to 
download from the [conda website](https://docs.conda.io/en/latest/miniconda.html).

## Required Packages
This code utilized the following packages, all of which should be installable via `conda install`

```` 
  - python=3.7
  - pandas
  - plotly
  - numpy
  - scipy
  - matplotlib
  - jax
  - openpyxl
  - geopandas
  - slycot
  - control
````

# Running the examples

## STS_13

The STS_13 code consists of an example of the vehicle orbiting, an example of an entry comparison 
utilizing two different atmosphere models, and an example with LQR control with uncertain initial conditions.

Each of these python scripts can be executed via `python [script.py]`. Plotting for the entry examples has a script in 
the `plotting` directory. The APOLLO, STS_13 orbit, and STS_13_comparison should generate some plots after running.

There is saved data in the `plotting` directory, so that the sampled trajectories can be plotted efficiently.




