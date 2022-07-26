# Installation

## Virtual environment
This code was run using a `miniconda` virtual environment, available to 
download from the [conda website](https://docs.conda.io/en/latest/miniconda.html).

## Required Packages
In your new virtual environment, utilize pip or conda to install the following packages
```` 
  - pandas
  - numpy
  - scipy
  - matplotlib
  - control
````

On my system, I used `python3.7` and installed the above utilizing `conda install`.

Due to package interdependencies, it is usually faster to install `geopandas` via `pip`. 

`pip install geopandas`

The final step to be able to run all the examples is to install the development package locally. First change directories
into the `Project` folder. In this directory there should be the `edl_control` folder, as well as `setup.py`. Next run `pip install -e .`
This will make the `edl_control` development package available to your virtualenv, and you should be able to import it. 

# Running the examples
Make sure that you are in the `Project/edl_control/examples` directory when running these.

## STS_13

The STS_13 code consists of an example of the vehicle orbiting, an example of an entry comparison 
utilizing two different atmosphere models, and an example with LQR control with uncertain initial conditions.

Each of these python scripts can be executed via `python [script.py]`. Plotting for the entry examples has a script in 
the `plotting` directory. The APOLLO, STS_13 orbit, and STS_13_comparison should generate some plots after running.

There is saved data in the `plotting` directory, so that the sampled trajectories can be plotted efficiently.




