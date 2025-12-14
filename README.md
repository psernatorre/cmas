# CMAS

Welcome! This repository contains cmaspy, a package for the Control of multi-agent systems (MAS). We have also developed its counterpart of cmaspy in Matlab, named cmasmat. 

## Installation (Python)

1. **Download cmaspy**: Make sure you have [python3.12](https://www.python.org/downloads/) (or greater) installed on your computer. Start by cloning this repository and navigating into the cmaspy directory.
    ```
    $ git clone https://github.com/REAM-lab/CMAS
    $ cd CMAS
    ```
    Next, create a virtual python environment and install cmaspy in the virtual environment. To this end, execute the following:
    ```
    $ python3.12 -m venv .venv 
    $ source .venv/bin/activate
    (.venv)$ pip install -e .  
    ```

2. **Install a solver for Semidefinite programming **: Most of the functions in this repository requires a solver, like MOSEK, to solve semidefinite programming. 

3. **Run examples**: To ensure that cmaspy was installed correctly navigate to the examples/python folder. Then, within your python virtual environment, launch python3.12 and execute the examples.

## Installation (Matlab)

1. **Download cmasmat**: Make sure you have [MATLAB>R2025a] (or greater) installed on your computer. Start by cloning this repository and navigating into the cmasmat directory.

2. **Add to your path folder**: On the Home tab, click Set Path, click Add folder, then click Add folder with Subfolders. Select the folder cmasmat and click Open. 

3. **Run examples**: To ensure that cmasmat was installed correctly navigate to the examples/matlab folder, and execute one of the examples.