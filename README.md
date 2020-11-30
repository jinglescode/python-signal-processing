# Python Signal Processing

Signal processing can be daunting; this repo contains tutorials on understanding and applying signal processing using NumPy and PyTorch.

**splearn** is a package for signal processing and machine learning with Python. It is built on top of [NumPy](https://numpy.org) and [SciPy](https://www.scipy.org), to provide easy to use functions from common signal processing tasks to machine learning. 

## Table of Contents

- [Tutorials](#tutorials)
- [Getting Started](#getting-started)
- [Disclaimer on Datasets](#disclaimer-on-datasets)

--- 

## Tutorials

We aim to bridge the gap for anyone who are new signal processings to get started, check out the [tutorials](https://github.com/jinglescode/python-signal-processing/tree/master/tutorials) to get started on signal processings.

### 1. Signal composition (time, sampling rate and frequency)

In order to begin the signal processing adventure, we need to understand what we are dealing with. In the first tutorial, we will uncover what is a signal, and what it is made up of. We will look at how the sampling rate and frequency can affect a signal. We will also see what happens when we combine multiple signals of different frequencies.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jinglescode/python-signal-processing/blob/master/tutorials/Signal%20composition%20-%20time%2C%20sampling%20rate%20and%20frequency.ipynb)

### 2. Fourier Transform

In the first tutorial, we learned that combining multiple signals will produce a new signal where all the frequencies are jumbled up. In this tutorial, we will learn about Fourier Transform and how it can take a complex signal and decompose it to the frequencies that made it up.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jinglescode/python-signal-processing/blob/master/tutorials/Fourier%20Transform.ipynb)

### 3. Denoising with mean-smooth filter

How can we apply the simplest filter to perform denoising? Introduce the running mean filter; we can remove noise that is normally distributed relative to the signal of interest.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jinglescode/python-signal-processing/blob/master/tutorials/Denoising%20with%20mean-smooth%20filter.ipynb)

## Getting Started

### Installation

Currently, this has not been released. Use Git or checkout with SVN, and install the dependencies:

```
git clone https://github.com/jinglescode/python-signal-processing.git
pip install -r requirements.txt
```

### Dependencies

See [requirements.txt](https://github.com/jinglescode/python-signal-processing/tree/master/requirements.txt).

### Usage

Let's generate a 2D-signal, sampled at 100-Hz. Design and apply a 4th-order bandpass Butterworth filter with a cutoff frequency between 5-Hz and 20-Hz.

```python
from splearn.data.generate import signal
from splearn.filter.butter import butter_bandpass

signal_2d = signal(
    length_seconds=4, 
    sampling_rate=100, 
    frequencies=[[4,7,11,17,40, 50],[1, 3]],
    plot=True
)

signal_2d_filtered = butter_bandpass(
    signal=signal_2d, 
    lowcut=5, 
    highcut=20, 
    sample_rate=100,
    type='sos',
    order=4, 
    plot=True,
    plot_xlim=[3,20]
)
```

## Disclaimer on Datasets

We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a GitHub issue. Thanks for your contribution to the ML community!
