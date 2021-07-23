# Python Signal Processing [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

This repository contains tutorials on understanding and applying signal processing using NumPy and PyTorch.

**splearn** is a package for signal processing and machine learning with Python. It is built on top of [NumPy](https://numpy.org) and [SciPy](https://www.scipy.org), to provide easy to use functions from common signal processing tasks to machine learning. 

## Contents

- [Tutorials](#tutorials)
- [Getting Started](#getting-started)
- [Disclaimer on Datasets](#disclaimer-on-datasets)

--- 

## Tutorials

Signal processing can be daunting; we aim to bridge the gap for anyone who are new signal processings to get started, check out the [tutorials](https://github.com/jinglescode/python-signal-processing/tree/main/tutorials) to get started on signal processings.

### 1. Signal composition (time, sampling rate and frequency)

In order to begin the signal processing adventure, we need to understand what we are dealing with. In the first tutorial, we will uncover what is a signal, and what it is made up of. We will look at how the sampling rate and frequency can affect a signal. We will also see what happens when we combine multiple signals of different frequencies.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jinglescode/python-signal-processing/blob/main/tutorials/Signal%20composition%20-%20time%2C%20sampling%20rate%20and%20frequency.ipynb)

### 2. Fourier Transform

Now we know what are signals made of and we learned that combining multiple signals of various frequencies will jumbled up all the frequencies. In this tutorial, we will learn about Fourier Transform and how it can take a complex signal and decompose it to the frequencies that made it up.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jinglescode/python-signal-processing/blob/main/tutorials/Fourier%20Transform.ipynb)

### 3. Denoising with mean-smooth filter

We know that signals can be noisy, and this tutorial will focus on removing these noise. We learn to apply the simplest filter to perform denoising, the running mean filter. We will also understand what are edge effects.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jinglescode/python-signal-processing/blob/main/tutorials/Denoising%20with%20mean-smooth%20filter.ipynb)

### 4. Denoising with Gaussian-smooth filter

Next, we will look at a slight adaptation of the mean-smooth filter, the Gaussian smoothing filter. This tends to smooth the data to be a bit smoother compared to mean-smooth filter. This does not mean that one is better than the other, it depends on the specific applications. So, it is important to be aware of different filters type and how to use them.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jinglescode/python-signal-processing/blob/main/tutorials/Denoising%20with%20Gaussian-smooth%20filter.ipynb)

### 5. Canonical correlation analysis

Canonical correlation analysis (CCA) is applied to analyze the frequency components of a signal. In this tutorials, we use CCA for feature extraction and classification.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jinglescode/python-signal-processing/blob/main/tutorials/Canonical%20Correlation%20Analysis.ipynb)

---

## Getting Started

### Installation

Currently, this has not been released. Use `git clone`, and install the dependencies:

```
git clone https://github.com/jinglescode/python-signal-processing.git
pip install -r requirements.txt
```

### Dependencies

See [requirements.txt](https://github.com/jinglescode/python-signal-processing/tree/main/requirements.txt).

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

---

## Disclaimer on Datasets

We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a GitHub issue. Thanks for your contribution to the ML community!
