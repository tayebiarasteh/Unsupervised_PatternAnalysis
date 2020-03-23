# Unsupervised Learning Techniques

### By Sacha Medaer and Soroosh Tayebi Arasteh

This project contains programming exercises of the **Pattern Analysis** course (SS19) offered by the [Pattern Recognition Lab (LME)](https://lme.tf.fau.de/) of the [Computer Science Department](https://www.informatik.uni-erlangen.de/) at University of Erlangen-Nuremberg (FAU).

## Prerequisites

To get the repository running, you will need several packages such as NumPy/SciPy, Matplotlib or scikit-learn.

You can obtain them easily by installing the conda environment file included in the repository. To do so, run the following command from the Conda Command Window:

```shell
$ conda env create -f environment.yml
$ activate PatternAnalysis
```

*__Note__:* This step might take a few minutes

## Contents

The main goal of the project is to illustrate different algorithms and to try them out in some real-world applications.

#### Overview of the project:

- **Density Estimation**: Implementation of the Parzen Window using different Kernels for probability distribution estimations.

- **HMM Signature Verification**: Application of Hidden Markov Models to perform human signature verification in order to distinguish original from fake signatures.

- **K-Means and Gap Statistics**: Implemenation of the K-Means clustering algorithm, overview and explanation of the model selection problem, and implementation of Gap Statistics to find out the optimal K for the algorithm.

- **Unsupervised as Supervised**: Here I illustrate how density estimation (inherently unsupervised) can be performed as a regression task (supervised learnning) using an auxiliar distribution and a Random Forest Regressor.

## Acknowledgment

Part of this `README` file is written based on the [Angel Villar's](https://github.com/angelvillar96) project.
