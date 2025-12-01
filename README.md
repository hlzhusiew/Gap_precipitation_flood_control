# Dual Autoencoder for Flood and Precipitation Analysis

This repository contains the implementation of a Dual Autoencoder model designed to analyze the correlation between urban flooding control features and precipitation data. The model utilizes contrastive learning to align the latent representations of both data modalities.

## Overview

The project implements a deep learning framework that:
1.  Extracts features from flood control data and precipitation records.
2.  Uses two separate Autoencoders to learn latent representations.
3.  Applies a contrastive loss function to align the embeddings of corresponding city-year pairs.
4.  Analyzes the reconstruction quality and correlation across different cities and years.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn, Plotly

To install dependencies:

```bash
pip install -r requirements.txt