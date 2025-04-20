# 2024 US Elections Twitter Data Bot Detector

## Environment Setup
1. Install anaconda or miniconda.
2. Run `conda env create -f environment.yml`
3. Activate the conda environment:
   - Windows command: `activate bot-detector` 
   - MacOS / Linux command: `conda activate bot-detector`

## Dataset
The Twibot20 dataset was requested from main author of "TwiBot-20: A Comprehensive Twitter Bot Detection Benchmark". His information can be found on https://github.com/BunsenFeng/TwiBot-20.

The US 2024 Elections data can be downloaded on https://github.com/sinking8/usc-x-24-us-election.
## Generate Embeddings
Under "Data_Pre_Processing", run all cells in the notebook to generate either Glove or Roberta Embeddings for both tweet data and metadata of Twibot20 or Elections 2024 dataset.

## Baseline Implementation
Based on “Deep Neural Networks for Bot Detection” paper by Kudugunta et al. which uses a Contextual LSTM (200D GloVE) model. Our goal is to match the following performance metric scores reported on “TwiBot-20 A Comprehensive Twitter Bot Detection Benchmark” paper by Feng et al. for this model.

Run all the cells in either notebooks (with or without auxiliary output) under "Baseline_Implementation" to get the performance metrics for the Twibot20 dataset.

## Improved Implementation
Our team ran the following experiments to improve the baseline implementation:
1. BiLSTM without Auxiliary Output and 200D Glove Embeddings
2. BiLSTM with Auxiliary Output and 200D Glove Embeddings
3. BiLSTM without Auxiliary Output and 200D Roberta Embeddings
4. DenseNet and using 200D Roberta Embeddings
5. BiLSTM without Auxiliary Output, DenseNet, and 200D Glove Embeddings
6. Transformer, BiLSTM without Auxiliary Output, and 200D Glove Embeddings

To run an implementation above, select the respective notebook under "Improved_Implementation" and run all cells just before grid search. the performance of these implementations are listed in order below:


