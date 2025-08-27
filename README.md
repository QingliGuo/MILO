MILO: Microsatellite Instability prediction using a Long-deletion signature
MILO is a command-line tool for predicting microsatellite instability (MSI) status from 83-channel indel mutation profiles. It supports a wide range of sample qualities, including shallow-sequencing data with coverage as low as 0.1× (or lower) and low-purity FF/FFPE samples with tumour content as low as 2–5% (without requiring a matched normal), as well as standard processed indel profiles.

Table of Contents
Installation

Input File Format

Usage

Predicting MSI Status (predict)

Training a Custom Model (train)

Test Data

License

Citation

Installation
Installing MILO in a dedicated Conda environment is strongly recommended to ensure reproducibility.

Create and activate a new Conda environment:

conda create -n milo_env python=3.9
conda activate milo_env

Install MILO from PyPI:
This will automatically install all the correct versions of the required dependencies.

pip install milo-mmrd

Verify the installation:
Check that the milo command and its sub-commands are available by displaying the help messages.

milo --help
milo predict --help
milo train --help

Input File Format
MILO expects a CSV or TSV file containing sample names and mutation counts across the 83 indel channels. The tool can automatically detect two formats:

Samples as rows: The first column contains sample IDs, and the header contains the 83 indel channel names.

Samples as columns: The header contains sample IDs, and the first column contains the 83 indel channel names.

The 83 channel names can be separated by either underscores (_) or colons (:). For example, 1_Del_C_0 and 1:Del:C:0 are both accepted.

Usage
MILO operates using two main sub-commands: predict and train.

1. Predicting MSI Status (predict)
This is the primary function of MILO. It uses built-in or custom models to classify samples.

Basic Prediction
For FFPE low-pass samples (ffpe_lp):

milo predict --input <path/to/your_data.csv> --sample_type ffpe_lp

For Fresh-Frozen low-pass samples (ff_lp):

milo predict --input <path/to/your_data.csv> --sample_type ff_lp

For standard, high-quality samples (standard):

milo predict --input <path/to/your_data.csv> --sample_type standard

Advanced Prediction Options
You can combine several flags for more detailed analysis:

Noise Correction (-c or --noise_correction): Correct for technical noise in low-pass samples.

MSI Intensity Score (-int or --msi_intensity): Calculate a quantitative score for the degree of instability.

Generate Plots (-p or --plot): Create and save ID83 profile plots for all samples.

Example of an advanced analysis run:

milo predict \
    --input ./test_data/built_in_model/MILO_test_data.csv \
    --sample_type ff_lp \
    --output ./milo_analysis \
    --noise_correction \
    --msi_intensity \
    --cov_norm ./test_data/built_in_model/cov_purity_simulated_data.csv \
    --purity_norm \
    --plot

Using a Custom Model
If you have trained your own model, you can use it for prediction instead of the built-in ones.

milo predict \
    --input ./test_data/custom_model/testing_dataset.csv \
    --custom_model ./new_milo_model/custom_milo_model.joblib \
    --output ./milo_custom_results

For a full list of options, run milo predict --help.

2. Training a Custom Model (train)
You can train a new Random Forest model using your own labeled dataset. Your training file must contain the 83 indel channels and a label column (e.g., MSI_status with values 0 for MMRp and 1 for MMRd).

Example training command:

milo train \
    --input ./test_data/custom_model/training_dataset.csv \
    --output ./new_milo_model

This will create a new_milo_model directory containing custom_milo_model.joblib, which you can then use with the predict command.

For a full list of options, run milo train --help.

Test Data
To help you get started, we provide example datasets for both prediction and training.

Download Link: test_data.zip

After downloading (wget https://github.com/QingliGuo/MILO/raw/main/test_data.zip) and unzipping (unzip test_data.zip), you can run the example commands provided in the Usage section.

License
This program is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/.

Citation
If you use MILO in your research, please cite:

Guo, Q. et al. Long deletion signatures in repetitive genomic regions track somatic evolution and enable sensitive detection of microsatellite instability. bioRxiv 2024.10.03.616572 (2024) doi:10.1101/2024.10.03.616572.

Link to Preprint