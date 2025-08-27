# MILO: Microsatellite Instability prediction using a Long-deletion signature

[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**MILO** is a command-line tool for predicting microsatellite instability (MSI) status from 83-channel indel mutation profiles. It supports a wide range of sample qualities, including shallow-sequencing data with coverage as low as 0.1× (or lower) and low-purity FF/FFPE samples with tumour content as low as 2–5% (without requiring a matched normal), as well as standard processed indel profiles.

## Table of Contents

* [Installation](#installation)
* [Input File Format](#input-file-format)
* [Test Data](#test-data)
* [Usage](#usage)
  * [Predicting MSI Status using Pre-trained Models (`predict`)](#1-predicting-msi-status-predict)
  * [Training a Custom Model (`train`)](#2-training-a-custom-model-train)
* [Output Files](#output-files)
* [License](#license)
* [Citation](#citation)

---

## Installation

Installing MILO in a dedicated Conda environment is strongly recommended to ensure reproducibility.

1.  **Create and activate a new Conda environment:**
    ```bash
    conda create -n milo_env python=3.9
    conda activate milo_env
    ```

2.  **Install MILO from PyPI:**
    This will automatically install all the correct versions of the required dependencies.
    ```bash
    pip install milo-mmrd
    ```

3.  **Verify the installation:**
    Check that the `milo` command and its sub-commands are available by displaying the help messages.
    ```bash
    milo --help
    milo predict --help
    milo train --help
    ```

---

## Input File Format

MILO `predict` mode requires a CSV or TSV input file (`--input <PATH-to-YOUR-FILE>` or `-i <PATH-to-YOUR-FILE>`) containing sample names and mutation counts across the 83 indel channels. The tool can automatically detect two formats:

1.  **Samples as rows:** The first column contains sample IDs, and the header contains the 83 indel channel names.
2.  **Samples as columns:** The header contains sample IDs, and the first column contains the 83 indel channel names.

The 83-channel names follow the [COSMIC(v3.4)](https://cancer.sanger.ac.uk/signatures/id/) indel spectra feature naming convention. All 83 channels must be present in the input file. Channel names may be separated by either underscores (_) or colons (:); for example, 1_Del_C_0 and 1:Del:C:0 are both accepted.

**Additional Requirement for train mode**

In addition to the format above, the input file for `milo train` must also contain a label column (e.g., MSI_status) with values 0 for MMRp and 1 for MMRd.

---

## Test Data

To help you get started, we provide example datasets for both prediction and training, intended only for demonstration purposes. To run MILO on your own data, simply replace the example files with your own.

- **Download Link:** [test_data.zip](https://github.com/QingliGuo/MILO/raw/main/test_data.zip)

After downloading and unzipping (`unzip test_data.zip`), you can run the example commands provided in the [Usage](#usage) section.

---

## Usage

MILO operates using two main sub-commands: `predict` and `train`. Here we use the example datasets to demonstrate the usage of MILO.

### 1. Predicting MSI Status (`predict`)

This is the primary function of MILO. It uses built-in or custom models to classify samples.

#### **Basic Prediction using Pre-trained models**

* **For FFPE low-pass samples (`ffpe_lp`):**
    ```bash
    milo predict --input ./test_data/MILO_test_data.csv --sample_type ffpe_lp
    ```

* **For Fresh-Frozen low-pass samples (`ff_lp`):**
    ```bash
    milo predict --input ./test_data/MILO_test_data.csv --sample_type ff_lp
    ```

* **For standard, high-quality samples (`standard`):**
    ```bash
    milo predict --input ./test_data/MILO_test_data.csv --sample_type standard
    ```
    
#### **Advanced Prediction Options**

You can combine several flags for more detailed analysis, for example:

* Noise Correction (`-c` or `--noise_correction`): Correct for technical noise in low-pass samples.
* MSI Intensity Score (`-int` or `--msi_intensity`): Calculate a quantitative score for the degree of instability.
* Generate Plots (`-p` or `--plot`): Create and save ID83 profile plots for all samples.

**Example of an advanced analysis run:**

```bash
milo predict \
    --input ./test_data/MILO_test_data.csv \
    --sample_type ff_lp \
    --output ./milo_analysis \
    --noise_correction \
    --msi_intensity \
    --cov_norm ./test_data/cov_purity_simulated_data.csv \
    --purity_norm \
    --plot
```
For a full list of options, run `milo predict --help`.

## 2. Training a Custom Model (`train`)

You can train a new **Random Forest model** using your own labeled dataset.  

### Example Training Command

```bash
milo train \
    --input ./test_data/training_dataset.csv \
    --output ./new_milo_model
```

This will create a `new_milo_model` directory containing `custom_milo_model.joblib`, which you can then use with the `predict` command.

For a full list of options, run `milo train --help`.

### Prediction of MSI using a custom model
  
You can use your trained your own model for prediction instead of the built-in ones.

```bash
milo predict \
    --input ./test_data/testing_dataset.csv \
    --custom_model ./new_milo_model/custom_milo_model.joblib
```
    
You can also combine the flags listed above for more advanced analysis when using your trained model. 

---

## Output Files

MILO creates an output directory (e.g., `./milo_results/` by default) containing the following files depending on the command and options used.

### `predict` Command Outputs

1.  **`MILO_predictions.csv`**
    This is the main results file. It contains the following columns for each sample:
    * `SampleID`: The identifier for the sample.
    * `Prob(MSI)`: The predicted probability of the sample being MSI-High, ranging from 0.0 to 1.0.
    * `MILO_prediction`: The final classification, which can be 'Yes', 'Maybe', or 'No' based on the probability thresholds. These thresholds can be adjusted using `--yes_threshold <FLOAT>` and/or `--maybe_threshold <FLOAT>`, with default values of 0.75 and 0.5, respectively.
    * MSI Intensity Score (Optional): This column is added if you use the `-int|--msi_intensity` flag. The column name changes based on the input data type:
        * `MSI intensity`: For standard, high-quality samples, calculated using original mutation counts.
        * `Relative MSI intensity`: For low-pass samples, calculated using normalised mutation propotions.
        * `MSI intensity (adjusted)`: For low-pass samples, calculated using coverage- and/or purity-adjusted mutation counts.
          - `--cov_norm <PATH>`: File with sample names and a coverage column (purity column optional).
          - `--purity_norm`: Enables purity-adjusted counts. If purity column is not provided, MILO will estimate it from noise correction results (`--noise_correction`).

2.  **`MILO_noise_corrected_profiles.csv` (Optional)**
    This file is generated only when using the `-c` or `--noise_correction` flag. It contains the noise-corrected 83-channel indel profiles for the samples classified as 'Yes'. When `--custom_noise_pattern` is specified, MILO automatically derives the noise pattern from the your input file and use it for noise correction.

4.  **`plots/` Directory (Optional)**
    This directory is created when using the `-p` or `--plot` flag. It contains several subdirectories with ID83 profile plots in PDF format:
    * `MMRd/`, `MMRp/`, `Maybe/`: These folders contain individual profile plots for each sample, categorized by their prediction status.
    * `Comparison/`: This folder contains mirrored comparison ID83 plots, such as the average MMRd vs. MMRp profile, or plots showing a sample's profile before and after noise correction.

### `train` Command Output

1.  **`custom_milo_model.joblib`**
    This file is the trained scikit-learn model object saved in a `.joblib` file. It can be used with the `--custom_model` argument in the `predict` command.

---

## License

This program is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License**.  
To view a copy of this license, visit [http://creativecommons.org/licenses/by-nc-sa/4.0/](http://creativecommons.org/licenses/by-nc-sa/4.0/).

---

## Citation

If you use MILO in your research, please cite:

Guo, Q. *et al.*  *Long deletion signatures in repetitive genomic regions track somatic evolution and enable sensitive detection of microsatellite instability.* *bioRxiv* 2024.10.03.616572 (2024) doi [10.1101/2024.10.03.616572](https://doi.org/10.1101/2024.10.03.616572)

[Link to Preprint](https://doi.org/10.1101/2024.10.03.616572)
