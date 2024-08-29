# MILO
MILO can be used for predicting **M**icrosatellite **I**nstability in **LO**w-quality samples. MILO takes indel profiles (83-channel mutation spectrum, see [here](https://cancer.sanger.ac.uk/signatures/id/) for more information) of given samples and predicts their MSI status. 

Low-quality samples could be `shallow WGS` (e.g., ~0.1X) and/or `low-purity` FF/FFPE samples (>2% for FF and >5% for FFPE). No matched normal is required.

## 1.1 Installation
MILO script can be downloaded [here](https://github.com/QingliGuo/MILO/blob/main/MILO_setup.py). You also need to download the two trained classifiers ([FFPE_rf](https://github.com/QingliGuo/MILO/blob/main/FFPE_rf.joblib) and [FF_rf](https://github.com/QingliGuo/MILO/blob/main/FF_rf.joblib)) to the same folder.

## 1.2 Dependencies
1) Python - MILO is tested on python 3.11.0.
2) Python packages used by MILO
+ joblib (1.2.0)
+ numpy (1.23.5)
+ pandas (2.0.0)
+ seaborn (0.13.2)
+ sys, getopt, datetime, sklearn
  
## 1.3 Usage
MILO predicts microsatellite instability in low-quality samples. Low-qulity samples could be 'shallow-sequencing' and/or 'low-purity' FF/FFPE samples. No matched normal is required also.

### 1.3.1 Example commands        
1) On FF samples without noise correction:
```
python MILO_setup.py [-I|--Input] ./MILO_test_data.csv [-T|--TissueType] FF
```
2) On FF samples with noise correction with plot using default noise profile:

```
python MILO_setup.py [-I|--Input] ./MILO_test_data.csv [-T|--TissueType] FF [-C|--NoiseCorrection] True [-P|--Plot] True
```
3) On FF samples with noise correction with user-specified noise profile: 

```
python MILO_setup.py [-I|--Input] ./MILO_test_data.csv [-T|--TissueType] FF [-C|--NoiseCorrection] True [-N|--Noise_file] ./test_noise.csv
```
4) On FFPE samples in above examples, using `[-T|--TissueType] FFPE`.

5) To print out help informaiton:

```
python MILO_setup.py [-h|--help]
```

### 1.3.2 Our tests
We tested MILO on our FF sWGS samples. Click [here](https://github.com/QingliGuo/MILO/tree/main/test_MILO) for the test data and results. Please format your input data as descirbed in the following section.

### 1.3.3 Arguments

1)  Required
           
+ **[-I|--Input] <path_of_input_file>** : the location of your CSV file containing the 83-channel mutational counts for your sample(s).

  + CSV Format: The file should be in CSV format with comma separators (,) between values.
  + Column Structure:
      + The file should contain 84 columns.
      + The first column must be the sample ID.
      + The following 83 columns should represent the 83-channel mutation profile.
  + Row Structure: Each row in the file represents a unique sample.
                
+ **[-T|TissueType] <FFPE|FF>**: MILO trained separate classifiers for FFPE and fresh frozen (FF) samples. Therefore, it is important that you specify your tissue type.
  
2)  Optional
  + [-C|--NoiseCorrection] <False|True>: determine if the noise correction will be carried out by MILO
  + [-N|--Noise_file] <path_to_noise_file>: users can specify a noise profile using this argument. If none is provided, MILO defaults to using cohort-specific noise profiles from high-confidence MMRp samples (Prob(MSI) < 0.1). If no such profiles are available, MILO uses the FF/FFPE noise pattern from our sWGS cohorts as described in our paper.
  + [-P|--Plot] <False|True>: determine if the PDFs of indel profiles will be generated
  + [-PC|--Prob_cutoff] <float>: specify the threshold for Prob(MSI) to determine high-confidence MSI samples. The default value is 0.75, meaning only samples with Prob(MSI) > 0.75 are considered high-confidence MSI predictions.

### 1.3.4 Output files

Here are the output files from MILO's prediction:
  + CSV file with indel profiles and the MILO predictions
  + (**Optional**) CSV file of MILO predicted long-deletion intensity in MSI positive files
  + (**Optional**) PDFs of indel profiles in predicted MMRd samples (and/or noise-corrected MMRd profiles) can be found in the folder `./plots/`.

# Citation


