# MILO
MILO can be used for predicting **M**icrosatellite **I**nstability in **LO**w-quality samples. MILO takes indel profiles (83-channel mutation spectrum, see [here](https://cancer.sanger.ac.uk/signatures/id/) for more information) of given samples and outputs the binary status of the input samples. 

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
  + [-C|--NoiseCorrection] <False|True>:
  + [-N|--Noise_file] <path-to-noise-file>: Default pattern is the averaged profile we observed in our FF/FFPE sWGS non-MSI samples. The users can also provide their own noise profile.
  + p: used to determine the high confidence MSI. Default value is 0.75.
  + [-P|--Plot] <False|True>:
  + [-PC|--Prob_cutoff] <float>:

### 1.3.4 Output files

Here are the output files from MILO's prediction:
  + CSV file with indel profiles and the MILO predictions
  + (**Optional**) CSV file of MILO predicted long-deletion intensity in MSI positive files
  + (**Optional**) PDFs of indel profiles in predicted MMRd samples (and/or noise-corrected MMRd profiles) can be found in the folder `./plots/`.

# Citation


