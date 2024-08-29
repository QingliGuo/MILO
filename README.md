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
  
To run MILO as a command-line script:
    
    1) for FF tissue without noise correction:
    
    ```
        python MILO_setup.py [-I|--Input] ./MILO_test_data.csv [-T|--TissueType] FF
    ```
    
    2) for FF tissue with noise correction with plot:
    ```    
    python MILO_setup.py [-I|--Input] ./MILO_test_data.csv [-T|--TissueType] FF [-C|--NoiseCorrection] True [-P|--Plot] True
    ```    
    
    3) for FF tissue with specified noise profile:
    ```
    python MILO_setup.py [-I|--Input] ./MILO_test_data.csv [-T|--TissueType] FF [-C|--NoiseCorrection] True [-N|--Noise_file] './test_noise.csv'
    ```
To run MILO prediction on FFPE samples, using **[-T|--TissueType] FFPE**.
  
To print out help informaiton, 
```
python MILO_setup.py [-h|--help]'
```
We tested MILO on our FF sWGS samples. Click [here](https://github.com/QingliGuo/MILO/tree/main/test_MILO) for the test data and results.

+ Run the command line:
```
python FFPEsig.py [--input|-i] <Path-to-the-DataFrame> [--sample|-s] <Sample_id> [--label|-l] <Unrepaired|Repaired> [--output_dir|-o] <Path-of-output-folder>
```
2. Example

```
python FFPEsig.py --input ./Data/simulated_PCAWG_FFPE_unrepaired.csv --sample ColoRect-AdenoCA::SP21528 --label Unrepaired --output_dir FFPEsig_OUTPUT
```
Or 

```
python FFPEsig.py -i ./Data/simulated_PCAWG_FFPE_unrepaired.csv -s ColoRect-AdenoCA::SP21528 -l Unrepaired -o FFPEsig_OUTPUT
```


# Analysis code
The links below include analysis codes used in our manuscript entitled "Long deletions at repetitive genomic regions reveal evolutionary dynamics and enable sensitive detection of microsatellite instability".

+ [Discovery of long_deletion signatures from deep WGS data](https://github.com/QingliGuo/MILO/blob/main/Notebooks/Long_deletion_sigs_discovery_from_DeepSeqData.ipynb)
+ [Evidence of stepwise accumulation of slippage errors](https://github.com/QingliGuo/MILO/blob/main/Notebooks/Evidence_of_stepwise_slippage_events.ipynb)
+ [Identification of long deletion signals in shallow WGS data](https://github.com/QingliGuo/MILO/blob/main/Notebooks/Long_deletion_sig_features_in_shallow_WGS.ipynb)
+ [Application of MILO](https://github.com/QingliGuo/MILO/blob/main/Notebooks/Application_of_MILO.ipynb)

# Citation


