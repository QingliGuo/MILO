#!/user/bin/python

## loading packages

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.random as npr
import sys, getopt
import os

from scipy.stats import *
from datetime import datetime, date
from sklearn.metrics import f1_score,confusion_matrix,accuracy_score, roc_auc_score

## loading trained models
rf_FFPE = joblib.load("./FFPE_rf.joblib")
rf_FF = joblib.load("./FF_rf.joblib")
deep_FF = joblib.load("./Deep_rf.joblib")

## Names of 83-channel indel signature
indel_channel_names = np.array(['1_Del_C_0', '1_Del_C_1', '1_Del_C_2', '1_Del_C_3', '1_Del_C_4',
                                '1_Del_C_5', '1_Del_T_0', '1_Del_T_1', '1_Del_T_2', '1_Del_T_3',
                                '1_Del_T_4', '1_Del_T_5', '1_Ins_C_0', '1_Ins_C_1', '1_Ins_C_2',
                                '1_Ins_C_3', '1_Ins_C_4', '1_Ins_C_5', '1_Ins_T_0', '1_Ins_T_1',
                                '1_Ins_T_2', '1_Ins_T_3', '1_Ins_T_4', '1_Ins_T_5', '2_Del_R_0',
                                '2_Del_R_1', '2_Del_R_2', '2_Del_R_3', '2_Del_R_4', '2_Del_R_5',
                                '3_Del_R_0', '3_Del_R_1', '3_Del_R_2', '3_Del_R_3', '3_Del_R_4',
                                '3_Del_R_5', '4_Del_R_0', '4_Del_R_1', '4_Del_R_2', '4_Del_R_3',
                                '4_Del_R_4', '4_Del_R_5', '5_Del_R_0', '5_Del_R_1', '5_Del_R_2',
                                '5_Del_R_3', '5_Del_R_4', '5_Del_R_5', '2_Ins_R_0', '2_Ins_R_1',
                                '2_Ins_R_2', '2_Ins_R_3', '2_Ins_R_4', '2_Ins_R_5', '3_Ins_R_0',
                                '3_Ins_R_1', '3_Ins_R_2', '3_Ins_R_3', '3_Ins_R_4', '3_Ins_R_5',
                                '4_Ins_R_0', '4_Ins_R_1', '4_Ins_R_2', '4_Ins_R_3', '4_Ins_R_4',
                                '4_Ins_R_5', '5_Ins_R_0', '5_Ins_R_1', '5_Ins_R_2', '5_Ins_R_3',
                                '5_Ins_R_4', '5_Ins_R_5', '2_Del_M_1', '3_Del_M_1', '3_Del_M_2',
                                '4_Del_M_1', '4_Del_M_2', '4_Del_M_3', '5_Del_M_1', '5_Del_M_2',
                                '5_Del_M_3', '5_Del_M_4', '5_Del_M_5'])

## Important indel signature features 
selected_features_FFPE = np.array (['2_Del_R_5', '3_Del_R_4', '3_Del_R_3', '5_Del_R_2', '4_Del_R_3',
       '2_Del_R_4', '4_Del_R_4', '5_Del_R_3', '1_Del_T_5', '4_Del_R_2'])

selected_features_FF = np.array (['4_Del_R_3', '3_Del_R_3', '5_Del_R_1', '5_Del_R_2', '3_Del_R_4',
       '2_Del_R_4', '4_Del_R_2', '1_Del_T_5', '5_Del_R_3', '2_Del_R_5'])

## Default noise pattern in FFPE sWGS samples (can be specified by the user)
noise_profile_ffpe = np.array([9.58484005e-02, 6.60390897e-02, 4.02413928e-02, 1.54575908e-02,
       7.55695567e-03, 3.16865183e-03, 7.05715556e-02, 5.29072416e-02,
       3.95923482e-02, 2.54772306e-02, 2.21216763e-02, 7.98942963e-02,
       5.22925856e-02, 1.18289031e-02, 6.55727933e-03, 2.11246377e-03,
       1.00279713e-03, 5.71238948e-04, 4.65637261e-02, 1.55468667e-02,
       7.10491762e-03, 6.96120666e-03, 7.82446590e-03, 4.76016116e-02,
       1.93955268e-02, 1.39312480e-02, 5.31706854e-03, 2.54735829e-03,
       3.29920071e-03, 2.42445056e-02, 5.79888134e-03, 3.32387804e-03,
       1.78766636e-03, 1.73240817e-03, 2.43110554e-03, 4.59258983e-03,
       2.18702507e-03, 2.61966568e-03, 2.14442340e-03, 2.26646779e-03,
       1.79829767e-03, 1.45206138e-03, 3.38281293e-03, 6.40780058e-03,
       3.26192841e-03, 1.77110157e-03, 5.48095911e-04, 2.17347820e-04,
       4.46892606e-02, 4.70127247e-03, 4.20654499e-03, 9.70381019e-04,
       1.08211208e-03, 8.47560736e-03, 1.80629629e-02, 1.85551068e-03,
       4.31187006e-04, 4.28221863e-04, 5.09686329e-04, 1.20810711e-03,
       1.04052167e-02, 8.86827782e-04, 3.11138184e-04, 3.49493070e-04,
       3.20469126e-04, 3.46872557e-04, 1.02986758e-02, 7.28470198e-04,
       4.42014244e-04, 2.56343293e-04, 1.11391818e-04, 5.56486698e-05,
       2.03733170e-02, 2.96848474e-03, 3.60742915e-03, 1.83824699e-03,
       1.01037217e-03, 1.65524334e-03, 2.39630446e-03, 1.36280483e-03,
       1.75303753e-03, 1.96515770e-03, 8.63322853e-03])

## Default noise pattern in FF sWGS samples (can be specified by the user)
noise_profile_ff = np.array([0.03188762, 0.10980923, 0.05596226, 0.0211853 , 0.01066379,
       0.00577608, 0.04073273, 0.09338299, 0.06889467, 0.04375247,
       0.03043592, 0.14727846, 0.00912034, 0.00207888, 0.00110229,
       0.00153728, 0.0014503 , 0.0032171 , 0.00751197, 0.00337621,
       0.00317826, 0.00455414, 0.00607843, 0.06148472, 0.01245731,
       0.01846886, 0.0051799 , 0.00298518, 0.00602975, 0.03921569,
       0.00449844, 0.00286332, 0.00157975, 0.00266543, 0.0032392 ,
       0.00579507, 0.00190535, 0.00204891, 0.00223124, 0.00285174,
       0.00244416, 0.00266482, 0.00297194, 0.00367331, 0.00357444,
       0.00236475, 0.00096298, 0.00043536, 0.01317631, 0.00159609,
       0.00109461, 0.00116069, 0.00273349, 0.02011619, 0.00562705,
       0.000706  , 0.00039768, 0.00089326, 0.00127011, 0.00264405,
       0.00558014, 0.00072013, 0.00060944, 0.00089572, 0.00085451,
       0.00104489, 0.00587526, 0.00116745, 0.00099917, 0.00069001,
       0.00031586, 0.00019611, 0.01686813, 0.0018647 , 0.00302536,
       0.00153427, 0.00093038, 0.0016307 , 0.00199248, 0.00090543,
       0.00094627, 0.00066515, 0.00171457])

def sig_extraction (V, W1, rank = 2, iteration = 3000, precision=0.95):
    
    """
    This function corrects noise (W1) in a given sample (V).   
        Author & Copyright: Qingli Guo <<qingliguo@outlook.com>
    Required:
        V: mutational counts in from a sample
        W1: noise proile
    Optional arguments(default values see above):
        rank: the number of signatures
        iteration: maximum iteration times for searching a solution
        precision: convergence ratio. The convergence ratio is computed as the average KL divergence from the last batch of 20 iterations divided by the second last batch of 20 iterations.
        
    Return:
        1) W: noise and signal signatures
        2) H:  weights/acticitites/attributions for noise and signal signatures
        3) the cost function changes for each iteration
    """
    
    n = V.shape[0]  ## num of features
    m = V.shape[1] ## num of sample
        
    ## initialize W2:
    W2 = npr.random (n)
    
    ## combine W1 and W2 to W;
    W = np.array ([W1,W2])
    W = W.T
    
    ## nomarlize W
    W = W / sum(W[:,])
    
    ## initialize H:
    H = npr.random ((rank,m))
    
    ## cost function records:
    Loss_KL = np.zeros (iteration)
    
    for ite in range (iteration):
        
        ## update H
        for a in range (rank):  
            denominator_H = sum(W[:,a])        
            np.seterr(divide='ignore')

            for u in range (m) :
                numerator_H = sum (W[:,a] * V[:,u] / (W @ H) [:,u])
                np.seterr(divide='ignore')
                H[a,u] *=  numerator_H / denominator_H

        ## only update W2
        a = 1
        denominator_W = sum(H[a,:])
        
        for i in range (n):            
            numerator_W = sum (H[a,:] * V[i,:] / (W @ H)[i,:])
            np.seterr(divide='ignore')
            W[i,a] *= numerator_W / denominator_W
    
        ## normlize W after upadating:
        W = W / sum(W[:,])
        
        ## record the costs
        if ite == 0 :
            Loss_KL [ite] = entropy(V, W @ H).sum()
            normlizer = 1/Loss_KL[0]
        
        Loss_KL [ite] = entropy(V, W @ H).sum() * normlizer
        
        if ite > 200:
            last_batch = np.mean(Loss_KL [ite-20:ite])
            previous_batch = np.mean(Loss_KL [ite-40:ite-20])
            change_rate = last_batch/previous_batch
            if change_rate >= precision and np.log(change_rate) <= 0:
                break
        
    return (W, H, Loss_KL[0:ite])

def correct_FFPE_profile(V, W1, sample_id="", precision = 0.95, ite = 100):
    """
    This function correct noise in a given sample.   
        Author & Copyright: Qingli Guo <qingliguo@outlook.com>
    Required:
        V: mutational counts in from a sample
        W1: noise proile
    Optional arguments(default values see above):
        sample_id: identifier used in dataframe for multiple solutions
        precision: convergence ratio. The convergence ratio is computed as the average KL divergence from the last batch of 20 iterations divided by the second last batch of 20 iterations.
        ite: how many solutions should be searched for
    Return:
        1) W: noise and signal signatures
        2) H:  weights/acticitites/attributions for noise and signal signatures
        3) the cost function changes for each iteration
    """
    
    df_tmp = pd.DataFrame()
    for i in range(ite):
        seed_i = i + 1
        npr.seed(seed_i)
        col_name = sample_id + "::rand" + str(seed_i)
        ## algorithm works on channels with mutation count > 0
        V_nonzeros = V[V > 0]
        w, h, loss = sig_extraction(V = V_nonzeros.reshape(len(V_nonzeros),1),
                                    W1 = W1[V > 0],
                                    precision = precision)
        predicted_V = np.zeros (len(V))
        predicted_V[V > 0] = w[:,1] * h[1]
        df_tmp[col_name] = predicted_V

    corrected_profile = df_tmp.mean(axis = 1).astype("int").to_numpy()
    
    return ([corrected_profile, df_tmp])

def ID83_plot(sig, name = "", norm = False, xticks_label = False, width = 11, height = 2.7, 
              bar_width = 0.8, grid = 0.2, label = "", file = ""):
    
    """
    This function plots 83-channel indel mutation profile.   
        Author & Copyright: Qingli Guo <qingliguo@outlook.com>
    Required:
        sig: 83-channel mutation counts/probabilities
        
    Optional arguments(default values see above):
        label: to identify the plot, e.g. sample ID
        name: to add extra information on top of the plot
        file: file path and name where to save the plot if given
        norm: to normlise provided 83-channel vector or not
        width: width of the plot
        height: height of the plot
        bar_width: bar_width of the plot
        xticks_label: to show the xticks channel information or not
        grid: grid of the plot    
    """
        
    channel = 83; col_list = []
    
    ## Setting up colors:
    color_names1 = ['sandybrown', 'darkorange','yellowgreen','g','peachpuff','coral','orangered',
                    'darkred', 'powderblue', 'skyblue','cornflowerblue','navy']
    for col in color_names1:
        col_list += [col] * 6

    col_list = col_list + ['thistle'] + ['mediumpurple'] * 2 + ['rebeccapurple'] * 3 + ['indigo'] *5
    
    col_set = color_names1 + ['thistle','mediumpurple','rebeccapurple','indigo']

    ## The top layer annotations:
    top_label = ['1bp Deletion', '1bp Insertion', '> 1bp Deletion at Repeats \n (Deletion Length)', 
             '>1bp Insertions at Repeats \n (Insertion Length)', 'Mircohomology \n (Deletion Length)']
    second_top_layer = ['C', 'T', 'C','T','2','3','4','5+','2','3','4','5+','2','3','4','5+']
    second_top_layer_color = ["black"] * 5 + ["white"] * 3 + ["black"] * 3 + ["white", "black"] + ["white"] * 3
    ## The bottom layer annotations:
    xlabel = ['1','2','3','4','5','6+','1','2','3','4','5','6+','0','1','2','3','4','5+','0','1','2','3','4','5+',
         '1','2','3','4','5','6+','1','2','3','4','5','6+','1','2','3','4','5','6+','1','2','3','4','5','6+',
         '0','1','2','3','4','5+','0','1','2','3','4','5+','0','1','2','3','4','5+','0','1','2','3','4','5+',
         '1','1','2','1','2','3','1','2','3','4','5+']
    bottom_layer = ['Homopolymer Length', 'Homopolymer Length', 'Number of Repeat Units',  'Number of Repeat Units','Microhomology Length']

    ## figuer configuration setting
    sns.set(rc={"figure.figsize":(width, height)})
    sns.set(style="whitegrid", color_codes=True, rc={"grid.linewidth": grid, 'grid.color': '.7', 'ytick.major.size': 2,
                                                'axes.edgecolor': '.3', 'axes.linewidth': 1.35,})
    ## Plot the normalized version 
    if norm:
        normed_sig = sig / np.sum(sig)
        plt.bar(range(channel),normed_sig , width = bar_width, color =col_list)
        plt.ylim (np.min(normed_sig)*1.05, np.max(normed_sig) * 1.2) ##
                  
        plt.ylabel("Proportions of\nINDELs")
        
        plt.annotate (name, (55, np.max(normed_sig) *0.9))

    
    ## plot the original version:
    else:
        plt.bar(range(channel), sig , width = bar_width, color =col_list)
        plt.xticks(rotation=90, size = 7, weight='bold')
        plt.ylim (np.min(sig)*1.05, np.max(sig)*1.2) #3
        plt.annotate (name, (62, np.max(sig) *0.92))
        plt.annotate ('Total =' + format(np.sum(np.abs(sig)), ','), (0, np.max(sig)*1.01))
        plt.ylabel("Counts")
    
    plt.xticks(range(channel), xlabel, ha = "center", va= "center",  size = 8) 
    plt.yticks( va= "center", size = 9)
        
    ## the 16 types of color rectangle and the annotation:
    length = [6] * 12 + [1,2,3,5]
    for i in range(16):
        ## The upper pannel:
        left, width = sum (length[:i])/84 + 0.005, 1/84 * length[i] -0.001       
        bottom, height = 1.003, 0.12
        right = left + width
        top = bottom + height
        ax = plt.gca()
        p = plt.Rectangle((left, bottom), width, height+0.02, fill=True, color = col_set[i])
        p.set_transform(ax.transAxes)
        p.set_clip_on(False)
        ax.add_patch(p)
        ax.text(0.5 * (left + right), 0.5 * (bottom + top), second_top_layer[i], 
                color = second_top_layer_color [i], weight='bold',
                horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)

        ## The bottom pannel:
        left, width = sum (length[:i])/84 + 0.005, 1/84 * length[i] - 0.001  
        bottom, height = -0.02, -0.03
        right = left + width
        top = bottom + height
        ax = plt.gca()
        p = plt.Rectangle((left, bottom), width, height, fill=True, color = col_set[i])
        p.set_transform(ax.transAxes)
        p.set_clip_on(False)
        ax.add_patch(p)
        
    ## The most top and bottom annotation
    length2 = [1,1,2,2,1]
    for i in range (5):
        ## The most top
        left, width = sum (length2[:i])/7 , 1/7 * length2[i]     
        bottom, height = 1.15, 0.3
        right = left + width
        top = bottom + height
        ax = plt.gca()
        p = plt.Rectangle((left, bottom), width, height, fill = True, color = 'w')
        p.set_transform(ax.transAxes)
        p.set_clip_on(False)
        ax.add_patch(p)
        ax.text(0.5 * (left + right), 0.5 * (bottom + top), top_label[i], color = 'black', size = 10,
                horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
        
        ## The most bottom
        if xticks_label:
            left, width = sum (length2[:i])/7 , 1/7 * length2[i] 
        
            bottom, height = -0.4, 0.3
            right = left + width
            top = bottom + height
            ax = plt.gca()
            p = plt.Rectangle((left, bottom), width, height, fill=True, color = 'w', alpha = 0)
            p.set_transform(ax.transAxes)
            p.set_clip_on(False)
            ax.add_patch(p)
            ax.text(0.5 * (left + right), 0.5 * (bottom + top), bottom_layer[i], color = 'black', size = 9, 
                horizontalalignment='center',verticalalignment='baseline', transform=ax.transAxes)
    
    ## plot the name annotation if there is any
    if label != "":
        left, width = 1.003, 0.04
        bottom, height = 0, 1
        right = left + width
        top = bottom + height
        ax = plt.gca()
        p = plt.Rectangle((left, bottom), width, height, fill=True, color = "silver",alpha = 0.2)
        p.set_transform(ax.transAxes)
        p.set_clip_on(False)
        ax.add_patch(p)
        ax.text(0.5 * (left + right), 0.5 * (bottom + top), name, color = 'black',size = 10,
            horizontalalignment='center',verticalalignment='center',transform=ax.transAxes , rotation = 90)        
    
    ax.margins(x = 0.007)
    
    plt.tight_layout()
    if file:
        plt.savefig(file, bbox_inches = "tight", dpi = 300) 
        plt.close(); f,ax = plt.subplots()
    else:
        plt.show()

def compute_msi_instability (sample_array, r=0.001, data_source = 'WGS', long_del_only = False):
    """
    Computes the MSI intensity using absolute mutation counts and log transformation.
    
    Parameters:
    1) sample_array (numpy array): Array containing mutation counts for different deletion lengths.
                               Example: np.array([555, 6, 0, 0, 0])
                               It counts deletion numbers at different length: 1, 2, 3, 4 and 5+.
                               
    2) r (float): Optional. Mutation rate per MS locus per cell division (default = 0.001).
    
    3) data_source (string): Optional. Use default value of 'WGS' if the mutation count is observed from deep-seq WGS data with matched normal applied.
                            for deep-seq WES data, using 'WES'.
    
    Returns:
    
    float: MSI intensity value.
    """
    if sum(sample_array) > 1:
        sample_array[sample_array < 10] = 0 ## remove untrust-worthy mutation count.
    
    weights = np.log10(r**-np.arange(len(sample_array)))
    
    if not long_del_only:
        weights[0] = 1 ## adjust weight for 1bp deletions
    
    if data_source == 'WGS':
        instability = np.sum(sample_array * weights)
        
    elif data_source == 'WES':
        instability = np.sum(sample_array * weights) * 10**2
        
    return instability

def aggerating_signals(df, add_ins = True):
    
    sample_size = df.shape[0]
    
    items = ['1_ID_T_5', '2_Del_R', '3_Del_R', '4_Del_R', '5_Del_R']
    
    df_tmp = pd.DataFrame ({i: np.zeros(sample_size) for i in items})
    df_tmp.index = df.index.values
    
    if add_ins:
        df_tmp[items[0]] = df['1_Del_T_5'].values + df['1_Ins_T_5'].values
    else:
        df_tmp[items[0]] = df['1_Del_T_5'].values

    for i, ite in enumerate(items[1:]):
        
        df_tmp[items[i+1]]= df.loc[:, [ite in c for c in df.columns]].sum(axis = 1)
    
    return df_tmp

def msi_intensity (df, add_ins = True):
    
    df_copy = df.copy()

    expected_col_nums = len (indel_channel_names)
    
    if df_copy.shape[1] == expected_col_nums:
        df_copy.columns = indel_channel_names
        df_tmp = aggerating_signals (df_copy, add_ins = add_ins)
        
        msi_intensity_list = df_tmp.apply(compute_msi_instability, axis=1)
        
        return msi_intensity_list
    
    else:
        raise ValueError(f"Error: Expected {expected_cols} columns, but found {df_copy.shape[1]}.")
        

def MILO (Input, TissueType, 
          NoiseCorrection = False, 
          NoisePattern_ff = noise_profile_ff, 
          NoisePattern_ffpe = noise_profile_ffpe, 
          Prob_cutoff = 0.75):
    """
    This function predicts MSI staus in given samples.   
        Author & Copyright: Qingli Guo <qingliguo@outlook.com>
    
    Required:
        Input: file path of 83-channel mutational counts of your sample(s) 
                (CSV format; shape: Nx84: N is the sample size and 84 is the column number;
                The first column is your sample name and the rest 83 column is indel mutation profile)
        TissueType: 'FFPE' or 'FF'
        
    Optional arguments:
        NoiseCorrection: True or False
        NoisePattern_ff/NoisePattern_ffpe: Default pattern is the averaged profile we observed in our FF/FFPE sWGS non-MSI samples. The users can also provide their own noise profile.
        p: used to determine the high confidence MSI. Default value is 0.75.
        
    Output:
        1) CSV file with indel profiles and the MILO predictions
        2) (optional) CSV file of MILO predicted long-deletion intensity in MSI positive files
        3) Indel profiles of MMRd samples (and the noise-corrected profiles if correction step is required) under the folder of './plots/'.
    """
    
    df = pd.read_csv (Input, index_col = 0)

    if df.shape [1] != 83:
        return "Indel profile should contain 83-channel vectors. The first column should be your sample IDs"
    
    if TissueType != 'FF' and TissueType != 'FFPE':
        return "TissueType must be either 'FF' or 'FFPE'"
    
    if TissueType == 'FFPE':
        model = rf_FFPE
        noise_pattern = NoisePattern_ffpe
        selected_features = selected_features_FFPE.copy()
        
    elif TissueType == 'FF':
        model = rf_FF
        noise_pattern = NoisePattern_ff
        selected_features = selected_features_FF.copy()
        
    ## Make sure the data type is correct
    if type(noise_pattern) == type(NoisePattern_ffpe):
        noise_pattern =  noise_pattern.astype("float64").copy()
    else:
        noise_pattern =  noise_pattern.astype("float64").to_numpy().copy()
        
    ## rename column of df
    old_channel_names = df.columns.values
    df = df.rename({i:indel_channel_names[i] 
                            for i in range (83)}, 
                            axis = 1).copy()
    ## normalise
    df_norm = (df.T / df.T.sum()).T.copy()
    
    ## prepare input data
    x_test = df_norm.loc[:, selected_features].to_numpy()
    
    ## Predict the prob(MSI) using pre-trained model
    df['Prob(MSI)'] = model.predict_proba(x_test)[:,1]
    
    ## Assign categorical status based ib prob(MSI)
    df['MILO_prediction'] = 'Maybe'
    df.loc[df['Prob(MSI)'] <= 0.5 , 'MILO_prediction'] = 'No'
    df.loc[df['Prob(MSI)'] > Prob_cutoff, 'MILO_prediction'] = 'Yes'
    
    ## remove noise in predicted MSI samples.
    if NoiseCorrection:
        if sum(df['Prob(MSI)'] > Prob_cutoff) == 0:
            print ('MILO did not find MSI positive samples in your cohort using 0.75 as cutoff for prob(MSI)')
            return (df)
        
        else:
            pred_MSI_samples = list(df.index[df['Prob(MSI)'] > Prob_cutoff])
            
            noise_corrected_MSI = pd.DataFrame()
            noise_corrected_MSI.index = pred_MSI_samples
            for id_channel in indel_channel_names:
                noise_corrected_MSI[id_channel] = [0] * len(pred_MSI_samples)
            
            #df.loc[pred_MSI_samples, :].copy()
            
            for s in pred_MSI_samples:
                
                bef_correction = df.loc[s, :][:83].astype("float64").to_numpy().copy()
                
                corrected_profile, corrected_df = correct_FFPE_profile(V = bef_correction, 
                                                            W1 = noise_pattern,
                                                            sample_id= s,
                                                            precision = 0.99)
                noise_corrected_MSI.loc[s, :] = corrected_profile

            ## normalise the corrected profile
            noise_corrected_MSI_norm = (noise_corrected_MSI.T/noise_corrected_MSI.T.sum()).T
                                  
            ## adding columns to output dataframe
            noise_corrected_MSI['Prob(MSI)'] = df.loc[pred_MSI_samples, 'Prob(MSI)']
            noise_corrected_MSI['MILO_prediction'] = df.loc[pred_MSI_samples, 'MILO_prediction']
            #noise_corrected_MSI['long_del_intensity'] = df_tmp['long_del_intensity'].values
            
            return (df, noise_corrected_MSI)
    else:
        return (df)
    

def main():
    
    usage = """
    
    MILO predicts microsatellite instability in low-quality samples.
    
    Low-qulity samples could be 'shallow-sequencing' and/or 'low-purity' FF/FFPE samples. No matched normal is required also.
    
    Author & Copyright: Qingli Guo <qingliguo@outlook.com>
    ----------------------------------------------------------------
    
    To run MILO as a command-line script:
    
    1) for FF tissue without noise correction:
    
        python MILO_setup.py [-I|--Input] ./MILO_test_data.csv [-T|--TissueType] FF
    
    2) for FF tissue with noise correction with plot:
        
        python MILO_setup.py [-I|--Input] ./MILO_test_data.csv [-T|--TissueType] FF [-C|--NoiseCorrection] True [-P|--Plot] True
        
    3) for FF tissue with specified noise profile:
    
        python MILO_setup.py [-I|--Input] ./MILO_test_data.csv [-T|--TissueType] FF [-C|--NoiseCorrection] True [-N|--Noise_file] './test_noise.csv'
    
    ----------------------------------------------------------------
    To run MILO prediction on FFPE samples, using '[-T|--TissueType] FFPE'.
    
    ----------------------------------------------------------------
    To print out this usage, 'python MILO_setup.py [-h|--help]'.
    
    """
    
    full_cmd_arguments = sys.argv
    
    argument_list = full_cmd_arguments [1:]
    
    short_options = "help:I:T:C:N:PC:P"
    long_options = ["help", 
                    "Input=", 
                    "TissueType=", 
                    "NoiseCorrection=", 
                    "Noise_file=", 
                    "Prob_cutoff=",
                    "Plot="]
    
    try:
        arguments, values = getopt.getopt(argument_list, short_options, long_options)
    except getopt.error as err:
        print (str(err))
        sys.exit(2)
    
    ## Initial values for the arguments:
    file, ty, noise_file = "", "" , ""
    correction_flag, plot = False, False
    cutoff = 0.75
    
    ## count required argument number
    arg_length = 0 
    
    # Evaluate given options
            
    for current_argument, current_value in arguments:
        
        if current_argument in ["-h", "--help"]:
            print (usage)
            sys.exit(2)
        
        elif current_argument in ["-I", "--Input"]:
            file = current_value
            arg_length += 1
            print (f'\tInput file path: {file}')
        
        elif current_argument in ["-T", "--TissueType"]:
            ty = current_value
            arg_length += 1
            print (f'\tTissue type is: {ty}')
            
        elif current_argument in ["-C", "--NoiseCorrection"]:
            correction_flag = current_value
                
        elif current_argument in ["-N", "--Noise_file"]:
            noise_file = current_value
            print (f"\tMILO will use specified pattern for noise correction.")
            
        elif current_argument in ["-PC", "--Prob_cutoff"]:
            cutoff = current_value
            print (f"\tMILO will use {cutoff} as threshold to determine high confidence MSI samples.")
        
        elif current_argument in ["-P", "--Plot"]:
            plot = current_value
    
    if arg_length != 2:
        print ("Missing two required arguements for MILO: data path and TissueType.\n")
        print (usage)
        sys.exit(2)
        
    today = date.today()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f"Start MILO at {current_time} on {today}")
    
    df1 = MILO(Input = file, TissueType = ty)
    df1.to_csv("./MILO_predictions.csv", encoding='utf-8')
        
    predicted_MSI_samples = list(df1.index[df1['MILO_prediction'] == 'Yes'])
    
    if plot:
        os.system("if [ -d ./plots ]; then rm -Rf ./plots; fi")
        os.system("mkdir ./plots")
        
        print ("Plotting the indel profiles for predicted MMRd samples")
        
        for s in predicted_MSI_samples:
            profile_tmp = df1.loc[s, indel_channel_names]
            f1 = ID83_plot(profile_tmp, name = s, 
                           file = "./plots/" + str(s) + "_bef_correction.pdf")
                              
    if correction_flag:
        print ("Correcting the noise in predicted MMRd samples. May take a while...")
        
        ## If the user does not specify a noise pattern, we use the averaged high-confidence MMRp sample profile from the input as the noise profile.
        
        mmrp_cutoff = 0.1
        
        if noise_file == '':
            print ("no input noise file")
            
            if sum(df1['Prob(MSI)'] < mmrp_cutoff) > 0:
                df1_norm = (df1.iloc[:,:83].T/df1.iloc[:,:83].T.sum()).T
                
                noise_profile = df1_norm.loc[df1['Prob(MSI)'] < mmrp_cutoff, indel_channel_names].mean().astype("float64").to_numpy()
                
                print ("using cohort MMRp profile")
                
                #ID83_plot (noise_profile, norm = True)
                
                df1, df2 = MILO(Input = file, TissueType = ty, 
                                NoiseCorrection = correction_flag,
                                NoisePattern_ff = noise_profile, 
                                NoisePattern_ffpe = noise_profile, 
                                Prob_cutoff = cutoff)
                
            else:
                ## if there is no high-confidence MMRp samples in the input samples, we use noise profiles derived from our datasets.
            
                df1, df2 = MILO(Input = file, TissueType = ty, 
                                NoiseCorrection = correction_flag,
                                NoisePattern_ff = noise_profile_ff, 
                                NoisePattern_ffpe = noise_profile_ffpe,
                                Prob_cutoff = cutoff)
                
        else: 
            print ("using user specific noise profile")
            ## if the user specify a noise profile to use by profiving the file path to noise profile:
            
            noise_profile = pd.read_csv(noise_file).iloc[:, 0].values[:83].astype("float64")
            
            noise_profile_norm = noise_profile/sum(noise_profile)
            
            df1, df2 = MILO(Input = file, TissueType = ty, 
                            NoiseCorrection = correction_flag,
                            NoisePattern_ff = noise_profile_norm, 
                            NoisePattern_ffpe = noise_profile_norm, 
                            Prob_cutoff = cutoff)
            
        df2.to_csv ("./MILO_MMRd_LongDel_intensity.csv", encoding='utf-8')
        
        predicted_MSI_samples = list(df1.index[df1['MILO_prediction'] == 'Yes'])
        
        if plot:
            
            print ("Plotting the indel profiles for predicted MMRd samples")
            
            for s in predicted_MSI_samples:
                f2 = ID83_plot(df2.loc[s, indel_channel_names], name = s, file = "./plots/" + str(s) + "_after_correction.pdf")
    
           
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print (f'Finished MILO at {current_time} on {today}')
    
if __name__ == "__main__":
    main()
