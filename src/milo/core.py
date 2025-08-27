# milo/core.py

"""
Core scientific logic for MILO.

This module contains the MILO class for MSI prediction, along with the
underlying matrix factorization and signal extraction algorithms for noise
correction.
"""

import sys
import joblib
import numpy as np
import pandas as pd
from importlib import resources
from scipy.stats import entropy
from typing import Dict, Any, Tuple
from . import constants

def sig_extraction(
    mutation_counts: np.ndarray,
    noise_profile: np.ndarray,
    iterations: int = 3000,
    precision: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """Separates a signal signature from a known noise profile.

    This function implements the core FFPEsig logic using Non-negative Matrix
    Factorization (NMF) with Kullback-Leibler (KL) divergence to extract a
    signal profile (W) and its contributions (H) from a mixed mutation count
    vector, given a fixed noise profile.

    Args:
        mutation_counts (np.ndarray): A 2D numpy array of mutation counts where
            rows are features and columns are samples.
        noise_profile (np.ndarray): A 1D numpy array representing the known
            noise signature (must have the same number of features as
            mutation_counts).
        iterations (int, optional): The maximum number of iterations for the
            NMF optimization. Defaults to 3000.
        precision (float, optional): The convergence precision threshold.
            The optimization stops if the rate of change of the loss falls
            below this value. Defaults to 0.95.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - W (np.ndarray): The matrix of signatures (column 0 is noise,
              column 1 is the extracted signal).
            - H (np.ndarray): The matrix of contributions for each signature.
    """
    n_features, n_samples = mutation_counts.shape
    rank = 2

    w2_signal = np.random.rand(n_features)
    W = np.array([noise_profile, w2_signal]).T
    W /= W.sum(axis=0)

    H = np.random.rand(rank, n_samples)
    loss_kl = np.zeros(iterations)

    for i in range(iterations):
        # Update H
        for a in range(rank):
            denominator_h = np.sum(W[:, a])
            with np.errstate(divide='ignore', invalid='ignore'):
                numerator_h = np.sum((W[:, a] / (W @ H).T).T * mutation_counts, axis=0)
                H[a, :] *= numerator_h / denominator_h

        # Update W (only for the signal component, a=1)
        a = 1
        denominator_w = np.sum(H[a, :])
        with np.errstate(divide='ignore', invalid='ignore'):
            numerator_w = np.sum((H[a, :] * mutation_counts) / (W @ H), axis=1)
            W[:, a] *= numerator_w / denominator_w
        
        W /= W.sum(axis=0)

        # Check for convergence
        loss_kl[i] = np.sum(entropy(mutation_counts, W @ H))
        if i > 200:
            last_batch_mean = np.mean(loss_kl[i-20:i])
            prev_batch_mean = np.mean(loss_kl[i-40:i-20])
            if prev_batch_mean > 0:
                change_rate = last_batch_mean / prev_batch_mean
                if change_rate >= precision and np.log(change_rate) <= 0:
                    break
    return W, H

def correct_single_profile(
    mutation_counts: np.ndarray,
    noise_profile: np.ndarray,
    sample_id: str
) -> np.ndarray:
    """Runs FFPEsig multiple times to get a stable noise-corrected profile.

    This function wraps the `sig_extraction` algorithm, running it 100 times
    with different random seeds to produce a stable, averaged estimate of the
    true signal profile by removing the noise component.

    Args:
        mutation_counts (np.ndarray): A 1D numpy array of the raw mutation
            counts for a single sample.
        noise_profile (np.ndarray): The noise profile to be removed.
        sample_id (str): The identifier for the sample, used for internal
            tracking during the runs.

    Returns:
        np.ndarray: A 1D numpy array representing the final, integer-based,
                    noise-corrected mutation profile.
    """
    solution_profiles = pd.DataFrame()
    non_zero_mask = mutation_counts > 0
    
    v_nonzeros = mutation_counts[non_zero_mask].reshape(-1, 1)
    w1_nonzeros = noise_profile[non_zero_mask]

    for i in range(100):
        np.random.seed(i + 1)
        w, h = sig_extraction(
            mutation_counts=v_nonzeros,
            noise_profile=w1_nonzeros,
            precision=0.99
        )
        predicted_v = np.zeros(len(mutation_counts))
        predicted_v[non_zero_mask] = w[:, 1] * h[1]
        solution_profiles[f"{sample_id}_run_{i+1}"] = predicted_v

    return solution_profiles.mean(axis=1).astype(int).to_numpy()

class MILO:
    """An object for predicting Microsatellite Instability (MSI) status."""
    def __init__(self, sample_type: str, yes_threshold: float = 0.75, maybe_threshold: float = 0.5, custom_noise_profile: np.ndarray = None):
        """Initializes the MILO predictor.

        Args:
            sample_type (str): The type of sample being analyzed. This
                determines which model and feature set to use.
                - Use 'standard', 'ffpe_lp', or 'ff_lp' for predictions.
            yes_threshold (float, optional): The probability threshold above
                which a prediction is classified as 'Yes'. Defaults to 0.75.
            maybe_threshold (float, optional): The probability threshold above
                which a prediction is classified as 'Maybe'. Defaults to 0.5.
            custom_noise_profile (np.ndarray, optional): A user-provided noise
                profile to use for correction, overriding the default.
                Defaults to None.
        """
        self.sample_type = sample_type
        self.yes_threshold = yes_threshold
        self.maybe_threshold = maybe_threshold
        
        self.model = None
        # Only load models if we are not in a calculation-only mode
        if self.sample_type != 'IntensityOnly':
            self._models = self._load_models()
            model_map = {'standard': 'Deep', 'ffpe_lp': 'FFPE', 'ff_lp': 'FF'}
            self.model = self._models[model_map[self.sample_type]]

        if self.sample_type == 'standard':
            self.features = constants.INDEL_CHANNEL_NAMES
        else:
            feature_map = {'ffpe_lp': 'FFPE', 'ff_lp': 'FF'}
            # Handle IntensityOnly case where sample_type might not be in the map
            if self.sample_type in feature_map:
                self.features = constants.SELECTED_FEATURES[feature_map[self.sample_type]]

        self.noise_profile = None
        if self.sample_type in ['ffpe_lp', 'ff_lp']:
            if custom_noise_profile is not None:
                self.noise_profile = custom_noise_profile
            else:
                noise_map = {'ffpe_lp': 'FFPE', 'ff_lp': 'FF'}
                self.noise_profile = constants.DEFAULT_NOISE_PROFILES[noise_map[self.sample_type]]

    def _load_models(self) -> Dict[str, Any]:
        """Loads the built-in scikit-learn models from package data.

        This method uses importlib.resources to safely access the model files
        that are included with the installed package.

        Returns:
            Dict[str, Any]: A dictionary mapping model names ('FFPE', 'FF',
                            'Deep') to the loaded model objects.
        
        Raises:
            SystemExit: If the model files cannot be found.
        """
        try:
            model_files_path = resources.files('milo') / 'model_files'
            return {
                'FFPE': joblib.load(model_files_path / "FFPE_rf.joblib"),
                'FF': joblib.load(model_files_path / "FF_rf.joblib"),
                'Deep': joblib.load(model_files_path / "Deep_rf.joblib"),
            }
        except (FileNotFoundError, ModuleNotFoundError) as e:
            print(f"Error: Built-in model file not found. Ensure the package is installed correctly. {e}")
            sys.exit(1)

    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Predicts MSI status for a batch of samples.

        Args:
            input_df (pd.DataFrame): A DataFrame of mutation profiles, with
                samples as rows and 83 indel channels as columns.

        Returns:
            pd.DataFrame: A DataFrame with SampleIDs as the index and columns
                for MSI probability ('Prob(MSI)') and the final classification
                ('MILO_prediction': 'Yes', 'Maybe', or 'No').
        """
        print(f"Running prediction for '{self.sample_type}' sample type...")

        if self.sample_type == 'standard':
            data_for_model = input_df
        else:
            row_sums = input_df.sum(axis=1)
            row_sums[row_sums == 0] = 1
            data_for_model = input_df.div(row_sums, axis=0)

        x_test = data_for_model[self.features].to_numpy()
        
        results_df = pd.DataFrame(index=input_df.index)
        results_df['Prob(MSI)'] = self.model.predict_proba(x_test)[:, 1]
        results_df['MILO_prediction'] = 'No'
        
        results_df.loc[results_df['Prob(MSI)'] > self.maybe_threshold, 'MILO_prediction'] = 'Maybe'
        results_df.loc[results_df['Prob(MSI)'] > self.yes_threshold, 'MILO_prediction'] = 'Yes'
        
        return results_df

    def run_noise_correction(self, results_df: pd.DataFrame, custom_profile_override: np.ndarray = None) -> pd.DataFrame:
        """Performs noise correction on high-confidence MMRd samples.

        This method identifies samples classified as 'Yes' and applies the
        noise correction algorithm to their mutation profiles.

        Args:
            results_df (pd.DataFrame): The input DataFrame containing the raw
                mutation counts and the 'MILO_prediction' column.
            custom_profile_override (np.ndarray, optional): A custom noise
                profile to use instead of the default. Defaults to None.

        Returns:
            pd.DataFrame: A new DataFrame containing the noise-corrected
                profiles for the 'Yes' samples. Returns an empty DataFrame
                if no 'Yes' samples are found.
        """
        if custom_profile_override is not None:
            noise_profile_to_use = custom_profile_override
            print("\nUsing derived custom noise profile for correction.")
        else:
            noise_profile_to_use = self.noise_profile
            print("\nUsing default noise profile for correction.")

        if noise_profile_to_use is None:
            print("Warning: Noise correction is not applicable for this run (no noise profile available).")
            return pd.DataFrame()

        msi_positive_samples = results_df[results_df['MILO_prediction'] == 'Yes']
        
        if msi_positive_samples.empty:
            print("No high-confidence MMRd samples found to perform noise correction on.")
            return pd.DataFrame()

        print(f"Performing noise correction on {len(msi_positive_samples)} sample(s)...")
        corrected_data = []
        for sample_id, row in msi_positive_samples.iterrows():
            original_profile = row[constants.INDEL_CHANNEL_NAMES].to_numpy().astype("float64")
            corrected_profile_arr = correct_single_profile(original_profile, noise_profile_to_use, sample_id)
            corrected_data.append(pd.Series(corrected_profile_arr, index=constants.INDEL_CHANNEL_NAMES, name=sample_id))
        
        if not corrected_data:
            return pd.DataFrame()

        corrected_df = pd.concat(corrected_data, axis=1).T
        corrected_df = corrected_df.join(msi_positive_samples[['Prob(MSI)', 'MILO_prediction']])
        return corrected_df

    def calculate_msi_intensity(self, indel_profile_df: pd.DataFrame, use_proportions: bool = False) -> pd.Series:
        """Calculates the MSI intensity score for each sample.

        This score is a weighted sum of counts in specific long-deletion
        and repeat-associated indel channels, designed to quantify the
        degree of microsatellite instability.

        Args:
            indel_profile_df (pd.DataFrame): DataFrame with samples as rows
                and 83 indel channels as columns.
            use_proportions (bool, optional): If True, treats values below 10
                as significant (for normalized data). If False, sets counts
                below 10 to zero (for absolute counts). Defaults to False.

        Returns:
            pd.Series: A pandas Series containing the MSI intensity score for
                each sample, indexed by SampleID.
        """
        df = indel_profile_df.copy()
        
        agg_items = ['1_ID_T_5', '2_Del_R', '3_Del_R', '4_Del_R', '5_Del_R']
        agg_df = pd.DataFrame(index=df.index, columns=agg_items, dtype=np.float64)

        agg_df['1_ID_T_5'] = df['1_Del_T_5'].values + df['1_Ins_T_5'].values
        for i, item in enumerate(agg_items[1:]):
            agg_df[item] = df.loc[:, [item in c for c in df.columns]].sum(axis=1)
        
        def _compute_score(sample_array: np.ndarray, r: float = 0.001) -> float:
            if not use_proportions:
                sample_array[sample_array < 10] = 0
            
            weights = np.log10(r**-np.arange(len(sample_array)))
            weights[0] = 1
            return np.sum(sample_array * weights)

        intensity_scores = agg_df.apply(_compute_score, axis=1)
        return intensity_scores
