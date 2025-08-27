# milo/cli.py

"""
Command-line interface (CLI) for MILO.

This module provides the main entry point for running MILO from the command
line. It handles argument parsing, subcommand logic, and orchestrates the
prediction and training pipelines.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from . import constants
from .core import MILO
from .plots import plot_id83_profile, plot_id83_comparison
from .utils import load_and_prepare_data, load_lp_norm_data

class CustomArgumentParser(argparse.ArgumentParser):
    """Custom parser to override the default error handling."""
    def error(self, message):
        """Prints a custom error message and the help text, then exits."""
        sys.stderr.write(f'Error: {message}\n\n')
        self.print_help()
        sys.exit(2)

def create_main_parser() -> argparse.ArgumentParser:
    """Creates and configures the main argument parser for the MILO CLI.

    Returns:
        argparse.ArgumentParser: The fully configured parser instance.
    """
    parser = CustomArgumentParser(
        prog='milo',
        description="MILO: Predicts microsatellite instability using Long-deletion signature. Use 'milo <command> --help' for details on a specific command.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="Author: Qingli Guo <qingliguo@outlook.com; qingliguo.ramina@gmail.com> | For full documentation, see the README.md file."
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    # --- PREDICT sub-command parser ---
    predict_parser = subparsers.add_parser('predict', 
                                         help='Predict MSI status using a built-in or custom model.',
                                         description='Predicts MSI status from 83-channel indel mutation profiles of standard or low-pass DNA-seq data.',
                                         formatter_class=argparse.RawTextHelpFormatter)
    
    predict_req_group = predict_parser.add_argument_group('Required Arguments')
    predict_req_group.add_argument("-i", "--input", type=Path, required=True, metavar='<PATH>', 
                                   help="Path to the 83-channel mutation profile (CSV or TSV).")
    
    model_group = predict_parser.add_argument_group('Model Selection (choose one)')
    model_exclusive_group = model_group.add_mutually_exclusive_group(required=True)
    model_exclusive_group.add_argument("-s", "--sample_type", type=str, metavar='<TYPE>', 
                                       help=("Specify the built-in model by sample type (ffpe_lp, ff_lp, standard).\n"
                                             "Required if --custom_model is not used."))
    model_exclusive_group.add_argument("-m", "--custom_model", type=Path, metavar='<PATH>',
                                       help=("Path to a custom trained model (*.joblib).\n"
                                             "Required if --sample_type is not used."))

    output_group = predict_parser.add_argument_group('Output Options')
    output_group.add_argument("-o", "--output", type=Path, default=Path("./milo_results"), metavar='<PATH>',
                                help="Directory to save prediction results and plots. (Default: milo_results/)")
    output_group.add_argument("-p", "--plot", action='store_true', 
                                help="A flag to generate plots for all samples in categorized subdirectories.")

    analysis_group = predict_parser.add_argument_group('Analysis Options')
    analysis_group.add_argument("-int", "--msi_intensity", action='store_true', 
                                help="A flag to calculate and add an MSI intensity score to the output.")
    analysis_group.add_argument("--cov_norm", type=Path, metavar='<PATH>',
                                help="Path to a file with 'coverage' and optional 'purity' for low-pass normalization.")
    analysis_group.add_argument("--purity_norm", action='store_true', 
                                help="A flag to enable purity-based normalization for low-pass data.")
    analysis_group.add_argument("-c", "--noise_correction", action='store_true', 
                                help="A flag to enable noise correction. Only for FFPE_lp and FF_lp modes.")
    analysis_group.add_argument("--custom_noise_pattern", action='store_true', 
                                help="A flag to derive a custom noise profile from low MSI-probability samples.")
    analysis_group.add_argument("--noise_threshold", type=float, default=0.1, metavar='<FLOAT>',
                                help="Probability threshold to identify samples for noise derivation. (Default: 0.1)")

    tuning_group = predict_parser.add_argument_group('Tuning Options')
    tuning_group.add_argument("--yes_threshold", type=float, default=0.75, metavar='<FLOAT>',
                              help="Minimum probability for a 'Yes' classification. (Default: 0.75)")
    tuning_group.add_argument("--maybe_threshold", type=float, default=0.5, metavar='<FLOAT>',
                              help="Minimum probability for a 'Maybe' classification. (Default: 0.5)")
    tuning_group.add_argument("--seed", type=int, default=1234, metavar='<INT>',
                              help="Random seed for reproducible predictions (e.g., for noise correction). (Default: 1234)")

    # --- TRAIN sub-command parser ---
    train_parser = subparsers.add_parser('train', 
                                       help='Train a new custom MSI classification model.',
                                       description='Trains a new Random Forest classifier for MSI prediction from a labeled dataset.',
                                       formatter_class=argparse.RawTextHelpFormatter)
    
    train_req_group = train_parser.add_argument_group('Required Arguments')
    train_req_group.add_argument("-i", "--input", type=Path, required=True, metavar='<PATH>',
                                 help="Path to the labeled training dataset (CSV/TSV).")
    
    train_opt_group = train_parser.add_argument_group('Optional Arguments')
    train_opt_group.add_argument("-o", "--output", type=Path, default=Path("new_model"), metavar='<PATH>',
                                 help=("Path to save the custom trained model.\n"
                                       "- If a directory is provided, the model is saved as 'custom_milo_model.joblib' inside it.\n"
                                       "- If a full filepath is provided, it is used directly.\n"
                                       "(Default: new_model/)"))
    train_opt_group.add_argument("--label_column", type=str, default='MSI_status', metavar='<NAME>',
                                 help="Name of the column in the training data that contains the labels (0 or 1).\n(Default: MSI_status)")
    train_opt_group.add_argument("--feature_set", type=str, default='whole_features', 
                                 choices=['whole_features', 'FFPE_features', 'FF_features'],
                                 help="Feature set to use for training the model. (Default: whole_features)")
    train_opt_group.add_argument("--train_abs", action='store_true',
                                 help="A flag to train the model on absolute indel counts instead of normalized proportions.")
    train_opt_group.add_argument("--seed", type=int, default=1234, metavar='<INT>',
                                 help="Random seed for reproducible model training. (Default: 1234)")

    return parser

def train_and_save_model(args: argparse.Namespace):
    """Handles the model training and saving workflow.

    This function reads a labeled dataset, trains a RandomForestClassifier
    based on the specified features and data format (absolute or proportional),
    and saves the trained model to a file.

    Args:
        args (argparse.Namespace): The parsed command-line arguments from the
            'train' subcommand.
    """
    np.random.seed(args.seed)
    print(f"--- Starting MILO Training Mode at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

    try:
        raw_df = pd.read_csv(args.input, sep=None, engine='python', index_col=0)
    except Exception as e:
        print(f"Error: Could not read the training file: {e}")
        sys.exit(1)

    required_features = set(constants.INDEL_CHANNEL_NAMES)
    if required_features.issubset(raw_df.columns):
        print("Samples x Features format detected for training.")
        train_df = raw_df
        train_df.rename(columns=lambda c: str(c).replace(':', '_'), inplace=True)
    elif required_features.issubset(raw_df.index):
        print("Features x Samples format detected for training.")
        train_df = raw_df.T
        train_df.rename(columns=lambda c: str(c).replace(':', '_'), inplace=True)
    else:
        print("Error: The training file must contain all 83 indel feature columns, either in the header or in the first column.")
        sys.exit(1)

    if args.label_column not in train_df.columns:
        print(f"Error: The label column '{args.label_column}' was not found in the training file.")
        sys.exit(1)
    
    print(f"Preparing data using feature set: '{args.feature_set}'...")
    feature_map = {
        'whole_features': constants.INDEL_CHANNEL_NAMES,
        'FFPE_features': constants.SELECTED_FEATURES['FFPE'],
        'FF_features': constants.SELECTED_FEATURES['FF']
    }
    features_to_use = feature_map[args.feature_set]
    
    X_raw = train_df[features_to_use]
    if args.train_abs:
        print("Training on ABSOLUTE counts.")
        X = X_raw.to_numpy()
        data_format_value = 'absolute'
    else:
        print("Training on NORMALIZED proportions.")
        row_sums = X_raw.sum(axis=1)
        row_sums[row_sums == 0] = 1
        X = X_raw.div(row_sums, axis=0).to_numpy()
        data_format_value = 'proportions'

    y_raw = train_df[args.label_column]
    if pd.api.types.is_string_dtype(y_raw):
        print("String labels detected. Using LabelEncoder.")
        encoder = LabelEncoder()
        y = encoder.fit_transform(y_raw)
    else:
        y = y_raw.to_numpy()

    print(f"Training RandomForestClassifier with random seed: {args.seed}...\n")
    rf = RandomForestClassifier(random_state=args.seed)
    rf.fit(X, y)
    print("Training complete.\n")

    if args.output.suffix in ['.joblib', '.pkl']:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        output_path = args.output / "custom_milo_model.joblib"
    
    model_payload = {
        'model': rf,
        'features': features_to_use,
        'data_format': data_format_value
    }
    joblib.dump(model_payload, output_path)
    print(f"\nCustom model and feature list saved to: {output_path}\n")
    print(f"--- MILO Training Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

def predict_with_custom_model(args: argparse.Namespace, input_df: pd.DataFrame) -> pd.DataFrame:
    """Loads a custom model and uses it for prediction.

    Args:
        args (argparse.Namespace): The parsed command-line arguments from the
            'predict' subcommand.
        input_df (pd.DataFrame): The DataFrame of mutation profiles to predict.

    Returns:
        pd.DataFrame: A DataFrame containing the prediction probabilities and
            final classifications.
    
    Raises:
        SystemExit: If the custom model cannot be loaded or is incompatible
            with the input data.
    """
    print("--- Running MILO using Custom Model for Prediction ---")
    
    try:
        model_payload = joblib.load(args.custom_model)
        model = model_payload['model']
        features = model_payload['features']
        data_format = model_payload.get('data_format', 'absolute')
    except Exception as e:
        print(f"Error: Could not load the custom model file: {e}")
        sys.exit(1)

    print(f"Custom model loaded. Applying '{data_format}' format for prediction.")
    
    if not set(features).issubset(input_df.columns):
        print("Error: The input data is missing some of the features the custom model was trained on.")
        sys.exit(1)

    data_to_predict = input_df[features]
    if data_format == 'proportions':
        row_sums = data_to_predict.sum(axis=1)
        row_sums[row_sums == 0] = 1
        x_test = data_to_predict.div(row_sums, axis=0).to_numpy()
    else:
        x_test = data_to_predict.to_numpy()

    results_df = pd.DataFrame(index=input_df.index)
    results_df['Prob(MSI)'] = model.predict_proba(x_test)[:, 1]
    results_df['MILO_prediction'] = 'No'
    
    results_df.loc[results_df['Prob(MSI)'] > args.maybe_threshold, 'MILO_prediction'] = 'Maybe'
    results_df.loc[results_df['Prob(MSI)'] > args.yes_threshold, 'MILO_prediction'] = 'Yes'
    
    return results_df

def run_prediction_pipeline(args: argparse.Namespace):
    """The main pipeline for the 'predict' sub-command.

    This function orchestrates the entire prediction workflow, including data
    loading, model prediction, optional noise correction, optional intensity
    score calculation, and saving all results and plots.

    Args:
        args (argparse.Namespace): The parsed command-line arguments from the
            'predict' subcommand.
    """
    np.random.seed(args.seed)
    print(f"--- Starting MILO prediction at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    
    try:
        full_df = load_and_prepare_data(args.input)
    except (IOError, ValueError) as e:
        print(f"Error processing input file: {e}")
        sys.exit(1)
    
    if args.sample_type:
        args.sample_type = args.sample_type.lower()

    if args.custom_model:
        prediction_results = predict_with_custom_model(args, full_df)
        model_payload = joblib.load(args.custom_model)
        data_format = model_payload.get('data_format', 'absolute')
        original_sample_type = 'ffpe_lp' if data_format == 'proportions' else 'standard'
        args.sample_type = 'custom'
    else:
        allowed_types = ['standard', 'ffpe_lp', 'ff_lp']
        if args.sample_type not in allowed_types:
            print(f"Error: invalid choice for --sample_type: '{args.sample_type}' (choose from {', '.join(allowed_types)})")
            sys.exit(1)
        
        if args.sample_type in ['ffpe_lp', 'ff_lp'] and args.noise_threshold > 0.5:
             parser = create_main_parser()
             parser.error("--noise_threshold cannot be greater than 0.5.")

        milo_tool = MILO(
            sample_type=args.sample_type, 
            yes_threshold=args.yes_threshold, 
            maybe_threshold=args.maybe_threshold
        )
        prediction_results = milo_tool.predict(full_df)
        original_sample_type = args.sample_type

    final_output_df = prediction_results.copy()
    corrected_df = None
    
    if args.noise_correction:
        if args.sample_type == 'custom':
            if original_sample_type in ['ffpe_lp', 'ff_lp']:
                print("\nNoise correction with custom model: Automatically deriving noise profile from data...\n")
                
                noise_samples_mask = prediction_results['Prob(MSI)'] < args.noise_threshold
                noise_samples_df = full_df[noise_samples_mask]
                custom_noise_profile = None

                if not noise_samples_df.empty:
                    print(f"Deriving custom noise profile from {len(noise_samples_df)} samples.")
                    row_sums = noise_samples_df.sum(axis=1)
                    row_sums[row_sums == 0] = 1
                    normalized_noise_df = noise_samples_df.div(row_sums, axis=0)
                    custom_noise_profile = normalized_noise_df.mean(axis=0).to_numpy()
                
                    milo_tool_for_correction = MILO(sample_type=original_sample_type)
                    temp_df_for_correction = full_df.join(prediction_results)
                    corrected_df = milo_tool_for_correction.run_noise_correction(
                        temp_df_for_correction, 
                        custom_profile_override=custom_noise_profile
                    )
                else:
                    print("Warning: Could not derive a custom noise profile as no samples were found below the threshold. Skipping noise correction.")
            else:
                 print("\nWarning: Noise correction is not applicable for custom models trained on 'standard' (absolute count) data. Skipping step.")

        elif args.sample_type in ['ffpe_lp', 'ff_lp']:
            custom_noise_profile = None
            if args.custom_noise_pattern:
                print(f"\nAttempting to derive custom noise profile using threshold < {args.noise_threshold}...\n")
                noise_samples_mask = prediction_results['Prob(MSI)'] < args.noise_threshold
                noise_samples_df = full_df[noise_samples_mask]

                if not noise_samples_df.empty:
                    print(f"Deriving custom noise profile from {len(noise_samples_df)} samples.")
                    row_sums = noise_samples_df.sum(axis=1)
                    row_sums[row_sums == 0] = 1
                    normalized_noise_df = noise_samples_df.div(row_sums, axis=0)
                    custom_noise_profile = normalized_noise_df.mean(axis=0).to_numpy()
                else:
                    print("Warning: No samples found below the threshold to derive a custom noise profile. Will fall back to default profile for correction.")

            milo_tool_for_correction = MILO(sample_type=args.sample_type)
            temp_df_for_correction = full_df.join(prediction_results)
            corrected_df = milo_tool_for_correction.run_noise_correction(
                temp_df_for_correction, 
                custom_profile_override=custom_noise_profile
            )

    if corrected_df is not None and not corrected_df.empty:
        args.output.mkdir(parents=True, exist_ok=True)
        corrected_path = args.output / "MILO_noise_corrected_profiles.csv"
        corrected_df.to_csv(corrected_path)
        print(f"Noise-corrected profiles saved to {corrected_path}\n")
    
    if args.msi_intensity:
        milo_intensity_calculator = MILO(sample_type='IntensityOnly')
        run_mode = args.sample_type
        if run_mode == 'custom':
            model_payload = joblib.load(args.custom_model)
            data_format = model_payload.get('data_format', 'absolute')
            run_mode = 'standard' if data_format == 'absolute' else 'ffpe_lp'
        
        if run_mode == 'standard':
            intensity_scores = milo_intensity_calculator.calculate_msi_intensity(full_df, use_proportions=False)
            final_output_df['MSI intensity'] = intensity_scores
        
        elif run_mode in ['ffpe_lp', 'ff_lp']:
            if not args.custom_model:
                 print("\n[Warning] You have requested MSI intensity scores for low-pass data.\n"
                       "It is recommended to compare the derived intensity scores between samples\n"
                       "that have been processed under similar protocol and platform.\n\n"
                       "Note that high background noise can dilute signal in low-purity MMRd samples.\n"
                       "It is recommended to use '--noise_correction' to minimise this dilution effect.\n")
            
            if args.cov_norm is None:
                print("\nCalculating 'Relative MSI intensity'...")
                row_sums = full_df.sum(axis=1)
                row_sums[row_sums == 0] = 1
                normalized_df = full_df.div(row_sums, axis=0)
                intensity_scores = milo_intensity_calculator.calculate_msi_intensity(normalized_df, use_proportions=True)
                final_output_df['Relative MSI intensity'] = intensity_scores
            
            else:
                print("\nCalculating 'MSI intensity (adjusted)'...")
                try:
                    norm_df = load_lp_norm_data(args.cov_norm)
                    merged_df = full_df.join(norm_df).join(prediction_results)
                    
                    missing_mask = merged_df['coverage'].isnull()
                    if missing_mask.all():
                        print("\nError: No sample IDs in the input coverage file match the expected IDs. MSI intensity calculation is skipped")
                    else:
                        if missing_mask.any():
                            missing_samples = merged_df[missing_mask].index.tolist()
                            print(f"\n[Warning] Coverage info from the --cov_norm file was not found for {len(missing_samples)} sample(s):\n"
                                  f"{missing_samples}\n"
                                  "These samples will be excluded from the 'MSI intensity (adjusted)' calculation.")

                        valid_mask = merged_df['coverage'].notnull()
                        valid_samples_df = merged_df[valid_mask].copy()
                        invalid_samples_index = merged_df[~valid_mask].index

                        valid_samples_df['final_purity'] = 1.0
                        counts_to_use = full_df.loc[valid_samples_df.index].copy()
                        
                        if args.purity_norm:
                            yes_samples_mask = valid_samples_df['MILO_prediction'] == 'Yes'
                            
                            if 'purity' in valid_samples_df.columns:
                                print("\nCalculating 'MSI intensity (adjusted)' using coverage and user-provided purity normalization...")
                                valid_samples_df.loc[yes_samples_mask, 'final_purity'] = valid_samples_df.loc[yes_samples_mask, 'purity']
                            
                            elif corrected_df is not None and not corrected_df.empty:
                                print("\nPurity information not found in --cov_norm file.\n"
                                      "Estimating MMRd signal ratio from noise correction results.\n"
                                      "Calculating 'MSI intensity (adjusted)' using coverage and estimated purity normalization...")
                                
                                corrected_valid_samples = corrected_df.index.intersection(valid_samples_df.index)
                                corrected_counts_sum = corrected_df.loc[corrected_valid_samples, constants.INDEL_CHANNEL_NAMES].sum(axis=1)
                                original_counts_sum = full_df.loc[corrected_valid_samples].sum(axis=1)
                                estimated_purity = corrected_counts_sum / original_counts_sum
                                
                                valid_samples_df.loc[estimated_purity.index, 'final_purity'] = estimated_purity
                                counts_to_use.loc[corrected_df.index] = corrected_df.loc[corrected_df.index, constants.INDEL_CHANNEL_NAMES]
                            else:
                                 print("\nWarning: Purity normlization is requested. But purity info not found and could not be estimated from noise correction results.\n"
                                       "Skipping purity normlization\n")
                        else:
                            print("\nCalculating 'MSI intensity (adjusted)' using coverage normalization...")

                        denominator = valid_samples_df['coverage'] * valid_samples_df['final_purity']
                        denominator[denominator == 0] = 1 
                        adjusted_counts = counts_to_use.div(denominator, axis=0)
                        
                        valid_scores = milo_intensity_calculator.calculate_msi_intensity(adjusted_counts, use_proportions=False)
                        invalid_scores = pd.Series(np.nan, index=invalid_samples_index)
                        intensity_scores = pd.concat([valid_scores, invalid_scores])
                        final_output_df['MSI intensity (adjusted)'] = intensity_scores

                except Exception as e:
                    print(f"\nError processing --cov_norm file: {e}. Skipping adjusted intensity calculation.")

    args.output.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output / "MILO_predictions.csv"
    final_output_df.to_csv(predictions_path)
    print(f"\nPrediction results saved to {predictions_path}")
    
    if args.plot:
        plot_dir = args.output / "plots"
        mmrd_dir = plot_dir / "MMRd"
        mmrp_dir = plot_dir / "MMRp"
        maybe_dir = plot_dir / "Maybe"
        comparison_dir = plot_dir / "Comparison"
        for d in [mmrd_dir, mmrp_dir, maybe_dir, comparison_dir]:
            d.mkdir(parents=True, exist_ok=True)

        plot_df = full_df.join(prediction_results)
        
        mmrd_samples = plot_df[plot_df['MILO_prediction'] == 'Yes']
        mmrp_samples = plot_df[plot_df['MILO_prediction'] == 'No']
        maybe_samples = plot_df[plot_df['MILO_prediction'] == 'Maybe']

        print(f"\nGenerating plots in {plot_dir}...")
        
        for sample_id, row in mmrd_samples.iterrows():
            plot_id83_profile(
                row[constants.INDEL_CHANNEL_NAMES].to_numpy(),
                plot_title=f"{sample_id} (Prob(MSI) = {row['Prob(MSI)']:.2f})",
                output_path=mmrd_dir / f"{sample_id}_profile.pdf"
            )
        for sample_id, row in mmrp_samples.iterrows():
            plot_id83_profile(
                row[constants.INDEL_CHANNEL_NAMES].to_numpy(),
                plot_title=f"{sample_id} (Prob(MSI) = {row['Prob(MSI)']:.2f})",
                output_path=mmrp_dir / f"{sample_id}_profile.pdf"
            )
        for sample_id, row in maybe_samples.iterrows():
            plot_id83_profile(
                row[constants.INDEL_CHANNEL_NAMES].to_numpy(),
                plot_title=f"{sample_id} (Prob(MSI) = {row['Prob(MSI)']:.2f})",
                output_path=maybe_dir / f"{sample_id}_profile.pdf"
            )
        
        if corrected_df is not None:
            print(f"Generating noise correction comparison plots...")
            for sample_id, corrected_row in corrected_df.iterrows():
                original_row = plot_df.loc[sample_id]
                plot_id83_comparison(
                    sig1=original_row[constants.INDEL_CHANNEL_NAMES].to_numpy(),
                    sig2=corrected_row[constants.INDEL_CHANNEL_NAMES].to_numpy(),
                    name1='Before noise correction',
                    name2='After noise correction',
                    plot_title=f"{sample_id} (Prob(MSI) = {original_row['Prob(MSI)']:.2f})",
                    output_path=comparison_dir / f"{sample_id}_noise_correction_comparison.pdf"
                )

        if not mmrd_samples.empty and not mmrp_samples.empty:
            print("Generating average MMRd vs MMRp comparison plot...")
            mmrd_mean_profile = mmrd_samples[constants.INDEL_CHANNEL_NAMES].mean().to_numpy()
            mmrp_mean_profile = mmrp_samples[constants.INDEL_CHANNEL_NAMES].mean().to_numpy()

            plot_id83_comparison(
                sig1=mmrd_mean_profile,
                sig2=mmrp_mean_profile,
                name1="MMR-deficient",
                name2="MMR-proficient",
                plot_title="Average MMRd vs MMRp Profiles",
                output_path=comparison_dir / "MMRd_vs_MMRp_ID83.pdf"
            )
    print(f"\n--- MILO finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

def main():
    """Main entry point for the MILO command-line tool."""
    parser = create_main_parser()
    args = parser.parse_args()
    
    if args.command == 'train':
        train_and_save_model(args)
    elif args.command == 'predict':
        run_prediction_pipeline(args)

if __name__ == "__main__":
    main()
