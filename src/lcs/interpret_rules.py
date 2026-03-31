import numpy as np
import pandas as pd
import librosa
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pickle
from scipy.stats import pearsonr

def extract_acoustic_features(wav_path, sr=16000):
    """Extract named acoustic features from a wav file"""
    audio, _ = librosa.load(wav_path, sr=sr)

    features = {}

    # MFCCs (13 coefficients)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])

    # Spectral centroid — where the centre of mass of the spectrum is
    # high value = brighter/higher frequency sound
    spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features['spectral_centroid_mean'] = np.mean(spec_centroid)
    features['spectral_centroid_std'] = np.std(spec_centroid)

    # Spectral bandwidth — spread of frequencies around centroid
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    features['spectral_bandwidth_mean'] = np.mean(spec_bandwidth)
    features['spectral_bandwidth_std'] = np.std(spec_bandwidth)

    # Spectral rolloff — frequency below which 85% of energy is contained
    spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features['spectral_rolloff_mean'] = np.mean(spec_rolloff)
    features['spectral_rolloff_std'] = np.std(spec_rolloff)

    # Zero crossing rate — how often signal crosses zero
    # high value = more noise-like or high frequency content
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    features['zero_crossing_rate_mean'] = np.mean(zcr)
    features['zero_crossing_rate_std'] = np.std(zcr)

    # RMS energy — overall loudness
    rms = librosa.feature.rms(y=audio)
    features['rms_energy_mean'] = np.mean(rms)
    features['rms_energy_std'] = np.std(rms)

    # Spectral contrast — difference between peaks and valleys in spectrum
    # relates to harmonic vs percussive content
    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    for i in range(spec_contrast.shape[0]):
        features[f'spectral_contrast_{i+1}_mean'] = np.mean(spec_contrast[i])

    # Mel spectrogram energy in low/mid/high bands
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features['low_freq_energy'] = np.mean(mel_db[:42, :])    # 0-33% of mel bins
    features['mid_freq_energy'] = np.mean(mel_db[42:85, :])  # 33-66% of mel bins
    features['high_freq_energy'] = np.mean(mel_db[85:, :])   # 66-100% of mel bins

    return features


def build_acoustic_feature_matrix(segments_csv, max_samples=2000):
    """Extract acoustic features from a sample of audio segments"""
    df = pd.read_csv(segments_csv)

    # sample evenly from normal and anomalous
    normal = df[df['label'] == 0].sample(
        min(max_samples // 2, len(df[df['label'] == 0])),
        random_state=42
    )
    anomalous = df[df['label'] == 1].sample(
        min(max_samples // 2, len(df[df['label'] == 1])),
        random_state=42
    )
    sample_df = pd.concat([normal, anomalous]).reset_index(drop=True)

    print(f"Extracting acoustic features from {len(sample_df)} segments...")
    print(f"Normal: {len(normal)}, Anomalous: {len(anomalous)}")

    all_features = []
    for idx, row in sample_df.iterrows():
        if idx % 200 == 0:
            print(f"  Processing {idx}/{len(sample_df)}...")
        try:
            feats = extract_acoustic_features(row['filepath'])
            all_features.append(feats)
        except Exception as e:
            print(f"  Error on {row['filepath']}: {e}")
            all_features.append({})

    acoustic_df = pd.DataFrame(all_features)
    acoustic_df['label'] = sample_df['label'].values
    acoustic_df['filepath'] = sample_df['filepath'].values

    return acoustic_df, sample_df


def correlate_cnn_with_acoustic(
    cnn_features_path,
    segments_csv,
    selected_indices_path,
    output_dir,
    max_samples=2000
):
    """Correlate selected CNN features with named acoustic features"""

    os.makedirs(output_dir, exist_ok=True)

    # load selected CNN feature indices
    selected_indices = np.load(selected_indices_path)
    print(f"Selected CNN feature indices: {len(selected_indices)}")

    # load CNN features
    print("\nLoading CNN features...")
    X_all = np.load(cnn_features_path)
    print(f"CNN features shape: {X_all.shape}")

    # load segments to get filepaths
    segments_df = pd.read_csv(segments_csv)
    train_df = segments_df[segments_df['split'] == 'train'].reset_index(drop=True)

    # sample same indices for acoustic features
    normal = train_df[train_df['label'] == 0].sample(
        min(max_samples // 2, len(train_df[train_df['label'] == 0])),
        random_state=42
    )
    anomalous = train_df[train_df['label'] == 1].sample(
        min(max_samples // 2, len(train_df[train_df['label'] == 1])),
        random_state=42
    )
    sample_df = pd.concat([normal, anomalous]).reset_index(drop=True)

    # get CNN features for these same samples
    sample_indices = sample_df.index.tolist()
    X_sample = X_all[sample_indices]

    # extract only the selected features
    X_selected = X_sample[:, selected_indices]

    print(f"\nExtracting acoustic features from {len(sample_df)} segments...")
    all_acoustic = []
    for idx, row in sample_df.iterrows():
        if idx % 200 == 0:
            print(f"  Processing {idx}/{len(sample_df)}...")
        try:
            feats = extract_acoustic_features(row['filepath'])
            all_acoustic.append(feats)
        except Exception as e:
            all_acoustic.append({})

    acoustic_df = pd.DataFrame(all_acoustic).fillna(0)
    acoustic_feature_names = [c for c in acoustic_df.columns]
    acoustic_matrix = acoustic_df.values

    print(f"\nAcoustic features extracted: {len(acoustic_feature_names)}")
    print("Computing correlations...")

    # correlate each selected CNN feature with each acoustic feature
    correlation_results = []
    for i, cnn_idx in enumerate(selected_indices):
        cnn_feat = X_selected[:, i]
        best_corr = 0
        best_acoustic = ''
        best_abs_corr = 0

        for j, acoustic_name in enumerate(acoustic_feature_names):
            acoustic_feat = acoustic_matrix[:, j]
            try:
                corr, _ = pearsonr(cnn_feat, acoustic_feat)
                if abs(corr) > best_abs_corr:
                    best_abs_corr = abs(corr)
                    best_corr = corr
                    best_acoustic = acoustic_name
            except:
                continue

        correlation_results.append({
            'cnn_feature_index': int(cnn_idx),
            'best_acoustic_match': best_acoustic,
            'correlation': round(best_corr, 4),
            'abs_correlation': round(best_abs_corr, 4),
            'direction': 'positive' if best_corr > 0 else 'negative'
        })

        print(f"  CNN feature {cnn_idx:4d} → {best_acoustic:40s} (r={best_corr:.3f})")

    results_df = pd.DataFrame(correlation_results)
    results_df.to_csv(f'{output_dir}/cnn_acoustic_correlations.csv', index=False)
    print(f"\nCorrelations saved to {output_dir}/cnn_acoustic_correlations.csv")

    return results_df


def generate_readable_rules(
    rules_csv,
    correlations_csv,
    output_dir
):
    """Generate human readable rule descriptions using acoustic correlations"""

    rules_df = pd.read_csv(rules_csv)
    corr_df = pd.read_csv(correlations_csv)

    # build lookup: cnn_feature_index -> acoustic description
    feature_lookup = {}
    for _, row in corr_df.iterrows():
        idx = int(row['cnn_feature_index'])
        acoustic = row['best_acoustic_match']
        corr = row['correlation']
        abs_corr = row['abs_correlation']

        # create human readable description
        if abs_corr > 0.3:
            strength = 'strongly' if abs_corr > 0.5 else 'moderately'
            direction = 'positively' if corr > 0 else 'negatively'
            description = f"{acoustic} ({strength} {direction} correlated, r={corr:.2f})"
        else:
            description = f"CNN activation {idx} (weak acoustic correlation)"

        feature_lookup[idx] = description

    print("\n=== Human Readable Rules ===\n")
    readable_rules = []

    for idx, row in rules_df.iterrows():
        try:
            original_indices = eval(row['original_feature_indices'])
            conditions = eval(row['condition'])
            prediction = row['prediction']
            accuracy = row['accuracy']
            numerosity = row['numerosity']

            rule_parts = []
            for feat_idx, condition in zip(original_indices, conditions):
                acoustic_desc = feature_lookup.get(feat_idx, f"CNN activation {feat_idx}")
                low, high = condition[0], condition[1]
                rule_parts.append(
                    f"    {acoustic_desc}\n"
                    f"      is between {low:.3f} and {high:.3f}"
                )

            rule_text = (
                f"Rule {idx+1}:\n"
                f"  IF:\n" +
                "\n  AND\n".join(rule_parts) +
                f"\n  THEN: {prediction}\n"
                f"  Accuracy: {accuracy:.4f} | Numerosity: {numerosity}\n"
            )

            print(rule_text)
            readable_rules.append({
                'rule_number': idx + 1,
                'readable_rule': rule_text,
                'prediction': prediction,
                'accuracy': accuracy,
                'numerosity': numerosity
            })

        except Exception as e:
            continue

    readable_df = pd.DataFrame(readable_rules)
    readable_df.to_csv(f'{output_dir}/readable_rules.csv', index=False)
    print(f"\nReadable rules saved to {output_dir}/readable_rules.csv")

    return readable_df


if __name__ == "__main__":

    FEATURES_DIR = '/content/drive/MyDrive/cnn-lcs/features'
    LCS_DIR = '/content/drive/MyDrive/cnn-lcs/lcs'
    SEGMENTS_CSV = 'data/segments_split.csv'
    OUTPUT_DIR = '/content/drive/MyDrive/cnn-lcs/interpretability'

    # Step 1: correlate CNN features with acoustic features
    print("=" * 60)
    print("STEP 1: Correlating CNN features with acoustic properties")
    print("=" * 60)
    corr_df = correlate_cnn_with_acoustic(
        cnn_features_path=f'{FEATURES_DIR}/train_features.npy',
        segments_csv=SEGMENTS_CSV,
        selected_indices_path=f'{LCS_DIR}/selected_feature_indices.npy',
        output_dir=OUTPUT_DIR,
        max_samples=2000
    )

    # Step 2: generate readable rules
    print("\n" + "=" * 60)
    print("STEP 2: Generating human readable rules")
    print("=" * 60)
    readable_df = generate_readable_rules(
        rules_csv=f'{LCS_DIR}/rules.csv',
        correlations_csv=f'{OUTPUT_DIR}/cnn_acoustic_correlations.csv',
        output_dir=OUTPUT_DIR
    )