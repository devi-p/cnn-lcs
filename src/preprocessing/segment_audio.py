import librosa
import soundfile as sf
import pandas as pd
from pathlib import Path

def segment_audio(row, output_dir, sr=16000, segment_length=1):
    """Cut a single clip into 1-second segments"""
    filepath = row['filepath']
    label = row['label']
    machine = row['machine_type']
    split = row['split']
    
    # load audio at target sample rate
    audio, _ = librosa.load(filepath, sr=sr)
    
    segment_samples = sr * segment_length
    segments_info = []
    
    for i, start in enumerate(range(0, len(audio), segment_samples)):
        segment = audio[start:start + segment_samples]
        
        # skip incomplete segments
        if len(segment) < segment_samples:
            continue
        
        # build output path: data/segments/bearing/train/normal/filename_seg0.wav
        label_str = 'normal' if label == 0 else 'anomaly'
        out_folder = Path(output_dir) / machine / split / label_str
        out_folder.mkdir(parents=True, exist_ok=True)
        
        stem = Path(filepath).stem
        out_path = out_folder / f"{stem}_seg{i}.wav"
        
        sf.write(str(out_path), segment, sr)
        
        segments_info.append({
            'filepath': str(out_path),
            'machine_type': machine,
            'label': label,
            'split': split,
            'parent_clip': filepath,
            'segment_index': i
        })
    
    return segments_info

def run_segmentation(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    all_segments = []
    total = len(df)
    
    for idx, row in df.iterrows():
        print(f"Processing {idx+1}/{total}: {Path(row['filepath']).name}")
        segments = segment_audio(row, output_dir)
        all_segments.extend(segments)
    
    segments_df = pd.DataFrame(all_segments)
    segments_df.to_csv('data/segments_split.csv', index=False)
    
    print(f"\nSegmentation Complete")
    print(f"Total segments: {len(segments_df)}")
    print(f"Training segments: {len(segments_df[segments_df['split']=='train'])}")
    print(f"Test segments: {len(segments_df[segments_df['split']=='test'])}")

if __name__ == "__main__":
    run_segmentation('data/dataset_split.csv', 'data/segments')