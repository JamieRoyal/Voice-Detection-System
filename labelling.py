import os
import librosa
import numpy as np
import pandas as pd

# Lists to store audio features and labels
features = []
label_list = []

# Define audio file paths and labels
audio_dirs = ['dataset/human_audios/', 'dataset/ai_audios/']
audio_labels = [0, 1]  # 0 - Human made | 1 - AI made

# A loop to labels audios files in each directory
for audio_dirs, labels in zip(audio_dirs, audio_labels):
    for filename in os.listdir(audio_dirs):
        file_path = os.path.join(audio_dirs, filename)
        audio, sr = librosa.load(file_path, sr=None)

        # Extract Features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs_flattened = np.mean(mfccs.T, axis=0)

        # Append features and labels to lists
        features.append(mfccs_flattened)
        label_list.append(labels)

# Convert lists into Numpy Arrays
features = np.array(features)
label_list = np.array(label_list)

# Create a Pandas DataFrame
df = pd.DataFrame(features)
df['label'] = label_list

# Save DataFrame to CSV file
df.to_csv('audio_dataset.csv', index=False)


