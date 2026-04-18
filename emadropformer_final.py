import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ==========================================================
# DEVICE
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================================================
# CONFIG
# ==========================================================
DATASET_PATH = "/content/RAVDESS"
SR = 16000
MAX_SEC = 4
MAX_LEN = SR * MAX_SEC

BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-4
NUM_CLASSES = 8

# ==========================================================
# LABEL MAP
# ==========================================================
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

label_to_idx = {v:i for i,v in enumerate(emotion_map.values())}
idx_to_label = {i:v for v,i in label_to_idx.items()}

# ==========================================================
# LOAD DATA
# ==========================================================
files = glob.glob(os.path.join(DATASET_PATH, "Actor_*", "*.wav"))

rows = []
for f in files:
    name = os.path.basename(f)
    emotion_code = name.split("-")[2]
    emotion = emotion_map[emotion_code]
    rows.append([f, emotion, label_to_idx[emotion]])

df = pd.DataFrame(rows, columns=["path", "emotion", "label"])

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

print("Train:", len(train_df), " Test:", len(test_df))

# ==========================================================
# LOAD PRETRAINED MODEL
# ==========================================================
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

# Freeze all first
for param in wav2vec.parameters():
    param.requires_grad = False

# Unfreeze last 2 transformer layers
for name, param in wav2vec.named_parameters():
    if "encoder.layers.10" in name or "encoder.layers.11" in name:
        param.requires_grad = True

wav2vec = wav2vec.to(device)

# ==========================================================
# DATASET
# ==========================================================
class RavdessDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx, "path"]
        label = self.df.loc[idx, "label"]

        audio, sr = librosa.load(path, sr=SR)

        if len(audio) > MAX_LEN:
            audio = audio[:MAX_LEN]
        else:
            audio = np.pad(audio, (0, MAX_LEN-len(audio)))

        return audio, label

train_ds = RavdessDataset(train_df)
test_ds = RavdessDataset(test_df)

# ==========================================================
# COLLATE FUNCTION
# ==========================================================
def collate_fn(batch):
    audios = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])

    inputs = processor(
        audios,
        sampling_rate=SR,
        padding=True,
        return_tensors="pt"
    )

    return inputs.input_values, labels

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

# ==========================================================
# EMADropFormer
# ==========================================================
class EMADropFormer(nn.Module):
    def __init__(self, hidden=768, num_classes=8):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=8,
            batch_first=True
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid()
        )

        self.fc1 = nn.Linear(hidden * 2, 256)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x = [B,T,H]

        attn_out, _ = self.attn(x, x, x)

        # emotion gate
        g = self.gate(attn_out)
        x = attn_out * g

        # Mean + Max pooling
        mean_pool = torch.mean(x, dim=1)
        max_pool = torch.max(x, dim=1).values

        x = torch.cat([mean_pool, max_pool], dim=1)

        x = torch.relu(self.fc1(x))
        x = self.drop(x)

        return self.fc2(x)

model = EMADropFormer(num_classes=NUM_CLASSES).to(device)

# ==========================================================
# CLASS WEIGHTS
# ==========================================================
counts = train_df["label"].value_counts().sort_index().values
weights = 1.0 / counts
weights = weights / weights.sum()
weights = torch.tensor(weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)

# train both model + unfrozen wav2vec layers
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(filter(lambda p: p.requires_grad, wav2vec.parameters())),
    lr=LR
)

# ==========================================================
# TRAINING
# ==========================================================
for epoch in range(EPOCHS):

    model.train()
    wav2vec.train()

    losses = []

    for input_values, labels in tqdm(train_loader):

        input_values = input_values.to(device)
        labels = labels.to(device)

        outputs = wav2vec(input_values)
        feats = outputs.last_hidden_state

        preds = model(feats)

        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {np.mean(losses):.4f}")

# ==========================================================
# EVALUATION
# ==========================================================
model.eval()
wav2vec.eval()

all_preds = []
all_true = []

with torch.no_grad():
    for input_values, labels in tqdm(test_loader):

        input_values = input_values.to(device)

        outputs = wav2vec(input_values)
        feats = outputs.last_hidden_state

        preds = model(feats)
        preds = torch.argmax(preds, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_true.extend(labels.numpy())

acc = accuracy_score(all_true, all_preds)
f1 = f1_score(all_true, all_preds, average="weighted")

print("\n==============================")
print("FINAL TEST ACCURACY:", round(acc*100,2), "%")
print("FINAL WEIGHTED F1 :", round(f1*100,2), "%")
print("==============================\n")

print(classification_report(
    all_true,
    all_preds,
    target_names=list(label_to_idx.keys()),
    zero_division=0
))

# ==========================================================
# SAVE
# ==========================================================
torch.save(model.state_dict(), "final_emadropformer_best.pth")
print("Saved final model!")