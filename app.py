import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import soundfile as sf
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader

from model_source import AudioRNN

# ========================================================
# 1. GLOBAL SETTINGS & DATA CONFIGURATION
# ========================================================
print("Verifying local Google Speech Commands dataset extraction...")
base_data_path = os.path.join(".", "data", "SpeechCommands", "speech_commands_v0.02")

if not os.path.exists(base_data_path):
    import torchaudio
    os.makedirs("./data", exist_ok=True)
    print("Downloading raw audio files (this may take a few minutes)...")
    torchaudio.datasets.SPEECHCOMMANDS(root="./data", download=True)

LABELS = {"go": 0, "yes": 1, "no": 2}
REVERSE_LABELS = {0: "go", 1: "yes", 2: "no"}

mfcc_transform = T.MFCC(
    sample_rate=16000,
    n_mfcc=20,
    melkwargs={"n_fft": 1024, "hop_length": 160, "n_mels": 40},
)


def wav_to_mfcc(path):
    data, _ = sf.read(path)
    waveform = torch.FloatTensor(data).unsqueeze(0)
    if waveform.shape[1] < 16000:
        waveform = nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
    else:
        waveform = waveform[:, :16000]
    return mfcc_transform(waveform).squeeze(0).t()


# ========================================================
# 2. COMPETITION PIPELINE DATASET
# ========================================================
class AudioTrainDataset(Dataset):
    def __init__(self, base_dir, labels_dict, augment=False, indices=None):
        all_paths = []
        all_targets = []
        for word, label_idx in labels_dict.items():
            word_dir = os.path.join(base_dir, word)
            if os.path.exists(word_dir):
                files = sorted([f for f in os.listdir(word_dir) if f.endswith(".wav")])
                train_files = files[:-50]
                for f in train_files:
                    all_paths.append(os.path.join(word_dir, f))
                    all_targets.append(label_idx)

        if indices is None:
            self.file_paths = all_paths
            self.targets = all_targets
        else:
            self.file_paths = [all_paths[i] for i in indices]
            self.targets = [all_targets[i] for i in indices]
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        mfcc = wav_to_mfcc(self.file_paths[idx])
        if self.augment:
            mfcc = mfcc + torch.randn_like(mfcc) * 1.5
            shift = torch.randint(-5, 6, (1,)).item()
            if shift != 0:
                mfcc = torch.roll(mfcc, shifts=shift, dims=0)
        return mfcc, self.targets[idx]


# ========================================================
# 3. TRAIN / VAL SPLIT (training audio only — never touch held-out 50/class)
# ========================================================
full_size = len(AudioTrainDataset(base_data_path, LABELS))
val_size = int(0.1 * full_size)
train_size = full_size - val_size
generator = torch.Generator().manual_seed(42)
perm = torch.randperm(full_size, generator=generator).tolist()
train_idx = perm[:train_size]
val_idx = perm[train_size:]

train_set = AudioTrainDataset(base_data_path, LABELS, augment=True, indices=train_idx)
val_set = AudioTrainDataset(base_data_path, LABELS, augment=False, indices=val_idx)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=0)

print(f"Training samples: {train_size} | Validation samples: {val_size}")

# ========================================================
# 4. TRAINING LOOP
# ========================================================
model = AudioRNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

epochs = 50
best_val_acc = 0.0
best_state = None

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for features, labels in train_loader:
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for features, labels in val_loader:
            logits = model(features)
            val_correct += (logits.argmax(dim=1) == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total if val_total else 0.0
    scheduler.step(val_acc)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_acc = correct / total if total else 0.0
    lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch + 1}/{epochs} | loss={total_loss / len(train_loader):.4f} "
        f"| train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | lr={lr:.6f}"
    )

model.load_state_dict(best_state)
model.eval()
torch.save(model, "model.pt")
print(f"Saved model.pt (best val_acc={best_val_acc:.4f})")

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")

# ========================================================
# 5. INFERENCE ON PACKAGED LEADERBOARD FILE
# ========================================================
print("\nRunning inference on 'student_test_features.pt'...")

if not os.path.exists("student_test_features.pt"):
    raise FileNotFoundError("Please ensure 'student_test_features.pt' is in this folder!")

X_evaluation = torch.load("student_test_features.pt", map_location="cpu", weights_only=False)

with torch.no_grad():
    logits = model(X_evaluation)
    predictions = logits.argmax(dim=1).tolist()

with open("predictions.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["id", "keyword_class"])
    for idx, pred_idx in enumerate(predictions):
        writer.writerow([idx, REVERSE_LABELS[pred_idx]])

print("predictions.csv successfully generated!")
print("Submit predictions.csv and model_source.py (or model.pt) to the leaderboard portal.")
