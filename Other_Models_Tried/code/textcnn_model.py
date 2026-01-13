"""
TextCNN Model - CNN with GloVe embeddings for country classification.
Achieved ~81% accuracy with pretrained GloVe 300d embeddings.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "code"))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from utils import BASE_DIR, load_and_prepare_data, RANDOM_STATE

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "TextCNN"
GLOVE_PATH = BASE_DIR / "glove" / "glove.6B.300d.txt"

# Model configuration
MAX_LEN = 100
EMBED_DIM = 300
FILTER_SIZES = [2, 3, 4, 5]
NUM_FILTERS = 128
DROPOUT = 0.5
HIDDEN_DIM = 256
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Run: pip install torch")


def load_glove_embeddings(glove_path, vocab, embed_dim=300):
    """Load GloVe embeddings for vocabulary."""
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.rstrip().split(' ')
            word = values[0]
            if word in vocab:
                embeddings[word] = np.array(values[1:], dtype=np.float32)
    
    # Create embedding matrix
    embedding_matrix = np.random.randn(len(vocab), embed_dim).astype(np.float32) * 0.1
    for word, idx in vocab.items():
        if word in embeddings:
            embedding_matrix[idx] = embeddings[word]
    
    return embedding_matrix


def build_vocab(texts, max_vocab=10000):
    """Build vocabulary from texts."""
    from collections import Counter
    word_counts = Counter()
    for text in texts:
        words = str(text).lower().split()
        word_counts.update(words)
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in word_counts.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    
    return vocab


def tokenize(texts, vocab, max_len):
    """Convert texts to padded sequences."""
    sequences = []
    for text in texts:
        words = str(text).lower().split()
        seq = [vocab.get(w, 1) for w in words[:max_len]]
        seq = seq + [0] * (max_len - len(seq))
        sequences.append(seq)
    return np.array(sequences)


if TORCH_AVAILABLE:
    class TextCNN(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_classes, pretrained_embeddings=None):
            super().__init__()
            
            if pretrained_embeddings is not None:
                self.embedding = nn.Embedding.from_pretrained(
                    torch.tensor(pretrained_embeddings), 
                    freeze=False
                )
            else:
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            
            self.convs = nn.ModuleList([
                nn.Conv1d(embed_dim, NUM_FILTERS, k) for k in FILTER_SIZES
            ])
            
            self.hidden = nn.Linear(NUM_FILTERS * len(FILTER_SIZES), HIDDEN_DIM)
            self.dropout = nn.Dropout(DROPOUT)
            self.fc = nn.Linear(HIDDEN_DIM, num_classes)
        
        def forward(self, x):
            x = self.embedding(x).transpose(1, 2)
            conv_outputs = [torch.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
            x = torch.cat(conv_outputs, dim=1)
            x = self.dropout(x)
            x = torch.relu(self.hidden(x))
            x = self.dropout(x)
            return self.fc(x)
    
    class TextDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.long)
            self.y = torch.tensor(y, dtype=torch.long)
        
        def __len__(self):
            return len(self.y)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]


def train_textcnn():
    """Train TextCNN with GloVe embeddings."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Skipping TextCNN.")
        return None
    
    print("=" * 60)
    print("TextCNN WITH GLOVE EMBEDDINGS")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    df = load_and_prepare_data()
    labels = sorted(df["Country"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    
    X = df["Description"].values
    y = df["Country"].map(label2id).values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Build vocabulary
    vocab = build_vocab(X_train)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Tokenize
    X_train_seq = tokenize(X_train, vocab, MAX_LEN)
    X_test_seq = tokenize(X_test, vocab, MAX_LEN)
    
    # Load GloVe
    if GLOVE_PATH.exists():
        print("Loading GloVe embeddings...")
        embeddings = load_glove_embeddings(GLOVE_PATH, vocab, EMBED_DIM)
        print(f"Loaded embeddings for {len(embeddings)} words")
    else:
        print("GloVe not found, using random embeddings")
        embeddings = None
    
    # Create model
    model = TextCNN(len(vocab), EMBED_DIM, len(labels), embeddings).to(device)
    
    # Training setup
    train_loader = DataLoader(TextDataset(X_train_seq, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TextDataset(X_test_seq, y_test), batch_size=BATCH_SIZE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    print("\nTraining...")
    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        if acc > best_acc:
            best_acc = acc
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} - Acc: {acc:.4f}")
    
    # Final evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    
    # Save results
    report = classification_report(all_labels, all_preds, target_names=labels, digits=4)
    with open(OUTPUT_DIR / "report_textcnn.txt", "w", encoding="utf-8") as f:
        f.write("TextCNN RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Test Macro F1: {macro_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    
    return accuracy, macro_f1


if __name__ == "__main__":
    train_textcnn()
