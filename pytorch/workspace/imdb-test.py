import argparse
import random
import re
import sys
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset


IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
TOKEN_RE = re.compile(r"[A-Za-z']+")
PAD_IDX = 0
UNK_IDX = 1


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in tar.getmembers():
        member_path = (destination / member.name).resolve()
        if not str(member_path).startswith(str(destination)):
            raise RuntimeError(f"Unsafe archive member: {member.name}")
    tar.extractall(destination)


def download_imdb(root: Path) -> Path:
    archive = root / "aclImdb_v1.tar.gz"
    dataset = root / "aclImdb"
    if dataset.exists():
        return dataset
    root.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        print(f"Downloading IMDB dataset to {archive}...")
        urllib.request.urlretrieve(IMDB_URL, archive)
    print(f"Extracting IMDB dataset into {root}...")
    with tarfile.open(archive, "r:gz") as tar:
        safe_extract(tar, root)
    return dataset


class IMDBDataset(Dataset):
    def __init__(self, root: Path, split: str, max_samples: Optional[int], seed: int) -> None:
        samples: List[Tuple[Path, int]] = []
        for label_name, label in (("neg", 0), ("pos", 1)):
            for path in (root / split / label_name).glob("*.txt"):
                samples.append((path, label))
        random.Random(seed).shuffle(samples)
        self.samples = samples[:max_samples] if max_samples else samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[str, int]:
        path, label = self.samples[index]
        return path.read_text(encoding="utf-8"), label


def build_vocab(dataset: Iterable[Tuple[str, int]], max_vocab_size: int, min_freq: int) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for text, _ in dataset:
        counter.update(tokenize(text))
    vocab = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}
    for token, count in counter.most_common(max_vocab_size - len(vocab)):
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def vectorize(text: str, vocab: Dict[str, int], max_length: Optional[int]) -> List[int]:
    ids = [vocab.get(token, UNK_IDX) for token in tokenize(text)] or [UNK_IDX]
    return ids[:max_length] if max_length else ids


def make_collate_fn(vocab: Dict[str, int], max_length: Optional[int]):
    def collate_fn(batch: Sequence[Tuple[str, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequences = []
        lengths = []
        labels = []
        for text, label in batch:
            token_ids = torch.tensor(vectorize(text, vocab, max_length), dtype=torch.long)
            sequences.append(token_ids)
            lengths.append(token_ids.numel())
            labels.append(label)
        return (
            pad_sequence(sequences, batch_first=True, padding_value=PAD_IDX),
            torch.tensor(lengths, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    return collate_fn


class IMDBClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_dim * 2, 2)

    def forward(self, text: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(text))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.encoder(packed)
        features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.output(self.dropout(features))


def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for text, lengths, labels in loader:
        text, lengths, labels = text.to(device), lengths.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(text, lengths), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for text, lengths, labels in loader:
        text, lengths, labels = text.to(device), lengths.to(device), labels.to(device)
        predictions = model(text, lengths).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    return correct / total if total else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a stronger PyTorch IMDB sentiment classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("/workspace/data"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-vocab-size", type=int, default=20000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=4000)
    parser.add_argument("--max-test-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    data_root = download_imdb(args.data_dir)
    train_data = IMDBDataset(data_root, "train", args.max_train_samples, args.seed)
    test_data = IMDBDataset(data_root, "test", args.max_test_samples, args.seed)
    vocab = build_vocab(train_data, args.max_vocab_size, args.min_freq)
    collate_fn = make_collate_fn(vocab, args.max_length)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IMDBClassifier(len(vocab), args.embed_dim, args.hidden_dim, args.num_layers, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Using device: {device}")
    print(f"Train={len(train_data)} Test={len(test_data)} Vocab={len(vocab)} MaxLength={args.max_length}")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}: loss={loss:.4f} test_accuracy={acc:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
