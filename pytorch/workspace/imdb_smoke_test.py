import importlib.util
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def load_imdb_module():
    module_path = Path(__file__).with_name("imdb-test.py")
    spec = importlib.util.spec_from_file_location("imdb_test_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    imdb = load_imdb_module()
    samples = [
        ("a wonderful movie with great acting", 1),
        ("boring slow plot and bad dialogue", 0),
        ("fun charming story and lovable cast", 1),
        ("terrible waste of time and money", 0),
    ]

    vocab = imdb.build_vocab(samples, max_vocab_size=64, min_freq=1)
    collate_fn = imdb.make_collate_fn(vocab, max_length=12)
    loader = DataLoader(samples, batch_size=2, shuffle=False, collate_fn=collate_fn)
    text, lengths, labels = next(iter(loader))
    assert text.shape[0] == 2
    assert lengths.shape[0] == 2
    assert labels.tolist() == [1, 0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = imdb.IMDBClassifier(len(vocab), embed_dim=16, hidden_dim=8, num_layers=1, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    loss = imdb.train_epoch(model, loader, criterion, optimizer, device)
    accuracy = imdb.evaluate(model, loader, device)
    assert loss >= 0.0
    assert 0.0 <= accuracy <= 1.0

    batch_logits = model(text.to(device), lengths.to(device))
    assert batch_logits.shape == (2, 2)
    print(f"Smoke test passed on {device}: loss={loss:.4f}, accuracy={accuracy:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())