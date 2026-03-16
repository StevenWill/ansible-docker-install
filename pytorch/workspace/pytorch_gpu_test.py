import sys

import torch
import torch.nn as nn


def main() -> int:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Check Docker GPU runtime and NVIDIA toolkit.")
        return 1

    device_count = torch.cuda.device_count()
    current_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_index)

    print(f"GPU count: {device_count}")
    print(f"Using GPU {current_index}: {device_name}")

    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
    ).to(device)

    inputs = torch.randn(256, 1024, device=device)
    targets = torch.randint(0, 10, (256,), device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Running warm-up inference...")
    with torch.no_grad():
        outputs = model(inputs)
        print(f"Inference output shape: {tuple(outputs.shape)}")

    print("Running one training step on GPU...")
    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()

    print(f"Training step complete. Loss: {loss.item():.6f}")
    print(f"Max GPU memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MiB")
    print("SUCCESS: PyTorch model executed on the NVIDIA GPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
