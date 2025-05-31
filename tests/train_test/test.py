import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from xmixers.modules.token_mixers.deep_memory.deep_memory_unit import DeepMemoryUnit


def create_test_training_loop():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model parameters
    embed_dim = 256
    num_heads = 8
    batch_size = 4
    seq_len = 256
    num_epochs = 100

    # Initialize model
    model = DeepMemoryUnit(
        embed_dim=embed_dim,
        num_heads=num_heads,
    ).to(device)

    print(model)

    # Create synthetic data
    # Input data: random tensors
    X = torch.randn(batch_size, seq_len, embed_dim)
    # Target data: for testing, we can use the same data as target (autoencoding task)
    # or create some transformed data
    Y = X + 0.1 * torch.randn_like(X)  # Add some noise as target

    # Create dataset and dataloader
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    model.train()
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs, _ = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping (optional)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update parameters
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Print progress
            if batch_idx % 5 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}], Loss: {loss.item():.6f}"
                )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_loss:.6f}")
        print("-" * 50)

    print("Training completed!")

    # Evaluation mode
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, seq_len, embed_dim).to(device)
        test_output, _ = model(test_input)
        print(f"Test input shape: {test_input.shape}")
        print(f"Test output shape: {test_output.shape}")


def test_single_forward_pass():
    """Test single forward pass"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embed_dim = 256
    num_heads = 8
    seq_len = 256

    model = DeepMemoryUnit(
        embed_dim=embed_dim,
        num_heads=num_heads,
    ).to(device)

    print(model)

    # Test input
    x = torch.randn(2, seq_len, embed_dim).to(device)

    model.eval()
    with torch.no_grad():
        output, _ = model(x)
        print(f"✓ Forward pass successful!")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    print("Testing single forward pass...")
    test_single_forward_pass()
    print("\n" + "=" * 50 + "\n")
    print("Starting training loop...")
    create_test_training_loop()
