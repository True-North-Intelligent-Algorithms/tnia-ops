"""What the torch-based ops share."""


def device():
    """Where to run a network: "cuda" if there is a GPU, else "cpu"."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
