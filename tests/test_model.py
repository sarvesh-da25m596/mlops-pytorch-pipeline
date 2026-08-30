import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import get_model  # noqa: E402


def test_output_shape():
    model = get_model("resnet18", num_classes=10)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 10)


def test_rejects_unknown_architecture():
    try:
        get_model("vgg16")
        assert False, "expected ValueError for unsupported architecture"
    except ValueError:
        pass
