"""Inference API for the CIFAR-10 classifier trained by train.py."""
import io
import os

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from model import get_model

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt")
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

app = FastAPI(title="cifar10-classifier")
model = None

_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
])


@app.on_event("startup")
def load_model() -> None:
    global model
    if not os.path.exists(CHECKPOINT_PATH):
        return  # let the pod start; readiness probe will keep failing until checkpoint is mounted
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    net = get_model("resnet18", num_classes=len(CLASSES))
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    model = net


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    raw = await image.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    tensor = _transform(img).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze(0)

    return {CLASSES[i]: round(probs[i].item(), 4) for i in range(len(CLASSES))}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
