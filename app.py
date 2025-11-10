import asyncio
import websockets
import cv2
import numpy as np
import base64
import json
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
import albumentations as A
from albumentations.pytorch import ToTensorV2
import math
import time
import os

# ============================================================
#                 DETR MODEL AND HELPERS
# ============================================================

def _get_1d_sincos_pos_embed(length: int, dim: int, temperature: float = 10000.0, device=None):
    """1D sine-cosine positional encoding."""
    assert dim % 2 == 0
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(temperature) / dim)
    )
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def build_2d_sincos_position_embedding(height: int, width: int, dim: int, device=None):
    """2D sine-cosine positional encoding of shape (1, H*W, dim)."""
    assert dim % 2 == 0
    dim_half = dim // 2
    pe_y = _get_1d_sincos_pos_embed(height, dim_half, device=device)
    pe_x = _get_1d_sincos_pos_embed(width, dim_half, device=device)
    pos = torch.zeros(height, width, dim, device=device, dtype=torch.float32)
    pos[:, :, :dim_half] = pe_y[:, None, :].expand(-1, width, -1)
    pos[:, :, dim_half:] = pe_x[None, :, :].expand(height, -1, -1)
    pos = pos.view(1, height * width, dim)
    return pos


class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=1, num_decoder_layers=1, num_queries=25):
        super().__init__()

        # Backbone: ResNet-50
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()

        # Convert feature maps to transformer hidden dimension
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # Transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers,
            batch_first=True, dropout=0.1
        )

        # Heads
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # Object queries
        self.num_queries = num_queries
        self.query_pos = nn.Parameter(torch.randn(self.num_queries, hidden_dim))

        # Normalization layers
        self.norm_src = nn.LayerNorm(hidden_dim)
        self.norm_tgt = nn.LayerNorm(hidden_dim)

    def forward(self, inputs):
        # Feature extraction
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        feat = self.conv(x)
        bsz, d_model, Hf, Wf = feat.shape

        src = feat.flatten(2).permute(0, 2, 1)
        pos = build_2d_sincos_position_embedding(Hf, Wf, d_model, device=feat.device)
        src = self.norm_src(src + pos)

        tgt = torch.zeros(bsz, self.num_queries, d_model, device=feat.device)
        query_pos = self.query_pos.unsqueeze(0).expand(bsz, -1, -1)
        tgt = self.norm_tgt(tgt + query_pos)

        hs = self.transformer(src=src, tgt=tgt)

        return {
            'pred_logits': self.linear_class(hs),
            'pred_boxes': self.linear_bbox(hs).sigmoid()
        }

    def load_pretrained(self, checkpoint_path: str, device='cpu'):
        """Safely load pretrained weights."""
        try:
            state = torch.load(checkpoint_path, map_location=device)
            self.load_state_dict(state)
            print(f"✅ Loaded pretrained model from {checkpoint_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")


# ============================================================
#                 BBOX RESCALING UTIL
# ============================================================

def rescale_bboxes(out_bbox, size):
    """Convert model output boxes (cx, cy, w, h) in [0,1] to (x1, y1, x2, y2) in image size."""
    img_w, img_h = size
    b = out_bbox
    x_c, y_c, w, h = b.unbind(-1)
    x1 = (x_c - 0.5 * w) * img_w
    y1 = (y_c - 0.5 * h) * img_h
    x2 = (x_c + 0.5 * w) * img_w
    y2 = (y_c + 0.5 * h) * img_h
    return torch.stack([x1, y1, x2, y2], dim=-1)


# ============================================================
#                 MODEL INITIALIZATION
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_WEIGHTS = "/app/model/4426_model.pt"

print(f"🔧 Loading DETR model on {DEVICE} from {MODEL_WEIGHTS}...")
model = DETR(num_classes=3)
model.eval()
model.load_pretrained(MODEL_WEIGHTS, device=DEVICE)
model.to(DEVICE)

CLASSES = ['hello', 'iloveyou', 'thankyou']

# Albumentations preprocessing
transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


# ============================================================
#                 INFERENCE WORKER
# ============================================================

class InferenceWorker:
    def __init__(self, interval_ms=150):
        self.latest_pred = []
        self.interval_ms = interval_ms
        self.frame_queue = asyncio.Queue(maxsize=1)
        self._stop = False

    async def push_frame(self, frame_bgr):
        if self.frame_queue.full():
            try:
                _ = self.frame_queue.get_nowait()
                self.frame_queue.task_done()
            except asyncio.QueueEmpty:
                pass
        await self.frame_queue.put(frame_bgr)

    async def run(self):
        last_run = 0
        while not self._stop:
            frame = await self.frame_queue.get()
            now = time.time() * 1000
            if now - last_run < self.interval_ms:
                self.frame_queue.task_done()
                continue
            last_run = now

            detections = await asyncio.to_thread(self.detect_objects, frame)
            self.latest_pred = detections
            print("Detections:", detections)

            self.frame_queue.task_done()

    def detect_objects(self, frame):
        transformed = transforms(image=frame)
        img_tensor = transformed['image'].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_tensor)

        probabilities = outputs['pred_logits'].softmax(-1)[..., :-1]
        max_probs, max_classes = probabilities.max(-1)
        keep_mask = max_probs > 0.3

        batch_indices, query_indices = torch.where(keep_mask)
        h, w = frame.shape[:2]
        boxes = rescale_bboxes(outputs['pred_boxes'][batch_indices, query_indices, :], (w, h))
        classes = max_classes[batch_indices, query_indices]
        probas = max_probs[batch_indices, query_indices]

        detections = []
        for bclass, bprob, bbox in zip(classes, probas, boxes):
            bclass_idx = int(bclass.cpu().numpy())
            bprob_val = float(bprob.cpu().numpy())
            x1, y1, x2, y2 = bbox.cpu().numpy().tolist()
            detections.append({
                'class': CLASSES[bclass_idx],
                'confidence': round(bprob_val, 3),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })
        return detections

    def stop(self):
        self._stop = True


# ============================================================
#                 WEBSOCKET SERVER
# ============================================================

worker = InferenceWorker(interval_ms=150)

async def process_frame(websocket):
    worker_task = asyncio.create_task(worker.run())
    print("⚡ Client connected!")
    try:
        async for message in websocket:
            data = json.loads(message)
            img_data = base64.b64decode(data['frame'])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                print("❌ Invalid frame received")
                continue

            print("✅ Frame received:", frame.shape)
            await worker.push_frame(frame)
            await websocket.send(json.dumps(worker.latest_pred))

    except websockets.ConnectionClosed:
        print("❌ Client disconnected.")
    finally:
        worker.stop()
        await asyncio.gather(worker_task, return_exceptions=True)


# ============================================================
#                 MAIN ENTRYPOINT
# ============================================================

async def main():
    async with websockets.serve(process_frame, "0.0.0.0", 8765, max_size=8*1024*1024):
        print("🟢 DETR object detection server running on ws://0.0.0.0:8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
