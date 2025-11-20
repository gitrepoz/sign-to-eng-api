import asyncio
import websockets
import cv2
import numpy as np
import base64
import json
import math
import time
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ======================================================
# 🔧 CONFIGURABLE VARIABLES (edit here, not in code body)
# ======================================================
HOST = "0.0.0.0"
PORT = 8765
INTERVAL_MS = 150  # per-client inference interval
NUM_CLASSES = 11
WEIGHTS_PATH = "/app/model/999_demo_11_model.pt"
CLASS_NAMES = [
    "bye",
    "drink",
    "eat",
    "evening",
    "good",
    "hello",
    "I love you",
    "love <3",
    "morning",
    "sorry",
    "thankyou",
]
CONF_THRESHOLD = 0.3
IMAGE_SIZE = 224
HIDDEN_DIM = 256
NUM_QUERIES = 25

# Optional: reduce CPU thread contention on shared servers
torch.set_num_threads(1)


class SignDETRServer:
    def __init__(self):
        self.host = HOST
        self.port = PORT
        self.interval_ms = INTERVAL_MS
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classes = CLASS_NAMES
        self.conf_thresh = CONF_THRESHOLD

        # Build + load model once for all users
        self.model = self._build_detr(NUM_CLASSES)
        self.model.to(self.device).eval()
        self._load_pretrained(WEIGHTS_PATH)

        self.transforms = A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        # Serialize inference to avoid torch/GPU concurrency issues
        self.infer_sema = asyncio.Semaphore(1)

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

    # ---------------- DETR Backbone ----------------
    def _get_1d_sincos_pos_embed(self, length, dim, temperature=10000.0):
        assert dim % 2 == 0
        position = torch.arange(length, device=self.device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=self.device, dtype=torch.float32)
            * (-math.log(temperature) / dim)
        )
        pe = torch.zeros(length, dim, device=self.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _build_2d_sincos_position_embedding(self, H, W, dim):
        dim_half = dim // 2
        pe_y = self._get_1d_sincos_pos_embed(H, dim_half)
        pe_x = self._get_1d_sincos_pos_embed(W, dim_half)
        pos = torch.zeros(H, W, dim, device=self.device)
        pos[:, :, :dim_half] = pe_y[:, None, :].expand(-1, W, -1)
        pos[:, :, dim_half:] = pe_x[None, :, :].expand(H, -1, -1)
        return pos.view(1, H * W, dim)

    def _build_detr(
        self,
        num_classes,
        hidden_dim=HIDDEN_DIM,
        nheads=8,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_queries=NUM_QUERIES,
    ):
        class DETR(nn.Module):
            def __init__(self, outer):
                super().__init__()
                self.outer = outer
                self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
                self.backbone.fc = nn.Identity()

                self.conv = nn.Conv2d(2048, hidden_dim, 1)
                self.transformer = nn.Transformer(
                    hidden_dim,
                    nheads,
                    num_encoder_layers,
                    num_decoder_layers,
                    batch_first=True,
                    dropout=0.1,
                )
                self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
                self.linear_bbox = nn.Linear(hidden_dim, 4)

                self.num_queries = num_queries
                self.query_pos = nn.Parameter(torch.randn(num_queries, hidden_dim))
                self.norm_src = nn.LayerNorm(hidden_dim)
                self.norm_tgt = nn.LayerNorm(hidden_dim)

            def forward(self, inputs):
                # backbone
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

                src = feat.flatten(2).permute(0, 2, 1)  # (B, HW, C)
                pos = self.outer._build_2d_sincos_position_embedding(Hf, Wf, d_model)
                src = self.norm_src(src + pos)

                tgt = torch.zeros(bsz, self.num_queries, d_model, device=feat.device)
                query_pos = self.query_pos.unsqueeze(0).expand(bsz, -1, -1)
                tgt = self.norm_tgt(tgt + query_pos)

                hs = self.transformer(src=src, tgt=tgt)

                return {
                    "pred_logits": self.linear_class(hs),
                    "pred_boxes": self.linear_bbox(hs).sigmoid(),
                }

        return DETR(self)

    def _load_pretrained(self, path):
        try:
            state = torch.load(path, map_location=self.device)

            # handle common checkpoint formats
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]

            self.model.load_state_dict(state, strict=False)
            print(f"✅ Loaded pretrained weights from {path}")
        except Exception as e:
            print(f"⚠️ Could not load weights: {e}")

    # ---------------- Utils ----------------
    def _rescale_bboxes(self, out_bbox, size):
        w, h = size
        x_c, y_c, bw, bh = out_bbox.unbind(-1)
        x1 = (x_c - 0.5 * bw) * w
        y1 = (y_c - 0.5 * bh) * h
        x2 = (x_c + 0.5 * bw) * w
        y2 = (y_c + 0.5 * bh) * h
        return torch.stack([x1, y1, x2, y2], dim=-1)

    # ---------------- Inference ----------------
    def _detect_objects_sync(self, frame):
        transformed = self.transforms(image=frame)
        img_tensor = transformed["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)

        probs = outputs["pred_logits"].softmax(-1)[..., :-1]
        max_probs, max_classes = probs.max(-1)

        keep = max_probs > self.conf_thresh
        batch_idx, query_idx = torch.where(keep)

        h, w = frame.shape[:2]
        boxes = self._rescale_bboxes(
            outputs["pred_boxes"][batch_idx, query_idx, :], (w, h)
        )

        classes = max_classes[batch_idx, query_idx]
        probas = max_probs[batch_idx, query_idx]

        detections = []
        for c, p, b in zip(classes, probas, boxes):
            detections.append(
                {
                    "class": self.classes[int(c)],
                    "confidence": round(float(p), 3),
                    "bbox": [float(v) for v in b.cpu()],
                }
            )
        return detections

    async def detect_objects(self, frame, timeout_s=2.0):
        # Serialize access to the model + prevent permanent hangs
        async with self.infer_sema:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(self._detect_objects_sync, frame),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                return {"error": "inference_timeout"}
            except Exception as e:
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                return {"error": str(e)}

    # ---------------- WebSocket ----------------
    async def _client_handler(self, websocket):
        print("⚡ Client connected!")

        # Send handshake so client knows server is alive
        try:
            await websocket.send(json.dumps({"status": "ready"}))
        except Exception:
            return

        # PER-CLIENT state (not shared with other users)
        last_pred = []
        last_run_ms = 0.0

        try:
            async for message in websocket:
                # decode incoming frame
                try:
                    data = json.loads(message)
                    img_data = base64.b64decode(data["frame"])
                    nparr = np.frombuffer(img_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except Exception:
                    await websocket.send(json.dumps({"error": "bad_frame"}))
                    continue

                if frame is None:
                    await websocket.send(json.dumps({"error": "bad_frame"}))
                    continue

                now_ms = time.time() * 1000.0

                # Per-client rate limiting
                if now_ms - last_run_ms >= self.interval_ms:
                    last_run_ms = now_ms
                    last_pred = await self.detect_objects(frame)

                # Always respond to keep frontend smooth
                await websocket.send(json.dumps(last_pred))

        except websockets.ConnectionClosed:
            print("❌ Client disconnected.")
        except Exception as e:
            print(f"❌ Client error: {e}")

    async def run(self):
        print(f"🟢 Starting Sign DETR WebSocket server on ws://{self.host}:{self.port}")
        async with websockets.serve(
            self._client_handler,
            self.host,
            self.port,
            max_size=8 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
        ):
            await asyncio.Future()  # run forever


# ======================================================
#                  MAIN ENTRY POINT
# ======================================================
if __name__ == "__main__":
    server = SignDETRServer()
    asyncio.run(server.run())
