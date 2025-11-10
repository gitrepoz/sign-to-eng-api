import asyncio
import websockets
import cv2
import numpy as np
import base64
import json
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.boxes import rescale_bboxes
from utils.model import DETR
import time
import os

# ==================== MODEL SETUP ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "/app/model/4426_model.pt")

print(f"🔧 Loading DETR model on {DEVICE} from {MODEL_WEIGHTS}...")
model = DETR(num_classes=3)
model.eval()
model.load_pretrained(MODEL_WEIGHTS)
model.to(DEVICE)

CLASSES = CLASSES = ['hello', 'iloveyou','thankyou',]

# Albumentations transform
transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ==================== DETECTION WORKER ====================
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
        keep_mask = max_probs > 0.3  # lower threshold for web demo

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


worker = InferenceWorker(interval_ms=150)

# ==================== WEBSOCKET HANDLER ====================
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


# ==================== MAIN ====================
async def main():
    async with websockets.serve(process_frame, "0.0.0.0", 8765, max_size=8*1024*1024):
        print("🟢 DETR object detection server running on ws://0.0.0.0:8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
