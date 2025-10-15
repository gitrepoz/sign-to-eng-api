import asyncio
import websockets
import cv2
import numpy as np
import base64
import json
import tensorflow as tf
import mediapipe as mp
from collections import deque
import time
from tensorflow.keras import layers, models, regularizers 
import os

# ==================== MODEL & ACTION SETUP ====================
actions = np.array(['beautiful', 'bye', 'call', 'love', 'hello'])
MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "/app/model/model_30.h5")
num_classes = len(actions)
threshold = 0.2

def build_model(timesteps=30, features=1662, classes=num_classes):
    inp = layers.Input(shape=(timesteps, features))
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.LayerNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
    x = layers.LayerNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(classes, activation="softmax")(x)
    return models.Model(inp, out)

model = build_model()
model.load_weights(MODEL_WEIGHTS)

# Warm-up
_ = model.predict(np.zeros((1, 30, 1662), dtype=np.float32), verbose=0)

# ==================== MEDIAPIPE SETUP ====================
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0
)

def extract_keypoints(results):
    pose = np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh]).astype(np.float32)

def mediapipe_detection_bgr(frame):
    # Downscale for faster processing
    h, w = frame.shape[:2]
    scale = 0.5
    frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = holistic.process(rgb)
    return results


# ==================== INFERENCE WORKER ====================
class InferenceWorker:
    def __init__(self, interval_ms=300):
        self.sequence = deque(maxlen=30)
        self.latest_pred = {"word": "", "confidence": 0.0}
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

            results = await asyncio.to_thread(mediapipe_detection_bgr, frame)
            keypoints = extract_keypoints(results)
            self.sequence.append(keypoints)

            if len(self.sequence) == 30:
                seq = np.expand_dims(np.asarray(self.sequence, dtype=np.float32), axis=0)
                res = await asyncio.to_thread(model.predict, seq, 0)
                res = res[0]
                idx = int(np.argmax(res))
                word, conf = actions[idx], float(res[idx])

                if conf > threshold:
                    self.latest_pred = {"word": word, "confidence": round(conf, 3)}
                else:
                    self.latest_pred = {"word": f"Low conf: {word}", "confidence": round(conf, 3)}

            self.frame_queue.task_done()

    def stop(self):
        self._stop = True

worker = InferenceWorker(interval_ms=100)

# ==================== WEBSOCKET HANDLER ====================
async def process_frame(websocket):
    worker_task = asyncio.create_task(worker.run())
    try:
        async for message in websocket:
            data = json.loads(message)
            img_data = base64.b64decode(data['frame'])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            await worker.push_frame(frame)
            await websocket.send(json.dumps(worker.latest_pred))
    finally:
        worker.stop()
        await asyncio.gather(worker_task, return_exceptions=True)

# ==================== MAIN ====================
async def main():
    async with websockets.serve(process_frame, "localhost", 8765, max_size=4*1024*1024):
        print("🟢 Sign language translation server running on ws://localhost:8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
