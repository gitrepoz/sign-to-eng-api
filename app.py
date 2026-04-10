import asyncio
import websockets
import cv2
import numpy as np
import base64
import json
import time
import os
from collections import deque

import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers

# ======================================================
# 🔧 CONFIGURABLE VARIABLES (edit here, not in code body)
# ======================================================
import os

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8765"))
INTERVAL_MS = int(os.getenv("INTERVAL_MS", "100"))
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/3_words_AB.keras")

# IMPORTANT:
# Keep CLASS_NAMES in the exact same order used during training.
CLASS_NAMES = [
    "hello",
    "thanks",
    "iloveyou",
]

CONF_THRESHOLD = 0.80
MAX_LEN = 48
PAD_VALUE = 0.0
SMOOTHING_ALPHA = 0.65
CONSENSUS_FRAMES = 3
MIN_FRAMES_BEFORE_PRED = 4

# MediaPipe landmark subsets
POSE_IDX = [0, 11, 12, 13, 14, 15, 16]
FACE_IDX = [33, 263, 61, 291, 13, 14, 78, 308]
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12

mp_holistic = mp.solutions.holistic


# ======================================================
#                 MODEL / FEATURE HELPERS
# ======================================================
class AttentionPooling1D(layers.Layer):
    def __init__(self, units=128, **kwargs):
        super().__init__(**kwargs)
        self.proj = layers.Dense(units, activation="tanh")
        self.score = layers.Dense(1)

    def call(self, inputs, mask=None):
        logits = self.score(self.proj(inputs))
        if mask is not None:
            mask = tf.cast(mask[:, :, tf.newaxis], tf.float32)
            logits = logits + (1.0 - mask) * (-1e9)
        weights = tf.nn.softmax(logits, axis=1)
        return tf.reduce_sum(inputs * weights, axis=1)

    def compute_mask(self, inputs, mask=None):
        return None


def infer_actions_from_data(data_path):
    if not os.path.isdir(data_path):
        return []
    actions = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    return sorted(actions)


def extract_selected_keypoints(results):
    pose = np.zeros((len(POSE_IDX), 4), dtype=np.float32)
    face = np.zeros((len(FACE_IDX), 3), dtype=np.float32)
    lhand = np.zeros((21, 3), dtype=np.float32)
    rhand = np.zeros((21, 3), dtype=np.float32)

    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark
        for j, idx in enumerate(POSE_IDX):
            lm = pose_landmarks[idx]
            pose[j] = [lm.x, lm.y, lm.z, lm.visibility]

    if results.face_landmarks:
        face_landmarks = results.face_landmarks.landmark
        for j, idx in enumerate(FACE_IDX):
            lm = face_landmarks[idx]
            face[j] = [lm.x, lm.y, lm.z]

    if results.left_hand_landmarks:
        for j, lm in enumerate(results.left_hand_landmarks.landmark):
            lhand[j] = [lm.x, lm.y, lm.z]

    if results.right_hand_landmarks:
        for j, lm in enumerate(results.right_hand_landmarks.landmark):
            rhand[j] = [lm.x, lm.y, lm.z]

    return np.concatenate([
        pose.flatten(),
        face.flatten(),
        lhand.flatten(),
        rhand.flatten()
    ]).astype(np.float32)


def raw1662_to_compact(raw):
    raw = np.asarray(raw, dtype=np.float32).flatten()

    compact_dim = len(POSE_IDX) * 4 + len(FACE_IDX) * 3 + 21 * 3 + 21 * 3
    if raw.shape[0] == compact_dim:
        return raw.astype(np.float32)

    if raw.shape[0] != 1662:
        raise ValueError(
            f"Unexpected feature dimension {raw.shape[0]}. Expected 1662 or {compact_dim}."
        )

    pose = raw[:132].reshape(33, 4)
    face = raw[132:132 + 1404].reshape(468, 3)
    lhand = raw[132 + 1404:132 + 1404 + 63].reshape(21, 3)
    rhand = raw[132 + 1404 + 63:].reshape(21, 3)

    pose_sel = pose[POSE_IDX].reshape(-1)
    face_sel = face[FACE_IDX].reshape(-1)

    compact = np.concatenate([
        pose_sel,
        face_sel,
        lhand.reshape(-1),
        rhand.reshape(-1)
    ]).astype(np.float32)

    return compact


def _split_compact_frame(frame):
    pose_size = len(POSE_IDX) * 4
    face_size = len(FACE_IDX) * 3
    hand_size = 21 * 3

    pose = frame[:pose_size].reshape(len(POSE_IDX), 4)
    face = frame[pose_size:pose_size + face_size].reshape(len(FACE_IDX), 3)
    lhand = frame[pose_size + face_size:pose_size + face_size + hand_size].reshape(21, 3)
    rhand = frame[pose_size + face_size + hand_size:].reshape(21, 3)
    return pose, face, lhand, rhand


def normalize_compact_sequence(sequence):
    sequence = np.asarray(sequence, dtype=np.float32)
    out = []

    left_idx = POSE_IDX.index(LEFT_SHOULDER_IDX) if LEFT_SHOULDER_IDX in POSE_IDX else None
    right_idx = POSE_IDX.index(RIGHT_SHOULDER_IDX) if RIGHT_SHOULDER_IDX in POSE_IDX else None

    for frame in sequence:
        pose, face, lhand, rhand = _split_compact_frame(frame)
        origin = np.zeros(3, dtype=np.float32)
        scale = 1.0

        if left_idx is not None and right_idx is not None:
            left_shoulder = pose[left_idx, :3]
            right_shoulder = pose[right_idx, :3]

            if np.any(left_shoulder) and np.any(right_shoulder):
                origin = 0.5 * (left_shoulder + right_shoulder)
                scale = float(np.linalg.norm(left_shoulder - right_shoulder))
                if scale < 1e-5:
                    scale = 1.0

        pose_xyz = (pose[:, :3] - origin) / scale
        pose_vis = pose[:, 3:4]
        face_xyz = (face - origin) / scale
        lhand_xyz = (lhand - origin) / scale
        rhand_xyz = (rhand - origin) / scale

        norm_frame = np.concatenate([
            np.concatenate([pose_xyz, pose_vis], axis=1).reshape(-1),
            face_xyz.reshape(-1),
            lhand_xyz.reshape(-1),
            rhand_xyz.reshape(-1),
        ]).astype(np.float32)

        out.append(norm_frame)

    return np.stack(out, axis=0)


def add_temporal_deltas(sequence):
    deltas = np.diff(sequence, axis=0, prepend=sequence[:1])
    return np.concatenate([sequence, deltas], axis=-1).astype(np.float32)


def pad_or_truncate(sequence, max_len=MAX_LEN):
    seq = np.asarray(sequence, dtype=np.float32)
    if len(seq) >= max_len:
        return seq[:max_len]

    pad = np.full((max_len - len(seq), seq.shape[1]), PAD_VALUE, dtype=np.float32)
    return np.concatenate([seq, pad], axis=0)


def preprocess_sequence(frames, max_len=MAX_LEN):
    compact = np.stack([raw1662_to_compact(f) for f in frames], axis=0)
    compact = normalize_compact_sequence(compact)
    compact = add_temporal_deltas(compact)
    compact = pad_or_truncate(compact, max_len=max_len)
    return compact.astype(np.float32)


# ======================================================
#                     SERVER CLASS
# ======================================================
class SignDETRServer:
    def __init__(self):
        self.host = HOST
        self.port = PORT
        self.interval_ms = INTERVAL_MS
        self.classes = CLASS_NAMES if CLASS_NAMES else infer_actions_from_data(DATA_PATH)
        self.conf_thresh = CONF_THRESHOLD
        self.max_len = MAX_LEN
        self.smoothing_alpha = SMOOTHING_ALPHA
        self.consensus_frames = CONSENSUS_FRAMES
        self.min_frames_before_pred = MIN_FRAMES_BEFORE_PRED

        # Build + load model once for all users
        self.model = None
        self._load_pretrained(MODEL_PATH)

        # Serialize inference to avoid TF concurrency issues
        self.infer_sema = asyncio.Semaphore(1)

    # ---------------- Model Loading ----------------
    def _load_pretrained(self, path):
        try:
            self.model = tf.keras.models.load_model(
                path,
                custom_objects={"AttentionPooling1D": AttentionPooling1D}
            )
            print(f"✅ Loaded pretrained weights from {path}")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            raise

    # ---------------- Utils ----------------
    def _decode_frame(self, message):
        try:
            data = json.loads(message)
            img_data = base64.b64decode(data["frame"])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

    def _extract_keypoints(self, frame, holistic):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        return extract_selected_keypoints(results)

    def _build_warmup_response(self, state):
        return {
            "type": "prediction",
            "ready": False,
            "current": "warming_up",
            "confidence": 0.0,
            "sentence": state["sentence"],
            "sentence_text": " ".join(state["sentence"]),
            "frames_seen": len(state["raw_window"]),
            "frames_needed": max(0, self.min_frames_before_pred - len(state["raw_window"])),
            "top": []
        }

    def _build_prediction_response(self, probs, label, confidence, state):
        top_indices = np.argsort(probs)[::-1][:min(5, len(self.classes))]
        top = []
        for i in top_indices:
            top.append({
                "label": self.classes[int(i)],
                "probability": round(float(probs[int(i)]), 4)
            })

        return {
            "type": "prediction",
            "ready": True,
            "current": label,
            "confidence": round(float(confidence), 4),
            "sentence": state["sentence"],
            "sentence_text": " ".join(state["sentence"]),
            "frames_seen": len(state["raw_window"]),
            "frames_needed": 0,
            "top": top
        }

    # ---------------- Inference ----------------
    def _predict_sync(self, sequence_np):
        probs = self.model.predict(np.expand_dims(sequence_np, axis=0), verbose=0)[0]
        return np.asarray(probs, dtype=np.float32)

    async def detect_objects(self, client_state, timeout_s=3.0):
        # Serialize access to the model + prevent permanent hangs
        async with self.infer_sema:
            try:
                if len(client_state["raw_window"]) < self.min_frames_before_pred:
                    return self._build_warmup_response(client_state)

                window_np = preprocess_sequence(
                    list(client_state["raw_window"]),
                    max_len=self.max_len
                )

                probs = await asyncio.wait_for(
                    asyncio.to_thread(self._predict_sync, window_np),
                    timeout=timeout_s,
                )

                if client_state["smoothed_probs"] is None:
                    client_state["smoothed_probs"] = probs
                else:
                    client_state["smoothed_probs"] = (
                        self.smoothing_alpha * client_state["smoothed_probs"]
                        + (1.0 - self.smoothing_alpha) * probs
                    )

                smoothed_probs = client_state["smoothed_probs"]

                best_idx = int(np.argmax(smoothed_probs))
                best_prob = float(smoothed_probs[best_idx])
                label = self.classes[best_idx] if best_prob >= self.conf_thresh else "unknown"

                client_state["recent_labels"].append(label)

                if (
                    len(client_state["recent_labels"]) == self.consensus_frames
                    and len(set(client_state["recent_labels"])) == 1
                ):
                    stable_label = client_state["recent_labels"][-1]
                    if stable_label != "unknown":
                        if not client_state["sentence"] or stable_label != client_state["sentence"][-1]:
                            client_state["sentence"].append(stable_label)
                            client_state["sentence"] = client_state["sentence"][-5:]

                return self._build_prediction_response(
                    smoothed_probs,
                    label,
                    best_prob,
                    client_state
                )

            except asyncio.TimeoutError:
                return {"type": "error", "error": "inference_timeout"}
            except Exception as e:
                return {"type": "error", "error": str(e)}

    # ---------------- WebSocket ----------------
    async def _client_handler(self, websocket):
        print("⚡ Client connected!")

        # Send handshake so client knows server is alive
        try:
            await websocket.send(json.dumps({
                "status": "ready",
                "type": "ready",
                "actions": self.classes,
                "max_len": self.max_len,
                "threshold": self.conf_thresh,
                "consensus_frames": self.consensus_frames
            }))
        except Exception:
            return

        # PER-CLIENT state (not shared with other users)
        last_pred = {
            "type": "prediction",
            "ready": False,
            "current": "waiting",
            "confidence": 0.0,
            "sentence": [],
            "sentence_text": "",
            "frames_seen": 0,
            "frames_needed": self.min_frames_before_pred,
            "top": []
        }
        last_run_ms = 0.0

        client_state = {
            "raw_window": deque(maxlen=self.max_len),
            "recent_labels": deque(maxlen=self.consensus_frames),
            "sentence": [],
            "smoothed_probs": None,
        }

        with mp_holistic.Holistic(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        ) as holistic:
            try:
                async for message in websocket:
                    # decode incoming frame
                    frame = self._decode_frame(message)
                    if frame is None:
                        await websocket.send(json.dumps({"type": "error", "error": "bad_frame"}))
                        continue

                    # extract keypoints for every frame so sequence keeps updating
                    try:
                        keypoints = self._extract_keypoints(frame, holistic)
                        client_state["raw_window"].append(keypoints)
                    except Exception as e:
                        await websocket.send(json.dumps({"type": "error", "error": f"mediapipe_failed: {e}"}))
                        continue

                    now_ms = time.time() * 1000.0

                    # Per-client rate limiting
                    if now_ms - last_run_ms >= self.interval_ms:
                        last_run_ms = now_ms
                        last_pred = await self.detect_objects(client_state)

                    # Always respond to keep frontend smooth
                    await websocket.send(json.dumps(last_pred))

            except websockets.ConnectionClosed:
                print("❌ Client disconnected.")
            except Exception as e:
                print(f"❌ Client error: {e}")

    async def run(self):
        print(f"🟢 Starting Sign Detection WebSocket server on ws://{self.host}:{self.port}")
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