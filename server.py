import asyncio
import websockets
import cv2
import numpy as np
import base64
import json
import tensorflow as tf
import mediapipe as mp
import os
from tensorflow.keras import layers, regularizers

# ==== labels ====
actions = np.array(['hello', 'call', 'sorry', 'bye', 'love'])
threshold = 0.4

# === Model  ===
MODEL_WEIGHTS = r'D:\SL_Frontend\NewSL\sign-to-eng-api\model\updated15words.h5'

num_classes = actions.shape[0]

# ================== Model ==================
inp = tf.keras.layers.Input(shape=(45, 1662))
x = tf.keras.layers.Masking(mask_value=0.0)(inp)
x = tf.keras.layers.LayerNormalization()(x)

x = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
)(x)
x = tf.keras.layers.LayerNormalization()(x)

x = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
)(x)

avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
x = tf.keras.layers.Concatenate()([avg_pool, max_pool])

x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-5))(x)
x = tf.keras.layers.LayerNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)

out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inp, out)


model.load_weights(MODEL_WEIGHTS)

mp_holistic = mp.solutions.holistic
sequence = []
sentence = []

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def mediapipe_detection(image, model_):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model_.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

async def process_frame(websocket):
    global sequence, sentence
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    try:
        async for message in websocket:
            data = json.loads(message)
            img_data = base64.b64decode(data['frame'])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            _, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            output = ""
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                prediction = actions[np.argmax(res)]
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if prediction != sentence[-1]:
                            sentence.append(prediction)
                    else:
                        sentence.append(prediction)
                    if len(sentence) > 5:
                        sentence = sentence[-5:]
                    output = " ".join(sentence)
                else:
                    output = ""
            else:
                output = ""

            await websocket.send(json.dumps({'translation': output}))
    finally:
        holistic.close()

async def main():
    async with websockets.serve(process_frame, "0.0.0.0", 8765):
        print("Sign language translation server started on ws://0.0.0.0:8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
