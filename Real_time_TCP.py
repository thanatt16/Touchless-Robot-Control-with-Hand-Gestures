import cv2
import pickle
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import json
import time
import warnings
import socket
import threading
from sklearn.exceptions import InconsistentVersionWarning

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

CONFIDENCE_THRESHOLD = 0.75

# Nastavení TCP serveru
HOST = '0.0.0.0'
PORT = 65432

client_conn = None

def start_server():
    global client_conn
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()

        print(f"[BĚŽÍ] TCP server naslouchá na {HOST}:{PORT}")
        print("Čekám na klienta...")

        client_conn, addr = s.accept()
        print(f"[PŘIPOJEN] Klient: {addr}")

    except Exception as e:
        print(f"Chyba: {e}")

server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()

# Načtení modelu
try:
    model_dict = pickle.load(open(r"RF_model_vsb.p", 'rb'))
    model = model_dict['RF_model_vsb']
    scaler = model_dict.get('scaler', None)
    print("Model a scaler načten.")

except FileNotFoundError:
    print("Model nebyl nalezen.")
    exit()

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False,min_detection_confidence=0.5,min_tracking_confidence=0.5,max_num_hands=1)

labels_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4'}

# Kamera
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

try:
    pipeline.start(config)
    print("RealSense spuštěn.")
except Exception as e:
    print("Nelze spustit RealSense:", e)
    exit()

print("Detekce běží. 'Q' ukonči")

def gesture_to_json(predicted_character, confidence, hand_landmarks, depth_frame, W, H):
    return {
        "Gesto": int(predicted_character)
    }

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        text_color = (0, 0, 255)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS,mp_styles.get_default_hand_landmarks_style(),mp_styles.get_default_hand_connections_style())
                data_aux = []
                x_, y_ = [], []

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                min_x, min_y = min(x_), min(y_)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)

                if len(data_aux) == 42:
                    input_data = np.asarray(data_aux).reshape(1, -1)

                    if scaler:
                        input_data = scaler.transform(input_data)

                    probas = model.predict_proba(input_data)[0]
                    max_prob = np.max(probas)
                    pred_index = np.argmax(probas)

                    if max_prob > CONFIDENCE_THRESHOLD:
                        predicted_character = labels_dict.get(pred_index, "?")
                        display_text = predicted_character
                        text_color = (0, 255, 0)

                        x1, y1 = int(min_x * W), int(min_y * H)
                        x2, y2 = int(max(x_) * W), int(max(y_) * H)

                        cv2.rectangle(frame,(x1 - 20, y1 - 20),(x2 + 40, y2 + 20),text_color,4)

                        cv2.putText(frame,f"{display_text} ({max_prob * 100:.1f}%)",(x1 + 40, y1 - 40),cv2.FONT_HERSHEY_SIMPLEX,0.8,text_color,2,cv2.LINE_AA)

                        gesture_json = gesture_to_json(predicted_character,max_prob,hand_landmarks,depth_frame,W,H)

                        json_ = json.dumps(gesture_json, ensure_ascii=False)

                        print(predicted_character)

                        # Odeslání TCP klientovi
                        if client_conn is not None:
                            try:
                                client_conn.sendall((json_ + "\n").encode("utf-8"))
                            except Exception as e:
                                print(f"Chyba při odesílání klientovi: {e}")
                                client_conn = None

                    else:
                        display_text = f"Nizka pravdepodobnost ({max_prob * 100:.1f}%)"
                        text_color = (0, 165, 255)

        cv2.imshow("Rozpoznavani gest", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()