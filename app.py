import os
import gdown
import streamlit as st
import cv2
import numpy as np
import joblib
import mediapipe as mp
import time
import threading
from collections import deque, Counter
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import av

# -------------------------------------------------------------------
# Page configuration & custom CSS
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main { background-color: #f5f7fa; }
    .header { text-align: center; padding: 1.5rem 0 0.5rem 0; }
    .header h1 { font-size: 3rem; color: #1E3A8A; margin-bottom: 0.2rem; }
    .header p { font-size: 1.1rem; color: #475569; max-width: 700px; margin: 0 auto; }
    .stButton button {
        background-color: #2563EB; color: white; border-radius: 12px;
        padding: 0.6rem 2rem; font-weight: 600; border: none; transition: background-color 0.2s;
        margin: 20px auto; display: block;
    }
    .stButton button:hover { background-color: #1D4ED8; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Load model once, cached
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "models/sign_model.pkl"

    if not os.path.exists(model_path):
        os.makedirs("models", exist_ok=True)

        file_id = "15EdH8xObL9C2WmApvWX_zUuSAEulGm-9"
        url = f"https://drive.google.com/uc?id={file_id}"

        gdown.download(url, model_path, quiet=False)

    return joblib.load(model_path)

model = load_model()

# -------------------------------------------------------------------
# Shared application state (thread‑safe)
# -------------------------------------------------------------------
class AppState:
    def __init__(self):
        self.word = ""
        self.lock = threading.Lock()
        self.current_prediction = "?"
        self.confidence = 0.0
        self.hand_detected = False

    def add_letter(self, char: str):
        with self.lock:
            self.word += char

    def delete_last(self):
        with self.lock:
            self.word = self.word[:-1]

    def clear_word(self):
        with self.lock:
            self.word = ""

    def update_display(self, pred_char: str, conf: float, hand: bool):
        self.current_prediction = pred_char
        self.confidence = conf
        self.hand_detected = hand

    def get_word(self) -> str:
        with self.lock:
            return self.word

# -------------------------------------------------------------------
# Video processor – draws everything on the frame
# -------------------------------------------------------------------
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self, model, app_state: AppState):
        self.model = model
        self.app_state = app_state

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.prediction_buffer = deque(maxlen=10)
        self.stable_char = "nothing"
        self.stable_start_time = None
        self.appended_this_stable = False
        self.last_stable_char = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        hand_detected = False
        raw_pred_class = "nothing"
        confidence = 0.0

        if results.multi_hand_landmarks:
            hand_detected = True
            hand_landmarks = results.multi_hand_landmarks[0]

            self.mp_draw.draw_landmarks(img_rgb, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # 63 features
            features = []
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
            features = np.array(features).reshape(1, -1)

            # Predict returns the label directly
            raw_pred_class = self.model.predict(features)[0]
            proba = self.model.predict_proba(features)[0]
            confidence = np.max(proba) * 100

        self.prediction_buffer.append(raw_pred_class)

        if self.prediction_buffer:
            stable_char = Counter(self.prediction_buffer).most_common(1)[0][0]
        else:
            stable_char = "nothing"

        now = time.monotonic()
        if stable_char != self.last_stable_char:
            self.last_stable_char = stable_char
            self.stable_start_time = now
            self.appended_this_stable = False
        else:
            if self.stable_start_time is not None and (now - self.stable_start_time) >= 1.5:
                if not self.appended_this_stable and stable_char != "nothing":
                    if stable_char == "space":
                        self.app_state.add_letter(" ")
                    elif stable_char == "del":
                        self.app_state.delete_last()
                    else:
                        self.app_state.add_letter(stable_char)
                    self.appended_this_stable = True

        self.app_state.update_display(stable_char, confidence, hand_detected)

        # --- Draw overlay on the video frame ---
        # Convert back to BGR for OpenCV drawing
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h, w, _ = img_bgr.shape

        # Prediction box (top‑left)
        pred_char = stable_char if stable_char != "nothing" else "?"
        cv2.rectangle(img_bgr, (10, 10), (160, 110), (37, 99, 235), -1)  # blue background
        cv2.putText(img_bgr, "Prediction", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(img_bgr, pred_char, (20, 95), cv2.FONT_HERSHEY_DUPLEX, 2.5, (255,255,255), 3)

        # Confidence box (top‑right)
        conf_text = f"{confidence:.1f}%"
        (tw, th), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(img_bgr, (w-210, 10), (w-10, 80), (16, 185, 129), -1)  # green background
        cv2.putText(img_bgr, "Confidence", (w-200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(img_bgr, conf_text, (w-200, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        # Hand status (bottom‑left)
        status = "Hand Detected" if hand_detected else "No Hand Detected"
        status_color = (16, 185, 129) if hand_detected else (239, 68, 68)  # green or red
        cv2.putText(img_bgr, status, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Current word (bottom‑center, large)
        current_word = self.app_state.get_word()
        if current_word == "":
            current_word = " "  # avoid empty string
        (tw, th), _ = cv2.getTextSize(current_word, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        # Semi‑transparent background for readability
        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (w//2 - tw//2 - 20, h-70), (w//2 + tw//2 + 20, h-10), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.3, img_bgr, 0.7, 0, img_bgr)
        cv2.putText(img_bgr, current_word, (w//2 - tw//2, h-25), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

# -------------------------------------------------------------------
# Main Streamlit app – no infinite loop, no st.rerun()
# -------------------------------------------------------------------
def main():
    st.markdown(
        '<div class="header"><h1>🤟 Sign Language Recognition</h1>'
        '<p>Recognize American Sign Language (ASL) alphabet gestures in real time using MediaPipe hand landmarks and a Random Forest machine learning model.</p></div>',
        unsafe_allow_html=True
    )

    # Initialize shared state
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState()
    app_state = st.session_state.app_state

    # WebRTC streamer – video updates are self‑contained
    webrtc_ctx = webrtc_streamer(
        key="asl-recognition",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: SignLanguageProcessor(model, app_state),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Only show the clear button – everything else is on the video
    if webrtc_ctx.state.playing:
        if st.button("Clear Word"):
            app_state.clear_word()
    else:
        st.info("👆 Click **Start** to begin the webcam. Make sure your hand is clearly visible.")

if __name__ == "__main__":
    main()