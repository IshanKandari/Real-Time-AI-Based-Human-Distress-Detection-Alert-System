from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import sounddevice as sd
import geocoder
import logging
import numpy as np
import tensorflow as tf
import time
from datetime import datetime, timezone
import yaml
import librosa

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

# --- Load Configuration ---
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# --- Global Variables ---
inference_thread = None
stop_event = threading.Event()
current_gps_location = {}

# --- Helper Function ---
def to_mel_spectrogram(audio, audio_cfg):
    spec = librosa.feature.melspectrogram(y=audio, sr=audio_cfg['sample_rate'], n_mels=128, n_fft=2048, hop_length=512)
    return librosa.power_to_db(spec, ref=np.max)

# --- The Core Inference Engine ---
class InferenceEngine:
    def __init__(self, config):
        self.config = config
        self.model = tf.keras.models.load_model("models/distress_model.h5")

    def _get_best_location(self):
        global current_gps_location
        if current_gps_location: return current_gps_location, "high"
        g = geocoder.ip('me');
        if g.ok and g.latlng: return {"lat": g.latlng[0], "lon": g.latlng[1]}, "low"
        return {"lat": 28.7041, "lon": 77.1025}, "fallback"

    def process_chunk(self, audio_chunk: np.ndarray):
        audio_cfg = self.config['audio']
        mel_spec = to_mel_spectrogram(audio_chunk, audio_cfg)
        processed_chunk = np.expand_dims(np.expand_dims(mel_spec, axis=0), axis=-1)
        
        prediction = self.model.predict(processed_chunk, verbose=0)[0][0]

        socketio.emit('update_data', {'distress_confidence': float(prediction)})

        if prediction > self.config['inference']['distress_threshold']:
            best_location, loc_confidence = self._get_best_location()
            payload = {
                "confidence": float(prediction),
                "location": best_location,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            socketio.emit('distress_alert', payload)
            time.sleep(5)

    def audio_listener(self, stop_event):
        audio_cfg = self.config['audio']
        chunk_samples = int(audio_cfg['chunk_duration'] * audio_cfg['sample_rate'])
        def audio_callback(indata, frames, time, status):
            self.process_chunk(indata[:, 0].astype('float32'))
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=audio_cfg['sample_rate'], blocksize=chunk_samples):
            while not stop_event.is_set():
                socketio.sleep(0.1)
        logging.info("Audio listener has stopped.")

# --- Flask & SocketIO Routes ---
@app.route('/')
def index(): return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    logging.info("Frontend connected.")
    socketio.emit('status', {'message': 'Connected to backend.'})

@socketio.on('start_listening')
def handle_start():
    global inference_thread, stop_event
    if inference_thread is None or not inference_thread.is_alive():
        logging.info("START command received.")
        stop_event.clear()
        engine = InferenceEngine(config)
        inference_thread = socketio.start_background_task(target=engine.audio_listener, stop_event=stop_event)
        socketio.emit('status', {'message': 'Listening...'})

@socketio.on('stop_listening')
def handle_stop():
    global stop_event
    logging.info("STOP command received.")
    stop_event.set()
    socketio.emit('status', {'message': 'Stopped.'})

@socketio.on('update_location')
def handle_location_update(json):
    global current_gps_location
    current_gps_location = json
    logging.info(f"Received GPS update from browser: {current_gps_location}")

if __name__ == '__main__':
    logging.info("Starting Flask-SocketIO server at http://127.0.0.1:5000")
    socketio.run(app, debug=False)