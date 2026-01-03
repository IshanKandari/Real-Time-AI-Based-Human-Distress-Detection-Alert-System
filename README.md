🛡️ **Real-Time AI-Based Human Distress Detection and Alert System**

This project is a real-time, AI-driven emergency alert system that detects human distress using voice signals and automatically sends alerts with the user’s live GPS location. It is designed to help in situations where a victim may be unconscious, panicked, injured, or unable to manually trigger emergency services.

🎯 **Key Features**

✔ Real-time distress detection from live microphone input
✔ Deep Learning–based CRNN (Convolutional Recurrent Neural Network) model
✔ Works with screams or keywords like “Help / Bachao”
✔ Noise-resistant audio classification
✔ Live GPS tracking and mapping using Leaflet.js
✔ Instant alert broadcast using Flask-SocketIO (WebSockets)
✔ Responsive monitoring dashboard
✔ Lightweight + fast + practical for real-world safety use

🧠 **Tech Stack**

Frontend: HTML, CSS, JavaScript, Leaflet.js
Backend: Python Flask + Flask-SocketIO
AI / ML: TensorFlow, Keras, Librosa, NumPy
Environment: Jupyter Notebook, Anaconda / venv

🚀 **How It Works**

1️⃣ User opens the web app and grants microphone + location access
2️⃣ System continuously listens to 2-second audio chunks
3️⃣ Audio is converted into Mel-spectrograms
4️⃣ CRNN model predicts distress probability
5️⃣ If probability > threshold → Alert Triggered
6️⃣ GPS coordinates are sent to dashboard
7️⃣ Distress marker is shown on live map instantly

📌 **Use Cases**

Women safety

Senior citizen monitoring

Healthcare emergencies

Child security

Public safety surveillance

📷 **Screenshots**

Model Training Output
<img width="945" height="656" alt="image" src="https://github.com/user-attachments/assets/7ec069cf-681a-48aa-a055-3debff2ceae6" />

Live Inference Test
<img width="745" height="849" alt="image" src="https://github.com/user-attachments/assets/678fc626-7a63-4fa7-87a1-73833afc48a0" />

Dashboard Listening Mode
<img width="626" height="524" alt="image" src="https://github.com/user-attachments/assets/7d526f39-079c-4666-b2cb-c17fda71a087" />

Alert Triggered with Location
![915c8746-f4ac-48b0-a61e-393486e02d81](https://github.com/user-attachments/assets/3609ea58-b397-4f70-ac92-e8d81c05a77b)


**Installation**
pip install -r requirements.txt
python app.py


Then open:

http://127.0.0.1:5000/

**🏁 Status**

✔ Successfully implemented
✔ Tested with multiple scenarios
✔ Achieved ~97% accuracy
✔ Low latency (~100–200ms)

**🔮 Future Enhancements**

Mobile app version (Android / iOS)

Integration with police/emergency APIs

Multi-language distress recognition

Offline/Edge AI processing

Background detection support

**🙌 Author**

Ishan Kandari
BCA — Institute of Information Technology and Management, Janakpuri
Guru Gobind Singh Indraprastha University
