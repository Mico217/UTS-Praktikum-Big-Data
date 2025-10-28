import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import time
from collections import defaultdict, Counter
import threading
import os 

# ==========================
#UI
# ==========================
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px;
    }
    .stButton>button {
        background: linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%);
        border: none;
        border-radius: 10px;
        color: white;
        padding: 10px 20px;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }
    .card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .footer {
        text-align: center;
        font-size: 12px;
        color: #ccc;
        margin-top: 50px;
    }
    .stat-num {
        font-size: 24px;
        font-weight: 800;
        color: #333;
    }
    .stat-label {
        font-size: 14px;
        color: #666;
    }
    .sidebar-card {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: white;
    }
    .sidebar-tip {
        font-size: 14px;
        margin: 5px 0;
    }
    .spotlight-item {
        margin: 10px 0;
        padding: 10px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/MikoSigma.pt")
    classifier = tf.keras.models.load_model("model/MikoCihuy.h5", compile=False)
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
#Session State
# ==========================
if "od_unique" not in st.session_state:
    st.session_state.od_unique = set()  
if "od_history" not in st.session_state:
    st.session_state.od_history = []  
if "total_od" not in st.session_state:
    st.session_state.total_od = 0  
if "cls_history" not in st.session_state:
    st.session_state.cls_history = [] 
if "total_cls" not in st.session_state:
    st.session_state.total_cls = 0
if "webrtc_started" not in st.session_state:
    st.session_state.webrtc_started = False

def add_od_history(label):
    if label not in st.session_state.od_unique:
        st.session_state.od_unique.add(label)
        st.session_state.od_history.append(label)
        st.session_state.total_od = len(st.session_state.od_unique)  

def add_cls_history(label, conf):
    st.session_state.cls_history.append({'label': label, 'confidence': conf})
    st.session_state.total_cls += 1

#Biar keren aja
def get_object_fact(label):
    facts = {
        "person": "🧑 Orang adalah spesies paling adaptif di Bumi, mampu hidup di berbagai lingkungan!",
        "cell phone": "📱 Telepon genggam pertama kali diciptakan pada 1973, tapi sekarang hampir semua orang punya!",
        "car": "🚗 Mobil pertama berjalan pada 1886, dan sekarang ada lebih dari 1 miliar mobil di dunia!",
        "dog": "🐶 Anjing adalah hewan peliharaan tertua, dengan sejarah domestikasi lebih dari 15.000 tahun!",
        "cat": "🐱 Kucing bisa melompat hingga 6 kali panjang tubuhnya – atlet sejati!",
        "bottle": "🍾 Botol kaca pertama kali dibuat sekitar 1500 SM di Mesir!",
        "chair": "🪑 Kursi pertama kali digunakan oleh orang Mesir kuno sekitar 3000 SM!",
        "book": "📖 Buku cetak pertama adalah Bible Gutenberg pada 1455!",
        "laptop": "💻 Laptop pertama muncul pada 1981, dan sekarang tak tergantikan!",
        "cup": "☕ Cangkir teh pertama kali dari Cina pada abad ke-9!",
        "banana": "🍌 Pisang adalah salah satu buah paling tua di dunia, dibudidayakan sejak 7.000 tahun lalu dan secara teknis tergolong berry!",
        "apple": "🍎 Buah apel mengapung di air karena 25%-nya terdiri dari udara – dan ada lebih dari 7.500 jenis apel di dunia!",
        "orange": "🍊 Jeruk adalah sumber vitamin C yang sangat kaya, dan secara genetik merupakan hasil persilangan alami antara jeruk bali dan jeruk keprok!"
    }

    # Kembalikan fakta jika label ditemukan, jika tidak berikan default
    return facts.get(label, "ℹ️ Fakta belum tersedia untuk objek ini.")


#Simpan file feedback
def save_feedback(feedback):
    with open("feedbacks.txt", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {feedback}\n")

# ==========================
#Buka kamera
# ==========================
class ObjectDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_annotated = None
        self.counters = Counter()
        self.lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        display_img = img.copy()
        frame_skip = 2
        conf_thresh = 30 / 100.0
        show_boxes = True
        
        if (self.frame_count % frame_skip) == 0:
            target_w, target_h = 640, 360
            img_resized = cv2.resize(img, (target_w, target_h))
            results = yolo_model(img_resized, verbose=False)  # Gunakan model kamera
            boxes = results[0].boxes
            scale_x = img.shape[1] / target_w
            scale_y = img.shape[0] / target_h

            annotated = display_img.copy()
            local_counter = Counter()

            for box in boxes:
                conf = float(box.conf[0])
                if conf < conf_thresh:
                    continue
                cls_id = int(box.cls[0])
                label = yolo_model.names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = box.xyxy[0]
                x1 = int(x1 * scale_x); x2 = int(x2 * scale_x)
                y1 = int(y1 * scale_y); y2 = int(y2 * scale_y)

                local_counter[label] += 1

                if show_boxes:
                    conf_p = int(conf * 100)
                    text = f"{label} {conf_p}%"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
                    
            with self.lock:
                for k, v in local_counter.items():
                    self.counters[k] += v

            self.last_annotated = annotated
            out_frame = annotated
        else:
            out_frame = self.last_annotated if self.last_annotated is not None else display_img

        return av.VideoFrame.from_ndarray(out_frame, format="bgr24")

    def pop_counters(self):
        with self.lock:
            popped = dict(self.counters)
            self.counters.clear()
        return popped

# ==========================
#Sidebar
# ==========================
st.sidebar.markdown("""
    <div style="text-align: center; color: white; font-size: 26x; font-weight: bold; margin-bottom: 10px;">
        Object Detection dibuat menggunakan model YOLO & Klasifikasi dibuat menggunakan model Tensorflow
    </div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
# Tambahan: Teks kredit kecil di bawah
st.sidebar.markdown("""
    <div style="font-size: 10px; color: rgba(255, 255, 255, 0.7); text-align: center; margin-top: 10px;">
        by Tuah Mico Ananda
    </div>
""", unsafe_allow_html=True)

#Tutorial nih cihuy
with st.sidebar.expander("📚 Tutorial Penggunaan", expanded=True):
    st.markdown("""
    <div class="sidebar-tip">
    <strong>Object Detection:</strong><br>
    - Klik "Start Camera" untuk memulai deteksi real-time.<br>
    - Untuk menghentikan dan menyimpan hasil ke dashboard statistik, klik tombol "Stop Camera" yang ada di atas (bukan di bawah), agar deteksi objek tercatat dengan benar.<br>
    - Jika kamera tidak bisa dibuka, silahkan refresh website nya.<br>
    - Upload gambar maka akan otomatin deteksi objek pada gambar.<br>
    <strong>Klasifikasi Kacamata:</strong><br>
    - Unggah gambar wajah, maka akan otomatis klasifikasi apakah memakai kacamata atau tidak.<br>
    <strong>Dashboard Statistik:</strong><br>
    - Lihat ringkasan deteksi dan history.
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")

#Object Spotlight
st.sidebar.markdown("### 💡 Object Spotlight")
if st.session_state.od_unique:
    last_unique = list(st.session_state.od_unique)[-1] 
    fact = get_object_fact(last_unique)
    st.sidebar.markdown(f"""
    <div class="spotlight-item">
    <strong>{last_unique.capitalize()}</strong><br>
    {fact}
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("*Belum ada objek terdeteksi untuk spotlight.*")
st.sidebar.markdown("---")

#Reset Statistik
st.sidebar.markdown("### 🔄 Reset Statistik")
if st.sidebar.button("Reset Semua Statistik", key="reset_stats"):
    st.session_state.od_unique.clear()
    st.session_state.od_history.clear()
    st.session_state.total_od = 0
    st.session_state.cls_history.clear()
    st.session_state.total_cls = 0
    st.sidebar.success("Statistik telah direset!")
st.sidebar.markdown("---")

#Feedback
st.sidebar.markdown("### 📝 Feedback")
feedback = st.sidebar.text_area("Berikan saran atau kritik Anda:", height=100)
if st.sidebar.button("Kirim Feedback", key="send_feedback"):
    if feedback.strip():
        save_feedback(feedback)
        st.sidebar.success("Terima kasih atas feedback Anda! Feedback telah disimpan.")
    else:
        st.sidebar.warning("Feedback tidak boleh kosong.")

# ==========================
#Main UI
# ==========================
st.title("Real-Time Object Detection & Klasifikasi Kacamata")
st.markdown("Selamat datang! Sebelum menggunakan silahkan baca tutorial dan penggunaan di sidebar.")

tabs = st.tabs(["📷 Object Detection", "🕶️ Klasifikasi Kacamata", "📊 Dashboard Statistik"])

# ==========================
#Tab 1 Object Detection
# ==========================
with tabs[0]:
    st.subheader("📷 Object Detection")
    st.markdown("Pilih mode deteksi: Upload gambar atau gunakan kamera real-time.")
    
    # Ubah kolom menjadi 3: kiri, separator, kanan
    col_left, col_sep, col_right = st.columns([1, 0.05, 1])  # Kolom separator sempit
    
    # Kolom Kiri: Upload Gambar
    with col_left:
        st.markdown("### 📤 Upload Gambar")
        uploaded_image = st.file_uploader(
            "Unggah gambar (format .jpg, .jpeg, .png)",
            type=["jpg", "jpeg", "png"],
            key="upload_image"
        )
        
        if uploaded_image is not None:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Gambar yang diunggah", use_container_width=True)
            
            with st.spinner("🔍 Mendeteksi objek..."):
                img_array = np.array(image)
                orig_h, orig_w = img_array.shape[:2]  # Ukuran asli gambar
                target_w, target_h = 640, 360  # Ukuran resize untuk model
                img_resized = cv2.resize(img_array, (target_w, target_h))
                results = yolo_model(img_resized, verbose=False)
                boxes = results[0].boxes
                
                # Hitung scale factor untuk menyesuaikan bounding box ke ukuran asli
                scale_x = orig_w / target_w
                scale_y = orig_h / target_h
                
                annotated_img = img_array.copy()
                detected_labels = set()
                
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf < 0.3:  # Threshold default
                        continue
                    cls_id = int(box.cls[0])
                    label = yolo_model.names.get(cls_id, str(cls_id))
                    x1, y1, x2, y2 = box.xyxy[0]
                    
                    # Sesuaikan koordinat bounding box ke ukuran asli
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    
                    detected_labels.add(label)
                    
                    conf_p = int(conf * 100)
                    text = f"{label} {conf_p}%"
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated_img, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
                
                st.image(annotated_img, caption="Hasil Deteksi", use_container_width=True)
                
                if detected_labels:
                    st.success(f"Objek terdeteksi: {', '.join(detected_labels)}")
                    for label in detected_labels:
                        add_od_history(label)
                else:
                    st.info("Tidak ada objek terdeteksi.")
    
    # Kolom Separator: Garis vertikal pendek
    with col_sep:
        st.markdown("""
        <div style="border-left: 2px solid rgba(255, 255, 255, 0.5); height: 300px; margin: 20px auto 0;"></div>
        """, unsafe_allow_html=True)
    
    # Kolom Kanan: Kamera Real-Time
    with col_right:
        st.markdown("### 📹 Real-Time Camera")
        st.markdown("Gunakan kamera untuk deteksi objek secara real-time.")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            start_btn = st.button("🔵 Start Camera", key="start_cam")
        with col2:
            stop_btn = st.button("⛔ Stop Camera", key="stop_cam")
        
        if start_btn:
            st.session_state.webrtc_started = True
        if stop_btn:
            st.session_state.webrtc_started = False
            if hasattr(st.session_state, 'current_webrtc_ctx') and st.session_state.current_webrtc_ctx:
                vp = st.session_state.current_webrtc_ctx.video_processor
                if vp is not None:
                    new_counts = vp.pop_counters()
                    if new_counts:
                        for label in new_counts.keys(): 
                            add_od_history(label) 
        
        webrtc_placeholder = st.empty()
        if st.session_state.webrtc_started:
            webrtc_ctx = webrtc_streamer(
                key="object-detection",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=ObjectDetectionProcessor,
                media_stream_constraints={
                    "audio": False,
                    "video": {
                        "width": {"ideal": 640, "max": 640},
                        "height": {"ideal": 360, "max": 360},
                        "frameRate": {"ideal": 30}
                    },
                },
                async_processing=True,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            )
            st.session_state.current_webrtc_ctx = webrtc_ctx 
        else:
            st.info("Kamera dimatikan. Klik 'Start Camera' untuk memulai deteksi.")


# ==========================
# Tab 2 Klasifikasi Kacamata
# ==========================
with tabs[1]:
    st.subheader("🕶️ Deteksi Pemakaian Kacamata")
    st.markdown("Unggah gambar wajah untuk klasifikasi apakah memakai kacamata atau tidak.")
    
    uploaded_file = st.file_uploader(
        "Unggah gambar wajah (format .jpg, .jpeg, .png)",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Gambar yang diunggah", use_container_width=True)
        
        with col2:
            with st.spinner("🔍 Memproses gambar..."):
                img_array = np.array(image)
                img_resized = cv2.resize(img_array, (150, 150))
                img_norm = img_resized / 255.0
                input_data = np.expand_dims(img_norm, axis=0)
                
                pred = classifier.predict(input_data)[0][0]
                
                if pred > 0.5:
                    label = "tidak pakai kacamata"
                    confidence = pred * 100
                    emot = "👀"
                else:
                    label = "pakai kacamata"
                    confidence = (1 - pred) * 100
                    emot = "🕶️"
                
                st.success(f"{emot} **{confidence:.0f}% {label}** — Mantaf ga? Mantaf dong😎👍")
                
                add_cls_history(label, confidence)

# ==========================
# Tab 3 Dashboard Statistik
# ==========================
with tabs[2]:
    st.subheader("📊 Dashboard Statistik")
    st.markdown("Ringkasan deteksi objek dan klasifikasi yang telah dilakukan.")
    
    col_left, col_right = st.columns([1, 1])
    
    #Object Detection
    with col_left:
        st.markdown("### 🚀 Object Detection")
        od_total = len(st.session_state.od_unique)  # Total objek unik
        st.markdown(f"<div class='stat-num'>{od_total}</div><div class='stat-label'>Total Objek Unik Terdeteksi</div>", unsafe_allow_html=True)
        
        if od_total > 0:
            last_od = st.session_state.od_history[-5:] 
            st.markdown("**5 History Deteksi Terakhir:**")
            for label in last_od:
                st.markdown(f"- **{label}**")
        else:
            st.markdown("*Belum ada deteksi objek.*")
    
#Klasifikasi
with col_right:
    st.markdown("### 🕶️ Klasifikasi Kacamata")
    cls_total = st.session_state.total_cls
    st.markdown(f"<div class='stat-num'>{cls_total}</div><div class='stat-label'>Total Prediksi Dilakukan</div>", unsafe_allow_html=True)
    
    if cls_total > 0:
        last_cls = st.session_state.cls_history[-5:]  # 5 terakhir
        st.markdown("**5 History Prediksi Terakhir:**")
        for item in last_cls:
            label = item['label']
            conf = item['confidence']
            st.markdown(f"- **{label}**: {conf:.0f}%")
    else:
        st.markdown("*Belum ada prediksi klasifikasi.*")

# ==========================
# Footer
# ==========================
st.markdown("""
    <div class="footer">
        Bismillah 100 boleh kali bang. Udah capek kali bikin nya ni bg hehe.
    </div>
""", unsafe_allow_html=True)
#ALHAMDULILAH UDAH GADA ERORRRR
