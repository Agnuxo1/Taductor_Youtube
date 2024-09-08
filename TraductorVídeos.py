""" Traducto de vídeos de Youtube Audio a Audio de cualquier idioma a español. 
Francisco ANgulo de Lafuente 
https://youtu.be/5MEUshT_xi4
https://github.com/Agnuxo1
8 de Agosto de 2024
Funcionalidad principal:
Reconocimiento automático de idioma: El programa utiliza el modelo Whisper de OpenAI para transcribir audio de cualquier idioma a texto. No es necesario especificar el idioma de origen, ya que Whisper lo detecta automáticamente.
Traducción al español: Una vez que el audio se transcribe a texto, el programa utiliza la biblioteca googletrans para traducir el texto al español.
Síntesis de voz: El programa utiliza el modelo TTS (Text-to-Speech) VITS de la biblioteca TTS para convertir el texto traducido en audio en español.
Reproducción de audio: El audio traducido se reproduce al usuario a través de la biblioteca sounddevice.
Componentes principales:
Interfaz gráfica de usuario (GUI): El programa utiliza PyQt5 para crear una GUI intuitiva que permite a los usuarios:
Introducir una URL de vídeo de YouTube.
Cargar y reproducir el vídeo.
Controlar el volumen del audio original y traducido.
Ajustar la velocidad de reproducción del audio traducido.
Activar/desactivar el modo nocturno para una mejor visualización.
Obtener la URL del vídeo que se está reproduciendo actualmente.
Recargar el programa para traducir un nuevo vídeo.
Procesamiento de audio:
AudioProcessingThread: Esta clase se encarga de extraer el audio del vídeo de YouTube, transcribirlo a texto utilizando Whisper y traducir el texto al español.
AudioPlaybackThread: Esta clase se encarga de reproducir el audio traducido generado por el modelo TTS.
Gestión de búfer: El programa utiliza un búfer para almacenar el audio traducido antes de reproducirlo. Esto ayuda a garantizar una reproducción fluida, incluso si la traducción lleva algún tiempo.
Flujo de trabajo:
El usuario introduce la URL del vídeo de YouTube en la GUI.
El programa descarga el audio del vídeo y lo guarda en un archivo WAV.
Se inicia un hilo AudioProcessingThread para procesar el audio.
El hilo AudioProcessingThread extrae el audio del archivo WAV, lo transcribe a texto utilizando Whisper y traduce el texto al español.
El texto traducido se envía al hilo AudioPlaybackThread.
El hilo AudioPlaybackThread convierte el texto traducido en audio utilizando el modelo TTS y lo reproduce al usuario.
Ventajas:
Reconocimiento automático de idioma: Elimina la necesidad de que el usuario especifique el idioma de origen.
Traducción en tiempo real: Proporciona una traducción casi instantánea del audio.
Interfaz fácil de usar: La GUI intuitiva facilita el uso del programa."""

import os
import sys
import torch
from transformers import pipeline
import warnings
import numpy as np
from TTS.api import TTS
import sounddevice as sd
import threading
import queue
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLineEdit, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QSlider, QLabel)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage, QWebEngineProfile
from PyQt5.QtGui import QIcon
import ffmpeg
import yt_dlp
from googletrans import Translator
from scipy import signal
from urllib.parse import urlparse, parse_qs, urlencode

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Global configuration
MAX_TOKENS = 100
TEMPERATURA = 0.5
BUFFER_SIZE = 10  # Buffer size in seconds

# Determine available device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize translator
translator = Translator()

# Initialize TTS model
tts = TTS(model_name="tts_models/es/css10/vits", progress_bar=False).to(device)

# Audio queue for generation
audio_queue = queue.Queue()

# Initialize Whisper model
whisper_model = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)

class AudioProcessingThread(QThread):
    translation_ready = pyqtSignal(str, float)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.running = True

    def run(self):
        try:
            # Extract audio from video
            stream = ffmpeg.input(self.video_path)
            audio = stream.audio.output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar='16000')
            out, _ = audio.run(capture_stdout=True)

            # Process audio with Whisper
            audio_data = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
            result = whisper_model(audio_data)

            # Process transcription
            transcription = result['text']
            segments = self.split_transcription(transcription)

            for segment in segments:
                if not self.running:
                    break
                translation = self.translate_text(segment['text'])
                self.translation_ready.emit(translation, segment['start'])

        except Exception as e:
            print(f"Error in audio processing: {str(e)}")

    def split_transcription(self, transcription, max_length=100):
        words = transcription.split()
        segments = []
        current_segment = {'text': '', 'start': 0}
        word_count = 0

        for i, word in enumerate(words):
            current_segment['text'] += word + ' '
            word_count += 1

            if word_count >= max_length or i == len(words) - 1:
                segments.append(current_segment)
                current_segment = {'text': '', 'start': (i + 1) / len(words)}
                word_count = 0

        return segments

    def translate_text(self, text):
        try:
            # Usar el reconocimiento automático de idiomas
            translation = translator.translate(text, dest='es').text
            return translation
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return text

    def stop(self):
        self.running = False

class AudioPlaybackThread(QThread):
    def __init__(self):
        super().__init__()
        self.running = True
        self.current_audio = None
        self.is_playing = False
        self.stop_signal = threading.Event()
        self.speed = 1.0
        self.volume = 0.5  # Default volume

    def run(self):
        while self.running:
            if not audio_queue.empty() and not self.is_playing:
                self.current_audio = audio_queue.get()
                self.is_playing = True
                self.stop_signal.clear()
                
                # Adjust audio speed and volume
                resampled_audio = self.resample_audio(self.current_audio, self.speed)
                volume_adjusted_audio = resampled_audio * self.volume
                
                sd.play(volume_adjusted_audio, tts.synthesizer.output_sample_rate)
                while sd.get_stream().active and not self.stop_signal.is_set():
                    time.sleep(0.1)
                sd.stop()
                self.is_playing = False
            else:
                time.sleep(0.1)

    def resample_audio(self, audio, speed):
        # Calculate new audio length based on speed
        new_length = int(len(audio) / speed)
        # Use scipy's resample to change audio speed
        return signal.resample(audio, new_length)

    def set_speed(self, speed):
        self.speed = speed

    def set_volume(self, volume):
        self.volume = volume

    def stop_thread(self):
        self.running = False
        self.stop_signal.set()

class CustomWebEnginePage(QWebEnginePage):
    def __init__(self, profile, parent=None):
        super().__init__(profile, parent)

    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        print(f"JS Console ({level}): {message} (Line {lineNumber}, Source: {sourceID})")

    def certificateError(self, error):
        print(f"Certificate Error: {error.errorDescription()}")
        return False  # Reject the certificate

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YouTube Video Translator")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        # URL input and controls
        url_layout = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter YouTube video link")
        url_layout.addWidget(self.url_input)

        self.load_button = QPushButton("Load Video")
        self.load_button.clicked.connect(self.load_video)
        url_layout.addWidget(self.load_button)

        self.live_button = QPushButton("Live")
        self.live_button.clicked.connect(self.get_current_video_url)
        url_layout.addWidget(self.live_button)

        self.reload_button = QPushButton()
        self.reload_button.setIcon(QIcon.fromTheme("view-refresh"))
        self.reload_button.setToolTip("Reload")
        self.reload_button.clicked.connect(self.reload_program)
        url_layout.addWidget(self.reload_button)

        self.layout.addLayout(url_layout)

        # Video player
        self.web_profile = QWebEngineProfile("youtube_profile", self)
        self.web_profile.setHttpUserAgent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        self.video_player = QWebEngineView()
        self.custom_page = CustomWebEnginePage(self.web_profile, self.video_player)
        self.video_player.setPage(self.custom_page)
        self.video_player.setUrl(QUrl("https://www.youtube.com/@Francisco_Angulo_de_Lafuente"))
        self.layout.addWidget(self.video_player)

        # Audio controls
        audio_layout = QHBoxLayout()

        self.original_volume_slider = QSlider(Qt.Horizontal)
        self.original_volume_slider.setRange(0, 100)
        self.original_volume_slider.setValue(50)
        self.original_volume_slider.valueChanged.connect(self.update_original_volume)
        audio_layout.addWidget(QLabel("Original Volume:"))
        audio_layout.addWidget(self.original_volume_slider)

        self.translated_volume_slider = QSlider(Qt.Horizontal)
        self.translated_volume_slider.setRange(0, 100)
        self.translated_volume_slider.setValue(50)
        self.translated_volume_slider.valueChanged.connect(self.update_translated_volume)
        audio_layout.addWidget(QLabel("Translated Volume:"))
        audio_layout.addWidget(self.translated_volume_slider)

        self.layout.addLayout(audio_layout)

        # Audio speed control
        speed_layout = QHBoxLayout()
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(50, 200)  # 0.5x to 2.0x
        self.speed_slider.setValue(100)  # 1.0x by default
        self.speed_slider.valueChanged.connect(self.update_audio_speed)
        speed_layout.addWidget(QLabel("Audio Speed:"))
        speed_layout.addWidget(self.speed_slider)
        self.layout.addLayout(speed_layout)

        # Buffer size control
        buffer_layout = QHBoxLayout()
        self.buffer_size_slider = QSlider(Qt.Horizontal)
        self.buffer_size_slider.setRange(1, 30)
        self.buffer_size_slider.setValue(BUFFER_SIZE)
        self.buffer_size_slider.valueChanged.connect(self.update_buffer_size)
        buffer_layout.addWidget(QLabel("Buffer Size (seconds):"))
        buffer_layout.addWidget(self.buffer_size_slider)
        self.layout.addLayout(buffer_layout)

        # Night mode button
        self.night_mode_button = QPushButton("Night Mode")
        self.night_mode_button.clicked.connect(self.toggle_night_mode)
        self.layout.addWidget(self.night_mode_button)

        self.central_widget.setLayout(self.layout)

        self.audio_processing_thread = None
        self.translation_buffer = []
        self.audio_playback_thread = AudioPlaybackThread()
        self.audio_playback_thread.start()

        self.audio_speed = 1.0
        self.night_mode = False

        # Initialize translated audio volume
        self.update_translated_volume(50)

        self.apply_style()

    def get_current_video_url(self):
        js = """
        (function() {
            var videoElement = document.querySelector('video');
            if (videoElement) {
                var videoUrl = window.location.href;
                if (videoUrl.includes('youtube.com/watch')) {
                    return videoUrl;
                }
            }
            return 'No YouTube video found';
        })();
        """
        self.custom_page.runJavaScript(js, self.update_url_input)

    def update_url_input(self, url):
        if url != 'No YouTube video found':
            self.url_input.setText(url)
            print(f"URL updated: {url}")
        else:
            print("No YouTube video found")

    def load_video(self):
        url = self.url_input.text()
        self.process_video_url(url)

    def process_video_url(self, url):
        try:
            # Parse the URL and add autoplay parameter
            parsed_url = urlparse(url)
            query_params = parse_qs(parsed_url.query)
            query_params['autoplay'] = ['1']
            new_query = urlencode(query_params, doseq=True)
            new_url = parsed_url._replace(query=new_query).geturl()

            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'outtmpl': 'downloaded_audio.%(ext)s'
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_title = info['title']

            self.video_player.setUrl(QUrl(new_url))
            
            # Inject JavaScript to ensure video starts playing
            js = """
            (function() {
                var attemptPlay = function() {
                    var video = document.querySelector('video');
                    if (video) {
                        video.play();
                    } else {
                        setTimeout(attemptPlay, 1000);
                    }
                };
                attemptPlay();
            })();
            """
            self.custom_page.runJavaScript(js)

            if self.audio_processing_thread:
                self.audio_processing_thread.stop()
                self.audio_processing_thread.wait()

            self.audio_processing_thread = AudioProcessingThread('downloaded_audio.wav')
            self.audio_processing_thread.translation_ready.connect(self.handle_translation)
            self.audio_processing_thread.start()

        except Exception as e:
            print(f"Error loading video: {str(e)}")

    def handle_translation(self, translation, timestamp):
        self.translation_buffer.append((translation, timestamp))
        if len(self.translation_buffer) * BUFFER_SIZE >= 10:  # Start playback when buffer has 10 seconds
            self.start_playback()

    def update_audio_speed(self, value):
        self.audio_speed = value / 100.0
        if self.audio_playback_thread:
            self.audio_playback_thread.set_speed(self.audio_speed)

    def update_original_volume(self, value):
        volume = value / 100.0
        js = f"document.querySelector('video').volume = {volume};"
        self.custom_page.runJavaScript(js)

    def update_translated_volume(self, value):
        volume = value / 100.0
        self.audio_playback_thread.set_volume(volume)

    def start_playback(self):
        for translation, timestamp in self.translation_buffer:
            print(f"[{timestamp:.2f}s] {translation}")
            wav = tts.tts(translation)
            audio_queue.put(wav)
        self.translation_buffer.clear()

    def update_buffer_size(self, value):
        global BUFFER_SIZE
        BUFFER_SIZE = value

    def toggle_night_mode(self):
        self.night_mode = not self.night_mode
        self.apply_style()

    def reload_program(self):
        # Stop current execution
        if self.audio_processing_thread:
            self.audio_processing_thread.stop()
            self.audio_processing_thread.wait()
        
        # Clear audio queue and stop playback
        while not audio_queue.empty():
            audio_queue.get()
        self.audio_playback_thread.stop_signal.set()
        
        # Clear translation buffer
        self.translation_buffer.clear()
        
        # Reset URL input
        self.url_input.clear()
        
        # Reset video player
        self.video_player.setUrl(QUrl("https://www.youtube.com"))
        
        print("Program reloaded and ready for a new video.")

    def apply_style(self):
        if self.night_mode:
            self.setStyleSheet("""
                QWidget {
                    background-color: #2E2E2E;
                    color: #E0E0E0;
                }
                QPushButton {
                    background-color: #4A4A4A;
                    color: #E0E0E0;
                    border: 1px solid #6A6A6A;
                    padding: 5px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #5A5A5A;
                }
                QLineEdit, QTextEdit {
                    background-color: #3E3E3E;
                    color: #E0E0E0;
                    border: 1px solid #6A6A6A;
                    padding: 3px;
                }
                QSlider::groove:horizontal {
                    background: #4A4A4A;
                    height: 8px;
                    border-radius: 4px;
                }
                QSlider::handle:horizontal {
                    background: #B39DDB;
                    width: 18px;
                    margin: -5px 0;
                    border-radius: 9px;
                }
            """)
        else:
            self.setStyleSheet("""
                QWidget {
                    background-color: #F0F0F0;
                    color: #333333;
                }
                QPushButton {
                    background-color: #E0E0E0;
                    color: #333333;
                    border: 1px solid #CCCCCC;
                    padding: 5px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #D0D0D0;
                }
                QLineEdit, QTextEdit {
                    background-color: #FFFFFF;
                    color: #333333;
                    border: 1px solid #CCCCCC;
                    padding: 3px;
                }
                QSlider::groove:horizontal {
                    background: #CCCCCC;
                    height: 8px;
                    border-radius: 4px;
                }
                QSlider::handle:horizontal {
                    background: #9575CD;
                    width: 18px;
                    margin: -5px 0;
                    border-radius: 9px;
                }
            """)

    def closeEvent(self, event):
        if self.audio_processing_thread:
            self.audio_processing_thread.stop()
            self.audio_processing_thread.wait()
        self.audio_playback_thread.stop_thread()
        self.audio_playback_thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())




