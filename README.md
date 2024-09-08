# Taductor_Youtube
![Leonardo_Phoenix_A_vibrant_eyecatching_thumbnail_for_a_YouTube_0](https://github.com/user-attachments/assets/cd496960-fe0a-4c3c-a2d6-8f4fe21b54d0)

Funcionalidad principal: Reconocimiento automático de idioma: El programa utiliza el modelo Whisper de OpenAI para transcribir audio de cualquier idioma a texto. No es necesario especificar el idioma de origen, ya que Whisper lo detecta automáticamente. Traducción al español: Una vez que el audio se transcribe a texto.

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
Interfaz fácil de usar: La GUI intuitiva facilita el uso del programa.
