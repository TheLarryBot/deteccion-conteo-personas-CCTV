"""
=============================================================
  DETECTOR DE PERSONAS EN CAMARAS DE SEGURIDAD
  Detecta, enmarca y cuenta personas en dos videos
  simultáneamente. Guarda los registros en tiempo real a CSV.
=============================================================
Dependencias:
    pip install ultralytics opencv-python pandas

Uso:
    python detector_camaras.py \
        --video1 "camara1.mp4" --lugar1 "Entrada Principal" \
        --video2 "camara2.mp4" --lugar2 "Pasillo Norte"

Opciones adicionales:
    --salida    Nombre del archivo CSV (default: registros_deteccion.csv)
    --confianza Umbral de confianza 0.0-1.0 (default: 0.40)
    --mostrar   Muestra ventanas de video en pantalla (flag)
    --guardar   Guarda los videos procesados con anotaciones
=============================================================
"""

import argparse
import csv
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────
MODELO_YOLO   = "yolov8n.pt"     # yolov8n (nano) se descarga automáticamente
CLASE_PERSONA = 0                # Índice COCO de "person"
COLOR_BOX     = (0, 255, 0)      # Verde para el recuadro
COLOR_TEXTO   = (0, 255, 0)
FUENTE        = cv2.FONT_HERSHEY_SIMPLEX
INTERVALO_LOG = 1.0              # Segundos entre registros CSV por cámara


# ─────────────────────────────────────────────
#  ESCRITURA THREAD-SAFE AL CSV
# ─────────────────────────────────────────────
class CSVWriter:
    """Gestiona escritura concurrente desde múltiples hilos."""

    ENCABEZADOS = [
        "fecha", "hora", "timestamp_unix",
        "camara_id", "lugar",
        "total_personas", "ids_detectados",
        "frame_numero", "fps_video"
    ]

    def __init__(self, ruta_csv: str):
        self.ruta = ruta_csv
        self._lock = threading.Lock()
        self._inicializar_csv()

    def _inicializar_csv(self):
        existe = Path(self.ruta).exists()
        with open(self.ruta, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.ENCABEZADOS)
            if not existe:
                w.writeheader()
        print(f"[CSV] Archivo listo: {self.ruta}")

    def escribir(self, fila: dict):
        with self._lock:
            with open(self.ruta, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self.ENCABEZADOS)
                w.writerow(fila)


# ─────────────────────────────────────────────
#  PROCESADOR DE UN VIDEO / CÁMARA
# ─────────────────────────────────────────────
class ProcesadorCamara:

    def __init__(
        self,
        video_path: str,
        camara_id: str,
        lugar: str,
        modelo: YOLO,
        csv_writer: CSVWriter,
        confianza: float = 0.40,
        mostrar: bool = True,
        guardar: bool = False,
    ):
        self.video_path  = video_path
        self.camara_id   = camara_id
        self.lugar       = lugar
        self.modelo      = modelo
        self.csv_writer  = csv_writer
        self.confianza   = confianza
        self.mostrar     = mostrar
        self.guardar     = guardar

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

        self.fps      = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.ancho    = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.alto     = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.writer_video = None
        if guardar:
            nombre_salida = f"procesado_{camara_id}_{Path(video_path).stem}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer_video = cv2.VideoWriter(
                nombre_salida, fourcc, self.fps, (self.ancho, self.alto)
            )
            print(f"[{camara_id}] Video procesado se guardará en: {nombre_salida}")

        self._ultimo_log  = 0.0
        self._frame_num   = 0
        self._activo      = True

    # ── Anotaciones visuales ──────────────────
    def _anotar_frame(self, frame, detecciones, cuenta):
        """Dibuja recuadros, IDs y contador sobre el frame."""
        for det in detecciones:
            x1, y1, x2, y2 = map(int, det["bbox"])
            track_id = det.get("id", "?")
            conf     = det["conf"]

            # Recuadro
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)

            # Etiqueta ID + confianza
            label = f"#{track_id}  {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, FUENTE, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), COLOR_BOX, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        FUENTE, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # Panel superior — lugar + conteo + tiempo
        ahora = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        panel = f"  {self.lugar}   |   Personas: {cuenta}   |   {ahora}"
        cv2.rectangle(frame, (0, 0), (self.ancho, 34), (0, 0, 0), -1)
        cv2.putText(frame, panel, (6, 24),
                    FUENTE, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

        # Barra de progreso inferior
        progreso = int(self.ancho * self._frame_num / max(self.total_frames, 1))
        cv2.rectangle(frame, (0, self.alto - 5), (progreso, self.alto), (0, 200, 255), -1)

        return frame

    # ── Ciclo principal ───────────────────────
    def procesar(self):
        nombre_ventana = f"Cámara {self.camara_id} — {self.lugar}"
        print(f"[{self.camara_id}] Iniciando procesamiento: {self.video_path}")
        print(f"[{self.camara_id}] Resolución: {self.ancho}×{self.alto}  |  FPS: {self.fps:.1f}  |  Frames: {self.total_frames}")

        while self._activo:
            ret, frame = self.cap.read()
            if not ret:
                print(f"[{self.camara_id}] Fin del video.")
                break

            self._frame_num += 1
            ahora_unix = time.time()

            # ── Inferencia YOLO con tracking ──────────────
            resultados = self.modelo.track(
                frame,
                persist=True,
                classes=[CLASE_PERSONA],
                conf=self.confianza,
                verbose=False,
                tracker="bytetrack.yaml",
            )

            detecciones = []
            if resultados and resultados[0].boxes is not None:
                boxes = resultados[0].boxes
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    tid  = int(boxes.id[i].cpu().numpy()) if boxes.id is not None else i
                    detecciones.append({"bbox": bbox, "conf": conf, "id": tid})

            cuenta = len(detecciones)
            ids_str = ",".join(str(d["id"]) for d in detecciones)

            # ── Registro CSV (throttled por INTERVALO_LOG) ─
            if ahora_unix - self._ultimo_log >= INTERVALO_LOG:
                self._ultimo_log = ahora_unix
                dt = datetime.now()
                self.csv_writer.escribir({
                    "fecha":           dt.strftime("%Y-%m-%d"),
                    "hora":            dt.strftime("%H:%M:%S"),
                    "timestamp_unix":  f"{ahora_unix:.3f}",
                    "camara_id":       self.camara_id,
                    "lugar":           self.lugar,
                    "total_personas":  cuenta,
                    "ids_detectados":  ids_str if ids_str else "ninguno",
                    "frame_numero":    self._frame_num,
                    "fps_video":       f"{self.fps:.2f}",
                })

            # ── Anotar y mostrar ──────────────────────────
            frame_anotado = self._anotar_frame(frame, detecciones, cuenta)

            if self.guardar and self.writer_video:
                self.writer_video.write(frame_anotado)

            if self.mostrar:
                cv2.imshow(nombre_ventana, frame_anotado)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord("q") or tecla == 27:   # Q o ESC para salir
                    print(f"[{self.camara_id}] Detenido por el usuario.")
                    self._activo = False
                    break

        self._liberar()

    def _liberar(self):
        self.cap.release()
        if self.writer_video:
            self.writer_video.release()
        if self.mostrar:
            cv2.destroyAllWindows()
        print(f"[{self.camara_id}] Recursos liberados.")


# ─────────────────────────────────────────────
#  RESUMEN FINAL
# ─────────────────────────────────────────────
def imprimir_resumen(ruta_csv: str):
    try:
        df = pd.read_csv(ruta_csv)
        print("\n" + "═" * 55)
        print("  RESUMEN DE DETECCIÓN")
        print("═" * 55)
        for lugar, grupo in df.groupby("lugar"):
            print(f"\n  📍 {lugar}")
            print(f"     Registros totales : {len(grupo)}")
            print(f"     Máx. personas/seg : {grupo['total_personas'].max()}")
            print(f"     Promedio personas  : {grupo['total_personas'].mean():.1f}")
            print(f"     Hora inicio        : {grupo['hora'].iloc[0]}")
            print(f"     Hora fin           : {grupo['hora'].iloc[-1]}")
        print("\n" + "═" * 55)
        print(f"  CSV guardado en: {ruta_csv}")
        print("═" * 55 + "\n")
    except Exception as e:
        print(f"[RESUMEN] Error al leer CSV: {e}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Detector de personas en cámaras de seguridad (2 videos simultáneos)"
    )
    parser.add_argument("--video1",    required=True,  help="Ruta al primer video")
    parser.add_argument("--lugar1",    default="Cámara 1", help="Nombre del lugar cámara 1")
    parser.add_argument("--video2",    required=True,  help="Ruta al segundo video")
    parser.add_argument("--lugar2",    default="Cámara 2", help="Nombre del lugar cámara 2")
    parser.add_argument("--salida",    default="registros_deteccion.csv", help="Archivo CSV de salida")
    parser.add_argument("--confianza", type=float, default=0.40, help="Umbral de confianza (0.0 - 1.0)")
    parser.add_argument("--mostrar",   action="store_true", help="Mostrar ventanas de video")
    parser.add_argument("--guardar",   action="store_true", help="Guardar videos procesados")
    args = parser.parse_args()

    print("\n" + "═" * 55)
    print("  SISTEMA DE DETECCIÓN — CÁMARAS DE SEGURIDAD")
    print("═" * 55)
    print(f"  Modelo  : {MODELO_YOLO}")
    print(f"  Video 1 : {args.video1}  →  {args.lugar1}")
    print(f"  Video 2 : {args.video2}  →  {args.lugar2}")
    print(f"  CSV     : {args.salida}")
    print(f"  Conf.   : {args.confianza}")
    print("═" * 55 + "\n")

    # Carga del modelo (se comparte entre hilos — YOLO es thread-safe en inferencia)
    print("[MODELO] Cargando YOLOv8... (se descarga si es la primera vez)")
    modelo = YOLO(MODELO_YOLO)
    print("[MODELO] Listo.\n")

    csv_writer = CSVWriter(args.salida)

    cam1 = ProcesadorCamara(
        video_path=args.video1,
        camara_id="CAM-01",
        lugar=args.lugar1,
        modelo=modelo,
        csv_writer=csv_writer,
        confianza=args.confianza,
        mostrar=args.mostrar,
        guardar=args.guardar,
    )
    cam2 = ProcesadorCamara(
        video_path=args.video2,
        camara_id="CAM-02",
        lugar=args.lugar2,
        modelo=modelo,
        csv_writer=csv_writer,
        confianza=args.confianza,
        mostrar=args.mostrar,
        guardar=args.guardar,
    )

    # Procesamiento paralelo en hilos
    hilo1 = threading.Thread(target=cam1.procesar, name="Hilo-CAM01", daemon=True)
    hilo2 = threading.Thread(target=cam2.procesar, name="Hilo-CAM02", daemon=True)

    hilo1.start()
    hilo2.start()

    hilo1.join()
    hilo2.join()

    imprimir_resumen(args.salida)


if __name__ == "__main__":
    main()