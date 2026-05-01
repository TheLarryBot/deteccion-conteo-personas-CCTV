"""
=============================================================
  DETECTOR DE PERSONAS — 3 CAMARAS DE SEGURIDAD
  - Una sola ventana con mosaico de 3 camaras
  - Cada hilo carga su propio modelo YOLO de forma aislada
  - Guarda registros en tiempo real en CSV
=============================================================
Dependencias:
    pip install ultralytics opencv-python pandas

Uso:
    python deteccion.py
=============================================================
"""

import csv
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────
#  ★  CONFIGURA AQUÍ TUS VIDEOS Y LUGARES  ★
# ─────────────────────────────────────────────────────────────
VIDEOS = [
    {
        "id":    "CAM-01",
        "path":  r"Ruta video 1",
        "lugar": "Pasillo",
    },
    {
        "id":    "CAM-02",
        "path":  r"Ruta video 2",
        "lugar": "Entrada",
    },
    {
        "id":    "CAM-03",
        "path":  r"Ruta video 3",
        "lugar": "Caja",
    },
]

ARCHIVO_CSV   = r"deteccion-personas-cctv\registros_deteccion.csv"
MODELO_YOLO   = "yolov8s.pt"   # yolov8n=rapido, yolov8m=balanceado, yolov8l=preciso
CONFIANZA     = 0.35
IOU           = 0.45
ANCHO_CAM     = 640            # px por camara en el mosaico
ALTO_CAM      = 360
INTERVALO_LOG = 1.0            # segundos entre registros CSV

# ─────────────────────────────────────────────────────────────
COLORES = [
    (0,  255,  80),    # CAM-01 Verde
    (0,  200, 255),    # CAM-02 Cyan
    (255, 140,  0),    # CAM-03 Naranja
]
FUENTE = cv2.FONT_HERSHEY_SIMPLEX

# Flag global para detener todos los hilos
DETENER = threading.Event()


# ─────────────────────────────────────────────
#  CSV THREAD-SAFE
# ─────────────────────────────────────────────
class CSVWriter:
    COLS = [
        "fecha", "hora", "timestamp_unix", "camara_id", "lugar",
        "total_personas", "ids_detectados", "frame_numero", "fps_video",
    ]

    def __init__(self, ruta):
        self._lock = threading.Lock()
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        self.ruta = ruta
        existe = Path(ruta).exists()
        with open(ruta, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.COLS)
            if not existe:
                w.writeheader()
        print(f"[CSV] {ruta}")

    def escribir(self, fila):
        with self._lock:
            with open(self.ruta, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.COLS).writerow(fila)


# ─────────────────────────────────────────────
#  WORKER — cada hilo carga su propio YOLO
# ─────────────────────────────────────────────
class WorkerCamara(threading.Thread):

    def __init__(self, cfg, idx, csv_writer):
        super().__init__(daemon=True)
        self.cfg        = cfg
        self.camara_id  = cfg["id"]
        self.lugar      = cfg["lugar"]
        self.video_path = cfg["path"]
        self.color      = COLORES[idx % len(COLORES)]
        self.csv_writer = csv_writer

        # Frame compartido con hilo principal
        self._frame   = np.zeros((ALTO_CAM, ANCHO_CAM, 3), dtype=np.uint8)
        self._cuenta  = 0
        self._mu      = threading.Lock()

        self._ultimo_log  = 0.0
        self.frame_num    = 0
        self.total_frames = 1
        self.fps          = 25.0
        self.terminado    = False

    # acceso thread-safe
    def get_frame(self):
        with self._mu:
            return self._frame.copy(), self._cuenta

    def _put_frame(self, frame, cuenta):
        with self._mu:
            self._frame  = frame
            self._cuenta = cuenta

    def _anotar(self, frame, dets, cuenta):
        for d in dets:
            x1, y1, x2, y2 = map(int, d["bbox"])
            tid, conf = d["id"], d["conf"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, 2)
            lbl = f"#{tid} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(lbl, FUENTE, 0.50, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), self.color, -1)
            cv2.putText(frame, lbl, (x1 + 2, y1 - 4),
                        FUENTE, 0.50, (0, 0, 0), 1, cv2.LINE_AA)

        # Barra superior
        cv2.rectangle(frame, (0, 0), (ANCHO_CAM, 30), (20, 20, 20), -1)
        cv2.putText(frame, f"  {self.lugar}  |  Personas: {cuenta}",
                    (6, 21), FUENTE, 0.60, self.color, 1, cv2.LINE_AA)

        # Barra de progreso inferior
        prog = int(ANCHO_CAM * self.frame_num / max(self.total_frames, 1))
        cv2.rectangle(frame, (0, ALTO_CAM - 4), (prog, ALTO_CAM), self.color, -1)
        return frame

    def run(self):
        # ★ Modelo se crea AQUÍ dentro del hilo — completamente aislado ★
        print(f"[{self.camara_id}] Cargando modelo...")
        modelo = YOLO(MODELO_YOLO)
        print(f"[{self.camara_id}] Modelo listo.")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[{self.camara_id}] ERROR: No se pudo abrir {self.video_path}")
            self.terminado = True
            return

        self.fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ancho_orig        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto_orig         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[{self.camara_id}] {ancho_orig}x{alto_orig} | "
              f"{self.fps:.1f} fps | {self.total_frames} frames")

        while not DETENER.is_set():
            ret, frame = cap.read()
            if not ret:
                print(f"[{self.camara_id}] Fin del video.")
                break

            self.frame_num += 1
            ts = time.time()

            # Redimensionar al tamaño del mosaico
            frame_r = cv2.resize(frame, (ANCHO_CAM, ALTO_CAM))

            # Inferencia YOLO (modelo propio de este hilo)
            try:
                res = modelo.track(
                    frame_r,
                    persist=True,
                    classes=[0],
                    conf=CONFIANZA,
                    iou=IOU,
                    verbose=False,
                    tracker="bytetrack.yaml",
                )
            except Exception as e:
                print(f"[{self.camara_id}] Error inferencia: {e}")
                continue

            dets = []
            if res and res[0].boxes is not None:
                boxes = res[0].boxes
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    tid  = int(boxes.id[i].cpu().numpy()) if boxes.id is not None else i
                    dets.append({"bbox": bbox, "conf": conf, "id": tid})

            cuenta  = len(dets)
            ids_str = ",".join(str(d["id"]) for d in dets) or "ninguno"

            # Registro CSV cada INTERVALO_LOG segundos
            if ts - self._ultimo_log >= INTERVALO_LOG:
                self._ultimo_log = ts
                dt = datetime.now()
                self.csv_writer.escribir({
                    "fecha":          dt.strftime("%Y-%m-%d"),
                    "hora":           dt.strftime("%H:%M:%S"),
                    "timestamp_unix": f"{ts:.3f}",
                    "camara_id":      self.camara_id,
                    "lugar":          self.lugar,
                    "total_personas": cuenta,
                    "ids_detectados": ids_str,
                    "frame_numero":   self.frame_num,
                    "fps_video":      f"{self.fps:.2f}",
                })

            frame_anot = self._anotar(frame_r.copy(), dets, cuenta)
            self._put_frame(frame_anot, cuenta)

        cap.release()
        self.terminado = True
        print(f"[{self.camara_id}] Hilo terminado.")


# ─────────────────────────────────────────────
#  MOSAICO  (solo se arma en el hilo principal)
# ─────────────────────────────────────────────
def construir_mosaico(workers):
    frames, cuentas = [], []
    for w in workers:
        f, c = w.get_frame()
        frames.append(f)
        cuentas.append(c)

    # Fila 1: CAM-01 | sep | CAM-02
    sep_v  = np.full((ALTO_CAM, 4, 3), 60, dtype=np.uint8)
    fila1  = np.hstack([frames[0], sep_v, frames[1]])
    W      = fila1.shape[1]   # ancho real incluyendo separador

    # Fila 2: CAM-03 centrada
    pad_l  = (W - ANCHO_CAM) // 2
    pad_r  = W - ANCHO_CAM - pad_l
    fila2  = np.hstack([
        np.zeros((ALTO_CAM, pad_l, 3), dtype=np.uint8),
        frames[2],
        np.zeros((ALTO_CAM, pad_r, 3), dtype=np.uint8),
    ])

    sep_h  = np.full((4, W, 3), 60, dtype=np.uint8)

    # Barra de estado
    total  = sum(cuentas)
    ahora  = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    barra  = np.zeros((36, W, 3), dtype=np.uint8)
    txt    = f"  TOTAL PERSONAS: {total}   |   {ahora}   |   Q = salir"
    cv2.putText(barra, txt, (10, 25), FUENTE, 0.60,
                (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([fila1, sep_h, fila2, sep_h, barra])


# ─────────────────────────────────────────────
#  RESUMEN FINAL
# ─────────────────────────────────────────────
def resumen(ruta):
    try:
        df = pd.read_csv(ruta)
        print("\n" + "=" * 55)
        print("  RESUMEN FINAL")
        print("=" * 55)
        for lugar, g in df.groupby("lugar"):
            print(f"\n  {lugar}")
            print(f"    Registros     : {len(g)}")
            print(f"    Max personas  : {g['total_personas'].max()}")
            print(f"    Promedio      : {g['total_personas'].mean():.1f}")
            print(f"    Inicio / Fin  : {g['hora'].iloc[0]}  →  {g['hora'].iloc[-1]}")
        print("\n" + "=" * 55)
        print(f"  CSV → {ruta}")
        print("=" * 55 + "\n")
    except Exception as e:
        print(f"[RESUMEN] {e}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "=" * 55)
    print("  SISTEMA CCTV — 3 CAMARAS")
    print("=" * 55)
    for v in VIDEOS:
        print(f"  {v['id']} → {v['lugar']}")
    print(f"  Modelo : {MODELO_YOLO}  |  Conf: {CONFIANZA}")
    print("=" * 55 + "\n")

    csv_w   = CSVWriter(ARCHIVO_CSV)
    workers = [WorkerCamara(cfg, idx, csv_w) for idx, cfg in enumerate(VIDEOS)]

    # Esperar que los 3 modelos estén listos antes de abrir ventana
    print("[SISTEMA] Iniciando los 3 hilos (cada uno carga su modelo)...")
    for w in workers:
        w.start()

    # Pequeña espera para que los primeros frames lleguen
    time.sleep(3)

    ventana = "CCTV — Deteccion de Personas  (Q para salir)"
    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ventana, ANCHO_CAM * 2 + 4, ALTO_CAM * 2 + 44)
    print("[VENTANA] Abierta. Presiona Q o ESC para salir.\n")

    while True:
        mosaico = construir_mosaico(workers)
        cv2.imshow(ventana, mosaico)

        tecla = cv2.waitKey(30) & 0xFF
        if tecla in (ord("q"), ord("Q"), 27):
            print("[SISTEMA] Detenido por el usuario.")
            DETENER.set()
            break

        if all(w.terminado for w in workers):
            print("[SISTEMA] Todos los videos terminaron.")
            break

    cv2.destroyAllWindows()
    for w in workers:
        w.join(timeout=5)
    resumen(ARCHIVO_CSV)


if __name__ == "__main__":
    main()
