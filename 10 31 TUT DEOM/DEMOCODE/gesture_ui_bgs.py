# gesture_ui_bgs.py — YOLO + BGS-ROI (single view, EN UI)
# Left panel: settings; Right big panel: detection only (no BGS mask view)
# BGS is used only for webcam/video to crop ROI; image runs full-frame.
#
# Requirements:
#   pip install -U ultralytics opencv-python opencv-contrib-python pillow
#
# Usage:
#   1) Put this file with bgs_roi_gsoc.py in the same folder
#   2) python gesture_ui_bgs.py
#   3) Choose model, select Webcam/Video/Image, press F5 to start, F6 to stop

from ultralytics import YOLO
import cv2, time
from collections import Counter, deque
from pathlib import Path
from tkinter import Tk, StringVar, IntVar, DoubleVar, BooleanVar, ttk, filedialog, Text, END, NORMAL, DISABLED
from PIL import Image, ImageTk

# Your BGS wrapper (the updated module you asked for)
from bgs_roi_gsoc import BGSROI, BGSParams


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO + BGS-ROI (Detection Only)")
        self.root.geometry("1180x820")

        # ====== runtime params ======
        self.model_path  = StringVar(value=r"E:/AMME5710 MP BGS/Handout/train/weights/best.pt")
        self.source_mode = StringVar(value="webcam")  # webcam | video | image
        self.webcam_idx  = IntVar(value=0)
        self.video_path  = StringVar(value="")
        self.image_path  = StringVar(value="")
        self.conf_thr    = DoubleVar(value=0.35)
        self.iou_thr     = DoubleVar(value=0.50)
        self.vid_stride  = IntVar(value=1)
        self.smooth_N    = IntVar(value=7)
        self.use_bgs     = BooleanVar(value=True)   # applied only for webcam/video

        # BGS params (keep the essentials; others use defaults in BGSParams)
        self.bgs_warmup  = IntVar(value=30)
        self.bgs_minarea = IntVar(value=800)
        self.bgs_maxr    = DoubleVar(value=0.5)
        self.bgs_pad     = DoubleVar(value=0.30)
        self.bgs_ema     = DoubleVar(value=0.6)

        # ====== UI ======
        self._left_panel()
        self._display_panel()
        self._log_panel()

        # ====== state ======
        self.cap = None
        self.model = None
        self.bgs = None
        self.history = deque(maxlen=self.smooth_N.get())
        self.last_fps_t = time.time()
        self.fps = 0.0
        self.running = False

        # shortcuts
        self.root.bind("<F5>", lambda e: self.start())
        self.root.bind("<F6>", lambda e: self.stop())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------------- UI ----------------
    def _left_panel(self):
        lf = ttk.LabelFrame(self.root, text="Settings", padding=8)
        lf.place(x=10, y=10, width=360, height=600)

        ttk.Label(lf, text="Model (.pt)").grid(row=0, column=0, sticky="w")
        ttk.Entry(lf, textvariable=self.model_path, width=34).grid(row=1, column=0, columnspan=2, sticky="we")
        ttk.Button(lf, text="Browse…", command=self._browse_model, width=8).grid(row=1, column=2, padx=4)

        ttk.Label(lf, text="Source").grid(row=2, column=0, sticky="w", pady=(6,0))
        ttk.Radiobutton(lf, text="Webcam",     variable=self.source_mode, value="webcam").grid(row=3, column=0, sticky="w")
        ttk.Radiobutton(lf, text="Video file", variable=self.source_mode, value="video").grid(row=3, column=1, sticky="w")
        ttk.Radiobutton(lf, text="Image",      variable=self.source_mode, value="image").grid(row=3, column=2, sticky="w")

        ttk.Label(lf, text="Webcam index").grid(row=4, column=0, sticky="w")
        ttk.Entry(lf, textvariable=self.webcam_idx, width=8).grid(row=4, column=1, sticky="w")
        ttk.Button(lf, text="Probe", command=self._find_cam, width=6).grid(row=4, column=2, padx=4)

        ttk.Label(lf, text="Video path").grid(row=5, column=0, sticky="w", pady=(6,0))
        ttk.Entry(lf, textvariable=self.video_path, width=28).grid(row=6, column=0, columnspan=2, sticky="we")
        ttk.Button(lf, text="Choose…", command=self._browse_video, width=8).grid(row=6, column=2, padx=4)

        ttk.Label(lf, text="Image path").grid(row=7, column=0, sticky="w", pady=(6,0))
        ttk.Entry(lf, textvariable=self.image_path, width=28).grid(row=8, column=0, columnspan=2, sticky="we")
        ttk.Button(lf, text="Choose…", command=self._browse_image, width=8).grid(row=8, column=2, padx=4)

        ttk.Separator(lf).grid(row=9, column=0, columnspan=3, sticky="we", pady=8)

        ttk.Label(lf, text="YOLO conf").grid(row=10, column=0, sticky="w")
        ttk.Scale(lf, from_=0.05, to=0.95, variable=self.conf_thr, orient="horizontal").grid(row=10, column=1, columnspan=2, sticky="we")
        ttk.Label(lf, text="YOLO IoU").grid(row=11, column=0, sticky="w")
        ttk.Scale(lf, from_=0.10, to=0.90, variable=self.iou_thr, orient="horizontal").grid(row=11, column=1, columnspan=2, sticky="we")
        ttk.Label(lf, text="Video stride").grid(row=12, column=0, sticky="w")
        ttk.Entry(lf, textvariable=self.vid_stride, width=8).grid(row=12, column=1, sticky="w")
        ttk.Label(lf, text="Smoothing N").grid(row=13, column=0, sticky="w")
        ttk.Entry(lf, textvariable=self.smooth_N, width=8).grid(row=13, column=1, sticky="w")

        ttk.Separator(lf).grid(row=14, column=0, columnspan=3, sticky="we", pady=8)

        ttk.Checkbutton(
            lf,
            text="Enable BGS-ROI (Webcam/Video only)",
            variable=self.use_bgs
        ).grid(row=15, column=0, columnspan=3, sticky="w")

        ttk.Label(lf, text="BGS warmup").grid(row=16, column=0, sticky="w")
        ttk.Entry(lf, textvariable=self.bgs_warmup, width=8).grid(row=16, column=1, sticky="w")
        ttk.Label(lf, text="Min area").grid(row=17, column=0, sticky="w")
        ttk.Entry(lf, textvariable=self.bgs_minarea, width=8).grid(row=17, column=1, sticky="w")
        ttk.Label(lf, text="Max area ratio").grid(row=18, column=0, sticky="w")
        ttk.Entry(lf, textvariable=self.bgs_maxr, width=8).grid(row=18, column=1, sticky="w")
        ttk.Label(lf, text="Pad ratio").grid(row=19, column=0, sticky="w")
        ttk.Entry(lf, textvariable=self.bgs_pad, width=8).grid(row=19, column=1, sticky="w")
        ttk.Label(lf, text="EMA α").grid(row=20, column=0, sticky="w")
        ttk.Entry(lf, textvariable=self.bgs_ema, width=8).grid(row=20, column=1, sticky="w")

        btnf = ttk.Frame(lf); btnf.grid(row=21, column=0, columnspan=3, pady=10)
        ttk.Button(btnf, text="Start (F5)", command=self.start, width=14).grid(row=0, column=0, padx=6)
        ttk.Button(btnf, text="Stop (F6)",  command=self.stop,  width=14).grid(row=0, column=1, padx=6)

        for i in range(3):
            lf.columnconfigure(i, weight=1)

    def _display_panel(self):
        vf = ttk.LabelFrame(self.root, text="Detection", padding=6)
        vf.place(x=380, y=10, width=780, height=600)
        self.lbl_det = ttk.Label(vf)
        self.lbl_det.pack(side="left", fill="both", expand=True)

    def _log_panel(self):
        lf2 = ttk.LabelFrame(self.root, text="Console", padding=6)
        lf2.place(x=10, y=620, width=1150, height=180)
        self.log = Text(lf2, height=8, wrap="word")
        self.log.pack(fill="both", expand=True)
        self._log("Tip: BGS-ROI is used to crop the region for YOLO on webcam/video. Images run full-frame.")

    # -------------- interactions --------------
    def _browse_model(self):
        p = filedialog.askopenfilename(title="Choose YOLO weights (.pt)",
                                       filetypes=[("PyTorch Weights","*.pt"),("All files","*.*")])
        if p: self.model_path.set(p)

    def _browse_video(self):
        p = filedialog.askopenfilename(title="Choose video file",
                                       filetypes=[("Video","*.mp4;*.avi;*.mov;*.mkv"),("All files","*.*")])
        if p: self.video_path.set(p)

    def _browse_image(self):
        p = filedialog.askopenfilename(title="Choose image",
                                       filetypes=[("Image","*.jpg;*.jpeg;*.png;*.bmp"),("All files","*.*")])
        if p: self.image_path.set(p)

    def _find_cam(self):
        self._log("Probing webcam indexes 0..5:")
        ok_idx = None
        for i in range(6):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if not (cap and cap.isOpened()):
                cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
            if not (cap and cap.isOpened()):
                cap = cv2.VideoCapture(i)
            ok = cap and cap.isOpened()
            self._log(f"index {i}: {'OK' if ok else 'FAIL'}")
            if ok and ok_idx is None: ok_idx = i
            if ok: cap.release()
        if ok_idx is not None:
            self.webcam_idx.set(ok_idx); self._log(f"Using index {ok_idx}")
        else:
            self._log("No usable webcam found.")

    def _log(self, s: str):
        self.log.configure(state=NORMAL)
        self.log.insert(END, s + "\n")
        self.log.see(END)
        self.log.configure(state=DISABLED)

    # -------------- start/stop --------------
    def start(self):
        if self.running: return

        weights = Path(self.model_path.get())
        if not weights.exists():
            self._log(f"Model not found: {weights}"); return

        # Load model
        self._log(f"Loading model: {weights}")
        self.model = YOLO(str(weights))
        try:
            import torch
            if torch.cuda.is_available():
                self.model.to("cuda"); self._log("GPU enabled")
            else:
                self._log("CPU mode")
        except Exception as e:
            self._log(f"CUDA probe warning: {e}")

        mode = self.source_mode.get()

        if mode == "webcam":
            idx = int(self.webcam_idx.get())
            self._log(f"Opening webcam {idx} (DSHOW/MSMF/fallback)")
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not (cap and cap.isOpened()):
                cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
            if not (cap and cap.isOpened()):
                cap = cv2.VideoCapture(idx)
            if not (cap and cap.isOpened()):
                self._log("Cannot open webcam"); return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self._log("Webcam opened")
            self.cap = cap

        elif mode == "video":
            vp = self.video_path.get()
            if not Path(vp).exists():
                self._log("Please choose a valid video"); return
            self.cap = cv2.VideoCapture(vp)
            self._log(f"Video opened: {vp}")

        else:  # image
            ip = self.image_path.get()
            if not Path(ip).exists():
                self._log("Please choose a valid image"); return
            self._run_image_once(ip)
            return  # no loop for image

        # configure BGS (only for webcam/video)
        if self.use_bgs.get():
            self.bgs = BGSROI(BGSParams(
                warmup=int(self.bgs_warmup.get()),
                # keep defaults for GSOC pipeline; ROI/crop params below:
                min_area=int(self.bgs_minarea.get()),
                max_area_ratio=float(self.bgs_maxr.get()),
                pad_ratio=float(self.bgs_pad.get()),
                ema_alpha=float(self.bgs_ema.get()),
                # ensure BGS is ROI-only helper (memory off):
                decay=0.0
            ))
            self._log("BGS-ROI enabled (webcam/video)")
        else:
            self.bgs = None
            self._log("BGS-ROI disabled (full-frame YOLO)")

        # reset runtime stats
        self.history = deque(maxlen=max(1, int(self.smooth_N.get())))
        self.last_fps_t = time.time(); self.fps = 0.0
        self.running = True
        self._loop()

    def stop(self):
        self.running = False
        if self.cap is not None:
            try: self.cap.release()
            except: pass
            self.cap = None
        self._log("Stopped.")

    # -------------- main loop --------------
    def _loop(self):
        if not self.running: return
        ok, frame = self.cap.read()
        if not ok:
            self._log("End of stream / read error"); self.stop(); return

        # video stride (skip frames)
        do_infer = True
        if self.source_mode.get() == "video":
            stride = max(1, int(self.vid_stride.get()))
            cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            do_infer = (cur % stride == 0)

        # ----- BGS ROI (only webcam/video) -----
        roi = None
        if self.bgs is not None:
            roi, _vis, _mask = self.bgs.apply(frame)
        # select region for inference
        if roi is not None:
            x1, y1, x2, y2 = roi
            infer_img = frame[y1:y2, x1:x2]
            offx, offy = x1, y1
        else:
            infer_img = frame
            offx, offy = 0, 0

        # ----- YOLO inference -----
        names = {}
        counts = Counter()
        top_label = None
        if do_infer:
            results = self.model(infer_img, conf=float(self.conf_thr.get()),
                                 iou=float(self.iou_thr.get()), verbose=False)
            for r in results:
                names = r.names
                for b in r.boxes:
                    cls = int(b.cls.item()); conf = float(b.conf.item())
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    X1, Y1, X2, Y2 = x1+offx, y1+offy, x2+offx, y2+offy
                    cv2.rectangle(frame, (X1, Y1), (X2, Y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{names[cls]} {conf:.2f}", (X1, max(20, Y1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    counts[names[cls]] += 1
                    if top_label is None or conf > top_label[1]:
                        top_label = (names[cls], conf)

        if top_label:
            self.history.append(top_label[0])
        vote = Counter(self.history).most_common(1)[0][0] if self.history else "-"

        # FPS
        now = time.time()
        dt = now - self.last_fps_t
        if dt > 0:
            self.fps = 0.9*self.fps + 0.1*(1.0/dt)
        self.last_fps_t = now

        # info bar
        info = f"BGS={'ON' if self.bgs is not None else 'OFF'}  conf={self.conf_thr.get():.2f}  iou={self.iou_thr.get():.2f}  FPS≈{self.fps:4.1f}"
        cv2.rectangle(frame, (0,0), (max(360, len(info)*9), 28), (0,0,0), -1)
        cv2.putText(frame, info, (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        # show detection only
        self._show_img(self.lbl_det, frame, (750, 560))

        # concise log: count, labels, smoothed
        if do_infer:
            total = sum(counts.values())
            label_str = ", ".join(f"{k}:{v}" for k,v in counts.items()) if total else "-"
            self._log(f"count={total} | labels: {label_str} | smoothed: {vote}")

        self.root.after(1, self._loop)

    # -------------- image one-shot --------------
    def _run_image_once(self, image_path: str):
        self._log(f"Image opened: {image_path}")
        frame = cv2.imread(image_path)
        if frame is None:
            self._log("Failed to read image."); return

        # full-frame inference (no BGS on single image)
        results = self.model(frame, conf=float(self.conf_thr.get()),
                             iou=float(self.iou_thr.get()), verbose=False)

        names = {}
        counts = Counter()
        top_label = None
        for r in results:
            names = r.names
            for b in r.boxes:
                cls = int(b.cls.item()); conf = float(b.conf.item())
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{names[cls]} {conf:.2f}", (x1, max(20, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                counts[names[cls]] += 1
                if top_label is None or conf > top_label[1]:
                    top_label = (names[cls], conf)

        if top_label:
            self.history.append(top_label[0])
        vote = Counter(self.history).most_common(1)[0][0] if self.history else "-"

        # show image
        self._show_img(self.lbl_det, frame, (750, 560))

        # concise log
        total = sum(counts.values())
        label_str = ", ".join(f"{k}:{v}" for k,v in counts.items()) if total else "-"
        self._log(f"[IMAGE] count={total} | labels: {label_str} | smoothed: {vote}")

    # -------------- utils --------------
    def _show_img(self, label, bgr, size):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize(size)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)

    def on_close(self):
        self.stop()
        self.root.destroy()


def main():
    root = Tk()
    try:
        style = ttk.Style(root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
