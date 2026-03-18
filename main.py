import os
import re
import json
import platform
import subprocess
import shutil
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QFileDialog, QLabel, QMessageBox, QComboBox, QGroupBox,
    QListWidget, QProgressBar, QCheckBox, QInputDialog, QSlider,
    QListWidgetItem
)
from PyQt6.QtGui import QIcon, QDragEnterEvent, QDropEvent, QColor, QAction
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl, QTimer
import sys

# Try to import multimedia for preview/playback
try:
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
    HAS_MULTIMEDIA = True
except ImportError:
    HAS_MULTIMEDIA = False

# ---------- CONFIG ----------
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".aac", ".m4a")

# ---------- THEMES ----------
DARK_THEME = {
    "window_bg": "#0a1a3a",
    "frame_bg": "#102050",
    "input_bg": "#0d1a35",
    "text_color": "white",
    "secondary_text": "#bbb",
    "accent_text": "#ccc",
    "log_bg": "#0a1228",
    "log_text": "#0f0",
    "btn_bg": "#1a3060",
    "btn_run_bg": "#3fa34d",
    "btn_run_hover": "#4ab85a",
    "btn_cancel_bg": "#c0392b",
    "progress_bg": "#0d1a35",
    "progress_chunk": "#3fa34d",
    "explanation_text": "#aaa",
    "border_color": "#1a3060",
    "menubar_bg": "#0d1530",
    "statusbar_bg": "#0a1228",
}
LIGHT_THEME = {
    "window_bg": "#e8edf2",
    "frame_bg": "#ffffff",
    "input_bg": "#f5f5f5",
    "text_color": "#222",
    "secondary_text": "#555",
    "accent_text": "#333",
    "log_bg": "#ffffff",
    "log_text": "#222",
    "btn_bg": "#d0d0d0",
    "btn_run_bg": "#27ae60",
    "btn_run_hover": "#2ecc71",
    "btn_cancel_bg": "#e74c3c",
    "progress_bg": "#ddd",
    "progress_chunk": "#27ae60",
    "explanation_text": "#666",
    "border_color": "#ccc",
    "menubar_bg": "#f0f0f0",
    "statusbar_bg": "#ddd",
}


def load_config():
    defaults = {
        "default_input_dir": os.path.expanduser("~"),
        "default_output_dir": os.path.expanduser("~"),
        "theme": "dark",
        "use_gpu": True,
        "output_format": "flac",
        "presets": {},
    }
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            defaults.update(json.load(f))
    return defaults


def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)


def check_dependency(name):
    return shutil.which(name) is not None


def check_cuda():
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip() == "True"
    except Exception:
        return False


# ===================== WORKER THREAD =====================
class DemucsWorker(QThread):
    progress_text = pyqtSignal(str)
    file_started = pyqtSignal(int, str)      # (index, filename)
    file_finished = pyqtSignal(int, str)     # (index, status: "done"/"failed")
    stem_progress = pyqtSignal(int)          # 0-100
    all_done = pyqtSignal(str)               # last output folder path
    error = pyqtSignal(str)

    def __init__(self, files, output_dir, model, output_format, use_gpu):
        super().__init__()
        self.files = files
        self.output_dir = output_dir
        self.model = model
        self.output_format = output_format
        self.use_gpu = use_gpu
        self._cancelled = False
        self._process = None

    def cancel(self):
        self._cancelled = True
        if self._process and self._process.poll() is None:
            self._process.terminate()

    def run(self):
        last_folder = self.output_dir

        for idx, file_path in enumerate(self.files):
            if self._cancelled:
                self.progress_text.emit("Cancelled by user.")
                return

            song_name = os.path.splitext(os.path.basename(file_path))[0]
            self.file_started.emit(idx, song_name)
            self.stem_progress.emit(0)
            self.progress_text.emit(f"\n▶ Splitting: {song_name} (model: {self.model})")

            # 1 — RUN DEMUCS
            cmd = ["demucs", "-n", self.model, "-o", self.output_dir]
            if not self.use_gpu:
                cmd += ["--device", "cpu"]
            cmd.append(file_path)

            try:
                self._process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                while True:
                    if self._cancelled:
                        self._process.terminate()
                        self.progress_text.emit("Cancelled by user.")
                        return
                    line = self._process.stderr.readline()
                    if not line and self._process.poll() is not None:
                        break
                    if line:
                        match = re.search(r"(\d+)%", line)
                        if match:
                            self.stem_progress.emit(int(match.group(1)))
                        self.progress_text.emit(line.strip())

                if self._process.returncode != 0:
                    self.progress_text.emit(f"❌ Demucs failed for {song_name}")
                    self.file_finished.emit(idx, "failed")
                    continue
            except Exception as e:
                self.progress_text.emit(f"❌ Error: {e}")
                self.file_finished.emit(idx, "failed")
                continue

            self.stem_progress.emit(50)

            # 2 — Locate output
            model_folder = os.path.join(self.output_dir, self.model)
            song_folder = os.path.join(model_folder, song_name)

            if not os.path.exists(song_folder):
                self.progress_text.emit(f"❌ ERROR: Demucs output missing for {song_name}")
                self.file_finished.emit(idx, "failed")
                continue

            self.progress_text.emit(f"✔ Found split folder: {song_folder}")

            # 3 — Create final folder
            final_folder = os.path.join(self.output_dir, song_name)
            if os.path.exists(final_folder):
                shutil.rmtree(final_folder)
            os.makedirs(final_folder)

            # 4 — Move WAV files
            for f in os.listdir(song_folder):
                shutil.move(os.path.join(song_folder, f), os.path.join(final_folder, f))

            # 5 — Remove model folder
            shutil.rmtree(model_folder, ignore_errors=True)

            # 6 — Convert from WAV to chosen format
            wav_files = [f for f in os.listdir(final_folder) if f.endswith(".wav")]
            for i, f in enumerate(wav_files):
                if self._cancelled:
                    self.progress_text.emit("Cancelled by user.")
                    return

                wav = os.path.join(final_folder, f)

                if self.output_format == "wav":
                    self.progress_text.emit(f"✔ Keeping {f} as WAV")
                else:
                    out_ext = self.output_format
                    out_file = os.path.join(final_folder, f.replace(".wav", f".{out_ext}"))

                    ffmpeg_cmd = ["ffmpeg", "-y", "-i", wav]
                    if self.output_format == "mp3":
                        ffmpeg_cmd += ["-b:a", "320k"]
                    elif self.output_format == "ogg":
                        ffmpeg_cmd += ["-c:a", "libvorbis", "-q:a", "6"]
                    ffmpeg_cmd.append(out_file)

                    self.progress_text.emit(f"→ Converting {f} → {out_ext.upper()}")
                    subprocess.run(ffmpeg_cmd, capture_output=True)

                    if os.path.exists(out_file):
                        os.remove(wav)
                        self.progress_text.emit(f"✔ {f} → {os.path.basename(out_file)}")
                    else:
                        self.progress_text.emit(f"❌ Conversion failed for: {f}")

                progress = 50 + int(50 * (i + 1) / len(wav_files))
                self.stem_progress.emit(progress)

            last_folder = final_folder
            self.progress_text.emit(f"✔ Finished: {song_name}")
            self.file_finished.emit(idx, "done")

        self.stem_progress.emit(100)
        self.progress_text.emit("\n🎉 ALL DONE ✔")
        self.all_done.emit(last_folder)


# ===================== MAIN GUI =====================
class DemucsGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.config = load_config()
        self.dark_mode = self.config.get("theme", "dark") == "dark"
        self.cuda_available = False
        self.worker = None
        self.last_output_folder = None

        self.setWindowTitle("Demucs Stem Splitter")
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.setMinimumWidth(1100)
        self.setAcceptDrops(True)

        self.default_input_dir = self.config["default_input_dir"]
        self.default_output_dir = self.config["default_output_dir"]
        self.input_files = []
        self.output_dir = self.default_output_dir

        # ==================== MENU BAR ====================
        menubar = self.menuBar()

        # -- File menu --
        file_menu = menubar.addMenu("File")

        act_open = QAction("Open Files...", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self.choose_files)
        file_menu.addAction(act_open)

        act_output = QAction("Set Output Folder...", self)
        act_output.triggered.connect(self.choose_output)
        file_menu.addAction(act_output)

        file_menu.addSeparator()

        act_exit = QAction("Exit", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # -- Presets menu --
        presets_menu = menubar.addMenu("Presets")

        act_save_preset = QAction("Save Preset...", self)
        act_save_preset.triggered.connect(self.save_preset)
        presets_menu.addAction(act_save_preset)

        self.load_preset_menu = presets_menu.addMenu("Load")
        self.delete_preset_menu = presets_menu.addMenu("Delete")
        self._refresh_presets()

        # -- View menu --
        view_menu = menubar.addMenu("View")
        self.dark_mode_action = QAction("Dark Mode", self)
        self.dark_mode_action.setCheckable(True)
        self.dark_mode_action.setChecked(self.dark_mode)
        self.dark_mode_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(self.dark_mode_action)

        view_menu.addSeparator()

        act_open_folder = QAction("Open Output Folder", self)
        act_open_folder.triggered.connect(self.open_output_folder)
        view_menu.addAction(act_open_folder)

        # -- Help menu --
        help_menu = menubar.addMenu("Help")

        act_check_deps = QAction("Check Dependencies", self)
        act_check_deps.triggered.connect(self._check_dependencies)
        help_menu.addAction(act_check_deps)

        act_about = QAction("About", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

        # ==================== STATUS BAR ====================
        self.gpu_status_label = QLabel("GPU: checking...")
        self.statusBar().addPermanentWidget(self.gpu_status_label)
        self.statusBar().showMessage("Ready")

        # ==================== CENTRAL WIDGET ====================
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setSpacing(10)
        outer.setContentsMargins(12, 12, 12, 12)

        # ============ TWO-COLUMN LAYOUT ============
        columns = QHBoxLayout()
        columns.setSpacing(12)

        # -------- LEFT COLUMN: Setup (Steps 1-3) --------
        left_col = QVBoxLayout()
        left_col.setSpacing(10)

        # STEP 1 — SELECT FILES
        step1 = QGroupBox("Step 1 — Select Audio Files")
        step1_layout = QVBoxLayout(step1)

        self.lbl_files = QLabel("No files selected.")
        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(100)

        file_btn_row = QHBoxLayout()
        self.btn_select_files = QPushButton("Select Files...")
        self.btn_select_files.clicked.connect(self.choose_files)
        self.btn_remove_file = QPushButton("Remove Selected")
        self.btn_remove_file.clicked.connect(self.remove_selected_file)
        file_btn_row.addWidget(self.btn_select_files)
        file_btn_row.addWidget(self.btn_remove_file)

        self.drag_hint = QLabel("Tip: You can also drag & drop audio files here")

        step1_layout.addWidget(self.lbl_files)
        step1_layout.addWidget(self.file_list)
        step1_layout.addLayout(file_btn_row)
        step1_layout.addWidget(self.drag_hint)
        left_col.addWidget(step1)

        # STEP 2 — OUTPUT FOLDER
        step2 = QGroupBox("Step 2 — Output Folder")
        step2_layout = QVBoxLayout(step2)

        self.lbl_output = QLabel(self.output_dir)
        self.lbl_output.setWordWrap(True)

        out_btn_row = QHBoxLayout()
        btn_change_output = QPushButton("Change Folder")
        btn_change_output.clicked.connect(self.choose_output)
        self.btn_open_folder = QPushButton("Open in File Manager")
        self.btn_open_folder.clicked.connect(self.open_output_folder)
        out_btn_row.addWidget(btn_change_output)
        out_btn_row.addWidget(self.btn_open_folder)

        step2_layout.addWidget(self.lbl_output)
        step2_layout.addLayout(out_btn_row)
        left_col.addWidget(step2)

        # STEP 3 — SETTINGS
        step3 = QGroupBox("Step 3 — Settings")
        step3_layout = QVBoxLayout(step3)

        # Model
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_box = QComboBox()
        self.model_box.addItem("htdemucs — Balanced (pop, EDM, most music)", "htdemucs")
        self.model_box.addItem("mdx_extra_q — Best vocals (lyric-heavy)", "mdx_extra_q")
        self.model_box.addItem("mdx_extra — Instrumentals only (no vocals)", "mdx_extra")
        self.model_box.addItem("mdx — Fast (lower quality)", "mdx")
        model_row.addWidget(self.model_box, stretch=1)
        step3_layout.addLayout(model_row)

        self.explanation = QLabel(
            "<b>htdemucs</b> = balanced | "
            "<b>mdx_extra_q</b> = best vocals | "
            "<b>mdx_extra</b> = instrumentals | "
            "<b>mdx</b> = fast"
        )
        step3_layout.addWidget(self.explanation)

        # Format + GPU
        fmt_gpu_row = QHBoxLayout()
        fmt_gpu_row.addWidget(QLabel("Format:"))
        self.format_box = QComboBox()
        self.format_box.addItem("FLAC (lossless)", "flac")
        self.format_box.addItem("WAV (lossless)", "wav")
        self.format_box.addItem("MP3 320kbps", "mp3")
        self.format_box.addItem("OGG Vorbis", "ogg")

        saved_fmt = self.config.get("output_format", "flac")
        for i in range(self.format_box.count()):
            if self.format_box.itemData(i) == saved_fmt:
                self.format_box.setCurrentIndex(i)
                break

        fmt_gpu_row.addWidget(self.format_box, stretch=1)
        fmt_gpu_row.addSpacing(20)

        self.gpu_checkbox = QCheckBox("Use GPU (CUDA)")
        self.gpu_checkbox.setChecked(self.config.get("use_gpu", True))
        fmt_gpu_row.addWidget(self.gpu_checkbox)

        step3_layout.addLayout(fmt_gpu_row)
        left_col.addWidget(step3)

        left_col.addStretch()
        columns.addLayout(left_col, stretch=1)

        # -------- RIGHT COLUMN: Process + Preview + Log --------
        right_col = QVBoxLayout()
        right_col.setSpacing(10)

        # STEP 4 — PROCESS
        step4 = QGroupBox("Step 4 — Process")
        step4_layout = QVBoxLayout(step4)

        self.batch_label = QLabel("Select files and click Start Processing")
        step4_layout.addWidget(self.batch_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumHeight(28)
        step4_layout.addWidget(self.progress_bar)

        # BIG RUN BUTTON
        self.btn_run = QPushButton("Start Processing")
        self.btn_run.setMinimumHeight(60)
        self.btn_run.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run.clicked.connect(self.run_demucs)
        step4_layout.addWidget(self.btn_run)

        # Cancel + Reset row
        action_row = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.cancel_processing)
        self.btn_cancel.setVisible(False)
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_all)
        action_row.addWidget(self.btn_cancel)
        action_row.addWidget(self.btn_reset)
        step4_layout.addLayout(action_row)

        right_col.addWidget(step4)

        # PREVIEW / PLAYBACK
        if HAS_MULTIMEDIA:
            preview_box = QGroupBox("Preview / Playback")
            preview_layout = QVBoxLayout(preview_box)

            stem_row = QHBoxLayout()
            stem_row.addWidget(QLabel("Stem:"))
            self.stem_combo = QComboBox()
            self.stem_combo.setMinimumWidth(150)
            stem_row.addWidget(self.stem_combo, stretch=1)

            self.btn_play = QPushButton("▶ Play")
            self.btn_play.clicked.connect(self.play_stem)
            self.btn_pause = QPushButton("⏸ Pause")
            self.btn_pause.clicked.connect(self.pause_stem)
            self.btn_stop = QPushButton("⏹ Stop")
            self.btn_stop.clicked.connect(self.stop_stem)
            stem_row.addWidget(self.btn_play)
            stem_row.addWidget(self.btn_pause)
            stem_row.addWidget(self.btn_stop)
            preview_layout.addLayout(stem_row)

            seek_row = QHBoxLayout()
            self.seek_slider = QSlider(Qt.Orientation.Horizontal)
            self.seek_slider.setRange(0, 0)
            self.seek_slider.sliderMoved.connect(self.seek_stem)
            self.time_label = QLabel("0:00 / 0:00")
            seek_row.addWidget(self.seek_slider, stretch=1)
            seek_row.addWidget(self.time_label)
            preview_layout.addLayout(seek_row)

            vol_row = QHBoxLayout()
            vol_row.addWidget(QLabel("Volume:"))
            self.vol_slider = QSlider(Qt.Orientation.Horizontal)
            self.vol_slider.setRange(0, 100)
            self.vol_slider.setValue(80)
            self.vol_slider.valueChanged.connect(self._on_volume_changed)
            self.vol_label = QLabel("80%")
            vol_row.addWidget(self.vol_slider, stretch=1)
            vol_row.addWidget(self.vol_label)
            preview_layout.addLayout(vol_row)

            right_col.addWidget(preview_box)
            self.preview_box = preview_box

            self.player = QMediaPlayer()
            self.audio_output = QAudioOutput()
            self.audio_output.setVolume(0.8)
            self.player.setAudioOutput(self.audio_output)
            self.player.positionChanged.connect(self._on_position_changed)
            self.player.durationChanged.connect(self._on_duration_changed)
        else:
            self.preview_box = None

        # LOG
        log_label = QLabel("Log Output")
        log_label.setProperty("log_title", True)
        right_col.addWidget(log_label)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(100)
        right_col.addWidget(self.log, stretch=1)

        columns.addLayout(right_col, stretch=1)
        outer.addLayout(columns, stretch=1)

        # ==================== APPLY THEME ====================
        self.apply_theme()

        # ==================== STARTUP CHECKS ====================
        self._check_dependencies()
        QTimer.singleShot(100, self._check_cuda_async)

    # ==================== THEME ====================
    def _get_theme(self):
        return DARK_THEME if self.dark_mode else LIGHT_THEME

    def apply_theme(self):
        t = self._get_theme()
        self.setStyleSheet(f"""
            QMainWindow {{
                background: {t['window_bg']};
            }}
            QWidget {{
                color: {t['text_color']};
                font-size: 14px;
            }}
            QMenuBar {{
                background: {t['menubar_bg']};
                color: {t['text_color']};
                font-size: 14px;
                padding: 2px;
            }}
            QMenuBar::item:selected {{
                background: {t['frame_bg']};
            }}
            QMenu {{
                background: {t['frame_bg']};
                color: {t['text_color']};
                border: 1px solid {t['border_color']};
            }}
            QMenu::item:selected {{
                background: {t['input_bg']};
            }}
            QGroupBox {{
                background: {t['frame_bg']};
                border: 1px solid {t['border_color']};
                border-radius: 8px;
                margin-top: 14px;
                padding: 18px 12px 12px 12px;
                font-size: 15px;
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                color: {t['text_color']};
                font-size: 15px;
            }}
            QComboBox {{
                padding: 8px;
                font-size: 14px;
                background: {t['input_bg']};
                color: {t['text_color']};
                border: 1px solid {t['border_color']};
                border-radius: 6px;
            }}
            QComboBox QAbstractItemView {{
                background: {t['input_bg']};
                color: {t['text_color']};
                selection-background-color: {t['frame_bg']};
            }}
            QPushButton {{
                padding: 10px 16px;
                font-size: 14px;
                background: {t['btn_bg']};
                color: {t['text_color']};
                border: 1px solid {t['border_color']};
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background: {t['frame_bg']};
            }}
            QListWidget {{
                background: {t['input_bg']};
                color: {t['text_color']};
                border: 1px solid {t['border_color']};
                border-radius: 6px;
                padding: 4px;
                font-size: 13px;
            }}
            QCheckBox {{
                font-size: 14px;
                color: {t['text_color']};
            }}
            QProgressBar {{
                background: {t['progress_bg']};
                border-radius: 6px;
                text-align: center;
                color: {t['text_color']};
                font-size: 13px;
                border: 1px solid {t['border_color']};
            }}
            QProgressBar::chunk {{
                background: {t['progress_chunk']};
                border-radius: 5px;
            }}
            QSlider::groove:horizontal {{
                background: {t['input_bg']};
                height: 6px;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {t['progress_chunk']};
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }}
            QStatusBar {{
                background: {t['statusbar_bg']};
                color: {t['text_color']};
                font-size: 13px;
            }}
        """)

        # Per-widget overrides
        self.lbl_output.setStyleSheet(f"""
            background: {t['input_bg']};
            color: {t['accent_text']};
            font-size: 13px;
            padding: 10px;
            border: 1px solid {t['border_color']};
            border-radius: 6px;
        """)
        self.drag_hint.setStyleSheet(f"color: {t['explanation_text']}; font-size: 12px; font-style: italic;")
        self.explanation.setStyleSheet(f"color: {t['explanation_text']}; font-size: 12px;")
        self.log.setStyleSheet(f"""
            background: {t['log_bg']};
            color: {t['log_text']};
            font-family: monospace;
            font-size: 13px;
            padding: 8px;
            border: 1px solid {t['border_color']};
            border-radius: 6px;
        """)
        self.batch_label.setStyleSheet(f"font-weight: bold; color: {t['text_color']};")

        # Run button — big and green
        self.btn_run.setStyleSheet(f"""
            QPushButton {{
                padding: 16px;
                font-size: 20px;
                font-weight: bold;
                background: {t['btn_run_bg']};
                color: white;
                border: none;
                border-radius: 10px;
            }}
            QPushButton:hover {{
                background: {t['btn_run_hover']};
            }}
            QPushButton:disabled {{
                background: #555;
                color: #999;
            }}
        """)
        self.btn_cancel.setStyleSheet(f"""
            QPushButton {{
                padding: 10px 16px;
                font-size: 14px;
                font-weight: bold;
                background: {t['btn_cancel_bg']};
                color: white;
                border: none;
                border-radius: 6px;
            }}
        """)

        # Log title label
        for lbl in self.findChildren(QLabel):
            if lbl.property("log_title"):
                lbl.setStyleSheet(f"font-weight: bold; font-size: 14px; color: {t['text_color']};")

    def toggle_theme(self):
        if isinstance(self.sender(), QAction):
            self.dark_mode = self.dark_mode_action.isChecked()
        else:
            self.dark_mode = not self.dark_mode
            self.dark_mode_action.setChecked(self.dark_mode)
        self.apply_theme()
        self.config["theme"] = "dark" if self.dark_mode else "light"
        save_config(self.config)

    # ==================== HELPERS ====================
    def _check_dependencies(self):
        missing = []
        if not check_dependency("demucs"):
            missing.append("demucs")
        if not check_dependency("ffmpeg"):
            missing.append("ffmpeg")
        if missing:
            msg = f"Missing required dependencies: {', '.join(missing)}\n\n"
            if "demucs" in missing:
                msg += "Install demucs: pip install demucs\n"
            if "ffmpeg" in missing:
                msg += "Install ffmpeg: sudo apt install ffmpeg\n"
            QMessageBox.warning(self, "Missing Dependencies", msg)
            self.log.append(f"⚠ Missing: {', '.join(missing)}")
        else:
            self.log.append("✔ All dependencies found (demucs, ffmpeg)")

    def _check_cuda_async(self):
        self.cuda_available = check_cuda()
        if self.cuda_available:
            self.gpu_status_label.setText("GPU: CUDA")
        else:
            self.gpu_checkbox.setChecked(False)
            self.gpu_checkbox.setEnabled(False)
            self.gpu_checkbox.setToolTip("CUDA not available (torch not found or no GPU)")
            self.gpu_status_label.setText("GPU: CPU only")

    def _show_about(self):
        QMessageBox.about(
            self, "About Demucs Stem Splitter",
            "Demucs Stem Splitter\n\n"
            "Split audio tracks into stems (vocals, drums, bass, other)\n"
            "using Facebook's Demucs AI model.\n\n"
            "Supports FLAC, WAV, MP3, and OGG output."
        )

    # ==================== DRAG & DROP ====================
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith(AUDIO_EXTENSIONS):
                    event.acceptProposedAction()
                    t = self._get_theme()
                    self.file_list.setStyleSheet(
                        f"background:{t['input_bg']}; color:{t['text_color']}; "
                        f"border: 2px dashed #3fa34d; border-radius:6px; padding:4px;"
                    )
                    return

    def dragLeaveEvent(self, event):
        self.apply_theme()

    def dropEvent(self, event: QDropEvent):
        files = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(AUDIO_EXTENSIONS):
                files.append(path)

        if files:
            self.input_files.extend(files)
            self.lbl_files.setText(f"{len(self.input_files)} file(s) selected ✔")
            for f in files:
                self.file_list.addItem(f)
                self.log.append(f" + Dropped: {f}")
        self.apply_theme()

    # ==================== FILE / FOLDER SELECTION ====================
    def remove_selected_file(self):
        row = self.file_list.currentRow()
        if row >= 0 and row < len(self.input_files):
            removed = self.input_files.pop(row)
            self.file_list.takeItem(row)
            self.log.append(f" - Removed: {os.path.basename(removed)}")
            if self.input_files:
                self.lbl_files.setText(f"{len(self.input_files)} file(s) selected ✔")
            else:
                self.lbl_files.setText("No files selected.")

    def choose_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio Files", self.default_input_dir,
            "Audio (*.mp3 *.wav *.flac *.aac *.m4a)"
        )
        if files:
            self.input_files = files
            self.lbl_files.setText(f"{len(files)} file(s) selected ✔")
            self.log.append(f"Loaded {len(files)} files:")
            self.file_list.clear()
            for f in files:
                self.file_list.addItem(f)
                self.log.append(f" - {f}")

    def choose_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose Output Folder", self.output_dir)
        if folder:
            self.output_dir = folder
            self.lbl_output.setText(folder)
            self.log.append(f"Output folder changed to: {folder}")

    def open_output_folder(self):
        path = self.last_output_folder or self.output_dir
        if not os.path.exists(path):
            path = self.output_dir
        system = platform.system()
        if system == "Linux":
            subprocess.Popen(["xdg-open", path])
        elif system == "Darwin":
            subprocess.Popen(["open", path])
        elif system == "Windows":
            os.startfile(path)

    def reset_all(self):
        self.input_files = []
        self.file_list.clear()
        self.lbl_files.setText("No files selected.")
        self.progress_bar.setValue(0)
        self.batch_label.setText("Select files and click Start Processing")
        self.log.clear()
        self.log.append("✔ Reset completed.")
        self.statusBar().showMessage("Ready")
        if HAS_MULTIMEDIA:
            self.stem_combo.clear()

    # ==================== PRESETS ====================
    def _refresh_presets(self):
        self.load_preset_menu.clear()
        self.delete_preset_menu.clear()
        presets = self.config.get("presets", {})
        if not presets:
            act_none1 = QAction("(no presets saved)", self)
            act_none1.setEnabled(False)
            self.load_preset_menu.addAction(act_none1)
            act_none2 = QAction("(no presets saved)", self)
            act_none2.setEnabled(False)
            self.delete_preset_menu.addAction(act_none2)
            return
        for name in presets:
            act_load = QAction(name, self)
            act_load.triggered.connect(lambda _, n=name: self.load_preset(n))
            self.load_preset_menu.addAction(act_load)

            act_del = QAction(name, self)
            act_del.triggered.connect(lambda _, n=name: self.delete_preset(n))
            self.delete_preset_menu.addAction(act_del)

    def save_preset(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if ok and name.strip():
            name = name.strip()
            self.config.setdefault("presets", {})[name] = {
                "model": self.model_box.currentData(),
                "format": self.format_box.currentData(),
                "use_gpu": self.gpu_checkbox.isChecked(),
            }
            save_config(self.config)
            self._refresh_presets()
            self.log.append(f"✔ Preset '{name}' saved.")

    def load_preset(self, name):
        presets = self.config.get("presets", {})
        if name not in presets:
            return
        p = presets[name]

        for i in range(self.model_box.count()):
            if self.model_box.itemData(i) == p.get("model"):
                self.model_box.setCurrentIndex(i)
                break
        for i in range(self.format_box.count()):
            if self.format_box.itemData(i) == p.get("format"):
                self.format_box.setCurrentIndex(i)
                break
        if self.gpu_checkbox.isEnabled():
            self.gpu_checkbox.setChecked(p.get("use_gpu", True))
        self.log.append(f"✔ Preset '{name}' loaded.")

    def delete_preset(self, name):
        presets = self.config.get("presets", {})
        if name in presets:
            del presets[name]
            save_config(self.config)
            self._refresh_presets()
            self.log.append(f"✔ Preset '{name}' deleted.")

    # ==================== PROCESSING ====================
    def run_demucs(self):
        if not self.input_files:
            QMessageBox.warning(self, "Error", "No audio files selected.")
            return

        model = self.model_box.currentData()
        fmt = self.format_box.currentData()
        use_gpu = self.gpu_checkbox.isChecked()

        # Update file list with pending status
        self.file_list.clear()
        for f in self.input_files:
            item = QListWidgetItem(f"⏳ {os.path.basename(f)}")
            item.setForeground(QColor("#999"))
            self.file_list.addItem(item)

        self.btn_run.setEnabled(False)
        self.btn_cancel.setVisible(True)
        self.progress_bar.setValue(0)
        self.batch_label.setText(f"Processing 0 of {len(self.input_files)}...")
        self.statusBar().showMessage(f"Processing 0 of {len(self.input_files)}...")

        self.worker = DemucsWorker(self.input_files, self.output_dir, model, fmt, use_gpu)
        self.worker.progress_text.connect(self.log.append)
        self.worker.file_started.connect(self._on_file_started)
        self.worker.file_finished.connect(self._on_file_finished)
        self.worker.stem_progress.connect(self.progress_bar.setValue)
        self.worker.all_done.connect(self._on_all_done)
        self.worker.start()

    def cancel_processing(self):
        if self.worker:
            self.worker.cancel()
            self.log.append("⏹ Cancelling...")
            self.btn_cancel.setVisible(False)
            self.btn_run.setEnabled(True)
            self.batch_label.setText("Cancelled.")
            self.statusBar().showMessage("Cancelled")

    def _on_file_started(self, idx, name):
        msg = f"Processing {idx + 1} of {len(self.input_files)}: {name}"
        self.batch_label.setText(msg)
        self.statusBar().showMessage(msg)
        if idx < self.file_list.count():
            item = self.file_list.item(idx)
            item.setText(f"🔄 {name}")
            item.setForeground(QColor("#f1c40f"))

    def _on_file_finished(self, idx, status):
        if idx < self.file_list.count():
            item = self.file_list.item(idx)
            name = os.path.basename(self.input_files[idx])
            if status == "done":
                item.setText(f"✔ {name}")
                item.setForeground(QColor("#2ecc71"))
            else:
                item.setText(f"❌ {name}")
                item.setForeground(QColor("#e74c3c"))

    def _on_all_done(self, last_folder):
        self.last_output_folder = last_folder
        self.btn_run.setEnabled(True)
        self.btn_cancel.setVisible(False)
        self.batch_label.setText("All done!")
        self.statusBar().showMessage("All done!")
        QMessageBox.information(self, "Done", "All tracks processed successfully!")

        # System notification
        try:
            system = platform.system()
            if system == "Linux":
                subprocess.Popen(["notify-send", "Demucs", "All tracks processed successfully!"])
            elif system == "Darwin":
                subprocess.Popen(["osascript", "-e",
                    'display notification "All tracks processed successfully!" with title "Demucs"'])
        except Exception:
            pass

        # Populate preview stems
        if HAS_MULTIMEDIA and last_folder and os.path.isdir(last_folder):
            self.stem_combo.clear()
            for f in sorted(os.listdir(last_folder)):
                full = os.path.join(last_folder, f)
                if os.path.isfile(full):
                    self.stem_combo.addItem(f, full)

    # ==================== PREVIEW / PLAYBACK ====================
    def play_stem(self):
        if not HAS_MULTIMEDIA:
            return
        path = self.stem_combo.currentData()
        if path and os.path.exists(path):
            self.player.setSource(QUrl.fromLocalFile(path))
            self.player.play()
            self.log.append(f"▶ Playing: {os.path.basename(path)}")

    def pause_stem(self):
        if HAS_MULTIMEDIA:
            self.player.pause()

    def stop_stem(self):
        if HAS_MULTIMEDIA:
            self.player.stop()

    def _on_volume_changed(self, value):
        if HAS_MULTIMEDIA:
            self.audio_output.setVolume(value / 100.0)
            self.vol_label.setText(f"{value}%")

    def seek_stem(self, position):
        if HAS_MULTIMEDIA:
            self.player.setPosition(position)

    def _on_position_changed(self, pos):
        self.seek_slider.setValue(pos)
        dur = self.player.duration()
        self.time_label.setText(f"{self._fmt_time(pos)} / {self._fmt_time(dur)}")

    def _on_duration_changed(self, dur):
        self.seek_slider.setRange(0, dur)

    @staticmethod
    def _fmt_time(ms):
        s = ms // 1000
        return f"{s // 60}:{s % 60:02d}"


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DemucsGUI()
    window.show()
    sys.exit(app.exec())
