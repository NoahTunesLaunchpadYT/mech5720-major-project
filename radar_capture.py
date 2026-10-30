"""
Radar capture + mask stepping for the coded-mask rig.

The radar streams continuously and the plots show the live amplitude vs distance
(200 mm - 2 m). Captures take the next FRAMES_PER_CAPTURE frames from that stream.

Masks are numbered 0..23 (mask m = columns m..m+3 of mask_pattern.csv).

  C / Enter        -> capture FRAMES_PER_CAPTURE radar frames at the current mask
  Right arrow      -> move the carriage one position (7 mm) away from the switch
  Left arrow       -> move the carriage one position towards the switch
  Space            -> stop the stepper (and abort a scan or calibration)
  "Scan all 24"    -> go to position 1, then capture + advance through every position
  "Advance after capture" moves to the next position once a single capture is saved.
  Mask 0..23 buttons -> go straight to that mask
  "Calibrate"      -> re-run the homing pass, then go to mask 23
  "Capture 24 mask ambient" -> same as Scan all 24, saved to captures/ambient/
                      (overwriting the previous ambient scan).
  "Subtract ambient" -> subtracts each mask's ambient mean from the live plots and
                      the scene map. Display only: saved captures stay raw. Enabled
                      once captures/ambient/scene_map.npz exists.
  Scene 00..88 buttons -> choose the scene folder that captures and scans save to,
                      and show that scene's map. Colour: grey = nothing captured,
                      orange = some masks, green = all 24.

Captures are saved to captures/<scene>/mask<NN>.npz (overwriting that mask's
previous capture); a completed scan also writes captures/<scene>/scene_map.npz,
the 24 x distance mean-amplitude map with row m = mask m. Each capture file holds:
  frames       complex array (frames, sweeps_per_frame, num_points)
  distances_m  range of each point
  distance_mm  carriage distance from the limit switch when the capture started
  position     carriage position 1..24 (1 = nearest the limit switch)
  mask_index   mask number 0..23
  frame_errors corrupted radar messages discarded during this capture
  mask         the 4x4 mask in front of the sensor
  sensor_config, server_info (JSON strings)

Setup: pip install -r requirements.txt, then
  python radar_capture.py                          (auto-detects both devices)
  python radar_capture.py --arduino COM12 --radar COM11
  python radar_capture.py --mock                   (fake radar data, for testing)

Notes:
  - Close the Arduino Serial Monitor and the Exploration Tool app first.
  - Opening the Arduino port resets it, so it re-homes when this app starts.
  - Uses the arduino/arduino.ino protocol: 'F'/'B' = one 7 mm move, 'a'..'x' = go to
    position 1..24, 'C' = re-home, 'S' = stop, 'H' = heartbeat.
"""

import argparse
import json
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import serial
import serial.tools.list_ports
import tkinter as tk
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from acconeer.exptool import a121
from acconeer.exptool._core.communication.message_stream import MessageStream


class FrameErrors:
    radar = 0
    arduino = 0


def robust_stream(self):
    """Replacement for MessageStream._get_stream that discards corrupted messages
    instead of disconnecting.

    Each message is a JSON header line, optionally followed by payload_size binary
    bytes. Payloads are read one byte long: in an intact stream that extra byte is
    the '{' opening the next header, so anything else means bytes were dropped.
    After an error, each line is searched for a '{"' that starts a valid header,
    since the next header usually arrives glued to leftover payload bytes.
    """
    carry = b""
    in_error = False
    while True:
        try:
            line = carry + self._link.recv_until(self.protocol.end_sequence)
        except Exception as e:
            self._error_callback(e)
        carry = b""
        try:
            start = 0
            while True:
                try:
                    header = json.loads(line[start:])
                    break
                except ValueError:
                    start = line.find(b'{"', start + 1)
                    if start < 0:
                        raise
            if start > 0 and not in_error:
                FrameErrors.radar += 1
            size = header.get("payload_size", 0)
            payload = b""
            if size:
                try:
                    data = self._link.recv(size + 1)
                except Exception as e:
                    self._error_callback(e)
                payload, carry = data[:-1], data[-1:]
                if carry != b"{":
                    carry = b""
                    raise ValueError("payload length mismatch")
            msg = self.protocol.parse_message(header, payload)
        except Exception:
            if not in_error:
                FrameErrors.radar += 1
            in_error = True
            continue
        in_error = False
        yield msg


MessageStream._get_stream = robust_stream

ARDUINO_LINE = re.compile(r"Voltage:-?\d+\.\d\d,Distance_mm:-?\d+\.\d\d")

BAUD = 115200
RADAR_BAUD = 1_000_000   # XM125 over CH340: 2 Mbaud (the default) drops bytes
ARDUINO_KEYWORDS = ("arduino", "usbmodem", "ttyacm")
RADAR_KEYWORDS = ("ch340", "wch")
UI_MS = 300              # status/label refresh interval
HEARTBEAT_S = 0.1        # sent from its own thread; the Arduino stops after 300 ms of silence
PLOT_MS = 25             # poll for new radar frames faster than the 20 fps frame rate
FRAMES_PER_CAPTURE = 20
SAVE_DIR = Path(__file__).parent / "captures"
AMBIENT_DIR = SAVE_DIR / "ambient"       # latest ambient scan; scene_map.npz is the reference
AMBIENT_FILE = AMBIENT_DIR / "scene_map.npz"
SCENES = [f"{r}{c}" for r in range(9) for c in range(9)]   # 00 .. 88
SCENE_NONE, SCENE_PARTIAL, SCENE_FULL = "#d9d9d9", "#ffc56b", "#8fd18f"

# Mask positions (must match arduino.ino)
START_OFFSET_MM = 3.9
STEP_MM = 7.0
NUM_POSITIONS = 24
ARRIVE_TOLERANCE_MM = 0.05
SETTLE_S = 0.3           # position must hold this long before a scan captures
MOVE_TIMEOUT_S = 60
HOMING_TIMEOUT_S = 90    # full 200 mm homing pass takes ~40 s
CALIBRATE_TO_MASK = 23   # where Calibrate leaves the carriage

# Mask pattern: 4 rows x 27 columns; mask m (0..23) is columns m .. m+3.
# MASK_REVERSED: carriage position 1 (nearest the switch) shows the last mask.
MASK = np.loadtxt(Path(__file__).parent / "mask_patterns" / "mask_impulse.csv",
                  delimiter=",", dtype=int)
MASK_REVERSED = True
MASK_ON = "#7fd4ff"      # 1s: light blue
MASK_OFF = "#000000"     # 0s: black

# Radar settings - tune for the experiment. 200 mm .. 2000 mm in 10 mm steps.
# Keep points x sweeps x frame_rate x 4 bytes well under ~80 kB/s at 1 Mbaud.
SENSOR_CONFIG = a121.SensorConfig(
    start_point=80,          # x 2.5 mm = 200 mm
    num_points=181,          # last point = 80 + 180 x 4 = 800 -> 2000 mm
    step_length=4,           # x 2.5 mm = 10 mm
    profile=a121.Profile.PROFILE_2,
    hwaas=16,
    sweeps_per_frame=4,
    frame_rate=15.0,
)
DISTANCES_M = (SENSOR_CONFIG.start_point
               + np.arange(SENSOR_CONFIG.num_points) * SENSOR_CONFIG.step_length) * 0.0025


def find_port(name, keywords):
    for p in serial.tools.list_ports.comports():
        text = f"{p.device} {p.description} {p.manufacturer}".lower()
        if any(k in text for k in keywords):
            return p.device
    names = ", ".join(p.device for p in serial.tools.list_ports.comports()) or "none found"
    sys.exit(f"Couldn't identify the {name}. Use --{name.lower()} <port>\nPorts: {names}")


def position_from_mm(mm):
    return int(round((mm - START_OFFSET_MM) / STEP_MM)) + 1


def mm_from_position(pos):
    return START_OFFSET_MM + (pos - 1) * STEP_MM


def mask_index(pos):
    """Mask number 0..23 (CSV column order) shown at carriage position pos (1..24)."""
    return NUM_POSITIONS - pos if MASK_REVERSED else pos - 1


def position_of_mask(m):
    return NUM_POSITIONS - m if MASK_REVERSED else m + 1


def mask_at(pos):
    col = mask_index(pos)
    return MASK[:, col:col + 4]


def load_scene(folder):
    """Scene map from the mask<NN>.npz captures in folder (NaN rows = not captured)."""
    stack = np.full((NUM_POSITIONS, len(DISTANCES_M)), np.nan)
    for f in folder.glob("mask*.npz"):
        with np.load(f) as d:
            stack[int(f.stem[4:])] = np.abs(d["frames"]).mean(axis=(0, 1))
    return stack


def scene_colour(scene):
    n = len(list((SAVE_DIR / scene).glob("mask*.npz")))
    return SCENE_FULL if n >= NUM_POSITIONS else SCENE_PARTIAL if n else SCENE_NONE


class ScanAborted(Exception):
    pass


class CaptureApp:
    def __init__(self, root, ser, client):
        self.root = root
        self.ser = ser
        self.client = client
        self.latest = {}
        self.message = "Homing..."
        self.status = ""
        self.busy = False          # capture, scan or calibration in progress
        self.abort = False
        self.running = True
        self.ready = threading.Event()   # set when the Arduino reports READY after homing
        self.profile = None        # latest intensity vs distance
        self.profile_dirty = False
        self.ymax = 1.0
        self.radar_frames = 0      # counters for the fps readout
        self.drawn_frames = 0
        self.fps_time = time.time()
        self.recording = None      # frames being collected by the radar thread
        self.record_done = threading.Event()
        self.shown_pos = None
        self.advance = tk.BooleanVar(value=False)
        # Scene map: one row per mask, each the mean amplitude of one capture
        self.scene = tk.StringVar(value=SCENES[0])
        self.stack = load_scene(SAVE_DIR / self.scene.get())
        self.stack_name = f"Scene {self.scene.get()}"
        self.stack_dirty = False
        self.scene_saved = False   # a capture was saved; recolour the scene button
        # Ambient reference (mask-indexed rows like the scene map), subtracted for display only
        self.ambient = np.load(AMBIENT_FILE)["scene"] if AMBIENT_FILE.exists() else None
        self.subtract = tk.BooleanVar(value=False)
        SAVE_DIR.mkdir(exist_ok=True)

        root.title("Radar Capture")
        root.geometry("1150x820")

        # ---- Left: position, mask, controls ----
        left = tk.Frame(root)
        left.pack(side="left", fill="y", padx=10, pady=6)
        self.lbl_pos = tk.Label(left, text="Mask --", font=("Helvetica", 28, "bold"))
        self.lbl_dist = tk.Label(left, text="-- mm", font=("Helvetica", 13))
        self.lbl_pos.pack()
        self.lbl_dist.pack()

        mrow = tk.Frame(left)
        mrow.pack(pady=4)
        self.cell = 32
        self.mask_canvas = tk.Canvas(mrow, width=4 * self.cell, height=4 * self.cell,
                                     highlightthickness=1, highlightbackground="#888")
        self.mask_canvas.pack(side="left", padx=(0, 10))
        # Go-to buttons: one per mask, sends 'a'..'x' (absolute move to its position)
        grid = tk.Frame(mrow)
        grid.pack(side="left")
        self.goto_buttons = []
        for m in range(NUM_POSITIONS):
            b = tk.Button(grid, text=str(m), width=3,
                          command=lambda m=m: self.goto(position_of_mask(m)))
            b.grid(row=m // 6, column=m % 6, padx=1, pady=1)
            self.goto_buttons.append(b)
        self.goto_bg = self.goto_buttons[0].cget("bg")

        self.lbl_status = tk.Label(left, text="", font=("Helvetica", 12))
        self.lbl_msg = tk.Label(left, text="", font=("Helvetica", 10), fg="#b00", wraplength=380)
        self.lbl_errors = tk.Label(left, text="", font=("Helvetica", 9))
        self.lbl_status.pack()
        self.lbl_msg.pack()
        self.lbl_errors.pack()
        self.errors_shown = (0, 0)

        row = tk.Frame(left)
        row.pack(pady=3)
        self.btn_prev = tk.Button(row, text="◀ Prev", width=8, command=lambda: self.move("F"))
        self.btn_capture = tk.Button(row, text=f"Capture {FRAMES_PER_CAPTURE}", width=12,
                                     command=self.capture)
        self.btn_next = tk.Button(row, text="Next ▶", width=8, command=lambda: self.move("B"))
        for b in (self.btn_prev, self.btn_capture, self.btn_next):
            b.pack(side="left", padx=3)

        actions = tk.Frame(left)
        actions.pack(pady=3)
        self.btn_scan = tk.Button(actions, text=f"Scan all {NUM_POSITIONS}", width=20,
                                  command=self.scan)
        self.btn_ambient = tk.Button(actions, text=f"Capture {NUM_POSITIONS} mask ambient",
                                     width=20, command=self.ambient_scan)
        self.btn_calibrate = tk.Button(actions, text="Calibrate", width=20,
                                       command=self.calibrate)
        btn_stop = tk.Button(actions, text="STOP", width=20, fg="#b00", command=self.stop)
        for i, b in enumerate((self.btn_scan, self.btn_ambient, self.btn_calibrate, btn_stop)):
            b.grid(row=i // 2, column=i % 2, padx=2, pady=2)

        checks = tk.Frame(left)
        checks.pack()
        tk.Checkbutton(checks, text="Advance after capture",
                       variable=self.advance).pack(side="left")
        self.chk_subtract = tk.Checkbutton(checks, text="Subtract ambient (display only)",
                                           variable=self.subtract, command=self.on_subtract)
        self.chk_subtract.pack(side="left")
        tk.Label(left, text="C/Enter capture  ·  ←/→ move  ·  Space stop",
                 font=("Helvetica", 9), fg="#666").pack(pady=2)

        # Scene selector: one folder per scene, coloured by how much has been captured
        tk.Label(left, text="Scene (save folder)", font=("Helvetica", 11, "bold")).pack(pady=(6, 0))
        sgrid = tk.Frame(left)
        sgrid.pack()
        self.scene_buttons = {}
        for i, s in enumerate(SCENES):
            colour = scene_colour(s)
            b = tk.Radiobutton(sgrid, text=s, variable=self.scene, value=s, indicatoron=False,
                               width=3, font=("Helvetica", 9), bg=colour, selectcolor=colour,
                               offrelief="raised", command=self.select_scene)
            b.grid(row=i // 9, column=i % 9, padx=1, pady=1)
            self.scene_buttons[s] = b
        legend = tk.Frame(left)
        legend.pack(pady=2)
        for text, colour in (("none", SCENE_NONE), ("partial", SCENE_PARTIAL),
                             ("all 24", SCENE_FULL)):
            tk.Label(legend, text=text, bg=colour, font=("Helvetica", 9),
                     padx=6).pack(side="left", padx=2)
        self.mark_selected_scene()

        # ---- Right: intensity plots ----
        right = tk.Frame(root)
        right.pack(side="right", fill="both", expand=True)
        self.lbl_fps = tk.Label(right, text="", font=("Helvetica", 10), fg="#666")
        self.lbl_fps.pack(anchor="e", padx=10)
        fig = Figure(figsize=(7, 7.8), dpi=100)
        self.ax_line, self.ax_color, self.ax_stack = fig.subplots(
            3, 1, sharex=True, gridspec_kw={"height_ratios": [3, 0.6, 4]})
        dist_mm = DISTANCES_M * 1000
        # animated artists are left out of full redraws and blitted on top each frame
        self.line, = self.ax_line.plot(dist_mm, np.zeros_like(dist_mm), animated=True)
        self.ax_line.set_ylabel("Intensity")
        self.ax_line.set_title("Live radar: mean amplitude vs distance")
        self.ax_line.grid(alpha=0.3)
        half = (dist_mm[1] - dist_mm[0]) / 2
        self.img = self.ax_color.imshow(np.zeros((1, len(dist_mm))), aspect="auto",
                                        cmap="viridis", animated=True,
                                        extent=(dist_mm[0] - half, dist_mm[-1] + half, 0, 1))
        self.ax_color.set_yticks([])
        # grey = not captured yet
        self.sequential = matplotlib.colormaps["viridis"].with_extremes(bad="#dddddd")
        self.diverging = matplotlib.colormaps["RdBu_r"].with_extremes(bad="#dddddd")
        self.stack_img = self.ax_stack.imshow(
            self.stack, aspect="auto", cmap=self.sequential, interpolation="nearest",
            extent=(dist_mm[0] - half, dist_mm[-1] + half, NUM_POSITIONS - 0.5, -0.5))
        self.ax_stack.set_yticks([0, 4, 8, 12, 16, 20, 23])
        self.ax_stack.set_ylabel("Mask")
        self.ax_stack.set_xlabel("Distance (mm)")
        self.ax_stack.set_title(" ", fontsize=10)   # reserve space; text set in draw_stack
        fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.backgrounds = None
        self.canvas.mpl_connect("draw_event", self.on_full_draw)
        self.stack_dirty = True

        root.bind("<Left>", lambda e: self.move("F"))
        root.bind("<Right>", lambda e: self.move("B"))
        root.bind("<space>", lambda e: self.stop())
        root.bind("<Return>", lambda e: self.capture())
        root.bind("c", lambda e: self.capture())
        root.protocol("WM_DELETE_WINDOW", self.close)

        threading.Thread(target=self.reader, daemon=True).start()
        threading.Thread(target=self.heartbeat, daemon=True).start()
        self.radar_thread = threading.Thread(target=self.radar_loop, daemon=True)
        self.radar_thread.start()
        self.tick()
        self.plot_tick()

    # ---- Display ----
    def draw_mask(self, pos):
        self.mask_canvas.delete("all")
        if not 1 <= pos <= NUM_POSITIONS:
            return
        c = self.cell
        for r, row in enumerate(mask_at(pos)):
            for k, v in enumerate(row):
                self.mask_canvas.create_rectangle(k * c, r * c, (k + 1) * c, (r + 1) * c,
                                                  fill=MASK_ON if v else MASK_OFF,
                                                  outline="#444")

    def on_full_draw(self, event):
        # Axes, labels and grid were just redrawn: keep them as the blit background
        self.backgrounds = [self.canvas.copy_from_bbox(ax.bbox)
                            for ax in (self.ax_line, self.ax_color)]
        self.blit()

    def blit(self):
        for ax, bg, artist in zip((self.ax_line, self.ax_color), self.backgrounds,
                                  (self.line, self.img)):
            self.canvas.restore_region(bg)
            ax.draw_artist(artist)
            self.canvas.blit(ax.bbox)

    def on_subtract(self):
        self.ymax = 0.0          # force a y-axis rescale on the next frame
        self.stack_dirty = True

    def draw_profile(self):
        p = self.profile
        pos = self.shown_pos
        if self.subtract.get() and pos and 1 <= pos <= NUM_POSITIONS:
            p = p - np.nan_to_num(self.ambient[mask_index(pos)])
        self.line.set_ydata(p)
        self.img.set_data(p[None, :])
        self.img.set_clim(p.min(), p.max())
        # Rescaling the y-axis needs a slow full redraw, so only do it when the
        # signal outgrows the axis or falls well below it
        peak = max(np.abs(p).max(), 1.0)
        if peak > self.ymax or peak < 0.4 * self.ymax:
            self.ymax = peak * 1.25
            self.ax_line.set_ylim(-self.ymax if self.subtract.get() else 0, self.ymax)
            self.canvas.draw()   # triggers on_full_draw, which blits the new data
        elif self.backgrounds:
            self.blit()

    def draw_stack(self):
        shown = self.stack
        if self.subtract.get():
            shown = shown - np.nan_to_num(self.ambient)
            lim = np.nanmax(np.abs(shown)) if not np.isnan(shown).all() else 1.0
            self.stack_img.set_cmap(self.diverging)   # 0 = white, so +/- vs ambient stands out
            self.stack_img.set_clim(-lim, lim)
        else:
            self.stack_img.set_cmap(self.sequential)
            if not np.isnan(shown).all():
                self.stack_img.set_clim(np.nanmin(shown), np.nanmax(shown))
        self.stack_img.set_data(shown)
        self.ax_stack.set_title(f"{self.stack_name}: {FRAMES_PER_CAPTURE}-frame mean per mask",
                                fontsize=10)
        self.canvas.draw()   # full redraw; on_full_draw re-captures the blit backgrounds

    def select_scene(self):
        self.select_scene_data()
        self.mark_selected_scene()

    def mark_selected_scene(self):
        for s, b in self.scene_buttons.items():
            b.config(font=("Helvetica", 9, "bold" if s == self.scene.get() else "normal"))

    def plot_tick(self):
        if not self.running:
            return
        if self.stack_dirty:
            self.stack_dirty = False
            self.draw_stack()
        if self.profile_dirty:
            self.profile_dirty = False
            self.draw_profile()
            self.drawn_frames += 1
        elapsed = time.time() - self.fps_time
        if elapsed >= 1:
            self.lbl_fps.config(text=f"Radar {self.radar_frames / elapsed:.1f} fps  ·  "
                                     f"display {self.drawn_frames / elapsed:.1f} fps")
            self.radar_frames = self.drawn_frames = 0
            self.fps_time = time.time()
        self.root.after(PLOT_MS, self.plot_tick)

    # ---- Stepper ----
    def send(self, cmd):
        try:
            self.ser.write(cmd.encode())
        except serial.SerialException:
            self.message = "Arduino connection lost"

    def move(self, cmd):
        if self.busy:
            self.message = "Wait for the capture to finish before moving"
            return
        self.send(cmd)

    def goto(self, pos):
        if self.busy:
            self.message = "Wait for the capture to finish before moving"
            return
        self.send(chr(ord("a") + pos - 1))

    def stop(self):
        self.abort = True
        self.send("S")

    def heartbeat(self):
        # Own thread so GUI stalls (window creation, full redraws) can't starve the Arduino
        while self.running:
            self.send("H")
            time.sleep(HEARTBEAT_S)

    def tick(self):
        if not self.running:
            return
        if "Distance_mm" in self.latest:
            mm = self.latest["Distance_mm"]
            pos = position_from_mm(mm)
            valid = 1 <= pos <= NUM_POSITIONS
            self.lbl_dist.config(text=f"Position {pos}  ·  {mm:.2f} mm")
            self.lbl_pos.config(text=f"Mask {mask_index(pos)}" if valid else "Mask --")
            if pos != self.shown_pos:
                self.shown_pos = pos
                self.draw_mask(pos)
                for m, b in enumerate(self.goto_buttons):
                    b.config(bg=MASK_ON if valid and m == mask_index(pos) else self.goto_bg)
        if self.scene_saved:
            self.scene_saved = False
            for s, b in self.scene_buttons.items():
                colour = scene_colour(s)
                b.config(bg=colour, selectcolor=colour)
        self.lbl_status.config(text=self.status)
        self.lbl_msg.config(text=self.message)
        errors = (FrameErrors.radar, FrameErrors.arduino)
        if errors != self.errors_shown:
            self.errors_shown = errors
            self.message = "Frame error - corrupted message discarded"
        self.lbl_errors.config(text=f"Frame errors: radar {errors[0]} · arduino {errors[1]}",
                               fg="#b00" if any(errors) else "#666")
        state = "disabled" if self.busy else "normal"
        for b in (self.btn_capture, self.btn_scan, self.btn_ambient, self.btn_calibrate,
                  self.btn_prev, self.btn_next, *self.goto_buttons,
                  *self.scene_buttons.values()):
            b.config(state=state)
        self.chk_subtract.config(state="normal" if self.ambient is not None else "disabled")
        self.root.after(UI_MS, self.tick)

    def reader(self):
        while self.running:
            try:
                line = self.ser.readline().decode(errors="ignore").strip()
            except serial.SerialException:
                self.message = "Arduino connection lost"
                return
            if not line:
                continue
            if ARDUINO_LINE.fullmatch(line):
                for part in line.split(","):
                    name, _, value = part.partition(":")
                    self.latest[name] = float(value)
            elif ":" in line or not line.isprintable():
                FrameErrors.arduino += 1   # garbled telemetry line
            elif line.startswith("READY"):
                self.message = ""
                self.ready.set()
            else:
                self.message = line

    def wait_for_position(self, pos):
        target = mm_from_position(pos)
        deadline = time.time() + MOVE_TIMEOUT_S
        settled_since = None
        while time.time() < deadline:
            if self.abort:
                raise ScanAborted
            if abs(self.latest.get("Distance_mm", -1) - target) < ARRIVE_TOLERANCE_MM:
                settled_since = settled_since or time.time()
                if time.time() - settled_since >= SETTLE_S:
                    return
            else:
                settled_since = None
            time.sleep(0.05)
        raise ScanAborted(f"Timed out moving to position {pos}")

    # ---- Radar ----
    def radar_loop(self):
        """Streams continuously: feeds the live plot and any capture in progress."""
        self.client.start_session()
        while self.running:
            try:
                frame = self.client.get_next().frame
            except Exception as e:
                self.message = f"Radar stopped: {e}"
                return
            self.profile = np.abs(frame).mean(axis=0)
            self.profile_dirty = True
            self.radar_frames += 1
            rec = self.recording
            if rec is not None:
                rec.append(frame)
                if len(rec) == FRAMES_PER_CAPTURE:
                    self.record_done.set()

    def record(self, folder, pos):
        m = mask_index(pos)
        mm = self.latest.get("Distance_mm", float("nan"))
        errors_before = FrameErrors.radar
        self.record_done.clear()
        self.recording = []
        while not self.record_done.wait(0.1):
            self.status = f"Mask {m}: frame {len(self.recording)}/{FRAMES_PER_CAPTURE}"
            if self.abort or not self.radar_thread.is_alive():
                self.recording = None
                raise ScanAborted
        frames = np.stack(self.recording[:FRAMES_PER_CAPTURE])
        self.recording = None
        self.stack[m] = np.abs(frames).mean(axis=(0, 1))
        self.stack_dirty = True
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(
            folder / f"mask{m:02d}.npz",
            frames=frames,
            distances_m=DISTANCES_M,
            distance_mm=mm,
            position=pos,
            mask_index=m,
            frame_errors=FrameErrors.radar - errors_before,
            mask=mask_at(pos),
            sensor_config=self.client.session_config.sensor_config.to_json(),
            server_info=self.client.server_info.to_json(),
        )
        self.scene_saved = True

    def run_job(self, job):
        if self.busy:
            return
        self.busy = True
        self.abort = False
        threading.Thread(target=self.job_wrapper, args=(job,), daemon=True).start()

    def job_wrapper(self, job):
        try:
            job()
        except ScanAborted as e:
            self.status = str(e) or "Aborted"
        except Exception as e:
            self.message = f"Capture failed: {e}"
            self.status = ""
        finally:
            self.busy = False

    def capture(self):
        self.run_job(self.capture_job)

    def capture_job(self):
        pos = position_from_mm(self.latest.get("Distance_mm", 0.0))
        if not 1 <= pos <= NUM_POSITIONS:
            self.message = "Not at a mask position - calibrate first"
            return
        self.record(SAVE_DIR / self.scene.get(), pos)
        self.status = f"Saved {self.scene.get()}/mask{mask_index(pos):02d}.npz"
        if self.advance.get():
            self.send("B")

    def calibrate(self):
        self.run_job(self.calibrate_job)

    def calibrate_job(self):
        self.ready.clear()
        self.send("C")
        self.status = "Calibrating (homing)..."
        deadline = time.time() + HOMING_TIMEOUT_S
        while not self.ready.wait(0.1):
            if self.abort or self.message.startswith("EMERGENCY"):
                raise ScanAborted("Calibration stopped")
            if time.time() > deadline:
                raise ScanAborted("Calibration timed out")
        pos = position_of_mask(CALIBRATE_TO_MASK)
        self.send(chr(ord("a") + pos - 1))
        self.status = f"Moving to mask {CALIBRATE_TO_MASK}"
        self.wait_for_position(pos)
        self.status = f"Calibrated - at mask {CALIBRATE_TO_MASK}"

    def scan(self):
        self.run_job(lambda: self.scan_job(SAVE_DIR / self.scene.get(),
                                           f"Scene {self.scene.get()}"))

    def ambient_scan(self):
        self.run_job(self.ambient_scan_job)

    def ambient_scan_job(self):
        try:
            self.scan_job(AMBIENT_DIR, "Ambient")
            # Only adopt the new reference once all 24 masks were captured
            self.ambient = self.stack.copy()
        finally:
            self.select_scene_data()

    def select_scene_data(self):
        """Back to showing the selected scene (thread-safe part of select_scene)."""
        self.stack = load_scene(SAVE_DIR / self.scene.get())
        self.stack_name = f"Scene {self.scene.get()}"
        self.stack_dirty = True

    def scan_job(self, folder, name):
        self.stack = np.full((NUM_POSITIONS, len(DISTANCES_M)), np.nan)
        self.stack_name = name
        self.stack_dirty = True
        self.status = "Moving to position 1"
        self.send("a")   # absolute move to position 1
        for pos in range(1, NUM_POSITIONS + 1):
            if pos > 1:
                self.send("B")
            self.status = f"Moving to mask {mask_index(pos)}"
            self.wait_for_position(pos)
            self.record(folder, pos)
        np.savez(folder / "scene_map.npz", scene=self.stack, distances_m=DISTANCES_M)
        self.status = f"Scan saved to {folder.name}"

    def close(self):
        self.running = False
        self.abort = True
        self.send("S")
        self.ser.close()
        self.radar_thread.join(timeout=1)
        self.client.close()
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arduino", help="Arduino serial port, e.g. COM12")
    ap.add_argument("--radar", help="Radar serial port, e.g. COM11 (default: auto-detect)")
    ap.add_argument("--mock", action="store_true", help="use simulated radar data")
    args = ap.parse_args()

    if args.mock:
        client = a121.Client.open(mock=True)
    else:
        radar_port = args.radar or find_port("Radar", RADAR_KEYWORDS)
        client = a121.Client.open(serial_port=radar_port, override_baudrate=RADAR_BAUD)
    client.setup_session(SENSOR_CONFIG)

    port = args.arduino or find_port("Arduino", ARDUINO_KEYWORDS)
    try:
        ser = serial.Serial(port, BAUD, timeout=0.1)
    except serial.SerialException as e:
        client.close()
        sys.exit(f"Couldn't open {port}: {e}\nIs the Serial Monitor/Plotter still open?")

    root = tk.Tk()
    CaptureApp(root, ser, client)
    root.mainloop()


if __name__ == "__main__":
    main()
