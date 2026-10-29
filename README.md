# MECH5720 major project: coded-mask radar rig

An Acconeer XM125 radar looks through a sliding 4x4 binary mask (24 positions, 7 mm apart)
driven by a stepper on an Arduino Uno. `radar_capture.py` shows the live radar feed, steps
the mask, and records captures per mask and per scene.

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Upload `arduino/arduino.ino` to the Uno (needs the AccelStepper library:
Arduino IDE > Tools > Manage Libraries).

## Run

```
python radar_capture.py                           # auto-detects radar and Arduino
python radar_capture.py --arduino COM12 --radar COM11
python radar_capture.py --mock                    # simulated radar (Arduino still required)
```

Close the Arduino Serial Monitor and the Acconeer Exploration Tool first: only one program
can use each port. Controls and the saved file format are described at the top of
`radar_capture.py`. Captures are saved to `captures/` (not tracked by git).

## Files

- `radar_capture.py` - capture app
- `arduino/arduino.ino` - stepper, homing and serial command firmware
- `mask_patterns/mask_pattern.csv` - the 4 x 27 binary mask pattern
- `main.py` - older arrow-key jog tool
