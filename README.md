# NeuroDrive

**NeuroDrive is an extension of NeuroWave EEG**, adding manual and EEG-based
wheelchair control to its acquisition, data collection, training, and prediction
workflows.

This Windows desktop application connects an OpenBCI Cyton + Daisy EEG board
through BrainFlow and controls a wheelchair through an Arduino, USB or HC-06
Bluetooth, and two MCP4725 DACs. Its interface is built with PyQt5 and its EEG
classifier uses scikit-learn's Random Forest.

## Install and run

Run these commands from the repository root with Python 3.11:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python 03_control_app/run_control.py
```

If you already use a Conda environment, activate it and run the install and
launch commands with that environment's Python instead.

From inside `03_control_app`, run `python run_control.py`.
Alternatively, run `python 03_control_app/code/main.py` from the repository root.
The launcher resolves paths relative to its own location, so it also works when
launched from the IDE or another working directory.

### Python dependencies

[requirements.txt](requirements.txt) contains the active app's direct runtime dependencies:

| Package | Purpose |
| --- | --- |
| BrainFlow | OpenBCI board connection and EEG acquisition |
| NumPy | EEG arrays and feature calculations |
| PyQt5 | Desktop interface, timers, worker threads, and SVG dropdown arrows |
| PySerial (`pyserial`) | Wheelchair communication and serial-port discovery |
| scikit-learn | Random Forest training, evaluation, and inference |
| joblib | Saving and loading trained models |

Tests use Python's built-in `unittest` module along with the runtime dependencies.
Install the full requirements before connecting hardware or running the complete
test suite.

## Project layout

```text
00_archive/                         Standalone controller reference
01_arduino_code/arduino_code.ino      Current wheelchair firmware
03_control_app/
  run_control.py                    Main launcher
  code/
    main.py                         Application startup
    config.py                       Paths and EEG defaults
    app_theme.py                    Shared interface styling
    core/                           Acquisition, ML, decisions, and serial control
    ui/                             Main window and EEG workflow windows
    images/                         App icon, dropdown arrow, and class cues
    streamer_data/                  CSV sources offered for replay
  dataset/                          Collected raw EEG recordings
  trained_model/                    Model bundles and training reports
  tests/                            Automated checks for the current app
README.md
requirements.txt
```

`00_archive/` contains standalone controller reference files; the active app does
not import them. Datasets, trained models, and images resolve relative to `03_control_app/`,
independently of the directory used to launch the app. The main app's fixed
voltage definitions are in `03_control_app/code/core/wheelchair_outputs.py`.

## Application overview

The main page contains EEG Connect/Start/Stop, the EEG workflow bar, and all
wheelchair controls in three compact columns, with a shared command-status area.
It fits a laptop window without scrolling. The footer shows EEG state, received
sample count, incoming sample rate and live-prediction activity.

The EEG workflow buttons open Data Collection, Train Model, Load Model, and
Realtime Prediction windows on demand. Incoming EEG feeds recording and
prediction directly. The main window shows prediction text, confidence,
configured output voltages, and command status.

Recordings, model bundles, and training reports save to their selected locations.
Errors appear in the interface and console; the app does not create session log files.

## Wheelchair firmware and connection

Flash [01_arduino_code/arduino_code.ino](01_arduino_code/arduino_code.ino) with the
Adafruit MCP4725 library installed when setting up the Arduino. This is the
firmware used by the current app's single-character command protocol.

Select the wheelchair COM port and connection type:

- **Bluetooth (HC-06):** the paired Bluetooth COM port at 9600 baud.
- **USB (Arduino):** the Arduino USB COM port at 115200 baud.

The EEG connection remains independent of the wheelchair connection.

Firmware wiring: MCP4725 X address `0x60`, Y address `0x61`;
HC-06 TXD to Arduino D10, Arduino D11 through a voltage divider to HC-06 RXD.
Firmware voltage conversion uses a configured DAC supply of **5.04 V**.

## Fixed wheelchair outputs

Both Manual and EEG modes use these fixed firmware targets:

| Command | X target | Y target | Wire command |
| --- | --- | --- | --- |
| Stop | 2.50 V | 2.50 V | `x` |
| Forward | 2.50 V | 0.55 V | `w` |
| Backward | 2.50 V | 4.45 V | `s` |
| Left | 0.55 V | 2.50 V | `a` |
| Right | 4.45 V | 2.50 V | `d` |

The Arduino converts millivolts to 12-bit DAC codes using
`min(4095, (millivolts * 4096 + 2520) // 5040)`.
Neutral is DAC code **2032**, low is **447**, and high is **3617**.
Voltage values and raw DAC codes are distinct units.

The Arduino converts the direction command into DAC outputs. The app displays
configured targets rather than measured hardware feedback. Voltage targets are
defined in the firmware and mirrored in
`03_control_app/code/core/wheelchair_outputs.py`.

## Manual and EEG control

### Manual control

The app starts disarmed. Connect the wheelchair controller and explicitly arm output before
movement. In Manual mode, hold W/A/S/D or a direction button to move; release
to stop. X, Space, or the STOP button stops motion. Losing focus while holding
a manual direction also stops it. Emergency Stop disables output in the app.

### EEG control and saved settings

EEG control requires a running EEG stream, realtime prediction, EEG Control
mode, a model containing a Rest/Stop class, and explicit arming.

| Setting | Default | How to change it |
| --- | --- | --- |
| Confidence threshold | 75% | Enter 25–99% with up to one decimal place, then click its **Save** button |
| Movement duration | 3 seconds | Enter 0.1–60 seconds with up to one decimal place, then click its **Save** button |
| Prediction voting | No voting | Select No voting, 2, 3, 4, or 5 votes; applies immediately |

Typing, pressing Enter, leaving a text field, and mouse scrolling do not apply
threshold or duration edits. Each Save button applies only its own field.
Changing voting uses the last saved threshold and duration. Invalid entries
leave the active setting unchanged and display an explanation in Command Status.
These settings apply to the current app session; restarting restores the defaults.

Saving either text field or changing voting stops any current EEG movement and
clears the vote history. Fresh predictions must meet the saved settings before
another run can begin.

No voting starts movement on the first qualifying prediction. Voting requires
the chosen number of agreeing predictions among the latest five qualifying predictions.
An accepted Forward, Left, or Right command runs for the saved **Movement
duration** (default **3 seconds**), then sends STOP and clears voting history. The next run requires a
fresh qualifying prediction or fresh votes, according to the selected mode.
Prediction results received during a run are ignored for control: movement,
Rest/Stop, unknown classes, and low-confidence results cannot interrupt,
redirect, or extend the accepted direction. They are not queued for later use.
Prediction timeout, stream shutdown, communication failure, Emergency Stop,
disarming, mode changes, or saving decision settings can still stop a run early.
Manual controls still use hold-to-move behavior.

Between movement runs, Rest/Stop predictions send STOP immediately and never
start a hold or cooldown of their own. Rest predicted during an active run does
not interrupt it. With No voting selected, the next fresh movement prediction
meeting the confidence threshold starts a new run using the saved duration immediately
when its result arrives. With voting enabled, only the selected vote count is
required; there is no additional cooldown. Predictions arriving at movement
expiry are processed instead of being discarded. EEG acquisition windows and
model inference still determine when each prediction becomes available.

For example, with **5 seconds** saved and No voting selected:

1. A Forward prediction meets the saved confidence threshold and starts movement.
2. Rest, another direction, or low confidence predicted during that run does not
   change its direction or extend the deadline.
3. At 5 seconds, the app sends STOP and clears the vote history.
4. The next fresh qualifying prediction can start another run without a cooldown.

The prediction stream must remain active. A gap of more than **1.5 seconds**
without any prediction result triggers the prediction watchdog, which can stop
movement early even when the saved duration is longer. Received Rest results
count as stream activity during a run; they do not interrupt the movement.
Explicit stop controls and stream/communication failure handling remain active.

The app sends lowercase direction characters and refreshes movement every
200 ms. The Arduino returns to neutral after 800 ms without a movement command,
at startup, or after unexpected input. Arming is an **application-side gate**;
the single-character firmware does not implement an ARM/DISARM handshake.

Serial write failures cancel refreshes, attempt STOP, and disconnect/disarm
the app. Reconnection requires fresh arming.

The software controls supplement an independent physical emergency stop.
Bench-test with the wheelchair unoccupied and drive wheels raised.

## EEG training and Rest balancing

Use the EEG workflow buttons as needed. Streaming is required for collection
and realtime prediction; training and model loading can be done offline:

1. **Data Collection:** configure labels and collect guided raw EEG recordings
   with `label` and `trial_id` columns.
2. **Train Model:** select recorded CSVs, configure the training run, and train
   the EEG Random Forest classifier.
3. **Load Model:** load the saved `.joblib` bundle for use in live prediction.
4. **Realtime Prediction:** start predictions, then select EEG Control and arm
   wheelchair output when ready.

Evaluation splits keep complete recording trials together, so overlapping windows
from one trial cannot appear in both training and evaluation partitions.

Training automatically undersamples excess `Rest` windows, matching the smallest
non-Rest class in the training partition. For example, training counts of
Forward=100, Left=100, Right=100 and Rest=300 become 100 per class. When movement
counts differ, all movement windows are retained and Rest is capped at the
smallest movement count. Labels match `Rest` regardless of case or outer spaces.

The training random seed makes the selected Rest subset repeatable.
No trimming occurs if Rest is absent or already at or below the target count.
The operation runs after the trial-grouped train/test split and separately within
each cross-validation training fold. Test and validation windows retain their
original counts. Collection settings and source CSV recordings stay intact.

Each training run saves a `.joblib` model bundle, `training_setup.json`,
`training_results.json`, and `training_summary.txt` under the selected model
output folder. Results include holdout accuracy, weighted F1, cross-validation
accuracy, and training class counts before and after Rest balancing. Load the
desired model bundle before starting realtime prediction.

## CSV replay without an EEG board

Select a `csv://...` source in the EEG connection dropdown, then Connect and
Start. CSVs in `03_control_app/code/streamer_data/` are listed automatically.
Replay loops continuously at the configured default of 125 Hz.

An optional CSV endpoint selector is available with:

```powershell
python 03_control_app/code/simulator_main.py
```

Replay replaces the EEG input only. Wheelchair output still requires a separate
controller connection and explicit arming.

## Tests

From the repository root, with the runtime dependencies installed:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m unittest discover -s 03_control_app/tests -v
```

The suite checks board connection cleanup, CSV replay, grouped training splits,
Rest balancing, fixed voltage targets, serial commands, UI layout, Save behavior,
movement timing, and stop conditions. Hardware interactions use mocks or replay;
passing tests do not verify physical DAC voltages or wheelchair motion.

Coding Assistant: Codex
