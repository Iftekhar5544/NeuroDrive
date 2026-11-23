import sys
import serial
import serial.tools.list_ports

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

BAUD_RATE = 9600

# Arduino fail-safe is 800 ms.
# Refresh movement commands well inside that window.
SEND_INTERVAL_MS = 200


class WheelchairController(QMainWindow):
    def __init__(self):
        super().__init__()

        self.serial_port = None
        self.active_command = None

        self.setWindowTitle("Wheelchair Bluetooth Controller")
        self.setFixedSize(470, 390)

        self.command_timer = QTimer(self)
        self.command_timer.setInterval(SEND_INTERVAL_MS)
        self.command_timer.timeout.connect(self.send_active_command)

        self.build_ui()
        self.refresh_ports()

    # =========================================================
    # UI
    # =========================================================
    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        title = QLabel("Wheelchair Bluetooth Controller")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size: 20px; font-weight: 600; padding: 8px;"
        )
        main_layout.addWidget(title)

        # ---------------- COM connection ----------------
        connection_layout = QHBoxLayout()

        self.port_combo = QComboBox()
        self.refresh_button = QPushButton("Refresh")
        self.connect_button = QPushButton("Connect")
        self.disconnect_button = QPushButton("Disconnect")

        self.refresh_button.clicked.connect(self.refresh_ports)
        self.connect_button.clicked.connect(self.connect_serial)
        self.disconnect_button.clicked.connect(self.disconnect_serial)

        self.disconnect_button.setEnabled(False)

        connection_layout.addWidget(self.port_combo, 1)
        connection_layout.addWidget(self.refresh_button)
        connection_layout.addWidget(self.connect_button)
        connection_layout.addWidget(self.disconnect_button)

        main_layout.addLayout(connection_layout)

        # ---------------- Controls ----------------
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        self.forward_button = QPushButton("FORWARD\nW")
        self.left_button = QPushButton("LEFT\nA")
        self.stop_button = QPushButton("STOP\nSpace / X")
        self.right_button = QPushButton("RIGHT\nD")
        self.backward_button = QPushButton("BACKWARD\nS")

        direction_buttons = [
            self.forward_button,
            self.left_button,
            self.right_button,
            self.backward_button,
        ]

        for button in direction_buttons:
            button.setMinimumSize(125, 72)
            button.setStyleSheet(
                "font-size: 14px; font-weight: 600;"
            )

        self.stop_button.setMinimumSize(125, 72)
        self.stop_button.setStyleSheet(
            "font-size: 14px;"
            "font-weight: 700;"
            "background-color: #b71c1c;"
            "color: white;"
        )

        grid.addWidget(self.forward_button, 0, 1)
        grid.addWidget(self.left_button, 1, 0)
        grid.addWidget(self.stop_button, 1, 1)
        grid.addWidget(self.right_button, 1, 2)
        grid.addWidget(self.backward_button, 2, 1)

        main_layout.addLayout(grid)

        # Hold-to-move behavior
        self.setup_hold_button(self.forward_button, "w", "FORWARD")
        self.setup_hold_button(self.backward_button, "s", "BACKWARD")
        self.setup_hold_button(self.left_button, "a", "LEFT")
        self.setup_hold_button(self.right_button, "d", "RIGHT")

        self.stop_button.clicked.connect(self.stop_motion)

        # ---------------- Status ----------------
        self.status_label = QLabel("Status: Disconnected")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            "font-size: 14px; padding: 8px;"
        )
        main_layout.addWidget(self.status_label)

        self.command_label = QLabel("Command: STOP")
        self.command_label.setAlignment(Qt.AlignCenter)
        self.command_label.setStyleSheet(
            "font-size: 16px; font-weight: 600; padding: 5px;"
        )
        main_layout.addWidget(self.command_label)

        hint = QLabel(
            "Hold W/A/S/D or a direction button to move. "
            "Release to stop."
        )
        hint.setAlignment(Qt.AlignCenter)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555;")
        main_layout.addWidget(hint)

    def setup_hold_button(self, button, command, name):
        button.pressed.connect(
            lambda c=command, n=name: self.start_motion(c, n)
        )
        button.released.connect(self.stop_motion)

    # =========================================================
    # COM PORTS
    # =========================================================
    def refresh_ports(self):
        previous_port = self.port_combo.currentData()

        self.port_combo.clear()
        ports = list(serial.tools.list_ports.comports())

        if not ports:
            self.port_combo.addItem("No COM ports found", None)
            return

        selected_index = 0

        for index, port in enumerate(ports):
            label = f"{port.device} - {port.description}"
            self.port_combo.addItem(label, port.device)

            if port.device == previous_port:
                selected_index = index

        self.port_combo.setCurrentIndex(selected_index)

    # =========================================================
    # CONNECTION
    # =========================================================
    def connect_serial(self):
        port_name = self.port_combo.currentData()

        if not port_name:
            QMessageBox.warning(
                self,
                "No COM Port",
                "Select the HC-06 Bluetooth COM port first.",
            )
            return

        try:
            self.serial_port = serial.Serial(
                port=port_name,
                baudrate=BAUD_RATE,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,
                write_timeout=0.2,
            )

            # Force STOP immediately on connection.
            self.send_command("x")

            self.status_label.setText(
                f"Status: Connected to {port_name} @ {BAUD_RATE} baud"
            )
            self.command_label.setText("Command: STOP")

            self.port_combo.setEnabled(False)
            self.refresh_button.setEnabled(False)
            self.connect_button.setEnabled(False)
            self.disconnect_button.setEnabled(True)

        except serial.SerialException as exc:
            self.serial_port = None

            QMessageBox.critical(
                self,
                "Connection Error",
                f"Could not open {port_name}.\n\n{exc}",
            )

    def disconnect_serial(self):
        self.command_timer.stop()
        self.active_command = None

        if self.serial_port and self.serial_port.is_open:
            # Try to stop before closing the link.
            try:
                self.serial_port.write(b"x")
                self.serial_port.flush()
            except Exception:
                pass

            try:
                self.serial_port.close()
            except Exception:
                pass

        self.serial_port = None

        self.status_label.setText("Status: Disconnected")
        self.command_label.setText("Command: STOP")

        self.port_combo.setEnabled(True)
        self.refresh_button.setEnabled(True)
        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)

    # =========================================================
    # SERIAL SEND
    # =========================================================
    def send_command(self, command):
        if not self.serial_port or not self.serial_port.is_open:
            return False

        try:
            self.serial_port.write(command.encode("ascii"))
            self.serial_port.flush()
            return True

        except (serial.SerialException, serial.SerialTimeoutException):
            self.command_timer.stop()
            self.active_command = None

            QMessageBox.critical(
                self,
                "Bluetooth Connection Lost",
                "The serial connection was lost. "
                "The Arduino fail-safe should return the wheelchair to STOP.",
            )

            self.disconnect_serial()
            return False

    # =========================================================
    # MOVEMENT
    # =========================================================
    def start_motion(self, command, name):
        if not self.serial_port or not self.serial_port.is_open:
            QMessageBox.warning(
                self,
                "Not Connected",
                "Connect to the HC-06 COM port first.",
            )
            return

        self.active_command = command

        # Send immediately.
        if self.send_command(command):
            # Then refresh continuously while held.
            self.command_timer.start()
            self.command_label.setText(f"Command: {name}")

    def send_active_command(self):
        if self.active_command:
            self.send_command(self.active_command)

    def stop_motion(self):
        self.command_timer.stop()
        self.active_command = None

        if self.serial_port and self.serial_port.is_open:
            self.send_command("x")

        self.command_label.setText("Command: STOP")

    # =========================================================
    # KEYBOARD
    # =========================================================
    def keyPressEvent(self, event):
        # We generate our own repeat using QTimer.
        if event.isAutoRepeat():
            return

        key = event.key()

        if key == Qt.Key_W:
            self.start_motion("w", "FORWARD")
        elif key == Qt.Key_S:
            self.start_motion("s", "BACKWARD")
        elif key == Qt.Key_A:
            self.start_motion("a", "LEFT")
        elif key == Qt.Key_D:
            self.start_motion("d", "RIGHT")
        elif key in (Qt.Key_Space, Qt.Key_X):
            self.stop_motion()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            return

        if event.key() in (
            Qt.Key_W,
            Qt.Key_A,
            Qt.Key_S,
            Qt.Key_D,
        ):
            self.stop_motion()
        else:
            super().keyReleaseEvent(event)

    # =========================================================
    # WINDOW CLOSE
    # =========================================================
    def closeEvent(self, event):
        self.disconnect_serial()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = WheelchairController()
    window.show()

    sys.exit(app.exec_())
