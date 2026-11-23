#include <Wire.h>
#include <Adafruit_MCP4725.h>
#include <SoftwareSerial.h>

// ============================================================
// WHEELCHAIR BLUETOOTH CONTROLLER
// Arduino Uno/Nano + HC-06 + 2x MCP4725
//
// MCP4725 0x60 = X axis
// MCP4725 0x61 = Y axis
//
// Verified wheelchair voltages:
// STOP      : X=2.50 V, Y=2.50 V
// FORWARD   : X=2.50 V, Y=0.55 V
// BACKWARD  : X=2.50 V, Y=4.45 V
// LEFT      : X=0.55 V, Y=2.50 V
// RIGHT     : X=4.45 V, Y=2.50 V
//
// HC-06 commands:
// W = Forward
// S = Backward
// A = Left
// D = Right
// X or Space = Stop
// ============================================================


// ---------------- HC-06 ----------------
// SoftwareSerial(RX, TX)
// HC-06 TXD -> Arduino D10
// Arduino D11 -> voltage divider -> HC-06 RXD
SoftwareSerial HC06(10, 11);


// ---------------- DACs ----------------
Adafruit_MCP4725 dacX;
Adafruit_MCP4725 dacY;

const uint8_t DAC_X_ADDRESS = 0x60;
const uint8_t DAC_Y_ADDRESS = 0x61;


// ---------------- Measured regulated supply ----------------
// LM2587 output measured about 5.04 V.
// Change this if your final measured DAC VCC is different.
const uint16_t DAC_VDD_MV = 5040;


// ---------------- Verified joystick voltages ----------------
const uint16_t CENTER_MV = 2500;
const uint16_t LOW_MV    = 550;
const uint16_t HIGH_MV   = 4450;


// ---------------- Communication fail-safe ----------------
// Python app re-sends a held movement command every 200 ms.
// If no movement command arrives for 800 ms, force STOP.
const unsigned long COMMAND_TIMEOUT_MS = 800;

bool movementActive = false;
unsigned long lastMovementCommand = 0;


// ============================================================
// Convert target millivolts to MCP4725 12-bit DAC code
// ============================================================
uint16_t voltageToDAC(uint16_t millivolts)
{
  uint32_t value =
      ((uint32_t)millivolts * 4096UL + (DAC_VDD_MV / 2))
      / DAC_VDD_MV;

  if (value > 4095)
    value = 4095;

  return (uint16_t)value;
}


// ============================================================
// Set X and Y outputs
// ============================================================
void setXY(uint16_t x_mV, uint16_t y_mV)
{
  uint16_t xCode = voltageToDAC(x_mV);
  uint16_t yCode = voltageToDAC(y_mV);

  // false = do not write movement values to EEPROM
  dacX.setVoltage(xCode, false);
  dacY.setVoltage(yCode, false);
}


// ============================================================
// USB debug output
// ============================================================
void printState(const char* state, uint16_t x_mV, uint16_t y_mV)
{
  Serial.print("STATE: ");
  Serial.print(state);

  Serial.print(" | X=");
  Serial.print(x_mV);
  Serial.print("mV (DAC ");
  Serial.print(voltageToDAC(x_mV));
  Serial.print(")");

  Serial.print(" | Y=");
  Serial.print(y_mV);
  Serial.print("mV (DAC ");
  Serial.print(voltageToDAC(y_mV));
  Serial.println(")");
}


// ============================================================
// STOP
// ============================================================
void stopWheelchair()
{
  setXY(CENTER_MV, CENTER_MV);
  movementActive = false;

  printState("STOP", CENTER_MV, CENTER_MV);
}


// ============================================================
// Apply a movement command
// ============================================================
void moveWheelchair(const char* state, uint16_t x_mV, uint16_t y_mV)
{
  setXY(x_mV, y_mV);

  movementActive = true;
  lastMovementCommand = millis();

  printState(state, x_mV, y_mV);
}


// ============================================================
// Process one character command
// ============================================================
void processCommand(char command)
{
  // Convert uppercase ASCII to lowercase
  if (command >= 'A' && command <= 'Z')
    command += ('a' - 'A');

  switch (command)
  {
    case 'w':
      // Forward: X=2.50 V, Y=0.55 V
      moveWheelchair("FORWARD", CENTER_MV, LOW_MV);
      break;

    case 's':
      // Backward: X=2.50 V, Y=4.45 V
      moveWheelchair("BACKWARD", CENTER_MV, HIGH_MV);
      break;

    case 'a':
      // Left: X=0.55 V, Y=2.50 V
      moveWheelchair("LEFT", LOW_MV, CENTER_MV);
      break;

    case 'd':
      // Right: X=4.45 V, Y=2.50 V
      moveWheelchair("RIGHT", HIGH_MV, CENTER_MV);
      break;

    case 'x':
    case ' ':
      stopWheelchair();
      break;

    // Ignore line endings
    case '\r':
    case '\n':
      break;

    default:
      // Any unexpected character is treated as STOP
      stopWheelchair();
      break;
  }
}


// ============================================================
// SETUP
// ============================================================
void setup()
{
  // USB serial for debugging
  Serial.begin(115200);

  // HC-06 default data-mode baud is commonly 9600
  HC06.begin(9600);

  Wire.begin();

  dacX.begin(DAC_X_ADDRESS);
  dacY.begin(DAC_Y_ADDRESS);

  delay(100);

  // Always start at neutral/stop
  stopWheelchair();

  Serial.println();
  Serial.println("Wheelchair Bluetooth Controller Ready");
  Serial.println("W=Forward, S=Backward, A=Left, D=Right, X/Space=Stop");
  Serial.println();
}


// ============================================================
// LOOP
// ============================================================
void loop()
{
  // Bluetooth commands
  while (HC06.available() > 0)
  {
    char command = HC06.read();
    processCommand(command);
  }

  // USB Serial commands also remain available for bench testing
  while (Serial.available() > 0)
  {
    char command = Serial.read();
    processCommand(command);
  }

  // Fail-safe: if movement refresh stops, force neutral
  if (movementActive &&
      (millis() - lastMovementCommand > COMMAND_TIMEOUT_MS))
  {
    stopWheelchair();
  }
}
