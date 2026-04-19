/*
 * HEAPGrasp — Servo Gripper Controller
 * Arduino Nano · MG996R servo
 *
 * Wiring:
 *   MG996R brown/black → GND
 *   MG996R red         → battery 5-6 V (NOT Arduino 5V — servo draws up to 2.5 A)
 *   MG996R orange      → D9  (PWM signal, share GND with Arduino)
 *
 * Serial protocol (115200 baud, CR+LF terminated):
 *   OPEN          → move to OPEN_ANGLE
 *   CLOSE         → move to CLOSE_ANGLE
 *   ANGLE <deg>   → move to exact angle (0–180)
 *   STATUS        → reply current angle
 *   STOP          → detach servo (saves power / stops jitter)
 *
 * Replies: "OK <data>\r\n"  or  "ERR <reason>\r\n"
 *
 * Calibration:
 *   Adjust OPEN_ANGLE and CLOSE_ANGLE to match your gripper geometry.
 *   Run ANGLE commands from the dashboard to find the right values first.
 */

#include <Servo.h>

#define SERVO_PIN   9
#define OPEN_ANGLE  30    // degrees — jaws fully open
#define CLOSE_ANGLE 150   // degrees — jaws fully closed

Servo gripper;
int currentAngle = OPEN_ANGLE;

void setup() {
    Serial.begin(115200);
    gripper.attach(SERVO_PIN);
    gripper.write(OPEN_ANGLE);
    currentAngle = OPEN_ANGLE;
}

// ── Command parser ────────────────────────────────────────────────────────────
String inputBuffer = "";

void setAngle(int deg) {
    deg = constrain(deg, 0, 180);
    gripper.attach(SERVO_PIN);   // re-attach in case STOP was called
    gripper.write(deg);
    currentAngle = deg;
}

void processCommand(String cmd) {
    cmd.trim();

    if (cmd == "STATUS") {
        Serial.print("OK ");
        Serial.println(currentAngle);

    } else if (cmd == "OPEN") {
        setAngle(OPEN_ANGLE);
        Serial.println("OK OPEN");

    } else if (cmd == "CLOSE") {
        setAngle(CLOSE_ANGLE);
        Serial.println("OK CLOSE");

    } else if (cmd == "STOP") {
        gripper.detach();
        Serial.println("OK STOPPED");

    } else if (cmd.startsWith("ANGLE ")) {
        int deg = cmd.substring(6).toInt();
        setAngle(deg);
        Serial.print("OK ANGLE ");
        Serial.println(currentAngle);

    } else if (cmd.length() > 0) {
        Serial.print("ERR UNKNOWN:");
        Serial.println(cmd);
    }
}

void loop() {
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\n') {
            processCommand(inputBuffer);
            inputBuffer = "";
        } else if (c != '\r') {
            inputBuffer += c;
        }
    }
}
