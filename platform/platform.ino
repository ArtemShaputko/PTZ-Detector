#include <Servo.h>

constexpr char DELIMITER     = ':';
constexpr int  OBJECT_ZONE   = 3;
constexpr int  MOVE_INTERVAL = 25;
constexpr int  MAX_STEP      = 5;

// ─── Pair ────────────────────────────────────────────────────────────────────

struct Pair {
    int h;
    int v;
};

// ─── ServoController ─────────────────────────────────────────────────────────

class ServoController {
public:
    ServoController(int pin, int initial_angle = 90)
        : _pin(pin), _current(initial_angle), _target(initial_angle) {}

    void attach() {
        _servo.attach(_pin);
        _servo.write(_current);
    }

    void set_target(int target) {
        _target = target;
    }

    int current() const { return _current; }

    bool step() {
        int diff = _target - _current;
        if (abs(diff) <= OBJECT_ZONE) return false;

        int s = clamp(diff / 4, -MAX_STEP, MAX_STEP);
        if (s == 0) s = (diff > 0) ? 1 : -1;

        _current = clamp(_current + s, 0, 180);
        _servo.write(_current);
        return true;
    }

private:
    Servo _servo;
    int   _pin;
    int   _current;
    int   _target;

    static int clamp(int val, int lo, int hi) {
        return (val < lo) ? lo : (val > hi) ? hi : val;
    }
};

// ─── CoordParser ─────────────────────────────────────────────────────────────

class CoordParser {
public:
    bool read(Pair& out) {
        if (!Serial.available()) return false;

        int len = Serial.readBytesUntil('\n', _buf, sizeof(_buf) - 1);
        _buf[len] = '\0';

        char* delim = strchr(_buf, DELIMITER);
        if (!delim) return false;

        *delim = '\0';
        out.h = atoi(_buf);
        out.v = atoi(delim + 1);
        return true;
    }

private:
    char _buf[32];
};

// ─── AngleCalculator ─────────────────────────────────────────────────────────

class AngleCalculator {
public:
    AngleCalculator(Pair fov, Pair resolution)
        : _fov(fov), _resolution(resolution) {}

    int calculate(float coord, bool horizontal) const {
        float size = horizontal ? _resolution.h : _resolution.v;
        float fov  = horizontal ? _fov.h        : _fov.v;
        float bias = coord - size / 2.0f;
        return int(bias * fov / size);
    }

private:
    Pair _fov;
    Pair _resolution;
};

// ─── CameraTracker ───────────────────────────────────────────────────────────

class CameraTracker {
public:
    CameraTracker()
        : _horizontal(HORIZONTAL_PIN),
          _vertical(VERTICAL_PIN),
          _calculator({58, 33}, {1920, 1080}),
          _last_move(0) {}

    void setup() {
        _horizontal.attach();
        _vertical.attach();
    }

    void update() {
        Pair coords;
        if (_parser.read(coords)) {
            _horizontal.set_target(_horizontal.current()
                                   + _calculator.calculate(coords.h, true));
            _vertical.set_target(_vertical.current()
                                 + _calculator.calculate(coords.v, false));
        }

        unsigned long now = millis();
        if (now - _last_move >= MOVE_INTERVAL) {
            bool moved = _horizontal.step() | _vertical.step();
            if (moved) _last_move = now;
        }
    }

private:
    static constexpr int HORIZONTAL_PIN = 4;
    static constexpr int VERTICAL_PIN   = 3;

    ServoController  _horizontal;
    ServoController  _vertical;
    CoordParser      _parser;
    AngleCalculator  _calculator;
    unsigned long    _last_move;
};

// ─── main ────────────────────────────────────────────────────────────────────

CameraTracker tracker;

void setup() {
    Serial.begin(115200);
    tracker.setup();
}

void loop() {
    tracker.update();
}