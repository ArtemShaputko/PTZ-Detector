#include <Servo.h>

constexpr char DELIMITER = ':';
constexpr int OBJECT_ZONE = 4;
constexpr int MOVE_INTERVAL = 50;
constexpr int MAX_STEP = 3;

// ─── Pair ────────────────────────────────────────────────────────────────────

struct Pair
{
    int h;
    int v;
};
// ─── ServoController ─────────────────────────────────────────────────────────

class ServoController
{
public:
    ServoController(int pin, int object_zone = OBJECT_ZONE, int low_angle = 180, int high_angle = 0, int initial_angle = 90)
        : _pin(pin), _current(initial_angle), _target(initial_angle), _low_angle(low_angle), _high_angle(high_angle), _object_zone(object_zone) {}

    void attach()
    {
        _servo.attach(_pin);
        _servo.write(_current);
    }

    void set_target(int target)
    {
        _target = target;
    }

    int current() const { return _current; }

    bool step()
    {
        int diff = _target - _current;
        if (abs(diff) <= _object_zone)
            return false;

        int s = clamp(diff / 8, -MAX_STEP, MAX_STEP);
        if (s == 0)
            s = (diff > 0) ? 1 : -1;

        _current = clamp(_current + s, _low_angle, _high_angle);
        _servo.write(_current);
        return true;
    }

private:
    Servo _servo;
    int _pin;
    int _current;
    int _target;
    int _low_angle;
    int _high_angle;
    int _object_zone;

    static int clamp(int val, int lo, int hi)
    {
        return (val < lo) ? lo : (val > hi) ? hi
                                            : val;
    }
};

// ─── CoordParser ─────────────────────────────────────────────────────────────

class CoordParser
{
public:
    bool read(Pair &out)
    {
        if (!Serial.available())
            return false;

        int len = Serial.readBytesUntil('\n', _buf, sizeof(_buf) - 1);
        _buf[len] = '\0';

        char *delim = strchr(_buf, DELIMITER);
        if (!delim)
            return false;

        *delim = '\0';
        out.h = atoi(_buf);
        out.v = atoi(delim + 1);
        return true;
    }

private:
    char _buf[32];
};

// ─── AngleCalculator ─────────────────────────────────────────────────────────

class AngleCalculator
{
public:
    AngleCalculator(Pair fov, Pair resolution)
        : _fov(fov), _resolution(resolution) {}

    int calculate(float coord, bool horizontal) const
    {
        float size = horizontal ? _resolution.h : _resolution.v;
        float fov = horizontal ? _fov.h : _fov.v;
        float bias = coord - size / 2.0f;
        return int(bias * fov / size);
    }

private:
    Pair _fov;
    Pair _resolution;
};

// ─── CameraTracker ───────────────────────────────────────────────────────────

class CameraTracker
{
public:
    CameraTracker()
        : _horizontal(HORIZONTAL_PIN, int(OBJECT_ZONE * ratio) , LOW_ANGLES.h,  HIGH_ANGLES.h, 90),
          _vertical(VERTICAL_PIN, int(OBJECT_ZONE / ratio), LOW_ANGLES.v, HIGH_ANGLES.v),
          _calculator(FOVS, RES),
          _last_move(0) {}

    void setup()
    {
        _horizontal.attach();
        _vertical.attach();
    }

    void update()
    {
        Pair coords;
        if (_parser.read(coords))
        {
            _horizontal.set_target(_horizontal.current() + _calculator.calculate(coords.h, true));
            _vertical.set_target(_vertical.current() + _calculator.calculate(coords.v, false));
        }

        unsigned long now = millis();
        if (now - _last_move >= MOVE_INTERVAL)
        {
            _horizontal.step() | _vertical.step();
            _last_move = now;
        }
    }

private:
    static constexpr int HORIZONTAL_PIN = 4;
    static constexpr int VERTICAL_PIN = 3;
    static inline constexpr Pair FOVS = {58, 33};
    static inline constexpr Pair RES = {1920, 1080};
    static inline constexpr float ratio = RES.h/RES.v;
    static inline constexpr Pair LOW_ANGLES = {0, 39};
    static inline constexpr Pair HIGH_ANGLES = {180, 140};

    ServoController _horizontal;
    ServoController _vertical;
    CoordParser _parser;
    AngleCalculator _calculator;
    unsigned long _last_move;
};

// ─── main ────────────────────────────────────────────────────────────────────

CameraTracker tracker;

void setup()
{
    Serial.begin(9600);
    tracker.setup();
}

void loop()
{
    tracker.update();
}