#include <Servo.h>

constexpr char DELIMITER = ':';
constexpr int OBJECT_ZONE = 1;    // мёртвая зона в градусах
constexpr int MOVE_INTERVAL = 10; // мс между обновлениями скорости

// 360° серво: 1500 мкс = стоп, <1500 = одна сторона, >1500 = другая
constexpr int STOP_US = 1500;
constexpr int MAX_SPEED_US = 200; // макс. отклонение от 1500 (диапазон 1300–1700)
constexpr int MIN_SPEED_US = 50;  // минимум чтобы мотор тронулся

struct Pair
{
    int h;
    int v;
};

// ─── ServoController360 ───────────────────────────────────────────────────────
class ServoController360
{
public:
    ServoController360(int pin, int object_zone = OBJECT_ZONE)
        : _pin(pin), _error(0), _speed_current(0.0f), _object_zone(object_zone) {}

    void attach()
    {
        _servo.attach(_pin);
        stop();
    }

    void add_error(int delta)
    {
        // Сглаживаем входящие данные
        _error = (int)(_error * 0.6f + (_error + delta) * 0.4f);
    }

    int error() const { return _error; }

    bool step()
    {
        if (abs(_error) <= _object_zone)
        {
            // Плавное торможение
            _speed_current *= 0.7f;
            if (abs(_speed_current) < 5.0f) {
                _speed_current = 0;
                stop();
                return false;
            }
            _servo.writeMicroseconds(STOP_US + (int)_speed_current);
            return true;
        }

        float speed_target = (float)clamp(_error * _err_mult, -MAX_SPEED_US, MAX_SPEED_US);

        if (speed_target > 0 && speed_target <  MIN_SPEED_US) speed_target =  MIN_SPEED_US;
        if (speed_target < 0 && speed_target > -MIN_SPEED_US) speed_target = -MIN_SPEED_US;

        // Low-pass фильтр скорости
        _speed_current = _speed_current * (1.0f - _alpha) + speed_target * _alpha;

        _servo.writeMicroseconds(STOP_US + (int)_speed_current);
        _error -= (int)_speed_current / _err_mult;

        return true;
    }

    void stop()
    {
        _servo.writeMicroseconds(STOP_US);
        _speed_current = 0.0f;
    }

private:
    Servo  _servo;
    int    _pin;
    int    _error;
    float  _speed_current;
    int    _object_zone;
    int    _err_mult = 20;
    float  _alpha    = 0.6f; // плавность: 0.1–0.5

    static int clamp(int val, int lo, int hi)
    {
        return (val < lo) ? lo : (val > hi) ? hi : val;
    }
};

// ─── AngleParser ──────────────────────────────────────────────────────────────
// Читает строки вида "12:-8\n" — угловые смещения в градусах
class AngleParser
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

        Serial.write((String(out.h) + " -- " + out.v + "\n").c_str());
        return true;
    }

private:
    char _buf[32];
};

// ─── CameraTracker ────────────────────────────────────────────────────────────
class CameraTracker
{
public:
    CameraTracker()
        : _horizontal(HORIZONTAL_PIN, OBJECT_ZONE),
          _vertical(VERTICAL_PIN, OBJECT_ZONE),
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
            // coords.h и coords.v — угловое смещение объекта от центра кадра
            _horizontal.add_error(coords.h);
            _vertical.add_error(coords.v);
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

    ServoController360 _horizontal;
    ServoController360 _vertical;
    AngleParser _parser;
    unsigned long _last_move;
};

// ─── Точка входа ──────────────────────────────────────────────────────────────
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