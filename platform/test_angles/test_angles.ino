#include <Servo.h>

int HORIZONTAL_PIN = 4;
int VERTICAL_PIN = 3;

Servo horiz;
Servo vert;

void setup() {
  horiz.attach(HORIZONTAL_PIN);
  vert.attach(VERTICAL_PIN);
  Serial.begin(115200);
  //   // left
  // for(int i = 45; i >= 0; i--) {
  //   horiz.write(i);
  //   delay(1000);
  //   Serial.println(String("Left ") + i);
  // } // 0
  // //right
  // for(int i = 135; i <= 180; i++) {
  //   horiz.write(i);
  //   delay(1000);
  //   Serial.println(String("Right ") + i);
  // } // 180
  //top
  // for(int i = 60; i >= 39; i--) {
  //   vert.write(i);
  //   delay(1000);
  //   Serial.println(String("Top ") + i);
  // }  // 40
  //bottom
  horiz.write(180);
  for(int i = 135; i <= 140; i++) {
    vert.write(i);
    delay(1000);
    Serial.println(String("Bottom ") + i);
  } // 168
}

void loop() {
  delay(10000);
}
