int counter = 0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  while (!Serial) {}
}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available() > 0) {
    String message = Serial.readStringUntil("\n");
    message = message + "" + String(counter);
    counter++;
    Serial.println(message);
  }
}
