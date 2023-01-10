const int PWM = 8;
const int STEERING_1 = 6;
const int STEERING_2 = 7;
const int FOWARD_RIGHT_1 = 2;
const int FOWARD_RIGHT_2 = 3;
const int FOWARD_LEFT_1 = 4;
const int FOWARD_LEFT_2 = 5;
const int POT = A1;
const int STEERING_SPEED = 128;
const int FOWARD_SPEED = 255;
int angle = 0, straight = 0, str_spd = 100, resistance = 0, mapped_resistance = 0;
char aaa = 'q', sss = 'w';

void right(){
  analogWrite(STEERING_1, STEERING_SPEED);
  analogWrite(STEERING_2, LOW);
}

void left(){
  analogWrite(STEERING_1, LOW);
  analogWrite(STEERING_2, STEERING_SPEED);
}

void stay(){
  analogWrite(STEERING_1, LOW);
  analogWrite(STEERING_2, LOW);
}

void foward(int st){
//  analogWrite(FOWARD_RIGHT_1, FOWARD_SPEED);
  analogWrite(FOWARD_RIGHT_1, st);
  analogWrite(FOWARD_RIGHT_2, LOW);
//  analogWrite(FOWARD_LEFT_1, FOWARD_SPEED);
  analogWrite(FOWARD_LEFT_1, st);
  analogWrite(FOWARD_LEFT_2, LOW);
}

void reverse(int st){
  analogWrite(FOWARD_RIGHT_1, LOW);
//  analogWrite(FOWARD_RIGHT_2, FOWARD_SPEED);
  analogWrite(FOWARD_RIGHT_2, st);
  analogWrite(FOWARD_LEFT_1, LOW);
//  analogWrite(FOWARD_LEFT_2, FOWARD_SPEED);
  analogWrite(FOWARD_LEFT_2, st);
}

void hold(){
  analogWrite(FOWARD_RIGHT_1, LOW);
  analogWrite(FOWARD_RIGHT_2, LOW);
  analogWrite(FOWARD_LEFT_1, LOW);
  analogWrite(FOWARD_LEFT_2, LOW);
}

void setup() {  
  Serial.begin(9600);
  pinMode(POT, INPUT);
  pinMode(STEERING_1, OUTPUT);
  pinMode(STEERING_2, OUTPUT);
  pinMode(FOWARD_RIGHT_1, OUTPUT);
  pinMode(FOWARD_RIGHT_2, OUTPUT);
  pinMode(FOWARD_LEFT_1, OUTPUT);
  pinMode(FOWARD_LEFT_2, OUTPUT);
  pinMode(PWM, OUTPUT);
  digitalWrite(PWM, HIGH);
}


void loop() {

  if (Serial.available()){
    if(Serial.peek() == 'a'){
      aaa = Serial.read();
      angle = Serial.parseInt();
      
    }
    if(Serial.peek() == 's'){
      sss = Serial.read();
      straight = Serial.parseInt();
    }
    if(0){
    if (angle >= 50 || angle <= -50){
      int straight_temp = angle;
      angle = straight;
      straight = straight_temp;
    }

    if ( (-15 <= straight && straight < 0) || (0 < straight && straight <= 15)){
      int angle_temp = straight;
      straight = angle;
      angle = angle_temp;
    }
    }
    if(0){
    Serial.print("straight: "); 
    Serial.print(sss);
    Serial.println(straight);
    Serial.print(" angle: ");
    Serial.print(aaa);
    Serial.print(angle);
    }
    resistance = analogRead(POT);
    mapped_resistance = map(resistance, 872, 757, -7, 7);
    
    if(1){
    Serial.print(" Read/Map [A1]/[b]: ");  
    Serial.print(resistance);
    Serial.print(" / ");
    Serial.println(mapped_resistance);
    }
    
    if (straight > 0){
      //Serial.println("-----------");
      foward(straight);
    }
    else if (straight == 0){
      hold();
    }
    else if (straight < 0){
      //Serial.println("-----------");
      reverse(abs(straight));
    }

    if (mapped_resistance == angle){
      stay();
    }
    else if (mapped_resistance > angle){
      left();
    }
    else if (mapped_resistance < angle){
      right();
    }
  }
  else{
//    resistance = analogRead(POT);
//    mapped_resistance = map(resistance, 160, 275, -5, 5);
//
//    if (straight > 0){
//      Serial.println("-----------");
//      foward(straight);
//    }
//    else if (straight == 0){
//      hold();
//    }
//    else if (straight < 0){
//      Serial.println("-----------");
//      reverse(abs(straight));
//    }
//
//    if (mapped_resistance == angle){
//      stay();
//    }
//    else if (mapped_resistance > angle){
//      left();
//    }
//    else if (mapped_resistance < angle){
//      right();
//    }
  }
}
