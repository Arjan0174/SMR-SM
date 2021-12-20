 #define outputA 2
 #define outputB 3
 #define inputHall 4
 #define outputPulse 5

 int counter = 0;
 int Pulse = 0;                   //pulses for knowing place of dakjes, gets reset when hall is active
 int aState;
 int aLastState;  
 int afstand;
 int stapafstand;
 long PosDif = 0;
 long TimeDif = 0;
 long newposition;
 long oldposition = 0;
 unsigned long newtime;
 unsigned long oldtime = 0;
 long vel;
 unsigned long SpeedCheckTime;

 void setup() { 
   pinMode (outputA,INPUT);
   pinMode (outputB,INPUT);
   pinMode (inputHall, INPUT);
   pinMode (outputPulse, OUTPUT);
   
   Serial.begin (115200);
   aLastState = digitalRead(outputA);     
 } 

 void loop(){
   float stapafstand = 1.2911;
   
   aState = digitalRead(outputA); // Reads the "current" state of the outputA
   // If the previous and the current state of the outputA are different, that means a Pulse has occured
   if (aState != aLastState){     
     // If the outputB state is different to the outputA state, that means the encoder is rotating clockwise
     if (digitalRead(outputB) != aState) { 
       counter ++;
       Pulse ++;
     }
     //else {
       //counter --;
       //Pulse --;

     if (digitalRead(inputHall) == 1) {
      delay (10);
      if (digitalRead(inputHall) == 1) {
      Pulse = 0;
      //Serial.println("HallSensor aan");
      delay (5);
     }
   }
   

     //Serial.print("Pulse :");
     //Serial.println(Pulse);
     float afstand = ((counter*stapafstand)/100);
   } 
   aLastState = aState; // Updates the previous state of the outputA with the current state 
   
   if (Pulse == 65 || Pulse == 160) {
    digitalWrite(outputPulse, HIGH);
    delay(10);
    digitalWrite(outputPulse,LOW);
   }

   
   if (millis() - SpeedCheckTime >= 0.5*1000UL)
   {
    SpeedCheckTime = millis();
    float newposition = (counter*stapafstand);
    Serial.print (oldposition,2);
    Serial.print(',');
    float PosDif = (newposition-oldposition);
   
    newtime = millis();
    TimeDif = (newtime-oldtime);

    float vel = (PosDif/(TimeDif/10));
    Serial.println (vel*1000,2);


    oldposition = newposition;
    oldtime = newtime;
   }
 }


 
 // D = ong 8.2cm
 // Omtrek = ong 28.76 cm
 // lengte band pak = ong 2.48m
 // lengte band leg = ong 2.48m
