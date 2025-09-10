/* 
  Partimos de: 1_EscaneoyVuelos
  Escanea --> Procesa la imagen --> Inferencia en Python --> Simulacion de vuelo

  Vamos a ir desilenciando los fragmentos pertinentes a la interefencia poco a poco:
  - Desilenciar la parte Global:  #include<MicroTFLite.h> ...        Ya le cuesta más compilar y, a veces, falla en trasmitir 1-2. PERO funcionando bien 
  - Desilenciar la parte Setup: Serial.println("Initializing ....    Le cuesta compilar.  PERO inesperadamente, sigue funcionando y NO ha fallado en la transmision
  - Desilenciar la parte Loop

  FUNCIONAAAAAAAAAAAAAAAAAA HACE LA INFERENCIA EN EL ARDUINOOOOOOO
  

*/

// Libreriac comunes a ServosCNN y Vuelos:
#include <Arduino.h>

// Libreria sendos cores:
#include <Scheduler.h>   // Se ha de iniciar con .startLoop(loop_2) en el setup


/*####################################################################################################################################################################################################################### 
################################################################################## Servos  y  C N N (bloque 1) ##########################################################################################################
####################################################################################################################################################################################################################### */
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - Lo pertinente al E S C A N E O - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  //
#include <Servo.h>   
#include <LIDARLite.h>

Servo servoX, servoY; 
LIDARLite myLidarLite;

#define ArcX 90               //Poner arcos PARES. Lio innecesario sino ArcX/2. Ej: Si angleX0 = 90 y ArcX = 60. l barrido irá de 60 a 120
#define ArcY 45
#define deltaX 1
#define deltaY 1
#define angleX0 90            // CUIDADO no poner un angleX0 < ArcX/2 obviamente

int angleX = 0;
int angleY = 0;
volatile char mssg = '\0';    // Inicializado con carácter nulo (char es mas eficiente, ocupa menos memoria que un int)    Volatile --> Le dice al compilador “¡Oye! Esta variable puede cambiar en cualquier momento, incluso fuera del flujo normal del programa. No la optimices ni la caches.”
float distance = 0;

int pos = 0;  // Contador de la posicion del input_tensor
float max_distance = 0, min_distance = 829; // Smart XDD
float input_tensor[(ArcY+1)*(ArcX+1)];
float ultima_fila[ArcX+1];

bool Inferir = false;   // Esto evita meter la inferencia en el If mssg==N asi se "separa la logica de eventos" quedando más limpio. Se inicializa FALSE, al final de If mssg==N se cambia a TRUE, de modo que se 

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - Lo pertinente a la I N F E R E N C I A - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
#include<MicroTFLite.h>
#include "ModeloCNN_8-2A_F1avg0.82_quant.h"  // "ModeloCNNLight_5_1x1N4_quantized_SinDW.h"// Accede al archivo de cabecera desde este script .ino
//#include "15_H135_float32.h"  // modelo de aeronave en vector pto. FLOTANTE, hay 4 diferentes.   NO se pueden evaluar varios modelos a la vez (se desborda la memoria RAM)
constexpr int kTensorArenaSize = 120 * 1024; // mínimo 5*tamaño dado en el archivo .h de la CNN. Cuidado con el tamaño máximo!
alignas(16) uint8_t tensorArena[kTensorArenaSize];

#define NUM_INPUTS 46*91
#define NUM_OUTPUTS 4  // Numero de CLASES 
float maxProb = -1.0;  // Probabilidad máxima encontrada (para poder mandar el vuelo)
/*####################################################################################################################################################################################################################### 
################################################################################# Servos  y  C N N (fin bloque 1) ########################################################################################################
####################################################################################################################################################################################################################### */


/*####################################################################################################################################################################################################################### 
############################################################################### Simulaciones V U E L O (bloque 1) #######################################################################################################
####################################################################################################################################################################################################################### */
// Librerias de los vuelos:
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
/*------------------------------------------------------------
  Parámetros globales (igual espíritu que en Python)
------------------------------------------------------------*/
float N_gain = 4.0;  // Pro-Nav Gain (típicamente 3–5) 

float p_m[3] = {0.0, 0.0, 0.0};        // Misil [N, E, D]  (m) 
float v_m[3] = {300.0, 300.0, -750.0}; // [vN, vE, vD] (m/s) 
float p_t[3] = {15000.0, 15000.0, -11000.0}; // Target [N, E, D] (m) 
float v_t[3] = {0.0, 0.0, 0.0};              // [vN, vE, vD] (m/s) 

float dk = 0.01;        // Paso temporal (s)   NO se puede usar dt porque dt ya está definido en 'double_tap_usb_boot.cpp' .Parte del sistema de arranque del Arduino Nano RP2040 Connect. Lo que provoca un conflicto de nombres
float S  = 80.0;        // Duración simulación (s) 
int   N_pasos;          // Número de iteraciones 

float u_max = 40.0;     // Límite de aceleración (m/s^2). Se ajusta por caso de vuelo 
float eg[3] = {0.0, 0.0, 1.0};   // Vector unitario gravedad (Down positivo) 
float blast_radius = 170.0;      // Distancia de intercepción (m) 

volatile int vuelo = 3;         // CASO DE VUELO
              // Vuelo 0: Destaca como el misil va A BUSCAR al avion      Lineas: 
              // Vuelo 1: Evasion, ascenso ELEGANTE                       Lineas: 3400
              // Vuelo 2: Aire - Aire al CHAFF                            Lineas: 
              // Vuelo 3: Posicion de espera                              Lineas: 7300

float centro[3] = {0.0, 0.0, 0.0}; // Centro de órbita para Vuelo3 

float chaff[3] = {0,0,0};      // En que posicion se dispara el chaff
int chaff_k = 4000;            // En que k se dispara el chaff
float p_m_last[3] = {0,0,0}; 
float p_t_temp[3] = {0.0, 0.0, 0.0};

//int mssg = 0; // Ya definido en el Bloque 1 de la CNN
bool sim_started = false;

/*------------------------------------------------------------
  Utilidades vectoriales
------------------------------------------------------------*/
void ProductoVectorial(const float a[3], const float b[3], float r[3]) {
    r[0] = a[1]*b[2] - a[2]*b[1];
    r[1] = a[2]*b[0] - a[0]*b[2];
    r[2] = a[0]*b[1] - a[1]*b[0];
}

float Normalizar(float v[3]) {
    float m = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);

    if (m < 0.001) {
      Serial.println("Maginutd 0!!");
    } 
    v[0] /= m; v[1] /= m; v[2] /= m;
    return m;
}

/*------------------------------------------------------------
  Perfiles de vuelo del target 
------------------------------------------------------------*/
void Vuelo1(float p_t[3], float v_t[3]) {
    /* Acel. simple: E y D cambian, N constante */
    v_t[0] += 0.0;             /* [N, E, D] */
    v_t[1] += 15.0 * dk;
    v_t[2] += -15.0 * dk;

    p_t[0] += v_t[0] * dk;
    p_t[1] += v_t[1] * dk;
    p_t[2] += v_t[2] * dk;
}

void Vuelo2(float p_m[3], float p_t[3], float p_t_temp[3], float chaff[3], int k) {      // CHAFFS 
    float p_rel[3];
    // Posición relativa
    for (int i = 0; i < 3; i++) {
        p_rel[i] = p_t[i] - p_m[i];
    }

    // ANTES del Chaff
    if (k <= chaff_k) {
      v_t[0] += 2.5 * dk;
      v_t[1] += 0 * dk;

      p_t[0] += v_t[0] * dk;
      p_t[1] += v_t[1] * dk;
      p_t[2] += v_t[2] * dk;

      p_t_temp[0] = p_t[0];
      p_t_temp[1] = p_t[1];
      p_t_temp[2] = p_t[2];

      for (int i = 0; i < 3; i++) {
          chaff[i] = p_t[i];
      }
    }

    //  DISPERSION del Chaff
    else if (k > chaff_k && k < 5000) {
      v_t[0] += 0.5 * dk;
      v_t[1] += 0 * dk;

      p_t_temp[0] += v_t[0] * dk;
      p_t_temp[1] += v_t[1] * dk;
      p_t_temp[2] += v_t[2] * dk;

      p_t[0] = chaff[0] + random(-3000, 3001);  // random(a, b) en Arduino es [a, b)
      p_t[1] = chaff[1] + random(-1500, 1501);
      p_t[2] = chaff[2] + random(-100, 101);
    }

    // Fase final del vuelo
    else if (k >= 5000) {
      v_t[0] += 0.1 * dk;
      v_t[1] += 0 * dk;

      p_t_temp[0] += v_t[0] * dk;
      p_t_temp[1] += v_t[1] * dk;
      p_t_temp[2] += v_t[2] * dk;

      p_t[0] = p_t_temp[0];
      p_t[1] = p_t_temp[1];
      p_t[2] = p_t_temp[2];
    }
}

void Vuelo3(float p_t_[3], float v_t_[3], int k) {    // Orbita circular a altura constante (v ~ const) 
    // Parámetros de la circunferencia 
    const float R = 2500.0;  // ¡Debe ser consistente con 'centro'!
    const float v = 150.0;   // Velocidad lineal aprox. 
    const float a_n = (v*v) / R;

    // Vector desde posición a centro (dirección hacia el centro) 
    float a_dir[3] = { centro[0] - p_t_[0], centro[1] - p_t_[1], centro[2] - p_t_[2] };
    /* Ángulo en el plano N-E */
    float theta = atan2(a_dir[1], a_dir[0]);

    float a_n_N = a_n * cos(theta);
    float a_n_E = a_n * sin(theta);

    v_t_[0] += a_n_N * dk;
    v_t_[1] += a_n_E * dk;
    v_t_[2] += 0.0;

    p_t_[0] += v_t_[0] * dk;
    p_t_[1] += v_t_[1] * dk;
    /* Mantener altura constante (centro[2]) */
    p_t_[2] = centro[2];
}


/*------------------------------------------------------------
  Runge - Kutta
------------------------------------------------------------*/
void Limitar_aceleracion(float a_cmd[3], float eg[3], float v_m[3]) {
  float ey[3], ez[3];

  // Calculo de los ejes Y y Z montados en el misil
  ProductoVectorial(eg, v_m, ey);
  Normalizar(ey);

  ProductoVectorial(v_m, ey, ez);
  Normalizar(ez);

  // Componente lateral y vertical
  float u_y = ey[0]*a_cmd[0] + ey[1]*a_cmd[1] + ey[2]*a_cmd[2];
  float u_z = ez[0]*a_cmd[0] + ez[1]*a_cmd[1] + ez[2]*a_cmd[2];

  float modulo = sqrt(u_y*u_y + u_z*u_z);

  float u_y_lim, u_z_lim;

  if (modulo > u_max) {
    float factor = u_max / modulo;
    u_y_lim = u_y * factor;
    u_z_lim = u_z * factor;

    // Reconstruir el vector de aceleración corregido
    for (int i = 0; i < 3; i++) {
      a_cmd[i] = u_y_lim * ey[i] + u_z_lim * ez[i];
    }

    /*Serial.print("Limitadas -> u_z_lim: ");
    Serial.print(u_z_lim);
    Serial.print("  u_y_lim: ");
    Serial.print(u_y_lim);
    Serial.print("  módulo corregido: ");
    Serial.print(sqrt(u_y_lim*u_y_lim + u_z_lim*u_z_lim));
    Serial.println(" m/s");*/
  } else {
    u_y_lim = u_y;
    u_z_lim = u_z;
    for (int i = 0; i < 3; i++) {
      a_cmd[i] = a_cmd[i];
    }
  }

  float modulo_final = sqrt(u_y_lim*u_y_lim + u_z_lim*u_z_lim);
  if (modulo_final > (u_max + 1.0)) {
    Serial.println("Misil volatilizado por exceso de Gs");
  }
}

void Calcular_aceleracion(const float p_m_temp[3], const float v_m_temp[3],   // Remind float a_cmd[3] =  float* a_cmd asiq, en
                          const float p_t_temp[3], const float v_t_temp[3],   float* a_cmd) {   
    float p_rel[3], v_rel[3];
    for (int i = 0; i < 3; ++i) {  // Posición y velocidad relativas
        p_rel[i] = p_t_temp[i] - p_m_temp[i];
        v_rel[i] = v_t_temp[i] - v_m_temp[i];
    }
    // lambda_dot = (p_rel x v_rel) / |p_rel|^2 
    float lambda_dot[3];
    ProductoVectorial(p_rel, v_rel, lambda_dot);
    
    float dist_sq; // El modulo al cuadrado  de p_rel,   dist_sq = |p_rel|^2
    dist_sq = p_rel[0]*p_rel[0] + p_rel[1]*p_rel[1] + p_rel[2]*p_rel[2];

    for (int i = 0; i < 3; i++) {
      lambda_dot[i] = lambda_dot[i] / dist_sq;
    }

    // a_cmd = N * (v_rel x lambda_dot) 
    ProductoVectorial(v_rel, lambda_dot, a_cmd);

    a_cmd[0] = N_gain * a_cmd[0];
    a_cmd[1] = N_gain * a_cmd[1];
    a_cmd[2] = N_gain * a_cmd[2];

    /* Limitar módulo a u_max */      // El 2 a_cmd del argumento es para reescribirlo
    Limitar_aceleracion(a_cmd,eg, v_m);
}

void RK4_step(float p_m_[3], float v_m_[3],
                     const float p_t_[3], const float v_t_[3]) {
    float a1[3], a2[3], a3[3], a4[3];   // <-- a_cmd
    float k1_p[3], k1_v[3];
    float k2_p[3], k2_v[3];
    float k3_p[3], k3_v[3];
    float k4_p[3], k4_v[3];

    float p_temp[3], v_temp[3];
    float p_t_temp[3] = { p_t_[0], p_t_[1], p_t_[2] };
    float v_t_temp[3] = { v_t_[0], v_t_[1], v_t_[2] };

    /* k1 */
    Calcular_aceleracion(p_m_, v_m_, p_t_temp, v_t_temp, a1);
    for (int i = 0; i < 3; ++i) { k1_p[i] = v_m_[i]; k1_v[i] = a1[i]; }

    /* k2 */
    for (int i = 0; i < 3; ++i) {
        p_temp[i] = p_m_[i] + 0.5 * dk * k1_p[i];
        v_temp[i] = v_m_[i] + 0.5 * dk * k1_v[i];
    }
    Calcular_aceleracion(p_temp, v_temp, p_t_temp, v_t_temp, a2);
    for (int i = 0; i < 3; ++i) { k2_p[i] = v_temp[i]; k2_v[i] = a2[i]; }

    /* k3 */
    for (int i = 0; i < 3; ++i) {
        p_temp[i] = p_m_[i] + 0.5 * dk * k2_p[i];
        v_temp[i] = v_m_[i] + 0.5 * dk * k2_v[i];
    }
    Calcular_aceleracion(p_temp, v_temp, p_t_temp, v_t_temp, a3);
    for (int i = 0; i < 3; ++i) { k3_p[i] = v_temp[i]; k3_v[i] = a3[i]; }

    /* k4 */
    for (int i = 0; i < 3; ++i) {
        p_temp[i] = p_m_[i] + dk * k3_p[i];
        v_temp[i] = v_m_[i] + dk * k3_v[i];
    }
    Calcular_aceleracion(p_temp, v_temp, p_t_temp, v_t_temp, a4);
    for (int i = 0; i < 3; ++i) { k4_p[i] = v_temp[i]; k4_v[i] = a4[i]; }

    /* Actualizar estado del misil */
    for (int i = 0; i < 3; ++i) {
        p_m_[i] += (dk/6.0) * (k1_p[i] + 2.0*k2_p[i] + 2.0*k3_p[i] + k4_p[i]);
        v_m_[i] += (dk/6.0) * (k1_v[i] + 2.0*k2_v[i] + 2.0*k3_v[i] + k4_v[i]);
    }
}
/*####################################################################################################################################################################################################################### 
############################################################################# Simulaciones V U E L O (fin bloque 1) #####################################################################################################
####################################################################################################################################################################################################################### */



void setup() {
  Serial.begin(115200);
  delay(1500);

  /*####################################################################################################################################################################################################################### 
  ################################################################################## Servos  y  C N N (bloque 2) ##########################################################################################################
  #######################################################################################################################################################################################################################*/
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - Lo pertinente al E S C A N E O - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -//
  //Wire.begin();                // Con el Lite no hace falta, pero por si acaso...
  myLidarLite.begin(0, false); // Set configuration to default (false) and I2C to 100 kHz
  myLidarLite.configure(0);    // Change this number to try out alternate configurations

  // Config para usar el Tools --> Board --> Arduino Mbed OS --> Arduino Nano RP240 Connect
  servoX.attach(9);           //servo.attach(9) indicará a la biblioteca que hemos conectado el servo 1 al pin 9, servo HORIZONTAL
  servoY.attach(10);          //pin 10, servo VERTICAL

  //PRUEBA SERVOS:
  servoX.write(45);      
  servoY.write(45);
  delay(1500);          
  servoX.write(90);       // NO poner 180 xq pega el tornillo lidar-placa y se va desaflojando el tornillo placa-servoY
  servoY.write(90);
  delay(1500);
  servoX.write(angleX0);       
  servoY.write(0);
  delay(1500);            
  //Este último delay es para COLOCAR A MANO EL SOPORTE del LIDAR xq con los tornillos flojos queda descentrado
  
  angleX = angleX0 - ArcX/2;      //Esto situa el LIDAR en el MINIMO angulo de X, para poder empezar a barrer del tiron +deltaX
  servoX.write(angleX); 

  pinMode(LED_BUILTIN, OUTPUT);  // Configura el pin del LED como salida

 // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - Lo pertinente a la I N F E R E N C I A - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -//
  // I N I C I A L I Z A R    E L    M O D E L O    CNN (según el Readme)
  Serial.println("Initializing TensorFlow Lite Micro Interpreter...");
  if (!ModelInit(modelo_tflite, tensorArena, kTensorArenaSize)){//ModeoCNNLight_5_1x1N4_quantized_SinDW_tflite, tensorArena,kTensorArenaSize))//
    Serial.println("Model initialization failed!");
    while (true);
  }
  Serial.println("Model initialization done.");
  // Las tres líneas dan información sobre el modelo de CNN incluido en el .h por el puerto serie (si queremos verlas usar el SERIAL DEL IDE DE ARDUINO)
  // Se puede hacer "esperar" a que Python mande 'R' para que no s epierda esta info. Pero entonces el Arduino se raya mucho
  ModelPrintMetadata();
  ModelPrintTensorQuantizationParams();
  ModelPrintTensorInfo();
  delay(100);
  /*####################################################################################################################################################################################################################### 
  ################################################################################# Servos  y  C N N (fin bloque 2) ########################################################################################################
  ####################################################################################################################################################################################################################### */


  /*####################################################################################################################################################################################################################### 
  ############################################################################### Simulaciones V U E L O (bloque 2) #######################################################################################################
  ####################################################################################################################################################################################################################### */
  
  // vuelo == Random  o cualquier otro: usar los valores por defecto que ya están puestos 

  /*####################################################################################################################################################################################################################### 
  ############################################################################# Simulaciones V U E L O (fin bloque 2) #####################################################################################################
  ####################################################################################################################################################################################################################### */

  Scheduler.startLoop(loop_2);  // inicia el segundo proceso, con el nombre que quieras ponerle

}
 
void loop() {

  if (Serial.available() > 0){  // Si tienes varios bloques Serial.read() dispersos, pueden competir por los mismos datos y leer cosas incompletas o equivocadas.
      mssg = Serial.read();    //Leer el comando enviado desde el script de Python
  }

  /*####################################################################################################################################################################################################################### 
  ############################################################################## Servos  y  C N N (bloque 3 - loop) #######################################################################################################
  ####################################################################################################################################################################################################################### */
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - Lo pertinente al E S C A N E O - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
  
    if(mssg == 'S'){              //Si el comando es 'S'tart Comienza la lectura  
      delay(80);
      servoY.write(angleY);       //El servo se situa a la altura fijada
      delay(80);
     
      if(angleY < ArcY){          //Mientras la altura se mantiene ArcY, se va sumando 1º (deltaY)
        angleY += deltaY;
      } 
      else{ 
        angleY = 0;               //En el momento que >90ºC reiniciamos el angleY
      }
      
      if(angleX < angleX0 + ArcX/2){           //Si X=0 i.e. posición inicial. A la altura dada, se barre horizontalmente
        for(angleX = angleX0 - ArcX/2; angleX <= angleX0 + ArcX/2; angleX += deltaX){
          servoX.write(angleX);
          delay(15);

          distance = myLidarLite.distance();
          delay(10);
          // Rarisimo pero los OUTLIERS se han disparado pese a que los codigos de escaneo son los mismos. Asi que se corrigen a mano:
          if(distance > 1000){distance = random(195, 206);;}  // Con esto corregimos el problema de los outliers y se colocan a la distancia del fondo estándar (mi pared)
          if(distance < 2){distance = 50;}  

          Serial.println(distance);   // Realmente lo "mas correcto" es agrupar datos y enviar la agrupacion para no "saturar el buffer" pero vamos, que así funciona bien
          
          input_tensor[pos] = distance;
          pos += 1;
          if(distance > max_distance){max_distance = distance;}
          if(distance < min_distance){min_distance = distance;}
                    
        }
      }

      //Vuelve despacito
      for(angleX = angleX0 + ArcX/2; angleX >= angleX0 - ArcX/2; angleX -= 2*deltaX){
          servoX.write(angleX);
          delay(10);
      }
      servoX.write(angleX0 - ArcX/2);
      delay(10);

      Serial.flush();
      mssg = '\0';
    }
    
    else if (mssg == 'N'){
      servoX.write(angleX0); servoY.write(0);
      delay(50);
      servoX.detach(); servoY.detach();
      // - - - - - - - - - - - - - - - - N o r m a l i z a r   a   1   y    m a n d a r   a   P y t h o n - - - - - - /
      for (pos = 0; pos < ((ArcY + 1) * (ArcX + 1)); pos++) { // Pasar a escala de grises "PROCESADO"
        // Paso 1: Normalización entre 0 y 255   -   Pasar de RAW a [0,255] se hace en "2_PasarTXTaPNG_VariosTXT"   
        input_tensor[pos] = ((input_tensor[pos] - min_distance) / (max_distance - min_distance)) * 255;

        // Paso 2: Escalado entre 0 y 1      -     El /255 para que quede decimal lo añadió Nicolas en el "3_Aplana_input_float"
        input_tensor[pos] = input_tensor[pos] / 255.0;
      }

      /* 
      for (pos = 0; pos < ArcX+1; pos++) {    // Cambiar ultima fila <--> primera fila. De momento lo dejo silencido xq PARECE que no hce falta (a veces...)
        ultima_fila[pos] = input_tensor[pos];
        input_tensor[pos] = input_tensor[(ArcY+1)*(ArcX+1) - ArcX+1 + pos];   // smart
        input_tensor[(ArcY+1)*(ArcX+1) - ArcX+1 + pos] = ultima_fila;
      }*/
      for(pos = 0; pos < ((ArcY+1)*(ArcX+1)); pos++){ // Enviar a Python
        Serial.println(input_tensor[pos]);
        delay(1); // 1 ms entre envios para no saturar el Buffer
        //Serial.println("142 ");

      }
 
      // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - /
      pos = 0;   // Se deja todo ready para empezar otro if"S"
      //Inferir = true; 
      max_distance = 0, min_distance = 829;

      Serial.flush();
      mssg = '\0';
      Inferir = true;
    }
  
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - Lo pertinente a la I N F E R E N C I A - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
  //Quizas ESTO debería ir dentro del propio     else if (mssg == 'E')    pero bueno, de momento pruebo si fuinciona así
  if(Inferir){
    Serial.println("Inferencia del RP2040:"); // En prueba
    for (int i = 0; i < NUM_INPUTS; i++){
      if (!ModelSetInput(input_tensor[i],i)) // input_tensor definido en faero1.h, es el vector de imagen. Se introduce componente a componente 
        Serial.println("ooohhh");//inputSetFailed = true;
    }
  
    if (!ModelRunInference()){
      Serial.println("RunInference Failed!");
      return;
    }

    Serial.println(" Class0_F-16   Class1_Su-35    Class2_H-135   Class3_A-50");    Serial.print(" ");
    for (int i = 0; i < NUM_OUTPUTS; i++){   // get output values and print as percentage
      Serial.print(ModelGetOutput(i), 6);    // Salidas 1 a 1.
      Serial.print(",     ");
    } 
    for (int i = 0; i < NUM_OUTPUTS; i++) {  // Determinar la clase --> Tipo de vuelo
      if (ModelGetOutput(i) > maxProb) {
        maxProb = ModelGetOutput(i);
        vuelo = i;  
      }
    }
    Serial.print("\nClase con mayor probabilidad: "); Serial.println(vuelo);
      // - - - - - - - - - Setup de vuelos - - - - - - - - - - - - //
        if (vuelo == 0) {   // Vuelo cte
          p_m[0]=0; p_m[1]=0; p_m[2]=0;
          v_m[0]=250; v_m[1]=250; v_m[2]=-600;
          p_t[0]=15000; p_t[1]=11000; p_t[2]=-3000;
          v_t[0]=-450; v_t[1]=0; v_t[2]=0;
          u_max = 390;
          blast_radius = 20;
        }
        if (vuelo == 1) {   // Intento de evasion elegante
          p_m[0]=0; p_m[1]=0; p_m[2]=0;
          v_m[0]=250; v_m[1]=250; v_m[2]=-600;
          p_t[0]=5000; p_t[1]=20000; p_t[2]=-3000;
          v_t[0]=-400; v_t[1]=0; v_t[2]=0;
          u_max = 390;
          blast_radius = 20;
        }
        if (vuelo == 2) {   // Helicoptero Chafs
          p_m[0]=0.0;   p_m[1]=0.0;   p_m[2]=-5200.0;
          v_m[0]=400.0; v_m[1]=400.0; v_m[2]=0.0;
          p_t[0]=23200.0; p_t[1]=10000.0; p_t[2]=-5000.0;
          v_t[0]=0.0; v_t[1]=100.0; v_t[2]=0.0;
          u_max = 390.0;
          blast_radius = 15.0;
          randomSeed(41);  // Fijamos la semilla para números aleatorios   NOOOO es necesario #include <random>
        }
        if (vuelo == 3) {   // Vuelo en espera
          p_m[0]=0; p_m[1]=0; p_m[2]=0;
          v_m[0]=200; v_m[1]=200; v_m[2]=-350;
          p_t[0]=27500; p_t[1]=25000; p_t[2]=-10000;
          centro[0]=25000; centro[1]=25000; centro[2]=-10000;
          v_t[0]=0; v_t[1]=150; v_t[2]=0;
          u_max = 120;
          blast_radius = 60;
        }
    Serial.println("FIN");
    Inferir = false;    // loop() se ejecuta de principio a fin antes de comenzar una nueva iteración. Con esto garantizamos que la inferencia se realiza 1 sola vez
    delay(1000);
  }
}


void loop_2() {  // NO tiene setup
  
  // Pin de TEST para comprobar que el Loop 2 está funcionando mientras el LiDAR escanea
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);
  delay(30);
  digitalWrite(LED_BUILTIN, LOW);
  delay(30);

  /*####################################################################################################################################################################################################################### 
  ########################################################################### Simulaciones V U E L O (bloque 3 - loop) ####################################################################################################
  ####################################################################################################################################################################################################################### */
    if (!sim_started && mssg == 'F') {  // Si el comando es 'F'ire Comienza el cálculo
      //servoX.write(40); servoY.write(0);  // Si en 'F' no estás actualizando el servo, pero tampoco lo detienes, puede quedar en un estado de “flotación” que lo hace temblar.
      sim_started = true;

      // Número de pasos 
      N_pasos = (int)(S / dk);

      // Init sim: avanzar una vez posiciones según velocidad actual (igual que Python) 
      for (int i = 0; i < 3; ++i) {
        p_m[i] += v_m[i] * dk;
        p_t[i] += v_t[i] * dk;
      }

      //  # # # # # # # # # # # # # # # # # # #   B U C L E   P R I N C I P A L   # # # # # # # # # # # # # # # # # # # //
      for (int k = 0; k < N_pasos; ++k) {
        // Integrar misil con RK4 (target fijo durante subpasos) 
        RK4_step(p_m, v_m, p_t, v_t);

        // Actualizar target según perfil 
        if (vuelo == 0) {
          p_t[0] += v_t[0] * dk;
          p_t[1] += v_t[1] * dk;
          p_t[2] += v_t[2] * dk;
        } else if (vuelo == 1) {
          Vuelo1(p_t, v_t);
        } else if (vuelo == 2) {
          Vuelo2(p_m, p_t, p_t_temp, chaff, k);
          if (k == 5099) {
            for (int i = 0; i < 3; ++i) {
              p_m_last[i] = p_m[i];
            }
          }
          if (k >= 5100) {
            for (int i = 0; i < 3; ++i) {
              p_m[i] = p_m_last[i];
            }
          }
        } else if (vuelo == 3) {
          Vuelo3(p_t, v_t, k);
        }

        // - - - - - - - - - -   Imprimir state - - - - - - - - - - //
        //Serial.print("p_m: ");
        for (int i = 0; i < 3; i++) {
          Serial.print(p_m[i], 0);           //Serial.print(p_m[i], 3);  // la ,3 indica los decimales
          Serial.print(" ");
        }
        //Serial.print("p_t: ");
        for (int i = 0; i < 3; i++) {
          Serial.print(p_t[i], 0);             
          Serial.print(" ");
        }
        Serial.println();
        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //

        /* Chequeo de intercepción con distancia actualizada */
        float dN = p_t[0] - p_m[0];
        float dE = p_t[1] - p_m[1];
        float dD = p_t[2] - p_m[2];
        float p_rel_mod = sqrt(dN * dN + dE * dE + dD * dD);
        if (p_rel_mod < blast_radius) {
          float dvN = v_t[0] - v_m[0];
          float dvE = v_t[1] - v_m[1];
          float dvD = v_t[2] - v_m[2];
          float vel_impacto = sqrt(dvN * dvN + dvE * dvE + dvD * dvD);

          Serial.print("FIN");
          Serial.print("  *** ¡Impacto! *** a ");
          Serial.print(p_rel_mod, 1);  // 1 decimal
          Serial.println(" m");

          Serial.print("  Tiempo de vuelo ");
          Serial.print(k, 3);  // 3 decimales
          Serial.print(" s, alcance relativo ");
          Serial.print(vel_impacto * 3.6, 1);  // km/h con 1 decimal
          Serial.println(" km/h");
          break;
        }
      }
      Serial.println("FIN");

      Serial.println("States transmitidos, simulación de Arduino finalizada.\n");
      //delay(500);
      Serial.flush();
      mssg = '\0';
    }
    
    if (mssg == 'E') {
      sim_started = false;
      Serial.flush();
      mssg = '\0';
    }
}