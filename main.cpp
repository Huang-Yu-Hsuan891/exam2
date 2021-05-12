#include "mbed.h"
#include "mbed_rpc.h"
#include "uLCD_4DGL.h"
#include "accelerometer_handler.h"
#include "config.h"

#include "magic_wand_model_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "stm32l475e_iot01_accelero.h"

constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
uLCD_4DGL uLCD(D1, D0, D2);
BufferedSerial pc(USBTX, USBRX);

InterruptIn btnRecord(USER_BUTTON);
//EventQueue queue(32 * EVENTS_EVENT_SIZE);
//Thread t;

EventQueue queue_gesture(32 * EVENTS_EVENT_SIZE);
void gesture_capture(Arguments *in, Reply *out);
RPCFunction gestureui(&gesture_capture, "gesture_capture");
Thread t1_gestureui;

int success_ang;

int PredictGesture(float* output) {
  // How many times the most recent gesture has been matched in a row
  static int continuous_count = 0;
  // The result of the last prediction
  static int last_predict = -1;

  // Find whichever output has a probability > 0.8 (they sum to 1)
  int this_predict = -1;
  for (int i = 0; i < label_num; i++) {
    if (output[i] > 0.8) this_predict = i;
  }

  // No gesture was detected above the threshold
  if (this_predict == -1) {
    continuous_count = 0;
    last_predict = label_num;
    return label_num;
  }

  if (last_predict == this_predict) {
    continuous_count += 1;
  } else {
    continuous_count = 0;
  }
  last_predict = this_predict;

  // If we haven't yet had enough consecutive matches for this gesture,
  // report a negative result
  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
    return label_num;
  }
  // Otherwise, we've seen a positive result, so clear all our variables
  // and report it
  continuous_count = 0;
  last_predict = -1;

  return this_predict;
}

int select_angle;

void print(int gesture_index) {
    uLCD.cls();
    uLCD.background_color(WHITE);
    uLCD.color(BLUE);
    uLCD.text_width(4); //4X size text
    uLCD.text_height(4);
    uLCD.textbackground_color(WHITE);

    if (gesture_index == 0) 
      uLCD.printf("\nring:30\n");
    else if (gesture_index == 1) 
      uLCD.printf("\nslope:45\n");
    else if (gesture_index == 2) 
      uLCD.printf("\nline:60\n");
    else uLCD.printf("\nline\n");
}
void accelerator_data() {
    int i = 0;
    int16_t pDataXYZ[3] = {0};
    int16_t pDataXYZ1[3] = {0};
    int16_t pDataXYZ2[3] = {0};
    int16_t pDataXYZ3[3] = {0};
    int16_t pDataXYZ4[3] = {0};
    int16_t pDataXYZ5[3] = {0};
    int16_t pDataXYZ6[3] = {0};
    int16_t pDataXYZ7[3] = {0};
    int16_t pDataXYZ8[3] = {0};
    int16_t pDataXYZ9[3] = {0};
    int16_t pDataXYZ10[3] = {0};
    float cosangle;
    float long1;
    float long2;
    float cos_select;

    bool should_clear_buffer = false;
    bool got_data = false;
    // The gesture index of the prediction
    int gesture_index;

    // Set up logging.
    static tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    //return -1;
  }
  static tflite::MicroOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                               tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                               tflite::ops::micro::Register_RESHAPE(), 1);

  // Build an interpreter to run the model with
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  tflite::MicroInterpreter* interpreter = &static_interpreter;
  // Allocate memory from the tensor_arena for the model's tensors
  interpreter->AllocateTensors();

  // Obtain pointer to the model's input tensor
  TfLiteTensor* model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != config.seq_length) ||
      (model_input->dims->data[2] != kChannelNumber) ||
      (model_input->type != kTfLiteFloat32)) {
    error_reporter->Report("Bad input tensor parameters in model");
    //return -1;
  }

  int input_length = model_input->bytes / sizeof(float);

  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
  if (setup_status != kTfLiteOk) {
    error_reporter->Report("Set up failed\n");
    //return -1;
  }
  error_reporter->Report("Set up successful...\n");
  BSP_ACCELERO_AccGetXYZ(pDataXYZ);
  printf("initial Accelerometer values: (%d, %d, %d)\r\n", pDataXYZ[0], pDataXYZ[1], pDataXYZ[2]);
  ThisThread::sleep_for(1000ms);
  while (true) {

    // Attempt to read new data from the accelerometer
    got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                 input_length, should_clear_buffer);

    // If there was no new data,
    // don't try to clear the buffer again and wait until next time
    if (!got_data) {
      should_clear_buffer = false;
      continue;
    }
    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on index: %d\n", begin_index);
      continue;
    }
    // Analyze the results to obtain a prediction
    gesture_index = PredictGesture(interpreter->output(0)->data.f);
    // Clear the buffer next time we read data
    should_clear_buffer = gesture_index < label_num;
    // Produce an output
    if (gesture_index < label_num) {
      error_reporter->Report(config.output_message[gesture_index]);
      print(gesture_index);
      /*if (gesture_index == 0) select_angle = 30;
      else if (gesture_index == 1) select_angle = 45;
      else if (gesture_index == 2) select_angle = 60;
      else select_angle = 0;*/
    }
    select_angle = 30;
    cos_select = 0.866;
    //BSP_ACCELERO_AccGetXYZ(pDataXYZ);
    //printf("initial Accelerometer values: (%d, %d, %d)\r\n", pDataXYZ[0], pDataXYZ[1], pDataXYZ[2]);
    //ThisThread::sleep_for(100ms);
    if (gesture_index == 2) {
    BSP_ACCELERO_AccGetXYZ(pDataXYZ1);
    printf("initial Accelerometer values: (%d, %d, %d)\r\n", pDataXYZ1[0], pDataXYZ1[1], pDataXYZ1[2]);
    ThisThread::sleep_for(100ms);
    BSP_ACCELERO_AccGetXYZ(pDataXYZ2);
    printf("initial Accelerometer values: (%d, %d, %d)\r\n", pDataXYZ2[0], pDataXYZ2[1], pDataXYZ2[2]);
    ThisThread::sleep_for(100ms);
    BSP_ACCELERO_AccGetXYZ(pDataXYZ3);
    printf("initial Accelerometer values: (%d, %d, %d)\r\n", pDataXYZ3[0], pDataXYZ3[1], pDataXYZ3[2]);
    ThisThread::sleep_for(100ms);
    BSP_ACCELERO_AccGetXYZ(pDataXYZ4);
    printf("initial Accelerometer values: (%d, %d, %d)\r\n", pDataXYZ4[0], pDataXYZ4[1], pDataXYZ4[2]);
    ThisThread::sleep_for(100ms);
    BSP_ACCELERO_AccGetXYZ(pDataXYZ5);
    printf("initial Accelerometer values: (%d, %d, %d)\r\n", pDataXYZ5[0], pDataXYZ5[1], pDataXYZ5[2]);
    ThisThread::sleep_for(100ms);
    BSP_ACCELERO_AccGetXYZ(pDataXYZ6);
    printf("initial Accelerometer values: (%d, %d, %d)\r\n", pDataXYZ6[0], pDataXYZ6[1], pDataXYZ6[2]);
    ThisThread::sleep_for(100ms);
    BSP_ACCELERO_AccGetXYZ(pDataXYZ7);
    printf("initial Accelerometer values: (%d, %d, %d)\r\n", pDataXYZ7[0], pDataXYZ7[1], pDataXYZ7[2]);
    ThisThread::sleep_for(100ms);
    BSP_ACCELERO_AccGetXYZ(pDataXYZ8);
    printf("initial Accelerometer values: (%d, %d, %d)\r\n", pDataXYZ8[0], pDataXYZ8[1], pDataXYZ8[2]);
    ThisThread::sleep_for(100ms);
    BSP_ACCELERO_AccGetXYZ(pDataXYZ9);
    printf("initial Accelerometer values: (%d, %d, %d)\r\n", pDataXYZ9[0], pDataXYZ9[1], pDataXYZ9[2]);
    ThisThread::sleep_for(100ms);
    BSP_ACCELERO_AccGetXYZ(pDataXYZ10);
    printf("initial Accelerometer values: (%d, %d, %d)\r\n", pDataXYZ10[0], pDataXYZ10[1], pDataXYZ10[2]);
    ThisThread::sleep_for(100ms);

    int j[10];
    long1 = sqrt(pDataXYZ[0] * pDataXYZ[0] + pDataXYZ[1] * pDataXYZ[1] + pDataXYZ[2] * pDataXYZ[2]);
    long2 = sqrt(pDataXYZ1[0] * pDataXYZ1[0] + pDataXYZ1[1] * pDataXYZ1[1] + pDataXYZ1[2] * pDataXYZ1[2]);
    cosangle = (pDataXYZ[0] * pDataXYZ1[0] + pDataXYZ[1] * pDataXYZ1[1] + pDataXYZ[2] * pDataXYZ1[2]) / (long1 * long2);
    if (cosangle < cos_select) {
      success_ang = 1;
      i++;
      //mqtt_queue1.call(&publish_message, &client);
      printf("success_ang = %d\r\n", success_ang);
      j[i] = success_ang;
    } else {
       success_ang = 0; 
       i++;
       printf("success_ang = %d\r\n", success_ang);
       j[i] = success_ang;
    }
    ThisThread::sleep_for(100ms);

    long1 = sqrt(pDataXYZ[0] * pDataXYZ[0] + pDataXYZ[1] * pDataXYZ[1] + pDataXYZ[2] * pDataXYZ[2]);
    long2 = sqrt(pDataXYZ2[0] * pDataXYZ2[0] + pDataXYZ2[1] * pDataXYZ2[1] + pDataXYZ2[2] * pDataXYZ2[2]);
    cosangle = (pDataXYZ[0] * pDataXYZ2[0] + pDataXYZ[1] * pDataXYZ2[1] + pDataXYZ[2] * pDataXYZ2[2]) / (long1 * long2);
    if (cosangle < cos_select) {
      success_ang = 1;
      i++;
      //mqtt_queue1.call(&publish_message, &client);
      printf("success_ang = %d\r\n", success_ang);
       j[i] = success_ang;
    } else {
       success_ang = 0; 
       i++;
       printf("success_ang = %d\r\n", success_ang);
       j[i] = success_ang;
    }
    ThisThread::sleep_for(100ms);
    }
    /*while (i<=10) {
    BSP_ACCELERO_AccGetXYZ(pDataXYZ1);
    printf("detect Accelerometer values: (%d, %d, %d)\r\n", pDataXYZ1[0], pDataXYZ1[1], pDataXYZ1[2]);
    long1 = sqrt(pDataXYZ[0] * pDataXYZ[0] + pDataXYZ[1] * pDataXYZ[1] + pDataXYZ[2] * pDataXYZ[2]);
    long2 = sqrt(pDataXYZ1[0] * pDataXYZ1[0] + pDataXYZ1[1] * pDataXYZ1[1] + pDataXYZ1[2] * pDataXYZ1[2]);
    cosangle = (pDataXYZ[0] * pDataXYZ1[0] + pDataXYZ[1] * pDataXYZ1[1] + pDataXYZ[2] * pDataXYZ1[2]) / (long1 * long2);
    if (cosangle < cos_select) {
      success_ang = 1;
      i++;
      //mqtt_queue1.call(&publish_message, &client);
    }
    else {
      success_ang = 0; 
      i = i;
    }*/
    //printf("success_ang = %d\r\n", success_ang);
    //print1(success_ang);
    //ThisThread::sleep_for(200ms);
    //if (i>10){break;}
  }
  //}
}

int main() {

    BSP_ACCELERO_Init();
    //t1_gestureui.start(callback(&queue3, &EventQueue::dispatch_forever));
    char buf[256], outbuf[256];
    FILE *devin = fdopen(&pc, "r");
    FILE *devout = fdopen(&pc, "w");
    while(1) {
      memset(buf, 0, 256);
      for (int i = 0; ; i++) {
          char recv = fgetc(devin);
          if (recv == '\n') {
              printf("\r\n");
              break;
          }
          buf[i] = fputc(recv, devout);
      }
    RPC::call(buf, outbuf);
    printf("%s\r\n", outbuf);
    }
}
void gesture_capture(Arguments *in, Reply *out) {
    t1_gestureui.start(callback(&queue_gesture, &EventQueue::dispatch_forever));
    queue_gesture.call(accelerator_data);
    //btnRecord.fall(queue_gesture.event(startRecord));
    //btnRecord.rise(queue_gesture.event(stopRecord));
}