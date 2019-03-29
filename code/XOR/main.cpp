#include "models/xor_relu.hpp"  //gernerated model file
#include "tensor.hpp"  //useful tensor classes
#include "mbed.h"
#include <stdio.h>

Serial pc(USBTX, USBRX, 115200);  //baudrate := 115200

int main(void) {
  printf("XOR uTensor cli example (device)\n");

  Context ctx;  //creating the context class, the stage where inferences take place 
  //wrapping the input data in a tensor class

  const float input_data [ 2 ] = { 0.0, 1.0 };

  Tensor* input_x = new WrappedRamTensor<float>({1, 2}, (float*) input_data);

  get_xor_relu_ctx(ctx, input_x);  // pass the tensor to the context
  S_TENSOR pred_tensor = ctx.get("add_1:0");  // getting a reference to the output tensor
  ctx.eval(); //trigger the inference

  float pred_label = *(pred_tensor->read<float>(0, 1));  //getting the result back
  printf("Predicted label: %f\r\n", pred_label);

  return 0;
}
