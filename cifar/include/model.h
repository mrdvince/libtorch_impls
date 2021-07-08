#pragma once
#include <torch/torch.h>

using namespace torch;
class ModelImpl : public torch::nn::Module {
   public:
    ModelImpl();
    torch::Tensor foward(torch::Tensor x);

   private:
    nn::Conv2d conv1;
    nn::MaxPool2d pool;
    nn::Conv2d conv2;
    nn::Linear fc1;
    nn::Linear fc2;
    nn::Linear fc3;
};
TORCH_MODULE(Model);