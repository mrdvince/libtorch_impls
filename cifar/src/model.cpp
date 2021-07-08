#include "model.h"
using namespace torch;

ModelImpl::ModelImpl() : conv1(nn::Conv2dOptions(3, 6, 5)),
                         pool(nn::MaxPool2dOptions(2)),
                         conv2(nn::Conv2dOptions(5, 16, 5)),
                         fc1(nn::LinearOptions(16 * 5 * 5, 120)),
                         fc2(nn::LinearOptions(120, 84)),
                         fc3(nn::LinearOptions(84, 10)) {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
}

torch::Tensor ModelImpl::foward(Tensor x) {
    auto out = conv1->forward(x);  // 28,28,6
    out = torch::relu(out);
    out = pool->forward(out);  // 14,14,6
    out = conv2->forward(x);   // 10, 10, 16
    out = torch::relu(out);
    out = pool->forward(out);          // 5,5,16
    out = out.view({-1, 16 * 5 * 5});  // flatten
    out = fc1->forward(out);
    out = torch::relu(out);
    out = fc2->forward(out);
    out = torch::relu(out);
    out = fc3->forward(out);
    return out;  //apply softmax later
}