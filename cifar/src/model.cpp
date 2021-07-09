#include "model.h"
using namespace torch;

ModelImpl::ModelImpl() : conv1(nn::Conv2dOptions(3, 6, 5)),
                         pool(nn::MaxPool2dOptions(2)),
                         conv2(nn::Conv2dOptions(6, 16, 5)),
                         fc1(nn::LinearOptions(16 * 5 * 5, 120)),
                         fc2(nn::LinearOptions(120, 84)),
                         fc3(nn::LinearOptions(84, 10)) {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
}


torch::Tensor ModelImpl::forward(torch::Tensor x) {
    auto out = conv1->forward(x); 
    out = torch::relu(out); 
    out = pool->forward(out);
    out = conv2->forward(out);
    out= torch::relu(out);
    out = pool->forward(out);
    out = out.view({-1, 16 * 5 * 5});
    out = fc1->forward(out);
    out = torch::relu(out);
    out = fc2->forward(out);
    out = torch::relu(out);
    return fc3->forward(out); // apply softmax later
}