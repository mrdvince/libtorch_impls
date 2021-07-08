#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <fstream>
#include <string>

class CIFAR10 : public torch::data::datasets::Dataset<CIFAR10> {
   public:
    enum Mode { kTrain,
                kTest };
    // load cifar10 dataset from path
    explicit CIFAR10(const std::string &root, Mode mode = Mode::kTrain);
    // get item
    torch::data::Example<> get(size_t index) override;
    // if training -> true
    bool is_train() const noexcept;
    // all images stacked into a tensor
    const torch::Tensor &images;
    // targets stacked into tensor
    const torch::Tensor &targets;

   private:
    torch::Tensor images_;
    torch::Tensor targets_;
    Mode mode_;
};