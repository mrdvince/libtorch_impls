#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <fstream>
#include <string>

class CIFAR10 : public torch::data::datasets::Dataset<CIFAR10> {
   public:
    enum Mode { kTrain,
                kTest };

    // load cifar1 dataset from path
    explicit CIFAR10(const std::string &root, Mode mode = Mode::kTrain);

    // get item
    torch::data::Example<> get(size_t index) override;

    // size of dataset
    torch::optional<size_t> size() const override;

    // if training -> true
    bool is_train() const noexcept;

    // all images stacked into a tensor
    torch::Tensor &images();

    // targets stacked into tensor
    torch::Tensor &targets();

   private:
    torch::Tensor images_;
    torch::Tensor targets_;
    Mode mode_;
};