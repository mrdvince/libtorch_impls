#include <iostream>

#include "cifar10.h"
#include "model.h"
#include "trainer.h"

using namespace torch;

int main() {
    std::cout << "Let's train this bad boy" << std::endl;
    // load and normalize data
    const std::string DATA_DIR = "../data/cifar10";
    auto train_dataset = CIFAR10(DATA_DIR)
                             .map(data::transforms::Normalize<>({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
                             .map(data::transforms::Stack<>());
    auto trainloader = data::make_data_loader<data::samplers::RandomSampler>(
        std::move(train_dataset), 4);

    auto test_dataset = CIFAR10(DATA_DIR, CIFAR10::Mode::kTest)
                            .map(torch::data::transforms::Normalize<>({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
                            .map(torch::data::transforms::Stack<>());
    auto testloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), 4);

    std::string classes[10] = {"plane", "car", "bird", "cat",
                               "deer", "dog", "frog", "horse", "ship", "truck"};

    //model
    Model model = Model();
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available())
        device = torch::kCUDA;
    model->to(device);

    // loss func and optimizer
    nn::CrossEntropyLoss criterion;
    optim::Adam optimizer(model->parameters(), optim::AdamOptions(0.001));
   
}