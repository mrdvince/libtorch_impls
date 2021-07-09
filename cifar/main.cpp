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
    std::cout << model;
    for (size_t epoch = 0; epoch < 10; ++epoch) {
        float running_loss = 0.0;
        int i = 0;
        for (auto& batch : *trainloader) {
            auto inputs = batch.data.to(device);
            auto labels = batch.target.to(device);
            // zero gradients
            optimizer.zero_grad();
            // foward pass, backprop
            auto outputs = model(inputs);
            auto loss = torch::cross_entropy_loss(outputs, labels);
            loss.backward();
            optimizer.step();
            // running stats
            running_loss += loss.item<float>();
            // if (i % 10000 == 0) {  // print every 2000 mini-batches
            //     std::cout << "[" << epoch + 1 << ", " << i + 1 << "] loss: "
            //               << running_loss / 2000 << '\n';
            //     running_loss = 0.0;
            // }
            i++;
        }
        std::cout << "[ " << epoch + 1 << " ] "<< "loss: "<< running_loss / (1999*6) << '\n';
    }

    std::cout << "Finished training\n";
    std::string MODEL_PATH = "./cifar.pth";
    torch::save(model, MODEL_PATH);

    // load and test model on test set
    model = Model();
    torch::load(model, MODEL_PATH);

    int correct = 0;
    int total = 0;
    for (const auto& batch : *testloader) {
        auto images = batch.data.to(torch::kCPU);
        auto labels = batch.target.to(torch::kCPU);

        auto outputs = model->forward(images);

        auto out_tuple = torch::max(outputs, 1);
        auto predicted = std::get<1>(out_tuple);
        total += labels.size(0);
        correct += (predicted == labels).sum().item<int>();
    }

    std::cout << "Overall accuracy: "
              << (100 * correct / total) << "%\n\n";

    float class_correct[10];
    float class_total[10];

    torch::NoGradGuard no_grad;

    for (const auto& batch : *testloader) {
        auto images = batch.data.to(torch::kCPU);
        auto labels = batch.target.to(torch::kCPU);

        auto outputs = model->forward(images);

        auto out_tuple = torch::max(outputs, 1);
        auto predicted = std::get<1>(out_tuple);
        auto c = (predicted == labels).squeeze();

        for (int i = 0; i < 4; ++i) {
            auto label = labels[i].item<int>();
            class_correct[label] += c[i].item<float>();
            class_total[label] += 1;
        }
    }

    for (int i = 0; i < 10; ++i)
        std::cout << "Accuracy of " << classes[i] << " "
                  << 100 * class_correct[i] / class_total[i] << "%\n";
}