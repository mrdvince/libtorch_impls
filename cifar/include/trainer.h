#pragma once
#include <torch/torch.h>

template <typename Dataloader>
void train(torch::nn::Module model, Dataloader &trainloader);
