#include <ATen/ATen.h>
#include <torch/torch.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <tuple>
#include <vector>

// load and convert image to tensor from a location arg
torch::Tensor read_data(std::string location) {
    // read and return img tensor
    cv::Mat img = cv::imread(location, 1);
    // image resize
    cv::resize(img, img, cv::Size(224, 224), cv::INTER_CUBIC);
    std::cout << "Sizes" << img.size() << std::endl;
    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
    img_tensor = img_tensor.permute({2, 0, 1});  // channels x height x width
    return img_tensor.clone();
}

// labels to tensor
torch::Tensor read_label(int label) {
    // read label and convert to tensor
    torch::Tensor label_tensor = torch::full({1}, label);
    return label_tensor.clone();
}

// load images to tensor types
std::vector<torch::Tensor> process_images(std::vector<std::string> list_images) {
    std::cout << "Reading images..." << std::endl;
    std::vector<torch::Tensor> states;
    for (std::vector<std::string>::iterator it = list_images.begin(); it != list_images.end(); ++it) {
        torch::Tensor img = read_data(*it);
        states.push_back(img);
    }
    return states;
}

// load labels to tensor types
std::vector<torch::Tensor> process_labels(std::vector<int> list_labels) {
    printf("Reading labels......");
    std::vector<torch::Tensor> labels;
    for (std::vector<int>::iterator it = list_labels.begin(); it != list_labels.end(); ++it) {
        torch::Tensor label = read_label(*it);
        labels.push_back(label);
    }
    return labels;
}

class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {
   private:
    // 2 vectors of tensors, images and labels
    std::vector<torch::Tensor> images, labels;

   public:
    // constructor
    CustomDataset(std::vector<std::string> list_images, std::vector<int> list_labels) {
        images = process_images(list_images);
        labels = process_labels(list_labels);
    };
    // override get() func to return tensor at loc index
    torch::data::Example<> get(size_t index) override {
        torch::Tensor sample_img = images.at(index);
        torch::Tensor sample_label = labels.at(index);
        return {sample_img.clone(), sample_label.clone()};
    }
    // return length of data
    torch::optional<size_t> size() const override {
        return labels.size();
    };
};

int main(int argc, char** argv) {
    std::vector<std::string> list_images;
    std::vector<int> list_labels;
    auto custom_dataset = CustomDataset(list_images, list_labels).map(torch::data::transforms::Stack<>());
}