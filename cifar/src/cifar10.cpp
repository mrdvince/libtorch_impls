#include "cifar10.h"
namespace {
constexpr uint32_t kTrainSize = 50000;
constexpr uint32_t kTestSize = 10000;
constexpr uint32_t kSizePerBatch = 10000;
constexpr uint32_t kImageRows = 32;
constexpr uint32_t kImageColumns = 32;
// shape -> (3,32,32) for each color channel, R-> 1024, G->1024, B->1024
constexpr uint32_t kBytesPerRow = 3072+1;
// 32 x 32 -> single color channel
constexpr uint32_t kBytesPerChannelPerRow = 1024;
constexpr uint32_t kBytesPerBatchFile = kBytesPerRow * kSizePerBatch;

const std::vector<std::string> kTrainDataBatchFiles = {
    "data_batch_1.bin",
    "data_batch_2.bin",
    "data_batch_3.bin",
    "data_batch_4.bin",
    "data_batch_5.bin",
};

const std::vector<std::string> kTestDataBatchFiles = {
    "test_batch.bin"};

std::string join_paths(std::string head, const std::string &tail) {
    if (head.back() != '/') {
        head.push_back('/');
    };
    head += tail;
    return head;
}

// pair of image tensors and labels
std::pair<torch::Tensor, torch::Tensor> read_data(std::string root, bool train) {
    auto &files = train ? kTrainDataBatchFiles : kTestDataBatchFiles;
    auto num_samples = train ? kTrainSize : kTestSize;
    std::vector<char> data_buffer;
    data_buffer.reserve(files.size() * kBytesPerBatchFile);

    for (auto &file : files) {
        // /path/to/image/ + data_bin_file -> /path/to/data_bin_file = data/cifar10/data_batch_1.bin
        auto path = join_paths(root, file);
        // std::cout << path << std::endl;
        std::ifstream data(path, std::ios::binary);
        TORCH_CHECK(data, "Error opening file at: ", path);
        data_buffer.insert(data_buffer.end(), std::istreambuf_iterator<char>(data), {});
    }
    // size assertion
    std::cout<<data_buffer.size()<<" " << (files.size() * kBytesPerBatchFile)<<std::endl;
    
    TORCH_CHECK(data_buffer.size() == files.size() * kBytesPerBatchFile, "Unexpected file sizes");
    // shape -> sample_size x 3 x 32 x 32
    auto images = torch::empty({num_samples, 3, kImageRows, kImageColumns}, torch::kByte);
    auto targets = torch::empty(num_samples, torch::kByte);

    for (uint32_t i = 0; i != num_samples; ++i) {
        uint32_t start_index = i * kBytesPerRow;
        targets[i] = data_buffer[start_index];
        uint32_t image_start = start_index + 1;
        uint32_t image_end = image_start + 3 * kBytesPerChannelPerRow;  // or image_start + kBytesPerRow
        std::copy(data_buffer.begin() + image_start, data_buffer.begin() + image_end, reinterpret_cast<char *>(images[i].data_ptr()));
    }
    // normalize -> 0,255 -> 0,1
    return {images.to(torch::kFloat32).div_(255), targets.to(torch::kInt64)};
}
}  // namespace

CIFAR10::CIFAR10(const std::string &root, Mode mode) : mode_(mode) {
    auto data = read_data(root, mode == Mode::kTrain);

    images_ = std::move(data.first);
    targets_ = std::move(data.second);
}

torch::data::Example<> CIFAR10::get(size_t index) {
    return {images_[index], targets_[index]};
}

torch::optional<size_t> CIFAR10::size() const {
    return images_.size(0);
}

bool CIFAR10::is_train() const noexcept {
    return mode_ == Mode::kTrain;
}

torch::Tensor &CIFAR10::images() {
    return images_;
}

torch::Tensor &CIFAR10::targets() {
    return targets_;
}