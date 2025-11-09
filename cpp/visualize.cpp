#include <torch/torch.h>
#include <iostream>

// Print 28x28 image in ASCII
void print_mnist_image(const torch::Tensor& img) {
    for (int r = 0; r < 28; ++r) {
        for (int c = 0; c < 28; ++c) {
            float pixel = img[r][c].item<float>();
            if (pixel > 0.7) std::cout << "@";
            else if (pixel > 0.4) std::cout << "*";
            else if (pixel > 0.1) std::cout << ".";
            else std::cout << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    // Load MNIST dataset (training)
    auto dataset = torch::data::datasets::MNIST("./data")
                       .map(torch::data::transforms::Normalize<>(0.0, 1.0))
                       .map(torch::data::transforms::Stack<>());

    // DataLoader for iteration
    auto data_loader = torch::data::make_data_loader(
        dataset, /*batch_size=*/1);

    std::cout << "Visualizing first 5 MNIST images:\n";

    int count = 0;
    for (auto& batch : *data_loader) {
        auto img = batch.data.view({28, 28});
        auto label = batch.target.item<int64_t>();

        std::cout << "Label: " << static_cast<int>(label) << "\n";
        print_mnist_image(img);
        std::cout << "---------------------------------\n";

        if (++count >= 5) break; // show only first 5 images
    }

    return 0;
}
