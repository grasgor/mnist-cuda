#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include "forward_functions.h" // your matmul, relu, softmax, etc.

using namespace std;

const int input_dim = 784;
const int hidden_dim = 256;
const int num_classes = 10;
const int batch_dim = 1;

void print_mnist_image(const torch::Tensor& img) {
    for (int r = 0; r < 28; ++r) {
        for (int c = 0; c < 28; ++c) {
            float pixel = img[r][c].item<float>();
            if (pixel > 0.7) cout << "@";
            else if (pixel > 0.4) cout << "*";
            else if (pixel > 0.1) cout << ".";
            else cout << " ";
        }
        cout << "\n";
    }
}

void forward(float* images, float* W1, float* b1, float* W2, float* b2, 
                float *Z1, float* A1, float *Z2, float* logits,
                const int batch_dim, const int input_dim, const int hidden_dim, const int num_classes){

    matmul(images, W1, Z1, batch_dim, 1, input_dim, hidden_dim);

    for (int b = 0; b < batch_dim; b++)
        for (int h = 0; h < hidden_dim; h++)
            Z1[b * hidden_dim + h] += b1[h];

    relu(Z1, A1, batch_dim, 1, hidden_dim);
    matmul(A1, W2, Z2, batch_dim, 1, hidden_dim, num_classes);

    for (int b = 0; b < batch_dim; b++)
        for (int k = 0; k < num_classes; k++)
            Z2[b * num_classes + k] += b2[k];

    softmax(Z2, logits, batch_dim, 1, num_classes);
}

void load_weights(const char* filename,
                  float* W1, float* b1,
                  float* W2, float* b2,
                  int size_W1, int size_b1,
                  int size_W2, int size_b2) {
    ifstream ifs(filename, ios::binary);
    if (!ifs) { cerr << "Cannot open " << filename << endl; exit(1); }
    ifs.read(reinterpret_cast<char*>(W1), size_W1 * sizeof(float));
    ifs.read(reinterpret_cast<char*>(b1), size_b1 * sizeof(float));
    ifs.read(reinterpret_cast<char*>(W2), size_W2 * sizeof(float));
    ifs.read(reinterpret_cast<char*>(b2), size_b2 * sizeof(float));
    ifs.close();
}

int main() {
    torch::manual_seed(0);

    auto dataset = torch::data::datasets::MNIST("./data", torch::data::datasets::MNIST::Mode::kTest)
                       .map(torch::data::transforms::Normalize<>(0.0, 1.0))
                       .map(torch::data::transforms::Stack<>());

    auto data_loader = torch::data::make_data_loader(dataset, torch::data::DataLoaderOptions().batch_size(1));

    // Allocate model parameters
    float* W1 = new float[input_dim * hidden_dim];
    float* b1 = new float[hidden_dim];
    float* W2 = new float[hidden_dim * num_classes];
    float* b2 = new float[num_classes];
    float* Z1 = new float[batch_dim * hidden_dim];
    float* A1 = new float[batch_dim * hidden_dim];
    float* Z2 = new float[batch_dim * num_classes];
    float* logits = new float[batch_dim * num_classes];

    load_weights("model_weights.bin", W1, b1, W2, b2,
                 input_dim * hidden_dim, hidden_dim,
                 hidden_dim * num_classes, num_classes);

    cout << "Running inference on first 5 test images:\n";

    int count = 0;
    for (auto it = data_loader->begin(); it != data_loader->end() && count < 5; ++it) {
        auto batch = *it;
        auto img = batch.data[0].view({28, 28});
        auto label = batch.target[0].item<int64_t>();

        cout << "\nSample " << count + 1 << " — True Label: " << label << "\n";
        print_mnist_image(img);

        // forward pass
        auto img_flat = batch.data.view({1, input_dim}).contiguous();
        float* images = img_flat.data_ptr<float>();

        forward(images, W1, b1, W2, b2, Z1, A1, Z2, logits,
                batch_dim, input_dim, hidden_dim, num_classes);

        // predicted label
        int pred = 0;
        float max_val = logits[0];
        for (int c = 1; c < num_classes; ++c) {
            if (logits[c] > max_val) { max_val = logits[c]; pred = c; }
        }
        cout << "Predicted Label: " << pred << "\n";

        count++;
    }

    // cleanup
    delete[] W1; delete[] b1; delete[] W2; delete[] b2;
    delete[] Z1; delete[] A1; delete[] Z2; delete[] logits;

    return 0;
}
