#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <sys/resource.h>
#include <torch/torch.h>
#include <torch/data/dataloader.h>

#include "forward_functions.h"
#include "backward_functions.h"

using namespace std;

const int num_epochs = 10;
const float lr = 0.01f;
const int batch_dim = 16;
const int num_classes = 10;
const int input_dim = 784;
const int hidden_dim = 256;
const float train_split_ratio = 0.7f;

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


void update_param(float* param, float *grad, float lr, int size){
    for(int i = 0; i<size; i++){
        param[i] = param[i] - lr * grad[i];
    }
}

void gradient_descent(float* W1, float* b1, float* W2, float* b2, 
                        float* dW1, float* db1, float* dW2, float* db2, float lr,
                        const int input_dim, const int hidden_dim, const int num_classes){
    update_param(W1, dW1, lr, input_dim*hidden_dim);
    update_param(W2, dW2, lr, hidden_dim*num_classes);
    update_param(b1, db1, lr, hidden_dim);
    update_param(b2, db2, lr, num_classes);
}


void backward(float* images, float* logits, const int64_t* targets, float* one_hot_targets,
    float* Z1, float* A1, const int batch_dim, const int input_dim,
    const int hidden_dim, const int num_classes, float* b1, float* b2,
    float* W1, float* W2, float* dW2, float* db2, float* dW1, float* db1, float batch_loss){

    one_hot_encode(targets, one_hot_targets, batch_dim, num_classes);
    calculate_grad(images, logits, one_hot_targets, Z1, A1, batch_dim, 
                    input_dim, hidden_dim, num_classes, W1, W2,
                    dW2, db2, dW1, db1, batch_loss);

    gradient_descent(W1, b1, W2, b2, dW1, db1, dW2, db2, lr, input_dim, hidden_dim, num_classes);
}


template <typename BatchType>
void train_step(BatchType &batch,
                float* W1, float* b1, float* W2, float* b2,
                float* Z1, float* A1, float* Z2, float* logits,
                float* one_hot_targets,
                float* dW1, float* db1, float* dW2, float* db2,
                int batch_dim, int input_dim,
                int hidden_dim, int num_classes,
                float batch_loss){
    auto torch_images = batch.data.view({batch_dim, 1, input_dim}).contiguous();
    auto torch_targets = batch.target.to(torch::kCPU).contiguous();

    float* images = torch_images.template data_ptr<float>();
    const int64_t* targets = torch_targets.template data_ptr<int64_t>();

    forward(images, W1, b1, W2, b2, Z1, A1, Z2, logits,
                batch_dim, input_dim, hidden_dim, num_classes);           
    
    backward(images, logits, targets, one_hot_targets, Z1, A1, batch_dim, 
                    input_dim, hidden_dim, num_classes, b1, b2, W1, W2,
                    dW2, db2, dW1, db1, batch_loss);
 }

float compute_accuracy(const float* logits, const int64_t* targets,
                       int batch_dim, int num_classes) {
    int correct = 0;
    for (int b = 0; b < batch_dim; ++b) {
        int pred = 0;
        float max_val = logits[b * num_classes];
        for (int c = 1; c < num_classes; ++c) {
            float val = logits[b * num_classes + c];
            if (val > max_val) {
                max_val = val;
                pred = c;
            }
        }
        if (pred == targets[b]) correct++;
    }
    return static_cast<float>(correct) / batch_dim;
}

//tqdm-like bar
void show_progress(int current, int total) {
    int bar_width = 30;
    float progress = float(current) / total;
    cout << "\r[";
    int pos = bar_width * progress;
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) cout << "=";
        else if (i == pos) cout << ">";
        else cout << " ";
    }

    cout << "] " << setw(3) << int(progress * 100.0) << "%";
    flush(cout);
}


// save weights to binary
void save_weights(const char* filename,
                  const float* W1, const float* b1,
                  const float* W2, const float* b2,
                  int size_W1, int size_b1, int size_W2, int size_b2) {
    ofstream ofs(filename, ios::binary);
    ofs.write(reinterpret_cast<const char*>(W1), size_W1 * sizeof(float));
    ofs.write(reinterpret_cast<const char*>(b1), size_b1 * sizeof(float));
    ofs.write(reinterpret_cast<const char*>(W2), size_W2 * sizeof(float));
    ofs.write(reinterpret_cast<const char*>(b2), size_b2 * sizeof(float));
    ofs.close();
}

int main() {
    // Load MNIST dataset (normalized and stacked)
    auto dataset = torch::data::datasets::MNIST("./data")
        .map(torch::data::transforms::Normalize<>(0.0, 1.0))
        .map(torch::data::transforms::Stack<>());

    // DataLoader
    auto data_loader = torch::data::make_data_loader(
        dataset,
        torch::data::DataLoaderOptions().batch_size(batch_dim).workers(2)
    );

    // Allocate model parameters and buffers
    float* W1 = new float[input_dim * hidden_dim];
    float* b1 = new float[hidden_dim];
    float* W2 = new float[hidden_dim * num_classes];
    float* b2 = new float[num_classes];
    float* Z1 = new float[batch_dim * hidden_dim];
    float* A1 = new float[batch_dim * hidden_dim];
    float* Z2 = new float[batch_dim * num_classes];
    float* logits = new float[batch_dim * num_classes];

    float* dW1 = new float[input_dim * hidden_dim];
    float* db1 = new float[hidden_dim];
    float* dW2 = new float[hidden_dim * num_classes];
    float* db2 = new float[num_classes];

    float* one_hot_targets = new float[batch_dim * num_classes];
    float batch_loss = 0.0f;

    // Initialize weights and biases
    xavier_weight_init(W1, input_dim, hidden_dim);
    xavier_weight_init(W2, hidden_dim, num_classes);
    memset(b1, 0, hidden_dim * sizeof(float));
    memset(b2, 0, num_classes * sizeof(float));

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        int batch_idx = 0;
        float total_acc = 0.0f;
        int total_batches = 0;

        auto start_time = chrono::high_resolution_clock::now();

        for (auto& batch : *data_loader) {
            train_step(batch, W1, b1, W2, b2, Z1, A1, Z2, logits,
                       one_hot_targets, dW1, db1, dW2, db2,
                       batch_dim, input_dim, hidden_dim, num_classes, batch_loss);

            // compute accuracy
            auto targets = batch.target;
            auto targets_cpu = targets.to(torch::kCPU);
            auto target_ptr = targets_cpu.data_ptr<int64_t>();

            float acc = compute_accuracy(logits, target_ptr, batch_dim, num_classes);
            total_acc += acc;
            total_batches++;
            
            batch_idx++;
            
            show_progress(batch_idx, dataset.size().value() / batch_dim);
        }

        auto end_time = chrono::high_resolution_clock::now();
        double duration = chrono::duration<double>(end_time - start_time).count();

        cout << " | Epoch " << epoch + 1 << "/" << num_epochs
                  << " | Accuracy: " << (total_acc / total_batches) * 100.0f << "%"
                  << " | Time: " << fixed << setprecision(2) << duration << "s";    
        }

    // Save trained weights
    save_weights("model_weights.bin", W1, b1, W2, b2,
                 input_dim * hidden_dim, hidden_dim,
                 hidden_dim * num_classes, num_classes);

    // Cleanup
    delete[] W1;
    delete[] b1;
    delete[] W2;
    delete[] b2;
    delete[] Z1;
    delete[] A1;
    delete[] Z2;
    delete[] logits;
    delete[] one_hot_targets;
    delete[] dW1;
    delete[] db1;
    delete[] dW2;
    delete[] db2;

    return 0;
}
