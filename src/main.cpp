#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <cmath>
#include <iterator>
#include <iomanip>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/AutoDiff>
#include <filesystem>
#include <cerrno>
#include <cstring>
#include "Layers.h"
#include "ConvLayer.h"
#include "BatchNorm.h"
#include "Activation.h"
#include "DenseBlock.h"
#include "Pooling.h"
#include "LinearLayer.h"

using namespace Eigen;
using namespace std;

class Cifar10data {
    private:
        std::vector<std::string> filenames;
        std::vector<std::pair<MatrixXf, MatrixXf>> dataset;
    public:
        Cifar10data(std::vector<std::string> &filenames) {
            this->filenames = filenames;
            read_dataset();
        }

        void read_dataset() {
            for (const auto& filename : filenames) {
                std::ifstream file(filename, std::ios::binary);
                if (!file.is_open()) {
                    std::cerr << "Could not open file: " << filename << std::endl;
                    return;
                }

                for (int i = 0; i < 10000; i++) {
                    std::vector<unsigned char> bytes(3073);
                    file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());

                    MatrixXf image(1, 3072);
                    for (int j = 0; j < 3072; j++) {
                        image(0, j) = static_cast<float>(bytes[j + 1]) / 255.0;
                    }

                    MatrixXf label = MatrixXf::Zero(1, 10);
                    label(0, bytes[0]) = 1.0;

                    dataset.push_back({image, label});
                }
                

                file.close();
            }
        }

        std::vector<std::pair<MatrixXf, MatrixXf>> get_dataset() {
            return dataset;
        }
};

class MnistData {
    private:
        std::string images;
        std::string labels;
        std::vector<std::pair<MatrixXf, MatrixXf>> dataset;
    public:
            MnistData(const std::string &images, const std::string &labels) {
                this->images = images;
                this->labels = labels;
                read_dataset();
            }

        int read_int(std::ifstream& file) {
            unsigned char buffer[4];
            file.read(reinterpret_cast<char*>(buffer), 4);
            return (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];
        }

        void read_dataset() {
            std::ifstream image_file(images, std::ios::binary);
            std::ifstream label_file(labels, std::ios::binary);
            if (!image_file.is_open() || !label_file.is_open()) {
                std::filesystem::path p_images = std::filesystem::absolute(images);
                std::filesystem::path p_labels = std::filesystem::absolute(labels);
                std::cerr << "Could not open MNIST files.\n"
                          << "  images: " << p_images << " exists=" << std::filesystem::exists(p_images) << "\n"
                          << "  labels: " << p_labels << " exists=" << std::filesystem::exists(p_labels) << "\n"
                          << "  cwd: " << std::filesystem::current_path() << "\n"
                          << "  std::strerror(errno): " << std::strerror(errno) << "\n";
                return;
            }

            int magic_number = read_int(image_file);
            if (magic_number != 2051) {
                std::cerr << "Invalid magic number for images file" << std::endl;
                return;
            }
            magic_number = read_int(label_file);
            if (magic_number != 2049) {
                std::cerr << "Invalid magic number for labels file" << std::endl;
                return;
            }

            int num_images = read_int(image_file);
            int num_labels = read_int(label_file);
            if (num_images != num_labels) {
                std::cerr << "Number of images and labels do not match" << std::endl;
                return;
                 
            }

            int rows = read_int(image_file);
            int cols = read_int(image_file);

            std::vector<MatrixXf> images;
            for (int i = 0; i < num_images; i++) {
                MatrixXf image(1, rows*cols);
                for (int r = 0; r < rows*cols; r++) {
                    unsigned char pixel;
                    image_file.read(reinterpret_cast<char*>(&pixel), 1);
                    image(0, r) = static_cast<float>(pixel) / 255.0f;
                }
                unsigned char label;
                label_file.read(reinterpret_cast<char*>(&label), 1);
                MatrixXf target = MatrixXf::Zero(1, 10);
                target(0, label) = 1.0f;
                dataset.push_back({image, target});
            }
        }

        std::vector<std::pair<MatrixXf, MatrixXf>> get_dataset() {
            return dataset;
        }
};


// Utility function to print 4D tensors for debugging
void printTensor4D(const Tensor<float, 4>& tensor, const std::string& name = "Tensor", 
                   int max_channels = -1, int max_height = -1, int max_width = -1) {
    int batches = tensor.dimension(0);
    int channels = tensor.dimension(1);
    int height = tensor.dimension(2);
    int width = tensor.dimension(3);
    
    // Limit what we print if tensor is too large
    int print_channels = (max_channels > 0 && max_channels < channels) ? max_channels : channels;
    int print_height = (max_height > 0 && max_height < height) ? max_height : height;
    int print_width = (max_width > 0 && max_width < width) ? max_width : width;
    
    std::cout << "\n========== " << name << " ==========\n";
    std::cout << "Shape: [" << batches << ", " << channels << ", " << height << ", " << width << "]\n";
    
    for (int b = 0; b < batches; b++) {
        std::cout << "\n--- Batch " << b << " ---\n";
        for (int c = 0; c < print_channels; c++) {
            std::cout << "  Channel " << c << ":\n";
            for (int h = 0; h < print_height; h++) {
                std::cout << "    ";
                for (int w = 0; w < print_width; w++) {
                    std::cout << std::setw(8) << std::setprecision(4) << tensor(b, c, h, w) << " ";
                }
                if (print_width < width) {
                    std::cout << "... (" << (width - print_width) << " more)";
                }
                std::cout << "\n";
            }
            if (print_height < height) {
                std::cout << "    ... (" << (height - print_height) << " more rows)\n";
            }
        }
        if (print_channels < channels) {
            std::cout << "  ... (" << (channels - print_channels) << " more channels)\n";
        }
    }
    std::cout << "====================================\n\n";
}

// Convert dataset (MatrixXf pairs) into vectors of 4D tensors (NCHW)
void convert_dataset_to_tensors(const std::vector<std::pair<MatrixXf, MatrixXf>>& dataset,
                                const std::string& kind,
                                std::vector<Tensor<float,4>>& inputs,
                                std::vector<Tensor<float,4>>& targets) {
    inputs.clear(); targets.clear();
    inputs.reserve(dataset.size()); targets.reserve(dataset.size());

    if (kind == "mnist") {
        const int H = 28, W = 28, C = 1;
        for (const auto &p : dataset) {
            const MatrixXf &img = p.first; // 1 x (H*W)
            Tensor<float,4> t(1, C, H, W);
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j)
                    t(0,0,i,j) = img(0, i*W + j);
            inputs.push_back(std::move(t));

            Tensor<float,4> lab(1, 10, 1, 1);
            lab.setZero();
            int label_col; p.second.row(0).maxCoeff(&label_col);
            lab(0, label_col, 0, 0) = 1.0f;
            targets.push_back(std::move(lab));
        }
    } else if (kind == "cifar10") {
        const int H = 32, W = 32, C = 3;
        for (const auto &p : dataset) {
            const MatrixXf &img = p.first; // 1 x 3072
            Tensor<float,4> t(1, C, H, W);
            for (int c = 0; c < C; ++c)
                for (int i = 0; i < H; ++i)
                    for (int j = 0; j < W; ++j)
                        t(0,c,i,j) = img(0, c*H*W + i*W + j);
            inputs.push_back(std::move(t));

            Tensor<float,4> lab(1, 10, 1, 1);
            lab.setZero();
            int label_col; p.second.row(0).maxCoeff(&label_col);
            lab(0, label_col, 0, 0) = 1.0f;
            targets.push_back(std::move(lab));
        }
    }
}

int main(int argc, char** argv) {
    Tensor<float, 4> tensor(32, 3, 32, 32);
    tensor.setRandom();
    /*
    std::vector<std::unique_ptr<Layer>> dense_net;
    dense_net.push_back(make_unique<BatchNorm>(3));
    dense_net.push_back(make_unique<ActivationFunction>(Activation::relu));
    dense_net.push_back(make_unique<ConvLayer>(3, 64, 7, 2, 1));
    dense_net.push_back(make_unique<PoolingLayer>(PoolingLayer::MAX, 3, 2));
    dense_net.push_back(make_unique<DenseBlock>(64, 32, 6));
    dense_net.push_back(make_unique<ConvLayer>(256, int(256*0.5), 1, 1, 0));
    dense_net.push_back(make_unique<BatchNorm>(128));
    dense_net.push_back(make_unique<ActivationFunction>(Activation::relu));
    dense_net.push_back(make_unique<PoolingLayer>(PoolingLayer::MAX, 2, 2));
    dense_net.push_back(make_unique<DenseBlock>(128, 32, 12));
    dense_net.push_back(make_unique<ConvLayer>(512, int(512*0.5), 1, 1, 0));
    dense_net.push_back(make_unique<BatchNorm>(256));
    dense_net.push_back(make_unique<ActivationFunction>(Activation::relu));
    dense_net.push_back(make_unique<PoolingLayer>(PoolingLayer::AVERAGE, 2, 2));
    dense_net.push_back(make_unique<DenseBlock>(256, 32, 24));
    dense_net.push_back(make_unique<ConvLayer>(1024, int(1024*0.5), 1, 1, 0));
    dense_net.push_back(make_unique<BatchNorm>(512));
    dense_net.push_back(make_unique<ActivationFunction>(Activation::relu));
    dense_net.push_back(make_unique<PoolingLayer>(PoolingLayer::AVERAGE, 2, 2));
    dense_net.push_back(make_unique<DenseBlock>(512, 32, 16));
    dense_net.push_back(make_unique<PoolingLayer>(PoolingLayer::AVERAGE, 7, 1, PoolingLayer::GLOBAL));
    dense_net.push_back(make_unique<LinearLayer>(1024, 10));
    dense_net.push_back(make_unique<ActivationFunction>(Activation::softmax));


    Tensor<float, 4> output;

    for (int j = 0; j<100; j++) {
        // Reset output to input tensor at the start of each iteration
        output = tensor;

        for (size_t i = 0; i < dense_net.size(); ++i) {
            output = dense_net[i]->forward(output);
        }
    
        printTensor4D(output, "Final output of densenet", 1000, 5, 5);
    
        // Create target one-hot with same batch size as final softmax output
        int B_out = output.dimension(0);
        Tensor<float, 4> testResults(B_out, 10, 1, 1);
        testResults.setZero();
        int correctClass = 3; // hypothetical correct class index for sample 0
        if (B_out > 0) testResults(0, correctClass, 0, 0) = 1.0f;

        // Cross-entropy loss: average over batch
        const float eps = 1e-6f;
        float loss = 0.0f;
        for (int b = 0; b < B_out; ++b) {
            for (int c = 0; c < 10; ++c) {
                float p = output(b, c, 0, 0);
                float t = testResults(b, c, 0, 0);
                if (t > 0.0f) {
                    loss -= std::log(std::max(p, eps));
                }
            }
        }
        if (B_out > 0) loss /= static_cast<float>(B_out);
        std::cout << "Cross-entropy loss (batch): " << loss << "\n";

        // Gradient at logits for softmax + cross-entropy: grad = p - t (per sample)
        Tensor<float, 4> grad(B_out, 10, 1, 1);
        for (int b = 0; b < B_out; ++b) {
            for (int c = 0; c < 10; ++c) {
                grad(b, c, 0, 0) = output(b, c, 0, 0) - testResults(b, c, 0, 0);
            }
        }
    
        // Set a learning rate for parameter updates during backward
        Layer::learning_rate = 0.1f;
    
        // Manual backward pass through softmax (identity), linear and pooling
        for (int i = static_cast<int>(dense_net.size()) - 1; i >= 0; --i) {
            grad = dense_net[i]->backward(grad);
        }
    
        // Optionally, print gradient w.r.t. input to inspect backprop result
        printTensor4D(grad, "Grad wrt input (after pooling backward)", 3, 7, 7);
        //std::cout << grad << std::endl;
    }*/


    // --- small training entry (use existing loaders and helper above) ---
    // parse simple args
    
    /*
    std::string train_flag = "";
    int n_samples = 100;
    int epochs = 3;
    int batch_size = 1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--train" && i + 1 < argc) train_flag = argv[++i];
        else if (a == "--n" && i + 1 < argc) n_samples = std::stoi(argv[++i]);
        else if (a == "--epochs" && i + 1 < argc) epochs = std::stoi(argv[++i]);
        else if (a == "--batch" && i + 1 < argc) batch_size = std::stoi(argv[++i]);
    }

    if (!train_flag.empty()) {
        std::cout << "Preparing small training run: " << train_flag << " n=" << n_samples << " epochs=" << epochs << " batch=" << batch_size << "\n";
        
        std::vector<std::pair<MatrixXf, MatrixXf>> dataset;
        if (train_flag == "mnist") {
            std::string img = "mnist/train-images.idx3-ubyte";
            std::string lbl = "mnist/train-labels.idx1-ubyte";
            MnistData loader(img, lbl);
            dataset = loader.get_dataset();
        } else if (train_flag == "cifar10") {
            std::vector<std::string> files = {"cifar-10-batches-bin/data_batch_1.bin"};
            Cifar10data loader(files);
            dataset = loader.get_dataset();
        } else {
            std::cerr << "Unknown dataset: " << train_flag << "\n";
            return 1;
        }

        if (n_samples < (int)dataset.size()) dataset.resize(n_samples);

        std::vector<Tensor<float,4>> inputs, targets;
        convert_dataset_to_tensors(dataset, train_flag, inputs, targets);

            std::cout << "Loaded dataset size: " << dataset.size() << " -> inputs=" << inputs.size() << " targets=" << targets.size() << "\n";
            if (!inputs.empty()) {
                // print basic stats for first input
                const auto &t0 = inputs[0];
                float s = 0.0f; float mn = t0(0,0,0,0), mx = mn;
                for (int c = 0; c < t0.dimension(1); ++c)
                    for (int i = 0; i < t0.dimension(2); ++i)
                        for (int j = 0; j < t0.dimension(3); ++j) {
                            float v = t0(0,c,i,j);
                            s += v;
                            mn = std::min(mn, v);
                            mx = std::max(mx, v);
                        }
                std::cout << "first input stats sum=" << s << " min=" << mn << " max=" << mx << "\n";
            }

            

    // build a small, stable model for quick tests (MNIST/CIFAR small subset)
    // Architecture: Conv(3x3, same) -> BatchNorm -> ReLU -> GlobalAvgPool -> Linear -> Softmax
    std::vector<std::unique_ptr<Layer>> model;
    int in_ch = (train_flag == "mnist") ? 1 : 3;
    // small conv: 16 filters, 3x3 kernel, stride 1, padding 1 keeps spatial dims
    model.push_back(make_unique<ConvLayer>(in_ch, 64, 7, 2, 1));
    model.push_back(make_unique<BatchNorm>(64));
    model.push_back(make_unique<ActivationFunction>(Activation::relu));
    model.push_back(make_unique<PoolingLayer>(PoolingLayer::MAX, 3, 2));
    model.push_back(make_unique<DenseBlock>(64, 32, 6));
    model.push_back(make_unique<ConvLayer>(256, int(256*0.5), 1, 1, 0));
    model.push_back(make_unique<BatchNorm>(128));
    model.push_back(make_unique<ActivationFunction>(Activation::relu));
    model.push_back(make_unique<PoolingLayer>(PoolingLayer::MAX, 2, 2));
    model.push_back(make_unique<DenseBlock>(128, 32, 12));
    model.push_back(make_unique<ConvLayer>(512, int(512*0.5), 1, 1, 0));
    model.push_back(make_unique<BatchNorm>(256));
    model.push_back(make_unique<ActivationFunction>(Activation::relu));
    model.push_back(make_unique<PoolingLayer>(PoolingLayer::AVERAGE, 2, 2));
    model.push_back(make_unique<DenseBlock>(256, 32, 24));
    model.push_back(make_unique<ConvLayer>(1024, int(1024*0.5), 1, 1, 0));
    model.push_back(make_unique<BatchNorm>(512));
    model.push_back(make_unique<ActivationFunction>(Activation::relu));
    model.push_back(make_unique<PoolingLayer>(PoolingLayer::AVERAGE, 2, 2));
    model.push_back(make_unique<DenseBlock>(512, 32, 16));
    model.push_back(make_unique<PoolingLayer>(PoolingLayer::AVERAGE, 7, 1, PoolingLayer::GLOBAL));
    model.push_back(make_unique<LinearLayer>(1024, 10));
    model.push_back(make_unique<ActivationFunction>(Activation::softmax));


    // reasonable learning rate for this tiny model
    Layer::learning_rate = 0.1f;

        // training loop (simple SGD, per-sample or small batches)
        std::mt19937 rng(std::random_device{}());
        std::vector<int> idx(inputs.size()); std::iota(idx.begin(), idx.end(), 0);

        for (int e = 0; e < epochs; ++e) {
            std::shuffle(idx.begin(), idx.end(), rng);
            double epoch_loss = 0.0;
            for (size_t ii = 0; ii < idx.size(); ii += batch_size) {
                int bs = std::min(batch_size, (int)(idx.size() - ii));
                // process each sample individually (you can vectorize later)
                for (int b = 0; b < bs; ++b) {
                    int id = idx[ii + b];
                    Tensor<float,4> x = inputs[id];
                    Tensor<float,4> out = x;
                    for (size_t li = 0; li < model.size(); ++li) out = model[li]->forward(out);
                    // loss
                    //std::cout << "Logits: ";
                    //std::cout << out << "\n";
                    //std::cout << "\n";
                    int correct = -1;
                    float bestv = -1.0f;
                    for (int c = 0; c < out.dimension(1); ++c) {
                        float v = targets[id](0, c, 0, 0);
                        if (v > bestv) { bestv = v; correct = c; }
                    }
                    if (correct < 0) correct = 0;
                    float p = out(0, correct, 0, 0);
                    float sample_loss = -std::log(std::max(p, 1e-7f));
                    epoch_loss += sample_loss;
                    // debug: print first few sample probabilities in each epoch

                    std::cout << " debug epoch " << e << " sample " << b << " correct=" << correct << " bestv=" << bestv
                                << " out_dim=" << out.dimension(1) << " p=" << p << " loss=" << sample_loss << "\n";
                    std::cout << "  target vector: ";
                    for (int cc = 0; cc < out.dimension(1); ++cc) std::cout << targets[id](0,cc,0,0) << " ";
                    std::cout << "\n";
                    
                    // grad
                    Tensor<float,4> grad = out;
                    for (int c = 0; c < 10; ++c) grad(0,c,0,0) = out(0,c,0,0) - targets[id](0,c,0,0);
                        for (int li = (int)model.size() - 1; li >= 0; --li) {
                            grad = model[li]->backward(grad);
                            // check for NaNs in grad
                            bool found_nan = false;
                            for (int b2 = 0; b2 < grad.dimension(0) && !found_nan; ++b2)
                                for (int c2 = 0; c2 < grad.dimension(1) && !found_nan; ++c2)
                                    for (int h2 = 0; h2 < grad.dimension(2) && !found_nan; ++h2)
                                        for (int w2 = 0; w2 < grad.dimension(3) && !found_nan; ++w2)
                                            if (std::isnan(grad(b2,c2,h2,w2))) { found_nan = true; break; }
                            if (found_nan) {
                                std::cerr << "NaN encountered in gradient after backward of layer " << li
                                          << " type=" << typeid(*model[li]).name() << "\n";
                                // dump a little info and abort this run
                                std::abort();
                            }
                        }
                }
            }
            std::cout << "Epoch " << e << " loss=" << (epoch_loss / inputs.size()) << "\n";

            // print model output for first sample to see if it changes
            if (!inputs.empty()) {
                Tensor<float,4> out0 = inputs[0];
                for (size_t li = 0; li < model.size(); ++li) out0 = model[li]->forward(out0);
                std::cout << " first_sample_output: ";
                for (int c = 0; c < out0.dimension(1); ++c) std::cout << out0(0,c,0,0) << " ";
                std::cout << "\n";
            }
        }
        std::cout << "Done training small set." << std::endl;
    }
    */


    int batches = 2;
    int in_channels = 3;
    int out_channels = 8;
    int in_h = 10;
    int in_w = 10;
    int kernel_h = 3;
    int kernel_w = 3;

    const int total = out_channels * in_channels * kernel_h * kernel_w;
    Tensor<float, 4> weights(out_channels, in_channels, kernel_h, kernel_w);
    int idx = 0;
    for (int o = 0; o < out_channels; o++) {
        for (int c = 0; c <in_channels; c++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    float v = (idx / float(total - 1)) - 0.5f; // in [-0.5, 0.5]
                    weights(o, c, kh, kw) = v;
                    idx++;
                }
            }
        }
    }


    Tensor<float, 4> input(batches, in_channels, in_h, in_w);
    for (int n = batches; n < 2; ++n) {
        for (int h = 0; h < in_h; ++h) {
            for (int w = 0; w < in_w; ++w) {
                input(n, 0, h, w) = 5.0f;
                input(n, 1, h, w) = static_cast<float>(n * 4 + h * 2 + w);
                input(n, 2, h, w) = 1.0f;
            }
        }
    }

    Tensor<float, 1> bias(8);
    bias.setValues({0.4f, 0.2f, 0.5f, 0.5f, 0.2f, 0.9f, 0.7f, 0.5f});

    auto convolution = ConvLayer(3, 8, 3);
    convolution.setBias(bias);
    convolution.setWeights(weights);
    printTensor4D(convolution.getWeights());

    Tensor<float, 4> output = convolution.forward(input);

    printTensor4D(output, "convolution forward output", 8, 10, 10);


    return 0;
}
