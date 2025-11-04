#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <random>
#include <cmath>
#include <iterator>
#include <iomanip>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/AutoDiff>
#include <chrono>
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

Eigen::MatrixXf EMPTY(0,0);

MatrixXf sigmoid(const MatrixXf& x) {
    return 1.0f / (1.0f + (-x).array().exp());
}

MatrixXf sigmoid_derivative(const MatrixXf& x) {
    MatrixXf s = sigmoid(x);
    return s.array() * (1.0f - s.array());
}

MatrixXf relu_derivative(const MatrixXf& x) {
    return (x.array() > 0.0).cast<float>();
}

/*class Layer {
    public:
        MatrixXf output, input, weights;

        virtual void forward(const MatrixXf& input) = 0;
        virtual void backward(const MatrixXf& grad, const MatrixXf& weight = EMPTY) = 0;
        virtual void update(float learning_rate) = 0;
        virtual ~Layer() = default;
};

class DenseLayer : public Layer {
    private:
        MatrixXf bias;
        MatrixXf dW;
        MatrixXf db;
    public:

        DenseLayer(int input_size, int output_size) {
            weights = MatrixXf::Random(input_size, output_size) * sqrt(2.0/input_size);
            bias = MatrixXf::Zero(1, output_size);
        }

        void forward(const MatrixXf& input) override {
            this->input = input;
            output = relu((input*weights) + bias.replicate(input.rows(), 1));
        }

        void backward(const MatrixXf& grad, const MatrixXf& weight) override {
            MatrixXf error = grad * weight.transpose();
            output = relu_derivative(output).array() * error.array();
            dW = (input.transpose() * output);
            db = output.colwise().sum() / input.rows();
        }

        void update(float learning_rate) override {
            weights -= learning_rate * dW;
            bias -= learning_rate * db;
        }
};

class DenseBlock {
    private:
        std::vector<std::unique_ptr<Layer>> layers;
    public:
        DenseBlock(int num_layers, int input_size, int growth_rate) {
            for (int i = 0; i < num_layers; i++) {
                layers.emplace_back(std::make_unique<DenseLayer>(input_size + i * growth_rate, growth_rate));
            }
        }

        Eigen::MatrixXf forward(const Eigen::MatrixXf& input) {
            Eigen::MatrixXf current_input = input;
            Eigen::MatrixXf concat_output = input;

            for (auto& layer : layers) {
                layer->forward(current_input);
                Eigen::MatrixXf output = layer->output;
                concat_output.conservativeResize(concat_output.rows(), concat_output.cols() + output.cols());
                concat_output.rightCols(output.cols()) = output;
                current_input = concat_output;
            }
            return concat_output;
        }
};

class OutputLayer : public Layer  {
    private:
        MatrixXf bias;
        MatrixXf dW;
        MatrixXf db;
    public:
        OutputLayer(int input_size, int output_size) {
            float limit = sqrt(6.0 / (input_size + output_size));
            weights = MatrixXf::Random(input_size, output_size) * limit;
            bias = MatrixXf::Zero(1, output_size);
        }

        void forward(const MatrixXf& input) override {
            this->input = input;
            output = sigmoid((input*weights) + bias.replicate(input.rows(), 1));
        }

        void backward(const MatrixXf& grad, const MatrixXf& weight) override {
            MatrixXf sigmoid = sigmoid_derivative(output);
            output = grad.array() * sigmoid.array(); 
            dW = (input.transpose() * output);
            db = output.colwise().sum();
        }

        void update(float learning_rate) override {
            weights -= learning_rate * dW;
            bias -= learning_rate * db;
        }
};


class NeuralNetwork {
    private:
        std::vector<std::unique_ptr<Layer>> layers;
    public:

        void add(unique_ptr<Layer> layer) {
            layers.push_back(std::move(layer));
        }

        MatrixXf forward(const MatrixXf& input) {
            MatrixXf out = input;

            for (auto& layer : layers) {
                layer->forward(out);
                out = layer->output;
            }

            return out;
        }

        void backward(const MatrixXf& grad_output) {
            MatrixXf grad = grad_output;
            MatrixXf weight;
            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                (*it)->backward(grad, weight);
                grad = (*it)->output;
                weight = (*it)->weights;
            }
        }

        void update(float learning_rate) {
            for (auto& layer : layers) {
                layer->update(learning_rate);
            }
        }
};

MatrixXf generate_random_matrix(int rows, int cols) {
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_real_distribution<float> dist(-1.0, 1.0);
    MatrixXf mat(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat(i, j) = dist(rng);
        }
    }
    return mat;
}

*/

std::vector<std::pair<MatrixXf, MatrixXf>> generate_dataset(std::vector<MatrixXf>& X) {
    std::vector<std::pair<MatrixXf, MatrixXf>> dataset;

    for (auto& x : X) {
        MatrixXf y = x.array().sin();
        y = (y.array() - y.minCoeff()) / (y.maxCoeff() - y.minCoeff());
        dataset.push_back({x, y});
    }
    return dataset;
}

/*void visualize_layer(const MatrixXf& activations, int layer_num) {
    std::ofstream outfile("layer" + std::to_string(layer_num) + "_activations.txt");

    for (int i = 0; i < activations.rows(); i++) {
        for (int j = 0; j < activations.cols(); j++) {
            outfile << activations(i, j) << " ";
        }
        outfile << "\n";
    }
    outfile.close();

    std::ofstream gpfile("plot_layer" + std::to_string(layer_num) + ".gp");
    gpfile << "set terminal png\n";
    gpfile << "set output 'layer" + std::to_string(layer_num) + "_heatmap.png'\n";
    gpfile << "set view map\n";
    gpfile << "plot 'layer" + std::to_string(layer_num) + "_activations.txt' matrix with image\n";
    gpfile.close();

    system(("gnuplot plot_layer" + std::to_string(layer_num) + ".gp").c_str());
}*/
/*
void train_model(NeuralNetwork& model, const std::vector<std::pair<MatrixXf, MatrixXf>>& dataset, int epochs, float learning_rate, int batch_size) {
    int num_batches = dataset.size() / batch_size;

    for (int epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0;

        auto shuffled_dataset = dataset;
        std::random_shuffle(shuffled_dataset.begin(), shuffled_dataset.end());
        float previous_loss = std::numeric_limits<float>::max();
        for (int batch = 0; batch < num_batches; batch++) {
            std::vector<MatrixXf> X_batch, y_batch;
            for (int i = 0; i < batch_size; i++) {
                X_batch.push_back(shuffled_dataset[batch * batch_size + i].first);
                y_batch.push_back(shuffled_dataset[batch * batch_size + i].second);
            }

            MatrixXf predictions = MatrixXf::Zero(y_batch[0].rows(), y_batch[0].cols());
            MatrixXf batch_targets = MatrixXf::Zero(y_batch[0].rows(), y_batch[0].cols());
            for (size_t i = 0; i < batch_size; i++) {
                predictions += model.forward(X_batch[i]);
                batch_targets += y_batch[i];
            }
            predictions /= batch_size;
            batch_targets /= batch_size;

            float batch_loss = (predictions - batch_targets).array().square().sum() / predictions.rows();
            epoch_loss += batch_loss;          

            MatrixXf grad = 2 * (predictions - batch_targets) / predictions.rows();
                model.backward(grad);
            model.update(learning_rate);
        }

        if (epoch_loss < 1e-6) {
            cout << "Early stopping at epoch " << epoch << " with loss: " << epoch_loss << endl;
            break;
        }

        if (abs(previous_loss - epoch_loss) < 1e-8) {
            cout << "Converged at epoch " << epoch << " with loss: " << epoch_loss << endl;
            break;
        }

        previous_loss = epoch_loss;

        std::cout << "Epoch " << epoch << " Loss: " << epoch_loss / num_batches << endl;
    }
}
*/
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
    //std::vector<std::string> filenames = {"cifar-10-batches-bin/data_batch_1.bin"};
    //std::vector<std::string> test_filenames = {"cifar-10-batches-bin/test_batch.bin"}; 
    //"cifar-10-batches-bin/data_batch_2.bin", "cifar-10-batches-bin/data_batch_3.bin", 
    //"cifar-10-batches-bin/data_batch_4.bin", "cifar-10-batches-bin/data_batch_5.bin"};
    /*std::string training_images = "mnist/train-images.idx3-ubyte";
    std::string training_labels = "mnist/train-labels.idx1-ubyte";
    std::string test_images = "mnist/t10k-images.idx3-ubyte";
    std::string test_labels = "mnnist/t10k-labels.idx1-ubyte";

    MnistData training_mnist(training_images, training_labels);
    MnistData test_mnist(test_images, test_labels);
    std::vector<std::pair<MatrixXf, MatrixXf>> training_dataset = training_mnist.get_dataset();
    std::vector<std::pair<MatrixXf, MatrixXf>> test_dataset = test_mnist.get_dataset();*/
    

    /*Cifar10data cifar(filenames);
    std::vector<std::pair<MatrixXf, MatrixXf>> dataset = cifar.get_dataset();

    Cifar10data test_cifar(test_filenames);
    std::vector<std::pair<MatrixXf, MatrixXf>> test_dataset = test_cifar.get_dataset();
    */

    //NeuralNetwork model;

    //model.add(std::make_unique<DenseLayer>(784, 56));
    //model.add(std::make_unique<OutputLayer>(56, 10));

    /*train_model(model, training_dataset, 1000, 0.1, 100);

    int correct = 0;
    for (auto& [input, label] : test_dataset) {
        MatrixXf output = model.forward(input);

        Eigen::Index pred_col;
        output.row(0).maxCoeff(&pred_col);

        Eigen::Index label_col;
        label.row(0).maxCoeff(&label_col);

        if (pred_col == label_col) {
            correct++;
        }

        std::cout << "Predicted: " << output << std::endl;
        std::cout << " Actual: " << label << std::endl;
    }

    float accuracy = static_cast<float>(correct) / test_dataset.size();

    std::cout << "Accuracy: " << accuracy << std::endl;
    */
    int temp_batch_size = 1;
    int in_channels = 3;
    int height = 230;
    int width = 230;
    int kernel_size = 3;
    int out_channels = 1;
    int kernel_height = 3;
    int kernel_width = 3;
    int padding = 0;
    int stride = 1;

    using Derivative = Eigen::VectorXd;
    using ADScalar = Eigen::AutoDiffScalar<Derivative>;

    //Tensor<float, 4> tensor(batch_size, in_channels, height, width);
    std::srand(static_cast<unsigned int>(std::time(0)));

    int output_height = height - kernel_size + 1;  // 5 - 3 + 1 = 3
    int output_width = width - kernel_size + 1;
    
    Tensor<float, 4> tensor(32, 3, 32, 32);
    tensor.setRandom();
    /*tensor.setValues({{{
        {  1024,   2,   3,   4,   50,   6,   7,   8,   9,  1000,  110,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32},
        { 33,  34,  35,  36,  37,  308,  39,  400,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64},
        { 65,  66,  67,  68,  69,  700,  71,  72,  73,  74,  75,  706,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96},
        { 97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128},
        {129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160},
        {161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192},
        {193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224},
        {225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256},
        {257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288},
        {289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320},
        {321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352},
        {353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384},
        {385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416},
        {417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448},
        {449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480},
        {481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512},
        {513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544},
        {545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576},
        {577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608},
        {609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640},
        {641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672},
        {673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704},
        {705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736},
        {737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768},
        {769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800},
        {801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832},
        {833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864},
        {865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896},
        {897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928},
        {929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960},
        {961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992},
        {993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024}},
        {{  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  170,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32},
        { 33,  34,  35,  36,  37,  38,  39,  40,  401,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64},
        { 65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96},
        { 97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128},
        {129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160},
        {161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192},
        {193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224},
        {225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256},
        {257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288},
        {289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320},
        {321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352},
        {353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384},
        {385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416},
        {417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448},
        {449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480},
        {481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512},
        {513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544},
        {545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576},
        {577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608},
        {609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640},
        {641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672},
        {673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704},
        {705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736},
        {737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768},
        {769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800},
        {801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832},
        {833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864},
        {865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896},
        {897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928},
        {929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960},
        {961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992},
        {993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024}},
        {{  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32},
        { 33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64},
        { 65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96},
        { 97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128},
        {129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160},
        {161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192},
        {193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224},
        {225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256},
        {257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288},
        {289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320},
        {321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352},
        {353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384},
        {385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416},
        {417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448},
        {449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480},
        {481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512},
        {513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544},
        {545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576},
        {577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608},
        {609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640},
        {641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672},
        {673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704},
        {705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736},
        {737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768},
        {769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800},
        {801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832},
        {833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864},
        {865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896},
        {897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928},
        {929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960},
        {961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992},
        {993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024}}
        }});

    tensor = tensor / 1024.0f;*/
    /*
    Tensor<float, 4> smallTensor()

    ConvLayer conv(3, 3, 7, 2);
    BatchNorm batch(3, 3);
    ActivationFunction activationLayer(3, 3, Activation::relu);
    PoolingLayer pool(3, 3, PoolingLayer::MAX, 3, 2);

    Tensor<float, 4> result = conv.forward(tensor);
    Tensor<float, 4> relu_result = activationLayer.forward(result);
    Tensor<float, 4> norm_result = batch.forward(relu_result);
    Tensor<float, 4> pool_result = pool.forward(norm_result);
    //Tensor<float, 4> dense_result = db.forward(pool_result);

    // Print tensor shapes and sample values
    std::cout << "\n=== Layer Outputs ===\n";
    std::cout << "Conv output shape: " << result.dimensions() << std::endl;
    //std::cout << "Dense block output shape: " << dense_result.dimensions() << std::endl;
    
    // Use the printTensor4D function to inspect layer outputs
    // Limit printing to avoid overwhelming output for large tensors
    printTensor4D(result, "Conv Output", 3, 5, 5);           // First 2 channels, 5x5 pixels
    printTensor4D(relu_result, "ReLU Output", 1, 3, 3);      // First channel, 3x3 pixels
    printTensor4D(norm_result, "BatchNorm Output", 1, 3, 3); // First channel, 3x3 pixels
    printTensor4D(pool_result, "Pooling Output", 2, 4, 4);   // First 2 channels, 4x4 pixels
    //printTensor4D(dense_result, "DenseBlock Output", 3, 3, 3); // First 3 channels, 3x3 pixels
    */
     // Create a proper 224×224 test input for DenseNet (normalized to [0,1])
     //Tensor<float, 4> testInput(1, 3, 224, 224);
     //testInput.setRandom();  // Random values in [-1, 1]
     //testInput = (testInput + 1.0f) * 0.5f;  // Scale to [0, 1]


    /*
    auto spatial_dims = Eigen::array<int, 3>({0, 2, 3});  // mean over batch, height, width
    Tensor<float, 1> tensorMean = tensor.mean(spatial_dims);
    std::cout << "mean before batch" << tensorMean;

    printTensor4D(tensor, "tensor before batch norm", 3, 5, 5);
    BatchNorm batchLayer(3);
    Tensor<float, 4> summable(tensor.dimension(0), tensor.dimension(1), tensor.dimension(2), tensor.dimension(3));
    float sum = 0.f;
    Eigen::array<int, 3>dims{0, 2, 3};
    Eigen::array<ptrdiff_t, 4> bcast({tensor.dimension(0), 1, tensor.dimension(2), tensor.dimension(3)});
    Eigen::array<ptrdiff_t, 4> resize({1, tensor.dimension(1), 1, 1});

    Tensor<float, 1> testGamma(tensor.dimension(1));
    testGamma.setValues({1.0f, 2.0f, 3.0f});
    // Time the BatchNorm forward pass (small test area)
    auto t0 = std::chrono::steady_clock::now();
    /*for (int b = 0; b < tensor.dimension(0); b++) {
        for (int c = 0; c < tensor.dimension(1); c++) {
            for (int h = 0; h < tensor.dimension(2); h++) {
                for (int w = 0; w < tensor.dimension(3); w++) {
                    summable(b, c, h, w) = tensor(b, c, h, w) * testGamma(c);
                }
            }
        }
    }*/
    
    //Tensor<float, 4> batchTest = batchLayer.forward(tensor);
    //auto t1 = std::chrono::steady_clock::now();
    //auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    //auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    //std::cout << "BatchNorm forward time: " << elapsed_us << " us (" << elapsed_ms << " ms)" << std::endl;
    
    //printTensor4D(batchTest, "batch test", 3, 5, 5);

    //Tensor<float, 1> mean = batchTest.mean(spatial_dims);
    //std::cout << "mean after batch" << mean;
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
            // Print intermediate outputs to debug
            if (i == 0) {
                printTensor4D(output, "After first Conv7x7", 5, 3, 3);
            }
            if (i == 4) {
                printTensor4D(output, "After first DenseBlock", 10, 2, 2);
            }
            if (i == 12) {
                printTensor4D(output, "After second DenseBlock", 10, 2, 2);
            }
            if (i == 15) {
                printTensor4D(output, "After third DenseBlock before", 10, 2, 2);
            }
            if (i == 16) {
                printTensor4D(output, "After third DenseBlock", 10, 2, 2);
            }
            if (i == dense_net.size() - 6) {
                printTensor4D(output, "After ReLU before last DenseBlock", 10, 2, 2);
            }
            if (i == dense_net.size() - 5) {
                printTensor4D(output, "Before last DenseBlock", 10, 2, 2);
            }
            if (i == dense_net.size() - 4) {
                printTensor4D(output, "After last DenseBlock", 10, 2, 2);
            }
            if (i == dense_net.size() - 3) {
                printTensor4D(output, "After Global Pool (before Linear)", 10, 1, 1);
            }
            if (i == dense_net.size() - 2) {
                printTensor4D(output, "After Linear (logits before softmax)", 10, 1, 1);
            }
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


    


    

    /*for (int b = 0; b < tensor.dimension(0); b++) {
        for (int c = 0; c < tensor.dimension(1); c++) {
            for (int h = 0; h < tensor.dimension(2); h++) {
                for (int w = 0; w < tensor.dimension(3); w++) {
                    std::cout << norm_result(b, c, h, w) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }*/


    
    //PoolingLayer pool(PoolingLayer::AVERAGE, 2, 2);
    //Tensor<float, 4> poolTEst = pool.forward(newt_test);

    //std::vector<std::unique_ptr<Layer>> denseBlockLayers;

    

    //std::cout << poolTEst << std::endl;

    // --- small training entry (use existing loaders and helper above) ---
    // parse simple args
    
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
    
    return 0;
}


//2.30259
//2.31201
//2.23396
//2.17344
//2.12826
//2.08933
//2.06554
//2.0454
//2.03368
//2.02248
//2.20732
//2.01298
//2.00791
//2.00115
//1.99185
//2.17109
//2.16435
//2.15776
