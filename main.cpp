#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <random>
#include <cmath>
#include <iterator>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/AutoDiff>
#include "Layers.h"
#include "ConvLayer.h"
#include "BatchNorm.h"
#include "Activation.h"
#include "DenseConstruct.h"
#include "DenseBlock.h"
#include "Pooling.h"

using namespace Eigen;
using namespace std;

Eigen::MatrixXd EMPTY(0,0);

MatrixXd sigmoid(const MatrixXd& x) {
    return 1.0 / (1.0 + (-x).array().exp());
}

MatrixXd sigmoid_derivative(const MatrixXd& x) {
    MatrixXd s = sigmoid(x);
    return s.array() * (1.0 - s.array());
}

MatrixXd relu_derivative(const MatrixXd& x) {
    return (x.array() > 0.0).cast<double>();
}

/*class Layer {
    public:
        MatrixXd output, input, weights;

        virtual void forward(const MatrixXd& input) = 0;
        virtual void backward(const MatrixXd& grad, const MatrixXd& weight = EMPTY) = 0;
        virtual void update(double learning_rate) = 0;
        virtual ~Layer() = default;
};

class DenseLayer : public Layer {
    private:
        MatrixXd bias;
        MatrixXd dW;
        MatrixXd db;
    public:

        DenseLayer(int input_size, int output_size) {
            weights = MatrixXd::Random(input_size, output_size) * sqrt(2.0/input_size);
            bias = MatrixXd::Zero(1, output_size);
        }

        void forward(const MatrixXd& input) override {
            this->input = input;
            output = relu((input*weights) + bias.replicate(input.rows(), 1));
        }

        void backward(const MatrixXd& grad, const MatrixXd& weight) override {
            MatrixXd error = grad * weight.transpose();
            output = relu_derivative(output).array() * error.array();
            dW = (input.transpose() * output);
            db = output.colwise().sum() / input.rows();
        }

        void update(double learning_rate) override {
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

        Eigen::MatrixXd forward(const Eigen::MatrixXd& input) {
            Eigen::MatrixXd current_input = input;
            Eigen::MatrixXd concat_output = input;

            for (auto& layer : layers) {
                layer->forward(current_input);
                Eigen::MatrixXd output = layer->output;
                concat_output.conservativeResize(concat_output.rows(), concat_output.cols() + output.cols());
                concat_output.rightCols(output.cols()) = output;
                current_input = concat_output;
            }
            return concat_output;
        }
};

class OutputLayer : public Layer  {
    private:
        MatrixXd bias;
        MatrixXd dW;
        MatrixXd db;
    public:
        OutputLayer(int input_size, int output_size) {
            double limit = sqrt(6.0 / (input_size + output_size));
            weights = MatrixXd::Random(input_size, output_size) * limit;
            bias = MatrixXd::Zero(1, output_size);
        }

        void forward(const MatrixXd& input) override {
            this->input = input;
            output = sigmoid((input*weights) + bias.replicate(input.rows(), 1));
        }

        void backward(const MatrixXd& grad, const MatrixXd& weight) override {
            MatrixXd sigmoid = sigmoid_derivative(output);
            output = grad.array() * sigmoid.array(); 
            dW = (input.transpose() * output);
            db = output.colwise().sum();
        }

        void update(double learning_rate) override {
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

        MatrixXd forward(const MatrixXd& input) {
            MatrixXd out = input;

            for (auto& layer : layers) {
                layer->forward(out);
                out = layer->output;
            }

            return out;
        }

        void backward(const MatrixXd& grad_output) {
            MatrixXd grad = grad_output;
            MatrixXd weight;
            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                (*it)->backward(grad, weight);
                grad = (*it)->output;
                weight = (*it)->weights;
            }
        }

        void update(double learning_rate) {
            for (auto& layer : layers) {
                layer->update(learning_rate);
            }
        }
};

MatrixXd generate_random_matrix(int rows, int cols) {
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_real_distribution<double> dist(-1.0, 1.0);
    MatrixXd mat(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat(i, j) = dist(rng);
        }
    }
    return mat;
}

*/

std::vector<std::pair<MatrixXd, MatrixXd>> generate_dataset(std::vector<MatrixXd>& X) {
    std::vector<std::pair<MatrixXd, MatrixXd>> dataset;

    for (auto& x : X) {
        MatrixXd y = x.array().sin();
        y = (y.array() - y.minCoeff()) / (y.maxCoeff() - y.minCoeff());
        dataset.push_back({x, y});
    }
    return dataset;
}

/*void visualize_layer(const MatrixXd& activations, int layer_num) {
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
void train_model(NeuralNetwork& model, const std::vector<std::pair<MatrixXd, MatrixXd>>& dataset, int epochs, double learning_rate, int batch_size) {
    int num_batches = dataset.size() / batch_size;

    for (int epoch = 0; epoch < epochs; epoch++) {
        double epoch_loss = 0.0;

        auto shuffled_dataset = dataset;
        std::random_shuffle(shuffled_dataset.begin(), shuffled_dataset.end());
        double previous_loss = std::numeric_limits<double>::max();
        for (int batch = 0; batch < num_batches; batch++) {
            std::vector<MatrixXd> X_batch, y_batch;
            for (int i = 0; i < batch_size; i++) {
                X_batch.push_back(shuffled_dataset[batch * batch_size + i].first);
                y_batch.push_back(shuffled_dataset[batch * batch_size + i].second);
            }

            MatrixXd predictions = MatrixXd::Zero(y_batch[0].rows(), y_batch[0].cols());
            MatrixXd batch_targets = MatrixXd::Zero(y_batch[0].rows(), y_batch[0].cols());
            for (size_t i = 0; i < batch_size; i++) {
                predictions += model.forward(X_batch[i]);
                batch_targets += y_batch[i];
            }
            predictions /= batch_size;
            batch_targets /= batch_size;

            double batch_loss = (predictions - batch_targets).array().square().sum() / predictions.rows();
            epoch_loss += batch_loss;          

            MatrixXd grad = 2 * (predictions - batch_targets) / predictions.rows();
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

class Cifar10data {
    private:
        std::vector<std::string> filenames;
        std::vector<std::pair<MatrixXd, MatrixXd>> dataset;
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

                    MatrixXd image(1, 3072);
                    for (int j = 0; j < 3072; j++) {
                        image(0, j) = static_cast<double>(bytes[j + 1]) / 255.0;
                    }

                    MatrixXd label = MatrixXd::Zero(1, 10);
                    label(0, bytes[0]) = 1.0;

                    dataset.push_back({image, label});
                }
                

                file.close();
            }
        }

        std::vector<std::pair<MatrixXd, MatrixXd>> get_dataset() {
            return dataset;
        }
};
*/
class MnistData {
    private:
        std::string images;
        std::string labels;
        std::vector<std::pair<MatrixXd, MatrixXd>> dataset;
    public:
        MnistData(std::string &images, std::string &labels) {
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
                std::cerr << "Could not open file: " << images << std::endl;
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

            std::vector<MatrixXd> images;
            for (int i = 0; i < num_images; i++) {
                MatrixXd image(1, rows*cols);
                for (int r = 0; r < rows*cols; r++) {
                    unsigned char pixel;
                    image_file.read(reinterpret_cast<char*>(&pixel), 1);
                    image(0, r) = static_cast<double>(pixel) / 255.0;
                }
                unsigned char label;
                label_file.read(reinterpret_cast<char*>(&label), 1);
                MatrixXd target = MatrixXd::Zero(1, 10);
                target(0, label) = 1.0;
                dataset.push_back({image, target});
            }
        }

        std::vector<std::pair<MatrixXd, MatrixXd>> get_dataset() {
            return dataset;
        }
};

namespace Activation {
    Tensor<double, 4> relu(const Tensor<double, 4> &input) {
        return input.cwiseMax(0.0);
    }

    Tensor<double, 4> relu_grad(const Tensor<double, 4> &grad, const Tensor<double, 4> &output) {
        return output;
    }
}

template<typename LayerType, typename... Args>
std::unique_ptr<ConstructInfo> makeLayer(Args&&... args) {
    return std::make_unique<LayerConstructInfo<LayerType, Args...>>(
        std::forward<Args>(args)...
    );
}

int main() {
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
    std::vector<std::pair<MatrixXd, MatrixXd>> training_dataset = training_mnist.get_dataset();
    std::vector<std::pair<MatrixXd, MatrixXd>> test_dataset = test_mnist.get_dataset();*/
    

    /*Cifar10data cifar(filenames);
    std::vector<std::pair<MatrixXd, MatrixXd>> dataset = cifar.get_dataset();

    Cifar10data test_cifar(test_filenames);
    std::vector<std::pair<MatrixXd, MatrixXd>> test_dataset = test_cifar.get_dataset();
    */

    //NeuralNetwork model;

    //model.add(std::make_unique<DenseLayer>(784, 56));
    //model.add(std::make_unique<OutputLayer>(56, 10));

    /*train_model(model, training_dataset, 1000, 0.1, 100);

    int correct = 0;
    for (auto& [input, label] : test_dataset) {
        MatrixXd output = model.forward(input);

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

    double accuracy = static_cast<double>(correct) / test_dataset.size();

    std::cout << "Accuracy: " << accuracy << std::endl;
    */
    int batch_size = 1;
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

    //Tensor<double, 4> tensor(batch_size, in_channels, height, width);
    std::srand(static_cast<unsigned int>(std::time(0)));

    int output_height = height - kernel_size + 1;  // 5 - 3 + 1 = 3
    int output_width = width - kernel_size + 1;
    
    Tensor<double, 4> tensor(1, 3, height, width);
    tensor.setRandom();
    /*tensor.setValues({{{
        {  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32},
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
        }});*/


    ConvLayer conv(3, 3, 7, 2);
    BatchNorm batch(3, 3);
    ActivationFunction activationLayer(3, 3, Activation::relu);
    PoolingLayer pool(3, 3, PoolingLayer::MAX, 3, 2);
    DenseBlock db(3, 32, 4, 3);
    db.addCompositeLayer(
        makeLayer<BatchNorm>(),
        makeLayer<ActivationFunction>(Activation::relu),
        makeLayer<ConvLayer>(1, 1),
        makeLayer<BatchNorm>(),
        makeLayer<ActivationFunction>(Activation::relu),
        makeLayer<ConvLayer>(3, 1)
    );

    Tensor<double, 4> result = conv.forward(tensor);
    Tensor<double, 4> relu_result = activationLayer.forward(result);
    Tensor<double, 4> norm_result = batch.forward(relu_result);
    Tensor<double, 4> pool_result = pool.forward(norm_result);
    Tensor<double, 4> dense_result = db.forward(pool_result);


    std::cout << result.dimensions() << std::endl;
    std::cout << dense_result.dimensions() << std::endl;

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

    Tensor<double, 4> test(2, 3, 2, 2);
    test.setValues(
    {{{{3, 3},
    {3, 3}},
    {{2, 2},
    {2, 2}},
    {
        {1, 2},
        {3, 4}
    }},
    {
        {{3, 3},
    {3, 3}},
    {{2, 2},
    {2, 2}},
    {
        {2, 2},
        {3, 4}
    }
    }});

    Tensor<double, 4> test2(2, 3, 2, 2);
    test2.setValues(
    {{{{3, 3},
    {3, 3}},
    {{2, 2},
    {2, 2}},
    {
        {1, 2},
        {3, 4}
    }},
    {
        {{3, 3},
    {3, 3}},
    {{2, 2},
    {2, 2}},
    {
        {2, 2},
        {3, 4}
    }
    }});

    Tensor<double, 4> newt_test(1, 1, 4, 4);
    newt_test.setValues({{{{1, 2, 3, 4},
                            {5, 6, 7, 8},
                            {9, 10, 11, 12},
                            {13, 14, 15, 16}
                            }}});

    
    //PoolingLayer pool(PoolingLayer::AVERAGE, 2, 2);
    //Tensor<double, 4> poolTEst = pool.forward(newt_test);

    //std::vector<std::unique_ptr<Layer>> denseBlockLayers;

    

    //std::cout << poolTEst << std::endl;

    Tensor<double, 2> chipped = test.chip(0, 0).chip(0, 0);
    Tensor<double, 4> out(1, 1, 1, 1);
    out.setZero();
    Tensor<double, 0> max_val = chipped.maximum();
    double new_max = max_val(0);
    out(0, 0, 0, 0) = new_max;

    std::cout << out << std::endl;

    return 0;
}


//TODO
//Make dense block creation more modular, instead of being static
