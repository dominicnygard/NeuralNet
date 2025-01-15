#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <random>
#include <cmath>
#include <Eigen/Dense>


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

MatrixXd relu(const MatrixXd& x) {
    return x.array().max(0.0);
}

MatrixXd relu_derivative(const MatrixXd& x) {
    return (x.array() > 0.0).cast<double>();
}

class Layer {
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


int main() {
    //std::vector<std::string> filenames = {"cifar-10-batches-bin/data_batch_1.bin"};
    //std::vector<std::string> test_filenames = {"cifar-10-batches-bin/test_batch.bin"}; 
    //"cifar-10-batches-bin/data_batch_2.bin", "cifar-10-batches-bin/data_batch_3.bin", 
    //"cifar-10-batches-bin/data_batch_4.bin", "cifar-10-batches-bin/data_batch_5.bin"};
    std::string training_images = "mnist/train-images.idx3-ubyte";
    std::string training_labels = "mnist/train-labels.idx1-ubyte";
    std::string test_images = "mnist/t10k-images.idx3-ubyte";
    std::string test_labels = "mnnist/t10k-labels.idx1-ubyte";

    MnistData training_mnist(training_images, training_labels);
    MnistData test_mnist(test_images, test_labels);
    std::vector<std::pair<MatrixXd, MatrixXd>> training_dataset = training_mnist.get_dataset();
    std::vector<std::pair<MatrixXd, MatrixXd>> test_dataset = test_mnist.get_dataset();
    

    /*Cifar10data cifar(filenames);
    std::vector<std::pair<MatrixXd, MatrixXd>> dataset = cifar.get_dataset();

    Cifar10data test_cifar(test_filenames);
    std::vector<std::pair<MatrixXd, MatrixXd>> test_dataset = test_cifar.get_dataset();
    */

    NeuralNetwork model;

    model.add(std::make_unique<DenseLayer>(784, 56));
    model.add(std::make_unique<OutputLayer>(56, 10));

    train_model(model, training_dataset, 1000, 0.1, 100);

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

    return 0;
}