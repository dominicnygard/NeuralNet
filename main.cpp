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

void readfile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        return;
    }

    std::vector<unsigned char> bytes(3073);
    file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());

    MatrixXd image(3, 1024);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 1024; j++) {
            image(i, j) = static_cast<double>(bytes[i*1024 + j + 1]) / 255.0;
        }
    }

    MatrixXd label = MatrixXd::Zero(1, 10);
    label(0, bytes[0]) = 1.0;

    std::cout << "Label: " << bytes[0] << std::endl;
    std::cout << "Image: " << image << std::endl;
    std::cout << "Label: " << label << std::endl;
    

    file.close();
}

int main() {
    
    /*NeuralNetwork model;

    model.add(std::make_unique<DenseLayer>(3, 6));
    model.add(std::make_unique<OutputLayer>(6, 3));

    std::vector<MatrixXd> X;

    for (int i = 0; i < 1024; i++) {
        X.push_back(generate_random_matrix(4, 3));
    }

    std::vector<std::pair<MatrixXd, MatrixXd>> dataset = generate_dataset(X);

    train_model(model, dataset, 10000, 0.1, 32);

    MatrixXd test_input = generate_random_matrix(4, 3);
    MatrixXd test_output = model.forward(test_input);
    MatrixXd expected_output = test_input.array().sin();
    expected_output = (expected_output.array() - expected_output.minCoeff()) / (expected_output.maxCoeff() - expected_output.minCoeff());
    std::cout << "Test input:\n" << test_input << std::endl;
    std::cout << "Prediction:\n" << test_output << std::endl;
    std::cout << "Expected output:\n" << expected_output << std::endl;
    double accuracy = (test_output - expected_output).array().abs().sum() / test_output.rows();
    std::cout << "Inaccuracy: " << accuracy*100 << "%" << std::endl;
    */

    readfile("cifar-10-batches-bin/data_batch_1.bin");

    return 0;

}