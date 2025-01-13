#include <iostream>
#include <vector>
#include <memory>
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
            weights = MatrixXd::Random(input_size, output_size);
            bias = MatrixXd::Zero(1, output_size);
        }

        void forward(const MatrixXd& input) override {
            this->input = input;
            output = sigmoid((input*weights) + bias.replicate(input.rows(), 1));
        }

        void backward(const MatrixXd& grad, const MatrixXd& weight) override {
            MatrixXd error = grad * weight.transpose();
            output = sigmoid_derivative(output).array() * error.array();
            dW = (input.transpose() * output);
            db = output.colwise().sum();
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
            weights = MatrixXd::Random(input_size, output_size);
            bias = MatrixXd::Zero(1, output_size);
        }

        void forward(const MatrixXd& input) override {
            this->input = input;
            output = sigmoid((input*weights) + bias.replicate(input.rows(), 1));;
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

int main() {

    NeuralNetwork model;

    model.add(std::make_unique<DenseLayer>(4, 8));
    model.add(std::make_unique<DenseLayer>(8, 8));
    model.add(std::make_unique<DenseLayer>(8, 8));
    model.add(std::make_unique<DenseLayer>(8, 8));
    model.add(std::make_unique<DenseLayer>(8, 8));
    model.add(std::make_unique<DenseLayer>(8, 8));
    model.add(std::make_unique<OutputLayer>(8, 1));

    MatrixXd x = MatrixXd::Random(4, 4);

    MatrixXd y(4, 1); 
    y << 1, 0, 1, 1;
    std::cout << "Target:\n" << y << std::endl;

    double learning_rate = 0.5;

    for (int epoch = 0; epoch < 10000; epoch++) {
        MatrixXd predictions = model.forward(x);

        double loss = (predictions - y).array().square().sum();

        if (epoch % 100 == 0) {
            cout << "Epoch " << epoch << " Loss: " << loss << endl;
        }

        MatrixXd grad = predictions - y;
        model.backward(grad);

        model.update(learning_rate);
    }

    MatrixXd test_input = MatrixXd::Random(4, 4);

    MatrixXd test_output = model.forward(test_input);
    std::cout << "Test input:\n" << test_input << std::endl;
    std::cout << "Prediction:\n" << test_output << std::endl;

    return 0;
}
