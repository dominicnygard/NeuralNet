#ifndef POOLING_H
#define POOLING_H
#include "Layers.h"

class PoolingLayer : public Layer {
    public:
        enum PoolingType {MAX, AVERAGE};
        enum Mode {REGULAR, GLOBAL};

        PoolingLayer(int input_chan, int output_chan, PoolingType type, int pool_size, int stride, Mode mode = REGULAR); 

        Tensor<float, 4> forward(const Tensor<float, 4> &input) override;

        void backward(const MatrixXd& grad) override {}
        void update(float learnign_rate) override {}
    private:
        PoolingType type;
        int pool_size;
        int stride;
        Mode mode;
};

#endif