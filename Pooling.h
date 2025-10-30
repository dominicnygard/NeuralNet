#ifndef POOLING_H
#define POOLING_H
#include "Layers.h"

class PoolingLayer : public Layer {
public:
    enum PoolingType {MAX, AVERAGE};
    enum Mode {REGULAR, GLOBAL};

    PoolingLayer(PoolingType type, int pool_size, int stride, Mode mode = REGULAR); 

    Tensor<float, 4> forward(const Tensor<float, 4> &input) override;

    Eigen::Tensor<float, 4> backward(const MatrixXf& grad) override {}
    void update(float learnign_rate) override {}
private:
    PoolingType type;
    int pool_size;
    int stride;
    Mode mode;
};

#endif