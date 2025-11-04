#include "DenseBlock.h"


DenseBlock::DenseBlock(int input_channels, int growth_rate, int layers)
    : growth_rate(growth_rate), initial_channels(input_channels), layer_count(layers)
{

    for(int i = 1; i <= layers; i++) {
        compositeLayers.push_back(std::make_unique<DenseLayerComposite>(growth_rate, input_channels + growth_rate*(i-1)));
    }
}

Tensor<float, 4> DenseBlock::forward(const Tensor<float, 4> &input) {
    this->input = input;
    // Reset channel_sizes for this forward pass (was accumulating across calls)
    channel_sizes.clear();
    Tensor<float, 4> concatenated = input;
    channel_sizes.push_back(input.dimension(1));

    for (auto& layer : compositeLayers) {
        Tensor<float, 4> new_features = layer->forward(concatenated);
        channel_sizes.push_back(new_features.dimension(1));
        Tensor<float, 4> temp = concatenated.concatenate(new_features, 1);
        concatenated = temp;
    }
    return concatenated;
}

Eigen::Tensor<float, 4> DenseBlock::backward(const Eigen::Tensor<float, 4> &dY)
{
    const int B = input.dimension(0);
    const int H = input.dimension(2);
    const int W = input.dimension(3);

    // 1. Split dY into slices for each layer’s output contribution
    std::vector<Eigen::Tensor<float, 4>> dY_slices;
    dY_slices.reserve(channel_sizes.size());

    int start = 0;
    for (int c : channel_sizes)
    {
        Eigen::array<int, 4> offsets = {0, start, 0, 0};
        Eigen::array<int, 4> extents = {B, c, H, W};
        dY_slices.push_back(dY.slice(offsets, extents).eval());
        start += c;
    }

    // 2. Prepare gradient accumulators for each feature map (x0, x1, x2, ...)
    std::vector<Eigen::Tensor<float, 4>> grads;
    grads.resize(channel_sizes.size() + 1); // +1 for the original input
    for (auto& g : grads)
        g = Eigen::Tensor<float, 4>(B, 0, H, W); // empty init for clarity

    // 3. Start backward pass from the last layer
    // Note: The last slice of dY corresponds to the last layer’s output
    for (int i = (int)compositeLayers.size() - 1; i >= 0; --i)
    {
        // Each layer gets its gradient from dY_slices[i+1]
        Eigen::Tensor<float, 4> dYi = dY_slices[i + 1];

        // Backward through this composite layer
        Eigen::Tensor<float, 4> dXi = compositeLayers[i]->backward(dYi);

        // Split dXi into per-input parts (since each layer gets concat inputs)
        int start_ch = 0;
        for (int j = 0; j <= i; ++j)
        {
            int c = channel_sizes[j];
            Eigen::array<int, 4> offsets = {0, start_ch, 0, 0};
            Eigen::array<int, 4> extents = {B, c, H, W};
            Eigen::Tensor<float, 4> slice = dXi.slice(offsets, extents).eval();

            // Accumulate into gradient for the corresponding earlier input
            if (grads[j].dimension(1) == 0) {
                grads[j] = slice; // first write
            } else {
                grads[j] = grads[j] + slice; // accumulate from multiple paths
            }

            start_ch += c;
        }
    }

    // 4. Finally, the gradient wrt DenseBlock’s input is grads[0]
    Eigen::Tensor<float, 4> dX = grads[0];

    return dX;
}

std::vector<Tensor<float, 4>> DenseBlock::splitChannels(const Tensor<float, 4>& concatTensor, const std::vector<int>& channelSizes) {
    std::vector<Tensor<float, 4>> outputs;
    int start = 0;
    for (int c : channelSizes) {
        outputs.push_back(concatTensor.slice(
            Eigen::array<int, 4>({0, start, 0, 0}),
            Eigen::array<int, 4>({concatTensor.dimension(0), c,
                                  concatTensor.dimension(2), concatTensor.dimension(3)})));
        start += c;
    }
    return outputs;
}