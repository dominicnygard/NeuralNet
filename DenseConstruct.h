#ifndef DENSECONSTRUCT_H
#define DENSECONSTRUCT_H
#include "Layers.h"

class ConstructInfo {
    public:
        virtual std::unique_ptr<Layer> construct(int in_channels, int out_channels) = 0;
        virtual ~ConstructInfo() = default;
};

template<typename LayerType, typename... Args>
class LayerConstructInfo : public ConstructInfo {
    private:
        std::tuple<Args...> construction_args;
    public:
        LayerConstructInfo (Args... args)
            : construction_args(std::forward<Args>(args)...) {}

        std::unique_ptr<Layer> construct(int in_channels, int out_channels) override {
            return constructHelper(in_channels, out_channels, std::make_index_sequence<sizeof...(Args)>{});
        }

    private:
        template<size_t... Is>
        std::unique_ptr<Layer> constructHelper(int in_channels, int out_channels, std::index_sequence<Is...>) {
            return std::make_unique<LayerType>(
                in_channels,
                out_channels,
                std::get<Is>(construction_args)...
            );
            
          
        }
};

#endif