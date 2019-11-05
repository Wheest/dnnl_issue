#include <assert.h>

#include <chrono>
#include <numeric>
#include <vector>
#include <unordered_map>
#include "example_utils.hpp"

using namespace dnnl;

void run_network()
{

    int kdim1;
    int kdim2;
    int stride;
    int pad_hl;
    int pad_hr;
    int pad_wl;
    int pad_wr;
    int in_c;
    int num_filters;
    int in_h;
    int in_w;
    int out_h;
    int out_w;
    std::vector<float> inputs_data;
    std::vector<float> kernels;
    std::vector<float> targets;
    std::vector<float> biases;

    kdim1 = 2;
    kdim2 = 2;
    stride = 1;
    pad_hl = 0;
    pad_hr = 0;
    pad_wl = 0;
    pad_wr = 0;
    in_c = 3;
    num_filters = 2;
    in_h = 2;
    in_w = 2;
    inputs_data = {0, -2,
                   -2, -1,

                   -5,  2,
                   0,  2,

                   -2,  4,
                   -3,  1};
    kernels = {3,  2,
               4, -5,

               5, -4,
               -2, -2,

               1,  4,
               -2, -4,


               3,  3,
               -1, -3,

               -4,  0,
               -5,  3,

               2,  3,
               0, -2};
    biases = {-2, -2};
    targets = {-30, 29};
    out_h = (in_h + pad_hl + pad_hr - kdim1) / stride + 1;
    out_w = (in_w + pad_wl + pad_wr - kdim2) / stride + 1;

    // following example of mkl-dnn/examples/cnn_inference_f32.cpp
    engine eng(engine::kind::cpu, 0);
    stream s(eng);

    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    memory::dims conv1_src_tz = {1, in_c, in_h, in_w};
    memory::dims conv1_weights_tz = {num_filters, in_c, kdim1, kdim1};
    memory::dims conv1_bias_tz = {num_filters};
    memory::dims conv1_dst_tz = {1, num_filters, out_h, out_w};
    memory::dims conv1_strides = {stride, stride};
    memory::dims conv1_padding = {pad_hl, pad_hr};

    std::cout << "writing to memory" << std::endl;
    auto user_src_memory = memory({{conv1_src_tz}, dt::f32, tag::nchw}, eng);
    write_to_dnnl_memory(inputs_data.data(), user_src_memory);
    auto user_weights_memory
        = memory({{conv1_weights_tz}, dt::f32, tag::oihw}, eng);
    write_to_dnnl_memory(kernels.data(), user_weights_memory);
    auto conv1_user_bias_memory
        = memory({{conv1_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(biases.data(), conv1_user_bias_memory);
    auto user_dst_memory = memory({{conv1_dst_tz}, dt::f32, tag::nchw}, eng);

    auto conv1_src_md = memory::desc({conv1_src_tz}, dt::f32, tag::any);
    auto conv1_bias_md = memory::desc({conv1_bias_tz}, dt::f32, tag::any);
    auto conv1_weights_md = memory::desc({conv1_weights_tz}, dt::f32, tag::any);
    auto conv1_dst_md = memory::desc({conv1_dst_tz}, dt::f32, tag::any);

    auto conv1_desc = convolution_forward::desc(prop_kind::forward_inference,
                                                algorithm::convolution_direct, conv1_src_md, conv1_weights_md,
                                                conv1_bias_md, conv1_dst_md, conv1_strides, conv1_padding,
                                                conv1_padding);

    auto conv1_prim_desc = convolution_forward::primitive_desc(conv1_desc, eng);
    auto conv1_src_memory = user_src_memory;

    // reorder src to HW specific format
    if (conv1_prim_desc.src_desc() != user_src_memory.get_desc()) {
        conv1_src_memory = memory(conv1_prim_desc.src_desc(), eng);
        net.push_back(reorder(user_src_memory, conv1_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, user_src_memory},
                {DNNL_ARG_TO, conv1_src_memory}});
    }

    // reorder weights to HW specific format (offline)
    auto conv1_weights_memory = user_weights_memory;
    if (conv1_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
        conv1_weights_memory = memory(conv1_prim_desc.weights_desc(), eng);
        reorder(user_weights_memory, conv1_weights_memory)
                .execute(s, user_weights_memory, conv1_weights_memory);
    }

    // create HW specific buffer for dst
    auto conv1_dst_memory = memory(conv1_prim_desc.dst_desc(), eng);

    // run convolution
    std::cout << "creating network " << std::endl;
    net.push_back(convolution_forward(conv1_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv1_src_memory},
                        {DNNL_ARG_WEIGHTS, conv1_weights_memory},
                        {DNNL_ARG_BIAS, conv1_user_bias_memory},
                        {DNNL_ARG_DST, conv1_dst_memory}});

    // reorder dst to HW specific format
    net.push_back(reorder(conv1_dst_memory, user_dst_memory));
    net_args.push_back({{DNNL_ARG_FROM, conv1_dst_memory},
            {DNNL_ARG_TO, user_dst_memory}});

    int times = 1000;
    assert(net.size() == net_args.size() && "something is missing");
    std::vector<float> outputs(num_filters * out_h * out_w);
    for(int j = 0; j < times; ++j)
    {
        for (size_t i = 0; i < net.size(); ++i)
            net.at(i).execute(s, net_args.at(i));
        read_from_dnnl_memory(outputs.data(), user_dst_memory);
        for (int i = 0; i < outputs.size(); i++)
            assert(((outputs[i] - targets[i]) < 1e-9));

    }

    s.wait();

    std::cout
        << "model run successfully about "
        << times
        << " times"
        << std::endl;

}

int main(int argc, char *argv[])
{
    run_network();
    return 0;
}
