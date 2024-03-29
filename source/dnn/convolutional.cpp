/***************************************************************************
 *            dnn/convolutional.cpp
 *
 *  Copyright  2021  Mirco De Marchi
 *
 ****************************************************************************/

/*
 *  This file is part of EdgeLearning.
 *
 *  EdgeLearning is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  EdgeLearning is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with EdgeLearning.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "convolutional.hpp"

#include <algorithm>
#include <cstdio>
#include <utility>

namespace EdgeLearning {

static inline SizeType convolutional_output_side(
    SizeType input_side, SizeType kernel_side,
    SizeType stride, SizeType padding
    )
{
    return input_side == 0 ?
        0 : ((input_side - kernel_side + (2 * padding)) / stride) + 1ULL;
}

static inline DLMath::Shape3d convolutional_output_shape(
    DLMath::Shape3d input_shape, DLMath::Shape2d kernel_shape,
    DLMath::Shape2d stride, DLMath::Shape2d padding, SizeType n_filters
)
{
    return {
        convolutional_output_side(input_shape.height(), kernel_shape.height(),
                                  stride.height(), padding.height()),
        convolutional_output_side(input_shape.width(), kernel_shape.width(),
                                  stride.width(), padding.width()),
        n_filters};
}

const std::string ConvolutionalLayer::TYPE = "Conv";

ConvolutionalLayer::ConvolutionalLayer(std::string name,
    DLMath::Shape3d input_shape, DLMath::Shape2d kernel_shape,
    SizeType n_filters, DLMath::Shape2d stride, DLMath::Shape2d padding)
    : FeedforwardLayer(input_shape,
        convolutional_output_shape(input_shape, kernel_shape,
                                  stride, padding, n_filters),
        std::move(name), "convolutional_layer_")
    , _kernel_shape(kernel_shape)
    , _n_filters(n_filters)
    , _stride(stride)
    , _padding(padding)
{
    // The weight parameters are composed by n_filters of kernel size.
    _weights.resize(_kernel_shape.size() * input_shape.channels() * n_filters);

    // The bias is incremented to the result of each filter.
    _biases.resize(n_filters);

    _weight_gradients.resize(_kernel_shape.size() * input_shape.channels()
                             * n_filters);
    _bias_gradients.resize(n_filters);
}

void ConvolutionalLayer::init(InitializationFunction init,
                              ProbabilityDensityFunction pdf,
                              RneType rne)
{
    auto dist = DLMath::initialization_pdf<NumType>(init, pdf, input_size());

    for (NumType& w: _weights)
    {
        w = dist(rne);
    }

    for (NumType& b: _biases)
    {
        b = 0.01;
    }
}

const std::vector<NumType>& ConvolutionalLayer::forward(
    const std::vector<NumType>& inputs)
{
    /*
     * Perform convolution with n_filters of kernel size contained in
     * _weights vector on the input 3D matrix.
     */
    DLMath::cross_correlation<NumType>(
        _output_activations.data(), inputs.data(),
        _shared_fields->input_shape().shape(),
        _weights.data(), _kernel_shape, _n_filters, _stride, _padding);

    return FeedforwardLayer::forward(_output_activations);
}

const std::vector<NumType>& ConvolutionalLayer::backward(
    const std::vector<NumType>& gradients)
{
    /*
     * Bias gradient. Calculate dJ/db = dJ/dz.
     *
     * Shape of gradients: out_height * out_width * n_filters.
     * Shape of bias: n_filters.
     */
    for (SizeType f = 0; f < _n_filters; ++f)
    {
        NumType sum = 0;
        for (SizeType r = 0; r < _shared_fields->output_shape().height(); ++r)
        {
            auto step = _shared_fields->output_shape().width() * _n_filters;
            for (SizeType c = 0; c < _shared_fields->output_shape().width(); ++c)
            {
                sum += gradients[r * step + c * _n_filters + f];
            }
        }
        _bias_gradients[f] = sum;
    }

    /*
     * Weight gradient. Calculate dJ/dw_i_j = dJ/dz * x_j.
     * Input gradient. Calculate dJ/dx = dJ/dz * W.
     *
     * Shape of gradients: out_height * out_width * n_filters.
     * Shape of input: in_height * in_width * channels.
     * Shape of input gradients: in_height * in_width * channels.
     * Shape of kernel: k_height * k_width * channels.
     * Shape of weight: k_height * k_width * channels * n_filters.
     * Shape of weight gradients: k_height * k_width * channels * n_filters.
     */
    std::fill(_input_gradients.begin(), _input_gradients.end(), 0);
    auto gradients_op = [&](
        NumType* dst, DLMath::Shape2d dst_shape, DLMath::Coord2d dst_coord,
        const NumType* src, DLMath::Shape3d src_shape,
        const NumType* k, DLMath::Shape2d k_shape, SizeType n_filters,
        int64_t row, int64_t col)
    {
        (void) dst;
        auto k_size = k_shape.size() * src_shape.channels();
        auto k_step = k_shape.width() * src_shape.channels();
        auto src_step = src_shape.width() * src_shape.channels();
        for (SizeType f = 0; f < n_filters; ++f)
        {
            auto output_gradient = gradients[
                dst_coord.row * dst_shape.width() * n_filters
                + dst_coord.col * n_filters + f];
            for (SizeType k_i = 0; k_i < k_size; ++k_i)
            {
                auto row_k = k_i / k_step;
                auto col_k = k_i % k_step;
                auto row_src = row + static_cast<int64_t>(row_k);
                auto col_src = col + static_cast<int64_t>(col_k);
                if (col_src < 0 || row_src < 0 ||
                    col_src >= static_cast<int64_t>(src_step) ||
                    row_src >= static_cast<int64_t>(src_shape.height()))
                {
                    continue; //< zero-padding.
                }
                _input_gradients[static_cast<std::size_t>(
                    row_src * static_cast<int64_t>(src_step)
                    + col_src)] += k[k_i * n_filters + f] * output_gradient;
                _weight_gradients[k_i * n_filters + f]
                    += src[row_src * static_cast<int64_t>(src_step) + col_src]
                        * output_gradient;
            }
        }
    };
    DLMath::kernel_slide<NumType>(
        gradients_op, nullptr, _last_input,
        _shared_fields->input_shape().shape(),
        _weights.data(), _kernel_shape,
        _n_filters, _stride, _padding);

    return FeedforwardLayer::backward(_input_gradients);
}

NumType& ConvolutionalLayer::param(SizeType index)
{
    if (index >= param_count())
    {
        throw std::runtime_error("index overflow");
    }
    if (index < _weights.size())
    {
        return _weights[index];
    }
    return _biases[index - _weights.size()];
}

NumType& ConvolutionalLayer::gradient(SizeType index)
{
    if (index >= param_count())
    {
        throw std::runtime_error("index overflow");
    }
    if (index < _weight_gradients.size())
    {
        return _weight_gradients[index];
    }
    return _bias_gradients[index - _weight_gradients.size()];
}

void ConvolutionalLayer::print() const
{
    std::cout << _shared_fields->name() << std::endl;
    std::cout << "Weights ("
        << _kernel_shape.height() << " x "
        << _kernel_shape.width()  << " x "
        << _shared_fields->input_shape().channels()  << " x "
        << _n_filters << ")" << std::endl;

    for (SizeType r = 0; r < _kernel_shape.height(); ++r)
    {
        SizeType r_offset = r * _kernel_shape.width()
                          * _shared_fields->input_shape().channels()
                          * _n_filters;
        for (SizeType c = 0; c < _kernel_shape.width(); ++c)
        {
            SizeType c_offset = c * _shared_fields->input_shape().channels()
                              * _n_filters;
            for (SizeType ch = 0;
                 ch < _shared_fields->input_shape().channels();
                 ++ch)
            {
                SizeType ch_offset = ch * _n_filters;
                std::cout << "\t[" << r << "," << c << "," << ch << ",0:"
                          << _n_filters << "]" << std::endl;
                for (SizeType f = 0; f < _n_filters - 1; ++f)
                {
                    std::cout << _weights[r_offset + c_offset + ch_offset + f]
                        << ", ";
                }
                std::cout << _weights[r_offset + c_offset + ch_offset
                                      + _n_filters - 1] << std::endl;
            }
        }
    }
    std::cout << "Biases (1 x 1 x " << _n_filters << ")" << std::endl;
    for (SizeType i = 0; i < _n_filters; ++i)
    {
        std::cout << "\t" << _biases[i] << std::endl;
    }
    std::cout << std::endl;
}

Json ConvolutionalLayer::dump() const
{
    Json out = FeedforwardLayer::dump();

    Json weights;
    for (SizeType r = 0; r < _kernel_shape.height(); ++r)
    {
        SizeType r_offset = r * _kernel_shape.width()
                          * _shared_fields->input_shape().channels()
                          * _n_filters;
        Json weights_row;
        for (SizeType c = 0; c < _kernel_shape.width(); ++c)
        {
            SizeType c_offset = c * _shared_fields->input_shape().channels()
                              * _n_filters;
            Json weights_col;
            for (SizeType ch = 0;
                 ch < _shared_fields->input_shape().channels();
                 ++ch)
            {
                SizeType ch_offset = ch * _n_filters;
                Json weights_channel;
                for (SizeType f = 0; f < _n_filters; ++f)
                {
                    weights_channel.append(
                        _weights[r_offset + c_offset + ch_offset + f]);
                }
                weights_col.append(weights_channel);
            }
            weights_row.append(weights_col);
        }
        weights.append(weights_row);
    }

    Json biases;
    for (SizeType i = 0; i < _n_filters; ++i)
    {
        biases.append(_biases[i]);
    }

    Json others;
    std::vector<std::size_t> kernel_size = {
        _kernel_shape.height(), _kernel_shape.width()
    };
    others["kernel_size"] = Json(kernel_size);
    others["n_filters"] = _n_filters;
    std::vector<std::size_t> stride = { _stride.height(), _stride.width() };
    others["stride"] = Json(stride);
    std::vector<std::size_t> padding = { _padding.height(), _padding.width() };
    others["padding"] = Json(padding);

    out[dump_fields.at(DumpFields::WEIGHTS)] = weights;
    out[dump_fields.at(DumpFields::BIASES)] = biases;
    out[dump_fields.at(DumpFields::OTHERS)] = others;
    return out;
}

void ConvolutionalLayer::load(const Json& in)
{
    FeedforwardLayer::load(in);

    auto kernel_size = in.at(dump_fields.at(DumpFields::OTHERS)).at("kernel_size")
        .as_vec<std::size_t>();
    _kernel_shape = DLMath::Shape2d(kernel_size.at(0), kernel_size.at(1));
    _n_filters = in.at(dump_fields.at(DumpFields::OTHERS)).at("n_filters")
        .as<SizeType>();
    auto stride = in.at(dump_fields.at(DumpFields::OTHERS)).at("stride")
        .as_vec<std::size_t>();
    _stride = DLMath::Shape2d(stride.at(0), stride.at(1));
    auto padding = in.at(dump_fields.at(DumpFields::OTHERS)).at("padding")
        .as_vec<std::size_t>();
    _padding = DLMath::Shape2d(padding.at(0), padding.at(1));

    _weights.resize(_kernel_shape.size()
                    * _shared_fields->input_shape().channels()
                    * _n_filters);
    _biases.resize(_n_filters);
    _weight_gradients.resize(_kernel_shape.size()
                             * _shared_fields->input_shape().channels()
                             * _n_filters);
    _bias_gradients.resize(_n_filters);

    for (SizeType r = 0; r < _kernel_shape.height(); ++r)
    {
        SizeType r_offset = r * _kernel_shape.width()
                          * _shared_fields->input_shape().channels()
                          * _n_filters;
        for (SizeType c = 0; c < _kernel_shape.width(); ++c)
        {
            SizeType c_offset = c * _shared_fields->input_shape().channels()
                              * _n_filters;
            for (SizeType ch = 0;
                 ch < _shared_fields->input_shape().channels();
                 ++ch)
            {
                SizeType ch_offset = ch * _n_filters;
                for (SizeType f = 0; f < _n_filters; ++f)
                {
                    _weights[r_offset + c_offset + ch_offset + f] = in.at(
                        dump_fields.at(DumpFields::WEIGHTS))
                            .at(r).at(c).at(ch).at(f);
                }
            }
        }
    }

    for (SizeType i = 0; i < _n_filters; ++i)
    {
        _biases[i] = in.at(dump_fields.at(DumpFields::BIASES)).at(i);
    }
}

DLMath::Shape3d ConvolutionalLayer::calculate_output_shape(
    DLMath::Shape3d input_shape, DLMath::Shape2d kernel_shape,
    DLMath::Shape2d stride, DLMath::Shape2d padding, SizeType n_filters)
{
    return convolutional_output_shape(
        input_shape, kernel_shape, stride, padding, n_filters);
}

void ConvolutionalLayer::_set_input_shape(LayerShape input_shape)
{
    FeedforwardLayer::_set_input_shape(input_shape);
    _weights.resize(_kernel_shape.size() * input_shape.shape().channels() * _n_filters);
    _weight_gradients.resize(_kernel_shape.size() * input_shape.shape().channels()
                             * _n_filters);

    // Update input and output shape accordingly (see this constructor).
    _shared_fields->input_shape() = input_shape;
    _shared_fields->output_shape() = convolutional_output_shape(
        input_shape.shape(), _kernel_shape, _stride, _padding, _n_filters);

    // Update output size accordingly (see Layer and FeedforwardLayer constr.).
    _output_activations.resize(output_size());
}

} // namespace EdgeLearning
