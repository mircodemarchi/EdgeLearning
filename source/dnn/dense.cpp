/***************************************************************************
 *            dnn/dense.cpp
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

#include "dense.hpp"

#include "dlmath.hpp"

#include "concmanager/include/task_manager.hpp"

#include <algorithm>
#include <cstdio>

namespace EdgeLearning {

const std::string DenseLayer::TYPE = "Dense";

DenseLayer::DenseLayer(std::string name,
    SizeType input_size, SizeType output_size)
    : FeedforwardLayer(input_size, output_size,
                       std::move(name), "dense_layer_")
{
    // std::cout << _name << ": " << input_size
    //    << " -> " << output_size << std::endl;

    // The weight parameters of a FF-layer are an NxM matrix.
    _weights.resize(output_size * input_size);

    // Each node in this layer is assigned a bias.
    _biases.resize(output_size);

    _weight_gradients.resize(output_size * input_size);
    _bias_gradients.resize(output_size);
}

void DenseLayer::init(InitializationFunction init,
                      ProbabilityDensityFunction pdf,
                      RneType rne)
{
    /*
     * The C++ standard does not guarantee that the results obtained from a 
     * distribution function will be identical given the same inputs across 
     * different compilers and platforms, therefore I use my own 
     * distributions to provide deterministic results.
     */
    auto dist = DLMath::initialization_pdf<NumType>(init, pdf, input_size());

    for (NumType& w: _weights)
    {
        w = dist(rne);
    }

    /*
     * Setting biases to zero is a common practice, as is initializing the 
     * bias to a small value (e.g. on the order of 0.01). The thinking is
     * that a non-zero bias will ensure that the neuron always "fires" at 
     * the beginning to produce a signal.
     */
    for (NumType& b: _biases)
    {
        b = 0.01; ///< You can try also with 0.0 or other strategies.
    }
}

const std::vector<NumType>& DenseLayer::forward(
    const std::vector<NumType>& inputs)
{
    SizeType in_size = inputs.size();
    SizeType out_size = _output_activations.size();

    // std::cout << "[" << _name << "] forward" << std::endl;

    /* 
     * Compute the product of the input data with the weight add the bias.
     * z = W * x + b
     */
#if 0 // Enable thread optimization.
    auto& tm = ConcManager::TaskManager::instance();
    tm.set_maximum_concurrency();
    std::vector<ConcManager::Future<void>> futures;
    for (SizeType i = 0; i < out_size; ++i)
    {
        futures.push_back(tm.enqueue(
            [&](SizeType out_idx) {
                _output_activations[out_idx] = _biases[out_idx];
                for (SizeType j = 0; j < in_size; ++j)
                {
                    _output_activations[out_idx] += _weights[(out_idx * in_size) + j] * inputs[j];
                }
            }, i));
    }
    for (auto& f: futures) f.get();
#else
    for (SizeType i = 0; i < out_size; ++i)
    {
        _output_activations[i] = _biases[i];
        for (SizeType j = 0; j < in_size; ++j)
        {
            _output_activations[i] += _weights[(i * in_size) + j] * inputs[j];
        }
    }
#endif
    return FeedforwardLayer::forward(_output_activations);
}

const std::vector<NumType>& DenseLayer::backward(
    const std::vector<NumType>& gradients)
{
    SizeType out_size = gradients.size();
    SizeType in_size = _shared_fields->input_size();

#if 0 // Enable thread optimization.
    auto& tm = ConcManager::TaskManager::instance();
    tm.set_maximum_concurrency();
    std::vector<ConcManager::Future<void>> futures;
    for (SizeType i = 0; i < out_size; ++i)
    {
        futures.push_back(tm.enqueue(
            [&](SizeType out_idx) {
                _bias_gradients[out_idx] += gradients[out_idx];
                for (SizeType j = 0; j < in_size; ++j)
                {
                    _weight_gradients[(out_idx * in_size) + j] +=
                        gradients[out_idx] * _last_input[j];
                    _input_gradients[j] +=
                        gradients[out_idx] * _weights[(out_idx * in_size) + j];
                }
            }, i));
    }
    for (auto& f: futures) f.get();
#else
    std::fill(_input_gradients.begin(), _input_gradients.end(), 0);
    for (SizeType i = 0; i < out_size; ++i)
    {
        _bias_gradients[i] += gradients[i];
        for (SizeType j = 0; j < in_size; ++j)
        {
            _weight_gradients[(i * in_size) + j] +=
                gradients[i] * _last_input[j];
            _input_gradients[j] +=
                gradients[i] * _weights[(i * in_size) + j];
        }
    }
#endif

    return FeedforwardLayer::backward(_input_gradients);;
}

NumType& DenseLayer::param(SizeType index)
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

NumType& DenseLayer::gradient(SizeType index)
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

void DenseLayer::print() const 
{
    std::cout << _shared_fields->name() << std::endl;
    std::cout << "Weights (" << output_size() << " x " << input_size() << ")"
        << std::endl;
    for (SizeType i = 0; i < output_size(); ++i)
    {
        SizeType offset = i * input_size();
        for (SizeType j = 0; j < input_size(); ++j)
        {
            std::cout << "\t[" << (offset + j) << "]" << _weights[offset + j]
                << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << "Biases (" << output_size() << " x 1)" << std::endl;
    for (SizeType i = 0; i < output_size(); ++i)
    {
        std::cout << "\t" << _biases[i] << std::endl;
    }
    std::cout << std::endl;
}

Json DenseLayer::dump() const
{
    Json out = FeedforwardLayer::dump();

    Json weights;
    for (SizeType i = 0; i < output_size(); ++i)
    {
        SizeType offset = i * input_size();

        Json weights_row;
        for (SizeType j = 0; j < input_size(); ++j)
        {
            weights_row.append(_weights[offset + j]);
        }
        weights.append(weights_row);
    }

    Json biases;
    for (SizeType i = 0; i < output_size(); ++i)
    {
        biases.append(_biases[i]);
    }

    out[dump_fields.at(DumpFields::WEIGHTS)] = weights;
    out[dump_fields.at(DumpFields::BIASES)] = biases;
    return out;
}

void DenseLayer::load(const Json& in)
{
    FeedforwardLayer::load(in);

    _weights.resize(output_size() * input_size());
    _biases.resize(output_size());
    _weight_gradients.resize(output_size() * input_size());
    _bias_gradients.resize(output_size());

    for (SizeType i = 0; i < output_size(); ++i)
    {
        for (SizeType j = 0; j < input_size(); ++j)
        {
            _weights[i * input_size() + j] = in.at(
                dump_fields.at(DumpFields::WEIGHTS)).at(i).at(j);
        }
    }

    for (SizeType i = 0; i < output_size(); ++i)
    {
        _biases[i] = in.at(dump_fields.at(DumpFields::BIASES)).at(i);
    }
}

void DenseLayer::_set_input_shape(LayerShape input_shape)
{
    FeedforwardLayer::_set_input_shape(input_shape);
    _weights.resize(output_size() * input_shape.size());
    _weight_gradients.resize(output_size() * input_shape.size());
}

} // namespace EdgeLearning
