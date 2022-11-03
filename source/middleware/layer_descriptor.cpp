/***************************************************************************
 *            middleware/layer_descriptor.cpp
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

#include "middleware/layer_descriptor.hpp"


namespace EdgeLearning {

LayerSetting::LayerSetting(LayerShape units)
    : _units(units)
    , _n_filters(0)
    , _kernel_shape(0)
    , _stride(0)
    , _padding(0)
    , _drop_probability(0.0)
{ }

LayerSetting::LayerSetting(SizeType n_filters,
                           DLMath::Shape2d kernel_shape,
                           DLMath::Shape2d stride,
                           DLMath::Shape2d padding)
    : _units(0)
    , _n_filters(n_filters)
    , _kernel_shape(kernel_shape)
    , _stride(stride)
    , _padding(padding)
    , _drop_probability(0.0)
{ }

LayerSetting::LayerSetting(DLMath::Shape2d kernel_shape,
                           DLMath::Shape2d stride)
    : _units(0)
    , _n_filters(0)
    , _kernel_shape(kernel_shape)
    , _stride(stride)
    , _padding(0)
    , _drop_probability(0.0)
{ }

LayerSetting::LayerSetting(NumType drop_probability)
    : _units(0)
    , _n_filters(0)
    , _kernel_shape(0)
    , _stride(0)
    , _padding(0)
    , _drop_probability(drop_probability)
{ }

LayerSetting::LayerSetting()
    : _units(0)
    , _n_filters(0)
    , _kernel_shape(0)
    , _stride(0)
    , _padding(0)
    , _drop_probability(0.0)
{ }

LayerShape LayerSetting::units() const
{ return _units; }
void LayerSetting::units(LayerShape units)
{ _units = units; }

SizeType LayerSetting::n_filters() const
{ return _n_filters; }
void LayerSetting::n_filters(SizeType n_filters)
{ _n_filters = n_filters; }

const DLMath::Shape2d& LayerSetting::kernel_shape() const
{ return _kernel_shape; }
void LayerSetting::kernel_shape(const DLMath::Shape2d &kernel_shape)
{ _kernel_shape = kernel_shape; }

const DLMath::Shape2d &LayerSetting::stride() const
{ return _stride; }
void LayerSetting::stride(const DLMath::Shape2d &stride)
{ _stride = stride; }

const DLMath::Shape2d &LayerSetting::padding() const
{ return _padding; }
void LayerSetting::padding(const DLMath::Shape2d &padding)
{ _padding = padding; }

NumType LayerSetting::drop_probability() const
{ return _drop_probability; }
void LayerSetting::drop_probability(NumType drop_probability)
{ _drop_probability = drop_probability; }

LayerDescriptor::LayerDescriptor(std::string name,
                                 LayerType type,
                                 LayerSetting setting,
                                 ActivationType activation_type)
    : _name(name)
    , _type(type)
    , _setting(setting)
    , _activation_type(activation_type)
{ }

const std::string& LayerDescriptor::name() const { return _name; }
void LayerDescriptor::name(const std::string& name) { _name = name; }

LayerType LayerDescriptor::type() const { return _type; }
void LayerDescriptor::type(LayerType type) { _type = type; }

const LayerSetting& LayerDescriptor::setting() const { return _setting; }
void LayerDescriptor::setting(const LayerSetting& setting) { _setting = setting; }

ActivationType LayerDescriptor::activation_type() const { return _activation_type; }
void LayerDescriptor::activation_type(ActivationType activation_type)
{ _activation_type = activation_type; }

Input::Input(std::string name, LayerShape input_size)
    : LayerDescriptor(
        name, LayerType::Input,
        LayerSetting(input_size))
{ }

Dense::Dense(std::string name,
             SizeType hidden_nodes,
             ActivationType activation_type)
    : LayerDescriptor(
        name, LayerType::Dense,
        LayerSetting(LayerShape(hidden_nodes)),
        activation_type)
{ }

Conv::Conv(std::string name,
           ConvSetting setting,
           ActivationType activation_type)
    : LayerDescriptor(
        name, LayerType::Conv,
        LayerSetting(setting.n_filters,
                     setting.kernel_shape,
                     setting.stride,
                     setting.padding),
        activation_type)
{ }

MaxPool::MaxPool(std::string name,
                 MaxPoolSetting setting,
                 ActivationType activation_type)
    : LayerDescriptor(
        name, LayerType::MaxPool,
        LayerSetting(setting.kernel_shape,
                     setting.stride),
        activation_type)
{ }

AvgPool::AvgPool(std::string name,
                 AvgPoolSetting setting,
                 ActivationType activation_type)
    : LayerDescriptor(
        name, LayerType::AvgPool,
        LayerSetting(setting.kernel_shape,
                     setting.stride),
        activation_type)
{ }

Dropout::Dropout(std::string name,
                 NumType drop_probability,
                 ActivationType activation_type)
    : LayerDescriptor(
        name, LayerType::Dropout,
        LayerSetting(drop_probability),
        activation_type)
{ }

} // namespace EdgeLearning
