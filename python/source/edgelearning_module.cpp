/***************************************************************************
 *            data_module.cpp
 *
 *  Copyright  2007-20  Mirco De Marchi
 *
 ****************************************************************************/

/*
 *  This file is part of Ariadne.
 *
 *  Ariadne is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Ariadne is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Ariadne.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <pybind11/pybind11.h>

#include "data_submodule.hpp"
#include "parser_submodule.hpp"
#include "dnn_submodule.hpp"

#include "middleware/layer_descriptor.hpp"
#include "middleware/nn.hpp"
#include "middleware/fnn.hpp"

namespace EdgeLearning {

} // namespace EdgeLearning

namespace py = pybind11;

using namespace pybind11::literals;
using namespace EdgeLearning;

using PyNumType = NumType;

template<typename T = PyNumType>
class PyNN : public CompileNeuralNetwork<T> {
public:
    using CompileNeuralNetwork<T>::CompileNeuralNetwork;
    using PyNNDataset = Dataset<T>;

    PyNNDataset predict(PyNNDataset& data) override {
        PYBIND11_OVERRIDE_PURE(
            PyNNDataset, CompileNeuralNetwork<T>, predict, data
        );
    }

    void fit(PyNNDataset& data,
             SizeType epochs = 1,
             SizeType batch_size = 1,
             NumType learning_rate = 0.03,
             RneType::result_type seed = 0) override {
        PYBIND11_OVERRIDE_PURE(
            void, CompileNeuralNetwork<T>, fit, data, epochs, batch_size, learning_rate, seed
        );
    }

    void compile(LossType loss = LossType::MSE,
                 OptimizerType optimizer = OptimizerType::ADAM,
                 InitType init = InitType::AUTO) override {
        PYBIND11_OVERRIDE_PURE(
            void, CompileNeuralNetwork<T>, compile, loss, optimizer, init
        );
    }

    typename CompileNeuralNetwork<T>::EvaluationResult evaluate(Dataset<T>& data) override {
        PYBIND11_OVERRIDE_PURE(
            typename CompileNeuralNetwork<T>::EvaluationResult, CompileNeuralNetwork<T>, compile, data
        );
    }

    SizeType input_size() override {
        PYBIND11_OVERRIDE_PURE(
            SizeType, CompileNeuralNetwork<T>, input_size,
        );
    }

    SizeType output_size() override {
        PYBIND11_OVERRIDE_PURE(
            SizeType, CompileNeuralNetwork<T>, output_size,
        );
    }
};

template <
    Framework MM_F,
    LossType MM_LT,
    InitType MM_IT,
    ParallelizationLevel MM_PL,
    typename MM_T>
struct PyMapFeedforwardModel {
    using type = typename MapModel<MM_F, MM_LT, MM_IT, MM_PL, MM_T>::feedforward_model;
};

template<
    ParallelizationLevel PL = ParallelizationLevel::SEQUENTIAL,
    typename T = PyNumType>
using PyEdgeDynamicFeedforwardNeuralNetwork = DynamicNeuralNetwork<
    PyMapFeedforwardModel, Framework::EDGE_LEARNING, PL, T>;

#if ENABLE_MLPACK
template<typename T = PyNumType>
using PyMlpackDynamicFeedforwardNeuralNetwork = DynamicNeuralNetwork<
    PyMapFeedforwardModel, Framework::MLPACK, ParallelizationLevel::SEQUENTIAL, T>;
#endif

static void submodules(pybind11::module& m)
{
    auto parser_subm = m.def_submodule("parser");
    parser_submodule(parser_subm);

    auto data_subm = m.def_submodule("data");
    data_submodule(data_subm);

    auto dnn_subm = m.def_submodule("dnn");
    dnn_submodule(dnn_subm);
}

static void layer_descriptor_class(pybind11::module& m)
{
    auto conv_layer_setting = [&](SizeType n_filters, py::tuple kernel_shape,
        py::tuple stride = py::make_tuple(1), py::tuple padding = py::make_tuple(0)) {
        return LayerSetting(
            n_filters,
            get_shape2d(kernel_shape),
            get_shape2d(stride),
            get_shape2d(padding));
    };

    auto pool_layer_setting = [&](py::tuple kernel_shape, py::tuple stride = py::make_tuple(1)){
        return LayerSetting(
            get_shape2d(kernel_shape),
            get_shape2d(stride));
    };

    py::class_<LayerSetting> layer_setting_class(m, "LayerSetting");
    layer_setting_class.def(py::init<>());
    layer_setting_class.def(py::init<LayerShape>());
    layer_setting_class.def(py::init([](std::vector<py::tuple> shapes){
        return get_layer_shape_from(shapes);
    }));
    layer_setting_class.def(py::init([](py::tuple shape){
        return get_layer_shape_from(shape);
    }));
    layer_setting_class.def(py::init<SizeType, DLMath::Shape2d, DLMath::Shape2d, DLMath::Shape2d>());
    layer_setting_class.def(py::init(conv_layer_setting));
    layer_setting_class.def(py::init<DLMath::Shape2d, DLMath::Shape2d>());
    layer_setting_class.def(py::init(pool_layer_setting));
    layer_setting_class.def(py::init<NumType>());
    layer_setting_class.def_property(
        "units",
        [](const LayerSetting& ls) { return ls.units(); },
        [](LayerSetting& ls, LayerShape shape) { ls.units(shape); });
    layer_setting_class.def_property(
        "n_filters",
        [](const LayerSetting& ls) { return ls.n_filters(); },
        [](LayerSetting& ls, SizeType n_filters) { ls.n_filters(n_filters); });
    layer_setting_class.def_property(
        "kernel_shape",
        [](const LayerSetting& ls) { return ls.kernel_shape(); },
        [](LayerSetting& ls, const DLMath::Shape2d& kernel_shape) { ls.kernel_shape(kernel_shape); });
    layer_setting_class.def_property(
        "strides",
        [](const LayerSetting& ls) { return ls.stride(); },
        [](LayerSetting& ls, const DLMath::Shape2d& stride) { ls.stride(stride); });
    layer_setting_class.def_property(
        "padding",
        [](const LayerSetting& ls) { return ls.padding(); },
        [](LayerSetting& ls, const DLMath::Shape2d& padding) { ls.padding(padding); });
    layer_setting_class.def_property(
        "drop_probability",
        [](const LayerSetting& ls) { return ls.drop_probability(); },
        [](LayerSetting& ls, NumType drop_probability) { ls.drop_probability(drop_probability); });

    py::class_<LayerDescriptor> layer_descriptor_class(m, "LayerDescriptor");
    layer_descriptor_class.def(py::init<std::string, LayerType, LayerSetting, ActivationType>());
    layer_descriptor_class.def_property(
        "name",
        [](const LayerDescriptor& ld) { return ld.name(); },
        [](LayerDescriptor& ld, const std::string& name) { ld.name(name); });
    layer_descriptor_class.def_property(
        "type",
        [](const LayerDescriptor& ld) { return ld.type(); },
        [](LayerDescriptor& ld, LayerType type) { ld.type(type); });
    layer_descriptor_class.def_property(
        "setting",
        [](const LayerDescriptor& ld) { return ld.setting(); },
        [](LayerDescriptor& ld, const LayerSetting& setting) { ld.setting(setting); });
    layer_descriptor_class.def_property(
        "activation_type",
        [](const LayerDescriptor& ld) { return ld.activation_type(); },
        [](LayerDescriptor& ld, ActivationType activation_type) { ld.activation_type(activation_type); });

    py::class_<Input, LayerDescriptor> input_layer_descriptor_class(m, "Input");
    input_layer_descriptor_class.def(py::init<std::string, LayerShape>());
    input_layer_descriptor_class.def(py::init<std::string, SizeType>());
    input_layer_descriptor_class.def(py::init(
        [](std::string name, std::vector<py::tuple> shapes) {
            return Input(name, get_layer_shape_from(shapes));
        }));
    input_layer_descriptor_class.def(py::init(
        [](std::string name, py::tuple shape) {
            return Input(name, get_layer_shape_from(shape));
        }));

    py::class_<Dense, LayerDescriptor> dense_layer_descriptor_class(m, "Dense");
    dense_layer_descriptor_class.def(py::init<std::string, SizeType, ActivationType>());

    py::class_<Conv, LayerDescriptor> conv_layer_descriptor_class(m, "Conv");
    conv_layer_descriptor_class.def(py::init<std::string, Conv::ConvSetting, ActivationType>());
    conv_layer_descriptor_class.def(py::init(
        [&](std::string name, py::tuple conv_setting, ActivationType activation_type)
        {
            auto layer_setting = conv_layer_setting(
                conv_setting[0].cast<SizeType>(),
                conv_setting[1].cast<py::tuple>(),
                conv_setting.size() > 2 ? conv_setting[2].cast<py::tuple>() : py::make_tuple(1),
                conv_setting.size() > 3 ? conv_setting[3].cast<py::tuple>() : py::make_tuple(0));
            return Conv{
                name,
                Conv::ConvSetting(
                    layer_setting.n_filters(),
                    layer_setting.kernel_shape(),
                    layer_setting.stride(),
                    layer_setting.padding()),
                activation_type
            };
        }));
    py::class_<Conv::ConvSetting> conv_setting(
        conv_layer_descriptor_class, "Setting");
    conv_setting.def(
        py::init<SizeType, DLMath::Shape2d, DLMath::Shape2d, DLMath::Shape2d>(),
        "filters"_a, "kernel_size"_a,
        "strides"_a=DLMath::Shape2d(1), "padding"_a=DLMath::Shape2d(0));
    conv_setting.def(py::init(
        [&](SizeType filters, py::tuple kernel_size, py::tuple strides, py::tuple padding) {
            return Conv::ConvSetting(
                filters,
                get_shape2d(kernel_size),
                get_shape2d(strides),
                get_shape2d(padding));
        }), "filters"_a, "kernel_size"_a, "strides"_a=py::make_tuple(1), "padding"_a=py::make_tuple(0));


    py::class_<MaxPool, LayerDescriptor> max_pool_layer_descriptor_class(m, "MaxPool");
    max_pool_layer_descriptor_class.def(py::init<std::string, MaxPool::MaxPoolSetting, ActivationType>());
    max_pool_layer_descriptor_class.def(py::init(
        [&](std::string name, py::tuple pool_setting, ActivationType activation_type)
        {
            auto layer_setting = pool_layer_setting(
                pool_setting[0].cast<py::tuple>(),
                pool_setting.size() > 1 ? pool_setting[1].cast<py::tuple>() : py::make_tuple(1));
            return MaxPool{
                name,
                MaxPool::MaxPoolSetting(
                    layer_setting.kernel_shape(),
                    layer_setting.stride()),
                activation_type
            };
        }));
    py::class_<MaxPool::MaxPoolSetting> max_pool_setting(
        max_pool_layer_descriptor_class, "Setting");
    max_pool_setting.def(
        py::init<DLMath::Shape2d, DLMath::Shape2d>(),
        "kernel_size"_a, "strides"_a=DLMath::Shape2d(1));
    max_pool_setting.def(py::init(
        [&](py::tuple kernel_size, py::tuple strides) {
            return MaxPool::MaxPoolSetting(
                get_shape2d(kernel_size),
                get_shape2d(strides));
        }), "kernel_size"_a, "strides"_a=py::make_tuple(1));

    py::class_<AvgPool, LayerDescriptor> avg_pool_layer_descriptor_class(m, "AvgPool");
    avg_pool_layer_descriptor_class.def(py::init<std::string, AvgPool::AvgPoolSetting, ActivationType>());
    avg_pool_layer_descriptor_class.def(py::init(
        [&](std::string name, py::tuple pool_setting, ActivationType activation_type)
        {
            auto layer_setting = pool_layer_setting(
                pool_setting[0].cast<py::tuple>(),
                pool_setting.size() > 1 ? pool_setting[1].cast<py::tuple>() : py::make_tuple(1));
            return AvgPool{
                name,
                AvgPool::AvgPoolSetting(
                    layer_setting.kernel_shape(),
                    layer_setting.stride()),
                activation_type
            };
        }));
    py::class_<AvgPool::AvgPoolSetting> avg_pool_setting(
        avg_pool_layer_descriptor_class, "Setting");
    avg_pool_setting.def(
        py::init<DLMath::Shape2d, DLMath::Shape2d>(),
        "kernel_size"_a, "strides"_a=DLMath::Shape2d(1));
    avg_pool_setting.def(py::init(
        [&](py::tuple kernel_size, py::tuple strides) {
            return AvgPool::AvgPoolSetting(
                get_shape2d(kernel_size),
                get_shape2d(strides));
        }), "kernel_size"_a, "strides"_a=py::make_tuple(1));

    py::class_<Dropout, LayerDescriptor> dropout_layer_descriptor_class(m, "Dropout");
    dropout_layer_descriptor_class.def(py::init<std::string, Dropout::DropoutSetting, ActivationType>());
    dropout_layer_descriptor_class.def(py::init<std::string, NumType, ActivationType>());
    py::class_<Dropout::DropoutSetting> dropout_setting(
        dropout_layer_descriptor_class, "Setting");
    dropout_setting.def(py::init<NumType>());

}

static void nn_class(pybind11::module& m)
{
    py::enum_<Framework> framework_enum(m, "Framework");
    framework_enum.value("EDGE_LEARNING", Framework::EDGE_LEARNING);
#if ENABLE_MLPACK
    framework_enum.value("MLPACK", Framework::MLPACK);
#endif
    framework_enum.export_values();

    py::enum_<ParallelizationLevel> parallelization_level_enum(m, "ParallelizationLevel");
    parallelization_level_enum.value("SEQUENTIAL", ParallelizationLevel::SEQUENTIAL);
    parallelization_level_enum.value("THREAD_PARALLELISM_ON_DATA_ENTRY", ParallelizationLevel::THREAD_PARALLELISM_ON_DATA_ENTRY);
    parallelization_level_enum.value("THREAD_PARALLELISM_ON_DATA_BATCH", ParallelizationLevel::THREAD_PARALLELISM_ON_DATA_BATCH);
    parallelization_level_enum.export_values();

    py::enum_<LayerType> layer_type_enum(m, "LayerType");
    layer_type_enum.value("DENSE", LayerType::Dense);
    layer_type_enum.value("CONV", LayerType::Conv);
    layer_type_enum.value("MAX_POOL", LayerType::MaxPool);
    layer_type_enum.value("AVG_POOL", LayerType::AvgPool);
    layer_type_enum.value("DROPOUT", LayerType::Dropout);
    layer_type_enum.value("INPUT", LayerType::Input);
    layer_type_enum.export_values();

    py::enum_<ActivationType> activation_type_enum(m, "ActivationType");
    activation_type_enum.value("RELU", ActivationType::ReLU);
    activation_type_enum.value("ELU", ActivationType::ELU);
    activation_type_enum.value("SOFTMAX", ActivationType::Softmax);
    activation_type_enum.value("TANH", ActivationType::TanH);
    activation_type_enum.value("SIGMOID", ActivationType::Sigmoid);
    activation_type_enum.value("LINEAR", ActivationType::Linear);
    activation_type_enum.value("NONE", ActivationType::None);
    activation_type_enum.export_values();

    py::enum_<LossType> loss_type_enum(m, "LossType");
    loss_type_enum.value("CCE", LossType::CCE);
    loss_type_enum.value("MSE", LossType::MSE);
    loss_type_enum.export_values();

    py::enum_<OptimizerType> optimizer_type_enum(m, "OptimizerType");
    optimizer_type_enum.value("GRADIENT_DESCENT", OptimizerType::GRADIENT_DESCENT);
    optimizer_type_enum.value("ADAM", OptimizerType::ADAM);
    optimizer_type_enum.export_values();

    py::enum_<InitType> init_type_enum(m, "InitType");
    init_type_enum.value("HE", InitType::HE_INIT);
    init_type_enum.value("XAVIER", InitType::XAVIER_INIT);
    init_type_enum.value("AUTO", InitType::AUTO);
    init_type_enum.export_values();

    py::class_<CompileNeuralNetwork<PyNumType>, PyNN<PyNumType>> nn_class(m, "NN");
    nn_class.def(py::init<std::string>());
    nn_class.def("compile", &CompileNeuralNetwork<PyNumType>::compile,
                 "loss"_a=LossType::MSE, "optimizer"_a=OptimizerType::ADAM, "init"_a=InitType::AUTO);
    nn_class.def("predict", &CompileNeuralNetwork<PyNumType>::predict, "data"_a);
    nn_class.def("fit", &CompileNeuralNetwork<PyNumType>::fit,
                 "data"_a, "epochs"_a=1, "batch_size"_a=1, "learning_rate"_a=0.03, "seed"_a=0);
    nn_class.def("evaluate", &CompileNeuralNetwork<PyNumType>::evaluate,
                 "data"_a);
    nn_class.def("input_size", &CompileNeuralNetwork<PyNumType>::input_size);
    nn_class.def("output_size", &CompileNeuralNetwork<PyNumType>::output_size);

    py::class_<CompileNeuralNetwork<PyNumType>::EvaluationResult> nn_evaluation_result(
        nn_class, "EvaluationResult");
    nn_evaluation_result.def(py::init<>());
    nn_evaluation_result.def(py::init<PyNumType, PyNumType>());
    nn_evaluation_result.def_readwrite("loss", &CompileNeuralNetwork<PyNumType>::EvaluationResult::loss);
    nn_evaluation_result.def_readwrite("accuracy", &CompileNeuralNetwork<PyNumType>::EvaluationResult::accuracy);
    nn_evaluation_result.def_readwrite("accuracy_perc", &CompileNeuralNetwork<PyNumType>::EvaluationResult::accuracy_perc);
    nn_evaluation_result.def_readwrite("error_rate", &CompileNeuralNetwork<PyNumType>::EvaluationResult::error_rate);
    nn_evaluation_result.def_readwrite("error_rate_perc", &CompileNeuralNetwork<PyNumType>::EvaluationResult::error_rate_perc);
}

static void fnn_class(pybind11::module& m)
{
    using EdgeFNN = PyEdgeDynamicFeedforwardNeuralNetwork<ParallelizationLevel::SEQUENTIAL, PyNumType>;
    py::class_<EdgeFNN, CompileNeuralNetwork<PyNumType>> edge_fnn_class(m, "EdgeFNN");
    edge_fnn_class.def(py::init<NeuralNetworkDescriptor, std::string>());
    edge_fnn_class.def("compile", &EdgeFNN::compile,
                       "loss"_a=LossType::MSE, "optimizer"_a=OptimizerType::ADAM, "init"_a=InitType::AUTO);
    edge_fnn_class.def("predict", &EdgeFNN::predict, "data"_a);
    edge_fnn_class.def("fit", &EdgeFNN::fit,
                       "data"_a, "epochs"_a=1, "batch_size"_a=1, "learning_rate"_a=0.03, "seed"_a=0);
    edge_fnn_class.def("evaluate", &EdgeFNN::evaluate, "data"_a);
    edge_fnn_class.def("input_size", &EdgeFNN::input_size);
    edge_fnn_class.def("output_size", &EdgeFNN::output_size);

#if ENABLE_MLPACK
    using MlpackFNN = PyMlpackDynamicFeedforwardNeuralNetwork<PyNumType>;
    py::class_<MlpackFNN, CompileNeuralNetwork<PyNumType>> mlpack_fnn_class(m, "MlpackFNN");
    mlpack_fnn_class.def(py::init<NeuralNetworkDescriptor, std::string>());
    mlpack_fnn_class.def("compile", &MlpackFNN::compile,
                         "loss"_a=LossType::MSE, "optimizer"_a=OptimizerType::ADAM, "init"_a=InitType::AUTO);
    mlpack_fnn_class.def("predict", &MlpackFNN::predict, "data"_a);
    mlpack_fnn_class.def("fit", &MlpackFNN::fit,
                         "data"_a, "epochs"_a=1, "batch_size"_a=1, "learning_rate"_a=0.03, "seed"_a=0);
    mlpack_fnn_class.def("evaluate", &MlpackFNN::evaluate, "data"_a);
    mlpack_fnn_class.def("input_size", &MlpackFNN::input_size);
    mlpack_fnn_class.def("output_size", &MlpackFNN::output_size);

    m.attr("FNN") = mlpack_fnn_class;
#else
    m.attr("FNN") = edge_fnn_class;
#endif

}

static void middleware(pybind11::module& m)
{
    layer_descriptor_class(m);
    nn_class(m);
    fnn_class(m);
}

PYBIND11_MODULE(pyedgelearning, m) {
    m.doc() = "Python EdgeLearning core module";

    submodules(m);
    middleware(m);
}