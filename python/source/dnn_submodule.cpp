/***************************************************************************
 *            dnn_submodule.cpp
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

#include "dnn_submodule.hpp"

#include "dnn/layer.hpp"
#include "dnn/dlmath.hpp"

namespace EdgeLearning {

} // namespace EdgeLearning

using namespace EdgeLearning;


DLMath::Shape2d get_shape2d(py::tuple shape2d)
{
    if (shape2d.empty()) throw std::runtime_error("Empty tuple for shape2d");
    SizeType h = 1, w = 1;
    if (shape2d.size() >= 1) h = shape2d[0].cast<SizeType>();
    if (shape2d.size() >= 2) w = shape2d[1].cast<SizeType>();
    return {h, w};
}

DLMath::Shape3d get_shape3d(py::tuple shape3d)
{
    if (shape3d.empty()) throw std::runtime_error("Empty tuple for shape3d");
    if (shape3d.size() < 3) return get_shape2d(shape3d);
    return {
        shape3d[0].cast<SizeType>(),
        shape3d[1].cast<SizeType>(),
        shape3d[2].cast<SizeType>()
    };
}

LayerShape get_shape4d(py::tuple shape4d)
{
    if (shape4d.empty()) throw std::runtime_error("Empty tuple for layer shape");
    if (shape4d.size() < 4) return get_shape3d(shape4d);
    return {
        std::vector<DLMath::Shape3d>(
            shape4d[0].cast<SizeType>(),
            {
                shape4d[1].cast<SizeType>(),
                shape4d[2].cast<SizeType>(),
                shape4d[3].cast<SizeType>()
            }
        )
    };
}


LayerShape get_layer_shape_from(std::vector<py::tuple> shapes)
{
    std::vector<DLMath::Shape3d> ret;
    for (const auto& s: shapes)
    {
        ret.push_back(get_shape3d(s));
    }
    return ret;
}

LayerShape get_layer_shape_from(py::tuple shape)
{
    return get_shape4d(shape);
}

class PyLayer : public Layer {
public:
    using Layer::Layer;

    void init(InitializationFunction init_type = InitializationFunction::KAIMING,
              ProbabilityDensityFunction pdf = ProbabilityDensityFunction::NORMAL,
              RneType rne = RneType(std::random_device{}())) override {
        PYBIND11_OVERRIDE_PURE(
            void, Layer, init, init_type, pdf, rne
        );
    }

    const std::vector<NumType>& forward(const std::vector<NumType>& inputs) override {
        PYBIND11_OVERRIDE(
            const std::vector<NumType>&, Layer, forward, inputs
        );
    }

    const std::vector<NumType>& training_forward(const std::vector<NumType>& inputs) override {
        PYBIND11_OVERRIDE(
            const std::vector<NumType>&, Layer, training_forward, inputs
        );
    }

    const std::vector<NumType>& backward(const std::vector<NumType>& gradients) override {
        PYBIND11_OVERRIDE(
            const std::vector<NumType>&, Layer, backward, gradients
        );
    }

    inline const std::string& type() const override {
        PYBIND11_OVERRIDE(
            const std::string&, Layer, type,
        );
    }

    const std::vector<NumType>& last_input_gradient() override {
        PYBIND11_OVERRIDE_PURE(
            const std::vector<NumType>&, Layer, last_input_gradient,
        );
    }

    const std::vector<NumType>& last_output() override {
        PYBIND11_OVERRIDE_PURE(
            const std::vector<NumType>&, Layer, last_output,
        );
    }

    SizeType param_count() const noexcept override {
        PYBIND11_OVERRIDE_PURE(
            SizeType, Layer, param_count,
        );
    }

    NumType& param(SizeType index) override {
        PYBIND11_OVERRIDE_PURE(
            NumType&, Layer, param, index
        );
    }

    NumType& gradient(SizeType index) override {
        PYBIND11_OVERRIDE_PURE(
            NumType&, Layer, gradient, index
        );
    }

    SharedPtr clone() const override {
        PYBIND11_OVERRIDE_PURE(
            SharedPtr, Layer, clone,
        );
    }

    void print() const override {
        PYBIND11_OVERRIDE_PURE(
            void, Layer, print,
        );
    }

    Json dump() const override {
        PYBIND11_OVERRIDE(
            Json, Layer, dump,
        );
    }

    void load(const Json& out) override {
        PYBIND11_OVERRIDE(
            void, Layer, load, out
        );
    }
};

static void dlmath_class(pybind11::module& subm)
{
    auto dlmath = subm.def_submodule("math");

    py::class_<DLMath::Coord2d> coord2d_class(dlmath, "Coord2d");
    coord2d_class.def(py::init<SizeType, SizeType>(), "row"_a, "col"_a);
    coord2d_class.def_readwrite("row", &DLMath::Coord2d::row);
    coord2d_class.def_readwrite("col", &DLMath::Coord2d::col);

    py::class_<DLMath::Coord3d> coord3d_class(dlmath, "Coord3d");
    coord3d_class.def(
        py::init<SizeType, SizeType, SizeType>(),
        "row"_a, "col"_a, "channel"_a=0);
    coord3d_class.def_readwrite("row", &DLMath::Coord3d::row);
    coord3d_class.def_readwrite("col", &DLMath::Coord3d::col);
    coord3d_class.def_readwrite("channel", &DLMath::Coord3d::channel);

    py::class_<DLMath::Shape> shape_class(dlmath, "Shape");
    shape_class.def(py::init<std::vector<SizeType>>());
    shape_class.def("size", &DLMath::Shape::size);

    py::class_<DLMath::Shape2d, DLMath::Shape> shape2d_class(dlmath, "Shape2d");
    shape2d_class.def(py::init<SizeType, SizeType>());
    shape2d_class.def(py::init<SizeType>());
    shape2d_class.def_property(
        "height",
        [](const DLMath::Shape2d& shape) { return shape.height(); },
        [](DLMath::Shape2d& shape, SizeType height) { shape.height() = height; });
    shape2d_class.def_property(
        "width",
        [](const DLMath::Shape2d& shape) { return shape.width(); },
        [](DLMath::Shape2d& shape, SizeType width) { shape.width() = width; });

    py::class_<DLMath::Shape3d, DLMath::Shape> shape3d_class(dlmath, "Shape3d");
    shape3d_class.def(py::init<SizeType, SizeType, SizeType>(),
        "height"_a, "width"_a=1, "channels"_a=1);
    shape3d_class.def_property(
        "height",
        [](const DLMath::Shape3d& shape) { return shape.height(); },
        [](DLMath::Shape3d& shape, SizeType height) { shape.height() = height; });
    shape3d_class.def_property(
        "width",
        [](const DLMath::Shape3d& shape) { return shape.width(); },
        [](DLMath::Shape3d& shape, SizeType width) { shape.width() = width; });
    shape3d_class.def_property(
        "channels",
        [](const DLMath::Shape3d& shape) { return shape.channels(); },
        [](DLMath::Shape3d& shape, SizeType channels) { shape.channels() = channels; });

    py::enum_<DLMath::ProbabilityDensityFunction> probability_density_function_enum(
        dlmath, "ProbabilityDensityFunction");
    probability_density_function_enum.value("NORMAL", DLMath::ProbabilityDensityFunction::NORMAL);
    probability_density_function_enum.value("UNIFORM", DLMath::ProbabilityDensityFunction::UNIFORM);
    probability_density_function_enum.export_values();

    py::enum_<DLMath::InitializationFunction> initialization_function_enum(
            dlmath, "InitializationFunction");
    initialization_function_enum.value("XAVIER", DLMath::InitializationFunction::XAVIER);
    initialization_function_enum.value("KAIMING", DLMath::InitializationFunction::KAIMING);
    initialization_function_enum.export_values();

}

static void layer_class(pybind11::module& subm)
{
    py::class_<LayerShape> layer_shape_class(subm, "LayerShape");
    layer_shape_class.def(py::init<std::vector<DLMath::Shape3d>>());
    layer_shape_class.def(py::init<DLMath::Shape3d>());
    layer_shape_class.def(py::init<SizeType>());
    layer_shape_class.def(py::init<>());
    layer_shape_class.def(py::init(
        [](std::vector<py::tuple> shapes)
        {
            return get_layer_shape_from(shapes);
        }));
    layer_shape_class.def(py::init(
        [](py::tuple shape)
        {
            return get_layer_shape_from(shape);
        }));
    layer_shape_class.def("shapes", &LayerShape::shapes);
    layer_shape_class.def("shape", &LayerShape::shape, "idx"_a=0);
    layer_shape_class.def("size", &LayerShape::size, "idx"_a=0);
    layer_shape_class.def("width", &LayerShape::width, "idx"_a=0);
    layer_shape_class.def("height", &LayerShape::height, "idx"_a=0);
    layer_shape_class.def("channels", &LayerShape::channels, "idx"_a=0);
    layer_shape_class.def("amount_shapes", &LayerShape::amount_shapes);

    py::class_<Layer, PyLayer> layer_class(subm, "Layer");
    layer_class.def(
        py::init<std::string, LayerShape, LayerShape, std::string>(),
        "name"_a="", "input_shape"_a=LayerShape(0), "output_shape"_a=LayerShape(0), "prefix_name"_a="");
    layer_class.def("init", &Layer::init,
                    "init_function"_a=Layer::InitializationFunction::KAIMING,
                    "pdf"_a=Layer::ProbabilityDensityFunction::NORMAL,
                    "rne"_a=std::random_device{}());
    layer_class.def("forward", &Layer::forward, "inputs"_a);
    layer_class.def("training_forward", &Layer::training_forward, "inputs"_a);
    layer_class.def("backward", &Layer::backward, "inputs"_a);
    layer_class.def_property_readonly("type_name", &Layer::type);
    layer_class.def_property_readonly("last_input", &Layer::last_input);
    layer_class.def_property_readonly("last_input_gradient", &Layer::last_input_gradient);
    layer_class.def_property_readonly("last_output", &Layer::last_output);
    layer_class.def_property_readonly("param_count", &Layer::param_count);
    layer_class.def("param", &Layer::param, "index"_a);
    layer_class.def("gradient", &Layer::gradient, "index"_a);
    layer_class.def("clone", &Layer::clone);
    layer_class.def("print", &Layer::print);
    layer_class.def_property_readonly("name", &Layer::name);
    layer_class.def_property(
        "input_shape",
        [](const Layer& l){ return l.input_shape(); },
        [](Layer& l, const LayerShape& l_shape){ l.input_shape(l_shape); });
    layer_class.def_property_readonly("output_shape", &Layer::output_shape);
    layer_class.def_property_readonly("input_shapes", &Layer::input_shapes);
    layer_class.def_property_readonly("output_shapes", &Layer::output_shapes);
    layer_class.def("input_size", &Layer::input_size, "input_idx"_a=0);
    layer_class.def("output_size", &Layer::output_size, "output_idx"_a=0);
    layer_class.def_property_readonly("input_layers", &Layer::input_layers);
    layer_class.def_property_readonly("output_layers", &Layer::output_layers);
    layer_class.def("dump", &Layer::dump);
    layer_class.def("load", &Layer::load, "in"_a);
}

void dnn_submodule(pybind11::module& subm)
{
    subm.doc() = "Python Edge Learning submodule for "
                 "Deep Neural Network components";

    dlmath_class(subm);
    layer_class(subm);
}