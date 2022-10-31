#!/usr/bin/python3

##############################################################################
#            test_import.py
#
#  Copyright  2006-20  Mirco De Marchi
##############################################################################

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.

from pyedgelearning.dnn.math import (
    Coord2d, Coord3d,
    Shape, Shape2d, Shape3d,
    ProbabilityDensityFunction,
    InitializationFunction
)
from pyedgelearning.dnn import LayerShape, Layer


def test_coord():
    coord2d = Coord2d(1, 2)
    assert(coord2d.row == 1)
    assert(coord2d.col == 2)
    coord2d.row = 3
    coord2d.col = 4
    assert(coord2d.row == 3)
    assert(coord2d.col == 4)

    coord3d = Coord3d(1, 2, 3)
    assert(coord3d.row == 1)
    assert(coord3d.col == 2)
    assert(coord3d.channel == 3)
    coord3d.row = 3
    coord3d.col = 4
    coord3d.channel = 5
    assert(coord2d.row == 3)
    assert(coord2d.col == 4)
    assert(coord3d.channel == 5)


def test_shape():
    shape = Shape([1, 2, 3])
    assert(shape.size() == 6)

    shape2d = Shape2d(2,4)
    assert(shape2d.size() == 8)
    assert(shape2d.height == 2)
    assert(shape2d.width == 4)
    shape2d.height = 5
    shape2d.width = 3
    assert(shape2d.size() == 15)
    assert(shape2d.height == 5)
    assert(shape2d.width == 3)

    shape2d = Shape2d(2)
    assert(shape2d.size() == 4)
    assert(shape2d.height == 2)
    assert(shape2d.width == 2)
    shape2d.width = 3
    assert(shape2d.size() == 6)
    assert(shape2d.height == 2)
    assert(shape2d.width == 3)

    shape3d = Shape3d(2, 4, 6)
    assert(shape3d.size() == 48)
    assert(shape3d.height == 2)
    assert(shape3d.width == 4)
    assert(shape3d.channels == 6)
    shape3d.height = 5
    shape3d.width = 3
    shape3d.channels = 2
    assert(shape3d.size() == 30)
    assert(shape3d.height == 5)
    assert(shape3d.width == 3)
    assert(shape3d.channels == 2)


def test_probability_density_function():
    assert(ProbabilityDensityFunction.NORMAL)
    assert(ProbabilityDensityFunction.UNIFORM)


def test_initialization_function():
    assert(InitializationFunction.XAVIER)
    assert(InitializationFunction.KAIMING)


def test_layer_shape():
    layer_shape = LayerShape([Shape3d(1, 2, 3), Shape3d(4, 5, 6)])
    assert(len(layer_shape.shapes()) == 2)
    assert(layer_shape.amount_shapes() == 2)
    assert(layer_shape.shape().size() == 6)
    assert(layer_shape.shape(1).size() == 120)
    assert(layer_shape.size() == 6)
    assert(layer_shape.size(1) == 120)
    assert(layer_shape.height() == 1)
    assert(layer_shape.width() == 2)
    assert(layer_shape.channels() == 3)
    assert(layer_shape.height(1) == 4)
    assert(layer_shape.width(1) == 5)
    assert(layer_shape.channels(1) == 6)

    layer_shape = LayerShape(Shape3d(1, 2, 3))
    assert(len(layer_shape.shapes()) == 1)
    assert(layer_shape.amount_shapes() == 1)
    assert(layer_shape.shape().size() == 6)
    assert(layer_shape.size() == 6)
    assert(layer_shape.height() == 1)
    assert(layer_shape.width() == 2)
    assert(layer_shape.channels() == 3)

    layer_shape = LayerShape(100)
    assert(len(layer_shape.shapes()) == 1)
    assert(layer_shape.amount_shapes() == 1)
    assert(layer_shape.shape().size() == 100)
    assert(layer_shape.size() == 100)
    assert(layer_shape.height() == 100)
    assert(layer_shape.width() == 1)
    assert(layer_shape.channels() == 1)

    layer_shape = LayerShape()
    assert(len(layer_shape.shapes()) == 0)
    assert(layer_shape.amount_shapes() == 0)

    layer_shape = LayerShape((1, 2, 3))
    assert(len(layer_shape.shapes()) == 1)
    assert(layer_shape.amount_shapes() == 1)
    assert(layer_shape.shape().size() == 6)
    assert(layer_shape.size() == 6)
    assert(layer_shape.height() == 1)
    assert(layer_shape.width() == 2)
    assert(layer_shape.channels() == 3)

    layer_shape = LayerShape([(1, 2, 3), (4, 5, 6)])
    assert(len(layer_shape.shapes()) == 2)
    assert(layer_shape.amount_shapes() == 2)
    assert(layer_shape.shape().size() == 6)
    assert(layer_shape.shape(1).size() == 120)
    assert(layer_shape.size() == 6)
    assert(layer_shape.size(1) == 120)
    assert(layer_shape.height() == 1)
    assert(layer_shape.width() == 2)
    assert(layer_shape.channels() == 3)
    assert(layer_shape.height(1) == 4)
    assert(layer_shape.width(1) == 5)
    assert(layer_shape.channels(1) == 6)

    layer_shape = LayerShape((2, 1, 2, 3))
    assert(len(layer_shape.shapes()) == 2)
    assert(layer_shape.amount_shapes() == 2)
    assert(layer_shape.shape().size() == 6)
    assert(layer_shape.shape(1).size() == 6)
    assert(layer_shape.size() == 6)
    assert(layer_shape.size(1) == 6)
    assert(layer_shape.height() == 1)
    assert(layer_shape.width() == 2)
    assert(layer_shape.channels() == 3)
    assert(layer_shape.height(1) == 1)
    assert(layer_shape.width(1) == 2)
    assert(layer_shape.channels(1) == 3)


def test_layer():
    class CustomLayer(Layer):
        def __init__(self, input_size=0, output_size=0, name=""):
            Layer.__init__(
                self, "custom_layer_test" if len(name) == 0 else name,
                LayerShape(input_size), LayerShape(output_size))

        def init(self):
            pass

        def forward(self, inputs):
            return inputs

        def backward(self, gradients):
            return gradients

        def last_input_gradient(self):
            raise Exception("")

        def last_output(self):
            raise Exception("")

        def param_count(self):
            return 0

        def param(self, index):
            raise Exception("")

        def gradient(self, index):
            raise Exception("")

        def clone(self):
            return self

        def print(self):
            pass

        def type(self):
            return "CustomType"

    l = Layer()
    l.forward([])
    l.training_forward([])
    l.backward([])
    assert(l.type_name == "None")
    assert(l.last_input == [])
    assert(l.input_shape.amount_shapes() == 1)
    assert(l.input_shape.size() == LayerShape(0).size())
    assert(l.input_shapes[0].size() == Shape3d(0).size())
    assert(len(l.input_shapes) == 1)
    assert(l.input_size() == Shape3d(0).size())
    assert(l.input_layers == 1)
    assert(l.output_shape.amount_shapes() == 1)
    assert(l.output_shape.size() == LayerShape(0).size())
    assert(l.output_shapes[0].size() == Shape3d(0).size())
    assert(len(l.output_shapes) == 1)
    assert(l.output_size() == Shape3d(0).size())
    assert(l.output_layers == 1)
    l.input_shape = LayerShape((2, 1, 1, 1))
    assert(l.input_shape.amount_shapes() == 2)
    assert(l.input_shape.size() == LayerShape((2, 1, 1, 1)).size())
    assert(l.input_shape.size(1) == LayerShape((2, 1, 1, 1)).size(1))
    assert(l.input_shapes[0].size() == Shape3d(1, 1, 1).size())
    assert(l.input_shapes[1].size() == Shape3d(1, 1, 1).size())
    assert(len(l.input_shapes) == 2)
    assert(l.input_size() == Shape3d(1, 1, 1).size())
    assert(l.input_size(1) == Shape3d(1, 1, 1).size())

    l = CustomLayer(10, 20)
    l.init()
    l.print()
    assert(l.forward([]) == [])
    assert(l.training_forward([0.0] * 10) == ([0.0] * 10))
    assert(l.backward([]) == [])
    assert(l.type_name == "CustomType")
    assert(len(l.last_input) == len([0.0] * 10))
    assert(l.input_shape.amount_shapes() == 1)
    assert(l.input_shape.size() == 10)
    assert(l.input_shapes[0].size() == 10)
    assert(len(l.input_shapes) == 1)
    assert(l.input_size() == 10)
    assert(l.input_layers == 1)
    assert(l.output_shape.amount_shapes() == 1)
    assert(l.output_shape.size() == 20)
    assert(l.output_shapes[0].size() == 20)
    assert(len(l.output_shapes) == 1)
    assert(l.output_size() == 20)
    assert(l.output_layers == 1)
    assert(l.name == "custom_layer_test")
    assert(l.clone() == l)
    assert(l.param_count() == 0)
    try:
        l.param(0)
        assert(False)
    except:
        assert(True)
    try:
        l.gradient(0)
        assert(False)
    except:
        assert(True)
    try:
        print(l.last_input_gradient)
        assert(False)
    except:
        assert(True)
    try:
        print(l.last_output)
        assert(False)
    except:
        assert(True)



