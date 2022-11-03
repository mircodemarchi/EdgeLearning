#!/usr/bin/python3

##############################################################################
#            test_fnn.py
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

from pyedgelearning import FNN, NN
from pyedgelearning import (
    LayerSetting,
    LayerDescriptor,
    Input,
    Dense,
    Conv,
    MaxPool,
    AvgPool,
    Dropout
)
from pyedgelearning import (
    Framework,
    ParallelizationLevel,
    LayerType,
    ActivationType,
    LossType,
    OptimizerType,
    InitType
)
from pyedgelearning.data import Dataset
from pyedgelearning.dnn import LayerShape
from pyedgelearning.dnn.math import Shape2d


def test_layer_descriptor():
    ls = LayerSetting()
    assert(ls.units.size() == LayerShape(0).size())
    assert(ls.n_filters == 0)
    assert(ls.kernel_shape.size() == Shape2d(0).size())
    assert(ls.strides.size() == Shape2d(0).size())
    assert(ls.padding.size() == Shape2d(0).size())
    assert(ls.drop_probability == 0)
    ls.units = LayerShape((1, 2, 3))
    ls.n_filters = 10
    ls.kernel_shape = Shape2d(3, 3)
    ls.strides = Shape2d(2, 2)
    ls.padding = Shape2d(1, 1)
    ls.drop_probability = 0.2
    assert(ls.units.size() == LayerShape((1, 2, 3)).size())
    assert(ls.n_filters == 10)
    assert(ls.kernel_shape.size() == Shape2d(3, 3).size())
    assert(ls.strides.size() == Shape2d(2, 2).size())
    assert(ls.padding.size() == Shape2d(1, 1).size())
    assert(ls.drop_probability == 0.2)

    ls = LayerSetting(LayerShape((1, 2, 3)))
    assert(ls.units.size() == LayerShape((1, 2, 3)).size())
    assert(ls.n_filters == 0)
    assert(ls.kernel_shape.size() == Shape2d(0).size())
    assert(ls.strides.size() == Shape2d(0).size())
    assert(ls.padding.size() == Shape2d(0).size())
    assert(ls.drop_probability == 0)
    ls = LayerSetting((1, 2, 3))
    assert(ls.units.size() == LayerShape((1, 2, 3)).size())
    ls = LayerSetting((2, 1, 2, 3))
    assert(ls.units.size() == LayerShape((1, 2, 3)).size())
    assert(ls.units.amount_shapes() == 2)

    ls = LayerSetting(16, Shape2d(3, 3), Shape2d(2, 2), Shape2d(1, 1))
    assert(ls.units.size() == LayerShape(0).size())
    assert(ls.n_filters == 16)
    assert(ls.kernel_shape.size() == Shape2d(3, 3).size())
    assert(ls.strides.size() == Shape2d(2, 2).size())
    assert(ls.padding.size() == Shape2d(1, 1).size())
    assert(ls.drop_probability == 0)
    ls = LayerSetting(16, (3, 3), (2, 2), (1, 1))
    assert(ls.n_filters == 16)
    assert(ls.kernel_shape.size() == Shape2d(3, 3).size())
    assert(ls.strides.size() == Shape2d(2, 2).size())
    assert(ls.padding.size() == Shape2d(1, 1).size())

    ls = LayerSetting(Shape2d(3, 3), Shape2d(2, 2))
    assert(ls.units.size() == LayerShape(0).size())
    assert(ls.n_filters == 0)
    assert(ls.kernel_shape.size() == Shape2d(3, 3).size())
    assert(ls.strides.size() == Shape2d(2, 2).size())
    assert(ls.padding.size() == Shape2d(0).size())
    assert(ls.drop_probability == 0)
    ls = LayerSetting((3, 3), (2, 2))
    assert(ls.kernel_shape.size() == Shape2d(3, 3).size())
    assert(ls.strides.size() == Shape2d(2, 2).size())

    ls = LayerSetting(0.2)
    assert(ls.units.size() == LayerShape(0).size())
    assert(ls.n_filters == 0)
    assert(ls.kernel_shape.size() == Shape2d(0).size())
    assert(ls.strides.size() == Shape2d(0).size())
    assert(ls.padding.size() == Shape2d(0).size())
    assert(ls.drop_probability == 0.2)

    ld = LayerDescriptor("layer_descriptor_test", LayerType.DENSE, LayerSetting(), ActivationType.LINEAR)
    assert(ld.name == "layer_descriptor_test")
    ld.name = "layer_descriptor_test2"
    assert(ld.name == "layer_descriptor_test2")
    assert(ld.type == LayerType.DENSE)
    ld.type = LayerType.CONV
    assert(ld.type == LayerType.CONV)
    assert(ld.setting.units.size() == LayerShape(0).size())
    assert(ld.setting.n_filters == 0)
    assert(ld.setting.kernel_shape.size() == Shape2d(0).size())
    assert(ld.setting.strides.size() == Shape2d(0).size())
    assert(ld.setting.padding.size() == Shape2d(0).size())
    assert(ld.setting.drop_probability == 0)
    ls = LayerSetting()
    ls.units = LayerShape((1, 2, 3))
    ls.n_filters = 10
    ls.kernel_shape = Shape2d(3, 3)
    ls.strides = Shape2d(2, 2)
    ls.padding = Shape2d(1, 1)
    ls.drop_probability = 0.2
    ld.setting = ls
    assert(ls.units.size() == LayerShape((1, 2, 3)).size())
    assert(ls.n_filters == 10)
    assert(ls.kernel_shape.size() == Shape2d(3, 3).size())
    assert(ls.strides.size() == Shape2d(2, 2).size())
    assert(ls.padding.size() == Shape2d(1, 1).size())
    assert(ls.drop_probability == 0.2)
    assert(ld.activation_type == ActivationType.LINEAR)
    ld.activation_type = ActivationType.RELU
    assert(ld.activation_type == ActivationType.RELU)

    input_ld = Input("input_layer_descriptor", LayerShape(10))
    assert(input_ld.name == "input_layer_descriptor")
    assert(input_ld.type == LayerType.INPUT)
    assert(input_ld.activation_type == ActivationType.LINEAR)
    assert(input_ld.setting.units.size() == 10)
    input_ld = Input("input_layer_descriptor", LayerShape((1, 2, 3)))
    assert(input_ld.setting.units.size() == 6)
    input_ld = Input("input_layer_descriptor", 10)
    assert(input_ld.setting.units.size() == 10)
    input_ld = Input("input_layer_descriptor", (1, 2, 3))
    assert(input_ld.setting.units.size() == 6)
    input_ld = Input("input_layer_descriptor", [(1, 2, 3), (2, 2, 3)])
    assert(input_ld.setting.units.size() == 6)
    assert(input_ld.setting.units.size(1) == 12)

    dense_ld = Dense("dense_layer_descriptor", 10, ActivationType.RELU)
    assert(dense_ld.name == "dense_layer_descriptor")
    assert(dense_ld.type == LayerType.DENSE)
    assert(dense_ld.activation_type == ActivationType.RELU)
    assert(dense_ld.setting.units.size() == 10)

    conv_ld = Conv("conv_layer_descriptor", (16, (3, 3)), ActivationType.RELU)
    assert(conv_ld.name == "conv_layer_descriptor")
    assert(conv_ld.type == LayerType.CONV)
    assert(conv_ld.activation_type == ActivationType.RELU)
    assert(conv_ld.setting.n_filters == 16)
    assert(conv_ld.setting.kernel_shape.size() == 9)
    assert(conv_ld.setting.strides.size() == 1)
    assert(conv_ld.setting.padding.size() == 0)
    conv_ld = Conv("conv_layer_descriptor", (16, (3, 3), (2, 2)), ActivationType.RELU)
    assert(conv_ld.setting.n_filters == 16)
    assert(conv_ld.setting.kernel_shape.size() == 9)
    assert(conv_ld.setting.strides.size() == 4)
    assert(conv_ld.setting.padding.size() == 0)
    conv_ld = Conv("conv_layer_descriptor", (16, (3, 3), (2, 2), (1, 1)), ActivationType.RELU)
    assert(conv_ld.setting.n_filters == 16)
    assert(conv_ld.setting.kernel_shape.size() == 9)
    assert(conv_ld.setting.strides.size() == 4)
    assert(conv_ld.setting.padding.size() == 1)
    conv_ld = Conv("conv_layer_descriptor", Conv.Setting(16, Shape2d(3, 3), Shape2d(2, 2), Shape2d(1, 1)), ActivationType.RELU)
    assert(conv_ld.setting.n_filters == 16)
    assert(conv_ld.setting.kernel_shape.size() == 9)
    assert(conv_ld.setting.strides.size() == 4)
    assert(conv_ld.setting.padding.size() == 1)
    conv_ld = Conv("conv_layer_descriptor", Conv.Setting(16, (3, 3), (2, 2), (1, 1)), ActivationType.RELU)
    assert(conv_ld.setting.n_filters == 16)
    assert(conv_ld.setting.kernel_shape.size() == 9)
    assert(conv_ld.setting.strides.size() == 4)
    assert(conv_ld.setting.padding.size() == 1)

    max_pool_ld = MaxPool("max_pool_layer_descriptor", ((3, 3), ), ActivationType.RELU)
    assert(max_pool_ld.name == "max_pool_layer_descriptor")
    assert(max_pool_ld.type == LayerType.MAX_POOL)
    assert(max_pool_ld.activation_type == ActivationType.RELU)
    assert(max_pool_ld.setting.kernel_shape.size() == 9)
    assert(max_pool_ld.setting.strides.size() == 1)
    max_pool_ld = MaxPool("max_pool_layer_descriptor", ((3, 3), (2, 2)), ActivationType.RELU)
    assert(max_pool_ld.setting.kernel_shape.size() == 9)
    assert(max_pool_ld.setting.strides.size() == 4)
    max_pool_ld = MaxPool("max_pool_layer_descriptor", MaxPool.Setting(Shape2d(3, 3), Shape2d(2, 2)), ActivationType.RELU)
    assert(max_pool_ld.setting.kernel_shape.size() == 9)
    assert(max_pool_ld.setting.strides.size() == 4)
    max_pool_ld = MaxPool("max_pool_layer_descriptor", MaxPool.Setting((3, 3), (2, 2)), ActivationType.RELU)
    assert(max_pool_ld.setting.kernel_shape.size() == 9)
    assert(max_pool_ld.setting.strides.size() == 4)

    avg_pool_ld = AvgPool("avg_pool_layer_descriptor", ((3, 3), ), ActivationType.RELU)
    assert(avg_pool_ld.name == "avg_pool_layer_descriptor")
    assert(avg_pool_ld.type == LayerType.AVG_POOL)
    assert(avg_pool_ld.activation_type == ActivationType.RELU)
    assert(avg_pool_ld.setting.kernel_shape.size() == 9)
    assert(avg_pool_ld.setting.strides.size() == 1)
    avg_pool_ld = AvgPool("avg_pool_layer_descriptor", ((3, 3), (2, 2)), ActivationType.RELU)
    assert(avg_pool_ld.setting.kernel_shape.size() == 9)
    assert(avg_pool_ld.setting.strides.size() == 4)
    avg_pool_ld = AvgPool("avg_pool_layer_descriptor", AvgPool.Setting(Shape2d(3, 3), Shape2d(2, 2)), ActivationType.RELU)
    assert(avg_pool_ld.setting.kernel_shape.size() == 9)
    assert(avg_pool_ld.setting.strides.size() == 4)
    avg_pool_ld = AvgPool("avg_pool_layer_descriptor", AvgPool.Setting((3, 3), (2, 2)), ActivationType.RELU)
    assert(avg_pool_ld.setting.kernel_shape.size() == 9)
    assert(avg_pool_ld.setting.strides.size() == 4)

    dropout_ld = Dropout("dropout_layer_descriptor", 0.2, ActivationType.RELU)
    assert(dropout_ld.name == "dropout_layer_descriptor")
    assert(dropout_ld.type == LayerType.DROPOUT)
    assert(dropout_ld.activation_type == ActivationType.RELU)
    assert(dropout_ld.setting.drop_probability == 0.2)


def test_nn():
    assert(Framework.EDGE_LEARNING)

    assert(ParallelizationLevel.SEQUENTIAL)
    assert(ParallelizationLevel.THREAD_PARALLELISM_ON_DATA_ENTRY)
    assert(ParallelizationLevel.THREAD_PARALLELISM_ON_DATA_BATCH)

    assert(LayerType.DENSE)
    assert(LayerType.CONV)
    assert(LayerType.MAX_POOL)
    assert(LayerType.AVG_POOL)
    assert(LayerType.DROPOUT)
    assert(LayerType.INPUT)

    assert(ActivationType.RELU)
    assert(ActivationType.ELU)
    assert(ActivationType.SOFTMAX)
    assert(ActivationType.TANH)
    assert(ActivationType.SIGMOID)
    assert(ActivationType.LINEAR)
    assert(ActivationType.NONE)

    assert(LossType.CCE)
    assert(LossType.MSE)

    assert(OptimizerType.GRADIENT_DESCENT)
    assert(OptimizerType.ADAM)

    assert(InitType.HE)
    assert(InitType.XAVIER)
    assert(InitType.AUTO)

    class TestNN(NN):
        def __init__(self):
            NN.__init__(self, "test_nn")
            self.loss = None
            self.opt = None
            self.init_type = None

        def compile(self):
            self.loss = LossType.MSE
            self.opt = OptimizerType.ADAM
            self.init_type = InitType.AUTO

        def predict(self, data):
            return data

        def fit(self, data, epochs=1, batch_size=1, learning_rate=0.03):
            return data

        def evaluate(self, data):
            return NN.EvaluationResult()

        def input_size(self):
            return 0

        def output_size(self):
            return 0

    data = Dataset([1, 2, 3, 4, 5])
    nn = TestNN()
    nn.compile()
    assert(nn.predict(data) == data)
    assert(nn.fit(data) == data)
    assert(nn.evaluate(data).loss == NN.EvaluationResult().loss)
    assert(nn.evaluate(data).accuracy == NN.EvaluationResult().accuracy)
    assert(nn.evaluate(data).accuracy_perc == NN.EvaluationResult().accuracy_perc)
    assert(nn.evaluate(data).error_rate == NN.EvaluationResult().error_rate)
    assert(nn.evaluate(data).error_rate_perc == NN.EvaluationResult().error_rate_perc)
    assert(nn.input_size() == 0)
    assert(nn.output_size() == 0)

    score = NN.EvaluationResult()
    assert((score.accuracy * 100.0) == score.accuracy_perc)
    assert((score.error_rate * 100.0) == score.error_rate_perc)
    score = NN.EvaluationResult(0.2, 0.8)
    assert((score.accuracy * 100.0) == score.accuracy_perc)
    assert((score.error_rate * 100.0) == score.error_rate_perc)


def test_fnn():
    data = [
        [10.0, 1.0, 10.0, 1.0, 1.0, 0.0],
        [1.0,  3.0, 8.0,  3.0, 0.0, 1.0],
        [8.0,  1.0, 8.0,  1.0, 1.0, 0.0],
        [1.0,  1.5, 8.0,  1.5, 0.0, 1.0],
        [8.0,  1.0, 8.0,  1.0, 1.0, 0.0],
        [1.0,  1.5, 8.0,  1.5, 0.0, 1.0],
    ]
    dataset = Dataset(data, 1, {4, 5})

    layers = [
        Input("input_layer", 4),
        Dense("hidden_layer_relu0", 8, ActivationType.RELU),
        Dense("hidden_layer_relu1", 8, ActivationType.RELU),
        Dense("hidden_layer_relu2", 8, ActivationType.RELU),
        Dense("hidden_layer_relu3", 8, ActivationType.RELU),
        Dense("hidden_layer_relu4", 8, ActivationType.RELU),
        Dense("output_layer", 2, ActivationType.RELU),
    ]
    model = FNN(layers, "model")
    assert(model.input_size() == 4)
    assert(model.output_size() == 2)
    model.compile(LossType.MSE, OptimizerType.GRADIENT_DESCENT, InitType.AUTO)
    model.fit(dataset, epochs=1, batch_size=8, learning_rate=0.03)
    prediction = model.predict(dataset.trainset())
    assert(prediction.size() == 6)
    assert(prediction.feature_size() == 2)
    assert(prediction.sequence_size == 1)
    assert(prediction.labels_idx == list())
    score = model.evaluate(dataset)
    assert((score.accuracy * 100.0) == score.accuracy_perc)
    assert((score.error_rate * 100.0) == score.error_rate_perc)




