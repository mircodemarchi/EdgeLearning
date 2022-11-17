#!/usr/bin/python3

##############################################################################
#            mnist_dense.py
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

import pyedgelearning as el


def main():
    # Load MNIST dataset

    layers = [
        el.Input("input_layer",        4),
        el.Dense("hidden_layer_relu0", 8, el.ActivationType.RELU),
        el.Dense("hidden_layer_relu1", 8, el.ActivationType.RELU),
        el.Dense("hidden_layer_relu2", 8, el.ActivationType.RELU),
        el.Dense("hidden_layer_relu3", 8, el.ActivationType.RELU),
        el.Dense("hidden_layer_relu4", 8, el.ActivationType.RELU),
        el.Dense("output_layer",       2, el.ActivationType.RELU),
    ]
    model = el.FNN(layers, "model")


if __name__ == '__main__':
    main()

