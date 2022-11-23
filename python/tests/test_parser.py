#!/usr/bin/python3

##############################################################################
#            test_parser.py
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

import os
from pyedgelearning.parser import *


def test_parser():
    assert(True)
    return # Working test with all datasets in data folder.
    resource_path = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    train_mnist, test_mnist = load_mnist(resource_path)

    assert(train_mnist.size() == 60000)
    assert(train_mnist.feature_size() == (28 * 28 + 10))
    assert(train_mnist.sequence_size == 1)
    assert(train_mnist.label_idx == list(range(28 * 28, 28 * 28 + 10)))

    assert(test_mnist.size() == 10000)
    assert(test_mnist.feature_size() == (28 * 28 + 10))
    assert(test_mnist.sequence_size == 1)
    assert(test_mnist.label_idx == list(range(28 * 28, 28 * 28 + 10)))

    train_cifar10, test_cifar10 = load_cifar10(resource_path)

    assert(train_cifar10.size() == 50000)
    assert(train_cifar10.feature_size() == (32 * 32 * 3 + 10))
    assert(train_cifar10.sequence_size == 1)
    assert(train_cifar10.label_idx == list(range(32 * 32 * 3, 32 * 32 * 3 + 10)))

    assert(test_cifar10.size() == 10000)
    assert(test_cifar10.feature_size() == (32 * 32 * 3 + 10))
    assert(test_cifar10.sequence_size == 1)
    assert(test_cifar10.label_idx == list(range(32 * 32 * 3, 32 * 32 * 3 + 10)))

    train_cifar100, test_cifar100 = load_cifar100(resource_path)

    assert(train_cifar100.size() == 10000)
    assert(train_cifar100.feature_size() == (32 * 32 * 3 + 100))
    assert(train_cifar100.sequence_size == 1)
    assert(train_cifar100.label_idx == list(range(32 * 32 * 3, 32 * 32 * 3 + 100)))

    assert(test_cifar100.size() == 10000)
    assert(test_cifar100.feature_size() == (32 * 32 * 3 + 100))
    assert(test_cifar100.sequence_size == 1)
    assert(test_cifar100.label_idx == list(range(32 * 32 * 3, 32 * 32 * 3 + 100)))

