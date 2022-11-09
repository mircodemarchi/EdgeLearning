#!/usr/bin/python3

##############################################################################
#            test_data.py
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

import numpy as np

from pyedgelearning.data import Dataset


def test_constructors():
    ds = Dataset()
    assert(ds.size() == 0)
    assert(ds.feature_size() == 0)
    assert(ds.sequence_size == 0)
    assert(ds.label_idx == [])

    l = [1,2,3,4]
    labels = [1]
    ds = Dataset(l, 2, 1, set(labels))
    assert(ds.size() == 2)
    assert(ds.feature_size() == 2)
    assert(ds.sequence_size == 1)
    assert(ds.label_idx == labels)

    l = [[1,2],[3,4]]
    labels = [1]
    ds = Dataset(l, 1, set(labels))
    assert(ds.size() == 2)
    assert(ds.feature_size() == 2)
    assert(ds.sequence_size == 1)
    assert(ds.label_idx == labels)

    l = [[[1,2],[3,4]], [[1,2],[3,4]]]
    labels = [1]
    ds = Dataset(l, set(labels))
    assert(ds.size() == 4)
    assert(ds.feature_size() == 2)
    assert(ds.sequence_size == 2)
    assert(ds.label_idx == labels)


def test_from_numpy():
    arr = np.array([1,2,3,4,1,2,3,4])
    labels = [0]
    ds = Dataset(arr, set(labels))
    assert(ds.size() == 8)
    assert(ds.feature_size() == 1)
    assert(ds.sequence_size == 1)
    assert(ds.label_idx == labels)

    arr = arr.reshape((-1, 2))
    labels = [1]
    ds = Dataset(arr, set(labels))
    assert(ds.size() == 4)
    assert(ds.feature_size() == 2)
    assert(ds.sequence_size == 1)
    assert(ds.label_idx == labels)

    arr = arr.reshape((-1, 2, 2))
    labels = [1]
    ds = Dataset(arr, set(labels))
    assert(ds.size() == 4)
    assert(ds.feature_size() == 2)
    assert(ds.sequence_size == 2)
    assert(ds.label_idx == labels)


def test_to_numpy():
    l = [1,2,3,4]
    arr = np.array(l)
    ds = Dataset(l)
    assert((np.array(ds) == arr).all())


def test_methods():
    l = [[[1,2],[3,4]], [[1,2],[3,4]]]
    labels = [1]
    ds = Dataset(l, set(labels))
    assert(ds.size() == 4)
    assert(ds.feature_size() == 2)
    assert(ds.sequence_size == 2)
    assert(ds.label_idx == labels)

    assert(not ds.empty())
    assert(Dataset().empty())

    ds.sequence_size = 1
    assert(ds.sequence_size == 1)
    ds.sequence_size = 2
    assert(ds.sequence_size == 2)

    assert(ds.entry(0) == [1,2])
    assert(ds.entry_seq(0) == [1,2,3,4])

    assert(ds.input_idx() == [0])
    assert((np.array(ds.inputs()) == np.array([1,3,1,3])).all())
    assert(ds.input(0) == [1])
    assert(ds.inputs_seq(0) == [1,3])