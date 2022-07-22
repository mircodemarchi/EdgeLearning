/***************************************************************************
 *            type.hpp
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

/*! \file  type.hpp
 *  \brief Global types of the whole library.
 */

#ifndef EDGE_LEARNING_DNN_TYPE_HPP
#define EDGE_LEARNING_DNN_TYPE_HPP

#include <random>
#include <cstddef>
#include <memory>
#include <vector>


namespace EdgeLearning {

using NumType = double;

/**
 * Random number engine: 64-bit Mersenne Twister by Matsumoto and
 * Nishimura, 1998.
 */
using RneType = std::mt19937_64;

using SizeType = std::size_t;

/**
 * \brief Learning parameters of a layer that can't be shared and will be
 * copied.
 */
using Params = std::vector<NumType>;

/**
 * \brief Learning parameters of a layer that can be shared.
 */
class SharedParams {
public:
    struct Iterator
    {
        using pointer   = NumType*;
        using reference = NumType&;

        Iterator(Params::iterator ptr) : _iter(ptr) {}

        reference operator*() const { return _iter.operator*(); }
        pointer operator->() { return _iter.operator->(); }
        Iterator& operator++() { _iter++; return *this; }
        Iterator operator++(int)
        { Iterator tmp = *this; ++(*this); return tmp; }
        friend bool operator== (const Iterator& a, const Iterator& b)
        { return a._iter == b._iter; };
        friend bool operator!= (const Iterator& a, const Iterator& b)
        { return a._iter != b._iter; };

    private:
        Params::iterator _iter;
    };

    SharedParams()
        : _p(std::make_shared<Params>())
    { }

    void resize(std::size_t length) const { (*_p).resize(length); }
    NumType& operator[](std::size_t i) const { return (*_p)[i]; }
    [[nodiscard]] const NumType& at(std::size_t i) const
    { return (*_p).at(i); }
    NumType* data() { return (*_p).data(); }
    std::size_t size() { return (*_p).size(); }

    Iterator begin() { return Iterator((*_p).begin()); }
    Iterator end()   { return Iterator((*_p).end());   }

private:
    std::shared_ptr<Params> _p;
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_DNN_TYPE_HPP
