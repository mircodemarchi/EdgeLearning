/***************************************************************************
 *            dataset.hpp
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

/*! \file dataset.hpp
 *  \brief TODO
 */

#ifndef EDGE_LEARNING_MIDDLEWARE_DATASET_HPP
#define EDGE_LEARNING_MIDDLEWARE_DATASET_HPP


#include <cstddef>
#include <vector>
#include <algorithm>
#include <cmath>

namespace EdgeLearning {

// template<typename T>
// using Sequence = std::vector<T>;

template<typename T = double>
class Dataset {
public:
    using Vec = std::vector<T>;
    using Mat = std::vector<Vec>;
    using Cub = std::vector<Mat>;

    Dataset(Vec data = Vec(), 
        std::size_t feature_size = 0, 
        std::size_t sequence_size = 1)
        : _data{data}
        , _feature_size{std::min(feature_size, data.size())}
        , _dataset_size{std::ceil((data.size() - 1) / _feature_size) + 1}
        , _sequence_size{std::min(sequence_size, _dataset_size)}
    {
        // Ceil the dataset size.
        _data.resize(_dataset_size * _feature_size);   
    }

    Dataset(Mat data = Mat(), std::size_t sequence_size = 1) 
        : _data{}
        , _dataset_size{data.size()}
        , _sequence_size{std::min(sequence_size, _dataset_size)}
    {
        if (data.empty())
        {
            _feature_size = 0;
        }
        else 
        {
            _feature_size = data[0].size();
            for (const auto& v: data)
            {
                _feature_size = std::min(v.size(), _feature_size);
            }
            if (_feature_size != 0)
            {
                for (const auto& v: data)
                {
                    _data.insert(data.end(), 
                        v.begin(), v.begin() + _feature_size);
                }
            }
        }
    } 

    Dataset(Cub data = Cub()) 
        : _data{}
        , _dataset_size{0}
    {
        if (data.empty())
        {
            _sequence_size = 0;
            _feature_size = 0;
        }
        else if (data[0].empty())
        {
            _sequence_size = data.size();
            _feature_size = 0;
        }
        else 
        {
            _sequence_size = data[0].size();
            _feature_size = data[0][0].size();
            for (const auto& m: data)
            {
                _sequence_size = std::min(m.size(), _sequence_size);
                for (const auto& v: m)
                {
                    _feature_size = std::min(v.size(), _feature_size);
                }
            }
            if (_sequence_size != 0 && _feature_size != 0)
            {
                for (const auto& m: data)
                {
                    for (const auto& v: m)
                    {
                        _data.insert(data.end(), 
                            v.begin(), v.begin() + _feature_size);
                    }
                }
            }
        }
    } 

    ~Dataset() {};

    /**
     * @brief Getter and setter of feature_size param.
     * @return std::size_t& 
     */
    void feature_size(std::size_t s);
    const std::size_t& feature_size() const { return _feature_size; };

    /**
     * @brief Getter and setter of sequence_size param.
     * @return std::size_t& 
     */
    void sequence_size(std::size_t s);
    const std::size_t& sequence_size() const { return _sequence_size; };

    /**
     * @brief Return the number of entries of the dataset.
     * @return std::size_t the number of entries of the dataset.
     */
    std::size_t size() const { return _dataset_size; };

private:
    std::vector<T> _data;
    std::size_t _feature_size;
    std::size_t _dataset_size;
    std::size_t _sequence_size;
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_MIDDLEWARE_DATASET_HPP
