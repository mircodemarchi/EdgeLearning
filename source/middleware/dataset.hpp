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
 *  \brief The Dataset class implementation for training set, validation set 
 * and test set. 
 */

#ifndef EDGE_LEARNING_MIDDLEWARE_DATASET_HPP
#define EDGE_LEARNING_MIDDLEWARE_DATASET_HPP


#include <cstddef>
#include <vector>
#include <algorithm>
#include <cmath>
#if ENABLE_MLPACK
#include <armadillo>
#endif // ENABLE_MLPACK

namespace EdgeLearning {

/**
 * @brief Dataset class for training and Armadillo conversions. 
 * @tparam T 
 */
template<typename T = double>
class Dataset {
public:
    using Vec = std::vector<T>;
    using Mat = std::vector<Vec>;
    using Cub = std::vector<Mat>;

    /**
     * @brief Construct a new Dataset object.
     * @param data          The vector dataset. 
     * @param feature_size  The size of the features in number of elements. 
     * @param sequence_size The size of the sequence in number of feature entry. 
     */
    Dataset(Vec data = Vec(), 
        std::size_t feature_size = 1, 
        std::size_t sequence_size = 1)
        : _data{data}
        , _feature_size{std::min(feature_size, _data.size())}
        , _sequence_size{std::min(sequence_size, 
            std::size_t(_data.size() / _feature_size))}
        , _dataset_size{
            std::size_t(_data.size() / (_feature_size * _sequence_size))
            * (_feature_size * _sequence_size)}
        , _feature_amount{_dataset_size / _feature_size}
        , _sequence_amount{_dataset_size / (_feature_size * _sequence_size)}
    {
        _data.resize(_dataset_size);
    }

    /**
     * @brief Construct a new Dataset object.
     * @param data          The matrix dataset. 
     * @param sequence_size The size of the sequence in number of feature entry. 
     */
    Dataset(Mat data = Mat(), std::size_t sequence_size = 1) 
        : _data{}
        , _feature_amount{data.size()}
        , _sequence_size{std::min(sequence_size, _feature_amount)}
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
        _dataset_size = _feature_amount * _feature_size;
        _sequence_amount = _feature_amount / _sequence_size;
    } 

    /**
     * @brief Construct a new Dataset object.
     * @param data The cube dataset. 
     */
    Dataset(Cub data = Cub()) 
        : _data{}
        , _sequence_amount{data.size()}
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
                    for (std::size_t i = 0; i < _sequence_size; ++i)
                    {
                        const auto& v = m[i];
                        _data.insert(data.end(), 
                            v.begin(), v.begin() + _feature_size);
                    }
                }
            }
        }
        _feature_amount = _sequence_amount * _sequence_size;
        _dataset_size = _feature_amount * _feature_size;
    } 

    /**
     * @brief Destroy the Dataset object.
     */
    ~Dataset() {};

#if ENABLE_MLPACK
    /**
     * @brief Convert the dataset in Armadillo Vector. 
     * @return arma::Vec<T> 
     */
    operator arma::Vec<T>()
    {   
        arma::Mat<T> ret(_data);
        ret.reshape(_feature_amount, _feature_size);
        return ret;
    }

    /**
     * @brief Convert the dataset in Armadillo Matrix. 
     * @return arma::Mat<T> 
     */
    operator arma::Mat<T>()
    {   
        arma::Mat<T> ret(_data);
        ret.reshape(_feature_amount, _feature_size);
        return ret;
    }

    /**
     * @brief Convert the dataset in Armadillo Cube. 
     * @return arma::Cube<T> 
     */
    operator arma::Cube<T>()
    {   
        arma::Cube<T> ret(_data);
        ret.reshape(_sequence_amount, _sequence_size, _feature_size);
        return ret;
    }

    /**
     * @brief Convert the dataset in Armadillo format. 
     * @tparam ARMA_T arma::Cube<T>, arma::Mat<T> or arma::Vec<T>
     * @return ARMA_T arma::Cube<T>, arma::Mat<T> or arma::Vec<T>
     */
    template<typename ARMA_T>
    ARMA_T to_arma()
    {
        return operator ARMA_T();
    }
#endif // ENABLE_MLPACK

    /**
     * @brief Getter and setter of feature_size param.
     * @return const std::size_t& 
     */
    const std::size_t& feature_size() const { return _feature_size; };

    /**
     * @brief Getter and setter of sequence_size param.
     * @return std::size_t& 
     */
    void sequence_size(std::size_t s)
    {
        _sequence_size = std::min(s, _feature_amount);
        _sequence_amount = _feature_amount / _sequence_size;
        _dataset_size = _sequence_size * _sequence_amount;
        _data.resize(_dataset_size);
    }
    const std::size_t& sequence_size() const { return _sequence_size; };

    /**
     * @brief Return the number of entries of the dataset.
     * @return std::size_t the number of entries of the dataset.
     */
    std::size_t size() const { return _feature_amount; };

private:
    std::vector<T> _data;

    /**
     * @brief The size of a single entry of the dataset, called feature. 
     */
    std::size_t _feature_size;

    /**
     * @brief The size of a sequence of features entry in the dataset. The 
     * sequence size is in feature entries number granularity. 
     */
    std::size_t _sequence_size;

    /**
     * @brief The size of the dataset, that is the size of the data field.
     * Element-wise granularity. 
     */
    std::size_t _dataset_size;

    /**
     * @brief The number of feature entry of dimension `feature_size` in the 
     * dataset. The feature amount is in feature entries number granularity. 
     */
    std::size_t _feature_amount;

    /**
     * @brief The number of sequences in the dataset. The sequence amount is in 
     * sequence number granularity. 
     */
    std::size_t _sequence_amount;
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_MIDDLEWARE_DATASET_HPP
