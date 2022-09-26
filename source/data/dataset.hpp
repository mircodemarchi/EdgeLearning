/***************************************************************************
 *            data/dataset.hpp
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

/*! \file data/dataset.hpp
 *  \brief The Dataset class implementation for training set, validation set 
 * and test set. 
 */

#ifndef EDGE_LEARNING_MIDDLEWARE_DATASET_HPP
#define EDGE_LEARNING_MIDDLEWARE_DATASET_HPP

#include "type.hpp"
#include "parser/parser.hpp"
#include "dnn/dlmath.hpp"

#include <cstddef>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <tuple>
#if ENABLE_MLPACK
#include <mlpack/core.hpp>
#include <armadillo>
#endif // ENABLE_MLPACK

namespace EdgeLearning {

/**
 * \brief Dataset class for training and Armadillo conversions.
 * \tparam T
 */
template<typename T = double>
class Dataset {
public:
    using Vec = std::vector<T>;
    using Mat = std::vector<Vec>;
    using Cub = std::vector<Mat>;

    /**
     * \brief Dataset split in training set and testing set.
     */
    struct SplitDataset {
        /**
         * \brief Constructor of the split dataset.
         * \param train_set Dataset<T> Training set.
         * \param test_set  Dataset<T> Testing set.
         */
        SplitDataset(Dataset<T> train_set, Dataset<T> test_set)
            : training_set(train_set)
            , testing_set(test_set)
        { }

        Dataset<T> training_set; ///< \brief The dataset for training.
        Dataset<T> testing_set;  ///< \brief The dataset for testing.
    };

    /**
     * \brief Empty construct a new Dataset object.
     */
    Dataset()
        : _data{}
        , _entry_labels_cache{}
        , _entry_trainset_cache{}
        , _feature_size{0}
        , _sequence_size{0}
        , _dataset_size{0}
        , _feature_amount{0}
        , _sequence_amount{0}
        , _labels_idx{}
        , _trainset_idx{}
    {

    }

    /**
     * \brief Construct a new Dataset object.
     * \param data          Vec The vector dataset.
     * \param feature_size  SizeType The size of the features in number of
     *                      elements.
     * \param sequence_size SizeType The size of the sequence in number of
     *                      feature entry.
     * \param labels_idx    std::set<SizeType> The index of the feature to
     *                      use as ground truth.
     */
    Dataset(Vec data,
            SizeType feature_size = 1,
            SizeType sequence_size = 1,
            std::set<SizeType> labels_idx = {})
        : _data{data}
        , _entry_labels_cache{}
        , _entry_trainset_cache{}
        , _feature_size{std::min(feature_size, _data.size())}
        , _sequence_size{std::min(sequence_size, 
            SizeType(_data.size() / std::max(_feature_size, SizeType(1))))}
        , _dataset_size{
            SizeType(_data.size() 
                / std::max(_feature_size * _sequence_size, SizeType(1)))
            * (_feature_size * _sequence_size)}
        , _feature_amount{_dataset_size / std::max(_feature_size, SizeType(1))}
        , _sequence_amount{_dataset_size 
            / std::max(_feature_size * _sequence_size, SizeType(1))}
        , _labels_idx{labels_idx}
        , _trainset_idx{}
    {
        _data.resize(_dataset_size);

        // Init indexes set.
        for (auto it = _labels_idx.begin(); it != _labels_idx.end(); )
        {
            if (*it >= _feature_size) _labels_idx.erase(it++); else ++it;
        }
        for (SizeType idx_val = 0; idx_val < _feature_size; ++idx_val)
        {
            if (_labels_idx.find(idx_val) == _labels_idx.end())
            {
                _trainset_idx.insert(idx_val);
            }
        }
    }

    /**
     * \brief Construct a new Dataset object.
     * \param data          Mat The matrix dataset.
     * \param sequence_size SizeType The size of the sequence in number of
     *                      feature entry.
     * \param labels_idx    std::set<SizeType> The index of the feature to use
     *                      as ground truth.
     */
    Dataset(Mat data, SizeType sequence_size = 1,
            std::set<SizeType> labels_idx = {})
        : _data{}
        , _entry_labels_cache{}
        , _entry_trainset_cache{}
        , _feature_amount{data.size()}
        , _labels_idx{labels_idx}
        , _trainset_idx{}
    {
        _sequence_size = std::min(sequence_size, _feature_amount);
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
                    _data.insert(_data.end(), 
                        v.begin(), v.begin() + long(_feature_size));
                }
            }
        }
        _sequence_amount = _feature_amount 
            / std::max(_sequence_size, SizeType(1));
        _dataset_size = _sequence_amount * _sequence_size * _feature_size;
        _data.resize(_dataset_size);

        // Init indexes set.
        for (auto it = _labels_idx.begin(); it != _labels_idx.end(); )
        {
            if (*it >= _feature_size) _labels_idx.erase(it++); else ++it;
        }
        for (SizeType idx_val = 0; idx_val < _feature_size; ++idx_val)
        {
            if (_labels_idx.find(idx_val) == _labels_idx.end())
            {
                _trainset_idx.insert(idx_val);
            }
        }
    } 

    /**
     * \brief Construct a new Dataset object.
     * \param data       Cub The cube dataset.
     * \param labels_idx std::set<SizeType> The index of the feature to use
     * as ground truth.
     */
    Dataset(Cub data, std::set<SizeType> labels_idx = {})
        : _data{}
        , _entry_labels_cache{}
        , _entry_trainset_cache{}
        , _sequence_amount{data.size()}
        , _labels_idx{labels_idx}
        , _trainset_idx{}
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
                    for (SizeType i = 0; i < _sequence_size; ++i)
                    {
                        const auto& v = m[i];
                        _data.insert(_data.end(), v.begin(), 
                            v.begin() + long(_feature_size));
                    }
                }
            }
        }
        _feature_amount = _sequence_amount * _sequence_size;
        _dataset_size = _feature_amount * _feature_size;

        // Init indexes set.
        for (auto it = _labels_idx.begin(); it != _labels_idx.end(); )
        {
            if (*it >= _feature_size) _labels_idx.erase(it++); else ++it;
        }
        for (SizeType idx_val = 0; idx_val < _feature_size; ++idx_val)
        {
            if (_labels_idx.find(idx_val) == _labels_idx.end())
            {
                _trainset_idx.insert(idx_val);
            }
        }
    } 

    /**
     * \brief Destroy the Dataset object.
     */
    ~Dataset() {};

#if ENABLE_MLPACK
    /**
     * \brief Convert the dataset in Armadillo Vector.
     * \return arma::Col<T>
     */
    operator arma::Col<T>()
    {   
        arma::Col<T> ret(_data);
        return ret;
    }

    /**
     * \brief Convert the dataset in Armadillo Vector.
     * \return arma::Row<T>
     */
    operator arma::Row<T>()
    {   
        arma::Row<T> ret(_data);
        return ret;
    }

    /**
     * \brief Convert the dataset in Armadillo Matrix.
     * \return arma::Mat<T>
     */
    operator arma::Mat<T>()
    {   
        arma::Mat<T> ret(_data);
        ret.reshape(_feature_size, _feature_amount);
        return ret;
    }

    /**
     * \brief Convert the dataset in Armadillo Cube.
     * \return arma::Cube<T>
     */
    operator arma::Cube<T>()
    {   
        arma::Cube<T> ret(1, 1, _dataset_size);
        ret.row(0) = arma::conv_to<arma::Mat<T>>::from(_data);
        ret.reshape(_feature_size, _sequence_size, _sequence_amount);

        // // Uncomment to transpose the cube.
        // arma::Cube<T> ret_trans(
        //     _sequence_size, _feature_size, _sequence_amount);
        // for (SizeType s = 0; s < _sequence_amount; ++s)
        // {
        //     ret_trans.slice(s) = ret.slice(s).t();
        // } 
        // return ret_trans;

        return ret;
    }

    /**
     * \brief Convert the dataset in Armadillo format.
     * \tparam ARMA_T arma::Cube<T>, arma::Mat<T> or arma::Col<T>
     * \return ARMA_T arma::Cube<T>, arma::Mat<T> or arma::Col<T>
     */
    template<typename ARMA_T>
    ARMA_T to_arma()
    {
        return ARMA_T(*this);
    }
#endif // ENABLE_MLPACK

    /**
     * \brief Getter and setter of feature_size param.
     * \return const SizeType& The number of features, labels included, of the
     * dataset.
     */
    const SizeType& feature_size() const { return _feature_size; };

    /**
     * \brief Getter and setter of sequence_size param.
     * \return SizeType The size of the sequence of the resulting dataset.
     */
    void sequence_size(SizeType s)
    {
        _sequence_size = std::min(s, _feature_amount);
        _sequence_amount = _feature_amount / _sequence_size;
        _dataset_size = _sequence_size * _sequence_amount * _feature_size;
        _feature_amount = _dataset_size / _feature_size;
        _data.resize(_dataset_size);
    }
    const SizeType& sequence_size() const { return _sequence_size; };

    /**
     * \brief Return the number of entries of the dataset.
     * \return SizeType the number of entries of the dataset.
     */
    SizeType size() const { return _feature_amount; };

    /**
     * \brief Return if the dataset is empty.
     * \return True if the dataset is empty, else False.
     */
    [[nodiscard]] bool empty() const { return _data.empty(); };

    /**
     * \brief Return the data vector.
     * \return const std::vector<T>& The data vector reference.
     */
    const std::vector<T>& data() const { return _data; };

    /**
     * \brief Retrieve an entry of the dataset at index.
     * \param row_idx SizeType The index of the dataset.
     * \return const std::vector<T>& A reference vector to the dataset entry.
     */
    const std::vector<T>& entry(SizeType row_idx)
    {
        if (row_idx >= _feature_amount)
        {
            _entry_trainset_cache.clear();
            return _entry_trainset_cache;
        } 
        _entry_trainset_cache.resize(_feature_size);

        std::copy(
            _data.begin() + long(row_idx * _feature_size),
            _data.begin() + long(row_idx * _feature_size) + long(_feature_size),
            _entry_trainset_cache.begin());
        return _entry_trainset_cache;
    }

    /**
     * \brief Retrieve an entry sequence of the dataset at sequence index.
     * \param seq_idx SizeType The sequence index of the dataset.
     * \return const std::vector<T>& A reference vector to the dataset sequence.
     */
    const std::vector<T>& entry_seq(SizeType seq_idx)
    {
        if (seq_idx >= _sequence_amount)
        {
            _entry_trainset_cache.clear();
            return _entry_trainset_cache;
        } 
        _entry_trainset_cache.resize(_sequence_size * _feature_size);

        std::copy(
            _data.begin() + long(seq_idx * _sequence_size * _feature_size),
            _data.begin() + long(seq_idx * _sequence_size * _feature_size) 
                + long(_sequence_size * _feature_size),
            _entry_trainset_cache.begin());
        return _entry_trainset_cache;
    }

    /**
     * \brief Retrieve the indexes of the feature for training.
     * \return std::vector<SizeType> Vector of indexes of training features.
     */
    std::vector<SizeType> trainset_idx() const 
    { 
        return std::vector<SizeType>(
            _trainset_idx.begin(), _trainset_idx.end());
    }

    /**
     * \brief Retrieve an train feature entry of the dataset at row index.
     * \param row_idx SizeType The row index of the dataset.
     * \return const std::vector<T>& A reference vector to the dataset train
     * features.
     */
    const std::vector<T>& trainset(SizeType row_idx)
    {
        if (row_idx >= _feature_amount)
        {
            _entry_trainset_cache.clear();
            return _entry_trainset_cache;
        } 
        if (_trainset_idx.size() == _feature_size)
        {
            return entry(row_idx);
        }
        return _field_from_row_idx(
            _entry_trainset_cache, row_idx, _trainset_idx);
    }

    /**
     * \brief Retrieve the training feature part of the dataset.
     * \return Dataset<T> The training feature dataset.
     */
    Dataset<T> trainset()
    {
        auto ret = Vec(std::size_t(_trainset_idx.size() * _feature_amount));
        for (std::size_t row_i = 0; row_i < _feature_amount; ++row_i)
        {
            std::size_t trainset_col_i = 0;
            for (const auto& col_i: _trainset_idx)
            {
                ret[(row_i * _trainset_idx.size()) + trainset_col_i++]
                    = _data[(row_i * _feature_size) + col_i];
            }
        }
        return Dataset<T>(ret, _trainset_idx.size(), _sequence_size);
    }

    /**
     * \brief Retrieve a train feature sequence of the dataset at
     * sequence index.
     * \param seq_idx SizeType The sequence index of the dataset.
     * \return const std::vector<T>& A reference vector to the dataset train
     * features sequence.
     */
    const std::vector<T>& trainset_seq(SizeType seq_idx)
    {
        if (seq_idx >= _sequence_amount)
        {
            _entry_trainset_cache.clear();
            return _entry_trainset_cache;
        }
        if (_trainset_idx.size() == _feature_size)
        {
            return entry_seq(seq_idx);
        }
        return _field_from_seq_idx(
            _entry_trainset_cache, seq_idx, _trainset_idx);
    }

    /**
     * \brief Retrieve the indexes of the feature labels.
     * \return std::vector<SizeType> Vector of indexes of label features.
     */
    std::vector<SizeType> labels_idx() const 
    { 
        return {_labels_idx.begin(), _labels_idx.end()};
    }

    /**
     * \brief Setter of the feature labels indexes.
     * \param set std::set<SizeType> The set of indexes for feature labels.
     */
    void labels_idx(std::set<SizeType> set) 
    { 
        _labels_idx = set;
        for (auto it = _labels_idx.begin(); it != _labels_idx.end(); )
        {
            if (*it >= _feature_size) _labels_idx.erase(it++); else ++it;
        }
        _trainset_idx.clear();
        for (SizeType idx_val = 0; idx_val < _feature_size; ++idx_val)
        {
            if (_labels_idx.find(idx_val) == _labels_idx.end())
            {
                _trainset_idx.insert(idx_val);
            }
        }
    }

    /**
     * \brief Retrieve a label feature entry of the dataset at row index.
     * \param row_idx SizeType The row index of the dataset.
     * \return const std::vector<T>& A reference vector to the dataset label
     * features.
     */
    const std::vector<T>& labels(SizeType row_idx)
    {
        if (row_idx >= _feature_amount || _labels_idx.empty())
        {
            _entry_labels_cache.clear();
            return _entry_labels_cache;
        } 
        return _field_from_row_idx(_entry_labels_cache, row_idx, _labels_idx);
    }

    /**
     * \brief Retrieve the label feature part of the dataset.
     * \return Dataset<T> The label feature dataset.
     */
    Dataset<T> labels()
    {
        auto ret = Vec(std::size_t(_labels_idx.size() * _feature_amount));
        for (std::size_t row_i = 0; row_i < _feature_amount; ++row_i)
        {
            std::size_t labels_col_i = 0;
            for (const auto& col_i: _labels_idx)
            {
                ret[(row_i * _labels_idx.size()) + labels_col_i++]
                    = _data[(row_i * _feature_size) + col_i];
            }
        }
        return Dataset<T>(ret, _labels_idx.size(), _sequence_size);
    }

    /**
     * \brief Retrieve a label feature sequence of the dataset at
     * sequence index.
     * \param seq_idx SizeType The sequence index of the dataset.
     * \return const std::vector<T>& A reference vector to the dataset label
     * features sequence.
     */
    const std::vector<T>& labels_seq(SizeType seq_idx)
    {
        if (seq_idx >= _sequence_amount || _labels_idx.empty())
        {
            _entry_labels_cache.clear();
            return _entry_labels_cache;
        } 
        return _field_from_seq_idx(_entry_labels_cache, seq_idx, _labels_idx);
    }

    /**
     * \brief Retrieve a subsequence of the dataset from index to index.
     * \param from SizeType The index entry from take the subsequence.
     * \param to   SizeType The index entry to take the subsequence.
     * \return Dataset<T> The subsequence dataset from index to index.
     */
    Dataset<T> subdata(SizeType from, SizeType to)
    {
        if (to > _feature_amount)
        {
            to = _feature_amount;
        }

        if (from > to)
        {
            throw std::runtime_error("The argument 'from' exceeds 'to'");
        }

        auto subvector = Vec(
            _data.begin() + long(from * _feature_size),
            _data.begin() + long(to * _feature_size)
            );
        return Dataset<T>(subvector,
                          _feature_size, _sequence_size, _labels_idx);
    }

    /**
     * \brief Retrieve the first percentage of the dataset.
     * \param perc Percentage value from 0 to 1.
     * \return Dataset<T> The subsequence dataset in percentage.
     */
    Dataset<T> subdata(NumType perc)
    {
        if (perc > 1.0) perc = 1.0;
        if (perc < 0.0) perc = 0.0;
        auto to = static_cast<SizeType>(
            perc * static_cast<NumType>(_feature_amount));
        return subdata(0, to);
    }

    /**
     * \brief Split the dataset in training set and testing set, given a
     * percentage of the training set.
     * \param perc NumType Percentage of the training set.
     * \return SplitDataset Dataset split in training set and testing set.
     */
    SplitDataset split(NumType perc)
    {
        if (perc > 1.0) perc = 1.0;
        if (perc < 0.0) perc = 0.0;
        auto to = static_cast<SizeType>(perc
            * static_cast<NumType>(_feature_amount)
            / static_cast<NumType>(_sequence_size)) * _sequence_size;
        return SplitDataset(subdata(0, to), subdata(to, _feature_amount));
    }

    /**
     * \brief Shuffle the entries of the dataset.
     * \param rne RneType Random generator value.
     * \return Dataset<T>& The reference to this.
     */
    Dataset<T>& shuffle(RneType rne = RneType(std::random_device{}()))
    {
        auto shuffle_indexes = DLMath::unique_rand_sequence(
            0, _feature_amount, rne);
        std::vector<T> new_data(_data.size());

        for (std::size_t curr_idx = 0; curr_idx < _feature_amount; ++curr_idx)
        {
            auto new_idx =  shuffle_indexes[curr_idx];
            for (std::size_t i = 0; i < _feature_size; ++i)
            {
                new_data[new_idx * _feature_size + i]
                    = _data[curr_idx * _feature_size + i];
            }
        }

        _data = new_data;
        return *this;
    }

    Dataset<T>& min_max_normalization(T min, T max)
    {
        if (min == max) {
            throw std::runtime_error(
                "normalization error: min and max cannot be equal");
        }


        for (std::size_t row = 0; row < _feature_amount; ++row)
        {
            for (std::size_t col = 0; col < _feature_size; ++col)
            {
                _data[row * _feature_size + col]
                    = (_data[row * _feature_size + col] - min) / (max - min);
            }
        }
        return *this;
    }

    Dataset<T>& min_max_normalization()
    {
        std::vector<T> min_vec(_feature_size);
        std::vector<T> max_vec(_feature_size);

        // Init min and max vectors with first row.
        for (std::size_t col = 0; col < _feature_size; ++col)
        {
            min_vec[col] = _data[col];
            max_vec[col] = _data[col];
        }

        // Update min and max vectors.
        for (std::size_t row = 1; row < _feature_amount; ++row)
        {
            for (std::size_t col = 0; col < _feature_size; ++col)
            {
                auto idx = row * _feature_size + col;
                if (_data[idx] > max_vec[col]) max_vec[col] = _data[idx];
                if (_data[idx] < min_vec[col]) min_vec[col] = _data[idx];
            }
        }

        // Apply min max normalization.
        for (std::size_t row = 0; row < _feature_amount; ++row)
        {
            for (std::size_t col = 0; col < _feature_size; ++col)
            {
                _data[row * _feature_size + col]
                    = (_data[row * _feature_size + col] - min_vec[col])
                        / (max_vec[col] - min_vec[col]);
            }
        }
        return *this;
    }

    /**
     * \brief Parse a dataset from a given parser and label encoding.
     * Static method.
     * \param dataset_parser DatasetParser& The dataset to parse.
     * \param label_encoding DatasetParser::LabelEncoding The label encoding.
     * \param sequence_size  SizeType The sequence size of the return dataset.
     * \return Dataset<T> The resulting dataset parsed.
     */
    static Dataset<T> parse(
        DatasetParser& dataset_parser,
        DatasetParser::LabelEncoding label_encoding = DatasetParser::LabelEncoding::DEFAULT_ENCODING,
        SizeType sequence_size = 1)
    {
        return Dataset<T>(
                dataset_parser.data_to_encoding(label_encoding),
                dataset_parser.encoding_feature_size(label_encoding),
                sequence_size,
                dataset_parser.encoding_labels_idx(label_encoding)
            );
    }

private:
    /**
     * \brief Retrieve a part of the features from a dataset entry given by
     * the row index.
     * \param dst     std::vector<T>& The destination vector to fill with
     *                the retrieved values.
     * \param row_idx SizeType The row entry index.
     * \param set_idx const std::set<SizeType>& The set of feature indexes to
     *                retrieve.
     * \return const std::vector<T>& The reference to dst.
     */
    const std::vector<T>& _field_from_row_idx(
        std::vector<T>& dst, 
        SizeType row_idx, 
        const std::set<SizeType>& set_idx)
    {
        dst.resize(set_idx.size());
        auto data_entry_idx = row_idx * _feature_size;
        SizeType i = 0;
        for (const auto& idx: set_idx)
        {
            dst[i++] = _data[data_entry_idx + idx];
        }
        return dst;
    }

    /**
     * \brief Retrieve a part of the features from a dataset sequence given by
     * the sequence index.
     * \param dst     std::vector<T>& The destination vector to fill with
     *                the retrieved values.
     * \param seq_idx SizeType The sequence entry index.
     * \param set_idx const std::set<SizeType>& The set of feature indexes to
     *                retrieve.
     * \return const std::vector<T>& The reference to dst.
     */
    const std::vector<T>& _field_from_seq_idx(
        std::vector<T>& dst, 
        SizeType seq_idx, 
        const std::set<SizeType>& set_idx)
    {
        dst.resize(_sequence_size * set_idx.size());
        auto data_entry_idx = seq_idx * _sequence_size * _feature_size;
        SizeType i = 0;
        for (SizeType t = 0; t < _sequence_size; ++t)
        {
            for (const auto& idx: set_idx)
            {
                dst[i++] = _data[data_entry_idx + idx];
            }
            data_entry_idx += _feature_size;
        }
        return dst;
    }

    std::vector<T> _data;
    std::vector<T> _entry_labels_cache;
    std::vector<T> _entry_trainset_cache;

    /**
     * \brief The size of a single entry of the dataset, called feature.
     */
    SizeType _feature_size;

    /**
     * \brief The size of a sequence of features entry in the dataset. The
     * sequence size is in feature entries number granularity. 
     */
    SizeType _sequence_size;

    /**
     * \brief The size of the dataset, that is the quantity of data fields.
     * Element-wise granularity. 
     */
    SizeType _dataset_size;

    /**
     * \brief The number of feature entry of dimension `feature_size` in the
     * dataset. The feature amount is in feature entries number granularity. 
     */
    SizeType _feature_amount;

    /**
     * \brief The number of sequences in the dataset. The sequence amount is in
     * sequence number granularity. 
     */
    SizeType _sequence_amount;

    /**
     * \brief Set of indexes for the labels features.
     */
    std::set<SizeType> _labels_idx;

    /**
     * \brief Set of indexes for the training features.
     */
    std::set<SizeType> _trainset_idx;
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_MIDDLEWARE_DATASET_HPP
