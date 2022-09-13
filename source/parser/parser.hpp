/***************************************************************************
 *            parser/parser.hpp
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

/*! \file  parser/parser.hpp
 *  \brief Generic parser implementation.
 */

#ifndef EDGE_LEARNING_PARSER_PARSER_HPP
#define EDGE_LEARNING_PARSER_PARSER_HPP

#include "type_checker.hpp"
#include "type.hpp"

#include <set>
#include <map>


namespace EdgeLearning {

/**
 * \brief Generic class for parser the shares the TypeChecker entity.
 */
class Parser 
{
public:
    /**
     * \brief Construct a new Parser object.
     */
    Parser() : _tc() {}

    /**
     * \brief Destroy the Parser object.
     */
    virtual ~Parser() = default;

protected:
    TypeChecker _tc; ///< TypeChecker entity to perform parsing and convertion.
};


class DatasetParser : public Parser
{
public:

    enum class LabelEncoding {
        ONE_HOT_ENCODING,
        DEFAULT_ENCODING
    };

    using OneHotLabelMap = std::map<NumType, SizeType>;

    DatasetParser()
        : Parser()
    { }

    virtual ~DatasetParser() = default;

    virtual std::vector<NumType> entry(SizeType i) = 0;
    virtual SizeType entries_amount() const = 0;
    virtual SizeType feature_size() const = 0;
    virtual std::set<SizeType> labels_idx() const = 0;

    std::set<NumType> unique(SizeType idx)
    {
        std::vector<NumType> ret;

        for (SizeType i = 0; i < entries_amount(); ++i)
        {
            ret.push_back(entry(i)[idx]);
        }

        return {ret.begin(), ret.end()};
    }

    OneHotLabelMap unique_map(SizeType idx)
    {
        auto unique_set = unique(idx);
        OneHotLabelMap value_to_index;

        SizeType i = 0;
        for (const auto& l: unique_set)
        {
            value_to_index[l] = i++;
        }

        return value_to_index;
    }

    SizeType encoding_feature_size(
        LabelEncoding label_encoding = LabelEncoding::DEFAULT_ENCODING)
    {
        switch(label_encoding)
        {
            case LabelEncoding::ONE_HOT_ENCODING:
            {
                SizeType entry_size = feature_size() - labels_idx().size();
                for (const auto& label_idx: labels_idx())
                {
                    auto one_hot_label_map = unique(label_idx);
                    entry_size += one_hot_label_map.size();
                }
                return entry_size;
            }
            case LabelEncoding::DEFAULT_ENCODING:
            default:
            {
                return feature_size();
            }
        }
    }

    std::set<SizeType> encoding_labels_idx(
        LabelEncoding label_encoding = LabelEncoding::DEFAULT_ENCODING)
    {
        switch(label_encoding)
        {
            case LabelEncoding::ONE_HOT_ENCODING:
            {
                std::set<SizeType> encoding_label_indexes;
                auto encoding_entry_size = encoding_feature_size(label_encoding);
                SizeType trainset_idx_offset = feature_size() - labels_idx().size();
                for (SizeType i = trainset_idx_offset; i < encoding_entry_size; ++i)
                {
                    encoding_label_indexes.insert(i);
                }
                return encoding_label_indexes;
            }
            case LabelEncoding::DEFAULT_ENCODING:
            default:
            {
                return labels_idx();
            }
        }
    }

    std::vector<NumType> data_to_encoding(
        LabelEncoding label_encoding = LabelEncoding::DEFAULT_ENCODING)
    {
        switch(label_encoding)
        {
            case LabelEncoding::ONE_HOT_ENCODING:
            {
                auto label_indexes = labels_idx();
                std::vector<OneHotLabelMap> one_hot_label_maps;
                SizeType entry_size = feature_size() - label_indexes.size();
                for (const auto& label_idx: label_indexes)
                {
                    auto one_hot_label_map = unique_map(label_idx);
                    one_hot_label_maps.push_back(one_hot_label_map);
                    entry_size += one_hot_label_map.size();
                }

                std::vector<NumType> ret(entry_size * entries_amount());
                std::vector<SizeType> label_indexes_vec(
                    label_indexes.begin(), label_indexes.end());
                for (SizeType row_idx = 0; row_idx < entries_amount(); ++row_idx)
                {
                    auto row = entry(row_idx);

                    // Copy the trainset.
                    for (SizeType col_idx = 0; col_idx < feature_size(); ++col_idx)
                    {
                        if (label_indexes.find(col_idx) == label_indexes.end())
                        {
                            ret[row_idx * entry_size + col_idx] = row[col_idx];
                        }
                    }

                    // Copy the one hot vector for each label.
                    SizeType col_offset = feature_size() - label_indexes.size();
                    for (SizeType i = 0; i < label_indexes_vec.size(); ++i)
                    {
                        const OneHotLabelMap& one_hot_label_map = one_hot_label_maps[i];
                        auto label_value = row[label_indexes_vec[i]];

                        // Init the one hot label vector.
                        auto one_hot_size = one_hot_label_map.size();
                        auto one_hot_idx = one_hot_label_map.at(label_value);
                        for(SizeType col_idx = 0; col_idx < one_hot_size; ++col_idx)
                        {
                            ret[row_idx * entry_size + col_offset + col_idx] =
                                col_idx == one_hot_idx ? NumType(1) : NumType(0);
                        }

                        col_offset += one_hot_size;
                    }
                }
                return ret;
            }
            case LabelEncoding::DEFAULT_ENCODING:
            default:
            {
                std::vector<NumType> ret(feature_size() * entries_amount());
                for (SizeType row_idx = 0; row_idx < entries_amount(); ++row_idx)
                {
                    auto row = entry(row_idx);
                    for (SizeType col_idx = 0; col_idx < feature_size(); ++col_idx)
                    {
                        ret[row_idx * feature_size() + col_idx] = row[col_idx];
                    }
                }
                return ret;
            }
        }
    }


};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_PARSER_PARSER_HPP
