/***************************************************************************
 *            parser/parser.cpp
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

#include "test.hpp"
#include "parser/parser.hpp"

#include <algorithm>


using namespace std;
using namespace EdgeLearning;


class ExampleDatasetParser : public DatasetParser
{
public:
    ExampleDatasetParser(const std::vector<NumType>& data,
                         SizeType feature_size,
                         std::set<SizeType> labels_idx = {})
        : _data(data)
        , _feature_size(feature_size)
        , _entries_amount(data.size() / std::max(feature_size, SizeType(1)))
        , _labels_idx(labels_idx)
    {

    }

    std::vector<NumType> entry(SizeType i) override {
        if (i >= _entries_amount) return {};

        std::vector<NumType> ret(_feature_size);
        auto offset = static_cast<std::int64_t>(i * _feature_size);
        std::copy(_data.begin() + offset,
                  _data.begin() + offset + static_cast<std::int64_t>(_feature_size),
                  ret.begin());
        return ret;
    }

    SizeType entries_amount() const override {
        return _entries_amount;
    }

    SizeType feature_size() const override {
        return _feature_size;
    }

    std::set<SizeType> labels_idx() const override {
        return _labels_idx;
    }

    std::vector<NumType> _data;
    SizeType _feature_size;
    SizeType _entries_amount;
    std::set<SizeType> _labels_idx;
};


class TestParser {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_parser());
        EDGE_LEARNING_TEST_CALL(test_dataset_parser());
    }

private:
    void test_parser() {
        EDGE_LEARNING_TEST_TRY(Parser());
    }

    void test_dataset_parser() {
        std::vector<NumType> v({
            5,5,5,5,5,0,1,
            5,5,5,5,5,1,1,
            5,5,5,5,5,0,2,
            5,5,5,5,5,1,3,
            5,5,5,5,5,0,1,
            5,5,5,5,5,0,1,
        });
        std::set<SizeType> v_idx({5,6});
        SizeType feature_size = 7;

        EDGE_LEARNING_TEST_TRY(ExampleDatasetParser(v, feature_size));
        auto edp = ExampleDatasetParser(v, feature_size);
        EDGE_LEARNING_TEST_EQUAL(edp.labels_idx().size(), 0);

        EDGE_LEARNING_TEST_TRY(ExampleDatasetParser(v, feature_size, v_idx));
        edp = ExampleDatasetParser(v, feature_size, v_idx);
        EDGE_LEARNING_TEST_EQUAL(edp.feature_size(), feature_size);
        EDGE_LEARNING_TEST_EQUAL(edp.entries_amount(), 6);
        EDGE_LEARNING_TEST_EQUAL(edp.entry(0).size(), feature_size);
        EDGE_LEARNING_TEST_EQUAL(edp.entry(1).size(), feature_size);
        EDGE_LEARNING_TEST_EQUAL(edp.entry(0)[0], 5);
        EDGE_LEARNING_TEST_EQUAL(edp.entry(edp.entries_amount()).size(), 0);
        EDGE_LEARNING_TEST_EQUAL(edp.labels_idx().size(), 2);

        EDGE_LEARNING_TEST_TRY((void) edp.unique(5));
        auto label5_values = edp.unique(5);
        for (const auto& val: label5_values)
        {
            EDGE_LEARNING_TEST_PRINT(val);
        }
        EDGE_LEARNING_TEST_ASSERT(label5_values.count(0) > 0);
        EDGE_LEARNING_TEST_ASSERT(label5_values.count(1) > 0);
        EDGE_LEARNING_TEST_ASSERT(label5_values.count(2) == 0);

        EDGE_LEARNING_TEST_TRY((void) edp.unique_map(6));
        auto label6_values_map = edp.unique_map(6);
        for (const auto& val: label6_values_map)
        {
            EDGE_LEARNING_TEST_PRINT(val.first);
            EDGE_LEARNING_TEST_PRINT(val.second);
        }
        EDGE_LEARNING_TEST_ASSERT(label6_values_map.count(0) == 0);
        EDGE_LEARNING_TEST_ASSERT(label6_values_map.count(1) > 0);
        EDGE_LEARNING_TEST_ASSERT(label6_values_map.count(2) > 0);
        EDGE_LEARNING_TEST_ASSERT(label6_values_map.count(3) > 0);
        EDGE_LEARNING_TEST_ASSERT(label6_values_map.count(4) == 0);
        EDGE_LEARNING_TEST_EQUAL(label6_values_map[1], 0);
        EDGE_LEARNING_TEST_EQUAL(label6_values_map[2], 1);
        EDGE_LEARNING_TEST_EQUAL(label6_values_map[3], 2);

        EDGE_LEARNING_TEST_TRY((void) edp.data_to_encoding(
            ExampleDatasetParser::LabelEncoding::DEFAULT_ENCODING));
        auto data_default_encoding = edp.data_to_encoding(
            ExampleDatasetParser::LabelEncoding::DEFAULT_ENCODING);
        EDGE_LEARNING_TEST_EQUAL(v.size(), data_default_encoding.size());
        EDGE_LEARNING_TEST_EQUAL(
            edp.encoding_feature_size(
                ExampleDatasetParser::LabelEncoding::DEFAULT_ENCODING),
                edp.feature_size());
        EDGE_LEARNING_TEST_EQUAL(
            edp.encoding_labels_idx(
                ExampleDatasetParser::LabelEncoding::DEFAULT_ENCODING).size(),
                edp.labels_idx().size());
        for (SizeType i = 0; i < v.size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(v[i], data_default_encoding[i]);
        }

        EDGE_LEARNING_TEST_TRY((void) edp.data_to_encoding(
            ExampleDatasetParser::LabelEncoding::ONE_HOT_ENCODING));
        auto data_one_hot_encoding = edp.data_to_encoding(
            ExampleDatasetParser::LabelEncoding::ONE_HOT_ENCODING);
        EDGE_LEARNING_TEST_EQUAL(edp.encoding_feature_size(
                ExampleDatasetParser::LabelEncoding::ONE_HOT_ENCODING), 5+2+3);
        EDGE_LEARNING_TEST_EQUAL(edp.encoding_labels_idx(
            ExampleDatasetParser::LabelEncoding::ONE_HOT_ENCODING).size(), 2+3);

        std::vector<NumType> test_v({
           5,5,5,5,5,1,0,1,0,0,
           5,5,5,5,5,0,1,1,0,0,
           5,5,5,5,5,1,0,0,1,0,
           5,5,5,5,5,0,1,0,0,1,
           5,5,5,5,5,1,0,1,0,0,
           5,5,5,5,5,1,0,1,0,0,
        });
        for (SizeType i = 0; i < test_v.size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(data_one_hot_encoding[i], test_v[i]);
        }
    }
};

int main() {
    TestParser().test();
    return EDGE_LEARNING_TEST_FAILURES;
}



