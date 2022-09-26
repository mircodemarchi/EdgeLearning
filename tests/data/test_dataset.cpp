/***************************************************************************
 *            middleware/test_dataset.cpp
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
#include "data/dataset.hpp"

#include <vector>

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


class TestDataset {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_dataset_vec());
        EDGE_LEARNING_TEST_CALL(test_dataset_mat());
        EDGE_LEARNING_TEST_CALL(test_dataset_cub());
        EDGE_LEARNING_TEST_CALL(test_dataset_entry());
        EDGE_LEARNING_TEST_CALL(test_dataset_labels());
        EDGE_LEARNING_TEST_CALL(test_dataset_trainset());
        EDGE_LEARNING_TEST_CALL(test_dataset_parse());
        EDGE_LEARNING_TEST_CALL(test_dataset_shuffle());
        EDGE_LEARNING_TEST_CALL(test_dataset_normalization());
    }
private:

    void test_dataset_vec() {
        Dataset<double>::Vec data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        EDGE_LEARNING_TEST_TRY(Dataset<double> d(data));
        Dataset<double> d(data);
        EDGE_LEARNING_TEST_TRY(d.feature_size());
        EDGE_LEARNING_TEST_TRY(d.size());
        EDGE_LEARNING_TEST_TRY(d.sequence_size());

        EDGE_LEARNING_TEST_EQUAL(
            Dataset<double>(data, 2, 1).feature_size(), 2);
        EDGE_LEARNING_TEST_EQUAL(
            Dataset<double>(data, 10, 1).feature_size(), 10);
        EDGE_LEARNING_TEST_EQUAL(
            Dataset<double>(data, 15, 1).feature_size(), 10);

        EDGE_LEARNING_TEST_EQUAL(Dataset<double>(data, 2, 1).size(), 5);
        EDGE_LEARNING_TEST_EQUAL(Dataset<double>(data, 3, 1).size(), 3);
        EDGE_LEARNING_TEST_EQUAL(Dataset<double>(data, 4, 1).size(), 2);
        EDGE_LEARNING_TEST_EQUAL(Dataset<double>(data, 5, 1).size(), 2);
        EDGE_LEARNING_TEST_EQUAL(Dataset<double>(data, 6, 1).size(), 1);

        EDGE_LEARNING_TEST_EQUAL(
            Dataset<double>(data, 2, 1).data().size(), 10);
        EDGE_LEARNING_TEST_EQUAL(
            Dataset<double>(data, 3, 1).data().size(), 9);
        EDGE_LEARNING_TEST_EQUAL(
            Dataset<double>(data, 4, 1).data().size(), 8);
        EDGE_LEARNING_TEST_EQUAL(
            Dataset<double>(data, 5, 1).data().size(), 10);
        EDGE_LEARNING_TEST_EQUAL(
            Dataset<double>(data, 6, 1).data().size(), 6);

        EDGE_LEARNING_TEST_EQUAL(
            Dataset<double>(data, 2, 1).data()[9], 9);
        EDGE_LEARNING_TEST_EQUAL(
            Dataset<double>(data, 3, 1).data()[8], 8);
        EDGE_LEARNING_TEST_EQUAL(
            Dataset<double>(data, 4, 1).data()[7], 7);
        EDGE_LEARNING_TEST_EQUAL(
            Dataset<double>(data, 5, 1).data()[9], 9);
        EDGE_LEARNING_TEST_EQUAL(
            Dataset<double>(data, 6, 1).data()[5], 5);

        EDGE_LEARNING_TEST_EQUAL(
            Dataset<double>(data, 2, 2).sequence_size(), 2);
        EDGE_LEARNING_TEST_EQUAL(
            Dataset<double>(data, 3, 2).sequence_size(), 2);
        EDGE_LEARNING_TEST_EQUAL(
            Dataset<double>(data, 4, 2).sequence_size(), 2);
        EDGE_LEARNING_TEST_EQUAL(
            Dataset<double>(data, 6, 2).sequence_size(), 1);
        EDGE_LEARNING_TEST_EQUAL(
            Dataset<double>(data, 15, 2).sequence_size(), 1);

        EDGE_LEARNING_TEST_EQUAL(
            Dataset<double>(data, 2, 2).data().size(), 8);
        EDGE_LEARNING_TEST_EQUAL(
            Dataset<double>(data, 3, 2).data().size(), 6);
        EDGE_LEARNING_TEST_EQUAL(
            Dataset<double>(data, 15, 2).data().size(), 10);
        
        Dataset<double> ds(data, 2, 1);
        ds.sequence_size(2);
        EDGE_LEARNING_TEST_EQUAL(ds.feature_size(), 2);
        EDGE_LEARNING_TEST_EQUAL(ds.sequence_size(), 2);
        EDGE_LEARNING_TEST_EQUAL(ds.data().size(), 8);
        EDGE_LEARNING_TEST_EQUAL(ds.data()[7], 7);
        ds.sequence_size(3);
        EDGE_LEARNING_TEST_EQUAL(ds.data().size(), 6);
        EDGE_LEARNING_TEST_EQUAL(ds.sequence_size(), 3);

#if ENABLE_MLPACK
        auto arma_col = Dataset<double>(data).to_arma<arma::Col<double>>();
        EDGE_LEARNING_TEST_PRINT(arma_col);
        EDGE_LEARNING_TEST_EQUAL(arma_col.n_rows, 10);
        EDGE_LEARNING_TEST_EQUAL(arma_col.n_cols, 1);

        auto arma_row = Dataset<double>(data).to_arma<arma::Row<double>>();
        EDGE_LEARNING_TEST_PRINT(arma_row);
        EDGE_LEARNING_TEST_EQUAL(arma_row.n_rows, 1);
        EDGE_LEARNING_TEST_EQUAL(arma_row.n_cols, 10);
#endif // ENABLE_MLPACK

        Dataset<double> d_empty;
        EDGE_LEARNING_TEST_EQUAL(d_empty.feature_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_empty.sequence_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_empty.data().size(), 0);

        Dataset<double>::Vec data_empty1{};
        Dataset<double> d_empty1(data_empty1);
        EDGE_LEARNING_TEST_EQUAL(d_empty1.feature_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_empty1.sequence_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_empty1.data().size(), 0);

        Dataset<double> d_subdata(data, 2, 1);
        EDGE_LEARNING_TEST_FAIL(d_subdata.subdata(4, 2));
        EDGE_LEARNING_TEST_THROWS(d_subdata.subdata(4, 2),
                                  std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(
            d_subdata.subdata(1, 2).feature_size(), d_subdata.feature_size());
        EDGE_LEARNING_TEST_EQUAL(
            d_subdata.subdata(1, 2).sequence_size(), d_subdata.sequence_size());
        EDGE_LEARNING_TEST_ASSERT(d_subdata.subdata(1, 2).labels_idx().empty());
        EDGE_LEARNING_TEST_ASSERT(d_subdata.subdata(2, 2).data().empty());
        EDGE_LEARNING_TEST_EQUAL(d_subdata.subdata(0, 1).data().size(), 2);
        EDGE_LEARNING_TEST_EQUAL(d_subdata.subdata(0, 2).data().size(), 4);
        EDGE_LEARNING_TEST_EQUAL(
            d_subdata.subdata(0, d_subdata.size()).data().size(), 10);
        EDGE_LEARNING_TEST_EQUAL(
            d_subdata.subdata(0, 100).data().size(), 10);
        EDGE_LEARNING_TEST_EQUAL(d_subdata.subdata(0, 1).size(), 1);
        EDGE_LEARNING_TEST_EQUAL(d_subdata.subdata(0, 2).size(), 2);
        EDGE_LEARNING_TEST_EQUAL(
            d_subdata.subdata(0, d_subdata.size()).size(), 5);
        EDGE_LEARNING_TEST_EQUAL(d_subdata.subdata(0, 100).size(), 5);
        EDGE_LEARNING_TEST_EQUAL(
            d_subdata.subdata(0, 2).data()[0], d_subdata.data()[0]);
        EDGE_LEARNING_TEST_EQUAL(
            d_subdata.subdata(1, 2).data()[0], d_subdata.data()[2]);
        EDGE_LEARNING_TEST_EQUAL(
            d_subdata.subdata(1, 2).data()[1], d_subdata.data()[3]);
        EDGE_LEARNING_TEST_EQUAL(
            d_subdata.subdata(0, 100).data()[d_subdata.data().size() - 1],
            d_subdata.data()[d_subdata.data().size() - 1]);

        EDGE_LEARNING_TEST_EQUAL(d_subdata.subdata(0.4).size(), 2);
        EDGE_LEARNING_TEST_EQUAL(d_subdata.subdata(1.0).size(), 5);
        EDGE_LEARNING_TEST_EQUAL(d_subdata.subdata(1.5).size(), 5);
        EDGE_LEARNING_TEST_EQUAL(d_subdata.subdata(0.0).size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_subdata.subdata(-1.0).size(), 0);

        EDGE_LEARNING_TEST_TRY((void) d_subdata.split(0.4));
        auto d_split = d_subdata.split(0.4);
        EDGE_LEARNING_TEST_EQUAL(d_split.training_set.size(), 2);
        EDGE_LEARNING_TEST_EQUAL(d_split.testing_set.size(), 3);
        EDGE_LEARNING_TEST_TRY((void) d_subdata.split(1.0));
        d_split = d_subdata.split(1.0);
        EDGE_LEARNING_TEST_EQUAL(d_split.training_set.size(), 5);
        EDGE_LEARNING_TEST_EQUAL(d_split.testing_set.size(), 0);
        EDGE_LEARNING_TEST_TRY((void) d_subdata.split(1.5));
        d_split = d_subdata.split(1.5);
        EDGE_LEARNING_TEST_EQUAL(d_split.training_set.size(), 5);
        EDGE_LEARNING_TEST_EQUAL(d_split.testing_set.size(), 0);
        EDGE_LEARNING_TEST_TRY((void) d_subdata.split(0.0));
        d_split = d_subdata.split(0.0);
        EDGE_LEARNING_TEST_EQUAL(d_split.training_set.size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_split.testing_set.size(), 5);
        EDGE_LEARNING_TEST_TRY((void) d_subdata.split(-1.0));
        d_split = d_subdata.split(-1.0);
        EDGE_LEARNING_TEST_EQUAL(d_split.training_set.size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_split.testing_set.size(), 5);
    }

    void test_dataset_mat() {
        Dataset<double>::Mat data = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}};
        Dataset<double> d(data);
        EDGE_LEARNING_TEST_TRY(d.feature_size());
        EDGE_LEARNING_TEST_TRY(d.size());
        EDGE_LEARNING_TEST_TRY(d.sequence_size());

        EDGE_LEARNING_TEST_EQUAL(
            Dataset<double>(data, 1).feature_size(), 2);
        EDGE_LEARNING_TEST_EQUAL(Dataset<double>(data, 1).size(), 5);
        EDGE_LEARNING_TEST_EQUAL(
            Dataset<double>(data, 1).data().size(), 10);
        EDGE_LEARNING_TEST_EQUAL(
            Dataset<double>(data, 1).data()[9], 9);

        EDGE_LEARNING_TEST_EQUAL(
            Dataset<double>(data, 2).sequence_size(), 2);
        EDGE_LEARNING_TEST_EQUAL(
            Dataset<double>(data, 2).data().size(), 8);
        
        Dataset<double> ds(data, 1);
        ds.sequence_size(2);
        EDGE_LEARNING_TEST_EQUAL(ds.feature_size(), 2);
        EDGE_LEARNING_TEST_EQUAL(ds.sequence_size(), 2);
        EDGE_LEARNING_TEST_EQUAL(ds.data().size(), 8);
        EDGE_LEARNING_TEST_EQUAL(ds.data()[7], 7);
        ds.sequence_size(3);
        EDGE_LEARNING_TEST_EQUAL(ds.data().size(), 6);
        EDGE_LEARNING_TEST_EQUAL(ds.sequence_size(), 3);

#if ENABLE_MLPACK
        auto arma_col = Dataset<double>(data).to_arma<arma::Col<double>>();
        EDGE_LEARNING_TEST_PRINT(arma_col);
        EDGE_LEARNING_TEST_EQUAL(arma_col.n_rows, 10);
        EDGE_LEARNING_TEST_EQUAL(arma_col.n_cols, 1);

        auto arma_row = Dataset<double>(data).to_arma<arma::Row<double>>();
        EDGE_LEARNING_TEST_PRINT(arma_row);
        EDGE_LEARNING_TEST_EQUAL(arma_row.n_rows, 1);
        EDGE_LEARNING_TEST_EQUAL(arma_row.n_cols, 10);

        auto arma_mat = Dataset<double>(data).to_arma<arma::Mat<double>>();
        EDGE_LEARNING_TEST_PRINT(arma_mat);
        EDGE_LEARNING_TEST_EQUAL(arma_mat.n_rows, 2);
        EDGE_LEARNING_TEST_EQUAL(arma_mat.n_cols, 5);
#endif // ENABLE_MLPACK

        Dataset<double>::Mat data_empty1{};
        Dataset<double> d_empty1(data_empty1);
        EDGE_LEARNING_TEST_EQUAL(d_empty1.feature_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_empty1.sequence_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_empty1.data().size(), 0);

        Dataset<double> d_empty2((Dataset<double>::Mat({})));
        EDGE_LEARNING_TEST_EQUAL(d_empty2.feature_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_empty2.sequence_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_empty2.data().size(), 0);

        Dataset<double> d_subdata(data, 1);
        EDGE_LEARNING_TEST_FAIL(d_subdata.subdata(4, 2));
        EDGE_LEARNING_TEST_THROWS(d_subdata.subdata(4, 2),
                                  std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(
            d_subdata.subdata(1, 2).feature_size(), d_subdata.feature_size());
        EDGE_LEARNING_TEST_EQUAL(
            d_subdata.subdata(1, 2).sequence_size(), d_subdata.sequence_size());
        EDGE_LEARNING_TEST_ASSERT(d_subdata.subdata(1, 2).labels_idx().empty());
        EDGE_LEARNING_TEST_EQUAL(
            d_subdata.subdata(1, 2).data()[0], d_subdata.data()[2]);

        EDGE_LEARNING_TEST_EQUAL(d_subdata.subdata(0.4).size(), 2);
        EDGE_LEARNING_TEST_EQUAL(d_subdata.subdata(1.0).size(), 5);
        EDGE_LEARNING_TEST_EQUAL(d_subdata.subdata(1.5).size(), 5);
        EDGE_LEARNING_TEST_EQUAL(d_subdata.subdata(0.0).size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_subdata.subdata(-1.0).size(), 0);

        EDGE_LEARNING_TEST_TRY((void) d_subdata.split(0.4));
        auto d_split = d_subdata.split(0.4);
        EDGE_LEARNING_TEST_EQUAL(d_split.training_set.size(), 2);
        EDGE_LEARNING_TEST_EQUAL(d_split.testing_set.size(), 3);
        EDGE_LEARNING_TEST_TRY((void) d_subdata.split(1.0));
        d_split = d_subdata.split(1.0);
        EDGE_LEARNING_TEST_EQUAL(d_split.training_set.size(), 5);
        EDGE_LEARNING_TEST_EQUAL(d_split.testing_set.size(), 0);
        EDGE_LEARNING_TEST_TRY((void) d_subdata.split(1.5));
        d_split = d_subdata.split(1.5);
        EDGE_LEARNING_TEST_EQUAL(d_split.training_set.size(), 5);
        EDGE_LEARNING_TEST_EQUAL(d_split.testing_set.size(), 0);
        EDGE_LEARNING_TEST_TRY((void) d_subdata.split(0.0));
        d_split = d_subdata.split(0.0);
        EDGE_LEARNING_TEST_EQUAL(d_split.training_set.size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_split.testing_set.size(), 5);
        EDGE_LEARNING_TEST_TRY((void) d_subdata.split(-1.0));
        d_split = d_subdata.split(-1.0);
        EDGE_LEARNING_TEST_EQUAL(d_split.training_set.size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_split.testing_set.size(), 5);
    }

    void test_dataset_cub() {
        Dataset<double>::Cub data = {{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}};
        Dataset<double> d(data);
        EDGE_LEARNING_TEST_TRY(d.feature_size());
        EDGE_LEARNING_TEST_TRY(d.size());
        EDGE_LEARNING_TEST_TRY(d.sequence_size());

        EDGE_LEARNING_TEST_EQUAL(Dataset<double>(data).feature_size(), 2);
        EDGE_LEARNING_TEST_EQUAL(Dataset<double>(data).size(), 4);
        EDGE_LEARNING_TEST_EQUAL(Dataset<double>(data).data().size(), 8);
        EDGE_LEARNING_TEST_EQUAL(Dataset<double>(data).data()[7], 7);

        EDGE_LEARNING_TEST_EQUAL(Dataset<double>(data).sequence_size(), 2);
        
        Dataset<double> ds(data);
        ds.sequence_size(1);
        EDGE_LEARNING_TEST_EQUAL(ds.feature_size(), 2);
        EDGE_LEARNING_TEST_EQUAL(ds.sequence_size(), 1);
        EDGE_LEARNING_TEST_EQUAL(ds.data().size(), 8);
        EDGE_LEARNING_TEST_EQUAL(ds.data()[7], 7);
        ds.sequence_size(4);
        EDGE_LEARNING_TEST_EQUAL(ds.data().size(), 8);
        EDGE_LEARNING_TEST_EQUAL(ds.sequence_size(), 4);
        ds.sequence_size(3);
        EDGE_LEARNING_TEST_EQUAL(ds.data().size(), 6);
        EDGE_LEARNING_TEST_EQUAL(ds.sequence_size(), 3);
        ds.sequence_size(4);
        EDGE_LEARNING_TEST_EQUAL(ds.data().size(), 6);
        EDGE_LEARNING_TEST_EQUAL(ds.sequence_size(), 3);

#if ENABLE_MLPACK
        auto arma_col = Dataset<double>(data).to_arma<arma::Col<double>>();
        EDGE_LEARNING_TEST_PRINT(arma_col);
        EDGE_LEARNING_TEST_EQUAL(arma_col.n_rows, 8);
        EDGE_LEARNING_TEST_EQUAL(arma_col.n_cols, 1);

        auto arma_row = Dataset<double>(data).to_arma<arma::Row<double>>();
        EDGE_LEARNING_TEST_PRINT(arma_row);
        EDGE_LEARNING_TEST_EQUAL(arma_row.n_rows, 1);
        EDGE_LEARNING_TEST_EQUAL(arma_row.n_cols, 8);

        auto arma_mat = Dataset<double>(data).to_arma<arma::Mat<double>>();
        EDGE_LEARNING_TEST_PRINT(arma_mat);
        EDGE_LEARNING_TEST_EQUAL(arma_mat.n_rows, 2);
        EDGE_LEARNING_TEST_EQUAL(arma_mat.n_cols, 4);

        auto arma_cub = Dataset<double>(data).to_arma<arma::Cube<double>>();
        EDGE_LEARNING_TEST_PRINT(arma_cub);
        EDGE_LEARNING_TEST_EQUAL(arma_cub.n_rows, 2);
        EDGE_LEARNING_TEST_EQUAL(arma_cub.n_cols, 2);
        EDGE_LEARNING_TEST_EQUAL(arma_cub.n_slices, 2);

        Dataset<double>::Cub struct_data = {
            {{0, 1, 2, 4}, {1, 2, 3, 4}, {2, 3, 4, 5}}, 
            {{3, 4, 5, 6}, {4, 5, 6, 7}, {5, 6, 7, 8}}
        };
        auto arma_struct_cub = Dataset<double>(struct_data)
            .to_arma<arma::Cube<double>>();
        EDGE_LEARNING_TEST_PRINT(arma_struct_cub);
        EDGE_LEARNING_TEST_EQUAL(arma_struct_cub.n_rows, 4);
        EDGE_LEARNING_TEST_EQUAL(arma_struct_cub.n_cols, 3);
        EDGE_LEARNING_TEST_EQUAL(arma_struct_cub.n_slices, 2);

#endif // ENABLE_MLPACK

        Dataset<double>::Cub data_empty1{};
        Dataset<double> d_empty1(data_empty1);
        EDGE_LEARNING_TEST_EQUAL(d_empty1.feature_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_empty1.sequence_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_empty1.data().size(), 0);

        Dataset<double> d_empty2((Dataset<double>::Cub({})));
        EDGE_LEARNING_TEST_EQUAL(d_empty2.feature_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_empty2.sequence_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_empty2.data().size(), 0);

        Dataset<double> d_empty3((Dataset<double>::Cub({{}})));
        EDGE_LEARNING_TEST_EQUAL(d_empty3.feature_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_empty3.sequence_size(), 1);
        EDGE_LEARNING_TEST_EQUAL(d_empty3.data().size(), 0);

        Dataset<double> d_subdata(data);
        EDGE_LEARNING_TEST_FAIL(d_subdata.subdata(4, 2));
        EDGE_LEARNING_TEST_THROWS(d_subdata.subdata(4, 2),
                                  std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(
            d_subdata.subdata(1, 2).feature_size(), d_subdata.feature_size());
        EDGE_LEARNING_TEST_EQUAL(
            d_subdata.subdata(1, 2).sequence_size(), 1);
        EDGE_LEARNING_TEST_EQUAL(
            d_subdata.subdata(0, d_subdata.size()).sequence_size(),
            d_subdata.sequence_size());
        EDGE_LEARNING_TEST_ASSERT(d_subdata.subdata(1, 2).labels_idx().empty());
        EDGE_LEARNING_TEST_EQUAL(
            d_subdata.subdata(1, 2).data()[0], d_subdata.data()[2]);

        EDGE_LEARNING_TEST_EQUAL(d_subdata.subdata(0.25).size(), 1);
        EDGE_LEARNING_TEST_EQUAL(d_subdata.subdata(0.25).sequence_size(), 1);
        EDGE_LEARNING_TEST_EQUAL(d_subdata.subdata(0.5).size(), 2);
        EDGE_LEARNING_TEST_EQUAL(d_subdata.subdata(0.5).sequence_size(), 2);
        EDGE_LEARNING_TEST_EQUAL(d_subdata.subdata(1.0).size(), 4);
        EDGE_LEARNING_TEST_EQUAL(d_subdata.subdata(1.0).sequence_size(), 2);
        EDGE_LEARNING_TEST_EQUAL(d_subdata.subdata(1.5).size(), 4);
        EDGE_LEARNING_TEST_EQUAL(d_subdata.subdata(1.5).sequence_size(), 2);
        EDGE_LEARNING_TEST_EQUAL(d_subdata.subdata(0.0).size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_subdata.subdata(0.0).sequence_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_subdata.subdata(-1.0).size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_subdata.subdata(-1.0).sequence_size(), 0);

        EDGE_LEARNING_TEST_TRY((void) d_subdata.split(0.25));
        auto d_split = d_subdata.split(0.25);
        EDGE_LEARNING_TEST_EQUAL(d_split.training_set.size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_split.training_set.sequence_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_split.testing_set.size(), 4);
        EDGE_LEARNING_TEST_EQUAL(d_split.testing_set.sequence_size(), 2);
        EDGE_LEARNING_TEST_TRY((void) d_subdata.split(0.5));
        d_split = d_subdata.split(0.5);
        EDGE_LEARNING_TEST_EQUAL(d_split.training_set.size(), 2);
        EDGE_LEARNING_TEST_EQUAL(d_split.training_set.sequence_size(), 2);
        EDGE_LEARNING_TEST_EQUAL(d_split.testing_set.size(), 2);
        EDGE_LEARNING_TEST_EQUAL(d_split.testing_set.sequence_size(), 2);
        EDGE_LEARNING_TEST_TRY((void) d_subdata.split(1.0));
        d_split = d_subdata.split(1.0);
        EDGE_LEARNING_TEST_EQUAL(d_split.training_set.size(), 4);
        EDGE_LEARNING_TEST_EQUAL(d_split.training_set.sequence_size(), 2);
        EDGE_LEARNING_TEST_EQUAL(d_split.testing_set.size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_split.testing_set.sequence_size(), 0);
        EDGE_LEARNING_TEST_TRY((void) d_subdata.split(1.5));
        d_split = d_subdata.split(1.5);
        EDGE_LEARNING_TEST_EQUAL(d_split.training_set.size(), 4);
        EDGE_LEARNING_TEST_EQUAL(d_split.training_set.sequence_size(), 2);
        EDGE_LEARNING_TEST_EQUAL(d_split.testing_set.size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_split.testing_set.sequence_size(), 0);
        EDGE_LEARNING_TEST_TRY((void) d_subdata.split(0.0));
        d_split = d_subdata.split(0.0);
        EDGE_LEARNING_TEST_EQUAL(d_split.training_set.size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_split.training_set.sequence_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_split.testing_set.size(), 4);
        EDGE_LEARNING_TEST_EQUAL(d_split.testing_set.sequence_size(), 2);
        EDGE_LEARNING_TEST_TRY((void) d_subdata.split(-1.0));
        d_split = d_subdata.split(-1.0);
        EDGE_LEARNING_TEST_EQUAL(d_split.training_set.size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_split.training_set.sequence_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(d_split.testing_set.size(), 4);
        EDGE_LEARNING_TEST_EQUAL(d_split.testing_set.sequence_size(), 2);
    }

    void test_dataset_entry() {
        Dataset<double>::Vec data_vec = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        EDGE_LEARNING_TEST_TRY(Dataset<double> d(data_vec));
        Dataset<double> d_vec(data_vec, 2);

        EDGE_LEARNING_TEST_EQUAL(d_vec.entry(0).size(), 2);
        EDGE_LEARNING_TEST_ASSERT(d_vec.entry(5).empty());
        EDGE_LEARNING_TEST_EQUAL(d_vec.entry(2)[0], 4);
        EDGE_LEARNING_TEST_EQUAL(d_vec.entry(2)[1], 5);

        EDGE_LEARNING_TEST_EQUAL(d_vec.entry_seq(0).size(), 2);
        EDGE_LEARNING_TEST_ASSERT(d_vec.entry_seq(5).empty());
        EDGE_LEARNING_TEST_EQUAL(d_vec.entry_seq(2)[0], 4);
        EDGE_LEARNING_TEST_EQUAL(d_vec.entry_seq(2)[1], 5);
        d_vec.sequence_size(2);
        EDGE_LEARNING_TEST_EQUAL(d_vec.entry_seq(0).size(), 4);
        EDGE_LEARNING_TEST_ASSERT(d_vec.entry_seq(2).empty());
        EDGE_LEARNING_TEST_EQUAL(d_vec.entry_seq(1)[0], 4);
        EDGE_LEARNING_TEST_EQUAL(d_vec.entry_seq(1)[1], 5);

        Dataset<double>::Mat data_mat = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}};
        EDGE_LEARNING_TEST_TRY(Dataset<double> d(data_mat));
        Dataset<double> d_mat(data_mat);

        EDGE_LEARNING_TEST_EQUAL(d_mat.entry(0).size(), 2);
        EDGE_LEARNING_TEST_ASSERT(d_mat.entry(5).empty());
        EDGE_LEARNING_TEST_EQUAL(d_mat.entry(2)[0], 4);
        EDGE_LEARNING_TEST_EQUAL(d_mat.entry(2)[1], 5);

        EDGE_LEARNING_TEST_EQUAL(d_mat.entry_seq(0).size(), 2);
        EDGE_LEARNING_TEST_ASSERT(d_mat.entry_seq(5).empty());
        EDGE_LEARNING_TEST_EQUAL(d_mat.entry_seq(2)[0], 4);
        EDGE_LEARNING_TEST_EQUAL(d_mat.entry_seq(2)[1], 5);
        d_mat.sequence_size(2);
        EDGE_LEARNING_TEST_EQUAL(d_mat.entry_seq(0).size(), 4);
        EDGE_LEARNING_TEST_ASSERT(d_mat.entry_seq(2).empty());
        EDGE_LEARNING_TEST_EQUAL(d_mat.entry_seq(1)[0], 4);
        EDGE_LEARNING_TEST_EQUAL(d_mat.entry_seq(1)[1], 5);

        Dataset<double>::Cub data_cub = {{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}};
        EDGE_LEARNING_TEST_TRY(Dataset<double> d(data_cub));
        Dataset<double> d_cub(data_cub);

        EDGE_LEARNING_TEST_EQUAL(d_mat.entry(0).size(), 2);
        EDGE_LEARNING_TEST_ASSERT(d_mat.entry(4).empty());
        EDGE_LEARNING_TEST_EQUAL(d_mat.entry(2)[0], 4);
        EDGE_LEARNING_TEST_EQUAL(d_mat.entry(2)[1], 5);

        EDGE_LEARNING_TEST_EQUAL(d_mat.entry_seq(0).size(), 4);
        EDGE_LEARNING_TEST_ASSERT(d_mat.entry_seq(2).empty());
        EDGE_LEARNING_TEST_EQUAL(d_mat.entry_seq(1)[0], 4);
        EDGE_LEARNING_TEST_EQUAL(d_mat.entry_seq(1)[1], 5);
        d_mat.sequence_size(4);
        EDGE_LEARNING_TEST_EQUAL(d_mat.entry_seq(0).size(), 8);
        EDGE_LEARNING_TEST_ASSERT(d_mat.entry_seq(1).empty());
        EDGE_LEARNING_TEST_EQUAL(d_mat.entry_seq(0)[4], 4);
        EDGE_LEARNING_TEST_EQUAL(d_mat.entry_seq(0)[5], 5);
    }

    void test_dataset_labels() {
        Dataset<double>::Vec data_vec = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        EDGE_LEARNING_TEST_TRY(Dataset<double> d(data_vec, 2, 1, {1}));
        Dataset<double> d_vec(data_vec, 2, 1, {1});

        EDGE_LEARNING_TEST_EQUAL(d_vec.labels_idx().size(), 1);
        EDGE_LEARNING_TEST_EQUAL(d_vec.labels_idx()[0], 1);
        d_vec.labels_idx({1, 2, 3, 4, 5, 6});
        EDGE_LEARNING_TEST_EQUAL(d_vec.labels_idx().size(), 1);
        EDGE_LEARNING_TEST_EQUAL(d_vec.labels_idx()[0], 1);
        d_vec.labels_idx({0, 1, 2, 3, 4, 5, 6});
        EDGE_LEARNING_TEST_EQUAL(d_vec.labels_idx().size(), 2);
        EDGE_LEARNING_TEST_EQUAL(d_vec.labels_idx()[0], 0);
        EDGE_LEARNING_TEST_EQUAL(d_vec.labels_idx()[1], 1);
        d_vec.labels_idx({});
        EDGE_LEARNING_TEST_ASSERT(d_vec.labels_idx().empty());

        d_vec.labels_idx({0, 1});
        EDGE_LEARNING_TEST_EQUAL(d_vec.labels(0).size(), 2);
        EDGE_LEARNING_TEST_ASSERT(d_vec.labels(5).empty());
        EDGE_LEARNING_TEST_EQUAL(d_vec.labels(2)[0], 4);
        EDGE_LEARNING_TEST_EQUAL(d_vec.labels(2)[1], 5);
        d_vec.labels_idx({1});
        EDGE_LEARNING_TEST_EQUAL(d_vec.labels(0).size(), 1);
        EDGE_LEARNING_TEST_ASSERT(d_vec.labels(5).empty());
        EDGE_LEARNING_TEST_EQUAL(d_vec.labels(2)[0], 5);
        d_vec.labels_idx({0});
        EDGE_LEARNING_TEST_EQUAL(d_vec.labels(0).size(), 1);
        EDGE_LEARNING_TEST_ASSERT(d_vec.labels(5).empty());
        EDGE_LEARNING_TEST_EQUAL(d_vec.labels(2)[0], 4);
        d_vec.labels_idx({});
        EDGE_LEARNING_TEST_ASSERT(d_vec.labels(0).empty());

        d_vec.labels_idx({1});
        EDGE_LEARNING_TEST_EQUAL(d_vec.labels_seq(0).size(), 1);
        EDGE_LEARNING_TEST_ASSERT(d_vec.labels_seq(5).empty());
        EDGE_LEARNING_TEST_EQUAL(d_vec.labels_seq(2)[0], 5);
        d_vec.sequence_size(2);
        EDGE_LEARNING_TEST_EQUAL(d_vec.labels_seq(0).size(), 2);
        EDGE_LEARNING_TEST_ASSERT(d_vec.labels_seq(2).empty());
        EDGE_LEARNING_TEST_EQUAL(d_vec.labels_seq(1)[0], 5);
        EDGE_LEARNING_TEST_EQUAL(d_vec.labels_seq(1)[1], 7);
        d_vec.labels_idx({});
        EDGE_LEARNING_TEST_ASSERT(d_vec.labels_seq(0).empty());

        d_vec.labels_idx({1});
        EDGE_LEARNING_TEST_EQUAL(d_vec.labels().data()[0], d_vec.labels(0)[0]);
        d_vec.labels_idx({});
        EDGE_LEARNING_TEST_ASSERT(d_vec.labels().empty());

        Dataset<double>::Mat data_mat = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}};
        EDGE_LEARNING_TEST_TRY(Dataset<double> d(data_mat, 1, {1}));
        Dataset<double> d_mat(data_mat, 1, {1});

        EDGE_LEARNING_TEST_EQUAL(d_mat.labels_idx().size(), 1);
        EDGE_LEARNING_TEST_EQUAL(d_mat.labels_idx()[0], 1);
        d_mat.labels_idx({1, 2, 3, 4, 5, 6});
        EDGE_LEARNING_TEST_EQUAL(d_mat.labels_idx().size(), 1);
        EDGE_LEARNING_TEST_EQUAL(d_mat.labels_idx()[0], 1);
        d_mat.labels_idx({0, 1, 2, 3, 4, 5, 6});
        EDGE_LEARNING_TEST_EQUAL(d_mat.labels_idx().size(), 2);
        EDGE_LEARNING_TEST_EQUAL(d_mat.labels_idx()[0], 0);
        EDGE_LEARNING_TEST_EQUAL(d_mat.labels_idx()[1], 1);

        d_mat.labels_idx({0, 1});
        EDGE_LEARNING_TEST_EQUAL(d_mat.labels(0).size(), 2);
        EDGE_LEARNING_TEST_ASSERT(d_mat.labels(5).empty());
        EDGE_LEARNING_TEST_EQUAL(d_mat.labels(2)[0], 4);
        EDGE_LEARNING_TEST_EQUAL(d_mat.labels(2)[1], 5);
        d_mat.labels_idx({1});
        EDGE_LEARNING_TEST_EQUAL(d_mat.labels(0).size(), 1);
        EDGE_LEARNING_TEST_ASSERT(d_mat.labels(5).empty());
        EDGE_LEARNING_TEST_EQUAL(d_mat.labels(2)[0], 5);
        d_mat.labels_idx({0});
        EDGE_LEARNING_TEST_EQUAL(d_mat.labels(0).size(), 1);
        EDGE_LEARNING_TEST_ASSERT(d_mat.labels(5).empty());
        EDGE_LEARNING_TEST_EQUAL(d_mat.labels(2)[0], 4);
        d_mat.labels_idx({});
        EDGE_LEARNING_TEST_ASSERT(d_mat.labels(0).empty());

        d_mat.labels_idx({1});
        EDGE_LEARNING_TEST_EQUAL(d_mat.labels_seq(0).size(), 1);
        EDGE_LEARNING_TEST_ASSERT(d_mat.labels_seq(5).empty());
        EDGE_LEARNING_TEST_EQUAL(d_mat.labels_seq(2)[0], 5);
        d_mat.sequence_size(2);
        EDGE_LEARNING_TEST_EQUAL(d_mat.labels_seq(0).size(), 2);
        EDGE_LEARNING_TEST_ASSERT(d_mat.labels_seq(2).empty());
        EDGE_LEARNING_TEST_EQUAL(d_mat.labels_seq(1)[0], 5);
        EDGE_LEARNING_TEST_EQUAL(d_mat.labels_seq(1)[1], 7);
        d_mat.labels_idx({});
        EDGE_LEARNING_TEST_ASSERT(d_mat.labels_seq(0).empty());

        d_vec.labels_idx({1});
        EDGE_LEARNING_TEST_EQUAL(d_vec.labels().data()[0],
                                 d_vec.labels(0)[0]);
        d_vec.labels_idx({});
        EDGE_LEARNING_TEST_ASSERT(d_vec.labels().empty());

        Dataset<double>::Cub data_cub = {{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}};
        EDGE_LEARNING_TEST_TRY(Dataset<double> d(data_cub, {1}));
        Dataset<double> d_cub(data_cub, {1});

        EDGE_LEARNING_TEST_EQUAL(d_cub.labels_idx().size(), 1);
        EDGE_LEARNING_TEST_EQUAL(d_cub.labels_idx()[0], 1);
        d_cub.labels_idx({1, 2, 3, 4, 5, 6});
        EDGE_LEARNING_TEST_EQUAL(d_cub.labels_idx().size(), 1);
        EDGE_LEARNING_TEST_EQUAL(d_cub.labels_idx()[0], 1);
        d_cub.labels_idx({0, 1, 2, 3, 4, 5, 6});
        EDGE_LEARNING_TEST_EQUAL(d_cub.labels_idx().size(), 2);
        EDGE_LEARNING_TEST_EQUAL(d_cub.labels_idx()[0], 0);
        EDGE_LEARNING_TEST_EQUAL(d_cub.labels_idx()[1], 1);

        d_cub.labels_idx({0, 1});
        EDGE_LEARNING_TEST_EQUAL(d_cub.labels(0).size(), 2);
        EDGE_LEARNING_TEST_ASSERT(d_cub.labels(4).empty());
        EDGE_LEARNING_TEST_EQUAL(d_cub.labels(2)[0], 4);
        EDGE_LEARNING_TEST_EQUAL(d_cub.labels(2)[1], 5);
        d_cub.labels_idx({1});
        EDGE_LEARNING_TEST_EQUAL(d_cub.labels(0).size(), 1);
        EDGE_LEARNING_TEST_ASSERT(d_cub.labels(4).empty());
        EDGE_LEARNING_TEST_EQUAL(d_cub.labels(2)[0], 5);
        d_cub.labels_idx({0});
        EDGE_LEARNING_TEST_EQUAL(d_cub.labels(0).size(), 1);
        EDGE_LEARNING_TEST_ASSERT(d_cub.labels(4).empty());
        EDGE_LEARNING_TEST_EQUAL(d_cub.labels(2)[0], 4);
        d_cub.labels_idx({});
        EDGE_LEARNING_TEST_ASSERT(d_cub.labels(0).empty());

        d_cub.labels_idx({1});
        EDGE_LEARNING_TEST_EQUAL(d_cub.labels_seq(0).size(), 2);
        EDGE_LEARNING_TEST_ASSERT(d_cub.labels_seq(2).empty());
        EDGE_LEARNING_TEST_EQUAL(d_cub.labels_seq(1)[0], 5);
        d_cub.sequence_size(4);
        EDGE_LEARNING_TEST_EQUAL(d_cub.labels_seq(0).size(), 4);
        EDGE_LEARNING_TEST_ASSERT(d_cub.labels_seq(1).empty());
        EDGE_LEARNING_TEST_EQUAL(d_cub.labels_seq(0)[0], 1);
        EDGE_LEARNING_TEST_EQUAL(d_cub.labels_seq(0)[1], 3);
        d_cub.labels_idx({});
        EDGE_LEARNING_TEST_ASSERT(d_cub.labels_seq(0).empty());

        d_vec.labels_idx({1});
        EDGE_LEARNING_TEST_EQUAL(d_vec.labels().data()[0],
                                 d_vec.labels(0)[0]);
        EDGE_LEARNING_TEST_EQUAL(d_vec.labels().sequence_size(),
                                 d_vec.sequence_size());
        d_vec.labels_idx({});
        EDGE_LEARNING_TEST_ASSERT(d_vec.labels().empty());
    }

    void test_dataset_trainset() {
        Dataset<double>::Vec data_vec = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        EDGE_LEARNING_TEST_TRY(Dataset<double> d(data_vec, 2, 1, {1}));
        Dataset<double> d_vec(data_vec, 2, 1, {1});

        EDGE_LEARNING_TEST_EQUAL(d_vec.trainset_idx().size(), 1);
        EDGE_LEARNING_TEST_EQUAL(d_vec.trainset_idx()[0], 0);
        d_vec.labels_idx({1, 2, 3, 4, 5, 6});
        EDGE_LEARNING_TEST_EQUAL(d_vec.trainset_idx().size(), 1);
        EDGE_LEARNING_TEST_EQUAL(d_vec.trainset_idx()[0], 0);
        d_vec.labels_idx({0, 1, 2, 3, 4, 5, 6});
        EDGE_LEARNING_TEST_ASSERT(d_vec.trainset_idx().empty());
        d_vec.labels_idx({});
        EDGE_LEARNING_TEST_EQUAL(d_vec.trainset_idx().size(), 2);
        EDGE_LEARNING_TEST_EQUAL(d_vec.trainset_idx()[0], 0);
        EDGE_LEARNING_TEST_EQUAL(d_vec.trainset_idx()[1], 1);

        d_vec.labels_idx({});
        EDGE_LEARNING_TEST_EQUAL(
            d_vec.trainset(0).size(), d_vec.feature_size());
        EDGE_LEARNING_TEST_ASSERT(d_vec.trainset(5).empty());
        EDGE_LEARNING_TEST_EQUAL(d_vec.trainset(2)[0], 4);
        EDGE_LEARNING_TEST_EQUAL(d_vec.trainset(2)[1], 5);
        d_vec.labels_idx({0});
        EDGE_LEARNING_TEST_EQUAL(d_vec.trainset(0).size(), 1);
        EDGE_LEARNING_TEST_ASSERT(d_vec.trainset(5).empty());
        EDGE_LEARNING_TEST_EQUAL(d_vec.trainset(2)[0], 5);
        d_vec.labels_idx({1});
        EDGE_LEARNING_TEST_EQUAL(d_vec.trainset(0).size(), 1);
        EDGE_LEARNING_TEST_ASSERT(d_vec.trainset(5).empty());
        EDGE_LEARNING_TEST_EQUAL(d_vec.trainset(2)[0], 4);
        d_vec.labels_idx({0, 1});
        EDGE_LEARNING_TEST_ASSERT(d_vec.trainset(0).empty());

        d_vec.labels_idx({1});
        EDGE_LEARNING_TEST_EQUAL(d_vec.trainset_seq(0).size(), 1);
        EDGE_LEARNING_TEST_ASSERT(d_vec.trainset_seq(5).empty());
        EDGE_LEARNING_TEST_EQUAL(d_vec.trainset_seq(2)[0], 4);
        d_vec.sequence_size(2);
        EDGE_LEARNING_TEST_EQUAL(d_vec.trainset_seq(0).size(), 2);
        EDGE_LEARNING_TEST_ASSERT(d_vec.trainset_seq(2).empty());
        EDGE_LEARNING_TEST_EQUAL(d_vec.trainset_seq(1)[0], 4);
        EDGE_LEARNING_TEST_EQUAL(d_vec.trainset_seq(1)[1], 6);
        d_vec.labels_idx({});
        EDGE_LEARNING_TEST_EQUAL(
            d_vec.trainset_seq(0).size(), 
            d_vec.sequence_size() * d_vec.feature_size());

        d_vec.labels_idx({0});
        EDGE_LEARNING_TEST_EQUAL(d_vec.trainset().data()[0],
                                 d_vec.trainset(0)[0]);
        d_vec.labels_idx({0, 1});
        EDGE_LEARNING_TEST_ASSERT(d_vec.trainset().empty());

        Dataset<double>::Mat data_mat = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}};
        EDGE_LEARNING_TEST_TRY(Dataset<double> d(data_mat, 1, {1}));
        Dataset<double> d_mat(data_mat, 1, {1});

        EDGE_LEARNING_TEST_EQUAL(d_mat.trainset_idx().size(), 1);
        EDGE_LEARNING_TEST_EQUAL(d_mat.trainset_idx()[0], 0);
        d_mat.labels_idx({1, 2, 3, 4, 5, 6});
        EDGE_LEARNING_TEST_EQUAL(d_mat.trainset_idx().size(), 1);
        EDGE_LEARNING_TEST_EQUAL(d_mat.trainset_idx()[0], 0);
        d_mat.labels_idx({0, 1, 2, 3, 4, 5, 6});
        EDGE_LEARNING_TEST_ASSERT(d_mat.trainset_idx().empty());
        d_mat.labels_idx({});
        EDGE_LEARNING_TEST_EQUAL(d_mat.trainset_idx().size(), 2);
        EDGE_LEARNING_TEST_EQUAL(d_mat.trainset_idx()[0], 0);
        EDGE_LEARNING_TEST_EQUAL(d_mat.trainset_idx()[1], 1);

        d_mat.labels_idx({});
        EDGE_LEARNING_TEST_EQUAL(
            d_mat.trainset(0).size(), d_mat.feature_size());
        EDGE_LEARNING_TEST_ASSERT(d_mat.trainset(5).empty());
        EDGE_LEARNING_TEST_EQUAL(d_mat.trainset(2)[0], 4);
        EDGE_LEARNING_TEST_EQUAL(d_mat.trainset(2)[1], 5);
        d_mat.labels_idx({0});
        EDGE_LEARNING_TEST_EQUAL(d_mat.trainset(0).size(), 1);
        EDGE_LEARNING_TEST_ASSERT(d_mat.trainset(5).empty());
        EDGE_LEARNING_TEST_EQUAL(d_mat.trainset(2)[0], 5);
        d_mat.labels_idx({1});
        EDGE_LEARNING_TEST_EQUAL(d_mat.trainset(0).size(), 1);
        EDGE_LEARNING_TEST_ASSERT(d_mat.trainset(5).empty());
        EDGE_LEARNING_TEST_EQUAL(d_mat.trainset(2)[0], 4);
        d_mat.labels_idx({0, 1});
        EDGE_LEARNING_TEST_ASSERT(d_mat.trainset(0).empty());

        d_mat.labels_idx({1});
        EDGE_LEARNING_TEST_EQUAL(d_mat.trainset_seq(0).size(), 1);
        EDGE_LEARNING_TEST_ASSERT(d_mat.trainset_seq(5).empty());
        EDGE_LEARNING_TEST_EQUAL(d_mat.trainset_seq(2)[0], 4);
        d_mat.sequence_size(2);
        EDGE_LEARNING_TEST_EQUAL(d_mat.trainset_seq(0).size(), 2);
        EDGE_LEARNING_TEST_ASSERT(d_mat.trainset_seq(2).empty());
        EDGE_LEARNING_TEST_EQUAL(d_mat.trainset_seq(1)[0], 4);
        EDGE_LEARNING_TEST_EQUAL(d_mat.trainset_seq(1)[1], 6);
        d_mat.labels_idx({});
        EDGE_LEARNING_TEST_EQUAL(
            d_mat.trainset_seq(0).size(), 
            d_mat.sequence_size() * d_mat.feature_size());

        d_vec.labels_idx({0});
        EDGE_LEARNING_TEST_EQUAL(d_vec.trainset().data()[0],
                                 d_vec.trainset(0)[0]);
        d_vec.labels_idx({0, 1});
        EDGE_LEARNING_TEST_ASSERT(d_vec.trainset().empty());

        Dataset<double>::Cub data_cub = {{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}};
        EDGE_LEARNING_TEST_TRY(Dataset<double> d(data_cub, {1}));
        Dataset<double> d_cub(data_cub, {1});

        EDGE_LEARNING_TEST_EQUAL(d_cub.trainset_idx().size(), 1);
        EDGE_LEARNING_TEST_EQUAL(d_cub.trainset_idx()[0], 0);
        d_cub.labels_idx({1, 2, 3, 4, 5, 6});
        EDGE_LEARNING_TEST_EQUAL(d_cub.trainset_idx().size(), 1);
        EDGE_LEARNING_TEST_EQUAL(d_cub.trainset_idx()[0], 0);
        d_cub.labels_idx({0, 1, 2, 3, 4, 5, 6});
        EDGE_LEARNING_TEST_ASSERT(d_cub.trainset_idx().empty());
        d_cub.labels_idx({});
        EDGE_LEARNING_TEST_EQUAL(d_cub.trainset_idx().size(), 2);
        EDGE_LEARNING_TEST_EQUAL(d_cub.trainset_idx()[0], 0);
        EDGE_LEARNING_TEST_EQUAL(d_cub.trainset_idx()[1], 1);

        d_cub.labels_idx({});
        EDGE_LEARNING_TEST_EQUAL(d_cub.trainset(0).size(), 2);
        EDGE_LEARNING_TEST_ASSERT(d_cub.trainset(4).empty());
        EDGE_LEARNING_TEST_EQUAL(d_cub.trainset(2)[0], 4);
        EDGE_LEARNING_TEST_EQUAL(d_cub.trainset(2)[1], 5);
        d_cub.labels_idx({0});
        EDGE_LEARNING_TEST_EQUAL(d_cub.trainset(0).size(), 1);
        EDGE_LEARNING_TEST_ASSERT(d_cub.trainset(4).empty());
        EDGE_LEARNING_TEST_EQUAL(d_cub.trainset(2)[0], 5);
        d_cub.labels_idx({1});
        EDGE_LEARNING_TEST_EQUAL(d_cub.trainset(0).size(), 1);
        EDGE_LEARNING_TEST_ASSERT(d_cub.trainset(4).empty());
        EDGE_LEARNING_TEST_EQUAL(d_cub.trainset(2)[0], 4);
        d_cub.labels_idx({0, 1});
        EDGE_LEARNING_TEST_ASSERT(d_cub.trainset(0).empty());

        d_cub.labels_idx({1});
        EDGE_LEARNING_TEST_EQUAL(d_cub.trainset_seq(0).size(), 2);
        EDGE_LEARNING_TEST_ASSERT(d_cub.trainset_seq(2).empty());
        EDGE_LEARNING_TEST_EQUAL(d_cub.trainset_seq(1)[0], 4);
        d_cub.sequence_size(4);
        EDGE_LEARNING_TEST_EQUAL(d_cub.trainset_seq(0).size(), 4);
        EDGE_LEARNING_TEST_ASSERT(d_cub.trainset_seq(1).empty());
        EDGE_LEARNING_TEST_EQUAL(d_cub.trainset_seq(0)[0], 0);
        EDGE_LEARNING_TEST_EQUAL(d_cub.trainset_seq(0)[1], 2);
        d_cub.labels_idx({});
        EDGE_LEARNING_TEST_EQUAL(
            d_cub.trainset_seq(0).size(),
            d_cub.sequence_size() * d_cub.feature_size());

        d_vec.labels_idx({1});
        EDGE_LEARNING_TEST_EQUAL(d_vec.trainset().data()[0],
                                 d_vec.trainset(0)[0]);
        EDGE_LEARNING_TEST_EQUAL(d_vec.trainset().sequence_size(),
                                 d_vec.sequence_size());
        d_vec.labels_idx({0, 1});
        EDGE_LEARNING_TEST_ASSERT(d_vec.trainset().empty());
    }

    void test_dataset_parse()
    {
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
        auto edp = ExampleDatasetParser(v, feature_size, v_idx);

        EDGE_LEARNING_TEST_TRY(Dataset<NumType>::parse(edp));
        auto ds = Dataset<NumType>::parse(edp);
        EDGE_LEARNING_TEST_EQUAL(ds.feature_size(), feature_size);
        EDGE_LEARNING_TEST_EQUAL(ds.size(), 6);
        EDGE_LEARNING_TEST_EQUAL(ds.sequence_size(), 1);
        EDGE_LEARNING_TEST_EQUAL(ds.trainset(0).size(), 5);
        EDGE_LEARNING_TEST_EQUAL(ds.labels(0).size(), 2);
        auto truth_trainset_idx = std::vector<SizeType>({0,1,2,3,4});
        EDGE_LEARNING_TEST_EQUAL(truth_trainset_idx.size(), ds.trainset_idx().size());
        for (std::size_t i = 0; i < truth_trainset_idx.size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(truth_trainset_idx[i],
                                     ds.trainset_idx()[i]);
        }
        auto truth_labels_idx = std::vector<SizeType>({5,6});
        EDGE_LEARNING_TEST_EQUAL(truth_labels_idx.size(), ds.labels_idx().size());
        for (std::size_t i = 0; i < truth_labels_idx.size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(truth_labels_idx[i],
                                     ds.labels_idx()[i]);
        }

        EDGE_LEARNING_TEST_TRY(Dataset<NumType>::parse(
            edp, DatasetParser::LabelEncoding::ONE_HOT_ENCODING));
        ds = Dataset<NumType>::parse(
            edp, DatasetParser::LabelEncoding::ONE_HOT_ENCODING);
        EDGE_LEARNING_TEST_EQUAL(ds.feature_size(), 5+2+3);
        EDGE_LEARNING_TEST_EQUAL(ds.size(), 6);
        EDGE_LEARNING_TEST_EQUAL(ds.sequence_size(), 1);
        EDGE_LEARNING_TEST_EQUAL(ds.trainset(0).size(), 5);
        EDGE_LEARNING_TEST_EQUAL(ds.labels(0).size(), 2+3);
        EDGE_LEARNING_TEST_EQUAL(truth_trainset_idx.size(), ds.trainset_idx().size());
        for (std::size_t i = 0; i < truth_trainset_idx.size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(truth_trainset_idx[i],
                                     ds.trainset_idx()[i]);
        }
        truth_labels_idx = std::vector<SizeType>({5,6,7,8,9});
        EDGE_LEARNING_TEST_EQUAL(truth_labels_idx.size(), ds.labels_idx().size());
        for (std::size_t i = 0; i < truth_labels_idx.size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(truth_labels_idx[i],
                                     ds.labels_idx()[i]);
        }

        EDGE_LEARNING_TEST_TRY(Dataset<NumType>::parse(
            edp, DatasetParser::LabelEncoding::ONE_HOT_ENCODING, 2));
        ds = Dataset<NumType>::parse(
            edp, DatasetParser::LabelEncoding::ONE_HOT_ENCODING, 2);
        EDGE_LEARNING_TEST_EQUAL(ds.feature_size(), 5+2+3);
        EDGE_LEARNING_TEST_EQUAL(ds.size(), 6);
        EDGE_LEARNING_TEST_EQUAL(ds.sequence_size(), 2);
        EDGE_LEARNING_TEST_EQUAL(ds.trainset(0).size(), 5);
        EDGE_LEARNING_TEST_EQUAL(ds.labels(0).size(), 2+3);
        EDGE_LEARNING_TEST_EQUAL(ds.trainset_seq(0).size(), 5*2);
        EDGE_LEARNING_TEST_EQUAL(ds.labels_seq(0).size(), (2+3)*2);
        for (std::size_t i = 0; i < truth_trainset_idx.size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(truth_trainset_idx[i],
                                     ds.trainset_idx()[i]);
        }
        truth_labels_idx = std::vector<SizeType>({5,6,7,8,9});
        EDGE_LEARNING_TEST_EQUAL(truth_labels_idx.size(), ds.labels_idx().size());
        for (std::size_t i = 0; i < truth_labels_idx.size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(truth_labels_idx[i],
                                     ds.labels_idx()[i]);
        }
    }

    void test_dataset_shuffle() {
        std::vector<NumType> v({
            5, 5, 5, 5, 5, 0, 1,
            5, 5, 5, 5, 5, 1, 1,
            5, 5, 5, 5, 5, 0, 2,
            5, 5, 5, 5, 5, 1, 3,
            5, 5, 5, 5, 5, 0, 1,
            5, 5, 5, 5, 5, 0, 1,
        });
        std::set<SizeType> v_idx({5,6});
        SizeType feature_size = 7;
        auto ds = Dataset<NumType>(v, feature_size, 1, v_idx);
        Dataset<NumType> ds_copy = ds;

        EDGE_LEARNING_TEST_TRY(ds.shuffle());
        EDGE_LEARNING_TEST_EQUAL(ds.feature_size(), ds_copy.feature_size());
        EDGE_LEARNING_TEST_EQUAL(ds.size(), ds_copy.size());

        for (std::size_t i = 0; i < ds.size(); ++i)
        {
            auto train_entry = ds.trainset(i);
            for (const auto& e: train_entry)
            {
                EDGE_LEARNING_TEST_EQUAL(e, 5);
            }

            bool exist_label = false;
            auto label_entry = ds.labels(i);
            EDGE_LEARNING_TEST_EQUAL(label_entry.size(), 2);
            for (std::size_t i_ds_copy = 0; i_ds_copy < ds.size(); ++i_ds_copy)
            {
                auto label_entry_origin = ds_copy.labels(i_ds_copy);
                EDGE_LEARNING_TEST_EQUAL(label_entry_origin.size(), 2);
                if (label_entry_origin[0] == label_entry[0]
                    && label_entry_origin[1] == label_entry[1])
                {
                    exist_label = true;
                }
            }
            EDGE_LEARNING_TEST_ASSERT(exist_label);
        }

        for (std::size_t i = 0; i < ds.size(); ++i)
        {
            EDGE_LEARNING_TEST_PRINT(ds.labels(i)[0]);
            EDGE_LEARNING_TEST_PRINT(ds.labels(i)[1]);
        }
    }

    void test_dataset_normalization() {
        std::vector<NumType> v({
            0, 5, 5, 5, 5, 0, 1,
            5, 0, 5, 5, 5, 1, 1,
            5, 5, 0, 5, 5, 0, 2,
            5, 5, 5, 0, 5, 1, 3,
            5, 5, 5, 5, 0, 0, 1,
            5, 5, 5, 5, 5, 0, 1,
        });
        auto ds = Dataset<NumType>(v, 7, 1, {5,6});
        Dataset<NumType> ds_copy = ds;

        EDGE_LEARNING_TEST_TRY(ds.min_max_normalization(0, 5));
        EDGE_LEARNING_TEST_EQUAL(ds.feature_size(), ds_copy.feature_size());
        EDGE_LEARNING_TEST_EQUAL(ds.size(), ds_copy.size());
        EDGE_LEARNING_TEST_EQUAL(ds.sequence_size(), ds_copy.sequence_size());
        EDGE_LEARNING_TEST_EQUAL(ds.trainset_idx().size(),
                                 ds_copy.trainset_idx().size());
        for (const auto& e: ds.trainset().data())
        {
            EDGE_LEARNING_TEST_EQUAL(e, 1.0);
        }

        EDGE_LEARNING_TEST_FAIL(ds.min_max_normalization(0, 0));
        EDGE_LEARNING_TEST_THROWS(ds.min_max_normalization(0, 0),
                                  std::runtime_error);

        ds = ds_copy;
        EDGE_LEARNING_TEST_TRY(ds.min_max_normalization());
        for (const auto& e: ds.data()) {
            EDGE_LEARNING_TEST_ASSERT(0.0 <= e);
            EDGE_LEARNING_TEST_ASSERT(e <= 1.0);
        }
    }
};

int main() {
    TestDataset().test();
    return EDGE_LEARNING_TEST_FAILURES;
}



