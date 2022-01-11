/***************************************************************************
 *            tests/middleware/test_dataset.cpp
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
#include "middleware/dataset.hpp"

#include <vector>

using namespace std;
using namespace EdgeLearning;

class TestDataset {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_dataset_vec());
        EDGE_LEARNING_TEST_CALL(test_dataset_mat());
        EDGE_LEARNING_TEST_CALL(test_dataset_cub());
        EDGE_LEARNING_TEST_CALL(test_dataset_entry());
        EDGE_LEARNING_TEST_CALL(test_dataset_labels());
        EDGE_LEARNING_TEST_CALL(test_dataset_trainset());
    }
private:

    void test_dataset_vec() {
        Dataset<double>::Vec data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        EDGE_LEARNING_TEST_TRY(Dataset<double> d(data));
        Dataset<double> d(data);
        EDGE_LEARNING_TEST_TRY(d.feature_size());
        EDGE_LEARNING_TEST_TRY(d.size());
        EDGE_LEARNING_TEST_TRY(d.data().size());
        EDGE_LEARNING_TEST_TRY(d.data()[0]);
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
    }

    void test_dataset_mat() {
        Dataset<double>::Mat data = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}};
        Dataset<double> d(data);
        EDGE_LEARNING_TEST_TRY(d.feature_size());
        EDGE_LEARNING_TEST_TRY(d.size());
        EDGE_LEARNING_TEST_TRY(d.data().size());
        EDGE_LEARNING_TEST_TRY(d.data()[0]);
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
    }

    void test_dataset_cub() {
        Dataset<double>::Cub data = {{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}};
        Dataset<double> d(data);
        EDGE_LEARNING_TEST_TRY(d.feature_size());
        EDGE_LEARNING_TEST_TRY(d.size());
        EDGE_LEARNING_TEST_TRY(d.data().size());
        EDGE_LEARNING_TEST_TRY(d.data()[0]);
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
    }
};

int main() {
    TestDataset().test();
    return EDGE_LEARNING_TEST_FAILURES;
}



