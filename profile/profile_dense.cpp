/***************************************************************************
 *            profile_dense.cpp
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

#include "profile.hpp"

#include "dnn/dlmath.hpp"

#include <vector>
#include <tuple>
#include <functional>


class ProfileDense : public Profile {
public:

    struct Info {
        SizeType input_size;
        SizeType output_size;
    };

    ProfileDense() : Profile(100, "profile_dlmath_dense")
        , _seed(std::random_device{}())
    { }

    void run() {
        std::vector<Info> dense_params({
            {10,    10},
            {10,    100},
            {100,   100},
            {100,   1000},
            {1000,  1000},
            {1000,  10000},
            {10000, 10000},
        });

        profile_dense("sequential", DLMath::dense<NumType>, dense_params);
        profile_dense("thread_opt", DLMath::dense_thread_opt<NumType>, dense_params);
        profile_dense("simd_opt", DLMath::dense_simd_opt, dense_params);

        profile_dense_1("sequential", DLMath::dense_1<NumType>, dense_params);
        profile_dense_1("thread_opt", DLMath::dense_1_thread_opt<NumType>, dense_params);
    }

private:

    void profile_dense(
        std::string type,
        std::function<NumType*(NumType*,
                               const NumType*, const NumType*, const NumType*,
                               SizeType, SizeType)> dense_f,
        const std::vector<Info>& dense_params)
    {
        for (const auto& dense_p: dense_params)
        {
            std::vector<NumType> input(dense_p.input_size);
            std::vector<NumType> weights(
                dense_p.input_size * dense_p.output_size);
            std::vector<NumType> bias(
                dense_p.output_size);
            std::vector<NumType> output(dense_p.output_size);

            for (auto& e: input) e = DLMath::rand(-10, +10, _seed);
            for (auto& e: weights) e = DLMath::rand(-10, +10, _seed);
            for (auto& e: bias) e = DLMath::rand(-10, +10, _seed);

            profile(
                "dense math " + type +  " algorithm in forward with input_size="
                + std::to_string(dense_p.input_size) + " and output_size="
                + std::to_string(dense_p.output_size),
                [&](SizeType i) {
                    (void) i;
                    dense_f(output.data(), input.data(),
                            weights.data(), bias.data(),
                            dense_p.input_size, dense_p.output_size);
                },
                100,
                "dense_on_" + type + "_" + std::to_string(dense_p.input_size)
                + "x" + std::to_string(dense_p.output_size));
        }
    }

    void profile_dense_1(
        std::string type,
        std::function<NumType*(NumType*, NumType*, NumType*,
                               const NumType*, const NumType*, const NumType*,
                               SizeType, SizeType)> dense_f,
        const std::vector<Info>& dense_params)
    {
        for (const auto& dense_p: dense_params)
        {
            std::vector<NumType> gradients(
                dense_p.output_size);
            std::vector<NumType> input_gradients(
                dense_p.input_size);
            std::vector<NumType> weight_gradients(
                dense_p.input_size * dense_p.output_size);
            std::vector<NumType> bias_gradients(
                dense_p.output_size);
            std::vector<NumType> last_input(
                dense_p.input_size);
            std::vector<NumType> weights(
                dense_p.input_size * dense_p.output_size);

            for (auto& e: gradients) e = DLMath::rand(-10, +10, _seed);
            for (auto& e: last_input) e = DLMath::rand(-10, +10, _seed);
            for (auto& e: weights) e = DLMath::rand(-10, +10, _seed);

            profile(
                "dense math " + type +  " algorithm in backward with input_size="
                + std::to_string(dense_p.input_size) + " and output_size="
                + std::to_string(dense_p.output_size),
                [&](SizeType i) {
                    (void) i;
                    dense_f(input_gradients.data(), weight_gradients.data(),
                            bias_gradients.data(), gradients.data(),
                            last_input.data(), weights.data(),
                            dense_p.input_size, dense_p.output_size);
                },
                100,
                "dense_1_on_" + type + "_" + std::to_string(dense_p.input_size)
                + "x" + std::to_string(dense_p.output_size));
        }
    }

    RneType _seed;
};

int main() {
    ProfileDense().run();
}