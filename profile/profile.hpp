/***************************************************************************
 *            profile.hpp
 *
 *  Copyright  2021  Luca Geretti, Mirco De Marchi
 *
 ****************************************************************************/

/*
 * This file is part of Opera, under the MIT license.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef EDGE_LEARNING_PROFILE_HPP
#define EDGE_LEARNING_PROFILE_HPP

#include "stopwatch.hpp"
#include "type.hpp"
#include "data/dataset.hpp"
#include "data/path.hpp"
#include "middleware/fnn.hpp"

#include <functional>
#include <iostream>
#include <sstream>
#include <utility>
#include <algorithm>
#include <numeric>

using namespace EdgeLearning;

#define EDGE_LEARNING_PROFILE_TITLE(comment)                                   \
    {                                                                          \
        std::cout << "****************************************\n"              \
                  << comment << "\n"                                           \
                  << "****************************************\n" << std::endl;\
    }

#define EDGE_LEARNING_PROFILE_CALL(function)                                   \
    {                                                                          \
        std::cout << "*** PROFILING " << #function << " ***" << std::endl;     \
        try {                                                                  \
            function;                                                          \
        } catch(const std::exception& except) {                                \
            std::cout << "ERROR: exception '" << except.what() << "' in "      \
                      << #function << ": "                                     \
                      << except.what() << std::endl;                           \
            std::cerr << "ERROR: " << __FILE__ << ":"                          \
                      << __LINE__ << ": calling "                              \
                      << #function << ": " << except.what() << std::endl;      \
            std::cout << std::endl;                                            \
        }                                                                      \
    }                                                                          \

struct Randomizer {
    static NumType get(double min, double max) {
        return (max - min) * rand() / RAND_MAX + min;
    }
};

inline bool _init_randomizer() {
    srand(static_cast<unsigned int>(time(nullptr)));
    return true;
}
static const bool init_randomizer = _init_randomizer();

using NsCount = long long unsigned int;

class Profiler {
public:
    Profiler(SizeType num_tries, std::string name = std::string())
        : _num_tries(num_tries)
        , _name(name.empty() ? "profiler" : name)
    {
        std::filesystem::path path(_name);
        if (std::filesystem::exists(path))
        {
            if (!std::filesystem::is_directory(path))
            {
                throw std::runtime_error("path is not a directory");
            }
        }
        else
        {
            std::filesystem::create_directories(path);
        }
    }

    [[nodiscard]] SizeType num_tries() const { return _num_tries; }

    [[nodiscard]] Randomizer const& rnd() const { return _rnd; }

    void profile(
        std::function<void(SizeType)> function, SizeType num_tries)
    {
        for (SizeType i = 0; i < num_tries; ++i) {
            _sw.restart();
            function(i);
            _sw.click();
        }
    }

    void profile(
        std::string msg,
        std::function<void(SizeType)> function,
        SizeType num_tries, std::string profile_name)
    {
        std::cout << msg << std::endl;
        profile(std::move(function), num_tries);
        std::cout << "completed " << _pretty_print() << std::endl;
        std::string fn(profile_name + ".csv");
        _sw.dump(std::filesystem::path(_name) / fn, "time");
        _sw.reset();
    }

    void profile(std::string msg, std::function<void(SizeType)> function,
                 std::string profile_name)
    {
        profile(std::move(msg), std::move(function), _num_tries, profile_name);
    }

private:

    [[nodiscard]] std::string _pretty_print(NsCount const& cnt) const
    {
        std::stringstream ss;
        if (cnt < 1000)
            ss << cnt << " ns";
        else if (cnt < 1000000)
            ss << static_cast<NumType>(cnt)/1000 << " us";
        else if (cnt < 1000000000)
            ss << static_cast<NumType>(cnt)/1000000 << " ms";
        else if (cnt < 1000000000000)
            ss << static_cast<NumType>(cnt)/1000000000 << " sec";
        else
            ss << static_cast<NumType>(cnt)/60000000000 << " min";
        return ss.str();
    }

    [[nodiscard]] std::string _pretty_print()
    {
        auto mean = static_cast<NsCount>(_sw.mean() * 1000);
        auto median = static_cast<NsCount>(_sw.median() * 1000);
        auto std = static_cast<NsCount>(_sw.std() * 1000);
        return "mean: " + _pretty_print(mean)
            + " median: " +  _pretty_print(median)
            + " std: " +  _pretty_print(std);
    }

    Stopwatch<Microseconds> _sw;
    Randomizer _rnd;
    const SizeType _num_tries;
    std::string _name;
};


class DNNProfiler : public Profiler {
public:
    DNNProfiler(SizeType num_tries, std::string name = std::string())
        : Profiler(num_tries, name)
    { }

    template<typename Model>
    void training(std::string info, std::string profile_name,
                  SizeType iteration_amount,
                  Dataset<NumType>& data_training,
                  Dataset<NumType>& data_evaluation,
                  const LayerDescriptorVector& layers_descriptor,
                  SizeType epochs,
                  SizeType batch_size,
                  NumType learning_rate)
    {
        profile(
            info,
            [&](SizeType i){
                (void) i;
                Model m(layers_descriptor, "training_profiling_model");
                m.fit(data_training, epochs, batch_size, learning_rate);
                auto metrics = m.evaluate(data_evaluation);
                std::cout
                    << "evaluation: { "
                    << " accuracy: " << metrics.accuracy_perc << "%, "
                    << " error_rate: " << metrics.error_rate_perc << "%, "
                    << " avg_loss: " << metrics.loss << ", "
                    << " } "<< std::endl;
            },
            iteration_amount, profile_name);
    }

    template<typename Model>
    void predict(std::string info, std::string profile_name,
                 SizeType iteration_amount,
                 Dataset<NumType>& data,
                 const LayerDescriptorVector& layers_descriptor,
                 SizeType epochs,
                 SizeType batch_size,
                 NumType learning_rate)
    {
        Model m(layers_descriptor, "predict_profiling_model");
        m.fit(data, epochs, batch_size, learning_rate);
        profile(
            info,
            [&](SizeType i){
                (void) i;
                auto input = data.trainset();
                auto prediction = m.predict(input);
                (void) prediction;
            },
            iteration_amount,
            profile_name);
    }
};

#endif // EDGE_LEARNING_PROFILE_HPP