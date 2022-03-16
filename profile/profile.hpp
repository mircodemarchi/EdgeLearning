/***************************************************************************
 *            profile.hpp
 *
 *  Copyright  2021  Luca Geretti
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

#include <functional>
#include <iostream>
#include <sstream>
#include <utility>
#include <algorithm>
#include <numeric>

using namespace EdgeLearning;

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
    Profiler(SizeType num_tries) : _num_tries(num_tries) { }

    [[nodiscard]] SizeType num_tries() const { return _num_tries; }

    [[nodiscard]] Randomizer const& rnd() const { return _rnd; }

    std::vector<NsCount> profile(std::function<void(SizeType)> function, SizeType num_tries)
    {
        std::vector<NsCount> duration_times{};
        duration_times.resize(num_tries);
        std::cout << "profile tries: ";
        for (SizeType i = 0; i < num_tries; ++i) {
            std::cout << i << "..";
            _sw.restart();
            function(i);
            _sw.click();
            auto duration = static_cast<NsCount>(
                _sw.duration().count() * 1000 //< to ns
            );
            duration_times[i] = duration;
        }
        std::cout << std::endl;
        return duration_times;
    }

    std::vector<NsCount> profile(
        std::string msg,
        std::function<void(SizeType)> function,
        SizeType num_tries)
    {
        std::cout << msg << std::endl;
        auto cnt = profile(std::move(function), num_tries);
        std::cout << "completed " << _pretty_print(cnt) << std::endl;
        return cnt;
    }

    std::vector<NsCount> profile(std::string msg, std::function<void(SizeType)> function)
    {
        return profile(std::move(msg), std::move(function), _num_tries);
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

    [[nodiscard]] std::string
    _pretty_print(std::vector<NsCount> const& cnt) const
    {
        std::string ret = "[ ";
        for (std::size_t i = 0; i < cnt.size() - 1; ++i)
        {
            ret += _pretty_print(cnt[i]) + ", ";
        }
        ret += _pretty_print(cnt[cnt.size() - 1]) + "]";

        auto mean = std::accumulate(cnt.begin(), cnt.end(), 0.0) / cnt.size();
        ret = "{ mean: " + _pretty_print(static_cast<NsCount>(mean))
            + ", arr: " + ret + " }";
        return ret;
    }

private:
    Stopwatch<Microseconds> _sw;
    Randomizer _rnd;
    const SizeType _num_tries;
};

#endif // EDGE_LEARNING_PROFILE_HPP