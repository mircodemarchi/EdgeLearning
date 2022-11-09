/***************************************************************************
 *            stopwatch.hpp
 *
 *  Copyright  2021  Luca Geretti
 *
 ****************************************************************************/

/*
 * This file is part of EdgeLearning, under the MIT license.
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

#ifndef EDGE_LEARNING_STOPWATCH_HPP
#define EDGE_LEARNING_STOPWATCH_HPP

#include "data/path.hpp"

#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>

namespace EdgeLearning {

using Seconds = std::chrono::seconds;
using Milliseconds = std::chrono::milliseconds;
using Microseconds = std::chrono::microseconds;

template<class D> class Stopwatch {
public:
    using ResolutionType = std::chrono::high_resolution_clock;
    using TimePointType = std::chrono::time_point<ResolutionType>;
    using DurationVecType = std::vector<double>;

    Stopwatch()
        : _initial()
        , _clicked()
        , _durations()
    {
        restart();
    }

    void reset()
    {
        _durations.clear();
        restart();
    }

    //! \brief Get the duration in the given type
    D duration() const
    {
        if (_durations.empty()) return -1;
        return _durations.back();
    }

    //! \brief Get the duration in seconds, in double precision
    [[nodiscard]] double elapsed_seconds() const
    {
        return std::chrono::duration_cast<std::chrono::duration<double>>(
            duration()).count();
    }

    //! \brief Restart the watch time to zero
    Stopwatch& restart()
    {
        _initial = ResolutionType::now();
        _clicked = _initial;
        return *this;
    }

    //! \brief Save the current time
    Stopwatch& click()
    {
        _clicked = ResolutionType::now();
        auto d = std::chrono::duration_cast<D>(_clicked - _initial);
        _durations.push_back(d.count());
        return *this;
    }

    template <typename R = double>
    R median() const
    {
        std::size_t mid = (_durations.size() - 1) / 2;
        DurationVecType durations_copy(_durations);
        std::nth_element(durations_copy.begin(),
                         durations_copy.begin() + static_cast<long>(mid),
                         durations_copy.end());
        return durations_copy[mid];
    }

    template <typename R = double>
    R mean() const
    {
        return std::accumulate(_durations.begin(), _durations.end(), R(0.0))
            / _durations.size();
    }

    template <typename R = double>
    R std() const
    {
        R m = mean<R>();
        std::vector<R> diff(_durations.size());
        std::transform(
            _durations.begin() , _durations.end(),
            diff.begin(),
            [m](R x) {
                return x - m;
            });
        R sqsum = std::inner_product(
            diff.begin(), diff.end(), diff.begin(), R(0.0));
        R ret = std::sqrt(sqsum / diff.size());
        return ret;
    }

    void dump(std::filesystem::path path, std::string header = "data")
    {
        std::ofstream f;
        if (!std::filesystem::exists(path)) {
            f.open(path);
            f << header << std::endl;
            f.close();
        }

        f.open(path, std::ofstream::out | std::ofstream::app);
        if(!f.is_open() || !f.good())
        {
            throw std::runtime_error("dump file open error");
        }
        for (std::size_t i = 0; i < _durations.size(); ++i)
        {
            f << _durations[i] << std::endl;
        }
        f.close();
    }

private:
    TimePointType _initial;
    TimePointType _clicked;
    DurationVecType _durations;
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_STOPWATCH_HPP