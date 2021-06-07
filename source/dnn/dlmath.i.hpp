/***************************************************************************
 *            dlmath.hpp
 *
 *  Copyright  2021  Mirco De Marchi
 *
 ****************************************************************************/

/*
 *  This file is part of Ariadne.
 *
 *  Ariadne is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Ariadne is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Ariadne.  If not, see <https://www.gnu.org/licenses/>.
 */

/*! \file dlmath.hpp
 *  \brief Simply replace me.
 */

#ifndef ARIADNE_DL_DLMATH_HPP
#define ARIADNE_DL_DLMATH_HPP

#include <cmath>
#include <functional>


namespace Ariadne {

namespace dlmath {

/**
 * \brief Gaussian Probability Density Function.
 * @tparam T Input and output type.
 * \param x       Input value to compute.
 * \param mean    Mean of the probability distribution required.
 * \param std_dev Standard Deviation of the probability distribution required.
 * \return T
 */
template <typename T>
std::function<T(rne_t)> normal_pdf(float mean, float std_dev)
{
    static const double inv_sqrt_2pi = 0.3989422804014327;
    double inv_sqrt_2pi_std_dev = (inv_sqrt_2pi / std_dev);
    
    std::function<T(rne_t)> ret = [inv_sqrt_2pi_std_dev, mean, std_dev](rne_t x) 
    {
        double a = (double(x()) - mean) / std_dev;
        return T(inv_sqrt_2pi_std_dev * std::exp(-0.5f * a * a));
    };
    return ret;
}

} // namespace dlmath

} // namespace Ariadne

#endif // ARIADNE_DL_DLMATH_HPP
