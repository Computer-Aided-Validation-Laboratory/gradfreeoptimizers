/**
 * @file ackley.hpp
 * @author Lavya D. Ghanwat (Pizzaman2629)
 * @brief Ackley example problem. Non-convex, many shallow local minima around one narrow global basin.
 *
 * f(x) = -a exp(-b sqrt(sum z_k^2 / d)) - exp(sum cos(c z_k) / d) + a + e,   z = x - shift
 * Domain: [-32.768, 32.768]^d. Minimum f = 0 at x_k = shift.
 *
 * @version 0.1
 * @date 2026-11-16
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once
#include <cstddef>
#include <cmath>
#include <numbers>

namespace problems {

/**
 * @brief Ackley function.
 *
 * @param x Position (d consecutive doubles, physical coordinates).
 * @param d Dimension.
 * @param shift Location of the minimum in every dimension.
 * @param a Depth of the outer envelope.
 * @param b Width of the outer envelope.
 * @param c Frequency of the ripples.
 * @return double
 */
inline double ackley(const double* x, size_t d, double shift = 2.0, double a = 20.0, double b = 0.2,
                     double c = 2.0 * std::numbers::pi){
    double sum1 {0.0};
    double sum2 {0.0};
    for(size_t k = 0; k < d; k++){
        double z {x[k] - shift};
        sum1 += z * z;
        sum2 += std::cos(c * z);
    }
    double dd {static_cast<double>(d)};
    return -a * std::exp(-b * std::sqrt(sum1 / dd)) - std::exp(sum2 / dd) + a + std::exp(1.0);
}

/**
 * @brief Ackley problem: parameters, default domain and known optimum. Callable as double(const double*, size_t).
 */
struct Ackley {
    static constexpr const char* name {"Ackley"};
    double shift {2.0};                     //Location of the minimum in every dimension.
    double a {20.0};                        //Depth of the outer envelope.
    double b {0.2};                         //Width of the outer envelope.
    double c {2.0 * std::numbers::pi};      //Frequency of the ripples.
    double lo {-32.768};                    //Default lower bound (every dimension).
    double hi {32.768};                     //Default upper bound (every dimension).

    double operator()(const double* x, size_t d) const { return ackley(x, d, shift, a, b, c); }
    double x_opt() const { return shift; }
    double f_opt() const { return 0.0; }
};

} //namespace problems
