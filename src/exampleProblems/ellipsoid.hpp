/**
 * @file ellipsoid.hpp
 * @author Lavya D. Ghanwat (Pizzaman2629)
 * @brief Axis parallel hyper-ellipsoid example problem. Convex, unimodal, but ill conditioned (weights 1 to d).
 *
 * f(x) = sum_k (k + 1) (x_k - shift)^2
 * Domain: [-5.12, 5.12]^d. Minimum f = 0 at x_k = shift.
 *
 * @version 0.1
 * @date 2026-11-16
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once
#include <cstddef>

namespace problems {

/**
 * @brief Axis parallel hyper-ellipsoid function.
 *
 * @param x Position (d consecutive doubles, physical coordinates).
 * @param d Dimension.
 * @param shift Location of the minimum in every dimension.
 * @return double
 */
inline double ellipsoid(const double* x, size_t d, double shift = 2.0){
    double sum {0.0};
    for(size_t k = 0; k < d; k++){
        double z {x[k] - shift};
        sum += static_cast<double>(k + 1) * z * z;
    }
    return sum;
}

/**
 * @brief Ellipsoid problem: parameters, default domain and known optimum. Callable as double(const double*, size_t).
 */
struct Ellipsoid {
    static constexpr const char* name {"Ellipsoid"};
    double shift {2.0};     //Location of the minimum in every dimension.
    double lo {-5.12};      //Default lower bound (every dimension).
    double hi {5.12};       //Default upper bound (every dimension).

    double operator()(const double* x, size_t d) const { return ellipsoid(x, d, shift); }
    double x_opt() const { return shift; }
    double f_opt() const { return 0.0; }
};

} //namespace problems
