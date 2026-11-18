/**
 * @file griewank.hpp
 * @author Lavya D. Ghanwat (Pizzaman2629)
 * @brief Griewank example problem. Non-convex, many regularly spaced local minima, couples dimensions through a product.
 *
 * f(x) = sum_k z_k^2 / scale - prod_k cos(z_k / sqrt(k + 1)) + 1,   z = x - shift
 * Domain: [-600, 600]^d. Minimum f = 0 at x_k = shift.
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

namespace problems {

/**
 * @brief Griewank function.
 *
 * @param x Position (d consecutive doubles, physical coordinates).
 * @param d Dimension.
 * @param shift Location of the minimum in every dimension.
 * @param scale Divisor of the quadratic term (4000 in the standard form).
 * @return double
 */
inline double griewank(const double* x, size_t d, double shift = 100.0, double scale = 4000.0){
    double sum {0.0};
    double prod {1.0};
    for(size_t k = 0; k < d; k++){
        double z {x[k] - shift};
        sum += z * z / scale;
        prod *= std::cos(z / std::sqrt(static_cast<double>(k + 1)));
    }
    return sum - prod + 1.0;
}

/**
 * @brief Griewank problem: parameters, default domain and known optimum. Callable as double(const double*, size_t).
 */
struct Griewank {
    static constexpr const char* name {"Griewank"};
    double shift {100.0};       //Location of the minimum in every dimension.
    double scale {4000.0};      //Divisor of the quadratic term.
    double lo {-600.0};         //Default lower bound (every dimension).
    double hi {600.0};          //Default upper bound (every dimension).

    double operator()(const double* x, size_t d) const { return griewank(x, d, shift, scale); }
    double x_opt() const { return shift; }
    double f_opt() const { return 0.0; }
};

} //namespace problems
