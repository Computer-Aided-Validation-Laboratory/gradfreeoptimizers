/**
 * @file rastrigin.hpp
 * @author Lavya D. Ghanwat (Pizzaman2629)
 * @brief Rastrigin example problem. Non-convex, highly multimodal with a regular grid of local minima.
 *
 * f(x) = A d + sum_k (z_k^2 - A cos(2 pi z_k)),   z = x - shift
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
#include <cmath>
#include <numbers>

namespace problems {

/**
 * @brief Rastrigin function.
 *
 * @param x Position (d consecutive doubles, physical coordinates).
 * @param d Dimension.
 * @param shift Location of the minimum in every dimension.
 * @param amplitude Height of the local minima ripples (A).
 * @return double
 */
inline double rastrigin(const double* x, size_t d, double shift = 2.0, double amplitude = 10.0){
    double sum {0.0};
    for(size_t k = 0; k < d; k++){
        double z {x[k] - shift};
        sum += z * z - amplitude * std::cos(2.0 * std::numbers::pi * z);
    }
    return amplitude * static_cast<double>(d) + sum;
}

/**
 * @brief Rastrigin problem: parameters, default domain and known optimum. Callable as double(const double*, size_t).
 */
struct Rastrigin {
    static constexpr const char* name {"Rastrigin"};
    double shift {2.0};         //Location of the minimum in every dimension.
    double amplitude {10.0};    //Height of the ripples (A).
    double lo {-5.12};          //Default lower bound (every dimension).
    double hi {5.12};           //Default upper bound (every dimension).

    double operator()(const double* x, size_t d) const { return rastrigin(x, d, shift, amplitude); }
    double x_opt() const { return shift; }
    double f_opt() const { return 0.0; }
};

} //namespace problems
