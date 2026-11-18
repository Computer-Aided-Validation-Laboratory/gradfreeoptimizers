/**
 * @file schwefel.hpp
 * @author Lavya D. Ghanwat (Pizzaman2629)
 * @brief Schwefel example problem. Non-convex and deceptive: the global minimum is far from the next best local minima.
 *
 * f(x) = 418.9829 d - sum_k z_k sin(sqrt(|z_k|)),   z = x - shift
 * Not shifted by default, its minimum is already far from the centre.
 * Domain: [-500, 500]^d. Minimum f = 0 (to about 1e-11 per dimension) at x_k = 420.9687462275036 + shift.
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

//Location of the minimum of z sin(sqrt|z|) per dimension, and the matching constant that makes the minimum 0.
inline constexpr double SCHWEFEL_X_OPT {420.9687462275036};
inline constexpr double SCHWEFEL_CONSTANT {418.982887272433799807913601398};

/**
 * @brief Schwefel function.
 *
 * @param x Position (d consecutive doubles, physical coordinates).
 * @param d Dimension.
 * @param shift Shift applied to every coordinate (0 = the standard function).
 * @return double
 */
inline double schwefel(const double* x, size_t d, double shift = 0.0){
    double sum {0.0};
    for(size_t k = 0; k < d; k++){
        double z {x[k] - shift};
        sum += z * std::sin(std::sqrt(std::fabs(z)));
    }
    return SCHWEFEL_CONSTANT * static_cast<double>(d) - sum;
}

/**
 * @brief Schwefel problem: parameters, default domain and known optimum. Callable as double(const double*, size_t).
 */
struct Schwefel {
    static constexpr const char* name {"Schwefel"};
    double shift {0.0};     //Shift applied to every coordinate.
    double lo {-500.0};     //Default lower bound (every dimension).
    double hi {500.0};      //Default upper bound (every dimension).

    double operator()(const double* x, size_t d) const { return schwefel(x, d, shift); }
    double x_opt() const { return SCHWEFEL_X_OPT + shift; }
    double f_opt() const { return 0.0; }
};

} //namespace problems
