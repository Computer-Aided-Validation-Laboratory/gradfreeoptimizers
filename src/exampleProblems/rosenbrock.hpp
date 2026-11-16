/**
 * @file rosenbrock.hpp
 * @author Lavya D. Ghanwat (Pizzaman2629)
 * @brief Rosenbrock example problem. Non-convex, narrow curved valley, hard to converge along the valley floor.
 *
 * f(x) = sum_{k=0}^{d-2} b (z_{k+1} - z_k^2)^2 + (a - z_k)^2,   z = x - shift
 * Domain: [-5, 10]^d. Minimum f = 0 at x_k = shift + a.
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
 * @brief Rosenbrock function.
 *
 * @param x Position (d consecutive doubles, physical coordinates).
 * @param d Dimension (d = 1 has no terms and returns 0).
 * @param shift Shift applied to every coordinate.
 * @param a Valley floor location (the minimum is at z_k = a).
 * @param b Valley steepness.
 * @return double
 */
inline double rosenbrock(const double* x, size_t d, double shift = 2.0, double a = 1.0, double b = 100.0){
    double sum {0.0};
    for(size_t k = 0; k + 1 < d; k++){
        double z1 {x[k] - shift};
        double z2 {x[k + 1] - shift};
        double t1 {z2 - z1 * z1};
        double t2 {a - z1};
        sum += b * t1 * t1 + t2 * t2;
    }
    return sum;
}

/**
 * @brief Rosenbrock problem: parameters, default domain and known optimum. Callable as double(const double*, size_t).
 */
struct Rosenbrock {
    static constexpr const char* name {"Rosenbrock"};
    double shift {2.0};     //Shift applied to every coordinate.
    double a {1.0};         //Valley floor location.
    double b {100.0};       //Valley steepness.
    double lo {-5.0};       //Default lower bound (every dimension).
    double hi {10.0};       //Default upper bound (every dimension).

    double operator()(const double* x, size_t d) const { return rosenbrock(x, d, shift, a, b); }
    double x_opt() const { return shift + a; }
    double f_opt() const { return 0.0; }
};

} //namespace problems
