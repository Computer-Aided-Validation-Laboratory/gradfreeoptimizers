/**
 * @file sphere.hpp
 * @author Lavya D. Ghanwat (Pizzaman2629)
 * @brief Sphere example problem. Convex, unimodal, separable.
 *
 * f(x) = sum_k (x_k - shift)^2
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
 * @brief Sphere function.
 *
 * @param x Position (d consecutive doubles, physical coordinates).
 * @param d Dimension.
 * @param shift Location of the minimum in every dimension. Shifted off the centre because swarms are biased towards it.
 * @return double
 */
inline double sphere(const double* x, size_t d, double shift = 2.0){
    double sum {0.0};
    for(size_t k = 0; k < d; k++){
        double z {x[k] - shift};
        sum += z * z;
    }
    return sum;
}

/**
 * @brief Sphere problem: parameters, default domain and known optimum. Callable as double(const double*, size_t).
 */
struct Sphere {
    static constexpr const char* name {"Sphere"};
    double shift {2.0};     //Location of the minimum in every dimension.
    double lo {-5.12};      //Default lower bound (every dimension).
    double hi {5.12};       //Default upper bound (every dimension).

    double operator()(const double* x, size_t d) const { return sphere(x, d, shift); }
    double x_opt() const { return shift; }      //Optimum coordinate (every dimension).
    double f_opt() const { return 0.0; }        //Optimum value.
};

} //namespace problems
