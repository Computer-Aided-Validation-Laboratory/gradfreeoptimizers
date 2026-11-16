/**
 * @file exampleProblems.hpp
 * @author Lavya D. Ghanwat (Pizzaman2629)
 * @brief Every C++ example problem in one include.
 *
 * Each problem has its own header with two things in it:
 *  1. A free function with default valued parameters, e.g. problems::rastrigin(x, d, shift = 2.0, amplitude = 10.0).
 *  2. A small struct holding those parameters plus the default domain and known optimum, e.g. problems::Rastrigin.
 *     It is callable as double(const double* x, size_t d), so it can be passed straight to run_pso:
 *
 *          problems::Rastrigin problem;            //All defaults.
 *          problem.shift = 1.0;                    //Change what you like.
 *          std::vector<double> lo(d, problem.lo), hi(d, problem.hi);
 *          PSOResult result = run_pso(problem, d, lo, hi, settings);
 *
 * Problem      | Type                          | Default domain        | Minimum
 * -------------|-------------------------------|-----------------------|-------------------------------
 * Sphere       | convex, unimodal              | [-5.12, 5.12]^d       | f = 0 at x_k = shift (2)
 * Ellipsoid    | convex, ill conditioned       | [-5.12, 5.12]^d       | f = 0 at x_k = shift (2)
 * Rosenbrock   | non-convex, curved valley     | [-5, 10]^d            | f = 0 at x_k = shift + a (3)
 * Rastrigin    | non-convex, multimodal grid   | [-5.12, 5.12]^d       | f = 0 at x_k = shift (2)
 * Ackley       | non-convex, one narrow basin  | [-32.768, 32.768]^d   | f = 0 at x_k = shift (2)
 * Griewank     | non-convex, coupled           | [-600, 600]^d         | f = 0 at x_k = shift (100)
 * Schwefel     | non-convex, deceptive         | [-500, 500]^d         | f = 0 at x_k = 420.9687 + shift (0)
 *
 * Several problems are shifted so the minimum is not at the centre of the domain.
 * Swarms are known to be biased towards the centre of the search space, so an unshifted test at the origin can look better than it should.
 *
 * Every problem is a pure function of x, so it is safe to call from many threads at once (run_pso does).
 * Timings of the objective are collected by the optimizer itself (PSOSettings::diagnostics), for any objective.
 *
 * @version 0.1
 * @date 2026-11-16
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

//CONVEX.
#include "sphere.hpp"
#include "ellipsoid.hpp"

//NON-CONVEX.
#include "rosenbrock.hpp"
#include "rastrigin.hpp"
#include "ackley.hpp"
#include "griewank.hpp"
#include "schwefel.hpp"
