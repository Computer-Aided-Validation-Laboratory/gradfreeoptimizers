/**
 * @file main.cpp
 * @author Lavya D. Ghanwat (Pizzaman2629)
 * @brief C++ example: using the particle swarm optimizer and the example problems without Python.
 *
 * Build:  python scripts/build.py --example-only     ->  build/gfo_example(.exe)
 * Plot:   python scripts/plot_optimizer.py -r cpp_example_run   (after running the example)
 *
 * @version 0.1
 * @date 2026-11-16
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "pso/pso.hpp"
#include "exampleProblems/exampleProblems.hpp"


//Runs one example problem with its default domain and prints one summary line.
template <typename Problem>
void run_summary(const Problem& problem, size_t d, const PSOSettings& settings){
    std::vector<double> lo(d, problem.lo), hi(d, problem.hi);
    PSOResult result {run_pso(problem, d, lo, hi, settings)};
    std::cout << " | " << std::left << std::setw(11) << Problem::name << std::right << " | d = " << std::setw(2) << d
              << " | best value = " << std::scientific << std::setprecision(3) << result.best_val
              << " | error = " << result.best_val - problem.f_opt()
              << " | time = " << std::fixed << std::setprecision(4) << result.diagnostics.total_s << " s |" << std::endl;
    std::cout << std::defaultfloat << std::setprecision(6);
}


int main(){

    auto start {std::chrono::steady_clock::now()};

    //---------------------------------------------------------------------------
    //1. ONE RUN WITH EVERYTHING SWITCHED ON.
    //Verbose and diagnostics output, and the files gfo.post needs (loss.csv, and history.csv/domain.csv because d = 2).
    {
        problems::Rastrigin rastrigin;                  //All defaults: shift 2, amplitude 10, domain [-5.12, 5.12].
        const size_t d {2};
        std::vector<double> lo(d, rastrigin.lo), hi(d, rastrigin.hi);

        PSOSettings settings;                           //Every field has a default, set only what you need.
        settings.iterations = 200;
        settings.particles = 24;
        settings.verbose = true;
        settings.verbose_interval = 20;
        settings.diagnostics = true;
        settings.run_dir = "cpp_example_run";           //Files go to cpp_example_run/optimizer/.
        settings.save_loss = true;
        settings.save_history = true;

        PSOResult result {run_pso(rastrigin, d, lo, hi, settings)};
        std::cout << " | Known minimum: x = " << rastrigin.x_opt() << " in every dimension, f = " << rastrigin.f_opt() << "\n" << std::endl;
    }

    //---------------------------------------------------------------------------
    //2. CHANGING A PROBLEM'S PARAMETERS.
    //Every parameter has a default, change what you like (C++20 designated initializers work too).
    {
        problems::Rosenbrock valley {.shift = 0.0, .b = 50.0};
        const size_t d {5};
        std::vector<double> lo(d, valley.lo), hi(d, valley.hi);

        PSOSettings settings;
        settings.iterations = 1000;
        settings.particles = 30;
        PSOResult result {run_pso(valley, d, lo, hi, settings)};
        std::cout << " | Rosenbrock (shift 0, b 50), d = 5: best value = " << result.best_val
                  << " at x[0] = " << result.best_pos[0] << " (expected " << valley.x_opt() << ")\n" << std::endl;
    }

    //---------------------------------------------------------------------------
    //3. YOUR OWN OBJECTIVE.
    //Anything callable as double f(const double* x, size_t d) works: a function, a lambda, a functor.
    //It is called from several threads at once, so it must not modify shared state.
    {
        auto bowl = [](const double* x, size_t d){
            double sum {0.0};
            for(size_t k = 0; k < d; k++) sum += std::pow(x[k] - 0.5, 4);
            return sum;
        };
        const size_t d {3};
        std::vector<double> lo(d, -1.0), hi(d, 1.0);

        PSOSettings settings;
        settings.iterations = 300;
        PSOResult result {run_pso(bowl, d, lo, hi, settings)};
        std::cout << " | Custom quartic bowl, d = 3: best value = " << result.best_val << "\n" << std::endl;
    }

    //---------------------------------------------------------------------------
    //4. EVERY EXAMPLE PROBLEM ONCE.
    {
        const size_t d {10};
        PSOSettings settings;
        settings.iterations = 2000;
        settings.particles = 20 + 2 * d;               //Particle count grows with dimension.
        settings.seed = 1000;

        std::cout << std::string(99, '-') << "\n" << "EXAMPLE PROBLEMS (d = " << d << ", one run each)" << "\n" << std::string(99, '-') << std::endl;
        run_summary(problems::Sphere{}, d, settings);
        run_summary(problems::Ellipsoid{}, d, settings);
        run_summary(problems::Rosenbrock{}, d, settings);
        run_summary(problems::Rastrigin{}, d, settings);
        run_summary(problems::Ackley{}, d, settings);
        run_summary(problems::Griewank{}, d, settings);
        run_summary(problems::Schwefel{}, d, settings);
        std::cout << std::string(99, '-') << std::endl;
    }

    std::chrono::duration<double> total {std::chrono::steady_clock::now() - start};
    std::cout << "PROGRAM FINISHED." << "\n" << "Total time: " << std::fixed << std::setprecision(2) << total.count() << "s" << std::endl;
    return 0;
}
