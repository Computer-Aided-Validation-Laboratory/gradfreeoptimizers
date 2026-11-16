/**
 * @file pso_bindings.cpp
 * @author Lavya D. Ghanwat (Pizzaman2629)
 * @brief Python binding (pybind11) of the particle swarm optimizer and the C++ example problems: the gfo._pso module.
 *
 * Two ways to run from Python:
 *  1. optimize_python(objective, d, lo, hi, settings)
 *     objective(X) gets an (n, d) numpy array (one row per particle, PHYSICAL coordinates) and returns n values.
 *     It is called ONCE per iteration, from the calling thread. The swarm update in between runs in C++ (OpenMP) with
 *     the GIL released, so no OpenMP thread ever touches Python.
 *  2. optimize_problem(problem, d, lo, hi, settings)
 *     problem is one of the C++ example problems (gfo._pso.problems). The whole run, objective included, stays in C++
 *     with the GIL released and every particle evaluated in parallel.
 *
 * Build with scripts/build.py, which puts the module in src/gfo/. Most users call gfo.pso(...) instead of these directly.
 *
 * @version 0.3
 * @date 2026-11-16
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include <chrono>
#include <cstring>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "pso.hpp"
#include "exampleProblems/exampleProblems.hpp"

namespace py = pybind11;

namespace {

using Clock = std::chrono::steady_clock;
using InputArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

//How often a C++ run checks for Ctrl+C. Checking needs the GIL, so it is rate limited rather than done every iteration.
constexpr auto SIGNAL_CHECK_INTERVAL {std::chrono::milliseconds(100)};

py::array_t<double> to_numpy(const std::vector<double>& values){
    return py::array_t<double>(static_cast<py::ssize_t>(values.size()), values.data());
}


/**
 * @brief Iteration hook that raises KeyboardInterrupt (or any pending Python signal error) during long C++ runs.
 */
struct SignalCheck {
    Clock::time_point last {Clock::now()};

    void operator()(int){
        const auto now {Clock::now()};
        if(now - last < SIGNAL_CHECK_INTERVAL) return;
        last = now;
        py::gil_scoped_acquire gil;
        if(PyErr_CheckSignals() != 0) throw py::error_already_set();
    }
};


/**
 * @brief Run fn with the GIL released, sending std::cout to Python's sys.stdout if the run prints anything.
 *
 * The redirect is created and destroyed while the GIL is held. pybind11's stream buffer takes the GIL itself whenever
 * it flushes, and the optimizer only prints from the calling thread, so printing with the GIL released is safe.
 */
template <typename Function>
PSOResult run_without_gil(const PSOSettings& settings, Function&& fn){
    std::optional<py::scoped_ostream_redirect> redirect;
    if(settings.verbose || settings.diagnostics) redirect.emplace();

    py::gil_scoped_release release;
    return fn();
}


/**
 * @brief Optimize a Python objective. It is called once per iteration with the whole swarm.
 */
PSOResult optimize_python(py::object objective, size_t d, const std::vector<double>& lo, const std::vector<double>& hi,
                          const PSOSettings& settings){

    if(!PyCallable_Check(objective.ptr())) throw py::type_error("objective must be callable: objective(X) -> n values");

    //Called by the optimizer once per iteration, from this thread, with the GIL released.
    //Only references are captured, so copying this lambda never touches a Python reference count without the GIL.
    auto evaluate = [&objective](const double* X, size_t n, size_t dim, double* out) -> double {
        py::gil_scoped_acquire gil;

        //A fresh array every call, so Python can keep or modify X without touching the optimizer.
        py::array_t<double> batch({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(dim)});
        std::memcpy(batch.mutable_data(), X, n * dim * sizeof(double));

        const auto t0 {Clock::now()};
        py::object returned {objective(batch)};
        const double inner {std::chrono::duration<double>(Clock::now() - t0).count()};

        InputArray values {InputArray::ensure(returned)};
        if(!values){
            throw py::type_error("The objective must return n numbers (one per row of X), got an object of type "
                                 + std::string(py::str(py::type::handle_of(returned).attr("__name__"))));
        }
        if(static_cast<size_t>(values.size()) != n || values.ndim() > 2){
            std::ostringstream message;
            message << "The objective got X with shape (" << n << ", " << dim << ") and must return " << n
                    << " values (one per row), but returned " << values.size() << " value(s) with " << values.ndim()
                    << " dimension(s). If your function takes a single point, pass vectorized=False to gfo.pso.";
            throw py::value_error(message.str());
        }
        std::memcpy(out, values.data(), n * sizeof(double));
        return inner;
    };

    return run_without_gil(settings, [&]{ return run_pso_batch(evaluate, d, lo, hi, settings); });
}


/**
 * @brief Optimize a C++ example problem. Everything stays in C++, particles are evaluated in parallel.
 */
template <typename Problem>
PSOResult optimize_problem(const Problem& problem, size_t d, const std::vector<double>& lo, const std::vector<double>& hi,
                           const PSOSettings& settings){
    const Problem local {problem};  //Own copy, so changing the Python object mid-run cannot race with the threads.
    return run_without_gil(settings, [&]{ return run_pso(local, d, lo, hi, settings, SignalCheck{}); });
}


/**
 * @brief Evaluate a C++ problem from Python: points of shape (n, d) give n values.
 */
template <typename Problem>
py::array_t<double> evaluate_problem(const Problem& problem, const InputArray& points){
    if(points.ndim() != 2){
        throw py::value_error("Expected points with shape (n, d), got " + std::to_string(points.ndim())
                              + " dimension(s). Use points[np.newaxis, :] for a single point.");
    }

    const size_t n {static_cast<size_t>(points.shape(0))};
    const size_t d {static_cast<size_t>(points.shape(1))};
    py::array_t<double> values(static_cast<py::ssize_t>(n));
    double* out {values.mutable_data()};
    const double* rows {points.data()};
    {
        py::gil_scoped_release release;
        for(size_t k = 0; k < n; k++) out[k] = problem(rows + k * d, d);
    }
    return values;
}


/**
 * @brief Common Python interface of every C++ example problem.
 *
 * Problem(...) with keyword arguments for every parameter, readable/writable attributes, evaluate(points)
 * (shape (n, d) -> (n,)), calc_x_opt(d), f_opt and name. The same interface as the Python example problems in
 * gfo.problems.
 */
template <typename Problem>
py::class_<Problem> bind_problem(py::module_& m, const char* doc, std::vector<std::string> fields){
    py::class_<Problem> cls(m, Problem::name, doc);
    cls.attr("name") = Problem::name;
    cls.attr("fields") = py::cast(fields);
    cls.attr("f_opt") = Problem{}.f_opt();
    cls.def("evaluate", &evaluate_problem<Problem>, py::arg("points"),
            "Evaluate every row of points, shape (n, d). Returns the n values, shape (n,).");
    cls.def("calc_x_opt", [](const Problem& p, size_t d){
        return to_numpy(std::vector<double>(d, p.x_opt()));
    }, py::arg("d"), "Location of the known minimum in d dimensions, shape (d,).");
    cls.def_readwrite("lo", &Problem::lo, "Default lower bound (every dimension).");
    cls.def_readwrite("hi", &Problem::hi, "Default upper bound (every dimension).");
    cls.def("__repr__", [](py::object self){
        std::string text {std::string(py::str(self.attr("name"))) + "("};
        bool first {true};
        for(py::handle field : self.attr("fields")){
            text += (first ? "" : ", ") + std::string(py::str(field)) + "=" + std::string(py::repr(self.attr(field)));
            first = false;
        }
        return text + ")";
    });
    return cls;
}


std::string settings_repr(const PSOSettings& s){
    std::ostringstream out;
    out << "PSOSettings(iterations=" << s.iterations << ", particles=" << s.particles << ", inertia=" << s.inertia
        << ", acceleration1=" << s.acceleration1 << ", acceleration2=" << s.acceleration2
        << ", max_velocity_rate=" << s.max_velocity_rate << ", initial_velocity_rate=" << s.initial_velocity_rate
        << ", seed=" << s.seed << ", default_bounds=" << (s.default_bounds ? "True" : "False") << ", threads=" << s.threads
        << ", verbose=" << (s.verbose ? "True" : "False") << ", diagnostics=" << (s.diagnostics ? "True" : "False")
        << ", run_dir='" << s.run_dir << "', optimizer_dir='" << s.optimizer_dir << "', save_loss="
        << (s.save_loss ? "True" : "False") << ", save_history=" << (s.save_history ? "True" : "False") << ")";
    return out.str();
}

} //namespace


PYBIND11_MODULE(_pso, m){

    m.doc() = "Particle swarm optimizer (C++) with Python and C++ objectives. Use gfo.pso(...) for the friendly interface.";

#ifdef _OPENMP
    m.attr("openmp") = true;
    m.attr("max_threads") = omp_get_max_threads();
#else
    m.attr("openmp") = false;
    m.attr("max_threads") = 1;
#endif

    //SETTINGS.
    py::class_<PSOSettings>(m, "PSOSettings", "Settings for one PSO run. Every field has a default (the original v0.2 values).")
        .def(py::init<>())
        .def_readwrite("iterations", &PSOSettings::iterations)
        .def_readwrite("particles", &PSOSettings::particles)
        .def_readwrite("inertia", &PSOSettings::inertia)
        .def_readwrite("acceleration1", &PSOSettings::acceleration1)
        .def_readwrite("acceleration2", &PSOSettings::acceleration2)
        .def_readwrite("max_velocity_rate", &PSOSettings::max_velocity_rate)
        .def_readwrite("initial_velocity_rate", &PSOSettings::initial_velocity_rate)
        .def_readwrite("seed", &PSOSettings::seed)
        .def_readwrite("default_bounds", &PSOSettings::default_bounds)
        .def_readwrite("threads", &PSOSettings::threads)
        .def_readwrite("verbose", &PSOSettings::verbose)
        .def_readwrite("verbose_interval", &PSOSettings::verbose_interval)
        .def_readwrite("diagnostics", &PSOSettings::diagnostics)
        .def_readwrite("diagnostics_interval", &PSOSettings::diagnostics_interval)
        .def_readwrite("run_dir", &PSOSettings::run_dir)
        .def_readwrite("optimizer_dir", &PSOSettings::optimizer_dir)
        .def_readwrite("save_loss", &PSOSettings::save_loss)
        .def_readwrite("save_history", &PSOSettings::save_history)
        .def("__repr__", &settings_repr);

    //DIAGNOSTICS.
    py::class_<PSODiagnostics>(m, "PSODiagnostics", "Time spent in each section of a run (seconds).")
        .def_readonly("total_s", &PSODiagnostics::total_s)
        .def_readonly("setup_s", &PSODiagnostics::setup_s)
        .def_readonly("swarm_s", &PSODiagnostics::swarm_s)
        .def_readonly("evaluate_s", &PSODiagnostics::evaluate_s)
        .def_readonly("objective_s", &PSODiagnostics::objective_s)
        .def_readonly("update_s", &PSODiagnostics::update_s)
        .def_readonly("best_s", &PSODiagnostics::best_s)
        .def_readonly("output_s", &PSODiagnostics::output_s)
        .def_readonly("evaluations", &PSODiagnostics::evaluations)
        .def_readonly("threads", &PSODiagnostics::threads)
        .def_readonly("batch", &PSODiagnostics::batch)
        .def_readonly("thread_times", &PSODiagnostics::thread_times)
        .def("to_dict", [](const PSODiagnostics& s){
            py::dict out;
            out["total_s"] = s.total_s;
            out["setup_s"] = s.setup_s;
            out["swarm_s"] = s.swarm_s;
            out["evaluate_s"] = s.evaluate_s;
            out["objective_s"] = s.objective_s;
            out["update_s"] = s.update_s;
            out["best_s"] = s.best_s;
            out["output_s"] = s.output_s;
            out["evaluations"] = s.evaluations;
            out["threads"] = s.threads;
            out["batch"] = s.batch;
            out["thread_times"] = s.thread_times;
            return out;
        })
        .def("__repr__", [](const PSODiagnostics& s){
            std::ostringstream out;
            out << "PSODiagnostics(total_s=" << s.total_s << ", swarm_s=" << s.swarm_s << ", evaluate_s=" << s.evaluate_s
                << ", objective_s=" << s.objective_s << ", evaluations=" << s.evaluations << ", threads=" << s.threads << ")";
            return out.str();
        });

    //RESULT.
    py::class_<PSOResult>(m, "PSOResult", "Result of a PSO run. Positions are in physical coordinates.")
        .def_property_readonly("best_pos", [](const PSOResult& r){ return to_numpy(r.best_pos); })
        .def_readonly("best_val", &PSOResult::best_val)
        .def_property_readonly("loss_history", [](const PSOResult& r){ return to_numpy(r.loss_history); })
        .def_readonly("evaluations", &PSOResult::evaluations)
        .def_readonly("diagnostics", &PSOResult::diagnostics)
        .def("__repr__", [](const PSOResult& r){
            std::ostringstream out;
            out << std::setprecision(10) << "<PSOResult best_val=" << r.best_val << " evaluations=" << r.evaluations
                << " time=" << std::setprecision(4) << r.diagnostics.total_s << "s>";
            return out.str();
        });

    //OPTIMIZERS.
    m.def("optimize_python", &optimize_python,
        py::arg("objective"), py::arg("d"), py::arg("lo"), py::arg("hi"), py::arg("settings") = PSOSettings{},
        "Run PSO on a vectorized Python objective f(X) -> n values, X has shape (n, d) in physical coordinates.");

    //EXAMPLE PROBLEMS.
    py::module_ pm = m.def_submodule("problems", "C++ example problems. Same interface as gfo.problems (the Python versions).");

    const problems::Sphere sphere_def {};
    bind_problem<problems::Sphere>(pm, "Sphere: convex, unimodal. f = sum (x - shift)^2.", {"shift", "lo", "hi"})
        .def(py::init([](double shift, double lo, double hi){ return problems::Sphere{.shift = shift, .lo = lo, .hi = hi}; }),
             py::arg("shift") = sphere_def.shift, py::arg("lo") = sphere_def.lo, py::arg("hi") = sphere_def.hi)
        .def_readwrite("shift", &problems::Sphere::shift);

    const problems::Ellipsoid ellipsoid_def {};
    bind_problem<problems::Ellipsoid>(pm, "Ellipsoid: convex, ill conditioned. f = sum (k + 1)(x - shift)^2.", {"shift", "lo", "hi"})
        .def(py::init([](double shift, double lo, double hi){ return problems::Ellipsoid{.shift = shift, .lo = lo, .hi = hi}; }),
             py::arg("shift") = ellipsoid_def.shift, py::arg("lo") = ellipsoid_def.lo, py::arg("hi") = ellipsoid_def.hi)
        .def_readwrite("shift", &problems::Ellipsoid::shift);

    const problems::Rosenbrock rosenbrock_def {};
    bind_problem<problems::Rosenbrock>(pm, "Rosenbrock: non-convex curved valley. Minimum at x = shift + a.", {"shift", "a", "b", "lo", "hi"})
        .def(py::init([](double shift, double a, double b, double lo, double hi){
                 return problems::Rosenbrock{.shift = shift, .a = a, .b = b, .lo = lo, .hi = hi}; }),
             py::arg("shift") = rosenbrock_def.shift, py::arg("a") = rosenbrock_def.a, py::arg("b") = rosenbrock_def.b,
             py::arg("lo") = rosenbrock_def.lo, py::arg("hi") = rosenbrock_def.hi)
        .def_readwrite("shift", &problems::Rosenbrock::shift)
        .def_readwrite("a", &problems::Rosenbrock::a)
        .def_readwrite("b", &problems::Rosenbrock::b);

    const problems::Rastrigin rastrigin_def {};
    bind_problem<problems::Rastrigin>(pm, "Rastrigin: non-convex, multimodal grid of local minima.", {"shift", "amplitude", "lo", "hi"})
        .def(py::init([](double shift, double amplitude, double lo, double hi){
                 return problems::Rastrigin{.shift = shift, .amplitude = amplitude, .lo = lo, .hi = hi}; }),
             py::arg("shift") = rastrigin_def.shift, py::arg("amplitude") = rastrigin_def.amplitude,
             py::arg("lo") = rastrigin_def.lo, py::arg("hi") = rastrigin_def.hi)
        .def_readwrite("shift", &problems::Rastrigin::shift)
        .def_readwrite("amplitude", &problems::Rastrigin::amplitude);

    const problems::Ackley ackley_def {};
    bind_problem<problems::Ackley>(pm, "Ackley: non-convex, shallow local minima around one narrow basin.", {"shift", "a", "b", "c", "lo", "hi"})
        .def(py::init([](double shift, double a, double b, double c, double lo, double hi){
                 return problems::Ackley{.shift = shift, .a = a, .b = b, .c = c, .lo = lo, .hi = hi}; }),
             py::arg("shift") = ackley_def.shift, py::arg("a") = ackley_def.a, py::arg("b") = ackley_def.b,
             py::arg("c") = ackley_def.c, py::arg("lo") = ackley_def.lo, py::arg("hi") = ackley_def.hi)
        .def_readwrite("shift", &problems::Ackley::shift)
        .def_readwrite("a", &problems::Ackley::a)
        .def_readwrite("b", &problems::Ackley::b)
        .def_readwrite("c", &problems::Ackley::c);

    const problems::Griewank griewank_def {};
    bind_problem<problems::Griewank>(pm, "Griewank: non-convex, couples dimensions through a product.", {"shift", "scale", "lo", "hi"})
        .def(py::init([](double shift, double scale, double lo, double hi){
                 return problems::Griewank{.shift = shift, .scale = scale, .lo = lo, .hi = hi}; }),
             py::arg("shift") = griewank_def.shift, py::arg("scale") = griewank_def.scale,
             py::arg("lo") = griewank_def.lo, py::arg("hi") = griewank_def.hi)
        .def_readwrite("shift", &problems::Griewank::shift)
        .def_readwrite("scale", &problems::Griewank::scale);

    const problems::Schwefel schwefel_def {};
    bind_problem<problems::Schwefel>(pm, "Schwefel: non-convex and deceptive. Minimum at x = 420.9687 + shift.", {"shift", "lo", "hi"})
        .def(py::init([](double shift, double lo, double hi){ return problems::Schwefel{.shift = shift, .lo = lo, .hi = hi}; }),
             py::arg("shift") = schwefel_def.shift, py::arg("lo") = schwefel_def.lo, py::arg("hi") = schwefel_def.hi)
        .def_readwrite("shift", &problems::Schwefel::shift);

    //One optimize_problem overload per problem type (pybind11 picks the matching one).
    const char* problem_doc {"Run PSO on a C++ example problem. The whole run stays in C++ with every particle evaluated in parallel."};
    m.def("optimize_problem", &optimize_problem<problems::Sphere>, py::arg("problem"), py::arg("d"), py::arg("lo"), py::arg("hi"), py::arg("settings") = PSOSettings{}, problem_doc);
    m.def("optimize_problem", &optimize_problem<problems::Ellipsoid>, py::arg("problem"), py::arg("d"), py::arg("lo"), py::arg("hi"), py::arg("settings") = PSOSettings{}, problem_doc);
    m.def("optimize_problem", &optimize_problem<problems::Rosenbrock>, py::arg("problem"), py::arg("d"), py::arg("lo"), py::arg("hi"), py::arg("settings") = PSOSettings{}, problem_doc);
    m.def("optimize_problem", &optimize_problem<problems::Rastrigin>, py::arg("problem"), py::arg("d"), py::arg("lo"), py::arg("hi"), py::arg("settings") = PSOSettings{}, problem_doc);
    m.def("optimize_problem", &optimize_problem<problems::Ackley>, py::arg("problem"), py::arg("d"), py::arg("lo"), py::arg("hi"), py::arg("settings") = PSOSettings{}, problem_doc);
    m.def("optimize_problem", &optimize_problem<problems::Griewank>, py::arg("problem"), py::arg("d"), py::arg("lo"), py::arg("hi"), py::arg("settings") = PSOSettings{}, problem_doc);
    m.def("optimize_problem", &optimize_problem<problems::Schwefel>, py::arg("problem"), py::arg("d"), py::arg("lo"), py::arg("hi"), py::arg("settings") = PSOSettings{}, problem_doc);
}

