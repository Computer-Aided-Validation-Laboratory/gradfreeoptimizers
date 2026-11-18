/**
 * @file pso.hpp
 * @author Lavya D. Ghanwat (Pizzaman2629)
 * @brief Particle swarm optimizer (header).
 *
 * Declares the global-topology swarm structure, initializer, boundary handling, settings, result structures and the standard PSO routines.
 * Optimization runs in the unit hypercube [0,1]^d; the objective is rescaled to physical bounds by the optimizer.
 *
 * Two ways to evaluate the objective:
 *  1. run_pso        : a C++ objective double f(const double* x, size_t d), evaluated for every particle in parallel (OpenMP).
 *  2. run_pso_batch  : a batch objective that gets the whole swarm at once, called once per iteration from one thread.
 *                      This is what the Python binding uses: the swarm update runs in C++ (in parallel), Python evaluates the batch.
 *
 * Both share the same swarm update, so for the same seed they visit exactly the same points.
 *
 * @version 0.1
 * @date 2026-11-16
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <iomanip>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <functional>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Structure for a Particle Swarm with Global Swarm Topology
 *
 * Contains two size_t variables: n and d which contain the number of particles and problem dimensionality respectively.
 *
 * Contains double vectors:
 * pos --> Position vectors for all particles. Flattened index = n_ * d + d_
 * vel --> Velocity vectors for all particles. Flattened index = n_ * d + d_
 * pbest_pos --> Best position vectors for all particles. Flattened index = n_ * d + d_
 * pbest_val --> Best function value for each particle. Indexed as n_
 * gbest_pos --> Global best position vector. Indexed as d_
 * gbest_val --> Global best value.
 * Here, n_ and d_ are meant to be indices you can loop over to extract particle number and coordinate dimension respectively.
 *
 * Contains pointers:
 * x --> Acts as an easy way to go to the first coordinate of a particle without much indexing.
 * y --> Similar to x but for velocity.
 *
 * Used extensively in the PSO optimizer to describe the particle swarm state.
 */
struct GBSwarmTopology {
    size_t n;
    size_t d;

    std::vector<double> pos;
    std::vector<double> vel;
    std::vector<double> pbest_pos;
    std::vector<double> pbest_val;
    std::vector<double> gbest_pos;
    double gbest_val = std::numeric_limits<double>::infinity();

    GBSwarmTopology(size_t n_, size_t d_)
        : n(n_), d(d_),
          pos(n_*d_), vel(n_*d_),
          pbest_pos(n_*d_),
          pbest_val(n_, std::numeric_limits<double>::infinity()),
          gbest_pos(d_) {}

    double* x(size_t i) { return &pos[i*d]; }
    double* v(size_t i) { return &vel[i*d]; }
};


/**
 * @brief Settings for one PSO run.
 *
 * Every field has a default, so only the ones you care about need setting, e.g.
 *      PSOSettings settings;
 *      settings.iterations = 500;
 *      settings.verbose = true;
 *
 * The defaults reproduce the original optimizer (v0.1) exactly.
 */
struct PSOSettings {
    int iterations {100};                   //Iterations to run the loop for.
    size_t particles {20};                  //Number of particles.
    double inertia {0.729};                 //Inertia term. Default is 0.729 to ensure stability (Clerc-Kennedy).
    double acceleration1 {1.49445};         //Acceleration (Cognitive Influence) term. Defaults to 1.49445 (stability).
    double acceleration2 {1.49445};         //Acceleration (Social Influence) term. Defaults to 1.49445 (stability).
    double max_velocity_rate {0.15};        //Velocity clamp, as a fraction of the domain width (unit hypercube). Same meaning as pymoo's max_velocity_rate.
    double initial_velocity_rate {0.02};    //Initial velocities are uniform in [-rate, rate] (unit hypercube). 0 starts the swarm at rest.
    unsigned long long seed {42};           //Seed for reproducibility.
    bool default_bounds {false};            //Ignore lo/hi and optimize purely in the unit hypercube.
    int threads {0};                        //OpenMP threads for the particle loop. 0 = OpenMP default (all cores).

    //Console output.
    bool verbose {false};                   //Start/end summary and a line every verbose_interval iterations.
    int verbose_interval {1};               //Print every this many iterations when verbose.
    bool diagnostics {false};               //Time every section of the loop and print a breakdown.
    int diagnostics_interval {0};           //Print a diagnostics line every this many iterations. 0 = about 10 lines per run.

    //Files. Everything goes in <run_dir>/<optimizer_dir>/. An empty run_dir means the current folder.
    std::string run_dir {""};               //Run folder the optimizer folder is created in.
    std::string optimizer_dir {"optimizer"};//Name of the optimizer folder inside run_dir.
    bool save_loss {false};                 //Save the global best value of every iteration to loss.csv (columns: iteration, f).
    bool save_history {false};              //2D only: save every particle of every iteration to history.csv (iteration, x, y, f) and the bounds to domain.csv.
};


/**
 * @brief Time spent in each section of a run.
 *
 * The wall times (all *_s fields except the thread times) are always measured, they cost a couple of clock reads per iteration.
 * The per-call thread times are only measured with diagnostics on, because they need clock reads for every particle.
 */
struct PSODiagnostics {
    double total_s {0.0};               //Whole run, from the call to the return.
    double setup_s {0.0};               //Validation, allocation and swarm initialization.
    double swarm_s {0.0};               //Wall time in the particle loop(s). C++ objective: update + objective. Batch objective: update only.
    double evaluate_s {0.0};            //Batch objective only: wall time of the batch calls (including copying data in and out).
    double objective_s {0.0};           //Batch objective: time inside the objective itself. C++ objective (diagnostics only): summed over all threads.
    double update_s {0.0};              //Diagnostics only, C++ objective: particle update time summed over all threads. Batch objective: equals swarm_s.
    double best_s {0.0};                //Personal and global best bookkeeping outside the particle loop.
    double output_s {0.0};              //Writing files and printing to the console.
    long long evaluations {0};          //Number of objective evaluations.
    int threads {1};                    //Threads used for the particle loop.
    bool batch {false};                 //True if the run used a batch objective.
    bool thread_times {false};          //True if objective_s/update_s hold per-call thread times (C++ objective with diagnostics on).
};


/**
 * @brief Result from PSO
 *
 * Contains an array to store the minima locations and the smallest obtained functional value.
 */
struct PSOResult {
    std::vector<double> best_pos;       //Best position in PHYSICAL coordinates.
    double best_val;                    //Objective value at best_pos.
    std::vector<double> loss_history;   //Global best value after every iteration.
    long long evaluations {0};          //Number of objective evaluations.
    PSODiagnostics diagnostics;         //Timings of the run.
};


/**
 * @brief One particle of a 2D run as it is saved in the history file: the two physical coordinates and the objective value.
 */
struct PSOHistoryPoint {
    double x;
    double y;
    double f;
};


/**
 * @brief Called once per iteration (after the global best update) with the iteration number.
 *
 * Optional. The Python binding uses it to check for Ctrl+C during long C++ runs. Throwing from it stops the run.
 */
using PSOIterationHook = std::function<void(int)>;


/**
 * @brief Swarm Initializer
 *
 * Takes the swarm state and initializes it.
 * Uses random numbers to initialize a swarm that follows the global best topology.
 *
 * @param swarmState Input swarm state, overwritten to be initialized.
 * @param gen Random number generator for initialization (seeded ideally)
 * @param vmax Initial velocities are uniform in [-vmax, vmax]. Default 0.02, we don't want the particles to whizz around like crazy.
 */
void gbinitializer(GBSwarmTopology& swarmState, std::mt19937_64& gen, double vmax = 0.02);


/**
 * @brief Boundary Function
 *
 * Deals with the boundary in the PSO iteration loop.
 * Reflects the particle back so it always bounces around in the unit hypercube.
 *
 * @param x Particle position.
 * @param v Particle velocity.
 */
void reflect_unit(double& x, double& v);


//---------------------------------------------------------------------------
//INTERNALS. Shared by run_pso and run_pso_batch, not meant to be called directly.
namespace pso_detail {

//Setting an alias for clock.
using Clock = std::chrono::steady_clock;

//Helper function for getting a duration.
inline double seconds_between(Clock::time_point start, Clock::time_point end){
    return std::chrono::duration<double>(end - start).count();
}

//Helper function to resolve threads. This sets 0 requested threads as the OpenMP default, and returns 1 if OpenMP is not available.
inline int resolve_threads(int requested){
#ifdef _OPENMP
    return requested > 0 ? requested : omp_get_max_threads();
#else
    (void)requested;
    return 1;
#endif
}

/**
 * @brief Velocity and position update of one particle mapped to physical coordinates.
 *
 * This is the main update function. 
 * Every (iteration, particle) pair gets its own generator, seeded from the run seed to keep it thread safe.
 */
inline void update_particle(GBSwarmTopology& swarmState, size_t n_, int i, const PSOSettings& settings,
                            const std::vector<double>& lo_, const std::vector<double>& hi_, double* xphys){

    const size_t d {swarmState.d};
    const unsigned long long seed {settings.seed};
    const double vmax {settings.max_velocity_rate};

    //Random Number Generator.
    //This guy takes a seed and shuffles its bits around deterministically to create a unique seed for every instance.
    std::seed_seq seed_sequence{
        static_cast<std::uint32_t>(seed),
        static_cast<std::uint32_t>(seed >> 32),
        static_cast<std::uint32_t>(i),
        static_cast<std::uint32_t>(n_)
    };
    std::mt19937_64 gen_p(seed_sequence);
    std::uniform_real_distribution<double> dist_p(0.0, 1.0);

    //Loop over dimensions.
    for(size_t d_ = 0; d_ < d; d_++){

        //This index is the usual 1D flattened array logic.
        size_t index {n_ * d + d_};

        double cognitive_influence = swarmState.pbest_pos[index] - swarmState.pos[index];
        double social_influence = swarmState.gbest_pos[d_] - swarmState.pos[index];

        double r1 = dist_p(gen_p);
        double r2 = dist_p(gen_p);

        //Velocity Update (skipped on i == 0 as a slight burn in)
        if(i != 0){
        swarmState.vel[index] = (settings.inertia * swarmState.vel[index])  //Inertia Term.
        + (settings.acceleration1 * r1 * cognitive_influence)               //Cognitive Influence Term.
        + (settings.acceleration2 * r2 * social_influence);                 //Social Influence Term.
        }

        swarmState.vel[index] = std::clamp(swarmState.vel[index], -vmax, vmax);

        //Particle Position Update
        swarmState.pos[index] = swarmState.pos[index] + swarmState.vel[index];

        //Boundary pass. Basically, if we hit any boundaries of the unit hypercube, reflect things back.
        reflect_unit(swarmState.pos[index], swarmState.vel[index]);
    }

    //Map to physical coordinates for the objective.
    const double* pos_ptr {swarmState.x(n_)};
    for(size_t d_ = 0; d_ < d; d_++){
        xphys[d_] = lo_[d_] + pos_ptr[d_] * (hi_[d_] - lo_[d_]);
    }
}

/**
 * @brief Compute Local Best for one particle.
 */
inline void update_personal_best(GBSwarmTopology& swarmState, size_t n_, double f_val){
    if(f_val < swarmState.pbest_val[n_]){
        swarmState.pbest_val[n_] = f_val;
        const double* pos_ptr {swarmState.x(n_)};
        std::copy(pos_ptr, pos_ptr + swarmState.d, &swarmState.pbest_pos[n_*swarmState.d]);
    }
}

/**
 * @brief Check for global best (serial, after every particle of the iteration is done).
 */
inline void update_global_best(GBSwarmTopology& swarmState){
    const size_t d {swarmState.d};

    //Loop over all particles, check personal best to current gbest, and update if necessary.
    for(size_t n_ = 0; n_ < swarmState.n; n_++){
        if(swarmState.pbest_val[n_] < swarmState.gbest_val){
            swarmState.gbest_val = swarmState.pbest_val[n_];
            std::copy(&swarmState.pbest_pos[n_*d], &swarmState.pbest_pos[n_*d] + d, swarmState.gbest_pos.begin());
        }
    }
}

//Console line separator, same width as the rest of the project's banners.
inline const std::string& separator(){
    static const std::string line(99, '-');
    return line;
}

//Format a position as [a, b, c], cutting long vectors short.
inline std::string format_position(const std::vector<double>& x, size_t max_entries = 10){
    std::ostringstream out;
    out << std::setprecision(6) << "[";
    size_t shown {std::min(x.size(), max_entries)};
    for(size_t k = 0; k < shown; k++){
        out << (k ? ", " : "") << x[k];
    }
    if(x.size() > shown) out << ", ... (" << x.size() - shown << " more)";
    out << "]";
    return out.str();
}

inline double percent(double part, double whole){
    return whole > 0.0 ? 100.0 * part / whole : 0.0;
}

//Saves std::cout's number formatting and puts it back, so a run never leaves the caller's stream changed.
struct CoutStateGuard {
    std::ios_base::fmtflags flags;
    std::streamsize precision;
    CoutStateGuard() : flags(std::cout.flags()), precision(std::cout.precision()) {}
    void restore() const { std::cout.flags(flags); std::cout.precision(precision); }
    ~CoutStateGuard() { restore(); }
};

/**
 * @brief Validate the inputs and build the bounds used by the optimizer.
 */
inline void validate_and_build_bounds(size_t d, const std::vector<double>& lo, const std::vector<double>& hi,
                                      const PSOSettings& settings, std::vector<double>& lo_, std::vector<double>& hi_){

    //Some control flows for the input.
    if(settings.particles == 0 || d == 0) throw std::invalid_argument("Number of particles and dimensions must not be 0!");
    if(settings.iterations < 0) throw std::invalid_argument("Iterations must not be negative!");
    if(!(settings.max_velocity_rate > 0.0)) throw std::invalid_argument("max_velocity_rate must be greater than 0!");
    if(!(settings.initial_velocity_rate >= 0.0)) throw std::invalid_argument("initial_velocity_rate must not be negative!");
    if(settings.threads < 0) throw std::invalid_argument("threads must not be negative (0 = OpenMP default)!");
    if(settings.verbose_interval < 1) throw std::invalid_argument("verbose_interval must be at least 1!");
    if(settings.diagnostics_interval < 0) throw std::invalid_argument("diagnostics_interval must not be negative!");
    if(settings.optimizer_dir.empty() && (settings.save_loss || settings.save_history))
        throw std::invalid_argument("optimizer_dir must not be empty when saving files!");

    //Create a lo_, hi_ array that is used in this optimizer.
    //It is overwritten with user passed defaults.
    lo_.assign(d, 0.0);
    hi_.assign(d, 1.0);
    if(!settings.default_bounds){
        
        //Size mismatch arguments.
        if(lo.size() != d || hi.size() != d)
            throw std::invalid_argument("Lower bound and upper bound arrays must have d entries");

        //Ensure that lower bounds are actually lower than higher bounds (and are real numbers).
        for(size_t k = 0; k < d; k++){
            if(!std::isfinite(lo[k]) || !std::isfinite(hi[k])) throw std::invalid_argument("Bounds must be finite numbers!");
            if(!(lo[k] < hi[k])) throw std::invalid_argument("Need the lower bound to be smaller than the higher bound!");
        }
        lo_ = lo;
        hi_ = hi;
    }
}


/**
 * @brief The PSO loop shared by both evaluation modes.
 *
 * Batch == false: objective is a C++ objective double(const double* x, size_t d). The particle update and the objective
 *                 run together in one parallel loop.
 * Batch == true : objective is double(const double* X, size_t n, size_t d, double* f). It fills f[0..n) for the n x d
 *                 row-major positions in X and returns the seconds spent inside the objective itself (or a negative
 *                 number if it did not measure it). It is called from one thread, outside any parallel region.
 */
template <bool Batch, typename Objective>
PSOResult run_engine(Objective& objective, size_t d, const std::vector<double>& lo, const std::vector<double>& hi,
                     const PSOSettings& settings, const PSOIterationHook& on_iteration){

    const auto run_start {Clock::now()};
    const CoutStateGuard cout_state;

    std::vector<double> lo_, hi_;
    validate_and_build_bounds(d, lo, hi, settings, lo_, hi_);

    const size_t n {settings.particles};
    const int it {settings.iterations};
    const int threads {resolve_threads(settings.threads)};
    const bool diag {settings.diagnostics};
    const bool talk {settings.verbose || settings.diagnostics};
    const int diag_interval {settings.diagnostics_interval > 0 ? settings.diagnostics_interval : std::max(1, it / 10)};

    PSODiagnostics stats;
    stats.threads = threads;
    stats.batch = Batch;
    stats.thread_times = (!Batch && diag);

    //Create the particle swarm, seed and random number generator.
    GBSwarmTopology swarmState(n, d);
    std::mt19937_64 gen(settings.seed);

    //Initialize the swarm.
    gbinitializer(swarmState, gen, settings.initial_velocity_rate);

    //Physical coordinates of every particle (row-major, n x d) and their objective values.
    //Allocated once, each particle only ever touches its own row, so this is thread safe.
    std::vector<double> xphys(n * d);
    std::vector<double> fvals(n);

    //Loss history: the global best value after every iteration.
    PSOResult result;
    result.loss_history.reserve(static_cast<size_t>(it));

    //All optimizer files go in <run_dir>/<optimizer_dir>/, so they sit next to the rest of the run's data.
    const std::filesystem::path optimizerDir {std::filesystem::path(settings.run_dir) / settings.optimizer_dir};

    //Setup history and domain tracking for 2D.
    //This will create a file if it doesn't exist, but if the folder doesn't exist it will throw an error.
    const bool trackHistory {d == 2 && settings.save_history};
    std::ofstream histFile;
    if(trackHistory){
        std::filesystem::create_directories(optimizerDir);
        histFile.open(optimizerDir / "history.csv");
        if(!histFile) throw std::runtime_error("Could not open " + (optimizerDir / "history.csv").string());
        histFile << std::setprecision(12);
        histFile << "iteration,x,y,f\n";

        std::ofstream domFile(optimizerDir / "domain.csv");
        if(!domFile) throw std::runtime_error("Could not open " + (optimizerDir / "domain.csv").string());
        domFile << std::setprecision(12);
        domFile << "lo_x,lo_y,hi_x,hi_y\n";
        domFile << lo_[0] << "," << lo_[1] << "," << hi_[0] << "," << hi_[1] << "\n";
    }

    //Start banner.
    if(talk){
        const auto& line {separator()};
        std::cout << line << "\n" << "PARTICLE SWARM OPTIMIZATION" << "\n" << line << "\n";
        std::cout << " | Particles = " << n << " | Dimensions = " << d << " | Iterations = " << it
                  << " | Threads = " << threads << " | Objective = " << (Batch ? "batch (one call per iteration)" : "C++ (parallel)") << " |\n";
        std::cout << " | Inertia = " << settings.inertia << " | Acceleration 1 = " << settings.acceleration1
                  << " | Acceleration 2 = " << settings.acceleration2 << " | Max velocity rate = " << settings.max_velocity_rate
                  << " | Initial velocity rate = " << settings.initial_velocity_rate << " | Seed = " << settings.seed << " |\n";
        std::cout << " | Bounds = " << (settings.default_bounds ? "unit hypercube" : "physical")
                  << " | Diagnostics = " << (diag ? "on" : "off");
        if(settings.save_loss || trackHistory) std::cout << " | Output folder = " << optimizerDir.string();
        if(settings.save_history && d != 2) std::cout << " | history.csv skipped (only saved for d = 2)";
        std::cout << " |\n" << line << std::endl;
    }

    std::vector<PSOHistoryPoint> iterationPoints;
    if(trackHistory) iterationPoints.resize(n);

    //Diagnostics window (reset every diag_interval iterations).
    PSODiagnostics window;
    int window_start {0};

    stats.setup_s = seconds_between(run_start, Clock::now());

    //Iteration Loop.
    for(int i = 0; i < it; i++){
        const auto iter_start {Clock::now()};
        double swarm_s {0.0}, evaluate_s {0.0}, objective_s {0.0}, update_s {0.0};

        if constexpr (!Batch){
            //Particles are independent within an iteration, so they are evaluated in parallel.
            //Static scheduling is being used here for some more robustness.
            double update_acc {0.0}, objective_acc {0.0};
            #pragma omp parallel for schedule(static, 1) num_threads(threads) reduction(+:update_acc, objective_acc)
            for(size_t n_ = 0; n_ < n; n_++){
                double* x {&xphys[n_ * d]};
                if(diag){
                    const auto t0 {Clock::now()};
                    update_particle(swarmState, n_, i, settings, lo_, hi_, x);
                    const auto t1 {Clock::now()};
                    double f_val = objective(static_cast<const double*>(x), d);
                    const auto t2 {Clock::now()};
                    fvals[n_] = f_val;
                    update_personal_best(swarmState, n_, f_val);
                    update_acc += seconds_between(t0, t1) + seconds_between(t2, Clock::now());
                    objective_acc += seconds_between(t1, t2);
                }
                else{
                    update_particle(swarmState, n_, i, settings, lo_, hi_, x);
                    double f_val = objective(static_cast<const double*>(x), d);
                    fvals[n_] = f_val;
                    update_personal_best(swarmState, n_, f_val);
                }
            }
            swarm_s = seconds_between(iter_start, Clock::now());
            update_s = update_acc;
            objective_s = objective_acc;
        }
        else{
            //Move every particle in parallel, then hand the whole swarm to the batch objective in one call.
            #pragma omp parallel for schedule(static, 1) num_threads(threads)
            for(size_t n_ = 0; n_ < n; n_++){
                update_particle(swarmState, n_, i, settings, lo_, hi_, &xphys[n_ * d]);
            }
            const auto t_moved {Clock::now()};
            swarm_s = seconds_between(iter_start, t_moved);
            update_s = swarm_s;

            double inner {objective(static_cast<const double*>(xphys.data()), n, d, fvals.data())};
            const auto t_evaluated {Clock::now()};
            evaluate_s = seconds_between(t_moved, t_evaluated);
            objective_s = inner >= 0.0 ? inner : evaluate_s;

            for(size_t n_ = 0; n_ < n; n_++){
                update_personal_best(swarmState, n_, fvals[n_]);
            }
        }

        //Check for global best.
        const auto t_best {Clock::now()};
        update_global_best(swarmState);
        result.loss_history.push_back(swarmState.gbest_val);
        const auto t_output {Clock::now()};
        double best_s {seconds_between(t_best, t_output)};
        if constexpr (Batch){
            //The personal best loop above also counts as best bookkeeping.
            best_s += seconds_between(iter_start, t_best) - swarm_s - evaluate_s;
        }

        // Append iteration history to CSV file
        if(trackHistory){
            for(size_t n_ = 0; n_ < n; n_++){
                iterationPoints[n_] = {xphys[n_ * d], xphys[n_ * d + 1], fvals[n_]};
            }
            for(const auto& p : iterationPoints){
                histFile << i << "," << p.x << "," << p.y << "," << p.f << "\n";
            }
        }

        const auto iter_end {Clock::now()};
        const double elapsed_ms {1000.0 * seconds_between(iter_start, iter_end)};

        //Print values out each iteration (iteration 0 is the burn in).
        if(settings.verbose && i > 0 && (i % settings.verbose_interval == 0 || i == it - 1)){
            std::cout << " | Iteration = " << i << " | Global Best Value = " << swarmState.gbest_val << " | Time = " << elapsed_ms << " ms |" << std::endl;
        }

        //Accumulate the timings.
        double output_s {seconds_between(t_output, Clock::now())};
        for(PSODiagnostics* acc : {&stats, &window}){
            acc->swarm_s += swarm_s;
            acc->evaluate_s += evaluate_s;
            acc->objective_s += objective_s;
            acc->update_s += update_s;
            acc->best_s += best_s;
            acc->output_s += output_s;
            acc->evaluations += static_cast<long long>(n);
        }

        //Diagnostics line every diag_interval iterations (and at the end).
        if(diag && ((i + 1) % diag_interval == 0 || i == it - 1)){
            const auto t_diag {Clock::now()};
            const double window_s {window.swarm_s + window.evaluate_s + window.best_s + window.output_s};
            const double per_eval_us {window.evaluations > 0 ? 1e6 * window.objective_s / static_cast<double>(window.evaluations) : 0.0};
            std::cout << std::fixed << std::setprecision(2)
                      << " | [diag] Iterations " << window_start << "-" << i
                      << " | Loop = " << 1000.0 * window_s << " ms";
            if(Batch){
                std::cout << " | Update = " << 1000.0 * window.swarm_s << " ms (" << percent(window.swarm_s, window_s) << "%)"
                          << " | Objective call = " << 1000.0 * window.evaluate_s << " ms (" << percent(window.evaluate_s, window_s) << "%)";
            }
            else{
                const double thread_total {window.update_s + window.objective_s};
                std::cout << " | Particle loop = " << 1000.0 * window.swarm_s << " ms (" << percent(window.swarm_s, window_s) << "%)"
                          << " | Thread time: objective " << percent(window.objective_s, thread_total) << "% / update " << percent(window.update_s, thread_total) << "%";
            }
            std::cout << " | Per evaluation = " << per_eval_us << " us"
                      << " | Best = " << std::scientific << std::setprecision(4) << swarmState.gbest_val << " |" << std::endl;
            cout_state.restore();
            window = PSODiagnostics{};
            window_start = i + 1;
            stats.output_s += seconds_between(t_diag, Clock::now());
        }

        if(on_iteration) on_iteration(i);
    }

    //Write the loss history to a csv file: iteration in the first column, global best value in the second.
    const auto t_files {Clock::now()};
    if(trackHistory){
        histFile.close();
    }
    if(settings.save_loss){
        std::filesystem::create_directories(optimizerDir);
        const std::string lossFilename {(optimizerDir / "loss.csv").string()};
        std::ofstream lossFile(lossFilename);
        if(!lossFile) throw std::runtime_error("Could not open " + lossFilename);
        lossFile << std::setprecision(12);
        lossFile << "iteration,f\n";
        for(size_t k = 0; k < result.loss_history.size(); k++){
            lossFile << k << "," << result.loss_history[k] << "\n";
        }
        if(talk) std::cout << " | Saved loss history to: " << lossFilename << std::endl;
    }
    if(trackHistory && talk){
        std::cout << " | Saved particle history to: " << (optimizerDir / "history.csv").string() << std::endl;
    }
    stats.output_s += seconds_between(t_files, Clock::now());

    //Return the best point in physical coordinates.
    result.best_val = swarmState.gbest_val;
    result.best_pos.resize(d);
    for(size_t k = 0; k < d; k++){
        result.best_pos[k] = lo_[k] + swarmState.gbest_pos[k] * (hi_[k] - lo_[k]);
    }
    result.evaluations = stats.evaluations;
    stats.total_s = seconds_between(run_start, Clock::now());

    //End summary.
    if(talk){
        const auto& line {separator()};
        std::cout << line << "\n";
        std::cout << " | PSO finished | Best value = " << std::setprecision(10) << result.best_val
                  << " | Evaluations = " << result.evaluations << " | Total time = " << std::setprecision(4) << stats.total_s << " s |\n";
        std::cout << " | Best position = " << format_position(result.best_pos) << "\n";
        cout_state.restore();
    }

    //Final diagnostics breakdown.
    if(diag){
        const auto& line {separator()};
        const double total {stats.total_s};
        const double accounted {stats.setup_s + stats.swarm_s + stats.evaluate_s + stats.best_s + stats.output_s};
        const double evals {static_cast<double>(std::max<long long>(stats.evaluations, 1))};
        auto row = [&](const std::string& name, double seconds){
            std::cout << " | " << std::left << std::setw(34) << name << std::right << std::fixed << std::setprecision(4)
                      << std::setw(12) << seconds << " s" << std::setw(10) << std::setprecision(2) << percent(seconds, total) << " % |\n";
        };
        std::cout << line << "\n" << "PSO DIAGNOSTICS (wall time)" << "\n" << line << "\n";
        row("Setup (validation, initialization)", stats.setup_s);
        if(Batch){
            row("Particle update (parallel)", stats.swarm_s);
            row("Objective batch calls", stats.evaluate_s);
            row("  of which inside the objective", stats.objective_s);
            row("  of which data transfer/overhead", std::max(0.0, stats.evaluate_s - stats.objective_s));
        }
        else{
            row("Particle loop (update + objective)", stats.swarm_s);
        }
        row("Best bookkeeping", stats.best_s);
        row("Output (files, console)", stats.output_s);
        row("Unaccounted", std::max(0.0, total - accounted));
        row("Total", total);
        std::cout << line << "\n";
        std::cout << std::setprecision(3);
        if(!Batch){
            const double thread_total {stats.update_s + stats.objective_s};
            std::cout << " | Thread time (summed over " << threads << " threads): objective = " << stats.objective_s << " s ("
                      << percent(stats.objective_s, thread_total) << "%) | update = " << stats.update_s << " s ("
                      << percent(stats.update_s, thread_total) << "%) |\n";
            std::cout << " | Parallel efficiency of the particle loop = "
                      << percent(thread_total, stats.swarm_s * threads) << "% (thread time / (loop wall time x threads)) |\n";
        }
        std::cout << " | Mean objective time per evaluation = " << 1e6 * stats.objective_s / evals << " us"
                  << " | Mean iteration time = " << (it > 0 ? 1000.0 * (total - stats.setup_s) / it : 0.0) << " ms"
                  << " | Evaluations per second = " << std::setprecision(0) << (total > 0.0 ? stats.evaluations / total : 0.0) << " |\n";
        std::cout << line << std::endl;
        cout_state.restore();
    }

    result.diagnostics = stats;
    return result;
}

} //namespace pso_detail


/**
 * @brief Standard Particle Swarm Optimizer (C++ objective, parallel evaluation).
 *
 * This function runs a standard particle swarm optimization on an input loss function.
 * Every particle is moved and evaluated in parallel (OpenMP), so the objective must be thread safe.
 *
 * @tparam ObjectiveFunction Anything callable as double f(const double* x, size_t d): a function, a lambda, or an example problem (see exampleProblems/).
 * @param f Input objective function. Gets the position in PHYSICAL coordinates.
 * @param d Number of dimensions.
 * @param lo Array for lower bounds. (These should be physical for proper resizing)
 * @param hi Array for upper bounds. (Should be physical for proper resizing)
 * @param settings Iterations, particles, coefficients, output... (see PSOSettings, every field has a default).
 * @param on_iteration Optional hook called after every iteration.
 * @return PSOResult
 */
template <typename ObjectiveFunction>
PSOResult run_pso(ObjectiveFunction f, size_t d, const std::vector<double>& lo, const std::vector<double>& hi,
                  const PSOSettings& settings = PSOSettings{}, const PSOIterationHook& on_iteration = {}){
    return pso_detail::run_engine<false>(f, d, lo, hi, settings, on_iteration);
}


/**
 * @brief Standard Particle Swarm Optimizer (batch objective, one call per iteration).
 *
 * The particle update still runs in parallel, but the objective is called once per iteration with the whole swarm,
 * from the calling thread and outside any parallel region. This is the mode for objectives that are not thread safe
 * or that evaluate many points at once more efficiently (e.g. vectorized Python/numpy code).
 *
 * @tparam BatchObjective Callable as double f(const double* X, size_t n, size_t d, double* out).
 *                        X holds n positions (row-major, n x d, PHYSICAL coordinates), out must be filled with the n values.
 *                        Return the seconds spent inside the objective itself, or a negative number if not measured.
 */
template <typename BatchObjective>
PSOResult run_pso_batch(BatchObjective f, size_t d, const std::vector<double>& lo, const std::vector<double>& hi,
                        const PSOSettings& settings = PSOSettings{}, const PSOIterationHook& on_iteration = {}){
    return pso_detail::run_engine<true>(f, d, lo, hi, settings, on_iteration);
}


/**
 * @brief Standard Particle Swarm Optimizer, original (v0.1) call signature.
 *
 * Kept so existing code still compiles. It forwards to run_pso, see PSOSettings for what every argument means.
 *
 * @param optimizer_name Run folder the optimizer files are saved in (inside its optimizer/ subfolder).
 *                       2D runs also save history.csv (iteration, x, y, f for every particle) and domain.csv (the bounds) when this is set.
 *                       If empty, only the loss history is saved, to ./optimizer/.
 * @param loss_history Save the global best value of every iteration to <optimizer_name>/optimizer/loss.csv (columns: iteration, f).
 * @return PSOResult
 */
template <typename ObjectiveFunction>
PSOResult spsoptimzer(
    ObjectiveFunction f,
    int it,
    size_t n,
    size_t d,
    const std::vector<double>& lo,
    const std::vector<double>& hi,
    bool default_bounds,
    double inertia = 0.729,
    double acceleration1 = 1.49445,
    double acceleration2 = 1.49445,
    unsigned long long seed = 42,
    bool verbose = true,
    const std::string& optimizer_name = "",
    bool loss_history = false)
{
    PSOSettings settings;
    settings.iterations = it;
    settings.particles = n;
    settings.default_bounds = default_bounds;
    settings.inertia = inertia;
    settings.acceleration1 = acceleration1;
    settings.acceleration2 = acceleration2;
    settings.seed = seed;
    settings.verbose = verbose;
    settings.run_dir = optimizer_name;
    settings.save_history = !optimizer_name.empty();
    settings.save_loss = loss_history;
    return run_pso(f, d, lo, hi, settings);
}
