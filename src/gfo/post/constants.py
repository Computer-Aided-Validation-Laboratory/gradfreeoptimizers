"""Constants of the post-processing package.

File and column names mirror what the C++ optimizer (src/pso/pso.hpp) writes
in ``<run_dir>/<optimizer_dir>/``. 

If the backend naming changes, this is the only place that needs editing.
"""

HISTORY_FILENAME = "history.csv"  # Only written for 2D runs.
DOMAIN_FILENAME = "domain.csv"  # Only written for 2D runs.
LOSS_FILENAME = "loss.csv"

HISTORY_COLUMNS = ("iteration", "x", "y", "f")
# Read in the order of OptimizerHistory.domain: (x_lo, x_hi, y_lo, y_hi).
DOMAIN_COLUMNS = ("lo_x", "hi_x", "lo_y", "hi_y")
LOSS_COLUMNS = ("iteration", "f")

FIGURE_DPI = 300
ANIMATION_DPI = 100  # Far lower than FIGURE_DPI, animations are many frames.
ANIMATION_FPS = 12
MAX_ANIMATION_FRAMES = 120  # Iterations are skipped evenly above this.
