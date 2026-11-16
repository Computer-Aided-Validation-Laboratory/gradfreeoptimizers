"""Shared constants for the post-processing package.

File and column names here mirror what the C++ optimizer (src/pso/pso.hpp)
writes. If the backend naming changes, this is the only place that needs
editing.
"""

# Console formatting (matches the C++ banners).
BANNER_WIDTH = 99

# Output settings.
POST_SUFFIX = "_post"
DEFAULT_DPI = 300

# Optimizer files, saved by the backend in <run_dir>/<optimizer_dir>/.
OPT_SUBDIR = "optimizer"              # Default of PSOSettings::optimizer_dir.
OPT_HISTORY_FILENAME = "history.csv"  # Only written for 2D runs.
OPT_DOMAIN_FILENAME = "domain.csv"    # Only written for 2D runs.
OPT_LOSS_FILENAME = "loss.csv"
OPT_POST_SUBDIR = "optimizer"         # Subfolder of <run>_post/ for optimizer figures.

# Optimizer animation settings.
ANIMATION_DPI = 100       # Far lower than DEFAULT_DPI, animations are many frames.
ANIMATION_FPS = 12
MAX_ANIMATION_FRAMES = 120  # Iterations are skipped evenly above this.

# Column names used in the CSVs.
COL_X = "x"
COL_Y = "y"
COL_VALUE = "f"
COL_ITERATION = "iteration"
DOMAIN_COLUMNS = ("lo_x", "lo_y", "hi_x", "hi_y")
