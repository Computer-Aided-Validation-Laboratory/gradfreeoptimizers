"""Test configuration: figures are only saved, never shown, so no GUI backend."""

import matplotlib

matplotlib.use("Agg")
