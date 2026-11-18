"""Build the C++ extension (gfo._pso) and/or the C++ example program.

Usage (from anywhere, with the project's Python environment active):

    python scripts/build.py                    # extension -> src/gfo/_pso<suffix>
    python scripts/build.py --target example   # C++ example -> build/gfo_example
    python scripts/build.py --target all --cxx path/to/g++ --native

On Windows the default compiler is MSYS2 ucrt64 g++
(C:/msys64/ucrt64/bin/g++.exe), override it with --cxx or the CXX environment
variable. The GCC runtime, OpenMP and winpthread are linked statically, so the
module only needs the Python DLL.
"""

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
from enum import Enum
from pathlib import Path

try:
    import pybind11
except ImportError as exc:
    raise ImportError(
        'pybind11 is not installed. Run: pip install -e ".[build]"'
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
PACKAGE_DIR = SRC_DIR / "gfo"
BUILD_DIR = ROOT / "build"

MODULE_NAME = "_pso"
EXTENSION_SOURCES = (SRC_DIR / "pso" / "pso.cpp", SRC_DIR / "pso" / "pso_bindings.cpp")
EXAMPLE_SOURCES = (SRC_DIR / "main.cpp", SRC_DIR / "pso" / "pso.cpp")
EXAMPLE_NAME = "gfo_example"

WINDOWS_DEFAULT_CXX = Path("C:/msys64/ucrt64/bin/g++.exe")
CXX_STANDARD = "-std=c++23"
IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"


class ETarget(Enum):
    """What to build."""

    EXTENSION = "extension"
    EXAMPLE = "example"
    ALL = "all"


def run_command(cmd: list[str], env: dict[str, str]) -> None:
    """Print and run a command, exit on failure."""
    print("Executing:\n  " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        sys.exit(f"\nBuild failed (exit code {result.returncode}).")


def main() -> None:
    """Parse the arguments and build."""
    parser = argparse.ArgumentParser(description="Build the gfo C++ extension.")
    parser.add_argument(
        "--target",
        choices=[target.value for target in ETarget],
        default=ETarget.EXTENSION.value,
        help="What to build (default: extension).",
    )
    parser.add_argument(
        "--cxx", help="C++ compiler (default: $CXX, MSYS2 g++ on Windows, g++)."
    )
    parser.add_argument(
        "--no-openmp", action="store_true", help="Build without OpenMP (serial)."
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help="Optimize for this CPU (-march=native). It may not run elsewhere.",
    )
    parser.add_argument("--debug", action="store_true", help="Debug build (-O0 -g).")
    args = parser.parse_args()
    target = ETarget(args.target)

    # Compiler: --cxx, then $CXX, then MSYS2 (Windows), then g++ on the PATH.
    candidates: list[str | None] = [args.cxx, os.environ.get("CXX")]
    if IS_WINDOWS:
        candidates.append(str(WINDOWS_DEFAULT_CXX))
    candidates.append("g++")
    cxx: Path | None = None
    for candidate in candidates:
        resolved: str | None = shutil.which(candidate) if candidate else None
        if resolved is not None:
            cxx = Path(resolved)
            break
    if cxx is None:
        sys.exit("ERROR: no C++ compiler found. Pass --cxx or set CXX.")
    print(f"Compiler: {cxx}")

    # The compiler's own folder goes first on the PATH (MinGW needs its DLLs).
    env: dict[str, str] = dict(os.environ)
    env["PATH"] = str(cxx.parent) + os.pathsep + env.get("PATH", "")

    common_flags: list[str] = [CXX_STANDARD, f"-I{SRC_DIR}", "-Wall", "-Wextra"]
    common_flags += ["-O0", "-g"] if args.debug else ["-O3", "-DNDEBUG"]
    if args.native:
        common_flags.append("-march=native")
    if not args.no_openmp:
        common_flags.append("-fopenmp")
    if IS_WINDOWS:
        # Static GCC runtime, OpenMP and winpthread: no MSYS2 DLLs at run time.
        common_flags.append("-static")

    if target in (ETarget.EXTENSION, ETarget.ALL):
        suffix: str = sysconfig.get_config_var("EXT_SUFFIX")
        output: Path = PACKAGE_DIR / f"{MODULE_NAME}{suffix}"

        # Windows links the Python import library, Linux and macOS resolve the
        # Python symbols when the module is loaded.
        link_flags: list[str] = []
        if IS_MACOS:
            link_flags = ["-undefined", "dynamic_lookup"]
        if IS_WINDOWS:
            version = f"{sys.version_info.major}{sys.version_info.minor}"
            library = Path(sys.base_prefix) / "libs" / f"python{version}.lib"
            if not library.is_file():
                sys.exit(f"ERROR: Python import library not found: {library}")
            link_flags = [str(library)]

        include_dirs: list[str] = [
            pybind11.get_include(),
            sysconfig.get_path("include"),
        ]
        cmd: list[str] = [str(cxx), "-shared"] + common_flags
        if not IS_WINDOWS:
            cmd += ["-fPIC", "-fvisibility=hidden"]
        cmd += [f"-I{path}" for path in dict.fromkeys(include_dirs)]
        cmd += [str(source) for source in EXTENSION_SOURCES]
        cmd += ["-o", str(output)] + link_flags

        print(f"Building {output.name} for Python {sys.version.split()[0]}")
        run_command(cmd, env)
        print(f"Built: {output}\n")

        # Import the fresh module in a clean interpreter.
        check_code = (
            "import gfo._pso as m; "
            "print(f'Import check passed: openmp={m.openmp}, "
            "max_threads={m.max_threads}')"
        )
        check_env: dict[str, str] = dict(os.environ)
        check_env["PYTHONPATH"] = (
            str(SRC_DIR) + os.pathsep + check_env.get("PYTHONPATH", "")
        )
        check = subprocess.run(
            [sys.executable, "-c", check_code], env=check_env, check=False
        )
        if check.returncode != 0:
            sys.exit("Import check FAILED: the module was built but cannot be loaded.")

    if target in (ETarget.EXAMPLE, ETarget.ALL):
        BUILD_DIR.mkdir(exist_ok=True)
        executable_suffix: str = ".exe" if IS_WINDOWS else ""
        example: Path = BUILD_DIR / f"{EXAMPLE_NAME}{executable_suffix}"
        cmd = [str(cxx)] + common_flags
        cmd += [str(source) for source in EXAMPLE_SOURCES]
        cmd += ["-o", str(example)]

        print(f"Building the C++ example {example.name}")
        run_command(cmd, env)
        print(f"Built: {example}\n")


if __name__ == "__main__":
    main()
