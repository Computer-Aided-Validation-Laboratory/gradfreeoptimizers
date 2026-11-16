"""Build the C++ extension (gfo._pso) and, optionally, the C++ example program.

Usage (from anywhere, with the project's Python environment active):

    python scripts/build.py              # Python extension -> src/gfo/_pso<suffix>
    python scripts/build.py --example    # also the C++ example -> build/gfo_example(.exe)
    python scripts/build.py --cxx path/to/g++ --native

On Windows the default compiler is MSYS2 ucrt64 g++ (C:/msys64/ucrt64/bin/g++.exe),
override it with --cxx or the CXX environment variable. The GCC runtime, OpenMP
and winpthread are linked statically, so the module only needs the Python DLL.
"""

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

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


def find_compiler(requested: str | None) -> Path:
    """Pick the C++ compiler: --cxx, then $CXX, then MSYS2 (Windows), then g++ on PATH."""
    candidates = [requested, os.environ.get("CXX")]
    if IS_WINDOWS:
        candidates.append(str(WINDOWS_DEFAULT_CXX))
    candidates.append("g++")

    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return Path(resolved)
    sys.exit("ERROR: no C++ compiler found. Pass --cxx or set CXX.")


def compiler_env(cxx: Path) -> dict:
    """Environment with the compiler's own folder first on PATH (MinGW needs its DLLs)."""
    env = dict(os.environ)
    env["PATH"] = str(cxx.parent) + os.pathsep + env.get("PATH", "")
    return env


def get_python_include_flags() -> list[str]:
    """Include flags for pybind11 and the Python headers."""
    try:
        import pybind11
    except ImportError:
        sys.exit("ERROR: pybind11 is not installed. Run: pip install -e \".[build]\"")
    includes = [pybind11.get_include(), sysconfig.get_path("include")]
    return [f"-I{path}" for path in dict.fromkeys(includes)]


def get_python_link_flags() -> list[str]:
    """Windows needs the Python import library, Linux/macOS resolve symbols at load time."""
    if IS_MACOS:
        return ["-undefined", "dynamic_lookup"]
    if not IS_WINDOWS:
        return []

    version = f"{sys.version_info.major}{sys.version_info.minor}"
    library = Path(sys.base_prefix) / "libs" / f"python{version}.lib"
    if not library.is_file():
        sys.exit(f"ERROR: Python import library not found: {library}")
    return [str(library)]


def get_common_flags(args: argparse.Namespace) -> list[str]:
    """Flags shared by the extension and the example program."""
    flags = [CXX_STANDARD, f"-I{SRC_DIR}", "-Wall", "-Wextra"]
    flags += ["-O0", "-g"] if args.debug else ["-O3", "-DNDEBUG"]
    if args.native:
        flags.append("-march=native")
    if not args.no_openmp:
        flags.append("-fopenmp")
    if IS_WINDOWS:
        # Static GCC runtime, OpenMP and winpthread: no MSYS2 DLLs needed at run time.
        flags.append("-static")
    return flags


def run_command(cmd: list[str], env: dict) -> None:
    """Print and run a command, exit on failure."""
    print("Executing:\n  " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        sys.exit(f"\nBuild failed (exit code {result.returncode}).")


def build_extension(cxx: Path, args: argparse.Namespace) -> Path:
    """Compile gfo._pso into src/gfo/."""
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or (".pyd" if IS_WINDOWS else ".so")
    output = PACKAGE_DIR / f"{MODULE_NAME}{suffix}"

    flags = ["-shared"] + get_common_flags(args)
    if not IS_WINDOWS:
        flags += ["-fPIC", "-fvisibility=hidden"]

    cmd = [str(cxx)] + flags + get_python_include_flags()
    cmd += [str(source) for source in EXTENSION_SOURCES]
    cmd += ["-o", str(output)] + get_python_link_flags()

    print(f"Building {output.name} for Python {sys.version.split()[0]} ({sys.executable})")
    run_command(cmd, compiler_env(cxx))
    print(f"Built: {output}\n")
    return output


def build_example(cxx: Path, args: argparse.Namespace) -> Path:
    """Compile the C++ example program into build/."""
    BUILD_DIR.mkdir(exist_ok=True)
    output = BUILD_DIR / (EXAMPLE_NAME + (".exe" if IS_WINDOWS else ""))
    cmd = [str(cxx)] + get_common_flags(args)
    cmd += [str(source) for source in EXAMPLE_SOURCES]
    cmd += ["-o", str(output)]

    print(f"Building the C++ example {output.name}")
    run_command(cmd, compiler_env(cxx))
    print(f"Built: {output}\n")
    return output


def check_import() -> None:
    """Import the fresh module in a clean interpreter and report what it was built with."""
    code = ("import gfo._pso as m; "
            "print(f'Import check passed: openmp={m.openmp}, max_threads={m.max_threads}')")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-c", code], env=env)
    if result.returncode != 0:
        sys.exit("Import check FAILED: the module was built but could not be imported.")


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(description="Build the gfo C++ extension.")
    parser.add_argument("--cxx", help="C++ compiler (default: $CXX, MSYS2 g++ on Windows, g++).")
    parser.add_argument("--example", action="store_true",
                        help="Also build the C++ example program (src/main.cpp).")
    parser.add_argument("--example-only", action="store_true",
                        help="Only build the C++ example program.")
    parser.add_argument("--no-openmp", action="store_true", help="Build without OpenMP (serial).")
    parser.add_argument("--native", action="store_true",
                        help="Optimize for this CPU (-march=native). The binary may not run elsewhere.")
    parser.add_argument("--debug", action="store_true", help="Debug build (-O0 -g).")
    return parser


def main() -> None:
    """Parse the arguments and build."""
    args = build_parser().parse_args()
    cxx = find_compiler(args.cxx)
    print(f"Compiler: {cxx}")

    if not args.example_only:
        build_extension(cxx, args)
        check_import()
    if args.example or args.example_only:
        build_example(cxx, args)


if __name__ == "__main__":
    main()
