# Matrix Multiplication Benchmarks

**Semester project (B6B36PCC)** — C++ console utility comparing matrix multiplication algorithms.

| | |
|---|---|
| **Stack** | C++20, CMake |
| **Algorithms** | Classic, loop-optimized, multi-threaded, Strassen, cache-blocked |
| **Report** | [PDF results](Zubkov__PCC_semetr%C3%A1lka__n%C3%A1soben%C3%AD_matic.pdf) |
| **Downloads** | [Windows](https://github.com/vladimirzubkov/sem-pcc-matrix_multiplication/releases/latest/download/sem-release-windows.exe) · [Linux](https://github.com/vladimirzubkov/sem-pcc-matrix_multiplication/releases/latest/download/sem-release-linux) |

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Binary: `bin/Release/sem-release` (Linux/macOS) or `bin/Release/sem-release.exe` (Windows).

## Usage

```bash
sem-release -help
sem-release -c 500 100 2000
sem-release -c 1000 200 3000 -algo classic optimized
sem-release -test 400x500 500x300 -algo blocked classic
```

Options:

- `-c <start> <step> <max>` — benchmark with generated matrices (default: 500 500 3000)
- `-algo <name> ...` — `classic`, `optimized`, `multi-threaded`, `strassen`, `blocked`, or `all`
- `-test <AxB> <BxC> ...` — generate matrices, multiply, compare results across algorithms

## CI

GitHub Actions builds Windows and Linux binaries on every push to `main` and publishes them to [Releases](https://github.com/vladimirzubkov/sem-pcc-matrix_multiplication/releases).
