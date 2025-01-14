#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <random>
#include <cmath>
#include <fstream> // For CSV output
#include <iterator>
#include <map>
#include <set>
#include <string>


// Classical single-threaded matrix multiplication
std::vector<std::vector<double>> multiplyClassic(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, size_t& operations) {
    size_t rowsA = A.size();
    size_t colsA = A[0].size();
    size_t colsB = B[0].size();

    std::vector<std::vector<double>> C(rowsA, std::vector<double>(colsB, 0.0));
    operations = 0;

    for (size_t i = 0; i < rowsA; ++i) {
        for (size_t j = 0; j < colsB; ++j) {
            for (size_t k = 0; k < colsA; ++k) {
                C[i][j] += A[i][k] * B[k][j];
                ++operations;
            }
        }
    }
    return C;
}

// Optimized matrix multiplication function
// This function improves cache efficiency by reorganizing the loops
// and reducing repeated memory accesses. Instead of iterating over
// columns of B in the innermost loop, we loop over rows of A and columns of B,
// while keeping access to matrix A sequential and reducing redundant fetches.
// This optimization enhances performance for large matrices due to better
// cache utilization.

std::vector<std::vector<double>> multiplyOptimized(const std::vector<std::vector<double>>& A,
                                                   const std::vector<std::vector<double>>& B,
                                                   size_t& operations) {
    size_t rowsA = A.size();
    size_t colsA = A[0].size();
    size_t colsB = B[0].size();

    // Initialize the result matrix with zeros
    std::vector<std::vector<double>> C(rowsA, std::vector<double>(colsB, 0.0));
    operations = 0;

    // Loop through each row of matrix A
    for (size_t i = 0; i < rowsA; ++i) {
        // Access each element in the current row of A
        for (size_t k = 0; k < colsA; ++k) {
            double a = A[i][k]; // Pre-fetch the element from matrix A to avoid redundant memory accesses
            // Update each column in the result matrix
            for (size_t j = 0; j < colsB; ++j) {
                C[i][j] += a * B[k][j]; // Accumulate the product while accessing B sequentially
                ++operations; // Count the multiplication operation
            }
        }
    }

    return C; // Return the resulting matrix
}

// // Multi-threaded matrix multiplication - not used
// std::vector<std::vector<double>> multiplyMultiThreaded(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, size_t numThreads, size_t& operations) {
//     size_t rowsA = A.size();
//     size_t colsA = A[0].size();
//     size_t colsB = B[0].size();
//
//     std::vector<std::vector<double>> C(rowsA, std::vector<double>(colsB, 0.0));
//     std::vector<std::thread> threads;
//     operations = 0;
//
//     auto worker = [&](size_t startRow, size_t endRow, size_t& localOps) {
//         localOps = 0;
//         for (size_t i = startRow; i < endRow; ++i) {
//             for (size_t j = 0; j < colsB; ++j) {
//                 for (size_t k = 0; k < colsA; ++k) {
//                     C[i][j] += A[i][k] * B[k][j];
//                     ++localOps;
//                 }
//             }
//         }
//     };
//
//     size_t rowsPerThread = rowsA / numThreads;
//     std::vector<size_t> localOps(numThreads, 0);
//
//     for (size_t t = 0; t < numThreads; ++t) {
//         size_t startRow = t * rowsPerThread;
//         size_t endRow = (t == numThreads - 1) ? rowsA : startRow + rowsPerThread;
//         threads.emplace_back(worker, startRow, endRow, std::ref(localOps[t]));
//     }
//
//     for (size_t t = 0; t < threads.size(); ++t) {
//         threads[t].join();
//         operations += localOps[t];
//     }
//
//     return C;
// }

// Multi-threaded matrix multiplication using optimized approach
std::vector<std::vector<double>> multiplyMultiThreadedOptimized(const std::vector<std::vector<double>>& A,
                                                                const std::vector<std::vector<double>>& B,
                                                                size_t numThreads, size_t& operations) {
    size_t rowsA = A.size();
    size_t colsA = A[0].size();
    size_t colsB = B[0].size();

    std::vector<std::vector<double>> C(rowsA, std::vector<double>(colsB, 0.0));
    std::vector<std::thread> threads;
    operations = 0;

    // Worker function to handle a subset of rows of A
    auto worker = [&](size_t startRow, size_t endRow, size_t& localOps) {
        localOps = 0;
        for (size_t i = startRow; i < endRow; ++i) {
            for (size_t k = 0; k < colsA; ++k) {
                double a = A[i][k]; // Pre-fetch the element from matrix A
                for (size_t j = 0; j < colsB; ++j) {
                    C[i][j] += a * B[k][j]; // Sequential access to B and update C
                    ++localOps; // Count multiplication operations
                }
            }
        }
    };

    size_t rowsPerThread = rowsA / numThreads;
    std::vector<size_t> localOps(numThreads, 0);

    // Launch threads
    for (size_t t = 0; t < numThreads; ++t) {
        size_t startRow = t * rowsPerThread;
        size_t endRow = (t == numThreads - 1) ? rowsA : startRow + rowsPerThread;
        threads.emplace_back(worker, startRow, endRow, std::ref(localOps[t]));
    }

    // Wait for threads to finish and aggregate operations
    for (size_t t = 0; t < threads.size(); ++t) {
        threads[t].join();
        operations += localOps[t];
    }

    return C;
}

// ------------------- Strassen
// Helper function to pad a matrix with zeros to make it square and of size 2^n
std::vector<std::vector<double>> padMatrix(const std::vector<std::vector<double>>& mat, size_t newSize) {
    size_t rows = mat.size();
    size_t cols = mat[0].size();
    std::vector<std::vector<double>> padded(newSize, std::vector<double>(newSize, 0.0));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            padded[i][j] = mat[i][j];
        }
    }
    return padded;
}

// Helper function to unpad a matrix to its original size
std::vector<std::vector<double>> unpadMatrix(const std::vector<std::vector<double>>& mat, size_t originalRows, size_t originalCols) {
    std::vector<std::vector<double>> unpadded(originalRows, std::vector<double>(originalCols));

    for (size_t i = 0; i < originalRows; ++i) {
        for (size_t j = 0; j < originalCols; ++j) {
            unpadded[i][j] = mat[i][j];
        }
    }
    return unpadded;
}

// Add two matrices
std::vector<std::vector<double>> addMatrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    size_t n = A.size();
    std::vector<std::vector<double>> C(n, std::vector<double>(n));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}

// Subtract two matrices
std::vector<std::vector<double>> subtractMatrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    size_t n = A.size();
    std::vector<std::vector<double>> C(n, std::vector<double>(n));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
    return C;
}

// Strassen's matrix multiplication with threshold for switching to classical
std::vector<std::vector<double>> strassenMultiply(const std::vector<std::vector<double>>& A,
                                                  const std::vector<std::vector<double>>& B,
                                                  size_t& operations, size_t threshold = 64) {
    size_t n = A.size();
    if (n <= threshold) {
        return multiplyOptimized(A, B, operations);
    }

    size_t newSize = n / 2;
    std::vector<std::vector<double>> A11(newSize, std::vector<double>(newSize));
    std::vector<std::vector<double>> A12(newSize, std::vector<double>(newSize));
    std::vector<std::vector<double>> A21(newSize, std::vector<double>(newSize));
    std::vector<std::vector<double>> A22(newSize, std::vector<double>(newSize));

    std::vector<std::vector<double>> B11(newSize, std::vector<double>(newSize));
    std::vector<std::vector<double>> B12(newSize, std::vector<double>(newSize));
    std::vector<std::vector<double>> B21(newSize, std::vector<double>(newSize));
    std::vector<std::vector<double>> B22(newSize, std::vector<double>(newSize));

    // Ensure proper bounds for splitting matrices
    for (size_t i = 0; i < newSize; ++i) {
        for (size_t j = 0; j < newSize; ++j) {
            A11[i][j] = (i < A.size() && j < A[0].size()) ? A[i][j] : 0.0;
            A12[i][j] = (i < A.size() && j + newSize < A[0].size()) ? A[i][j + newSize] : 0.0;
            A21[i][j] = (i + newSize < A.size() && j < A[0].size()) ? A[i + newSize][j] : 0.0;
            A22[i][j] = (i + newSize < A.size() && j + newSize < A[0].size()) ? A[i + newSize][j + newSize] : 0.0;

            B11[i][j] = (i < B.size() && j < B[0].size()) ? B[i][j] : 0.0;
            B12[i][j] = (i < B.size() && j + newSize < B[0].size()) ? B[i][j + newSize] : 0.0;
            B21[i][j] = (i + newSize < B.size() && j < B[0].size()) ? B[i + newSize][j] : 0.0;
            B22[i][j] = (i + newSize < B.size() && j + newSize < B[0].size()) ? B[i + newSize][j + newSize] : 0.0;
        }
    }

    operations += 7 * newSize * newSize; // Additions and subtractions for M1-M7

    // M1 = (A11 + A22) * (B11 + B22)
    operations += newSize * newSize; // Addition for (A11 + A22)
    operations += newSize * newSize; // Addition for (B11 + B22)
    auto M1 = strassenMultiply(addMatrices(A11, A22), addMatrices(B11, B22), operations, threshold);

    // M2 = (A21 + A22) * B11
    operations += newSize * newSize; // Addition for (A21 + A22)
    auto M2 = strassenMultiply(addMatrices(A21, A22), B11, operations, threshold);

    // M3 = A11 * (B12 - B22)
    operations += newSize * newSize; // Subtraction for (B12 - B22)
    auto M3 = strassenMultiply(A11, subtractMatrices(B12, B22), operations, threshold);

    // M4 = A22 * (B21 - B11)
    operations += newSize * newSize; // Subtraction for (B21 - B11)
    auto M4 = strassenMultiply(A22, subtractMatrices(B21, B11), operations, threshold);

    // M5 = (A11 + A12) * B22
    operations += newSize * newSize; // Addition for (A11 + A12)
    auto M5 = strassenMultiply(addMatrices(A11, A12), B22, operations, threshold);

    // M6 = (A21 - A11) * (B11 + B12)
    operations += newSize * newSize; // Subtraction for (A21 - A11)
    operations += newSize * newSize; // Addition for (B11 + B12)
    auto M6 = strassenMultiply(subtractMatrices(A21, A11), addMatrices(B11, B12), operations, threshold);

    // M7 = (A12 - A22) * (B21 + B22)
    operations += newSize * newSize; // Subtraction for (A12 - A22)
    operations += newSize * newSize; // Addition for (B21 + B22)
    auto M7 = strassenMultiply(subtractMatrices(A12, A22), addMatrices(B21, B22), operations, threshold);

    // C11 = M1 + M4 - M5 + M7
    operations += newSize * newSize * 3; // Two additions and one subtraction
    auto C11 = addMatrices(subtractMatrices(addMatrices(M1, M4), M5), M7);

    // C12 = M3 + M5
    operations += newSize * newSize; // One addition
    auto C12 = addMatrices(M3, M5);

    // C21 = M2 + M4
    operations += newSize * newSize; // One addition
    auto C21 = addMatrices(M2, M4);

    // C22 = M1 + M3 - M2 + M6
    operations += newSize * newSize * 3; // Two additions and one subtraction
    auto C22 = addMatrices(subtractMatrices(addMatrices(M1, M3), M2), M6);

    std::vector<std::vector<double>> C(n, std::vector<double>(n));
    for (size_t i = 0; i < newSize; ++i) {
        for (size_t j = 0; j < newSize; ++j) {
            C[i][j] = C11[i][j];
            C[i][j + newSize] = C12[i][j];
            C[i + newSize][j] = C21[i][j];
            C[i + newSize][j + newSize] = C22[i][j];
        }
    }

    // Clear temporary matrices
    A11.clear();
    A12.clear();
    A21.clear();
    A22.clear();
    B11.clear();
    B12.clear();
    B21.clear();
    B22.clear();
    M1.clear();
    M2.clear();
    M3.clear();
    M4.clear();
    M5.clear();
    M6.clear();
    M7.clear();
    C11.clear();
    C12.clear();
    C21.clear();
    C22.clear();

    return C;
}

std::vector<std::vector<double>> strassen(const std::vector<std::vector<double>>& A,
                                          const std::vector<std::vector<double>>& B,
                                          size_t& operations) {
    size_t originalRows = A.size();
    size_t originalCols = B[0].size();

    size_t newSize = std::pow(2, std::ceil(std::log2(std::max({A.size(), A[0].size(), B.size(), B[0].size()}))));

    auto paddedA = padMatrix(A, newSize);
    auto paddedB = padMatrix(B, newSize);

    auto paddedC = strassenMultiply(paddedA, paddedB, operations);

    return unpadMatrix(paddedC, originalRows, originalCols);
}

size_t calculateBlockSize(size_t cacheSize, size_t elementSize = sizeof(double), int matrices = 3) {
    // Return fixed block size
    return 36; // Block size is fixed to 36
}

void multiplyBlocked(const std::vector<std::vector<double>>& A,
                     const std::vector<std::vector<double>>& B,
                     std::vector<std::vector<double>>& C,
                     size_t blockSize, size_t& operations) {
    // Get dimensions of the input matrices
    size_t rowsA = A.size();       // Number of rows in matrix A
    size_t colsA = A[0].size();    // Number of columns in matrix A
    size_t colsB = B[0].size();    // Number of columns in matrix B

    operations = 0; // Initialize the counter for multiplication operations

    // Iterate over row blocks of A
    for (size_t iBlock = 0; iBlock < rowsA; iBlock += blockSize) {
        // Iterate over column blocks of B
        for (size_t jBlock = 0; jBlock < colsB; jBlock += blockSize) {
            // Iterate over column blocks of A (or row blocks of B)
            for (size_t kBlock = 0; kBlock < colsA; kBlock += blockSize) {
                // Process individual elements within the current block
                for (size_t i = iBlock; i < std::min(iBlock + blockSize, rowsA); ++i) {
                    for (size_t k = kBlock; k < std::min(kBlock + blockSize, colsA); ++k) {
                        double a = A[i][k]; // Pre-fetch A[i][k] to minimize redundant memory access
                        for (size_t j = jBlock; j < std::min(jBlock + blockSize, colsB); ++j) {
                            C[i][j] += a * B[k][j]; // Multiply and accumulate into C[i][j]
                            ++operations;           // Increment operation counter
                        }
                    }
                }
            }
        }
    }
}


// ------------------- Helper and run functions

// List of files
std::vector<std::string> createdFiles;

// Delete all files
void cleanupAllFiles() {
    for (const auto& file : createdFiles) {
        if (std::remove(file.c_str()) == 0) {
            std::cout << "Deleted file: " << file << std::endl;
        }
    }
    createdFiles.clear(); // Очистка списка файлов
}

// Generate a random matrix with specified number of rows and columns
std::vector<std::vector<double>> generateMatrix(size_t rows, size_t cols) {
    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 10.0);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            matrix[i][j] = dist(rng);
        }
    }
    return matrix;
}

// Runner
void processAndRecord(const std::vector<std::vector<double>>& A,
                      const std::vector<std::vector<double>>& B,
                      size_t rowsA, size_t colsA, size_t rowsB, size_t colsB,
                      size_t totalSize, std::ofstream& csvFile, size_t& csvRowNo,
                      const std::set<std::string>& algorithms) {
    size_t operations = 0;
    size_t blockSize = 36; // Fixed block size
    size_t numThreads = std::thread::hardware_concurrency();

    std::set<std::string> localAlgorithms = algorithms; // Create a local copy
    // If "all" is present, expand to all algorithms
    if (localAlgorithms.count("all") > 0) {
        localAlgorithms = {"classic", "optimized", "multi-threaded", "strassen", "blocked"};
    }

    std::cout << "Algorithms to run: ";
    for (const auto& algo : algorithms) {
        std::cout << algo << " ";
    }
    std::cout << std::endl;

    // Check if CSV file is open
    if (!csvFile.is_open()) {
        std::cerr << "Error: Unable to write to CSV file. File not open.\n";
        return;
    }

    std::cout << "Using fixed block size: " << blockSize << " (assumed 32 KB cache)\n";
    std::cout << "Processing:\n";
    std::cout << "  Matrix A: " << rowsA << "x" << colsA << "\n";
    std::cout << "  Matrix B: " << rowsB << "x" << colsB << "\n";
    std::cout << "  Total size: " << totalSize << "\n";

    if (algorithms.count("classic") > 0) {
        auto start = std::chrono::high_resolution_clock::now();
        auto C1 = multiplyClassic(A, B, operations);
        auto end = std::chrono::high_resolution_clock::now();
        size_t singleThreadedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "  Single-threaded: Time = " << singleThreadedTime
                  << " ms, Operations = " << operations << "\n";
        csvFile << csvRowNo++ << "," << rowsA << "x" << colsA << "," << rowsB << "x" << colsB << ","
                << totalSize << ",Single-threaded," << singleThreadedTime << "," << operations << ",1" << std::endl;
        C1.clear();
    }

    if (algorithms.count("optimized") > 0) {
        operations = 0;
        auto start = std::chrono::high_resolution_clock::now();
        auto C3 = multiplyOptimized(A, B, operations);
        auto end = std::chrono::high_resolution_clock::now();
        size_t optimizedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "  Single-threaded Optimized: Time = " << optimizedTime
                  << " ms, Operations = " << operations << "\n";
        csvFile << csvRowNo++ << "," << rowsA << "x" << colsA << "," << rowsB << "x" << colsB << ","
                << totalSize << ",Optimized," << optimizedTime << "," << operations << ",1" << std::endl;
        C3.clear();
    }

    if (algorithms.count("multi-threaded") > 0) {
        operations = 0;
        auto start = std::chrono::high_resolution_clock::now();
        auto C2 = multiplyMultiThreadedOptimized(A, B, numThreads, operations);
        auto end = std::chrono::high_resolution_clock::now();
        size_t multiThreadedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "  Multi-threaded Optimized: Time = " << multiThreadedTime
                  << " ms, Operations = " << operations
                  << ", Threads = " << numThreads << "\n";
        csvFile << csvRowNo++ << "," << rowsA << "x" << colsA << "," << rowsB << "x" << colsB << ","
                << totalSize << ",Multi-threaded," << multiThreadedTime << "," << operations << "," << numThreads << std::endl;
        C2.clear();
    }

    if (algorithms.count("strassen") > 0) {
        operations = 0;
        auto start = std::chrono::high_resolution_clock::now();
        auto C4 = strassen(A, B, operations);
        auto end = std::chrono::high_resolution_clock::now();
        size_t strassenTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "  Strassen Optimized: Time = " << strassenTime
                  << " ms, Operations = " << operations << "\n";
        csvFile << csvRowNo++ << "," << rowsA << "x" << colsA << "," << rowsB << "x" << colsB << ","
                << totalSize << ",Strassen," << strassenTime << "," << operations << ",1" << std::endl;
        C4.clear();
    }

    if (algorithms.count("blocked") > 0) {
        operations = 0;
        try {
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<std::vector<double>> C5(rowsA, std::vector<double>(colsB, 0.0));
            multiplyBlocked(A, B, C5, blockSize, operations);
            auto end = std::chrono::high_resolution_clock::now();
            size_t blockedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "  Blocked Optimized: Time = " << blockedTime
                      << " ms, Operations = " << operations << "\n";
            csvFile << csvRowNo++ << "," << rowsA << "x" << colsA << "," << rowsB << "x" << colsB << ","
                    << totalSize << ",Blocked Optimized," << blockedTime << "," << operations << ",1" << std::endl;
            C5.clear();
        } catch (const std::exception& e) {
            std::cerr << "  Blocked Optimized failed: " << e.what() << "\n";
        }
    }
}


// ------------------- Command line help and other operations
void printHelp() {
    std::cout << "Matrix Multiplication Program\n";
    std::cout << "Usage:\n";
    std::cout << "  sem-release.exe [options]\n\n";

    std::cout << "Options:\n";
    std::cout << "  -help                Display this help message.\n\n";

    std::cout << "  -c <start> <step> <max>\n";
    std::cout << "                      Specify the start size, step size, and maximum size\n";
    std::cout << "                      for default matrix processing.\n";
    std::cout << "                      Example: -c 1000 200 3000\n\n";

    std::cout << "  -algo <algorithm>   Specify the algorithm to use. Available options are:\n";
    std::cout << "                      - classic: Classical single-threaded multiplication\n";
    std::cout << "                      - optimized: Optimized single-threaded multiplication\n";
    std::cout << "                      - multi-threaded: Multi-threaded multiplication\n";
    std::cout << "                      - strassen: Strassen's algorithm\n";
    std::cout << "                      - blocked: Blocked multiplication\n";
    std::cout << "                      - all: Use all algorithms (default)\n";
    std::cout << "                      Example: -algo optimized\n\n";

    std::cout << "  -test <A_dims> <B_dims> [...]\n";
    std::cout << "                      Test matrix multiplication for the specified pairs of\n";
    std::cout << "                      dimensions. Matrices will be generated, saved to disk,\n";
    std::cout << "                      multiplied, and the results compared.\n";
    std::cout << "                      Example: -test 400x500 500x300\n\n";

    std::cout << "All switches can be used separately or together. Examples:\n";

    std::cout << "  1. Run all algorithms for generated matrices with custom sizes:\n";
    std::cout << "     sem-release.exe -c 500 100 2000\n\n";

    std::cout << "  2. Run custom matrices processing with single algorithm:\n";
    std::cout << "     sem-release.exe -c 1000 200 3000 -algo classic\n\n";

    std::cout << "  3. Test blocked multiplication for specific matrices:\n";
    std::cout << "     sem-release.exe -test 400x500 500x300 -algo blocked classic\n\n";

    std::cout << "  4. Show this help message:\n";
    std::cout << "     sem-release.exe -help\n";
}

// For command line text matrix saving
void saveMatrixToText(const std::vector<std::vector<double>>& matrix, const std::string& filename) {
    std::ofstream file(filename);
    for (const auto& row : matrix) {
        for (const auto& value : row) {
            file << value << " ";
        }
        file << "\n";
    }
    file.close();
}

std::vector<std::vector<double>> loadMatrixFromText(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> matrix;
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<double> row((std::istream_iterator<double>(iss)), std::istream_iterator<double>());
        matrix.push_back(row);
    }

    file.close();
    return matrix;
}

std::pair<size_t, size_t> parseDimensions(const std::string& dimensions) {
    auto xPos = dimensions.find('x');
    if (xPos == std::string::npos) {
        throw std::invalid_argument("Invalid dimensions format. Expected 'NxM'.");
    }

    size_t rows = std::stoul(dimensions.substr(0, xPos));
    size_t cols = std::stoul(dimensions.substr(xPos + 1));
    return {rows, cols};
}

bool compareMatrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, double epsilon = 1e-6) {
    if (A.size() != B.size() || A[0].size() != B[0].size()) {
        return false;
    }
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            if (std::fabs(A[i][j] - B[i][j]) > epsilon) {
                return false;
            }
        }
    }
    return true;
}

void defaultMatrixProcessing(size_t startSize, size_t step, size_t maxSize, const std::set<std::string>& algorithms = {"all"}) {
    size_t rowsA = startSize, colsA = startSize;
    size_t rowsB = startSize, colsB = startSize;

    size_t operationNo = 1;  // For tracking logical operations
    size_t csvRowNo = 1;     // For tracking rows in the CSV file

    // Open CSV file to log results
    std::ofstream csvFile("results.csv");
    csvFile << "No,MatrixSizeA,MatrixSizeB,TotalSize,Algorithm,Time(ms),Operations,Threads" << std::endl;

    // Check if initial condition is met
    if ((rowsA + colsA) + (rowsB + colsB) > maxSize) {
        std::cout << "Initial condition not met: total size ("
                  << (rowsA + colsA) + (rowsB + colsB)
                  << ") exceeds maxSize (" << maxSize << "). No processing performed.\n";
        return;
    }

    while ((rowsA + colsA) + (rowsB + colsB) <= maxSize) {
        // Generate matrices with current dimensions
        auto A = generateMatrix(rowsA, colsA);
        auto B = generateMatrix(rowsB, colsB);

        // Check matrix compatibility
        if (colsA != rowsB) {
            std::cerr << "Incompatible matrices: A (" << rowsA << "x" << colsA
                      << ") and B (" << rowsB << "x" << colsB << ")" << std::endl;
            break;
        }

        // Save matrices to disk
        std::string filenameA = "matrix_A_" + std::to_string(rowsA) + "x" + std::to_string(colsA) + ".txt";
        std::string filenameB = "matrix_B_" + std::to_string(rowsB) + "x" + std::to_string(colsB) + ".txt";
        saveMatrixToText(A, filenameA);
        createdFiles.push_back(filenameA);
        saveMatrixToText(B, filenameB);
        createdFiles.push_back(filenameB);

        // Calculate total size
        size_t totalSize = (rowsA + colsA) + (rowsB + colsB);

        // Process matrices using the specified algorithms
        processAndRecord(A, B, rowsA, colsA, rowsB, colsB, totalSize, csvFile, csvRowNo, algorithms);

        // Log the operation number
        std::cout << "Operation No: " << operationNo << std::endl;

        // Update dimensions
        switch ((operationNo - 1) % 5) {
            case 0:  // Step 1: Increase rows in A
                rowsA += step;
                break;
            case 1:  // Step 2: Decrease rows in A, increase columns in B
                rowsA -= step;
                colsB += step;
                break;
            case 2:  // Step 3: Match rows in A to columns in B
                rowsA = colsB;
                break;
            case 3:  // Step 4: Increase columns in A and rows in B, but decrease rows in A and columns in B
                colsA += step;
                rowsB += step;
                rowsA -= step;
                colsB -= step;
                break;
            case 4:  // Step 5: Increase rows in A and columns in B
                rowsA = rowsB;   // Increase rows in A
                colsB = colsA;   // Increase columns in B
                break;
        }

        operationNo++;
    }

    csvFile.close();

    // Clean up all generated files
    cleanupAllFiles();

    std::cout << "Default matrix processing completed with algorithms.\n";
}

int runTests(const std::vector<std::pair<std::string, std::string>>& testPairs, const std::set<std::string>& algorithms) {
    std::vector<std::string> tempFiles;

    for (const auto& [dimA, dimB] : testPairs) {
        try {
            size_t operations = 0;
            auto [rowsA, colsA] = parseDimensions(dimA);
            auto [rowsB, colsB] = parseDimensions(dimB);

            if (colsA != rowsB) {
                std::cerr << "Matrix dimensions are incompatible for multiplication: "
                          << dimA << " and " << dimB << "\n";
                return 1;
            }

            auto A = generateMatrix(rowsA, colsA);
            auto B = generateMatrix(rowsB, colsB);

            std::map<std::string, std::vector<std::vector<double>>> results;

            if (algorithms.count("classic")) {
                auto start = std::chrono::high_resolution_clock::now();
                auto result = multiplyClassic(A, B, operations);
                auto end = std::chrono::high_resolution_clock::now();
                size_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

                results["classic"] = result;
                saveMatrixToText(result, "result_classic.txt");
                tempFiles.push_back("result_classic.txt");
                std::cout << "Classic multiplication completed. Operations: " << operations
                          << ", Time: " << elapsed << " ms\n";
            }

            if (algorithms.count("optimized")) {
                operations = 0;
                auto start = std::chrono::high_resolution_clock::now();
                auto result = multiplyOptimized(A, B, operations);
                auto end = std::chrono::high_resolution_clock::now();
                size_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

                results["optimized"] = result;
                saveMatrixToText(result, "result_optimized.txt");
                tempFiles.push_back("result_optimized.txt");
                std::cout << "Optimized multiplication completed. Operations: " << operations
                          << ", Time: " << elapsed << " ms\n";
            }

            if (algorithms.count("blocked")) {
                operations = 0;
                auto start = std::chrono::high_resolution_clock::now();
                std::vector<std::vector<double>> CBlocked(rowsA, std::vector<double>(colsB, 0.0));
                multiplyBlocked(A, B, CBlocked, 36, operations);
                auto end = std::chrono::high_resolution_clock::now();
                size_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

                results["blocked"] = CBlocked;
                saveMatrixToText(CBlocked, "result_blocked.txt");
                tempFiles.push_back("result_blocked.txt");
                std::cout << "Blocked multiplication completed. Operations: " << operations
                          << ", Time: " << elapsed << " ms\n";
            }

            if (algorithms.count("strassen")) {
                operations = 0;
                auto start = std::chrono::high_resolution_clock::now();
                auto result = strassen(A, B, operations);
                auto end = std::chrono::high_resolution_clock::now();
                size_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

                results["strassen"] = result;
                saveMatrixToText(result, "result_strassen.txt");
                tempFiles.push_back("result_strassen.txt");
                std::cout << "Strassen multiplication completed. Operations: " << operations
                          << ", Time: " << elapsed << " ms\n";
            }

            // Compare results
            if (results.size() > 1) {
                auto reference = results.begin()->second;
                bool allMatch = true;

                for (const auto& [algoName, result] : results) {
                    if (!compareMatrices(reference, result)) {
                        std::cerr << "Mismatch found in results of algorithm: " << algoName << "\n";
                        allMatch = false;
                    }
                }

                if (allMatch) {
                    std::cout << "All results match for test case: " << dimA << " and " << dimB << "\n";
                } else {
                    std::cerr << "Test failed: Results do not match.\n";
                    return 1;
                }
            }

        } catch (const std::exception& e) {
            std::cerr << "Error processing dimensions " << dimA << " and " << dimB << ": " << e.what() << "\n";
            return 1;
        }
    }

    // Cleanup temporary files
    std::cout << "Cleaning up temporary files...\n";
    for (const auto& file : tempFiles) {
        if (std::remove(file.c_str()) == 0) {
            std::cout << "Deleted file: " << file << std::endl;
        }
    }
    std::cout << "Cleanup completed.\n";
    return 0;
}

int main(int argc, char* argv[]) {
    size_t startSize = 500, step = 500, maxSize = 3000;
    std::set<std::string> algorithms = {"all"}; // Default: all algorithms
    bool isTestMode = false;
    std::vector<std::pair<std::string, std::string>> testPairs;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string command = argv[i];

        if (command == "-help") {
            printHelp();
            return 0;
        } else if (command == "-c" && i + 3 < argc) {
            // Parse size configuration
            startSize = std::stoul(argv[++i]);
            step = std::stoul(argv[++i]);
            maxSize = std::stoul(argv[++i]);
        } else if (command == "-algo") {
            // Parse algorithms to run
            algorithms.clear(); // Remove the default "all"
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                algorithms.insert(argv[++i]); // Add specified algorithms
            }
        } else if (command == "-test") {
            // Parse test mode and test pairs
            isTestMode = true;
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                std::string dimA = argv[++i];
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    std::string dimB = argv[++i];
                    testPairs.emplace_back(dimA, dimB);
                } else {
                    std::cerr << "Invalid arguments for -test. Dimensions must be provided in pairs.\n";
                    return 1;
                }
            }
        } else {
            std::cerr << "Unknown or invalid arguments: " << command << ". Use -help for usage.\n";
            return 1;
        }
    }

    // If no algorithms are specified or "all" is included, expand to all algorithms
    if (algorithms.empty() || algorithms.count("all") > 0) {
        algorithms = {"classic", "optimized", "multi-threaded", "strassen", "blocked"};
    }

    // Output selected algorithms for debugging
    std::cout << "Selected algorithms: ";
    for (const auto& algo : algorithms) {
        std::cout << algo << " ";
    }
    std::cout << std::endl;
    std::cout << "startSize: " << startSize << std::endl;
    std::cout << "step: " << step << std::endl;
    std::cout << "maxSize: " << maxSize << std::endl;

    // Run in test mode if specified
    if (isTestMode) {
        return runTests(testPairs, algorithms);
    }

    // Default matrix processing
    defaultMatrixProcessing(startSize, step, maxSize, algorithms);
    return 0;
}
