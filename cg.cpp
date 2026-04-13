#include <cmath>
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

// Function prototypes
void matrix_vector_multiply_csr(const double* values, const int* col_indices, const int* row_start, const double* x, double* result, int n);

void conjugate_gradient_csr(const double* values, const int* col_indices, const int* row_start, const double* b, double* x, int n, int max_iterations, double tolerance) {
    double* r = new double[n];
    double* p = new double[n];
    double* Ap = new double[n];
    double* Ax = new double[n];

    int nnz = row_start[n]; // Total number of non-zero entries in the matrix

#pragma omp target data map(alloc: r[0:n], p[0:n], Ap[0:n], Ax[0:n]) \
                        map(to: values[0:nnz], col_indices[0:nnz], row_start[0:n+1], b[0:n]) \
                        map(tofrom: x[0:n])
{
    // Initial step: compute r = b - A*x
    matrix_vector_multiply_csr(values, col_indices, row_start, x, Ax, n);
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < n; ++i) {
        r[i] = b[i] - Ax[i];
        p[i] = r[i];
    }

    double rsold = 0.0;
    #pragma omp target teams distribute parallel for reduction(+:rsold)
    for (int i = 0; i < n; ++i) {
        rsold += r[i] * r[i];
    }

    for (int i = 0; i < max_iterations; ++i) {
        matrix_vector_multiply_csr(values, col_indices, row_start, p, Ap, n);
        double pAp = 0.0;
        #pragma omp target teams distribute parallel for reduction(+:pAp)
        for (int j = 0; j < n; ++j) {
            pAp += p[j] * Ap[j];
        }
        double alpha = rsold / pAp;

        #pragma omp target teams distribute parallel for
        for (int j = 0; j < n; ++j) {
            x[j] += alpha * p[j];
            r[j] -= alpha * Ap[j];
        }

        double rsnew = 0.0;
        #pragma omp target teams distribute parallel for reduction(+:rsnew)
        for (int j = 0; j < n; ++j) {
            rsnew += r[j] * r[j];
        }

        if (sqrt(rsnew) < tolerance) {
	    std::cout << "Final residual " << sqrt(rsnew) << std::endl;
            break;
        } else if (i%100 == 0) {
	    std::cout << i << " residual " << sqrt(rsnew) << std::endl;
	}

        #pragma omp target teams distribute parallel for
        for (int j = 0; j < n; ++j) {
            p[j] = r[j] + (rsnew / rsold) * p[j];
        }

        rsold = rsnew;
    }

    }  // end of target data

    delete[] r;
    delete[] p;
    delete[] Ap;
    delete[] Ax;
}

void matrix_vector_multiply_csr(const double* values, const int* col_indices, const int* row_start, const double* x, double* result, int n) {
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < n; ++i) {
        result[i] = 0.0;
        for (int j = row_start[i]; j < row_start[i + 1]; ++j) {
            result[i] += values[j] * x[col_indices[j]];
        }
    }
}


int main() {
    const int gridSize = 2000;
    const int n = gridSize * gridSize;
    std::vector<double> values;
    std::vector<int> col_indices;
    std::vector<int> row_start(n + 1);
    std::vector<double> b(n, 1.0); // Heat source vector
    std::vector<double> x(n, 0.0); // Solution vector

    // Build CSR representation for A
    int nnz = 0; // Non-zero count
    for (int i = 0; i < n; ++i) {
        row_start[i] = nnz;
        // Diagonal element
        values.push_back(4.0);
        col_indices.push_back(i);
        nnz++;

        // Off-diagonal elements
        if (i >= gridSize) { // Upper neighbor
            values.push_back(-1.0);
            col_indices.push_back(i - gridSize);
            nnz++;
        }
        if (i % gridSize != 0) { // Left neighbor
            values.push_back(-1.0);
            col_indices.push_back(i - 1);
            nnz++;
        }
        if ((i + 1) % gridSize != 0) { // Right neighbor
            values.push_back(-1.0);
            col_indices.push_back(i + 1);
            nnz++;
        }
        if (i < n - gridSize) { // Lower neighbor
            values.push_back(-1.0);
            col_indices.push_back(i + gridSize);
            nnz++;
        }
    }
    row_start[n] = nnz;

    // Convert std::vector to raw pointers for the solver
    double* val_array = &values[0];
    int* col_array = &col_indices[0];
    int* row_start_array = &row_start[0];
    double* b_array = &b[0];
    double* x_array = &x[0];

    // Solve the system using a CSR-based Conjugate Gradient method
    int max_iterations = 1000;
    double tolerance = 1e-8;
    auto t1 = std::chrono::high_resolution_clock::now();
    conjugate_gradient_csr(val_array, col_array, row_start_array, b_array, x_array, n, max_iterations, tolerance);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms\n";
    // Output the solution
    std::cout << "Temperature distribution:" << std::endl;
    for (int i = n/2; i < n/2+1; ++i) {
        std::cout << "Temperature at (" << i / gridSize << ", " << i % gridSize << ") = " << x[i] << std::endl;
        if ((i + 1) % gridSize == 0) std::cout << std::endl;
    }

    return 0;
}