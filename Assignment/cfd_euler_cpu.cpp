#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <sstream>

#include <omp.h>
#include <chrono>

using namespace std;

// ------------------------------------------------------------
// A simple struct to hold loop statistics
// ------------------------------------------------------------
struct LoopStat {
    const char* name;
    int calls = 0;
    double total_ms = 0.0;
};

// -----------------------------------------------------------------
// Function to print loop statistics
// -----------------------------------------------------------------
static inline void print_stat(std::ostream& os, const LoopStat& s) {
    const double total_s = s.total_ms * 1e-3;
    const double avg_us   = (s.calls > 0) ? (s.total_ms * 1e3) / s.calls : 0.0;

    os << std::setw(40) << std::left << s.name
       << " ran " << std::setw(4) << s.calls << " times;" 
       << " total " << std::setw(7) << std::fixed << std::setprecision(6) << total_s << " s;"
       << " avg " << std::fixed << std::setprecision(3) << avg_us << " µs\n";
}

// ------------------------------------------------------------
// Global parameters
// ------------------------------------------------------------
const double gamma_val = 1.4;   // Ratio of specific heats
const double CFL = 0.5;         // CFL number

// ------------------------------------------------------------
// Compute pressure from the conservative variables
// ------------------------------------------------------------
double pressure(double rho, double rhou, double rhov, double E) {
    double u = rhou / rho;
    double v = rhov / rho;
    double kinetic = 0.5 * rho * (u * u + v * v);
    return (gamma_val - 1.0) * (E - kinetic);
}

// ------------------------------------------------------------
// Compute flux in the x-direction
// ------------------------------------------------------------
void fluxX(double rho, double rhou, double rhov, double E, 
           double& frho, double& frhou, double& frhov, double& fE) {
    double u = rhou / rho;
    double p = pressure(rho, rhou, rhov, E);
    frho = rhou;
    frhou = rhou * u + p;
    frhov = rhov * u;
    fE = (E + p) * u;
}

// ------------------------------------------------------------
// Compute flux in the y-direction
// ------------------------------------------------------------
void fluxY(double rho, double rhou, double rhov, double E,
           double& frho, double& frhou, double& frhov, double& fE) {
    double v = rhov / rho;
    double p = pressure(rho, rhou, rhov, E);
    frho = rhov;
    frhou = rhou * v;
    frhov = rhov * v + p;
    fE = (E + p) * v;
}

// ------------------------------------------------------------
// Main simulation routine
// ------------------------------------------------------------
int main(){
    // Open output file for writing results
    std::ofstream out("output_cpu_16x.txt");
    // Check if output file opened successfully
    if (!out) {
        std::cerr << "Error opening output_cpu_16x.txt\n";
        return 1;
    }

    // Create loop statistics variables
    LoopStat st_main{"MAIN time-stepping loop"};

    LoopStat st_bc_left{"Boundary condition left (inflow)"};
    LoopStat st_bc_right{"Boundary condition right (outflow)"};
    LoopStat st_bc_bottom{"Boundary condition bottom (reflective)"};
    LoopStat st_bc_top{"Boundary condition top (reflective)"};

    LoopStat st_update{"Update interior (Lax-Friedrichs)"};
    LoopStat st_copy{"Copy back new -> old"};
    LoopStat st_kin{"Kinetic energy sum"};

    // ----- Grid and domain parameters -----
    const int Nx = 3200;         // Number of cells in x (excluding ghost cells)
    const int Ny = 1600;         // Number of cells in y
    const double Lx = 2.0;      // Domain length in x
    const double Ly = 1.0;      // Domain length in y
    const double dx = Lx / Nx;
    const double dy = Ly / Ny;

    // Create flat arrays (with ghost cells)
    const int total_size = (Nx + 2) * (Ny + 2);
    
    vector<double> rho(total_size);
    vector<double> rhou(total_size);
    vector<double> rhov(total_size);
    vector<double> E(total_size);
    
    vector<double> rho_new(total_size);
    vector<double> rhou_new(total_size);
    vector<double> rhov_new(total_size);
    vector<double> E_new(total_size);
    
    // A mask to mark solid cells (inside the cylinder)
    vector<int> solid(total_size, 0);

    // ----- Obstacle (cylinder) parameters -----
    const double cx = 0.5;      // Cylinder center x
    const double cy = 0.5;      // Cylinder center y
    const double radius = 0.1;  // Cylinder radius

    // ----- Free-stream initial conditions (inflow) -----
    const double rho0 = 1.0;
    const double u0 = 1.0;
    const double v0 = 0.0;
    const double p0 = 1.0;
    const double E0 = p0/(gamma_val - 1.0) + 0.5*rho0*(u0*u0 + v0*v0);

    // ----- Initialize grid and obstacle mask -----
    for (int i = 0; i < Nx+2; i++){
        for (int j = 0; j < Ny+2; j++){
            // Compute cell center coordinates
            double x = (i - 0.5) * dx;
            double y = (j - 0.5) * dy;
            // Mark cell as solid if inside the cylinder
            if ((x - cx)*(x - cx) + (y - cy)*(y - cy) <= radius * radius) {
                solid[i*(Ny+2)+j] = true;
                // For a wall, we set zero velocity
                rho[i*(Ny+2)+j] = rho0;
                rhou[i*(Ny+2)+j] = 0.0;
                rhov[i*(Ny+2)+j] = 0.0;
                E[i*(Ny+2)+j] = p0/(gamma_val - 1.0);
            } else {
                solid[i*(Ny+2)+j] = false;
                rho[i*(Ny+2)+j] = rho0;
                rhou[i*(Ny+2)+j] = rho0 * u0;
                rhov[i*(Ny+2)+j] = rho0 * v0;
                E[i*(Ny+2)+j] = E0;
            }
        }
    }

    // ----- Determine time step from CFL condition -----
    double c0 = sqrt(gamma_val * p0 / rho0);
    double dt = CFL * min(dx, dy) / (fabs(u0) + c0)/2.0;

    // ----- Time stepping parameters -----
    const int nSteps = 2000;

    auto t1_main = std::chrono::high_resolution_clock::now();
    // ----- Main time-stepping loop -----
    for (int n = 0; n < nSteps; n++){
        st_bc_left.calls++;
        auto t1_bc_left = std::chrono::high_resolution_clock::now();
        // --- Apply boundary conditions on ghost cells ---
        // Left boundary (inflow): fixed free-stream state
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < Ny+2; j++){
            rho[0*(Ny+2)+j] = rho0;
            rhou[0*(Ny+2)+j] = rho0*u0;
            rhov[0*(Ny+2)+j] = rho0*v0;
            E[0*(Ny+2)+j] = E0;
        }
        auto t2_bc_left = std::chrono::high_resolution_clock::now();
        st_bc_left.total_ms += std::chrono::duration<double, std::milli>(t2_bc_left - t1_bc_left).count();

        st_bc_right.calls++;
        auto t1_bc_right = std::chrono::high_resolution_clock::now();
        // Right boundary (outflow): copy from the interior
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < Ny+2; j++){
            rho[(Nx+1)*(Ny+2)+j] = rho[Nx*(Ny+2)+j];
            rhou[(Nx+1)*(Ny+2)+j] = rhou[Nx*(Ny+2)+j];
            rhov[(Nx+1)*(Ny+2)+j] = rhov[Nx*(Ny+2)+j];
            E[(Nx+1)*(Ny+2)+j] = E[Nx*(Ny+2)+j];
        }
        auto t2_bc_right = std::chrono::high_resolution_clock::now();
        st_bc_right.total_ms += std::chrono::duration<double, std::milli>(t2_bc_right - t1_bc_right).count();

        st_bc_bottom.calls++;
        auto t1_bc_bottom = std::chrono::high_resolution_clock::now();
        // Bottom boundary: reflective
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Nx+2; i++){
            rho[i*(Ny+2)+0] = rho[i*(Ny+2)+1];
            rhou[i*(Ny+2)+0] = rhou[i*(Ny+2)+1];
            rhov[i*(Ny+2)+0] = -rhov[i*(Ny+2)+1];
            E[i*(Ny+2)+0] = E[i*(Ny+2)+1];
        }
        auto t2_bc_bottom = std::chrono::high_resolution_clock::now();
        st_bc_bottom.total_ms += std::chrono::duration<double, std::milli>(t2_bc_bottom - t1_bc_bottom).count();

        st_bc_top.calls++;
        auto t1_bc_top = std::chrono::high_resolution_clock::now();
        // Top boundary: reflective
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Nx+2; i++){
            rho[i*(Ny+2)+(Ny+1)] = rho[i*(Ny+2)+Ny];
            rhou[i*(Ny+2)+(Ny+1)] = rhou[i*(Ny+2)+Ny];
            rhov[i*(Ny+2)+(Ny+1)] = -rhov[i*(Ny+2)+Ny];
            E[i*(Ny+2)+(Ny+1)] = E[i*(Ny+2)+Ny];
        }
        auto t2_bc_top = std::chrono::high_resolution_clock::now();
        st_bc_top.total_ms += std::chrono::duration<double, std::milli>(t2_bc_top - t1_bc_top).count();

        st_update.calls++;
        auto t1_update = std::chrono::high_resolution_clock::now();
        // --- Update interior cells using a Lax-Friedrichs scheme ---
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i <= Nx; i++){
            for (int j = 1; j <= Ny; j++){
                // If the cell is inside the solid obstacle, do not update it
                if (solid[i*(Ny+2)+j]) {
                    rho_new[i*(Ny+2)+j] = rho[i*(Ny+2)+j];
                    rhou_new[i*(Ny+2)+j] = rhou[i*(Ny+2)+j];
                    rhov_new[i*(Ny+2)+j] = rhov[i*(Ny+2)+j];
                    E_new[i*(Ny+2)+j] = E[i*(Ny+2)+j];
                    continue;
                }

                // Compute a Lax averaging of the four neighboring cells
                rho_new[i*(Ny+2)+j] = 0.25 * (rho[(i+1)*(Ny+2)+j] + rho[(i-1)*(Ny+2)+j] + 
                                            rho[i*(Ny+2)+(j+1)] + rho[i*(Ny+2)+(j-1)]);
                rhou_new[i*(Ny+2)+j] = 0.25 * (rhou[(i+1)*(Ny+2)+j] + rhou[(i-1)*(Ny+2)+j] + 
                                            rhou[i*(Ny+2)+(j+1)] + rhou[i*(Ny+2)+(j-1)]);
                rhov_new[i*(Ny+2)+j] = 0.25 * (rhov[(i+1)*(Ny+2)+j] + rhov[(i-1)*(Ny+2)+j] + 
                                            rhov[i*(Ny+2)+(j+1)] + rhov[i*(Ny+2)+(j-1)]);
                E_new[i*(Ny+2)+j] = 0.25 * (E[(i+1)*(Ny+2)+j] + E[(i-1)*(Ny+2)+j] + 
                                        E[i*(Ny+2)+(j+1)] + E[i*(Ny+2)+(j-1)]);

                // Compute fluxes
                double fx_rho1, fx_rhou1, fx_rhov1, fx_E1;
                double fx_rho2, fx_rhou2, fx_rhov2, fx_E2;
                double fy_rho1, fy_rhou1, fy_rhov1, fy_E1;
                double fy_rho2, fy_rhou2, fy_rhov2, fy_E2;

                fluxX(rho[(i+1)*(Ny+2)+j], rhou[(i+1)*(Ny+2)+j], rhov[(i+1)*(Ny+2)+j], E[(i+1)*(Ny+2)+j],
                    fx_rho1, fx_rhou1, fx_rhov1, fx_E1);
                fluxX(rho[(i-1)*(Ny+2)+j], rhou[(i-1)*(Ny+2)+j], rhov[(i-1)*(Ny+2)+j], E[(i-1)*(Ny+2)+j],
                    fx_rho2, fx_rhou2, fx_rhov2, fx_E2);
                fluxY(rho[i*(Ny+2)+(j+1)], rhou[i*(Ny+2)+(j+1)], rhov[i*(Ny+2)+(j+1)], E[i*(Ny+2)+(j+1)],
                    fy_rho1, fy_rhou1, fy_rhov1, fy_E1);
                fluxY(rho[i*(Ny+2)+(j-1)], rhou[i*(Ny+2)+(j-1)], rhov[i*(Ny+2)+(j-1)], E[i*(Ny+2)+(j-1)],
                    fy_rho2, fy_rhou2, fy_rhov2, fy_E2);

                // Apply flux differences
                double dtdx = dt / (2 * dx);
                double dtdy = dt / (2 * dy);
                
                rho_new[i*(Ny+2)+j] -= dtdx * (fx_rho1 - fx_rho2) + dtdy * (fy_rho1 - fy_rho2);
                rhou_new[i*(Ny+2)+j] -= dtdx * (fx_rhou1 - fx_rhou2) + dtdy * (fy_rhou1 - fy_rhou2);
                rhov_new[i*(Ny+2)+j] -= dtdx * (fx_rhov1 - fx_rhov2) + dtdy * (fy_rhov1 - fy_rhov2);
                E_new[i*(Ny+2)+j] -= dtdx * (fx_E1 - fx_E2) + dtdy * (fy_E1 - fy_E2);
            }
        }
        auto t2_update = std::chrono::high_resolution_clock::now();
        st_update.total_ms += std::chrono::duration<double, std::milli>(t2_update - t1_update).count();

        st_copy.calls++;
        auto t1_copy = std::chrono::high_resolution_clock::now();
        // Copy updated values back
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i <= Nx; i++){
            for (int j = 1; j <= Ny; j++){
                rho[i*(Ny+2)+j] = rho_new[i*(Ny+2)+j];
                rhou[i*(Ny+2)+j] = rhou_new[i*(Ny+2)+j];
                rhov[i*(Ny+2)+j] = rhov_new[i*(Ny+2)+j];
                E[i*(Ny+2)+j] = E_new[i*(Ny+2)+j];
            }
        }
        auto t2_copy = std::chrono::high_resolution_clock::now();
        st_copy.total_ms += std::chrono::duration<double, std::milli>(t2_copy - t1_copy).count();

        st_kin.calls++;
        auto t1_kin = std::chrono::high_resolution_clock::now();
        // Calculate total kinetic energy
        double total_kinetic = 0.0;
        #pragma omp parallel for collapse(2) reduction(+:total_kinetic) schedule(static)
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                double u = rhou[i*(Ny+2)+j] / rho[i*(Ny+2)+j];
                double v = rhov[i*(Ny+2)+j] / rho[i*(Ny+2)+j];
                total_kinetic += 0.5 * rho[i*(Ny+2)+j] * (u * u + v * v);
            }
        }
        auto t2_kin = std::chrono::high_resolution_clock::now();
        st_kin.total_ms += std::chrono::duration<double, std::milli>(t2_kin - t1_kin).count();

        if ((n % 50 == 0) || n == nSteps - 1) {
            out << "Step " << n << " completed, total kinetic energy: " << total_kinetic << "\n";
        }
    }

    auto t2_main = std::chrono::high_resolution_clock::now();
    st_main.calls = 1;
    st_main.total_ms = std::chrono::duration<double, std::milli>(t2_main - t1_main).count();
    

    // Print statistics
    out << "\n============================================ TIMING SUMMARY ============================================\n";
    print_stat(out, st_main);
    print_stat(out, st_bc_left);
    print_stat(out, st_bc_right);
    print_stat(out, st_bc_bottom);
    print_stat(out, st_bc_top);
    print_stat(out, st_update);
    print_stat(out, st_copy);
    print_stat(out, st_kin);
    out << "========================================================================================================\n";

    return 0;
}