#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <tuple>

std::tuple<std::vector<double>, std::vector<double>> run_particle_filter(
    int N, int T, std::vector<double> velocity,
    double sigma_motion, double sigma_sensor)
{
    std::vector<std::vector<double>> particles(N, std::vector<double>(2));
    std::vector<double> weights(N, 1.0 / N);
    std::vector<double> true_pos = {0.0, 0.0};

    std::default_random_engine gen;
    std::uniform_real_distribution<> uniform_dist(0.0, 10.0);
    std::normal_distribution<> motion_dist(0.0, sigma_motion);
    std::normal_distribution<> sensor_dist(0.0, sigma_sensor);

    // Inicializar partículas aleatorias
    for (int i = 0; i < N; ++i) {
        particles[i][0] = uniform_dist(gen);
        particles[i][1] = uniform_dist(gen);
    }

    for (int t = 0; t < T; ++t) {
        // Movimiento del estado real
        true_pos[0] += velocity[0] + motion_dist(gen);
        true_pos[1] += velocity[1] + motion_dist(gen);

        // Medición con ruido
        std::vector<double> z = {
            true_pos[0] + sensor_dist(gen),
            true_pos[1] + sensor_dist(gen)
        };

        // Mover partículas
        for (int i = 0; i < N; ++i) {
            particles[i][0] += velocity[0] + motion_dist(gen);
            particles[i][1] += velocity[1] + motion_dist(gen);
        }

        // Calcular pesos
        double total_weight = 0.0;
        for (int i = 0; i < N; ++i) {
            double dx = particles[i][0] - z[0];
            double dy = particles[i][1] - z[1];
            double dist2 = dx * dx + dy * dy;
            weights[i] = std::exp(-0.5 * dist2 / (sigma_sensor * sigma_sensor));
            total_weight += weights[i];
        }

        if (total_weight > 0) {
            for (int i = 0; i < N; ++i) weights[i] /= total_weight;
        } else {
            for (int i = 0; i < N; ++i) weights[i] = 1.0 / N;
        }

        // Reamostrado
        std::discrete_distribution<> dist(weights.begin(), weights.end());
        std::vector<std::vector<double>> new_particles(N);
        for (int i = 0; i < N; ++i) {
            int idx = dist(gen);
            new_particles[i] = particles[idx];
        }
        particles = new_particles;
    }

    // Estimación
    std::vector<double> estimate = {0.0, 0.0};
    for (int i = 0; i < N; ++i) {
        estimate[0] += particles[i][0];
        estimate[1] += particles[i][1];
    }
    estimate[0] /= N;
    estimate[1] /= N;

    return std::make_tuple(estimate, true_pos);
}

// Binding de PyBind11
PYBIND11_MODULE(pf_cbind, m) {
    m.doc() = "Filtro de partículas optimizado en C++";
    
    m.def("run_particle_filter", &run_particle_filter, "Ejecuta el filtro de partículas",
            pybind11::arg("N") = 1000, pybind11::arg("T") = 50,
            pybind11::arg("velocity") = std::vector<double>{1.0, 0.5},
            pybind11::arg("sigma_motion") = 0.5,
            pybind11::arg("sigma_sensor") = 1.0);
}