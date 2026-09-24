#include <igl/boundary_loop.h>
#include <igl/harmonic.h>
#include <igl/lscm.h>
#include <igl/arap.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/read_triangle_mesh.h>
#include <igl/writeOBJ.h> // <-- Cambiado para exportar UVs explícitamente

#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>

// Variables globales para el estado de la aplicación
Eigen::MatrixXd V, V_uv; // vértices de la malla 3d y de la parametrización 2d
Eigen::MatrixXi F;       // triángulos
Eigen::VectorXi bnd;     // vértices del borde
Eigen::MatrixXd Sigmas;

// Variables para las estadísticas
double compute_time = 0.0;
int num_flips = 0;
double avg_mips = 0.0, max_mips = 0.0;
double avg_l2 = 0.0, max_l2 = 0.0;

// -------------------------------------------------------------------
// 1. CÁLCULO DE VALORES SINGULARES
// -------------------------------------------------------------------
void compute_sigmas()
{
    Sigmas.resize(F.rows(), 2);
    for (int i = 0; i < F.rows(); i++)
    {
        int v1 = F(i, 0), v2 = F(i, 1), v3 = F(i, 2);
        Eigen::Vector3d p1 = V.row(v1), p2 = V.row(v2), p3 = V.row(v3);
        Eigen::Vector2d u1 = V_uv.row(v1).head<2>(), u2 = V_uv.row(v2).head<2>(), u3 = V_uv.row(v3).head<2>();

        Eigen::Matrix<double, 3, 2> dP;
        dP.col(0) = p2 - p1; dP.col(1) = p3 - p1;

        Eigen::Matrix<double, 2, 2> dU;
        dU.col(0) = u2 - u1; dU.col(1) = u3 - u1;

        if (dU.determinant() == 0) {
            Sigmas(i, 0) = 0.0; Sigmas(i, 1) = 0.0;
            continue;
        }

        Eigen::Matrix<double, 3, 2> J = dP * dU.inverse();
        Eigen::Matrix2d I = J.transpose() * J;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(I);
        Eigen::Vector2d eigenvalues = eigensolver.eigenvalues();

        Sigmas(i, 0) = std::sqrt(std::max(0.0, eigenvalues(1))); // Mayor (sigma 1)
        Sigmas(i, 1) = std::sqrt(std::max(0.0, eigenvalues(0))); // Menor (sigma 2)
    }
}

// -------------------------------------------------------------------
// 2. CÁLCULO DE ESTADÍSTICAS GLOBALES
// -------------------------------------------------------------------
void compute_stats()
{
    num_flips = 0;
    max_mips = 0.0; max_l2 = 0.0;
    double sum_mips = 0.0, sum_l2 = 0.0;
    double total_area_3d = 0.0;

    for (int i = 0; i < F.rows(); ++i)
    {
        // 1. Calcular área 3D del triángulo original para ponderar
        Eigen::Vector3d p1 = V.row(F(i, 0));
        Eigen::Vector3d p2 = V.row(F(i, 1));
        Eigen::Vector3d p3 = V.row(F(i, 2));
        double area_3d = 0.5 * ((p2 - p1).cross(p3 - p1)).norm();
        total_area_3d += area_3d;

        // 2. Detectar Triángulos Invertidos (Flips) usando el área 2D con signo
        Eigen::Vector2d u1 = V_uv.row(F(i, 0)).head<2>();
        Eigen::Vector2d u2 = V_uv.row(F(i, 1)).head<2>();
        Eigen::Vector2d u3 = V_uv.row(F(i, 2)).head<2>();

        // Producto cruzado en 2D
        double area_2d = 0.5 * ((u2.x() - u1.x()) * (u3.y() - u1.y()) - (u2.y() - u1.y()) * (u3.x() - u1.x()));

        if (area_2d <= 0.0) {
            num_flips++;
        }

        // 3. Obtener valores singulares previamente calculados
        double s1 = Sigmas(i, 0); // Mayor
        double s2 = Sigmas(i, 1); // Menor

        if (s1 > 0 && s2 > 0) {
            // Métrica MIPS (Conformal, ideal = 2.0)
            double mips = (s1 / s2) + (s2 / s1);
            max_mips = std::max(max_mips, mips);
            sum_mips += mips * area_3d;

            // Métrica L2 Stretch (ideal = 1.0)
            double l2 = std::sqrt((s1 * s1 + s2 * s2) / 2.0);
            max_l2 = std::max(max_l2, l2);
            sum_l2 += l2 * area_3d;
        }
    }

    // Calcular las medias ponderadas
    if (total_area_3d > 0) {
        avg_mips = sum_mips / total_area_3d;
        avg_l2 = sum_l2 / total_area_3d;
    }
}

// -------------------------------------------------------------------
// 3. EJECUCIÓN DE LOS MÉTODOS DE PARAMETRIZACIÓN
// -------------------------------------------------------------------
void compute_parameterization(int method)
{
    if (bnd.size() == 0) {
        std::cerr << "Error: La malla no tiene bordes." << std::endl;
        return;
    }

    // --- INICIO CRONÓMETRO ---
    auto start_time = std::chrono::high_resolution_clock::now();

    if (method == 0) { // HARMONIC
        Eigen::MatrixXd bnd_uv;
        igl::map_vertices_to_circle(V, bnd, bnd_uv);
        igl::harmonic(V, F, bnd, bnd_uv, 1, V_uv);
    }
    else if (method == 1) { // LSCM
        Eigen::VectorXi b(2, 1);
        b(0) = bnd(0);
        b(1) = bnd(bnd.size() / 2);
        Eigen::MatrixXd bc(2, 2);
        bc << 0, 0, 1, 0;
        igl::lscm(V, F, b, bc, V_uv);
    }
    else if (method == 2) { // ARAP
        Eigen::VectorXi b(2, 1);
        b(0) = bnd(0); b(1) = bnd(bnd.size() / 2);
        Eigen::MatrixXd bc(2, 2); bc << 0, 0, 1, 0;

        Eigen::MatrixXd V_uv_initial;
        igl::lscm(V, F, b, bc, V_uv_initial); // Initial guess

        igl::ARAPData arap_data;
        arap_data.max_iter = 100;
        igl::arap_precomputation(V, F, 2, b, arap_data);

        V_uv = V_uv_initial;
        igl::arap_solve(bc, arap_data, V_uv);
    }

    // --- FIN CRONÓMETRO ---
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    compute_time = diff.count();

    // Calculamos valores singulares y estadísticas globales
    compute_sigmas();
    compute_stats();
}

// -------------------------------------------------------------------
// MAIN (CLI sin interfaz gráfica)
// -------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc < 4) {
        std::cerr << "Uso: " << argv[0] << " <input_malla.obj> <metodo> <output_malla.obj>" << std::endl;
        std::cerr << "  Metodos disponibles:" << std::endl;
        std::cerr << "    0 = Harmonic (Tutte)" << std::endl;
        std::cerr << "    1 = LSCM" << std::endl;
        std::cerr << "    2 = ARAP" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    int method = std::stoi(argv[2]);
    std::string output_file = argv[3];

    if (method < 0 || method > 2) {
        std::cerr << "Error: Metodo no valido. Seleccione 0, 1 o 2." << std::endl;
        return 1;
    }

    // 1. Cargar la malla
    if (!igl::read_triangle_mesh(input_file, V, F)) {
        std::cerr << "Error al cargar la malla de entrada: " << input_file << std::endl;
        return 1;
    }

    // 2. Extraer bordes
    igl::boundary_loop(F, bnd);
    if (bnd.size() == 0) {
        std::cerr << "Error: La malla de entrada no tiene bordes abiertos (es cerrada o hermetica)." << std::endl;
        return 1;
    }

    // 3. Ejecutar algoritmo
    std::cout << "Calculando parametrizacion..." << std::endl;
    compute_parameterization(method);

    // 4. Imprimir estadisticas a la consola
    std::cout << "\n================ ESTADISTICAS ================" << std::endl;
    std::cout << "Metodo utilizado: " << (method == 0 ? "Harmonic" : (method == 1 ? "LSCM" : "ARAP")) << std::endl;
    std::cout << "Tiempo de CPU: " << compute_time << " segundos" << std::endl;
    std::cout << "Triangulos invertidos (Flips): " << num_flips << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "Distorsion Conformal (MIPS, ideal=2.0)" << std::endl;
    std::cout << "  Media ponderada: " << avg_mips << std::endl;
    std::cout << "  Maximo error: " << max_mips << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "Distorsion por Estiramiento (L2, ideal=1.0)" << std::endl;
    std::cout << "  Media ponderada: " << avg_l2 << std::endl;
    std::cout << "  Maximo error: " << max_l2 << std::endl;
    std::cout << "==============================================\n" << std::endl;

    // 5. Guardar la malla 3D con sus atributos UV en formato OBJ
    if (V_uv.rows() > 0) {
        // Matrices vacías para normales (las calcularemos después o las ignoramos si solo queremos UVs)
        Eigen::MatrixXd CN;
        Eigen::MatrixXi FN;

        // F actúa también como FTC (Índices de caras de las UVs), 
        // ya que cada vértice i tiene su respectiva coordenada UV i.
        if (igl::writeOBJ(output_file, V, F, CN, FN, V_uv, F)) {
            std::cout << "Malla 3D y coordenadas UV guardadas exitosamente en: " << output_file << std::endl;
        }
        else {
            std::cerr << "Error al guardar el archivo de salida." << std::endl;
            return 1;
        }
    }

    return 0;
}