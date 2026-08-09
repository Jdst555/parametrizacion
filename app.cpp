
#include <igl/readOBJ.h>
#include <igl/colormap.h> // Para los mapas de calor

// Cabeceras del visor e ImGui
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>

#include <Eigen/Dense>
#include <iostream>
//#include <string>
//#include <vector>
//#include <algorithm>

// Variables globales para el estado de la aplicación
Eigen::MatrixXd V, V_uv, N;//vértices, coordenadas uv, normales
Eigen::MatrixXi F, FTC, FN;//triángulos
Eigen::MatrixXd Sigmas;

int current_method = 1; // 0: Harmonic, 1: LSCM, 2: ARAP
int current_metric = 1; // 0: Ninguna, 1: MIPS, 2: L2 Stretch, 3: Area
bool show_2d = false;

// Variables para controlar el mapa de color
bool auto_scale = true;
float metric_min = 2.0f;
float metric_max = 5.0f;

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
// CÁLCULO DE ESTADÍSTICAS GLOBALES
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
        double s1 = Sigmas(i, 0); // Mayor (L_infinito)
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
// 2. ACTUALIZAR COLORES (HEATMAP)
// -------------------------------------------------------------------
void update_colors(igl::opengl::glfw::Viewer& viewer)
{
    if (current_metric == 0 || Sigmas.rows() == 0) {
        viewer.data().set_colors(Eigen::RowVector3d(0.8, 0.8, 0.8)); // Gris por defecto
        return;
    }

    Eigen::VectorXd metric_values(F.rows());
    for (int i = 0; i < F.rows(); ++i) {
        double s1 = Sigmas(i, 0);
        double s2 = Sigmas(i, 1);

        if (s1 == 0 && s2 == 0) { metric_values(i) = 0; continue; }

        if (current_metric == 1) { // MIPS (Conformal)
            metric_values(i) = (s1 / s2) + (s2 / s1);
        }
        else if (current_metric == 2) { // L2 Stretch
            metric_values(i) = std::sqrt((s1 * s1 + s2 * s2) / 2.0);
        }
        else if (current_metric == 3) { // Area
            metric_values(i) = s1 * s2;
        }
    }

    // Mapear los valores a una escala de color jet
    Eigen::MatrixXd C;

    if (auto_scale) {
        // Modo por defecto: usa el min y max reales de toda la malla
        igl::colormap(igl::COLOR_MAP_TYPE_TURBO, metric_values, true, C);
    }
    else {
        // Modo manual: clampeamos los colores entre nuestros valores
        igl::colormap(igl::COLOR_MAP_TYPE_TURBO, metric_values, (double)metric_min, (double)metric_max, C);
    }

    viewer.data().set_colors(C);
}

// -------------------------------------------------------------------
// MAIN
// -------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <ruta_malla.obj>" << std::endl;
        return 1;
    }

    // 1. Cargar la malla
    if (!igl::readOBJ(argv[2], V, V_uv, N, F, FTC, FN)) {
        std::cerr << "Error al cargar la malla." << std::endl;
        return 1;
    }

    // Configurar el visor de libigl
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V, F);
    viewer.data().show_lines = false;
    viewer.core().lighting_factor = 0.0;

    // Valores por defecto mejorados para el visor
    viewer.core().background_color << 0.8f, 0.8f, 0.8f, 1.0f; // Fondo gris claro (no tan negro)
    viewer.core().light_position << 0.0f, 5.0f, 10.0f;        // Luz al frente y un poco arriba
    viewer.data().double_sided = true;                        // Prevenir caras oscuras por normales invertidas

    // Configurar ImGui
    igl::opengl::glfw::imgui::ImGuiPlugin plugin;
    viewer.plugins.push_back(&plugin);

    igl::opengl::glfw::imgui::ImGuiMenu menu;
    plugin.widgets.push_back(&menu);

    // Definir la interfaz de usuario
    menu.callback_draw_viewer_window = [&]()
        {
            ImGui::Text("Analisis de Parametrizacion");

            ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

            // Selector de Métrica
            ImGui::Text("Análisis de Distorsión");
            if (ImGui::Combo("Métrica", &current_metric, "Ninguna (Color sólido)\0MIPS (Conformal)\0L2 Stretch (Distancias)\0Cambio de Area\0")) {
                update_colors(viewer); // Refrescar color al cambiar de métrica
            }

            // --- NUEVOS CONTROLES DE RANGO DE COLOR ---
            if (current_metric != 0) {
                ImGui::Indent(); // Tabulamos un poco para que quede bonito
                if (ImGui::Checkbox("Auto-Escalar Color", &auto_scale)) {
                    update_colors(viewer);
                }

                if (!auto_scale) {
                    // Si el usuario mueve estos sliders, repintamos en tiempo real
                    if (ImGui::DragFloat("Rango Min", &metric_min, 0.05f)) update_colors(viewer);
                    if (ImGui::DragFloat("Rango Max", &metric_max, 0.05f)) update_colors(viewer);
                }
                ImGui::Unindent();
            }
            ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

            // Controles de Vista
            ImGui::Text("Vista");
            if (ImGui::RadioButton("Modelo 3D", !show_2d)) {
                show_2d = false;
                viewer.data().set_vertices(V);
                //viewer.data().compute_normals();
                viewer.core().align_camera_center(V, F);
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Espacio UV (2D)", show_2d)) {
                if (V_uv.rows() > 0) {
                    show_2d = true;
                    // En libigl, pasamos las uvs como vértices, rellenando Z con 0
                    Eigen::MatrixXd V_flat = Eigen::MatrixXd::Zero(V_uv.rows(), 3);
                    V_flat.leftCols(2) = V_uv;
                    viewer.data().set_vertices(V_flat);
                    viewer.core().align_camera_center(V_flat, F);
                }
            }
            ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

            // --- PANEL DE ESTADÍSTICAS ---
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Resultados Numericos"); // Título en color verde
            ImGui::Text("Tiempo de CPU: %.4f s", compute_time);

            if (num_flips > 0) {
                ImGui::TextColored(ImVec4(1.0f, 0.2f, 0.2f, 1.0f), "Triangulos invertidos (Flips): %d", num_flips);
            }
            else {
                ImGui::Text("Triangulos invertidos: 0 (Biyectivo)");
            }

            ImGui::Spacing();
            ImGui::Text("Distorsion Conformal (MIPS, ideal=2.0)");
            ImGui::BulletText("Media ponderada: %.3f", avg_mips);
            ImGui::BulletText("Maximo error: %.3f", max_mips);

            ImGui::Spacing();
            ImGui::Text("Distorsion por Estiramiento (L2, ideal=1.0)");
            ImGui::BulletText("Media ponderada: %.3f", avg_l2);
            ImGui::BulletText("Maximo error: %.3f", max_l2);

            ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
            ImGui::Text("Configuracion de Renderizado");

            // 1. Doble cara (Ayuda muchísimo si las normales están invertidas)
            ImGui::Checkbox("Iluminar ambas caras (Double Sided)", &viewer.data().double_sided);

            

            // 3. Mover la posición de la luz
            // Extraemos la posición actual de la luz a un array compatible con ImGui
            float light_pos[3] = {
                viewer.core().light_position(0),
                viewer.core().light_position(1),
                viewer.core().light_position(2)
            };
            if (ImGui::DragFloat3("Posicion Luz", light_pos, 0.1f)) {
                // Si el usuario mueve el slider, actualizamos la luz en libigl
                viewer.core().light_position << light_pos[0], light_pos[1], light_pos[2];
            }

            // 4. Cambiar el color de fondo (Un fondo más claro ayuda al contraste)
            float bg_color[3] = {
                viewer.core().background_color(0),
                viewer.core().background_color(1),
                viewer.core().background_color(2)
            };
            if (ImGui::ColorEdit3("Color de Fondo", bg_color)) {
                viewer.core().background_color << bg_color[0], bg_color[1], bg_color[2], 1.0f;
            }
        };

    
    viewer.data().set_uv(V_uv, FTC);
    update_colors(viewer);

    viewer.launch();
    return 0;
}