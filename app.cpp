#include <igl/readOBJ.h>
#include <igl/colormap.h> 

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/stb/write_image.h>

#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <vector>

// Variables globales para el estado de la aplicación
Eigen::MatrixXd V, V_uv, N;
Eigen::MatrixXi F, FTC, FN;
Eigen::MatrixXd Sigmas;
Eigen::VectorXd metric_values;
float min_metric_value = 0;
float max_metric_value = 0;

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

// PARA LOS HISTOGRAMAS
const int NUM_BINS = 50; // Cantidad de barras en el histograma
std::vector<float> hist_mips(NUM_BINS, 0.0f);
std::vector<float> hist_l2(NUM_BINS, 0.0f);
std::vector<float> hist_area(NUM_BINS, 0.0f);

// -------------------------------------------------------------------
// 1. CÁLCULO DE VALORES SINGULARES
// -------------------------------------------------------------------
void compute_sigmas()
{
    Sigmas.resize(F.rows(), 2);
    for (int i = 0; i < F.rows(); i++)
    {
        // Índices 3D
        int v1 = F(i, 0), v2 = F(i, 1), v3 = F(i, 2);
        // Índices 2D (UVs) - ¡CRUCIAL USAR FTC!
        int uv1 = FTC(i, 0), uv2 = FTC(i, 1), uv3 = FTC(i, 2);

        Eigen::Vector3d p1 = V.row(v1), p2 = V.row(v2), p3 = V.row(v3);
        Eigen::Vector2d u1 = V_uv.row(uv1).head<2>(), u2 = V_uv.row(uv2).head<2>(), u3 = V_uv.row(uv3).head<2>();

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

    // Limpiar los histogramas anteriores
    std::fill(hist_mips.begin(), hist_mips.end(), 0.0f);
    std::fill(hist_l2.begin(), hist_l2.end(), 0.0f);
    std::fill(hist_area.begin(), hist_area.end(), 0.0f);

    // Definir los rangos para los histogramas (ajustar maximos)
    double min_mips = 2.0, max_mips_hist = 5.0; // MIPS ideal es 2.0
    double min_l2 = 1.0, max_l2_hist = 3.0;     // L2 ideal es 1.0
    double min_area = 0.0, max_area_hist = 2.0; // Area ideal es 1.0


    for (int i = 0; i < F.rows(); ++i)
    {
        Eigen::Vector3d p1 = V.row(F(i, 0));
        Eigen::Vector3d p2 = V.row(F(i, 1));
        Eigen::Vector3d p3 = V.row(F(i, 2));
        double area_3d = 0.5 * ((p2 - p1).cross(p3 - p1)).norm();
        total_area_3d += area_3d;

        // ¡USAR FTC PARA UVS!
        int uv1 = FTC(i, 0), uv2 = FTC(i, 1), uv3 = FTC(i, 2);
        Eigen::Vector2d u1 = V_uv.row(uv1).head<2>();
        Eigen::Vector2d u2 = V_uv.row(uv2).head<2>();
        Eigen::Vector2d u3 = V_uv.row(uv3).head<2>();

        double area_2d = 0.5 * ((u2.x() - u1.x()) * (u3.y() - u1.y()) - (u2.y() - u1.y()) * (u3.x() - u1.x()));

        if (area_2d <= 0.0) {
            num_flips++;
        }

        double s1 = Sigmas(i, 0);
        double s2 = Sigmas(i, 1);

        if (s1 > 0 && s2 > 0) {
            double mips = (s1 / s2) + (s2 / s1);
            max_mips = std::max(max_mips, mips);
            sum_mips += mips * area_3d;

            int bin_mips = (int)(((mips - min_mips) / (max_mips_hist - min_mips)) * NUM_BINS);
            bin_mips = std::max(0, std::min(NUM_BINS - 1, bin_mips));
            hist_mips[bin_mips] += (float)area_3d; // Ponderado por área!

            double l2 = std::sqrt((s1 * s1 + s2 * s2) / 2.0);
            max_l2 = std::max(max_l2, l2);
            sum_l2 += l2 * area_3d;

            int bin_l2 = (int)(((l2 - min_l2) / (max_l2_hist - min_l2)) * NUM_BINS);
            bin_l2 = std::max(0, std::min(NUM_BINS - 1, bin_l2));
            hist_l2[bin_l2] += (float)area_3d;

            double area_metric = s1 * s2;
            int bin_a = (int)(((area_metric - min_area) / (max_area_hist - min_area)) * NUM_BINS);
            bin_a = std::max(0, std::min(NUM_BINS - 1, bin_a));
            hist_area[bin_a] += (float)area_3d;
        }
    }

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
        viewer.data().set_colors(Eigen::RowVector3d(0.8, 0.8, 0.8));
        return;
    }

    metric_values.resize(F.rows());
    for (int i = 0; i < F.rows(); ++i) {
        double s1 = Sigmas(i, 0);
        double s2 = Sigmas(i, 1);

        if (s1 == 0 && s2 == 0) { metric_values(i) = 0; continue; }

        if (current_metric == 1) {
            metric_values(i) = (s1 / s2) + (s2 / s1);
        }
        else if (current_metric == 2) {
            metric_values(i) = std::sqrt((s1 * s1 + s2 * s2) / 2.0);
        }
        else if (current_metric == 3) {
            metric_values(i) = s1 * s2;
        }
    }

    Eigen::MatrixXd C;
    if (auto_scale) {
        igl::colormap(igl::COLOR_MAP_TYPE_TURBO, metric_values, true, C);
    }
    else {
        igl::colormap(igl::COLOR_MAP_TYPE_TURBO, metric_values, (double)metric_min, (double)metric_max, C);
    }
    min_metric_value = metric_values.minCoeff();
    max_metric_value = metric_values.maxCoeff();
    viewer.data().set_colors(C);
}
//imprimir matrices
template <typename Derived>
void print_mat(const Eigen::MatrixBase<Derived>& mat,
    std::ostream& os = std::cout)
{
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            os << mat(i, j);
            if (j + 1 < mat.cols()) os << " ";
        }
        if (i + 1 < mat.rows()) os << "\n";
    }
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

    // CORRECCIÓN 1: Leer desde argv[1], no argv[2]
    if (!igl::readOBJ(argv[1], V, V_uv, N, F, FTC, FN)) {
        std::cerr << "Error al cargar la malla." << std::endl;
        return 1;
    }
	
    // CORRECCIÓN 2: ¡Llamar a las funciones de cálculo!
    compute_sigmas();
    compute_stats();

    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V, F);
    viewer.data().set_uv(V_uv, FTC);
    viewer.data().show_lines = false;
    viewer.core().lighting_factor = 0.0;
    viewer.core().background_color << 0.8f, 0.8f, 0.8f, 1.0f;
    viewer.core().light_position << 0.0f, 5.0f, 10.0f;
    viewer.data().double_sided = true;

    igl::opengl::glfw::imgui::ImGuiPlugin plugin;
    viewer.plugins.push_back(&plugin);
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    plugin.widgets.push_back(&menu);

    menu.callback_draw_viewer_window = [&]()
        {
            ImGui::Text("Analisis de Parametrizacion");
            ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

            ImGui::Text("Analisis de Distorsion");
            if (ImGui::Combo("Metrica", &current_metric, "Ninguna (Color sólido)\0MIPS (Conformal)\0L2 Stretch (Distancias)\0Cambio de Area\0")) {
                update_colors(viewer);
            }

            if (current_metric != 0) {
                ImGui::Indent();
                if (ImGui::Checkbox("Auto-Escalar Color", &auto_scale)) {
                    update_colors(viewer);
                }
                if (!auto_scale) {
                    if (ImGui::DragFloat("Rango Min", &metric_min, 0.05f)) update_colors(viewer);
                    if (ImGui::DragFloat("Rango Max", &metric_max, 0.05f)) update_colors(viewer);
                }
                ImGui::Unindent();
            }
            ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

            ImGui::Text("Vista");
            if (ImGui::RadioButton("Modelo 3D", !show_2d)) {
                show_2d = false;
                viewer.data().clear();
                viewer.data().set_mesh(V, F);
                viewer.data().set_uv(V_uv, FTC);
                viewer.data().show_lines = true;
                update_colors(viewer);
                viewer.core().align_camera_center(V, F);
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Espacio UV (2D)", show_2d)) {
                if (V_uv.rows() > 0) {
                    show_2d = true;
                    Eigen::MatrixXd V_flat = Eigen::MatrixXd::Zero(V_uv.rows(), 3);
                    V_flat.leftCols(2) = V_uv;

                    viewer.data().clear();
                    // CORRECCIÓN 4: Renderizar usando FTC para V_flat
                    viewer.data().set_mesh(V_flat, FTC);
                    viewer.data().show_lines = false;
                    update_colors(viewer);
                    viewer.core().align_camera_center(V_flat, FTC);
                }
            }
            ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
            ImGui::Text("Exportar");

            if (ImGui::Button("Guardar PNG (Alta Resolucion)")) {
                if (V_uv.rows() > 0) {
                    // 1. Prepare the 2D flat vertices
                    Eigen::MatrixXd V_flat = Eigen::MatrixXd::Zero(V_uv.rows(), 3);
                    V_flat.leftCols(2) = V_uv;

                    // 2. Set the geometry to the 2D map and apply the heatmap colors
                    viewer.data().clear();
                    viewer.data().set_mesh(V_flat, FTC);
                    viewer.data().show_lines = true;
                    update_colors(viewer);

                    // 3. FORCE PERFECT 2D CAMERA SETTINGS
                    viewer.core().align_camera_center(V_flat, FTC);
                    viewer.core().orthographic = true;
                    viewer.core().trackball_angle = Eigen::Quaternionf::Identity(); // Perfect top-down
                    viewer.core().camera_zoom = 0.85f; // Zoom out slightly to leave a nice margin

                    // 4. Allocate memory for a High-Res Image (e.g., 2048 x 2048 pixels)
                    // By specifying a size here, it ignores your tiny window size and renders in 4K!
                    int img_size = 2048;
                    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R(img_size, img_size);
                    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> G(img_size, img_size);
                    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> B(img_size, img_size);
                    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> A(img_size, img_size);

                    // 5. Render the scene off-screen to the matrices!
                    // We pass 'true' as the second argument so the viewer recalculates the matrices 
                    // to apply the orthographic settings we just forced.
                    viewer.core().draw_buffer(viewer.data(), true, R, G, B, A);

                    // 6. Write out the PNG file to your project folder
                    igl::stb::write_image("parametrizacion_2d.png", R, G, B, A);

                    // 7. Sync the UI so the user knows they are now looking at the 2D view
                    show_2d = true;

                    std::cout << "¡Imagen guardada exitosamente como parametrizacion_2d.png!" << std::endl;
                }
            }
            ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Resultados Numericos");
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

            ImGui::Spacing();
            ImGui::Text("Valores min y max de la metrica seleccionada");
            ImGui::BulletText(std::to_string(min_metric_value).c_str());
			ImGui::BulletText(std::to_string(max_metric_value).c_str());

            ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), "Distribucion de Distorsion (Por Area)");

            // Elegir qué histograma mostrar según la métrica actual seleccionada por el usuario
            if (current_metric == 1) {
                ImGui::Text("Histograma: MIPS (Ideal = 2.0)");
                ImGui::PlotHistogram("##mips", hist_mips.data(), NUM_BINS, 0, NULL, 0.0f, FLT_MAX, ImVec2(0, 80));
            }
            else if (current_metric == 2) {
                ImGui::Text("Histograma: L2 Stretch (Ideal = 1.0)");
                ImGui::PlotHistogram("##l2", hist_l2.data(), NUM_BINS, 0, NULL, 0.0f, FLT_MAX, ImVec2(0, 80));
            }
            else if (current_metric == 3) {
                ImGui::Text("Histograma: Cambio de Area (Ideal = 1.0)");
                ImGui::PlotHistogram("##area", hist_area.data(), NUM_BINS, 0, NULL, 0.0f, FLT_MAX, ImVec2(0, 80));
            }
            else {
                ImGui::TextDisabled("Seleccione una metrica arriba para ver el histograma.");
            }


            ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
            ImGui::Text("Configuracion de Renderizado");

            ImGui::Checkbox("Iluminar ambas caras (Double Sided)", &viewer.data().double_sided);

            float light_pos[3] = {
                viewer.core().light_position(0),
                viewer.core().light_position(1),
                viewer.core().light_position(2)
            };
            if (ImGui::DragFloat3("Posicion Luz", light_pos, 0.1f)) {
                viewer.core().light_position << light_pos[0], light_pos[1], light_pos[2];
            }

            float bg_color[3] = {
                viewer.core().background_color(0),
                viewer.core().background_color(1),
                viewer.core().background_color(2)
            };
            if (ImGui::ColorEdit3("Color de Fondo", bg_color)) {
                viewer.core().background_color << bg_color[0], bg_color[1], bg_color[2], 1.0f;
            }

        };
    update_colors(viewer);
    viewer.launch();
    
    return 0;
}