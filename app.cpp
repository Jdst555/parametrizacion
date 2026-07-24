#include <igl/boundary_loop.h>
#include <igl/harmonic.h>
#include <igl/lscm.h>
#include <igl/arap.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/read_triangle_mesh.h>
#include <igl/writePLY.h>
#include <igl/colormap.h> // Para los mapas de calor

// Cabeceras del visor e ImGui
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>

#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

// Variables globales para el estado de la aplicación
Eigen::MatrixXd V, V_uv;//vértices de la malla 3d y de la parametrización 2d
Eigen::MatrixXi F;//triángulos
Eigen::VectorXi bnd;//vértices del borde
Eigen::MatrixXd Sigmas;

int current_method = 1; // 0: Harmonic, 1: LSCM, 2: ARAP
int current_metric = 1; // 0: Ninguna, 1: MIPS, 2: L2 Stretch, 3: Area
bool show_2d = false;

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
    igl::colormap(igl::COLOR_MAP_TYPE_JET, metric_values, true, C);
    viewer.data().set_colors(C);
}

// -------------------------------------------------------------------
// 3. EJECUCIÓN DE LOS MÉTODOS DE PARAMETRIZACIÓN
// -------------------------------------------------------------------
void compute_parameterization()
{
    if (bnd.size() == 0) {
        std::cerr << "Error: La malla no tiene bordes." << std::endl;
        return;
    }

    if (current_method == 0) { // HARMONIC
        Eigen::MatrixXd bnd_uv;
        igl::map_vertices_to_circle(V, bnd, bnd_uv);
        igl::harmonic(V, F, bnd, bnd_uv, 1, V_uv);
    }
    else if (current_method == 1) { // LSCM
        Eigen::VectorXi b(2, 1);
        b(0) = bnd(0);
        b(1) = bnd(bnd.size() / 2);
        Eigen::MatrixXd bc(2, 2);
        bc << 0, 0, 1, 0;
        igl::lscm(V, F, b, bc, V_uv);
    }
    else if (current_method == 2) { // ARAP
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

    // Tras parametrizar, calculamos los sigmas automáticamente
    compute_sigmas();
}

// -------------------------------------------------------------------
// MAIN
// -------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <ruta_malla.off/obj/ply>" << std::endl;
        return 1;
    }

    // 1. Cargar la malla
    if (!igl::read_triangle_mesh(argv[1], V, F)) {
        std::cerr << "Error al cargar la malla." << std::endl;
        return 1;
    }

    // 2. Extraer bordes
    igl::boundary_loop(F, bnd);

    // Configurar el visor de libigl
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V, F);
    viewer.data().show_lines = false;

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
            ImGui::Text("TFM - Parametrización de Mallas");
            ImGui::Spacing();

            // Selector de método
            ImGui::Combo("Algoritmo", &current_method, "Harmonic (Tutte)\0LSCM\0ARAP\0");

            if (ImGui::Button("Calcular Parametrización", ImVec2(-1, 0))) {
                compute_parameterization();
                // Actualizar vista
                if (show_2d) viewer.data().set_vertices(V_uv);
                viewer.data().set_uv(V_uv);
                update_colors(viewer);
            }

            ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

            // Selector de Métrica
            ImGui::Text("Análisis de Distorsión");
            if (ImGui::Combo("Métrica", &current_metric, "Ninguna (Color sólido)\0MIPS (Conformal)\0L2 Stretch (Distancias)\0Cambio de Area\0")) {
                update_colors(viewer); // Refrescar color al cambiar de métrica
            }

            ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

            // Controles de Vista
            ImGui::Text("Vista");
            if (ImGui::RadioButton("Modelo 3D", !show_2d)) {
                show_2d = false;
                viewer.data().set_vertices(V);
                viewer.data().compute_normals();
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

            // Botón opcional para guardar PLY
            if (ImGui::Button("Guardar Malla Plana (.ply)", ImVec2(-1, 0))) {
                if (V_uv.rows() > 0) {
                    Eigen::MatrixXd V_flat = Eigen::MatrixXd::Zero(V_uv.rows(), 3);
                    V_flat.leftCols(2) = V_uv;
                    igl::writePLY("parametrizacion_exportada.ply", V_flat, F);
                    std::cout << "Malla 2D guardada exitosamente." << std::endl;
                }
                else {
                    std::cerr << "Debe calcular una parametrización primero." << std::endl;
                }
            }

            ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
            ImGui::Text("Configuracion de Renderizado");

            // 1. Doble cara (Ayuda muchísimo si las normales están invertidas)
            ImGui::Checkbox("Iluminar ambas caras (Double Sided)", &viewer.data().double_sided);

            // 2. Factor de Iluminación (Si lo bajas a 0, ves los colores puros sin sombras)
            ImGui::SliderFloat("Factor de Iluminacion", &viewer.core().lighting_factor, 0.0f, 2.0f);

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

    // Calcular una vez por defecto al iniciar
    compute_parameterization();
    viewer.data().set_uv(V_uv);
    update_colors(viewer);

    viewer.launch();
    return 0;
}