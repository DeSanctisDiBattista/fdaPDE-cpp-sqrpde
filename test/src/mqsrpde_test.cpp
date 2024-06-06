// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <fdaPDE/core.h>
#include <gtest/gtest.h>   // testing framework

#include <cstddef>
using fdapde::core::advection;
using fdapde::core::diffusion;
using fdapde::core::FEM;
using fdapde::core::fem_order;

using fdapde::core::laplacian;
using fdapde::core::PDE;
using fdapde::core::Triangulation;

#include "../../fdaPDE/models/regression/mqsrpde.h"
#include "../../fdaPDE/models/sampling_design.h"
using fdapde::models::SpaceOnly;
using fdapde::models::SRPDE;
using fdapde::models::MQSRPDE;

#include "utils/constants.h"
#include "utils/mesh_loader.h"
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_csv;


// // TEST TESI

// // test 7 (run multiple & PP)
// //    domain:       unit square
// //    sampling:     locations != nodes
// //    penalization: constant coefficients PDE
// //    covariates:   no
// //    BC:           no
// //    order FE:     1
// TEST(mqsrpde_test7, laplacian_nonparametric_samplingatlocations) {

//     // path test   
//     std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/models/multiple_quantiles/Tests/Test_7"; 

//     // define domain
//     MeshLoader<Triangulation<2, 2>> domain("unit_square_test7");
//     const std::string lambda_selection = "gcv_smooth_eps1e-1"; 
//     const std::string pde_type = "_lap";    // "_Ktrue" "_lap" "_casc"
//     const bool single_est = true;
//     const bool mult_est = true; 
//     const std::vector<std::string> methods = {"mult", "PP"};    // "mult", "PP", "PP_new"

//     // rhs 
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);

//     // lap 
//     if(pde_type != "_lap")
//         std::cout << "ERROR: YOU WANT TO USE K = I BUT YOU ARE USING SOMETHING ELSE" << std::endl; 
//     auto L = -laplacian<FEM>(); 
//     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);  

//     // // K = K_true
//     // if(pde_type != "_Ktrue")
//     //     std::cout << "ERROR: YOU WANT TO USE K = K_true BUT YOU ARE USING SOMETHING ELSE" << std::endl; 
//     // SMatrix<2> K = read_csv<double>(R_path + "/data/true/K_true.csv"); 
//     // auto L = -diffusion<FEM>(K);   // anisotropic diffusion 
//     // PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//     // define statistical model
//     std::vector<double> alphas = {0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 
//                                   0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99}; 

//     // Read locs
//     DMatrix<double> loc = read_csv<double>(R_path + "/data/locs.csv"); 

//     // Simulations 
//     const unsigned int n_sim = 20; 
//     std::vector<unsigned int> simulations = {25}; //  {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}; 
//     // Single estimations
//     if(single_est){
//         std::cout << "-----------------------SINGLE running---------------" << std::endl;
//         for(auto sim : simulations){

//                 std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl; 

//                 // // K = K_est
//                 // if(pde_type != "_casc")
//                 //     std::cout << "ERROR: YOU WANT TO USE K = K_est BUT YOU ARE USING SOMETHING ELSE" << std::endl; 
//                 // SMatrix<2> K = read_csv<double>(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/K.csv"); 
//                 // auto L = -diffusion<FEM>(K);   // anisotropic diffusion  
//                 // PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//                 // load data from .csv files
//                 DMatrix<double> y = read_csv<double>(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/y.csv");
//                 unsigned int idx = 0; 
//                 std::string solution_path = R_path + "/data/simulations/sim_" + std::to_string(sim) + "/single_est" + pde_type +  "/" + lambda_selection; 

//                 for(double alpha : alphas){
//                     QSRPDE<SpaceOnly> model(problem, Sampling::pointwise, alpha);
//                     model.set_spatial_locations(loc);
//                     unsigned int alpha_int = alphas[idx]*100;  
//                     double lambda; 
//                     std::ifstream fileLambda(solution_path + "/lambdas_opt_alpha_" + std::to_string(alpha_int) + ".csv");
//                     if(fileLambda.is_open()){
//                         fileLambda >> lambda; 
//                         fileLambda.close();
//                     }
//                     model.set_lambda_D(lambda);

//                     // set model data
//                     BlockFrame<double, int> df;
//                     df.insert(OBSERVATIONS_BLK, y);
//                     model.set_data(df);

//                     // solve smoothing problem
//                     model.init();
//                     model.solve();

//                     // Save solution
//                     DMatrix<double> computedF = model.f();
//                     const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filef(solution_path + "/f_" + std::to_string(alpha_int) + ".csv");
//                     if(filef.is_open()){
//                         filef << computedF.format(CSVFormatf);
//                         filef.close();
//                     }

//                     DMatrix<double> computedFn = model.Psi()*model.f();
//                     const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filefn(solution_path + "/fn_" + std::to_string(alpha_int) + ".csv");
//                     if(filefn.is_open()){
//                         filefn << computedFn.format(CSVFormatfn);
//                         filefn.close();
//                     }

//                     idx++;
//                 }

//         }


//     }


//     // Simultaneous estimations
//     if(mult_est){
//         for(std::string method : methods){  
//             bool force; 
//             bool processing; 
//             if(method == "mult"){
//                 processing = false; 
//                 force = false; 
//                 std::cout << "-------------------------MULTIPLE running-----------------" << std::endl;
//             }
//             if(method == "PP"){
//                 processing = true; 
//                 force = false; 
//                 std::cout << "-------------------------PP running-----------------" << std::endl;
//             }
//             if(method == "PP_new"){
//                 processing = true; 
//                 force = true; 
//                 std::cout << "-------------------------PP new running-----------------" << std::endl;
//             }

//             for(auto sim : simulations){

//                 std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl; 

//                 // // K = K_est
//                 // if(pde_type != "_casc")
//                 //     std::cout << "ERROR: YOU WANT TO USE K = K_est BUT YOU ARE USING SOMETHING ELSE" << std::endl; 
//                 // SMatrix<2> K = read_csv<double>(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/K.csv"); 
//                 // auto L = -diffusion<FEM>(K);   // anisotropic diffusion  
//                 // PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//                 MQSRPDE<SpaceOnly> model(problem, Sampling::pointwise, alphas);
//                 model.set_spatial_locations(loc);
//                 model.set_preprocess_option(processing); 
//                 model.set_forcing_option(force);

//                 std::string solution_path = R_path + "/data/simulations/sim_" + std::to_string(sim) + "/mult_est" + pde_type + "/" + lambda_selection;
//                 std::string lambda_path = R_path + "/data/simulations/sim_" + std::to_string(sim) + "/single_est" + pde_type + "/" + lambda_selection; 

//                 // use optimal lambda to avoid possible numerical issues
//                 DMatrix<double> lambdas;
//                 DVector<double> lambdas_temp; 
//                 lambdas_temp.resize(alphas.size());
//                 for(std::size_t idx = 0; idx < alphas.size(); ++idx){
//                     unsigned int alpha_int = alphas[idx]*100;  
//                     std::ifstream fileLambdas(lambda_path + "/lambdas_opt_alpha_" + std::to_string(alpha_int) + ".csv");
//                     if(fileLambdas.is_open()){
//                         fileLambdas >> lambdas_temp(idx); 
//                         fileLambdas.close();
//                     }
//                 }
//                 lambdas = lambdas_temp;                
//                 model.setLambdas_D(lambdas);

//                 // load data from .csv files
//                 DMatrix<double> y = read_csv<double>(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/y.csv");

//                 // set model data
//                 BlockFrame<double, int> df;
//                 df.insert(OBSERVATIONS_BLK, y);
//                 model.set_data(df);

//                 // solve smoothing problem
//                 model.init();
//                 model.solve();

//                 // Save solution
//                 if(method == "mult"){
//                     DMatrix<double> computedF = model.f();
//                     const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filef(solution_path + "/f_all.csv");
//                     if(filef.is_open()){
//                         filef << computedF.format(CSVFormatf);
//                         filef.close();
//                     }

//                     DMatrix<double> computedFn = model.Psi_mult()*model.f();
//                     const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filefn(solution_path + "/fn_all.csv");
//                     if(filefn.is_open()){
//                         filefn << computedFn.format(CSVFormatfn);
//                         filefn.close();
//                     }
//                 } 
//                 if(method == "PP"){
//                     DMatrix<double> computedF = model.f();
//                     const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filef(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/competitors/f_all" + pde_type + "_postproc.csv");
//                     if(filef.is_open()){
//                         filef << computedF.format(CSVFormatf);
//                         filef.close();
//                     }

//                     DMatrix<double> computedFn = model.Psi_mult()*model.f();
//                     const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filefn(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/competitors/fn_all" + pde_type + "_postproc.csv");
//                     if(filefn.is_open()){
//                         filefn << computedFn.format(CSVFormatfn);
//                         filefn.close();
//                     }

//                 }
//                 if(method == "PP_new"){
//                     DMatrix<double> computedF = model.f();
//                     const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filef(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/competitors/f_all" + pde_type + "_postproc_new.csv");
//                     if(filef.is_open()){
//                         filef << computedF.format(CSVFormatf);
//                         filef.close();
//                     }

//                     DMatrix<double> computedFn = model.Psi_mult()*model.f();
//                     const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filefn(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/competitors/fn_all" + pde_type + "_postproc_new.csv");
//                     if(filefn.is_open()){
//                         filefn << computedFn.format(CSVFormatfn);
//                         filefn.close();
//                     }

//                 }


//             }

//         }

//     }


// }




// // test 8 (run multiple & PP)
// //    domain:       unit square
// //    sampling:     locations != nodes
// //    penalization: constant coefficients PDE
// //    covariates:   yes
// //    BC:           no
// //    order FE:     1
// TEST(mqsrpde_test8, laplacian_nonparametric_samplingatlocations) {

//     // path test   
//     std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/models/multiple_quantiles/Tests/Test_8"; 

//     // define domain
//     MeshLoader<Triangulation<2, 2>> domain("unit_square_test8");
//     const std::string lambda_selection = "gcv_smooth_eps1e-1"; 
//     const std::string pde_type = "_casc";    // "_Ktrue" "_lap" "_casc"
//     const bool single_est = false;
//     const bool mult_est = true; 
//     const std::vector<std::string> methods = {"mult"};    // "mult", "PP", "PP_new"

//     // rhs 
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);

//     // // lap 
//     // if(pde_type != "_lap")
//     //     std::cout << "ERROR: YOU WANT TO USE K = I BUT YOU ARE USING SOMETHING ELSE" << std::endl; 
//     // auto L = -laplacian<FEM>(); 
//     // PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);  

//     // // K = K_true
//     // if(pde_type != "_Ktrue")
//     //     std::cout << "ERROR: YOU WANT TO USE K = K_true BUT YOU ARE USING SOMETHING ELSE" << std::endl; 
//     // SMatrix<2> K = read_csv<double>(R_path + "/data/true/K_true.csv"); 
//     // auto L = -diffusion<FEM>(K);   // anisotropic diffusion 
//     // PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//     // define statistical model
//     // std::vector<double> alphas = {0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 
//     //                               0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99}; 
//     std::vector<double> alphas = {0.25, 0.50}; 

//     // Read locs and X 
//     DMatrix<double> loc = read_csv<double>(R_path + "/data/locs.csv");
//     DMatrix<double> X = read_csv<double>(R_path + "/data/X.csv");  

//     // Simulations 
//     const unsigned int n_sim = 10; 
//     std::vector<unsigned int> simulations = {1,2,3,4,5,6,7,8,9,10}; 
//     // Single estimations
//     if(single_est){
//         std::cout << "-----------------------SINGLE running---------------" << std::endl;
//         for(auto sim : simulations){

//                 std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl; 

//                 // K = K_est
//                 if(pde_type != "_casc")
//                     std::cout << "ERROR: YOU WANT TO USE K = K_est BUT YOU ARE USING SOMETHING ELSE" << std::endl; 
//                 SMatrix<2> K = read_csv<double>(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/K.csv"); 
//                 auto L = -diffusion<FEM>(K);   // anisotropic diffusion  
//                 PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//                 // load data from .csv files
//                 DMatrix<double> y = read_csv<double>(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/y.csv");
//                 unsigned int idx = 0; 
//                 std::string solution_path = R_path + "/data/simulations/sim_" + std::to_string(sim) + "/single_est" + pde_type +  "/" + lambda_selection; 

//                 for(double alpha : alphas){
//                     QSRPDE<SpaceOnly> model(problem, Sampling::pointwise, alpha);
//                     model.set_spatial_locations(loc);
//                     unsigned int alpha_int = alphas[idx]*100;  
//                     double lambda; 
//                     std::ifstream fileLambda(solution_path + "/lambdas_opt_alpha_" + std::to_string(alpha_int) + ".csv");
//                     if(fileLambda.is_open()){
//                         fileLambda >> lambda; 
//                         fileLambda.close();
//                     }
//                     model.set_lambda_D(lambda);

//                     // set model data
//                     BlockFrame<double, int> df;
//                     df.insert(OBSERVATIONS_BLK, y);
//                     df.insert(DESIGN_MATRIX_BLK, X);
//                     model.set_data(df);

//                     // solve smoothing problem
//                     model.init();
//                     model.solve();

//                     // // Save solution
//                     // DMatrix<double> computedF = model.f();
//                     // const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     // std::ofstream filef(solution_path + "/f_" + std::to_string(alpha_int) + ".csv");
//                     // if(filef.is_open()){
//                     //     filef << computedF.format(CSVFormatf);
//                     //     filef.close();
//                     // }

//                     // DMatrix<double> computedFn = model.Psi()*model.f();
//                     // const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     // std::ofstream filefn(solution_path + "/fn_" + std::to_string(alpha_int) + ".csv");
//                     // if(filefn.is_open()){
//                     //     filefn << computedFn.format(CSVFormatfn);
//                     //     filefn.close();
//                     // }

//                     // DMatrix<double> computedBeta = model.beta(); 
//                     // const static Eigen::IOFormat CSVFormatBeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     // std::ofstream filebeta(solution_path + "/beta_" + std::to_string(alpha_int) + ".csv");
//                     // if(filebeta.is_open()){
//                     //     filebeta << computedBeta.format(CSVFormatBeta);
//                     //     filebeta.close();
//                     // }

//                     idx++;
//                 }

//         }


//     }


//     // Simultaneous estimations
//     if(mult_est){
//         for(std::string method : methods){  
//             bool force; 
//             bool processing; 
//             if(method == "mult"){
//                 processing = false; 
//                 force = false; 
//                 std::cout << "-------------------------MULTIPLE running-----------------" << std::endl;
//             }
//             if(method == "PP"){
//                 processing = true; 
//                 force = false; 
//                 std::cout << "-------------------------PP running-----------------" << std::endl;
//             }
//             if(method == "PP_new"){
//                 processing = true; 
//                 force = true; 
//                 std::cout << "-------------------------PP new running-----------------" << std::endl;
//             }

//             for(auto sim : simulations){

//                 std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl; 

//                 // K = K_est
//                 if(pde_type != "_casc")
//                     std::cout << "ERROR: YOU WANT TO USE K = K_est BUT YOU ARE USING SOMETHING ELSE" << std::endl; 
//                 SMatrix<2> K = read_csv<double>(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/K.csv"); 
//                 auto L = -diffusion<FEM>(K);   // anisotropic diffusion  
//                 PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//                 MQSRPDE<SpaceOnly> model(problem, Sampling::pointwise, alphas);
//                 model.set_spatial_locations(loc);
//                 model.set_preprocess_option(processing); 
//                 model.set_forcing_option(force);

//                 std::string solution_path = R_path + "/data/simulations/sim_" + std::to_string(sim) + "/mult_est" + pde_type + "/" + lambda_selection;
//                 std::string lambda_path = R_path + "/data/simulations/sim_" + std::to_string(sim) + "/single_est" + pde_type + "/" + lambda_selection; 

//                 // use optimal lambda to avoid possible numerical issues
//                 DMatrix<double> lambdas;
//                 DVector<double> lambdas_temp; 
//                 lambdas_temp.resize(alphas.size());
//                 for(std::size_t idx = 0; idx < alphas.size(); ++idx){
//                     unsigned int alpha_int = alphas[idx]*100;  
//                     std::ifstream fileLambdas(lambda_path + "/lambdas_opt_alpha_" + std::to_string(alpha_int) + ".csv");
//                     if(fileLambdas.is_open()){
//                         fileLambdas >> lambdas_temp(idx); 
//                         fileLambdas.close();
//                     }
//                 }
//                 lambdas = lambdas_temp;                
//                 model.setLambdas_D(lambdas);

//                 // load data from .csv files
//                 DMatrix<double> y = read_csv<double>(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/y.csv");

//                 // set model data
//                 BlockFrame<double, int> df;
//                 df.insert(OBSERVATIONS_BLK, y);
//                 df.insert(DESIGN_MATRIX_BLK, X);
//                 model.set_data(df);

//                 // solve smoothing problem
//                 model.init();
//                 model.solve();

//             //     // Save solution
//             //     if(method == "mult"){
//             //         DMatrix<double> computedF = model.f();
//             //         const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             //         std::ofstream filef(solution_path + "/f_all.csv");
//             //         if(filef.is_open()){
//             //             filef << computedF.format(CSVFormatf);
//             //             filef.close();
//             //         }

//             //         DMatrix<double> computedFn = model.Psi_mult()*model.f();
//             //         const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             //         std::ofstream filefn(solution_path + "/fn_all.csv");
//             //         if(filefn.is_open()){
//             //             filefn << computedFn.format(CSVFormatfn);
//             //             filefn.close();
//             //         }

//             //         DMatrix<double> computedBeta = model.beta(); 
//             //         const static Eigen::IOFormat CSVFormatBeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             //         std::ofstream filebeta(solution_path + "/beta_all.csv");
//             //         if(filebeta.is_open()){
//             //             filebeta << computedBeta.format(CSVFormatBeta);
//             //             filebeta.close();
//             //         }
//             //     } 
//             //     if(method == "PP"){
//             //         DMatrix<double> computedF = model.f();
//             //         const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             //         std::ofstream filef(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/competitors/f_all" + pde_type + "_postproc.csv");
//             //         if(filef.is_open()){
//             //             filef << computedF.format(CSVFormatf);
//             //             filef.close();
//             //         }

//             //         DMatrix<double> computedFn = model.Psi_mult()*model.f();
//             //         const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             //         std::ofstream filefn(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/competitors/fn_all" + pde_type + "_postproc.csv");
//             //         if(filefn.is_open()){
//             //             filefn << computedFn.format(CSVFormatfn);
//             //             filefn.close();
//             //         }

//             //         DMatrix<double> computedBeta = model.beta(); 
//             //         const static Eigen::IOFormat CSVFormatBeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             //         std::ofstream filebeta(solution_path + "/beta_all.csv");
//             //         if(filebeta.is_open()){
//             //             filebeta << computedBeta.format(CSVFormatBeta);
//             //             filebeta.close();
//             //         }

//             //     }
//             //     if(method == "PP_new"){
//             //         DMatrix<double> computedF = model.f();
//             //         const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             //         std::ofstream filef(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/competitors/f_all" + pde_type + "_postproc_new.csv");
//             //         if(filef.is_open()){
//             //             filef << computedF.format(CSVFormatf);
//             //             filef.close();
//             //         }

//             //         DMatrix<double> computedFn = model.Psi_mult()*model.f();
//             //         const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             //         std::ofstream filefn(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/competitors/fn_all" + pde_type + "_postproc_new.csv");
//             //         if(filefn.is_open()){
//             //             filefn << computedFn.format(CSVFormatfn);
//             //             filefn.close();
//             //         }

//             //         DMatrix<double> computedBeta = model.beta(); 
//             //         const static Eigen::IOFormat CSVFormatBeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             //         std::ofstream filebeta(solution_path + "/beta_all.csv");
//             //         if(filebeta.is_open()){
//             //             filebeta << computedBeta.format(CSVFormatBeta);
//             //             filebeta.close();
//             //         }

//             //     }


//             }

//         }

//     }


// }



// // //// 

// // // TEST OBS RIPETUTE


// // test 7 OBS RIPETUTE
// //    domain:       unit square
// //    sampling:     locations != nodes
// //    penalization: constant coefficients PDE
// //    covariates:   no
// //    BC:           no
// //    order FE:     1
// TEST(mqsrpde_test7_obs_rip, laplacian_nonparametric_samplingatlocations) {

//     // path test   
//     std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/models/multiple_quantiles/Tests/Test_7_obs_ripetute"; 
//     // std::string R_path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/Thesis_shared/models/multiple_quantiles/Tests/Test_7_obs_ripetute"; 

//     // define domain
//     MeshLoader<Triangulation<2, 2>> domain("unit_square_test7");
//     const std::string lambda_selection = "gcv_smooth_eps1e-1"; 
//     const std::string pde_type = "_lap";    // "_Ktrue" "_lap" "_casc"
//     const bool single_est = false;
//     const bool mult_est = true; 

//     const std::string max_iter_string = "";     // "_maxiter400" per il caso in cui max_iter di fpirls è stato alzato a 400 (si parla di stima singola)
//                                                     // "" per il caso classico con max_iter=200

//     // usare le obs ripetute?
//     bool bool_obs_rip = true;

//     const std::vector<std::string> methods = {"mult"};    // "mult", "PP", "PP_new"

//     // rhs 
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);

//     // lap 
//     if(pde_type != "_lap")
//         std::cout << "ERROR: YOU WANT TO USE K = I BUT YOU ARE USING SOMETHING ELSE" << std::endl; 
//     auto L = -laplacian<FEM>();  
//     PDE<Triangulation<2, 2>, decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);  

//     // // K = K_true
//     // if(pde_type != "_Ktrue")
//     //     std::cout << "ERROR: YOU WANT TO USE K = K_true BUT YOU ARE USING SOMETHING ELSE" << std::endl; 
//     // SMatrix<2> K = read_csv<double>(R_path + "/data/true/K_true.csv"); 
//     // auto L = -diffusion<FEM>(K);   // anisotropic diffusion 
//     // PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//     // define statistical model
//     std::vector<double> alphas = {0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 
//                                   0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99};

//     // std::vector<double> alphas = {0.01, 0.02, 0.05, 0.10, 0.25, 0.75, 
//     //                              0.91, 0.92, 0.93, 0.94, 0.96, 0.97, 0.98};


//     // std::vector<double> alphas = {0.50, 0.90, 0.95, 0.99}; 

//     // Read locs
//     DMatrix<double> loc ; 
//     if(bool_obs_rip)
//         loc = read_csv<double>(R_path + "/data" + "/locs_ripetute.csv"); 
//     else
//         loc = read_csv<double>(R_path + "/data" + "/locs.csv"); 

//     // Simulations 
//     const unsigned int n_sim = 1; 
//     // Single estimations
//     if(single_est){
//         std::cout << "-----------------------SINGLE running---------------" << std::endl;
//         for(auto sim = 1; sim <= n_sim; ++sim){

//                 std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl; 

//                 std::string solution_path = R_path + "/data/simulations/sim_" + std::to_string(sim) + "/single_est" + pde_type + "/" + lambda_selection; 

//                 if(bool_obs_rip)
//                     solution_path = solution_path + "/obs_ripetute"; 
//                 else
//                     solution_path = solution_path + "/obs_singola"; 

//                 // // K = K_est
//                 // if(pde_type != "_casc")
//                 //     std::cout << "ERROR: YOU WANT TO USE K = K_est BUT YOU ARE USING SOMETHING ELSE" << std::endl; 
//                 // SMatrix<2> K = read_csv<double>(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/K.csv"); 
//                 // auto L = -diffusion<FEM>(K);   // anisotropic diffusion  
//                 // PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//                 // load data from .csv files
//                 DMatrix<double> y; 
//                 if(bool_obs_rip)
//                     y = read_csv<double>(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/y_ripetute.csv");
//                 else
//                     y = read_csv<double>(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/y.csv");

//                 // check dimensions
//                 std::cout << "dim loc " << loc.rows() << " , " << loc.cols() << std::endl;
//                 std::cout << "dim y " << y.rows() << std::endl;


//                 for(double alpha : alphas){
//                     QSRPDE<SpaceOnly> model(problem, Sampling::pointwise, alpha);
//                     model.set_spatial_locations(loc);
//                     unsigned int alpha_int = alpha*100;  
//                     double lambda; 
//                     std::ifstream fileLambda(solution_path + "/lambdas_opt_alpha_" + std::to_string(alpha_int) + ".csv");
//                     if(fileLambda.is_open()){
//                         fileLambda >> lambda; 
//                         fileLambda.close();
//                     }
//                     model.set_lambda_D(lambda);

//                     // set model data
//                     BlockFrame<double, int> df;
//                     df.insert(OBSERVATIONS_BLK, y);
//                     model.set_data(df);

//                     // solve smoothing problem
//                     model.init();
//                     model.solve();

//                     // Save solution
//                     DMatrix<double> computedF = model.f();
//                     const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filef(solution_path + "/f_" + std::to_string(alpha_int) + max_iter_string + ".csv");
//                     if(filef.is_open()){
//                         filef << computedF.format(CSVFormatf);
//                         filef.close();
//                     }

//                     DMatrix<double> computedFn = model.Psi()*model.f();
//                     const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filefn(solution_path + "/fn_" + std::to_string(alpha_int) + max_iter_string + ".csv");
//                     if(filefn.is_open()){
//                         filefn << computedFn.format(CSVFormatfn);
//                         filefn.close();
//                     }

//                     // Save number of iterations
//                     unsigned int computedNiter = model.n_iter_qsrpde();
//                     std::ofstream ofile_niter;
//                     ofile_niter.open(solution_path + "/niter_" + std::to_string(alpha_int) + max_iter_string + ".txt");
//                     ofile_niter << computedNiter; 
//                     ofile_niter.close(); 
 
//                 }

//         }


//     }


//     // Simultaneous estimations
//     if(mult_est){
//         for(std::string method : methods){  
//             bool force; 
//             bool processing; 
//             if(method == "mult"){
//                 processing = false; 
//                 force = false; 
//                 std::cout << "-------------------------MULTIPLE running-----------------" << std::endl;
//             }
//             if(method == "PP"){
//                 processing = true; 
//                 force = false; 
//                 std::cout << "-------------------------PP running-----------------" << std::endl;
//             }
//             if(method == "PP_new"){
//                 processing = true; 
//                 force = true; 
//                 std::cout << "-------------------------PP new running-----------------" << std::endl;
//             }

//             for(auto sim = 1; sim <= n_sim; ++sim){

//                 std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl; 

//                 // // K = K_est
//                 // if(pde_type != "_casc")
//                 //     std::cout << "ERROR: YOU WANT TO USE K = K_est BUT YOU ARE USING SOMETHING ELSE" << std::endl; 
//                 // SMatrix<2> K = read_csv<double>(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/K.csv"); 
//                 // auto L = -diffusion<FEM>(K);   // anisotropic diffusion  
//                 // PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//                 MQSRPDE<SpaceOnly> model(problem, Sampling::pointwise, alphas);
//                 model.set_spatial_locations(loc);
//                 model.set_preprocess_option(processing); 
//                 model.set_forcing_option(force);

//                 std::string solution_path = R_path + "/data/simulations/sim_" + std::to_string(sim) + "/mult_est" + pde_type + "/" + lambda_selection;
//                 std::string lambda_path = R_path + "/data/simulations/sim_" + std::to_string(sim) + "/single_est" + pde_type + "/" + lambda_selection; 

//                 if(bool_obs_rip){
//                     solution_path = solution_path + "/obs_ripetute";
//                     lambda_path = lambda_path + "/obs_ripetute"; 
//                 }else{
//                     solution_path = solution_path + "/obs_singola"; 
//                     lambda_path = lambda_path + "/obs_singola";
//                 }



//                 // use optimal lambda to avoid possible numerical issues
//                 DMatrix<double> lambdas;
//                 DVector<double> lambdas_temp; 
//                 lambdas_temp.resize(alphas.size());
//                 for(std::size_t idx = 0; idx < alphas.size(); ++idx){
//                     unsigned int alpha_int = alphas[idx]*100;  
//                     std::ifstream fileLambdas(lambda_path + "/lambdas_opt_alpha_" + std::to_string(alpha_int) + ".csv");
//                     if(fileLambdas.is_open()){
//                         fileLambdas >> lambdas_temp(idx); 
//                         fileLambdas.close();
//                     }
//                 }
//                 lambdas = lambdas_temp;                
//                 model.setLambdas_D(lambdas);


//                 // load data from .csv files
//                 DMatrix<double> y; 
//                 if(bool_obs_rip)
//                     y = read_csv<double>(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/y_ripetute.csv");
//                 else
//                     y = read_csv<double>(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/y.csv");

//                 // check dimensions
//                 std::cout << "dim loc " << loc.rows() << " , " << loc.cols() << std::endl;
//                 std::cout << "dim y " << y.rows() << std::endl;

//                 // set model data
//                 BlockFrame<double, int> df;
//                 df.insert(OBSERVATIONS_BLK, y);
//                 model.set_data(df);

//                 // solve smoothing problem
//                 model.init();
//                 model.solve();

//                 // Save solution
//                 if(method == "mult"){
//                     DMatrix<double> computedF = model.f();
//                     const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filef(solution_path + "/f_all_prova.csv");
//                     if(filef.is_open()){
//                         filef << computedF.format(CSVFormatf);
//                         filef.close();
//                     }

//                     DMatrix<double> computedFn = model.Psi_mult()*model.f();
//                     const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filefn(solution_path + "/fn_all_prova.csv");
//                     if(filefn.is_open()){
//                         filefn << computedFn.format(CSVFormatfn);
//                         filefn.close();
//                     }
//                 } 
//                 if(method == "PP"){
//                     DMatrix<double> computedF = model.f();
//                     const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filef(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/competitors/f_all" + pde_type + "_postproc.csv");
//                     if(filef.is_open()){
//                         filef << computedF.format(CSVFormatf);
//                         filef.close();
//                     }

//                     DMatrix<double> computedFn = model.Psi_mult()*model.f();
//                     const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filefn(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/competitors/fn_all" + pde_type + "_postproc.csv");
//                     if(filefn.is_open()){
//                         filefn << computedFn.format(CSVFormatfn);
//                         filefn.close();
//                     }

//                 }
//                 if(method == "PP_new"){
//                     DMatrix<double> computedF = model.f();
//                     const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filef(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/competitors/f_all" + pde_type + "_postproc_new.csv");
//                     if(filef.is_open()){
//                         filef << computedF.format(CSVFormatf);
//                         filef.close();
//                     }

//                     DMatrix<double> computedFn = model.Psi_mult()*model.f();
//                     const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filefn(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/competitors/fn_all" + pde_type + "_postproc_new.csv");
//                     if(filefn.is_open()){
//                         filefn << computedFn.format(CSVFormatfn);
//                         filefn.close();
//                     }

//                 }


//             }

//         }

//     }


// }





// test OBS RIPETUTE
//    domain:       unit square
//    sampling:     locations != nodes
//    penalization: constant coefficients PDE
//    covariates:   no
//    BC:           no
//    order FE:     1
TEST(msqrpde_test_obs_rip, pde_nonparametric_samplingatlocations_spaceonly_gridexact) {

    bool mean_estimation = false;
    bool quantile_estimation = !mean_estimation;  

    bool corr = true; 
    // path test  
    std::string R_path; 
    if(!corr){
        if(mean_estimation){
            R_path = "";
        } else{
            R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/models/multiple_quantiles/Tests/Test_obs_ripetute";
        }
         
    } else{
        if(mean_estimation){
            R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/models/srpde/Tests/Test_obs_ripetute_cor";
        } else{
            R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/models/multiple_quantiles/Tests/Test_obs_ripetute_cor";
        }
    }

    std::vector<unsigned int> max_reps = {10, 20, 30, 40, 50};   // max number of repetitions 
    std::vector<std::string> data_types = {"data"}; 
    for(auto max_rep : max_reps){
        data_types.push_back("data_rip_" + std::to_string(max_rep));
    }

    std::string diffusion_type = "_lap"; 

    std::vector<double> alphas = {0.50};
    bool single = true; 
    bool multiple = false;  
    const std::string gcv_refinement = "fine";    // "lasco" "fine"

    // model selection parameters
    std::string smooth_type_mean = "GCV";    
    std::vector<std::string> smooth_types_quantile = {"GCV_eps1e-1"};   
    
    bool compute_rmse = false;
    bool compute_gcv = true;    

    const unsigned int n_sim = 10;

    // define domain
    std::string domain_str; 
    MeshLoader<Triangulation<2, 2>> domain("unit_square_25");

    // define regularizing PDE
    auto L = -laplacian<FEM>();   
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<Triangulation<2, 2>, decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

    for(auto sim = 1; sim <= n_sim; ++sim){
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl; 
        
        for(std::string data_type : data_types){
            // Read locations and data 
            std::string loc_str; 
            if(data_type.length() >= 2 && data_type.substr(data_type.length() - 2) != "ta"){
                loc_str = "/locs_rip.csv";
            } else{
                loc_str = "/locs.csv"; 
            }
            std::string data_path = R_path + "/" + data_type; 
            DMatrix<double> loc = read_csv<double>(data_path + loc_str); 
            DMatrix<double> y = read_csv<double>(data_path + "/simulations/sim_" + std::to_string(sim) + "/y.csv");
            BlockFrame<double, int> df;
            df.insert(OBSERVATIONS_BLK, y);

            if(mean_estimation){

                std::cout << "------------------MEAN REGRESSION-----------------" << std::endl;

                std::string gcv_path = data_path + "/simulations/sim_" + std::to_string(sim) + "/mean/" + smooth_type_mean + "/est" + diffusion_type; 
                std::string rmse_path = data_path + "/simulations/sim_" + std::to_string(sim) + "/mean/RMSE/est" + diffusion_type; 
                if(compute_gcv){
                    std::cout << "------------------gcv selection-----------------" << std::endl;
                    // Read lambda 
                    double best_lambda;
                    std::ifstream fileLambda(gcv_path + "/lambda_s_opt.csv");
                    if(fileLambda.is_open()){
                        fileLambda >> best_lambda; 
                        fileLambda.close();
                    }
                    std::cout << "best lambda=" << best_lambda << std::endl; 

                    SRPDE model_mean(problem, Sampling::pointwise);
                    model_mean.set_lambda_D(best_lambda);
                    // set model's data
                    model_mean.set_spatial_locations(loc);  
                    model_mean.set_data(df);           
                    // solve smoothing problem
                    model_mean.init();
                    model_mean.solve();

                    // save
                    DMatrix<double> computedF = model_mean.f();
                    const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
                    std::ofstream filef(gcv_path + "/f.csv");
                    if(filef.is_open()){
                        filef << computedF.format(CSVFormatf);
                        filef.close();
                    }

                    DMatrix<double> computedFn = model_mean.Psi()*model_mean.f();
                    const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
                    std::ofstream filefn(gcv_path + "/fn.csv");
                    if(filefn.is_open()){
                        filefn << computedFn.format(CSVFormatfn);
                        filefn.close();
                    }

                } 
                if(compute_rmse){
                    std::cout << "------------------rmse selection-----------------" << std::endl;
                    // Read lambda 
                    double best_lambda;
                    std::ifstream fileLambda(rmse_path + "/lambda_s_opt.csv");
                    if(fileLambda.is_open()){
                        fileLambda >> best_lambda; 
                        fileLambda.close();
                    }
                    std::cout << "best lambda=" << best_lambda << std::endl; 

                    SRPDE model_mean(problem, Sampling::pointwise);
                    model_mean.set_lambda_D(best_lambda);
                    // set model's data
                    model_mean.set_spatial_locations(loc);
                    model_mean.set_data(df);
                    // solve smoothing problem
                    model_mean.init();
                    model_mean.solve();

                    // save
                    DMatrix<double> computedF = model_mean.f();
                    const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
                    std::ofstream filef(rmse_path + "/f.csv");
                    if(filef.is_open()){
                        filef << computedF.format(CSVFormatf);
                        filef.close();
                    }

                    DMatrix<double> computedFn = model_mean.Psi()*model_mean.f();
                    const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
                    std::ofstream filefn(rmse_path + "/fn.csv");
                    if(filefn.is_open()){
                        filefn << computedFn.format(CSVFormatfn);
                        filefn.close();
                    }

                }

            }
    
            if(quantile_estimation){

                std::cout << "------------------QUANTILE REGRESSION-----------------" << std::endl; 
              
                if(compute_gcv){
                    std::cout << "------------------gcv computation-----------------" << std::endl;

                    for(auto smooth_type : smooth_types_quantile){
                        const int eps_power = std::stoi(smooth_type.substr(smooth_type.size() - 2));
                        if(single){
                            for(auto alpha : alphas){

                                unsigned int alpha_int = alpha*100; 
                                std::string alpha_string = std::to_string(alpha_int); 

                                std::cout << "------------------alpha=" << alpha_string << "-----------------" << std::endl; 
                                std::string gcv_path = data_path + "/simulations/sim_" + std::to_string(sim) + "/quantile/" + smooth_type + "/single_est" + diffusion_type + "/alpha_" + alpha_string;
                                // Read lambda 
                                double best_lambda;
                                std::ifstream fileLambda(gcv_path + "/lambda_s_opt_" + gcv_refinement + ".csv");
                                if(fileLambda.is_open()){
                                    fileLambda >> best_lambda; 
                                    fileLambda.close();
                                }
                                std::cout << "best lambda=" << best_lambda << std::endl; 

                                QSRPDE<SpaceOnly> model_quantile(problem, Sampling::pointwise, alpha);
                                model_quantile.set_lambda_D(best_lambda);
                                // set model's data
                                model_quantile.set_spatial_locations(loc);  
                                model_quantile.set_data(df);           
                                // solve smoothing problem
                                model_quantile.init();
                                model_quantile.solve();

                                // save
                                DMatrix<double> computedF = model_quantile.f();
                                const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
                                std::ofstream filef(gcv_path + "/f.csv");
                                if(filef.is_open()){
                                    filef << computedF.format(CSVFormatf);
                                    filef.close();
                                }

                                DMatrix<double> computedFn = model_quantile.Psi()*model_quantile.f();
                                const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
                                std::ofstream filefn(gcv_path + "/fn.csv");
                                if(filefn.is_open()){
                                    filefn << computedFn.format(CSVFormatfn);
                                    filefn.close();
                                }
                            }
                        }
                        if(multiple){

                            MQSRPDE<SpaceOnly> model_quantile(problem, Sampling::pointwise, alphas);
                            std::string solution_mult_path = data_path + "/simulations/sim_" + std::to_string(sim) + "/quantile/" + smooth_type + "/mult_est" + diffusion_type;
                            std::string lambda_path = data_path + "/simulations/sim_" + std::to_string(sim) + "/quantile/" + smooth_type + "/single_est" + diffusion_type;

                            // use optimal lambda to avoid possible numerical issues
                            DMatrix<double> lambdas;
                            DVector<double> lambdas_temp; 
                            lambdas_temp.resize(alphas.size());
                            for(std::size_t idx = 0; idx < alphas.size(); ++idx){
                                unsigned int alpha_int = alphas[idx]*100;  
                                std::ifstream fileLambdas(lambda_path  + "/alpha_" + std::to_string(alpha_int) + "/lambda_s_opt_" + gcv_refinement + ".csv");
                                if(fileLambdas.is_open()){
                                    fileLambdas >> lambdas_temp(idx); 
                                    fileLambdas.close();
                                }
                            }
                            lambdas = lambdas_temp;                
                            model_quantile.setLambdas_D(lambdas);

                            // set model data
                            model_quantile.set_data(df);
                            model_quantile.set_spatial_locations(loc); 

                            // solve smoothing problem
                            model_quantile.init();
                            model_quantile.solve();

                            // Save solution
                            DMatrix<double> computedF = model_quantile.f();
                            const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
                            std::ofstream filef(solution_mult_path + "/f_all.csv");
                            if(filef.is_open()){
                                filef << computedF.format(CSVFormatf);
                                filef.close();
                            }

                            DMatrix<double> computedFn = model_quantile.Psi_mult()*model_quantile.f();
                            const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
                            std::ofstream filefn(solution_mult_path + "/fn_all.csv");
                            if(filefn.is_open()){
                                filefn << computedFn.format(CSVFormatfn);
                                filefn.close();
                            }

                        }

                    }
                } 
                if(compute_rmse){
                    std::cout << "------------------rmse computation-----------------" << std::endl;
                    
                    if(single){

                        for(auto alpha : alphas){

                            unsigned int alpha_int = alpha*100; 
                            std::string alpha_string = std::to_string(alpha_int); 

                            std::cout << "------------------alpha=" << alpha_string << "-----------------" << std::endl; 
                            std::string rmse_path = data_path + "/simulations/sim_" + std::to_string(sim) + "/quantile/RMSE/single_est" + diffusion_type + "/alpha_" + alpha_string;
                            // Read lambda 
                            double best_lambda;
                            std::ifstream fileLambda(rmse_path + "/lambda_s_opt_" + gcv_refinement + ".csv");
                            if(fileLambda.is_open()){
                                fileLambda >> best_lambda; 
                                fileLambda.close();
                            }
                            std::cout << "best lambda=" << best_lambda << std::endl; 

                            QSRPDE<SpaceOnly> model_quantile(problem, Sampling::pointwise, alpha);
                            model_quantile.set_lambda_D(best_lambda);
                            // set model's data
                            model_quantile.set_spatial_locations(loc);  
                            model_quantile.set_data(df);           
                            // solve smoothing problem
                            model_quantile.init();
                            model_quantile.solve();

                            // save
                            DMatrix<double> computedF = model_quantile.f();
                            const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
                            std::ofstream filef(rmse_path + "/f.csv");
                            if(filef.is_open()){
                                filef << computedF.format(CSVFormatf);
                                filef.close();
                            }

                            DMatrix<double> computedFn = model_quantile.Psi()*model_quantile.f();
                            const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
                            std::ofstream filefn(rmse_path + "/fn.csv");
                            if(filefn.is_open()){
                                filefn << computedFn.format(CSVFormatfn);
                                filefn.close();
                            }

                        }
                    }
                    if(multiple){

                        MQSRPDE<SpaceOnly> model_quantile(problem, Sampling::pointwise, alphas);
                        std::string solution_mult_path = data_path + "/simulations/sim_" + std::to_string(sim) + "/quantile/RMSE/mult_est" + diffusion_type;
                        std::string lambda_path = data_path + "/simulations/sim_" + std::to_string(sim) + "/quantile/RMSE/single_est" + diffusion_type;

                        // use optimal lambda to avoid possible numerical issues
                        DMatrix<double> lambdas;
                        DVector<double> lambdas_temp; 
                        lambdas_temp.resize(alphas.size());
                        for(std::size_t idx = 0; idx < alphas.size(); ++idx){
                            unsigned int alpha_int = alphas[idx]*100;  
                            std::ifstream fileLambdas(lambda_path + "/alpha_" + std::to_string(alpha_int) +  "/lambda_s_opt_" + gcv_refinement + ".csv");
                            if(fileLambdas.is_open()){
                                fileLambdas >> lambdas_temp(idx); 
                                fileLambdas.close();
                            }
                        }
                        lambdas = lambdas_temp;                
                        model_quantile.setLambdas_D(lambdas);

                        // set model data
                        model_quantile.set_data(df);
                        model_quantile.set_spatial_locations(loc); 

                        // solve smoothing problem
                        model_quantile.init();
                        model_quantile.solve();

                        // Save solution
                        DMatrix<double> computedF = model_quantile.f();
                        const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
                        std::ofstream filef(solution_mult_path + "/f_all.csv");
                        if(filef.is_open()){
                            filef << computedF.format(CSVFormatf);
                            filef.close();
                        }

                        DMatrix<double> computedFn = model_quantile.Psi_mult()*model_quantile.f();
                        const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
                        std::ofstream filefn(solution_mult_path + "/fn_all.csv");
                        if(filefn.is_open()){
                            filefn << computedFn.format(CSVFormatfn);
                            filefn.close();
                        }

                    }
                    



                }                    

            }
                            
        }
 
                        

    }

}












// // //// 

// // // NUOVI TEST POST-TESI


// // test 1 (run multiple & PP)
// //    domain:       unit square
// //    sampling:     locations != nodes
// //    penalization: constant coefficients PDE
// //    covariates:   no
// //    BC:           no
// //    order FE:     1
// TEST(mqsrpde_test1, laplacian_nonparametric_samplingatlocations) {

//     // path test   
//     std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/PhD/Codice/models/MQSRPDE/Tests/Test_1"; 

//     const unsigned int n_sim = 20; 
//     const std::string gcv_refinement = "fine"; 
//     double lambdas_step; 
//     if(gcv_refinement == "lasco"){
//         lambdas_step = 0.5; 
//     } 
//     if(gcv_refinement == "fine"){
//         lambdas_step = 0.1; 
//     } 

//     // define domain
//     MeshLoader<Triangulation<2, 2>> domain("unit_square_test7");
//     const std::string lambda_selection = "gcv_smooth_eps1e-1"; 
//     //const std::string pde_type = "_lap";    // "_Ktrue" "_lap" "_casc"
//     const bool single_est = true;
//     const bool mult_est = true; 
//     const std::vector<std::string> methods = {"mult"};    // "mult", "PP", "PP_new"

//     // rhs 
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);

//     // // lap 
//     // if(pde_type != "_lap")
//     //     std::cout << "ERROR: YOU WANT TO USE K = I BUT YOU ARE USING SOMETHING ELSE" << std::endl; 
//     // auto L = -laplacian<FEM>(); 
//     // PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);  

//     // // K = K_true
//     // if(pde_type != "_Ktrue")
//     //     std::cout << "ERROR: YOU WANT TO USE K = K_true BUT YOU ARE USING SOMETHING ELSE" << std::endl; 
//     // SMatrix<2> K = read_csv<double>(R_path + "/data/true/K_true.csv"); 
//     // auto L = -diffusion<FEM>(K);   // anisotropic diffusion 
//     // PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//     // define statistical model
//     std::vector<double> alphas = {0.01, 0.02, 0.03,   // ATT: aggiunto 3%
//                                   0.05, 0.10, 0.25, 0.50, 0.75, 
//                                   0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99}; 

//     // Read locs
//     DMatrix<double> loc = read_csv<double>(R_path + "/locs.csv"); 

//     // Simulations 
//     // std::vector<unsigned int> simulations = {25}; //  {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}; 
    
    
//     // Single estimations
//     if(single_est){
//         std::cout << "-----------------------SINGLE running---------------" << std::endl;
//         for(auto sim = 1; sim <= n_sim; ++sim){

//                 std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl; 

//                 // K = K_est
//                 // if(pde_type != "_casc")
//                 //     std::cout << "ERROR: YOU WANT TO USE K = K_est BUT YOU ARE USING SOMETHING ELSE" << std::endl; 
//                 SMatrix<2> K = read_csv<double>(R_path + "/simulations/sim_" + std::to_string(sim) + "/K.csv"); 
//                 auto L = -diffusion<FEM>(K);   // anisotropic diffusion  
//                 PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//                 // load data from .csv files
//                 DMatrix<double> y = read_csv<double>(R_path + "/simulations/sim_" + std::to_string(sim) + "/y.csv");
//                 unsigned int idx = 0; 
//                 //std::string solution_path = R_path + "/simulations/sim_" + std::to_string(sim) + "/single_est" + pde_type +  "/" + lambda_selection; 
//                 std::string solution_path = R_path + "/simulations/sim_" + std::to_string(sim) + "/single" +  "/" + lambda_selection + "/" + gcv_refinement; 

//                 for(double alpha : alphas){
//                     QSRPDE<SpaceOnly> model(problem, Sampling::pointwise, alpha);
//                     model.set_spatial_locations(loc);
//                     unsigned int alpha_int = alphas[idx]*100;  
//                     double lambda; 
//                     std::ifstream fileLambda(solution_path + "/lambdas_opt_alpha_" + std::to_string(alpha_int) + ".csv");
//                     if(fileLambda.is_open()){
//                         fileLambda >> lambda; 
//                         fileLambda.close();
//                     }
//                     model.set_lambda_D(lambda);

//                     // set model data
//                     BlockFrame<double, int> df;
//                     df.insert(OBSERVATIONS_BLK, y);
//                     model.set_data(df);

//                     // solve smoothing problem
//                     model.init();
//                     model.solve();

//                     // Save solution
//                     DMatrix<double> computedF = model.f();
//                     const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filef(solution_path + "/f_" + std::to_string(alpha_int) + ".csv");
//                     if(filef.is_open()){
//                         filef << computedF.format(CSVFormatf);
//                         filef.close();
//                     }

//                     DMatrix<double> computedFn = model.Psi()*model.f();
//                     const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filefn(solution_path + "/fn_" + std::to_string(alpha_int) + ".csv");
//                     if(filefn.is_open()){
//                         filefn << computedFn.format(CSVFormatfn);
//                         filefn.close();
//                     }

//                     idx++;
//                 }

//         }


//     }


//     // Simultaneous estimations
//     if(mult_est){
//         for(std::string method : methods){  
//             bool force; 
//             bool processing; 
//             if(method == "mult"){
//                 processing = false; 
//                 force = false; 
//                 std::cout << "-------------------------MULTIPLE running-----------------" << std::endl;
//             }
//             if(method == "PP"){
//                 processing = true; 
//                 force = false; 
//                 std::cout << "-------------------------PP running-----------------" << std::endl;
//             }
//             if(method == "PP_new"){
//                 processing = true; 
//                 force = true; 
//                 std::cout << "-------------------------PP new running-----------------" << std::endl;
//             }

//             for(auto sim = 1; sim <= n_sim; ++sim){

//                 std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl; 

//                 // K = K_est
//                 // if(pde_type != "_casc")
//                 //     std::cout << "ERROR: YOU WANT TO USE K = K_est BUT YOU ARE USING SOMETHING ELSE" << std::endl; 
//                 SMatrix<2> K = read_csv<double>(R_path + "/simulations/sim_" + std::to_string(sim) + "/K.csv"); 
//                 auto L = -diffusion<FEM>(K);   // anisotropic diffusion  
//                 PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//                 MQSRPDE<SpaceOnly> model(problem, Sampling::pointwise, alphas);
//                 model.set_spatial_locations(loc);
//                 model.set_preprocess_option(processing); 
//                 model.set_forcing_option(force);

//                 // std::string solution_path = R_path + "/data/simulations/sim_" + std::to_string(sim) + "/mult_est" + pde_type + "/" + lambda_selection;
//                 // std::string lambda_path = R_path + "/data/simulations/sim_" + std::to_string(sim) + "/single_est" + pde_type + "/" + lambda_selection; 
//                 std::string solution_path = R_path + "/simulations/sim_" + std::to_string(sim) + "/multiple" + "/" + lambda_selection + "/" + gcv_refinement;
//                 std::string lambda_path = R_path + "/simulations/sim_" + std::to_string(sim) + "/single" + "/" + lambda_selection + "/" + gcv_refinement; 


//                 // use optimal lambda to avoid possible numerical issues
//                 DMatrix<double> lambdas;
//                 DVector<double> lambdas_temp; 
//                 lambdas_temp.resize(alphas.size());
//                 for(std::size_t idx = 0; idx < alphas.size(); ++idx){
//                     unsigned int alpha_int = alphas[idx]*100;  
//                     std::ifstream fileLambdas(lambda_path + "/lambdas_opt_alpha_" + std::to_string(alpha_int) + ".csv");
//                     if(fileLambdas.is_open()){
//                         fileLambdas >> lambdas_temp(idx); 
//                         fileLambdas.close();
//                     }
//                 }
//                 lambdas = lambdas_temp;                
//                 model.setLambdas_D(lambdas);

//                 // load data from .csv files
//                 DMatrix<double> y = read_csv<double>(R_path + "/simulations/sim_" + std::to_string(sim) + "/y.csv");

//                 // set model data
//                 BlockFrame<double, int> df;
//                 df.insert(OBSERVATIONS_BLK, y);
//                 model.set_data(df);

//                 // solve smoothing problem
//                 model.init();
//                 model.solve();

//                 // Save solution
//                 if(method == "mult"){
//                     DMatrix<double> computedF = model.f();
//                     const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filef(solution_path + "/f_all.csv");
//                     if(filef.is_open()){
//                         filef << computedF.format(CSVFormatf);
//                         filef.close();
//                     }

//                     DMatrix<double> computedFn = model.Psi_mult()*model.f();
//                     const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filefn(solution_path + "/fn_all.csv");
//                     if(filefn.is_open()){
//                         filefn << computedFn.format(CSVFormatfn);
//                         filefn.close();
//                     }
//                 } 
//                 // if(method == "PP"){
//                 //     DMatrix<double> computedF = model.f();
//                 //     const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                 //     std::ofstream filef(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/competitors/f_all" + pde_type + "_postproc.csv");
//                 //     if(filef.is_open()){
//                 //         filef << computedF.format(CSVFormatf);
//                 //         filef.close();
//                 //     }

//                 //     DMatrix<double> computedFn = model.Psi_mult()*model.f();
//                 //     const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                 //     std::ofstream filefn(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/competitors/fn_all" + pde_type + "_postproc.csv");
//                 //     if(filefn.is_open()){
//                 //         filefn << computedFn.format(CSVFormatfn);
//                 //         filefn.close();
//                 //     }

//                 // }
//                 // if(method == "PP_new"){
//                 //     DMatrix<double> computedF = model.f();
//                 //     const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                 //     std::ofstream filef(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/competitors/f_all" + pde_type + "_postproc_new.csv");
//                 //     if(filef.is_open()){
//                 //         filef << computedF.format(CSVFormatf);
//                 //         filef.close();
//                 //     }

//                 //     DMatrix<double> computedFn = model.Psi_mult()*model.f();
//                 //     const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                 //     std::ofstream filefn(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/competitors/fn_all" + pde_type + "_postproc_new.csv");
//                 //     if(filefn.is_open()){
//                 //         filefn << computedFn.format(CSVFormatfn);
//                 //         filefn.close();
//                 //     }

//                 // }

//             }

//         }

//     }


// }


// // /////

// // TEST x PULL-REQUEST

// // parametri multiple
// // double gamma0_ = 5.;                   // crossing penalty   
// // double eps_ = 1e-6;                    // crossing tolerance 
// // double C_ = 1.5;                       // crossing penalty factor
// // double tolerance_ = 1e-5;              // convergence tolerance 
// // double tol_weights_ = 1e-6;            // weights tolerance
// // std::size_t max_iter_ = 200;           // max number of inner iterations 
// // std::size_t max_iter_global_ = 100;    // max number of outer iterations 

// // test 1 
// //    domain:       unit square
// //    sampling:     locations != nodes
// //    penalization: constant coefficients PDE
// //    covariates:   no
// //    BC:           no
// //    order FE:     1
// TEST(mqsrpde_test1_PullRequest, laplacian_nonparametric_samplingatlocations) {

//     // define domain
//     MeshLoader<Triangulation<2, 2>> domain("unit_square_coarse");
//     // import data from files
//     DMatrix<double> locs = read_csv<double>("../data/models/mqsrpde/2D_test1/locs.csv");
//     DMatrix<double> y = read_csv<double>("../data/models/mqsrpde/2D_test1/y.csv");

//     // define regularizing PDE
//     auto L = -laplacian<FEM>();
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
//     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
//     // define model
//     std::vector<double> alphas = {0.05, 0.10, 0.50, 0.90, 0.95}; 
//     DMatrix<double> lambdas;
//     lambdas.resize(alphas.size(), 1); 
//     double lambda = 1.778279 * std::pow(0.1, 4);
//     for(auto i = 0; i < lambdas.rows(); ++i) lambdas(i, 0) = lambda;   
//     MQSRPDE<SpaceOnly> model(problem, Sampling::pointwise, alphas);
//     model.set_spatial_locations(locs);
//     model.setLambdas_D(lambdas);
//     // set model's data
//     BlockFrame<double, int> df;
//     df.insert(OBSERVATIONS_BLK, y);
//     model.set_data(df);
//     // solve smoothing problem
//     model.init();
//     model.solve();
//     // test correctness
//     EXPECT_TRUE(almost_equal(model.f(), "../data/models/mqsrpde/2D_test1/sol.mtx"));

//     // // save solutions
//     // DMatrix<double> computedF = model.f();
//     // Eigen::saveMarket(computedF, "../data/models/mqsrpde/2D_test1/sol.mtx");

// }

// // test 2 
// //    domain:       unit square
// //    sampling:     locations != nodes
// //    penalization: constant coefficients PDE
// //    covariates:   yes
// //    BC:           no
// //    order FE:     1
// TEST(mqsrpde_test2_PullRequest, laplacian_semiparametric_samplingatlocations) {

//     // define domain and regularizing PDE
//     MeshLoader<Triangulation<2, 2>> domain("unit_square_coarse");
//     // import data from files
//     DMatrix<double> locs = read_csv<double>("../data/models/mqsrpde/2D_test1/locs.csv");
//     DMatrix<double> y = read_csv<double>("../data/models/mqsrpde/2D_test2/y.csv");
//     DMatrix<double> X = read_csv<double>("../data/models/mqsrpde/2D_test2/X.csv");
//     // define regularizing PDE
//     auto L = -laplacian<FEM>();
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
//     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
//     // define statistical model
//     std::vector<double> alphas = {0.05, 0.10, 0.50, 0.90, 0.95}; 
//     DMatrix<double> lambdas;
//     lambdas.resize(alphas.size(), 1); 
//     double lambda = 1.778279 * std::pow(0.1, 4);
//     for(auto i = 0; i < lambdas.rows(); ++i) lambdas(i, 0) = lambda;   
//     MQSRPDE<SpaceOnly> model(problem, Sampling::pointwise, alphas);
//     model.set_spatial_locations(locs);
//     model.setLambdas_D(lambdas);
//     // set model data
//     BlockFrame<double, int> df;
//     df.insert(OBSERVATIONS_BLK, y);
//     df.insert(DESIGN_MATRIX_BLK, X);
//     model.set_data(df);
//     // solve smoothing problem
//     model.init();
//     model.solve();
//     // test correctness
//     EXPECT_TRUE(almost_equal(model.f()   , "../data/models/mqsrpde/2D_test2/sol.mtx" ));
//     EXPECT_TRUE(almost_equal(model.beta(), "../data/models/mqsrpde/2D_test2/beta.mtx"));

//     // // save solutions
//     // DMatrix<double> computedF = model.f();
//     // Eigen::saveMarket(computedF, "../data/models/mqsrpde/2D_test2/sol.mtx");

//     // DMatrix<double> computedBeta = model.beta();
//     // Eigen::saveMarket(computedBeta, "../data/models/mqsrpde/2D_test2/beta.mtx");

// }