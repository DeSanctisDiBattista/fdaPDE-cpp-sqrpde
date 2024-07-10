// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <fdaPDE/core.h>
#include <gtest/gtest.h>   // testing framework

#include <cstddef>
using fdapde::core::advection;
using fdapde::core::diffusion;
using fdapde::core::fem_order;
using fdapde::core::FEM;
using fdapde::core::Grid;
using fdapde::core::laplacian;
using fdapde::core::PDE;
using fdapde::core::Triangulation;

#include "../../fdaPDE/models/regression/mqsrpde.h"
#include "../../fdaPDE/models/sampling_design.h"
using fdapde::models::SRPDE;
using fdapde::models::MQSRPDE;
using fdapde::models::SpaceOnly;

#include "../../fdaPDE/models/regression/gcv.h"
using fdapde::models::ExactEDF;
using fdapde::models::GCV;
using fdapde::models::StochasticEDF;
using fdapde::models::Sampling;

#include "utils/constants.h"
#include "utils/mesh_loader.h"
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_mtx;
using fdapde::testing::read_csv;


// helper functions
double RMSE_metric(DVector<double> v1, DVector<double> v2){
    double res = 0.; 
    if(v1.size() != v2.size())
        std::cout << std::endl << "----------ERROR IN RMSE COMPUTATION---------" << std::endl; 
    for(auto i = 0; i < v1.size(); ++i){
        res += (v1[i]-v2[i])*(v1[i]-v2[i]); 
    }
    return std::sqrt(1./(v1.size())*res); 
}

    
DMatrix<double> collapse_rows(DMatrix<double> m, std::vector<bool> unique_flags, unsigned int num_unique){
    DMatrix<double> m_ret; 
    m_ret.resize(num_unique, m.cols()); 
    if(unique_flags.size() != m.rows()){
        std::cout << "problems in collapse_rows..." << std::endl; 
    }
    for(int j=0; j<m.cols(); ++j){
        unsigned int count = 0; 
        for(int i=0; i<m.rows(); ++i){
            if(unique_flags[i]){
                m_ret(count,j) = m(i,j); 
                count++; 
            }     
        }
    }
    return m_ret; 
}


// // TEST TESI

// // test 7
// //    domain:       unit square
// //    sampling:     locations != nodes
// //    penalization: constant PDE coefficients 
// //    covariates:   no
// //    BC:           no
// //    order FE:     1
// //    GCV optimization: grid exact
// TEST(gcv_msqrpde_test7, pde_nonparametric_samplingatlocations_spaceonly_gridexact) {

//     // path test  
//     std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/models/multiple_quantiles/Tests/Test_7"; 
//     //std::string R_path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/Thesis_shared/models/multiple_quantiles/Tests/Test_7"; 

//     const std::string pde_type = "_lap";    // "_lap" "_Ktrue" "_casc"

//     // define domain
//     MeshLoader<Triangulation<2, 2>> domain("unit_square_test7");

//     // rhs 
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);

//     // define regularizing PDE

//     // lap 
//     if(pde_type != "_lap")
//         std::cout << "ERROR: YOU WANT TO USE K = I BUT YOU ARE USING SOMETHING ELSE" << std::endl; 
//     auto L = -laplacian<FEM>();   
//     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//     // // K = K_true
//     // if(pde_type != "_Ktrue")
//     //     std::cout << "ERROR: YOU WANT TO USE K = K_true BUT YOU ARE USING SOMETHING ELSE" << std::endl;
//     // SMatrix<2> K = read_csv<double>(R_path + "/data/true/K_true.csv"); 
//     // auto L = -diffusion<FEM>(K);  
//     // PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//     // define statistical model
//     std::vector<double> alphas = {0.01, 0.02, 0.05, 0.10, 0.25, 
//                                   0.50, 0.75, 0.90, 0.91, 0.92, 
//                                   0.93, 0.94, 0.95, 0.96, 0.97, 
//                                   0.98, 0.99};  

//     // define grid of lambda values
//     std::vector<std::string> lambda_selection_types = {"gcv_smooth_eps1e-1"}; // {"gcv", "gcv_smooth_eps1e-3", "gcv_smooth_eps1e-2", "gcv_smooth_eps1e-1.5", "gcv_smooth_eps1e-1"}; // {"gcv", "gcv_smooth_eps1e-3", "gcv_smooth_eps1e-2", "gcv_smooth_eps1e-1.5", "gcv_smooth_eps1e-1"};     
//     std::vector<double> lambdas_1; std::vector<double> lambdas_2; std::vector<double> lambdas_5;
//     std::vector<double> lambdas_10; std::vector<double> lambdas_25; std::vector<double> lambdas_50;
//     std::vector<double> lambdas_75; std::vector<double> lambdas_90; std::vector<double> lambdas_91; 
//     std::vector<double> lambdas_92; std::vector<double> lambdas_93;  std::vector<double> lambdas_94; 
//     std::vector<double> lambdas_95; std::vector<double> lambdas_96;  std::vector<double> lambdas_97; 
//     std::vector<double> lambdas_98; std::vector<double> lambdas_99; 
//     for(double x = -7.5; x <= -3.5; x += 0.1) lambdas_1.push_back(std::pow(10, x));
//     for(double x = -7.5; x <= -3.5; x += 0.1) lambdas_2.push_back(std::pow(10, x));
//     for(double x = -8.0; x <= -2.0; x += 0.1) lambdas_5.push_back(std::pow(10, x)); 
//     for(double x = -7.5; x <= -3.0; x += 0.1) lambdas_10.push_back(std::pow(10, x)); 
//     for(double x = -7.0; x <= -3.5; x += 0.1) lambdas_25.push_back(std::pow(10, x));
//     for(double x = -6.0; x <= -3.0; x += 0.1) lambdas_50.push_back(std::pow(10, x)); 
//     for(double x = -6.0; x <= -3.0; x += 0.1) lambdas_75.push_back(std::pow(10, x)); 
//     for(double x = -7.5; x <= -3.5; x += 0.1) lambdas_90.push_back(std::pow(10, x));
//     for(double x = -6.5; x <= -3.5; x += 0.1) lambdas_91.push_back(std::pow(10, x));
//     for(double x = -6.5; x <= -3.5; x += 0.1) lambdas_92.push_back(std::pow(10, x));
//     for(double x = -6.5; x <= -3.5; x += 0.1) lambdas_93.push_back(std::pow(10, x));
//     for(double x = -6.5; x <= -4.0; x += 0.1) lambdas_94.push_back(std::pow(10, x));
//     for(double x = -7.5; x <= -3.5; x += 0.1) lambdas_95.push_back(std::pow(10, x));
//     for(double x = -6.5; x <= -4.0; x += 0.1) lambdas_96.push_back(std::pow(10, x));
//     for(double x = -6.5; x <= -4.0; x += 0.1) lambdas_97.push_back(std::pow(10, x));
//     for(double x = -7.5; x <= -3.5; x += 0.1) lambdas_98.push_back(std::pow(10, x)); 
//     for(double x = -8.0; x <= -3.5; x += 0.1) lambdas_99.push_back(std::pow(10, x));
//     double best_lambda; 

//     // Read covariates and locations
//     DMatrix<double> loc = read_csv<double>(R_path + "/data" + "/locs.csv"); 

//     // Simulations 
//     std::vector<unsigned int> simulations = {25}; // {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}; 
//     for(auto sim : simulations){
//         std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl; 

//         std::string solutions_path_rmse = R_path + "/data/simulations/sim_" + std::to_string(sim) + "/single_est" + pde_type + "/RMSE"; 

//         // // K = K_est
//         // if(pde_type != "_casc")
//         //     std::cout << "ERROR: YOU WANT TO USE K = K_est BUT YOU ARE USING SOMETHING ELSE" << std::endl; 
//         // SMatrix<2> K = read_csv<double>(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/K.csv"); 
//         // auto L = -diffusion<FEM>(K);   // anisotropic diffusion  
//         // PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//         for(auto alpha : alphas){

//             unsigned int alpha_int = alpha*100; 
//             std::string alpha_string = std::to_string(alpha_int); 

//             std::cout << "------------------alpha=" << alpha_string << "-----------------" << std::endl; 

//             // load data from .csv files
//             DMatrix<double> y = read_csv<double>(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/y.csv");
//             BlockFrame<double, int> df;
//             df.insert(OBSERVATIONS_BLK, y);

//             // GCV:
//             for(auto lambda_selection_type : lambda_selection_types){
                
//                 std::string solutions_path_gcv = R_path + "/data/simulations/sim_" + std::to_string(sim) + "/single_est" + pde_type + "/" + lambda_selection_type; 
                
//                 QSRPDE<SpaceOnly> model_gcv(problem, Sampling::pointwise, alpha);
//                 model_gcv.set_spatial_locations(loc);

//                 std::vector<double> lambdas;
//                 if(almost_equal(alpha, 0.01)){
//                     lambdas = lambdas_1; 
//                 }  
//                 if(almost_equal(alpha, 0.02)){
//                     lambdas = lambdas_2; 
//                 }  
//                 if(almost_equal(alpha, 0.05)){
//                     lambdas = lambdas_5; 
//                 }  
//                 if(almost_equal(alpha, 0.10)){
//                     lambdas = lambdas_10; 
//                 }  
//                 if(almost_equal(alpha, 0.25)){
//                     lambdas = lambdas_25; 
//                 }  
//                 if(almost_equal(alpha, 0.50)){
//                     lambdas = lambdas_50; 
//                 }  
//                 if(almost_equal(alpha, 0.75)){
//                     lambdas = lambdas_75; 
//                 }  
//                 if(almost_equal(alpha, 0.90)){
//                     lambdas = lambdas_90; 
//                 } 
//                 if(almost_equal(alpha, 0.91)){
//                     lambdas = lambdas_91; 
//                 }   
//                 if(almost_equal(alpha, 0.92)){
//                     lambdas = lambdas_92; 
//                 }  
//                 if(almost_equal(alpha, 0.93)){
//                     lambdas = lambdas_93; 
//                 } 
//                 if(almost_equal(alpha, 0.94)){
//                     lambdas = lambdas_94; 
//                 }   
//                 if(almost_equal(alpha, 0.95)){
//                     lambdas = lambdas_95; 
//                 } 
//                 if(almost_equal(alpha, 0.96)){
//                     lambdas = lambdas_96; 
//                 }    
//                 if(almost_equal(alpha, 0.97)){
//                     lambdas = lambdas_97; 
//                 }    
//                 if(almost_equal(alpha, 0.98)){
//                     lambdas = lambdas_98; 
//                 }  
//                 if(almost_equal(alpha, 0.99)){
//                     lambdas = lambdas_99; 
//                 }  
//                 // define lambda sequence as matrix 
//                 DMatrix<double> lambdas_mat;
//                 lambdas_mat.resize(lambdas.size(), 1); 
//                 for(auto i = 0; i < lambdas_mat.rows(); ++i){
//                    lambdas_mat(i,0) = lambdas[i]; 
//                  }
//                 // set model's data
//                 model_gcv.set_exact_gcv(lambda_selection_type == "gcv"); 

//                 if(lambda_selection_type == "gcv_smooth_eps1e-3"){
//                     model_gcv.set_eps_power(-3.0); 
//                 }
//                 if(lambda_selection_type == "gcv_smooth_eps1e-2"){
//                     model_gcv.set_eps_power(-2.0); 
//                 }
//                 if(lambda_selection_type == "gcv_smooth_eps1e-1.5"){
//                     model_gcv.set_eps_power(-1.5); 
//                 }
//                 if(lambda_selection_type == "gcv_smooth_eps1e-1"){
//                     model_gcv.set_eps_power(-1.0); 
//                 }
                
//                 model_gcv.set_data(df);
//                 model_gcv.init();

//                 // define GCV function and grid of \lambda_D values
//                 auto GCV = model_gcv.gcv<ExactEDF>();
//                 // optimize GCV
//                 Grid<fdapde::Dynamic> opt;
//                 opt.optimize(GCV, lambdas_mat);
                
//                 best_lambda = opt.optimum()(0,0);
        
//                 std::cout << "Best lambda is: " << std::setprecision(16) << best_lambda << std::endl; 

//                 // Save lambda sequence 
//                 std::ofstream fileLambdaS(solutions_path_gcv + "/lambdas_seq_alpha_" + alpha_string + ".csv");
//                 for(std::size_t i = 0; i < lambdas.size(); ++i) 
//                     fileLambdaS << std::setprecision(16) << lambdas[i] << "\n"; 
//                 fileLambdaS.close();

//                 // Save lambda GCVopt for all alphas
//                 std::ofstream fileLambdaoptS(solutions_path_gcv + "/lambdas_opt_alpha_" + alpha_string + ".csv");
//                 if(fileLambdaoptS.is_open()){
//                     fileLambdaoptS << std::setprecision(16) << best_lambda;
//                     fileLambdaoptS.close();
//                 }

//                 // Save GCV 
//                 std::ofstream fileGCV_scores(solutions_path_gcv + "/score_alpha_" + alpha_string + ".csv");
//                 for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
//                     fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.gcvs()[i]) << "\n"; 
//                 fileGCV_scores.close();
//             }

//         }


//     }
// }



// // test 8
// //    domain:       unit square
// //    sampling:     locations != nodes
// //    penalization: constant PDE coefficients 
// //    covariates:   yes
// //    BC:           no
// //    order FE:     1
// //    GCV optimization: grid exact
// TEST(gcv_msqrpde_test8, pde_nonparametric_samplingatlocations_spaceonly_gridexact) {

//     // path test  
//     std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/models/multiple_quantiles/Tests/Test_8"; 
//     //std::string R_path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/Thesis_shared/models/multiple_quantiles/Tests/Test_8"; 

//     const std::string pde_type = "_casc";    // "_lap" "_Ktrue" "_casc"
//     const bool save = false; 

//     // define domain
//     MeshLoader<Triangulation<2, 2>> domain("unit_square_test8");

//     // rhs 
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);

//     // define regularizing PDE

//     // // lap 
//     // if(pde_type != "_lap")
//     //     std::cout << "ERROR: YOU WANT TO USE K = I BUT YOU ARE USING SOMETHING ELSE" << std::endl; 
//     // auto L = -laplacian<FEM>();   
//     // PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//     // // K = K_true
//     // if(pde_type != "_Ktrue")
//     //     std::cout << "ERROR: YOU WANT TO USE K = K_true BUT YOU ARE USING SOMETHING ELSE" << std::endl;
//     // SMatrix<2> K = read_csv<double>(R_path + "/data/true/K_true.csv"); 
//     // auto L = -diffusion<FEM>(K);  
//     // PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//     // define statistical model
//     std::vector<double> alphas = {0.50, 0.75}; 
//     // {0.01, 0.02, 0.05, 0.10, 
//     //                               0.25, 0.50, 0.75, 
//     //                               0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99};  

//     // define grid of lambda values
//     std::vector<std::string> lambda_selection_types = {"gcv_smooth_eps1e-1"}; // {"gcv", "gcv_smooth_eps1e-3", "gcv_smooth_eps1e-2", "gcv_smooth_eps1e-1.5", "gcv_smooth_eps1e-1"}; // {"gcv", "gcv_smooth_eps1e-3", "gcv_smooth_eps1e-2", "gcv_smooth_eps1e-1.5", "gcv_smooth_eps1e-1"};     
//     std::vector<double> lambdas_1; std::vector<double> lambdas_2; std::vector<double> lambdas_5;
//     std::vector<double> lambdas_10; std::vector<double> lambdas_25; std::vector<double> lambdas_50;
//     std::vector<double> lambdas_75; std::vector<double> lambdas_90; std::vector<double> lambdas_91; 
//     std::vector<double> lambdas_92; std::vector<double> lambdas_93;  std::vector<double> lambdas_94; 
//     std::vector<double> lambdas_95; std::vector<double> lambdas_96;  std::vector<double> lambdas_97; 
//     std::vector<double> lambdas_98; std::vector<double> lambdas_99; 
//     for(double x = -7.5; x <= -3.5; x += 0.1) lambdas_1.push_back(std::pow(10, x));
//     for(double x = -7.5; x <= -3.5; x += 0.1) lambdas_2.push_back(std::pow(10, x));
//     for(double x = -8.0; x <= -2.0; x += 0.1) lambdas_5.push_back(std::pow(10, x)); 
//     for(double x = -7.5; x <= -3.0; x += 0.1) lambdas_10.push_back(std::pow(10, x)); 
//     for(double x = -7.0; x <= -3.5; x += 0.1) lambdas_25.push_back(std::pow(10, x));
//     for(double x = -6.0; x <= -3.0; x += 1.0) lambdas_50.push_back(std::pow(10, x)); 
//     for(double x = -6.0; x <= -3.0; x += 1.0) lambdas_75.push_back(std::pow(10, x)); 
//     for(double x = -7.5; x <= -3.5; x += 0.1) lambdas_90.push_back(std::pow(10, x));
//     for(double x = -6.5; x <= -3.5; x += 0.1) lambdas_91.push_back(std::pow(10, x));
//     for(double x = -6.5; x <= -3.5; x += 0.1) lambdas_92.push_back(std::pow(10, x));
//     for(double x = -6.5; x <= -3.5; x += 0.1) lambdas_93.push_back(std::pow(10, x));
//     for(double x = -6.5; x <= -4.0; x += 0.1) lambdas_94.push_back(std::pow(10, x));
//     for(double x = -7.5; x <= -3.5; x += 0.1) lambdas_95.push_back(std::pow(10, x));
//     for(double x = -6.5; x <= -4.0; x += 0.1) lambdas_96.push_back(std::pow(10, x));
//     for(double x = -6.5; x <= -4.0; x += 0.1) lambdas_97.push_back(std::pow(10, x));
//     for(double x = -7.5; x <= -3.5; x += 0.1) lambdas_98.push_back(std::pow(10, x)); 
//     for(double x = -8.0; x <= -1.5; x += 0.1) lambdas_99.push_back(std::pow(10, x));
//     double best_lambda; 

//     // Read covariates and locations
//     DMatrix<double> loc = read_csv<double>(R_path + "/data" + "/locs.csv"); 
//     DMatrix<double> X = read_csv<double>(R_path + "/data" + "/X.csv"); 

//     // Simulations 
//     std::vector<unsigned int> simulations = {1}; // {1,2,3,4,5,6,7,8,9,10}; 
//     for(auto sim : simulations){

//         std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl; 

//         // K = K_est
//         if(pde_type != "_casc")
//             std::cout << "ERROR: YOU WANT TO USE K = K_est BUT YOU ARE USING SOMETHING ELSE" << std::endl; 
//         SMatrix<2> K = read_csv<double>(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/K.csv"); 
//         auto L = -diffusion<FEM>(K);   // anisotropic diffusion  
//         PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//         for(auto alpha : alphas){

//             unsigned int alpha_int = alpha*100; 
//             std::string alpha_string = std::to_string(alpha_int); 

//             std::cout << "------------------alpha=" << alpha_string << "-----------------" << std::endl; 

//             // load data from .csv files
//             DMatrix<double> y = read_csv<double>(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/y.csv");
//             BlockFrame<double, int> df;
//             df.insert(OBSERVATIONS_BLK, y);
//             df.insert(DESIGN_MATRIX_BLK, X);

//             // GCV:
//             for(auto lambda_selection_type : lambda_selection_types){
                
//                 std::string solutions_path_gcv = R_path + "/data/simulations/sim_" + std::to_string(sim) + "/single_est" + pde_type + "/" + lambda_selection_type; 
                
//                 QSRPDE<SpaceOnly> model_gcv(problem, Sampling::pointwise, alpha);
//                 model_gcv.set_spatial_locations(loc);

//                 std::vector<double> lambdas;
//                 if(almost_equal(alpha, 0.01)){
//                     lambdas = lambdas_1; 
//                 }  
//                 if(almost_equal(alpha, 0.02)){
//                     lambdas = lambdas_2; 
//                 }  
//                 if(almost_equal(alpha, 0.05)){
//                     lambdas = lambdas_5; 
//                 }  
//                 if(almost_equal(alpha, 0.10)){
//                     lambdas = lambdas_10; 
//                 }  
//                 if(almost_equal(alpha, 0.25)){
//                     lambdas = lambdas_25; 
//                 }  
//                 if(almost_equal(alpha, 0.50)){
//                     lambdas = lambdas_50; 
//                 }  
//                 if(almost_equal(alpha, 0.75)){
//                     lambdas = lambdas_75; 
//                 }  
//                 if(almost_equal(alpha, 0.90)){
//                     lambdas = lambdas_90; 
//                 } 
//                 if(almost_equal(alpha, 0.91)){
//                     lambdas = lambdas_91; 
//                 }   
//                 if(almost_equal(alpha, 0.92)){
//                     lambdas = lambdas_92; 
//                 }  
//                 if(almost_equal(alpha, 0.93)){
//                     lambdas = lambdas_93; 
//                 } 
//                 if(almost_equal(alpha, 0.94)){
//                     lambdas = lambdas_94; 
//                 }   
//                 if(almost_equal(alpha, 0.95)){
//                     lambdas = lambdas_95; 
//                 } 
//                 if(almost_equal(alpha, 0.96)){
//                     lambdas = lambdas_96; 
//                 }    
//                 if(almost_equal(alpha, 0.97)){
//                     lambdas = lambdas_97; 
//                 }    
//                 if(almost_equal(alpha, 0.98)){
//                     lambdas = lambdas_98; 
//                 }  
//                 if(almost_equal(alpha, 0.99)){
//                     lambdas = lambdas_99; 
//                 }  
//                 // define lambda sequence as matrix 
//                 DMatrix<double> lambdas_mat;
//                 lambdas_mat.resize(lambdas.size(), 1); 
//                 for(auto i = 0; i < lambdas_mat.rows(); ++i){
//                    lambdas_mat(i,0) = lambdas[i]; 
//                 }

//                 if(lambda_selection_type == "gcv_smooth_eps1e-3"){
//                     model_gcv.set_eps_power(-3.0); 
//                 }
//                 if(lambda_selection_type == "gcv_smooth_eps1e-2"){
//                     model_gcv.set_eps_power(-2.0); 
//                 }
//                 if(lambda_selection_type == "gcv_smooth_eps1e-1.5"){
//                     model_gcv.set_eps_power(-1.5); 
//                 }
//                 if(lambda_selection_type == "gcv_smooth_eps1e-1"){
//                     model_gcv.set_eps_power(-1.0); 
//                 }
                
//                 model_gcv.set_data(df);
//                 model_gcv.init();

//                 // define GCV function and grid of \lambda_D values
//                 auto GCV = model_gcv.gcv<ExactEDF>();
//                 // optimize GCV
//                 Grid<fdapde::Dynamic> opt;
//                 opt.optimize(GCV, lambdas_mat);
                
//                 best_lambda = opt.optimum()(0,0);
        
//                 std::cout << "Best lambda is: " << std::setprecision(16) << best_lambda << std::endl; 

//                 if(save){
//                     // Save lambda sequence 
//                     std::ofstream fileLambdaS(solutions_path_gcv + "/lambdas_seq_alpha_" + alpha_string + ".csv");
//                     for(std::size_t i = 0; i < lambdas.size(); ++i) 
//                         fileLambdaS << std::setprecision(16) << lambdas[i] << "\n"; 
//                     fileLambdaS.close();

//                     // Save lambda GCVopt for all alphas
//                     std::ofstream fileLambdaoptS(solutions_path_gcv + "/lambdas_opt_alpha_" + alpha_string + ".csv");
//                     if(fileLambdaoptS.is_open()){
//                         fileLambdaoptS << std::setprecision(16) << best_lambda;
//                         fileLambdaoptS.close();
//                     }

//                     // Save GCV 
//                     std::ofstream fileGCV_scores(solutions_path_gcv + "/score_alpha_" + alpha_string + ".csv");
//                     for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
//                         fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.gcvs()[i]) << "\n"; 
//                     fileGCV_scores.close();
//                 }

//             }

//         }


//     }
// }



// // //// 

// // // TEST OBS RIPETUTE

// // test 7 OBS RIPETUTE
// //    domain:       unit square
// //    sampling:     locations != nodes
// //    penalization: constant PDE coefficients 
// //    covariates:   no
// //    BC:           no
// //    order FE:     1
// //    GCV optimization: grid exact
// TEST(gcv_msqrpde_test7_obs_rip, pde_nonparametric_samplingatlocations_spaceonly_gridexact) {

//     // path test  
//     // std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/models/multiple_quantiles/Tests/Test_7"; 
//     std::string R_path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/Thesis_shared/models/multiple_quantiles/Tests/Test_7_obs_ripetute"; 

//     const std::string pde_type = "_lap";    // "_lap" "_Ktrue" "_casc"
    
//     // usare le obs ripetute?
//     bool bool_obs_rip = true;

//     // define domain
//     MeshLoader<Triangulation<2, 2>> domain("unit_square_test7");

//     // rhs 
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);

//     // define regularizing PDE

//     // lap 
//     if(pde_type != "_lap")
//         std::cout << "ERROR: YOU WANT TO USE K = I BUT YOU ARE USING SOMETHING ELSE" << std::endl; 
//     auto L = -laplacian<FEM>();   
//     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//     // // K = K_true
//     // if(pde_type != "_Ktrue")
//     //     std::cout << "ERROR: YOU WANT TO USE K = K_true BUT YOU ARE USING SOMETHING ELSE" << std::endl;
//     // SMatrix<2> K = read_csv<double>(R_path + "/data/true/K_true.csv"); 
//     // auto L = -diffusion<FEM>(K);  
//     // PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//     // // define statistical model
//     // std::vector<double> alphas = {0.01, 0.02, 0.05, 0.10, 0.25, 
//     //                               0.50, 0.75, 0.90, 0.91, 0.92, 
//     //                               0.93, 0.94, 0.95, 0.96, 0.97, 
//     //                               0.98, 0.99};  

//     std::vector<double> alphas = {0.01, 0.02, 0.05, 0.10, 0.25, 
//                                   0.75, 0.91, 0.92, 
//                                   0.93, 0.94, 0.96, 0.97, 
//                                   0.98};  

//     // std::vector<double> alphas = {0.50, 0.90, 0.95, 0.99};  

//     // define grid of lambda values
//     std::vector<std::string> lambda_selection_types = {"gcv_smooth_eps1e-1"}; // {"gcv", "gcv_smooth_eps1e-3", "gcv_smooth_eps1e-2", "gcv_smooth_eps1e-1.5", "gcv_smooth_eps1e-1"}; // {"gcv", "gcv_smooth_eps1e-3", "gcv_smooth_eps1e-2", "gcv_smooth_eps1e-1.5", "gcv_smooth_eps1e-1"};     
//     const std::string eps_string = "1e-1"; 
    

//     std::vector<double> lambdas_1; std::vector<double> lambdas_2; std::vector<double> lambdas_5;
//     std::vector<double> lambdas_10; std::vector<double> lambdas_25; std::vector<double> lambdas_50;
//     std::vector<double> lambdas_75; std::vector<double> lambdas_90; std::vector<double> lambdas_91; 
//     std::vector<double> lambdas_92; std::vector<double> lambdas_93;  std::vector<double> lambdas_94; 
//     std::vector<double> lambdas_95; std::vector<double> lambdas_96;  std::vector<double> lambdas_97; 
//     std::vector<double> lambdas_98; std::vector<double> lambdas_99; 

//     // for(double x = -10.0; x <= -3.0; x += 0.2) lambdas_50.push_back(std::pow(10, x));
//     // for(double x = -10.0; x <= -3.0; x += 0.2) lambdas_90.push_back(std::pow(10, x)); 
//     // for(double x = -10.0; x <= -3.0; x += 0.2) lambdas_95.push_back(std::pow(10, x));
//     // for(double x = -10.0; x <= -3.0; x += 0.2) lambdas_99.push_back(std::pow(10, x));

//     for(double x = -10.0; x <= -3.0; x += 0.2) lambdas_1.push_back(std::pow(10, x));
//     for(double x = -10.0; x <= -3.0; x += 0.2) lambdas_2.push_back(std::pow(10, x));
//     for(double x = -10.0; x <= -3.0; x += 0.2) lambdas_5.push_back(std::pow(10, x)); 
//     for(double x = -10.0; x <= -3.0; x += 0.2) lambdas_10.push_back(std::pow(10, x)); 
//     for(double x = -10.0; x <= -3.0; x += 0.2) lambdas_25.push_back(std::pow(10, x));
//     for(double x = -10.0; x <= -3.0; x += 0.2) lambdas_75.push_back(std::pow(10, x)); 
//     for(double x = -10.0; x <= -3.0; x += 0.2) lambdas_91.push_back(std::pow(10, x));
//     for(double x = -10.0; x <= -3.0; x += 0.2) lambdas_92.push_back(std::pow(10, x));
//     for(double x = -10.0; x <= -3.0; x += 0.2) lambdas_93.push_back(std::pow(10, x));
//     for(double x = -10.0; x <= -3.0; x += 0.2) lambdas_94.push_back(std::pow(10, x));
//     for(double x = -10.0; x <= -3.0; x += 0.2) lambdas_96.push_back(std::pow(10, x));
//     for(double x = -10.0; x <= -3.0; x += 0.2) lambdas_97.push_back(std::pow(10, x));
//     for(double x = -10.0; x <= -3.0; x += 0.2) lambdas_98.push_back(std::pow(10, x)); 

//     double best_lambda; 

//     // Read covariates and locations
//     DMatrix<double> loc ; 
//     if(bool_obs_rip)
//         loc = read_csv<double>(R_path + "/data" + "/locs_ripetute.csv"); 
//     else
//         loc = read_csv<double>(R_path + "/data" + "/locs.csv"); 

//     // Simulations 
//     const unsigned int n_sim = 10; 
//     for(auto sim = 1; sim <= n_sim; ++sim){
//         std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl; 

//         // std::string solutions_path_rmse = R_path + "/data/simulations/sim_" + std::to_string(sim) + "/single_est" + pde_type + "/RMSE"; 

//         // // K = K_est
//         // if(pde_type != "_casc")
//         //     std::cout << "ERROR: YOU WANT TO USE K = K_est BUT YOU ARE USING SOMETHING ELSE" << std::endl; 
//         // SMatrix<2> K = read_csv<double>(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/K.csv"); 
//         // auto L = -diffusion<FEM>(K);   // anisotropic diffusion  
//         // PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//         for(auto alpha : alphas){

//             unsigned int alpha_int = alpha*100; 
//             std::string alpha_string = std::to_string(alpha_int); 

//             std::cout << "------------------alpha=" << alpha_string << "-----------------" << std::endl; 

//             // load data from .csv files
//             DMatrix<double> y; 
//             if(bool_obs_rip)
//                 y = read_csv<double>(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/y_ripetute.csv");
//             else
//                 y = read_csv<double>(R_path + "/data/simulations/sim_" + std::to_string(sim) + "/y.csv");

//             // check dimensions
//             std::cout << "dim loc " << loc.rows() << " , " << loc.cols() << std::endl;
//             std::cout << "dim y " << y.rows() << std::endl;


//             BlockFrame<double, int> df;
//             df.insert(OBSERVATIONS_BLK, y);

//             // GCV:
//             for(auto lambda_selection_type : lambda_selection_types){
                
//                 std::string solutions_path_gcv = R_path + "/data/simulations/sim_" + std::to_string(sim) + "/single_est" + pde_type + "/" + lambda_selection_type; 

//                 if(bool_obs_rip)
//                     solutions_path_gcv = solutions_path_gcv + "/obs_ripetute"; 
//                 else
//                     solutions_path_gcv = solutions_path_gcv + "/obs_singola"; 


                
//                 QSRPDE<SpaceOnly> model_gcv(problem, Sampling::pointwise, alpha);
//                 model_gcv.set_spatial_locations(loc);

//                 std::vector<double> lambdas;
//                 if(almost_equal(alpha, 0.01)){
//                     lambdas = lambdas_1; 
//                 }  
//                 if(almost_equal(alpha, 0.02)){
//                     lambdas = lambdas_2; 
//                 }  
//                 if(almost_equal(alpha, 0.05)){
//                     lambdas = lambdas_5; 
//                 }  
//                 if(almost_equal(alpha, 0.10)){
//                     lambdas = lambdas_10; 
//                 }  
//                 if(almost_equal(alpha, 0.25)){
//                     lambdas = lambdas_25; 
//                 }  
//                 if(almost_equal(alpha, 0.50)){
//                     lambdas = lambdas_50; 
//                 }  
//                 if(almost_equal(alpha, 0.75)){
//                     lambdas = lambdas_75; 
//                 }  
//                 if(almost_equal(alpha, 0.90)){
//                     lambdas = lambdas_90; 
//                 } 
//                 if(almost_equal(alpha, 0.91)){
//                     lambdas = lambdas_91; 
//                 }   
//                 if(almost_equal(alpha, 0.92)){
//                     lambdas = lambdas_92; 
//                 }  
//                 if(almost_equal(alpha, 0.93)){
//                     lambdas = lambdas_93; 
//                 } 
//                 if(almost_equal(alpha, 0.94)){
//                     lambdas = lambdas_94; 
//                 }   
//                 if(almost_equal(alpha, 0.95)){
//                     lambdas = lambdas_95; 
//                 } 
//                 if(almost_equal(alpha, 0.96)){
//                     lambdas = lambdas_96; 
//                 }    
//                 if(almost_equal(alpha, 0.97)){
//                     lambdas = lambdas_97; 
//                 }    
//                 if(almost_equal(alpha, 0.98)){
//                     lambdas = lambdas_98; 
//                 }  
//                 if(almost_equal(alpha, 0.99)){
//                     lambdas = lambdas_99; 
//                 }  

//                 // define lambda sequence as matrix 
//                 DMatrix<double> lambdas_mat;
//                 lambdas_mat.resize(lambdas.size(), 1); 
//                 for(auto i = 0; i < lambdas_mat.rows(); ++i){
//                     lambdas_mat(i,0) = lambdas[i]; 
//                 }


//                 // set model's data
//                 if(eps_string == "1e-1"){
//                     model_gcv.set_eps_power(-1.0); 
//                 }
                
//                 model_gcv.set_data(df);
//                 model_gcv.init();

//                 // define GCV function and grid of \lambda_D values
//                 auto GCV = model_gcv.gcv<ExactEDF>();
//                 // optimize GCV
//                 Grid<fdapde::Dynamic> opt;
//                 opt.optimize(GCV, lambdas_mat);
                
//                 best_lambda = opt.optimum()(0,0);
        
//                 std::cout << "Best lambda is: " << std::setprecision(16) << best_lambda << std::endl; 

//                 // Save lambda sequence 
//                 std::ofstream fileLambdaS(solutions_path_gcv + "/lambdas_seq_alpha_" + alpha_string + ".csv");
//                 for(std::size_t i = 0; i < lambdas.size(); ++i) 
//                     fileLambdaS << std::setprecision(16) << lambdas[i] << "\n"; 
//                 fileLambdaS.close();

//                 // Save lambda GCVopt for all alphas
//                 std::ofstream fileLambdaoptS(solutions_path_gcv + "/lambdas_opt_alpha_" + alpha_string + ".csv");
//                 if(fileLambdaoptS.is_open()){
//                     fileLambdaoptS << std::setprecision(16) << best_lambda;
//                     fileLambdaoptS.close();
//                 }

//                 // Save GCV scores
//                 std::ofstream fileGCV_scores(solutions_path_gcv + "/score_alpha_" + alpha_string + ".csv");
//                 for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
//                     fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.gcvs()[i]) << "\n"; 
//                 fileGCV_scores.close();

//                 // Save GCV edf
//                 std::ofstream fileGCV_edfs(solutions_path_gcv + "/edf_alpha_" + alpha_string + ".csv");
//                 for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
//                     fileGCV_edfs << std::setprecision(16) << GCV.edfs()[i] << "\n"; 
//                 fileGCV_edfs.close();
//             }

//         }


//     }
// }



// //// 

// // NUOVI TEST POST-TESI

// // test 1
// //    domain:       unit square
// //    sampling:     locations != nodes
// //    penalization: constant PDE coefficients 
// //    covariates:   no
// //    BC:           no
// //    order FE:     1
// //    GCV optimization: grid exact
// TEST(gcv_msqrpde_test1, pde_nonparametric_samplingatlocations_spaceonly_gridexact) {

//     // path test  
//     std::string R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/PhD/Codice/models/MQSRPDE/Tests/Test_1"; 

//     // const std::string pde_type = "_lap";    // "_lap" "_Ktrue" "_casc"

//     const unsigned int n_sim = 20;
//     const std::string gcv_refinement = "fine";    // "lasco" "fine"
//     double lambdas_step; 
//     if(gcv_refinement == "lasco"){
//         lambdas_step = 0.5;
//     } 
//     if(gcv_refinement == "fine"){
//         lambdas_step = 0.1; 
//     } 

//     // define domain
//     MeshLoader<Triangulation<2, 2>> domain("unit_square_test7");  // il dominio è lo stesso del vecchio test 7 

//     // rhs 
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);

//     // define regularizing PDE

//     // // lap 
//     // if(pde_type != "_lap")
//     //     std::cout << "ERROR: YOU WANT TO USE K = I BUT YOU ARE USING SOMETHING ELSE" << std::endl; 
//     // auto L = -laplacian<FEM>();   
//     // PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//     // // K = K_true
//     // if(pde_type != "_Ktrue")
//     //     std::cout << "ERROR: YOU WANT TO USE K = K_true BUT YOU ARE USING SOMETHING ELSE" << std::endl;
//     // SMatrix<2> K = read_csv<double>(R_path + "/data/true/K_true.csv"); 
//     // auto L = -diffusion<FEM>(K);  
//     // PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//     // define statistical model
//     std::vector<double> alphas = {0.01, 0.02, 0.03,   // ATT: aggiunto 3%
//                                   0.05, 0.10, 0.25, 
//                                   0.50, 0.75, 0.90, 0.91, 0.92, 
//                                   0.93, 0.94, 0.95, 0.96, 0.97, 
//                                   0.98, 0.99};  

//     // define grid of lambda values
//     std::vector<std::string> lambda_selection_types = {"gcv_smooth_eps1e-1"}; // {"gcv", "gcv_smooth_eps1e-3", "gcv_smooth_eps1e-2", "gcv_smooth_eps1e-1.5", "gcv_smooth_eps1e-1"}; // {"gcv", "gcv_smooth_eps1e-3", "gcv_smooth_eps1e-2", "gcv_smooth_eps1e-1.5", "gcv_smooth_eps1e-1"};     
//     std::vector<double> lambdas_1; std::vector<double> lambdas_2; std::vector<double> lambdas_3; 
//     std::vector<double> lambdas_5;
//     std::vector<double> lambdas_10; std::vector<double> lambdas_25; std::vector<double> lambdas_50;
//     std::vector<double> lambdas_75; std::vector<double> lambdas_90; std::vector<double> lambdas_91; 
//     std::vector<double> lambdas_92; std::vector<double> lambdas_93;  std::vector<double> lambdas_94; 
//     std::vector<double> lambdas_95; std::vector<double> lambdas_96;  std::vector<double> lambdas_97; 
//     std::vector<double> lambdas_98; std::vector<double> lambdas_99; 
//     for(double x = -7.0; x <= -4.0; x += lambdas_step) lambdas_1.push_back(std::pow(10, x));
//     for(double x = -7.0; x <= -4.0; x += lambdas_step) lambdas_2.push_back(std::pow(10, x));
//     for(double x = -7.5; x <= -3.5; x += lambdas_step) lambdas_3.push_back(std::pow(10, x));
//     for(double x = -7.5; x <= -3.5; x += lambdas_step) lambdas_5.push_back(std::pow(10, x)); 
//     for(double x = -7.0; x <= -3.0; x += lambdas_step) lambdas_10.push_back(std::pow(10, x)); 
//     for(double x = -6.0; x <= -3.0; x += lambdas_step) lambdas_25.push_back(std::pow(10, x));
//     for(double x = -6.0; x <= -3.0; x += lambdas_step) lambdas_50.push_back(std::pow(10, x)); 
//     for(double x = -6.0; x <= -3.0; x += lambdas_step) lambdas_75.push_back(std::pow(10, x)); 
//     for(double x = -7.5; x <= -3.0; x += lambdas_step) lambdas_90.push_back(std::pow(10, x));
//     for(double x = -7.5; x <= -3.0; x += lambdas_step) lambdas_91.push_back(std::pow(10, x));
//     for(double x = -7.5; x <= -3.0; x += lambdas_step) lambdas_92.push_back(std::pow(10, x));
//     for(double x = -7.5; x <= -3.0; x += lambdas_step) lambdas_93.push_back(std::pow(10, x));
//     for(double x = -7.5; x <= -3.5; x += lambdas_step) lambdas_94.push_back(std::pow(10, x));
//     for(double x = -7.5; x <= -3.0; x += lambdas_step) lambdas_95.push_back(std::pow(10, x));
//     for(double x = -7.5; x <= -4.0; x += lambdas_step) lambdas_96.push_back(std::pow(10, x));
//     for(double x = -6.5; x <= -3.0; x += lambdas_step) lambdas_97.push_back(std::pow(10, x));
//     for(double x = -6.5; x <= -2.5; x += lambdas_step) lambdas_98.push_back(std::pow(10, x)); 
//     for(double x = -6.5; x <= -2.0; x += lambdas_step) lambdas_99.push_back(std::pow(10, x));
//     double best_lambda; 

//     // Read covariates and locations
//     DMatrix<double> loc = read_csv<double>(R_path + "/locs.csv"); 

//     // Simulations  
//     // std::vector<unsigned int> simulations = {25}; // {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}; 
//     for(auto sim = 1; sim <= n_sim; ++sim){
//         std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl; 

//         // std::string solutions_path_rmse = R_path + "/data/simulations/sim_" + std::to_string(sim) + "/single_est" + pde_type + "/RMSE"; 
//         std::string solutions_path_rmse = R_path + "/simulations/sim_" + std::to_string(sim) + "/single" + "/true_lambda"; 

//         // // K = K_est
//         // if(pde_type != "_casc")
//         //     std::cout << "ERROR: YOU WANT TO USE K = K_est BUT YOU ARE USING SOMETHING ELSE" << std::endl; 
//         SMatrix<2> K = read_csv<double>(R_path + "/simulations/sim_" + std::to_string(sim) + "/K.csv"); 

//         auto L = -diffusion<FEM>(K);   // anisotropic diffusion  
//         PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//         for(auto alpha : alphas){

//             unsigned int alpha_int = alpha*100; 
//             std::string alpha_string = std::to_string(alpha_int); 

//             std::cout << "------------------alpha=" << alpha_string << "-----------------" << std::endl; 

//             // load data from .csv files
//             DMatrix<double> y = read_csv<double>(R_path + "/simulations/sim_" + std::to_string(sim) + "/y.csv");
//             BlockFrame<double, int> df;
//             df.insert(OBSERVATIONS_BLK, y);

//             // GCV:
//             for(auto lambda_selection_type : lambda_selection_types){
                
//                 // std::string solutions_path_gcv = R_path + "/data/simulations/sim_" + std::to_string(sim) + "/single_est" + pde_type + "/" + lambda_selection_type; 
//                 std::string solutions_path_gcv = R_path + "/simulations/sim_" + std::to_string(sim) + "/single" + "/" + lambda_selection_type + "/" + gcv_refinement; 
                       
//                 QSRPDE<SpaceOnly> model_gcv(problem, Sampling::pointwise, alpha);
//                 model_gcv.set_spatial_locations(loc);

//                 std::vector<double> lambdas;
//                 if(almost_equal(alpha, 0.01)){
//                     lambdas = lambdas_1; 
//                 }  
//                 if(almost_equal(alpha, 0.02)){
//                     lambdas = lambdas_2; 
//                 }
//                 if(almost_equal(alpha, 0.03)){
//                     lambdas = lambdas_3; 
//                 }  
//                 if(almost_equal(alpha, 0.05)){
//                     lambdas = lambdas_5; 
//                 }  
//                 if(almost_equal(alpha, 0.10)){
//                     lambdas = lambdas_10; 
//                 }  
//                 if(almost_equal(alpha, 0.25)){
//                     lambdas = lambdas_25; 
//                 }  
//                 if(almost_equal(alpha, 0.50)){
//                     lambdas = lambdas_50; 
//                 }  
//                 if(almost_equal(alpha, 0.75)){
//                     lambdas = lambdas_75; 
//                 }  
//                 if(almost_equal(alpha, 0.90)){
//                     lambdas = lambdas_90; 
//                 } 
//                 if(almost_equal(alpha, 0.91)){
//                     lambdas = lambdas_91; 
//                 }   
//                 if(almost_equal(alpha, 0.92)){
//                     lambdas = lambdas_92; 
//                 }  
//                 if(almost_equal(alpha, 0.93)){
//                     lambdas = lambdas_93; 
//                 } 
//                 if(almost_equal(alpha, 0.94)){
//                     lambdas = lambdas_94; 
//                 }   
//                 if(almost_equal(alpha, 0.95)){
//                     lambdas = lambdas_95; 
//                 } 
//                 if(almost_equal(alpha, 0.96)){
//                     lambdas = lambdas_96; 
//                 }    
//                 if(almost_equal(alpha, 0.97)){
//                     lambdas = lambdas_97; 
//                 }    
//                 if(almost_equal(alpha, 0.98)){
//                     lambdas = lambdas_98; 
//                 }  
//                 if(almost_equal(alpha, 0.99)){
//                     lambdas = lambdas_99; 
//                 }  

//                 // define lambda sequence as matrix 
//                 DMatrix<double> lambdas_mat;
//                 lambdas_mat.resize(lambdas.size(), 1); 
//                 for(auto i = 0; i < lambdas_mat.rows(); ++i){
//                     lambdas_mat(i,0) = lambdas[i]; 
//                 }

//                 // set model's data
//                 // model_gcv.set_exact_gcv(lambda_selection_type == "gcv"); 

//                 if(lambda_selection_type == "gcv_smooth_eps1e-3"){
//                     model_gcv.set_eps_power(-3.0); 
//                 }
//                 if(lambda_selection_type == "gcv_smooth_eps1e-2"){
//                     model_gcv.set_eps_power(-2.0); 
//                 }
//                 if(lambda_selection_type == "gcv_smooth_eps1e-1.5"){
//                     model_gcv.set_eps_power(-1.5); 
//                 }
//                 if(lambda_selection_type == "gcv_smooth_eps1e-1"){
//                     model_gcv.set_eps_power(-1.0); 
//                 }
                
//                 model_gcv.set_data(df);
//                 model_gcv.init();

//                 // define GCV function and grid of \lambda_D values
//                 auto GCV = model_gcv.gcv<ExactEDF>();
//                 // optimize GCV
//                 Grid<fdapde::Dynamic> opt;
//                 opt.optimize(GCV, lambdas_mat);
                
//                 best_lambda = opt.optimum()(0,0);
        
//                 std::cout << "Best lambda is: " << std::setprecision(16) << best_lambda << std::endl; 

//                 // Save lambda sequence 
//                 std::ofstream fileLambdaS(solutions_path_gcv + "/lambdas_seq_alpha_" + alpha_string + ".csv");
//                 for(std::size_t i = 0; i < lambdas.size(); ++i) 
//                     fileLambdaS << std::setprecision(16) << lambdas[i] << "\n"; 
//                 fileLambdaS.close();

//                 // Save lambda GCVopt for all alphas
//                 std::ofstream fileLambdaoptS(solutions_path_gcv + "/lambdas_opt_alpha_" + alpha_string + ".csv");
//                 if(fileLambdaoptS.is_open()){
//                     fileLambdaoptS << std::setprecision(16) << best_lambda;
//                     fileLambdaoptS.close();
//                 }

//                 // Save GCV 
//                 std::ofstream fileGCV_scores(solutions_path_gcv + "/score_alpha_" + alpha_string + ".csv");
//                 for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
//                     fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.gcvs()[i]) << "\n"; 
//                 fileGCV_scores.close();
//             }

//         }


//     }
// }



// // test OBS RIPETUTE
// //    domain:       unit square
// //    sampling:     locations != nodes
// //    penalization: constant coefficients PDE
// //    covariates:   no
// //    BC:           no
// //    order FE:     1
// TEST(gcv_sqrpde_test_obs_rip, pde_nonparametric_samplingatlocations_spaceonly_gridexact) {

//     bool mean_estimation = true;    // false implies that QSRPDE is run!
//     bool quantile_estimation = !mean_estimation;  

//     bool corr = true; 
//     std::string cor_str = "3";   // ""  "_2" "_3" "_4" "_5" "_6" "_7"

//     std::string norm_loss = "_norm_loss";   // "" "_norm_loss"    // for SRPDE

//     // covariates
//     bool has_covariates = false; 
//     DVector<double> beta_true; 
//     beta_true.resize(2); 
//     beta_true << 1.0, -1.0; 
//     std::cout << "num of covariates = " << beta_true.size() << std::endl; 
//     // nb: nel caso semiparametrico l'RMSE viene calcolato sulle locazioni uniche

//     // path test  
//     std::string R_path; 
//     std::string simulations_string = "sims"; 
//     if(!corr){
//         if(mean_estimation){
//             R_path = "";
//         } else{
//             R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/models/multiple_quantiles/Tests/Test_obs_ripetute";
//         }
         
//     } else{
//         if(mean_estimation){
//             R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/models/srpde/Tests/Test_obs_ripetute_cor" + cor_str;
//         } else{
//             R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/models/multiple_quantiles/Tests/Test_obs_ripetute_cor" + cor_str;
//         }
//     }
//     if(quantile_estimation){
//         if(cor_str == "_7" || cor_str == "_8"){
//             R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/models/multiple_quantiles/Tests/Test_cor" + cor_str; 
//         }
//     }

       
//     // data parameters
//     std::vector<unsigned int> max_reps = {10, 20, 30, 40, 50};   // max number of repetitions 
//     std::vector<std::string> data_types = {"data"};   // ATT: tolto data   
//     for(auto max_rep : max_reps){
//         data_types.push_back("data_rip_" + std::to_string(max_rep));
//     }
//     for(auto d : data_types)
//         std::cout << "data_type=" << d << std::endl; 

//     // pde parameters 
//     std::string diffusion_type = "_lap"; 

//     // quantile parameters 
//     std::vector<double> alphas = {0.5, 0.95};
//     const std::string gcv_refinement = "fine";    // "lasco" "fine"
//     const std::string gcv_summary = "_II_appr";    // "" "_summary_mean" "_II_appr"

//     std::string strategy_gcv; 
//     if(gcv_summary == "_summary_mean")
//         strategy_gcv = "I";  
//     if(gcv_summary == "_II_appr")
//         strategy_gcv = "II";  
//     if(gcv_summary == "")
//         strategy_gcv = "";   
        
//     std::cout << "strategy_gcv=" << strategy_gcv << std::endl; 

//     double step_quantile; 
//     if(gcv_refinement == "lasco"){
//         step_quantile = 0.5;
//     } 
//     if(gcv_refinement == "fine"){
//         step_quantile = 0.1; 
//     } 

//     // model selection parameters
//     std::string smooth_type_mean = "GCV";    
//     std::vector<std::string> smooth_types_quantile = {"GCV_eps1e-1"};   
//     bool compute_rmse = false;
//     bool compute_gcv = true;    

//     const unsigned int n_sim = 10;

//     // define domain
//     std::string domain_str; 
//     MeshLoader<Triangulation<2, 2>> domain("unit_square_25");

//     // define regularizing PDE
//     auto L = -laplacian<FEM>();   
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
//     PDE<Triangulation<2, 2>, decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//     // lambdas sequences 
//     std::vector<double> lambdas_mean;
//     if(norm_loss == "")
//         for(double x = -7.5; x <= -2; x += 0.1) lambdas_mean.push_back(std::pow(10, x));
//     if(norm_loss == "_norm_loss")
//         for(double x = -11.0; x <= 2.0; x += 0.5) lambdas_mean.push_back(std::pow(10, x));

//     std::vector<double> lambdas_quantile;
//     for(double x = -9.5; x <= -1.5; x += step_quantile) lambdas_quantile.push_back(std::pow(10, x));
    
//     bool force_lambdas_longer = false; 
//     std::vector<double> lambdas_longer; 
//     for(double x = -3.0; x <= -1.0; x += step_quantile) lambdas_longer.push_back(std::pow(10, x));

//     double best_lambda; 


//     // Simulations 
//     for(auto sim = 1; sim <= n_sim; ++sim){
//         std::cout << std::endl;
//         std::cout << std::endl;
//         std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl; 

//         for(std::string data_type : data_types){  

//             std::cout << std::endl;    
//             std::cout << "------Data type = " << data_type << std::endl; 

//             std::string gcv_summary_tmp; 
//             std::string strategy_gcv_tmp; 
//             if(data_type == "data"){
//                 gcv_summary_tmp = ""; 
//                 strategy_gcv_tmp = ""; 
//             } else{
//                 gcv_summary_tmp = gcv_summary; 
//                 strategy_gcv_tmp = strategy_gcv; 
//             }

//             // Read locations and data 
//             std::string loc_str; 
//             if(data_type.length() >= 2 && data_type.substr(data_type.length() - 2) != "ta"){
//                 loc_str = "/locs_rip.csv"; 
//             } else{
//                 loc_str = "/locs.csv"; 
//             }
            
//             std::string data_path = R_path + "/" + data_type; 
//             DMatrix<double> loc = read_csv<double>(data_path + loc_str); 
//             std::cout << "locs size = " << loc.rows() << std::endl; 

//             DMatrix<double> y = read_csv<double>(data_path + "/" + simulations_string +  "/sim_" + std::to_string(sim) + "/y.csv");
//             std::cout << "size y in test =" << y.rows() << std::endl;

//             DMatrix<double> X; 
//             if(has_covariates){
//                 std::string str_rip; 
//                 if(data_type.length() >= 2 && data_type.substr(data_type.length() - 2) != "ta"){
//                    str_rip = "_rip"; 
//                 } else{
//                     str_rip = ""; 
//                 }
//                 X = read_csv<double>(data_path + "/X" + str_rip + ".csv");   
//                 std::cout << "dim X=" << X.rows() << ";" << X.cols() << std::endl;      
//             }

//             BlockFrame<double, int> df;
//             df.insert(OBSERVATIONS_BLK, y);
//             if(has_covariates)
//                 df.insert(DESIGN_MATRIX_BLK, X);


//             if(mean_estimation){

//                 std::cout << "-----MEAN REGRESSION-----------------" << std::endl;

//                 // define lambda sequence as matrix 
//                 DMatrix<double> lambdas_mat;
//                 std::vector<double> lambdas = lambdas_mean; 
          
//                 lambdas_mat.resize(lambdas.size(), 1); 
//                 for(auto i = 0; i < lambdas_mat.rows(); ++i){
//                     lambdas_mat(i,0) = lambdas[i]; 
//                 }

//                 if(compute_gcv){
//                     if(smooth_type_mean == "GCV"){

//                         std::cout << "------------------gcv computation-----------------" << std::endl;

//                         std::string gcv_path = data_path + "/" + simulations_string + "/sim_" + std::to_string(sim) + "/mean/" + smooth_type_mean + "/est" + diffusion_type; 
//                         std::cout << "gcv_path=" << gcv_path << std::endl;     

//                         SRPDE model_cv(problem, Sampling::pointwise);
//                         model_cv.set_spatial_locations(loc);
//                         model_cv.gcv_oss_rip_strategy_set(strategy_gcv_tmp); 

//                         model_cv.set_data(df);
//                         model_cv.init();

//                         // define GCV function and grid of \lambda_D values
//                         auto GCV = model_cv.gcv<ExactEDF>();  

//                         // optimize GCV
//                         Grid<fdapde::Dynamic> opt;
//                         opt.optimize(GCV, lambdas_mat);
                        
//                         best_lambda = opt.optimum()(0,0);

//                         // Save GCV score (no radice in SRPDE!)
//                         std::cout << "gcv_path=" << gcv_path << std::endl; 
//                         std::ofstream fileGCV_scores(gcv_path + "/score" + gcv_summary_tmp + norm_loss + ".csv");
//                         for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
//                             fileGCV_scores << std::setprecision(16) << GCV.gcvs()[i] << "\n"; 
//                         fileGCV_scores.close();

//                         // Save GCV edf
//                         std::ofstream fileGCV_edfs(gcv_path + "/edf" + gcv_summary_tmp + norm_loss + ".csv");
//                         for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
//                             fileGCV_edfs << std::setprecision(16) << GCV.edfs()[i] << "\n"; 
//                         fileGCV_edfs.close();
            
//                         std::cout << "Best lambda is: " << std::setprecision(16) << best_lambda << std::endl; 

//                         // Save lambda sequence 
//                         std::ofstream fileLambdaS(gcv_path + "/lambdas_seq" + gcv_summary_tmp + norm_loss + ".csv");
//                         for(std::size_t i = 0; i < lambdas_mat.rows(); ++i) 
//                             fileLambdaS << std::setprecision(16) << lambdas_mat(i,0) << "\n"; 
//                         fileLambdaS.close();

//                         // Save lambda GCVopt for all alphas
//                         std::ofstream fileLambdaoptS(gcv_path + "/lambda_s_opt" + gcv_summary_tmp + norm_loss + ".csv");
//                         if(fileLambdaoptS.is_open()){
//                             fileLambdaoptS << std::setprecision(16) << best_lambda;
//                             fileLambdaoptS.close();
//                         }
                        

//                     }
            
//                 }

//                 if(compute_rmse){
//                     std::cout << "------------------RMSE computation-----" << std::endl; 

//                     std::string rmse_path = data_path + "/" + simulations_string + "/sim_" + std::to_string(sim) + "/mean/RMSE/est" + diffusion_type; 
//                     // RMSE
//                     DMatrix<double> f_true = read_csv<double>(R_path + "/true/mean_true.csv");
//                     DMatrix<double> fn_true = read_csv<double>(R_path + "/true/mean_true_loc.csv");

//                     std::vector<double> rmse_score; 
//                     rmse_score.resize(lambdas_mean.size()); 
//                     double count_l = 0; 
//                     for(auto lambda : lambdas_mean){
//                         SRPDE model_rmse(problem, Sampling::pointwise);
//                         // set model's data
//                         model_rmse.set_spatial_locations(loc);
//                         model_rmse.set_lambda_D(lambda);           
                        
//                         model_rmse.set_data(df);
//                         model_rmse.init();
//                         model_rmse.solve();
                        
//                         DVector<double> sol; 
//                         DVector<double> sol_true; 
//                         if(has_covariates){
//                             sol = collapse_rows(model_rmse.Psi()*model_rmse.f(), model_rmse.unique_locs_flags(), model_rmse.num_unique_locs()) + model_rmse.X_II_approach()*model_rmse.beta(); 
//                             sol_true = fn_true +  model_rmse.X_II_approach()*beta_true;  
//                         } else{
//                             sol = model_rmse.f(); 
//                             sol_true = f_true; 
//                         }
//                         rmse_score[count_l] = RMSE_metric(sol, sol_true); 

//                         count_l = count_l+1; 
//                     }

//                     auto min_idx = std::distance(std::begin(rmse_score), std::min_element(std::begin(rmse_score), std::end(rmse_score))); 
                    
//                     // Save lambda sequence 
//                     std::ofstream fileLambdaS_rmse(rmse_path + "/lambdas_seq" + norm_loss + ".csv");
//                     for(std::size_t i = 0; i < lambdas.size(); ++i) 
//                         fileLambdaS_rmse << std::setprecision(16) << lambdas[i] << "\n"; 
//                     fileLambdaS_rmse.close();

//                     // Save lambda RMSEopt for all alphas
//                     std::ofstream fileLambdaoptS_rmse(rmse_path + "/lambda_s_opt" + norm_loss + ".csv");
//                     if(fileLambdaoptS_rmse.is_open()){
//                         fileLambdaoptS_rmse << std::setprecision(16) << lambdas[min_idx]; ;
//                         fileLambdaoptS_rmse.close();
//                     }

//                     // Save score 
//                     std::ofstream fileRMSE_scores(rmse_path + "/score" + norm_loss + ".csv");
//                     for(std::size_t i = 0; i < rmse_score.size(); ++i) 
//                         fileRMSE_scores << std::setprecision(16) << rmse_score[i] << "\n"; 
//                     fileRMSE_scores.close();
                
//                 }    

//             }


//             if(quantile_estimation){

//                 for(auto alpha : alphas){

//                     unsigned int alpha_int = alpha*100; 
//                     std::string alpha_string = std::to_string(alpha_int); 

//                     std::cout << "------------------alpha=" << alpha_string << "-----------------" << std::endl; 

//                     // define lambda sequence as matrix 
//                     DMatrix<double> lambdas_mat;
//                     std::vector<double> lambdas; 
//                     if(force_lambdas_longer){
//                         lambdas = lambdas_longer; 
//                     } else{
//                         lambdas = lambdas_quantile; 
//                     }
//                     lambdas_mat.resize(lambdas.size(), 1); 
//                     for(auto i = 0; i < lambdas_mat.rows(); ++i){
//                         lambdas_mat(i,0) = lambdas[i]; 
//                     }

//                     if(compute_gcv){
//                         std::cout << "-----GCV computation-----" << std::endl; 
//                         for(auto smooth_type : smooth_types_quantile){

//                             const int eps_power = std::stoi(smooth_type.substr(smooth_type.size() - 2));

//                             std::string gcv_path = data_path + "/" + simulations_string + "/sim_" + std::to_string(sim) + "/quantile/" + smooth_type + "/single_est" + diffusion_type + "/alpha_" + alpha_string; 
                                
//                             QSRPDE<SpaceOnly> model_cv(problem, Sampling::pointwise, alpha);
//                             model_cv.set_spatial_locations(loc);
//                             model_cv.set_eps_power(eps_power); 
                                            
//                             model_cv.set_data(df);
//                             model_cv.gcv_oss_rip_strategy_set(strategy_gcv_tmp); 
//                             model_cv.init();

//                             // define GCV function and grid of \lambda_D values
//                             auto GCV = model_cv.gcv<ExactEDF>();  

//                             // optimize GCV
//                             Grid<fdapde::Dynamic> opt;
//                             opt.optimize(GCV, lambdas_mat);
                            
//                             best_lambda = opt.optimum()(0,0);

//                             // Save GCV score
//                             std::ofstream fileGCV_scores(gcv_path + "/score_" + gcv_refinement + gcv_summary_tmp + ".csv");
//                             for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
//                                 fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.gcvs()[i]) << "\n"; 
//                             fileGCV_scores.close();

//                             // Save GCV edf
//                             std::ofstream fileGCV_edfs(gcv_path + "/edf_" + gcv_refinement + gcv_summary_tmp + ".csv");
//                             for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
//                                 fileGCV_edfs << std::setprecision(16) << GCV.edfs()[i] << "\n"; 
//                             fileGCV_edfs.close();
                
//                             std::cout << "Best lambda is: " << std::setprecision(16) << best_lambda << std::endl; 

//                             // Save lambda sequence 
//                             if(force_lambdas_longer){
//                                 std::cout << "forcing lambda sequence to be longer" << std::endl;
//                                 std::ofstream fileLambdaS(gcv_path + "/lambdas_seq_ext" + gcv_summary + ".csv");
//                                 for(std::size_t i = 0; i < lambdas_mat.rows(); ++i) 
//                                     fileLambdaS << std::setprecision(16) << lambdas_mat(i,0) << "\n"; 
//                                 fileLambdaS.close();

//                                 // Save lambda GCVopt for all alphas
//                                 std::ofstream fileLambdaoptS(gcv_path + "/lambda_s_opt_ext_" + gcv_refinement + gcv_summary + ".csv");
//                                 if(fileLambdaoptS.is_open()){
//                                     fileLambdaoptS << std::setprecision(16) << best_lambda;
//                                     fileLambdaoptS.close();
//                                 }
//                             } else{
//                                 std::ofstream fileLambdaS(gcv_path + "/lambdas_seq_" + gcv_refinement + gcv_summary_tmp + ".csv");
//                                 for(std::size_t i = 0; i < lambdas_mat.rows(); ++i) 
//                                     fileLambdaS << std::setprecision(16) << lambdas_mat(i,0) << "\n"; 
//                                 fileLambdaS.close();

//                                 // Save lambda GCVopt for all alphas
//                                 std::ofstream fileLambdaoptS(gcv_path + "/lambda_s_opt_" + gcv_refinement + gcv_summary_tmp + ".csv");
//                                 if(fileLambdaoptS.is_open()){
//                                     fileLambdaoptS << std::setprecision(16) << best_lambda;
//                                     fileLambdaoptS.close();
//                                 }
//                             }

//                         }
                
//                     }

//                     if(compute_rmse){
//                         std::cout << "-----RMSE computation-----" << std::endl; 

//                         std::string rmse_path = data_path + "/" + simulations_string  + "/sim_" + std::to_string(sim) + "/quantile/RMSE/single_est" + diffusion_type + "/alpha_" + alpha_string; 
//                         // RMSE
//                         DMatrix<double> f_true = read_csv<double>(R_path + "/true/f_true_" + alpha_string + ".csv");
//                         DMatrix<double> fn_true = read_csv<double>(R_path + "/true/fn_true_" + alpha_string + ".csv");

//                         std::vector<double> rmse_score; 
//                         rmse_score.resize(lambdas_quantile.size()); 
//                         double count_l = 0; 
//                         for(auto lambda : lambdas_quantile){
//                             QSRPDE<SpaceOnly> model_rmse(problem, Sampling::pointwise, alpha);
//                             // set model's data
//                             model_rmse.set_spatial_locations(loc);
//                             model_rmse.set_lambda_D(lambda);           
                            
//                             model_rmse.set_data(df);
//                             model_rmse.init();
//                             model_rmse.solve();

//                             DVector<double> sol; 
//                             DVector<double> sol_true; 
//                             if(has_covariates){
//                                 sol = collapse_rows(model_rmse.Psi()*model_rmse.f(), model_rmse.unique_locs_flags(), model_rmse.num_unique_locs()) + model_rmse.X_II_approach()*model_rmse.beta();   
//                                 sol_true = fn_true +  model_rmse.X_II_approach()*beta_true;  
//                             } else{
//                                 sol = model_rmse.f(); 
//                                 sol_true = f_true; 
//                             }

//                             rmse_score[count_l] = RMSE_metric(sol, sol_true); 

//                             count_l = count_l+1; 
//                         }

//                         auto min_idx = std::distance(std::begin(rmse_score), std::min_element(std::begin(rmse_score), std::end(rmse_score))); 
                        
//                         // Save lambda sequence 
//                         std::ofstream fileLambdaS_rmse(rmse_path + "/lambdas_seq_" + gcv_refinement + ".csv");
//                         for(std::size_t i = 0; i < lambdas.size(); ++i) 
//                             fileLambdaS_rmse << std::setprecision(16) << lambdas[i] << "\n"; 
//                         fileLambdaS_rmse.close();

//                         // Save lambda RMSEopt for all alphas
//                         std::ofstream fileLambdaoptS_rmse(rmse_path + "/lambda_s_opt_" + gcv_refinement + ".csv");
//                         if(fileLambdaoptS_rmse.is_open()){
//                             fileLambdaoptS_rmse << std::setprecision(16) << lambdas[min_idx]; ;
//                             fileLambdaoptS_rmse.close();
//                         }

//                         // Save score 
//                         std::ofstream fileRMSE_scores(rmse_path + "/score_" + gcv_refinement + ".csv");
//                         for(std::size_t i = 0; i < rmse_score.size(); ++i) 
//                             fileRMSE_scores << std::setprecision(16) << rmse_score[i] << "\n"; 
//                         fileRMSE_scores.close();
                    
//                     }        
//                 }

//             }


//         }


//     }

// }



// test OBS RIPETUTE NEW
//    domain:       unit square
//    sampling:     locations != nodes
//    penalization: constant coefficients PDE
//    covariates:   no
//    BC:           no
//    order FE:     1
TEST(gcv_sqrpde_test_obs_rip, pde_nonparametric_samplingatlocations_spaceonly_gridexact) {

    bool mean_estimation = false;    // false implies that QSRPDE is run!
    bool quantile_estimation = !mean_estimation;  

    bool corr = false; 
    std::string test_str = "9";   // "4"

    std::string norm_loss = "_norm_loss";   // "" "_norm_loss"    // for SRPDE

    // path test  
    std::string R_path; 
    std::string simulations_string = "sims"; 

    // for stocastich GCV 
    std::size_t seed = 438172;
    const unsigned int MC_run = 100; 

    std::vector<std::string> nxx_vec = {"13", "23", "39"}; 
    const std::string chosen_max_repetion = "10"; 
    const std::string chosen_nxx_loc = "13"; 


    if(mean_estimation){
        // marco 
        R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/models/srpde/Tests/Test_obs_ripetute_" + test_str;

        // ilenia
        // ... 
    } else{
        // marco
        R_path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/models/multiple_quantiles/Tests/Test_rip_" + test_str;
        
        // ilenia
        // ... 
    }
        


    // data parameters
    std::vector<unsigned int> max_reps = {10, 50};   // max number of repetitions 
    std::vector<std::string> data_types = {"data"};   // ATT: tolto data   
    for(auto max_rep : max_reps){
        data_types.push_back("data_rip_" + std::to_string(max_rep));
    }
    for(auto d : data_types)
        std::cout << "data_type=" << d << std::endl; 

    // pde parameters 
    std::string diffusion_type = "_lap"; 

    // quantile parameters 
    std::vector<double> alphas = {0.5, 0.95};

    const std::string gcv_summary = "";    // ""  "_II_appr"

    std::string strategy_gcv; 
    if(gcv_summary == "_II_appr")
        strategy_gcv = "II";  
    if(gcv_summary == "")
        strategy_gcv = "";   
        
    std::cout << "strategy_gcv=" << strategy_gcv << std::endl; 

    // model selection parameters
    std::string smooth_type_mean = "GCV";    
    std::vector<std::string> smooth_types_quantile = {"GCV_eps1e-1"};   
    bool compute_rmse = true;
    bool compute_gcv = true;    

    const unsigned int n_sim = 15;

    // define domain
    std::string domain_str; 
    MeshLoader<Triangulation<2, 2>> domain("unit_square_25");

    // define regularizing PDE
    auto L = -laplacian<FEM>();   
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<Triangulation<2, 2>, decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

    // lambdas sequences 
    std::vector<double> lambdas_mean;
    for(double x = -11.0; x <= 2.0; x += 0.1) lambdas_mean.push_back(std::pow(10, x));

    std::vector<double> lambdas_quantile;
    for(double x = -9.5; x <= -1.5; x += 0.1) lambdas_quantile.push_back(std::pow(10, x));

    double best_lambda; 

    // Simulations 
    for(auto sim = 1; sim <= n_sim; ++sim){
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << "--------------------Simulation #" << std::to_string(sim) << "-------------" << std::endl; 

        for(std::string data_type : data_types){  

            std::cout << std::endl;    
            std::cout << "------Data type = " << data_type << std::endl; 

            std::string gcv_summary_tmp; 
            std::string strategy_gcv_tmp; 
            if(data_type == "data"){
                gcv_summary_tmp = ""; 
                strategy_gcv_tmp = ""; 
            } else{
                gcv_summary_tmp = gcv_summary; 
                strategy_gcv_tmp = strategy_gcv; 
            }

            if(data_type == ("data_rip_" + chosen_max_repetion)){
                for(std::string nxx_loc : nxx_vec){
                    
                    std::string data_path = R_path + "/" + data_type; 
                    DMatrix<double> loc = read_csv<double>(data_path + "/loc_" + nxx_loc + "/loc_" + nxx_loc + ".csv"); 
                    std::cout << "locs size = " << loc.rows() << std::endl; 

                    DMatrix<double> y = read_csv<double>(data_path + "/loc_" + nxx_loc + "/" + simulations_string +  "/sim_" + std::to_string(sim) + "/y.csv");
                    std::cout << "size y in test =" << y.rows() << std::endl;

                    BlockFrame<double, int> df;
                    df.insert(OBSERVATIONS_BLK, y);


                    if(mean_estimation){

                        std::cout << "-----MEAN REGRESSION-----------------" << std::endl;

                        // define lambda sequence as matrix 
                        DMatrix<double> lambdas_mat;
                        std::vector<double> lambdas = lambdas_mean; 
                
                        lambdas_mat.resize(lambdas.size(), 1); 
                        for(auto i = 0; i < lambdas_mat.rows(); ++i){
                            lambdas_mat(i,0) = lambdas[i]; 
                        }

                        if(compute_gcv){
                            if(smooth_type_mean == "GCV"){

                                std::cout << "------------------gcv computation-----------------" << std::endl;

                                std::string gcv_path = data_path + "/loc_" + nxx_loc + "/" + simulations_string + "/sim_" + std::to_string(sim) + "/mean/" + smooth_type_mean + "/est" + diffusion_type; 
                                std::cout << "gcv_path=" << gcv_path << std::endl;     

                                SRPDE model_cv(problem, Sampling::pointwise);
                                model_cv.set_spatial_locations(loc);
                                model_cv.gcv_oss_rip_strategy_set(strategy_gcv_tmp); 

                                model_cv.set_data(df);
                                model_cv.init();

                                // define GCV function and grid of \lambda_D values
                                // std::cout << "Running EXACT GCV" << std::endl; 
                                // auto GCV = model_cv.gcv<ExactEDF>();  


                                std::cout << "Running STOCHASTIC GCV" << std::endl; 
                                auto GCV = model_cv.gcv<StochasticEDF>(MC_run, seed);


                                // optimize GCV
                                Grid<fdapde::Dynamic> opt;
                                opt.optimize(GCV, lambdas_mat);
                                
                                best_lambda = opt.optimum()(0,0);

                                // Save GCV score (no radice in SRPDE!)
                                std::cout << "gcv_path=" << gcv_path << std::endl; 
                                std::ofstream fileGCV_scores(gcv_path + "/score" + gcv_summary_tmp + norm_loss + ".csv");
                                for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
                                    fileGCV_scores << std::setprecision(16) << GCV.gcvs()[i] << "\n"; 
                                fileGCV_scores.close();

                                // Save GCV edf
                                std::ofstream fileGCV_edfs(gcv_path + "/edf" + gcv_summary_tmp + norm_loss + ".csv");
                                for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
                                    fileGCV_edfs << std::setprecision(16) << GCV.edfs()[i] << "\n"; 
                                fileGCV_edfs.close();
                    
                                std::cout << "Best lambda is: " << std::setprecision(16) << best_lambda << std::endl; 

                                // Save lambda sequence 
                                std::ofstream fileLambdaS(gcv_path + "/lambdas_seq" + gcv_summary_tmp + norm_loss + ".csv");
                                for(std::size_t i = 0; i < lambdas_mat.rows(); ++i) 
                                    fileLambdaS << std::setprecision(16) << lambdas_mat(i,0) << "\n"; 
                                fileLambdaS.close();

                                // Save lambda GCVopt for all alphas
                                std::ofstream fileLambdaoptS(gcv_path + "/lambda_s_opt" + gcv_summary_tmp + norm_loss + ".csv");
                                if(fileLambdaoptS.is_open()){
                                    fileLambdaoptS << std::setprecision(16) << best_lambda;
                                    fileLambdaoptS.close();
                                }
                                

                            }
                    
                        }

                        if(compute_rmse){
                            std::cout << "------------------RMSE computation-----" << std::endl; 

                            std::string rmse_path = data_path + "/loc_" + nxx_loc + "/" + simulations_string + "/sim_" + std::to_string(sim) + "/mean/RMSE/est" + diffusion_type; 
                            // RMSE
                            DMatrix<double> f_true = read_csv<double>(R_path + "/true/mean_true.csv");

                            std::vector<double> rmse_score; 
                            rmse_score.resize(lambdas_mean.size()); 
                            double count_l = 0; 
                            for(auto lambda : lambdas_mean){
                                SRPDE model_rmse(problem, Sampling::pointwise);
                                // set model's data
                                model_rmse.set_spatial_locations(loc);
                                model_rmse.set_lambda_D(lambda);           
                                
                                model_rmse.set_data(df);
                                model_rmse.init();
                                model_rmse.solve();
                                
                                DVector<double> sol; 
                                DVector<double> sol_true; 
                                sol = model_rmse.f(); 
                                sol_true = f_true; 
                                
                                rmse_score[count_l] = RMSE_metric(sol, sol_true); 

                                count_l = count_l+1; 
                            }

                            auto min_idx = std::distance(std::begin(rmse_score), std::min_element(std::begin(rmse_score), std::end(rmse_score))); 
                            
                            // Save lambda sequence 
                            std::ofstream fileLambdaS_rmse(rmse_path + "/lambdas_seq" + norm_loss + ".csv");
                            for(std::size_t i = 0; i < lambdas.size(); ++i) 
                                fileLambdaS_rmse << std::setprecision(16) << lambdas[i] << "\n"; 
                            fileLambdaS_rmse.close();

                            // Save lambda RMSEopt for all alphas
                            std::ofstream fileLambdaoptS_rmse(rmse_path + "/lambda_s_opt" + norm_loss + ".csv");
                            if(fileLambdaoptS_rmse.is_open()){
                                fileLambdaoptS_rmse << std::setprecision(16) << lambdas[min_idx]; ;
                                fileLambdaoptS_rmse.close();
                            }

                            // Save score 
                            std::ofstream fileRMSE_scores(rmse_path + "/score" + norm_loss + ".csv");
                            for(std::size_t i = 0; i < rmse_score.size(); ++i) 
                                fileRMSE_scores << std::setprecision(16) << rmse_score[i] << "\n"; 
                            fileRMSE_scores.close();
                        
                        }    

            }


                    if(quantile_estimation){

                        for(auto alpha : alphas){

                            unsigned int alpha_int = alpha*100; 
                            std::string alpha_string = std::to_string(alpha_int); 

                            std::cout << "------------------alpha=" << alpha_string << "-----------------" << std::endl; 

                            // define lambda sequence as matrix 
                            DMatrix<double> lambdas_mat;
                            std::vector<double> lambdas; 
                            lambdas = lambdas_quantile; 
                            
                            lambdas_mat.resize(lambdas.size(), 1); 
                            for(auto i = 0; i < lambdas_mat.rows(); ++i){
                                lambdas_mat(i,0) = lambdas[i]; 
                            }

                            if(compute_gcv){
                                std::cout << "-----GCV computation-----" << std::endl; 
                                for(auto smooth_type : smooth_types_quantile){

                                    const int eps_power = std::stoi(smooth_type.substr(smooth_type.size() - 2));

                                    std::string gcv_path = data_path + "/loc_" + nxx_loc + "/" + simulations_string + "/sim_" + std::to_string(sim) + "/alpha_" + alpha_string + "/" + smooth_type + "/est" + diffusion_type; 
                                        
                                    QSRPDE<SpaceOnly> model_cv(problem, Sampling::pointwise, alpha);
                                    model_cv.set_spatial_locations(loc);
                                    model_cv.set_eps_power(eps_power); 
                                                    
                                    model_cv.set_data(df);
                                    model_cv.gcv_oss_rip_strategy_set(strategy_gcv_tmp); 
                                    model_cv.init();

                                    // define GCV function and grid of \lambda_D values
                                    std::cout << "Running EXACT GCV" << std::endl; 
                                    auto GCV = model_cv.gcv<ExactEDF>();  


                                    // std::cout << "Running STOCHASTIC GCV" << std::endl; 
                                    // auto GCV = model_cv.gcv<StochasticEDF>(MC_run, seed);

                                    // optimize GCV
                                    Grid<fdapde::Dynamic> opt;
                                    opt.optimize(GCV, lambdas_mat);
                                    
                                    best_lambda = opt.optimum()(0,0);

                                    // Save GCV score
                                    std::ofstream fileGCV_scores(gcv_path + "/score" + gcv_summary_tmp + ".csv");
                                    for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
                                        fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.gcvs()[i]) << "\n"; 
                                    fileGCV_scores.close();

                                    // Save GCV edf
                                    std::ofstream fileGCV_edfs(gcv_path + "/edf" + gcv_summary_tmp + ".csv");
                                    for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
                                        fileGCV_edfs << std::setprecision(16) << GCV.edfs()[i] << "\n"; 
                                    fileGCV_edfs.close();
                        
                                    std::cout << "Best lambda is: " << std::setprecision(16) << best_lambda << std::endl; 

                                    // Save lambda sequence 
                                    std::ofstream fileLambdaS(gcv_path + "/lambdas_seq" + gcv_summary_tmp + ".csv");
                                    for(std::size_t i = 0; i < lambdas_mat.rows(); ++i) 
                                        fileLambdaS << std::setprecision(16) << lambdas_mat(i,0) << "\n"; 
                                    fileLambdaS.close();

                                    // Save lambda GCVopt for all alphas
                                    std::ofstream fileLambdaoptS(gcv_path + "/lambda_s_opt" + gcv_summary_tmp + ".csv");
                                    if(fileLambdaoptS.is_open()){
                                        fileLambdaoptS << std::setprecision(16) << best_lambda;
                                        fileLambdaoptS.close();
                                    }
                                    

                                }
                        
                            }

                            if(compute_rmse){
                                std::cout << "-----RMSE computation-----" << std::endl; 

                                std::string rmse_path = data_path + "/loc_" + nxx_loc + "/" + simulations_string  + "/sim_" + std::to_string(sim) + "/alpha_" + alpha_string + "/RMSE/est" + diffusion_type; 
                                // RMSE
                                DMatrix<double> f_true = read_csv<double>(R_path + "/true/f" + alpha_string + "_true.csv");

                                std::vector<double> rmse_score; 
                                rmse_score.resize(lambdas_quantile.size()); 
                                double count_l = 0; 
                                for(auto lambda : lambdas_quantile){
                                    QSRPDE<SpaceOnly> model_rmse(problem, Sampling::pointwise, alpha);
                                    // set model's data
                                    model_rmse.set_spatial_locations(loc);
                                    model_rmse.set_lambda_D(lambda);           
                                    
                                    model_rmse.set_data(df);
                                    model_rmse.init();
                                    model_rmse.solve();

                                    DVector<double> sol; 
                                    DVector<double> sol_true; 

                                    sol = model_rmse.f(); 
                                    sol_true = f_true; 
                                    

                                    rmse_score[count_l] = RMSE_metric(sol, sol_true); 

                                    count_l = count_l+1; 
                                }

                                auto min_idx = std::distance(std::begin(rmse_score), std::min_element(std::begin(rmse_score), std::end(rmse_score))); 
                                
                                // Save lambda sequence 
                                std::ofstream fileLambdaS_rmse(rmse_path + "/lambdas_seq.csv");
                                for(std::size_t i = 0; i < lambdas.size(); ++i) 
                                    fileLambdaS_rmse << std::setprecision(16) << lambdas[i] << "\n"; 
                                fileLambdaS_rmse.close();

                                // Save lambda RMSEopt for all alphas
                                std::ofstream fileLambdaoptS_rmse(rmse_path + "/lambda_s_opt.csv");
                                if(fileLambdaoptS_rmse.is_open()){
                                    fileLambdaoptS_rmse << std::setprecision(16) << lambdas[min_idx]; ;
                                    fileLambdaoptS_rmse.close();
                                }

                                // Save score 
                                std::ofstream fileRMSE_scores(rmse_path + "/score.csv");
                                for(std::size_t i = 0; i < rmse_score.size(); ++i) 
                                    fileRMSE_scores << std::setprecision(16) << rmse_score[i] << "\n"; 
                                fileRMSE_scores.close();
                            
                            }        
                        }

                    }

                }
            } else{
                //std::cout << "in else" << std::endl; 
                std::string nxx_loc = chosen_nxx_loc; 

                std::string data_path = R_path + "/" + data_type; 
                DMatrix<double> loc = read_csv<double>(data_path + "/loc_" + nxx_loc + "/loc_" + nxx_loc + ".csv"); 
                //std::cout << "locs size = " << loc.rows() << std::endl; 

                DMatrix<double> y = read_csv<double>(data_path + "/loc_" + nxx_loc + "/" + simulations_string +  "/sim_" + std::to_string(sim) + "/y.csv");
                //std::cout << "size y in test =" << y.rows() << std::endl;

                BlockFrame<double, int> df;
                df.insert(OBSERVATIONS_BLK, y);


                if(mean_estimation){

                    std::cout << "-----MEAN REGRESSION-----------------" << std::endl;

                    // define lambda sequence as matrix 
                    DMatrix<double> lambdas_mat;
                    std::vector<double> lambdas = lambdas_mean; 
            
                    lambdas_mat.resize(lambdas.size(), 1); 
                    for(auto i = 0; i < lambdas_mat.rows(); ++i){
                        lambdas_mat(i,0) = lambdas[i]; 
                    }

                    if(compute_gcv){
                        if(smooth_type_mean == "GCV"){

                            std::cout << "------------------gcv computation-----------------" << std::endl;

                            std::string gcv_path = data_path + "/loc_" + nxx_loc + "/" + simulations_string + "/sim_" + std::to_string(sim) + "/mean/" + smooth_type_mean + "/est" + diffusion_type; 
                            std::cout << "gcv_path=" << gcv_path << std::endl;     

                            SRPDE model_cv(problem, Sampling::pointwise);
                            model_cv.set_spatial_locations(loc);
                            model_cv.gcv_oss_rip_strategy_set(strategy_gcv_tmp); 

                            model_cv.set_data(df);
                            model_cv.init();

                            // define GCV function and grid of \lambda_D values
                            // std::cout << "Running EXACT GCV" << std::endl; 
                            // auto GCV = model_cv.gcv<ExactEDF>();  


                            std::cout << "Running STOCHASTIC GCV" << std::endl; 
                            auto GCV = model_cv.gcv<StochasticEDF>(MC_run, seed);


                            // optimize GCV
                            Grid<fdapde::Dynamic> opt;
                            opt.optimize(GCV, lambdas_mat);
                            
                            best_lambda = opt.optimum()(0,0);

                            // Save GCV score (no radice in SRPDE!)
                            std::cout << "gcv_path=" << gcv_path << std::endl; 
                            std::ofstream fileGCV_scores(gcv_path + "/score" + gcv_summary_tmp + norm_loss + ".csv");
                            for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
                                fileGCV_scores << std::setprecision(16) << GCV.gcvs()[i] << "\n"; 
                            fileGCV_scores.close();

                            // Save GCV edf
                            std::ofstream fileGCV_edfs(gcv_path + "/edf" + gcv_summary_tmp + norm_loss + ".csv");
                            for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
                                fileGCV_edfs << std::setprecision(16) << GCV.edfs()[i] << "\n"; 
                            fileGCV_edfs.close();
                
                            std::cout << "Best lambda is: " << std::setprecision(16) << best_lambda << std::endl; 

                            // Save lambda sequence 
                            std::ofstream fileLambdaS(gcv_path + "/lambdas_seq" + gcv_summary_tmp + norm_loss + ".csv");
                            for(std::size_t i = 0; i < lambdas_mat.rows(); ++i) 
                                fileLambdaS << std::setprecision(16) << lambdas_mat(i,0) << "\n"; 
                            fileLambdaS.close();

                            // Save lambda GCVopt for all alphas
                            std::ofstream fileLambdaoptS(gcv_path + "/lambda_s_opt" + gcv_summary_tmp + norm_loss + ".csv");
                            if(fileLambdaoptS.is_open()){
                                fileLambdaoptS << std::setprecision(16) << best_lambda;
                                fileLambdaoptS.close();
                            }
                            

                        }
                
                    }

                    if(compute_rmse){
                        std::cout << "------------------RMSE computation-----" << std::endl; 

                        std::string rmse_path = data_path + "/loc_" + nxx_loc + "/" + simulations_string + "/sim_" + std::to_string(sim) + "/mean/RMSE/est" + diffusion_type; 
                        // RMSE
                        DMatrix<double> f_true = read_csv<double>(R_path + "/true/mean_true.csv");

                        std::vector<double> rmse_score; 
                        rmse_score.resize(lambdas_mean.size()); 
                        double count_l = 0; 
                        for(auto lambda : lambdas_mean){
                            SRPDE model_rmse(problem, Sampling::pointwise);
                            // set model's data
                            model_rmse.set_spatial_locations(loc);
                            model_rmse.set_lambda_D(lambda);           
                            
                            model_rmse.set_data(df);
                            model_rmse.init();
                            model_rmse.solve();
                            
                            DVector<double> sol; 
                            DVector<double> sol_true; 
                            sol = model_rmse.f(); 
                            sol_true = f_true; 
                            
                            rmse_score[count_l] = RMSE_metric(sol, sol_true); 

                            count_l = count_l+1; 
                        }

                        auto min_idx = std::distance(std::begin(rmse_score), std::min_element(std::begin(rmse_score), std::end(rmse_score))); 
                        
                        // Save lambda sequence 
                        std::ofstream fileLambdaS_rmse(rmse_path + "/lambdas_seq" + norm_loss + ".csv");
                        for(std::size_t i = 0; i < lambdas.size(); ++i) 
                            fileLambdaS_rmse << std::setprecision(16) << lambdas[i] << "\n"; 
                        fileLambdaS_rmse.close();

                        // Save lambda RMSEopt for all alphas
                        std::ofstream fileLambdaoptS_rmse(rmse_path + "/lambda_s_opt" + norm_loss + ".csv");
                        if(fileLambdaoptS_rmse.is_open()){
                            fileLambdaoptS_rmse << std::setprecision(16) << lambdas[min_idx]; ;
                            fileLambdaoptS_rmse.close();
                        }

                        // Save score 
                        std::ofstream fileRMSE_scores(rmse_path + "/score" + norm_loss + ".csv");
                        for(std::size_t i = 0; i < rmse_score.size(); ++i) 
                            fileRMSE_scores << std::setprecision(16) << rmse_score[i] << "\n"; 
                        fileRMSE_scores.close();
                    
                    }    

                }


                if(quantile_estimation){

                    for(auto alpha : alphas){

                        unsigned int alpha_int = alpha*100; 
                        std::string alpha_string = std::to_string(alpha_int); 

                        std::cout << "------------------alpha=" << alpha_string << "-----------------" << std::endl; 

                        // define lambda sequence as matrix 
                        DMatrix<double> lambdas_mat;
                        std::vector<double> lambdas; 
                        lambdas = lambdas_quantile; 
                        
                        lambdas_mat.resize(lambdas.size(), 1); 
                        for(auto i = 0; i < lambdas_mat.rows(); ++i){
                            lambdas_mat(i,0) = lambdas[i]; 
                        }

                        if(compute_gcv){
                            std::cout << "-----GCV computation-----" << std::endl; 
                            for(auto smooth_type : smooth_types_quantile){

                                const int eps_power = std::stoi(smooth_type.substr(smooth_type.size() - 2));

                                std::string gcv_path = data_path + "/loc_" + nxx_loc + "/" + simulations_string + "/sim_" + std::to_string(sim) + "/alpha_" + alpha_string + "/" + smooth_type + "/est" + diffusion_type; 
                                    
                                QSRPDE<SpaceOnly> model_cv(problem, Sampling::pointwise, alpha);
                                model_cv.set_spatial_locations(loc);
                                model_cv.set_eps_power(eps_power); 
                                                
                                model_cv.set_data(df);
                                model_cv.gcv_oss_rip_strategy_set(strategy_gcv_tmp); 
                                model_cv.init();

                                //define GCV function and grid of \lambda_D values
                                std::cout << "Running EXACT GCV" << std::endl; 
                                auto GCV = model_cv.gcv<ExactEDF>();  


                                // std::cout << "Running STOCHASTIC GCV" << std::endl; 
                                // auto GCV = model_cv.gcv<StochasticEDF>(MC_run, seed);


                                // optimize GCV
                                Grid<fdapde::Dynamic> opt;
                                opt.optimize(GCV, lambdas_mat);
                                
                                best_lambda = opt.optimum()(0,0);

                                // Save GCV score
                                std::ofstream fileGCV_scores(gcv_path + "/score" + gcv_summary_tmp + ".csv");
                                for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
                                    fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.gcvs()[i]) << "\n"; 
                                fileGCV_scores.close();

                                // Save GCV edf
                                std::ofstream fileGCV_edfs(gcv_path + "/edf" + gcv_summary_tmp + ".csv");
                                for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
                                    fileGCV_edfs << std::setprecision(16) << GCV.edfs()[i] << "\n"; 
                                fileGCV_edfs.close();
                    
                                std::cout << "Best lambda is: " << std::setprecision(16) << best_lambda << std::endl; 

                                // Save lambda sequence 
                                std::ofstream fileLambdaS(gcv_path + "/lambdas_seq" + gcv_summary_tmp + ".csv");
                                for(std::size_t i = 0; i < lambdas_mat.rows(); ++i) 
                                    fileLambdaS << std::setprecision(16) << lambdas_mat(i,0) << "\n"; 
                                fileLambdaS.close();

                                // Save lambda GCVopt for all alphas
                                std::ofstream fileLambdaoptS(gcv_path + "/lambda_s_opt" + gcv_summary_tmp + ".csv");
                                if(fileLambdaoptS.is_open()){
                                    fileLambdaoptS << std::setprecision(16) << best_lambda;
                                    fileLambdaoptS.close();
                                }
                                

                            }
                    
                        }

                        if(compute_rmse){
                            std::cout << "-----RMSE computation-----" << std::endl; 

                            std::string rmse_path = data_path + "/loc_" + nxx_loc + "/" + simulations_string  + "/sim_" + std::to_string(sim) + "/alpha_" + alpha_string + "/RMSE/est" + diffusion_type; 
                            // RMSE
                            DMatrix<double> f_true = read_csv<double>(R_path + "/true/f" + alpha_string + "_true.csv");

                            std::vector<double> rmse_score; 
                            rmse_score.resize(lambdas_quantile.size()); 
                            double count_l = 0; 
                            for(auto lambda : lambdas_quantile){
                                QSRPDE<SpaceOnly> model_rmse(problem, Sampling::pointwise, alpha);
                                // set model's data
                                model_rmse.set_spatial_locations(loc);
                                model_rmse.set_lambda_D(lambda);           
                                
                                model_rmse.set_data(df);
                                model_rmse.init();
                                model_rmse.solve();

                                DVector<double> sol; 
                                DVector<double> sol_true; 

                                sol = model_rmse.f(); 
                                sol_true = f_true; 
                                

                                rmse_score[count_l] = RMSE_metric(sol, sol_true); 

                                count_l = count_l+1; 
                            }

                            auto min_idx = std::distance(std::begin(rmse_score), std::min_element(std::begin(rmse_score), std::end(rmse_score))); 
                            
                            // Save lambda sequence 
                            std::ofstream fileLambdaS_rmse(rmse_path + "/lambdas_seq.csv");
                            for(std::size_t i = 0; i < lambdas.size(); ++i) 
                                fileLambdaS_rmse << std::setprecision(16) << lambdas[i] << "\n"; 
                            fileLambdaS_rmse.close();

                            // Save lambda RMSEopt for all alphas
                            std::ofstream fileLambdaoptS_rmse(rmse_path + "/lambda_s_opt.csv");
                            if(fileLambdaoptS_rmse.is_open()){
                                fileLambdaoptS_rmse << std::setprecision(16) << lambdas[min_idx]; ;
                                fileLambdaoptS_rmse.close();
                            }

                            // Save score 
                            std::ofstream fileRMSE_scores(rmse_path + "/score.csv");
                            for(std::size_t i = 0; i < rmse_score.size(); ++i) 
                                fileRMSE_scores << std::setprecision(16) << rmse_score[i] << "\n"; 
                            fileRMSE_scores.close();
                        
                        }        
                    }

                }


            }


        }

    }

}




