#include <fdaPDE/core.h>
#include <gtest/gtest.h>   // testing framework

#include <cstddef>
using fdapde::core::advection;
using fdapde::core::diffusion;
using fdapde::core::laplacian;
using fdapde::core::bilaplacian;
using fdapde::core::fem_order;
using fdapde::core::FEM;
using fdapde::core::Grid; 
using fdapde::core::Mesh; 
using fdapde::core::SPLINE;
using fdapde::core::spline_order;
using fdapde::core::PDE;
using fdapde::core::DiscretizedMatrixField;
using fdapde::core::DiscretizedVectorField;

#include "../../fdaPDE/models/sampling_design.h"
#include "../../fdaPDE/models/regression/srpde.h"
#include "../../fdaPDE/models/regression/qsrpde.h"
#include "../../fdaPDE/models/regression/mqsrpde.h"
using fdapde::models::SRPDE;
using fdapde::models::QSRPDE;
using fdapde::models::MQSRPDE;

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



// // gcv 
// TEST(case_study_mqsrpde_gcv, NO2_restricted) {

//     // const std::string eps_string = "1e-1.5";   // "1e-0.25" "0"  "1e+0" "1e+0.5" "1e+1" "1e+2"
//     // -1.5 usato per una obs

//     const std::string eps_string = "1e-0.5";

//     std::string gcv_type = "stochastic";   // "exact" "stochastic"  ---> MODIFICA ANCHE GIU'!
//     std::size_t seed = 438172;
//     unsigned int MC_run = 100; 
//     const std::string model_type = "nonparametric";  // "nonparametric" "parametric"
//     const std::string pde_type = "";  // "transport" ""
//     const std::string u_string = "1";  // value of u in case of transport
//     const std::string intensity_string = "opt";
//     // double intensity_value = 1;
//     const std::string day_chosen = "11";
//     const std::string num_months  = "one_month";
//     const std::string covariate_type = "radiation_dens.new_log.elev.original";
//     const std::string mesh_type = "convex_hull";  // "square" "esagoni" "convex_hull"
//     const std::string pollutant = "NO2";
//     const bool return_smoothing = false;    // metti exact gcv!! 

//     const std::string str_obs_type = "/obs_ripetute_2"; // "/obs_ripetute" , "/obs_max"
//     const std::string n_max = "2";  // numero di massimi giornalieri estratti come dati 
    

//     std::string est_type = "quantile";    // mean quantile

//     // std::vector<double> alphas = {0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,  
//     //                               0.5, 
//     //                               0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99};
//     std::vector<double> alphas = {0.50, 0.90, 0.95, 0.99};  

//     // // Marco 
//     // std::string path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/case_study/ARPA/Lombardia_restricted"; 
//     // std::string path_data = path + "/data/MQSRPDE/" + pollutant + "/day_" + day_chosen;  
//     // std::string solutions_path; 
  
//     // Ilenia 
//     std::string path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/Thesis_shared/case_study/ARPA/Lombardia"; 
//     std::string path_data = path + "/data/MQSRPDE/" + pollutant + "/day_" + day_chosen; 
//     std::string solutions_path;

//     if(est_type == "mean"){
//         if(model_type == "nonparametric"){
//             if(pde_type == "")
//                 solutions_path = path + "/results/SRPDE/day_" + day_chosen + "/" + num_months + "/" + model_type + "/" + mesh_type + "/" + pollutant;
//             else
//                 solutions_path = path + "/results/SRPDE/day_" + day_chosen + "/" + num_months + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/pde_" + pde_type + "/u_" + u_string;
            
//         } else{
//             if(pde_type == "")
//                 solutions_path = path + "/results/SRPDE/day_" + day_chosen + "/" + num_months + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/" + covariate_type;
//             else
//                 solutions_path = path + "/results/SRPDE/day_" + day_chosen + "/" + num_months + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/" + covariate_type +
//                                 "/pde_" + pde_type + "/u_" + u_string ;
            
//         }
//     }


//     if(est_type == "quantile"){
//         if(model_type == "nonparametric"){
//             if(pde_type == "")
//                 solutions_path = path + "/results/MQSRPDE/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/eps_" + eps_string + "/data_NEW" + str_obs_type;  // ATT CAMBIATO PER I NUOVI DATI
//             else
//                 solutions_path = path + "/results/MQSRPDE/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/eps_" + eps_string + "/pde_" + pde_type + "/u_" + u_string + "/intensity_" + intensity_string;
//         } else{
//             if(pde_type == "")
//                 solutions_path = path + "/results/MQSRPDE/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/" + covariate_type + "/eps_" + eps_string;
//             else
//                 solutions_path = path + "/results/MQSRPDE/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/" + covariate_type + "/eps_" + eps_string +
//                                  "/pde_" + pde_type + "/u_" + u_string + "/intensity_" + intensity_string;
//         }
//     }

//     std::cout << "Sol path: " << solutions_path << std::endl ; 

//     // lambdas sequence 
//     std::vector<DVector<double>>  lambdas; 
//     std::vector<DVector<double>> lambdas_50; 
//     std::vector<DVector<double>>  lambdas_90; 
//     std::vector<DVector<double>> lambdas_95; 
//     std::vector<DVector<double>>  lambdas_99; 

//     // lambdas sequence for fine grid fìof quantiles 
//     std::vector<DVector<double>> lambdas_1_5;
//     std::vector<DVector<double>> lambdas_10_25;
//     std::vector<DVector<double>> lambdas_30_70;
//     std::vector<DVector<double>> lambdas_75_90;
//     std::vector<DVector<double>> lambdas_95_99;

//     if(est_type == "mean"){
//         if(!return_smoothing){
//             for(double xs = -4.0; xs <= +2.0; xs += 0.05)
//                 lambdas.push_back(SVector<1>(std::pow(10,xs)));   

//         } else{
//             double lambda_S;  
//             std::ifstream fileLambdaS_opt(solutions_path + "/lambda_s_opt.csv");
//             if(fileLambdaS_opt.is_open()){
//                 fileLambdaS_opt >> lambda_S; 
//                 fileLambdaS_opt.close();
//             }
//             lambdas.push_back(SVector<1>(lambda_S)); 
//         }
//     }

//     if(return_smoothing && lambdas.size() > 1){
//         std::cout << "ERROR: you want S, but you are providing more lambdas" << std::endl; 
//     } 

//     if(est_type == "quantile"){

//         // // sequenze fatte per la tesi 
//         // for(double x = -6.8; x <= -3.0; x += 0.1) lambdas_1_5.push_back(SVector<1>(std::pow(10, x))); 
//         // for(double x = -6.0; x <= 0.5; x += 0.1) lambdas_10_25.push_back(SVector<1>(std::pow(10, x))); 
//         // for(double x = -5.0; x <= 0.5; x += 0.1) lambdas_30_70.push_back(SVector<1>(std::pow(10, x))); 
//         // for(double x = -6.5; x <= 0.5; x += 0.1) lambdas_75_90.push_back(SVector<1>(std::pow(10, x))); 
//         // for(double x = -6.5; x <= -4.5; x += 0.1) lambdas_95_99.push_back(SVector<1>(std::pow(10, x)));

//         // sequenze per le locs ripetute
//         for(double x = -6.8; x <= -3.0; x += 0.1) lambdas_1_5.push_back(SVector<1>(std::pow(10, x))); 
//         for(double x = -6.0; x <= 0.5; x += 0.1) lambdas_10_25.push_back(SVector<1>(std::pow(10, x))); 
//         for(double x = -10.0; x <= -3.0; x += 0.5) lambdas_30_70.push_back(SVector<1>(std::pow(10, x))); 
//         for(double x = -10.0; x <= -3.0; x += 0.5) lambdas_75_90.push_back(SVector<1>(std::pow(10, x))); 
//         for(double x = -10.0; x <= -3.0; x += 0.5) lambdas_95_99.push_back(SVector<1>(std::pow(10, x)));

//     }

//     // define spatial domain and regularizing PDE
//     MeshLoader<Mesh2D> domain("mesh_lombardia_" + mesh_type);

//     // import data and locs from files
//     DMatrix<double> y; DMatrix<double> space_locs; DMatrix<double> X;  

//     // AGGIUNTO PER IL CONFRONTO obs_max VS obs_ripetute
//     if(str_obs_type == "/obs_max"){
//            y = read_csv<double>(path_data + "/y_rescale_NEW.csv");     // ATT: MESSO NEW PER CONSIDERARE I NUOVI DATI (GIORNI VERI)
//         space_locs = read_csv<double>(path_data + "/locs_NEW.csv");  // ATT: MESSO NEW PER CONSIDERARE I NUOVI DATI (GIORNI VERI)
//     }else{
//         y = read_csv<double>(path_data + "/y_rescale_ripetuto_" + n_max + ".csv");     // ATT: MESSO NEW PER CONSIDERARE I NUOVI DATI (GIORNI VERI)
//         space_locs = read_csv<double>(path_data + "/locs_ripetute_" + n_max + ".csv");    // ATT: MESSO NEW PER CONSIDERARE I NUOVI DATI (GIORNI VERI)
//     }

//     // y = read_csv<double>(path_data + "/y_rescale_NEW.csv");     // ATT: MESSO NEW PER CONSIDERARE I NUOVI DATI (GIORNI VERI)
//     // space_locs = read_csv<double>(path_data + "/locs_NEW.csv");  // ATT: MESSO NEW PER CONSIDERARE I NUOVI DATI (GIORNI VERI)
//     if(model_type == "parametric")
//         X = read_csv<double>(path_data + "/X_" + covariate_type + ".csv");
    
//     // check dimensions
//     std::cout << "dim space loc " << space_locs.rows() << " " << space_locs.cols() << std::endl;
//     std::cout << "dim y " << y.rows() << " " << y.cols() << std::endl;
//     std::cout << "dim X " << X.rows() << " " << X.cols() << std::endl;

//     BlockFrame<double, int> df;
//     df.insert(OBSERVATIONS_BLK, y);
//     if(model_type == "parametric")
//         df.insert(DESIGN_MATRIX_BLK, X);
   
//     // define regularizing PDE in space 
    
//     // Laplacian
//     if(pde_type == "transport")
//         std::cout << "ATT You want to run a model with transport but you are using the Laplacian"; 
//     auto L = -laplacian<FEM>();
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
//     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//     // // save quadrature nodes 
//     // DMatrix<double> quad_nodes = problem.quadrature_nodes();
//     // const static Eigen::IOFormat CSVFormat_quadnodes(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//     // std::ofstream file_quadnodes(path_data + "/quadrature_nodes.csv");
//     // if (file_quadnodes.is_open()){
//     //     file_quadnodes << quad_nodes.format(CSVFormat_quadnodes);
//     //     file_quadnodes.close();
//     // }
    

//     // Laplacian + transport 
//     if(pde_type == "")
//         std::cout << "ATT You want to run a model with only the Laplacian but you are using a PDE with transport"; 
//     // DMatrix<double> u = DMatrix<double>::Ones(domain.mesh.n_elements() * 3, 1); // *0.001;
//     DMatrix<double, Eigen::RowMajor> b_data  = read_csv<double>(path_data + "/b_" + u_string + "_opt.csv");
//     // std::cout << "b dimensions : " << b_data.rows() << " , " << b_data.cols() << std::endl ; 
//     // DMatrix<double> u = read_csv<double>(path_data + "/u_" + u_value + ".csv");
//     DMatrix<double> u = read_csv<double>(path_data + "/u_" + u_string + "_opt.csv");
//     // std::cout << "u dimensions : " << u.rows() << " , " << u.cols() << std::endl ; 
//     DiscretizedVectorField<2, 2> b(b_data);
//     // auto L = -intensity_value*laplacian<FEM>() + advection<FEM>(b);
//     auto L = -laplacian<FEM>() + advection<FEM>(b);
//     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//     std::cout << "-----------------------------GCV STARTS------------------------" << std::endl; 

//     if(est_type == "mean"){

//         SRPDE model(problem, Sampling::pointwise);
        
//         // set model's data
//         model.set_spatial_locations(space_locs);
        
//         model.set_data(df);
//         model.init();

//         // define GCV function and grid of \lambda_D values

//         // stochastic
//         auto GCV = model.gcv<StochasticEDF>(MC_run, seed);
//         if(return_smoothing){
//             std::cout << "ATTENTION: YOU WANT S, BUT STOCHASTIC GCV IS ACTIVATED";
//         }

//         // // exact
//         // auto GCV = model.gcv<ExactEDF>();
//         // if(!return_smoothing){
//         //     std::cout << "ATTENTION: YOU WANT TO RUN GCV, BUT EXACT GCV IS ACTIVATED"; 
//         // }

           
//         // optimize GCV
//         Grid<fdapde::Dynamic> opt;
//         opt.optimize(GCV, lambdas);
//         SVector<1> best_lambda = opt.optimum();

//         if(!return_smoothing){
//             // Save lambda sequence 
//         std::ofstream fileLambda_S_Seq(solutions_path + "/lambdas_S_seq.csv");
//         for(std::size_t i = 0; i < lambdas.size(); ++i) 
//             fileLambda_S_Seq << std::setprecision(16) << lambdas[i] << "\n"; 
//         fileLambda_S_Seq.close();

//         // Save Lambda opt
//         std::ofstream fileLambdaoptS(solutions_path + "/lambda_s_opt.csv");
//         if(fileLambdaoptS.is_open()){
//             fileLambdaoptS << std::setprecision(16) << best_lambda[0];
//             fileLambdaoptS.close();
//         }
//         // Save GCV scores
//         std::ofstream fileGCV_scores(solutions_path + "/gcv_scores.csv");
//         for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
//             fileGCV_scores << std::setprecision(16) << GCV.gcvs()[i] << "\n"; 
//         fileGCV_scores.close();

//         // Save edfs
//         std::ofstream fileEDF(solutions_path + "/edfs.csv");
//         for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
//             fileEDF << std::setprecision(16) << GCV.edfs()[i] << "\n"; 
//         fileEDF.close();

//         }
        

//         if(return_smoothing){
//             // Save S
//             DMatrix<double> computedS = GCV.S_get_gcv();
//             Eigen::saveMarket(computedS, solutions_path + "/S.mtx");
//         }

//     }

//     if(est_type == "quantile"){
        
//         for(auto alpha : alphas){

//             unsigned int alpha_int = alpha*100; 
//             std::string alpha_string = std::to_string(alpha_int); 

//             std::cout << "------------------alpha=" << alpha_string << "-----------------" << std::endl; 

                    
//                 QSRPDE<SpaceOnly> model_gcv(problem, Sampling::pointwise, alpha);
//                 model_gcv.set_spatial_locations(space_locs);

//                 std::vector<DVector<double>> lambdas;
                

//                 if(alpha < 0.06){
//                     lambdas = lambdas_1_5; 
//                 }
//                 if((0.06 < alpha) && (alpha < 0.28)){
//                     lambdas = lambdas_10_25; 
//                 }
//                 if((0.28 < alpha) && (alpha < 0.72)){
//                     lambdas = lambdas_30_70; 
//                 }
//                 if((0.73 < alpha) && (alpha < 0.92)){
//                     lambdas = lambdas_75_90; 
//                 }
//                 if(alpha > 0.93){
//                     lambdas = lambdas_95_99; 
//                 }

//                 // set model's data
//                 if(eps_string == "1e+1"){
//                     model_gcv.set_eps_power(+1.0); 
//                 }
//                 if(eps_string == "1e+0.5"){
//                     model_gcv.set_eps_power(+0.5); 
//                 }
//                 if(eps_string == "1e+0.25"){
//                     model_gcv.set_eps_power(+0.25); 
//                 }
//                 if(eps_string == "1e-0.5"){
//                     model_gcv.set_eps_power(-0.5); 
//                 }
//                 if(eps_string == "1e-1"){
//                     model_gcv.set_eps_power(-1.0); 
//                 }
//                 if(eps_string == "1e-1.5"){
//                     model_gcv.set_eps_power(-1.5); 
//                 }
//                 if(eps_string == "1e-2"){
//                     model_gcv.set_eps_power(-2.0); 
//                 }
//                 if(eps_string == "1e-3"){
//                     model_gcv.set_eps_power(-3.0); 
//                 }
                
//                 model_gcv.set_data(df);
//                 model_gcv.init();

//                 // define GCV function and grid of \lambda_D values
//                 // stochastic
//                 auto GCV = model_gcv.gcv<StochasticEDF>(MC_run, seed);
//                 // optimize GCV
//                 Grid<fdapde::Dynamic> opt;
//                 opt.optimize(GCV, lambdas);
                
//                 double best_lambda = opt.optimum()(0,0);
        
//                 std::cout << "Best lambda is: " << std::setprecision(16) << best_lambda << std::endl; 

//                 // Save lambda sequence 
//                 std::ofstream fileLambdaS(solutions_path + "/lambdas_seq_alpha_" + alpha_string + ".csv");
//                 for(std::size_t i = 0; i < lambdas.size(); ++i) 
//                     fileLambdaS << std::setprecision(16) << lambdas[i] << "\n"; 
//                 fileLambdaS.close();

//                 // Save lambda GCVopt for all alphas
//                 std::ofstream fileLambdaoptS(solutions_path + "/lambdas_opt_alpha_" + alpha_string + ".csv");
//                 if(fileLambdaoptS.is_open()){
//                     fileLambdaoptS << std::setprecision(16) << best_lambda;
//                     fileLambdaoptS.close();
//                 }

//                 // Save GCV 
//                 std::ofstream fileGCV_scores(solutions_path + "/score_alpha_" + alpha_string + ".csv");
//                 for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
//                     fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.gcvs()[i]) << "\n"; 
//                 fileGCV_scores.close();
//             }

//         }


// }


// // run time obs_max vs obs_ripetute
// TEST(case_study_mqsrpde_run, NO2_restricted) {

//     const std::string eps_string = "1e-1.5";   // "0" "1e+0" "1e+1"

//     std::string est_type = "quantile";    // mean quantile
//     bool single_est = false;
//     bool mult_est = true;
//     const std::string model_type = "nonparametric";  // "nonparametric" "parametric"
//     const std::string pde_type = "";   // "transport"
//     const std::string u_string = "1";
//     const std::string intensity_string = "opt";
//     // double intensity_value = 1;
//     const std::string day_chosen = "11";
//     const std::string num_months  = "one_month";
//     const std::string covariate_type = "dens.new_log.elev.original";  //  "dens_log.elev.original";
//     const std::string mesh_type = "convex_hull";  // "square" "esagoni" "convex_hull"
//     const std::string pollutant = "NO2"; 

//     const std::string str_obs_type = "/obs_ripetute_2"; // "/obs_ripetute" , "/obs_max"
//     const std::string n_max = "2";  // numero di massimi giornalieri estratti come dati 

//     // std::vector<double> alphas = {0.5, 0.9, 0.95, 0.99}; // {0.5, 0.90, 0.95}; 
//     // std::vector<double> alphas = {0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,  
//     //                               0.5, 
//     //                               0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99};

//     // std::vector<double> alphas = {0.01, 0.02, 0.05, 0.10, 0.25,  
//     //                               0.5, 
//     //                               0.75, 0.90, 0.95, 0.98, 0.99};

//     std::vector<double> alphas = {0.5, 0.9, 0.95, 0.99};

//     // // Marco 
//     // std::string path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/case_study/ARPA/Lombardia_restricted"; 
//     // std::string path_data = path + "/data/MQSRPDE/" + pollutant + "/day_" + day_chosen ; 
//     // std::string solutions_path; 

//     // Ilenia 
//     std::string path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/Thesis_shared/case_study/ARPA/Lombardia"; 
//     std::string path_data = path + "/data/MQSRPDE/" + pollutant + "/day_" + day_chosen ; 
//     std::string solutions_path;

//     if(est_type == "mean"){
//         if(model_type == "nonparametric"){
//             if(pde_type == "")
//                 solutions_path = path + "/results/SRPDE/day_" + day_chosen + "/" + num_months + "/" + model_type + "/" + mesh_type + "/" + pollutant;
//             else
//                 solutions_path = path + "/results/SRPDE/day_" + day_chosen + "/" + num_months + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/pde_" + pde_type + "/u_" + u_string;
            
//         } else{
//             if(pde_type == "")
//                 solutions_path = path + "/results/SRPDE/day_" + day_chosen + "/" + num_months + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/" + covariate_type;
//             else
//                 solutions_path = path + "/results/SRPDE/day_" + day_chosen + "/" + num_months + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/" + covariate_type +
//                                 "/pde_" + pde_type + "/u_" + u_string;
            
//         }
//     }


//     if(est_type == "quantile"){
//         if(model_type == "nonparametric"){
//             if(pde_type == "")
//                 solutions_path = path + "/results/MQSRPDE/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/eps_" + eps_string + "/data_NEW" + str_obs_type;   // ATT: cambiato per considerare nuovi dati (GIORNI VERI)
//             else
//                 solutions_path = path + "/results/MQSRPDE/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/eps_" + eps_string + "/pde_" + pde_type + "/u_" + u_string + "/intensity_" + intensity_string;
//         } else{
//             if(pde_type == "")
//                 solutions_path = path + "/results/MQSRPDE/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/" + covariate_type + "/eps_" + eps_string ;
//             else
//                 solutions_path = path + "/results/MQSRPDE/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/" + covariate_type + "/eps_" + eps_string +
//                                  "/pde_" + pde_type + "/u_" + u_string + "/intensity_" + intensity_string + "/prova_stesso_lambda";
//         }
//     }

//     std::cout << "Sol path: " << solutions_path << std::endl ; 

//     // define spatial domain and regularizing PDE
//     MeshLoader<Mesh2D> domain("mesh_lombardia_" + mesh_type);

//     // import data and locs from files
//     DMatrix<double> y; DMatrix<double> space_locs; DMatrix<double> X; 

//     // AGGIUNTO PER IL CONFRONTO obs_max VS obs_ripetute
//     if(str_obs_type == "/obs_max"){
//            y = read_csv<double>(path_data + "/y_rescale_NEW.csv");     // ATT: MESSO NEW PER CONSIDERARE I NUOVI DATI (GIORNI VERI)
//         space_locs = read_csv<double>(path_data + "/locs_NEW.csv");  // ATT: MESSO NEW PER CONSIDERARE I NUOVI DATI (GIORNI VERI)
//     }else{
//         y = read_csv<double>(path_data + "/y_rescale_ripetuto_" + n_max + ".csv");     // ATT: MESSO NEW PER CONSIDERARE I NUOVI DATI (GIORNI VERI)
//         space_locs = read_csv<double>(path_data + "/locs_ripetute_" + n_max + ".csv");   // ATT: MESSO NEW PER CONSIDERARE I NUOVI DATI (GIORNI VERI)
//     }

//     // y = read_csv<double>(path_data + "/y_rescale_NEW.csv");     // ATT: cambiato per considerare nuovi dati (GIORNI VERI)
//     // space_locs = read_csv<double>(path_data + "/locs_NEW.csv");  // ATT: cambiato per considerare nuovi dati (GIORNI VERI)


//     if(model_type == "parametric")
//         X = read_csv<double>(path_data + "/X_" + covariate_type + ".csv");
    
//     // check dimensions
//     std::cout << "dim space loc " << space_locs.rows() << " " << space_locs.cols() << std::endl;
//     std::cout << "dim y " << y.rows() << " " << y.cols() << std::endl;
//     std::cout << "dim X " << X.rows() << " " << X.cols() << std::endl;

//     BlockFrame<double, int> df;

//     df.insert(OBSERVATIONS_BLK, y);
//     if(model_type == "parametric")
//         df.insert(DESIGN_MATRIX_BLK, X);
   
//     // define regularizing PDE in space 

//     // Laplacian
//     if(pde_type == "transport")
//         std::cout << "ATT You want to run a model with transport but you are using the Laplacian";
//     auto L = -laplacian<FEM>();
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
//     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//     // // Laplacian + transport 
//     // if(pde_type == "")
//     //     std::cout << "ATT You want to run a model with only the Laplacian but you are using a PDE with transport";
//     // // DMatrix<double> u = DMatrix<double>::Ones(domain.mesh.n_elements() * 3, 1); // *0.001;
//     // DMatrix<double, Eigen::RowMajor> b_data  = read_csv<double>(path_data + "/b_" + u_string + "_opt.csv");
//     // // DMatrix<double> u = read_csv<double>(path_data + "/u_" + u_value + ".csv");
//     // DMatrix<double> u = read_csv<double>(path_data + "/u_" + u_string + "_opt.csv");
//     // DiscretizedVectorField<2, 2> b(b_data);
//     // // auto L = -intensity_value*laplacian<FEM>() + advection<FEM>(b);
//     // auto L = -laplacian<FEM>() + advection<FEM>(b);
//     // PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);


//     std::cout << "--------------------------------RUN STARTS--------------------------------" << std::endl; 
    
//     if(est_type == "mean"){

//         SRPDE model(problem, Sampling::pointwise);
    
//         // set model's data
//         model.set_spatial_locations(space_locs);

//         // Read optima lambdas 
//         double lambda_S; 
//         std::ifstream fileLambdaS_opt(solutions_path + "/lambda_s_opt.csv");
//         if(fileLambdaS_opt.is_open()){
//             fileLambdaS_opt >> lambda_S; 
//             fileLambdaS_opt.close();
//         }

//         std::cout << "lambda S " << lambda_S << std::endl;

//         model.set_lambda_D(lambda_S);
        
//         model.set_data(df);

//         model.init();
//         model.solve();

//         // Save C++ solution 
//         DMatrix<double> computedF = model.f();
//         const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//         std::ofstream filef(solutions_path + "/f.csv");
//         if (filef.is_open()){
//             filef << computedF.format(CSVFormatf);
//             filef.close();
//         }

//         DMatrix<double> computedFn = model.Psi(fdapde::models::not_nan())*model.f();
//         const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//         std::ofstream filefn(solutions_path + "/fn.csv");
//         if (filefn.is_open()){
//             filefn << computedFn.format(CSVFormatfn);
//             filefn.close();
//         }

//         if(model_type == "parametric"){
//             DMatrix<double> computedBeta = model.beta();
//             const static Eigen::IOFormat CSVFormatBeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream filebeta(solutions_path + "/beta.csv");
//             if (filebeta.is_open()){
//                 filebeta << computedBeta.format(CSVFormatBeta);
//                 filebeta.close();
//             }
//         }

//     }

//     if(est_type == "quantile"){

//         if(single_est){
//             std::cout << "-----------------------SINGLE running---------------" << std::endl;

//             std::size_t idx = 0;
//             for(double alpha : alphas){
//                 QSRPDE<SpaceOnly> model(problem, Sampling::pointwise, alpha);

//                 model.set_spatial_locations(space_locs);
//                 unsigned int alpha_int = alphas[idx]*100;  

//                 double lambda ;  
//                 // if((0.24 < alpha) && (alpha < 0.76)){
//                 //     std::cout << "ATT you are using lambda_opt_45 ! " << std::endl;
//                 //     std::ifstream fileLambda(solutions_path + "/lambdas_opt_alpha_45.csv");
//                 //     if(fileLambda.is_open()){
//                 //         fileLambda >> lambda; 
//                 //         fileLambda.close();
//                 //     }
//                 // }else{
//                     std::ifstream fileLambda(solutions_path + "/lambdas_opt_alpha_" + std::to_string(alpha_int) + ".csv");
//                         if(fileLambda.is_open()){
//                             fileLambda >> lambda; 
//                             fileLambda.close();
//                         }                   

//                 // }

//                 // ATT: forza a leggere il lambda selezionato da obs_max (non obs_rip)
//                 // std::string solutions_path_lambda = path + "/results/MQSRPDE/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/eps_" + eps_string + "/data_NEW/obs_max";
//                 // std::string str_lambda_obs_max = "/lambda_obs_max" ; 
//                 // solutions_path = solutions_path + str_lambda_obs_max; 

//                 // std::ifstream fileLambda(solutions_path_lambda + "/lambdas_opt_alpha_" + std::to_string(alpha_int) + ".csv");
//                 //     if(fileLambda.is_open()){
//                 //         fileLambda >> lambda; 
//                 //         fileLambda.close();
//                 //     }                   
//                 // std::cout << "Lambda = " << lambda << std::endl ; 


//                 model.set_lambda_D(lambda);

//                 // set model data
//                 model.set_data(df);

//                 // solve smoothing problem
//                 model.init();
//                 model.solve();

//                 // Save solution
//                 DMatrix<double> computedF = model.f();
//                 const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                 std::ofstream filef(solutions_path + "/f_" + std::to_string(alpha_int) + ".csv");
//                 if(filef.is_open()){
//                     filef << computedF.format(CSVFormatf);
//                     filef.close();
//                 }

//                 DMatrix<double> computedFn = model.Psi()*model.f();
//                 const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                 std::ofstream filefn(solutions_path + "/fn_" + std::to_string(alpha_int) + ".csv");
//                 if(filefn.is_open()){
//                     filefn << computedFn.format(CSVFormatfn);
//                     filefn.close();
//                 }

//                 if(model_type == "parametric"){
//                     DMatrix<double> computedBeta = model.beta(); 
//                     const static Eigen::IOFormat CSVFormatBeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                     std::ofstream filebeta(solutions_path + "/beta_" + std::to_string(alpha_int) + ".csv");
//                     if(filebeta.is_open()){
//                         filebeta << computedBeta.format(CSVFormatBeta);
//                         filebeta.close();
//                     }
//                 }
                

//                 idx++;
//             }

//         }
        
//         if(mult_est){
            
//             MQSRPDE<SpaceOnly> model(problem, Sampling::pointwise, alphas);
//             model.set_spatial_locations(space_locs);
//             model.set_preprocess_option(false); 
//             model.set_forcing_option(false);

//             // use optimal lambda to avoid possible numerical issues
//             DMatrix<double> lambdas;
//             DVector<double> lambdas_temp; 
//             lambdas_temp.resize(alphas.size());

//             for(std::size_t idx = 0; idx < alphas.size(); ++idx){
//                 unsigned int alpha_int = alphas[idx]*100;  
                
//                 // if((0.24 < alphas[idx]) && (alphas[idx] < 0.76)){
//                 //     std::cout << "ATT you are using lambda_opt_45 ! " << std::endl;
//                 //     std::ifstream fileLambdas(solutions_path + "/lambdas_opt_alpha_45.csv");
//                 //     if(fileLambdas.is_open()){
//                 //         fileLambdas >> lambdas_temp(idx); 
//                 //         fileLambdas.close();
//                 //     }
//                 // }else{
//                     std::ifstream fileLambdas(solutions_path + "/lambdas_opt_alpha_" + std::to_string(alpha_int) + ".csv");
//                     if(fileLambdas.is_open()){
//                         fileLambdas >> lambdas_temp(idx); 
//                         fileLambdas.close();
//                     }
//                 // }

//             }
//             lambdas = lambdas_temp;        


//             model.setLambdas_D(lambdas);

//             // set model data
//             model.set_data(df);

//             // solve smoothing problem
//             model.init();
//             model.solve();

//             // Save solution
//             DMatrix<double> computedF = model.f();
//             const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream filef(solutions_path + "/f_all.csv");
//             if(filef.is_open()){
//                 filef << computedF.format(CSVFormatf);
//                 filef.close();
//             }

//             DMatrix<double> computedFn = model.Psi_mult()*model.f();
//             const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//             std::ofstream filefn(solutions_path + "/fn_all.csv");
//             if(filefn.is_open()){
//                 filefn << computedFn.format(CSVFormatfn);
//                 filefn.close();
//             }
            
//             if(model_type == "parametric"){
//                 DMatrix<double> computedBeta = model.beta();
//                 const static Eigen::IOFormat CSVFormatbeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//                 std::ofstream filebeta(solutions_path + "/beta_all.csv");
//                 if(filebeta.is_open()){
//                     filebeta << computedBeta.format(CSVFormatbeta);
//                     filebeta.close();
//                 }
//             }
//         }
        

//     }        
// }




// prova run 
TEST(case_study_mqsrpde_run, NO2_restricted) {

    const std::string month = "gennaio";       // gennaio dicembre 
    const std::string day_chosen = "11"; 

    const std::string eps_string = "1e-1.5";   // "0" "1e+0" "1e+1"

    const std::string pde_type = "transport";  // "" "transport"
    const std::string u_string = "1"; 

    std::string est_type = "quantile";    // mean quantile
    bool single_est = false;
    bool mult_est = true;
    const std::string model_type = "parametric";  // "nonparametric" "parametric"
    const std::string num_months  = "one_month";    
    const std::string covariate_type = "dens_log.elev.original";
    const std::string mesh_type = "convex_hull";  // "square" "esagoni" "convex_hull"
    const std::string pollutant = "NO2"; 

    std::vector<double> alphas = {0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20, 0.25, 
                                  0.30, 0.35, 0.40, 0.45, 0.5, 0.55, 0.60, 0.65, 0.70, 
                                  0.75, 0.80, 0.85, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99};

    // // Marco 
    // std::string path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/case_study/ARPA/Lombardia"; 
    // std::string path_data = path + "/data/MQSRPDE/" + pollutant + "/" + month + "/day_" + day_chosen;  
    // std::string solutions_path; 
  
    // Ilenia 
    std::string path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/Thesis_shared/case_study/ARPA/Lombardia"; 
    std::string path_data = path + "/data/MQSRPDE/" + pollutant + "/" + month + "/day_" + day_chosen;  
    std::string solutions_path; 

    // std::cout << "ATT DATA PATH SOVRASCRITTO" << std::endl; 
    // path_data = path + "/data/MQSRPDE/" + pollutant + "/mix"; 

    if(est_type == "mean"){
        if(model_type == "nonparametric"){
            solutions_path = path + "/results/SRPDE/" + month + "/day_" + day_chosen + "/" + num_months + "/" + model_type + "/" + mesh_type + "/" + pollutant;
        } else{
            solutions_path = path + "/results/SRPDE/" + month + "/day_" + day_chosen + "/" + num_months + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/" + covariate_type;
        }

       solutions_path = solutions_path + "/pde_" + pde_type + "/u_" + u_string + "/intensity_opt"; 
    }

    if(est_type == "quantile"){
        if(model_type == "nonparametric"){
            solutions_path = path + "/results/MQSRPDE/" + month + "/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/eps_" + eps_string;
        } else{
            solutions_path = path + "/results/MQSRPDE/" + month + "/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/" + covariate_type + "/eps_" + eps_string;
        }

        solutions_path = solutions_path + "/pde_" + pde_type + "/u_" + u_string + "/intensity_opt"; 
    }

    std::cout << "path: " << solutions_path << std::endl; 


    // define spatial domain and regularizing PDE
    MeshLoader<Mesh2D> domain("mesh_lombardia_" + mesh_type);

    // import data and locs from files
    DMatrix<double> y; DMatrix<double> space_locs; DMatrix<double> X; 


    y = read_csv<double>(path_data + "/y_rescale.csv"); 
    space_locs = read_csv<double>(path_data + "/locs.csv");
    if(model_type == "parametric")
        X = read_csv<double>(path_data + "/X_" + covariate_type + ".csv");
    
    // check dimensions
    std::cout << "dim space loc " << space_locs.rows() << " " << space_locs.cols() << std::endl;
    std::cout << "dim y " << y.rows() << " " << y.cols() << std::endl;
    std::cout << "dim X " << X.rows() << " " << X.cols() << std::endl;
    std::cout << "sum X " << X.sum() << std::endl;

    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    if(model_type == "parametric")
        df.insert(DESIGN_MATRIX_BLK, X);


    // // Laplacian
    // if(pde_type == "transport")
    //     std::cout << "ATT You want to run a model with transport but you are using the Laplacian";
    // auto L = -laplacian<FEM>();
    // DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    // PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
   
    // Laplacian + transport 
    if(pde_type == "")
        std::cout << "ATT You want to run a model with only the Laplacian but you are using a PDE with transport"; 
    // DMatrix<double> u = DMatrix<double>::Ones(domain.mesh.n_elements() * 3, 1); // *0.001;
    DMatrix<double, Eigen::RowMajor> b_data  = read_csv<double>(path_data + "/b_" + u_string + "_opt.csv");
    // std::cout << "b dimensions : " << b_data.rows() << " , " << b_data.cols() << std::endl ; 
    // DMatrix<double> u = read_csv<double>(path_data + "/u_" + u_value + ".csv");
    DMatrix<double> u = read_csv<double>(path_data + "/u_" + u_string + "_opt.csv");
    // std::cout << "u dimensions : " << u.rows() << " , " << u.cols() << std::endl ; 
    DiscretizedVectorField<2, 2> b(b_data);
    // auto L = -intensity_value*laplacian<FEM>() + advection<FEM>(b);
    auto L = -laplacian<FEM>() + advection<FEM>(b);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);


    std::cout << "--------------------------------RUN STARTS--------------------------------" << std::endl; 
    if(est_type == "mean"){

        SRPDE model(problem, Sampling::pointwise);
    
        // set model's data
        model.set_spatial_locations(space_locs);

        // Read optima lambdas 
        double lambda_S; 
        std::ifstream fileLambdaS_opt(solutions_path + "/lambda_s_opt.csv");
        if(fileLambdaS_opt.is_open()){
            fileLambdaS_opt >> lambda_S; 
            fileLambdaS_opt.close();
        }

        std::cout << "lambda S" << lambda_S << std::endl;

        model.set_lambda_D(lambda_S);
        
        model.set_data(df);

        model.init();
        model.solve();

        // Save C++ solution 
        DMatrix<double> computedF = model.f();
        const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        std::ofstream filef(solutions_path + "/f.csv");
        if (filef.is_open()){
            filef << computedF.format(CSVFormatf);
            filef.close();
        }

        DMatrix<double> computedFn = model.Psi(fdapde::models::not_nan())*model.f();
        const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        std::ofstream filefn(solutions_path + "/fn.csv");
        if (filefn.is_open()){
            filefn << computedFn.format(CSVFormatfn);
            filefn.close();
        }

        if(model_type == "parametric"){
            DMatrix<double> computedBeta = model.beta();
            const static Eigen::IOFormat CSVFormatBeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
            std::ofstream filebeta(solutions_path + "/beta.csv");
            if (filebeta.is_open()){
                filebeta << computedBeta.format(CSVFormatBeta);
                filebeta.close();
            }
        }

    }

    

    if(est_type == "quantile"){

        if(single_est){
            std::cout << "-----------------------SINGLE running---------------" << std::endl;

            std::size_t idx = 0;
            for(double alpha : alphas){
                QSRPDE<SpaceOnly> model(problem, Sampling::pointwise, alpha);
                model.set_spatial_locations(space_locs);
                unsigned int alpha_int = alphas[idx]*100;  
                double lambda; 
                std::ifstream fileLambda(solutions_path + "/lambdas_opt_alpha_" + std::to_string(alpha_int) + ".csv");
                if(fileLambda.is_open()){
                    fileLambda >> lambda; 
                    fileLambda.close();
                }
                model.set_lambda_D(lambda);

                // set model data
                model.set_data(df);

                // solve smoothing problem
                model.init();
                model.solve();

                // Save solution
                DMatrix<double> computedF = model.f();
                const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
                std::ofstream filef(solutions_path + "/f_" + std::to_string(alpha_int) + ".csv");
                if(filef.is_open()){
                    filef << computedF.format(CSVFormatf);
                    filef.close();
                }

                // DMatrix<double> computedG = model.g();
                // const static Eigen::IOFormat CSVFormatg(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
                // std::ofstream fileg(solutions_path + "/g_" + std::to_string(alpha_int) + ".csv");
                // if(fileg.is_open()){
                //     fileg << computedG.format(CSVFormatg);
                //     fileg.close();
                // }

                DMatrix<double> computedFn = model.Psi()*model.f();
                const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
                std::ofstream filefn(solutions_path + "/fn_" + std::to_string(alpha_int) + ".csv");
                if(filefn.is_open()){
                    filefn << computedFn.format(CSVFormatfn);
                    filefn.close();
                }

                if(model_type == "parametric"){
                    DMatrix<double> computedBeta = model.beta(); 
                    const static Eigen::IOFormat CSVFormatBeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
                    std::ofstream filebeta(solutions_path + "/beta_" + std::to_string(alpha_int) + ".csv");
                    if(filebeta.is_open()){
                        filebeta << computedBeta.format(CSVFormatBeta);
                        filebeta.close();
                    }
                }
                

                idx++;
            }

    }
        
    if(mult_est){

        std::cout << "-----------------------MULTIPLE running---------------" << std::endl;
        
        MQSRPDE<SpaceOnly> model(problem, Sampling::pointwise, alphas);
        model.set_spatial_locations(space_locs);
        model.set_preprocess_option(false); 
        model.set_forcing_option(false);

        // use optimal lambda to avoid possible numerical issues
        DMatrix<double> lambdas;
        DVector<double> lambdas_temp; 
        lambdas_temp.resize(alphas.size());
        for(std::size_t idx = 0; idx < alphas.size(); ++idx){
            unsigned int alpha_int = alphas[idx]*100;  
            std::ifstream fileLambdas(solutions_path + "/lambdas_opt_alpha_" + std::to_string(alpha_int) + ".csv");
            if(fileLambdas.is_open()){
                fileLambdas >> lambdas_temp(idx); 
                fileLambdas.close();
            }

            std::cout << "Lambda = " << lambdas_temp(idx) << std::endl ; 
        }
        lambdas = lambdas_temp;                
        model.setLambdas_D(lambdas);

        // set model data
        model.set_data(df);

        // solve smoothing problem
        std::cout << "main 1" << std::endl; 
        model.init();
        std::cout << "main 2" << std::endl;
        model.solve();
        std::cout << "main 3" << std::endl;

        // // Save solution
        // DMatrix<double> computedF = model.f();
        // const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        // std::ofstream filef(solutions_path + "/f_all.csv");
        // if(filef.is_open()){
        //     filef << computedF.format(CSVFormatf);
        //     filef.close();
        // }

        // DMatrix<double> computedFn = model.Psi_mult()*model.f();
        // const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        // std::ofstream filefn(solutions_path + "/fn_all.csv");
        // if(filefn.is_open()){
        //     filefn << computedFn.format(CSVFormatfn);
        //     filefn.close();
        // }
        
        // if(model_type == "parametric"){
        //     DMatrix<double> computedBeta = model.beta();
        //     const static Eigen::IOFormat CSVFormatbeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        //     std::ofstream filebeta(solutions_path + "/beta_all.csv");
        //     if(filebeta.is_open()){
        //         filebeta << computedBeta.format(CSVFormatbeta);
        //         filebeta.close();
        //     }
        // }
    }
        

    }        
}





// TEST(gcv_srpde_test, ldensity_spaceonly_gridstochastic){
//     const std::string mesh_type = "esagoni";
//     const std::string log = "_log";

//     // define spatial domain and regularizing PDE
//     std::string path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/Thesis_shared/case_study/ARPA/Lombardia_restricted/covariates_nodes/density/SRPDE"; 
//     std::string path_data = path + "/data" ; 
//     std::string solutions_path = path ; 

//     std::cout << "Sol path: " << solutions_path << std::endl ; 

//     MeshLoader<Mesh2D> domain("mesh_lombardia_" + mesh_type);

//     // import data and locs from files
//     DMatrix<double> y; DMatrix<double> space_locs; 

//     y = read_csv<double>(path_data + "/y" + log + ".csv");    
//     space_locs = read_csv<double>(path_data + "/locs.csv");

//     std::cout << "Dim y = " << y.rows() << " , " << y.cols() << std::endl ; 
//     std::cout << "Dim locs = " << space_locs.rows() << " , " << space_locs.cols() << std::endl ; 

//     // define regularizing PDE
//     auto L = -laplacian<FEM>();
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
//     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
//     // define model
//     SRPDE model(problem, Sampling::pointwise);

//     // set model's data
    
//     model.set_spatial_locations(space_locs);

//     BlockFrame<double, int> df;
//     df.insert(OBSERVATIONS_BLK, y);
//     model.set_data(df);
//     model.init();

//     // define GCV function and grid of \lambda_D values
//     std::size_t seed = 476813;
//     auto GCV = model.gcv<StochasticEDF>(100, seed);
//     std::vector<DVector<double>> lambdas;
//     for (double x = -6.0; x <= -1.0; x += 0.1) lambdas.push_back(SVector<1>(std::pow(10, x)));

//     // optimize GCV
//     Grid<fdapde::Dynamic> opt;
//     opt.optimize(GCV, lambdas);

//     SVector<1> best_lambda = opt.optimum();


//     // Save lambda sequence 
//     std::ofstream fileLambda_S_Seq(solutions_path + "/lambdas_S_seq" + log + ".csv");
//     for(std::size_t i = 0; i < lambdas.size(); ++i) 
//         fileLambda_S_Seq << std::setprecision(16) << lambdas[i] << "\n"; 
//     fileLambda_S_Seq.close();

//     // Save Lambda opt
//     std::ofstream fileLambdaoptS(solutions_path + "/lambda_s_opt"+ log + ".csv");
//     if(fileLambdaoptS.is_open()){
//         fileLambdaoptS << std::setprecision(16) << best_lambda[0];
//         fileLambdaoptS.close();
//     }
    

//     // Save GCV scores
//     std::ofstream fileGCV_scores(solutions_path + "/gcv_scores" + log + ".csv");
//     for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
//         fileGCV_scores << std::setprecision(16) << GCV.gcvs()[i] << "\n"; 
//     fileGCV_scores.close();

//     // Save edfs
//     std::ofstream fileEDF(solutions_path + "/edfs"+ log +".csv");
//     for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
//         fileEDF << std::setprecision(16) << GCV.edfs()[i] << "\n"; 
//     fileEDF.close();

// }



// TEST(run_srpde_test, density_spaceonly_gridstochastic){
//     const std::string mesh_type = "esagoni";
//     const std::string log = "_log";

//     // define spatial domain and regularizing PDE
//     std::string path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/Thesis_shared/case_study/ARPA/Lombardia_restricted/covariates_nodes/density/SRPDE"; 
//     std::string path_data = path + "/data" ; 
//     std::string solutions_path = path ; 

//     std::cout << "Sol path: " << solutions_path << std::endl ; 

//     MeshLoader<Mesh2D> domain("mesh_lombardia_" + mesh_type);

//     // import data and locs from files
//     DMatrix<double> y; DMatrix<double> space_locs; 

//     y = read_csv<double>(path_data + "/y"+ log +".csv");    
//     space_locs = read_csv<double>(path_data + "/locs.csv");

//     std::cout << "Dim y = " << y.rows() << " , " << y.cols() << std::endl ; 
//     std::cout << "Dim locs = " << space_locs.rows() << " , " << space_locs.cols() << std::endl ; 

//     // define regularizing PDE
//     auto L = -laplacian<FEM>();
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
//     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
//     // define model
//     SRPDE model(problem, Sampling::pointwise);

//     // set model's data
//     model.set_spatial_locations(space_locs);

//     BlockFrame<double, int> df;
//     df.insert(OBSERVATIONS_BLK, y);

//     // Read optima lambdas 
//     double lambda_S; 
//     std::ifstream fileLambdaS_opt(solutions_path + "/lambda_s_opt.csv");
//     if(fileLambdaS_opt.is_open()){
//         fileLambdaS_opt >> lambda_S; 
//         fileLambdaS_opt.close();
//     }

//     std::cout << "lambda S " << lambda_S << std::endl;

//     model.set_lambda_D(lambda_S);

//     model.set_data(df);
//     model.init();
//     model.solve();

//     // Save C++ solution 
//     DMatrix<double> computedF = model.f();
//     const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//     std::ofstream filef(solutions_path + "/f"+ log + ".csv");
//     if (filef.is_open()){
//         filef << computedF.format(CSVFormatf);
//         filef.close();
//     }

//     DMatrix<double> computedFn = model.Psi(fdapde::models::not_nan())*model.f();
//     const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
//     std::ofstream filefn(solutions_path + "/fn"+ log + ".csv");
//     if (filefn.is_open()){
//         filefn << computedFn.format(CSVFormatfn);
//         filefn.close();
//     }

// }
 
    
