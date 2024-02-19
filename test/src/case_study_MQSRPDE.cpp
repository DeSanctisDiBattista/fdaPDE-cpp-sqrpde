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

//     const std::string eps_string = "1e-2";   // "1e-0.25" "0"  "1e+0" "1e+0.5" "1e+1" "1e+2"

//     std::string gcv_type = "stochastic";   // "exact" "stochastic"  ---> MODIFICA ANCHE GIU'!
//     std::size_t seed = 438172;
//     unsigned int MC_run = 100; 
//     const std::string model_type = "nonparametric";  // "nonparametric" "parametric"
//     const std::string day_chosen = "11";
//     const std::string covariate_type = "radiation_dens";
//     const std::string mesh_type = "convex_hull";  // "square" "esagoni" "convex_hull"
//     const std::string pollutant = "NO2";
//     const bool return_smoothing = false;    // metti exact gcv!! 

//     std::string est_type = "quantile";    // mean quantile
//     // std::vector<double> alphas = {0.5, 0.9, 0.95, 0.99};
//     std::vector<double> alphas = {0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,  
//                                   0.5, 
//                                   0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99};
//     // std::vector<double> alphas = {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,  
//     //                               0.5, 
//     //                               0.55, 0.60, 0.65, 0.70};  

//     // // Marco 
//     // std::string path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/case_study/ARPA/Lombardia_restricted"; 
//     // std::string path_data = path + "/data/MQSRPDE/" + pollutant + "/day_" + day_chosen;  
//     // std::string solutions_path; 
//     // 16.53
  
//     // Ilenia 
//     std::string path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/Thesis_shared/case_study/ARPA/Lombardia_restricted"; 
//     std::string path_data = path + "/data/MQSRPDE/" + pollutant + "/day_" + day_chosen; 
//     std::string solutions_path;

//     if(est_type == "mean"){
//         if(model_type == "nonparametric"){
//             solutions_path = path + "/results/SRPDE/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + pollutant;
//         } else{
//             solutions_path = path + "/results/SRPDE/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/" + covariate_type;
//         }
//     }

//     if(est_type == "quantile"){
//         if(model_type == "nonparametric"){
//             solutions_path = path + "/results/MQSRPDE/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/eps_" + eps_string;
//         } else{
//             solutions_path = path + "/results/MQSRPDE/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/" + covariate_type + "/eps_" + eps_string;
//         }
//     }

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

//     // if(est_type == "mean"){
//     //     if(!return_smoothing){
//     //         for(double xs = -3.0; xs <= +1.0; xs += 0.1)
//     //             lambdas.push_back(std::pow(10,xs));   

//     //     } else{
//     //         double lambda_S; double lambda_T;  
//     //         std::ifstream fileLambdaS_opt(solutions_path + "/lambda_s_opt.csv");
//     //         if(fileLambdaS_opt.is_open()){
//     //             fileLambdaS_opt >> lambda_S; 
//     //             fileLambdaS_opt.close();
//     //         }
//     //         lambdas_d.push_back(std::pow(10,lambda_S)); 

//     //         std::ifstream fileLambdaT_opt(solutions_path + "/lambda_t_opt.csv");
//     //         if(fileLambdaT_opt.is_open()){
//     //             fileLambdaT_opt >> lambda_T; 
//     //             fileLambdaT_opt.close();
//     //         }
//     //         lambdas_t.push_back(std::pow(10,lambda_T)); 
//     //     }


//     //     for(auto i = 0; i < lambdas_d.size(); ++i)
//     //         for(auto j = 0; j < lambdas_t.size(); ++j) 
//     //             lambdas_d_t.push_back(SVector<2>(lambdas_d[i], lambdas_t[j]));
//     // }

//     if(return_smoothing && lambdas.size() > 1){
//         std::cout << "ERROR: you want S, but you are providing more lambdas" << std::endl; 
//     } 

//     if(est_type == "quantile"){

//         // for(double x = -5.5; x <= -2.5; x += 0.1) lambdas_50.push_back(SVector<1>(std::pow(10, x))); 
//         // for(double x = -6.5; x <= -4.5; x += 0.1) lambdas_90.push_back(SVector<1>(std::pow(10, x))); 
//         // for(double x = -6.5; x <= -4.5; x += 0.1) lambdas_95.push_back(SVector<1>(std::pow(10, x)));
//         // for(double x = -7.0; x <= -5.0; x += 0.1) lambdas_99.push_back(SVector<1>(std::pow(10, x)));

//         for(double x = -6.8; x <= -3.0; x += 0.1) lambdas_1_5.push_back(SVector<1>(std::pow(10, x))); 
//         for(double x = -6.0; x <= -2.0; x += 0.1) lambdas_10_25.push_back(SVector<1>(std::pow(10, x)));
//         for(double x = -5.0; x <= -2.0; x += 0.1) lambdas_30_70.push_back(SVector<1>(std::pow(10, x))); 
//         for(double x = -6.5; x <= -4.0; x += 0.1) lambdas_75_90.push_back(SVector<1>(std::pow(10, x))); 
//         for(double x = -6.5; x <= -4.5; x += 0.1) lambdas_95_99.push_back(SVector<1>(std::pow(10, x)));

//     }

//     // define spatial domain and regularizing PDE
//     MeshLoader<Mesh2D> domain("mesh_lombardia_" + mesh_type);

//     // import data and locs from files
//     DMatrix<double> y; DMatrix<double> space_locs; DMatrix<double> X;  

//     y = read_csv<double>(path_data + "/y_rescale.csv");     // ATT
//     space_locs = read_csv<double>(path_data + "/locs.csv");
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
   
//     // define regularizing PDE 
//     auto L = -laplacian<FEM>();
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
//     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

//     std::cout << "-----------------------------GCV STARTS------------------------" << std::endl; 

//     // if(est_type == "mean"){

//     //     STRPDE<SpaceTimeSeparable, fdapde::monolithic> model(space_penalty, time_penalty, Sampling::pointwise);
        
//     //     // set model's data
//     //     model.set_spatial_locations(space_locs);
//     //     model.set_temporal_locations(time_locs);
        
//     //     model.set_data(df);
//     //     model.init();

//     //     // define GCV function and grid of \lambda_D values

//     //     // // stochastic
//     //     // auto GCV = model.gcv<StochasticEDF>(MC_run, seed);
//     //     // exact
//     //     auto GCV = model.gcv<ExactEDF>();

           
//     //     // optimize GCV
//     //     Grid<fdapde::Dynamic> opt;
//     //     opt.optimize(GCV, lambdas_d_t);
//     //     SVector<2> best_lambda = opt.optimum();

//     //     if(!return_smoothing){
//     //         // Save lambda sequence 
//     //     std::ofstream fileLambda_S_Seq(solutions_path + "/lambdas_S_seq.csv");
//     //     for(std::size_t i = 0; i < lambdas_d.size(); ++i) 
//     //         fileLambda_S_Seq << std::setprecision(16) << lambdas_d[i] << "\n"; 
//     //     fileLambda_S_Seq.close();

//     //     std::ofstream fileLambda_T_Seq(solutions_path + "/lambdas_T_seq.csv");
//     //     for(std::size_t i = 0; i < lambdas_t.size(); ++i) 
//     //         fileLambda_T_Seq << std::setprecision(16) << lambdas_t[i] << "\n"; 
//     //     fileLambda_T_Seq.close();

//     //     // Save Lambda opt
//     //     std::ofstream fileLambdaoptS(solutions_path + "/lambda_s_opt.csv");
//     //     if(fileLambdaoptS.is_open()){
//     //         fileLambdaoptS << std::setprecision(16) << best_lambda[0];
//     //         fileLambdaoptS.close();
//     //     }
//     //     std::ofstream fileLambdaoptT(solutions_path + "/lambda_t_opt.csv");
//     //     if (fileLambdaoptT.is_open()){
//     //         fileLambdaoptT << std::setprecision(16) << best_lambda[1];
//     //         fileLambdaoptT.close();
//     //     }
//     //     // Save GCV scores
//     //     std::ofstream fileGCV_scores(solutions_path + "/gcv_scores.csv");
//     //     for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
//     //         fileGCV_scores << std::setprecision(16) << GCV.gcvs()[i] << "\n"; 
//     //     fileGCV_scores.close();

//     //     // Save edfs
//     //     std::ofstream fileEDF(solutions_path + "/edfs.csv");
//     //     for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
//     //         fileEDF << std::setprecision(16) << GCV.edfs()[i] << "\n"; 
//     //     fileEDF.close();

//     //     }
        

//     //     if(return_smoothing){
//     //         // Save S
//     //         DMatrix<double> computedS = GCV.S_get_gcv();
//     //         Eigen::saveMarket(computedS, solutions_path + "/S.mtx");
//     //     }

        


//     // }

//     if(est_type == "quantile"){
        
//         for(auto alpha : alphas){

//             unsigned int alpha_int = alpha*100; 
//             std::string alpha_string = std::to_string(alpha_int); 

//             std::cout << "------------------alpha=" << alpha_string << "-----------------" << std::endl; 

                    
//                 QSRPDE<SpaceOnly> model_gcv(problem, Sampling::pointwise, alpha);
//                 model_gcv.set_spatial_locations(space_locs);

//                 std::vector<DVector<double>> lambdas;
                
//                 // if(almost_equal(alpha, 0.50)){
//                 //     lambdas = lambdas_50; 
//                 // } 
//                 // if(almost_equal(alpha, 0.90)){
//                 //     lambdas = lambdas_90; 
//                 // }
//                 // if(almost_equal(alpha, 0.95)){
//                 //     lambdas = lambdas_95; 
//                 // } 
//                 // if(almost_equal(alpha, 0.99)){
//                 //     lambdas = lambdas_99; 
//                 // }

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
//                     std::cout << "ciao sono qui" << std::endl ; 
//                     lambdas = lambdas_75_90; 
//                 }
//                 if(alpha > 0.93){
//                     lambdas = lambdas_95_99; 
//                 }

//                 // set model's data

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


// run time
TEST(case_study_mqsrpde_run, NO2_restricted) {

    const std::string eps_string = "1e-2";   // "0" "1e+0" "1e+1"

    std::string est_type = "quantile";    // mean quantile
    bool single_est = false;
    bool mult_est = true;
    const std::string model_type = "nonparametric";  // "nonparametric" "parametric"
    const std::string day_chosen = "11";
    const std::string covariate_type = "radiation_dens";
    const std::string mesh_type = "convex_hull";  // "square" "esagoni" "convex_hull"
    const std::string pollutant = "NO2"; 

    // // std::vector<double> alphas = {0.5, 0.9, 0.95, 0.99}; // {0.5, 0.90, 0.95}; 
    // std::vector<double> alphas = {0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,  
    //                               0.5, 
    //                               0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99};

    // std::vector<double> alphas = {0.01, 0.02, 0.05, 0.10, 0.25,  
    //                               0.5, 
    //                               0.75, 0.90, 0.95, 0.98, 0.99};

    std::vector<double> alphas = {0.5, 0.75, 0.90, 0.95, 0.98, 0.99};

    // Marco 
    std::string path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/case_study/ARPA/Lombardia_restricted"; 
    std::string path_data = path + "/data/MQSRPDE/" + pollutant + "/day_" + day_chosen ; 
    std::string solutions_path; 

    // // Ilenia 
    // std::string path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/Thesis_shared/case_study/ARPA/Lombardia_restricted"; 
    // std::string path_data = path + "/data/MQSRPDE/" + pollutant + "/day_" + day_chosen ; 
    // std::string solutions_path;

    if(est_type == "mean"){
        if(model_type == "nonparametric"){
            solutions_path = path + "/results/SRPDE/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + pollutant;
        } else{
            solutions_path = path + "/results/SRPDE/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/" + covariate_type;
        }
    }
        
    if(est_type == "quantile"){
        if(model_type == "nonparametric"){
            solutions_path = path + "/results/MQSRPDE/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/eps_" + eps_string;
        } else{
            solutions_path = path + "/results/MQSRPDE/day_" + day_chosen + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/" + covariate_type + "/eps_" + eps_string;
        }
    }

    // define spatial domain and regularizing PDE
    MeshLoader<Mesh2D> domain("mesh_lombardia_" + mesh_type);

    // import data and locs from files
    DMatrix<double> y; DMatrix<double> space_locs; DMatrix<double> X; 

    y = read_csv<double>(path_data + "/y_rescale.csv");     // ATT
    space_locs = read_csv<double>(path_data + "/locs.csv");
    if(model_type == "parametric")
        X = read_csv<double>(path_data + "/X_" + covariate_type + ".csv");
    
    // check dimensions
    std::cout << "dim space loc " << space_locs.rows() << " " << space_locs.cols() << std::endl;
    std::cout << "dim y " << y.rows() << " " << y.cols() << std::endl;
    std::cout << "dim X " << X.rows() << " " << X.cols() << std::endl;

    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    if(model_type == "parametric")
        df.insert(DESIGN_MATRIX_BLK, X);
   
    // define regularizing PDE in space 
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);


    std::cout << "--------------------------------RUN STARTS--------------------------------" << std::endl; 
    // if(est_type == "mean"){

    //     STRPDE<SpaceTimeSeparable, fdapde::monolithic> model(space_penalty, time_penalty, Sampling::pointwise);
    
    //     // set model's data
    //     model.set_spatial_locations(space_locs);
    //     model.set_temporal_locations(time_locs);

    //     // Read optima lambdas 
    //     double lambda_T; double lambda_S; 
    //     std::ifstream fileLambdaT_opt(solutions_path + "/lambda_t_opt.csv");
    //     if(fileLambdaT_opt.is_open()){
    //         fileLambdaT_opt >> lambda_T; 
    //         fileLambdaT_opt.close();
    //     }
    //     std::ifstream fileLambdaS_opt(solutions_path + "/lambda_s_opt.csv");
    //     if(fileLambdaS_opt.is_open()){
    //         fileLambdaS_opt >> lambda_S; 
    //         fileLambdaS_opt.close();
    //     }

    //     std::cout << "lambda S " << lambda_S << std::endl;
    //     std::cout << "lambda T " << lambda_T << std::endl;

    //     model.set_lambda_D(lambda_S);
    //     model.set_lambda_T(lambda_T);
        
    //     model.set_data(df);

    //     model.init();
    //     model.solve();

    //     // Save C++ solution 
    //     DMatrix<double> computedF = model.f();
    //     const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
    //     std::ofstream filef(solutions_path + "/f.csv");
    //     if (filef.is_open()){
    //         filef << computedF.format(CSVFormatf);
    //         filef.close();
    //     }

    //     DMatrix<double> computedFn = model.Psi(not_nan())*model.f();
    //     const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
    //     std::ofstream filefn(solutions_path + "/fn.csv");
    //     if (filefn.is_open()){
    //         filefn << computedFn.format(CSVFormatfn);
    //         filefn.close();
    //     }

    //     if(model_type == "parametric"){
    //         DMatrix<double> computedBeta = model.beta();
    //         const static Eigen::IOFormat CSVFormatBeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
    //         std::ofstream filebeta(solutions_path + "/beta.csv");
    //         if (filebeta.is_open()){
    //             filebeta << computedBeta.format(CSVFormatBeta);
    //             filebeta.close();
    //         }
    //     }

    // }

    

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
            }
            lambdas = lambdas_temp;                
            model.setLambdas_D(lambdas);

            // set model data
            model.set_data(df);

            // solve smoothing problem
            model.init();
            model.solve();

            // Save solution
            DMatrix<double> computedF = model.f();
            const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
            std::ofstream filef(solutions_path + "/f_all.csv");
            if(filef.is_open()){
                filef << computedF.format(CSVFormatf);
                filef.close();
            }

            DMatrix<double> computedFn = model.Psi_mult()*model.f();
            const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
            std::ofstream filefn(solutions_path + "/fn_all.csv");
            if(filefn.is_open()){
                filefn << computedFn.format(CSVFormatfn);
                filefn.close();
            }
            
            if(model_type == "parametric"){
                DMatrix<double> computedBeta = model.beta();
                const static Eigen::IOFormat CSVFormatbeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
                std::ofstream filebeta(solutions_path + "/beta_all.csv");
                if(filebeta.is_open()){
                    filebeta << computedBeta.format(CSVFormatbeta);
                    filebeta.close();
                }
            }
        }
        

    }        
}




