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
using fdapde::core::Triangulation;
using fdapde::core::SPLINE;
using fdapde::core::spline_order;
using fdapde::core::PDE;
using fdapde::core::DiscretizedMatrixField;
using fdapde::core::DiscretizedVectorField;

#include "../../fdaPDE/models/sampling_design.h"
#include "../../fdaPDE/models/regression/strpde.h"
#include "../../fdaPDE/models/regression/srpde.h"
#include "../../fdaPDE/models/regression/qsrpde.h"
using fdapde::models::STRPDE;
using fdapde::models::SRPDE;
using fdapde::models::QSRPDE;
using fdapde::models::SpaceTimeSeparable;
using fdapde::models::SpaceTimeParabolic;

#include "../../fdaPDE/models/regression/gcv.h"
using fdapde::models::ExactEDF;
using fdapde::models::GCV;
using fdapde::models::StochasticEDF;
using fdapde::models::Sampling;
using fdapde::models::not_nan;

#include "utils/constants.h"
#include "utils/mesh_loader.h"
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_mtx;
using fdapde::testing::read_csv;


// gcv time
TEST(case_study_gcv, laplacian_nonparametric_samplingatlocations_timelocations_separable_monolithic_missingdata) {

    const std::string eps_string = "1e-1";   // "1e-0.25" "0"  "1e+0" "1e+0.5" "1e+1" "1e+2"

    std::string gcv_type = "stochastic";   // "exact" "stochastic"  ---> MODIFICA ANCHE GIU'!
    std::size_t seed = 438172;
    unsigned int MC_run = 100; 
    const std::string num_months = "one_month";   // "one_month" "two_months"   
    const std::string model_type = "parametric";  // "nonparametric" "parametric"
    const std::string pde_type = "";        // "transport"
    const std::string u_string = "1";               // value of u in case of transport
    const std::string covariate_type = "dens_log.elev.original";
    // const std::vector<std::string> covariate_type_vec = {"dens"};
    const std::string mesh_type = "convex_hull";  // "square" "esagoni" "convex_hull"
    const std::string pollutant = "NO2";
    const bool return_smoothing = false ;    // metti exact gcv!! 

    std::string est_type = "quantile";    // mean quantile
    std::vector<double> alphas = {0.99}; //, 0.9, 0.95}; 

    // Marco 
    std::string path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/case_study/ARPA/Lombardia"; 
    std::string path_data = path + "/data/" + num_months + "/" + pollutant;  
    std::string solutions_path; 

    // // Ilenia 
    // std::string path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/Thesis_shared/case_study/ARPA/Lombardia"; 
    // std::string path_data = path + "/data/" + num_months + "/" + pollutant; 
    // std::string solutions_path;

    // for(std::string covariate_type : covariate_type_vec){

    if(est_type == "mean"){
        if(model_type == "nonparametric"){
            solutions_path = path + "/results/STRPDE/" + num_months + "/" + model_type + "/" + mesh_type + "/" + pollutant;
        } else{
            solutions_path = path + "/results/STRPDE/" + num_months + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/" + covariate_type;
        }
    }

    if(est_type == "quantile"){
        if(model_type == "nonparametric"){
            if(pde_type == ""){
                solutions_path = path + "/results/QSTRPDE/" + num_months + "/" + model_type + "/" + mesh_type + "/" + pollutant;
            }else{
                solutions_path = path + "/results/QSTRPDE/" + num_months + "/" + model_type + "/" + mesh_type + "/" + pollutant +
                                 "/pde_" + pde_type + "/u_" + u_string ;
            }
            
        } else{
            if(pde_type == ""){
                solutions_path = path + "/results/QSTRPDE/" + num_months + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/" + covariate_type;
            }else{
                solutions_path = path + "/results/QSTRPDE/" + num_months + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/" + covariate_type +
                                 "/pde_" + pde_type + "/u_" + u_string ;
            }
        }
    }

    std::cout << "Sol path = " << solutions_path << std::endl; 

    // lambdas sequence 
    std::vector<DVector<double>> lambdas_d_t; std::vector<double> lambdas_d; std::vector<double> lambdas_t;
    std::vector<double> lambdas50_d; std::vector<double> lambdas50_t; std::vector<DVector<double>> lambdas50_d_t; 
    std::vector<double> lambdas90_d; std::vector<double> lambdas90_t; std::vector<DVector<double>> lambdas90_d_t;
    std::vector<double> lambdas10_d; std::vector<double> lambdas10_t; std::vector<DVector<double>> lambdas10_d_t; 
    std::vector<double> lambdas95_d; std::vector<double> lambdas95_t; std::vector<DVector<double>> lambdas95_d_t; 
    std::vector<double> lambdas99_d; std::vector<double> lambdas99_t; std::vector<DVector<double>> lambdas99_d_t; 

    if(est_type == "mean"){
        if(!return_smoothing){
            for(double xs = -3.0; xs <= -1.0; xs += 0.1)
                lambdas_d.push_back(std::pow(10,xs));
            for(double xt = -3.0; xt <= 1.0; xt += 2.0)
                lambdas_t.push_back(std::pow(10,xt));    

        } else{
            double lambda_S; double lambda_T;  
            std::ifstream fileLambdaS_opt(solutions_path + "/lambda_s_opt.csv");
            if(fileLambdaS_opt.is_open()){
                fileLambdaS_opt >> lambda_S; 
                fileLambdaS_opt.close();
            }
            lambdas_d.push_back(lambda_S); 

            std::ifstream fileLambdaT_opt(solutions_path + "/lambda_t_opt.csv");
            if(fileLambdaT_opt.is_open()){
                fileLambdaT_opt >> lambda_T; 
                fileLambdaT_opt.close();
            }
            lambdas_t.push_back(lambda_T); 
        }


        for(auto i = 0; i < lambdas_d.size(); ++i)
            for(auto j = 0; j < lambdas_t.size(); ++j) 
                lambdas_d_t.push_back(SVector<2>(lambdas_d[i], lambdas_t[j]));
    }

    // define lambda sequence as matrix 
    DMatrix<double> lambdas_mat;
    lambdas_mat.resize(lambdas_d_t.size(), 2); 
    for(auto i = 0; i < lambdas_mat.rows(); ++i){
        lambdas_mat(i,0) = lambdas_d_t[0][i]; 
        lambdas_mat(i,1) = lambdas_d_t[1][i]; 
    }



    if(return_smoothing && lambdas_d.size() > 1){
        std::cout << "ERROR: you want S, but you are providing more lambdas" << std::endl; 
    } 

    if(est_type == "quantile"){

        // 50% 
        {
            for(double xs = -6.0; xs <= -4.9; xs += 0.1)
                lambdas50_d.push_back(std::pow(10,xs));

            for(double xt = -3.0; xt <= +1.0; xt += 2.0)  // con lambdaT=1e-3 la U è in (-6, -5)
                lambdas50_t.push_back(std::pow(10,xt));    

            for(auto i = 0; i < lambdas50_d.size(); ++i)
                for(auto j = 0; j < lambdas50_t.size(); ++j) 
                    lambdas50_d_t.push_back(SVector<2>(lambdas50_d[i], lambdas50_t[j]));
        }

        // 90% 
        {
            for(double xs = -8.5; xs <= -5.5; xs += 0.1)  // -6 
                lambdas90_d.push_back(std::pow(10,xs));   

            for(double xt = -3.0; xt <= -1.0; xt += 2.0) // -1
                lambdas90_t.push_back(std::pow(10,xt));    

            for(auto i = 0; i < lambdas90_d.size(); ++i)
                for(auto j = 0; j < lambdas90_t.size(); ++j) 
                    lambdas90_d_t.push_back(SVector<2>(lambdas90_d[i], lambdas90_t[j]));
        }

        // 10% 
        {
            for(double xs = -8.5; xs <= -6.0; xs += 0.1)  
                lambdas10_d.push_back(std::pow(10,xs));   

            for(double xt = -3.0; xt <= -2.0; xt += 2.0) 
                lambdas10_t.push_back(std::pow(10,xt));    

            for(auto i = 0; i < lambdas10_d.size(); ++i)
                for(auto j = 0; j < lambdas10_t.size(); ++j) 
                    lambdas10_d_t.push_back(SVector<2>(lambdas10_d[i], lambdas10_t[j]));
        }

        // 95% 
        {
            for(double xs = -9.0; xs <= -5.0; xs += 0.05)
                lambdas95_d.push_back(std::pow(10,xs));

            for(double xt = -3.0; xt <= -2.0; xt += 2.0)
                lambdas95_t.push_back(std::pow(10,xt));    

            for(auto i = 0; i < lambdas95_d.size(); ++i)
                for(auto j = 0; j < lambdas95_t.size(); ++j) 
                    lambdas95_d_t.push_back(SVector<2>(lambdas95_d[i], lambdas95_t[j]));
        }

        // 99% 
        {
            for(double xs = -8.5; xs <= -6.0; xs += 0.1)    // -8.0; xs <= -5.9; xs += 0.1)  
                lambdas99_d.push_back(std::pow(10,xs));

            for(double xt = -3.0; xt <= -2.0; xt += 2.0)
                lambdas99_t.push_back(std::pow(10,xt));    

            for(auto i = 0; i < lambdas99_d.size(); ++i)
                for(auto j = 0; j < lambdas99_t.size(); ++j) 
                    lambdas99_d_t.push_back(SVector<2>(lambdas99_d[i], lambdas99_t[j]));
        }

    }

    // define lambda sequence as matrix 
    // 10%
    DMatrix<double> lambdas10_mat;
    lambdas10_mat.resize(lambdas10_d_t.size(), 2); 
    for(auto i = 0; i < lambdas10_mat.rows(); ++i){
        lambdas10_mat(i,0) = lambdas10_d_t[i][0]; ;
        lambdas10_mat(i,1) = lambdas10_d_t[i][1]; 
    }
    // 50%
    DMatrix<double> lambdas50_mat;
    lambdas50_mat.resize(lambdas50_d_t.size(), 2); 
    for(auto i = 0; i < lambdas50_mat.rows(); ++i){
        lambdas50_mat(i,0) = lambdas50_d_t[i][0]; 
        lambdas50_mat(i,1) = lambdas50_d_t[i][1]; 
    }
    // 90%
    DMatrix<double> lambdas90_mat;
    lambdas90_mat.resize(lambdas90_d_t.size(), 2); 
    for(auto i = 0; i < lambdas90_mat.rows(); ++i){
        lambdas90_mat(i,0) = lambdas90_d_t[i][0]; 
        lambdas90_mat(i,1) = lambdas90_d_t[i][1]; 
    }
    // 95%
    DMatrix<double> lambdas95_mat;
    lambdas95_mat.resize(lambdas95_d_t.size(), 2); 
    for(auto i = 0; i < lambdas95_mat.rows(); ++i){
        lambdas95_mat(i,0) = lambdas95_d_t[i][0]; 
        lambdas95_mat(i,1) = lambdas95_d_t[i][1]; 
    }
    // 99%
    DMatrix<double> lambdas99_mat;
    lambdas99_mat.resize(lambdas99_d_t.size(), 2); 
    for(auto i = 0; i < lambdas99_mat.rows(); ++i){
        lambdas99_mat(i,0) = lambdas99_d_t[i][0]; 
        lambdas99_mat(i,1) = lambdas99_d_t[i][1]; 
    }

    // define temporal domain
    unsigned int M; double tf; 
    if(num_months == "two_months"){
        M = 30; 
        tf = 61.0; 
    }
    if(num_months == "one_month"){
        M = 11; 
        tf = 22.0; 
    }
    std::string M_string = std::to_string(M);
    Triangulation<1, 1> time_mesh(0, tf, M-1);
    // define spatial domain and regularizing PDE
    MeshLoader<Triangulation<2, 2>> domain("mesh_lombardia_" + mesh_type);

    // import data and locs from files
    DMatrix<double> y; DMatrix<double> space_locs; DMatrix<double> time_locs; DMatrix<double> X;  

    y = read_csv<double>(path_data + "/y_rescale_NEW.csv");     // ATT MESSO NEW PER VEDERE IL PERIODO NUOVO (VERO !!)
    time_locs = read_csv<double>(path_data + "/time_locs.csv"); 
    space_locs = read_csv<double>(path_data + "/locs.csv");
    if(model_type == "parametric")
        X = read_csv<double>(path_data + "/X_" + covariate_type + ".csv");
    
    // check dimensions
    std::cout << "dim space loc " << space_locs.rows() << " " << space_locs.cols() << std::endl;
    std::cout << "dim time loc " << time_locs.rows() << " " << time_locs.cols() << std::endl;
    std::cout << "dim y " << y.rows() << " " << y.cols() << std::endl;
    std::cout << "dim X " << X.rows() << " " << X.cols() << std::endl;

    BlockFrame<double, int> df;
    df.stack(OBSERVATIONS_BLK, y);
    if(model_type == "parametric")
        df.insert(DESIGN_MATRIX_BLK, X);
   
    // define regularizing PDE in space 

    // Laplacian
    if(pde_type == "transport")
        std::cout << "ATT You want to run a model with transport but you are using the Laplacian"; 

    auto Ld = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3 * time_mesh.n_nodes(), 1);
    PDE<Triangulation<2, 2>, decltype(Ld), DMatrix<double>, FEM, fem_order<1>> space_penalty(domain.mesh, Ld, u);

    // // Laplacian + transport 
    // if(pde_type == "")
    //     std::cout << "ATT You want to run a model with only the Laplacian but you are using a PDE with transport"; 

    // DMatrix<double, Eigen::RowMajor> b_data  = read_csv<double>(path_data + "/transport/b_" + u_string + "_opt.csv");  
    
    // DMatrix<double> u = read_csv<double>(path_data + "/transport/u_" + u_string + "_opt.csv");
  
    // //DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3 * time_mesh.n_nodes(), 1);

    // DiscretizedVectorField<2, 2> b(b_data);
    // auto Ld = -laplacian<FEM>() + advection<FEM>(b);
    // PDE<Triangulation<2, 2>, decltype(Ld), DMatrix<double>, FEM, fem_order<1>> space_penalty(domain.mesh, Ld, u);

    // define regularizing PDE in time
    auto Lt = -bilaplacian<SPLINE>();
    PDE<Triangulation<1, 1>, decltype(Lt), DMatrix<double>, SPLINE, spline_order<3>> time_penalty(time_mesh, Lt);

    std::cout << "-----------------------------GCV STARTS------------------------" << std::endl; 

    if(est_type == "mean"){

        STRPDE<SpaceTimeSeparable, fdapde::monolithic> model(space_penalty, time_penalty, Sampling::pointwise);
        
        // set model's data
        model.set_spatial_locations(space_locs);
        model.set_temporal_locations(time_locs);
        
        model.set_data(df);
        model.init();

        // define GCV function and grid of \lambda_D values

        // stochastic
        auto GCV = model.gcv<StochasticEDF>(MC_run, seed);
        // // exact
        // auto GCV = model.gcv<ExactEDF>();

           
        // optimize GCV
        Grid<fdapde::Dynamic> opt;
        opt.optimize(GCV, lambdas_mat);
        SVector<2> best_lambda = opt.optimum();

        if(!return_smoothing){
            // Save lambda sequence 
        std::ofstream fileLambda_S_Seq(solutions_path + "/lambdas_S_seq.csv");
        for(std::size_t i = 0; i < lambdas_d.size(); ++i) 
            fileLambda_S_Seq << std::setprecision(16) << lambdas_d[i] << "\n"; 
        fileLambda_S_Seq.close();

        std::ofstream fileLambda_T_Seq(solutions_path + "/lambdas_T_seq.csv");
        for(std::size_t i = 0; i < lambdas_t.size(); ++i) 
            fileLambda_T_Seq << std::setprecision(16) << lambdas_t[i] << "\n"; 
        fileLambda_T_Seq.close();

        // Save Lambda opt
        std::ofstream fileLambdaoptS(solutions_path + "/lambda_s_opt.csv");
        if(fileLambdaoptS.is_open()){
            fileLambdaoptS << std::setprecision(16) << best_lambda[0];
            fileLambdaoptS.close();
        }
        std::ofstream fileLambdaoptT(solutions_path + "/lambda_t_opt.csv");
        if (fileLambdaoptT.is_open()){
            fileLambdaoptT << std::setprecision(16) << best_lambda[1];
            fileLambdaoptT.close();
        }
        // Save GCV scores
        std::ofstream fileGCV_scores(solutions_path + "/gcv_scores.csv");
        for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
            fileGCV_scores << std::setprecision(16) << GCV.gcvs()[i] << "\n"; 
        fileGCV_scores.close();

        // Save edfs
        std::ofstream fileEDF(solutions_path + "/edfs.csv");
        for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
            fileEDF << std::setprecision(16) << GCV.edfs()[i] << "\n"; 
        fileEDF.close();

        }
        

        if(return_smoothing){
            // Save S
            DMatrix<double> computedS = GCV.S_get_gcv();
            Eigen::saveMarket(computedS, solutions_path + "/S.mtx");
        }

        


    }

    if(est_type == "quantile"){
        
        for(double alpha : alphas){

            unsigned int alpha_int = alpha*100; 
            std::string alpha_string = std::to_string(alpha_int); 

            std::cout << "----------------ALPHA = " << alpha_string << "----------------" << std::endl; 

            QSRPDE<SpaceTimeSeparable> model(space_penalty, time_penalty, Sampling::pointwise, alpha);
    
            // set model's data
            model.set_spatial_locations(space_locs);
            model.set_temporal_locations(time_locs);

            if(eps_string == "1e+2"){
                model.set_eps_power(2.0); 
            }
            if(eps_string == "1e+1"){
                model.set_eps_power(1.0); 
            }
            if(eps_string == "1e+0.75"){
                model.set_eps_power(0.75); 
            }
            if(eps_string == "1e+0.5"){
                model.set_eps_power(0.5); 
            }
            if(eps_string == "1e+0"){
                model.set_eps_power(0.0); 
            }
            if(eps_string == "1e-1"){
                model.set_eps_power(-1.0); 
            }
            if(eps_string == "1e-0.5"){
                model.set_eps_power(-0.5); 
            }
            if(eps_string == "1e-0.25"){
                model.set_eps_power(-0.25); 
            }
            if(eps_string == "1e-1.5"){
                model.set_eps_power(-1.5); 
            }
            if(eps_string == "1e-2"){
                model.set_eps_power(-2.0); 
            }
            if(eps_string == "1e-3"){
                model.set_eps_power(-3.0); 
            }
            
            model.set_data(df);
            model.init();

            // define GCV function and grid of \lambda_D values

            // stochastic
            auto GCV = model.gcv<StochasticEDF>(MC_run, seed);
            // // exact
            // auto GCV = model.gcv<ExactEDF>();

            DMatrix<double> lambdas_mat_quantile; 

            if(alpha_string == "10"){
                lambdas_d = lambdas10_d; 
                lambdas_t = lambdas10_t;
                lambdas_d_t = lambdas10_d_t;
                lambdas_mat_quantile = lambdas10_mat; 
            }

            if(alpha_string == "50"){
                lambdas_d = lambdas50_d; 
                lambdas_t = lambdas50_t;
                lambdas_d_t = lambdas50_d_t;
                lambdas_mat_quantile = lambdas50_mat;
            }
                 
            if(alpha_string == "90"){
                lambdas_d = lambdas90_d; 
                lambdas_t = lambdas90_t;
                lambdas_d_t = lambdas90_d_t;
                lambdas_mat_quantile = lambdas90_mat;
            }
                 
            if(alpha_string == "95"){
                lambdas_d = lambdas95_d; 
                lambdas_t = lambdas95_t;
                lambdas_d_t = lambdas95_d_t;
                lambdas_mat_quantile = lambdas95_mat;
            }

            if(alpha_string == "99"){
                lambdas_d = lambdas99_d; 
                lambdas_t = lambdas99_t;
                lambdas_d_t = lambdas99_d_t;
                lambdas_mat_quantile = lambdas99_mat;
            }


            // optimize GCV
            Grid<fdapde::Dynamic> opt;

            std::cout << "Start optimize" << std::endl ; 
            opt.optimize(GCV, lambdas_mat_quantile);
            std::cout << "End optimize" << std::endl ; 
            SVector<2> best_lambda = opt.optimum();

            // Save lambda sequences 
            std::ofstream fileLambda_S_Seq(solutions_path + "/alpha_" + alpha_string  + "/eps_" + eps_string + "/data_NEW" + "/lambdas_S_seq.csv");
            for(std::size_t i = 0; i < lambdas_d.size(); ++i) 
                fileLambda_S_Seq << std::setprecision(16) << lambdas_d[i] << "\n"; 
            fileLambda_S_Seq.close();
            for(std::size_t i = 0; i < lambdas_t.size(); ++i) 
                std::cout << lambdas_t[i] << "\n"; 
            std::ofstream fileLambda_T_Seq(solutions_path + "/alpha_" + alpha_string  + "/eps_" + eps_string + "/data_NEW" + "/lambdas_T_seq.csv");
            for(std::size_t i = 0; i < lambdas_t.size(); ++i) 
                fileLambda_T_Seq << std::setprecision(16) << lambdas_t[i] << "\n"; 
            fileLambda_T_Seq.close();

            // Save Lambda opt
            std::cout << "best_lambda[0] = " << best_lambda[0] << std::endl ; 
            std::ofstream fileLambdaoptS(solutions_path + "/alpha_" + alpha_string  + "/eps_" + eps_string + "/data_NEW" + "/lambda_s_opt.csv");
            if(fileLambdaoptS.is_open()){
                fileLambdaoptS << std::setprecision(16) << best_lambda[0];
                fileLambdaoptS.close();
            }

        

            std::ofstream fileLambdaoptT(solutions_path + "/alpha_" + alpha_string  + "/eps_" + eps_string + "/data_NEW" + "/lambda_t_opt.csv");
            if (fileLambdaoptT.is_open()){
                fileLambdaoptT << std::setprecision(16) << best_lambda[1];
                fileLambdaoptT.close();
            }
            // Save GCV scores
            std::ofstream fileGCV_scores(solutions_path + "/alpha_" + alpha_string  + "/eps_" + eps_string + "/data_NEW" + "/gcv_scores.csv");
            for(std::size_t i = 0; i < GCV.gcvs().size(); ++i){
                fileGCV_scores << std::setprecision(16) << std::sqrt(GCV.gcvs()[i]) << "\n"; 
                std::cout << "----- " << std::sqrt(GCV.gcvs()[i]) << std::endl ; 
            }
                
            fileGCV_scores.close();
            // Save edfs
            std::ofstream fileEDF(solutions_path + "/alpha_" + alpha_string  + "/eps_" + eps_string + "/data_NEW" + "/edfs.csv");
            for(std::size_t i = 0; i < GCV.edfs().size(); ++i) 
                fileEDF << std::setprecision(16) << GCV.edfs()[i] << "\n"; 
            fileEDF.close();  
            // Save gcv numerator
            std::ofstream fileGCVnum(solutions_path + "/alpha_" + alpha_string  + "/eps_" + eps_string + "/data_NEW" + "/gcv_num.csv");
            for(std::size_t i = 0; i < GCV.gcvs().size(); ++i) 
                fileGCVnum << std::setprecision(16) << std::sqrt(GCV.gcvs()[i]*model.n_obs()) << "\n"; 
            fileGCVnum.close();

        }

    }


// }     

}


// run time
TEST(case_study_run, laplacian_nonparametric_samplingatlocations_timelocations_separable_monolithic_missingdata) {

    const std::string eps_string = "1e-1";   // "0" "1e+0" "1e+1"

    std::string est_type = "quantile";    // mean quantile
    std::vector<double> alphas = {0.99}; // {0.5, 0.90, 0.95}; 
    std::string correct = "" ; // "" -> "_corrected" if you want to take the corrected lambdas

    const std::string num_months = "one_month";   // "one_month" "two_months"
    const std::string model_type = "parametric";  // "nonparametric" "parametric"
    const std::string pde_type = "";        // "transport"
    const std::string u_string = "1";               // value of u in case of transport
    const std::string covariate_type = "dens_log.elev.original";
    bool force_lambda = false ;   // ATT: cambia path di salvataggio 
    // const std::vector<std::string> covariate_type_vec = {"rain_wind_radiation_dens", "wind_radiation_dens", "radiation_dens", "dens"};
    const std::string mesh_type = "convex_hull";  // "square" "esagoni" "convex_hull"
    const std::string pollutant = "NO2"; 

    // Marco 
    std::string path = "/mnt/c/Users/marco/OneDrive - Politecnico di Milano/Corsi/Magistrale/Anno_II_Semestre_II/Thesis_shared/case_study/ARPA/Lombardia"; 
    std::string path_data = path + "/data/" + num_months + "/" + pollutant; 
    std::string solutions_path; 

    // // Ilenia 
    // std::string path = "/mnt/c/Users/ileni/OneDrive - Politecnico di Milano/Thesis_shared/case_study/ARPA/Lombardia"; 
    // std::string path_data = path + "/data/" + num_months + "/" + pollutant; 
    // std::string solutions_path;

    // for(std::string covariate_type : covariate_type_vec){

    if(est_type == "mean"){
        if(model_type == "nonparametric"){
            solutions_path = path + "/results/STRPDE/" + num_months + "/" + model_type + "/" + mesh_type + "/" + pollutant;
        } else{
            solutions_path = path + "/results/STRPDE/" + num_months + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/" + covariate_type;
        }
    }
        
    if(est_type == "quantile"){
        if(model_type == "nonparametric"){
            if(pde_type == ""){
                solutions_path = path + "/results/QSTRPDE/" + num_months + "/" + model_type + "/" + mesh_type + "/" + pollutant;
            }else{
                solutions_path = path + "/results/QSTRPDE/" + num_months + "/" + model_type + "/" + mesh_type + "/" + pollutant +
                                 "/pde_" + pde_type + "/u_" + u_string ;
            }
            
        } else{
            if(pde_type == ""){
                solutions_path = path + "/results/QSTRPDE/" + num_months + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/" + covariate_type;
            }else{
                solutions_path = path + "/results/QSTRPDE/" + num_months + "/" + model_type + "/" + mesh_type + "/" + pollutant + "/" + covariate_type +
                                 "/pde_" + pde_type + "/u_" + u_string ;
            }
        }
    }



    // define temporal domain
    unsigned int M; double tf; 
    if(num_months == "two_months"){
        M = 30; 
        tf = 61.0; 
    }
    if(num_months == "one_month"){
        M = 11; 
        tf = 22.0; 
    }
    std::string M_string = std::to_string(M);
    Triangulation<1, 1> time_mesh(0, tf, M-1);
    // define spatial domain and regularizing PDE
    MeshLoader<Triangulation<2, 2>> domain("mesh_lombardia_" + mesh_type);

    // import data and locs from files
    DMatrix<double> y; DMatrix<double> space_locs; DMatrix<double> time_locs; DMatrix<double> X; 

    y = read_csv<double>(path_data + "/y_rescale_NEW.csv");     // ATT  MESSO NEW (DATI NUOVI, PERIODO VERO9)
    time_locs = read_csv<double>(path_data + "/time_locs.csv"); 
    space_locs = read_csv<double>(path_data + "/locs.csv");
    if(model_type == "parametric")
        X = read_csv<double>(path_data + "/X_" + covariate_type + ".csv");
    
    // check dimensions
    std::cout << "dim space loc " << space_locs.rows() << " " << space_locs.cols() << std::endl;
    std::cout << "dim time loc " << time_locs.rows() << " " << time_locs.cols() << std::endl;
    std::cout << "dim y " << y.rows() << " " << y.cols() << std::endl;
    std::cout << "dim X " << X.rows() << " " << X.cols() << std::endl;

    BlockFrame<double, int> df;
    df.stack(OBSERVATIONS_BLK, y);
    if(model_type == "parametric")
        df.insert(DESIGN_MATRIX_BLK, X);
   
    // define regularizing PDE in space 

    // Laplacian
    if(pde_type == "transport")
        std::cout << "ATT You want to run a model with transport but you are using the Laplacian"; 

    auto Ld = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3 * time_mesh.n_nodes(), 1);
    PDE<Triangulation<2, 2>, decltype(Ld), DMatrix<double>, FEM, fem_order<1>> space_penalty(domain.mesh, Ld, u);

    // // Laplacian + transport 
    // if(pde_type == "")
    //     std::cout << "ATT You want to run a model with only the Laplacian but you are using a PDE with transport"; 

    // DMatrix<double, Eigen::RowMajor> b_data  = read_csv<double>(path_data + "/transport/b_" + u_string + "_opt.csv");  
    
    // DMatrix<double> u = read_csv<double>(path_data + "/transport/u_" + u_string + "_opt.csv");
  
    //DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3 * time_mesh.n_nodes(), 1);

    // DiscretizedVectorField<2, 2> b(b_data);
    // auto Ld = -laplacian<FEM>() + advection<FEM>(b);
    // PDE<Triangulation<2, 2>, decltype(Ld), DMatrix<double>, FEM, fem_order<1>> space_penalty(domain.mesh, Ld, u);

    // define regularizing PDE in time
    auto Lt = -bilaplacian<SPLINE>();
    PDE<Triangulation<1, 1>, decltype(Lt), DMatrix<double>, SPLINE, spline_order<3>> time_penalty(time_mesh, Lt);



    std::cout << "--------------------------------RUN STARTS--------------------------------" << std::endl; 
    if(est_type == "mean"){

        STRPDE<SpaceTimeSeparable, fdapde::monolithic> model(space_penalty, time_penalty, Sampling::pointwise);
    
        // set model's data
        model.set_spatial_locations(space_locs);
        model.set_temporal_locations(time_locs);

        // Read optima lambdas 
        double lambda_T; double lambda_S; 
        std::ifstream fileLambdaT_opt(solutions_path + "/lambda_t_opt.csv");
        if(fileLambdaT_opt.is_open()){
            fileLambdaT_opt >> lambda_T; 
            fileLambdaT_opt.close();
        }
        std::ifstream fileLambdaS_opt(solutions_path + "/lambda_s_opt.csv");
        if(fileLambdaS_opt.is_open()){
            fileLambdaS_opt >> lambda_S; 
            fileLambdaS_opt.close();
        }

        std::cout << "lambda S " << lambda_S << std::endl;
        std::cout << "lambda T " << lambda_T << std::endl;

        model.set_lambda_D(lambda_S);
        model.set_lambda_T(lambda_T);
        
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

        DMatrix<double> computedFn = model.Psi(not_nan())*model.f();
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
        
        for(double alpha : alphas){

            unsigned int alpha_int = alpha*100; 
            std::string alpha_string = std::to_string(alpha_int); 

            std::cout << "----------------ALPHA = " << alpha_string << "----------------" << std::endl; 

            QSRPDE<SpaceTimeSeparable> model(space_penalty, time_penalty, Sampling::pointwise, alpha);
    
            // set model's data
            model.set_spatial_locations(space_locs);
            model.set_temporal_locations(time_locs);

            // Read optima lambdas 
            double lambda_T; double lambda_S; 
            std::ifstream fileLambdaT_opt(solutions_path + "/alpha_" + alpha_string + "/eps_" + eps_string + "/data_NEW" + "/lambda_t_opt" + correct + ".csv");
            if(fileLambdaT_opt.is_open()){
                fileLambdaT_opt >> lambda_T; 
                fileLambdaT_opt.close();
            }
            std::ifstream fileLambdaS_opt(solutions_path + "/alpha_" + alpha_string + "/eps_" + eps_string + "/data_NEW" + "/lambda_s_opt" + correct + ".csv");
            if(fileLambdaS_opt.is_open()){
                fileLambdaS_opt >> lambda_S; 
                fileLambdaS_opt.close();
            }

            if(force_lambda){
                lambda_S = std::pow(10, -7); 
            }

            std::cout << "lambda S " << lambda_S << std::endl;
            std::cout << "lambda T " << lambda_T << std::endl;

            model.set_lambda_D(lambda_S);
            model.set_lambda_T(lambda_T);
            
            model.set_data(df);

            model.init();
            model.solve();

            // Save C++ solution 
            DMatrix<double> computedF = model.f();
            const static Eigen::IOFormat CSVFormatf(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
            std::ofstream filef(solutions_path + "/alpha_" + alpha_string + "/eps_" + eps_string + "/data_NEW" + "/f" + correct + "_bis.csv");
            if (filef.is_open()){
                filef << computedF.format(CSVFormatf);
                filef.close();
            }

            DMatrix<double> computedFn = model.Psi(not_nan())*model.f();
            const static Eigen::IOFormat CSVFormatfn(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
            std::ofstream filefn(solutions_path + "/alpha_" + alpha_string + "/eps_" + eps_string + "/data_NEW" + "/fn" + correct + "_bis.csv");
            if (filefn.is_open()){
                filefn << computedFn.format(CSVFormatfn);
                filefn.close();
            }

            if(model_type == "parametric"){
                DMatrix<double> computedBeta = model.beta();
                const static Eigen::IOFormat CSVFormatBeta(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
                std::ofstream filebeta(solutions_path + "/alpha_" + alpha_string + "/eps_" + eps_string + "/data_NEW" + "/beta" + correct + "_bis.csv");
                if (filebeta.is_open()){
                    filebeta << computedBeta.format(CSVFormatBeta);
                    filebeta.close();
                }
            }


        }

    }        
// }

}








