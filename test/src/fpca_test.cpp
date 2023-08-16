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

#include <cstddef>
#include <gtest/gtest.h> // testing framework
#include <unsupported/Eigen/SparseExtra>

#include "../fdaPDE/core/utils/Symbols.h"
#include "../fdaPDE/core/utils/IO/CSVReader.h"
#include "../fdaPDE/core/utils/multithreading/ThreadPool.h"
using fdaPDE::multithreading::ThreadPool;
#include "../fdaPDE/core/FEM/PDE.h"
using fdaPDE::core::FEM::PDE;
#include "core/MESH/Mesh.h"
#include "../fdaPDE/models/functional/fPCA.h"
using fdaPDE::models::FPCA;
#include "../fdaPDE/models/SamplingDesign.h"
#include "../../fdaPDE/models/ModelTraits.h"

#include "../utils/MeshLoader.h"
using fdaPDE::testing::MeshLoader;
#include "../utils/Constants.h"
using fdaPDE::testing::DOUBLE_TOLERANCE;
#include "../utils/Utils.h"
using fdaPDE::testing::almost_equal;

/* test 1
   domain:       unit square [1,1] x [1,1]
   sampling:     locations = nodes
   penalization: simple laplacian
   BC:           no
   order FE:     1
   missing data: no
 */
// TEST(FPCA, Test1_Laplacian_GeostatisticalAtNodes_Fixed) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   double lambda = 1e-2;
//   FPCA<decltype(problem), fdaPDE::models::SpaceOnly, fdaPDE::models::GeoStatMeshNodes,
//        fdaPDE::models::fixed_lambda> model(problem);
//   model.setLambdaS(lambda);

//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile("data/models/FPCA/2D_test1/y.csv");
//   DMatrix<double> y = yFile.toEigen();
  
//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK, y);
//   model.setData(df);
  
//   // solve smoothing problem
//   model.init();
//   model.solve();
  
//   //   **  test correctness of computed results  **
  
//   // loadings vector
//   SpMatrix<double> expectedLoadings;
//   Eigen::loadMarket(expectedLoadings, "data/models/FPCA/2D_test1/loadings.mtx");
//   DMatrix<double> computedLoadings = model.loadings();
//   EXPECT_TRUE( almost_equal(DMatrix<double>(expectedLoadings), computedLoadings) );

//   // scores vector
//   SpMatrix<double> expectedScores;
//   Eigen::loadMarket(expectedScores,   "data/models/FPCA/2D_test1/scores.mtx");
//   DMatrix<double> computedScores = model.scores();
//   EXPECT_TRUE( almost_equal(DMatrix<double>(expectedScores), computedScores) );  
// }

// /* test 2
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations != nodes
//    penalization: simple laplacian
//    BC:           no
//    order FE:     1
//    missing data: no
//    GCV smoothing parameter selection
//  */
// TEST(FPCA, Test2_Laplacian_GeostatisticalAtLocations_GCV) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // load data from .csv files
//   CSVReader<double> reader{};
//   // load locations where data are sampled
//   CSVFile<double> locFile;
//   locFile = reader.parseFile("data/models/FPCA/2D_test2/locs.csv");
//   DMatrix<double> loc = locFile.toEigen();

//   std::shared_ptr<ThreadPool> tp = std::make_shared<ThreadPool>(10);
  
//   // define statistical model
//   FPCA<decltype(problem), fdaPDE::models::SpaceOnly, fdaPDE::models::GeoStatLocations,
//        fdaPDE::models::gcv_lambda_selection> model(problem);
//   model.init_multithreading(tp);
//   model.set_spatial_locations(loc);

//   // grid of smoothing parameters
//   std::vector<SVector<1>> lambdas;
//   for(double x = -4; x <= -2; x += 0.1) lambdas.push_back(SVector<1>(std::pow(10, x)));
//   model.setLambda(lambdas);
  
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile("data/models/FPCA/2D_test2/y.csv");
//   DMatrix<double> y = yFile.toEigen();
  
//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK, y);
//   model.setData(df);
  
//   // solve smoothing problem
//   model.init();
//   model.solve();
  
//   //   **  test correctness of computed results  **
  
//   // loadings vector
//   SpMatrix<double> expectedLoadings;
//   Eigen::loadMarket(expectedLoadings, "data/models/FPCA/2D_test2/loadings.mtx");
//   DMatrix<double> computedLoadings = model.loadings();
//   EXPECT_TRUE( almost_equal(DMatrix<double>(expectedLoadings), computedLoadings) );

//   // scores vector
//   SpMatrix<double> expectedScores;
//   Eigen::loadMarket(expectedScores,   "data/models/FPCA/2D_test2/scores.mtx");
//   DMatrix<double> computedScores = model.scores();
//   EXPECT_TRUE( almost_equal(DMatrix<double>(expectedScores), computedScores) );  
// }

// /* test 3
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations != nodes
//    penalization: simple laplacian
//    BC:           no
//    order FE:     1
//    missing data: no
//    KCV smoothing parameter selection, 10 folds
//  */
// TEST(FPCA, Test3_Laplacian_GeostatisticalAtLocations_KFoldCV) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // load data from .csv files
//   CSVReader<double> reader{};
//   // load locations where data are sampled
//   CSVFile<double> locFile;
//   locFile = reader.parseFile("data/models/FPCA/2D_test3/locs.csv");
//   DMatrix<double> loc = locFile.toEigen();
  
//   // define statistical model
//   FPCA<decltype(problem), fdaPDE::models::SpaceOnly, fdaPDE::models::GeoStatLocations,
//        fdaPDE::models::kcv_lambda_selection> model(problem);
//   model.set_spatial_locations(loc);

//   // grid of smoothing parameters
//   std::vector<SVector<1>> lambdas;
//   for(double x = -4; x <= -2; x += 0.1) lambdas.push_back(SVector<1>(std::pow(10, x)));
//   model.setLambda(lambdas);
  
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile("data/models/FPCA/2D_test3/y.csv");
//   DMatrix<double> y = yFile.toEigen();
  
//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK, y);
//   model.setData(df);
  
//   // solve smoothing problem
//   model.init();
//   model.solve();
  
//   //   **  test correctness of computed results  **
//   Eigen::saveMarket(model.scores(), "scores.mtx");
//   Eigen::saveMarket(model.loadings(), "loadings.mtx");
  
//   // loadings vector
//   SpMatrix<double> expectedLoadings;
//   Eigen::loadMarket(expectedLoadings, "data/models/FPCA/2D_test3/loadings.mtx");
//   DMatrix<double> computedLoadings = model.loadings();
//   EXPECT_TRUE( almost_equal(DMatrix<double>(expectedLoadings), computedLoadings) );

//   // scores vector
//   SpMatrix<double> expectedScores;
//   Eigen::loadMarket(expectedScores,   "data/models/FPCA/2D_test3/scores.mtx");
//   DMatrix<double> computedScores = model.scores();
//   EXPECT_TRUE( almost_equal(DMatrix<double>(expectedScores), computedScores) );  
// }

/* test 2
   domain:       unit square [1,1] x [1,1]
   sampling:     locations = nodes
   penalization: simple laplacian
   BC:           no
   order FE:     1
   missing data: yes
 */
// TEST(FPCA, Test4_Laplacian_GeostatisticalAtNodes_MissingData) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square_coarse");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   // use optimal lambda to avoid possible numerical issues
//   double lambda = 1e-2;
//   FPCA<decltype(problem), fdaPDE::models::SpaceOnly, fdaPDE::models::GeoStatMeshNodes,
//        fdaPDE::models::fixed_lambda> model(problem);
//   model.setLambdaS(lambda);
  
//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile("data/models/FPCA/2D_test4/y.csv");
//   DMatrix<double> y = yFile.toEigen();
  
//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK, y);
//   model.setData(df);

//   // solve smoothing problem
//   model.init();
//   model.solve();

//   std::cout << model.scores() << std::endl;
  
//   //   **  test correctness of computed results  **

//   // loadings vector
//   // SpMatrix<double> expectedLoadings;
//   // Eigen::loadMarket(expectedLoadings, "data/models/FPCA/2D_test1/loadings.mtx");
//   // DMatrix<double> computedLoadings = model.loadings();
//   // EXPECT_TRUE( almost_equal(DMatrix<double>(expectedLoadings), computedLoadings) );

//   // // scores vector
//   // SpMatrix<double> expectedScores;
//   // Eigen::loadMarket(expectedScores,   "data/models/FPCA/2D_test1/scores.mtx");
//   // DMatrix<double> computedScores = model.scores();
//   // EXPECT_TRUE( almost_equal(DMatrix<double>(expectedScores), computedScores) );  
// }


// TEST(FPCA, Test4_Laplacian_GeostatisticalAtNodes_MissingData) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   // define statistical model
//   // use optimal lambda to avoid possible numerical issues
//   double lambda = 1.04713e-05;
//   FPCA<decltype(problem), fdaPDE::models::SpaceOnly, fdaPDE::models::GeoStatMeshNodes,
//        fdaPDE::models::fixed_lambda> model(problem);
//   model.setLambdaS(lambda);
  
//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile("data/models/FPCA/2D_test6/y.csv");
//   DMatrix<double> y = yFile.toEigen();
  
//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK, y);
//   model.setData(df);

//   // solve smoothing problem
//   model.init();
//   model.solve();

//   std::cout << model.scores() << std::endl;
  
//   //   **  test correctness of computed results  **

//   // loadings vector
//   // SpMatrix<double> expectedLoadings;
//   // Eigen::loadMarket(expectedLoadings, "data/models/FPCA/2D_test1/loadings.mtx");
//   // DMatrix<double> computedLoadings = model.loadings();
//   // EXPECT_TRUE( almost_equal(DMatrix<double>(expectedLoadings), computedLoadings) );

//   // // scores vector
//   // SpMatrix<double> expectedScores;
//   // Eigen::loadMarket(expectedScores,   "data/models/FPCA/2D_test1/scores.mtx");
//   // DMatrix<double> computedScores = model.scores();
//   // EXPECT_TRUE( almost_equal(DMatrix<double>(expectedScores), computedScores) );  
// }

// #include "../fdaPDE/models/functional/gap_fill.h"
// using fdapde::models::GapFill;

// TEST(GapFill, Test5_Laplacian_GeostatisticalAtNodes_MissingData) {
//   // define domain and regularizing PDE
//   MeshLoader<Mesh2D<>> domain("unit_square");
//   auto L = Laplacian();
//   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
//   PDE problem(domain.mesh, L, u); // definition of regularizing PDE

//   std::shared_ptr<ThreadPool> tp = std::make_shared<ThreadPool>(4);
  
//   // define statistical model
//   GapFill<decltype(problem), fdaPDE::models::SpaceOnly, fdaPDE::models::GeoStatMeshNodes> model(problem);
//   model.init_multithreading(tp);
  
//   std::vector<SVector<1>> lambdas;
//   for(double x = -4; x <= -2; x += 0.1) lambdas.push_back(SVector<1>(std::pow(10, x)));
//   model.setLambda(lambdas);

//   // load data from .csv files
//   CSVReader<double> reader{};
//   CSVFile<double> yFile; // observation file
//   yFile = reader.parseFile("data/models/FPCA/2D_test6/y.csv");
//   DMatrix<double> y = yFile.toEigen();
  
//   // set model data
//   BlockFrame<double, int> df;
//   df.insert(OBSERVATIONS_BLK, y);
//   model.setData(df);

//   // solve smoothing problem
//   model.init();
//   model.solve();

  
//   //   **  test correctness of computed results  **

//   // loadings vector
//   // SpMatrix<double> expectedLoadings;
//   // Eigen::loadMarket(expectedLoadings, "data/models/FPCA/2D_test1/loadings.mtx");
//   // DMatrix<double> computedLoadings = model.loadings();
//   // EXPECT_TRUE( almost_equal(DMatrix<double>(expectedLoadings), computedLoadings) );

//   // // scores vector
//   // SpMatrix<double> expectedScores;
//   // Eigen::loadMarket(expectedScores,   "data/models/FPCA/2D_test1/scores.mtx");
//   // DMatrix<double> computedScores = model.scores();
//   // EXPECT_TRUE( almost_equal(DMatrix<double>(expectedScores), computedScores) );  
// }
