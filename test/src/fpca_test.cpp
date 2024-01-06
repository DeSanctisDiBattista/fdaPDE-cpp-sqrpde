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

#include <fdaPDE/core.h>
using fdapde::core::FEM;
using fdapde::core::fem_order;
using fdapde::core::laplacian;
using fdapde::core::PDE;

#include "../../fdaPDE/models/functional/fpca.h"
#include "../../fdaPDE/models/sampling_design.h"
using fdapde::models::FPCA;
using fdapde::models::Sampling;
using fdapde::models::Calibration;

#include "utils/constants.h"
#include "utils/mesh_loader.h"
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_csv;
using fdapde::testing::read_mtx;

// test 1
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    BC:           no
//    order FE:     1
//    missing data: no
//    solver: sequential (power iteration)
TEST(fpca_test, laplacian_samplingatnodes_sequential) {
    // define domain
    MeshLoader<Mesh2D> domain("unit_square");
    // import data from files
    DMatrix<double> y = read_csv<double>("../data/models/fpca/2D_test1/y.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define model
    double lambda_D = 1e-2;
    FPCA<SpaceOnly, fdapde::sequential> model(problem, Sampling::mesh_nodes, Calibration::off);
    model.set_lambda_D(lambda_D);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // solve FPCA problem
    model.init();
    model.solve();
    // test correctness
    EXPECT_TRUE(almost_equal(model.loadings(), "../data/models/fpca/2D_test1/loadings_seq.mtx"));
    EXPECT_TRUE(almost_equal(model.scores(),   "../data/models/fpca/2D_test1/scores_seq.mtx"));
}

// test 2
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    BC:           no
//    order FE:     1
//    missing data: no
//    solver: monolithic (rsvd)
TEST(fpca_test, laplacian_samplingatnodes_monolithic) {
    // define domain
    MeshLoader<Mesh2D> domain("unit_square");
    // import data from files
    DMatrix<double> y = read_csv<double>("../data/models/fpca/2D_test1/y.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define model
    double lambda_D = 1e-2;
    FPCA<SpaceOnly, fdapde::monolithic> model(problem, Sampling::mesh_nodes);
    model.set_lambda_D(lambda_D);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // solve FPCA problem
    model.init();
    model.solve();
    // test correctness
    EXPECT_TRUE(almost_equal(model.loadings(), "../data/models/fpca/2D_test1/loadings_mon.mtx"));
    EXPECT_TRUE(almost_equal(model.scores(),   "../data/models/fpca/2D_test1/scores_mon.mtx"));
}

// test 2
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations != nodes
//    penalization: simple laplacian
//    BC:           no
//    order FE:     1
//    missing data: no
//    solver: sequential (power iteration) + GCV \lambda selection
TEST(fpca_test, laplacian_samplingatlocations_sequential_gcv) {
    // define domain
    MeshLoader<Mesh2D> domain("unit_square");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/fpca/2D_test2/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/fpca/2D_test2/y.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define model
    FPCA<SpaceOnly, fdapde::sequential> model(problem, Sampling::pointwise, Calibration::gcv);
    model.set_spatial_locations(locs);
    // grid of smoothing parameters
    std::vector<DVector<double>> lambda_grid;
    for (double x = -4; x <= -2; x += 0.1) { lambda_grid.push_back(SVector<1>(std::pow(10, x))); }
    model.set_lambda(lambda_grid);
    model.set_seed(78965);   // for reproducibility purposes in testing
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // solve FPCA problem
    model.init();
    model.solve();
    // test correctness
    EXPECT_TRUE(almost_equal(model.loadings(), "../data/models/fpca/2D_test2/loadings.mtx"));
    EXPECT_TRUE(almost_equal(model.scores(),   "../data/models/fpca/2D_test2/scores.mtx"  ));
}

/*
// test 3
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations != nodes
//    penalization: simple laplacian
//    BC:           no
//    order FE:     1
//    missing data: no
//    KCV smoothing parameter selection, 10 folds
TEST(fpca_test, laplacian_samplingatlocations_kcvcalibration) {
    // define domain
    MeshLoader<Mesh2D> domain("unit_square");
    // import data from files
    DMatrix<double> locs = read_csv<double>("../data/models/fpca/2D_test3/locs.csv");
    DMatrix<double> y    = read_csv<double>("../data/models/fpca/2D_test3/y.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define model
    FPCA<decltype(problem), SpaceOnly, GeoStatLocations, KCVCalibration> model(problem);
    model.set_spatial_locations(locs);
    // grid of smoothing parameters
    std::vector<SVector<1>> lambdas;
    for (double x = -4; x <= -2; x += 0.1) lambdas.push_back(SVector<1>(std::pow(10, x)));
    model.set_lambda(lambdas);
    model.set_seed(12654);   // for reproducibility purposes in testing
    model.set_nfolds(10);    // perform a 10 folds cross-validation
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // solve FPCA problem
    model.init();
    model.solve();
    // test correctness
    EXPECT_TRUE(almost_equal(model.loadings(), "../data/models/fpca/2D_test3/loadings.mtx"));
    EXPECT_TRUE(almost_equal(model.scores(),   "../data/models/fpca/2D_test3/scores.mtx"  ));
}

// test 4
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    BC:           no
//    order FE:     1
//    missing data: yes
TEST(fpca_test, laplacian_samplingatnodes_nocalibration_missingdata) {
    // define domain
    MeshLoader<Mesh2D> domain("unit_square_coarse");
    // import data from files
    DMatrix<double> y = read_csv<double>("../data/models/fpca/2D_test4/y.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define model
    double lambda_D = 1e-2;
    FPCA<decltype(problem), SpaceOnly, GeoStatMeshNodes, NoCalibration> model(problem);
    model.set_lambda_D(lambda_D);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);
    // solve FPCA problem
    model.init();
    model.solve();
    // test correctness
    EXPECT_TRUE(almost_equal(model.loadings(), "../data/models/fpca/2D_test4/loadings.mtx"));
    EXPECT_TRUE(almost_equal(model.scores(),   "../data/models/fpca/2D_test4/scores.mtx"  ));
}
*/
