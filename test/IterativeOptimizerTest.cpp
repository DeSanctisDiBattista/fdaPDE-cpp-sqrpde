#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h> // testing framework

#include "../fdaPDE/core/utils/Symbols.h"
#include "../fdaPDE/core/utils/fields/ScalarField.h"
using fdaPDE::core::ScalarField;
using fdaPDE::core::TwiceDifferentiableScalarField;
#include "../fdaPDE/core/OPT/optimizers/BFGS.h"
using fdaPDE::core::OPT::BFGSOptimizer;
#include "../fdaPDE/core/OPT/optimizers/Newton.h"
using fdaPDE::core::OPT::NewtonOptimizer;

// a templated test suite for testing iterative optimizers over 2D problems. Any iterative optimizer in OPT must satisfy this test suite

// create templated test fixture
template <typename T> class IterativeOptimizerTest : public ::testing::Test {};

// associate list of types to the test suite. If you insert a new *iterative* optimizer in OPT module it must be inserted in this
// type list. Any iterative optimizer MUST pass all the test cases here listed to guarantee it conforms to the adopted interface in OPT.
// You can further add specific test cases for the optimizer at hand in a different dedicated file.
using optList = ::testing::Types<BFGSOptimizer<2>, NewtonOptimizer<2>>;
TYPED_TEST_SUITE(IterativeOptimizerTest, optList);

// optimization using numerical approximation of derivatives works
TYPED_TEST(IterativeOptimizerTest, Optimize2DConvexNonExact) {
  // define objective function: x^2 + y^2
  ScalarField<2> field([](SVector<2> x) -> double { return std::pow(x[0],2) + std::pow(x[1], 2); });
  // do not define analytical gradient and hessian, instead resort to numerical approximations
  
  // define optimizer
  double tolerance = 0.0001;
  TypeParam opt(1000, tolerance, 0.01);
  // perform optimization
  SVector<2> init(1,1);
  opt.findMinimum(field, init);
  // optimum value should be near to zero, at least below opt tolerance
  double objValue = opt.getObjValue();
  EXPECT_TRUE(std::abs(objValue - 0) < tolerance);
}

// check that calling .findMinimum after a previous call to .findMinimum returns the same result, this means to check
// that the optimizer is left to a valid state after execution of .findMinimum
TYPED_TEST(IterativeOptimizerTest, SubsequentCallsSameResult) {
  // define objective function: x^2 + y^2
  ScalarField<2> field([](SVector<2> x) -> double { return std::pow(x[0],2) + std::pow(x[1], 2); });
  // do not define analytical gradient and hessian, instead resort to numerical approximations
  
  // define optimizer
  double tolerance = 0.0001;
  TypeParam opt(1000, tolerance, 0.01);
  // perform optimization
  SVector<2> init(1,1);
  opt.findMinimum(field, init);
  // optimum value should be near to zero, at least below opt tolerance
  double objValue = opt.getObjValue();
  EXPECT_TRUE(std::abs(objValue - 0) < tolerance);

  // optimize again using same datum, the solution found must be equal
  opt.findMinimum(field, init);
  double objValue2 = opt.getObjValue();
  EXPECT_DOUBLE_EQ(objValue, objValue2);

  // corner case in which the objective changes and the optimizer returns after 0 iterations (check no spurious state is left from
  // previous optimization)
  ScalarField<2> field2([](SVector<2> x) -> double { // x*e^{-x^2 - y^2} + (x^2 + y^2)/20
    return x[0]*std::exp(- std::pow(x[0],2) - std::pow(x[1], 2)) + (std::pow(x[0],2) + std::pow(x[1], 2))/20;
  });
  // set init to optimum point
  SVector<2> init2(-0.669071831647573, 0.000000004602598);
  // opt should return after 0 iterations
  opt.findMinimum(field2, init2);
  // optimum found should be equal to true optimum
  double exprectedOptimum = -0.405236870266690;
  EXPECT_DOUBLE_EQ(exprectedOptimum, opt.getObjValue()) << opt.iterations();
  
  // if the above assertion is not met, most likely the optimizer doesn't clean its state after a call to .findMinimum()
  // you should always clean the state of the optimizer before starting a new optimization
}

// supply an objective with exact expression of gradient and hessian triggers the optimizer to use these quantites
TYPED_TEST(IterativeOptimizerTest, Optimize2DConvexExact) {
  // define objective function: x^2 + y^2
  std::function<double(SVector<2>)> fieldExpr = [](SVector<2> x) -> double {
    return std::pow(x[0],2) + std::pow(x[1], 2);
  };
  // define analytical gradient vector
  std::function<double(SVector<2>)> dx = [](SVector<2> x) -> double { return 2*x[0]; };
  std::function<double(SVector<2>)> dy = [](SVector<2> x) -> double { return 2*x[1]; };
  // define analytical hessian matrix
  std::function<SMatrix<2>(SVector<2>)> hess = [](SVector<2> x) -> SMatrix<2> {
    return SMatrix<2> {{2,0}, {0,2}};
  };
  // field approximated using central differences
  ScalarField<2> field_approx(fieldExpr);
  // define the same field but supplying both analytical gradient and hessian
  TwiceDifferentiableScalarField<2> field(fieldExpr, {dx, dy}, hess);
  
  // define optimizer
  double tolerance = 0.0001;
  TypeParam opt(1000, tolerance, 0.01);
  // perform optimization using approximation 
  SVector<2> init(1,1);
  opt.findMinimum(field_approx, init);
  double objValue_approx = opt.getObjValue();
  // perform optimization using analytical gradient and hessian
  opt.findMinimum(field, init);
  double objValue_exact = opt.getObjValue();
  
  // expect objValue_exact below tolerance
  EXPECT_TRUE(std::abs(objValue_exact - 0) < tolerance);
  // expect the two values different (this means that the analytical expressions are actually used by the method)
  EXPECT_FALSE(objValue_exact == objValue_approx) << objValue_exact << " - " << opt.iterations();
  // expect solution obtained with exact expressions approximately equal to the approximated one
  EXPECT_TRUE(std::abs(objValue_exact - objValue_approx) < tolerance);
}

// use a non convex objective to stress the optimizer but with a unique minimum
TYPED_TEST(IterativeOptimizerTest, Optimize2DNonConvexUniqueMin) {
  // define non convex objective function having just one minimum : x*e^{-x^2 - y^2} + (x^2 + y^2)/20
  ScalarField<2> field([](SVector<2> x) -> double {
    return x[0]*std::exp(- std::pow(x[0],2) - std::pow(x[1], 2)) + (std::pow(x[0],2) + std::pow(x[1], 2))/20;
  });
  // do not define analytical gradient and hessian, instead resort to numerical approximations
  
  // define optimizer
  double tolerance = 0.0001;
  TypeParam opt(1000, tolerance, 0.01);
  // perform optimization
  SVector<2> init(-0.5, 0);
  opt.findMinimum(field, init);
  // optimum value should be near to expectedValue, at least below opt tolerance
  double expectedValue = -0.405236870266690;
  double objValue = opt.getObjValue();
  EXPECT_TRUE(std::abs(objValue - expectedValue) < tolerance) << objValue << " - " << opt.iterations();
}

// check that if tolerance is not reached maximum iteration is met
TYPED_TEST(IterativeOptimizerTest, StopOnMaxIter) {
  // select a convex function with a single minimum point: 2*x^2 + 4*y^2
  ScalarField<2> field([](SVector<2> x) -> double { return 2*std::pow(x[0],2) + 4*std::pow(x[1], 2); });

  // define optimizer, set tolerance too low to be reached
  double tolerance = std::pow(0.1, 20);
  TypeParam opt(1000, tolerance, 0.01);
  // perform optimization
  SVector<2> init(1,1);
  opt.findMinimum(field, init);
  // optimum value should be near to zero, at least below opt tolerance. If this condition is not met, the maximum
  // number of iterations must be reached
  double objValue = opt.getObjValue();
  EXPECT_TRUE(std::abs(objValue - 0) < tolerance || opt.iterations() == 1000);
}