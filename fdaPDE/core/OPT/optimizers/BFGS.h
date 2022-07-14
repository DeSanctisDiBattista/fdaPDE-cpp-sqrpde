#ifndef __BFGS_H__
#define __BFGS_H__

#include "../../utils/Symbols.h"
#include "../../utils/fields/ScalarField.h"
#include "../extensions/Extension.h"

namespace fdaPDE{
namespace core{
namespace OPT{
  
  // implementation of the BFGS optimizer
  template <unsigned int N>
  class BFGSOptimizer {
  private:
    // internal status of the optimizer 
    SVector<N> x_old_{};    // value of the optimization point before the update step
    SVector<N> x_new_{};    // value of the optimization point after the update step
    SVector<N> update_{};   // update vector computed at each step
    SVector<N> grad_old_{}; // value of the field's gradient before the update step
    SVector<N> grad_new_{}; // value of the field's gradient after the update step
    SMatrix<N> hessian_{};  // value of the hessian matrix approximation at iteration i.
    double error_ = 0;      // squared l^2 norm of the gradient after the update step
    
    // optimization problem data
    unsigned int maxIter_; // maximum number of iterations before forced stop
    double tolerance_; // tolerance on error
    unsigned int numIter_ = 0; // counter to keep track of the number of iterations executed
    double h_; // step employed by the optimization scheme.

    // results of the optimization
    SVector<N> minimumPoint_;
    double objectiveValue_;

  public:
    // constructor
    BFGSOptimizer(unsigned int maxIter, double tolerance, double h)
      : maxIter_(maxIter), tolerance_(tolerance), h_(h) {};

    // optimization routine, depending on the objective type the method employs an approximation or the exact expression
    // of gradient and/or hessian function.
    template <typename... Args>
    void findMinimum(const ScalarField<N>& objective, // objective to optimize
		     const SVector<N>& x0, // initial point
		     const Args&... args);

    // getters to internal state
    unsigned int iterations() const { return numIter_; }
    double error() const { return error_; }
    SVector<N> x_old() const { return x_old_; }
    SVector<N> x_new() const { return x_new_; }
    SVector<N> update_vector() const { return update_; }
    SVector<N> gradient_old() const { return grad_old_; }    
    SVector<N> gradient_new() const { return grad_new_; }
    SMatrix<N> hessian() const { return hessian_; }
    // getters to optimization solution
    SVector<N> getSolution() const { return minimumPoint_; }
    double getObjValue() const { return objectiveValue_; }    
  };

#include "BFGS.tpp"
}}}
#endif // __BFGS_H__