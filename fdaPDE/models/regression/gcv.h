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

#ifndef __GCV_H__
#define __GCV_H__

#include <fdaPDE/fields.h>
#include <fdaPDE/utils.h>

#include <functional>
#include <memory>
#include <type_traits>
using fdapde::core::ScalarField;
using fdapde::core::TwiceDifferentiableScalarField;
#include "../model_traits.h"
using fdapde::models::SpaceOnly;
#include "exact_edf.h"
#include "stochastic_edf.h"

namespace fdapde {
namespace models {
  
// base functor implementing the expression of the GCV index for model M (type-erased).
class GCV {
   private:
    using This = GCV;
    using VectorType = DVector<double>;
    using MatrixType = DMatrix<double>;
    // type-erased wrapper for Tr[S] computation strategy
    struct EDFStrategy__ {
        template <typename M> using fn_ptrs = fdapde::mem_fn_ptrs<&M::compute, &M::set_model, &M::S_get, &M::gcv_2_approach_set_trace>;
        // forwardings
        decltype(auto) compute() { return fdapde::invoke<double, 0>(*this); }
        void set_model(const RegressionView<void>& model) { fdapde::invoke<void, 1>(*this, model); }
        decltype(auto) S_get() const { return fdapde::invoke<const DMatrix<double>&   , 2>(*this); }  // M 
        void gcv_2_approach_set_trace(const bool& approach) { return fdapde::invoke<void, 3>(*this, approach); }  // M
    };

    RegressionView<void> model_;
    erase<heap_storage, EDFStrategy__> trS_;   // strategy used to evaluate the trace of smoothing matrix S
    std::vector<double> edfs_;                 // equivalent degrees of freedom q + Tr[S]
    std::vector<double> gcvs_;                 // computed values of GCV index
    // cache pairs (lambda, Tr[S]) for fast access if GCV is queried at an already computed point
    std::map<VectorType, double, fdapde::d_vector_compare<double>> cache_;

    std::vector<double> trace_debug;  // M debug

    // M 
    bool gcv_bool_approach_ = false;    // M: if you want to apply the second strategy for gcv with obs rip 

    // analytical expression of gcv at \lambda
    //
    // edf = n - (q + Tr[S])
    // GCV(\lambda) = n/(edf^2)*norm(y - \hat y)^2
    ScalarField<fdapde::Dynamic, double (This::*)(const VectorType&)> gcv_;
    double gcv_impl(const VectorType& lambda) {
        std::cout << "in GCV gcv_impl()" << std::endl; 
        // fit the model given current lambda
        model_.set_lambda(lambda);
        model_.init();
        model_.solve();
        // compute equivalent degrees of freedom given current lambda (if not already cached)
        if (cache_.find(lambda) == cache_.end()) { cache_[lambda] = trS_.compute(); }
        double trS = cache_[lambda];
        std::cout << "trS=" << trS << std::endl; 

        double q = model_.q();            // number of covariates
        std::size_t n = model_.n_obs();   // number of observations
        double dor;                       // residual degrees of freedom

        if(gcv_bool_approach_){               // M gcv for repeated observations
            std::cout << "gcv_impl with unique locs..." << std::endl; 
            dor = model_.num_unique_locs() - (model_.q() + trS);   // (n - (q + Tr[S])
        } else{
            std::cout << "gcv_impl with all locs..." << std::endl; 
            dor = n - (q + trS);       // residual degrees of freedom
        }

        edfs_.emplace_back(q + trS);      // store equivalent degrees of freedom
        // return gcv at point
        double gcv_value = model_.norm(model_.fitted(), model_.y()) / std::pow(dor, 2);  // M tolta costante causa rinormalizzazione loss 
        gcvs_.emplace_back(gcv_value);

        return gcv_value;
    }
   public:
    static constexpr int DomainDimension = fdapde::Dynamic;
    using EDFStrategy = erase<heap_storage, EDFStrategy__>;
    // constructors
    template <typename ModelType_, typename EDFStrategy_>
    GCV(const ModelType_& model, EDFStrategy_&& trS) : model_(model), trS_(trS), gcv_(this, &This::gcv_impl) {
        // set model pointer in edf computation strategy
        gcv_bool_approach_ = model.gcv_2_approach(); 
        trS_.set_model(model_);
        trS_.gcv_2_approach_set_trace(gcv_bool_approach_);
    }
    template <typename ModelType_> GCV(const ModelType_& model) : GCV(model, StochasticEDF()) { }
    GCV(const GCV& other) : model_(other.model_), trS_(other.trS_), gcv_(this, &This::gcv_impl) {
        // copy other GCV functor configuration
        gcv_.resize(other.gcv_.inner_size());
	gcv_.set_step(other.gcv_.step());
    }
    GCV() : gcv_(this, &This::gcv_impl) { }
    GCV& operator=(const GCV& other) {
        model_ = other.model_;
        trS_ = other.trS_;
        gcv_ = ScalarField<fdapde::Dynamic, double (This::*)(const VectorType&)>(this, &This::gcv_impl);
        return *this;
    }
  
    // call operator and numerical derivative approximations
    double operator()(const VectorType& lambda) { return gcv_(lambda); }
    std::function<VectorType(const VectorType&)> derive() const { return gcv_.derive(); }
    std::function<MatrixType(const VectorType&)> derive_twice() const { return gcv_.derive_twice(); }

    // returns GCV index of Model in its current state
    double eval() {
        std::cout << "in GCV eval()" << std::endl; 
        if (cache_.find(model_.lambda()) == cache_.end()) { cache_[model_.lambda()] = trS_.compute(); }
        double trS = cache_[model_.lambda()];

        trace_debug.push_back(trS); 

        if(gcv_bool_approach_){               // M gcv for repeated observations
            std::cout << "GCV computation with unique locs..." << std::endl; 
            double dor = model_.num_unique_locs() - (model_.q() + trS);   // (n - (q + Tr[S])
        } else{
            std::cout << "GCV computation with all locs..." << std::endl; 
            // GCV(\lambda) = n/((n - (q + Tr[S]))^2)*norm(y - \hat y)^2
            double dor = model_.n_obs() - (model_.q() + trS);   // (n - (q + Tr[S])
        }
        return (model_.norm(model_.fitted(), model_.y()) / std::pow(dor, 2)); // M 

    }

    // set edf_evaluation strategy
    template <typename EDFStrategy_> void set_edf_strategy(EDFStrategy_&& trS) {
        trS_ = trS;
	if(model_) {
        trS_.set_model(model_);
        trS_.gcv_2_approach_set_trace(gcv_bool_approach_);  // M 
    }
	edfs_.clear(); gcvs_.clear(); cache_.clear();
    }
    template <typename ModelType_> void set_model(ModelType_&& model) {
        model_ = model;
	if(trS_){
        trS_.set_model(model_);
        trS_.gcv_2_approach_set_trace(gcv_bool_approach_); // M 
    }
        edfs_.clear(); gcvs_.clear(); cache_.clear();
    }
    void set_step(double step) { gcv_.set_step(step); }

    void resize(int gcv_dynamic_inner_size) {
        fdapde_assert(gcv_dynamic_inner_size == 1 || gcv_dynamic_inner_size == 2);
        gcv_.resize(gcv_dynamic_inner_size);
    }
    // getters
    const std::vector<double>& edfs() const { return edfs_; }   // equivalent degrees of freedom q + Tr[S]
    const std::vector<double>& gcvs() const { return gcvs_; }   // computed values of GCV index
    const DMatrix<double>& S_get_gcv() const { return trS_.S_get(); }  // M 
    const DMatrix<double>& compute_IC() const { return trS_.S_get(); }  // M 
    const std::vector<double> get_trace() const { return trace_debug; }  // M
    int inner_size() const { return gcv_.inner_size(); }
    
};

// provides the analytical expresssion of GCV gradient and hessian, for newton-like optimization methods
/*template <typename M, typename RegularizationType> class ExactGCV;

// space only specialization of GCV exact derivatives
// expression of GCV derivatives:
//    edf = n - (q + Tr[S])
//    dGCV(\lambda)  = \frac{2n}{edf^2}[ \sigma^2 * Tr[dS] + a ]
//    ddGCV(\lambda) = \frac{2n}{edf^2}[ \frac{1}{edf}(3*\sigma^2*Tr[dS] + 4*a)*Tr[dS] + \sigma^2*Tr[ddS] + b ]
template <typename M> class ExactGCV<M, SpaceOnly> : public GCV<ExactEDF> {
   private:
    using GCV<ExactEDF>::model_;
    using GCV<ExactEDF>::trS_;
    DMatrix<double> L_;     // T^{-1}*R
    DMatrix<double> F_;     // (T^{-1}*R)*(T^{-1}*E)
    DVector<double> h_;     // h = (\lambda*L - I)*T^{-1}*R1^T*R0^{-1}*u
    DVector<double> p_;     // p = \Psi*h - dS*y
    DMatrix<double> S_;     // S = \Psi*T^{-1}*\Psi^T*Q
    DMatrix<double> dS_;    // dS = -\Psi*(T^{-1}*R)*(T^{-1}*E)
    DMatrix<double> ddS_;   // ddS = 2*\Psi*L*F

    // compute first derivative of matrix S: dS = -\Psi*(T^{-1}*R)*(T^{-1}*E)
    const DMatrix<double>& dS() {
        L_ = (trS_.invT_).solve(model_.R());     // T^{-1}*R
        F_ = L_ * (trS_.invT_).solve(trS_.E_);   // (T^{-1}*R)*(T^{-1}*E)
        dS_ = model_.Psi() * (-F_);
        return dS_;
    }
    // compute second derivative of matrix S: ddS = 2*\Psi*L*F
    const DMatrix<double>& ddS() {
        ddS_ = model_.Psi() * 2 * L_ * F_;
        return ddS_;
    }

    // computes the a term in the dGCV expression
    // a = p.dot(y - \hat y)
    //   p = \Psi*h - t
    //     h = (\lambda*L - I)*T^{-1}*g
    //       g = R1^T*R0^{-1}*u
    //     t = dS*y
    double a() {
        DMatrix<double> g = model_.R1().transpose() * model_.invR0().solve(model_.u());
        // cache h and p since needed for computation of second derivative
        h_ = (model_.lambda_D() * L_ - DMatrix<double>::Identity(model_.n_locs(), model_.n_locs())) *
             (trS_.invT_).solve(g);
        p_ = model_.Psi() * h_ - dS_ * model_.y();
        // return a = p.dot(y - \hat y)
        return ((model_.y() - model_.fitted()).transpose() * p_).coeff(0, 0);
    }

    // computes the b term in the ddGCV expression
    // b = p.dot(Q*p) + (-ddS*y - 2*\Psi*L*h).dot(y - \hat y)
    //   p = \Psi*h - t
    //     h = (\lambda*L - I)*T^{-1}*g
    //       g = R1^T*R0^{-1}*u
    //     t = dS*y
    double b() {
        DMatrix<double> C = 2 * L_ * h_;
        // perform efficient multiplication by permutation matrix Psi
        DMatrix<double> D(model_.n_locs(), 1);   // 2*\Psi*L*h
        for (std::size_t k = 0; k < model_.Psi().outerSize(); ++k) {
            for (SpMatrix<double>::InnerIterator it(model_.Psi(), k); it; ++it) { D.row(it.row()) = C.row(it.col()); }
        }
        DVector<double> Qp_;
        if (model_.has_covariates())
            Qp_ = model_.lmbQ(p_);   // efficient computation of Q*p
        else
            Qp_ = model_.W() * p_;
        // return b = p.dot(Q*p) + (-ddS*y - 2*\Psi*L*h).dot(y - \hat y)
        return ((model_.y() - model_.fitted()).transpose() * (-ddS_ * model_.y() - D)).coeff(0, 0) + p_.dot(Qp_);
    }
   public:
    ExactGCV(M& model) : GCV<M, ExactEDF>(model) {};

    // analytical expression of GCV first derivative
    //
    // edf      = n - (q + Tr[S])
    // \sigma^2 = \frac{norm(y - \hat y)^2}{n - (q + Tr[S])}
    // a        = p.dot(y - \hat y)
    // dGCV(\lambda) = \frac{2n}{edf^2}[ \sigma^2 * Tr[dS] + a ]
    std::function<SVector<1>(SVector<1>)> derive() {
        return [*this](SVector<1> lambda) mutable -> SVector<1> {
            // fit the model given current lambda
            model_.set_lambda(lambda);
            model_.init_model();
            model_.solve();
            // compute trace of matrix S and its first derivative given current lambda
            double trS = trS_.compute();
            double trdS = dS().trace();

            double q = model_.q();             // number of covariates
            std::size_t n = model_.n_locs();   // number of locations
            double edf = n - (q + trS);        // equivalent degrees of freedom
            // \sigma^2 = \frac{(y - \hat y).squaredNorm()}{n - (q + Tr[S])}
            double sigma = (model_.y() - model_.fitted()).squaredNorm() / edf;
            // return gradient of GCV at point
            return SVector<1>(2 * n / std::pow(n - (q + trS), 2) * (sigma * trdS + a()));
        };
    }

    // analytical expression of GCV second derivative
    //
    // edf      = n - (q + Tr[S])
    // \sigma^2 = \frac{norm(y - \hat y)^2}{n - (q + Tr[S])}
    // b        = p.dot(Q*p) + (-ddS*y - 2*\Psi*L*h).dot(y - \hat y)
    // ddGCV(\lambda) = \frac{2n}{edf^2}[ \frac{1}{edf}(3*\sigma^2*Tr[dS] + 4*a)*Tr[dS] + \sigma^2*Tr[ddS] + b ]
    std::function<SMatrix<1>(SVector<1>)> derive_twice() {
        return [*this](SVector<1> lambda) mutable -> SMatrix<1> {
            // fit the model given current lambda
            model_.set_lambda(lambda);
            model_.init_model();
            model_.solve();
            // compute trace of matrix S and its first and second derivative given current lambda
            double trS = trS_.compute();
            double trdS = dS().trace();
            double trddS = ddS().trace();

            double q = model_.q();             // number of covariates
            std::size_t n = model_.n_locs();   // number of locations
            double edf = n - (q + trS);        // equivalent degrees of freedom
            // \sigma^2 = \frac{norm(y - \hat y)^2}{n - (q + Tr[S])}
            double sigma = (model_.y() - model_.fitted()).squaredNorm() / edf;
            // return hessian of GCV at point
            return SMatrix<1>(
              2 * n / std::pow(edf, 2) * (trdS / edf * (3 * sigma * trdS + 4 * a()) + sigma * trddS + b()));
        };
    }
    };
*/

}   // namespace models
}   // namespace fdapde

#endif   // __GCV_H__
