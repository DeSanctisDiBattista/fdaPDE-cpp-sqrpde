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

#ifndef __QSRPDE_H__
#define __QSRPDE_H__

#include <fdaPDE/pde.h>
#include <fdaPDE/utils.h>

#include "../model_macros.h"
#include "fpirls.h"
#include "regression_base.h"

namespace fdapde {
namespace models {

template <typename RegularizationType_>
class QSRPDE : public RegressionBase<QSRPDE<RegularizationType_>, RegularizationType_> {
   public:
    using RegularizationType = std::decay_t<RegularizationType_>;
    using This = QSRPDE<RegularizationType>;
    using Base = RegressionBase<QSRPDE<RegularizationType>, RegularizationType>;
    // import commonly defined symbols from base
    IMPORT_REGRESSION_SYMBOLS
    using Base::invXtWX_;   // LU factorization of X^T*W*X
    using Base::lambda_D;   // smoothing parameter in space
    using Base::n_basis;    // number of spatial basis functions
    using Base::P;          // discretized penalty matrix
    using Base::W_;         // weight matrix
    using Base::XtWX_;      // q x q matrix X^T*W*X
    using Base::unique_locs_flags; // M 
    using Base::X_II_approach; // M 
    using Base::W_II_approach; // M 
    using Base::Psi_II_approach; // M 
    using Base::PsiTD_II_approach; // M 


    // constructor
    QSRPDE() = default;
    // space-only and space-time parabolic constructor
    QSRPDE(const Base::PDE& pde, Sampling s, double alpha = 0.5)
        requires(is_space_only<This>::value || is_space_time_parabolic<This>::value)
        : Base(pde, s), alpha_(alpha) {
        fpirls_ = FPIRLS<This>(this, tol_, max_iter_);
    }
    // space-time separable constructor
    QSRPDE(const Base::PDE& space_penalty, const Base::PDE& time_penalty, Sampling s, double alpha = 0.5)
        requires(is_space_time_separable<This>::value)
        : Base(space_penalty, time_penalty, s), alpha_(alpha) {
        fpirls_ = FPIRLS<This>(this, tol_, max_iter_);
    }
    // setter
    void set_fpirls_tolerance(double tol) { tol_ = tol; }
    void set_fpirls_max_iter(int max_iter) { max_iter_ = max_iter; }
    void set_alpha(double alpha) { alpha_ = alpha; }
    void set_eps_power(double eps) { eps_ = eps; }
    void set_weights_tolerance(double tol_weights) { tol_weights_ = tol_weights; }
    // M 
    void gcv_oss_rip_strategy_set(const std::string str){
        if(str == "I"){
            std::cout << "First strategy set" << std::endl; 
            gcv_oss_rip_I_strategy = true; 
        }
        if(str == "II"){
            std::cout << "Second strategy set" << std::endl; 
            gcv_oss_rip_II_strategy = true; 
            Base::gcv_2_approach_set(true); 
        }
    }

    void init_model() { 
        fpirls_.init();
    }
    void solve() {

        // execute FPIRLS_ for minimization of functional \norm{V^{-1/2}(y - \mu)}^2 + \lambda \int_D (Lf - u)^2
        fpirls_.compute();

        // Debug -> salva il numero di iterazioni (per il test obs ripetute)
        n_iter_qsrpde_ = fpirls_.n_iter();

        // fpirls_ converged: store solution estimates  
        W_ = fpirls_.solver().W();
        f_ = fpirls_.solver().f();
        g_ = fpirls_.solver().g();
        // parametric part, if problem was semi-parametric
        if (has_covariates()) {
            beta_ = fpirls_.solver().beta();
            XtWX_ = fpirls_.solver().XtWX();
            invXtWX_ = fpirls_.solver().invXtWX();
            U_ = fpirls_.solver().U();
            V_ = fpirls_.solver().V();
        }
        invA_ = fpirls_.solver().invA();

        // M 
        if(Base::gcv_2_approach() && Base::num_unique_locs() != n_locs()){
            std::cout << "Computing weigths for GCV II approach..." << std::endl; 
            // nb: calcoli ripresi da "fpirls_compute_step", che li fa ad ogni step di fpirls. Ma qui ci servono solo una volta, a convergenza di fpirls. 

            // compute summary of observations
            DVector<double> summary_vec = compute_summary_data(); 
            // compute mu_II_approach
            DVector<double> mu_II_approach = skip_repeated_locs(mu_); 

            // compute the weights 
            DVector<double> abs_res_II_approach = (summary_vec - mu_II_approach).array().abs();
            // W_i = 1/(2*n*(abs_res[i] + tol_weights_)) if abs_res[i] < tol_weights, W_i = 1/(2*n*abs_res[i]) otherwise
            pW_II_approach =
            (abs_res_II_approach.array() < tol_weights_)
                .select(
                (2 * (abs_res_II_approach.array() + tol_weights_)).inverse(), (2 * abs_res_II_approach.array()).inverse());
            py_II_approach = summary_vec - (1 - 2. * alpha_) * abs_res_II_approach;
            // NB: il fattore di normalizzazione verrà inserito dalla Base class, che usa sempre n*
            
            // NB: maschera NON applicata ==> non funziona con k-fold CV e con space-time !!
            Base::W_II_approach_set(pW_II_approach.asDiagonal());
            Base::invA_II_approach_set(); 

            if(Base::has_covariates()) {
                // Set XtWX_II_approach_set and its inverse in base class 
                std::cout << "in solve, setting XtWX, U, V II approach" << std::endl; 
                Base::XtWX_II_approach_set(X_II_approach().transpose() * W_II_approach() * X_II_approach()); // at this point W_II_approach has been filled
                Base::U_II_approach_set(); 
                Base::V_II_approach_set(); 
            }
        } 

        return;
    }

    // required by FPIRLS_ (see fpirls_.h for details)
    // initalizes mean vector \mu
    void fpirls_init() {
        // TODO: use a standardized solver (we can exploit the fpirls solver)
        // non-parametric and semi-parametric cases coincide here, since beta^(0) = 0

        // nota: qui il fattore di normalizzazione rimane in quanto questo è il sistema SRPDE con 
        //       pesi posti uguali all'identità.  
        SparseBlockMatrix<double, 2, 2> A(
          -PsiTD() * Psi() / n_obs(), 2 * lambda_D() * R1().transpose(),   // NB: note the 2 * here
          lambda_D() * R1(),          lambda_D() * R0()                );
        if constexpr (is_space_time_separable<This>::value) {
            A.block(0, 0) -= 2*Base::lambda_T() * Kronecker(Base::P1(), Base::pde().mass());
        }
        fdapde::SparseLU<SpMatrix<double>> invA;
        invA.compute(A);
        // assemble rhs of srpde problem
        DVector<double> b(A.rows());
        b.block(n_basis(), 0, n_basis(), 1) = lambda_D() * u();
        b.block(0, 0, n_basis(), 1) = -PsiTD() * y() / n_obs();
        mu_ = Psi(not_nan()) * (invA.solve(b)).head(n_basis());
    }
    // computes W^k = diag(1/(2*n*|y - X*beta - f|)) and y^k = y - (1-2*alpha)|y - X*beta - f|
    void fpirls_compute_step() {
        DVector<double> abs_res = (y() - mu_).array().abs();

        // M ora che la loss di SRPDE è normalizzata, non c'è più il fattore 1/n nei pesi dato
        //   che è inserito direttamente da SRPDE
        // W_i = 1/(2*(abs_res[i] + tol_weights_)) if abs_res[i] < tol_weights, W_i = 1/(2*abs_res[i]) otherwise
        pW_ =
          (abs_res.array() < tol_weights_)
            .select(
              (2 * (abs_res.array() + tol_weights_)).inverse(), (2 * abs_res.array()).inverse());
        py_ = y() - (1 - 2. * alpha_) * abs_res;

        for(std::size_t i=0; i<n_locs(); ++i){
            if(Base::nan_mask()[i]){
                py_(i)=0.; 
                pW_(i)=0.; 
            }
        }

    }
    // updates mean vector \mu after WLS solution
    void fpirls_update_step(const DMatrix<double>& hat_f, [[maybe_unused]] const DMatrix<double>& hat_beta) {
        mu_ = hat_f;
    }
    // returns the data loss \norm{diag(W)^{-1/2}(y - \mu)}^2
    double data_loss() const { return (pW_.cwiseSqrt().matrix().asDiagonal() * (py_ - mu_)).squaredNorm() / n_obs(); }
    // M aggiunto fattore di normalizzazione perchè pW_ non lo contiene

    const DVector<double>& py() const { return py_; }
    const DVector<double>& pW() const { return pW_; }
    const fdapde::SparseLU<SpMatrix<double>>& invA() const { return invA_; }
    //const double& alpha() const { return alpha_; }

    // Debug -> salva il numero di iterazioni (per il test obs ripetute)
    const std::size_t& n_iter_qsrpde() const {return n_iter_qsrpde_; }

    // GCV support
    double norm(const DMatrix<double>& op1, const DMatrix<double>& op2) const {

        double result = 0;

        if(gcv_oss_rip_I_strategy){

            std::cout << "Running GCV per obs ripetute (I approccio)" << std::endl;

            unsigned int n_unique_obs = 0;   // number of unique locations 
            bool new_loc;                    // flag for new location 

            for(std::size_t i = 0; i < n_locs(); ++i) {

                if(i==0){
                    new_loc = true; 
                } else{
                    new_loc = !( almost_equal(Base::locs().coeff(i,0), Base::locs().coeff(i-1,0)) && almost_equal(Base::locs().coeff(i,1), Base::locs().coeff(i-1,1)) ); 
                }

                if(!Base::masked_obs()[i] && new_loc){
                    
                    std::size_t j=0;   // size of the i_th block of locations 
                    double summary_data = 0.;    // statistical summary of the observations in location i 
                    // Compute a summary of the block
                    while(almost_equal(Base::locs().coeff(i,0), Base::locs().coeff(i+j,0)) && almost_equal(Base::locs().coeff(i,1), Base::locs().coeff(i+j,1)) ){    // nb: it always enter the first time 
                        summary_data += op2.coeff(i+j, 0);   // nb: op2 è il dato y 
                        j++; 
                    }
                    summary_data = summary_data / j;   // divide by block size to have the mean of the observations in the block i 
                    if(std::abs(summary_data) < 1e-10){
                        std::cout << "ATT in norm: very small summary" << std::endl; 
                        std::cout << "value=" << summary_data << std::endl; 
                    } 
                        
                    result += pinball_loss(summary_data - op1.coeff(i, 0), std::pow(10, eps_));
                    n_unique_obs++; 
                }
            }
            

            // Adjustment of result to account for repeated data (nb: in questo modo i calcoli in gcv.h sono invariati)
            result *= n_obs() / n_unique_obs; 
        } 

        if(gcv_oss_rip_II_strategy){

            std::cout << "Running GCV per obs ripetute (II approccio)" << std::endl;

            DVector<double> fit_II_approach = skip_repeated_locs(op1);
            DVector<double> summary_vec = compute_summary_data(); 

            for (int i = 0; i < Base::num_unique_locs(); ++i) {
                result += pinball_loss(summary_vec[i] - fit_II_approach(i), std::pow(10, eps_));
            }
        }
        
        if(!gcv_oss_rip_I_strategy && !gcv_oss_rip_II_strategy){
            for (int i = 0; i < n_locs(); ++i) {
                if (!Base::masked_obs()[i]) result += pinball_loss(op2.coeff(i, 0) - op1.coeff(i, 0), std::pow(10, eps_));
            }
            
        }

        return std::pow(result, 2);    // M tolto il fattore di normalizzazione in quanto è stato tolto anche da gcv.h
    }
   

    // GCV support (repeated observations)
    // computes the summary vector of the data 
    DVector<double> compute_summary_data() const{
        DVector<double> summary_vec; 
        summary_vec.resize(Base::num_unique_locs()); 

        unsigned int i = 0;   // index of the first observation in the subgroup
        unsigned int count = 0; 
        while(i < n_locs()){      
            // extract the sub-vector of the observations with same location i 
            std::vector<double> obs_loc_i;
            // sum of the observation in sub-group i 
            double sum = 0.; 
            
            // the first of the group should always be considered
            obs_loc_i.push_back(y()(i));
            sum += y()(i);    
            
            unsigned int j = i+1;    // index runnig on the subgroup starting from its second observation (nb: it is global as i it is)
            bool avvistati_na2 = false; 
            while(j < n_locs() && Base::unique_locs_flags()[j]==0){

                if(std::isnan(y()(j)) && !avvistati_na2){
                    std::cout << "avvistati nana in DATI in compute summary!!" << std::endl; 
                    avvistati_na2 = true; 
                }
                obs_loc_i.push_back(y()(j));  
                sum += y()(j); 
                j++;  
            }
            i = j; 

            avvistati_na2 = false; 
            if(std::isnan(sum) && !avvistati_na2){
                std::cout << "avvistati nana in SUM in compute summary!!" << std::endl; 
                avvistati_na2 = true; 
            }

            if(std::abs(sum / obs_loc_i.size()) < 1e-10){
                std::cout << "ATT in compute summary: very small summary" << std::endl; 
                std::cout << "value=" << sum / obs_loc_i.size() << std::endl; 
            } 
            //std::cout << "obs_loc_i.size() = " << obs_loc_i.size() << std::endl; 
 
            summary_vec(count) = sum / obs_loc_i.size();  
            count++; 

        }

        return summary_vec; 
    }

    // Given the vector v of length n_locs(), returns the vector selecting only the entries correspondent to unique locations
    DVector<double> skip_repeated_locs(DVector<double> v) const{
        
        DVector<double> v_reduced; 
        if(v.size()!=n_locs()){
            std::cout << "ATT: the vector you want to reduce is not of size n*" << std::endl;   
        }
        std::size_t j = 0; 
        v_reduced.resize(Base::num_unique_locs()); 
        for(std::size_t i = 0; i < n_locs(); ++i) {
            if(Base::unique_locs_flags()[i]==1){
                v_reduced(j) = v(i); 
                j++; 
            }
        }

        return v_reduced; 
    }


   private:
    double alpha_ = 0.5;      // quantile order (default to median)
    DVector<double> py_ {};   // y - (1-2*alpha)|y - X*beta - f|
    DVector<double> pW_ {};   // diagonal of W^k = 1/(2*n*|y - X*beta - f|)
    DVector<double> mu_;      // \mu^k = [ \mu^k_1, ..., \mu^k_n ] : quantile vector at step k
    fdapde::SparseLU<SpMatrix<double>> invA_;

    DVector<double> pW_II_approach;   // M 
    DVector<double> py_II_approach;   // M 

    FPIRLS<This> fpirls_;   // fpirls algorithm
    int max_iter_ = 200;    // maximum number of iterations in fpirls before forced stop
    double tol_ = 1e-6;     // fprils convergence tolerance
    double tol_weights_ = 1e-6;

    // Debug -> salva il numero di iterazioni (per il test obs ripetute)
    std::size_t n_iter_qsrpde_ = 0;

    bool gcv_oss_rip_I_strategy = false; 
    bool gcv_oss_rip_II_strategy = false; 

    double eps_ = -1.0;   // pinball loss smoothing factor
    double pinball_loss(double x, double eps) const {   // quantile check function
        return (alpha_ - 1) * x + eps * fdapde::log1pexp(x / eps);
    };
    double pinball_loss(double x) const { return 0.5 * std::abs(x) + (alpha_ - 0.5) * x; }   // non-smoothed pinball

};

}   // namespace models
}   // namespace fdapde

#endif   // __QSRPDE_H__
