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

#ifndef __SRPDE_H__
#define __SRPDE_H__

#include <fdaPDE/linear_algebra.h>
#include <fdaPDE/pde.h>
#include <fdaPDE/utils.h>

#include <memory>
#include <type_traits>

#include "../model_base.h"
#include "../model_macros.h"
#include "../sampling_design.h"
#include "regression_base.h"
using fdapde::core::SMW;
using fdapde::core::SparseBlockMatrix;

namespace fdapde {
namespace models {

class SRPDE : public RegressionBase<SRPDE, SpaceOnly> {
   private:
    using Base = RegressionBase<SRPDE, SpaceOnly>;
    SparseBlockMatrix<double, 2, 2> A_ {};         // system matrix of non-parametric problem (2N x 2N matrix)  
    fdapde::SparseLU<SpMatrix<double>> invA_ {};   // factorization of matrix A
    DVector<double> b_ {};                         // right hand side of problem's linear system (1 x 2N vector)
    SpMatrix<double> P1_{}; // ficticious 

    bool gcv_oss_rip_I_strategy = true; 
    bool gcv_oss_rip_II_strategy = false; 
    bool gcv_oss_rip_III_strategy = false; 
    bool gcv_oss_rip_IV_strategy = false; 
    
   public:
    IMPORT_REGRESSION_SYMBOLS
    using Base::lambda_D;   // smoothing parameter in space
    using Base::n_basis;    // number of spatial basis
    using Base::runtime;    // runtime model status
    using RegularizationType = SpaceOnly;
    using This = SRPDE;
    static constexpr int n_lambda = 1;
    // constructor
    SRPDE() = default;
    SRPDE(const Base::PDE& pde, Sampling s) : Base(pde, s) {};

    void init_model() {
        if (runtime().query(runtime_status::is_lambda_changed)) {
            // assemble system matrix for nonparameteric part
            A_ = SparseBlockMatrix<double, 2, 2>(
              -PsiTD() * W() * Psi(), lambda_D() * R1().transpose(),   
	      lambda_D() * R1(),      lambda_D() * R0()            );

            invA_.compute(A_);

            // prepare rhs of linear system
            b_.resize(A_.rows());
            b_.block(n_basis(), 0, n_basis(), 1) = lambda_D() * u();
            return;
        }
        if (runtime().query(runtime_status::require_W_update)) {
            // adjust north-west block of matrix A_ only
            A_.block(0, 0) = -PsiTD() * W() * Psi();
            invA_.compute(A_);
            return;
        }
    }
    void solve() {
        fdapde_assert(y().rows() != 0);
        DVector<double> sol;
        if (!has_covariates()) {   // nonparametric case
            // update rhs of SR-PDE linear system
            b_.block(0, 0, n_basis(), 1) = -PsiTD() * W() * y();   
            // solve linear system A_*x = b_
            sol = invA_.solve(b_);
            f_ = sol.head(n_basis());
        } else {   // parametric case
            // update rhs of SR-PDE linear system
            b_.block(0, 0, n_basis(), 1) = -PsiTD() * lmbQ(y());   // -\Psi^T*D*Q*z
            // matrices U and V for application of woodbury formula
            U_ = DMatrix<double>::Zero(2 * n_basis(), q());
            U_.block(0, 0, n_basis(), q()) = PsiTD() * W() * X();
            V_ = DMatrix<double>::Zero(q(), 2 * n_basis());
            V_.block(0, 0, q(), n_basis()) = X().transpose() * W() * Psi();
            // solve system (A_ + U_*(X^T*W_*X)*V_)x = b using woodbury formula from linear_algebra module
            sol = SMW<>().solve(invA_, U_, XtWX(), V_, b_);
            // store result of smoothing
            f_ = sol.head(n_basis());
            beta_ = invXtWX().solve(X().transpose() * W()) * (y() - Psi() * f_);
        }
        // store PDE misfit 
        g_ = sol.tail(n_basis());

        // M 
        if(!gcv_oss_rip_I_strategy){
            std::cout << "Computing W in solve srpde for GCV reduced..." << std::endl; 

            // for(std::size_t i = 0; i < Base::num_unique_locs(); ++i){
            //     Base::W_reduced_set(i, 1.0);   // identity matrix 
            // }  
            std::cout << "setting normal W_reduced" << std::endl; 
            Base::W_reduced_set(DVector<double>::Ones(Base::num_unique_locs()).asDiagonal());  
            
            std::cout << "Computing invA in solve srpde for GCV reduced..." << std::endl; 
            Base::invA_reduced_set(); 
            if(Base::has_covariates()) {
                // Set XtWX_reduced_set and its inverse in base class 
                std::cout << "in solve, setting XtWX, U, V reduced" << std::endl; 
                Base::XtWX_reduced_set(X_reduced().transpose()*W_reduced()*X_reduced()); 
                Base::U_reduced_set(); 
                Base::V_reduced_set(); 
            }
        } 


        return;
    }
    // GCV support
    double norm(const DMatrix<double>& op1, const DMatrix<double>& op2) const { 
        
        double result = 0.; 
        if(!gcv_oss_rip_I_strategy){
            std::cout << "Running GCV per obs ripetute in SRPDE" << std::endl; 
            DVector<double> fit_reduced = skip_repeated_locs(op1);
            DVector<double> summary_vec = compute_summary_data(); 
            result = (fit_reduced - summary_vec).squaredNorm();  
        } else{
            std::cout << "Running GCV per obs uniche in SRPDE" << std::endl;
            result = (op1 - op2).squaredNorm();  
        }
        
        return result;    
    }
    
    // M 
    void gcv_oss_rip_strategy_set(const std::string str){

        gcv_oss_rip_I_strategy = (str=="I"); 
        gcv_oss_rip_II_strategy = (str=="II"); 
        gcv_oss_rip_III_strategy = (str=="III"); 
        gcv_oss_rip_IV_strategy = (str=="IV"); 

        Base::gcv_approach_set(str); 
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
    
    // getters
    const SparseBlockMatrix<double, 2, 2>& A() const { return A_; }
    const fdapde::SparseLU<SpMatrix<double>>& invA() const { return invA_; }
    const SpMatrix<double>& P1() const { return P1_; }   // ficticious (otherwise compile error in regression_wrappers)

    virtual ~SRPDE() = default;
};

}   // namespace models
}   // namespace fdapde

#endif   // __SRPDE_H__
