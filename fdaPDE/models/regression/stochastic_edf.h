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

#ifndef __STOCHASTIC_EDF_H__
#define __STOCHASTIC_EDF_H__

#include <fdaPDE/linear_algebra.h>
#include <fdaPDE/utils.h>
#include "regression_type_erasure.h"

#include <random>
using fdapde::core::SMW;

namespace fdapde {
namespace models {
  
// computes an approximation of the trace of S = \Psi*T^{-1}*\Psi^T*Q using a monte carlo approximation.
class StochasticEDF {
   private:
    RegressionView<void> model_;
    int r_ = 100;   // number of monte carlo realizations
    DMatrix<double> Us_;    // sample from Rademacher distribution
    DMatrix<double> Bs_;    // \Psi^T*Q*Us_
    DMatrix<double> Y_;     // Us_^T*\Psi
    int seed_ = fdapde::random_seed;
    bool init_ = false;
    std::string gcv_approach_; // M 
   public:
    // constructor
    StochasticEDF(int r, int seed) :
        r_(r), seed_((seed == fdapde::random_seed) ? std::random_device()() : seed) { }
    StochasticEDF(int r) : StochasticEDF(r, std::random_device()()) { }
    StochasticEDF() : StochasticEDF(100) { }
  
    // evaluate trace of S exploiting a monte carlo approximation
    double compute() {
        if (!init_) {
            std::cout << "-----------------------STOCHASTIC GCV running-------------------------------" << std::endl; 
            // compute sample from Rademacher distribution
            std::mt19937 rng(seed_);
            std::bernoulli_distribution Be(0.5);   // bernulli distribution with parameter p = 0.5

            //std::cout << "in stochastic_edf: gcv_approach_=" << gcv_approach_ << std::endl; 
            if(gcv_approach_ == "first"){
                Us_.resize(model_.n_locs(), r_);       // preallocate memory for matrix Us
                // fill matrix
                for (int i = 0; i < model_.n_locs(); ++i) {
                    for (int j = 0; j < r_; ++j) {
                        if (Be(rng))
                            Us_(i, j) = 1.0;
                        else
                            Us_(i, j) = -1.0;
                    }
                }
                // prepare matrix Y
                Y_ = Us_.transpose() * model_.Psi();
            }

            if(gcv_approach_ == "II"){
                std::cout << "Trace computation (stoch) with II approach" << std::endl; 
                Us_.resize(model_.num_unique_locs(), r_);       // preallocate memory for matrix Us
                // fill matrix
                for (int i = 0; i < model_.num_unique_locs(); ++i) {
                    for (int j = 0; j < r_; ++j) {
                        if (Be(rng))
                            Us_(i, j) = 1.0;
                        else
                            Us_(i, j) = -1.0;
                    }
                }
                // prepare matrix Y
                Y_ = Us_.transpose() * model_.Psi_reduced();

            }

            if(gcv_approach_ == "III" || gcv_approach_ == "IV"){
                std::cout << "ATT: in stochastic edf manca implementazione III e IV !!!" << std::endl; 
            }

            init_ = true;   // never reinitialize again
        }
        // prepare matrix Bs_
        int n = model_.n_basis();
        Bs_ = DMatrix<double>::Zero(2 * n, r_);

        if(gcv_approach_ == "first"){
            //std::cout << "in stochastic_edf.h first approach" << std::endl; 
            if (!model_.has_covariates())   // non-parametric model
                Bs_.topRows(n) = -model_.PsiTD() * model_.W() * Us_;
            else   // semi-parametric model
                Bs_.topRows(n) = -model_.PsiTD() * model_.lmbQ(Us_);   
        }

        if(gcv_approach_ == "II"){
            if (!model_.has_covariates())   // non-parametric model
                Bs_.topRows(n) = -model_.PsiTD_reduced() * model_.W_reduced() * Us_;
            else   // semi-parametric model
                Bs_.topRows(n) = -model_.PsiTD_reduced() * model_.lmbQ_reduced(Us_);   
        }

        if(gcv_approach_ == "III" || gcv_approach_ == "IV"){
            std::cout << "ATT: in stochastic edf manca implementazione III e IV !!!" << std::endl; 
        }


        DMatrix<double> sol;              // room for problem solution

        if(gcv_approach_ == "first"){
            if (!model_.has_covariates()) {   // nonparametric case
                sol = model_.invA().solve(Bs_);
            } else {
                // solve system (A+UCV)*x = Bs via woodbury decomposition using matrices U and V cached by model_
                sol = SMW<>().solve(model_.invA(), model_.U(), model_.XtWX(), model_.V(), Bs_);
            }
        }

        if(gcv_approach_ == "II"){
            if (!model_.has_covariates()) {   // nonparametric case
                sol = model_.invA_reduced().solve(Bs_);
            } else {
                // solve system (A+UCV)*x = Bs via woodbury decomposition using matrices U and V cached by model_
                sol = SMW<>().solve(model_.invA_reduced(), model_.U_reduced(), model_.XtWX_reduced(), model_.V_reduced(), Bs_);
            }
        } 

        if(gcv_approach_ == "III" || gcv_approach_ == "IV"){
            std::cout << "ATT: in stochastic edf manca implementazione III e IV !!!" << std::endl; 
        }

        // compute approximated Tr[S] using monte carlo mean
        double MCmean = 0;
        for (int i = 0; i < r_; ++i) MCmean += Y_.row(i).dot(sol.col(i).head(n));
        return MCmean / r_;
    }
    // setter
    void set_model(RegressionView<void> model) { model_ = model; }
    void set_seed(int seed) { seed_ = seed; }
    void set_n_mc_samples(int r) { r_ = r; }

    // M 
    const DMatrix<double>& S_get() const { 
        std::cout << "ATTENTION: returning a ficticious value for the smoothing matrix" << std::endl; 
        return Us_; 
    }   // M : ficticious, just to compile, since we need this method in exact_edf

    // M 
    void gcv_approach_set_trace(const std::string& gcv_approach) { 
        //std::cout << "Setting strategy GCV in stochastic_edf.h" << std::endl; 
        //std::cout << "string value = " << gcv_approach << std::endl; 
        gcv_approach_ = gcv_approach; 
    }; 
};

}   // namespace models
}   // namespace fdapde

#endif   // __STOCHASTIC_EDF_H__
