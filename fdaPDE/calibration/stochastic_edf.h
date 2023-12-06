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

#include <random>
using fdapde::core::SMW;

#include "../models/regression/regression_base.h"
using fdapde::models::is_regression_model;

namespace fdapde {
namespace calibration {

// computes an approximation of the trace of S = \Psi*T^{-1}*\Psi^T*Q using a monte carlo approximation.
template <typename Model> class StochasticEDF {
    static_assert(is_regression_model<Model>::value);
   private:
    const Model& model_;
    std::size_t r_ = 100;   // number of monte carlo realizations
    std::size_t seed_;
    DMatrix<double> Us_;   // sample from Rademacher distribution
    DMatrix<double> Bs_;   // \Psi^T*Q*Us_
    DMatrix<double> Y_;    // Us_^T*\Psi

    bool init_ = false;
   public:
    // constructor
    StochasticEDF(const Model& model, std::size_t r, std::size_t seed) :
        model_(model), r_(r), seed_((seed == fdapde::random_seed) ? std::random_device()() : seed) { }
    StochasticEDF(const Model& model, std::size_t r) : StochasticEDF(model, r, std::random_device()()) { }

    // evaluate trace of S exploiting a monte carlo approximation
    double compute() {
        std::size_t n = model_.Psi().cols();   // number of basis functions
        if (!init_) {
            // compute sample from Rademacher distribution
            std::default_random_engine rng(seed_);
            std::bernoulli_distribution Be(0.5);   // bernulli distribution with parameter p = 0.5
            Us_.resize(model_.n_obs(), r_);        // preallocate memory for matrix Us
            // fill matrix
            for (std::size_t i = 0; i < model_.n_obs(); ++i) {
                for (std::size_t j = 0; j < r_; ++j) {
                    if (Be(rng))
                        Us_(i, j) = 1.0;
                    else
                        Us_(i, j) = -1.0;
                }
            }
            // prepare matrix Y
            Y_ = Us_.transpose() * model_.Psi();
            init_ = true;   // never reinitialize again
        }
        // prepare matrix Bs_
        Bs_ = DMatrix<double>::Zero(2 * n, r_);
        if (!model_.has_covariates())   // non-parametric model
            Bs_.topRows(n) = -model_.PsiTD() * model_.W() * Us_;
        else   // semi-parametric model
            Bs_.topRows(n) = -model_.PsiTD() * model_.lmbQ(Us_);

        DMatrix<double> sol;              // room for problem solution
        if (!model_.has_covariates()) {   // nonparametric case
            sol = model_.invA().solve(Bs_);
        } else {
            // solve system (A+UCV)*x = Bs via woodbury decomposition using matrices U and V cached by model_
            sol = SMW<>().solve(model_.invA(), model_.U(), model_.XtWX(), model_.V(), Bs_);
        }
        // compute approximated Tr[S] using monte carlo mean
        double MCmean = 0;
        for (std::size_t i = 0; i < r_; ++i) MCmean += Y_.row(i).dot(sol.col(i).head(n));

        return MCmean / r_;
    }

    // setter
    void set_seed(std::size_t seed) { seed_ = seed; }
};

}   // namespace calibration
}   // namespace fdapde

#endif   // __STOCHASTIC_EDF_H__