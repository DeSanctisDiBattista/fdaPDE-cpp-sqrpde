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

#ifndef __EXACT_EDF_H__
#define __EXACT_EDF_H__

#include <fdaPDE/utils.h>
#include "regression_type_erasure.h"

namespace fdapde {
namespace models {
  
// Evaluates exactly the trace of matrix S = \Psi*T^{-1}*\Psi^T*Q. Uses the cyclic property of the trace
// operator: Tr[S] = Tr[\Psi*T^{-1}*\Psi^T*Q] = Tr[Q*\Psi*T^{-1}*\Psi^T]
class ExactEDF {
   private:
    RegressionView<void> model_;
    bool gcv_2_approach_ = false; // M 
    // computes smoothing matrix S = Q*\Psi*T^{-1}*\Psi^T
    const DMatrix<double>& S() {
        // std::cout << "-----------------------EXACT GCV running-------------------------------" << std::endl; 
        // factorize matrix T

        if(gcv_2_approach_){   // NB: funziona solo nel caso NONparametrico (matrici di SMW non corrette) e space-only (maschere per NA obs non corrette)
            std::cout << "smoothing matrix computation with summarized locs" << std::endl; 
            invT_ = model_.T_II_approach().partialPivLu();
            //std::cout << "max norm invT= " << invT_.maxCoeff() << std::endl;
            DMatrix<double> E_ = model_.PsiTD_II_approach();    // need to cast to dense for PartialPivLU::solve()
            //std::cout << "max norm E= " << E_.maxCoeff() << std::endl; 
            //std::cout << "max norm Psi= " << model_.Psi_II_approach().maxCoeff() << std::endl; 
            //std::cout << "max norm W= " << model_.W_II_approach().diagonal().maxCoeff() << std::endl; 
            S_ = model_.lmbQ_II_approach(model_.Psi_II_approach() * invT_.solve(E_));   // \Psi*T^{-1}*\Psi^T*Q
            //std::cout << "max norm S= " << S_.maxCoeff() << std::endl; 
        } else{
            std::cout << "smoothing matrix computation with all locs" << std::endl; 
            invT_ = model_.T().partialPivLu();
            DMatrix<double> E_ = model_.PsiTD();    // need to cast to dense for PartialPivLU::solve()
            S_ = model_.lmbQ(model_.Psi() * invT_.solve(E_));   // \Psi*T^{-1}*\Psi^T*Q
        }

        return S_;
    };
   public:
    Eigen::PartialPivLU<DMatrix<double>> invT_ {};   // T^{-1}
    DMatrix<double> S_ {};                           // \Psi*T^{-1}*\Psi^T*Q = \Psi*V_

    ExactEDF() = default;
    double compute() { return S().trace(); }   // computes Tr[S]
    void set_model(const RegressionView<void> model) { model_ = model; }

    // M 
    const DMatrix<double>& S_get() const { return S_; }   // return S

    // M 
    void gcv_2_approach_set_trace(bool gcv_approach) { 
        std::cout << "Setting strategy GCV in exact_edf.h" << std::endl; 
        std::cout << "boolean value = " << gcv_approach << std::endl; 
        gcv_2_approach_ = gcv_approach; 
    }; 
    
};

}   // namespace models
}   // namespace fdapde

#endif   // __EXACT_EDF_H__
