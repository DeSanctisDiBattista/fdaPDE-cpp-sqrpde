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
    
    std::string gcv_approach_; // M 

    // computes smoothing matrix S = Q*\Psi*T^{-1}*\Psi^T
    const DMatrix<double>& S() {

        // std::cout << "-----------------------EXACT GCV running-------------------------------" << std::endl; 
        // factorize matrix T

        if(gcv_approach_ == "I"){
            std::cout << "smoothing matrix computation with I strategy" << std::endl;
            invT_ = model_.T().partialPivLu();
            DMatrix<double> E_ = model_.PsiTD();    // need to cast to dense for PartialPivLU::solve()
            S_ = model_.lmbQ(model_.Psi() * invT_.solve(E_));   // \Psi*T^{-1}*\Psi^T*Q
        }

        if(gcv_approach_ == "II" || gcv_approach_ == "III"){   // NB: funziona solo space-only (maschere per NA obs non corrette)
            std::cout << "smoothing matrix computation with " << gcv_approach_ << " strategy" << std::endl;
            invT_ = model_.T_reduced().partialPivLu();
            DMatrix<double> E_ = model_.PsiTD_reduced();    // need to cast to dense for PartialPivLU::solve()
            S_ = model_.lmbQ_reduced(model_.Psi_reduced() * invT_.solve(E_));   // \Psi*T^{-1}*\Psi^T*Q
        } 

        if(gcv_approach_ == "IV"){
            std::cout << "smoothing matrix computation with IV strategy" << std::endl;
            invT_ = model_.T().partialPivLu();
            DMatrix<double> E_ = model_.PsiTD_reduced();    // need to cast to dense for PartialPivLU::solve()
            S_ = model_.lmbQ_reduced(model_.Psi_reduced() * invT_.solve(E_));   // \Psi*T^{-1}*\Psi^T*Q
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
    void gcv_approach_set_trace(std::string gcv_approach) { 
        std::cout << "Setting strategy GCV in exact_edf.h" << std::endl; 
        std::cout << "string value = " << gcv_approach << std::endl; 
        gcv_approach_ = gcv_approach; 
    }; 

    
};

}   // namespace models
}   // namespace fdapde

#endif   // __EXACT_EDF_H__
