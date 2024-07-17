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

#ifndef __SAMPLING_DESIGN_H__
#define __SAMPLING_DESIGN_H__

#include <fdaPDE/linear_algebra.h>
#include <fdaPDE/geometry.h>
#include <fdaPDE/utils.h>
#include <fdaPDE/pde.h>
using fdapde::core::Kronecker;

#include "model_base.h"
#include "model_macros.h"
#include "model_traits.h"

namespace fdapde {
namespace models {

struct not_nan { };                               // tag to request the not-NaN corrected version of matrix \Psi
enum Sampling { mesh_nodes, pointwise, areal };   // supported sampling strategies

// base class for the implemetation of the different sampling designs.
// Here is computed the matrix of spatial basis evaluations \Psi = [\Psi]_{ij} = \psi_i(p_j) or its tensorization
// (for space-time problems)
template <typename Model> class SamplingBase {
    Sampling sampling_;
   protected:
    FDAPDE_DEFINE_MODEL_GETTER;    // import model() method (const and non-const access)
    SpMatrix<double> Psi_;         // n x N matrix \Psi = [\psi_{ij}] = \psi_j(p_i)
    SpMatrix<double> PsiTD_;       // N x n block \Psi^T*D, being D the matrix of subdomain measures

    SpMatrix<double> Psi_II_approach_;   
    SpMatrix<double> PsiTD_II_approach_;    
    DMatrix<double> X_II_approach_;   

    DiagMatrix<double> D_;         // for areal sampling, diagonal matrix of subdomains' measures, D_ = I_n otherwise
    DMatrix<double> locs_;         // matrix of spatial locations p_1, p2_, ... p_n, or subdomains D_1, D_2, ..., D_n

    std::vector<bool> unique_locs_flags_; // zeros identify repeated locations 
    unsigned int num_unique_locs_ = 0;
    DVector<double> num_obs_per_location_;  // number of observations in each location

    // for space-time models, perform a proper tensorization of matrix \Psi
    void tensorize_psi() {
        if constexpr (is_space_time_separable<Model>::value) Psi_ = Kronecker(model().Phi(), Psi_);
	if constexpr (is_space_time_parabolic<Model>::value) {
	    SpMatrix<double> Im(model().n_temporal_locs(), model().n_temporal_locs());   // m x m identity matrix
            Im.setIdentity();
            Psi_ = Kronecker(Im, Psi_);
	}
    }
   public:
    SamplingBase() = default;
    SamplingBase(Sampling sampling) : sampling_(sampling) { }
  
    void init_sampling(bool forced = false) {
        // compute once if not forced to recompute
        if (!is_empty(Psi_) && forced == false) return;
	
        switch (sampling_) {
        case Sampling::mesh_nodes: {  // data sampled at mesh nodes: \Psi is the identity matrix
            // preallocate space for Psi matrix
            int n = model().n_spatial_basis();
            int N = model().n_spatial_basis();
            Psi_.resize(n, N);
            std::vector<fdapde::Triplet<double>> triplet_list;
            triplet_list.reserve(n);
            // if data locations are equal to mesh nodes then \Psi is the identity matrix.
            // \psi_i(p_i) = 1 and \psi_i(p_j) = 0 \forall i \neq j
            for (int i = 0; i < n; ++i) triplet_list.emplace_back(i, i, 1.0);
            // finalize construction
            Psi_.setFromTriplets(triplet_list.begin(), triplet_list.end());
            Psi_.makeCompressed();
            model().tensorize_psi();   // tensorize \Psi for space-time problems
            PsiTD_ = Psi_.transpose();
            D_ = DVector<double>::Ones(Psi_.rows()).asDiagonal();
  	  } return;
        case Sampling::pointwise: {   // data sampled at general locations p_1, p_2, ... p_n
            // query pde to evaluate functional basis at given locations
            auto basis_evaluation = model().pde().eval_basis(core::eval::pointwise, locs_);
            Psi_ = basis_evaluation->Psi;
            model().tensorize_psi();   // tensorize \Psi for space-time problems
            D_ = DVector<double>::Ones(Psi_.rows()).asDiagonal();
	    PsiTD_ = Psi_.transpose();
	  } break;
        case Sampling::areal: {   // data sampled at subdomains D_1, D_2, ... D_d
            // query pde to evaluate functional basis at given locations
            auto basis_evaluation = model().pde().eval_basis(core::eval::areal, locs_);
            Psi_ = basis_evaluation->Psi;
            model().tensorize_psi();   // tensorize \Psi for space-time problems

            // here we must distinguish between space-only and space-time models
            if constexpr (is_space_time<Model>::value) {
                // store I_m \kron D
                int m = model().n_temporal_locs();
                int n = n_spatial_locs();
                DVector<double> IkronD(n * m);
                for (int i = 0; i < m; ++i) IkronD.segment(i * n, n) = basis_evaluation->D;
                // compute and store result
                D_ = IkronD.asDiagonal();
            } else {
                // for space-only problems store diagonal matrix D_ = diag(D_1, D_2, ... ,D_d) as it is
                D_ = basis_evaluation->D.asDiagonal();
            }
            PsiTD_ = Psi_.transpose() * D_;
	  } break;
        }

        // M:
        // fill the unique_locs_flags_ vector
        bool new_loc; 
        for(std::size_t i = 0; i < locs_.rows(); ++i) {
            if(i==0){
                new_loc = true;  
            } else{
                new_loc = !( almost_equal(locs_.coeff(i,0), locs_.coeff(i-1,0)) && almost_equal(locs_.coeff(i,1), locs_.coeff(i-1,1)) ); 
            }
            unique_locs_flags_.push_back(new_loc);  
        }

        // compute num_unique_locs_
        if(num_unique_locs_ == 0){  // non ancora calcolato
            for(auto idx = 0; idx < unique_locs_flags_.size(); ++idx){
                if(unique_locs_flags_[idx] == 1){
                    num_unique_locs_++; 
                }
            }
        } 

        num_obs_per_location_.resize(num_unique_locs_); 
        unsigned int obs_counter = 1;  // because the first is skipped
        unsigned int idx_counter = 0; 
        for(auto idx = 1; idx < unique_locs_flags_.size(); ++idx){  // skip the first since always true
            if(unique_locs_flags_[idx] == 1){
                std::cout << "obs in loc " << idx_counter+1 << "=" << obs_counter << std::endl; 
                num_obs_per_location_[idx_counter] = obs_counter; 
                idx_counter++; 
                obs_counter = 0;
            } 
            obs_counter++; 
        }
        std::cout << "obs in loc " << num_unique_locs_ << "=" << obs_counter << std::endl; 
        num_obs_per_location_[num_unique_locs_-1] = obs_counter; 


        //compute the reduced Psi matrix in case of repeated observations
        if(num_unique_locs() != n_spatial_locs()){
            //std::cout << "unique locs=" << num_unique_locs() << std::endl; 

            Psi_II_approach_.resize(num_unique_locs(), model().n_spatial_basis()); 
            PsiTD_II_approach_.resize(model().n_spatial_basis(), num_unique_locs()); 
            unsigned int count = 0; 

            std::vector<fdapde::Triplet<double>> triplet_list_psi;
            triplet_list_psi.reserve(num_unique_locs() * Psi_.cols());

            std::vector<fdapde::Triplet<double>> triplet_list_psiTD;
            triplet_list_psiTD.reserve(num_unique_locs() * PsiTD_.rows());

            for(std::size_t i = 0; i < locs_.rows(); ++i) {

                if(unique_locs_flags_[i]){
                    
                    for(int j = 0; j < Psi_.cols(); ++j){
                        triplet_list_psi.emplace_back(count, j, Psi_.coeff(i, j));
                    }
                    
                    for(int j = 0; j < PsiTD_.rows(); ++j){
                        triplet_list_psiTD.emplace_back(j, count, PsiTD_.coeff(j, i));
                    }  // fatto separatamente e non come il trasposto di Psi_II_approach_ così che abbiamo già dentro le eventuali D

                    count ++;
                }
            
            }

            Psi_II_approach_.setFromTriplets(triplet_list_psi.begin(), triplet_list_psi.end());
            Psi_II_approach_.makeCompressed();

            PsiTD_II_approach_.setFromTriplets(triplet_list_psiTD.begin(), triplet_list_psiTD.end());
            PsiTD_II_approach_.makeCompressed();

            // // check 
            // double maxPsi = 0.; 
            // for(int i = 0; i < Psi_II_approach_.rows(); ++i){
            //     for(int j = 0; j < Psi_II_approach_.cols(); ++j){
            //         if(Psi_II_approach_.coeff(i,j) > maxPsi){
            //             maxPsi = Psi_II_approach_.coeff(i,j); 
            //         }
            //         if(std::isnan(Psi_II_approach_.coeff(i,j)))
            //             std::cout << "avvistati nana in Psi_II_approach_!!" << std::endl; 
            //     }
            // }
            // double maxPsiTD = 0.; 
            // for(int i = 0; i < PsiTD_II_approach_.rows(); ++i){
            //     for(int j = 0; j < PsiTD_II_approach_.cols(); ++j){
            //         if(PsiTD_II_approach_.coeff(i,j) > maxPsiTD){
            //             maxPsiTD = PsiTD_II_approach_.coeff(i,j); 
            //         }
            //         if(std::isnan(PsiTD_II_approach_.coeff(i,j)))
            //             std::cout << "avvistati nana in PsiTD_II_approach_!!" << std::endl; 
            //     }
            // }

            if(model().has_covariates()){
                X_II_approach_.resize(num_unique_locs(), model().q());
                count = 0; 
                for(std::size_t i = 0; i < locs_.rows(); ++i) {
                    if(unique_locs_flags_[i]){
                        for(int j = 0; j < model().X().cols(); ++j){
                            X_II_approach_(count,j) = model().X().coeff(i, j);
                        }
                        count ++;
                    }

                }

                // check 
                double max_X = 0.; 
                for(int i = 0; i < X_II_approach_.rows(); ++i){
                    for(int j = 0; j < X_II_approach_.cols(); ++j){
                        if(X_II_approach_.coeff(i,j) > max_X){
                            max_X = X_II_approach_.coeff(i,j); 
                        }
                        if(std::isnan(X_II_approach_.coeff(i,j)))
                            std::cout << "avvistati nana in X_II_approach_!!" << std::endl; 
                    }
                }
            }
 
        }
 
    }

    // getters
    const SpMatrix<double>& Psi(not_nan) const { return Psi_; }
    const SpMatrix<double>& PsiTD(not_nan) const { return PsiTD_; }

    // M 
    const SpMatrix<double>& Psi_II_approach(not_nan) const { return Psi_II_approach_; }
    const SpMatrix<double>& PsiTD_II_approach(not_nan) const {return PsiTD_II_approach_; }
    const DMatrix<double>& X_II_approach() const {return X_II_approach_; }
    const std::vector<bool>& unique_locs_flags() const { return unique_locs_flags_; }

    const DVector<double>& num_obs_per_location() const {return num_obs_per_location_; }


    // M 
    const unsigned int num_unique_locs() const { 
        return num_unique_locs_;
    }


    int n_spatial_locs() const {
        return sampling_ == Sampling::mesh_nodes ? model().pde().n_dofs() : locs_.rows();
    }
    const DiagMatrix<double>& D() const { return D_; }
    DMatrix<double> locs() const { return sampling_ == Sampling::mesh_nodes ? model().pde().dof_coords() : locs_; }

    // setters
    void set_spatial_locations(const DMatrix<double>& locs) {
        if (sampling_ == Sampling::mesh_nodes) { return; }   // avoid a useless copy
        locs_ = locs;
    }
    Sampling sampling() const { return sampling_; }
    
};

}   // namespace models
}   // namespace fdapde

#endif   // __SAMPLING_DESIGN_H__
