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

    SpMatrix<double> Psi_reduced_;   
    SpMatrix<double> PsiTD_reduced_;  
    DMatrix<double> X_reduced_;   

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

            //std::cout << "dim loc_ in init_sampling " << locs_.rows() << ";" << locs_.cols() << std::endl; 
            // std::cout << "print head of locs_ in init_sampling" << std::endl; 
            // for(int i=0; i<4; ++i){
            //     std::cout << "loc_" << i << " = " << locs_(i,0) << "," << locs_(i,1) << std::endl; 
            // }

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
        unique_locs_flags_.resize(locs_.rows()); 
        bool new_loc; 
        std::vector<unsigned int> idxs_repeated_locs_;
        for(std::size_t i = 0; i < locs_.rows(); ++i) {
            if(i==0){
                new_loc = true;  
            } else{
                new_loc = !( almost_equal(locs_.coeff(i,0), locs_.coeff(i-1,0)) && almost_equal(locs_.coeff(i,1), locs_.coeff(i-1,1)) ); 
            }
            unique_locs_flags_[i] = new_loc;  

            if(!new_loc)
                idxs_repeated_locs_.push_back(i); 
        }

        // compute num_unique_locs_
        num_unique_locs_ = 0; 
        for(auto idx = 0; idx < unique_locs_flags_.size(); ++idx){
            if(unique_locs_flags_[idx] == 1){
                num_unique_locs_++; 
            }
        }     

        num_obs_per_location_.resize(num_unique_locs_); 
        unsigned int obs_counter = 1;  // because the first is skipped
        unsigned int idx_counter = 0; 
        for(auto idx = 1; idx < unique_locs_flags_.size(); ++idx){  // skip the first since always true
            if(unique_locs_flags_[idx] == 1){
                //std::cout << "obs in loc " << idx_counter+1 << "=" << obs_counter << std::endl; 
                num_obs_per_location_[idx_counter] = obs_counter; 
                idx_counter++; 
                obs_counter = 0;
            } 
            obs_counter++; 
        }      
        //std::cout << "obs in loc " << num_unique_locs_ << "=" << obs_counter << std::endl; 
        num_obs_per_location_[num_unique_locs_-1] = obs_counter; 

        //compute the reduced Psi matrix in case of repeated observations
        if(num_unique_locs() != n_spatial_locs()){
        
            Psi_reduced_ = remove_rows(Psi_); 
            PsiTD_reduced_ = remove_rows(SpMatrix<double>(PsiTD_.transpose())).transpose(); // per PsiTD dobbiamo rimuovere colonne -> passo trasposto e poi rifacciamo trasposto
            
            if(model().has_covariates()){
                X_reduced_ = remove_rows(model().X(), idxs_repeated_locs_);
            }

        }

    }

    // getters
    const SpMatrix<double>& Psi(not_nan) const { return Psi_; }
    const SpMatrix<double>& PsiTD(not_nan) const { return PsiTD_; }

    // M 
    const SpMatrix<double>& Psi_reduced(not_nan) const { return Psi_reduced_; }
    const SpMatrix<double>& PsiTD_reduced(not_nan) const {return PsiTD_reduced_; }
    const DMatrix<double>& X_reduced() const {return X_reduced_; }
    const std::vector<bool>& unique_locs_flags() const { return unique_locs_flags_; }
    const DVector<double>& num_obs_per_location() const {return num_obs_per_location_; }


    // M 
    const unsigned int num_unique_locs() const { 
        return num_unique_locs_;
    }

    const DMatrix<double> remove_rows(DMatrix<double> mat, std::vector<unsigned int> row_indexes_to_remove) {
        // Given the matrix mat, and a vector of row indexes row_indexes_to_remove, returns the matrix 
        // without the rows in row_indexes_to_remove
        for(unsigned int rowToRemove : row_indexes_to_remove){
            if(rowToRemove < mat.rows()){
                removeRow(mat, rowToRemove); 
                // decrease the indexes to remove by one since a row has been eliminated
                std::transform(std::begin(row_indexes_to_remove),std::end(row_indexes_to_remove),std::begin(row_indexes_to_remove),[](unsigned int x){return x-1;});
                // note: the previous indexes to remove are currupted by this, since they are also decremented by one, but they are not used anymore

            } else{
                std::cout << "Error: row to remove exceeds the matrix size" << std::endl; 
            }        
        }

        return(mat); 
    }


    void removeRow(DMatrix<double>& matrix, unsigned int rowToRemove){
        unsigned int numRows = matrix.rows()-1;
        unsigned int numCols = matrix.cols();

        if(rowToRemove < numRows)
            matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

        matrix.conservativeResize(numRows,numCols);
    }

    const SpMatrix<double> remove_rows(SpMatrix<double> mat) {
        // Given the sparse matrix mat returns the matrix with rows removed according to unique_locs_flags_
        SpMatrix<double> mat_ret; 
        mat_ret.resize(num_unique_locs(), mat.cols()); 
        unsigned int count = 0; 

        std::vector<fdapde::Triplet<double>> triplet_list;
        triplet_list.reserve(num_unique_locs() * mat.cols());

        for(std::size_t i = 0; i < locs_.rows(); ++i) {

            if(unique_locs_flags_[i]){            
                for(int j = 0; j < mat.cols(); ++j)
                    triplet_list.emplace_back(count, j, mat.coeff(i, j));
            
                count ++;
            }
        
        }

        mat_ret.setFromTriplets(triplet_list.begin(), triplet_list.end());
        mat_ret.makeCompressed();


        return(mat_ret); 
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
