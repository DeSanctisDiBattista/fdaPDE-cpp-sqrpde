#ifndef __ELEMENT_H__
#define __ELEMENT_H__

#include "../OPT/Utils.h"
#include "MeshUtils.h"
#include "Geometry.h"

#include <Eigen/LU>
#include <cstddef>
#include <limits>
#include <memory>
#include <iostream>
#include <type_traits>
#include <array>

namespace fdaPDE{
namespace core{
namespace MESH{
  
  template <unsigned int M, unsigned int N> class Element; // forward declaration

  // using declarations for manifold specialization
  using SurfaceElement = Element<2,3>;
  using NetworkElement = Element<1,2>;

  // A mesh element
  template <unsigned int M, unsigned int N>
  class Element{
  private:
    unsigned int ID;                                     // ID of this element
    std::array<SVector<N>, N_VERTICES(M,N)> coords;      // coordinates of vertices as N-dimensional points
    std::array<unsigned int, M+1> neighbors;             // ID of the neighboring elements

    // Functional information to assist the FEM module. FEsupport contains pairs <ID, point> where
    //    * ID:    is the global ID of the node in the mesh (as row index in the points_ table)
    //    * point: are the node's coordinates
    // Observe that 'FESupport' is different from 'coords' for elements of order greater than one: in this case
    // the nodes to support a functional basis are more than the geometrical vertices of the element. From here
    // the need to keep the geometrical informations separate from the functional ones.
    std::array<std::pair<unsigned, SVector<N>>, N_VERTICES(M,N)> FEsupport;
    
    // matrix defining the affine transformation from cartesian to barycentric coordinates and viceversa
    Eigen::Matrix<double, N, M> baryMatrix;
    Eigen::Matrix<double, M, N> invBaryMatrix{};

    // compute the inverse of the barycentric matrix. This will be specialized by manifold elements
    Eigen::Matrix<double, M, N> computeInvBaryMatrix(const Eigen::Matrix<double, N, M>& baryMatrix) {
      return baryMatrix.inverse();
    };
  
  public:
    Element() = default;
  
    Element(int ID_, std::array<std::pair<unsigned, SVector<N>>, N_VERTICES(M,N)> FEsupport_,
	    std::array<SVector<N>, N_VERTICES(M,N)> coords_, std::array<unsigned int, M+1> neighbors_);

    // getters
    std::array<std::pair<unsigned, SVector<N>>, N_VERTICES(M,N)> getFESupport() const { return FEsupport; }
    std::array<SVector<N>, N_VERTICES(M,N)> getCoords()  const { return coords;    }
    std::array<unsigned int, M+1> getNeighbors()         const { return neighbors; }
    unsigned int getID()                                 const { return ID;        }

    // computes the baricentric coordinates of the element
    SVector<M+1> computeBarycentricCoordinates(const SVector<N>& x) const;

    // check if a given point is inside the element
    template <bool is_manifold = (N!=M)> typename std::enable_if<!is_manifold, bool>::type
    contains(const SVector<N>& x) const;

    // specialization of contains() for manifold meshes
    template <bool is_manifold = (N!=M)> typename std::enable_if< is_manifold, bool>::type
    contains(const SVector<N>& x) const;

    // compute bounding box of this element. A bounding box is the smallest rectangle containing the element
    // this construction is usefull in searching problems.
    std::pair<SVector<N>, SVector<N>> computeBoundingBox() const;
  };

#include "Element.tpp"
}}}

#endif // __ELEMENT_H__