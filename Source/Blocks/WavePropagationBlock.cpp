/**
 * @file
 * This file is part of SWE.
 *
 * @author Alexander Breuer (breuera AT in.tum.de, http://www5.in.tum.de/wiki/index.php/Dipl.-Math._Alexander_Breuer)
 * @author Sebastian Rettenberger (rettenbs AT in.tum.de,
 * http://www5.in.tum.de/wiki/index.php/Sebastian_Rettenberger,_M.Sc.)
 * @author Michael Bader (bader AT in.tum.de, http://www5.in.tum.de/wiki/index.php/Michael_Bader)
 *
 * @section LICENSE
 *
 * SWE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SWE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SWE.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 * @section DESCRIPTION
 *
 * Implementation of Blocks::Block that uses solvers in the wave propagation formulation.
 */

#include "WavePropagationBlock.hpp"

#include <cmath>
#include <iostream>

Blocks::WavePropagationBlock::WavePropagationBlock(
    int nx, int ny, RealType dx, RealType dy, std::shared_ptr<Solvers::WavePropagationSolver<RealType>> solver
):
    Block(nx, ny, dx, dy),
    hNetUpdatesLeft_(nx + 1, ny, STRIDE),
    hNetUpdatesRight_(nx + 1, ny, STRIDE),
    huNetUpdatesLeft_(nx + 1, ny, STRIDE),
    huNetUpdatesRight_(nx + 1, ny, STRIDE),
    hNetUpdatesBelow_(nx, ny + 1, STRIDE),
    hNetUpdatesAbove_(nx, ny + 1, STRIDE),
    hvNetUpdatesBelow_(nx, ny + 1, STRIDE),
    hvNetUpdatesAbove_(nx, ny + 1, STRIDE)
{
    ASSERTION(solver, "No solver has been set to WavePropagationBlock!");
#if defined(ENABLE_VECTORIZATION) && defined(ENABLE_VECTORIZATION_WITH_SIMD)
    ASSERTION(nx >= 4 && ny >= 4 , "The permitted smallest grid size using SIMD solver is 4 points e.g. 2*2, 4*1, etc!");
#endif
    wavePropagationSolver_ = solver;
}

void Blocks::WavePropagationBlock::computeNumericalFluxes()
{
    PROFILER_INSTANCE(0);

    // ---------------------------------------------------------------------------------------------------------------
    // Grid indexing Ex: nx = 2, ny = 2
    //
    //        i    i+1   i+2   i+3   i+4
    //
    //  j     + --- + --- + --- + --- +
    //        |  g  |  g  |  g  |  g  |
    //  j+1   + --- + --- + --- + --- +
    //        |  g  |  c  |  c  |  g  |
    //  j+2   + --- + --- + --- + --- +    ny+2
    //        |  g  |  c  |  c  |  g  |
    //  j+3   + --- + --- + --- + --- +
    //        |  g  |  g  |  g  |  g  |
    //  j+4   + --- + --- + --- + --- +
    //
    //                  nx+2
    //
    // g = ghost cell
    // c = computational cell
    // Kernel layout options:
    //
    //         row-based decomposition           column-based decomposition           block-based decomposition
    //           kernel size = 2 * 4                 kernel size = 4 * 2                kernel size = 2 * 2
    //        + === + === + === + === +         ++ --- + --- ++ --- + --- ++         + === + === ++ === + === +
    //        |  g  |  g  |  g  |  g  |         ||  g  |  g  ||  g  |  g  ||         |  g  |  g  ||  g  |  g  |
    //        + --- + --- + --- + --- +         ++ --- + --- ++ --- + --- ++         + --- + --- ++ --- + --- +
    //        |  g  |  c  |  c  |  g  |         ||  g  |  c  ||  c  |  g  ||         |  g  |  c  ||  c  |  g  |
    //        + === + === + === + === +         ++ --- + --- ++ --- + --- ++         + === + === ++ === + === +
    //        |  g  |  c  |  c  |  g  |         ||  g  |  c  ||  c  |  g  ||         |  g  |  c  ||  c  |  g  |
    //        + --- + --- + --- + --- +         ++ --- + --- ++ --- + --- ++         + --- + --- ++ --- + --- +
    //        |  g  |  g  |  g  |  g  |         ||  g  |  g  ||  g  |  g  ||         |  g  |  g  ||  g  |  g  |
    //        + === + === + === + === +         ++ --- + --- ++ --- + --- ++         + === + === ++ === + === +
    // REVIEW Based on the Nsight Compute profiler, we will decide which decomposition is the best for our application.
    // ---------------------------------------------------------------------------------------------------------------

    // Maximum (linearized) wave speed within one iteration
    RealType maxWaveSpeed = utilities::smart_cast<RealType>(0.0);
    RealType maxEdgeSpeed[STRIDE];

    // Compute the net-updates for the vertical edges
    for (int i = 1; i < nx_ + 2; i++)
    {
        for (int j = 1; j < ny_ + 1; j += STRIDE)
        {
            wavePropagationSolver_->computeNetUpdates(
                h_[i - 1][j],
                h_[i][j],
                hu_[i - 1][j],
                hu_[i][j],
                b_[i - 1][j],
                b_[i][j],
                hNetUpdatesLeft_[i - 1][j - 1],
                hNetUpdatesRight_[i - 1][j - 1],
                huNetUpdatesLeft_[i - 1][j - 1],
                huNetUpdatesRight_[i - 1][j - 1],
                maxEdgeSpeed[0]
            );

            // Update the thread-local maximum wave speed
            maxWaveSpeed = std::max(maxWaveSpeed, maxEdgeSpeed[0]);
        }
    }

    // FIXME[epic=SWE,seq=52] vectorize
    for (size_t i = 0; i < STRIDE; i++)
    {
        maxEdgeSpeed[i] = utilities::smart_cast<RealType>(0.0);
    }

    // Compute the net-updates for the horizontal edges
    for (int i = 1; i < nx_ + 1; i++)
    {
        for (int j = 1; j < ny_ + 2; j += STRIDE)
        {
            wavePropagationSolver_->computeNetUpdates(
                h_[i][j - 1],
                h_[i][j],
                hv_[i][j - 1],
                hv_[i][j],
                b_[i][j - 1],
                b_[i][j],
                hNetUpdatesBelow_[i - 1][j - 1],
                hNetUpdatesAbove_[i - 1][j - 1],
                hvNetUpdatesBelow_[i - 1][j - 1],
                hvNetUpdatesAbove_[i - 1][j - 1],
                maxEdgeSpeed[0]
            );

            // Update the thread-local maximum wave speed
            maxWaveSpeed = std::max(maxWaveSpeed, maxEdgeSpeed[0]);
        }
    }

    if (maxWaveSpeed > 0.00001)
    {
        // Compute the time step width
        maxTimeStep_ = std::min(dx_ / maxWaveSpeed, dy_ / maxWaveSpeed);

        // Reduce maximum time step size by "safety factor"
        maxTimeStep_ *= utilities::smart_cast<RealType>(0.4); // CFL-number = 0.5
    }
    else
    {
        // Might happen in dry cells
        maxTimeStep_ = std::numeric_limits<RealType>::max();
    }
}

void Blocks::WavePropagationBlock::updateUnknowns(RealType dt)
{
    PROFILER_INSTANCE(0);

    // Update cell averages with the net-updates
    for (int i = 1; i < nx_ + 1; i++)
    {
        for (int j = 1; j < ny_ + 1; j++)
        {
            h_[i][j] -= dt / dx_ * (hNetUpdatesRight_[i - 1][j - 1] + hNetUpdatesLeft_[i][j - 1])
                        + dt / dy_ * (hNetUpdatesAbove_[i - 1][j - 1] + hNetUpdatesBelow_[i - 1][j]);
            hu_[i][j] -= dt / dx_ * (huNetUpdatesRight_[i - 1][j - 1] + huNetUpdatesLeft_[i][j - 1]);
            hv_[i][j] -= dt / dy_ * (hvNetUpdatesAbove_[i - 1][j - 1] + hvNetUpdatesBelow_[i - 1][j]);

            if (h_[i][j] < 0)
            {
#ifndef NDEBUG
                // Only print this warning when debug is enabled
                // Otherwise we cannot vectorize this loop
                if (h_[i][j] < -0.1)
                {
                    std::cerr << "Warning, negative height: (i,j)=(" << i << "," << j << ")=" << h_[i][j] << std::endl;
                    std::cerr << "         b: " << b_[i][j] << std::endl;
                }
#endif

                // Zero (small) negative depths
                h_[i][j] = hu_[i][j] = hv_[i][j] = utilities::smart_cast<RealType>(0.0);
            }
            else if (h_[i][j] < 0.1)
            {                                                                 // dryTol
                hu_[i][j] = hv_[i][j] = utilities::smart_cast<RealType>(0.0); // No water, no speed!
            }
        }
    }
}