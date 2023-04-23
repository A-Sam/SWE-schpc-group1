/**
 * @file
 * This file is part of SWE.
 *
 * @author Michael Bader, Kaveh Rahnema, Tobias Schnabel
 * @author Sebastian Rettenberger (rettenbs AT in.tum.de,
 * http://www5.in.tum.de/wiki/index.php/Sebastian_Rettenberger,_M.Sc.)
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
 * TODO
 */

#include "SplashingPoolScenario.hpp"

#include <cmath>

RealType Scenarios::SplashingPoolScenario::getWaterHeight(RealType x, RealType y) const {
  return utilities::smart_cast<RealType>(250.0 + (5.0 - (x + y) / 200));
}

RealType Scenarios::SplashingPoolScenario::getBathymetry([[maybe_unused]] RealType x, [[maybe_unused]] RealType y)
  const {
  return utilities::smart_cast<RealType>(-250.0);
}

double Scenarios::SplashingPoolScenario::getEndSimulationTime() const { return 15; }

RealType Scenarios::SplashingPoolScenario::getBoundaryPos(BoundaryEdge edge) const {
  if (edge == BoundaryEdge::Left) {
    return utilities::smart_cast<RealType>(0.0);
  } else if (edge == BoundaryEdge::Right) {
    return utilities::smart_cast<RealType>(1000.0);
  } else if (edge == BoundaryEdge::Bottom) {
    return utilities::smart_cast<RealType>(0.0);
  } else {
    return utilities::smart_cast<RealType>(1000.0);
  }
}
