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

#include "SeaAtRestScenario.hpp"

#include <cmath>

RealType Scenarios::SeaAtRestScenario::getWaterHeight(RealType x, RealType y) const {
  return utilities::smart_cast<RealType>((sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) < 0.1) ? 9.9 : 10.0);
}

RealType Scenarios::SeaAtRestScenario::getBathymetry([[maybe_unused]] RealType x, [[maybe_unused]] RealType y) const {
  return utilities::smart_cast<RealType>(sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) < 0.1) ? 0.1 : 0.0;
}
