/* This file is a part of the CompLaB program.
 *
 * The CompLaB softare is developed since 2022 by the University of Georgia
 * (United States) and Chungnam National University (South Korea).
 *
 * Contact:
 * Heewon Jung
 * Department of Geological Sciences
 * Chungnam National University
 * 99 Daehak-ro, Yuseong-gu
 * Daejeon 34134, South Korea
 * hjung@cnu.ac.kr
 *
 * The most recent release of CompLaB can be downloaded at
 * <https://CompLaB.unige.ch/>
 *
 * CompLaB is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * The library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


// The defineRxnKinetics function computes the rates at which different chemical species (substrates) are produced or consumed in a series of reactions, based on their current concentrations.
// Parameters
//     C: A vector of doubles representing the current concentrations of different substrates.
//     subsR: A reference to a vector of doubles that will store the calculated rates of change for each substrate.
//     mask: An integer indicating the type of space (e.g., pore or non-pore) for which the kinetics are being calculated.

using namespace plb;

void defineRxnKinetics(std::vector<double> &C, std::vector<double> &subsR, plint mask )
{
    if (mask == 0) { // Reaction only in pore space

        // Reaction rate based on kinetics
        double R = 0; // 'chemical' decays over time due to reaction

        // Update the subsR vector based on the reaction
        subsR[0] = R; // Update the rate of change of 'chemical' concentration due to reaction
    }

    else { // No reaction in non-pore space
        subsR[0] = 0;
    }


}
