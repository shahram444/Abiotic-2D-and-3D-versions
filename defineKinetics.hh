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


// The defineRxnKinetics function computes the rates at which different chemical species (substrates) are produced or consumed in a series of reactions, based on their current concentrations. This function is essential for updating the system's state over time. Let's dissect how it works based on the code you've provided:
// Parameters

//     C: A vector of doubles representing the current concentrations of different substrates.
//     subsR: A reference to a vector of doubles that will store the calculated rates of change (production or consumption) for each substrate.
//     mask: An integer indicating the type of space (e.g., pore or non-pore) for which the kinetics are being calculated.

// Process

//     Reaction Rate Constants:
//         k1, k2, and k3 are constants defining the speed of the respective chemical reactions.

//     Mask Check:
//         If mask is 0, it likely indicates a pore space where reactions can occur. For other values (non-pore space), no reactions are assumed, and all rates are set to 0.

//     Reaction Kinetics:
//         Based on the provided rate constants and the concentrations (C vector), the reaction rates for each substrate are calculated.

//     Reaction Mechanisms:
//     The reactions appear to be:
//         First Reaction: Involves H2O (water) turning into other products. The rate R1 is calculated as k1×[H2O]k1×[H2O] (assuming C[5] represents the concentration of H2O).
//         Second Reaction: Involves H2CO3 (carbonic acid) turning into other products. The rate R2 is k2×[H2CO3]k2×[H2CO3] (assuming C[0] is H2CO3).
//         Third Reaction: Involves HCO3 (bicarbonate) turning into other products. The rate R3 is k3×[HCO3]k3×[HCO3] (assuming C[3] is HCO3).

//     Updating Substrate Rates (subsR):
//         For each substrate, the subsR array is updated to reflect how quickly it is being produced or consumed:
//             subsR[0] = -R2: H2CO3 is consumed in the second reaction.
//             subsR[1] = R3: CO32 (carbonate) is produced in the third reaction.
//             subsR[2] = R1 + R2 + R3: H (hydrogen ion) is produced in all three reactions.
//             subsR[3] = -R3: HCO3 is consumed in the third reaction.
//             subsR[4] = R1: OH (hydroxide ion) is produced in the first reaction.
//             subsR[5] = -R1: H2O is consumed in the first reaction.

using namespace plb;

void defineRxnKinetics(std::vector<double> &C, std::vector<double> &subsR, plint mask )
{
    // Constants for reactions' rate
    //double k = 6.9e-3; //1/s
    //double k = 8;

    // Logging the initial concentrations and mask value

    if (mask == 0) { // Reaction only in pore space
        

        // Update the subsR vector based on the reaction
        //subsR[0] =-k * C[0]; // Update the rate of change of 'chemical' concentration due to reaction
        //subsR[0] = 0;
        //std::cout << "Reaction rate for mask 0: " << subsR[0] << std::endl;
        // Print the reaction rate constant 'k' and initial concentration 'C[0]'
        //std::cout << "Reaction constant (k): " << k << std::endl;
        //std::cout << "Initial concentration of the first substrate extracted from definkinetics (C[0]): " << C[0] << std::endl;


        // Reaction rate based on kinetics
        double R = 0; // 'chemical' decays over time due to reaction

        // Update the subsR vector based on the reaction
        subsR[0] = R; // Update the rate of change of 'chemical' concentration due to reaction
    }

    else { // No reaction in non-pore space
        subsR[0] = 0;
        //std::cout << "Reaction rate for mask 1: " << subsR[0] << std::endl;
    }


}
