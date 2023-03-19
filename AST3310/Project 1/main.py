import numpy as np
import matplotlib.pyplot as plt
from functions import *
from variables import *


if __name__ == "__main__":
    # Tried to make a more friendly new user "interface"  

    print("Do you wish a sanity check? [y/n]")
    choice1 = input()
    if  choice1 == "y":
        print("Sanity check 1 or 2? [1, 2]")
        choice2 = input()
        if choice2 == "1":
            T = 1.57e7 #K
        elif choice2 == "2":
            T = 1e8 #K
    elif choice1 == "n":
        print("Gamow peaks [1], Energy production [2] or Energy per reaction [3] ?")
        choice2 = input()
        T = np.logspace(4, 9, 10000)

    reaction_rates = get_reaction_rates(T)

    #this converts the reaction rates over to m^3/sinstead of cm^3/s/mole
    #Units: reactions m^3 per s
    for key in reaction_rates:
        for i, reaction in enumerate(reaction_rates[key]):
            reaction_rates[key][i] = cm_volume_to_m(reaction)

    r_reaction_rates = {}

    #calculating first iteration of r (no scaling yet)
    for key in reaction_rates:
        r_reaction_rates[key] = []
        for i, [reaction_rate, reaction_reactants, branches_step] in enumerate(zip(reaction_rates[key], reactants[key], branches_steps[key][1::2])):
            ni = number_density(fractions[reaction_reactants[0]][0], fractions[reaction_reactants[0]][1])
            nk = number_density(fractions[reaction_reactants[1]][0], fractions[reaction_reactants[1]][1])
            r = lamda_to_r(reaction_rate, ni, nk)
            r_reaction_rates[key].append(r)


    #only really used in the sanity checks, but it supports arrays, just never got it to work fully
    if choice1 == "y":
        scale_factor = {}
        accounted_for = {}
        #the scale factor is only calculated for reactions which depend on a reaction that other reactions depend on
        #such as for He32+He42 -> Be74, which relies on PP0
        for key in reactants:
            for i, [r, reactant, branches_step] in enumerate(zip(r_reaction_rates[key], reactants[key], branches_steps[key][1::2])):
                if reactant[2]:
                    if reactant[2] not in scale_factor:
                        scale_factor[reactant[2]] = 0

                    #reactant[0]+reactant[1] not in accounted_for returns True only the first time we encounterHe32+He42 -> Be74
                    #branches_steps[key][1 + 2*(i-1)] is the amount of times the reaction is repeated
                    scale_factor[reactant[2]] += r*branches_steps[key][1 + 2*(i-1)]*(reactant[0]+reactant[1] not in accounted_for)

                    # He32+He42 -> Be74 appears in two chains, we only need to add its reaction rate to the scale factor ones
                    accounted_for[reactant[0]+reactant[1]] = True
        scales = {}
        #all is float checks are now redundant, as this only runs when T is a float
        for key in r_reaction_rates:
            scales[key] = []
            for i, [r, reactant, branches_step] in enumerate(zip(r_reaction_rates[key], reactants[key], branches_steps[key][1::2])):
                scale = np.ones(np.shape(r))
                #if the reaction is not the only one who consumes a reactant
                if reactant[2]:
                    if r_reaction_rates[key][i-1] is float:
                        #if more is consumed than produced
                        if scale_factor[reactant[2]] > r_reaction_rates[key][i-1]:
                            scale = r/scale_factor[reactant[2]]
                            r_reaction_rates[key][i] = r_reaction_rates[key][i-1]
                    else:
                        r_reaction_rates[key][i] = np.where(scale_factor[reactant[2]] > r_reaction_rates[key][i-1], r_reaction_rates[key][i-1], r)
                        scale = np.where(scale_factor[reactant[2]] > r_reaction_rates[key][i-1], r/scale_factor[reactant[2]], 1)
                #this is not valid for the PP0 step
                if i!=0:
                    if r_reaction_rates[key][i-1] is float:
                        #if more is consumed than produced
                        if r_reaction_rates[key][i] > r_reaction_rates[key][i-1]:
                            r_reaction_rates[key][i] = r_reaction_rates[key][i-1]
                    else:
                        r_reaction_rates[key][i] = np.where(r_reaction_rates[key][i] > r_reaction_rates[key][i-1], r_reaction_rates[key][i-1], r_reaction_rates[key][i])

                #we finally scale the reaction rate
                r_reaction_rates[key][i] *= scale
                scales[key].append(scale)

    #couldnt manage to do it through for loops, so did it by hand
    if choice1 == "n":
        #applogies for the long lines of code, my naming was never supposed to be used like this

        #if more PP0 is consumed than produced, scale it, if not leave it
        r_reaction_rates["PP1"][1] = np.where(r_reaction_rates["PP0"][0] <= 2*r_reaction_rates["PP1"][1] + r_reaction_rates["PP2"][1], r_reaction_rates["PP0"][0]/2*(2*r_reaction_rates["PP1"][1]/(2*r_reaction_rates["PP1"][1] + r_reaction_rates["PP2"][1])), r_reaction_rates["PP1"][1])
        #define reaction rate of PP0 in PP1 
        r_reaction_rates["PP1"][0] = 2*r_reaction_rates["PP1"][1]

        #repeat
        r_reaction_rates["PP2"][1] = np.where(r_reaction_rates["PP0"][0] <= 2*r_reaction_rates["PP1"][1] + r_reaction_rates["PP2"][1], r_reaction_rates["PP0"][0]*(r_reaction_rates["PP2"][1]/(2*r_reaction_rates["PP1"][1] + r_reaction_rates["PP2"][1])), r_reaction_rates["PP2"][1]) 
        r_reaction_rates["PP3"][1] = r_reaction_rates["PP2"][1]
        r_reaction_rates["PP2"][2] = np.where(r_reaction_rates["PP2"][1] <= r_reaction_rates["PP2"][2] + r_reaction_rates["PP3"][2], r_reaction_rates["PP2"][1]*(r_reaction_rates["PP2"][2]/(r_reaction_rates["PP2"][2] + r_reaction_rates["PP3"][2])), r_reaction_rates["PP2"][2])
        r_reaction_rates["PP2"][3] = np.where(r_reaction_rates["PP2"][2] <= r_reaction_rates["PP2"][3], r_reaction_rates["PP2"][2], r_reaction_rates["PP2"][3])

        r_reaction_rates["PP3"][2] = np.where(r_reaction_rates["PP3"][1] <= r_reaction_rates["PP2"][2] + r_reaction_rates["PP3"][2], r_reaction_rates["PP3"][1]*(r_reaction_rates["PP3"][2]/(r_reaction_rates["PP2"][2] + r_reaction_rates["PP3"][2])), r_reaction_rates["PP3"][1])
        r_reaction_rates["PP3"][0] = r_reaction_rates["PP3"][1]
        r_reaction_rates["PP2"][0] = r_reaction_rates["PP2"][1]
    
    sanity_checks = {}
    #calculate the sanity checks
    for key in r_reaction_rates:
        sanity_checks[key] = []
        for i, [Q_, r, Qn, reactant] in enumerate(zip(Q[key], r_reaction_rates[key], branches_steps[key][::2], reactants[key])):
            Q_J = MeV_to_J(Q_)
            Qn_J = MeV_to_J(Qn[2])
            sanity_checks[key].append(r*(Q_J-Qn_J)*rho)

    #print the sanity checks
    if choice1 == "y":
        for key in sanity_checks:
            print(key, ":")
            for check, reactant in zip(sanity_checks[key], reactants[key]):
                print(f"   {reactant[0]} + {reactant[1]}")
                print(f"   {check:e} J/m^3/s\n")

    #plot the relative energy production
    if choice1 == "n" and choice2 == "2":
        total_production = 0
        branch_production = {}
        for key in sanity_checks:
            if key != "PP0":
                branch_production[key] = 0
                for i, check in enumerate(sanity_checks[key]):
                    branch_production[key] += check
                total_production += branch_production[key]

        #divide by total to get a relative energy production
        for key in branch_production:
            plt.plot(T, branch_production[key]/total_production, label=key)

        #add a line for the temperature of the sun
        plt.axvline(x = 1.57e7, color="black", linestyle="--")
        plt.ylim(-0.1, 1.1)
        plt.xscale("log")
        plt.legend()
        plt.xlabel("T [K]")
        plt.ylabel("Fraction of total energy production [*]")
        plt.show()

    #gamov peaks
    elif choice1 == "n" and choice2 == "1":
        E = np.logspace(-15.5, -13.5, 10000)
        T = 1.57e7 #K

        mb = MB(E, T)

        sigmas = {}
        for key in reactants:
            sigmas[key] = []
            for reactant in reactants[key]:
                i, k = reactant[0], reactant[1]
                mi, mk = fractions[i][1], fractions[k][1]
                zi, zk = fractions[i][2], fractions[k][2]
                new_sigma = sigma(E, mi, mk, zi, zk)
                sigmas[key].append(new_sigma)
                

        accounted_for = {}
        for key in sigmas:
            for reactant, sigma_ in zip(reactants[key], sigmas[key]):
                i, k = reactant[0], reactant[1]
                #calculate gamow peak
                gamow = mb*sigma_
                if i+k not in accounted_for:
                    #scale it accordingly
                    plt.plot(E, gamow/np.linalg.norm(gamow), label=f"{i} + {k}")
                    accounted_for[i+k] = True

        plt.xscale("log")
        plt.legend()
        plt.xlabel("E [J]")
        plt.ylabel("Relative probability [*]")
        plt.show()

    if choice1 == "n" and choice2 == "3":
        Q = {}
        print("Energy output per reaction w/ neutrinos")
        for key in branches_steps_energy_calc:
            print(key, ":")
            Q[key] = []
            for mass1, mass2, neutrino, protons in branches_steps_energy_calc[key][::2]:
                dm = abs(mass1 - mass2)
                Q[key].append(energy_out(dm))
                print("    ", Q[key][-1])

        Q_tot = {
        key : sum([reaction*Q_ for Q_, reaction in zip(Q[key], branches_steps_energy_calc[key][1::2])]) for key in Q
        }
        print("Total Energy output per reaction w/ neutrinos")
        for key in Q_tot:
            print(key, ":")
            print("    ", Q_tot[key])

        print("Energy percentage lost to neutrinos")
        for key in Q_tot:
            nu = 0
            for mass1, mass2, neutrino, protons in branches_steps_energy_calc[key][::2]:
                nu += neutrino
            print(key, ":")
            print("    ", nu/Q_tot[key]*100, "%")