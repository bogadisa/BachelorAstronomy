

scale_factor = 0

for r in reactions:
    if multiple_dependents:
        scale_factor += r

for r in reactions:
    if scale_factor > prev_r:
        r = prev_r * r/scale_factor

        
    for key in r_reaction_rates:
        print(key, ":")
        for i, [r, reactant, branches_step] in enumerate(zip(r_reaction_rates[key], reactants[key], branches_steps[key][1::2])):
            print("    ", reactant[2])
            if reactant[2]:
                print("    ", scale_factor[reactant[2]])
                r_reaction_rates[key][i] = np.where(r_reaction_rates[key][i-1] > scale_factor[reactant[2]], r_reaction_rates[key][i], r_reaction_rates[key][i-1]*r/scale_factor[reactant[2]])
                r_reaction_rates[key][i-1] = np.where(r_reaction_rates[key][i-1] > scale_factor[reactant[2]], r_reaction_rates[key][i-1], r_reaction_rates[key][i]) #*r/scale_factor[reactant[2]]
                # for j in range(i, 0, -1):
                    # r_reaction_rates[key][j-1] = np.where(r_reaction_rates[key][j-1] > r_reaction_rates[key][j]*branches_step, r_reaction_rates[key][j], r_reaction_rates[key][j-1])
                #     r_reaction_rates[key][j-1] = np.where(r_reaction_rates[key][j-1] > r_reaction_rates[key][j]*branches_step, r_reaction_rates[key][j-1]*r/scale_factor[reactant[2]], r_reaction_rates[key][j-1])
            elif i != 0:
                r_reaction_rates[key][i] = np.where(r_reaction_rates[key][i-1] > r_reaction_rates[key][i], r_reaction_rates[key][i], r_reaction_rates[key][i-1])
                # for j in range(i, 0, -1):
                #     r_reaction_rates[key][j-1] = np.where(r_reaction_rates[key][j-1] > r_reaction_rates[key][j]*branches_step, r_reaction_rates[key][j], r_reaction_rates[key][j-1])
                # r_reaction_rates[key][i-1] = np.where(r_reaction_rates[key][i-1] > r_reaction_rates[key][i]*branches_step, r_reaction_rates[key][i], r_reaction_rates[key][i-1])
            print("    ", i, r_reaction_rates[key][i])
    

    # for key in r_reaction_rates:
    #     k = len(r_reaction_rates[key]) - 1
    #     for i, reactant in enumerate(reactants[key]):
    #         if (k-i) != 0 and reactant[2]:
    #             r_reaction_rates[key][k-i-1] = np.where(r_reaction_rates[key][k-i-1] > r_reaction_rates[key][k-i]*branches_steps[key][1 + 2*(k-i)], r_reaction_rates[key][k-i], r_reaction_rates[key][k-i-1])