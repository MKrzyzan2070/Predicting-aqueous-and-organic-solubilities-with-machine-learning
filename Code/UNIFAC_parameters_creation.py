from rdkit import Chem
from thermo.unifac import UFIP, UFSG, UNIFAC
from rdkit import RDLogger
import json
import numpy as np

logger = RDLogger.logger()
logger.setLevel(RDLogger.ERROR)

def Make_UNIFAC_parameters_from_SMILES_Version1(solute_solvent_df, mole_fraction_list):

    UNIFAC_column_list = []
    inf_once = False
    for mole_fraction_tuple in mole_fraction_list:

        mole_fraction_solute = mole_fraction_tuple[0]
        mole_fraction_solvent = mole_fraction_tuple[1]
        ########################################################################################################
        # Declaring the lists for features
        gamma_solute_list = []
        gamma_solvent_list = []
        G_Excess_list = []

        gamma_inf_solute_list = []
        gamma_inf_solvent_list = []

        ########################################################################################################
        # Assigning the UFSG groups to the molecules
        with open('Code/JSON_files/UFSG Group Assignment.json', 'r') as file:
            UFSG_group_dict = json.load(file)

        for index_main in solute_solvent_df.index:

            # Two dictionaries are created. One for a solute and the other for a solvent
            solute_assignment = {}
            solvent_assignment = {}

            # This will be useful to make sure that atoms are not double matched
            # The groups in the UFSG_group_dict are sorted going from most complex
            # substructure to the least complex substructure. This ensured that most
            # complex substructures are matched first.
            matched_atoms_solute = set()
            matched_atoms_solvent = set()

            solute_smiles = solute_solvent_df.loc[index_main, "solute_smiles"]
            solvent_smiles = solute_solvent_df.loc[index_main, "solvent_smiles"]

            solute_molecule = Chem.MolFromSmiles(solute_smiles)
            solvent_molecule = Chem.MolFromSmiles(solvent_smiles)

            ##################################################
            ######## MATCHING WITH UFSG SUBSTRUCTURES ########
            ##################################################
            for index in UFSG_group_dict:
                UFSG_group_smarts = UFSG_group_dict[index]

                # Some substructures are a list, so that different substructures will be matched
                # under the same name
                if isinstance(UFSG_group_smarts, list):
                    got_matched_solute = False
                    got_matched_solvent = False
                    for smarts in UFSG_group_smarts:
                        substructure = Chem.MolFromSmarts(smarts)
                        matches_solute = solute_molecule.GetSubstructMatches(substructure)
                        matches_solvent = solvent_molecule.GetSubstructMatches(substructure)
                        how_many_match_solute = 0
                        how_many_match_solvent = 0

                        if got_matched_solute is False:
                            for match in matches_solute:
                                match_set = set(match)
                                if len(match_set & matched_atoms_solute) == 0:
                                    matched_atoms_solute.update(match_set)
                                    how_many_match_solute += 1
                            if how_many_match_solute > 0:
                                solute_assignment[int(index)] = how_many_match_solute
                                got_matched_solute = True

                        if got_matched_solvent is False:
                            for match in matches_solvent:
                                match_set = set(match)
                                if len(match_set & matched_atoms_solvent) == 0:
                                    matched_atoms_solvent.update(match_set)
                                    how_many_match_solvent += 1
                            if how_many_match_solvent > 0:
                                solvent_assignment[int(index)] = how_many_match_solvent
                                got_matched_solvent = True

                # The case when the substructure is just a string
                else:
                    substructure = Chem.MolFromSmarts(UFSG_group_smarts)
                    matches_solute = solute_molecule.GetSubstructMatches(substructure)
                    matches_solvent = solvent_molecule.GetSubstructMatches(substructure)
                    how_many_match_solute = 0
                    how_many_match_solvent = 0

                    for match in matches_solute:
                        match_set = set(match)
                        if len(match_set & matched_atoms_solute) == 0:
                            matched_atoms_solute.update(match_set)
                            how_many_match_solute += 1
                    if how_many_match_solute > 0:
                        solute_assignment[int(index)] = how_many_match_solute

                    for match in matches_solvent:
                        match_set = set(match)
                        if len(match_set & matched_atoms_solvent) == 0:
                            matched_atoms_solvent.update(match_set)
                            how_many_match_solvent += 1
                    if how_many_match_solvent > 0:
                        solvent_assignment[int(index)] = how_many_match_solvent

            ##################################################
            ########### OBTAINING UNIFAC FEATURES ############
            ##################################################
            # first value of molar fraction is a solute while the second value is the solvent
            # it is possible (especially if a purely inorganic SMILES sttring was passed) that no
            # UFSG substructure has been assigned to it. Hence, the try and except statement

            try:
                #### G_Excess features
                GE = UNIFAC.from_subgroups(chemgroups=[solute_assignment, solvent_assignment],
                                                 T=solute_solvent_df.loc[index_main, "temperature"],
                                                 xs=[mole_fraction_solute, mole_fraction_solvent],
                                                 interaction_data=UFIP, subgroups=UFSG)

                ##############################################################################################

                # Gamma inifinite dilutions:
                # This can be done once as it yields the same values regardless of the molar fraction (makes sense)
                if inf_once is False:
                    gamma_inf_solute = np.log10(GE.gammas_infinite_dilution()[0])
                    gamma_inf_solvent = np.log10(GE.gammas_infinite_dilution()[1])
                    gamma_inf_solute_list.append(gamma_inf_solute)
                    gamma_inf_solvent_list.append(gamma_inf_solvent)


                gamma_solute = np.log10(GE.gammas()[0])
                gamma_solvent = np.log10(GE.gammas()[1])
                # I won't use the G_excess anymore but let's leave it commented out:
                #G_Excess = GE.GE()

                # In case of extreme values of gamma, the G_excess will preserve this knowledge while the gamma_solute
                # and gamma_solvent need to be changed, because otherwise the min_max scaler will produce data that is
                # very close to one another
                solute_mol_fraction = mole_fraction_tuple[0]
                solvent_mol_fraction = mole_fraction_tuple[1]
                if solute_mol_fraction < solvent_mol_fraction:
                    gamma_solute_list.append(gamma_solute)
                elif solvent_mol_fraction < solute_mol_fraction:
                    gamma_solvent_list.append(gamma_solvent)
                else:
                    gamma_solute_list.append(gamma_solute)
                    gamma_solvent_list.append(gamma_solvent)

                #G_Excess_list.append(G_Excess)
            except:
                solute_mol_fraction = mole_fraction_tuple[0]
                solvent_mol_fraction = mole_fraction_tuple[1]
                if solute_mol_fraction < solvent_mol_fraction:
                    gamma_solute_list.append("ERROR")
                elif solvent_mol_fraction < solute_mol_fraction:
                    gamma_solvent_list.append("ERROR")
                else:
                    gamma_solute_list.append("ERROR")
                    gamma_solvent_list.append("ERROR")
                if inf_once is False:
                    gamma_inf_solute_list.append("ERROR")
                    gamma_inf_solvent_list.append("ERROR")

        ########################################################################################################

        ########################################################################################################
        # Lastly, it's time to add the lists to the dataframe
        #solute_solvent_df[f"G_Excess_{mole_fraction_solute}_{mole_fraction_solvent}"] = G_Excess_list
        if len(gamma_solute_list) - gamma_solute_list.count("ERROR") != 0:
            solute_solvent_df[f"gamma_solute_{mole_fraction_solute}_{mole_fraction_solvent}"] = gamma_solute_list
            UNIFAC_column_list.append(f"gamma_solute_{mole_fraction_solute}_{mole_fraction_solvent}")
        if len(gamma_solvent_list) - gamma_solvent_list.count("ERROR") != 0:
            solute_solvent_df[f"gamma_solvent_{mole_fraction_solute}_{mole_fraction_solvent}"] = gamma_solvent_list
            UNIFAC_column_list.append(f"gamma_solvent_{mole_fraction_solute}_{mole_fraction_solvent}")
        #UNIFAC_column_list.append(f"G_Excess_{mole_fraction_solute}_{mole_fraction_solvent}")

        if inf_once is False:
            solute_solvent_df[f"gamma_inf_solute"] = gamma_inf_solute_list
            solute_solvent_df[f"gamma_inf_solvent"] = gamma_inf_solvent_list
            UNIFAC_column_list.append(f"gamma_inf_solute")
            UNIFAC_column_list.append(f"gamma_inf_solvent")
            inf_once = True
        ########################################################################################################


    return solute_solvent_df, UNIFAC_column_list