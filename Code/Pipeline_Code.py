from rdkit import Chem
import matplotlib.pyplot as plt
import numpy as np
import os
import textwrap
from rdkit.Chem import Draw
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit.Chem import inchi
import pubchempy as pcp
pd.set_option('display.max_columns', None)


def Filter_immiscible_cosolvents(molecule_smiles_list, organic_solubility_prediction_df,
                                 aquatic_solubility_prediction_df, feature_type, model,
                                 cutoff_1, cutoff_2, cutoff_3):

    for molecule_smiles in molecule_smiles_list:
        the_solvents_df = organic_solubility_prediction_df[organic_solubility_prediction_df["Molecule_smiles"]
                                                           == molecule_smiles].copy()[["Solubility Prediction",
                                                                                       "Solvent_smiles",
                                                                                       "Solvent_InChIKey"]]
        the_solvents_df = the_solvents_df.copy().rename(
            columns={"Solubility Prediction": "Organic Solubility Prediction"})

        # Getting the InChiKey:
        molecule_inchikey = inchi.MolToInchiKey(Chem.MolFromSmiles(molecule_smiles))
        if "/" in molecule_smiles:
            molecule_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(molecule_smiles), isomericSmiles=False)

        the_aquatic_solubility_df = aquatic_solubility_prediction_df[["Solubility Prediction",
                                                                      "Molecule_smiles", "Molecule_InChIKey"]]
        the_aquatic_solubility_df = the_aquatic_solubility_df.copy().rename(
            columns={"Molecule_smiles": "Solvent_smiles", "Molecule_InChIKey": "Solvent_InChIKey",
                     "Solubility Prediction": "Aqueous Solubility Prediction"})


        the_pipeline_df = pd.merge(the_solvents_df, the_aquatic_solubility_df,
                                   on=['Solvent_InChIKey'],
                                   how='inner')

        # As usual, there is a discrepancy among the SMILES strings
        # The aquatic solubility SMILES column contains the correct SMILES strings
        the_pipeline_df.drop(columns=["Solvent_smiles_x"], inplace=True)
        the_pipeline_df.rename(columns={"Solvent_smiles_y": "Solvent_smiles"}, inplace=True)

        # Making the plot
        dpi = 800
        fig = plt.figure(figsize=(11, 7), dpi=dpi)
        ax = fig.add_axes([0.15, 0.20, 0.45, 0.75])

        # The cutoff line:
        plt.axhline(y=cutoff_1, color='green', linestyle='--', linewidth=3)
        plt.axhline(y=cutoff_2, color='red', linestyle='--', linewidth=3)
        plt.axhline(y=cutoff_3, color='green', linestyle='--', linewidth=3)

        # The cutoff color:
        ax.axhspan(ymin=min(list(the_pipeline_df['Aqueous Solubility Prediction']))-2,
                   ymax=cutoff_1, facecolor='red', alpha=0.2)
        ax.axhspan(ymin=cutoff_1, ymax=cutoff_3, facecolor='grey', alpha=0.5)
        ax.axhspan(ymin=cutoff_3, ymax=max(list(the_pipeline_df['Aqueous Solubility Prediction']))+2,
                   facecolor='green', alpha=0.2)

        # The scatter plot:
        ax.scatter(
            x=the_pipeline_df['Organic Solubility Prediction'],
            y=the_pipeline_df['Aqueous Solubility Prediction'],
            color="black",
            s=150,
            alpha=1,
            edgecolor='black',
            linewidths=1.4
        )

        # Coloring the axes:
        color_axes = True
        if color_axes is True:
            ax.spines['bottom'].set_color('#d49f00')
            ax.spines['bottom'].set_linewidth(3)
            ax.spines['left'].set_color('#0070c0')
            ax.spines['left'].set_linewidth(3)
            ax.set_xlabel('Organic Solubility Prediction \n log(x)', fontsize=18, color='#d49f00', labelpad=15)
            ax.set_ylabel('Aqueous Solubility Prediction \n log(S / mol/dm³)', fontsize=18, color='#0070c0', labelpad=15)
            ax.tick_params(axis='x', which='major', labelsize=16, length=10, width=2,
                           colors='#d49f00')
            ax.tick_params(axis='y', which='major', labelsize=16, length=10, width=2,
                           colors='#0070c0')
        else:
            ax.set_xlabel('Organic Solubility Prediction log(x)', fontsize=18)
            ax.set_ylabel('Aqueous Solubility Prediction log(S / mol/dm³)', fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=16, length=10, width=2)

        # Making the other spines thicker:
        ax.spines["top"].set_linewidth(2.2)
        ax.spines["right"].set_linewidth(2.2)

        # Adding the image of the mollecule of interest:
        mol = Chem.MolFromSmiles(molecule_smiles)
        img = Draw.MolToImage(mol, size=(300, 300))
        img = img.convert("RGBA")
        img = img.rotate(90)
        datas = img.getdata()
        newData = []
        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
        img.putdata(newData)

        image_size_inches = [img.size[0] / (3 * 300), img.size[1] / (3 * 300)]
        ax_img = fig.add_axes([0.72, 0.6, image_size_inches[0], image_size_inches[1]])
        ax_img.imshow(img, aspect='equal')
        ax_img.axis('off')

        ax.set_ylim(min(list(the_pipeline_df['Aqueous Solubility Prediction'])) - 1,
                    max(list(the_pipeline_df['Aqueous Solubility Prediction'])) + 1)


        # Saving the figure:
        main_path = f"Pipeline Predictions/Pipeline_Combined/{feature_type}_{model}/{molecule_smiles}"
        if not os.path.exists(main_path):
            os.makedirs(main_path)
            os.makedirs(main_path + "/Best CoSolvents Images")

        fig_path = main_path + f"/Pipeline_{molecule_inchikey}.png"
        plt.savefig(fig_path)
        plt.close()

        ########################################################################################
        InChIKey_solvent_list = list(the_pipeline_df["Solvent_InChIKey"])
        solvent_name_list = []
        for InChIKey_solvent in InChIKey_solvent_list:
            solvent_name = pcp.get_compounds(InChIKey_solvent, 'inchikey')[0].iupac_name
            solvent_name_list.append(solvent_name)
        the_pipeline_df["Solvent_name"] = solvent_name_list

        # Sorting based on the Organic solubility value:
        the_pipeline_df.sort_values(by="Organic Solubility Prediction", ascending=False, inplace=True)

        # And saving it:
        the_pipeline_df.to_csv(main_path + f"/Pipeline_{molecule_inchikey}.csv")
        ########################################################################################

        ########################################################################################
        # Filtering out the pipeline dataframe so that only the molecules miscible in water will in it:
        the_pipeline_df_water_misc = the_pipeline_df[the_pipeline_df['Aqueous Solubility Prediction'] > cutoff_3]

        # Adding the name of the solvent:
        InChIKey_solvent_list = list(the_pipeline_df_water_misc["Solvent_InChIKey"])
        solvent_name_list = []
        for InChIKey_solvent in InChIKey_solvent_list:
            solvent_name = pcp.get_compounds(InChIKey_solvent, 'inchikey')[0].iupac_name
            solvent_name_list.append(solvent_name)
        the_pipeline_df_water_misc["Solvent_name"] = solvent_name_list

        # Sorting based on the Organic solubility value:
        the_pipeline_df_water_misc.sort_values(by="Organic Solubility Prediction", ascending=False, inplace=True)

        # And saving it:
        the_pipeline_df_water_misc.to_csv(main_path + f"/Pipeline_water_miscible_{molecule_inchikey}.csv")
        ########################################################################################


def Draw_co_solvent_molecules(molecule_smiles_list, feature_type, model):

    for molecule_smiles in molecule_smiles_list:

        molecule_inchikey = inchi.MolToInchiKey(Chem.MolFromSmiles(molecule_smiles))
        if "/" in molecule_smiles:
            molecule_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(molecule_smiles), isomericSmiles=False)

        main_path = f"Pipeline Predictions/Pipeline_Combined/{feature_type}_{model}/{molecule_smiles}"
        the_pipeline_df = pd.read_csv(main_path + f"/Pipeline_water_miscible_{molecule_inchikey}.csv")

        #######################################################################################
        best_combined_solvents_dict = {}

        best_combined_solvents_dict["Solvent_smiles"] = list(the_pipeline_df["Solvent_smiles"])
        best_combined_solvents_dict["Organic_solubility"] = list(the_pipeline_df["Organic Solubility Prediction"])

        i = -1
        for organic_solubility in best_combined_solvents_dict["Organic_solubility"]:
            i += 1
            solvent_smiles = best_combined_solvents_dict["Solvent_smiles"][i]
            solvent_inchikey = inchi.MolToInchiKey(Chem.MolFromSmiles(solvent_smiles))
            mol_img = Draw.MolToImage(Chem.MolFromSmiles(solvent_smiles), size=(400, 400))
            img = Image.new('RGB', (500, 550), color='white')
            mol_x = (500 - 400) // 2
            mol_y = (550 - 400) // 2
            img.paste(mol_img, (mol_x, mol_y))

            draw = ImageDraw.Draw(img)
            title = f"Number {i + 1}"
            fontsize = 30
            font = ImageFont.load_default().font_variant(size=fontsize)

            # Centering the title
            title_bbox = draw.textbbox((0, 0), title, font=font)  # Get the bounding box of the title
            title_width = title_bbox[2] - title_bbox[0]  # Width from the bounding box
            draw.text(((500 - title_width) // 2, 30), title, fill="black", font=font)

            # Centering the score text
            score_text = f"Organic Solubility log(x): {round(organic_solubility, 3)}"
            score_bbox = draw.textbbox((0, 0), score_text, font=font)  # Get the bounding box of the score text
            score_width = score_bbox[2] - score_bbox[0]  # Width from the bounding box
            draw.text(((500 - score_width) // 2, 60), score_text, fill="black", font=font)

            file_path = (main_path + "/Best CoSolvents Images/" + f"Number_{i + 1}_{solvent_inchikey}.png")
            img.save(file_path)


def Make_co_solvent_ranking_plot(molecule_smiles_list, feature_type, model):
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    for molecule_smiles in molecule_smiles_list:
        molecule_inchikey = inchi.MolToInchiKey(Chem.MolFromSmiles(molecule_smiles))
        if "/" in molecule_smiles:
            molecule_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(molecule_smiles), isomericSmiles=False)

        main_path = f"Pipeline Predictions/Pipeline_Combined/{feature_type}_{model}/{molecule_smiles}"
        the_pipeline_df = pd.read_csv(main_path + f"/Pipeline_water_miscible_{molecule_inchikey}.csv")

        # Sort the dataframe by Organic Solubility Prediction
        the_pipeline_df = the_pipeline_df.sort_values(by="Organic Solubility Prediction", ascending=False)
        the_pipeline_df['Rank'] = range(1, len(the_pipeline_df) + 1)

        fig, ax = plt.subplots(figsize=(50, 10)) # Should be extended way to the right
        bars = ax.bar(the_pipeline_df['Rank'], the_pipeline_df["Organic Solubility Prediction"], color="skyblue")

        # Add solvent names and images on top of bars
        i = -1
        for bar, (_, row) in zip(bars, the_pipeline_df.iterrows()):
            i += 1

            # Adding the name on top of the bar:
            organic_solubility = float(the_pipeline_df["Organic Solubility Prediction"].iloc[i])
            if organic_solubility > -0.8:
                max_width = 10
            elif organic_solubility > -1.2:
                max_width = 14
            else:
                max_width = 18

            wrapped_name = "\n".join(textwrap.wrap(row['Solvent_name'], max_width))
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    wrapped_name, ha='center', va='bottom', rotation=90,
                    fontsize=22, fontweight='bold')  # Increase font size to 16

            # Setting the ylim right:
            min_solubility = the_pipeline_df["Organic Solubility Prediction"].min()
            max_solubility = the_pipeline_df["Organic Solubility Prediction"].max()
            data_range = max_solubility - min_solubility
            padding = data_range
            bottom_limit = min_solubility - padding
            ax.set_ylim(bottom_limit, 0)


            ax.set_xlim(0.5, 17.5)
            # Adding the image of the solvent on top of the bar:
            mol = Chem.MolFromSmiles(row['Solvent_smiles'])
            img = Draw.MolToImage(mol, size=(250, 250))

            # I dunno why it has to be done but otherwise things simply do not work:
            img_array = np.array(img)
            imagebox = OffsetImage(img_array, zoom=0.5)
            imagebox.image.axes = ax
            ab = AnnotationBbox(imagebox, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                xybox=(0, -80),
                                xycoords='data',
                                boxcoords="offset points",
                                pad=0.5,
                                frameon=False) # This just removed the stupid black box around the molecule
            ax.add_artist(ab)

        plt.ylabel("Organic Solubility log(x)", fontsize=36, labelpad=20)
        plt.xlabel("Solvent Rank", fontsize=36, labelpad=20)

        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        max_rank = the_pipeline_df['Rank'].max()
        plt.xticks(np.arange(1, max_rank + 1, step=1))

        # Saving the figure
        fig_path = main_path + f"/Solvent_selection_{molecule_inchikey}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


def Solvent_selection_visalisation(molecule_smiles_list, feature_type, model, cutoff_1, cutoff_2, cutoff_3):

    for molecule_smiles in molecule_smiles_list:
        molecule_inchikey = inchi.MolToInchiKey(Chem.MolFromSmiles(molecule_smiles))
        if "/" in molecule_smiles:
            molecule_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(molecule_smiles), isomericSmiles=False)

        main_path = f"Pipeline Predictions/Pipeline_Combined/{feature_type}_{model}/{molecule_smiles}"
        the_pipeline_df = pd.read_csv(main_path + f"/Pipeline_{molecule_inchikey}.csv")

        # The water miscible organic solvents dataframe:
        the_pipeline_df_water_misc = pd.read_csv(main_path + f"/Pipeline_water_miscible_{molecule_inchikey}.csv")
        the_pipeline_df_water_misc = the_pipeline_df_water_misc.sort_values(by="Organic Solubility Prediction",
                                                                      ascending=False)
        the_pipeline_df_water_misc['Rank'] = range(1, len(the_pipeline_df_water_misc) + 1)

        ################### ################### ################### ###################
        ################### THE FIRST PART  ################### ###################
        # Making the plot
        dpi = 800
        fig = plt.figure(figsize=(11, 7), dpi=dpi)
        ax = fig.add_axes([0.15, 0.20, 0.45, 0.75])

        # The cutoff line:
        plt.axhline(y=cutoff_1, color='green', linestyle='--', linewidth=3)
        plt.axhline(y=cutoff_2, color='red', linestyle='--', linewidth=3)
        plt.axhline(y=cutoff_3, color='green', linestyle='--', linewidth=3)

        # The cutoff color:
        ax.axhspan(ymin=min(list(the_pipeline_df['Aqueous Solubility Prediction'])) - 2,
                   ymax=cutoff_1, facecolor='red', alpha=0.2)
        ax.axhspan(ymin=cutoff_1, ymax=cutoff_3, facecolor='grey', alpha=0.5)
        ax.axhspan(ymin=cutoff_3, ymax=max(list(the_pipeline_df['Aqueous Solubility Prediction'])) + 2,
                   facecolor='green', alpha=0.2)

        # Those are the solvents that were selected for the experimental validation:
        solvents_to_highlight = {
            'FXHOOIRPVKKKFG-UHFFFAOYSA-N': 'blue',  # N,N-Dimethylacetamide
            'LFQSCWFLJHTTHZ-UHFFFAOYSA-N': 'blue',  # Ethanol
            'QTBSBXVTEAMEQO-UHFFFAOYSA-N': 'blue',  # Acetic acid
            'OKKJLVBELUTLKV-UHFFFAOYSA-N': 'blue',  # Methanol
            'LYCAIKOWRPUZTN-UHFFFAOYSA-N': 'blue'  # Ethylene glycol
        }

        for _, row in the_pipeline_df.iterrows():
            if row['Solvent_InChIKey'] in solvents_to_highlight:
                ax.scatter(
                    x=row['Organic Solubility Prediction'],
                    y=row['Aqueous Solubility Prediction'],
                    color=solvents_to_highlight[row['Solvent_InChIKey']],
                    s=450,
                    edgecolor='black',
                    linewidths=1.4,
                    zorder=3
                )

                # Reading the organic solubility rank from the water-miscible solvents:
                the_rank = int(the_pipeline_df_water_misc[the_pipeline_df_water_misc["Solvent_InChIKey"]
                                                    == row['Solvent_InChIKey']]["Rank"])
                x = row['Organic Solubility Prediction']
                y = row['Aqueous Solubility Prediction']
                ax.text(x, y, str(the_rank), fontsize=14, ha='center', va='center',
                        color="yellow")

            else:
                ax.scatter(
                    x=row['Organic Solubility Prediction'],
                    y=row['Aqueous Solubility Prediction'],
                    color='black',
                    s=150,
                    edgecolor='black',
                    linewidths=1.4,
                    zorder=2
                )

        # Coloring the axes:
        color_axes = True
        if color_axes is True:
            ax.spines['bottom'].set_color('#d49f00')
            ax.spines['bottom'].set_linewidth(3)
            ax.spines['left'].set_color('#0070c0')
            ax.spines['left'].set_linewidth(3)
            ax.set_xlabel('Organic Solubility Prediction \n log(x)', fontsize=18, color='#d49f00', labelpad=15)
            ax.set_ylabel('Aqueous Solubility Prediction \n log(S / mol/dm³)', fontsize=18, color='#0070c0', labelpad=15)
            ax.tick_params(axis='x', which='major', labelsize=16, length=10, width=2,
                           colors='#d49f00')
            ax.tick_params(axis='y', which='major', labelsize=16, length=10, width=2,
                           colors='#0070c0')
        else:
            ax.set_xlabel('Organic Solubility Prediction log(x)', fontsize=18)
            ax.set_ylabel('Aqueous Solubility Prediction log(S / mol/dm³)', fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=16, length=10, width=2)

        # Setting the ticks:
        if molecule_smiles == "CC1=CCC(CC1)C(=C)C":
            x_ticks = [-1.4, -1.2, -1.0, -0.8, -0.6]
            ax.set_xlim([-1.5, -0.5])
            ax.set_xticks(x_ticks)
        elif molecule_smiles == "C1=CC=C(C=C1)C=O":
            x_ticks = [-1.7, -1.5, -1.3, -1.1, -0.9, -0.7, -0.5]
            ax.set_xticks(x_ticks)
            ax.set_xlim([-1.8, -0.4])

        # Making the other spines thicker:
        ax.spines["top"].set_linewidth(2.2)
        ax.spines["right"].set_linewidth(2.2)

        # Adding the image of the mollecule of interest:
        mol = Chem.MolFromSmiles(molecule_smiles)
        img = Draw.MolToImage(mol, size=(300, 300))
        img = img.convert("RGBA")
        img = img.rotate(90)
        datas = img.getdata()
        newData = []
        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
        img.putdata(newData)

        image_size_inches = [img.size[0] / (3 * 300), img.size[1] / (3 * 300)]
        ax_img = fig.add_axes([0.72, 0.6, image_size_inches[0], image_size_inches[1]])
        ax_img.imshow(img, aspect='equal')
        ax_img.axis('off')

        ax.set_ylim(min(list(the_pipeline_df['Aqueous Solubility Prediction'])) - 1,
                    max(list(the_pipeline_df['Aqueous Solubility Prediction'])) + 1)


        ##############################################
        ##############################################
        # Saving the figure:
        main_path = f"Pipeline Predictions/Pipeline_Combined/{feature_type}_{model}/{molecule_smiles}"
        if not os.path.exists(main_path):
            os.makedirs(main_path)
            os.makedirs(main_path + "/Best CoSolvents Images")

        fig_path = main_path + f"/Highlighted_solvents_{molecule_inchikey}.png"
        plt.savefig(fig_path)
        plt.close()
