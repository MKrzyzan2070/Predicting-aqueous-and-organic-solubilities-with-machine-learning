# Predicting aqueous and organic solubilities with machine learning: a workflow for identifying organic co-solvents

This project introduces a machine learning workflow to identify organic co-solvents for a molecule of interest. The workflow combines two solubility models—one being the aqueous solubility model and the other being the organic solvent solubility model. With the aqueous solubility model, the miscibility of potential co-solvents in water is predicted so that only water-miscible solvents are considered. With the organic solvent solubility model, the solvents are ranked based on the predicted solubility of the molecule of interest in them. 


## Project Structure

- **Code/**: All the Python code files can be found here. The scripts in the main repository, such as `AqSolDB_main.py`, call the functions contained in these files.
- **Datasets/**: Contains data files.  
- **Plots Images/**: Generated visualizations from the analysis.  
- **Pickled Models/**: For transparency, all the LGBM models were saved along with the data files on which the models were trained or their performance verified.  
- **Hyperparameters/**: Configuration files for the hyperparameters of the models.  
- **Jupyter Notebook/**: Contains code for determining the miscibility threshold.  
- **Pipeline Predictions/**: Results from the workflow.  
- **Experimental Comparison/**: Data files for the case study analysis that evaluates the organic solubility model's generalizability.


## Scripts to run (found in the main repository) 

- `AqSolDB_main.py`: Calls the functions that generate the dataset for training, testing, and validation, as well as the subsequent analysis of aqueous solubility (including generating the plots found in the manuscript). **Please see the comments at the beginning of the file for detailed instructions on how to configure the script for different usage scenarios.**

- `BigSolDB_main.py`: Same as above, but for organic solvent solubility.

- `Solvents_Pipeline.py`: This file mainly contains code for creating the images and plots used to visualize the identification of co-solvents. **As such, `BigSolDB_main.py` and `AqSolDB_main.py` must be run beforehand to generate the relevant datasets (see comments at the beginning of the file).** 

- `Case_study_analysis.py`: Script for the case study analysis, i.e., the assessment of the generalizability of the model.

### **IMPORTANT INFORMATION**
The scripts can be run through the Python console or in an IDE such as PyCharm.
**All scripts must be run from the project root directory** Basically, when you clone the repo it can't be in a folder within a folder.
So, when cloning the repository, make sure that it is not placed inside an extra nested folder.

Some scripts access external websites. In certain cases, frequent or automated access may trigger a purge or rate-limit request from those sites.
In this case, run the code again some other time because the server might be overloaded. 

## Dependencies

- Python >= 3.10 
- pandas >= 2.2.2
- numpy >= 1.16.4
- rdkit >= 2023.9.6
- scikit-learn >= 1.5.0
- scipy >= 1.13.0
- lightgbm >= 4.3.0
- matplotlib >= 3.8.4
- thermo >= 0.2.27
- pubchempy >= 1.0.4







