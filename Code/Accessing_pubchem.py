import pubchempy as pcp
import requests


def inchikey_to_smiles(inchikey_list):

    smiles_list = []

    # 11111111      Pubchempy will be tried first      11111111
    for inchikey in inchikey_list:
        compound = pcp.get_compounds(inchikey, 'inchikey')[0]
        # Looking at all the possible options:
        smiles_options = [
            getattr(compound, 'canonical_smiles', None),
            getattr(compound, 'isomeric_smiles', None),
            getattr(compound, 'connectivity_smiles', None)
        ]

        for smiles in smiles_options:
            if smiles is not None and smiles.strip() != "":
                smiles_list.append(smiles)
                break
    if smiles_list:
        return smiles_list
    # 11111111      Pubchempy will be tried first      11111111


    # 22222222      Direct pubchem website access      22222222
    # If Pubchempy doesn't work, which is likely, the json version of the pubchem website will be
    # accessed and the relevant information will be extracted from it
    base_url = "https://pubchem.ncbi.nlm.nih.gov"
    # Query PubChem for the compound using InChIKey
    for inchikey in inchikey_list:
        url = f"{base_url}/rest/pug/compound/inchikey/{inchikey}/json"

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        props = data['PC_Compounds'][0]['props']
        for p in props:
            if p['urn']['label'] == 'SMILES':
                try:
                    smiles = p['value']['sval']
                    smiles_list.append(smiles)
                    break
                except:
                    continue
    # 22222222      Direct pubchem website access      22222222

    if not smiles_list:
        print("No SMILES string for the molecule of interest were found!")
        return None
    else:
        return smiles_list


def inchikey_to_name(inchikey_list):
    name_list = []

    # 11111111      Pubchempy will be tried first      11111111
    for inchikey in inchikey_list:
        compound = pcp.get_compounds(inchikey, 'inchikey')[0]
        # Looking at all the possible options:
        name_options = [
            getattr(compound, 'iupac_name', None),
            getattr(compound, 'synonyms', None)
        ]

        if name_options[0] is not None and name_options[0].strip() != "":
            name_list.append(name_options[0])

    if name_list:
        return name_list
    # 11111111      Pubchempy will be tried first      11111111

    # 22222222      Direct pubchem website access      22222222
    # If Pubchempy doesn't work, which is likely, the json version of the pubchem website will be
    # accessed and the relevant information will be extracted from it
    base_url = "https://pubchem.ncbi.nlm.nih.gov"
    # Query PubChem for the compound using InChIKey
    for inchikey in inchikey_list:
        url = f"{base_url}/rest/pug/compound/inchikey/{inchikey}/json"

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        props = data['PC_Compounds'][0]['props']
        for p in props:
            if p['urn']['label'] == 'IUPAC Name':
                try:
                    name = p['value']['sval']
                    name_list.append(name)
                    break
                except:
                    continue
    # 22222222      Direct pubchem website access      22222222

    if not name_list:
        print("No compound names for the molecule of interest were found!")
        return None
    else:
        return name_list



