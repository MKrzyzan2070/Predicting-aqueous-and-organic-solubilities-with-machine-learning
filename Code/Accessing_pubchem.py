import pubchempy as pcp
import requests
import time
from random import uniform


def make_request_with_retry(url, max_retries=3, base_delay=1):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 503 and attempt < max_retries - 1:
                # Wait with exponential backoff + jitter
                delay = base_delay * (2 ** attempt) + uniform(0, 1)
                print(f"Server busy (503), retrying in {delay:.2f} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            else:
                raise
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + uniform(0, 1)
                print(f"Request failed, retrying in {delay:.2f} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            else:
                raise


def inchikey_to_smiles(inchikey_list):
    smiles_list = []

    # 11111111      Pubchempy will be tried first      11111111
    for inchikey in inchikey_list:
        try:
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
        except Exception as e:
            print(f"Pubchempy failed for {inchikey}: {e}")
            continue

    if smiles_list:
        return smiles_list
    # 11111111      Pubchempy will be tried first      11111111

    # 22222222      Direct pubchem website access with retry logic      22222222
    base_url = "https://pubchem.ncbi.nlm.nih.gov"

    for i, inchikey in enumerate(inchikey_list):
        try:
            # Adding delay between requests to avoid overwhelming the server
            if i > 0:
                time.sleep(1)

            url = f"{base_url}/rest/pug/compound/inchikey/{inchikey}/json"
            print(f"Fetching data for InChIKey: {inchikey}")

            response = make_request_with_retry(url, max_retries=3, base_delay=2)
            data = response.json()

            props = data['PC_Compounds'][0]['props']
            for p in props:
                if p['urn']['label'] == 'SMILES':
                    try:
                        smiles = p['value']['sval']
                        smiles_list.append(smiles)
                        print(f"Successfully retrieved SMILES for {inchikey}")
                        break
                    except:
                        continue

        except Exception as e:
            print(f"Failed to retrieve data for {inchikey}: {e}")
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
        try:
            compound = pcp.get_compounds(inchikey, 'inchikey')[0]
            # Looking at all the possible options:
            name_options = [
                getattr(compound, 'iupac_name', None),
                getattr(compound, 'synonyms', None)
            ]

            if name_options[0] is not None and name_options[0].strip() != "":
                name_list.append(name_options[0])
        except Exception as e:
            print(f"Pubchempy failed for {inchikey}: {e}")
            continue

    if name_list:
        return name_list
    # 11111111      Pubchempy will be tried first      11111111

    # 22222222      Direct pubchem website access with retry logic      22222222
    base_url = "https://pubchem.ncbi.nlm.nih.gov"

    for i, inchikey in enumerate(inchikey_list):
        try:
            # Add delay between requests
            if i > 0:
                time.sleep(1)

            url = f"{base_url}/rest/pug/compound/inchikey/{inchikey}/json"
            print(f"Fetching name for InChIKey: {inchikey}")

            response = make_request_with_retry(url, max_retries=3, base_delay=2)
            data = response.json()

            props = data['PC_Compounds'][0]['props']
            for p in props:
                if p['urn']['label'] == 'IUPAC Name':
                    try:
                        name = p['value']['sval']
                        name_list.append(name)
                        print(f"Successfully retrieved name for {inchikey}")
                        break
                    except:
                        continue

        except Exception as e:
            print(f"Failed to retrieve name for {inchikey}: {e}")
            continue
    # 22222222      Direct pubchem website access      22222222

    if not name_list:
        print("No compound names for the molecule of interest were found!")
        return None
    else:
        return name_list