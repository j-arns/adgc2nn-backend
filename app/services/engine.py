import torch
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional, Any

# Standard relative import for a sibling file in the same directory.
# We removed the try/except block so that if 'rdkit' or 'torch' 
# is missing inside application_functions, you will see the real error.
from . import application_functions

class AdGC2NNEngine:
    """
    Encapsulates the adGC2NN model logic, weights, and normalizers.
    Replaces the global state from the original Flask app.
    """
    
    # Feature constants from the original Flask app
    SMILES_FEATURES_CONF = [
        "mass", "NumAtoms", "NumBonds", "NumSingleBonds", "NumDoubleBonds", "NumTripleBonds",
        "NumAromBonds", "AromC", "Charge", "C", "O", "H", "carboxyle", "hydroxyl", "ester", "carbonyl",
        'BertzCT', 'Ipc', 'TPSA', 'NHOHCount', 'MolMR', 'VSA_EState3', 'AvgIpc',
        'VSA_EState3', 'ExactMolWt', 'HeavyAtomMolWt', 'NumHDonors', 'Chi1', 'OC-ratio'
    ]

    SMILES_FEATURES_BROAD = [
        "mass", "NumAtoms", "NumBonds", "NumSingleBonds", "NumDoubleBonds", "NumTripleBonds",
        "NumAromBonds", "AromC", "Charge", "C", "O", "H", "Cl", "N", "I", "S", "F", "P", "Si", "Br", "B",
        "hydroxyl", "carboxyle", "ester", "amine", "amide", "carbonyl", "sulfide", "nitro", "nitrile",
        'BertzCT', 'Ipc', 'TPSA', 'NHOHCount', 'MolMR', 'VSA_EState3', 'AvgIpc',
        'VSA_EState3', 'ExactMolWt', 'HeavyAtomMolWt', 'NumHDonors', 'Chi1', 'OC-ratio'
    ]

    def __init__(self):
        self.conf_model = None
        self.broad_model = None
        self.conf_normalizer = None
        self.broad_normalizer = None
        self._load_resources()

    def _load_resources(self):
        """
        Loads models and pickles. 
        Paths are assumed to be relative to where the app is run or in services folder.
        """
        # Define paths - Update these to point to your actual file locations
        base_path = os.path.dirname(__file__)
        weights_conf = os.path.join(base_path, 'confined_model_weights.pth')
        weights_broad = os.path.join(base_path, 'broad_model_weights.pth')
        pickle_conf = os.path.join(base_path, 'conf_normalizer.pickle')
        pickle_broad = os.path.join(base_path, 'broad_normalizer.pickle')

        print(f"Loading resources from: {base_path}")

        # 1. Load Confined Model
        try:
            self.conf_model = application_functions.AdaptiveGC2NN()
            self.conf_model.load_state_dict(torch.load(weights_conf, map_location=torch.device('cpu')))
            self.conf_model.eval()
            print("✅ Confined Model loaded.")
        except Exception as e:
            raise RuntimeError(f"Failed to load Confined Model from {weights_conf}: {e}")

        # 2. Load Broad Model
        try:
            self.broad_model = application_functions.AdaptiveGC2NN(nodes_features=47, additional_input_size=43)
            self.broad_model.load_state_dict(torch.load(weights_broad, map_location=torch.device('cpu')))
            self.broad_model.eval()
            print("✅ Broad Model loaded.")
        except Exception as e:
            raise RuntimeError(f"Failed to load Broad Model from {weights_broad}: {e}")

        # 3. Load Normalizers with sklearn compatibility patch
        try:
            # Patch for older sklearn versions if needed
            import sklearn.preprocessing
            import sys
            if 'sklearn.preprocessing.data' not in sys.modules:
                sys.modules['sklearn.preprocessing.data'] = sklearn.preprocessing

            with open(pickle_conf, "rb") as f:
                self.conf_normalizer = pickle.load(f)
            print("✅ Confined Normalizer loaded.")

            with open(pickle_broad, "rb") as f:
                self.broad_normalizer = pickle.load(f)
            print("✅ Broad Normalizer loaded.")
            
        except ModuleNotFoundError as e:
            raise RuntimeError(f"Missing dependency for unpickling normalizers: {e}. Ensure scikit-learn is installed.")
        except Exception as e:
            # Common error with sklearn version mismatch
            if "No module named" in str(e):
                raise RuntimeError(f"scikit-learn version mismatch or missing module during unpickling: {e}")
            raise RuntimeError(f"Failed to load Normalizers: {e}")

        print("✅ All adGC2NN resources ready.")

    def predict_single(self, smiles: str) -> Tuple[Optional[float], str, Optional[str]]:
        """
        Logic from Flask's /predict endpoint.
        Returns: (prediction_value, model_name, error_message)
        """
        try:
            validity = application_functions.check_molecule(smiles)
            
            # CASE 1: Confined Model
            if validity == 1:
                graph_input = application_functions.create_pytorch_geometric_graph_data_list_from_smiles_and_labels([smiles])
                if graph_input[0] is None:
                    return None, "Error", "Invalid SMILES string"

                prediction = application_functions.get_pred(graph_input, self.conf_model, self.SMILES_FEATURES_CONF).reshape(-1, 1)
                pred_log = self.conf_normalizer.inverse_transform(prediction).flatten()
                return float(10 ** pred_log[0]), 'adGC2NN-confined', None

            # CASE 2: Broad Model
            elif validity == 0:
                graph_input = application_functions.create_pytorch_geometric_graph_data_list_from_smiles_and_labels([smiles], mode=1)
                if graph_input[0] is None:
                    return None, "Error", "Invalid SMILES string"

                prediction = application_functions.get_pred(graph_input, self.broad_model, self.SMILES_FEATURES_BROAD).reshape(-1, 1)
                pred_log = self.broad_normalizer.inverse_transform(prediction).flatten()
                return float(10 ** pred_log[0]), 'adGC2NN-broad', None

            # CASE 3: Invalid
            else:
                return None, "Error", "Not a valid SMILES string, or out of scope"

        except Exception as e:
            return None, "Error", f"Computation Error: {str(e)}"

    def predict_batch(self, smiles_list: List[str]) -> List[dict]:
        """
        Logic from Flask's /batch_predict endpoint.
        Optimized to process valid items in batch tensors rather than loops.
        """
        # 1. Check validity
        validity = [application_functions.check_molecule(s) for s in smiles_list]

        # 2. Sort into buckets
        smiles_conf = [smiles_list[i] for i in range(len(smiles_list)) if validity[i] == 1]
        smiles_broad = [smiles_list[i] for i in range(len(smiles_list)) if validity[i] == 0]

        # 3. Process Confined Batch
        pa_pred_conf = []
        if smiles_conf:
            graph_input = application_functions.create_pytorch_geometric_graph_data_list_from_smiles_and_labels(smiles_conf)
            # Filter valid graphs
            valid_indices = [i for i, g in enumerate(graph_input) if g is not None]
            valid_graphs = [graph_input[i] for i in valid_indices]
            
            if valid_graphs:
                pred_raw = application_functions.get_pred(valid_graphs, self.conf_model, self.SMILES_FEATURES_CONF).reshape(-1, 1)
                pred_log = self.conf_normalizer.inverse_transform(pred_raw).flatten()
                pa_pred_conf = [10 ** p for p in pred_log]

        # 4. Process Broad Batch
        pa_pred_broad = []
        if smiles_broad:
            graph_input = application_functions.create_pytorch_geometric_graph_data_list_from_smiles_and_labels(smiles_broad, mode=1)
            # Filter valid graphs
            valid_indices = [i for i, g in enumerate(graph_input) if g is not None]
            valid_graphs = [graph_input[i] for i in valid_indices]
            
            if valid_graphs:
                pred_raw = application_functions.get_pred(valid_graphs, self.broad_model, self.SMILES_FEATURES_BROAD).reshape(-1, 1)
                pred_log = self.broad_normalizer.inverse_transform(pred_raw).flatten()
                pa_pred_broad = [10 ** p for p in pred_log]

        # 5. Reconstruct Results in Order
        results = []
        conf_idx = 0
        broad_idx = 0

        for i in range(len(smiles_list)):
            v = validity[i]
            res_item = {"smiles": smiles_list[i], "prediction": None, "model": "Invalid"}

            if v == 1:
                if conf_idx < len(pa_pred_conf):
                    res_item["prediction"] = pa_pred_conf[conf_idx]
                    res_item["model"] = "adGC2NN-confined"
                    conf_idx += 1
                else:
                    res_item["model"] = "Invalid SMILES"
            elif v == 0:
                if broad_idx < len(pa_pred_broad):
                    res_item["prediction"] = pa_pred_broad[broad_idx]
                    res_item["model"] = "adGC2NN-broad"
                    broad_idx += 1
                else:
                    res_item["model"] = "Invalid SMILES"
            else:
                 res_item["model"] = "Out of Scope"
            
            results.append(res_item)
            
        return results

# Singleton instance
engine = AdGC2NNEngine()