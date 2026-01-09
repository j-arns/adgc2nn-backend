import numpy as np

# pytorch
from torch_geometric.loader import DataLoader
from torch.nn import Linear, BatchNorm1d, ModuleList, Module
from torch import manual_seed, tensor, arange, long, sigmoid, tanh, cat, zeros_like, full, from_numpy, stack
from torch import float as tensorfloat
from torch.nn.functional import leaky_relu, relu, dropout
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch

# rdkit
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem import Descriptors, MolFromSmiles, AddHs, GetFormalCharge, MolFromSmarts, GetDistanceMatrix, GetPeriodicTable
from rdkit.Chem.rdchem import BondType


def external_process_smiles(smiles, smiles_features):
    full_output_vector = []

    for smil in smiles:
        mol = MolFromSmiles(smil)
        molH = AddHs(mol)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        output_vector = []

        for feature in smiles_features:
            if feature == "mass":
                mass = Descriptors.MolWt(mol) / 1000
                output_vector.append(mass)
            elif feature == "ExactMolWt":
                value = Descriptors.ExactMolWt(mol) / 1000  
                output_vector.append(value)
            elif feature == 'TPSA':
                value = Descriptors.TPSA(mol) / 320
                output_vector.append(value)
            elif feature == 'HeavyAtomMolWt':
                value = Descriptors.HeavyAtomMolWt(mol) / 1000
                output_vector.append(value)
            elif feature == 'VSA_EState3':
                value = (Descriptors.VSA_EState3(mol) + 40) / 160
                output_vector.append(value)
            elif feature == 'NumHDonors':
                value = Descriptors.NumHDonors(mol) / mol.GetNumAtoms()
                output_vector.append(value)
            elif feature == 'Chi1':
                value = Descriptors.Chi1(mol) / 30
                output_vector.append(value)
            elif feature == 'BertzCT':
                value = Descriptors.BertzCT(mol) / 3000
                output_vector.append(value)
            elif feature == 'Ipc':
                value = Descriptors.Ipc(mol) / 60000000000000
                output_vector.append(value)
            elif feature == 'NHOHCount':
                value = Descriptors.NHOHCount(mol)/13
                output_vector.append(value)
            elif feature == 'MolMR':
                value = Descriptors.MolMR(mol) / 300
                output_vector.append(value)
            elif feature == 'AvgIpc':
                value = Descriptors.AvgIpc(mol) / 5
                output_vector.append(value)
            elif feature == "NumAtoms":
                numatoms = mol.GetNumAtoms() / 70
                output_vector.append(numatoms)
            elif feature == "NumBonds":
                numbonds = mol.GetNumBonds() / 70
                output_vector.append(numbonds)
            elif feature == "NumSingleBonds":
                try:
                    numsinglebonds = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == BondType.SINGLE) / mol.GetNumBonds()
                    output_vector.append(numsinglebonds)
                except:
                    output_vector.append(0)
            elif feature == "NumDoubleBonds":
                try:
                    numdoublebonds = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == BondType.DOUBLE) / mol.GetNumBonds()
                    output_vector.append(numdoublebonds)
                except:
                    output_vector.append(0)
            elif feature == "NumTripleBonds":
                try:
                    numtriplebonds = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == BondType.TRIPLE) / mol.GetNumBonds()
                    output_vector.append(numtriplebonds)
                except:
                    output_vector.append(0)
            elif feature == "NumAromBonds":
                try:
                    numarombonds = sum(1 for bond in mol.GetBonds() if bond.GetIsAromatic()) / mol.GetNumBonds()
                    output_vector.append(numarombonds)
                except:
                    output_vector.append(0)
            elif feature == "AromC":
                aromC = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C" and atom.GetIsAromatic()) / mol.GetNumAtoms()
                output_vector.append(aromC)
            elif feature == "Charge":
                charge = (GetFormalCharge(mol) + 2) / 4
                output_vector.append(charge)
            elif feature == "OC-ratio":
                C_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C")
                O_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "O")
                try:
                    output_vector.append((O_count/C_count)/8)
                except:
                    output_vector.append(1)
            elif feature == "H":
                atom_count = sum(1 for atom in molH.GetAtoms() if atom.GetSymbol() == "H") / molH.GetNumAtoms()
                output_vector.append(atom_count)
            elif len(feature) < 3:
                # if small, assume it's an element symbol and count the number of atoms
                atom_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == feature) / mol.GetNumAtoms()
                output_vector.append(atom_count)
            elif feature == "carboxyle":
                pattern = MolFromSmarts("C(=O)O")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "ester":
                pattern = MolFromSmarts("C(=O)OC")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "hydroxyl":
                pattern = MolFromSmarts("[OH]")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "carbonyl":
                pattern = MolFromSmarts("C=O")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "amine":
                pattern = MolFromSmarts("N")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "amide":
                pattern = MolFromSmarts("C(=O)N")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "sulfide":
                pattern = MolFromSmarts("S")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "nitro":
                pattern = MolFromSmarts("N(=O)=O")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "nitrile":
                pattern = MolFromSmarts("C#N")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "ketone":
                pattern = MolFromSmarts("C(=O)C")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "hydroperoxide":
                pattern = MolFromSmarts("OO")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "nitrate":
                pattern = MolFromSmarts("O[N+](=O)[O-]")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "aldehyde":
                pattern = MolFromSmarts("C=O")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "carboxylic acid":
                pattern = MolFromSmarts("C(=O)O")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "peroxide":
                pattern = MolFromSmarts("O-O")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "carbonylperoxynitrate":
                pattern = MolFromSmarts("C(=O)OON(=O)=O")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "ether":
                pattern = MolFromSmarts("C-O-C")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "nitroester":
                pattern = MolFromSmarts("C(=O)ON(=O)")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            else:
                raise ValueError("Feature " + feature + " not found.")

        full_output_vector.append(output_vector)

    return full_output_vector

class AdaptiveGC2NN(Module):
    "Full adaptive-depth GC2NN model."
    def __init__(self, nodes_features=29, embedding_sizes=[32, 16, 64, 16, 32], n_outputs=1, edge_dim=10, dropout_rates=[0], activations=["LeakyReLU", "LeakyReLU", "ReLU", "ReLU", "LeakyReLU"], activation_hidden="Tanh", activation_post="Tanh", pass_edge_attr=[0, 1, 0, 0, 1], layer_types=["GCN", "GAT", "GCN", "GCN", "GAT"], batch_norm_layers=[0], heads=[0, 6, 0, 0, 6], additional_input_size=29, hidden_layer_sizes=[32, 32], post_hidden_layer_sizes=[8]):
        super(AdaptiveGC2NN, self).__init__()
        manual_seed(42)

        # Expand architecture parameters
        self.embedding_sizes = embedding_sizes
        self.heads = self._expand_param(heads, len(embedding_sizes))
        self.dropout_rates = self._expand_param(dropout_rates, len(embedding_sizes))
        self.activations = self._expand_param(activations, len(embedding_sizes))
        self.activation_hidden = activation_hidden
        self.activation_post = activation_post
        self.pass_edge_attr = self._expand_param(pass_edge_attr, len(embedding_sizes))
        self.layer_types = self._expand_param(layer_types, len(embedding_sizes))
        self.batch_norm_layers = self._expand_param(batch_norm_layers, len(embedding_sizes))
        self.additional_input_size = additional_input_size

        # Initialize convolutional layers and batch normalization
        self.convs = ModuleList()
        self.bnss = ModuleList()
        self.additional_hl = ModuleList()
        self.out_layers = ModuleList()
        self.post_layers = ModuleList()
        self.post_layers_out = ModuleList()

        # additional hidden layer
        if len(hidden_layer_sizes) > 0:
            for i in range(len(hidden_layer_sizes)):
                in_channels = additional_input_size if i == 0 else hidden_layer_sizes[i - 1]
                out_channels = hidden_layer_sizes[i]
                hl = Linear(in_channels, out_channels)
                self.additional_hl.append(hl)
            additional_channels = hidden_layer_sizes[-1]
        else:
            additional_channels = additional_input_size

        for i in range(len(embedding_sizes)):
            in_channels = nodes_features if i == 0 else embedding_sizes[i - 1]
            out_channels = embedding_sizes[i]

            if "GAT" in self.layer_types[i]:
                conv = GATConv(in_channels, out_channels, edge_dim=edge_dim if self.pass_edge_attr[i] else None, heads=self.heads[i], concat=False)
            elif "GCN" in self.layer_types[i]:
                conv = GCNConv(in_channels, out_channels)

            self.convs.append(conv)
            self.bnss.append(BatchNorm1d(out_channels) if self.batch_norm_layers[i] else None)
            if len(post_hidden_layer_sizes) == 0:
                self.out_layers.append(Linear(out_channels * 2 + additional_channels, n_outputs))
            elif len(post_hidden_layer_sizes) == 1:
                self.out_layers.append(Linear(out_channels * 2 + additional_channels, post_hidden_layer_sizes[0]))
                self.post_layers_out.append(Linear(post_hidden_layer_sizes[0], n_outputs))
            elif len(post_hidden_layer_sizes) == 2:
                self.out_layers.append(Linear(out_channels * 2 + additional_channels, post_hidden_layer_sizes[0]))
                self.post_layers.append(Linear(post_hidden_layer_sizes[0], post_hidden_layer_sizes[1]))
                self.post_layers_out.append(Linear(post_hidden_layer_sizes[1], n_outputs))
            else:
                raise ValueError("Only a maximum of two post hidden layers are permitted!")

    def _expand_param(self, param, length):
        return [param[0] for _ in range(length)] if len(param) == 1 else param

    def process_smiles(self, smiles, smiles_features):
        return tensor(external_process_smiles(smiles, smiles_features), dtype=tensorfloat)

    def forward(self, x, edge_index, batch_index, edge_attr, smiles, smiles_features):

        if self.additional_input_size > 0:
            additional_input = self.process_smiles(smiles, smiles_features)

        graph_distances = [int(GetDistanceMatrix(MolFromSmiles(smil)).max() - 1) if int(GetDistanceMatrix(MolFromSmiles(smil)).max() - 1) > 0 else 1 for smil in smiles]

        outputs = []
        hidden_states = []

        # Iterate over each graph in the batch
        for graph_id in range(batch_index.max().item() + 1):
            # Mask to select nodes and edges for the current graph
            node_mask = (batch_index == graph_id)
            edge_mask = (batch_index[edge_index[0]] == graph_id) & (batch_index[edge_index[1]] == graph_id)

            # Extract subgraph components
            x_sub = x[node_mask]
            edge_index_sub = edge_index[:, edge_mask]
            edge_attr_sub = edge_attr[edge_mask] if edge_attr is not None else None

            # Re-index edge_index_sub to be relative to x_sub
            node_indices = arange(x.size(0))[node_mask]
            node_index_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(node_indices)}
            edge_index_sub = tensor(
                [[node_index_map[idx.item()] for idx in edge_index_sub_row] for edge_index_sub_row in edge_index_sub],
                dtype=long)

            # Determine the number of layers for this graph
            max_distance = graph_distances[graph_id]
            hidden = x_sub

            for n_conv in range(min(max_distance, len(self.convs))):
                conv = self.convs[n_conv]
                bn = self.bnss[n_conv]

                if self.pass_edge_attr[n_conv]:
                    hidden = conv(hidden, edge_index_sub, edge_attr=edge_attr_sub)
                else:
                    hidden = conv(hidden, edge_index_sub)

                if bn is not None:
                    hidden = bn(hidden)

                activation = self.activations[n_conv]
                if activation == "LeakyReLU":
                    hidden = leaky_relu(hidden)
                elif activation == "ReLU":
                    hidden = relu(hidden)
                elif activation == "Tanh":
                    hidden = tanh(hidden)
                elif activation == "Sigmoid":
                    hidden = sigmoid(hidden)

                if self.dropout_rates[n_conv] > 0:
                    hidden = dropout(hidden, p=self.dropout_rates[n_conv], training=self.training)

            # Global pooling for the current graph
            pooled_hidden = cat([gmp(hidden, zeros_like(batch_index[node_mask])),
                                       gap(hidden, zeros_like(batch_index[node_mask]))], dim=1)

            # Use the output layer corresponding to the number of layers used
            if len(self.additional_hl) == 0 and self.additional_input_size == 0: # no additional input
                out = self.out_layers[min(max_distance, len(self.convs)) - 1](pooled_hidden)
                if self.activation_hidden == "LeakyReLU":
                    out = leaky_relu(out)
                elif self.activation_hidden == "ReLU":
                    out = relu(out)
                elif self.activation_hidden == "Tanh":
                    out = tanh(out)
                elif self.activation_hidden == "Sigmoid":
                    out = sigmoid(out)
            elif len(self.additional_hl) == 0 and self.additional_input_size > 0: # smiles features directly in output layer
                add = additional_input[graph_id].view(1,-1)
                combined = cat([pooled_hidden, add], dim=1)
                out = self.out_layers[min(max_distance, len(self.convs)) - 1](combined)
                if self.activation_hidden == "LeakyReLU":
                    out = leaky_relu(out)
                elif self.activation_hidden == "ReLU":
                    out = relu(out)
                elif self.activation_hidden == "Tanh":
                    out = tanh(out)
                elif self.activation_hidden == "Sigmoid":
                    out = sigmoid(out)
            elif len(self.additional_hl) > 0:
                add = additional_input[graph_id].view(1,-1)
                for hl in range(len(self.additional_hl)):
                    add = self.additional_hl[hl](add)
                combined = cat([pooled_hidden, add], dim=1)
                out = self.out_layers[min(max_distance, len(self.convs)) - 1](combined)
                if self.activation_hidden == "LeakyReLU":
                    out = leaky_relu(out)
                elif self.activation_hidden == "ReLU":
                    out = relu(out)
                elif self.activation_hidden == "Tanh":
                    out = tanh(out)
                elif self.activation_hidden == "Sigmoid":
                    out = sigmoid(out)

            if len(self.post_layers) > 0:
                out = self.post_layers[min(max_distance, len(self.convs)) - 1](out)
                if self.activation_post == "LeakyReLU":
                    out = leaky_relu(out)
                elif self.activation_post == "ReLU":
                    out = relu(out)
                elif self.activation_post == "Tanh":
                    out = tanh(out)
                elif self.activation_post == "Sigmoid":
                    out = sigmoid(out)
            if len(self.post_layers_out) > 0:
                out = self.post_layers_out[min(max_distance, len(self.convs)) - 1](out)
                if self.activation_post == "LeakyReLU":
                    out = leaky_relu(out)
                elif self.activation_post == "ReLU":
                    out = relu(out)
                elif self.activation_post == "Tanh":
                    out = tanh(out)
                elif self.activation_post == "Sigmoid":
                    out = sigmoid(out)

            outputs.append(out)
            hidden_states.append(pooled_hidden)

        # Concatenate results for all graphs in the batch
        outputs = cat(outputs, dim=0)
        #hidden_states = torch.cat(hidden_states, dim=0)

        return outputs, hidden_states

def get_specific_atom_features(atom, features, permitted_atoms, add_hydrogen):
    """Possible features: atom_type, n_heavy_neighbors, formal_charge, hybridisation_type, is_in_a_ring, is_aromatic,
    atomic_mass_scaled, vdw_radius_scaled, covalent_radius_scaled, chirality_type, hydrogens_implicit"""

    # Consider 1-hot encoding
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_atoms)

    # Number of heavy atoms
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])

    # Formal Charge
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])

    # Hybridisation
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])

    # Atom in a ring or not
    is_in_a_ring_enc = [int(atom.IsInRing())]

    # Atom aromatic
    is_aromatic_enc = [int(atom.GetIsAromatic())]

    # Get Atomic mass
    atomic_mass_scaled = [float(atom.GetMass() / 130)] # largest element: iodine

    # Get Van Der Walls radius
    vdw_radius_scaled = [float(GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) / 2.1)]

    # Get Scaled covalent radius
    covalent_radius_scaled = [float(GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) / 1.4)]

    # Get 1-hot encoding of chirality
    chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])

    # Get 1-hot encoding of implicit hydrogens if activated (values from 1-hot encoding are )
    if not add_hydrogen:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])

    # Build atomic feature vector combining previous characteristics
    atom_feature_vector = atom_type_enc
    if "heavy_neighbors" in features:
        atom_feature_vector += n_heavy_neighbors_enc
    if "formal_charge" in features:
        atom_feature_vector += formal_charge_enc
    if "hybridisation_type" in features:
        atom_feature_vector += hybridisation_type_enc
    if "is_in_a_ring" in features:
        atom_feature_vector += is_in_a_ring_enc
    if "is_aromatic" in features:
        atom_feature_vector += is_aromatic_enc
    if "atomic_mass" in features:
        atom_feature_vector += atomic_mass_scaled
    if "vdw_radius" in features:
        atom_feature_vector += vdw_radius_scaled
    if "covalent_radius" in features:
        atom_feature_vector += covalent_radius_scaled
    if "chirality_type" in features:
        atom_feature_vector += chirality_type_enc
    if "hydrogens_implicit" in features and not add_hydrogen:
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)

def get_specific_bond_features(bond, features):
    """Possible features: bond_type, bond_is_conj, bond_is_in_ring, stereo_type"""

    # Define the type of bonds consider valid in molecules from dataset
    permitted_list_of_bond_types = [
        BondType.SINGLE,
        BondType.DOUBLE,
        BondType.TRIPLE,
        BondType.AROMATIC
    ]

    # Generate 1-hot encoding of bond types
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)

    # Bond is conjugated or not
    bond_is_conj_enc = [int(bond.GetIsConjugated())]

    # Bond is in ring or not
    bond_is_in_ring_enc = [int(bond.IsInRing())]

    stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])

    # Generate final bond feature vector by combining previous characteristics
    bond_feature_vector = bond_type_enc
    if "bond_is_conj" in features:
        bond_feature_vector += bond_is_conj_enc
    if "bond_is_in_ring" in features:
        bond_feature_vector += bond_is_in_ring_enc
    if "stereo_type" in features:
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)

def one_hot_encoding(x, permitted_list):
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding

class CustomBatchSMILES(Batch):
    @staticmethod
    def from_data_list(data_list):
        batch = CustomBatchSMILES()
        for key in data_list[0].keys:
            batch[key] = []
        batch.smiles = []

        for data in data_list:
            for key in data.keys:
                batch[key].append(data[key])
            batch.smiles.append(data.smiles)

        for key in batch.keys:
            batch[key] = cat(batch[key], dim=0)

        return batch

def custom_collate(batch):
    batch_x = cat([item.x for item in batch], dim=0)
    batch_edge_index = cat([item.edge_index for item in batch], dim=1)
    batch_edge_attr = cat([item.edge_attr for item in batch], dim=0)
    batch_smiles = [item.smiles for item in batch]
    batch_y = cat([item.y for item in batch], dim=0)
    # Creating the batch tensor for graph identification
    batch_size = len(batch)
    batch_list = []
    for i, item in enumerate(batch):
        num_nodes = item.x.size(0)
        batch_list.append(full((num_nodes,), i, dtype=long))
    batch_tensor = cat(batch_list, dim=0)

    return CustomBatchSMILES(batch_x, batch_edge_index, batch_tensor, batch_edge_attr, batch_y, batch_smiles)


def get_pred(graphs_list_for_pred, trained_model, smiles_features=[]) -> np.ndarray:

    full_graph_dataloader = DataLoader(graphs_list_for_pred, batch_size=64, shuffle=False, drop_last=False, collate_fn=CustomBatchSMILES.from_data_list)
    tensor_pred_list = []
    for (k, batch_p) in enumerate(full_graph_dataloader):
        pred, embedding = trained_model(batch_p.x.float(), batch_p.edge_index, batch_p.batch, batch_p.edge_attr, batch_p.smiles, smiles_features)
        np_pred = pred.detach().numpy().flatten()
        tensor_pred_list.append(pred)

    # Convert each tensor to a numpy array after detaching from the computation graph and flatten it
    pred_numpy_arrays = [t.detach().numpy().flatten() for t in tensor_pred_list]

    # Concatenate all numpy arrays into a single array
    array_of_predictions = np.concatenate(pred_numpy_arrays)

    return array_of_predictions



def create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, mode=0):
    data_list = []

    for smi in x_smiles:
        try:
            # convert SMILES to RDKit mol object
            mol = MolFromSmiles(smi)
            if mol is None:
                raise ValueError("Invalid SMILES string")

            # get feature dimensions for nodes and edges
            n_nodes = mol.GetNumAtoms()
            n_edges = 2 * mol.GetNumBonds()

            # construct node feature matrix X of shape (n_nodes, n_node_features)
            if mode == 0:
                X = np.zeros((n_nodes, 29))
            else:
                X = np.zeros((n_nodes, 47))

            for atom in mol.GetAtoms():
                if mode == 0:
                    X[atom.GetIdx(), :] = get_specific_atom_features(atom, ["atom_type", "heavy_neighbors", "hybridisation_type", "is_in_a_ring",
                        "atomic_mass", "vdw_radius", "covalent_radius", "chirality_type", "hydrogens_implicit"], ["C", "O"], False)
                else:
                    X[atom.GetIdx(), :] = get_specific_atom_features(atom,
                                                                     ["atom_type", "heavy_neighbors", "formal_charge",
                                                                      "hybridisation_type", "is_in_a_ring", "is_aromatic",
                                                                      "atomic_mass", "vdw_radius", "covalent_radius",
                                                                      "chirality_type", "hydrogens_implicit"], ['C', 'O', 'N', 'F', 'Br', 'Cl', 'S', 'P', 'I', 'B', 'Si'],
                                                                     False)

            X = tensor(X, dtype=tensorfloat)

            # Construct edge index array E of shape (2, n_edges)
            (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))

            torch_rows = from_numpy(rows.astype(np.int64)).to(long)
            torch_cols = from_numpy(cols.astype(np.int64)).to(long)
            E = stack([torch_rows, torch_cols], dim=0)

            # Construct edge feature array EF of shape (n_edges, n_edge_features)
            EF = np.zeros((n_edges, 10))

            for (k, (i, j)) in enumerate(zip(rows, cols)):
                EF[k] = get_specific_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)), ["bond_type", "bond_is_conj", "bond_is_in_ring", "stereo_type"])

            EF = tensor(EF, dtype=tensorfloat)

            # Construct label tensor
            y_tensor = tensor(np.array([[0]]), dtype=tensorfloat)

            # Construct Pytorch Geometric data object and append to data list
            data_list.append(Data(x=X, edge_index=E, edge_attr=EF, smiles=smi, y=y_tensor))

        except Exception as e:
            print(f"Error processing SMILES '{smi}': {e}")
            data_list.append(None)  # Append None for invalid SMILES

    return data_list


def check_molecule(input_string):
    # transform to rdkit molecule
    mol = MolFromSmiles(input_string)

    # if molecule is not valid, return -1
    if mol is None:
        return -1

    # get elements present in the molecule
    elements = {atom.GetSymbol() for atom in mol.GetAtoms()}

    # define allowed elements for conf and broad
    conf_elements = {'C', 'O', 'H'}
    broad_elements = {'C', 'O', 'H', 'Cl', 'N', 'I', 'S', 'F', 'P', 'Si', 'Br', 'B'}

    # check if molecule is aromatic
    is_aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())

    # check for conf validity: only C, O, H and no aromatics
    if elements <= conf_elements and not is_aromatic:
        return 1

    # check for broad validity: only allowed elements including aromatics
    if elements <= broad_elements:
        return 0

    # if neither conf nor broad return -1
    return -1