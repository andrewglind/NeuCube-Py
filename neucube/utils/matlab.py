import torch
import pandas as pd
import numpy as np
from scipy.io import savemat

def generate_datasetfile(file_name, data, feature_names, labels, attrs):
    """
    Generate NeuCube datafile
    """
    _data = np.transpose(data, (1, 2, 0))
    time, n_features, batch_size = _data.shape

    mat_data = {
        'file_name': file_name,
        'data': _data,
        'length_per_sample': float(time),
        'feature_number': float(n_features),
        'total_sample_number': float(batch_size),
        'feature_name': feature_names.reshape(n_features, 1),
        'number_of_class': float(len(labels.drop_duplicates())),
        'target_value':  labels * 1.0,
        'type': float(attrs['type']),
        'training_set_ratio': float(attrs['training_set_ratio']),
        'training_data': [], # ignored attribute
        'target_value_for_training': [], # ignored attribute
        'spike_state_for_training': [], # ignored attribute
        'training_time_length': float(time),
        'sample_amount_for_training': float(attrs['sample_amount_for_training']),
        'training_sample_id': [], # ignored attribute
        'validation_data': [], # ignored attribute
        'target_value_for_validation': [], # ignored attribute
        'spike_state_for_validation': [], # ignored attribute
        'validation_time_length': float(time),
        'sample_amount_for_validation': float(attrs['sample_amount_for_validation']),
        'predict_value_for_validation': [], # ignored attribute
        'encoding': {
            'method': float(attrs['method']),
            'spike_threshold': float(attrs['spike_threshold']),
            'window_size': float(attrs['window_size']),
            'filter_type': float(attrs['filter_type'])
        }
    }

    savemat(file_name, { 'dataset': mat_data })
    print(f"MATLAB file '{file_name}' has been generated.")

def generate_neucubefile(file_name, feature_names, brain_coordinates, eeg_mapping, neumids, weights, attrs):
    """
    Generate NeuCube cube
    """
    # TODO: hack to merge arrays of different shape
    input_mapping = np.array([eeg_mapping.astype(float),
                              np.transpose([np.insert(feature_names, len(feature_names), '')])],
                             dtype=object).reshape(1, 2)
    eeg_mapping_tensor = torch.tensor(eeg_mapping)
    brain_coordinates_tensor = torch.tensor(brain_coordinates)

    indices_of_input_neuron = torch.ones(len(feature_names), dtype=torch.int) * -1
    for k in range(eeg_mapping_tensor.size(0)):
        coord = eeg_mapping_tensor[k]
        L = torch.all(brain_coordinates_tensor == coord, dim=1)
        idx = torch.nonzero(L).squeeze()
        idx, _ = torch.sort(idx)
        indices_of_input_neuron[k] = idx[0]
    indices_of_input_neuron = indices_of_input_neuron[indices_of_input_neuron != -1] + 1

    n_neurons, _ = weights.shape

    mat_data = {
        'neuron_location': brain_coordinates.astype(float),
        'neucube_connection': torch.where(torch.tensor(weights) != 0.0, 1, weights).numpy(),
        'neucube_weight': weights.numpy(),
        'neumid': neumids.astype(float),
        'input_mapping': input_mapping,
        'indices_of_input_neuron': torch.tensor(indices_of_input_neuron.reshape((len(indices_of_input_neuron), 1)), dtype=float),
        'is_extended': attrs['is_extended'],
        'small_world_radius': attrs['small_world_radius'],
        'number_of_neucube_neural': float(n_neurons),
        'number_of_input': float(len(feature_names)),
        'STDP_rate': attrs['STDP_rate'],
        'threshold_of_firing': attrs['threshold_of_firing'],
        'potential_leak_rate': attrs['potential_leak_rate'],
        'refactory_time': float(attrs['refactory_time']),
        'LDC_probability': float(attrs['LDC_probability']),
        'LDC_initial_weight': attrs['LDC_initial_weight'],
        'training_round': float(attrs['training_round']),
        'step': float(attrs['step']),
        'type': float(attrs['type']),
        'classifier_flag': float(attrs['classifier_flag']),
        'spike_transmission_amount': [], # ignored
        'neucube_output': [], # ignored
        'classifier': {
            'mod': attrs['mod'],
            'drift': attrs['drift'],
            'K': float(attrs['K']),
            'sigma': float(attrs['sigma']),
            'output_neurals_weight': [] # ignored
        }
    }

    savemat(file_name, { 'neucube': mat_data })
    print(f"MATLAB file '{file_name}' has been generated.")

# Example usage
#from neucube import Reservoir
#from neucube.encoder import Delta
#from neucube.training import STDP, NRDP
#from sklearn.model_selection import train_test_split

#filenameslist = ['sam'+str(idx)+'_eeg.csv' for idx in range(1,61)]
#dfs = []
#for filename in filenameslist:
#  dfs.append(pd.read_csv('../../example_data/wrist_movement_eeg/'+filename, header=None))
#data = pd.concat(dfs)
#data = data.values.reshape(60, 128, 14)

#labels = pd.read_csv('../../example_data/wrist_movement_eeg/tar_class_labels.csv', header=None)
#feature_names = pd.read_csv('../../example_data/wrist_movement_eeg/feature_names_eeg.txt', header=None).values
#coordinates = pd.read_csv('../../example_data/wrist_movement_eeg/brain_coordinates.csv', header=None).values
#mapping = pd.read_csv('../../example_data/wrist_movement_eeg/eeg_mapping.csv', header=None).values

#generate_datasetfile('dataset.mat', data, feature_names, labels, {
#    'type': 1,
#    'training_set_ratio': 0.5,
#    'method': 1,
#    'spike_threshold': 0.5,
#    'window_size': 5,
#    'filter_type': 1,
#    'sample_amount_for_training': 0,
#    'validation_time_length': 0,
#    'sample_amount_for_validation': 0
#})

#X = torch.tensor(data)
#encoder = Delta(threshold=0.8)
#X = encoder.encode_dataset(X)
#y = labels.values.flatten()

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

#res = Reservoir(inputs=14, coordinates=torch.tensor(coordinates, dtype=float),
#                mapping=torch.tensor(mapping, dtype=float))
#weights = torch.tensor(pd.read_csv('../../weights.csv', header=None).values)
#connections = torch.tensor(pd.read_csv('./connections.csv', header=None).values)
#res.manual_init(weights)
# TODO: Work out how to generate neumids automatically
#neumids = pd.read_csv('./neumids.csv', header=None).values

#stdp = STDP(a_pos=0.0001, a_neg=-0.001, t_constant=3)
#nrdp = NRDP()
#res.simulate(X_train, mem_thr=0.1, refractory_period=5, verbose=True)
#reservoir_connections = res.get_reservoir_connections()
#min_value = torch.min(output_weights)
#max_value = torch.max(output_weights)
#normalised_output_weights = ((output_weights - min_value) / (max_value - min_value)) * 2 - 1

#generate_neucubefile('neucube.mat', feature_names, coordinates, mapping, neumids, weights, {
#    'is_extended': True,
#    'small_world_radius': 2.50,
#    'STDP_rate': 0.01,
#    'threshold_of_firing': 0.50,
#    'potential_leak_rate': 0.002,
#    'refactory_time': 6,
#    'LDC_probability': 0,
#    'LDC_initial_weight': 0.05,
#    'training_round': 1,
#    'step': 4,
#    'type': 1,
#    'classifier_flag': 1,
#    'mod': 0.8,
#    'drift': 0.005,
#    'K': 3,
#    'sigma': 1
#})
