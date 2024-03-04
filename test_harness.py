import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from tqdm import tqdm
from neucube import Reservoir
from neucube.encoder import Delta
from neucube.training import STDP, NRDP
from sklearn.model_selection import train_test_split

filenameslist = ['sam'+str(idx)+'_eeg.csv' for idx in range(1,61)]
dfs = []
for filename in filenameslist:
  dfs.append(pd.read_csv('./example_data/wrist_movement_eeg/'+filename, header=None))
data = pd.concat(dfs)
data = data.values.reshape(60, 128, 14)

labels = pd.read_csv('./example_data/wrist_movement_eeg/tar_class_labels.csv', header=None)
feature_names = pd.read_csv('./example_data/wrist_movement_eeg/feature_names_eeg.txt', header=None).values
brain_coordinates = pd.read_csv('./example_data/wrist_movement_eeg/brain_coordinates.csv', header=None).values
eeg_mapping = pd.read_csv('./example_data/wrist_movement_eeg/eeg_mapping.csv', header=None).values
brain_coordinates = torch.tensor(brain_coordinates, dtype=float)
eeg_mapping = torch.tensor(eeg_mapping, dtype=float)
num_neurons, _ = brain_coordinates.shape

# Parameters for 3D space
space_size = 10  # Size of the space in each dimension

X = torch.tensor(data)
encoder = Delta(threshold=0.5)
X = encoder.encode_dataset(X)
y = labels.values.flatten()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

res = Reservoir(inputs=14, coordinates=brain_coordinates, mapping=eeg_mapping, c_in=0.7, like_matlab=True)

stdp = STDP(a_pos=0.01, a_neg=-0.01, t_constant=3)
#nrdp = NRDP()
res.simulate(X_train, mem_thr=0.3, refractory_period=5, leak_rate=0.002, learning_rule=stdp, train=True, verbose=True)
weights = res.get_reservoir_connections()
connection_matrix = torch.where(weights != 0.0, 1.0, 0.0)

# Visualisation
# Scale survived weights between (0,1)
_weights = weights.clone().detach()
L = torch.isinf(_weights)
_weights[L] = 0
L = torch.abs(_weights) > 0.08

neucube_weight_scale = torch.abs(_weights)
m = torch.min(neucube_weight_scale[L])
neucube_weight_scale = neucube_weight_scale - m
neucube_weight_scale[~L] = 0
M = torch.max(_weights[L])
neucube_weight_scale = neucube_weight_scale / M
neucube_weight_scale[~L] = 0

# Discretize scaled weights into four ranks
L1 = L & (neucube_weight_scale < 0.25)
L2 = L & (neucube_weight_scale >= 0.25) & (neucube_weight_scale < 0.5)
L3 = L & (neucube_weight_scale >= 0.5) & (neucube_weight_scale < 0.75)
L4 = L & (neucube_weight_scale >= 0.75)
neucube_weight_scale[L1] = 1
neucube_weight_scale[L2] = 2
neucube_weight_scale[L3] = 2.5
neucube_weight_scale[L4] = 3

# Plot neurons and connections
fig = plt.figure(figsize=(15, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot neurons
for i in range(num_neurons):
    x, y, z = brain_coordinates[i]
    ax.scatter(y, x, z, color='k', alpha=0.5)  # Plot neurons as black dots with transparency
    #ax.scatter(y, x, z, s=8, c='#3258a8', zorder=0)
    # ax.text(y, x, z, str(i), color='k', fontsize=6)  # Uncomment to add neuron index label

mapping_np = eeg_mapping.numpy()
ax.scatter(mapping_np[:, 1], mapping_np[:, 0], mapping_np[:, 2], s=60, c='m', zorder=10)

# Plot connections
for i in tqdm(range(num_neurons)):
    for j in range(i + 1, num_neurons):
        if connection_matrix[i][j] == 1:
            weight = weights[i, j].item()
            x1, y1, z1 = brain_coordinates[i]
            x2, y2, z2 = brain_coordinates[j]
            if weight > 0:
                ax.plot([y1, y2], [x1, x2], [z1, z2], color='b', linewidth=neucube_weight_scale[i, j])
            elif weight < 0:
                ax.plot([y1, y2], [x1, x2], [z1, z2], color='r', linewidth=neucube_weight_scale[i, j])

# Set plot labels
ax.set_xlabel('Y')
ax.set_ylabel('X')
ax.set_zlabel('Z')
ax.invert_xaxis()
ax.grid(False)
plt.title('Neural Connections')
plt.show()