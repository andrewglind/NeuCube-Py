import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import kneighbors_graph

def visualize_reservoir_connections(neuron_coords, connection_weights, threshold=0.08, verbose=False):
    """
    Visualize connections in the reservoir
    """
    #min_val = connection_weights.min()
    #max_val = connection_weights.max()
    #normalised_connection_weights = torch.where(connection_weights == 0.0, 0.0,
    #                                            (connection_weights - min_val) / (max_val - min_val))
    #connectivity_matrix = ((connection_weights >= -0.5) &
    #                       (connection_weights <= 0.5)).int()
    connectivity_matrix = ((connection_weights >= -0.5) & (connection_weights <= 0.5)).int()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.scatter(neuron_coords[:, 0], neuron_coords[:, 1], neuron_coords[:, 2], color=(0, 0, 0, 0.20))

    for i in tqdm(range(len(neuron_coords)), disable=not verbose):
        for j in range(i + 1, len(neuron_coords)):
            if connectivity_matrix[i, j]:
                weight = connection_weights[i, j].item()
                if -threshold < weight < threshold:
                    intensity = (1-abs(weight))
                    color = (0, 0, intensity) if weight > 0.0 else (intensity, 0, 0)
                    ax.plot([neuron_coords[i, 0], neuron_coords[j, 0]],
                            [neuron_coords[i, 1], neuron_coords[j, 1]],
                            [neuron_coords[i, 2], neuron_coords[j, 2]], color=color)

    #for i in tqdm(range(len(neuron_coords)), disable = not verbose):
    #    for j in range(i+1, len(neuron_coords)):
    #        if connection_weights[i, j]:
    #            #weight = normalised_connection_weights[i, j].item()
    #            weight = connection_weights[i, j].item() + 0.5
    #            if threshold < weight < (1-threshold):
    #                color="k"
    #                if weight > threshold:
    #                    color = (abs(weight), 0, 0)
    #                if weight < (1-threshold):
    #                    color = (0, 0, weight)
    #                ax.plot([neuron_coords[i, 0], neuron_coords[j, 0]],
    #                        [neuron_coords[i, 1], neuron_coords[j, 1]],
    #                        [neuron_coords[i, 2], neuron_coords[j, 2]], color=color, linewidth=0.1)
                #weight = connection_weights[i, j].item()
                #print("connected: %f" % weight)
                #if (0.0 < weight < threshold) or (weight > (1 - threshold)):
                #if threshold < weight < (1 - threshold):
                    #color = "k"
                    #if threshold < weight:
                    #    color = ((1 - weight), 0, 0)
                    #if weight < (1 - threshold):
                    #    color = (0, 0, weight)
                    #ax.plot([neuron_coords[i, 0], neuron_coords[j, 0]],
                    #        [neuron_coords[i, 1], neuron_coords[j, 1]],
                    #        [neuron_coords[i, 2], neuron_coords[j, 2]], color=color)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Reservoir connections')

    #ax.view_init(elev=-90, azim=90) # plan view
    ax.view_init(elev=30, azim=-135) # emulate NeuCube viewport

    #plt.ion()
    plt.show()

# Example usage
import pandas as pd

#neuron_coords = pd.read_csv('./example_data/fastball/Talairach_coordinate.csv', header=None).values
#neuron_coords = torch.tensor(neuron_coords, dtype=float)
#shape = neuron_coords.shape[0]
#connection_weights = torch.randint(0, 2, (shape, shape))-torch.rand(shape)
#visualize_reservoir_connections(neuron_coords, connection_weights, threshold=0.04, verbose=True)

def visualize_network(activation_levels, neuron_coords, connection_weights, threshold=0.08):

    activation_levels_normalized = (activation_levels - activation_levels.min()) / (
                activation_levels.max() - activation_levels.min())
    x = torch.where(torch.abs(connection_weights) > 0.5, abs(connection_weights), 0)
    connectivity_matrix = kneighbors_graph(x.numpy(), 1, mode='connectivity', include_self=False)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Plot neurons with colors based on activation levels
    ax.scatter(neuron_coords[:, 0], neuron_coords[:, 1], neuron_coords[:, 2], c=activation_levels_normalized, cmap='coolwarm', s=2)

    # Plot connections
    for i in tqdm(range(len(neuron_coords))):
        for j in range(i + 1, len(neuron_coords)):
            if connectivity_matrix[i, j]:
                weight = connection_weights[i, j].item()
                if -threshold < weight < threshold:
                    color = (0, 0, weight) if weight > 0 else (abs(weight), 0, 0)
                    ax.plot([neuron_coords[i, 0], neuron_coords[j, 0]],
                            [neuron_coords[i, 1], neuron_coords[j, 1]],
                            [neuron_coords[i, 2], neuron_coords[j, 2]], color=color, linewidth=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Reservoir connections')

    #ax.view_init(elev=-90, azim=90) # plan view
    ax.view_init(elev=30, azim=-135) # emulate NeuCube viewport

    plt.show()

# Example usage
#neuron_coords = pd.read_csv('./example_data/fastball/Talairach_coordinate.csv', header=None).values
#neuron_coords = torch.tensor(neuron_coords, dtype=float)
#shape = neuron_coords.shape[0]
#activation_levels = torch.rand(shape)
#connection_weights = torch.randint(0, 2, (shape, shape))-torch.rand(shape)

#visualize_network(activation_levels, neuron_coords, connection_weights)