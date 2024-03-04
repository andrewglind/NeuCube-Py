import torch
from tqdm import tqdm

def small_world_connectivity(dist, c, l, like_matlab=False):
    """
    Calculates a small-world network connectivity matrix based on the given distance matrix.

    Args:
        dist (torch.Tensor): The distance matrix representing pairwise distances between nodes.
        c (float): Maximum connection probability
        l (float): Small world connection radius
        like_matlab (bool): Flag to make small-world connectivity behave more like the MATLAB implementation.

    Returns:
        torch.Tensor: Connectivity matrix.

    """

    # Normalize the distance matrix
    dist_norm = (dist - torch.min(dist, dim=1).values[:, None]) / (torch.max(dist, dim=1).values[:, None] - torch.min(dist, dim=1).values[:, None])

    # Calculate the connection probability matrix
    if like_matlab:
        conn_prob = c * torch.exp(-(dist / l) ** 2)
    else:
        conn_prob = c * torch.exp(-(dist_norm / l) ** 2)

    # Create the input connectivity matrix by selecting connections based on probability
    input_conn = torch.where(conn_prob < torch.rand_like(conn_prob), conn_prob, torch.zeros_like(conn_prob))

    return input_conn

def connectivity_matlab(inputs, coordinates, mapping, distance_threshold=25, verbose=True):
    """

    Args:
        inputs:
        coordinates:
        mapping:
        distance_threshold:
        verbose:

    Returns:

    """
    num_neurons, _ = coordinates.shape

    indices_of_input_neuron = torch.ones(inputs, dtype=torch.int) * -1
    for k in range(mapping.size(0)):
        coord = mapping[k]
        L = torch.all(coordinates == coord, dim=1)
        idx = torch.nonzero(L).squeeze()
        idx, _ = torch.sort(idx)
        indices_of_input_neuron[k] = idx[0]
    indices_of_input_neuron = indices_of_input_neuron[indices_of_input_neuron != -1] + 1

    LL = torch.zeros(num_neurons, dtype=torch.bool)
    LL[indices_of_input_neuron] = True

    aaa = num_neurons - inputs + 1

    connection_matrix = torch.ones(num_neurons, num_neurons)
    choice = torch.randint(0, 2, (num_neurons, num_neurons))

    distance_matrix = torch.cdist(coordinates, coordinates, p=2)

    L = distance_matrix == 0
    neudistance_inv = 1. / distance_matrix
    neudistance_inv[L] = 0

    max_value_for_neudist_inv = 0.1
    min_value_for_neudist_inv = 0
    neudistance_inv = (neudistance_inv - torch.min(neudistance_inv)) / (
            torch.max(neudistance_inv) - torch.min(neudistance_inv))
    range_val = (max_value_for_neudist_inv - min_value_for_neudist_inv)
    neudistance_inv = (neudistance_inv * range_val) + min_value_for_neudist_inv

    # 20 percent of the weight is a positive number, and 80 percent is negative
    weights = (torch.sign(torch.rand(num_neurons, num_neurons) - 0.2)
               * torch.rand(num_neurons, num_neurons) * neudistance_inv)

    for i in tqdm(range(num_neurons), disable=not verbose):
        for j in range(num_neurons):
            if distance_matrix[i][j].item() > distance_threshold or j >= aaa or LL[j]:
                connection_matrix[i, j] = 0.0
            elif connection_matrix[i][j].item() == 1 and connection_matrix[j][i].item() == 1:
                if choice[i][j].item() == 1:
                    connection_matrix[i, j] = 0.0
                else:
                    connection_matrix[j, i] = 0.0
            weights[i, j] = connection_matrix[i][j].item() * weights[i][j].item()
            if i >= aaa:
                weights[i, j] = 0.0
            elif LL[i]:
                weights[i, j] = 2 * abs(weights[i][j].item())

    return weights