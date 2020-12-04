import numpy as np

def calculate_hidden_dims(first_dim, n_layers=3, latent_dim=10, net_shape='geom'):
  if net_shape == 'linear':
    dims = np.linspace(first_dim, latent_dim, n_layers + 1)
  else:
    dims = np.geomspace(first_dim, latent_dim, n_layers + 1)

  dims = np.round(dims[1:]).astype(int)
  return tuple(dims)

