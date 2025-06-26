import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np

def gen_D_binary(S, M, key=None):
    """Generate a decoder matrix D where D[m,s] = P(m|s)"""
    if key is None:
        key = jax.random.PRNGKey(0)
    
    D = jnp.zeros((M, S))
    for col in range(S):
        subkey1, subkey2, key = jax.random.split(key, 3)
        # For each symbol, randomly assign probabilities to messages
        num_messages = jax.random.randint(subkey1, (), 1, M+1)
        selected_messages = jax.random.choice(subkey2, M, shape=(M,), replace=True)
        unique_messages = jnp.unique(selected_messages[:num_messages])
        D = D.at[unique_messages, col].set(1.0 / len(unique_messages))
    return D

def gen_optimal_encoder(D):
    """
    Generate the optimal encoder E given decoder D.
    D: (M, S) matrix where D[m,s] = P(m|s)
    Returns E: (S, M) matrix where E[s,m] = P(s|m)
    """
    M, S = D.shape
    
    # Compute marginal P(s) under uniform P(m) = 1/M
    P_m = 1.0 / M
    P_s = jnp.sum(D * P_m, axis=0)  # P(s) = sum_m P(m|s)P(m)
    
    # Compute encoder using Bayes rule
    # E(s|m) = D(m|s)P(s) / sum_s' D(m|s')P(s')
    # Vectorized computation
    numerator = D.T * P_s[:, jnp.newaxis]  # (S, M)
    denominator = jnp.sum(D * P_s[jnp.newaxis, :], axis=1)  # (M,)
    E = numerator / (denominator[jnp.newaxis, :] + 1e-10)
    
    return E

@jit
def calculate_mutual_information_jax(E_i, D_j, P_m=None):
    """
    JAX-compatible calculation of normalized mutual information.
    
    Parameters:
    -----------
    E_i : jnp.ndarray
        Encoder matrix for agent i, shape (|S|, |M|) where E_i[s,m] = P(s|m)
    D_j : jnp.ndarray  
        Decoder matrix for agent j, shape (|M|, |S|) where D_j[m',s] = P(m'|s)
    P_m : jnp.ndarray, optional
        Prior distribution over messages, shape (|M|,). If None, assumes uniform.
    
    Returns:
    --------
    I_ij_normalized : float
        Normalized mutual information I(M_i; M_j')/H(M), in range [0,1]
    """
    
    # Get dimensions
    num_symbols, num_messages = E_i.shape
    
    # Set uniform prior if not provided
    if P_m is None:
        P_m = jnp.ones(num_messages) / num_messages
    
    # Compute composite channel matrix C_ij
    C_ij = D_j @ E_i
    
    # Compute joint probability matrix
    P_joint = C_ij * P_m[jnp.newaxis, :]
    
    # Compute marginal P(m')
    P_m_prime = jnp.sum(P_joint, axis=1)
    
    # Compute mutual information using vectorized operations
    epsilon = 1e-10
    # Create outer product of marginals
    P_marginal_product = jnp.outer(P_m_prime, P_m)
    
    # Compute MI with numerical stability
    # Using xlogy for x * log(y) which handles x=0 case properly
    log_ratio = jnp.log(P_joint + epsilon) - jnp.log(P_marginal_product + epsilon)
    mutual_info = jnp.sum(P_joint * log_ratio)
    
    # Normalize by entropy of M
    H_M = jnp.log(num_messages)  # For uniform distribution
    I_ij_normalized = mutual_info / H_M
    
    return I_ij_normalized

def mutual_info_sum(D, E_arr, v=None):
    """
    Calculate weighted sum of mutual information for decoder D with multiple encoders.
    
    Parameters:
    -----------
    D : jnp.ndarray
        Decoder matrix, shape (|M|, |S|)
    E_arr : list of jnp.ndarray
        List of encoder matrices, each shape (|S|, |M|)
    v : jnp.ndarray, optional
        Weights for each encoder, default is uniform
    
    Returns:
    --------
    weighted_sum : float
        Weighted sum of mutual information values
    """
    n = len(E_arr)
    if v is None:
        v = jnp.ones(n) / n
    
    # Stack encoders for vectorized computation
    E_stack = jnp.stack(E_arr, axis=0)  # (n, |S|, |M|)
    
    # Vectorized mutual information calculation
    vmap_mi = vmap(lambda E: calculate_mutual_information_jax(E, D))
    mi_values = vmap_mi(E_stack)
    
    return jnp.sum(v * mi_values)

def info_grad(D, E_arr, v=None):
    """
    Compute gradient of mutual information sum with respect to decoder D.
    
    Parameters:
    -----------
    D : jnp.ndarray  
        Decoder matrix for agent j, shape (|M|, |S|) where D_j[m',s] = P(m'|s)
    E_arr : list[jnp.ndarray] of length n
        List of encoder matrices for agent i, shape (|S|, |M|) where E_i[s,m] = P(s|m)
    v : jnp.ndarray
        A n-vector of weights to prioritize the mutual information from different channels
        default = uniform weights
    
    Returns:
    --------
    grad_D: jnp.ndarray
        The gradient of the weighted mutual information sum with respect to D
    """
    # Create gradient function
    grad_fn = grad(mutual_info_sum, argnums=0)
    
    # Compute gradient
    grad_D = grad_fn(D, E_arr, v)
    
    return grad_D

def normalize_decoder(D):
    """
    Normalize decoder matrix so columns sum to 1.
    """
    return D / (jnp.sum(D, axis=0, keepdims=True) + 1e-10)

def update(D, E_arr, learning_rate=0.01, v=None):
    """
    Update decoder D to maximize weighted mutual information from encoders E_arr.
    
    Parameters:
    -----------
    D : jnp.ndarray
        Current decoder matrix
    E_arr : list[jnp.ndarray]
        List of encoder matrices to receive from
    learning_rate : float
        Step size for gradient ascent
    v : jnp.ndarray, optional
        Weights for each encoder
    
    Returns:
    --------
    D_new : jnp.ndarray
        Updated and normalized decoder matrix
    """
    # Compute gradient
    grad_D = info_grad(D, E_arr, v)
    
    # Gradient ascent step
    D_new = D + learning_rate * grad_D
    
    # Project back to probability simplex by normalization
    D_new = normalize_decoder(D_new)
    
    return D_new

# Additional utilities for the full system

def initialize_agents(N, S, M, key=None):
    """
    Initialize N agents with random decoders and optimal encoders.
    
    Parameters:
    -----------
    N : int
        Number of agents
    S : int
        Size of shared symbol space
    M : int
        Size of message space
    key : jax.random.PRNGKey
        Random key for initialization
    
    Returns:
    --------
    decoders : list of jnp.ndarray
        List of decoder matrices
    encoders : list of jnp.ndarray
        List of optimal encoder matrices
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    decoders = []
    encoders = []
    
    for i in range(N):
        key, subkey = jax.random.split(key)
        # Initialize random decoder
        D = jax.random.uniform(subkey, (M, S))
        D = normalize_decoder(D)
        decoders.append(D)
        
        # Compute optimal encoder
        E = gen_optimal_encoder(D)
        encoders.append(E)
    
    return decoders, encoders

def simulate_convergence(N, S, M, num_iterations=100, learning_rate=0.01, key=None):
    """
    Simulate the convergence of the multi-agent language system.
    
    Parameters:
    -----------
    N : int
        Number of agents
    S : int
        Size of shared symbol space
    M : int
        Size of message space
    num_iterations : int
        Number of update iterations
    learning_rate : float
        Learning rate for decoder updates
    key : jax.random.PRNGKey
        Random key for initialization
    
    Returns:
    --------
    history : dict
        Dictionary containing evolution history
    """
    # Initialize agents
    decoders, encoders = initialize_agents(N, S, M, key)
    
    # Track history
    history = {
        'mutual_info': [],
        'decoders': [decoders],
        'encoders': [encoders]
    }
    
    for iteration in range(num_iterations):
        # Update each agent's decoder
        new_decoders = []
        new_encoders = []
        
        for i in range(N):
            # Get encoders from all other agents
            other_encoders = [encoders[j] for j in range(N) if j != i]
            
            # Update decoder
            D_new = update(decoders[i], other_encoders, learning_rate)
            new_decoders.append(D_new)
            
            # Compute new optimal encoder
            E_new = gen_optimal_encoder(D_new)
            new_encoders.append(E_new)
        
        decoders = new_decoders
        encoders = new_encoders
        
        # Record mutual information matrix
        mi_matrix = jnp.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    mi_matrix = mi_matrix.at[i, j].set(
                        calculate_mutual_information_jax(encoders[j], decoders[i])
                    )
        
        history['mutual_info'].append(mi_matrix)
        history['decoders'].append(decoders)
        history['encoders'].append(encoders)
    
    return history

# Example usage and testing
if __name__ == "__main__":
    # Set parameters
    N = 3  # Number of agents
    S = 10  # Symbol space size
    M = 20  # Message space size
    
    # Initialize
    key = jax.random.PRNGKey(42)
    decoders, encoders = initialize_agents(N, S, M, key)
    
    # Test gradient computation
    D_test = decoders[0]
    E_test = encoders[1:]
    
    # Compute gradient
    grad_D = info_grad(D_test, E_test)
    print(f"Gradient shape: {grad_D.shape}")
    print(f"Gradient norm: {jnp.linalg.norm(grad_D):.4f}")
    
    # Test update
    D_new = update(D_test, E_test, learning_rate=0.1)
    print(f"Decoder still normalized: {jnp.allclose(jnp.sum(D_new, axis=0), 1.0)}")
    
    # Run short simulation
    history = simulate_convergence(N, S, M, num_iterations=10, learning_rate=0.05, key=key)
    
    print(f"\nInitial average mutual information: {jnp.mean(history['mutual_info'][0]):.4f}")
    print(f"Final average mutual information: {jnp.mean(history['mutual_info'][-1]):.4f}")