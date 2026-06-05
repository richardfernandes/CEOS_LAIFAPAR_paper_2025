import numpy as np

def weightedthielslopes(x, y, sample_weights):
    """
    Calculates the median of pairwise slopes, weighted by the Euclidean 
    combination of the user-specified weights for each sample.
    
    Parameters:
    - x: array-like, independent variable (length N)
    - y: array-like, dependent variable (length N)
    - sample_weights: array-like, user-specified weight for each sample (length N)
    
    Returns:
    - float: The weighted median slope.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(sample_weights, dtype=float)
    
    n = len(x)
    if n != len(y) or n != len(w):
        raise ValueError("x, y, and sample_weights must all have the same length.")
    if n < 2:
        raise ValueError("At least two data points are required.")
        
    slopes = []
    pairwise_weights = []
    
    # Compute pairwise slopes and Euclidean-combined weights
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            if dx == 0:
                continue  # Skip vertical lines to prevent division by zero
                
            slope = (y[j] - y[i]) / dx
            slopes.append(slope)
            
            # Calculate the Euclidean sum of the two sample weights
            pairwise_w = np.sqrt(w[i]**2 + w[j]**2)
            pairwise_weights.append(pairwise_w)
            
    slopes = np.array(slopes)
    pairwise_weights = np.array(pairwise_weights)
    
    num_slopes = len(slopes)
    if num_slopes == 0:
        raise ValueError("No valid pairwise slopes found (all x-values are identical).")
        
    # Sort slopes and weights together
    sort_idx = np.argsort(slopes)
    slopes = slopes[sort_idx]
    pairwise_weights = pairwise_weights[sort_idx]
    
    # Calculate the weighted median
    cum_weights = np.cumsum(pairwise_weights)

    
    # Find where the cumulative weight hits or crosses the 50% threshold, , 0.5-0.675/2, 0.5+0.765/2
    slope = slopes[np.maximum(0,np.minimum(num_slopes-1,np.searchsorted(cum_weights, cum_weights[-1] *0.5)))]
    upper_ci = slopes[np.maximum(0,np.minimum(num_slopes-1,np.searchsorted(cum_weights, cum_weights[-1] *(0.5-0.675/2))))]
    lower_ci= slopes[np.maximum(0,np.minimum(num_slopes-1,np.searchsorted(cum_weights, cum_weights[-1] *(0.5+0.675/2))))]
    
    return {'slope':slope,'upper_ci':upper_ci,'lower_ci':lower_ci}