import numpy as np
from scipy.stats import kendalltau

def compute_disagreement_score(attr1, attr2, threshold=15.0):
    """
    Computes the Disagreement Score (DS) between two explanation.
    Both attr1 and attr2 has to be of the same shape.

    Parameters:
    - attr1, attr2: Attributions.
    - threshold: Disagreement threshold in %.

    Returns:
    - ds: Disagreement Score (%)
    - disagreement_mask: Boolean array where disagreements occur
    - pass_threshold: Boolean, True if DS <= threshold
    """
    flat1 = attr1.flatten()
    flat2 = attr2.flatten()

    sign1 = np.sign(flat1)
    sign2 = np.sign(flat2)

    disagreement = sign1 != sign2
    ds = 100.0 * np.sum(disagreement) / disagreement.size

    return ds, disagreement, ds <= threshold

def compute_kendall_tau_positive(attr1, attr2):

    flat1 = attr1.flatten()
    flat2 = attr2.flatten()

    pos_mask = (flat1 > 0) & (flat2 > 0)
    pos1 = flat1[pos_mask]
    pos2 = flat2[pos_mask]

    if len(pos1) < 2:
        return np.nan, np.nan 
    return kendalltau(pos1, pos2)
