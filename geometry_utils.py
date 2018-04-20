import numpy as np
import random
import math

from debug_log import debug_log

def rotation_matrix(T):
    cos_T = math.cos(T)
    sin_T = math.sin(T)
    return np.array([[ cos_T, -sin_T ], [ sin_T, cos_T ]])


def reduce_line_segment(P_orig, alpha=10, beta=10, threshold=10):
    """
    Given a set of points P, attempt to find a line segment
    such that there exist a set of points P_line \subset P
    where all points are max distance alpha from the line
    segment and within max distance alpha of each other

    Crappy version of RANSAC
    """
    # Only use x,y coords for now
    P = P_orig[:,:2]

    # Pick a random point and pivot around it
    p_guess = random.choice(P)
    rotations = 120
    R = rotation_matrix(2 * math.pi / rotations)

    P_rot = P - p_guess
    for _ in range(rotations):
        # Determine how many points are within alpha of the line segment
        L = np.abs(P_rot[:,0]) < alpha
        # Expand outwards from the center and add points to
        # the line segment as long as the points are within
        # beta of each other
        if L.sum() >= threshold:
            # Indices of the points in the line segment
            D_indices = []
            indices = np.argsort(np.abs(P_rot[:,1]))
            v_min, v_max = 0, 0
            i_min, i_max = 0, 0
            for i in indices:
                # Only track indices in the mask
                if not L[i]:
                    continue
                v = P_rot[i,1]
                if 0 > v > v_min - beta:
                    D_indices.append(i)
                    v_min, i_min = v, i
                if 0 < v < v_max + beta:
                    D_indices.append(i)
                    v_max, i_max = v, i

            if len(D_indices) >= threshold:
                #debug_log(f'Found line segment of size {len(D_indices)} -> 2')
                # Get the two bounding points of the line segment
                R1, R2 = P_orig[i_min], P_orig[i_max]
                # Remove points that were associated with the line segment
                remainder_mask = np.ones((len(P),), dtype=np.bool)
                remainder_mask[D_indices] = False
                P_filtered = P_orig[remainder_mask]
                return (P_filtered, R1, R2)
            else:
                #debug_log(f'Index reduced below threshold from {L.sum()} -> {len(D_indices)}')
                pass

        P_rot = P_rot @ R

    # If we didn't find enough, return nothing
    return None


def fit_to_point_and_reduce(P_orig, L1, epsilon=10, iterations=17):
    """
    Attempt to fit points against random line segments that
    terminate at L1.

    P.shape should be (None, 2)
    epsilon is max distance to line to satisfy linearity
    iterations is the number of times to run RANSAC

    Returns the point that is best fitting and the inliers around
    the linear model fit
    """
    # Center space around L1 for ease of use
    P = P_orig - L1

    # Choose N random points
    indices = np.arange(P.shape[0])
    np.random.shuffle(indices)

    best_fit, best_fit_arg, best_inliers = 0, None, None
    for ind in P[:iterations]:
        # Define line as ax + by = 0
        L2_guess = P[ind]
        AB = np.array([1, -L2_guess[0] / L2_guess[1]])
        # Draw line between L1 and L2_guess, get the number of points
        # that fit that line within the given epsilon
        # D = (ax + by) ** 2 / (a**2 + b**2)
        D_P = np.sum(P * AB)**2 / (np.sum(AB ** 2))
        inliers = D_P < epsilon
        fit_points = inliers.sum()
        if fit_points > best_fit:
            best_fit = fit_points
            best_fit_arg = ind
            best_inliers = inliers

    # Partition the point space between inliers and outliers and
    # return a line segment from L1 to the furthest outlier
    return P_orig[best_fit_arg], best_inliers


def estimate_corner_points(P, radius=10):
    """
    TODO
    """

