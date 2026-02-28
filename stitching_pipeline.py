
class StitchingParameters:
    """All configurable parameters for the image stitching pipeline"""
    
    # Feature matching parameters
    MIN_MATCHES = 10  # Minimum number of matches required to consider an edge valid
    MIN_INLIER_RATIO = 0.2  # Minimum ratio of inliers to total matches (0.2 = 20%)
    SIFT_RATIO_THRESHOLD = 0.7  # Lowe's ratio test threshold for SIFT matching
    
    # RANSAC parameters
    RANSAC_REPROJ_THRESHOLD = 5.0  # Maximum reprojection error in pixels for RANSAC
    RANSAC_CONFIDENCE = 0.99  # Confidence level for RANSAC
    RANSAC_MAX_ITERS = 2000  # Maximum iterations for RANSAC
    
    # Levenberg-Marquardt optimization
    LM_MAX_NFEV = 1000  # Maximum number of function evaluations
    LM_FTOL = 1e-8  # Function tolerance for convergence
    LM_XTOL = 1e-8  # Parameter tolerance for convergence
    
    # Covariance computation
    COV_REGULARIZATION = 1e-12  # Regularization factor for covariance matrix inversion
    
    # GTSAM optimization
    GTSAM_PRIOR_SIGMA = 0.001  # Sigma for prior factor in GTSAM
    GTSAM_MAX_ITERATIONS = 100  # Maximum iterations for GTSAM optimizer
    GTSAM_REL_ERROR_TOL = 1e-8  # Relative error tolerance for GTSAM
    GTSAM_ABS_ERROR_TOL = 1e-8  # Absolute error tolerance for GTSAM
    
    # Visualization parameters
    GRAPH_NODE_SIZE = 2000  # Size of nodes in graph visualization
    MAX_DISPLAY_COLS = 3  # Maximum columns in grid displays
    
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from itertools import combinations
import networkx as nx
import gtsam
from gtsam import symbol

# ============================================================================
# CONFIGURABLE PARAMETERS
# ============================================================================

# Create global parameters instance
params = StitchingParameters()

# ============================================================================
# FEATURE DETECTION AND MATCHING
# ============================================================================

def find_and_match_features(img1, img2, ratio_threshold=None):
    """Find SIFT features and match them between two images"""
    if ratio_threshold is None:
        ratio_threshold = params.SIFT_RATIO_THRESHOLD
        
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        return [], [], []
    
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    
    good_matches = [m for m, n in matches 
                   if len([m, n]) == 2 and m.distance < ratio_threshold * n.distance]
    
    return kp1, kp2, good_matches

def homography_residuals(params_vec, src_pts, dst_pts):
    """Compute residuals for homography optimization"""
    H = np.array([
        [params_vec[0], params_vec[1], params_vec[2]],
        [params_vec[3], params_vec[4], params_vec[5]],
        [params_vec[6], params_vec[7], 1.0]
    ])
    
    src_hom = np.column_stack([src_pts, np.ones(len(src_pts))])
    transformed = (H @ src_hom.T).T
    transformed_cart = transformed[:, :2] / transformed[:, [2]]
    return (transformed_cart - dst_pts).flatten()

def compute_covariance_matrix(lm_result, src_pts, dst_pts):
    """Compute covariance matrix from Levenberg-Marquardt result"""
    try:
        # Get the Jacobian matrix from the optimization result
        jacobian = lm_result.jac
        
        # Compute residuals at the solution
        residuals = homography_residuals(lm_result.x, src_pts, dst_pts)
        
        # Estimate noise variance (sigma^2)
        n_points = len(src_pts)
        n_params = len(lm_result.x)  # 8 parameters for homography
        degrees_of_freedom = 2 * n_points - n_params  # 2 residuals per point
        
        if degrees_of_freedom > 0:
            sigma_squared = np.sum(residuals**2) / degrees_of_freedom
        else:
            sigma_squared = 1.0  # fallback
        
        # Compute covariance matrix: C = σ² * (J^T * J)^(-1)
        JTJ = jacobian.T @ jacobian
        
        # Add regularization to avoid singular matrix
        regularization = params.COV_REGULARIZATION * np.eye(JTJ.shape[0])
        JTJ_reg = JTJ + regularization
        
        try:
            covariance = sigma_squared * np.linalg.inv(JTJ_reg)
        except np.linalg.LinAlgError:
            # Use pseudoinverse if matrix is still singular
            covariance = sigma_squared * np.linalg.pinv(JTJ_reg)
        
        return covariance
        
    except Exception as e:
        print(f"Warning: Could not compute covariance matrix: {e}")
        # Return identity matrix as fallback
        return np.eye(8)

def print_covariance_analysis(covariance, title, homography=None):
    """Print detailed covariance analysis"""
    print(f"\n{title} - Covariance Analysis:")
    print("=" * 50)
    
    # Parameter labels
    param_labels = ['h00', 'h01', 'h02', 'h10', 'h11', 'h12', 'h20', 'h21']
    
    # Standard deviations (uncertainties)
    std_devs = np.sqrt(np.diag(covariance))
    print("\nParameter Uncertainties (Standard Deviations):")
    for i, (label, std) in enumerate(zip(param_labels, std_devs)):
        if homography is not None:
            h_val = homography.flatten()[i] if i < 8 else 1.0
            rel_uncertainty = (std / abs(h_val)) * 100 if abs(h_val) > 1e-10 else float('inf')
            print(f"  {label}: σ = {std:.6f} (rel: {rel_uncertainty:.1f}%)")
        else:
            print(f"  {label}: σ = {std:.6f}")
    
    # Condition number
    eigenvals = np.linalg.eigvals(covariance)
    condition_number = np.max(eigenvals) / np.min(eigenvals) if np.min(eigenvals) > 1e-15 else float('inf')
    print(f"\nCovariance Matrix Properties:")
    print(f"  Condition Number: {condition_number:.2e}")
    print(f"  Determinant: {np.linalg.det(covariance):.2e}")
    
    # Correlation matrix
    correlation = covariance / np.outer(std_devs, std_devs)
    print(f"\nStrong Correlations (|r| > 0.7):")
    for i in range(len(param_labels)):
        for j in range(i+1, len(param_labels)):
            if abs(correlation[i, j]) > 0.7:
                print(f"  {param_labels[i]} ↔ {param_labels[j]}: r = {correlation[i, j]:.3f}")
    
    # Overall uncertainty measures
    trace_cov = np.trace(covariance)
    frobenius_norm = np.linalg.norm(covariance, 'fro')
    print(f"\nOverall Uncertainty Measures:")
    print(f"  Trace (sum of variances): {trace_cov:.6f}")
    print(f"  Frobenius norm: {frobenius_norm:.6f}")

def optimize_homography_lm(src_pts, dst_pts, initial_h=None):
    """Optimize homography using Levenberg-Marquardt and compute covariance"""
    # Check if we have enough points for LM optimization (need at least 4 points for 8 parameters)
    if len(src_pts) < 4:
        return None, None, None
        
    if initial_h is None:
        initial_h = cv2.findHomography(
            src_pts.reshape(-1, 1, 2), dst_pts.reshape(-1, 1, 2), cv2.LMEDS)[0]
    
    if initial_h is None:
        return None, None, None
    
    # Use 'trf' method instead of 'lm' for better handling of edge cases
    result = least_squares(
        homography_residuals, initial_h.flatten()[:8], 
        args=(src_pts, dst_pts), method='trf', 
        max_nfev=params.LM_MAX_NFEV, 
        ftol=params.LM_FTOL, 
        xtol=params.LM_XTOL)
    
    if result.success:
        optimized_h = np.array([
            [result.x[0], result.x[1], result.x[2]],
            [result.x[3], result.x[4], result.x[5]],
            [result.x[6], result.x[7], 1.0]
        ])
        
        # Compute covariance matrix
        covariance = compute_covariance_matrix(result, src_pts, dst_pts)
        return optimized_h, result, covariance
    return initial_h, result, None

def estimate_homography_ransac_lm(kp1, kp2, matches, min_matches=None, img1_shape=None, img2_shape=None):
    """Estimate homography using RANSAC + Levenberg-Marquardt with covariance"""
    if min_matches is None:
        min_matches = params.MIN_MATCHES
        
    if len(matches) < min_matches:
        return None, [], None, None, None
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    H_ransac, mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, 
        ransacReprojThreshold=params.RANSAC_REPROJ_THRESHOLD, 
        confidence=params.RANSAC_CONFIDENCE, 
        maxIters=params.RANSAC_MAX_ITERS)
    
    if mask is None or H_ransac is None:
        return None, [], None, None, None
    
    # Validate initial RANSAC homography
    is_valid, reason = validate_homography_matrix(H_ransac)
    if not is_valid:
        print(f"Warning: RANSAC homography failed validation: {reason}")
        return None, [], None, None, None
    
    inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]
    inlier_src = np.array([kp1[m.queryIdx].pt for m in inlier_matches])
    inlier_dst = np.array([kp2[m.trainIdx].pt for m in inlier_matches])
    
    # Check if we have enough inliers for optimization
    if len(inlier_src) < 4:
        print(f"Warning: Not enough inliers for optimization: {len(inlier_src)}")
        return None, [], None, None, None
    
    H_lm, lm_result, covariance = optimize_homography_lm(inlier_src, inlier_dst, H_ransac)
    
    # Validate optimized homography
    if H_lm is not None:
        is_valid, reason = validate_homography_matrix(H_lm)
        if not is_valid:
            print(f"Warning: Optimized homography failed validation: {reason}")
            # Fall back to RANSAC result if optimization failed
            H_lm = H_ransac
            covariance = None
        else:
            # Test homography on image corners to detect exploding transformations
            if img1_shape is not None:
                test_valid, test_reason = test_homography_on_corners(H_lm, img1_shape)
                if not test_valid:
                    print(f"Warning: Homography causes exploding transformation: {test_reason}")
                    H_lm = H_ransac
                    covariance = None
    
    return H_lm, inlier_matches, lm_result, mask, covariance

def validate_homography_matrix(H, min_det=1e-6, max_det=1e6, max_cond=1e12):
    """Validate homography matrix for quality and numerical stability"""
    if H is None:
        return False, "Matrix is None"
    
    # Check determinant
    det = np.linalg.det(H)
    if abs(det) < min_det:
        return False, f"Determinant too small: {det:.2e}"
    if abs(det) > max_det:
        return False, f"Determinant too large: {det:.2e}"
    
    # Check condition number
    try:
        cond = np.linalg.cond(H)
        if cond > max_cond:
            return False, f"Condition number too large: {cond:.2e}"
    except:
        return False, "Cannot compute condition number"
    
    # Check for reasonable scale (h33 should be close to 1)
    if abs(H[2, 2]) < 1e-6:
        return False, f"h33 too small: {H[2, 2]:.2e}"
    
    # Check for reasonable transformation (no extreme scaling)
    scale_x = np.sqrt(H[0, 0]**2 + H[0, 1]**2)
    scale_y = np.sqrt(H[1, 0]**2 + H[1, 1]**2)
    
    if scale_x < 0.1 or scale_x > 10:
        return False, f"X scale unreasonable: {scale_x:.2f}"
    if scale_y < 0.1 or scale_y > 10:
        return False, f"Y scale unreasonable: {scale_y:.2f}"
    
    # Check for reasonable translation (no extreme shifts)
    tx = H[0, 2]
    ty = H[1, 2]
    max_translation = 10000  # pixels
    
    if abs(tx) > max_translation:
        return False, f"X translation too large: {tx:.1f}"
    if abs(ty) > max_translation:
        return False, f"Y translation too large: {ty:.1f}"
    
    # Check for reasonable perspective distortion
    perspective_x = abs(H[2, 0])
    perspective_y = abs(H[2, 1])
    max_perspective = 1e-3  # Very small perspective distortion allowed
    
    if perspective_x > max_perspective:
        return False, f"X perspective too large: {perspective_x:.2e}"
    if perspective_y > max_perspective:
        return False, f"Y perspective too large: {perspective_y:.2e}"
    
    # Check for reasonable rotation (no extreme rotations)
    # Extract rotation angle from the 2x2 upper-left submatrix
    try:
        R = H[:2, :2] / np.sqrt(scale_x * scale_y)  # Normalize to get rotation
        # Check if it's close to a rotation matrix
        should_be_identity = R @ R.T
        identity_error = np.linalg.norm(should_be_identity - np.eye(2))
        if identity_error > 0.1:  # Allow some numerical error
            return False, f"Not a proper rotation matrix: error={identity_error:.3f}"
    except:
        return False, "Cannot extract rotation from homography"
    
    return True, "Valid"

def test_homography_on_corners(H, img_shape):
    """Test homography transformation on image corners to detect exploding transformations"""
    if H is None:
        return False, "No homography matrix"
    
    h, w = img_shape[:2]
    corners = np.array([
        [0, 0],      # Top-left
        [w, 0],      # Top-right
        [w, h],      # Bottom-right
        [0, h]       # Bottom-left
    ], dtype=np.float32)
    
    try:
        # Transform corners
        corners_hom = np.column_stack([corners, np.ones(4)])
        transformed = (H @ corners_hom.T).T
        transformed_cart = transformed[:, :2] / transformed[:, [2]]
        
        # Check for reasonable transformed positions
        # Transformed corners should be within reasonable bounds
        max_reasonable_distance = 50000  # pixels
        
        for i, (x, y) in enumerate(transformed_cart):
            if abs(x) > max_reasonable_distance or abs(y) > max_reasonable_distance:
                return False, f"Corner {i} explodes to ({x:.1f}, {y:.1f})"
        
        # Check for reasonable area change
        # Compute area of original and transformed quadrilaterals
        def polygon_area(points):
            x, y = points[:, 0], points[:, 1]
            return 0.5 * abs(sum(x[i] * y[(i+1) % len(x)] - x[(i+1) % len(x)] * y[i] for i in range(len(x))))
        
        original_area = w * h
        transformed_area = polygon_area(transformed_cart)
        area_ratio = transformed_area / original_area
        
        if area_ratio < 0.01 or area_ratio > 100:
            return False, f"Area ratio too extreme: {area_ratio:.2f}"
        
        return True, "Valid transformation"
        
    except Exception as e:
        return False, f"Transformation failed: {str(e)}"

def compute_homography_quality_score(H, src_pts, dst_pts):
    """Compute a quality score for the homography matrix"""
    if H is None or src_pts is None or dst_pts is None:
        return 0.0
    
    try:
        # Transform source points
        src_hom = np.column_stack([src_pts, np.ones(len(src_pts))])
        transformed = (H @ src_hom.T).T
        transformed_cart = transformed[:, :2] / transformed[:, [2]]
        
        # Compute reprojection error
        errors = np.linalg.norm(transformed_cart - dst_pts, axis=1)
        mean_error = np.mean(errors)
        
        # Quality score: lower error = higher score
        # Also consider the number of points
        n_points = len(src_pts)
        quality = n_points / (1.0 + mean_error)
        
        return quality
    except:
        return 0.0

def compute_covariance_weight(covariance):
    """
    Compute weight from covariance matrix for graph edges.
    Lower uncertainty (smaller covariance values) should result in higher weights.
    """
    if covariance is None:
        return float('inf')  # Very low weight for edges without covariance
    
    # Use trace of covariance as uncertainty measure
    uncertainty = np.trace(covariance)
    
    # Convert uncertainty to weight (higher uncertainty = lower weight)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-12
    weight = 1.0 / (uncertainty + epsilon)
    
    return weight

def show_full_connection_graph(images, pairwise_data):
    """Display the full connection graph with all valid edges (including loops)"""
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Create full graph with all valid connections
    full_graph = nx.Graph()
    for i, (img_num, _, _) in enumerate(images):
        full_graph.add_node(i, image_num=img_num)
    
    # Add all edges from pairwise_data
    for (i, j), data in pairwise_data.items():
        if data.get('covariance') is not None:
            weight = compute_covariance_weight(data['covariance'])
            uncertainty = np.trace(data['covariance'])
        else:
            weight = 0
            uncertainty = float('inf')
        
        full_graph.add_edge(i, j, weight=weight, error=data['error'], 
                           homography=data['homography'], 
                           inliers=data['inliers'], 
                           uncertainty=uncertainty)
    
    # Layout
    if len(full_graph.nodes()) <= 6:
        pos = nx.circular_layout(full_graph, scale=3)
    else:
        pos = nx.spring_layout(full_graph, seed=42, k=4, iterations=100, scale=3)
    
    # Node labels
    node_labels = {i: f"Image {images[i][0]}" for i in range(len(images))}
    
    # Calculate edge properties for visualization
    edge_weights = []
    edge_widths = []
    edge_labels = {}
    
    for (i, j) in full_graph.edges():
        pair_key = tuple(sorted([i, j]))
        if pair_key in pairwise_data:
            data = pairwise_data[pair_key]
            if data.get('covariance') is not None:
                weight = compute_covariance_weight(data['covariance'])
            else:
                weight = 0
            
            matches = data['inliers']
            error = data['error']
            
            edge_weights.append(weight if np.isfinite(weight) else 0)
            edge_labels[(i, j)] = f"M: {matches}\nE: {error:.1f}px"
            
            # Vary edge width based on number of matches
            edge_widths.append(max(1, min(6, matches / 25)))
    
    # Color mapping for edges
    if edge_weights and any(w > 0 and np.isfinite(w) for w in edge_weights):
        finite_weights = [w for w in edge_weights if np.isfinite(w) and w > 0]
        if finite_weights and len(finite_weights) > 1:
            min_weight, max_weight = min(finite_weights), max(finite_weights)
            if max_weight > min_weight:
                norm_weights = [(w - min_weight) / (max_weight - min_weight) if np.isfinite(w) and w > 0 else 0 
                               for w in edge_weights]
            else:
                norm_weights = [1.0 if np.isfinite(w) and w > 0 else 0 for w in edge_weights]
        else:
            norm_weights = [1.0 if np.isfinite(w) and w > 0 else 0 for w in edge_weights]
        
        cmap = plt.cm.viridis
        edge_colors = [cmap(w) for w in norm_weights]
    else:
        edge_colors = ['gray'] * len(full_graph.edges())
    
    # Draw edges
    nx.draw_networkx_edges(full_graph, pos, 
                          edge_color=edge_colors,
                          width=edge_widths,
                          alpha=0.6,
                          style='solid')
    
    # Draw nodes
    node_sizes = [params.GRAPH_NODE_SIZE] * len(full_graph.nodes())
    
    # Shadow
    shadow_pos = {node: (x + 0.05, y - 0.05) for node, (x, y) in pos.items()}
    nx.draw_networkx_nodes(full_graph, shadow_pos, node_color='black', 
                          node_size=node_sizes, alpha=0.3)
    
    # Main nodes
    nx.draw_networkx_nodes(full_graph, pos, node_color='lightcoral', 
                          node_size=node_sizes, alpha=0.9,
                          edgecolors='darkred', linewidths=2)
    
    # Node labels
    for node, (x, y) in pos.items():
        ax.text(x, y, node_labels[node], ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
    
    # Edge labels
    for (i, j), label in edge_labels.items():
        x1, y1 = pos[i]
        x2, y2 = pos[j]
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        
        ax.text(mid_x, mid_y, label, ha='center', va='center',
                fontsize=8, fontweight='normal',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='gray', alpha=0.8))
    
    # Title and statistics
    ax.set_title('Full Connection Graph (All Valid Pairs with Homographies)\nBefore MST Selection', 
                fontsize=16, fontweight='bold', pad=20,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
    
    # Statistics box
    stats_text = f"Full Graph Statistics:\n"
    stats_text += f"• Nodes: {full_graph.number_of_nodes()}\n"
    stats_text += f"• Edges: {full_graph.number_of_edges()}\n"
    stats_text += f"• Possible Edges: {len(images)*(len(images)-1)//2}\n"
    stats_text += f"• Connection Rate: {100*full_graph.number_of_edges()/(len(images)*(len(images)-1)//2):.1f}%"
    stats_text += f"\n• Min Matches Required: {params.MIN_MATCHES}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           ha='left', va='top', fontsize=10,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
                    alpha=0.9, edgecolor='blue'))
    
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# ============= GTSAM BUNDLE ADJUSTMENT SECTION =============

def create_homography_factor(key_i, key_j, measured_H, noise_model):
    """Create a custom homography factor for GTSAM"""
    
    class HomographyFactor(gtsam.CustomFactor):
        def __init__(self, key_i, key_j, measured_H, noise_model):
            super().__init__(noise_model, [key_i, key_j])
            self.measured_H = measured_H
            
        def error_func(self, v, H):
            # v contains the poses [H_i, H_j]
            H_i = v.atPose3(self.keys()[0])  # Pose from image i to reference
            H_j = v.atPose3(self.keys()[1])  # Pose from image j to reference
            
            # Convert Pose3 to homography matrices
            H_i_mat = pose3_to_homography(H_i)
            H_j_mat = pose3_to_homography(H_j)
            
            # Compute predicted relative homography
            H_predicted = np.linalg.inv(H_j_mat) @ H_i_mat
            
            # Compute error between measured and predicted
            error = (H_predicted - self.measured_H).flatten()
            return error[:8]  # Return 8 DOF (ignoring scale)
    
    return HomographyFactor(key_i, key_j, measured_H, noise_model)

def homography_to_pose3(H):
    """Convert homography matrix to GTSAM Pose3 (simplified)"""
    # This is a simplified conversion assuming planar motion
    # Extract rotation and translation from homography
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]
    
    # Normalize to get rotation columns
    scale = np.linalg.norm(h1)
    r1 = h1 / scale
    r2 = h2 / scale
    r3 = np.cross(r1, r2)
    
    # Build rotation matrix
    R = np.column_stack([r1, r2, r3])
    
    # Ensure proper rotation matrix
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    
    # Translation
    t = h3 / scale
    
    # Create Pose3
    return gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(t[0], t[1], 0))

def pose3_to_homography(pose):
    """Convert GTSAM Pose3 back to homography matrix"""
    R = pose.rotation().matrix()
    t = pose.translation()
    
    # Build homography from R and t (assuming planar)
    H = np.eye(3)
    H[:2, :2] = R[:2, :2]
    H[:2, 2] = [t[0], t[1]]
    
    return H

def optimize_with_gtsam(images, pairwise_data, reference_idx=0):
    """
    Perform bundle adjustment using GTSAM to optimize all homographies jointly
    """
    print("\n" + "="*80)
    print("GTSAM BUNDLE ADJUSTMENT - OPTIMIZING HOMOGRAPHIES")
    print("="*80)
    
    # Create factor graph and initial estimates
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    
    # Add prior on reference image (identity transform)
    reference_key = symbol('H', reference_idx)
    identity_h = np.array([1, 0, 0, 0, 1, 0, 0, 0])  # Identity homography (minus h33=1)
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(8) * params.GTSAM_PRIOR_SIGMA)  # Very tight prior
    
    # Add the prior factor for the reference image
    graph.addPriorVector(reference_key, identity_h, prior_noise)
    initial.insert(reference_key, identity_h)
    
    # Initialize other poses using a simple breadth-first search from reference
    initialized = {reference_idx: np.eye(3)}
    to_initialize = set(range(len(images))) - {reference_idx}
    
    # BFS to initialize poses
    while to_initialize:
        found_connection = False
        for i in list(initialized.keys()):
            for j in list(to_initialize):
                pair_key = tuple(sorted([i, j]))
                if pair_key in pairwise_data:
                    H_ij = pairwise_data[pair_key]['homography']
                    
                    # Determine direction of homography
                    if pair_key[0] == i:
                        # H transforms from i to j
                        H_j = initialized[i] @ np.linalg.inv(H_ij)
                    else:
                        # H transforms from j to i
                        H_j = initialized[i] @ H_ij
                    
                    initialized[j] = H_j
                    to_initialize.remove(j)
                    
                    # Add to initial values (8 parameters, h33=1)
                    key_j = symbol('H', j)
                    h_vec = H_j.flatten()[:8] / H_j[2, 2]  # Normalize by h33
                    initial.insert(key_j, h_vec)
                    found_connection = True
                    break
            if found_connection:
                break
        
        if not found_connection and to_initialize:
            # Handle disconnected components
            print(f"Warning: Could not initialize images {to_initialize}")
            break
    
    # Add pairwise constraints using only one prior per non-reference node
    # This avoids conflicts from multiple constraints on the same variable
    factor_count = 0
    edges_used = []
    node_constraints = {}  # Track best constraint for each node
    
    for (i, j), data in pairwise_data.items():
        if i not in initialized or j not in initialized:
            continue
            
        H_measured = data['homography']
        covariance = data.get('covariance')
        
        # Create noise model from covariance
        if covariance is not None:
            sigmas = np.sqrt(np.diag(covariance))[:8]
            sigmas = np.clip(sigmas, 0.001, 10.0)
            uncertainty = np.trace(covariance)
            edges_used.append((i, j, uncertainty))
        else:
            sigmas = np.ones(8) * 0.1
            uncertainty = float('inf')
            edges_used.append((i, j, uncertainty))
        
        # For MST edges, we can safely add constraints since tree structure guarantees no conflicts
        # Add constraint for the non-reference node in each edge
        if i == reference_idx:
            # Add constraint for j based on reference
            H_j_expected = initialized[reference_idx] @ np.linalg.inv(H_measured) if (i, j) == tuple(sorted([i, j])) else initialized[reference_idx] @ H_measured
            h_j_expected = H_j_expected.flatten()[:8] / H_j_expected[2, 2]
            
            key_j = symbol('H', j)
            noise_model = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
            graph.addPriorVector(key_j, h_j_expected, noise_model)
            factor_count += 1
            
        elif j == reference_idx:
            # Add constraint for i based on reference
            H_i_expected = initialized[reference_idx] @ H_measured if (i, j) == tuple(sorted([i, j])) else initialized[reference_idx] @ np.linalg.inv(H_measured)
            h_i_expected = H_i_expected.flatten()[:8] / H_i_expected[2, 2]
            
            key_i = symbol('H', i)
            noise_model = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
            graph.addPriorVector(key_i, h_i_expected, noise_model)
            factor_count += 1
            
        else:
            # For non-reference edges, store the constraint and use the best one
            # This prevents over-constraining the system
            for node in [i, j]:
                if node == reference_idx:
                    continue
                    
                # Calculate expected position based on the other node
                other_node = j if node == i else i
                H_other = initialized[other_node]
                
                if tuple(sorted([i, j]))[0] == other_node:
                    # H_measured goes from other to node
                    H_node_expected = H_other @ np.linalg.inv(H_measured)
                else:
                    # H_measured goes from node to other
                    H_node_expected = H_other @ H_measured
                
                h_node_expected = H_node_expected.flatten()[:8] / H_node_expected[2, 2]
                
                # Keep only the best (most certain) constraint for each node
                if node not in node_constraints or uncertainty < node_constraints[node][1]:
                    node_constraints[node] = (h_node_expected, uncertainty, sigmas)
    
    # Add the best constraint for each non-reference node
    for node, (h_expected, uncertainty, sigmas) in node_constraints.items():
        key = symbol('H', node)
        noise_model = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
        graph.addPriorVector(key, h_expected, noise_model)
        factor_count += 1
    
    print(f"Graph construction complete with {len(initialized)} nodes")
    print(f"Added {factor_count} pairwise constraints weighted by covariance")
    
    # Print edge statistics
    if edges_used:
        finite_uncertainties = [u for _, _, u in edges_used if np.isfinite(u)]
        if finite_uncertainties:
            print(f"Edge uncertainty range: {min(finite_uncertainties):.2e} to {max(finite_uncertainties):.2e}")
    
    # Optimize using Levenberg-Marquardt
    print("Running GTSAM optimization...")
    gtsam_params = gtsam.LevenbergMarquardtParams()
    gtsam_params.setVerbosity('SILENT')
    gtsam_params.setMaxIterations(params.GTSAM_MAX_ITERATIONS)
    gtsam_params.setRelativeErrorTol(params.GTSAM_REL_ERROR_TOL)
    gtsam_params.setAbsoluteErrorTol(params.GTSAM_ABS_ERROR_TOL)
    
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, gtsam_params)
    
    try:
        result = optimizer.optimize()
        optimization_success = True
    except Exception as e:
        print(f"Optimization failed: {e}")
        result = initial  # Use initial values if optimization fails
        optimization_success = False
    
    # Extract optimized homographies
    optimized_transforms = {}
    for i in range(len(images)):
        key = symbol('H', i)
        if result.exists(key):
            h_vec = result.atVector(key)
            H_optimized = np.array([
                [h_vec[0], h_vec[1], h_vec[2]],
                [h_vec[3], h_vec[4], h_vec[5]],
                [h_vec[6], h_vec[7], 1.0]
            ])
            optimized_transforms[i] = H_optimized
        elif i in initialized:
            # Use initial value if not in result
            optimized_transforms[i] = initialized[i]
    
    if optimization_success:
        print(f"GTSAM optimization complete. Optimized {len(optimized_transforms)} poses.")
        
        # Compute optimization statistics
        initial_error = graph.error(initial)
        final_error = graph.error(result)
        print(f"Initial error: {initial_error:.4f}")
        print(f"Final error: {final_error:.4f}")
        if initial_error > 1e-10:  # Avoid division by zero
            print(f"Error reduction: {(1 - final_error/initial_error)*100:.2f}%")
        else:
            print(f"Error was already minimal")
    else:
        print(f"Using initial transforms for {len(optimized_transforms)} poses.")
    
    return optimized_transforms

def build_panorama_with_gtsam(images, transforms, title="GTSAM-Optimized Panorama"):
    """Build panorama using GTSAM-optimized transforms"""
    if not transforms:
        return None
    
    # Find reference (should be identity or close to it)
    ref_idx = min(transforms.keys(), 
                  key=lambda x: np.linalg.norm(transforms[x] - np.eye(3)))
    
    # Calculate global bounds
    global_corners = []
    for idx, H in transforms.items():
        img = images[idx][1]
        h, w = img.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
        corners_hom = np.column_stack([corners, np.ones(4)])
        transformed = (H @ corners_hom.T).T
        transformed_cart = transformed[:, :2] / transformed[:, [2]]
        global_corners.extend(transformed_cart)
    
    global_corners = np.array(global_corners)
    min_x, min_y = np.floor(global_corners.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(global_corners.max(axis=0)).astype(int)
    
    canvas_w, canvas_h = max_x - min_x, max_y - min_y
    offset_x, offset_y = -min_x, -min_y
    
    # Translation to center in canvas
    T = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)
    
    # Create result canvas
    result = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    
    # Warp and blend all images
    for idx in sorted(transforms.keys()):
        img = images[idx][1]
        final_H = T @ transforms[idx]
        warped = cv2.warpPerspective(img, final_H, (canvas_w, canvas_h))
        mask = warped > 0
        result[mask] = warped[mask]
    
    # Display result
    plt.figure(figsize=(16, 8))
    plt.imshow(result, cmap='gray')
    plt.title(title, fontsize=12, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return result

# ============= END OF GTSAM SECTION =============

def find_all_pairwise_matches(images, min_matches=None, min_inlier_ratio=None):
    """Find matches between all possible image pairs and return MST using covariance-based weights"""
    if min_matches is None:
        min_matches = params.MIN_MATCHES
    if min_inlier_ratio is None:
        min_inlier_ratio = params.MIN_INLIER_RATIO
        
    print(f"Analyzing {len(images)} images...")
    print(f"Parameters: min_matches={min_matches}, min_inlier_ratio={min_inlier_ratio:.1%}")
    
    pairwise_data = {}
    full_graph = nx.Graph()
    
    for i, (img_num, _, _) in enumerate(images):
        full_graph.add_node(i, image_num=img_num)
    
    # Find all valid pairs
    covariance_weights = {}  # Store covariance-based weights
    all_inlier_ratios = []  # Store all inlier ratios for histogram
    
    for i, j in combinations(range(len(images)), 2):
        img1_num, img1, _ = images[i]
        img2_num, img2, _ = images[j]
        
        kp1, kp2, matches = find_and_match_features(img1, img2)
        if len(matches) < min_matches:
            continue
        
        H, inliers, lm_result, mask, covariance = estimate_homography_ransac_lm(
            kp1, kp2, matches, min_matches, img1.shape, img2.shape)
        if H is None or lm_result is None:
            continue
        
        inlier_ratio = len(inliers) / len(matches)
        all_inlier_ratios.append(inlier_ratio)  # Store for histogram
        
        if inlier_ratio < min_inlier_ratio:
            continue
        
        inlier_src = np.array([kp1[m.queryIdx].pt for m in inliers])
        inlier_dst = np.array([kp2[m.trainIdx].pt for m in inliers])
        error = np.mean(np.abs(homography_residuals(H.flatten()[:8], inlier_src, inlier_dst)))
        
        # Additional quality checks
        quality_score = compute_homography_quality_score(H, inlier_src, inlier_dst)
        
        # Only accept if quality score is reasonable
        if quality_score > 1.0:  # Minimum quality threshold
            pairwise_data[(i, j)] = {
                'homography': H, 'matches': len(matches), 'inliers': len(inliers),
                'inlier_ratio': inlier_ratio, 'error': error, 
                'keypoints': (kp1, kp2), 'inlier_matches': inliers,
                'covariance': covariance, 'lm_result': lm_result,
                'quality_score': quality_score
            }
        else:
            print(f"Rejected pair Img{images[i][0]}↔Img{images[j][0]}: quality score too low ({quality_score:.2f})")
            continue
        
        # Compute covariance-based weight
        cov_weight = compute_covariance_weight(covariance)
        covariance_weights[(i, j)] = cov_weight
        
        # Handle case where covariance computation failed
        if covariance is not None:
            uncertainty_measure = np.trace(covariance)
        else:
            uncertainty_measure = float('inf')  # High uncertainty if computation failed
        
        # Use NEGATIVE covariance weight for minimum spanning tree
        # (NetworkX finds minimum spanning tree, but we want maximum weight edges)
        full_graph.add_edge(i, j, weight=-cov_weight, error=error, homography=H, 
                           inliers=len(inliers), covariance=covariance, 
                           uncertainty=uncertainty_measure, 
                           covariance_weight=cov_weight)
    
    print(f"Found {len(pairwise_data)} valid pairs")
    
    # Plot histogram of inlier ratios to help choose better threshold
    if all_inlier_ratios:
        plt.figure(figsize=(12, 8))
        
        # Create histogram
        plt.hist(all_inlier_ratios, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=min_inlier_ratio, color='red', linestyle='--', linewidth=2, 
                   label=f'Current threshold: {min_inlier_ratio:.1%}')
        
        # Add statistics
        mean_ratio = np.mean(all_inlier_ratios)
        median_ratio = np.median(all_inlier_ratios)
        std_ratio = np.std(all_inlier_ratios)
        
        plt.axvline(x=mean_ratio, color='green', linestyle='-', linewidth=2, 
                   label=f'Mean: {mean_ratio:.1%}')
        plt.axvline(x=median_ratio, color='orange', linestyle='-', linewidth=2, 
                   label=f'Median: {median_ratio:.1%}')
        
        # Calculate how many pairs would be included with different thresholds
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        included_counts = [sum(1 for ratio in all_inlier_ratios if ratio >= thresh) for thresh in thresholds]
        
        plt.title(f'Inlier Ratio Distribution\nTotal pairs analyzed: {len(all_inlier_ratios)}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Inlier Ratio', fontsize=12)
        plt.ylabel('Number of Image Pairs', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add text box with statistics
        stats_text = f'Statistics:\n'
        stats_text += f'Mean: {mean_ratio:.3f}\n'
        stats_text += f'Median: {median_ratio:.3f}\n'
        stats_text += f'Std: {std_ratio:.3f}\n'
        stats_text += f'Min: {min(all_inlier_ratios):.3f}\n'
        stats_text += f'Max: {max(all_inlier_ratios):.3f}\n\n'
        stats_text += f'Pairs included by threshold:\n'
        for thresh, count in zip(thresholds, included_counts):
            stats_text += f'{thresh:.0%}: {count} pairs\n'
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # Print recommendation
        print(f"\n{'='*60}")
        print("INLIER RATIO ANALYSIS")
        print(f"{'='*60}")
        print(f"Total pairs analyzed: {len(all_inlier_ratios)}")
        print(f"Current threshold ({min_inlier_ratio:.1%}) includes: {len(pairwise_data)} pairs")
        print(f"Mean inlier ratio: {mean_ratio:.1%}")
        print(f"Median inlier ratio: {median_ratio:.1%}")
        print(f"Standard deviation: {std_ratio:.1%}")
        
        # Find threshold that would include more pairs
        for thresh in [0.1, 0.15, 0.2, 0.25, 0.3]:
            count = sum(1 for ratio in all_inlier_ratios if ratio >= thresh)
            if count > len(pairwise_data):
                print(f"Threshold {thresh:.0%} would include {count} pairs (+{count - len(pairwise_data)} more)")
    
    # Display the full connection graph BEFORE creating MST
    if pairwise_data:
        show_full_connection_graph(images, pairwise_data)
    
    # Print covariance weight statistics
    if covariance_weights:
        weights_array = np.array(list(covariance_weights.values()))
        finite_weights = weights_array[np.isfinite(weights_array)]
        
        if len(finite_weights) > 0:
            print(f"\nCovariance Weight Statistics:")
            print(f"  Mean: {np.mean(finite_weights):.2e}")
            print(f"  Std:  {np.std(finite_weights):.2e}")
            print(f"  Min:  {np.min(finite_weights):.2e}")
            print(f"  Max:  {np.max(finite_weights):.2e}")
            print(f"  Edges with infinite weight: {len(weights_array) - len(finite_weights)}")
    
    # Create Maximum Spanning Tree based on covariance weights
    if full_graph.number_of_edges() > 0:
        # Find all connected components
        all_components = list(nx.connected_components(full_graph))
        print(f"\nConnected Components Analysis:")
        print(f"Total connected components: {len(all_components)}")
        
        for i, component in enumerate(all_components):
            print(f"  Component {i+1}: {len(component)} images - {[images[idx][0] for idx in component]}")
        
        # Find largest connected component first
        largest_cc = max(all_components, key=len)
        print(f"\nUsing largest component: {len(largest_cc)} images")
        print(f"Largest component images: {[images[idx][0] for idx in largest_cc]}")
        
        # Check for isolated images
        all_connected_images = set()
        for component in all_components:
            all_connected_images.update(component)
        
        isolated_images = set(range(len(images))) - all_connected_images
        if isolated_images:
            print(f"Isolated images (no connections): {[images[idx][0] for idx in isolated_images]}")
        
        subgraph = full_graph.subgraph(largest_cc).copy()
        
        # Get maximum spanning tree (minimum of negative covariance weights)
        mst = nx.minimum_spanning_tree(subgraph, weight='weight')
        
        # Update pairwise_data to only include MST edges
        mst_pairwise_data = {}
        for edge in mst.edges():
            pair_key = tuple(sorted(edge))
            if pair_key in pairwise_data:
                mst_pairwise_data[pair_key] = pairwise_data[pair_key]
        
        print(f"Covariance-weighted Maximum Spanning Tree: {mst.number_of_edges()} edges")
        
        # Print selected edges with their covariance weights
        print(f"\nSelected MST edges (sorted by covariance weight):")
        selected_edges = []
        for edge in mst.edges():
            pair_key = tuple(sorted(edge))
            if pair_key in covariance_weights:
                weight = covariance_weights[pair_key]
                uncertainty = mst_pairwise_data[pair_key]['covariance']
                uncertainty_val = np.trace(uncertainty) if uncertainty is not None else float('inf')
                img1_num, img2_num = images[edge[0]][0], images[edge[1]][0]
                selected_edges.append((pair_key, weight, uncertainty_val, img1_num, img2_num))
        
        # Sort by weight (descending - higher weight = lower uncertainty = better)
        selected_edges.sort(key=lambda x: x[1], reverse=True)
        
        for i, (pair_key, weight, uncertainty, img1_num, img2_num) in enumerate(selected_edges):
            print(f"  {i+1}. Img{img1_num}↔Img{img2_num}: weight={weight:.2e}, uncertainty={uncertainty:.2e}")
        
        # Return both all pairwise data and MST for comparison
        return pairwise_data, mst_pairwise_data, mst
    else:
        return {}, {}, nx.Graph()

def print_homography_matrix(matrix, title):
    """Print homography matrix with properties"""
    print(f"\n{title}:")
    print("┌" + " "*28 + "┐")
    for i in range(3):
        row = "│ " + " ".join([f"{matrix[i,j]:8.4f}" for j in range(3)]) + " │"
        print(row)
    print("└" + " "*28 + "┘")
    
    det = np.linalg.det(matrix)
    if matrix[2,2] != 0:
        tx, ty = matrix[0,2]/matrix[2,2], matrix[1,2]/matrix[2,2]
        print(f"Det: {det:.4f}, Translation: ({tx:.1f}, {ty:.1f})")

def compute_absolute_transforms(ref_idx, connected_indices, pairwise_data, 
                              graph, images, debug=True):
    """Compute absolute transformations to reference using the MST (which should be loop-free)"""
    print(f"Computing transforms to Img{images[ref_idx][0]} (reference)")
    
    transforms = {ref_idx: np.eye(3)}
    
    # BFS traversal from reference
    visited = {ref_idx}
    queue = [(ref_idx, np.eye(3))]
    
    while queue:
        current_idx, current_transform = queue.pop(0)
        
        for neighbor_idx in graph.neighbors(current_idx):
            if neighbor_idx in visited:
                continue
            
            pair_key = tuple(sorted([current_idx, neighbor_idx]))
            if pair_key not in pairwise_data:
                continue
            
            H_pair = pairwise_data[pair_key]['homography']
            
            # Determine homography direction
            if pair_key[0] == current_idx:
                step_H = np.linalg.inv(H_pair)
            else:
                step_H = H_pair
            
            neighbor_transform = current_transform @ step_H
            transforms[neighbor_idx] = neighbor_transform
            visited.add(neighbor_idx)
            queue.append((neighbor_idx, neighbor_transform))
    
    print(f"Computed transforms for {len(transforms)} images")
    
    return transforms

def stitch_images(images, transforms, ref_idx):
    """Stitch images using computed transforms"""
    # Calculate global bounds
    global_corners = []
    ref_img = images[ref_idx][1]
    h_ref, w_ref = ref_img.shape[:2]
    global_corners.extend([[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]])
    
    for idx, transform in transforms.items():
        if idx == ref_idx:
            continue
        img = images[idx][1]
        h, w = img.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
        corners_hom = np.column_stack([corners, np.ones(4)])
        transformed_corners = (transform @ corners_hom.T).T
        transformed_corners = transformed_corners[:, :2] / transformed_corners[:, [2]]
        global_corners.extend(transformed_corners)
    
    global_corners = np.array(global_corners)
    min_x, min_y = np.floor(global_corners.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(global_corners.max(axis=0)).astype(int)
    
    canvas_w, canvas_h = max_x - min_x, max_y - min_y
    offset_x, offset_y = -min_x, -min_y
    translation = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], 
                          dtype=np.float32)
    
    # Create result canvas
    result = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    
    for idx in sorted(transforms.keys()):
        img = images[idx][1]
        final_transform = translation @ (transforms[idx] if idx != ref_idx else np.eye(3))
        warped = cv2.warpPerspective(img, final_transform, (canvas_w, canvas_h))
        mask = warped > 0
        result[mask] = warped[mask]
    
    return result

def show_pairwise_stitches(images, pairwise_data):
    """Display enhanced pairwise stitching results with improved visual design"""
    if not pairwise_data:
        return
    
    sorted_pairs = sorted(pairwise_data.items(), 
                         key=lambda x: (images[x[0][0]][0], images[x[0][1]][0]))
    n_pairs = len(sorted_pairs)
    
    n_cols = min(2, n_pairs)  # Limit to 2 columns for larger display
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16*n_cols, 12*n_rows))
    fig.patch.set_facecolor('white')
    
    if n_pairs == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_cols > 1 else [axes]
    else:
        axes = axes.flatten()
    
    # Create a color map for different pairs
    colors = plt.cm.Set3(np.linspace(0, 1, n_pairs))
    
    for idx, ((i, j), data) in enumerate(sorted_pairs):
        img1_num, img1, _ = images[i]
        img2_num, img2, _ = images[j]
        H = data['homography']
        inliers = data['inliers']
        error = data['error']
        weight = compute_covariance_weight(data['covariance']) if data.get('covariance') is not None else 0
        
        # Simple pairwise stitch
        H_corrected = np.linalg.inv(H)
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Calculate bounds
        corners1 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]])
        corners2_hom = np.column_stack([corners2, np.ones(4)])
        corners2_transformed = (H_corrected @ corners2_hom.T).T
        corners2_transformed = corners2_transformed[:, :2] / corners2_transformed[:, [2]]
        
        all_corners = np.vstack([corners1, corners2_transformed])
        min_x, min_y = np.floor(all_corners.min(axis=0)).astype(int)
        max_x, max_y = np.ceil(all_corners.max(axis=0)).astype(int)
        
        canvas_w, canvas_h = max_x - min_x, max_y - min_y
        offset_x, offset_y = -min_x, -min_y
        T = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], 
                     dtype=np.float32)
        
        # Warp and blend
        warped1 = cv2.warpPerspective(img1, T, (canvas_w, canvas_h))
        warped2 = cv2.warpPerspective(img2, T @ H_corrected, (canvas_w, canvas_h))
        result = np.maximum(warped1, warped2)
        
        # Display with enhanced styling
        axes[idx].imshow(result, cmap='gray')
        
        # Enhanced title with more information
        title = f'Image {img1_num} ↔ Image {img2_num}\n'
        title += f'Matches: {inliers} | Error: {error:.1f}px\n'
        if np.isfinite(weight):
            title += f'Weight: {weight:.1e}'
        else:
            title += f'Weight: N/A'
            
        axes[idx].set_title(title, fontsize=8, fontweight='bold', 
                           pad=10, color='darkblue')
        
        # Add colored border based on weight quality
        if np.isfinite(weight) and weight > 0:
            # Normalize weight for color (higher weight = better = green)
            all_finite_weights = [compute_covariance_weight(d['covariance']) 
                                 for d in pairwise_data.values() 
                                 if d.get('covariance') is not None and 
                                 np.isfinite(compute_covariance_weight(d['covariance']))]
            if all_finite_weights:
                norm_weight = (weight - min(all_finite_weights)) / (max(all_finite_weights) - min(all_finite_weights))
                border_color = plt.cm.RdYlGn(norm_weight)
            else:
                border_color = 'gray'
        else:
            border_color = 'red'
        
        # Add colored border
        for spine in axes[idx].spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(4)
        
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
    
    # Hide unused subplots
    for i in range(n_pairs, len(axes)):
        axes[i].set_visible(False)
    
    # Enhanced main title
    plt.suptitle('Covariance-Weighted MST: Pairwise Image Stitching Results\n' + 
                f'{n_pairs} Selected Edges', 
                fontsize=16, fontweight='bold', y=0.98,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def show_covariance_analysis(images, pairwise_data):
    """Display detailed covariance analysis for each edge"""
    print(f"\n{'='*80}")
    print("DETAILED COVARIANCE ANALYSIS FOR COVARIANCE-WEIGHTED MST EDGES")
    print(f"{'='*80}")
    
    if not pairwise_data:
        print("No pairwise data available")
        return
    
    # Filter out edges with no covariance data and sort by uncertainty
    valid_edges = [(k, v) for k, v in pairwise_data.items() if v.get('covariance') is not None]
    
    if not valid_edges:
        print("No covariance data available for analysis")
        return
    
    sorted_edges = sorted(valid_edges, key=lambda x: np.trace(x[1]['covariance']))
    
    for edge_idx, ((i, j), data) in enumerate(sorted_edges):
        img1_num, img2_num = images[i][0], images[j][0]
        homography = data['homography']
        covariance = data['covariance']
        error = data['error']
        inliers = data['inliers']
        
        # Calculate covariance weight
        cov_weight = compute_covariance_weight(covariance)
        
        title = f"Edge {edge_idx+1}: Img{img1_num} ↔ Img{img2_num} ({inliers} inliers, {error:.1f}px, weight={cov_weight:.2e})"
        print_covariance_analysis(covariance, title, homography)
        
        if edge_idx < len(sorted_edges) - 1:
            print("\n" + "-"*50)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL COVARIANCE-WEIGHTED MST EDGES")
    print(f"{'='*80}")
    
    all_traces = [np.trace(data['covariance']) for data in pairwise_data.values() if data.get('covariance') is not None]
    all_errors = [data['error'] for data in pairwise_data.values()]
    all_inliers = [data['inliers'] for data in pairwise_data.values()]
    all_weights = [compute_covariance_weight(data['covariance']) for data in pairwise_data.values() if data.get('covariance') is not None]
    
    if all_traces:
        print(f"\nUncertainty Statistics (Trace of Covariance):")
        print(f"  Mean: {np.mean(all_traces):.2e}")
        print(f"  Std:  {np.std(all_traces):.2e}")
        print(f"  Min:  {np.min(all_traces):.2e}")
        print(f"  Max:  {np.max(all_traces):.2e}")
    else:
        print(f"\nNo uncertainty data available")
    
    if all_weights:
        finite_weights = [w for w in all_weights if np.isfinite(w)]
        if finite_weights:
            print(f"\nCovariance Weight Statistics:")
            print(f"  Mean: {np.mean(finite_weights):.2e}")
            print(f"  Std:  {np.std(finite_weights):.2e}")
            print(f"  Min:  {np.min(finite_weights):.2e}")
            print(f"  Max:  {np.max(finite_weights):.2e}")
    
    print(f"\nReprojection Error Statistics:")
    print(f"  Mean: {np.mean(all_errors):.2f}px")
    print(f"  Std:  {np.std(all_errors):.2f}px")
    print(f"  Min:  {np.min(all_errors):.2f}px")
    print(f"  Max:  {np.max(all_errors):.2f}px")
    
    print(f"\nInlier Count Statistics:")
    print(f"  Mean: {np.mean(all_inliers):.1f}")
    print(f"  Std:  {np.std(all_inliers):.1f}")
    print(f"  Min:  {np.min(all_inliers)}")
    print(f"  Max:  {np.max(all_inliers)}")

def show_connection_graph(images, pairwise_data, graph):
    """Display enhanced covariance-weighted MST graph with improved visual design"""
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Use different layout algorithms for better positioning
    if len(graph.nodes()) <= 6:
        pos = nx.circular_layout(graph, scale=3)
    else:
        pos = nx.spring_layout(graph, seed=42, k=4, iterations=100, scale=3)
    
    # Enhance node appearance
    node_labels = {i: f"Image {images[i][0]}" for i in range(len(images))}
    
    # Calculate edge properties for visualization
    edge_weights = []
    edge_widths = []
    edge_labels = {}
    edge_alphas = []
    
    for (i, j) in graph.edges():
        pair_key = tuple(sorted([i, j]))
        if pair_key in pairwise_data and pairwise_data[pair_key].get('covariance') is not None:
            weight = compute_covariance_weight(pairwise_data[pair_key]['covariance'])
            uncertainty = np.trace(pairwise_data[pair_key]['covariance'])
            matches = pairwise_data[pair_key]['inliers']
            error = pairwise_data[pair_key]['error']
            
            edge_weights.append(weight if np.isfinite(weight) else 0)
            edge_labels[(i, j)] = f"Matches: {matches}\nWeight: {weight:.1e}\nError: {error:.1f}px"
            
            # Vary edge width based on number of matches
            edge_widths.append(max(2, min(8, matches / 20)))  # Scale between 2-8
            edge_alphas.append(0.8)
        else:
            edge_weights.append(0)
            matches = pairwise_data.get(pair_key, {}).get('inliers', 0)
            edge_labels[(i, j)] = f"Matches: {matches}\nWeight: N/A"
            edge_widths.append(1)
            edge_alphas.append(0.4)
    
    # Create sophisticated color mapping for edges
    if edge_weights and max(edge_weights) > 0:
        finite_weights = [w for w in edge_weights if np.isfinite(w) and w > 0]
        if finite_weights and len(finite_weights) > 1:
            min_weight, max_weight = min(finite_weights), max(finite_weights)
            if max_weight > min_weight:
                norm_weights = [(w - min_weight) / (max_weight - min_weight) if np.isfinite(w) and w > 0 else 0 
                               for w in edge_weights]
            else:
                norm_weights = [1.0 if np.isfinite(w) and w > 0 else 0 for w in edge_weights]
        else:
            norm_weights = [1.0 if np.isfinite(w) and w > 0 else 0 for w in edge_weights]
        
        # Use a more sophisticated colormap
        cmap = plt.cm.plasma_r  # Purple (low) to Yellow (high)
        edge_colors = [cmap(w) for w in norm_weights]
    else:
        edge_colors = ['gray'] * len(graph.edges())
    
    # Draw all edges at once to avoid warnings
    nx.draw_networkx_edges(graph, pos, 
                          edge_color=edge_colors,
                          width=edge_widths,
                          alpha=0.8)
    
    # Draw nodes with gradient effect and shadows
    node_sizes = [2500] * len(graph.nodes())  # Larger nodes
    
    # Draw shadow first
    shadow_pos = {node: (x + 0.05, y - 0.05) for node, (x, y) in pos.items()}
    nx.draw_networkx_nodes(graph, shadow_pos, node_color='black', 
                          node_size=node_sizes, alpha=0.3)
    
    # Draw main nodes with gradient-like effect
    nx.draw_networkx_nodes(graph, pos, node_color='lightsteelblue', 
                          node_size=node_sizes, alpha=0.9,
                          edgecolors='navy', linewidths=3)
    
    # Add inner highlight to nodes
    nx.draw_networkx_nodes(graph, pos, node_color='white', 
                          node_size=[s*0.3 for s in node_sizes], alpha=0.6)
    
    # Enhanced node labels
    for node, (x, y) in pos.items():
        ax.text(x, y, node_labels[node], ha='center', va='center',
                fontsize=9, fontweight='bold', color='darkblue',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='navy', alpha=0.8))
    
    # Enhanced edge labels with better positioning
    edge_label_pos = {}
    for (i, j), label in edge_labels.items():
        x1, y1 = pos[i]
        x2, y2 = pos[j]
        # Position label slightly offset from edge midpoint
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Calculate perpendicular offset
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            # Perpendicular vector
            perp_x, perp_y = -dy / length, dx / length
            offset = 0.3
            label_x = mid_x + offset * perp_x
            label_y = mid_y + offset * perp_y
        else:
            label_x, label_y = mid_x, mid_y
        
        ax.text(label_x, label_y, label, ha='center', va='center',
                fontsize=7, fontweight='normal',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', 
                         edgecolor='orange', alpha=0.9, linewidth=1))
    
    # Create a custom colorbar for edge weights
    if finite_weights and len(finite_weights) > 1:
        sm = plt.cm.ScalarMappable(cmap=cmap, 
                                  norm=plt.Normalize(vmin=min(finite_weights), 
                                                   vmax=max(finite_weights)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.02)
        cbar.set_label('Covariance Weight\n(Higher = More Reliable)', 
                      fontsize=10, fontweight='bold')
        cbar.ax.tick_params(labelsize=8)
    
    # Enhanced title and layout
    ax.set_title('Covariance-Weighted Maximum Spanning Tree\nImage Connection Graph', 
                fontsize=12, fontweight='bold', pad=15,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='purple', lw=4, alpha=0.8, label='Low Weight (High Uncertainty)'),
        plt.Line2D([0], [0], color='yellow', lw=4, alpha=0.8, label='High Weight (Low Uncertainty)'),
        plt.Circle((0, 0), 0.1, color='lightsteelblue', label='Image Node')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98),
             frameon=True, fancybox=True, shadow=True, fontsize=9)
    
    # Add statistics box
    if pairwise_data:
        stats_text = f"Graph Statistics:\n"
        stats_text += f"• Nodes: {graph.number_of_nodes()}\n"
        stats_text += f"• Edges: {graph.number_of_edges()}\n"
        stats_text += f"• Min Matches Required: {params.MIN_MATCHES}\n"
        if finite_weights:
            stats_text += f"• Weight Range: {min(finite_weights):.1e} - {max(finite_weights):.1e}\n"
            stats_text += f"• Avg Matches: {np.mean([data['inliers'] for data in pairwise_data.values()]):.1f}\n"
        
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
               ha='right', va='bottom', fontsize=8,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', 
                        alpha=0.9, edgecolor='teal'))
    
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def compute_image_positions(images, transforms, ref_idx):
    """Compute absolute 2D positions of each image in panorama coordinates"""
    positions = {}
    
    # Reference image is at origin
    ref_img = images[ref_idx][1]
    h_ref, w_ref = ref_img.shape[:2]
    positions[ref_idx] = (w_ref/2, h_ref/2)  # Center of reference image
    
    # Compute positions for all other images
    for idx, transform in transforms.items():
        if idx == ref_idx:
            continue
            
        img = images[idx][1]
        h, w = img.shape[:2]
        
        # Transform image corners to get position
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
        corners_hom = np.column_stack([corners, np.ones(4)])
        transformed_corners = (transform @ corners_hom.T).T
        transformed_corners = transformed_corners[:, :2] / transformed_corners[:, [2]]
        
        # Use center of transformed image as position
        center_x = np.mean(transformed_corners[:, 0])
        center_y = np.mean(transformed_corners[:, 1])
        positions[idx] = (center_x, center_y)
    
    return positions

def show_position_graph(images, pairwise_data, graph, transforms, ref_idx):
    """Display graph with nodes positioned at their actual 2D image positions"""
    print("\nComputing absolute image positions...")
    
    # Compute absolute positions
    positions = compute_image_positions(images, transforms, ref_idx)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # Normalize positions to fit in plot
    if positions:
        pos_array = np.array(list(positions.values()))
        min_x, min_y = np.min(pos_array, axis=0)
        max_x, max_y = np.max(pos_array, axis=0)
        
        # Add some padding
        range_x = max_x - min_x
        range_y = max_y - min_y
        padding_x = range_x * 0.1
        padding_y = range_y * 0.1
        
        # Normalize to [0, 1] range
        normalized_pos = {}
        for idx, (x, y) in positions.items():
            norm_x = (x - min_x + padding_x) / (range_x + 2*padding_x)
            norm_y = (y - min_y + padding_y) / (range_y + 2*padding_y)
            normalized_pos[idx] = (norm_x, norm_y)
    else:
        normalized_pos = {}
    
    # Ensure all nodes have positions (use default position for missing ones)
    for i in range(len(images)):
        if i not in normalized_pos:
            normalized_pos[i] = (0.5, 0.5)  # Default center position
    
    # Create graph with actual positions
    pos_graph = nx.Graph()
    for i, (img_num, _, _) in enumerate(images):
        pos_graph.add_node(i, image_num=img_num)
    
    # Add edges from MST
    for (i, j) in graph.edges():
        if (i, j) in pairwise_data or (j, i) in pairwise_data:
            pos_graph.add_edge(i, j)
    
    # Node labels
    node_labels = {i: f"Img{images[i][0]}" for i in range(len(images))}
    
    # Edge labels with weights
    edge_labels = {}
    for (i, j) in pos_graph.edges():
        if (i, j) in pairwise_data:
            data = pairwise_data[(i, j)]
        elif (j, i) in pairwise_data:
            data = pairwise_data[(j, i)]
        else:
            continue
            
        if data.get('covariance') is not None:
            weight = compute_covariance_weight(data['covariance'])
            edge_labels[(i, j)] = f"{weight:.2e}"
        else:
            edge_labels[(i, j)] = "N/A"
    
    # Draw edges with weights
    edge_widths = []
    edge_colors = []
    for (i, j) in pos_graph.edges():
        if (i, j) in pairwise_data:
            data = pairwise_data[(i, j)]
        elif (j, i) in pairwise_data:
            data = pairwise_data[(j, i)]
        else:
            continue
            
        if data.get('covariance') is not None:
            weight = compute_covariance_weight(data['covariance'])
            edge_widths.append(max(1, min(8, weight * 2)))
            edge_colors.append('blue')
        else:
            edge_widths.append(1)
            edge_colors.append('gray')
    
    # Draw edges
    nx.draw_networkx_edges(pos_graph, normalized_pos,
                          edge_color=edge_colors,
                          width=edge_widths,
                          alpha=0.7)
    
    # Draw nodes
    node_sizes = [800] * len(pos_graph.nodes())
    nx.draw_networkx_nodes(pos_graph, normalized_pos,
                          node_color='lightcoral',
                          node_size=node_sizes,
                          alpha=0.8,
                          edgecolors='darkred',
                          linewidths=2)
    
    # Node labels
    for node, (x, y) in normalized_pos.items():
        ax.text(x, y, node_labels[node], ha='center', va='center',
                fontsize=10, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='darkred', alpha=0.8))
    
    # Edge labels
    for (i, j), label in edge_labels.items():
        if (i, j) in normalized_pos and (j, i) in normalized_pos:
            x1, y1 = normalized_pos[i]
            x2, y2 = normalized_pos[j]
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y, label, ha='center', va='center',
                   fontsize=8, color='blue', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))
    
    # Add title and info
    ax.set_title('Image Network with Absolute 2D Positions\n' + 
                f'Reference: Img{images[ref_idx][0]} at center, {len(positions)} images positioned',
                fontsize=12, fontweight='bold', pad=15)
    
    # Add position info
    info_text = f"Position Statistics:\n"
    if positions:
        pos_array = np.array(list(positions.values()))
        info_text += f"X range: {np.min(pos_array[:, 0]):.0f} to {np.max(pos_array[:, 0]):.0f}\n"
        info_text += f"Y range: {np.min(pos_array[:, 1]):.0f} to {np.max(pos_array[:, 1]):.0f}\n"
        info_text += f"Total span: {np.max(pos_array[:, 0]) - np.min(pos_array[:, 0]):.0f} x {np.max(pos_array[:, 1]) - np.min(pos_array[:, 1]):.0f}"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print position details
    print(f"\n{'='*60}")
    print("ABSOLUTE IMAGE POSITIONS IN PANORAMA COORDINATES")
    print(f"{'='*60}")
    print(f"Reference image: Img{images[ref_idx][0]} at origin (0, 0)")
    print()
    
    for idx in sorted(positions.keys()):
        if idx != ref_idx:
            x, y = positions[idx]
            img_num = images[idx][0]
            print(f"Img{img_num:2d}: ({x:8.1f}, {y:8.1f})")
    
    return positions

def analyze_homography_quality(pairwise_data):
    """Analyze and display homography quality statistics"""
    if not pairwise_data:
        return
    
    print(f"\n{'='*60}")
    print("HOMOGRAPHY QUALITY ANALYSIS")
    print(f"{'='*60}")
    
    quality_scores = []
    determinants = []
    condition_numbers = []
    errors = []
    
    for (i, j), data in pairwise_data.items():
        H = data.get('homography')
        if H is not None:
            # Quality score
            if 'quality_score' in data:
                quality_scores.append(data['quality_score'])
            
            # Matrix properties
            det = np.linalg.det(H)
            determinants.append(det)
            
            try:
                cond = np.linalg.cond(H)
                condition_numbers.append(cond)
            except:
                condition_numbers.append(float('inf'))
            
            # Reprojection error
            if 'error' in data:
                errors.append(data['error'])
    
    if quality_scores:
        print(f"Quality Scores:")
        print(f"  Mean: {np.mean(quality_scores):.2f}")
        print(f"  Std:  {np.std(quality_scores):.2f}")
        print(f"  Min:  {np.min(quality_scores):.2f}")
        print(f"  Max:  {np.max(quality_scores):.2f}")
    
    if determinants:
        print(f"\nDeterminants:")
        print(f"  Mean: {np.mean(determinants):.2e}")
        print(f"  Std:  {np.std(determinants):.2e}")
        print(f"  Min:  {np.min(determinants):.2e}")
        print(f"  Max:  {np.max(determinants):.2e}")
    
    if condition_numbers:
        finite_conds = [c for c in condition_numbers if np.isfinite(c)]
        if finite_conds:
            print(f"\nCondition Numbers (finite only):")
            print(f"  Mean: {np.mean(finite_conds):.2e}")
            print(f"  Std:  {np.std(finite_conds):.2e}")
            print(f"  Min:  {np.min(finite_conds):.2e}")
            print(f"  Max:  {np.max(finite_conds):.2e}")
        else:
            print(f"\nCondition Numbers: All infinite (singular matrices)")
    
    if errors:
        print(f"\nReprojection Errors:")
        print(f"  Mean: {np.mean(errors):.2f}px")
        print(f"  Std:  {np.std(errors):.2f}px")
        print(f"  Min:  {np.min(errors):.2f}px")
        print(f"  Max:  {np.max(errors):.2f}px")
    
    # Identify problematic matrices
    print(f"\nProblematic Homographies:")
    problem_count = 0
    for (i, j), data in pairwise_data.items():
        H = data.get('homography')
        if H is not None:
            is_valid, reason = validate_homography_matrix(H)
            if not is_valid:
                print(f"  Img{i}↔Img{j}: {reason}")
                problem_count += 1
    
    if problem_count == 0:
        print("  None - all homographies are valid!")
    else:
        print(f"  Total problematic: {problem_count}")

def analyze_rejected_homographies():
    """Display statistics about rejected homographies"""
    print(f"\n{'='*60}")
    print("REJECTED HOMOGRAPHIES ANALYSIS")
    print(f"{'='*60}")
    print("The following homography matrices were rejected due to:")
    print("1. Poor numerical properties (determinant, condition number)")
    print("2. Unreasonable transformations (extreme scaling, translation)")
    print("3. Exploding transformations (corners mapped to extreme distances)")
    print("4. Low quality scores")
    print("\nThis filtering helps prevent distorted panoramas and improves")
    print("overall stitching quality by using only reliable transformations.")

def gtsam_optimize_2d_positions(images, pairwise_data, initial_positions, reference_idx=0):
    """
    Optimize 2D positions using GTSAM with covariance information from edges
    """
    print(f"\n{'='*80}")
    print("GTSAM 2D POSITION OPTIMIZATION WITH COVARIANCE")
    print(f"{'='*80}")
    
    # Create factor graph
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    
    # Add variables for all images
    X = lambda idx: gtsam.symbol('x', idx)
    for idx in initial_positions.keys():
        pos = initial_positions[idx]
        initial_estimate.insert(X(idx), gtsam.Point2(pos[0], pos[1]))
    
    # Add prior factor for reference image (fixed at origin)
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01]))
    graph.addPriorPoint2(X(reference_idx), gtsam.Point2(0.0, 0.0), prior_noise)
    
    # Add between factors for pairwise measurements with covariance-based noise models
    factor_count = 0
    for (i, j), data in pairwise_data.items():
        if i not in initial_positions or j not in initial_positions:
            continue
            
        # Extract translation from homography
        H = data['homography']
        H_inv = np.linalg.inv(H)
        measured_translation = gtsam.Point2(H_inv[0, 2], H_inv[1, 2])
        
        # Create noise model based on covariance
        covariance = data.get('covariance')
        if covariance is not None:
            # Extract x,y translation uncertainties from covariance
            # Use diagonal elements corresponding to translation (h02, h12)
            sigma_x = np.sqrt(covariance[2, 2])  # h02 uncertainty
            sigma_y = np.sqrt(covariance[5, 5])   # h12 uncertainty
            
            # Ensure minimum noise to avoid singular matrices
            sigma_x = max(sigma_x, 0.1)
            sigma_y = max(sigma_y, 0.1)
            
            noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([sigma_x, sigma_y]))
            uncertainty = np.trace(covariance)
        else:
            # Fallback to error-based noise model
            error = data.get('error', 1.0)
            sigma = max(error * 0.5, 0.5)
            sigma_x = sigma  # ADD THIS LINE
            sigma_y = sigma  # ADD THIS LINE
            noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([sigma, sigma]))
            uncertainty = float('inf')
        
        # Add between factor
        graph.add(gtsam.BetweenFactorPoint2(X(i), X(j), measured_translation, noise_model))
        factor_count += 1
        
        print(f"Added factor ({i},{j}): translation=({H_inv[0,2]:.1f}, {H_inv[1,2]:.1f}), "
              f"noise=({sigma_x:.3f}, {sigma_y:.3f}), uncertainty={uncertainty:.2e}")
    
    print(f"Created factor graph with {factor_count} between factors")
    
    # Run GTSAM optimization
    print("Running GTSAM optimization...")
    parameters = gtsam.LevenbergMarquardtParams()
    parameters.setMaxIterations(100)
    parameters.setAbsoluteErrorTol(1e-6)
    parameters.setRelativeErrorTol(1e-6)
    parameters.setVerbosity('SILENT')
    
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, parameters)
    
    try:
        result = optimizer.optimize()
        optimization_success = True
        
        # Compute optimization statistics
        initial_error = graph.error(initial_estimate)
        final_error = graph.error(result)
        print(f"Initial error: {initial_error:.4f}")
        print(f"Final error: {final_error:.4f}")
        if initial_error > 1e-10:
            print(f"Error reduction: {(1 - final_error/initial_error)*100:.2f}%")
    except Exception as e:
        print(f"Optimization failed: {e}")
        result = initial_estimate
        optimization_success = False
    
    # Extract optimized positions
    optimized_positions = {}
    for idx in initial_positions.keys():
        if result.exists(X(idx)):
            point = result.atPoint2(X(idx))
            # Handle both Point2 objects and numpy arrays
            if hasattr(point, 'x') and hasattr(point, 'y'):
                optimized_positions[idx] = np.array([point.x(), point.y()])
            else:
                # If it's already a numpy array
                optimized_positions[idx] = np.array(point)
        else:
            optimized_positions[idx] = initial_positions[idx]
    
    if optimization_success:
        print(f"GTSAM optimization complete. Optimized {len(optimized_positions)} positions.")
    else:
        print(f"Using initial positions for {len(optimized_positions)} images.")
    
    return optimized_positions

def update_homographies_with_optimized_positions(images, transforms, initial_positions, optimized_positions, reference_idx=0):
    """
    Update homographies using GTSAM-optimized positions
    Apply translation to each homography based on position differences
    """
    print(f"\n{'='*80}")
    print("UPDATING HOMOGRAPHIES WITH GTSAM-OPTIMIZED POSITIONS")
    print(f"{'='*80}")
    
    updated_transforms = {}
    
    # Reference image stays at origin (no change needed)
    updated_transforms[reference_idx] = transforms[reference_idx]
    
    # Calculate position differences and update homographies
    for idx in transforms.keys():
        if idx == reference_idx:
            continue
            
        if idx in initial_positions and idx in optimized_positions:
            # Get position differences
            initial_pos = np.array(initial_positions[idx])
            optimized_pos = np.array(optimized_positions[idx])
            position_diff = optimized_pos - initial_pos
            
            tx, ty = position_diff[0], position_diff[1]
            
            # Create translation matrix
            T = np.array([
                [1, 0, tx],
                [0, 1, ty],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Apply translation AFTER the original homography
            # H_new = T @ H_original
            H_original = transforms[idx]
            H_updated = T @ H_original
            
            updated_transforms[idx] = H_updated
            
            print(f"Img{images[idx][0]:2d}: pos_diff=({tx:7.1f}, {ty:7.1f}) -> "
                  f"translation=({tx:7.1f}, {ty:7.1f})")
        else:
            # Keep original transform if no optimized position available
            updated_transforms[idx] = transforms[idx]
            print(f"Img{images[idx][0]:2d}: No optimized position, keeping original transform")
    
    print(f"Updated {len(updated_transforms)} homographies with GTSAM-optimized positions")
    return updated_transforms

def build_panorama_with_updated_homographies(images, updated_transforms, title="GTSAM Position-Optimized Panorama"):
    """Build panorama using homographies updated with GTSAM-optimized positions"""
    if not updated_transforms:
        return None
    
    # Find reference (should be identity or close to it)
    ref_idx = min(updated_transforms.keys(), 
                  key=lambda x: np.linalg.norm(updated_transforms[x] - np.eye(3)))
    
    # Calculate global bounds
    global_corners = []
    for idx, H in updated_transforms.items():
        img = images[idx][1]
        h, w = img.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
        corners_hom = np.column_stack([corners, np.ones(4)])
        transformed = (H @ corners_hom.T).T
        transformed_cart = transformed[:, :2] / transformed[:, [2]]
        global_corners.extend(transformed_cart)
    
    global_corners = np.array(global_corners)
    min_x, min_y = np.floor(global_corners.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(global_corners.max(axis=0)).astype(int)
    
    canvas_w, canvas_h = max_x - min_x, max_y - min_y
    offset_x, offset_y = -min_x, -min_y
    
    # Translation to center in canvas
    T = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)
    
    # Create result canvas
    result = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    
    # Warp and blend all images
    for idx in sorted(updated_transforms.keys()):
        img = images[idx][1]
        final_H = T @ updated_transforms[idx]
        warped = cv2.warpPerspective(img, final_H, (canvas_w, canvas_h))
        mask = warped > 0
        result[mask] = warped[mask]
    
    # Display result
    plt.figure(figsize=(16, 8))
    plt.imshow(result, cmap='gray')
    plt.title(title, fontsize=12, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return result

def show_position_comparison(images, pairwise_data, initial_positions, optimized_positions, reference_idx=0):
    """
    Display comparison between original and GTSAM-optimized 2D positions
    """
    print(f"\n{'='*80}")
    print("POSITION COMPARISON: ORIGINAL vs GTSAM-OPTIMIZED")
    print(f"{'='*80}")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    
    # Normalize positions for both plots
    all_positions = list(initial_positions.values()) + list(optimized_positions.values())
    pos_array = np.array(all_positions)
    min_x, min_y = np.min(pos_array, axis=0)
    max_x, max_y = np.max(pos_array, axis=0)
    
    # Add padding
    range_x = max_x - min_x
    range_y = max_y - min_y
    padding_x = range_x * 0.1
    padding_y = range_y * 0.1
    
    def normalize_positions(positions):
        normalized = {}
        for idx, (x, y) in positions.items():
            norm_x = (x - min_x + padding_x) / (range_x + 2*padding_x)
            norm_y = (y - min_y + padding_y) / (range_y + 2*padding_y)
            normalized[idx] = (norm_x, norm_y)
        return normalized
    
    norm_initial = normalize_positions(initial_positions)
    norm_optimized = normalize_positions(optimized_positions)
    
    # Create graphs for both plots - only include nodes that have positions
    graph = nx.Graph()
    for idx in initial_positions.keys():
        img_num = images[idx][0]
        graph.add_node(idx, image_num=img_num)
    
    # Add edges from pairwise_data
    for (i, j) in pairwise_data.keys():
        if i in initial_positions and j in initial_positions:
            graph.add_edge(i, j)
    
    # Plot 1: Original positions
    ax1.set_title('Original 2D Positions (MST-based)', fontsize=14, fontweight='bold')
    
    # Draw edges
    edge_widths = []
    edge_colors = []
    for (i, j) in graph.edges():
        if (i, j) in pairwise_data:
            data = pairwise_data[(i, j)]
            if data.get('covariance') is not None:
                uncertainty = np.trace(data['covariance'])
                edge_widths.append(max(1, min(6, 10/np.sqrt(uncertainty))))
                edge_colors.append('blue')
            else:
                edge_widths.append(2)
                edge_colors.append('gray')
        else:
            edge_widths.append(1)
            edge_colors.append('lightgray')
    
    nx.draw_networkx_edges(graph, norm_initial, ax=ax1,
                          edge_color=edge_colors,
                          width=edge_widths,
                          alpha=0.7)
    
    # Draw nodes
    node_sizes = [800] * len(graph.nodes())
    nx.draw_networkx_nodes(graph, norm_initial, ax=ax1,
                          node_color='lightcoral',
                          node_size=node_sizes,
                          alpha=0.8,
                          edgecolors='darkred',
                          linewidths=2)
    
    # Node labels
    for node, (x, y) in norm_initial.items():
        ax1.text(x, y, f"Img{images[node][0]}", ha='center', va='center',
                fontsize=10, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='darkred', alpha=0.8))
    
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Plot 2: GTSAM-optimized positions
    ax2.set_title('GTSAM-Optimized 2D Positions', fontsize=14, fontweight='bold')
    
    # Draw edges (same as before)
    nx.draw_networkx_edges(graph, norm_optimized, ax=ax2,
                          edge_color=edge_colors,
                          width=edge_widths,
                          alpha=0.7)
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, norm_optimized, ax=ax2,
                          node_color='lightgreen',
                          node_size=node_sizes,
                          alpha=0.8,
                          edgecolors='darkgreen',
                          linewidths=2)
    
    # Node labels
    for node, (x, y) in norm_optimized.items():
        ax2.text(x, y, f"Img{images[node][0]}", ha='center', va='center',
                fontsize=10, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='darkgreen', alpha=0.8))
    
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # Add statistics
    stats_text = f"Position Statistics:\n"
    stats_text += f"Images: {len(initial_positions)}\n"
    stats_text += f"Edges: {len(graph.edges())}\n"
    stats_text += f"Reference: Img{images[reference_idx][0]}\n"
    
    # Calculate position changes
    position_changes = []
    for idx in initial_positions.keys():
        if idx in optimized_positions:
            initial_pos = np.array(initial_positions[idx])
            optimized_pos = np.array(optimized_positions[idx])
            change = np.linalg.norm(optimized_pos - initial_pos)
            position_changes.append(change)
    
    if position_changes:
        stats_text += f"Avg position change: {np.mean(position_changes):.1f}px\n"
        stats_text += f"Max position change: {np.max(position_changes):.1f}px\n"
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('2D Position Comparison: Original vs GTSAM-Optimized', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print detailed position comparison
    print(f"\nDetailed Position Comparison:")
    print(f"{'Image':<8} {'Original (x,y)':<20} {'Optimized (x,y)':<20} {'Change (px)':<12}")
    print("-" * 70)
    
    for idx in sorted(initial_positions.keys()):
        if idx in optimized_positions:
            orig_pos = initial_positions[idx]
            opt_pos = optimized_positions[idx]
            change = np.linalg.norm(np.array(opt_pos) - np.array(orig_pos))
            print(f"Img{images[idx][0]:<6} ({orig_pos[0]:7.1f},{orig_pos[1]:7.1f})    "
                  f"({opt_pos[0]:7.1f},{opt_pos[1]:7.1f})    {change:8.1f}")
    
    return optimized_positions

def build_panorama_progressive(images, pairwise_data, graph):
    """Build panorama progressively with visualization"""
    print("Building progressive panorama using covariance-weighted MST...")
    
    if not pairwise_data:
        return None, None, None
    
    # Find largest connected component
    largest_cc = max(nx.connected_components(graph), key=len)
    if len(largest_cc) < 2:
        return None, None, None
    
    # Find endpoints of the graph (nodes with degree 1) for chain-like graphs
    # Or use the most connected node for star-like graphs
    degrees = dict(graph.degree())
    endpoints = [node for node in largest_cc if degrees[node] == 1]
    
    if len(endpoints) >= 2:
        # Graph is chain-like, start from one endpoint
        start_idx = endpoints[0]
        print(f"Graph is chain-like, starting from endpoint Image {images[start_idx][0]}")
    else:
        # Graph is more complex, use most connected node as reference
        start_idx = max(largest_cc, key=lambda x: degrees[x])
        print(f"Graph is complex, using most connected node Image {images[start_idx][0]} as reference")
    
    # Traverse the graph to build ordered list and compute transforms
    traversal_order = []
    transforms = {}
    visited = set()
    
    # BFS or DFS to traverse the graph and compute transforms incrementally
    queue = [(start_idx, np.eye(3))]
    visited.add(start_idx)
    transforms[start_idx] = np.eye(3)
    traversal_order.append(start_idx)
    
    while queue:
        current_idx, current_transform = queue.pop(0)
        
        # Get neighbors in the MST
        for neighbor_idx in graph.neighbors(current_idx):
            if neighbor_idx in visited:
                continue
            
            pair_key = tuple(sorted([current_idx, neighbor_idx]))
            if pair_key not in pairwise_data:
                continue
            
            H_pair = pairwise_data[pair_key]['homography']
            
            # Determine homography direction and accumulate transform
            if pair_key[0] == current_idx:
                # H transforms from current to neighbor
                step_H = np.linalg.inv(H_pair)
            else:
                # H transforms from neighbor to current
                step_H = H_pair
            
            # Accumulate transformation
            neighbor_transform = current_transform @ step_H
            transforms[neighbor_idx] = neighbor_transform
            visited.add(neighbor_idx)
            traversal_order.append(neighbor_idx)
            queue.append((neighbor_idx, neighbor_transform))
    
    print(f"Traversal order: {[images[idx][0] for idx in traversal_order]}")
    
    if len(traversal_order) < 2:
        return None, None, None
    
    # Calculate global canvas based on all transforms
    global_corners = []
    for idx in traversal_order:
        img = images[idx][1]
        h, w = img.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
        corners_hom = np.column_stack([corners, np.ones(4)])
        transformed = (transforms[idx] @ corners_hom.T).T
        transformed_cart = transformed[:, :2] / transformed[:, [2]]
        global_corners.extend(transformed_cart)
    
    global_corners = np.array(global_corners)
    min_x, min_y = np.floor(global_corners.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(global_corners.max(axis=0)).astype(int)
    
    canvas_w, canvas_h = max_x - min_x, max_y - min_y
    offset_x, offset_y = -min_x, -min_y
    translation = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], 
                          dtype=np.float32)
    
    # Progressive building with visualization - follow traversal order
    result_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    n_steps = len(traversal_order)
    n_cols = min(params.MAX_DISPLAY_COLS, n_steps)
    n_rows = (n_steps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_steps == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_cols > 1 else [axes]
    else:
        axes = axes.flatten()
    
    added_images = []
    for step, idx in enumerate(traversal_order):
        img_num = images[idx][0]
        img = images[idx][1]
        added_images.append(img_num)
        
        # Apply accumulated transform with translation
        final_transform = translation @ transforms[idx]
        
        # Add to canvas
        warped = cv2.warpPerspective(img, final_transform, (canvas_w, canvas_h))
        mask = warped > 0
        result_canvas[mask] = warped[mask]
        
        # Display
        axes[step].imshow(result_canvas.copy(), cmap='gray')
        axes[step].axis('off')
    
    # Hide unused subplots
    for i in range(n_steps, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Progressive Panorama Building (Graph Traversal Order)', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return result_canvas

def build_panorama_final(images, pairwise_data, graph):
    """Build final panorama without intermediate visualization"""
    if not pairwise_data:
        return None
    
    largest_cc = max(nx.connected_components(graph), key=len)
    if len(largest_cc) < 2:
        return None
    
    degrees = dict(graph.degree())
    ref_idx = max(largest_cc, key=lambda x: degrees[x])
    
    transforms = compute_absolute_transforms(
        ref_idx, largest_cc, pairwise_data, graph, images, debug=False)
    
    valid_indices = [idx for idx in largest_cc if idx in transforms]
    if len(valid_indices) < 2:
        return None, None, None
    
    panorama = stitch_images(images, transforms, ref_idx)
    return panorama, transforms, ref_idx

def display_original_images(images):
    """Display original input images"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, (img_num, img, _) in enumerate(images):
        if i < len(axes):
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Img {img_num}')
            axes[i].axis('off')
    
    for i in range(len(images), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Original Images')
    plt.tight_layout()
    plt.show()
