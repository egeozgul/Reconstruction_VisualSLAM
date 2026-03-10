# Image Stitching with Covariance-Weighted MST and GTSAM


A 2D image stitching pipeline that assembles overlapping images into a globally consistent panorama. The pipeline combines feature-based matching, robust homography estimation, covariance-weighted graph construction, and GTSAM-based bundle adjustment to minimize accumulated drift across long image sequences.

# Overview

Stitching multiple images into a coherent panorama is straightforward for short sequences, but pairwise chaining of homographies accumulates geometric error quickly — small misalignments compound with each added image, producing visible drift and seam artifacts in the final mosaic.
This project addresses that problem at two levels:

## Graph construction

Rather than chaining images in a fixed order, all pairwise overlaps are computed and organized into a covariance-weighted Minimum Spanning Tree (MST). Each edge weight reflects the geometric uncertainty of its homography estimate, so the MST selects the most reliable connections to form the stitching backbone. Unstable or poorly-conditioned matches are naturally deprioritized.

## Global refinement

The MST gives a good initial structure, but residual errors remain. A GTSAM pose graph then treats each image as a 2D node and each homography as a relative pose constraint. Critically, images that are spatially close but far apart in acquisition order — loop closures — are detected and added as additional constraints. These cross-links break the open chain topology that causes drift, and bundle adjustment then jointly optimizes all image positions against the full constraint set, distributing error globally.

---

## Dataset

The underwater imagery used in this project comes from the Skerki D wreck dataset, collected at Skerki Bank in the Mediterranean Sea using the Jason ROV by the Woods Hole Oceanographic Institution (WHOI), and originally presented in Pizarro and Singh (2003) in the IEEE Journal of Oceanic Engineering.

### Sample Images from the dataset
<img width="679" height="612" alt="mosaic" src="https://github.com/user-attachments/assets/bcc1fdb5-139f-4dd8-b6e1-b4ea7597ab92" />

## Technical Details

### Pipeline Overview

1. **Image loading** — Load `.tif` images from the `29images` folder (or download a sample set from the linked repository). Images are sorted for consistent ordering.
2. **Feature detection & matching** — **SIFT** features with **FLANN** matching and Lowe’s ratio test to get putative correspondences between all image pairs.
3. **Homography estimation** — For each pair:
   - **RANSAC** for initial inlier set and outlier rejection.
   - **Levenberg–Marquardt** refinement on inliers.
   - **Covariance** estimation from the LM solution (JᵀJ⁻¹) for uncertainty-aware edge weighting.
4. **Covariance-weighted MST** — Build a **maximum spanning tree** where edge weights are derived from homography quality (inverse covariance). This selects a connected set of reliable pairwise links.
5. **Absolute transforms** — **BFS** from a reference image to accumulate absolute homographies for every image in the MST.
6. **Panorama building** — Warp all images into a common frame with **OpenCV** `warpPerspective`, then blend onto a canvas (e.g. max or average).
7. **GTSAM 2D optimization** — Model image poses as 2D positions; add **between-factor** constraints from pairwise homographies and **prior** on the reference. Optimize with GTSAM to get globally consistent positions.
8. **Updated homographies** — Convert optimized 2D positions back into updated absolute homographies and build a **GTSAM-optimized panorama** for comparison.

### Main Algorithms & Libraries

| Component            | Method / library |
|---------------------|------------------|
| Features            | **SIFT** (OpenCV) |
| Matching            | **FLANN** (OpenCV), Lowe’s ratio test |
| Homography          | **RANSAC** + **Levenberg–Marquardt** (SciPy `least_squares`) |
| Covariance          | From LM Jacobian: σ²(JᵀJ + εI)⁻¹ |
| Graph / MST         | **NetworkX** (covariance-weighted MST) |
| Global optimization | **GTSAM** (NonlinearFactorGraph, 2D poses, Loop Closures) |
| Visualization       | **Matplotlib** |
| Image I/O & warp    | **OpenCV** (cv2) |

### Key Parameters (from `StitchingParameters`)

- **Matching:** `MIN_MATCHES = 15`, `MIN_INLIER_RATIO = 0.2`, `SIFT_RATIO_THRESHOLD = 0.7`
- **RANSAC:** `RANSAC_REPROJ_THRESHOLD = 5.0` px, `RANSAC_CONFIDENCE = 0.99`, `RANSAC_MAX_ITERS = 2000`
- **LM:** `LM_MAX_NFEV = 1000`, `LM_FTOL = LM_XTOL = 1e-8`
- **GTSAM:** prior sigma, max iterations, and convergence tolerances for 2D position optimization

---

## The pipeline in pictures

To connect the images we first have to find the same points in both. The pipeline uses **SIFT** to detect keypoints and build descriptors that stay stable across scale and rotation; then for every pair of images it **matches** those descriptors with **FLANN**, keeps only the clear correspondences with **Lowe’s ratio test**, and fits a **homography** with **RANSAC** and **Levenberg–Marquardt** so we get a geometric transform and can throw away outliers. Only pairs with enough good matches and a valid homography count as connected. Not every pair is equally reliable, so we need a way to see which images actually connect and how strong those connections are.

### Nomralizing images using CLAHE
<img width="1189" height="935" alt="download" src="https://github.com/user-attachments/assets/5164ca8e-f4d6-4b8b-9be9-8845689c23ed" />

## Keypoint detection with SIFT
<img width="2390" height="2313" alt="SIFT" src="https://github.com/user-attachments/assets/445f911f-9e77-4146-a5b1-a247f1f67297" />

Here each image is a node, and an edge means we found enough good matches and a valid homography between those two images. This is the **full connection graph**—all pairwise links before we decide which ones to use. Where the graph is dense, many views overlap; where it’s sparse or disconnected, we know we have to be careful. From this we don’t yet know *which* path through the images to use for building the panorama.

So we pick a single, clean path: a **covariance-weighted maximum spanning tree (MST)**. We prefer edges where the homography is more certain (lower covariance). That gives us one tree that touches every image with no cycles—exactly the backbone we need to chain transforms from a reference image to all others and keep drift under control.


This is that tree (the **factor graph**). Each edge is a link we trust enough to use when building the panorama.

We can also see how **GTSAM** refines the 2D positions: the factor graph before and after optimization shows how the global bundle adjustment pulls the image poses into a more consistent layout.

## Minimum Spanning Tree (MST) Factor Graph
<img width="1555" height="1589" alt="download" src="https://github.com/user-attachments/assets/28c1c3d4-3b48-4b8c-8f46-fbb01cd71d05" />

## Factor Graph Before After GTSAM Loop Closure Optimization 
<img width="2354" height="1181" alt="download" src="https://github.com/user-attachments/assets/a6c41948-3a31-4074-9f94-93f94f83a363" />

Once we’re happy with the links, we build the panorama by adding images one at a time in the order given by the tree (e.g. a BFS from the reference). The **progressive panorama** below shows the canvas after each new image is added. You can see the mosaic grow and spot early drift or blending issues before any global optimization.

## Iterative Panorama Buildup - Normalized
<img width="1615" height="3961" alt="download" src="https://github.com/user-attachments/assets/b765ac34-aa53-413e-9768-c488652a27b8" />

Finally we warp every image into one common frame and blend them.

## Full Panorama Before After GTSAM Optimization - Normalized
<img width="1489" height="785" alt="download" src="https://github.com/user-attachments/assets/64038169-f683-4591-a173-2ad8cb9281a4" />

## Full Panorama Before After GTSAM Optimization
<img width="1489" height="785" alt="download" src="https://github.com/user-attachments/assets/a02de77e-9436-491e-b19a-1b9e782a1052" />

---

## Summary

This project implements a **production-style image stitching pipeline** with:

- **SIFT + FLANN** for matching  
- **RANSAC + LM** homography estimation with **covariance**  
- **Covariance-weighted MST** for robust connection selection  
- **GTSAM** for global 2D position optimization using loop closures  
- **OpenCV** for warping and panorama assembly  

The README documents the **goal** (consistent panorama + global optimization), **technical details** (algorithms and parameters), and **visualizations** from the `Figures/` folder.
