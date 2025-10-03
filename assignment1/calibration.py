import cv2
import numpy as np
import os

# Function to display corners extracted from the image
def show_img_corners(img, corners):
    corners = np.int32(corners)
    for idx, corner in enumerate(corners):
        x, y = corner.ravel()  # Extract coordinates
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Draw a red dot, thickness -1 fills the inside
        cv2.putText(img, str(idx), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # resized_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    
    # View the result image
    cv2.imshow('Corners', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()   

# Function to find the Homography matrix of the image
def find_homography(img_coord):
    n = img_coord.shape[0]
    Q = []
    for i in range(n):
        X, Y = world_coord[i][:2]
        u, v = img_coord[i]
        Q.append([X, Y, 1, 0, 0, 0, -u*X, -u*Y, -u])
        Q.append([0, 0, 0, X, Y, 1, -v*X, -v*Y, -v])
    Q = np.array(Q)
    _, _, VT = np.linalg.svd(Q)
    h = VT[-1]
    h = h / h[-1]
    H = np.reshape(h, (3, 3))
    return H

# Function to find the linear equation that constitutes the linear system to find the b vector 
def a_ij(H, i, j):
    return np.array([
        H[0, i] * H[0, j],
        H[0, i] * H[1, j] + H[1, i] * H[0, j],
        H[1, i] * H[1, j],
        H[2, i] * H[0, j] + H[0, i] * H[2, j],
        H[2, i] * H[1, j] + H[1, i] * H[2, j],
        H[2, i] * H[2, j]
    ])

# Function to convert vector b to matrix B
def convert_b_to_B(b):
    B11, B12, B22, B13, B23, B33 = b
    B = np.array([[B11, B12, B13],
                  [B12, B22, B23],
                  [B13, B23, B33]])
    return B

# Function to find R and T using K
def get_extrinsics(K):
    K_inv = np.linalg.inv(K)
    extrinsics = []
    for H in homographies:
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = H[:, 2]

        lambd = 1 / np.linalg.norm(K_inv @ h1)
        r1 = lambd * K_inv @ h1
        r2 = lambd * K_inv @ h2
        t = lambd * K_inv @ h3

        r3 = np.cross(r1, r2)

        R = np.array([r1, r2, r3]).T
        R_t = np.hstack((R, t.reshape(3,1)))
        extrinsics.append(R_t)
    return extrinsics

# Function to visualize reprojection, calculate and save errors
def visualize_reprojection(title, img_paths, obj_points, img_points, K, dist, extrinsics):
    per_image_rmse = []
    for i in range(len(img_paths)):
        img = cv2.imread(img_paths[i])
        
        objp = obj_points[i]
        imgp = img_points[i]
        
        R = extrinsics[i][:, :3]
        t = extrinsics[i][:, 3]
        rvec, _ = cv2.Rodrigues(R)
        
        reprojected_points, _ = cv2.projectPoints(objp, rvec, t, K, dist)
        
        # Calculate reprojection error (RMSE) for the current image
        error = np.sqrt(np.sum((imgp - reprojected_points)**2) / len(objp))
        per_image_rmse.append(error)
        
        # Display error value on the image
        cv2.putText(img, f"RMSE: {error:.4f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Draw original detected corners (green circles) and display numbers
        for idx, point in enumerate(np.int32(imgp)):
            x, y = point.ravel()
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(img, str(idx), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
        # Draw reprojected corners (red circles) and display numbers
        for idx, point in enumerate(np.int32(reprojected_points)):
            x, y = point.ravel()
            cv2.circle(img, (x, y), 5, (0, 0, 255), 2)
            cv2.putText(img, str(idx), (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
        # View the result image
        cv2.imshow(f"{title} - Image {i+1}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return per_image_rmse

# Function to visualize lens distortion corrected photos
def visualize_undistortion(img_paths, K, dist):
    for i, img_path in enumerate(img_paths):
        img_original = cv2.imread(img_path)
        h, w = img_original.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
        img_undistorted = cv2.undistort(img_original, K, dist, None, new_camera_matrix)
        
        comparison_image = np.hstack((img_original, img_undistorted))

        cv2.imshow(f'Original vs Undistorted {i}', comparison_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


img_dir = './imgs'
img_list = os.listdir(img_dir)
img_paths = [os.path.join(img_dir, file_name) for file_name in img_list]

pattern_size = (13, 9)
square_len = 2.0
world_coord = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
world_coord[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_len
homographies = []

obj_points = [] # 3D points
img_points = [] # 2D points
gray = None

np.set_printoptions(precision=3, suppress=True) # Set to express results in more concise decimals

for i in range(len(img_paths)):
    # Extract corner coordinates from 2D image
    img = cv2.imread(img_paths[i])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size)

    if ret:
        obj_points.append(world_coord)
        
        # Improve corner coordinate accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        img_points.append(corners2)

        # show_img_corners(img, corners2)

        # Use refined corners for Homography calculation
        H = find_homography(corners2.reshape(-1, 2))
        homographies.append(H)

A = []
for H in homographies: # Create matrix A to solve Ab = 0
    A.append(a_ij(H, 0, 1))
    A.append(a_ij(H, 0, 0) - a_ij(H, 1, 1))

A = np.array(A) # (2N, 6) matrix
_, _, VT = np.linalg.svd(A)
b = VT[-1]

if b[-1] < 0: # The sign of the solution to Ab = 0 must be positive for B to be positive definite, allowing Cholesky decomposition
    b = -b
B = convert_b_to_B(b)

# Find K using Cholesky Decomposition
B_inv = np.linalg.inv(B)
K_cd = np.linalg.cholesky(B_inv).T
K_cd = K_cd / K_cd[2, 2]


# Directly calculate K's parameters from each element of matrix B
cy = (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2]) / (B[0, 0] * B[1, 1] - B[0, 1]**2)
lambda_val = B[2, 2] - (B[0, 2]**2 + cy * (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2])) / B[0, 0]
fx = np.sqrt(lambda_val / B[0, 0])
fy = np.sqrt(lambda_val * B[0, 0] / (B[0, 0] * B[1, 1] - B[0, 1]**2))
gamma = -B[0, 1] * fx**2 * fy / lambda_val
cx = gamma * cy / fx - B[0, 2] * fx**2 / lambda_val

K = np.array([[fx, gamma, cx],
              [0,  fy,    cy],
              [0,  0,     1]])

extrinsics = get_extrinsics(K)
extrinsics_cd = get_extrinsics(K_cd)

# Using cv2.calibrateCamera()
ret, K_cv, dist_cv, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

extrinsics_cv = []
for i in range(len(rvecs)):
    R_cv, _ = cv2.Rodrigues(rvecs[i])
    t_cv = tvecs[i]

    extrinsics_cv.append(np.hstack((R_cv, t_cv)))

# comparison
print("my intrinsic Matrix (K)")
print(K)
print("my intrinsic Matrix (K_cd)")
print(K_cd)
print("cv2 intrinsic matrix (K_cv)")
print(K_cv)
print("cv2 distortion parameters")
print(dist_cv)

for i in range(len(extrinsics)):
    print(f"for image {i}:")
    print("my extrinsic Matrix (R | t)")
    print(extrinsics[i])
    print("my extrinsic Matrix (R | t) from K_cd")
    print(extrinsics_cd[i])
    print("cv2 extrinsic Matrix (R | t)")
    print(extrinsics_cv[i])
    print()


# 1. Visualize reprojection with directly calculated parameters (assuming no distortion coefficients)
print("\nVisualizing my reprojection results...")
my_errors = visualize_reprojection("My Reprojection", img_paths, obj_points, img_points, K, np.zeros((5, 1)), extrinsics)

# 2. Visualize reprojection with parameters from Cholesky Decomposition (assuming no distortion coefficients)
print("\nVisualizing my reprojection results... from K_cd")
my_errors_cd = visualize_reprojection("My Reprojection", img_paths, obj_points, img_points, K_cd, np.zeros((5, 1)), extrinsics_cd)

# 3. Visualize reprojection with cv2.calibrateCamera() results
print("\nVisualizing cv2's reprojection results...")
cv2_errors =  visualize_reprojection("CV2 Reprojection", img_paths, obj_points, img_points, K_cv, dist_cv, extrinsics_cv)

# Create and print (N, 3) error array
errors = np.array([my_errors, cv2_errors, my_errors_cd]).T
print("\n--- Per-Image Reprojection Errors (RMSE) ---")
print("Column 1: My Errors | Column 2: CV2 Errors | Column 3: My Errors with Cd")
print(errors)

# Print total average error
mean_my_error = np.mean(errors[:, 0])
mean_cv2_error = np.mean(errors[:, 1])
mean_my_error_cd = np.mean(errors[:, 2])
print(f"\nMy Mean Reprojection Error: {mean_my_error}")
print(f"CV2 Mean Reprojection Error: {mean_cv2_error}")
print(f"My Mean Reprojection Error with Cd: {mean_my_error_cd}")

# visualize_undistortion(img_paths, K_cv, dist_cv)
# visualize_undistortion(img_paths, K, np.zeros((5, 1)))