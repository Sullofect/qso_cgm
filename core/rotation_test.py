import numpy as np
import matplotlib.pyplot as plt

def coord_rotation(theta):
    # Convert to radians
    # if theta[0] == 0.0: theta = [0.001, theta[1], theta[2]]
    # if theta[1] == 0.0: theta = [theta[0], 0.001, theta[2]]
    # if theta[2] == 0.0: theta = [theta[0], theta[1], 0.001]
    #
    # if theta[0] == 90.0: theta = [89.9, theta[1], theta[2]]
    # if theta[1] == 90.0: theta = [theta[0], 89.9, theta[2]]
    # if theta[2] == 90.0: theta = [theta[0], theta[1], 89.9]
    #
    # if theta[0] == -90.0: theta = [-89.9, theta[1], theta[2]]
    # if theta[1] == -90.0: theta = [theta[0], -89.9, theta[2]]
    # if theta[2] == -90.0: theta = [theta[0], theta[1], -89.9]

    theta_1_rad = theta[0] * np.pi / 180.0
    theta_2_rad = theta[1] * np.pi / 180.0
    theta_3_rad = theta[2] * np.pi / 180.0
    # The bicone and dust angles correspond to Euler angles which are
    # (e1,e2,e3) -> (rotation about z, rotation about x, rotation about z again)
    theta_1, theta_2, theta_3 = theta_1_rad, theta_2_rad, theta_3_rad
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta_1), -np.sin(theta_1)],
                    [0, np.sin(theta_1), np.cos(theta_1)]])
    R_y = np.array([[np.cos(theta_2), 0, np.sin(theta_2)],
                    [0, 1, 0],
                    [-np.sin(theta_2), 0, np.cos(theta_2)]])
    R_z = np.array([[np.cos(theta_3), -np.sin(theta_3), 0],
                    [np.sin(theta_3), np.cos(theta_3), 0],
                    [0, 0, 1]])
    # R  = np.dot(R_z, np.dot( R_y, R_x )) #np.dot(R_z, np.dot( R_y, R_x ))
    # RR = np.dot(R_x, np.dot( R_y, R_z ))
    # R = np.dot(R_z, np.dot(R_x, R_y))
    RR = np.dot(R_y, np.dot(R_x, R_z))  # extrinsic rotation
    return RR



v = np.array([0, 2, 4])
theta_1 = np.array([0, 0, 90])
v1 = np.dot(coord_rotation(theta_1), v)
theta_2 = np.array([0, 90, 90])
v2 = np.dot(coord_rotation(theta_2), v)
theta_3 = np.array([30, 45, 45])
v3 = np.dot(coord_rotation(theta_3), v)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(0, 0, 0, v[0], v[1], v[2], color='r')
ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='b')
ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='g')
ax.quiver(0, 0, 0, v3[0], v3[1], v3[2], color='y')
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 5])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(azim=45, elev=45)
plt.show()