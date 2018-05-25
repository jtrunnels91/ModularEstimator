import numpy as np
from pyquaternion import Quaternion
import numpy as _np


def euler2quaternion(eulerAngles):
    roll = eulerAngles[0]
    pitch = eulerAngles[1]
    yaw = eulerAngles[2]
    
    cy = _np.cos(yaw * 0.5)
    sy = _np.sin(yaw * 0.5)
    cr = _np.cos(roll * 0.5)
    sr = _np.sin(roll * 0.5)
    cp = _np.cos(pitch * 0.5)
    sp = _np.sin(pitch * 0.5)

    w = cy * cr * cp + sy * sr * sp
    x = cy * sr * cp - sy * cr * sp
    y = cy * cr * sp + sy * sr * cp
    z = sy * cr * cp - cy * sr * sp
    return Quaternion([w,x,y,z])


def quaternion2euler(q):
    if isinstance(q,Quaternion):
        q = q.q

    phi = _np.arctan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(_np.square(q[1]) + _np.square(q[2])))
    theta = _np.arcsin(2 * ((q[0] * q[2]) - (q[3] * q[1])))
    psi = _np.arctan2(2 * (q[0] * q[3] + q[1]*q[2]), 1 - 2*(_np.square(q[2]) + _np.square(q[3])))

    return [phi, theta, psi]


def eulerAngleDiff(angle1, angle2):
    if hasattr(angle1, "__len__"):
        angleDiff = angle1 * 0
        for angleIt in range(len(angle1)):
            angleDiff[angleIt] = eulerAngleDiff(angle1[angleIt], angle2[angleIt])
    else:
        angleDiff = angle1 - angle2
        if angleDiff > np.pi:
            angleDiff = angleDiff - (2*np.pi)
        elif angleDiff < - np.pi:
            angleDiff = (2*np.pi) + angleDiff
        
    return angleDiff
