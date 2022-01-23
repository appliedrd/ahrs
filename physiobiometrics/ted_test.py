import numpy as np
import pytest
import scipy.io as sio
import pandas as pd
from math import pi
from ahrs.common.orientation import q2euler

import ahrs
import ahrs.utils.io

DEG2RAD = ahrs.common.DEG2RAD


class Data:
    acc = None
    gyr = None
    mag = None
    gyr_rads = None
    Q = None


def data():
    fn = "ExampleData.mat"
    mat = sio.loadmat(fn)
    d = Data()
    d.acc = mat["Accelerometer"]
    d.gyr = mat["Gyroscope"]
    d.mag = mat["Magnetometer"]
    d.num_samples = len(d.acc)
    assert d.num_samples
    assert len(d.acc[0]) == 3
    assert len(d.gyr[0]) == 3
    assert len(d.mag[0]) == 3
    return d


def check_integrity(Q):
    assert Q is not None
    sz = Q.shape
    qts_ok = not np.allclose(np.sum(Q, axis=0), sz[0] * np.array([1.0, 0.0, 0.0, 0.0]))
    qnm_ok = np.allclose(np.linalg.norm(Q, axis=1).mean(), 1.0)
    assert qts_ok and qnm_ok


def Q(data):
    q = np.zeros((data.num_samples, 4))
    q[:, 0] = 1.0
    return q

def test_distance():
    a = np.random.random((2, 3))
    d = ahrs.utils.metrics.euclidean(a[0], a[1])
    assert np.allclose(d, np.linalg.norm(a[0] - a[1]))

def dataframe_from_backGait(filename):
    d = Data()
    df = pd.read_csv(filename)
    df.columns = df.columns.str.replace(' ', '')
    print(df.columns)
    d.acc = df[['AccX(g)', 'AccY(g)', 'AccZ(g)']].to_numpy()
    d.gyr = df[['GyroX(deg/s)', 'GyroY(deg/s)', 'GyroZ(deg/s)']].to_numpy()
    d.gyr_rads = d.gyr * (pi / 180)
    d.mag = df[['MagX(uT)', 'MagY(uT)', 'MagZ(uT)']].to_numpy()
    return d

def dataframe_from_repoIMU(filename):
    d = Data()
    df = pd.read_csv(filename, sep = ';')
    df.columns = df.columns.str.replace(' ', '')
    print(df.columns)
    d.acc = df[['AX', 'AY', 'AZ']].to_numpy()
    d.gyr = df[['GX', 'GY', 'GZ']].to_numpy()
    d.gyr_rads = d.gyr * (pi / 180)
    d.mag = df[['MX', 'MY', 'MZ']].to_numpy()
    d.Q = df[['W', 'QX', 'QY', 'QZ']].to_numpy()
    return d

# https://ahrs.readthedocs.io/en/latest/
test_distance()
d = data()
q0=[1.0, 0.0, 0.0, 0.0]
#madgwick = ahrs.filters.Madgwick(gyr=d.gyr, acc=d.acc, mag=d.mag, q0=q0)

# imuData = dataframe_from_backGait("/home/ted/git/quaternions/ahrs/physiobiometrics/bgait.csv")
# madgwick = ahrs.filters.Madgwick(gyr=imuData.gyr, acc=imuData.acc, mag=imuData.mag, q0=q0)
# print("PROCESSED BGAIT DATA")

imuData = dataframe_from_repoIMU("/home/ted/git/quaternions/ahrs/physiobiometrics/repoIMU.csv")
madgwick = ahrs.filters.Madgwick(gyr=imuData.gyr, acc=imuData.acc, mag=imuData.mag, q0=q0)
eulerangles = q2euler(madgwick.Q[0])
print("PROCESSED repo IMU")


