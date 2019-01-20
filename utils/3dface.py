from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math

xx = np.arange(0,100,1)
yy = np.arange(0,100,1)
xx,yy = np.meshgrid(xx, yy)

z = np.random.randint(0,100,size = [100,100])
fig = plt.figure()
ax = Axes3D(fig)
x = np.random.rand(100)
y = np.random.rand(100)
r = np.sqrt(x**2 + y**2)

def face(x,y,z):
    x = np.round(x*100)
    y = np.round(y*100)
    # z = np.random.randint(0,100,1)
    return ax.plot_surface(x,y,z,rstride=1, cstride=1, cmap='rainbow'), ax.contourf(x, y, z, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow'))

a = face(x,y,z)
plt.show()
