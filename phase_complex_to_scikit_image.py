#! /usr/bin/env python
# -*- coding: utf-8 -*-
""" Unwrap 2D phase with different methods """
__copyright__ = "Copyright 2022: CREF, Centro Ricerche Enrico Fermi, www.cref.it"

from matplotlib.pyplot import colorbar, show, subplots, cm
import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import unwrap_phase
from skimage import filters

measured_complex_field = np.load('data/measured_complex_field_2.npy')

# Let's look at the data
print('measured_complex_field:')
print('\tdim =', measured_complex_field.ndim)
print('\tshape =', measured_complex_field.shape)
print('\tdtype =', measured_complex_field.dtype)
print('\tfirst line =', measured_complex_field[0])
print('\tfirst number =', measured_complex_field[0][0])

measured_complex_field = measured_complex_field.reshape(1200, 1920)
print('measured_complex_field:')
print('\tdim =', measured_complex_field.ndim)
print('\tshape =', measured_complex_field.shape)
print('\tdtype =', measured_complex_field.dtype)
print('\tfirst line =', measured_complex_field[0])
print('\tfirst number =', measured_complex_field[0][0])

# Compute the phase
measured_phase = np.angle(measured_complex_field)
print('measured_phase:')
print('\tfirst line =', measured_phase[0])
print('\tfirst number =', measured_phase[0][0])
print('\tmin =', measured_phase.min())
print('\tmax =', measured_phase.max())

# Denoise with a Gaussian filter
measured_phase_denoise = filters.gaussian(measured_phase, sigma=0.4)

# Perform phase unwrapping
# 1D
unwrapped_phase = np.unwrap(measured_phase[600], discont=np.pi, axis=0)
# 2D
measured_phase_unwrapped_skimage = unwrap_phase(measured_phase)
measured_phase_denoise_unwrapped_skimage = unwrap_phase(measured_phase_denoise)

# Plot measured_phase 2D
fig, ((ax1, ax2), (ax3, ax4)) = subplots(2, 2)
img1 = ax1.imshow(measured_phase, cmap=cm.tab20c, interpolation='nearest', origin='lower')
img2 = ax2.imshow(measured_phase_unwrapped_skimage, cmap=cm.tab20c, interpolation='nearest', origin='lower')
img3 = ax3.imshow(measured_phase_denoise, cmap=cm.tab20c, interpolation='nearest', origin='lower')
img4 = ax4.imshow(measured_phase_denoise_unwrapped_skimage, cmap=cm.tab20c, interpolation='nearest', origin='lower')

bar1 = colorbar(img1)
bar2 = colorbar(img2)
bar3 = colorbar(img3)
bar4 = colorbar(img4)

ax1.set_title('Measured Phase')
ax2.set_title('Measured Phase Unwrapped')
ax3.set_title('Measured Phase Denoise')
ax4.set_title('Measured Phase Denoise Unwrapped')

fig.tight_layout(pad=1.0)
# plt.show()

# Plot 1D Phases
y_ver = np.arange(1200)
x_hor = np.arange(1920)
fig, (ax1, ax2, ax3, ax4, ax5) = subplots(5)
fig.suptitle('Plot 1D Phases')

ax1.set_title('Measured Phase 1D horizontal cut')
ax1.plot(x_hor, measured_phase[600])

ax2.set_title('Measured Phase 1D Gaussian denoise')
ax2.plot(x_hor, measured_phase_denoise[600])

ax3.set_title('Measured Phase Unwrapped with np.angle')
ax3.plot(x_hor, unwrapped_phase)

ax4.set_title('Measured Phase Unwrapped with Scikit-image 2D')
ax4.plot(x_hor, measured_phase_unwrapped_skimage[600])

ax5.set_title('Measured Phase denoise Unwrapped')
ax5.plot(x_hor, measured_phase_denoise_unwrapped_skimage[600])

fig.tight_layout(pad=1.0)
# show()

# Plot Phase in 3D
X, Y = np.meshgrid(x_hor, y_ver)
fig = plt.figure()
ax6 = fig.add_subplot(111, projection='3d')
# ax6.plot_wireframe(X, Y, measured_phase_unwrapped_skimage, rstride=100, cstride=100, linewidth=0.2)
img_3D = ax6.plot_surface(X, Y, measured_phase_denoise_unwrapped_skimage, cmap=cm.hsv)
bar_3D = colorbar(img_3D)
bar_3D.set_label('Phase (rad)')
ax6.set_xlabel('Pixel X')
ax6.set_ylabel('Pixel Y')
ax6.set_zlabel('Phase (rad)')
# ax6.axes.set_xlim3d(left=100, right=1800)
# ax6.axes.set_ylim3d(bottom=0, top=1199)
ax6.axes.set_zlim3d(bottom=-2*np.pi, top=2*np.pi)
plt.show()
