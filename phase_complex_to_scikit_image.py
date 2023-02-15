#! /usr/bin/env python
# -*- coding: utf-8 -*-
""" Unwrap 2D phase with different methods """
__copyright__ = "Copyright 2022: CREF, Centro Ricerche Enrico Fermi, www.cref.it"

from matplotlib.pyplot import colorbar, show, subplots, cm
import numpy as np
from skimage.restoration import unwrap_phase
from skimage import filters

measured_complex_field = np.load('data/measured_complex_field_2.npy')

# Let's look at the data
print('dim =', measured_complex_field.ndim)
print('shape =', measured_complex_field.shape)
print('dtype =', measured_complex_field.dtype)
print('first line =', measured_complex_field[0])
print('first number =', measured_complex_field[0][0])

# Compute the phase
measured_phase = np.angle(measured_complex_field)
print('first line =', measured_phase[0])
print('first number =', measured_phase[0][0])
print('min =', measured_phase.min())
print('max =', measured_phase.max())

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
show()
