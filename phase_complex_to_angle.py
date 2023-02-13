#! /usr/bin/env python
# -*- coding: utf-8 -*-
""" Just a first test """
__copyright__ = "Copyright 2022: CREF, Centro Ricerche Enrico Fermi, www.cref.it"

import matplotlib.pyplot as plt
import numpy as np
measured_complex_field = np.load('data/measured_complex_field.npy')

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

# Plot measured_phase 2D
# plt.imshow(measured_phase, cmap='viridis', interpolation='nearest')

fig, ax = plt.subplots()
# img = ax.imshow(measured_phase, cmap=plt.cm.Reds, interpolation='bicubic', origin='lower')
img = ax.imshow(measured_phase, cmap=plt.cm.tab20c, interpolation='nearest', origin='lower')
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')
bar = plt.colorbar(img)
bar.set_label('Phase')
plt.show()

# Plot horizontal cut
# x_hor = np.arange(1920)
# plt.plot(x_hor, measured_phase[600])
# plt.show()

# Unwrap phase
unwrapped_phase = np.unwrap(4*measured_phase[600])/4
x_hor = np.arange(1920)
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Forcing unwrapping')
ax1.plot(x_hor, measured_phase[600])
ax2.plot(x_hor, unwrapped_phase)
plt.show()
