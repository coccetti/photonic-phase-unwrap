#! /usr/bin/env python
# -*- coding: utf-8 -*-
""" Compute the phase complex filed from measured intensity """
__copyright__ = "Copyright 2022: CREF, Centro Ricerche Enrico Fermi, www.cref.it"

import numpy as np

# Load data files
frames = np.load('data/I_allframes.npy')
phases = np.load('data/allphases.npy') / np.pi
frames = frames.astype(np.float32)
print('Phases array =', phases)

# Save complex field
phi = frames[:,:,[0,2,4,6]]
print(phi.dtype)
complex_field = (phi[:,:,[0]] - phi[:,:,[2]]) + 1j * (phi[:,:,[1]] - phi[:,:,[3]])
np.save('data/measured_complex_field_2.npy', complex_field)
