import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from devito import *
from ultrawave import TimeAxis, Receiver
from ultrawave.lib.model_2d import Model

from functions import ToneBurstSource, plot_velocity, transducer_elements, Acoustic2DOperator, circle_mask, EqnBackward

spacing = (0.5e-3, 0.5e-3)
shape = (int(22e-2/spacing[0])+1, int(22e-2/spacing[1])+1)

origin = (0., 0.)  # The location of the top left corner.

vp_background = 1540 # [m/s]
rho_background = 1000 # [kg/m^3]

v0 = np.empty(shape, dtype=np.float32)
v0[:] = vp_background

rho0 = np.empty(shape, dtype=np.float32)
rho0[:] = rho_background

num_ele = 128
points = transducer_elements(num_ele, 100) * 1.e-3
points = points - np.array([[-11.e-2, -11.e-2]])

time_order = 2
space_order = 10
dt = 5.e-8 #[s]
nbl =40

model0 = Model(vp=v0, rho=rho0, origin=origin, shape=shape, spacing=spacing, space_order=space_order, dt=dt,  nbl=nbl, dtype=np.float64)

plot_velocity(model=model0, receiver=points, cmap='viridis')

t0 = 0.  # Simulation starts a t=0
tn = 260.e-6 # [s]

time_range = TimeAxis(start=t0, stop=tn, step=dt)
nt = time_range.num
time = np.linspace(t0, tn, time_range.num)

f0 = 0.4e6 # [Hz]

c0 = 1540
obs = np.load('brain_obs_data_2D_0p4MHz.npy')

fwi_iteration = 10
grad_mask = circle_mask(shape[0]+2*nbl, shape[1]+2*nbl, 90.e-3 // spacing[0], 3.e-3 // spacing[0])

batch_size = 4
for epoch in range(fwi_iteration):
    indices = np.random.permutation(num_ele)
    grad_new = np.zeros((shape[0] + 2 * nbl, shape[1] + 2 * nbl))
    for i in range(0, num_ele//batch_size):
        objective = 0.

        for jj in range(batch_size):
            j = indices[i*batch_size + jj]
            print(f'Epoch {epoch}, Batch {i}, Shot {j}')
            grad = Function(name="grad", grid=model0.grid)

            d_obs = np.transpose(np.squeeze(obs[j, :, :]))

            residual = Receiver(name='residual', grid=model0.grid, npoint=num_ele,
                                time_range=time_range)
            residual.coordinates.data[:, 0] = points[:, 0]
            residual.coordinates.data[:, 1] = points[:, 1]

            syn = Receiver(name='syn', grid=model0.grid, npoint=num_ele,
                           time_range=time_range)
            syn.coordinates.data[:, 0] = points[:, 0]
            syn.coordinates.data[:, 1] = points[:, 1]

            p = TimeFunction(name='p', grid=model0.grid, staggered=NODE, time_order=1, space_order=model0.space_order, save=nt)

            src0 = ToneBurstSource(name='src', grid=model0.grid, f0=f0, npoint=1, time_range=time_range)
            src0.coordinates.data[:, :] = points[j, :]

            op0, _ = Acoustic2DOperator(p=p, model=model0, source=src0, reciever=syn)
            op0.apply(dt=dt)

            residual.data[:] = syn.data[:] - d_obs[:]
            objective += .5 * norm(residual) ** 2

            u = TimeFunction(name='u', grid=model0.grid, time_order=2, space_order=model0.space_order)
            eqns_back = EqnBackward(model0, p=u)
            gradient_update = Inc(grad, -p * u.dt2)

            m = model0.m
            s = model0.grid.stepping_dim.spacing
            rec_grad = residual.inject(field=u.backward, expr=residual * s ** 2 / m)

            op_grad = Operator(eqns_back + rec_grad + [gradient_update], name='Gradient')

            op_grad.apply(time=time_range.num-200, dt=dt)

            print(f'Objective value for this shot :{.5 * norm(residual) ** 2}')

            grad_new += np.copy(grad_mask * grad.data)

            # plt.figure()
            # plt.imshow(grad_new[nbl:-nbl, nbl:-nbl])
            # plt.show()

        print(f'Objective value for this batch :{objective}')
        search_direction = -grad_new

        vp_old = np.copy(model0.vp.data)
        alpha = 0.02
        vp_new = vp_old + alpha * search_direction
        model0.vp.data[:] = vp_new

    plt.figure()
    plt.imshow(model0.vp.data[nbl:-nbl, nbl:-nbl])
    plt.show()