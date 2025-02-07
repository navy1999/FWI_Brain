import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ultrawave.lib.source import WaveletSource
from devito import *

class ToneBurstSource(WaveletSource):
    @property
    def wavelet(self):
        # transmit frequency response
        bw = 0.8  # bandwith
        bwr = -6  # [dB] Freq cutoff
        tpr = -100  # [dB]
        #tc = signal.gausspulse('cutoff', self.f0*1e3, bw, bwr, tpr)
        tc = signal.gausspulse('cutoff', self.f0, bw, bwr, tpr)

        #tt = np.linspace(-1 * tc, tc, int(2*tc/(self.time_range.step*1e-3)))
        #impulse_response = signal.gausspulse(tt, self.f0*1e3, bw, bwr, tpr)
        tt = np.linspace(-1 * tc, tc, int(2*tc/(self.time_range.step)))
        impulse_response = signal.gausspulse(tt, self.f0, bw, bwr, tpr)
        # source signal
        t_arr = np.zeros(self.time_range.num)
        t_arr[0] = 1
        src_p = signal.convolve(t_arr, impulse_response)
        src_p = src_p[:self.time_range.num]

        return src_p

def plot_velocity(model, source=None, receiver=None, colorbar=True, cmap="jet"):
    """
    Plot a two-dimensional velocity field from a seismic `Model`
    object. Optionally also includes point markers for sources and receivers.

    Parameters
    ----------
    model : Model
        Object that holds the velocity model.
    source : array_like or float
        Coordinates of the source point.
    receiver : array_like or float
        Coordinates of the receiver points.
    colorbar : bool
        Option to plot the colorbar.
    """
    domain_size = 1.e3 * np.array(model.domain_size)
    extent = [model.origin[0], model.origin[0] + domain_size[0],
              model.origin[1] + domain_size[1], model.origin[1]]

    slices = tuple(slice(model.nbl, -model.nbl) for _ in range(2))
    if getattr(model, 'vp', None) is not None:
        field = model.vp.data[slices]
    else:
        field = model.lam.data[slices]
    plot = plt.imshow(field, animated=True, cmap=cmap, #np.transpose(field)
                      vmin=np.min(field), vmax=np.max(field),
                      extent=extent)
    plt.xlabel('X position (mm)')
    plt.ylabel('Depth (mm)')

    # Plot source points, if provided
    if receiver is not None:
        plt.scatter(1e3*receiver[:, 0], 1e3*receiver[:, 1],
                    s=25, c='green', marker='D') #25 'D'

    # Plot receiver points, if provided
    if source is not None:
        plt.scatter(1e3*source[:, 0], 1e3*source[:, 1],
                    s=25, c='red', marker='o')

    # Ensure axis limits
    plt.xlim(model.origin[0], model.origin[0] + domain_size[0])
    plt.ylim(model.origin[1] + domain_size[1], model.origin[1])

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(plot, cax=cax)
        cbar.set_label('Speed of sound (m/s)')
    # plt.savefig('brain_w_skull_setting.png', dpi=300, bbox_inches="tight")
    plt.show()

def transducer_elements(num_elem=128, radius=100):
    coordinates = []
    for j in range(num_elem):
        phi = 2 * np.pi * j / num_elem  # Azimuthal angle

        x = radius * np.cos(phi)
        y = radius * np.sin(phi)

        # Append the coordinates to the list
        coordinates.append((x, y))

    coordinates = np.array(coordinates)
    # print(np.max(coordinates[:,2]))
    return coordinates

def Acoustic2DOperator(model, p=None, source=None, reciever=None, reciever2=None, save=None, nt=None):
    x, y = model.grid.dimensions

    p = p or TimeFunction(name='p', grid=model.grid, staggered=NODE, time_order=1, space_order=model.space_order, save=nt if save else None)
    v1x = TimeFunction(name='vx', grid=model.grid, time_order=1, space_order=model.space_order, staggered=x)
    v1y = TimeFunction(name='vy', grid=model.grid, time_order=1, space_order=model.space_order, staggered=y)
    rho1x = TimeFunction(name='rhox', grid=model.grid, staggered=NODE, time_order=1, space_order=model.space_order)
    rho1y = TimeFunction(name='rhoy', grid=model.grid, staggered=NODE, time_order=1, space_order=model.space_order)

    dt = model.critical_dt
    indices = np.floor(source.coordinates.data[0,:] / model.spacing).astype(int)
    c0 = model.vp.data[indices[0], indices[1]] # sound speed of the first source point
    src_rhox = source.inject(field=rho1x.forward, expr=source * (2 * dt / (2 * c0 * model.spacing[0])))
    src_rhoy = source.inject(field=rho1y.forward, expr=source * (2 * dt / (2 * c0 * model.spacing[0])))
    rec_term = reciever.interpolate(expr=p)
    if reciever2:
        rec_term2 = reciever2.interpolate(expr=p)

    p_dx = getattr(p, 'd%s' % p.space_dimensions[0].name)
    p_dy = getattr(p, 'd%s' % p.space_dimensions[1].name)
    vx_dx = getattr(v1x.forward, 'd%s' % v1x.space_dimensions[0].name)
    vy_dy = getattr(v1y.forward, 'd%s' % v1y.space_dimensions[1].name)

    eq_v_x = Eq(v1x.forward, v1x - dt * p_dx / model.rho, subdomain=model.grid.subdomains['main'])
    eq_v_y = Eq(v1y.forward, v1y - dt * p_dy / model.rho, subdomain=model.grid.subdomains['main'])
    eq_rho_x = Eq(rho1x.forward, rho1x - dt * model.rho * vx_dx, subdomain=model.grid.subdomains['main'])
    eq_rho_y = Eq(rho1y.forward, rho1y - dt * model.rho * vy_dy, subdomain=model.grid.subdomains['main'])
    eq_p = Eq(p.forward, model.vp * model.vp * (rho1x.forward + rho1y.forward))  # , subdomain=grid.subdomains['main'])

    alpha_max = 2 * 1540 / model.spacing[0]

    # Damping parameterisation
    d_l = (1 - x / model.nbl) ** 4  # Left side
    d_r = (1 - (model.grid.shape[0] - 1 - x) / model.nbl) ** 4  # Right side
    d_t = (1 - y / model.nbl) ** 4  # Top side
    d_b = (1 - (model.grid.shape[1] - 1 - y) / model.nbl) ** 4  # Base edge
    # staggered
    d_l_s = (1 - (x + 0.5) / model.nbl) ** 4  # Left side
    d_r_s = (1 - (model.grid.shape[0] - 1 - (x + 0.5)) / model.nbl) ** 4  # Right side
    d_t_s = (1 - (y + 0.5) / model.nbl) ** 4  # Top side
    d_b_s = (1 - (model.grid.shape[1] - 1 - (y + 0.5)) / model.nbl) ** 4  # Base edge

    # for the PML domain
    eq_v_damp_left_x = Eq(v1x.forward, (1 - dt * d_l_s * alpha_max) * v1x - dt * p_dx / model.rho,
                          subdomain=model.grid.subdomains['left'])
    eq_rho_damp_left_x = Eq(rho1x.forward, (1 - dt * d_l * alpha_max) * rho1x - dt * model.rho * vx_dx,
                            subdomain=model.grid.subdomains['left'])

    eq_v_damp_right_x = Eq(v1x.forward, (1 - dt * d_r_s * alpha_max) * v1x - dt * p_dx / model.rho,
                           subdomain=model.grid.subdomains['right'])
    eq_rho_damp_right_x = Eq(rho1x.forward, (1 - dt * d_r * alpha_max) * rho1x - dt * model.rho * vx_dx,
                             subdomain=model.grid.subdomains['right'])

    eq_v_damp_top_y = Eq(v1y.forward, (1 - dt * d_t_s * alpha_max) * v1y - dt * p_dy / model.rho,
                         subdomain=model.grid.subdomains['top'])
    eq_rho_damp_top_y = Eq(rho1y.forward, (1 - dt * d_t * alpha_max) * rho1y - dt * model.rho * vy_dy,
                           subdomain=model.grid.subdomains['top'])

    eq_v_damp_base_y = Eq(v1y.forward, (1 - dt * d_b_s * alpha_max) * v1y - dt * p_dy / model.rho,
                          subdomain=model.grid.subdomains['base'])
    eq_rho_damp_base_y = Eq(rho1y.forward, (1 - dt * d_b * alpha_max) * rho1y - dt * model.rho * vy_dy,
                            subdomain=model.grid.subdomains['base'])

    eqns = [eq_v_x, eq_v_y, eq_rho_x, eq_rho_y, eq_p, eq_v_damp_left_x, eq_rho_damp_left_x,
                   eq_v_damp_right_x, eq_rho_damp_right_x,
                   eq_v_damp_top_y, eq_rho_damp_top_y,
                   eq_v_damp_base_y, eq_rho_damp_base_y]

    if reciever2:
        op = Operator([eq_v_x, eq_v_y, eq_rho_x, eq_rho_y, eq_p, eq_v_damp_left_x, eq_rho_damp_left_x,
                       eq_v_damp_right_x, eq_rho_damp_right_x,
                       eq_v_damp_top_y, eq_rho_damp_top_y,
                       eq_v_damp_base_y, eq_rho_damp_base_y] + src_rhox + src_rhoy + rec_term + rec_term2, name='Forward')
    else:
        op = Operator([eq_v_x, eq_v_y, eq_rho_x, eq_rho_y, eq_p, eq_v_damp_left_x, eq_rho_damp_left_x,
                       eq_v_damp_right_x, eq_rho_damp_right_x,
                       eq_v_damp_top_y, eq_rho_damp_top_y,
                       eq_v_damp_base_y, eq_rho_damp_base_y] + src_rhox + src_rhoy + rec_term, name='Forward')
    return op, eqns

def circle_mask(rows, cols, radius, sigma):
    center_x, center_y = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    mask = np.ones((rows, cols))
    outside_circle = distance_from_center > radius
    mask[outside_circle] = np.exp(-((distance_from_center[outside_circle] - radius) ** 2) / (2 * sigma ** 2))
    return mask

def EqnBackward(model, p=None):
    x, y = model.grid.dimensions

    p = p or TimeFunction(name='p', grid=model.grid, staggered=NODE, time_order=1, space_order=model.space_order)
    v1x = TimeFunction(name='vx', grid=model.grid, time_order=1, space_order=model.space_order, staggered=x)
    v1y = TimeFunction(name='vy', grid=model.grid, time_order=1, space_order=model.space_order, staggered=y)
    rho1x = TimeFunction(name='rhox', grid=model.grid, staggered=NODE, time_order=1, space_order=model.space_order)
    rho1y = TimeFunction(name='rhoy', grid=model.grid, staggered=NODE, time_order=1, space_order=model.space_order)

    dt = model.critical_dt

    p_dx = getattr(p, 'd%s' % p.space_dimensions[0].name)
    p_dy = getattr(p, 'd%s' % p.space_dimensions[1].name)
    vx_dx = getattr(v1x.backward, 'd%s' % v1x.space_dimensions[0].name)
    vy_dy = getattr(v1y.backward, 'd%s' % v1y.space_dimensions[1].name)

    eq_v_x = Eq(v1x.backward, v1x + dt * p_dx / model.rho, subdomain=model.grid.subdomains['main'])
    eq_v_y = Eq(v1y.backward, v1y + dt * p_dy / model.rho, subdomain=model.grid.subdomains['main'])
    eq_rho_x = Eq(rho1x.backward, rho1x + dt * model.rho * vx_dx, subdomain=model.grid.subdomains['main'])
    eq_rho_y = Eq(rho1y.backward, rho1y + dt * model.rho * vy_dy, subdomain=model.grid.subdomains['main'])
    eq_p = Eq(p.backward, model.vp * model.vp * (rho1x.backward + rho1y.backward))  # , subdomain=grid.subdomains['main'])

    alpha_max = 2 * 1540 / model.spacing[0]

    # Damping parameterisation
    d_l = (1 - x / model.nbl) ** 4  # Left side
    d_r = (1 - (model.grid.shape[0] - 1 - x) / model.nbl) ** 4  # Right side
    d_t = (1 - y / model.nbl) ** 4  # Top side
    d_b = (1 - (model.grid.shape[1] - 1 - y) / model.nbl) ** 4  # Base edge
    # staggered
    d_l_s = (1 - (x + 0.5) / model.nbl) ** 4  # Left side
    d_r_s = (1 - (model.grid.shape[0] - 1 - (x + 0.5)) / model.nbl) ** 4  # Right side
    d_t_s = (1 - (y + 0.5) / model.nbl) ** 4  # Top side
    d_b_s = (1 - (model.grid.shape[1] - 1 - (y + 0.5)) / model.nbl) ** 4  # Base edge

    # for the PML domain
    eq_v_damp_left_x = Eq(v1x.backward, (1 - dt * d_l_s * alpha_max) * v1x + dt * p_dx / model.rho,
                          subdomain=model.grid.subdomains['left'])
    eq_rho_damp_left_x = Eq(rho1x.backward, (1 - dt * d_l * alpha_max) * rho1x + dt * model.rho * vx_dx,
                            subdomain=model.grid.subdomains['left'])

    eq_v_damp_right_x = Eq(v1x.backward, (1 - dt * d_r_s * alpha_max) * v1x + dt * p_dx / model.rho,
                           subdomain=model.grid.subdomains['right'])
    eq_rho_damp_right_x = Eq(rho1x.backward, (1 - dt * d_r * alpha_max) * rho1x + dt * model.rho * vx_dx,
                             subdomain=model.grid.subdomains['right'])

    eq_v_damp_top_y = Eq(v1y.backward, (1 - dt * d_t_s * alpha_max) * v1y + dt * p_dy / model.rho,
                         subdomain=model.grid.subdomains['top'])
    eq_rho_damp_top_y = Eq(rho1y.backward, (1 - dt * d_t * alpha_max) * rho1y + dt * model.rho * vy_dy,
                           subdomain=model.grid.subdomains['top'])

    eq_v_damp_base_y = Eq(v1y.backward, (1 - dt * d_b_s * alpha_max) * v1y + dt * p_dy / model.rho,
                          subdomain=model.grid.subdomains['base'])
    eq_rho_damp_base_y = Eq(rho1y.backward, (1 - dt * d_b * alpha_max) * rho1y + dt * model.rho * vy_dy,
                            subdomain=model.grid.subdomains['base'])

    eqns = [eq_v_x, eq_v_y, eq_rho_x, eq_rho_y, eq_p, eq_v_damp_left_x, eq_rho_damp_left_x,
                   eq_v_damp_right_x, eq_rho_damp_right_x,
                   eq_v_damp_top_y, eq_rho_damp_top_y,
                   eq_v_damp_base_y, eq_rho_damp_base_y]
    return eqns