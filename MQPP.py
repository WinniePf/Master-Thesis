import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.constants import hbar, e, c #, eV
import os
from scipy.special import sici
#import tracemalloc

plt.rcParams.update({
    "text.usetex": False,
})

def generate_lattice(a, cutoff_radius, basis, min_excl=False):
    # Convert the basis to a numpy array
    basis = np.array(basis)

    # Calculate the number of cells needed to cover the cutoff radius
    max_cells = int(np.ceil(cutoff_radius / a)) + 1

    # Create a grid of points using broadcasting
    grid = np.arange(-max_cells, max_cells + 1)
    i, j, k = np.meshgrid(grid, grid, grid, indexing='ij')

    # Stack i, j, k into a (N, 3) shape array for all points in the grid
    lattice_points = np.stack([i, j, k], axis=-1).reshape(-1, 3)

    # Multiply lattice points by the basis matrix
    positions = np.dot(lattice_points, basis.T)

    # Calculate distances of positions from the origin and filter based on the cutoff_radius
    distances = np.linalg.norm(positions, axis=1)
    valid_positions = positions[(distances > min_excl*a*1.001) & (distances <= cutoff_radius)]

    # Remove duplicates
    unique_positions = np.unique(valid_positions, axis=0)

    return unique_positions


def f_q(q, rho_dipole, rho_quadrupole, basis, filename, nu):
    
    # with float16 and complex64 takes 2x longer than float32 and complex64
    # select accuracy (32 bit is sweet spot)
    base = 32
    # set data types (there is no complex 32)
    data_type = "float" + str(base)
    data_type_c = "complex" + str([2*base if base != 16 else 4*base][0])
    
    q_pol = get_polarization_vectors(q).astype(data_type)

    ###
    startD = time.time()
    
    # Precompute rho-related terms 0.004s
    rho_norm_D = np.linalg.norm(rho_dipole, axis=1)
    rho_norm_D = rho_norm_D.astype(data_type)
    rho_unit_D = rho_dipole / rho_norm_D[:, np.newaxis]
    rho_unit_D = rho_unit_D[:, :, np.newaxis]
    rho_unit_D = rho_unit_D.astype(data_type)
    
    v_uc = nu
    rho_a_3_D = v_uc / (rho_norm_D) ** (3)
    
    
    # opt path reduces comp. time by 2x
    opt_path = True

    
    res_dipole = rho_unit_D @ rho_unit_D.transpose(0, 2, 1)  # Shape: (N,3,3)

    ##opt_path, _ = np.einsum_path('rij,qjk->rqik', res_dipole, np.transpose(q_pol, axes=(0,2,1)), optimize=optimization)
    res_dipole = np.einsum('rij,qjk->rqik', res_dipole, np.transpose(q_pol, axes=(0,2,1)), optimize=opt_path)

    ##opt_path, _ = np.einsum_path('qij,rqjk->rqik', q_pol, res_dipole, optimize=optimization)
    res_dipole = np.einsum('qij,rqjk->rqik', q_pol, res_dipole, optimize=opt_path)

    np.add(np.eye(3, dtype=data_type), - 3*res_dipole, out=res_dipole)
    
    phase = np.tensordot(rho_dipole, q.T, axes=1)  # Shape: (j, q)
    cos_q_rho_D = np.exp(1j * phase, dtype=data_type_c)  # Shape: (j, q)

    res_dipole = np.einsum('qj,jqlk->qlk', np.multiply(rho_a_3_D, cos_q_rho_D.T, dtype=data_type_c) / 2, res_dipole, optimize=True, dtype=data_type_c)
    
    endD = time.time()
    
    print("time dipoles: ", endD - startD)
    
    if quadrupoles:
        # quadrupole term
        startQ = time.time()
        
        
        rho_norm_Q = np.linalg.norm(rho_quadrupole, axis=1)
        rho_unit_Q = rho_quadrupole / rho_norm_Q[:, np.newaxis]
        rho_unit_Q = rho_unit_Q[:, :, np.newaxis]
    
        cos_q_rho_Q = np.exp(1j*np.tensordot(rho_quadrupole, np.transpose(q), axes=1))    
        rho_a_3_Q = v_uc / (rho_norm_Q) ** (3)
    
        # diagonal
        chi_v = [1 / np.sqrt(2) * np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
                 1 / np.sqrt(6) * np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 2]]),
                 1 / np.sqrt(2) * np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
                 1 / np.sqrt(2) * np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
                 1 / np.sqrt(2) * np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])]
        
        chi_v_q = np.einsum('vij,qkj->qvik', chi_v, q_pol)
        chi_v_q = np.einsum('qij,qvjk->qvik', q_pol, chi_v_q)
        
        
        # Precompute outer products for rho_unit[i]
        rho_outer_Q = np.einsum('ij,ik->ijk', rho_unit_Q[:,:,0], rho_unit_Q[:,:,0])  # Shape: (len(rho), 3, 3)
    
        # calculate X:nn, X*X:nn, X:X
        x_nn = np.trace(np.einsum('qvij,rjk->qrvik', chi_v_q, rho_outer_Q), axis1=3, axis2=4)
        xx = np.einsum('qvij,qwjk->qvwik', chi_v_q, chi_v_q)
    
        term = 35* np.einsum('qri,qrj->qrij', x_nn, x_nn)
        term += -20 * np.trace(np.einsum('qvwij,rjk->qrvwik', np.transpose(xx,axes=(0,2,1,3,4)), rho_outer_Q), axis1=4, axis2=5)
        term += 2 * np.trace(np.einsum('qvij,qwjk->qvwik', chi_v_q, chi_v_q), axis1=-1, axis2=-2)[:,np.newaxis,:,:]
            
        rho_a_scalar_Q = rho_a_3_Q ** (5 / 3) / 6  # This is of shape (len(rho))
        
        res_quad = np.einsum('r,qrij->qrij', rho_a_scalar_Q, term)
        res_quad = np.einsum('qrij,rq->qij', res_quad, cos_q_rho_Q)
        
        # offdiagonal (dipole quadrupole coupling) 
        rho_a_scalar = rho_a_3_Q ** (4 / 3) / 2
        
        ne_v = np.einsum('ri,qji->rqj', rho_unit_Q[:,:,0], q_pol)
        
        rho_dot_chi = np.einsum('rn,qvnm->qrvm', rho_unit_Q[:, :, 0], chi_v_q)  # shape (rho_len, q_len, 5)
        rho_chi_rho = np.einsum('qrvn,rn->qrv', rho_dot_chi, rho_unit_Q[:, :, 0])  # shape (rho_len, q_len, 5)
        
        # equation (22) 1)
        res_quad_offcenter = -5 * np.einsum('qrv,rqi->qrvi', rho_chi_rho, ne_v)
    
        q_dot_chi_v_q = np.einsum('qnm,qimj->qnij', q_pol, chi_v_q)
        
        # equation (22) 2)
        res_quad_offcenter += 2 * np.einsum('qimn,rn->qrmi', q_dot_chi_v_q, rho_unit_Q[:, :, 0]) # correct
        
        # divide by (R/R)^4 equation (22) 3)
        res_quad_offcenter *=  rho_a_scalar[np.newaxis,:, np.newaxis, np.newaxis]
        
        # multiplying the terms with the exponential equation (22) 4)
        res_quad_offcenter = np.einsum('qrij,rq->qji', res_quad_offcenter, cos_q_rho_Q)
        
        endQ = time.time()
        print("results quad time", endQ-startQ)
    
        # create matrix for structure function and save
        res = np.block([[res_dipole, res_quad_offcenter], [np.conj(np.transpose(res_quad_offcenter, axes=(0,2,1))), res_quad]])
    
    else:
        res = res_dipole
        
    # save fq file:
    abspath = os.path.abspath(os.getcwd()) + '/'
    with open(abspath + filename, 'wb') as f:
        np.save(f, res)

    return res


def check_if_file_exists(lattice, kpath, cutoff_radius_dipole, cutoff_radius_quadrupole, alpha, qpoints, path):
    global recalculate
    if quadrupoles:
        quad_string = 'qp'
    else:
        quad_string = 'dp'

    filename = 'fq_' + lattice + '_' + quad_string + '_' + ''.join(kpath) + "_" + str(round(cutoff_radius_dipole)) + "_" + str(round(cutoff_radius_quadrupole)) + "_" + str(round(alpha)) + "_" + str(qpoints)
    
    if not path:
        path = os.path.abspath(os.getcwd()) + os.sep + filename
    if os.path.exists(path + os.sep + filename) and not recalculate:
        print("Structure factor found")
        return True
    elif recalculate and os.path.exists(path + os.sep + filename):
        print("Forced recalculation!")
        return False    
    else:
        print("Structure factor not found, recalculating")
        return False    


def run_f_q_for_multiple_q(q_list, rho_dipole, rho_quadrupole, cutoff_radius_dipole, cutoff_radius_quadrupole, lattice, basis, alpha, kpath, qpoints, file_exists, path, nu):
    if quadrupoles:
        quad_string = 'qp'
    else:
        quad_string = 'dp'

    filename = 'fq_' + lattice + '_' + quad_string + '_' + ''.join(kpath) + "_" + str(round(cutoff_radius_dipole)) + "_" + str(round(cutoff_radius_quadrupole)) + "_" + str(round(alpha)) + "_" + str(qpoints)
    
    if file_exists:
        with open(path + os.sep + filename, 'rb') as f:
            results = np.load(f)
    else:
        results = f_q(q_list, rho_dipole, rho_quadrupole, basis, filename, nu)
    
    q_pol = get_polarization_vectors(q_list)
    
    for i in range(len(q_list)):
        qnorm = np.linalg.norm(q_list[i])
        if qnorm < alpha/cutoff_radius_dipole:
            corr = -2/3 * np.pi * (np.eye(3) - 3 * np.outer(q_list[i], q_list[i])/qnorm**2) # /(3*nu)
            results[i][0:3, 0:3] = np.dot(np.dot(q_pol[i], corr), q_pol[i].T)
            
    return results


def select_basis(lattice):
    if lattice=='FCC':
        basis = np.array([
            [1 / 2, 0, 1 / 2],
            [1 / 2, 1 / 2, 0],
            [0, 1 / 2, 1 / 2]
        ]) * np.sqrt(2)
        rmax = 1/2
        nu = np.linalg.det(basis)    

    elif lattice=='SC':
        basis = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        rmax = 1/2
        nu = np.linalg.det(basis)    
    else:
        print('Lattice not implemented')
    
    return basis, nu, rmax


def k_path(kpath, lattice, n_kpoints, basis, nu):
    G = 2 * np.pi * np.array([np.cross(basis[1], basis[2]), np.cross(basis[2], basis[0]), np.cross(basis[0], basis[1])]) / nu
    k_line = []
    for i in kpath:
        if i == 'K':
            K = np.dot(np.array([3 / 8, 3 / 4, 3 / 8]), G)
            k_line.append(K)
        elif i == 'G':
            Gamma = np.array([0.0, 0.0, 0.0])
            k_line.append(Gamma)
        elif i == 'L':
            L = np.dot(np.array([1 / 2, 1 / 2, 1 / 2]), G)
            k_line.append(L)
        elif i == 'X':
            if lattice == 'FCC':
                X = np.dot(np.array([0, 1 / 2, 1 / 2]), G)
            elif lattice == 'SC':
                X = np.dot(np.array([0, 0.5, 0]), G)
            else:
                X = np.dot(np.array([0, 1 / 2, 1 / 2]), G)
            k_line.append(X)
        elif i == 'W':
            W = np.dot(np.array([1 / 4, 3 / 4, 1 / 2]), G)
            k_line.append(W)
        elif i == 'U':
            U = np.dot(np.array([1 / 4, 5 / 8, 5 / 8]), G)
            k_line.append(U)
        elif i == 'R':
            U = np.dot(np.array([1 / 2, 1 / 2, 1 / 2]), G)
            k_line.append(U)
        elif i == 'M':
            U = np.dot(np.array([1 / 2, 1 / 2, 0]), G)
            k_line.append(U)
        elif i == 'G2':
            U = np.dot(np.array([0.99, 0.99, 0.99]), G)
            k_line.append(U)
        else:
            print('High symmetry point not set')

    # Linear interpolation between high-symmetry points
    def interpolate_kpath(start, end, num_points):
        return np.linspace(start, end, num_points)

    kpath = interpolate_kpath(k_line[0], k_line[1], n_kpoints)
    # Define the full path sc
    for l in range(len(k_line) - 2):
        kpath = np.concatenate([kpath, interpolate_kpath(k_line[l + 1], k_line[l + 2], n_kpoints)])

    # correction diverges for k = 0, remove such entries:
    for i in range(len(kpath)):
        if np.all(kpath[i] == ([0,0,0])):
            if i !=0:
                kpath[i] = kpath[i-1]*0.1
            else:
                kpath[i] = kpath[i+1]*0.1


    return kpath, G


def plot_Dispersion(eigvals, eigvects, n_kpoints, kpath, omega_D, title, ymax, ymin):
    #eigvals = np.abs(eigvals)
    
    plt.figure(figsize=(9, 6))
    plt.rcParams.update({'font.size': 18})
    
    lev = len(eigvals[0])//2
    # determine color of polariton branch depending on the dipole, quad, photon contributions. Norm 
    # first 8 and lev + 0, ..., 8 are matter, rest is photons 
    
    dp_s = np.sqrt(np.sum(np.abs(eigvects[:,0:3,:])**2, axis=1) + np.sum(np.abs(eigvects[:,lev:lev+3,:])**2, axis=1))
    # normalize contribution to 1
    dp_s /= np.max(dp_s)

    if quadrupoles:
        qd_s = np.sqrt(np.sum(np.abs(eigvects[:,3:8,:])**2, axis=1) + np.sum(np.abs(eigvects[:,lev + 3:lev + 8,:])**2, axis=1))
        qd_s /= np.max(qd_s)
        color_list = [qd_s, np.zeros(np.shape(dp_s)), dp_s]

    if photon_interaction:
        if quadrupoles:
            n_matt = 8
        else:
            n_matt = 3
        
        pt_s = np.sqrt(np.sum(np.abs(eigvects[:,n_matt:lev,:])**2, axis=1) + np.sum(np.abs(eigvects[:,lev+n_matt:,:])**2, axis=1))
        pt_s /= np.max(pt_s)
        if quadrupoles:
            color_list = [qd_s, pt_s, dp_s]
        else:
            color_list = [np.zeros(np.shape(dp_s)), pt_s, dp_s]

    if not quadrupoles and not photon_interaction:
        color_list = [np.zeros(np.shape(dp_s)), np.zeros(np.shape(dp_s)), dp_s]


    colors = np.stack(color_list, axis=-1).reshape(-1, 3)  # shape (N, 3)

    s = 4 # set size of points

    x = np.repeat(np.arange(len(eigvals))[:,np.newaxis], len(eigvals[0]), axis=1)
    
    y = eigvals * omega_D * (hbar/e)
    
    plt.scatter(x, y, c=colors, s=s, rasterized=True)
    
    kpath_plot = kpath
    for i in range(len(kpath_plot)):
        if kpath_plot[i] == 'G':
            kpath_plot[i] = 'Γ'
        elif kpath_plot[i] == 'G2':
            kpath_plot[i] = "Γ'"

            
    xticks = np.concatenate(([0], [(i+1)*n_kpoints-1 for i in range(len(kpath)-1)]))
    plt.xticks(xticks, kpath)
    plt.grid(axis='x')
    
    if photon_interaction:
        plt.ylabel('$\omega_{pp,q}$ (eV)')
    else:
        plt.ylabel('$\omega_{pl,q}$ (eV)')
        
    plt.ylim([ymin,ymax])
    plt.xlim([min(xticks),max(xticks)])
    plt.title(title)

    plt.show()


def plot_Decomp(eigvals, eigvects, n_kpoints, kpath, omega_D, title, ymax, ymin):
    #eigvals = np.abs(eigvals)
    # create figure with 3 subplots
    ncols = 1 + quadrupoles + photon_interaction
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(8, 8))
    
    if ncols == 1:
        ax = [ax]
    
    plt.rcParams.update({'font.size': 18})
    
    lev = len(eigvals[0])//2
    
    s_max = 10
    # determine the color for the individual excitations
    dipole_color = (0, 0, 1) # blue
    quadrupole_color = (1, 0, 0) # red
    photon_color = (0, 1, 0) # green

    # determine color of polariton branch depending on the dipole, quad, photon contributions. Norm 
    # first 8 and lev + 0, ..., 8 are matter, rest is photons 
    dp_s = np.sqrt(np.sum(np.abs(eigvects[:,0:3,:])**2, axis=1) + np.sum(np.abs(eigvects[:,lev:lev+3,:])**2, axis=1))
    
    # normalize contribution to smax
    dp_s /= np.max(dp_s) / s_max
    ax[0].scatter(np.repeat(np.arange(len(eigvals))[:,np.newaxis], len(eigvals[0]), axis=1), eigvals * omega_D * (hbar/e), color=dipole_color, s=dp_s)

    if quadrupoles:
        qd_s = np.sqrt(np.sum(np.abs(eigvects[:,3:8,:])**2, axis=1) + np.sum(np.abs(eigvects[:,lev + 3:lev + 8,:])**2, axis=1))
        qd_s /= np.max(qd_s) / s_max
        ax[1].scatter(np.repeat(np.arange(len(eigvals))[:,np.newaxis], len(eigvals[0]), axis=1), eigvals * omega_D * (hbar/e), color=quadrupole_color, s=qd_s)
        
    if photon_interaction:
        if quadrupoles:
            n_matt = 8
        else:
            n_matt = 3
        pt_s = np.sqrt(np.sum(np.abs(eigvects[:,n_matt:lev,:])**2, axis=1) + np.sum(np.abs(eigvects[:,lev+n_matt:,:])**2, axis=1))
        pt_s /= np.max(pt_s) / s_max
        ax[-1].scatter(np.repeat(np.arange(len(eigvals))[:,np.newaxis], len(eigvals[0]), axis=1), eigvals * omega_D * (hbar/e), color=photon_color, s=pt_s)

    
    kpath_plot = kpath
    # replace G by Γ
    for i in range(len(kpath_plot)):
        if kpath_plot[i] == 'G':
            kpath_plot[i] = 'Γ'
         
    xticks = np.concatenate(([0], [(i+1)*n_kpoints-1 for i in range(len(kpath)-1)]))
    
    if photon_interaction:
        ax[0].set_ylabel('$\omega_{pp,q}$ (eV)')
    else:
        ax[0].set_ylabel('$\omega_{pl,q}$ (eV)')

    
    for i in range(len(ax)):
        ax[i].set_xticks(xticks, kpath)
        ax[i].grid(axis='x')
        ax[i].set_ylim([ymin,ymax])
        ax[i].set_xlim([min(xticks),max(xticks)])
        if i != 0 and i != len(ax)-1:
            ax[i].set_yticks([])

    if ncols != 1:
        ax[-1].yaxis.set_label_position("right")
        ax[-1].yaxis.tick_right()
        
    plt.title(title)
    plt.subplots_adjust(wspace=0.1)
    plt.show()


def sphbesint(nu, x):
    if nu == 0:
        fct = 3 * (np.sin(x) - x * np.cos(x)) / x**3

    elif nu == 1:
        fct = 9 * (sici(x)[0] - np.sin(x)) / x**3

    elif nu == -1:
        fct = x/x
    else:
        print("error in sphbesint function")

    return fct


def get_polarization_vectors(q_vecs):
    norms = np.linalg.norm(q_vecs, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1, norms)
    q_normed = q_vecs / safe_norms

    z_axis = np.array([0, 0, 1])
    z_broadcasted = np.tile(z_axis, (q_vecs.shape[0], 1))
    qpol_1 = np.cross(z_broadcasted, q_normed)
    qpol_1 /= np.linalg.norm(qpol_1, axis=1, keepdims=True)
    qpol_2 = np.cross(q_normed, qpol_1)
    qpol_2 /= np.linalg.norm(qpol_2, axis=1, keepdims=True)

    return np.stack([qpol_1, qpol_2, q_normed], axis=1)  # (N, 3, 3)


def dispersion(f_q, eps_m, eps_d, omega_p, omega_D, omega_Q, f, nu, nBZ, photon_interaction, G, q, w0, r, a_latt, rmax):
    # function that either calculates the collective plasmon dispersion or the plasmon polariton dispersion 
    # depending on if photon_interaction is set to True
    
    Lambda_D = 3*eps_m*1/(2*(eps_d + 2 * eps_m))*f/nu

    if quadrupoles:
        matter_excitations = np.concatenate((np.array([omega_D]*3), np.array([omega_Q]*5)))/omega_D
        magic_factor = 2
        Lambda_Q = ((5*eps_m*omega_Q/omega_D) / (12*magic_factor*(eps_d + (3/2)*eps_m)))*f**(5/3)*(nu)**(-5/3)

        # manipulate matrix first:
        coupling_terms = np.concatenate((np.array([Lambda_D]*3), np.array([Lambda_Q]*5)))
        coupling_matrix = np.sqrt(np.outer(coupling_terms, coupling_terms))
        #print(np.shape(coupling_matrix))
        f_q = coupling_matrix * f_q

    else:
        matter_excitations = np.array([omega_D]*3)/omega_D
        coupling_terms = (np.array([Lambda_D]*3))
        coupling_matrix = np.sqrt(np.outer(coupling_terms, coupling_terms))
        f_q = coupling_matrix * f_q[:,0:3,0:3]


    if photon_interaction:

        # photon coupling
        # calculate parameters:
        cutoff_BZ = ( nBZ - 1 + 0.1 ) * np.linalg.norm(G,2)

        i_range = np.arange(-nBZ, nBZ + 1)
        i1,i2,i3 = np.meshgrid(i_range, i_range, i_range, indexing="ij")
        i_comb = np.vstack([i1.ravel(), i2.ravel(), i3.ravel()])        
        
        qGt = np.dot(G, i_comb)
        norms = np.linalg.norm(qGt, axis=0)
        valid_indices = norms < cutoff_BZ
        
        Gn0 = qGt[:, valid_indices]
        
        ind = np.argsort(norms[valid_indices], kind="stable")
        Gn0 = Gn0[:,ind]
        Gn = [Gn0[:, 0]]
        
        if nBZ > 1:
            nbz = 1
            gg = np.linalg.norm(Gn0[:,1])
            
            while nbz < nBZ:
                ii = np.asarray( (np.linalg.norm(Gn0, axis=0) < 1.01*gg) & (np.linalg.norm(Gn0, axis=0) > 0.99*gg)).nonzero()[0]
                Gn = np.vstack((Gn, Gn0[:,ii].T))
                nbz += 1

                if max(ii) < len(Gn0[0]):
                    gg = np.linalg.norm( Gn0[:, max(ii) + 1] )
                else:
                    nbz = nBZ
        
            ng = len(Gn[:,0])
            nG = len(ind)

                                
        # plasmon-photon interaction:

        qG = np.einsum("qi,j->qji", q, np.ones(ng)) + Gn
        qGnorm = np.linalg.norm(qG, axis=2)
        qGnorm = np.kron(qGnorm, [1, 1])
        
        w0 *= a_latt/c
        
        wpt = qGnorm / (w0) / np.sqrt(eps_m)
        # wpt is correct        
        PqQ = np.zeros((len(q), 8, 2*ng), dtype="complex64")
        
        q_pol_0 = get_polarization_vectors(q)

        for i in range(ng):
            qa = np.array(q + Gn[i])
            q_pol = get_polarization_vectors(qa)

            chi_v = [1 / np.sqrt(2) * np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
                     1 / np.sqrt(6) * np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 2]]),
                     1 / np.sqrt(2) * np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
                     1 / np.sqrt(2) * np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
                     1 / np.sqrt(2) * np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])]
            
            chi_v_q = np.einsum('vij,qkj->qvik', chi_v, q_pol_0)
            chi_v_q = np.einsum('qij,qvjk->qvik', q_pol_0, chi_v_q)

            lamb1 = q_pol[:,0]
            lamb2 = q_pol[:,1]
            eq = q_pol[:,2]

            Pd = [[lamb1[:,0], lamb2[:,0]], [lamb1[:,1], lamb2[:,1]], [lamb1[:,2], lamb2[:,2]]]

            # diagonal

            Pd = np.einsum("qij,jlq->qil", q_pol_0, Pd)
            
            chi_eq1 = np.einsum('qi,qvij,qj->qv', lamb1, chi_v_q, eq)
            chi_eq2 = np.einsum('qi,qvji,qj->qv', lamb2, chi_v_q, eq)
            chi_eq3 = np.einsum('qi,qvij,qj->qv', eq, chi_v_q, lamb1)
            chi_eq4 = np.einsum('qi,qvji,qj->qv', eq, chi_v_q, lamb2)

            Rb = nu**(1/3)

            z1 = -1j/2*Rb*np.linalg.norm(qa,axis=-1)*( chi_eq1.T + chi_eq3.T )
            z2 = -1j/2*Rb*np.linalg.norm(qa,axis=-1)*( chi_eq2.T + chi_eq4.T )

            # z's are correct

            PQ = np.transpose(np.array([z1,z2]),axes=(2,1,0))
            
            Pq = np.concatenate((Pd, PQ), axis=1)

            PqQ[:, :, 2 * i] = Pq[:, :, 0]
            PqQ[:, :, 2 * i + 1] = Pq[:, :, 1]


            # end loop over ig
        
        
        # calculate factor (dmatrix param mqpp)
        
        factor0 = sphbesint(0, r/rmax * qGnorm) # correct
        factor1 = sphbesint(1, r/rmax * qGnorm) # correct

        factor0 = np.repeat(factor0[:,:,np.newaxis], 3, axis=2)
        factor1 = np.repeat(factor1[:,:,np.newaxis], 5, axis=2)

        factor = np.concatenate((factor0, factor1), axis=2) #correct

        Gn2 = np.empty(( 2*len(Gn), 3), dtype=Gn.dtype)
        Gn2[0::2] = Gn
        Gn2[1::2] = Gn
        
        rp = [0,0,0]
        phase = np.exp(1j*np.einsum("i,ki->k", rp, Gn2))

        PqQ = np.einsum("qik,k->qik", PqQ, phase)
        PqQ *= np.swapaxes(factor, 2,1)
        xiq0 = np.sqrt(2*np.pi*np.einsum("i,qj->qij",coupling_terms, 1/wpt))
        if quadrupoles:
            xiq = xiq0 * PqQ # correct
        else:
            xiq = xiq0 * PqQ[:,0:3,:] # correct

        Xi = np.einsum("qji,jk,qkm->qim", xiq.conj(), np.diag(matter_excitations), xiq) # correct

        fplpt = 1j*np.einsum("jk,qkm->qjm", np.diag(matter_excitations), xiq) # correct

        diag_aptpt = np.einsum("qi,ij->qij", wpt, np.identity(len(wpt[1])))
                
        A_ptpt = diag_aptpt + 2 * Xi
        
        A_plpl = np.repeat(np.diag(matter_excitations)[np.newaxis,:,:], len(f_q[:,0,0]), 0) + 2 * f_q
        
        G_M = f_q + np.transpose(f_q, axes=(0, 2, 1))
        G_pt = Xi + np.transpose(Xi, axes=(0,2,1))
        A = np.block([[A_plpl, fplpt],[fplpt.swapaxes(2,1).conj(), A_ptpt]])
        G_M = np.block([[G_M,fplpt],[fplpt.swapaxes(2,1), G_pt]])
        
        D = np.block([[A, G_M], [-G_M.swapaxes(2, 1).conj(), -np.transpose(A, axes=(0,2,1))]])
    else:        
        A = np.repeat(np.diag(matter_excitations)[np.newaxis,:,:], len(f_q[:,0,0]), 0) + 2 * f_q

        G_M = f_q + np.transpose(f_q, axes=(0, 2, 1))
        D = np.block([[A, G_M], [-G_M.swapaxes(2, 1).conj(), -np.transpose(A, axes=(0,2,1))]])
    

    eigvals, eigenvects = np.linalg.eig(D) #np.linalg.eig(D)
    return eigvals, eigenvects


def MQPP_main(lattice, kpath, n_kpoints, cutoff_radius_dipole, cutoff_radius_quadrupole, eps_m, eps_d, omega_p, omega_D, omega_Q, f, title, path, a_latt, nBZ, plot=0, ymax=10, ymin=0, min_excl=False):
    # determine cutoff distance to Gamma point
    alpha = 2*cutoff_radius_dipole/3 / np.sqrt(2) / 2    
    # check if structure function exists
    file_exists = check_if_file_exists(lattice, kpath, cutoff_radius_dipole, cutoff_radius_quadrupole, alpha, n_kpoints, path)
    # create basis vectors
    basis, nu, rmax = select_basis(lattice)
    # convert the given kpath to a list of vectors in the brillouin zone
    k_line, G = k_path(kpath, lattice, n_kpoints, basis, nu)
    
    if file_exists:
        f_q = run_f_q_for_multiple_q(k_line, None, None, cutoff_radius_dipole, cutoff_radius_quadrupole, lattice, basis, alpha, kpath, n_kpoints, file_exists, path, nu)
    else:
        # create lattice positions up to a cutoff radius
        next_neighbors_dipole = generate_lattice(1, cutoff_radius_dipole, basis)
        if quadrupoles:
            next_neighbors_quadrupole = generate_lattice(1, cutoff_radius_quadrupole, basis)
        else:
            next_neighbors_quadrupole = None
        f_q = run_f_q_for_multiple_q(k_line, next_neighbors_dipole, next_neighbors_quadrupole, cutoff_radius_dipole, cutoff_radius_quadrupole, lattice, basis, alpha, kpath, n_kpoints, file_exists, path, nu)

    #constants
    f = f * (rmax)**3
    r = f**(1/3) * rmax

    eigvals, eigvects = dispersion(f_q, eps_m, eps_d, omega_p, omega_D, omega_Q, f, nu, nBZ, photon_interaction, G, k_line, omega_D, r, a_latt, rmax)
    
    if plot == 0:
        plot_Dispersion(eigvals, eigvects, n_kpoints, kpath, omega_D, title, ymax, ymin)
    else:
        plot_Decomp(eigvals, eigvects, n_kpoints, kpath, omega_D, title, ymax, ymin)

    return eigvals


# where to save the structure function files
path = "/home/winniep01/Documents/thesis/mqpp_python"
lattice = 'FCC' # lattice: FCC and SC so far implemented, others can be added easily


# path in the selected lattice, 'G' for gamma point 
kpath = ['K', 'G', 'X', 'W', 'L', 'G'] # full fcc path
#kpath = ['M', 'G', 'X', 'M', 'R', 'G']
#kpath = ["G", "L"]
#kpath = ["G", "L"]

quadrupoles = False             # flag to turn on/off quadrupoles
photon_interaction = False      # flag to turn on/off photon interactions
recalculate = False             # flag to force recalculation

n_kpoints = 10                  # number of points between 2 high symmetry points
a_latt = 60e-9                  # Lattice constant
cutoff_radius_dipole = 60       #     
cutoff_radius_quadrupole = 7    # 
nBZ = 2                         # number of brillouin zones
eps_m = 1
eps_d = 1
f = 0.60 #(280/330)**3 * 0.74   # fill factor (not relative but total)
omega_p = 4 * e / hbar          # plasma frequency of the metal
plot = 0                        # 0 for dispersion, 1 for decomp
ymax = 8 #omega_p * 1.2 / e * hbar # maximum y value to plot
ymin = 0

# dipole and quadrupole plasmon resonances in the quasistatic approx.
omega_D = omega_p / np.sqrt(eps_d + 2 * eps_m)
omega_Q = omega_p / np.sqrt(eps_d + 3 / 2 * eps_m)

# you can set the dipole and quadrupole plasmon resonances here
#omega_D = 1.75 * e / hbar
#omega_Q = 1.95 * e / hbar


# call the main function. If you want to further process the MQPP Bands, simply pass the return value to a variable
mqpp_bands = MQPP_main(lattice, kpath, n_kpoints, cutoff_radius_dipole, cutoff_radius_quadrupole, eps_m, eps_d, omega_p, omega_D, omega_Q, f, title=None, path=path, a_latt=a_latt, nBZ=nBZ, plot=plot, ymax=ymax, ymin=ymin, min_excl=False)

