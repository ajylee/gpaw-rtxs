import numpy as np
from gpaw.mpi import rank, size, world

def Bootstrap1(chi0_wGG, Nw, Kc_GG, printtxt, print_bootstrap):

    Nw_local = chi0_wGG.shape[0]
    npw = chi0_wGG.shape[1]
    
    
    # arxiv 1107.0199
    fxc_GG = np.zeros((npw, npw), dtype=complex)
    tmp_GG = np.eye(npw, npw)
    dminv_wGG = np.zeros((Nw_local, npw, npw), dtype=complex)
    dflocal_w = np.zeros(Nw_local, dtype=complex)
    df_w = np.zeros(Nw, dtype=complex)
                
    for iscf in range(120):
        dminvold_wGG = dminv_wGG.copy()
        Kxc_GG = Kc_GG + fxc_GG
        for iw in range(Nw_local):
            chi_GG = np.dot(chi0_wGG[iw], np.linalg.inv(tmp_GG - np.dot(Kxc_GG, chi0_wGG[iw])))
            dminv_wGG[iw] = tmp_GG + np.dot(Kc_GG, chi_GG)
        if world.rank == 0:
            alpha = dminv_wGG[0,0,0] / (Kc_GG[0,0] * chi0_wGG[0,0,0])
            fxc_GG = alpha * Kc_GG
        world.broadcast(fxc_GG, 0)
    
        error = np.abs(dminvold_wGG - dminv_wGG).sum()
        if world.sum(error) < 0.1:
            printtxt('Self consistent fxc finished in %d iterations ! ' %(iscf))
            break
        if iscf > 100:
            printtxt('Too many fxc scf steps !')
    
        if print_bootstrap:
            for iw in range(Nw_local):
                dflocal_w[iw] = np.linalg.inv(dminv_wGG[iw])[0,0]
            world.all_gather(dflocal_w, df_w)
            if world.rank == 0:
                f = open('df_scf%d' %(iscf), 'w')
                for iw in range(Nw):
                    print >> f, np.real(df_w[iw]), np.imag(df_w[iw])
                f.close()
            world.barrier()
        
    for iw in range(Nw_local):
        dflocal_w[iw] = np.linalg.inv(dminv_wGG[iw])[0,0]
        world.all_gather(dflocal_w, df_w)
    
    return df_w

def BootstrapSerial(chi0_wGG, Nw, Kc_GG, printtxt, print_bootstrap):
    #Work in parallel with Bootstrap1, adjusted to fit fxc.py

    if world.rank != 0:
        raiseError('Parallel calculation is not supported.')

    Nw_local = chi0_wGG.shape[0]
    npw = chi0_wGG.shape[1]
            
    dm_wGG = np.zeros((Nw_local, npw, npw), dtype = complex)
    Kxc_GG = np.zeros((npw, npw))
    tmp_GG = np.eye(npw, npw)
                
    dm0_wGG = np.zeros_like(dm_wGG)
    for iw in range(Nw_local):
        dm0_wGG[iw] = tmp_GG - Kc_GG * chi0_wGG[iw]

    dm_wGG = dm0_wGG.copy()

    dminv_wGG0 = np.linalg.inv(dm_wGG[0])

    error = 1.
    BS_iteration = 0

    while error > 1E-5:
        BS_iteration += 1
        tmp_dminv_wGG0 = dminv_wGG0

        # Bootstrap approximation for xc kernel
        # http://arxiv.org/abs/1107.0199
        Kxc_GG = -np.dot(dminv_wGG0, Kc_GG) * np.linalg.inv(dm0_wGG[0] - tmp_GG)

        A_wGG = chi0_wGG.copy()
        for iw in range(Nw_local):
            A_wGG[iw] = np.dot(chi0_wGG[iw], np.linalg.inv(tmp_GG - np.dot(Kxc_GG, chi0_wGG[iw])))
        for iw in range(Nw_local):
            dm_wGG[iw] = tmp_GG - Kc_GG * A_wGG[iw]

        dminv_wGG0 = np.linalg.inv(dm_wGG[0])

        if np.mod(BS_iteration,10) == 1:
            printtxt('Bootstrap iteration: %d, error: %f ' % (BS_iteration,error))
        if BS_iteration > 200:
            raiseError('Not converged!')

        error = abs(tmp_dminv_wGG0 - dminv_wGG0).sum()

    df_w = np.zeros(Nw_local, dtype=complex)
    for iw in range(Nw_local):
        tmp_GG = dm_wGG[iw]
        df_w[iw] = tmp_GG[0, 0]

    return df_w
