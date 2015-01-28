

def orthonormalize(calc, unocc_photoelectron_indices):

    idx = unocc_photoelectron_indices

    copy = [kpt.psit_nG.copy() for kpt in calc.wfs.kpt_u]

    calc.wfs.orthonormalize()

    for copy_psit_nG, calc_kpt in zip(copy, calc.wfs.kpt_u):
        calc_kpt.psit_nG[idx] = copy_psit_nG[idx]
