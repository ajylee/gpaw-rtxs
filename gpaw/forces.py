import numpy as np


class ForceCalculator:
    def __init__(self, timer):
        self.timer = timer
        self.reset()
        
    def reset(self):
        self.F_av = None

    def calculate(self, wfs, dens, ham):
        """Return the atomic forces."""

        if self.F_av is not None:
            return self.F_av

        natoms = len(wfs.setups)
        self.F_av = np.zeros((natoms, 3))

        # Force from projector functions (and basis set):
        wfs.calculate_forces(ham, self.F_av)
        
        try:
            # ODD functionals need force corrections for each spin
            correction = ham.xc.setup_force_corrections
        except AttributeError:
            pass
        else:
            correction(self.F_av)
        
        if wfs.band_comm.rank == 0 and wfs.kpt_comm.rank == 0:
            ham.calculate_forces(dens, self.F_av)

        wfs.world.broadcast(self.F_av, 0)
        
        # Add non-local contributions:
        for kpt in wfs.kpt_u:
            pass#XXXself.F_av += hamiltonian.xcfunc.get_non_local_force(kpt)
    
        self.F_av = wfs.symmetry.symmetrize_forces(self.F_av)

        return self.F_av
