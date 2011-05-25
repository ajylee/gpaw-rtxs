#This module is used to store and read some temperary data.
import cPickle
import os
from gpaw.mpi import world
from gpaw.transport.tools import collect_atomic_matrices

class Transport_IO:
    def __init__(self, kpt_comm, domain_comm):
        self.kpt_comm = kpt_comm
	self.domain_comm = domain_comm
	assert self.kpt_comm.size * self.domain_comm.size == world.size
	self.dir_name = 'temperary_data'
        if world.rank == 0:
            if not os.access(self.dir_name, os.F_OK):
                os.mkdir(self.dir_name)
        world.barrier()
        self.filenames = self.default_temperary_filenames()

    def default_temperary_filenames(self):
        filenames = {}
	for name in ['Lead']:
	    filenames[name] = name
	for i in range(self.kpt_comm.size):
	    for j in range(self.domain_comm.size):
	        name = 'KC_' + str(i) + '_DC_' + str(j) +'AD'
		#Local analysis data on kpt_comm i and domain_comm j
		filenames[name] = name
	return filenames

    def read_data(self, filename=None, option='Analysis'):
        if option == 'Lead':
	    if filename is None:
	        filename = self.filenames[option]
  	    fd = file(self.dir_name + '/' + filename, 'r')
	    data = cPickle.load(fd)
	    fd.close()
	return data
           
    def save_data(self, obj, filename=None, option='Analysis'):
        #       option ------map------ obj
        #       Analysis            Transport.Analysor
	#        Lead                Lead_Calc
        data = self.collect_data(obj, option)
	if option == 'Lead':
	    if world.rank == 0:
	        if filename is None:
		    filename = self.filenames[option]
  	        fd = file(self.dir_name + '/' + filename, 'wb')
		cPickle.dump(data, fd, 2)
		fd.close()
	elif option == 'Analysis':
	    name = 'KC_' + str(self.kpt_comm.rank) + '_DC_' + \
	                                     str(self.domain_comm.rank) +'AD'
	    fd = file(self.dir_name + '/' + self.filenames[name], 'wb')
	    cPickle.dump(data, fd, 2)
	    fd.close()
	else:
	    raise NotImplementError()
	
    def collect_data(self, obj, option):
        if option == 'Lead':
	    data = self.collect_lead_data(obj)
	elif option == 'Analysis':
	    data = self.collect_analysis_data(obj)
	else:
	    raise NotImplementError
	return data

    def collect_lead_data(self, obj): 	
        data = {}
        data['bzk_kc'] = obj.wfs.bzk_kc
        #data['bzk_qc'] = obj.wfs.bzk_qc
        data['ibzk_kc'] = obj.wfs.ibzk_kc
        data['ibzk_qc'] = obj.wfs.ibzk_qc
        #data['setups'] = obj.wfs.setups
	data['cell_cv'] = obj.gd.cell_cv
	data['parsize_c'] = obj.gd.parsize_c
	data['pbc_c'] = obj.gd.pbc_c
        data['N_c'] = obj.gd.N_c
	data['nao'] = obj.wfs.setups.nao
        data['fine_N_c'] = obj.finegd.N_c
        data['nspins'] = obj.wfs.nspins
	data['fermi'] = obj.get_fermi_level()

        den, ham = obj.density, obj.hamiltonian
        gd = obj.gd
        vt_sG = gd.collect(ham.vt_sG)
        nt_sG = gd.collect(den.nt_sG)
        data['vt_sG'] = vt_sG
        data['nt_sG'] = nt_sG

        finegd = obj.finegd
        vt_sg = finegd.collect(ham.vt_sg)
        nt_sg = finegd.collect(den.nt_sg)
        vHt_g = finegd.collect(ham.vHt_g)
        rhot_g = finegd.collect(den.rhot_g)
        data['vt_sg'] = vt_sg
        data['nt_sg'] = nt_sg
        data['vHt_g'] = vHt_g
        data['rhot_g'] = rhot_g

        D_asp = collect_atomic_matrices(den.D_asp, den.setups,
                                        den.nspins, den.gd.comm,
            			    den.rank_a)
        dH_asp = collect_atomic_matrices(ham.dH_asp, ham.setups,
                                         ham.nspins, ham.gd.comm,
            			     ham.rank_a)
        data['D_asp'] = D_asp
        data['dH_asp'] = dH_asp
        return data

    def collect_analysis_data(self, obj):
        return obj.data

        
            

	       
	    
        

