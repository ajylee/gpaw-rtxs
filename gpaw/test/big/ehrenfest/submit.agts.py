
def agts(queue):
    h2_osc = queue.add('h2_osc.py', ncpus=8, walltime=2*60)
    n2_osc = queue.add('n2_osc.py --state-parallelization=5',
                        ncpus=5*8, walltime=8*60)
    na2_osc = queue.add('na2_osc.py', ncpus=8, walltime=15*60)
    queue.add('oscfit.py', ncpus=1, walltime=5, deps=[h2_osc, n2_osc, na2_osc],
              creates=['h2_osc.png', 'n2_osc.png', 'na2_osc.png'])

