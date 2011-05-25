def agts(queue):
    bulk = queue.add('bulk.py', ncpus=4, walltime=6)
    surf = queue.add('surface.py', ncpus=4, walltime=6)
    queue.add('fig2.py', deps=surf, creates='fig2.png')
    queue.add('sigma.py', deps=[bulk, surf])
