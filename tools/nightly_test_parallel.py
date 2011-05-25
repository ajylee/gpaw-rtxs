#!/usr/bin/python
import os
import sys
import time
import glob
import tempfile

def fail(subject, filename='/dev/null'):
    assert os.system(
        'mail -s "%s" gpaw-developers@listserv.fysik.dtu.dk < %s' %
        (subject, filename)) == 0
    raise SystemExit

if '--dir' in sys.argv:
    i = sys.argv.index('--dir')
    sys.argv.pop(i)
    dir = sys.argv.pop(i)
else:
    dir = None

tmpdir = tempfile.mkdtemp(prefix='gpaw-parallel-', dir=dir)
os.chdir(tmpdir)

# Checkout a fresh version and install:
if os.system('svn export ' +
             'https://svn.fysik.dtu.dk/projects/gpaw/trunk gpaw') != 0:
    fail('Checkout of gpaw failed!')
if os.system('svn export ' +
             'https://svn.fysik.dtu.dk/projects/ase/trunk ase') != 0:
    fail('Checkout of ASE failed!')

os.chdir('gpaw')
if os.system('source /home/camp/modulefiles.sh&& ' +
             'module load NUMPY&& ' +
             'module load open64/4.2.3-0&& ' +
             'python setup.py --remove-default-flags ' +
             '--customize=' +
             'doc/install/Linux/Niflheim/el5-xeon-open64-goto2-1.13-acml-4.4.0.py ' +
             'install --home=%s 2>&1 | ' % tmpdir +
             'grep -v "c/libxc/src"') != 0:
    fail('Installation failed!')

os.system('mv ../ase/ase ../lib64/python')

os.system('wget --no-check-certificate --quiet ' +
          'http://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-latest.tar.gz')
os.system('tar xvzf gpaw-setups-latest.tar.gz')
setups = tmpdir + '/gpaw/' + glob.glob('gpaw-setups-[0-9]*')[0]

day = time.localtime()[6]
if '--debug' in sys.argv[1:]:
    args = '--debug'
    cpus = 2 ** (1 + (day+1) % 3)
else:
    args = ''
    cpus = 2 ** (1 + day % 3)
    
# Run test-suite:
print 'Run'
if os.system('source /home/camp/modulefiles.sh; ' +
             'module load NUMPY; ' +
             'module load SCIPY; ' +
             'module load openmpi/1.3.3-1.el5.fys.open64.4.2.3; ' +
             'export PYTHONPATH=%s/lib64/python:$PYTHONPATH; ' % tmpdir +
             'export GPAW_SETUP_PATH=%s; ' % setups +
             'export OMP_NUM_THREADS=1; ' +
             'mpiexec -np %d ' % cpus +
             tmpdir + '/bin/gpaw-python ' +
             'tools/gpaw-test %s >& test.out' % args) != 0:
    fail('Testsuite crashed!', 'test.out')

try:
    failed = open('failed-tests.txt').readlines()
except IOError:
    pass
else:
    # Send mail:
    n = len(failed)
    if n == 1:
        subject = 'One failed test: ' + failed[0][:-1]
    else:
        subject = '%d failed tests: %s, %s' % (n,
                                               failed[0][:-1], failed[1][:-1])
        if n > 2:
            subject += ', ...'
    fail(subject, 'test.out')

print 'Done'
os.system('cd; rm -r ' + tmpdir)
