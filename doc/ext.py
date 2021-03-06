# -*- coding: utf-8 -*-
import os
from sys import executable
import types
from os.path import join
from stat import ST_MTIME
from docutils import nodes, utils
from docutils.parsers.rst.roles import set_classes

def mol_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    n = []
    t = ''
    while text:
        if text[0] == '_':
            n.append(nodes.Text(t))
            t = ''
            n.append(nodes.subscript(text=text[1]))
            text = text[2:]
        else:
            t += text[0]
            text = text[1:]
    n.append(nodes.Text(t))
    return n, []

def svn_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    if text[-1] == '>':
        i = text.index('<')
        name = text[:i - 1]
        text = text[i + 1:-1]
    else:
        name = text
        if name[0] == '~':
            name = name.split('/')[-1]
            text = text[1:]
        if '?' in name:
            name = name[:name.index('?')]
    ref = 'http://svn.fysik.dtu.dk/projects/gpaw/branches/general_unit_cells/' + text
    ref = 'http://svn.fysik.dtu.dk/projects/gpaw/trunk/' + text
    set_classes(options)
    node = nodes.reference(rawtext, name, refuri=ref,
                           **options)
    return [node], []

def trac_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    if text[-1] == '>':
        i = text.index('<')
        name = text[:i - 1]
        text = text[i + 1:-1]
    else:
        name = text
        if name[0] == '~':
            name = name.split('/')[-1]
            text = text[1:]
        if '?' in name:
            name = name[:name.index('?')]
    ref = 'http://trac.fysik.dtu.dk/projects/gpaw/browser/branches/general_unit_cells/' + text
    ref = 'http://trac.fysik.dtu.dk/projects/gpaw/browser/trunk/' + text
    set_classes(options)
    node = nodes.reference(rawtext, name, refuri=ref,
                           **options)
    return [node], []

def epydoc_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    name = None
    if text[-1] == '>':
        i = text.index('<')
        name = text[:i - 1]
        text = text[i + 1:-1]

    components = text.split('.')
    if components[0] != 'gpaw':
        components.insert(0, 'gpaw')

    if name is None:
        name = components[-1]

    try:
        module = None
        for n in range(2, len(components) + 1):
            module = __import__('.'.join(components[:n]))
    except ImportError:
        if module is None:
            print 'epydoc: could not process: %s' % str(components)
            raise
        for component in components[1:n]:
            module = getattr(module, component)
            ref = '.'.join(components[:n])
            if isinstance(module, (type, types.ClassType)):
                ref += '-class.html'
            else:
                ref += '-module.html'
        if n < len(components):
            ref += '#' + components[-1]
    else:
        ref = '.'.join(components) + '-module.html'

    ref = 'http://wiki.fysik.dtu.dk/gpaw/epydoc/' + ref
    set_classes(options)
    node = nodes.reference(rawtext, name,
                           refuri=ref,
                           **options)
    return [node], []

def setup(app):
    app.add_role('mol', mol_role)
    app.add_role('svn', svn_role)
    app.add_role('trac', trac_role)
    app.add_role('epydoc', epydoc_role)
    #import atexit
    #atexit.register(fix_sidebar)
    create_png_files()

def create_png_files():
    for dirpath, dirnames, filenames in os.walk('.'):
        for filename in filenames:
            if filename.endswith('.py'):
                path = join(dirpath, filename)
                lines = open(path).readlines()
                try:
                    line = lines[0]
                except IndexError:
                    continue
                if 'coding: utf-8' in line:
                    line = lines[1]
                if line.startswith('# creates:'):
                    t0 = os.stat(path)[ST_MTIME]
                    run = False
                    for file in line.split()[2:]:
                        try:
                            t = os.stat(join(dirpath, file))[ST_MTIME]
                        except OSError:
                            run = True
                            break
                        else:
                            if t < t0:
                                run = True
                                break
                    if run:
                        print 'running:', path
                        e = os.system('cd %s; %s %s' % (dirpath, executable,
                                                        filename))
                        if e != 0:
                            raise RuntimeError('FAILED!')
                        for file in line.split()[2:]:
                            print dirpath, file
        if '.svn' in dirnames:
            dirnames.remove('.svn')
