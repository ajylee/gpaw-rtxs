# creates: citations.png citations.csv

import datetime

import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

months = [datetime.date(2000, m, 1).strftime('%B')[:3].upper()
          for m in range(1, 13)]


def f(filename):
    papers = []
    nodoi = 0
    lines = open(filename).readlines()
    n = 0
    while n < len(lines):
        line = lines[n]
        if line[0] == '@':
            title = day = month = year = doi = None
        elif line.startswith('Title'):
            title = line.split('{{')[1]
            while '}}' not in title:
                title += lines[n + 1]
                n += 1
            title = ' '.join(title.split('}}')[0].split())
        elif line.startswith('Year'):
            year = int(line.split('{{')[1].split('}}')[0])
        elif line.startswith('Month'):
            month = line.split('{{')[1].split('}}')[0]
            if ' ' in month:
                month, day = month.split()
                day = int(day)
            else:
                day = 15
            month = months.index(month) + 1
        elif line.startswith('DOI'):
            doi = line.split('{{')[1].split('}}')[0]
        elif line[0] == '}':
            if month is None:
                month, day = 6, 15
            date = datetime.date(year, month, day)
            if doi is None:
                doi = filename + str(nodoi)
                nodoi += 1
                print title
            papers.append((date, doi, title))
        n += 1

    papers.sort()
    return papers

plt.figure(figsize=(10, 5))
total = {}
for bib in ['gpaw1', 'tddft', 'gpaw2', 'response']:
    papers = f(bib + '.bib')
    plt.plot([paper[0] for paper in papers], range(1, len(papers) + 1),
             '-o', label=bib)
    for date, doi, title in papers:
        total[doi] = (date, title)
    x=dict([(p[1],0) for p in papers])
    print bib, len(papers), len(x), len(total)

allpapers = [(paper[0], doi, paper[1]) for doi, paper in total.items()]
allpapers.sort()
plt.plot([paper[0] for paper in allpapers], range(1, len(allpapers) + 1),
             '-o', label='total')

fd = open('citations.csv', 'w')
n = len(allpapers)
for date, doi, title in allpapers[::-1]:
    fd.write('%d,"`%s <http://dx.doi.org/%s>`__"\n' % (n, title, doi))
    n -= 1
fd.close()

plt.xlabel('date')
plt.ylabel('number of citations')
plt.legend(loc='upper left')
plt.savefig('citations.png')
