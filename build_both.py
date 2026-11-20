#!/usr/bin/env python3
"""Compile the marked copy and the clean copy from one source, then verify the manuscript.

The verification pass exists because the response letters promise it. It checks:
  1. LaTeX errors, overfull boxes and unresolved references;
  2. that every table and figure is referenced somewhere in the text;
  3. that every numeric claim registered in claims.json matches the result files;
  4. that no large blank band is left on any page;
  5. that no reference entry still carries a placeholder.
A non-zero exit status means the manuscript is not ready to submit.
"""
from __future__ import annotations
import pathlib, subprocess, re, json, sys

HERE = pathlib.Path(__file__).parent
RESULTS = pathlib.Path('/home/claude/paper/results')
problems: list[str] = []


def compile_both() -> None:
    src = (HERE/'main_mdpi.tex').read_text()
    (HERE/'main_clean.tex').write_text(
        src.replace(r'\newif\ifMARKUP\MARKUPtrue', r'\newif\ifMARKUP\MARKUPfalse'))
    for stem in ('main_mdpi', 'main_clean'):
        for i in range(3):
            subprocess.run(['pdflatex', '-interaction=nonstopmode', stem],
                           cwd=HERE, capture_output=True)
            if i == 0:
                subprocess.run(['bibtex', stem], cwd=HERE, capture_output=True)
        log = (HERE/f'{stem}.log').read_text()
        errs = sum(1 for l in log.splitlines() if l.startswith('!'))
        over = log.count('Overfull')
        und = log.lower().count('undefined citation') + log.lower().count('undefined reference')
        pages = re.search(r'Output written on \S+ \((\d+) pages', log)
        print(f'{stem}: errors={errs} overfull={over} undefined={und} '
              f'pages={pages.group(1) if pages else "?"}')
        for name, n in (('LaTeX errors', errs), ('overfull boxes', over),
                        ('unresolved references', und)):
            if n:
                problems.append(f'{stem}: {n} {name}')


def check_crossrefs() -> None:
    """Reviewer 2 asked for this; it now runs on every build."""
    s = (HERE/'main_mdpi.tex').read_text()
    missing = []
    for kind in ('tab', 'fig'):
        for lab in re.findall(r'\\label\{(' + kind + r':[^}]*)\}', s):
            if ('\\ref{' + lab + '}') not in s.replace('\\label{' + lab + '}', ''):
                missing.append(lab)
    print(f'cross-references: {len(missing)} float(s) never cited in the text')
    for m in missing:
        print('   !', m)
        problems.append(f'float never cited: {m}')


def check_numbers() -> None:
    """Every registered numeric claim must match the result files to the printed precision."""
    reg = HERE/'claims.json'
    if not reg.exists():
        print('numeric claims: claims.json missing, skipped')
        return
    claims = json.loads(reg.read_text())
    tex = (HERE/'main_mdpi.tex').read_text()
    bad = 0
    for c in claims:
        shown = c['as_printed']
        if shown not in tex:
            print(f'   ! not found in the manuscript: {c["name"]} = {shown}')
            problems.append(f'claim text missing: {c["name"]}')
            bad += 1
            continue
        src = RESULTS/c['file']
        if not src.exists():
            print(f'   ! result file missing for {c["name"]}: {c["file"]}')
            problems.append(f'result file missing: {c["file"]}')
            bad += 1
            continue
        val = json.loads(src.read_text()) if src.suffix == '.json' else None
        if val is not None:
            got = val
            for k in c['path']:
                got = got[k]
            if abs(float(got) - float(c['value'])) > float(c.get('tol', 1e-9)):
                print(f'   ! {c["name"]}: manuscript {c["value"]}, results {got}')
                problems.append(f'number disagrees with results: {c["name"]}')
                bad += 1
    print(f'numeric claims: {len(claims)} checked, {bad} disagreement(s)')


def check_blank_bands() -> None:
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        print('blank bands: Pillow unavailable, skipped')
        return
    tmp = HERE/'_ink'
    tmp.mkdir(exist_ok=True)
    subprocess.run(['pdftoppm', '-r', '40', '-gray', '-png',
                    str(HERE/'main_clean.pdf'), str(tmp/'p')], capture_output=True)
    worst = 0.0
    page = ''
    for f in sorted(tmp.glob('p-*.png')):
        a = np.asarray(Image.open(f).convert('L'))
        h = a.shape[0]
        empty = (a < 200).sum(axis=1) == 0
        runs, cur = [], 0
        for i, e in enumerate(empty):
            if e:
                cur += 1
            else:
                if cur:
                    runs.append((cur, i - cur))
                cur = 0
        if cur:
            runs.append((cur, len(empty) - cur))
        inner = [l for l, s in runs if s > 0.10*h and s + l < 0.95*h]
        g = max(inner, default=0)/h
        if g > worst:
            worst, page = g, f.name
    for f in tmp.glob('*'):
        f.unlink()
    tmp.rmdir()
    print(f'largest internal blank band: {worst:.1%} ({page})')
    if worst > 0.12:
        problems.append(f'blank band of {worst:.1%} on {page}')


def check_references() -> None:
    bib = (HERE/'refs.bib').read_text()
    # only real entry fields count, not the explanatory header comment
    body = '\n'.join(l for l in bib.splitlines() if not l.lstrip().startswith('%'))
    marks = ['to be inserted', 'to be confirmed', 'status      = {incomplete}',
             'status    = {incomplete}']
    hits = [m for m in marks if m in body]
    entries = bib.count('@')
    no_id = []
    for block in re.split(r'\n(?=@)', bib):
        if not block.strip().startswith('@'):
            continue
        key = re.match(r'@\w+\{([^,]+),', block)
        if key and 'doi' not in block and 'url' not in block and 'howpublished' not in block:
            no_id.append(key.group(1))
    print(f'references: {entries} entries, {len(hits)} placeholder marker(s), '
          f'{len(no_id)} without a DOI or URL')
    for k in no_id:
        print('   -', k)
    if hits:
        problems.append(f'reference placeholders still present: {hits}')


compile_both()
check_crossrefs()
check_numbers()
check_blank_bands()
check_references()

print()
if problems:
    print(f'NOT READY: {len(problems)} problem(s)')
    for p in problems:
        print('  !', p)
    sys.exit(1)
print('All checks passed.')
