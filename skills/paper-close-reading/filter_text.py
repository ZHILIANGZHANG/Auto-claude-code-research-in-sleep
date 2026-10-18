# usage: python3 -I filter_text.py PAPER.txt [body|app|all]
# Prints pdftotext output with short noise lines (figure fragments, table digits, latexit) removed.
import sys, re

path = sys.argv[1]
part = sys.argv[2] if len(sys.argv) > 2 else 'body'
L = open(path, encoding='utf-8', errors='ignore').read().replace('\x00', '').split('\n')

ref = None
for i, l in enumerate(L):
    if re.match(r'^\s*(References|REFERENCES|Bibliography)\s*$', l) and i > len(L) * 0.25:
        ref = i
        break
if ref is None:
    ref = len(L)
app = None
for i in range(ref, len(L)):
    if re.match(r'^\s*(Appendix|APPENDIX|A\s+[A-Z][a-z]|A\.\s*[A-Z]|Supplementary)', L[i]):
        app = i
        break


def emit(a, b, minlen):
    buf = []
    for l in L[a:b]:
        s = l.strip()
        if 'latexit' in s:
            continue
        if len(s) < minlen and not re.match(r'^(\d+(\.\d+)*|[A-Z](\.\d+)*)\s+[A-Z]', s):
            continue
        if re.match(r'^[\d\.\s%×x\-–]+$', s):
            continue
        buf.append(s)
    print('\n'.join(buf))


if part in ('body', 'all'):
    emit(0, ref, 30)
if part in ('app', 'all') and app:
    emit(app, len(L), 30)
print(f'\n[lines total={len(L)} refs@{ref} appendix@{app}]  (appendix detection is heuristic; grep the raw text if it is None)')
