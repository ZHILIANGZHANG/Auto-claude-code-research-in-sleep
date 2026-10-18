import json,sys
h=json.load(open('hits.json'))
k=sys.argv[1]; excl=[e for e in sys.argv[2:] if not e.startswith('+')]; inc=[e[1:] for e in sys.argv[2:] if e.startswith('+')]
seen=set()
for r in sorted(h[k],key=lambda r:r['venue']):
    if any(e in r['venue'] for e in excl): continue
    if inc and not any(i in r['venue'] for i in inc): continue
    key=(r['venue'],r['title'].lower())
    if key in seen: continue
    seen.add(key)
    inst=[i[:28] for i in dict.fromkeys(r['inst']) if i][:3]
    print(f"[{r['venue'][:7]}] {r['title'][:120]} | {'; '.join(inst)}")
