import json,re,sys
rows=[json.loads(l) for l in open('all.jsonl')]
def q(p,label,abstract=False):
    P=re.compile(p,re.I); seen=set(); out=[]
    for r in rows:
        t=r['title']; txt=t+(' '+r.get('abstract','') if abstract else '')
        if P.search(txt) and (r['venue'],t.lower()) not in seen:
            seen.add((r['venue'],t.lower())); out.append(f"  [{r['venue']}] {t[:130]} | {'; '.join([i[:25] for i in dict.fromkeys(r.get('inst',[])) if i][:2])}")
    print(f"== {label}: {len(out)}"); print('\n'.join(out))
if __name__=='__main__':
    q(sys.argv[1], sys.argv[1][:40], len(sys.argv)>2)
