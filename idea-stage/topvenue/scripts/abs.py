import re,html,json,sys,os,time,urllib.request,hashlib
HOST={'ICLR2026':'https://iclr.cc','ICML2026':'https://icml.cc','NeurIPS2026':'https://neurips.cc','NeurIPS2025':'https://neurips.cc','ECCV2026':'https://eccv.ecva.net','ICCV2025':'https://iccv.thecvf.com','CVPR2026':'https://cvpr.thecvf.com'}
os.makedirs('abscache',exist_ok=True)
def get_abs(r):
    if r.get('abstract'): return r['abstract']
    v=r['venue']
    if v.startswith('ACL') or v.startswith('EMNLP'):
        url=r['url']
    elif v=='CVPR2026' and 'openaccess' in (r.get('url') or ''):
        url=r['url']
    else:
        if not r.get('vurl'): return ''
        url=HOST[v]+r['vurl']
    f='abscache/'+hashlib.md5(url.encode()).hexdigest()
    if os.path.exists(f): s=open(f,encoding='utf-8',errors='ignore').read()
    else:
        try:
            s=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=30).read().decode('utf-8','ignore')
        except Exception as e: return 'ERR '+str(e)
        open(f,'w').write(s); time.sleep(0.3)
    m=re.search(r'id="abstractText"[^>]*>(.*?)</div>',s,re.S) or re.search(r'<div class="card-body acl-abstract">(.*?)</div>',s,re.S) or re.search(r'<div id="abstract"\s*>(.*?)</div>',s,re.S)
    if not m: return ''
    return re.sub(r'\s+',' ',html.unescape(re.sub('<[^>]+>',' ',m.group(1)))).strip()
if __name__=='__main__':
    pat=re.compile(sys.argv[1],re.I); vf=sys.argv[2] if len(sys.argv)>2 else ''
    seen=set()
    for l in open('all.jsonl'):
        r=json.loads(l)
        if not pat.search(r['title']) or (vf and vf not in r['venue']): continue
        k=(r['venue'],r['title'].lower())
        if k in seen: continue
        seen.add(k)
        inst='; '.join([i for i in dict.fromkeys(r.get('inst',[])) if i][:3])
        print(f"### [{r['venue']}] {r['title']} | {inst}\n{get_abs(r)[:1500]}\n")
