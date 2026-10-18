import json,re,collections,sys
B={
'kv_cache': r'\bkv\b|kv[- ]cache|key[- ]value cache',
'spec_decode': r'speculative|draft model|lookahead decoding|parallel decoding',
'quant_prune_llm': r'quantiz|low[- ]bit|\bw4a|\b[0-9]-bit|post-training (quantization|pruning)|(llm|language model).{0,40}prun|prun.{0,40}(llm|language model)|sparsity.{0,30}(llm|language)',
'sparse_attn_longctx': r'sparse attention|long[- ]context|context window|length extrapolation|prefill|attention sink',
'moe_infer': r'mixture[- ]of[- ]experts|\bmoe\b|expert (pruning|skipping|merging|offload)',
'early_exit_layer': r'early[- ]exit|layer skip|skipping layers|dynamic depth|adaptive computation',
'mllm_token': r'(visual|vision|image|video) tokens?|token (pruning|reduction|compression|merging|dropping)|tokens? (pruning|reduction|compression|merging)',
'tts': r'test[- ]time (scal|compute)|inference[- ]time (scal|compute)|best[- ]of[- ]n|self[- ]consistency|majority vot|parallel (thinking|reasoning|sampling)|compute[- ]optimal',
'efficient_reasoning': r'overthink|efficient reasoning|reasoning (length|budget|efficien)|(concise|shorter|adaptive|compress).{0,20}(reasoning|chain[- ]of[- ]thought|cot)|thinking (budget|length|token)|early (stop|exit).{0,20}(reason|think)|underthink',
'continual': r'continual|lifelong|catastrophic forgetting|forgetting|incremental learning',
'merging': r'model merging|merging models|task arithmetic|task vector|weight interpolation|model soup',
'editing': r'knowledge editing|model editing|editing (knowledge|llm|language)|unlearning',
'tta': r'test[- ]time (adaptation|training|learning)',
'steering': r'steering|activation (addition|engineering|patching)|representation engineering|refusal direction|concept vector|sparse autoencoder|\bsae\b',
'halluc': r'hallucinat|contrastive decoding|faithful',
'uncertainty': r'uncertainty|calibrat|confidence estimation|semantic entropy|\bprob(e|ing)\b',
'training_free': r'training[- ]free|tuning[- ]free|plug[- ]and[- ]play|zero[- ]shot.{0,20}(acceleration|compression)',
'dllm': r'diffusion (language|llm)|masked diffusion (language|model)|discrete diffusion.{0,30}(language|text)|\bdllm',
}
P={k:re.compile(v,re.I) for k,v in B.items()}
rows=[json.loads(l) for l in open('all.jsonl')]
def vgroup(v): return v
tab=collections.defaultdict(collections.Counter)
hits=collections.defaultdict(list)
for r in rows:
    t=r['title']
    for k,p in P.items():
        if p.search(t): tab[k][r['venue']]+=1; hits[k].append(r)
vens=['ICLR2026','NeurIPS2025','ICML2026','NeurIPS2026','CVPR2026','ICCV2025','ECCV2026','ACL2026','EMNLP2025','ACL2026-Findings','EMNLP2025-Findings']
print('bucket'.ljust(20)+''.join(v[:9].rjust(10) for v in vens)+'   total')
for k in B: print(k.ljust(20)+''.join(str(tab[k][v]).rjust(10) for v in vens)+'   '+str(sum(tab[k].values())))
json.dump({k:[dict(venue=r['venue'],title=r['title'],inst=r.get('inst',[]),authors=r.get('authors',[])[:6],url=r.get('url'),vurl=r.get('vurl'),abstract=r.get('abstract','')) for r in v] for k,v in hits.items()},open('hits.json','w'),ensure_ascii=False)
