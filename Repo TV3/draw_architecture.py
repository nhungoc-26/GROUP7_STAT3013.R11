import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
# (label, real size, nodes drawn)
L=[("Input\n8 features",8,8),("Hidden 1\n64 · tanh",64,9),("Hidden 2\n32 · tanh",32,8),("Hidden 3\n16 · tanh",16,7),("Output\n1 · sigmoid",1,1)]
xs=[0.08,0.30,0.52,0.74,0.93]
fig,ax=plt.subplots(figsize=(10,4.4)); ax.axis("off"); pos=[]
for (name,n,k),x in zip(L,xs):
    ys=[0.52+(i-(k-1)/2)*0.075 for i in range(k)]; pos.append([(x,y) for y in ys])
    ax.text(x,0.05,name,ha="center",fontsize=10)
    if n>k: ax.text(x,0.93,f"({k} of {n} shown)",ha="center",fontsize=8,color="#555")
for a,b in zip(pos,pos[1:]):
    for p in a:
        for q in b: ax.plot([p[0],q[0]],[p[1],q[1]],color="#c8c8c8",lw=0.35,zorder=1)
for l,c in zip(pos,["#4C78A8","#F58518","#F58518","#F58518","#54A24B"]):
    for p in l: ax.scatter(*p,s=220,color=c,edgecolor="k",zorder=2)
ax.text(0.93,0.64,"P(High)",ha="center",fontsize=9)
ax.text(0.08,0.98,"Nurse, Doctor, Medication, Discharge,\nClean, Quiet, Rating, Recommend",ha="center",va="top",fontsize=7.5)
ax.set_xlim(0,1);ax.set_ylim(0,1)
plt.tight_layout();import os; os.makedirs("results_mlp",exist_ok=True); plt.savefig("results_mlp/mlp_architecture.png",dpi=200)
