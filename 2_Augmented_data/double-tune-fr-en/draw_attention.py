#%%
import matplotlib.pyplot as plt
import seaborn
import numpy as np
seaborn.set_context(context="talk")
%matplotlib inline

#%%
def draw(data, x, y):
    sns_plot = seaborn.heatmap(data, 
                    xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0)
    sns_plot.figure.set_size_inches(15, 15)
    sns_plot.figure.savefig("output.svg", format="svg",
                            dpi=1200, transparent=True, bbox_inches='tight', pad_inches=0)


#%%
file = "att.txt"
with open(file, "r") as f:
    temp = f.readlines()

#%%
len(temp)


#%%
matrix = []
for line in temp:
    tokens = line.split()
    tmp = []
    for token in tokens[1:]:
        if '*' in token:
            token = token.replace('*','')
        tmp.append(token)
    matrix.append(tmp)



#%%
src = "A@@ aa@@ aa@@ aa@@ ah .... 8 ans après , je viens de percu@@ ter .... : o &apos; tain je me dis@@ ais bien que je pass@@ ais à côté d&apos; un tru@@ c vu les up@@ votes .".split()
tgt = "A@@ a@@ a@@ a@@ a@@ a@@ a@@ a@@ ah .... 8 years later , I just hit ... : o &apos; f@@ uck I was saying that I was missing something given the up@@ votes .".split()
matrix = np.array(matrix, dtype=np.float)
#%%
seaborn.set(font_scale=1.1, context='notebook', style='whitegrid',
            palette='deep', font='sans-serif', color_codes=False, rc=None)
draw(matrix,src,tgt)

#%%
