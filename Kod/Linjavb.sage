# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,sage:percent
#     text_representation:
#       extension: .sage
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: SageMath 10.4
#     language: sage
#     name: sagemath
# ---

# %%

# %% [markdown]
# # Linjära avbildningar på följder

# %% [markdown]
# ## En vektor

# %%
n=15
u = vector([j^2/10 for j in range(n)])
u

# %%

# %%
list_plot(u)

# %%

# %% [markdown]
# ## Linjär avbildning : skifta vänster

# %%
def ShiftDown(v):
    return vector(list(v[1:]) + [0])
    

# %%

# %%
ShiftDown(u)

# %%
V = u.parent()
V

# %%
for bv in V.basis():
    print(bv, ShiftDown(bv))

# %% [markdown]
# ### Dess avbildningsmatris

# %%
M = matrix(QQ,n,n,[ShiftDown(bv) for bv in V.basis()]).transpose()
M

# %%
M*u

# %%
M^2*u

# %% [markdown]
#

# %% [markdown]
# ### Itererade shift

# %%
add([list_plot(M**k*u, plotjoined=True, color = (k/n,0,1-k/n)) for k in range(n//2)])

# %% [markdown]
# ## Diskret derivering, differensoperator

# %%
def IdentityOp(v):
    return v

# %%
def DiscreteDiffOp(v):
    return IdentityOp(v) - ShiftDown(v)

# %%
DiscDiff = matrix(QQ,n,n,[DiscreteDiffOp(bv) for bv in V.basis()]).transpose()
DiscDiff

# %% [markdown]
# ### Matrisen till andraderivatan

# %%
DiscDiff**2

# %%
graphics_array([list_plot(u), list_plot(DiscDiff*u), list_plot(DiscDiff**2*u)])

# %% [markdown]
# ### Invers till derivering är integrering

# %%
DiscDiff.det()

# %%
DiscInt = DiscDiff.inverse()
DiscInt

# %%
u, DiscDiff*u

# %%
DiscInt*DiscDiff*u

# %%
## Vi löser en differentialekvation

# %%
sinvector = vector([sin(k*2*pi/n).numerical_approx(digits=5) for k in range(n)])
list_plot(sinvector)

# %%
list_plot(DiscDiff*sinvector)

# %% [markdown]
# #### y'' + y = 0, y(0) = 0, y'(0) = 1, det är sinus

# %%
DiffOp = DiscDiff**2 + identity_matrix(n)
DiffOp

# %%
DiffOp*sinvector

# %%
list_plot(sinvector,color='black') +list_plot(-DiscDiff**2*sinvector, color='red')

# %% [markdown]
# #### Vi glömde bort kedjeregeln

# %%
diff(sin(x*2*pi/n),x), diff(sin(x*2*pi/n),x,2)

# %%

# %%

# %%

# %%

# %%

# %%
homogen = DiffOp.right_kernel()
homogen

# %%
HLETT = vector([1 for _ in range(n)])
soln = DiffOp.solve_right(HLETT)

# %%
soln

# %%
list_plot(soln)

# %%
DiffOp*soln

# %%
soln2 = DiffOp.solve_right(sinvector)
soln2

# %%
list_plot(sinvector, color='black',plotjoined=True) + list_plot(soln2, color='red')

# %%

# %%

# %% [markdown]
# # Linjära avbildningar på matriser

# %%
n=30
Bild = matrix(n,n, lambda x,y: (sin(x^2/n^2 + 2*y^2/n^2)+x/n).numerical_approx(digits=5) + sage.misc.prandom.gauss(0, 1/n))
matrix_plot(Bild)

# %% [markdown]
# ## Matris till vektor

# %%
vector(Bild)

# %%
matrix_plot(matrix(n,n,vector(Bild)))

# %% [markdown]
# ## Vad är diskret diff på bilden?

# %%
matrix_plot(matrix(n,n,DiscreteDiffOp(vector(Bild))))

# %% [markdown]
# ### Andraderivatan

# %%
matrix_plot(matrix(n,n,DiscreteDiffOp(DiscreteDiffOp(vector(Bild)))))

# %%
def DiscreteDiffOpN(v,k):
    if k == 0:
        return v
    else:
        return DiscreteDiffOpN(DiscreteDiffOp(v), k-1)


# %%
matrix_plot(matrix(n,n,DiscreteDiffOpN(vector(Bild),3)))

# %%
