# P3 hand checks (dev split only)

Judgments made by the agent on 2026-11-05 by reading each printed sample against its gold answer. They feed the amended
scorer invariant (DECISIONS 2026-11-05) through `HANDCHECK` in [`analysis/p3_scorer_agreement.py`](../analysis/p3_scorer_agreement.py);
every count and rate is in the generated [03_scorer_agreement.md](03_scorer_agreement.md).

## Scorer disagreements (main scorer vs second scorers v1 and v2)

**v2 sample (the one the bound uses): no main-scorer error.** Every sampled row is a disagreement in which the main
scorer accepts an answer that is equivalent to the gold and the second scorer rejects it. The forms:

- the same set or tuple in another order or spacing (`1, -1, -2, 2` vs `-2,-1,1,2`; `\pm 3i` vs `3i,-3i`);
- a label before the value (`(a, b) = (7, 49)`, `y(t) = 6t - 5`, `T = (-11, 9)`), or an equation whose right side is
  the answer (`(p - 1)(q - 1) = 1`, `1 + 1 + 2 = 4`);
- a matrix or vector with fractions written as `\frac{a}{b}` against `a/b`, or with a scalar factored out
  (`\frac{8}{49} \begin{pmatrix} 2 \\ 6 \\ 3 \end{pmatrix}` vs `16/49, 48/49, 24/49`);
- equal radicals in another form (`\sqrt{\frac{2}{3}}` vs `\frac{\sqrt{6}}{3}`; `-\frac{1}{\sqrt{3}}` vs `-\frac{\sqrt{3}}{3}`);
- units or `\text{}` in the gold (`25 \, \text{km}`, `\text{E}`), and base subscripts with or without braces (`4210_7`);
- letters in another case on SCLI5 items. Those questions are case-free ("What letter comes after j?", gold `k`;
  "What letter comes before g?", gold `f`), so `K` and `F` are correct;
- two boxed roots that the main scorer joins for a multi-answer gold (`3i, -3i`), where v2 calls the two boxes a
  conflict.

Borderline but accepted: `1, -1, -2, -1, 1, 2` for "all possible values" with gold `-2,-1,1,2`. The values repeat, but
the set is the gold.

**The one disagreement in which the main scorer rejects and v2 accepts**: dnat-mathext-2870, 1.7B, F2. The question
asks for the quotient of a polynomial division. The box holds `x^2 - 2x + 8 \text{ with remainder } -25x + 8`. The
quotient is right, but the box adds the remainder. The main scorer's rejection is strict. v2 accepts only because the
MATH normalization cuts everything after `\text{ ` as a unit. I count it as a main-scorer error in the bound, which is
the conservative choice.

**v1 sample: no main-scorer error either.** The same forms recur, plus currency (`\$13.00` vs `13`, `0.5` vs
`\$0.50`), degrees (`120` vs `120^\circ`), an interval written as a set difference
(`(0, 36) \setminus \{-9, 9\}` vs `(0,9) \cup (9,36)`, equal since -9 is outside (0, 36)), and a matrix identical to the
gold that v1 rejects because it was boxed twice. This sample is drawn from the v1 disagreements, so it is reported
here but not used in the bound.
