# app/core/tools.py
import re
import sympy as sp

def maybe_calc(q: str) -> str | None:
    # trigger if it looks like a calc request
    if re.search(r"\bsolve\b|=|\^|\d+[\+\-\*\/]\d+", q):
        try:
            expr = sp.sympify(q.replace("^","**"))
            res = sp.N(expr)
            return f"{q} = {res}"
        except Exception:
            return None
    return None
