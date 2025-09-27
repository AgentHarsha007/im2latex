def tokens_to_latex(tokens: list[str]) -> str:
    """
    Reconstruct a LaTeX string from linearized tokens.
    """
    out = []
    stack = []
    for tok in tokens:
        if tok.startswith("[/"):  # closing tag
            name = tok[2:-1]  # e.g. '/FRAC' -> 'FRAC'
            if name in ("ARG", "GROUP"):
                out.append("}")
            stack.pop()
        elif tok.startswith("["):  # opening tag
            name = tok[1:-1]
            stack.append(name)
            if name == "ARG" or name == "GROUP":
                out.append("{")
            else:  # macro like [FRAC]
                out.append("\\" + name.lower())
        else:
            out.append(tok)

    return "".join(out)