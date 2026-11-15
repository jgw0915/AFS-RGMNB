"""Render the short professor-facing Markdown report with local XeLaTeX.

Requires XeLaTeX and an installed CJK font; no network or thesis writes. This is a
small intentional Markdown subset (headings, paragraphs, lists, simple tables,
inline emphasis/code/links), not a general-purpose Markdown/LaTeX converter.
Build files remain in ignored outputs/report_build. Run Poppler visual QA after
every substantive report change; successful compilation alone is insufficient.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
import subprocess


def escape_tex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}",
        "$": r"\$", "&": r"\&", "#": r"\#", "%": r"\%",
        "_": r"\_", "^": r"\textasciicircum{}", "~": r"\textasciitilde{}",
    }
    return "".join(replacements.get(char, char) for char in value)


INLINE = re.compile(r"(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\))")


def inline(value: str) -> str:
    parts = []
    for part in INLINE.split(value):
        if part.startswith("`") and part.endswith("`"):
            parts.append(r"{\small\texttt{" + escape_tex(part[1:-1]) + "}}")
        elif part.startswith("**") and part.endswith("**"):
            parts.append(r"\textbf{" + escape_tex(part[2:-2]) + "}")
        elif part.startswith("[") and "](" in part:
            # Readable titles suffice in a short offline report. The full source
            # Markdown retains links; avoid embedding filesystem paths in PDF.
            parts.append(escape_tex(part[1:part.index("](")]))
        else:
            parts.append(escape_tex(part))
    return "".join(parts)


def render_markdown(source: str) -> tuple[str, str]:
    lines = source.splitlines()
    title = "碩士論文研究進度報告"
    body = []
    paragraph = []
    list_kind = None

    def flush() -> None:
        if paragraph:
            body.append(inline(" ".join(paragraph)) + "\n\\par\n")
            paragraph.clear()

    def close_list() -> None:
        nonlocal list_kind
        if list_kind:
            body.append(r"\end{" + list_kind + "}")
            list_kind = None

    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            flush()
            close_list()
            index += 1
            continue
        if line.startswith("# "):
            flush()
            close_list()
            title = line[2:].strip()
        elif line.startswith("## ") or line.startswith("### "):
            flush()
            close_list()
            heading = re.sub(r"^#{2,3}\s+", "", line)
            body.append(r"\section*{" + inline(heading) + "}")
        elif line.startswith("|") and line.endswith("|"):
            flush()
            close_list()
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                row = [item.strip() for item in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", item.replace(" ", "")) for item in row):
                    rows.append(row)
                index += 1
            if rows:
                count = len(rows[0])
                if any(len(row) != count for row in rows):
                    raise ValueError("All Markdown table rows must have the same column count")
                spec = "|".join([r">{\raggedright\arraybackslash}X"] * count)
                body.append(r"\begin{center}\small\renewcommand{\arraystretch}{1.35}")
                body.append(r"\begin{tabularx}{\linewidth}{" + spec + "}\\hline")
                for row_index, row in enumerate(rows):
                    cells = [inline(cell) for cell in row]
                    if row_index == 0:
                        cells = [r"\textbf{" + cell + "}" for cell in cells]
                    body.append(" & ".join(cells) + r" \\ \hline")
                body.append(r"\end{tabularx}\end{center}")
            continue
        elif re.match(r"^(?:[-*] |\d+\. )", line):
            flush()
            kind = "enumerate" if line[0].isdigit() else "itemize"
            if list_kind != kind:
                close_list()
                body.append(r"\begin{" + kind + "}")
                list_kind = kind
            item = re.sub(r"^(?:[-*] |\d+\. )", "", line)
            body.append(r"\item " + inline(item))
        elif line.startswith("```"):
            raise ValueError("Code blocks are not supported in the professor summary")
        else:
            paragraph.append(line)
        index += 1
    flush()
    close_list()
    return title, "\n".join(body)


def build_document(title: str, body: str, font: str) -> str:
    preamble = r"""\documentclass[11pt,a4paper]{article}
\usepackage[top=20mm,bottom=20mm,left=22mm,right=22mm]{geometry}
\usepackage{fontspec,xeCJK}
\usepackage{setspace,enumitem,titlesec,fancyhdr,tabularx,array,xcolor}
\usepackage[unicode,hidelinks]{hyperref}
\setmainfont{Times New Roman}
\setmonofont{Consolas}
\setCJKmainfont{__FONT__}
\setCJKsansfont{__FONT__}
\setCJKmonofont{__FONT__}
\XeTeXlinebreaklocale "zh"
\XeTeXlinebreakskip=0pt plus 1pt
\setstretch{1.22}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlength{\headheight}{15pt}
\emergencystretch=2em
\definecolor{ink}{RGB}{31,46,61}
\definecolor{muted}{RGB}{86,99,112}
\titleformat{\section}{\large\bfseries\color{ink}}{}{0pt}{}
\titlespacing*{\section}{0pt}{12pt}{5pt}
\setlist{leftmargin=1.6em,itemsep=3pt,topsep=3pt,parsep=0pt}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\color{muted}AFS-RGMNB}
\fancyhead[R]{\small\color{muted}國立成功大學資訊管理研究所}
\fancyfoot[L]{\small\color{muted}研究進度與資料驗證}
\fancyfoot[R]{\small\color{muted}\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\renewcommand{\footrulewidth}{0pt}
\hypersetup{pdftitle={__TITLE__},pdfauthor={江冠緯}}
\begin{document}
{\LARGE\bfseries\color{ink}__TITLE__}\par
{\small\color{muted}江冠緯\quad 指導教授：翁慈宗教授\quad 2026 年 11 月 16 日}\par
\vspace{2pt}
__BODY__
\end{document}
"""
    return preamble.replace("__FONT__", escape_tex(font)).replace(
        "__TITLE__", escape_tex(title)
    ).replace("__BODY__", body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--build-dir", type=Path, default=Path("outputs/report_build"))
    parser.add_argument("--font", default="Microsoft JhengHei")
    args = parser.parse_args()
    executable = shutil.which("xelatex")
    if executable is None:
        parser.error("XeLaTeX is not installed or is not on PATH")
    title, body = render_markdown(args.source.read_text(encoding="utf-8"))
    build_dir = (args.build_dir / args.output.stem).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    tex_path = build_dir / "report.tex"
    tex_path.write_text(build_document(title, body, args.font), encoding="utf-8")
    command = [executable, "-interaction=nonstopmode", "-halt-on-error",
               "-no-shell-escape", "-output-directory", str(build_dir), str(tex_path)]
    result = subprocess.run(command, cwd=build_dir, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    (build_dir / "compile_output.txt").write_text(result.stdout + result.stderr,
                                                encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"XeLaTeX failed; inspect {build_dir / 'compile_output.txt'}")
    log = (build_dir / "report.log").read_text(encoding="utf-8", errors="replace")
    if "Missing character:" in log:
        raise RuntimeError(f"Missing glyphs in PDF; inspect {build_dir / 'report.log'}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(build_dir / "report.pdf", args.output)
    print(f"Created {args.output.resolve()}")
    if "Overfull \\hbox" in log or "Overfull \\vbox" in log:
        print("Layout overflow warning: review report.log and rendered pages")
    print("Next: render every page with pdftoppm and inspect visually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
