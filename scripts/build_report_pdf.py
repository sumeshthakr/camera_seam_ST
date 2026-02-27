#!/usr/bin/env python3
"""Build a polished REPORT.pdf from REPORT.md with rendered equations."""

from __future__ import annotations

import html
import re
import shutil
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def inline_fmt(text: str) -> str:
    t = html.escape(text)
    t = re.sub(r"`([^`]+)`", lambda m: f"<font name='Courier'>{m.group(1)}</font>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", lambda m: f"<b>{m.group(1)}</b>", t)
    t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", lambda m: f"<i>{m.group(1)}</i>", t)
    return t


def parse_table(block_lines: list[str]) -> list[list[str]]:
    rows = []
    for raw in block_lines:
        s = raw.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c or "") for c in cells):
            continue
        rows.append(cells)
    return rows


def normalize_math(expr: str) -> str:
    expr = expr.strip()
    expr = re.sub(r"\\text\{([^{}]*)\}", r"\\mathrm{\1}", expr)
    expr = expr.replace(r"\left", "").replace(r"\right", "")
    expr = expr.replace(r"\,", " ")
    expr = expr.replace("°", r"^{\circ}")
    return expr


def render_equation_image(expr: str, out_path: Path, fontsize: int = 14) -> tuple[float, float]:
    expr = normalize_math(expr)
    fig = plt.figure(figsize=(8, 0.5), dpi=200)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    txt = ax.text(0.5, 0.5, f"${expr}$", ha="center", va="center", fontsize=fontsize)

    # Resize figure tightly around equation.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = txt.get_window_extent(renderer=renderer).expanded(1.08, 1.35)
    width_in = max(2.0, bbox.width / fig.dpi)
    height_in = max(0.3, bbox.height / fig.dpi)
    fig.set_size_inches(width_in, height_in)
    fig.canvas.draw()
    fig.savefig(out_path, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    reader = ImageReader(str(out_path))
    return reader.getSize()


def build(src_md: Path, out_pdf: Path) -> None:
    lines = src_md.read_text(encoding="utf-8").splitlines()

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "H1x",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            "H2x",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            spaceBefore=8,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "H3x",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            spaceBefore=6,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            "Bodyx",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            spaceAfter=6,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            "Bulx",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            leftIndent=12,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            "Codex",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=9,
            leading=11,
            leftIndent=8,
            rightIndent=8,
            backColor=colors.whitesmoke,
            borderPadding=6,
            spaceBefore=4,
            spaceAfter=8,
        )
    )

    story = []
    tmpdir = Path(tempfile.mkdtemp(prefix="report_eq_"))
    try:
        i = 0
        eq_id = 0
        while i < len(lines):
            s = lines[i].strip()

            if not s:
                story.append(Spacer(1, 4))
                i += 1
                continue

            # Equation block: $$ ... $$ on one line
            m_eq = re.match(r"^\$\$(.*)\$\$$", s)
            if m_eq:
                eq = m_eq.group(1)
                img_path = tmpdir / f"eq_{eq_id}.png"
                eq_id += 1
                px_w, px_h = render_equation_image(eq, img_path)
                max_w = LETTER[0] - 108  # page width - left/right margins
                scale = min(1.0, max_w / px_w)
                img = Image(str(img_path), width=px_w * scale, height=px_h * scale)
                img.hAlign = "CENTER"
                story.append(Spacer(1, 2))
                story.append(img)
                story.append(Spacer(1, 6))
                i += 1
                continue

            if s.startswith("```"):
                i += 1
                code = []
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    code.append(lines[i])
                    i += 1
                if i < len(lines):
                    i += 1
                story.append(Preformatted("\n".join(code), styles["Codex"]))
                continue

            if s.startswith("### "):
                story.append(Paragraph(inline_fmt(s[4:]), styles["H3x"]))
                i += 1
                continue
            if s.startswith("## "):
                story.append(Paragraph(inline_fmt(s[3:]), styles["H2x"]))
                i += 1
                continue
            if s.startswith("# "):
                story.append(Paragraph(inline_fmt(s[2:]), styles["H1x"]))
                i += 1
                continue

            if s in ("---", "***"):
                story.append(Spacer(1, 8))
                i += 1
                continue

            if s.startswith("|"):
                tlines = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    tlines.append(lines[i])
                    i += 1
                rows = parse_table(tlines)
                if rows:
                    max_cols = max(len(r) for r in rows)
                    norm = []
                    for r in rows:
                        r = r + [""] * (max_cols - len(r))
                        norm.append([Paragraph(inline_fmt(c), styles["Bodyx"]) for c in r])
                    tbl = Table(norm, repeatRows=1)
                    tbl.setStyle(
                        TableStyle(
                            [
                                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f4f8")),
                                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                                ("TOPPADDING", (0, 0), (-1, -1), 4),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                            ]
                        )
                    )
                    story.append(tbl)
                    story.append(Spacer(1, 6))
                continue

            m_num = re.match(r"^(\d+)\.\s+(.*)$", s)
            if m_num:
                while i < len(lines):
                    sm = lines[i].strip()
                    mm = re.match(r"^(\d+)\.\s+(.*)$", sm)
                    if not mm:
                        break
                    story.append(
                        Paragraph(f"{mm.group(1)}. {inline_fmt(mm.group(2))}", styles["Bulx"])
                    )
                    i += 1
                continue

            if s.startswith("- "):
                while i < len(lines) and lines[i].strip().startswith("- "):
                    item = lines[i].strip()[2:]
                    story.append(Paragraph(inline_fmt(item), styles["Bulx"], bulletText="•"))
                    i += 1
                continue

            para = [s]
            i += 1
            while i < len(lines):
                nx = lines[i].strip()
                if not nx:
                    break
                if nx.startswith(("#", "-", "|", "```")) or re.match(r"^\d+\.\s+", nx) or nx in (
                    "---",
                    "***",
                ) or re.match(r"^\$\$.*\$\$$", nx):
                    break
                para.append(nx)
                i += 1
            story.append(Paragraph(inline_fmt(" ".join(para)), styles["Bodyx"]))

        def on_page(canvas, doc):
            canvas.setFont("Helvetica", 9)
            canvas.setFillColor(colors.grey)
            canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 20, f"Page {doc.page}")

        doc = SimpleDocTemplate(
            str(out_pdf),
            pagesize=LETTER,
            leftMargin=54,
            rightMargin=54,
            topMargin=54,
            bottomMargin=36,
            title="Baseball Orientation Detection Report",
            author="Sumesh",
        )
        doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    build(Path("REPORT.md"), Path("REPORT.pdf"))
    print("Built REPORT.pdf")
