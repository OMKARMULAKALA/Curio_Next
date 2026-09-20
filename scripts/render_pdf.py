"""Renders a (self-authored, structurally simple) Markdown report to a
professionally formatted PDF using reportlab -- no external binary (pandoc,
wkhtmltopdf, a GTK/Pango stack for weasyprint) required, which matters on a
Windows CPU-only machine where those are painful or impossible to install.

This is not a general Markdown parser: it supports exactly the subset of
Markdown docs/final_report.md actually uses (headings, paragraphs, bold/
italic/inline-code spans, images, pipe tables, fenced code blocks, bullet/
numbered lists, horizontal rules) plus a generated cover page.
"""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BRAND_BLUE = colors.HexColor("#2a78d6")
TEXT_PRIMARY = colors.HexColor("#0b0b0b")
TEXT_SECONDARY = colors.HexColor("#52514e")
GRID = colors.HexColor("#dedcd6")
HEADER_BG = colors.HexColor("#eef4fc")

_styles = getSampleStyleSheet()
STYLES = {
    "cover_title": ParagraphStyle("cover_title", parent=_styles["Title"], fontSize=25, leading=30, alignment=TA_CENTER, textColor=TEXT_PRIMARY, spaceAfter=14),
    "cover_subtitle": ParagraphStyle("cover_subtitle", parent=_styles["Normal"], fontSize=13.5, leading=18, alignment=TA_CENTER, textColor=TEXT_SECONDARY, spaceAfter=8),
    "cover_meta": ParagraphStyle("cover_meta", parent=_styles["Normal"], fontSize=10.5, leading=15, alignment=TA_CENTER, textColor=TEXT_SECONDARY),
    "h1": ParagraphStyle("h1", parent=_styles["Heading1"], fontSize=17, leading=21, spaceBefore=6, spaceAfter=10, textColor=BRAND_BLUE),
    "h2": ParagraphStyle("h2", parent=_styles["Heading2"], fontSize=13, leading=17, spaceBefore=12, spaceAfter=6, textColor=TEXT_PRIMARY),
    "h3": ParagraphStyle("h3", parent=_styles["Heading3"], fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=4, textColor=TEXT_PRIMARY),
    "body": ParagraphStyle("body", parent=_styles["Normal"], fontSize=9.7, leading=14, spaceAfter=7, textColor=TEXT_PRIMARY, alignment=TA_LEFT),
    "caption": ParagraphStyle("caption", parent=_styles["Normal"], fontSize=8.5, leading=11, spaceAfter=12, textColor=TEXT_SECONDARY, alignment=TA_CENTER, italic=True),
    "cell": ParagraphStyle("cell", parent=_styles["Normal"], fontSize=8.3, leading=11, textColor=TEXT_PRIMARY),
    "cell_head": ParagraphStyle("cell_head", parent=_styles["Normal"], fontSize=8.3, leading=11, textColor=colors.white, fontName="Helvetica-Bold"),
    "code": ParagraphStyle("code", parent=_styles["Code"], fontSize=8, leading=10.5, backColor=colors.HexColor("#f5f4f0"), borderPadding=6, spaceAfter=8),
}

PAGE_WIDTH, PAGE_HEIGHT = LETTER
CONTENT_WIDTH = PAGE_WIDTH - 1.3 * inch


def _inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+?)`", r'<font face="Courier">\1</font>', text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # links -> plain text
    return text


def _split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-+:?", cell) for cell in cells if cell != "")


def render_markdown_to_pdf(markdown_path: Path, pdf_path: Path, image_root: Path, title: str, subtitle: str, meta_lines: list[str]) -> None:
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    story: list = []

    # --- cover page ---
    story.append(Spacer(1, 2.1 * inch))
    story.append(Paragraph(title, STYLES["cover_title"]))
    story.append(Paragraph(subtitle, STYLES["cover_subtitle"]))
    story.append(Spacer(1, 0.4 * inch))
    story.append(HRFlowable(width="40%", thickness=1.1, color=BRAND_BLUE, hAlign="CENTER"))
    story.append(Spacer(1, 0.3 * inch))
    for line in meta_lines:
        story.append(Paragraph(line, STYLES["cover_meta"]))
    story.append(PageBreak())

    i = 0
    n = len(lines)
    first_h1_skipped = False
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped == "---":
            story.append(Spacer(1, 4))
            i += 1
            continue

        if stripped.startswith("# "):
            # The document's own H1 title becomes the cover; skip it here.
            if not first_h1_skipped:
                first_h1_skipped = True
                i += 1
                continue
            story.append(Paragraph(_inline(stripped[2:]), STYLES["h1"]))
            i += 1
            continue

        if stripped.startswith("## "):
            story.append(Paragraph(_inline(stripped[3:]), STYLES["h1"]))
            i += 1
            continue

        if stripped.startswith("### "):
            story.append(Paragraph(_inline(stripped[4:]), STYLES["h2"]))
            i += 1
            continue

        if stripped.startswith("```"):
            i += 1
            code_lines = []
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            story.append(Preformatted("\n".join(code_lines), STYLES["code"]))
            continue

        img_match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if img_match:
            alt, rel_path = img_match.group(1), img_match.group(2)
            img_path = (image_root / rel_path).resolve()
            if img_path.exists():
                from PIL import Image as PILImage

                with PILImage.open(img_path) as im:
                    w_px, h_px = im.size
                max_w = CONTENT_WIDTH
                max_h = 3.6 * inch
                scale = min(max_w / w_px, max_h / h_px)
                story.append(RLImage(str(img_path), width=w_px * scale, height=h_px * scale, hAlign="CENTER"))
                if alt:
                    story.append(Paragraph(_inline(alt), STYLES["caption"]))
            i += 1
            continue

        if stripped.startswith("|"):
            table_lines = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows = [_split_row(r) for r in table_lines]
            if len(rows) >= 2 and _is_separator_row(rows[1]):
                header, body_rows = rows[0], rows[2:]
            else:
                header, body_rows = rows[0], rows[1:]
            ncols = len(header)
            table_data = [[Paragraph(_inline(c), STYLES["cell_head"]) for c in header]]
            for row in body_rows:
                row = (row + [""] * ncols)[:ncols]
                table_data.append([Paragraph(_inline(c), STYLES["cell"]) for c in row])
            col_width = CONTENT_WIDTH / ncols
            tbl = Table(table_data, colWidths=[col_width] * ncols, repeatRows=1)
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), BRAND_BLUE),
                ("GRID", (0, 0), (-1, -1), 0.5, GRID),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, HEADER_BG]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 8))
            continue

        list_match = re.match(r"^(\d+)\.\s+(.*)", stripped)
        bullet_match = re.match(r"^[-*]\s+(.*)", stripped)
        if list_match or bullet_match:
            items = []
            is_numbered = bool(list_match)
            while i < n:
                s = lines[i].strip()
                lm = re.match(r"^(\d+)\.\s+(.*)", s)
                bm = re.match(r"^[-*]\s+(.*)", s)
                if is_numbered and lm:
                    items.append(lm.group(2))
                elif not is_numbered and bm:
                    items.append(bm.group(1))
                else:
                    break
                i += 1
            list_flow = ListFlowable(
                [ListItem(Paragraph(_inline(it), STYLES["body"])) for it in items],
                bulletType="1" if is_numbered else "bullet",
                start=1 if is_numbered else None,
                leftIndent=16,
            )
            story.append(list_flow)
            continue

        # plain paragraph (accumulate until blank line)
        para_lines = [stripped]
        i += 1
        while i < n and lines[i].strip() and not lines[i].strip().startswith(("#", "|", "```", "- ", "* ")) and not re.match(r"^\d+\.\s", lines[i].strip()) and not re.match(r"!\[", lines[i].strip()):
            para_lines.append(lines[i].strip())
            i += 1
        story.append(Paragraph(_inline(" ".join(para_lines)), STYLES["body"]))

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=LETTER,
        leftMargin=0.65 * inch, rightMargin=0.65 * inch,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        title=title,
    )

    def _footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(TEXT_SECONDARY)
        canvas.drawString(0.65 * inch, 0.45 * inch, "Audio Context Layer -- Final Report")
        canvas.drawRightString(PAGE_WIDTH - 0.65 * inch, 0.45 * inch, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
