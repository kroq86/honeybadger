#!/usr/bin/env python3
"""Build a simple machine-readable PDF from whitepaper Markdown."""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def convert_inline(text: str) -> str:
    escaped = html.escape(text)

    def repl(match: re.Match[str]) -> str:
        code = html.escape(match.group(1))
        return f"<font name='Courier'>{code}</font>"

    escaped = re.sub(r"`([^`]+)`", repl, escaped)
    escaped = escaped.replace("**", "")
    return escaped


def flush_paragraph(buf: list[str], story: list, style: ParagraphStyle) -> None:
    if not buf:
        return
    text = " ".join(part.strip() for part in buf if part.strip())
    if text:
        story.append(Paragraph(convert_inline(text), style))
        story.append(Spacer(1, 0.18 * cm))
    buf.clear()


def flush_bullets(buf: list[str], story: list, style: ParagraphStyle) -> None:
    if not buf:
        return
    items = [
        ListItem(Paragraph(convert_inline(item), style), leftIndent=0)
        for item in buf
    ]
    story.append(ListFlowable(items, bulletType="bullet", start="circle", leftIndent=14))
    story.append(Spacer(1, 0.18 * cm))
    buf.clear()


def flush_table(rows: list[list[str]], story: list, body_style: ParagraphStyle) -> None:
    if not rows:
        return
    rendered = [[Paragraph(convert_inline(cell), body_style) for cell in row] for row in rows]
    table = Table(rendered, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa5b1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.22 * cm))
    rows.clear()


def build_story(markdown: str) -> list:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCenter",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        spaceAfter=10,
    )
    author_style = ParagraphStyle(
        "AuthorCenter",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10.5,
        leading=13,
        spaceAfter=3,
    )
    h1_style = ParagraphStyle(
        "Heading1Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=17,
        spaceBefore=10,
        spaceAfter=6,
    )
    h2_style = ParagraphStyle(
        "Heading2Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=14,
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        spaceAfter=0,
    )
    bullet_style = ParagraphStyle(
        "BulletCustom",
        parent=body_style,
        leftIndent=0,
    )
    code_style = ParagraphStyle(
        "CodeBlock",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=8.8,
        leading=11,
    )

    story: list = []
    paragraph_buf: list[str] = []
    bullet_buf: list[str] = []
    table_rows: list[list[str]] = []
    code_buf: list[str] = []
    in_code = False

    lines = markdown.splitlines()
    for idx, raw_line in enumerate(lines):
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph(paragraph_buf, story, body_style)
            flush_bullets(bullet_buf, story, bullet_style)
            flush_table(table_rows, story, body_style)
            if in_code:
                story.append(Preformatted("\n".join(code_buf), code_style))
                story.append(Spacer(1, 0.18 * cm))
                code_buf.clear()
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_buf.append(line)
            continue

        if not stripped:
            flush_paragraph(paragraph_buf, story, body_style)
            flush_bullets(bullet_buf, story, bullet_style)
            flush_table(table_rows, story, body_style)
            continue

        if idx == 0 and stripped.startswith("# "):
            story.append(Paragraph(convert_inline(stripped[2:]), title_style))
            continue

        if idx in (2, 3, 4, 6):
            story.append(Paragraph(convert_inline(stripped), author_style))
            continue

        if stripped.startswith("## "):
            flush_paragraph(paragraph_buf, story, body_style)
            flush_bullets(bullet_buf, story, bullet_style)
            flush_table(table_rows, story, body_style)
            story.append(Paragraph(convert_inline(stripped[3:]), h1_style))
            continue

        if stripped.startswith("### "):
            flush_paragraph(paragraph_buf, story, body_style)
            flush_bullets(bullet_buf, story, bullet_style)
            flush_table(table_rows, story, body_style)
            story.append(Paragraph(convert_inline(stripped[4:]), h2_style))
            continue

        if stripped.startswith("- "):
            flush_paragraph(paragraph_buf, story, body_style)
            flush_table(table_rows, story, body_style)
            bullet_buf.append(stripped[2:])
            continue

        if "|" in stripped and stripped.startswith("|") and stripped.endswith("|"):
            flush_paragraph(paragraph_buf, story, body_style)
            flush_bullets(bullet_buf, story, bullet_style)
            if set(stripped.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
                continue
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            table_rows.append(cells)
            continue

        flush_bullets(bullet_buf, story, bullet_style)
        flush_table(table_rows, story, body_style)
        paragraph_buf.append(stripped)

    flush_paragraph(paragraph_buf, story, body_style)
    flush_bullets(bullet_buf, story, bullet_style)
    flush_table(table_rows, story, body_style)
    if code_buf:
        story.append(Preformatted("\n".join(code_buf), code_style))

    return story


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: build_whitepaper_pdf.py INPUT.md OUTPUT.pdf", file=sys.stderr)
        return 2

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    markdown = input_path.read_text(encoding="utf-8")

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        topMargin=1.7 * cm,
        bottomMargin=1.6 * cm,
        leftMargin=1.7 * cm,
        rightMargin=1.7 * cm,
        title="vmbench: A Formal VM Benchmark and Inspectable Reasoning Runtime",
        author="Kirill Ostapenko",
    )
    story = build_story(markdown)
    doc.build(story)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
