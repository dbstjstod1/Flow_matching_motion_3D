"""Repair the pandoc-produced SPIE .docx in place.

Two things pandoc 2.11 leaves broken for Word/LibreOffice:

1. Pipe tables are emitted with an EMPTY `<w:tblGrid/>` and `tblW = 0 pct`, so the renderer
   collapses every column but the first. We supply an explicit grid.
2. Table cell text inherits `Compact`; SPIE tables read better at 9 pt with the numeric
   columns centred.
3. Reference entries must be 10 pt per the SPIE template, and the list NUMBER needs its own
   paragraph-mark rPr or it keeps the inherited size.

    python scripts/finalize_spie_docx.py docs/spie/SPIE_manuscript.docx
"""

from __future__ import annotations

import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

TEXT_W = 6.75          # SPIE text block width, inches
SERIF = "Times New Roman"


def _el(parent, tag, **attrs):
    e = parent.makeelement(qn(tag), {})
    for k, v in attrs.items():
        e.set(qn(k), str(v))
    return e


def fix_table_grid(tbl, widths_in):
    """Give the table a real grid; without one the columns collapse."""
    tblPr = tbl._tbl.tblPr
    # total width = 100% of the text block, in fiftieths of a percent
    for old in tblPr.findall(qn("w:tblW")):
        tblPr.remove(old)
    tblPr.append(_el(tblPr, "w:tblW", **{"w:type": "dxa",
                                         "w:w": int(TEXT_W * 1440)}))
    for old in tblPr.findall(qn("w:tblLayout")):
        tblPr.remove(old)
    tblPr.append(_el(tblPr, "w:tblLayout", **{"w:type": "fixed"}))

    grid = tbl._tbl.find(qn("w:tblGrid"))
    if grid is not None:
        tbl._tbl.remove(grid)
    grid = _el(tbl._tbl, "w:tblGrid")
    for w in widths_in:
        grid.append(_el(grid, "w:gridCol", **{"w:w": int(w * 1440)}))
    tbl._tbl.insert(list(tbl._tbl).index(tblPr) + 1, grid)

    # per-cell widths must agree with the grid under a fixed layout
    for row in tbl.rows:
        for cell, w in zip(row.cells, widths_in):
            cell.width = Inches(w)


def style_table(tbl):
    last = len(tbl.rows) - 1
    for r, row in enumerate(tbl.rows):
        for c, cell in enumerate(row.cells):
            for p in cell.paragraphs:
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(1)
                # hold every row to the next one so the table never breaks across a page
                p.paragraph_format.keep_with_next = (r < last)
                p.alignment = (WD_ALIGN_PARAGRAPH.LEFT if c == 0
                               else WD_ALIGN_PARAGRAPH.CENTER)
                for run in p.runs:
                    run.font.size = Pt(9)
                    run.font.name = SERIF
                    run.font.color.rgb = RGBColor(0, 0, 0)


def style_reference_list(doc):
    """Hand the reference entries to the template's own `SPIE reference listing` style.

    That style is 10 pt justified AND carries the template's numbering (abstractNum 4,
    lvlText "[%1]"), i.e. SPIE's bracketed reference numbers. Pandoc emits the list as `Compact`
    paragraphs with its OWN numPr stamped directly on each one, which overrides the style — so the
    direct numPr has to be removed for the bracket format to show through.
    """
    seen, n = False, 0
    for p in doc.paragraphs:
        if p.text.strip().upper().startswith("REFERENCES"):
            seen = True
            continue
        if not seen or not p.text.strip():
            continue
        pPr = p._p.get_or_add_pPr()
        for direct in pPr.findall(qn("w:numPr")):
            pPr.remove(direct)              # let the style's [%1] numbering apply
        for ind in pPr.findall(qn("w:ind")):
            pPr.remove(ind)                 # ditto for pandoc's list indent
        p.style = doc.styles["SPIE reference listing"]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        for run in p.runs:
            run.font.size = Pt(10)
            run.font.name = SERIF
        n += 1
    print(f"reference list: {n} entries -> 'SPIE reference listing' (10 pt, [n] numbering)")


def bind_table_caption(doc):
    """SPIE puts the table caption ABOVE the table; keep the two on the same page."""
    body = list(doc.element.body)
    for i, el in enumerate(body):
        if el.tag == qn("w:tbl") and i and body[i - 1].tag == qn("w:p"):
            pPr = body[i - 1].get_or_add_pPr()
            if pPr.find(qn("w:keepNext")) is None:
                pPr.append(_el(pPr, "w:keepNext"))
            print("table caption bound to its table")


def main(path):
    doc = Document(path)

    for tbl in doc.tables:
        # tbl.columns is derived from tblGrid, which pandoc leaves EMPTY -- count the
        # cells in the header row instead.
        ncol = len(tbl.rows[0].cells)
        if ncol == 4:
            widths = [2.55, 1.45, 1.40, 1.35]        # Method | PSNR | SSIM | RPE
        else:
            widths = [TEXT_W / ncol] * ncol
        fix_table_grid(tbl, widths)
        style_table(tbl)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        print(f"table fixed: {ncol} cols, {len(tbl.rows)} rows")

    style_reference_list(doc)
    bind_table_caption(doc)

    doc.save(path)
    print(f"finalized {path}")


if __name__ == "__main__":
    main(sys.argv[1])
