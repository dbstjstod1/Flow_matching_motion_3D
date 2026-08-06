"""Turn the OFFICIAL SPIE Proceedings template into a pandoc `--reference-doc`.

We no longer hand-build SPIE's geometry from pandoc's default reference doc. Instead we take
`ProcSPIETemplate_A4.docx` itself — so the A4 page setup, the margins and every SPIE style come
from the authority — and only teach it the handful of style NAMES pandoc emits, by copying the
corresponding SPIE style's formatting onto them.

The official spec, read out of the template (textboxes on p.1-2 plus the style definitions):

    page            A4, 210 x 297 mm
    margins         top 2.54 cm (1.00 in), bottom 4.94 cm (1.95 in), left/right 1.93 cm (0.76 in)
                    -> text block 6.75 x 8.74 in
    paper title     16 pt bold, centred            (SPIE paper title)
    authors/affils  12 pt, centred                 (SPIE Authors-Affils)
    abstract title  11 pt bold CAPS, centred       (SPIE abstract title)
    section heading 11 pt bold CAPS, centred       (heading 1, alias "SPIE Section")
    subsection      10 pt bold, left               (heading 2, alias "SPIE Subsection")
    body            10 pt, justified               (SPIE body text)
    figure caption  9 pt, BELOW the figure, indented   (SPIE figure caption)
    table caption   9 pt, ABOVE the table, indented    (SPIE table caption)
    references      11 pt bold CAPS heading, entries 10 pt justified, numbered
    and: one column, no headers/footers/page numbers.

    python scripts/make_spie_reference_docx.py <template.docx> <out.docx>
"""

from __future__ import annotations

import copy
import sys

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn

# pandoc's style name  ->  the SPIE style whose formatting it should wear
MAP = {
    "Body Text":        "SPIE body text",
    "First Paragraph":  "SPIE body text",
    "Compact":          "SPIE body text",
    "Title":            "SPIE paper title",
    "Author":           "SPIE Authors-Affils",
    "Image Caption":    "SPIE figure caption",
    "Table Caption":    "SPIE table caption",
    "Caption":          "SPIE figure caption",
    "Figure":           "SPIE figure",
    "Captioned Figure": "SPIE figure",
    "Bibliography":     "SPIE reference listing",
    "Affiliation":      "SPIE Authors-Affils",
}


def by_name(doc, name):
    for st in doc.styles:
        if st.name == name:
            return st
    return None


def clone_format(doc, dst_name, src_name):
    """Give `dst_name` the paragraph+run formatting of `src_name`, creating it if needed."""
    src = by_name(doc, src_name)
    if src is None:
        raise SystemExit(f"template has no style {src_name!r}")
    dst = by_name(doc, dst_name)
    if dst is None:
        dst = doc.styles.add_style(dst_name, WD_STYLE_TYPE.PARAGRAPH)
    d, s = dst.element, src.element
    for tag in ("w:basedOn", "w:pPr", "w:rPr"):
        for old in d.findall(qn(tag)):
            d.remove(old)
        node = s.find(qn(tag))
        if node is not None:
            d.append(copy.deepcopy(node))
    return dst


def strip_numbering(style):
    """Drop a style's own automatic list numbering (necessary, but NOT sufficient — see below)."""
    pPr = style.element.find(qn("w:pPr"))
    if pPr is None:
        return
    for numPr in pPr.findall(qn("w:numPr")):
        pPr.remove(numPr)


def unbind_outline_numbering(doc, style_id="Heading1"):
    """Cut the OUTLINE numbering that auto-numbers every paragraph in `style_id`.

    The template numbers its sections from `numbering.xml`: abstractNum 3 has a level carrying
    `<w:pStyle w:val="Heading1"/>` and lvlText "%1.", which Word applies to EVERY Heading1
    paragraph regardless of the style's own numPr. Removing numPr from the style alone leaves it
    on, and our markdown's manual "1. INTRODUCTION" then renders as "2. 1. INTRODUCTION".
    We keep the manual numbers (they survive the pandoc round trip) and cut the automatic ones.
    """
    num = doc.part.numbering_part.element
    cut = 0
    for lvl in num.iter(qn("w:lvl")):
        ps = lvl.find(qn("w:pStyle"))
        if ps is not None and ps.get(qn("w:val")) == style_id:
            lvl.remove(ps)
            cut += 1
    return cut


def set_size(style, half_points):
    rPr = style.element.get_or_add_rPr()
    for tag in ("w:sz", "w:szCs"):
        e = rPr.find(qn(tag))
        if e is None:
            e = rPr.makeelement(qn(tag), {})
            rPr.append(e)
        e.set(qn("w:val"), str(half_points))


def main(src, dst):
    doc = Document(src)

    for pandoc_name, spie_name in MAP.items():
        clone_format(doc, pandoc_name, spie_name)

    for h in ("Heading 1", "heading 1"):
        st = by_name(doc, h)
        if st is not None:
            strip_numbering(st)
    cut = unbind_outline_numbering(doc, "Heading1")

    # Body text must be unambiguously 10 pt: the template leans on Word's application default,
    # which is 11 pt in current Word and would silently inflate the whole paper.
    for nm in ("Normal", "Body Text", "First Paragraph", "Compact"):
        st = by_name(doc, nm)
        if st is not None:
            set_size(st, 20)

    # Empty the body but KEEP its sectPr -- that is where pandoc reads the A4 page setup from.
    body = doc.element.body
    sectPr = body.find(qn("w:sectPr"))
    for child in list(body):
        if child is not sectPr:
            body.remove(child)

    doc.save(dst)

    sec = Document(dst).sections[0]
    E = 914400
    print(f"wrote {dst}   (unbound {cut} outline-numbering level(s) from Heading1)")
    print(f"  page    {sec.page_width/E:.2f} x {sec.page_height/E:.2f} in  (A4)")
    print(f"  margins T {sec.top_margin/E:.2f}  B {sec.bottom_margin/E:.2f}  "
          f"L/R {sec.left_margin/E:.2f} in")
    print(f"  text    {(sec.page_width-sec.left_margin-sec.right_margin)/E:.2f} x "
          f"{(sec.page_height-sec.top_margin-sec.bottom_margin)/E:.2f} in")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
