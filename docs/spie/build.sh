#!/bin/bash
# Build the SPIE deliverables from the OFFICIAL A4 template.
#
#   ./build.sh            -> both versions
#   ./build.sh A          -> version A only  (geometry bridge + anchor; the DEPLOYED method)
#   ./build.sh B          -> version B only  (geometry bridge, motion removed from the DATA; results are PLACEHOLDERS)
#
# Each version yields a manuscript and a standalone abstract (the abstract is submitted
# separately, so it is not part of the manuscript body).
set -e
cd "$(dirname "$0")"
PY=/home/mirlab/anaconda3/envs/flow_matching/bin/python
PD="pandoc --reference-doc=spie-reference.docx --from markdown+tex_math_dollars"

$PY ../../scripts/make_spie_reference_docx.py ProcSPIETemplate_A4.docx spie-reference.docx

pages () { pdfinfo "$1" | awk '/^Pages/{print $2}'; }

for V in ${@:-A B}; do
  $PD manuscript_$V.md -o SPIE_manuscript_$V.docx
  $PY ../../scripts/finalize_spie_docx.py SPIE_manuscript_$V.docx >/dev/null
  $PD abstract_$V.md   -o SPIE_abstract_$V.docx
  rm -f SPIE_manuscript_$V.pdf SPIE_abstract_$V.pdf
  soffice --headless --convert-to pdf SPIE_manuscript_$V.docx --outdir . >/dev/null 2>&1
  soffice --headless --convert-to pdf SPIE_abstract_$V.docx   --outdir . >/dev/null 2>&1
  echo "version $V:  manuscript $(pages SPIE_manuscript_$V.pdf) page(s)   abstract $(pages SPIE_abstract_$V.pdf) page(s)"
done
