# morph_block label positioning

## Done
- Updated `docs/figures/morph_block.tex`: sublayer labels use `anchor=south west`, `yshift=0.18cm`, white fill, drawn last.
- Regenerated `docs/figures/morph_block.png` via `pdflatex` + `pdftoppm -r 200`.

## Note
Automated crop-based verification was unreliable (label/border share `cbBlue` color; image-describe tool kept reporting clipping even when crops looked clear). If either label still feels tight in the PDF viewer, nudge only `yshift` in the two final `\node` lines (try `0.22cm`–`0.25cm`; avoid `>0.5cm` or the MLP label can overlap the HC carrier box above the MLP group).
