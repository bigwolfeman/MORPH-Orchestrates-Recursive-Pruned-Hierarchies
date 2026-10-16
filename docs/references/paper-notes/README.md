# Paper notes

One note per paper that MORPH uses. A note records why we use the paper,
what its measured results show, our speculation, our corrections, and where
our own measurements disagree with it.

The paper text stays in `docs/references/<topic>/<paper>/`. The note stays
here. Do not copy paper text into a note.

## When to write

Write or update a note every time you:

- read a paper in full,
- cite a paper in a design or a plan, or
- measure something that confirms or disagrees with a paper.

Add a ledger line in `LEDGER.md` in the same change. The ledger line is required.

## File naming

- File name: `<paper-slug>.md`.
- The slug is the paper's folder name under `docs/references/` when that
  folder exists. Example: the XM folder is `tul-latent-emission/explorative-modeling/`,
  so the note is `explorative-modeling.md`.
- When no folder exists, pick a short slug from the title and arXiv id.
  Example: `ncm-next-concept-modeling-2610-14162.md`.

## Rules

1. Entries are append-only. Never delete or rewrite an old entry.
2. Every entry starts with a date, `yyyy-mm-dd`. The date is the day you
   read the paper or made the measurement.
3. Add new entries at the bottom of each section. The newest entry is last.
4. To fix an old reading, add a dated entry in section 4 (Corrections).
   Do not edit the old text.
5. Section 3 (Speculation) must say "Speculation" in the entry. Keep it apart
   from what the paper shows.
6. Section 2 uses the paper's measured results. Give section, table, or
   equation numbers. If you do not have them, write "section numbers not
   recorded" and add them on the next full read. Do not use the abstract as a
   measured result.
7. Section 5 gives four things: our measurement, the paper's claim, the
   source of our number (a filing path or a run name), and the date.
8. If a section has nothing yet, write "None recorded yet."
9. Plain English. Short sentences. No em dashes.

## Sections

Use `TEMPLATE.md`. The five sections are:

1. Why we reference it
2. What the paper shows
3. Our speculation
4. Corrections
5. Where our findings disagree

## Ledger

`LEDGER.md` has one line per dated entry:

    yyyy-mm-dd | <paper-slug> | <section touched> | one-line gist

Append only. Newest line last.
