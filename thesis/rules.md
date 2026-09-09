# Thesis writing rules

This is the writing folder within adaptive-roi-rppg. The repository's root AGENTS.md applies here. Read these rules before writing; consult [evidence.md](evidence.md) for relevant sources and [plan.md](plan.md) for progress. The working document is [thesis.docx](thesis.docx), copied from the user's edited school template. Add files only as actual writing needs arise.

## Writing rules

- Preserve Atharva's voice and edits. Show substantive proposed rewrites before replacing user prose. Write connected academic prose with concrete subjects and direct verbs.
- Define abbreviations at first use. Explain the question and method before results; distinguish observations, interpretations, limitations, and incomplete work.
- Draft stable methods and background now. Keep working placeholders for missing evidence and avoid promising success or novelty.
- Avoid inflated significance, promotional adjectives, vague attribution, generic openings/conclusions, superficial trailing phrases, forced triads, synonym cycling, fake contrasts, and chatbot commentary.
- Vary sentence length naturally; preserve useful technical terms and qualifications. Avoid em dashes and excessive bold text or lists in thesis prose.
- Support related-work statements with actual literature. Claims such as “most methods use fixed regions” require verification.
- School template comments govern formatting and accessibility, not thesis prose. Preserve the downloaded original and use a working Word copy for submission formatting. Inspect rendered pages after edits; use accessible tables and figure alternative text. Verify uncertain caption requirements before submission.
- Do not publish, submit, or share externally without explicit approval.

## Evidence rules

- Use the latest accepted result under the relevant protocol. Compare checkpoint/method identity, preprocessing, ground-truth rule, split/cohort, metric, aggregation, information available to each method, and uncertainty before comparing versions.
- Define what “best” means and the selection set. A favorable test score alone does not justify model selection; disclose test-based selection and retain seed variation. Never substitute the best seed for a family mean.
- Older results can explain a failed direction, protocol change, or historical progression. Call them ablations only when controls support attributing the difference to the tested factor. Worse scores alone do not establish a bad hyperparameter.
- Recompute pivotal numbers from row-level artifacts before settling prose. Record source, denominator, aggregation, and uncertainty in evidence.md. Slides, filenames, prose summaries, and remote status messages are discovery aids; state verification limits.
- Identify current, historical, superseded, negative, diagnostic, and incomplete evidence in working notes. Oracle B/C are GT-informed offline diagnostics; seed means are not ensembles; MCD results do not establish transfer.
- MCD is the development dataset; MMPD remains frozen evaluation-only under root custody rules. Writing does not authorize dataset access, new experiments, or changes to legacy material.

## School template structure and approval workflow

`thesis.docx` contains the school-provided section headings and order. They are fixed. Do not rename, move, delete, merge, split, or add competing top-level headings. Draft and revise only the content that belongs under the existing template headings, preserving all required template material unless Atharva explicitly authorizes a specific change.

For each existing template section, use this approval-first workflow:

1. Agree on what the section needs to say and what it must not claim.
2. Use `plan.md` for the section outline, progress, and open questions; do not duplicate that planning structure here.
3. Wait for Atharva to approve the direction or redirect it.
4. Draft the section in Atharva's voice.
5. Add the prose under the matching existing heading in `thesis.docx` only after Atharva approves the wording.

Preserve approved wording. Render and inspect edited Word pages before treating a document edit as complete.

## Rule changes

Update rules here when the user establishes a preference, remove redundant wording, and append a dated note. Keep guidance together and link to existing research records instead of copying them.

- 2026-09-07: Established plain-language writing, source verification, and preservation of negative and incomplete findings.
- 2026-09-07: Consolidated to three files; use the root AGENTS.md only. Thesis is a writing folder in this project. Add files only for distinct content that needs them.
- 2026-09-07: Named this file rules.md to make its purpose explicit. Keep the school's comments in thesis.docx as drafting guidance; check each applicable instruction and remove comments only when preparing the approved final submission copy.
- 2026-09-07: Established the approval-first workflow for every template section and the requirement to add only approved prose to thesis.docx.
- 2026-09-07: Clarified that the school template's headings and order are fixed; work may only add or revise content under those existing headings.
- 2026-09-07: Removed the mistakenly added proposed content-section order. The five-step approval workflow applies separately to every existing school-template section.
- 2026-09-07: Clarified that plan.md, not rules.md, holds section outlines and planning detail.
