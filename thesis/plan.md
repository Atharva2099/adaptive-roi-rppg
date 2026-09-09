# Writing plan

The chapter structure is provisional for advisor discussion. No chapter prose has been drafted here yet.

| Section | Purpose and next action |
|---|---|
| Introduction | Draft the problem and research questions without promising success. |
| Background and related work | Explain rPPG, POS, regions, tracking, and control with verified citations. |
| Research design | Document datasets, splits, comparisons, metrics, and uncertainty. |
| Adaptive controller | Check candidate measurements, observations, tracker, actions, and learning against code and configuration. |
| Implementation and validation | Explain tests and evaluator; distinguish engineering checks from scientific results. |
| MCD results | Verify row-level model comparisons and Oracle diagnostics before adding numbers. |
| Behavioral and failure analysis | Describe supported analyses; leave missing findings open. |
| Discussion and limitations | Address interpretations, alternatives, causality, provenance, and transfer limits. |
| Conclusion | Finalize answers to the research questions after evidence review. |
| References and appendices | Add citations and useful technical detail as writing proceeds. |

## Open questions

- 2026-09-07: Created `thesis.docx` as an identical copy of `/Users/atharva/Downloads/Masters Thesis.docx`, preserving the user's edits and all 58 remaining comments. Name: Atharva Shirish Walawalkar. Degree: Master of Science in Data Science and Artificial Intelligence. Graduation entry: December 2026. Chair: Kazunori Okada, Ph.D., Professor. Thesis title and remaining committee entries are placeholders.

- What are the exact submission deadline, thesis title, remaining committee details, and citation style? December 2026 is the document's graduation entry, not a verified submission deadline.
- What does completed frozen MMPD evaluation support? Documented engineering progress does not establish a transfer result.
- Which behavioral analyses explain MCD differences, and which remain hypotheses?
- How much Oracle headroom can a deployable controller capture?
- Can checkpoint training lineage and configuration be verified?
- What is the impact of historical missing-data handling on causality and comparisons?
- Which older experiments support controlled ablations rather than historical progression alone?

Before new diagnostics, follow the root experiment protocol: question, required arms, negative control, interpretation limits, and stop/next rule. Update this file as questions are resolved and sections are drafted.
