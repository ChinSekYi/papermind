# Project Learnings

Honest retrospective on mistakes made and lessons learned building Papermind. Useful for interviews.

---

### 1. No clear problem statement

Started with a vague goal of "making research papers easier for juniors." Cycled through multiple directions — interview prep tool, collaborative reading, personalized relevance scoring — without validating any of them.

**Root cause:** Building a solution before confirming the problem was real and unsolved by existing tools.

**Fix:** Conduct user research first. Lock the problem statement before scoping features.

---

### 2. Differentiation against Claude and ChatPDF

Every product idea was vulnerable to "Claude already does this." Took extensive reasoning to identify that the real differentiation is not at the LLM layer — it is at the extraction layer upstream.

**Key learning:** Dense academic PDFs with equations, figures, and tables are poorly handled by generic tools. A purpose-built RAG pipeline for academic PDFs is the defensible technical artifact.

---

### 3. Personal pain ≠ validated user need

Initial assumptions were based entirely on personal experience — struggling with "Attention Is All You Need" while prepping for interviews. Had to conduct real user research to confirm which pain points were widely shared.

**Result:** "Not knowing if a paper is worth reading" and "prerequisite spiral" are the top two validated blockers across Reddit and Discord.

---

### 4. Over-engineering the parser benchmark

Built an elaborate evaluation framework with weighted rubrics (40/30/20/10 scoring), LLM-as-judge, and MLflow tracking before the problem was even validated. The weights were borrowed from an LLM suggestion with no principled justification.

**Key learning:** A simple, honest decision with clearly documented tradeoffs is more defensible in interviews than a complex benchmark that never ships.

---

### 5. Scope creep and direction changes

Project direction changed multiple times: paper understanding tool → interview prep → curiosity-driven learning → knowledge repo → RAG pipeline benchmark.

**Root cause:** Each pivot was driven by anxiety about differentiation rather than new validated evidence.

**Fix:** Reframe the project. The RAG pipeline itself is the project. The product wrapper is just a demo shell.

---

### 6. No concrete next steps → loss of momentum

The project stalled not because of lack of effort but because there was never a clear, bounded task for the next week. Each session restarted from the problem statement.

**Fix:** Define one output, one metric, one next step at a time. No scope changes until current phase is shipped.
