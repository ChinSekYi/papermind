# Papermind — User Research & Project Challenges

## Overview

**Channels:** Reddit (r/learnmachinelearning), Discord (Data Science server)  
**Method:** Open-ended public posts asking about pain points when reading research papers  
**Post title:** "What actually stops you from reading research papers?"

---

## Pain Point Tally

| Pain Point | Mentions |
|---|---|
| Not knowing if a paper is worth reading | 6 |
| Prerequisite spiral | 5 |
| Discovery / finding relevant papers | 3 |
| Too much noise — just show what matters | 2 |
| Losing the thread while reading | 1 |

---

## Raw Responses

### Reddit — r/learnmachinelearning
https://www.reddit.com/r/learnmachinelearning/comments/1trtg9k/what_actually_stops_you_from_reading_research/ <br>
**Response 1** *(ROI blocker + prerequisite spiral)*
> "The not knowing if it's worth reading one is the real blocker honestly. You can spend an hour on a paper and realise it's either not relevant or already superseded by something else. That's the tax that kills the habit. The prerequisite spiral is the other one — you open a paper and need to read 3 others first, and those need 2 more, and at some point you close everything and go back to Twitter."

---

**Response 2** *(Comprehensive breakdown)*
> "The not knowing if it's worth reading is bigger than people admit. A paper can take two hours to understand and there's no reliable signal before you start whether it's going to be relevant. The prerequisites problem compounds this. For junior engineers the framing gap is real too — papers are written for reviewers who already know the field. The motivation section assumes context you don't have yet. Discovery is also underrated — Google Scholar and arxiv are overwhelming without knowing how to navigate them."

---

**Response 3** *(Discovery + noise)*
> "Discovery is hard. Trust signals are hard. Papers are noisy — if there's like one or two interesting things just point it out at the beginning. Mapping what I know to what insight this new thing will give me is a problem that's relatively hard. I want all the info and less the noise and I want it tailored to me."

---

**Response 4** *(Choice fatigue)*
> "I have choice fatigue. I found that I actually read something when I have it literally pop in front of me, otherwise I completely forget. I have a higher chance reading from a newsletter than from an app or website because I get overwhelmed by having to choose where and what to read."

---

**Response 5** *(ROI blocker)*
> "Knowing what's worth reading or not is the biggest problem."

---

**Response 6** *(Motivation)*
> "I would ask myself why I start reading the paper instead."

---

**Response 7** *(Skills gap)*
> "To understand a research paper, students should understand what research is, how it is structured, how to evaluate results. That's the skill that can be obtained. That's what good universities teach you."

---

**Response 8** *(Prerequisites + concepts)*
> "Complex concepts and prerequisites!"

---

**Response 9** *(Losing the thread)*
> "Sometimes getting lost in words — like woah, I read till here and I can't recall what I read."

---

**Response 10** *(Job reality)*
> "A professional programming job paying a liveable wage is the main blocker."

---

### Discord — Data Science Server

**Response 1** *(Prerequisites + ROI — detailed)*
> "The biggest blocker for me is usually prerequisites and domain knowledge. My background is in biomedical sciences, and now I'm completing a master's in data science, so I often find papers that seem interesting but quickly realize I need to understand several concepts, methods, or previous papers before I can fully grasp what I'm reading. The second challenge is knowing whether a paper is worth investing time in. Research papers can be dense, and it can take a significant amount of effort just to determine if the methods or findings are relevant to what I'm trying to learn."

---

**Response 2** *(Prerequisites + learning path suggestion)*
> "Research papers often assume the reader already understands a lot of concepts. Sometimes I'll start reading a paper, then find myself opening multiple tabs just to understand the terminology, methods, or references to other papers. Before long, I'm learning about the prerequisites more than the original paper itself. A 'learning path' for a paper — showing key concepts, prerequisite knowledge, and beginner-friendly resources before diving in — would make it much easier to decide whether you're ready to read it."

---

**Response 3** *(Competitor challenge)*
> "What's the difference between that and me just asking Claude/Codex about it?"

---

## Key Insight

The competitor question from Discord is the most important signal to address.

Claude explains papers **reactively** — when you ask it a question, it answers. It does not:
- Tell you **before you start** whether the paper is worth your time
- Map out **what you personally need to know** before reading
- Extract prerequisites from **equations, figures, and tables** that generic PDF parsing misses

That gap is where Papermind's pipeline lives. The differentiation is not the LLM — it's the **extraction quality upstream**. A purpose-built RAG pipeline that handles dense academic PDFs properly gives the LLM better context, which produces better, more specific outputs downstream.

---

## Validated Problem Statement

> Junior engineers hit prerequisite hell when trying to read dense research papers — not knowing what they need to understand before starting, and getting lost in a spiral of prerequisite papers before reaching the actual contribution.

