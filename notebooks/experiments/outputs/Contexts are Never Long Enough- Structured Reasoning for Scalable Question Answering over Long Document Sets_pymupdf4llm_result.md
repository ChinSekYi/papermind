**==> picture [60 x 35] intentionally omitted <==**

**==> picture [56 x 35] intentionally omitted <==**

# **Contexts are Never Long Enough: Structured Reasoning for Scalable Question Answering over Long Document Sets** 

**Harshit Joshi Priyank Shethia Jadelynn Dao Monica S. Lam** Computer Science Department, Stanford University `{hj, lam}@cs.stanford.edu` 

## **Abstract** 

Real-world document question answering is challenging. Analysts must synthesize evidence across multiple documents and different parts of each document. However, any fixed LLM context window can be exceeded as document collections grow. A common workaround is to decompose documents into chunks and assemble answers from chunk-level outputs, but this introduces an _aggregation bottleneck_ : as the number of chunks grows, systems must still combine and reason over an increasingly large body of extracted evidence. We present SLIDERS, a framework for question answering over long document collections through _structured reasoning_ . SLIDERS extracts salient information into a relational database, enabling scalable reasoning over persistent structured state via SQL rather than concatenated text. To make this locally extracted representation globally coherent, SLIDERS introduces a data reconciliation stage that leverages provenance, extraction rationales, and metadata to detect and repair duplicated, inconsistent, and incomplete records. SLIDERS outperforms all baselines on three existing long-context benchmarks, despite all of them fitting within the context window of strong base LLMs, exceeding GPT-4.1 by 6.6 points on average. It also improves over the next best baseline by _∼_ 19 and _∼_ 32 points on two new benchmarks at 3.9M and 36M tokens, respectively. 

**==> picture [9 x 10] intentionally omitted <==**

```
https://sliders.genie.stanford.edu/https://github.com/stanford-oval/sliders
```

**==> picture [383 x 140] intentionally omitted <==**

**----- Start of picture text -----**<br>
Long-Context ( ≤ 360k tokens) Ultra-Long Benchmarks ( ≥ 3.9M tokens)<br>100<br>89 . 33<br>78 . 57 78 . 91<br>80<br>62 . 67 64 . 67<br>60 54 . 35 55 . 22<br>40 31 . 41<br>20 11 . 32<br>5 . 00<br>0 × × $$<br>FinanceBench Loong Oolong (256k) WikiCeleb100 FinQ100<br>3.9M tokens 36M tokens<br>RAG GPT 4.1 RLM SLIDERS<br>Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 1: Accuracy across long-context and ultra-long benchmarks. SLIDERS consistently outperforms all baselines and is the only system to scale beyond 3.9M tokens. _×_ denotes baselines that exceeded the model’s context window; $$ denotes prohibitively expensive runs. 

Preprint. 

## **1 Introduction** 

Answering questions over long or multi-document corpora accurately is essential for analysis in knowledge-intensive domains such as finance, healthcare, and the social sciences. Analysts routinely need to synthesize evidence scattered across multiple pages in a single report or across thousands of reports in a collection [Zhang et al., 2021, Van De Schoot et al., 2021, Del Fiol et al., 2014]. However, even with million-token context windows, modern LLMs fall short of real-world document analysis, where models must integrate and reason over distributed evidence. Realistic corpora remain far beyond what can be processed in a single pass, forcing systems to rely on selective retrieval or decomposition, where missing relevant evidence can lead to incorrect conclusions. At the same time, LLMs struggle to reliably combine information across distant sections or documents, often producing incomplete, duplicated, or contradictory outputs [Liu et al., 2023a, Hsieh et al., 2024, Goldman et al., 2024]. Finally, free-form outputs hinder auditing and compliance in high-stakes settings, while ultra-long-context inference remains computationally expensive [Gu and Dao, 2024]. 

A widely adopted workaround is to partition documents into smaller fragments and assemble answers from chunk outputs [Zhang et al., 2024, Zhao et al., 2024, Shankar et al., 2025]. While chunking avoids exceeding context limits and can improve attention to local details, it introduces a new bottleneck: as the number of chunks increases, systems must aggregate and reason over evidence extracted from all chunk outputs, ultimately recreating the very long-context problem it was designed to avoid, a fundamental limitation we refer to as the “Aggregation Bottleneck”. 

**==> picture [397 x 146] intentionally omitted <==**

**----- Start of picture text -----**<br>
Concatenated textual evidence grows with # chunks Aggregation Bottleneck<br>Existing Chunk-based methods
 Adele’s tv debut in  2007 (Doc 1) Must fit into context window<br>(Text Intermediate State) In 2006 adele featured as a vocalist on a single  (Doc 1) LLM<br>In 2017, ishowspeed started livestreaming (Doc 2) (Long Context)

 Final Answer<br>Watkins registered on youtube in 2016 (Doc 2) Reasons over huge<br>Chunk 1 ... (many more) concatenated context (Lossy, Prone to errors)<br>LLM extracts information (per chunk)<br>Chunk 2 Intermediate: unstructured text<br>Intermediate: relational database<br>... Extraction to structured relational tables rows<br>Document Set Chunk N Name Alt Name Debut Description Doc<br>Adele 2007 TV debut in Later ... Reconciliation

 Coding LLM<br>Partition into smaller chunks IshowspeedAAdeledele Watkins 200620020168 FRDeeatured as vocalistegistered on but song Hometown glyoutube ... Resolvesacross chunksconflicts 
 Generates SQL queries 
over tables (Accurate, auditable)Final Answer<br>SL(Structured relational database)IDERS: Structured reasoning
 Ishowspeed Watkins 2017 A...ctively started live stre...<br>LLM extracts rows with provenance<br>**----- End of picture text -----**<br>


Figure 2: Chunking based methods regenerate the long-context problem. SLIDERS mitigates it by using structured reasoning. 

**To overcome the** _**aggregation bottleneck**_ **in question answering over long document sets, we propose** _**structured reasoning**_ **.** Instead of relying on LLMs to reason over concatenated chunk outputs, we extract salient information into a relational database, transforming unstructured text into a persistent, structured state. This representation is inherently scalable, allowing us to store and organize arbitrarily large volumes of evidence in a compact, queryable form, rather than being constrained by context limits. Once information is explicitly stored, reasoning reduces to querying: we execute SQL over the database to aggregate, compare, and compute over the extracted evidence, enabling operations that LLMs struggle to perform reliably in free-form generation [Hsieh et al., 2024, Bai et al., 2023, Li et al., 2025, Mirzadeh et al., 2024]. In this way, long-context reasoning becomes database reasoning, where answers are derived by systematically inspecting and combining stored information without requiring all evidence to be reloaded into the model’s context (Figure 2). Leveraging the code-generation capabilities of LLMs, we bridge unstructured text and structured execution, using models to translate questions into queries while offloading symbolic computation to the database where it can be performed deterministically [Jimenez et al., 2023, Yang et al., 2025, Hong et al., 2025]. 

**To transform independently extracted evidence into a coherent and scalable database for reasoning, we introduce a** _**data reconciliation**_ **stage.** Because documents distribute, repeat, and refine information across sections at varying levels of granularity [Sollaci and Pereira, 2004, Mann and Thompson, 1988], independently extracted records often provide overlapping, partial, or conflicting views of the same underlying fact. Reconciliation groups related records, integrates partial evidence, 

2 

**==> picture [396 x 135] intentionally omitted <==**

**----- Start of picture text -----**<br>
Document Set Contextualized Extraction from self-contained chunks Automated Data Reconciliation<br>AppTech Payments, Battalion Oil Corp, Agape Corp, BioLargo Inc, 1847 Document Title: Biolargo Inc 2024 10Q
Chunk ID: 3 
Header: CONDENSED CONSOLIDATED BALANCE SHEETS Document Title: Biolargo Inc 2024 10Q
Chunk ID: Header: NOTES TO CONDENSED CONSOLIDATED 8
 SQL Coding Agent for Reconciliation 4<br>Amounts in thousands of dollars

 FINANCIAL STATEMENTS<br>QuestionChunk Contextualization - creating chunks, generating document metadataWhat is the accounts payable of  1 Current liabilities:
Accounts payable and accrued expenses $1,740Relevant? [3] Accounts and accrued expenses included Category            BioLargoAccounts payable       Accrued payroll        Total                  Payable and Accrued Expenses 
$ 200  13 (    ONin thousands$ 1,3    3M  ... 579 ... ......  $1,      77  $1,740):
 Totals663 Primary Key 
ResolutionSelectionEntity
 LLM reasons over
groups<br>BIOLARGO, INC.? Raw extracted table<br>Schema Induction 2 Extract chunk id company name accounts payable acc. payable quote acc. payable rationale Cleans the dirty database<br>LLMGenerate Schema AccountsPayable
company name (str)
reporting period (date, norm: mm-dd-yyyy)
accounts payable (float, scale: thousands)
Schema Guidelines 
Library Extracts quotes from chunks and reasoning Relevant?Relevant?Extract... 238 AppTechBioLargo IncBioLargo Inc 17481663970 Accounts payable and accrued expenses Accounts payable   
|$1,$266300 | $1,|3$51747 | 0 The consolidated balance sheet provides acccounts payable and accured expenses as The provides total accounts payable for all as financial statement $1,740$1,663 The accounts payable for BioLargo Inc is according to ...$1,663,000 AnsSynthesizerExecute SQLWrite SQLwer 
 5<br>currency (str) along with the values ... Final Answer<br>**----- End of picture text -----**<br>


Figure 3: SLIDERS overview showing the process of **(1) Contextualization** of chunks, **(2) Inducing the schema** , **(3) Structured Extraction** , **(4) Data Reconciliation** , and **(5) Question Answering** over the final database. An example from the Loong benchmark. 

and resolves inconsistencies into a unified representation. Our key insight is that this process can exploit how each record was produced: every entry is stored with its provenance, extraction rationale, and metadata, enabling a reconciliation agent to inspect supporting evidence and generate updates grounded in the original extraction context. This design also improves efficiency. Rather than performing pairwise comparisons over all extracted records, which scales quadratically with the number of entries, reconciliation restricts reasoning to key-based partitions, substantially reducing the effective problem size.. By combining LLM reasoning with database operations, reconciliation converts local extractions into a globally consistent, queryable state for downstream question answering. To our knowledge, this is the first approach to treat provenance and extraction rationale as first-class signals for reconciliation in LLM-based information extraction. 

We present **SLIDERS** (Scalable Long-document Integration through Decomposed Extraction and Reconciliation System), a framework for structured extraction and reconciliation over long document collections. On three long-context benchmarks that fit within the context window of strong base LLMs, SLIDERS outperforms all baselines, exceeding GPT-4.1 by 6.6 points on average. To evaluate reasoning beyond current context limits, we also introduce two new benchmarks of 100 documents each, totaling 3.9M tokens for Wikipedia articles and 36M tokens for financial filings. On these benchmarks, SLIDERS achieves 78.91% accuracy on Wikipedia and 55.22% on financial documents, compared to 59.80% and 5% for the next best baseline, respectively. 

## **2 From Documents to Structured Reasoning** 

Given a question _q_ and a collection of long documents _D_ , SLIDERS answers _q_ by transforming unstructured text into a globally coherent database. In the following, we overview the research questions (RQ) and our approach, which consists of five tasks, as shown in Figure 3. 

**RQ1:** How to decompose documents for extraction to preserve the required global context for correct interpretation? 

1. The _Contextualized Chunking Task_ . Decompose documents _D_ into context-aware chunks _Cd_ using extracted document-level and structural metadata _M_ , producing locally self-contained units suitable for extraction. 

**RQ2:** How to represent and extract information from chunks of a document with a relational database that preserves all information necessary for answering the question on the original documents? 

2. The _Schema Induction Task._ Derive a schema _S_ from question _q_ and document metadata _M_ , specifying the entities, attributes, and relationships to extract. 

3. The _Structure Extraction Task._ Extract relevant information from document _D_ with metadata _M_ according to schema _S_ for question _q_ to create a set of tables _T_ while preserving provenance and extraction rationales. 

3 

**RQ3:** How to construct a coherent global database from partial, redundant, and potentially conflicting extractions, and answer questions using the database? 

4. The _Data Reconciliation Task._ Cleans tables _T_ generated by the Structure Extraction Task, with respect to question _q_ to enhance its correctness with an SQL-coding agent. 

5. The _Question Answering Task_ . Synthesize the answer to question _q_ by generating and executing SQL queries over the reconciled database _T_ [Liu et al., 2024]. The QA agent iteratively generates queries over the reconciled database: (1) it is given the schema and database, (2) the query is executed and the result returned, (3) the agent refines the query if needed, repeating until a satisfactory answer is produced. 

## **3 Contextualized Extraction** 

## **3.1 The Contextualized Chunking Task** 

To ensure that extracted records are interpretable on their own, we retain the global and layout context as we extract information from documents to avoid issues such as headers detached from tables or orphaned paragraphs [Lewis et al., 2020, Jain et al., 2025]. SLIDERS augments each document _d ∈D_ with metadata _md_ = ( _m_[G] _d[, m] d_[L][)] _[ ∈M]_[ prior to chunking, where the global part] _[ m]_[G] _d_[consists] of a generated title and brief document-level description to provide high-level context shared across all chunks; the local part _m_[L] _d_[captures structural signals such as section headers, tables, and figure] captions, enabling chunk boundaries be formed that aligns with the document’s natural layout. 

Using this enriched representation, we partition each document into chunks _c ∈Cd_ while preserving semantic and structural coherence, avoiding splits within paragraphs, tables, code blocks, or captions. Each chunk is thus locally self-contained and associated with metadata including document identifiers, page indices, and structural tags. 

## **3.2 Schema Induction** 

Unlike free-form extraction, where an LLM generates unstructured natural language responses for each document chunk, relational databases fundamentally differ in their requirement for a rigid schema. In addition, free-form outputs are inherently polymorphic: one chunk might yield a verbose explanation, while another provides a concise numeric value or uses a different unit of measurement (e.g., 75 _[◦]_ F vs. 24 _[◦]_ C). We need to enforce a standard type and format for all values in each column to support SQL operations, like aggregation or filtering. To bridge this gap, we must constrain the extraction process with a task-specific schema. 

**Definition 1.** _The_ Schema Induction Task _takes a question q and extracted document metadata M and produces a relation schema with multiple tables S_ = _{S_ 1 _, S_ 2 _, . . . , Sk}; a table schema S_ = _⟨sn, f_ 1 _, . . . fn⟩ has a schema name sn, and a set of fields fi. A field is defined as a tuple_ 

**==> picture [84 x 11] intentionally omitted <==**

_where fn is the field name; d is a semantic text description; τ is the data type (e.g.,_ _`int` ,_ _`str` ); u is the unit of measurement (e.g., USD, kg); σ is the scale (e.g., millions, thousands); and ρ represents normalization rules (e.g., currency conversion, date formatting)._ 

We create a schema library that provides guidance for constructing task-appropriate relational schemas. Using the library, an LLM first classifies the query type and document type, retrieves the guidelines, and follows it to produce the desired schema. We provide the following question types: “Ordering”, “Multiple Choice”, “Others” and the following document types: “Narration”, “Policy”, “Dataset”, “Others”. More details on the guidelines are present in Appendix A.3. 

## **3.3 Contextualized Extraction with Relevance Gating** 

In our experiment to use in-context learning to extract knowledge for our schema, we observed empirically that hallucination is common when a chunk contains no mention of information for a field [Liu et al., 2023b, BBC, 2025]. The fact that our schema has strict type requirements appears to exacerbate this problem, which we hypothesize to be caused by training biases that prioritize structural compliance over factual abstinence. 

4 

To mitigate this, we introduce a _relevance gate_ prior to extraction. For each chunk, the model first determines if the text contains evidence relevant to the schema entities; we invoke the extraction on this chunk only when this gate passes. This two-stage process ensures that the extraction model is active only when presented with high-signal context, preventing the injection false positives into the database. We evaluate the relevance gate on 20 sampled incorrect predictions across benchmarks. Across 516 chunks, 282 were rejected by the relevance gate, with only 1 false negative, yielding a false-negative rate of 0.4%. This indicates that the relevance gate is not a major source of error; most failures arise downstream (e.g., schema mismatch or reconciliation). 

**Definition 2.** _The_ Structured Extraction Task _, SE, accepts as input question q, document D, metadata M, and schema S_ = _{S_ 1 _, S_ 2 _, . . . , Sk} to produce a set of rows, capturing relationships mentioned in the chunk, in tables T_ = _{T_ 1 _, T_ 2 _, . . . , Tk}, where Ti conforms to schema Si._ 

_Each entry of field f , ef_ = _⟨v, p, r⟩, where v is the normalized field value respecting the unit and scale specified for f , p is the provenance, the minimal textual span supporting the value extracted, and r describes the decision used to map the quote to the extracted value._ 

**Structure Extraction Task** . The Structured Extraction task for a document is formulated as 

**==> picture [184 x 11] intentionally omitted <==**

where _q_[e] is an adaptation of _q_ for the extraction process, and _∪_ denotes table union without removing duplicates. 

**==> picture [229 x 26] intentionally omitted <==**

where _R_ ( _q_[e] _, c, mc, S_ ) is an explicit relevance gate determining if _c_ is relevant to _q_[e] , to minimize hallucination. The extraction of information from a chunk, SELLM, is implemented with in-context learning, generating structured output as JSON objects. 

**Why extraction scales?** (i) **Unbounded corpus size:** The analysis is applied to a chunk at a time, whose size can be chosen to fit in any single model’s context window. (ii) **Parallelism:** The analysis of each chunk (gating and extraction) is independent of each other and can thus be parallelized. 

Table 1: Executor agents for data reconciliation. Each agent applies a distinct reconciliation operation over related records, using provenance and rationale to guide decisions and produce a coherent representation. 

|**Operation**|**Reconciliation Need**|**Decision Making**|**Action**|
|---|---|---|---|
|Deduplication|Semantically identical or near-|Prefer the most precise or ex-|Select a canonical represen-|
||identical rows expressed with|plicitly stated value; assess|tation and collapse redundant|
||different phrasing|specifcity via provenance|rows|
|Confict Reso-|Competing values for the same|Examine provenance and ratio-|Retain<br>the<br>best-supported|
|lution|attribute across related rows|nale to determine which value|value and remove incompati-|
|||is best supported in context|ble alternatives|
|Consolidation|Partial rows capturing comple-|Determine whether attributes|Merge complementary records|
||mentary attributes of the same|can be meaningfully combined|into a more complete repre-|
||entity or fact|across rows without contradic-|sentation and propagate shared|
|||tion|values where appropriate|



## **4 Data Reconciliation** 

Although information extraction from individual document chunks is often accurate in local context, aggregating these extractions across an entire document or document set can still yield an incoherent global representation. Documents distribute, repeat, and refine related information across sections at different levels of granularity. For example, a Wikipedia article may mention an artist’s date of birth and primary profession in the introduction, while later sections discuss additional professions. Each chunk may therefore yield a correct extraction in isolation, yet their aggregation can still produce missing attributes, redundant entries, or conflicting values. Reconciliation addresses this challenge by aligning overlapping, partial, or conflicting records into a coherent relational state. 

5 

**Definition 3.** _The_ Data Reconciliation Task _, DR, accepts as input question q and extracted tables T_ = _SE_ ( _q, D, M, S_ ) _and produces reconciled tables T[′]_ = _{T_ 1 _[′][, T]_ 2 _[ ′][,][ · · ·][ T] k[ ′][}][.]_ 

A naive approach would reason jointly over all extracted rows, requiring pairwise comparisons across the full table and becoming quadratic in the number of entries. Our key observation is that relational structure provides a natural decomposition. Our key observation is that reconciliation can exploit the structure of text itself: although information is distributed across sections, statements about the same entity or claim are typically anchored by a common identifying attribute or set of attributes. In the extracted relational representation, these attributes serve as a primary key. Grouping rows by this key allows us to reconcile overlapping evidence within small, semantically coherent partitions. 

Rather than relying on a fixed reconciliation program, we collect during extraction the evidence needed to reconcile each group: provenance, extraction rationale, and documentlevel metadata for every table entry. We then use an agent to reason over each key-based partition and dynamically generate SQL programs that align duplicate references, integrate complementary attributes, and resolve competing values. In this way, reconciliation converts independently extracted local views into a globally coherent and queryable database for downstream question answering. 

**Algorithm 1** Data Reconciliation (DR) **Require:** Question _q_ , Tables _T_ = _{T_ 1 _, . . . , Tk}_ **Ensure:** Reconciled _T[′]_ = _{T_ 1 _[′][, . . . , T] k[ ′][}]_ 1: _q_[r] _←_ adapt _q_ for reconciliation _% Phase 1: Partitioning for Reconciliation_ 2: **for** each table _Ti ∈T_ **do** 3: _pki ←_ SELECTPRIMARYKEY( _Ti, q_[r] ) _% doc-level, then table-level_ 4: _Ti ←_ RESOLVEPRIMARYKEYENTITIES( _Ti, pki_ ) 5: _Gi ←_ GROUPBYPRIMARYKEY( _Ti, pki_ ) 6: **end for** _% Phase 2: Data Reconciliation Agent_ 7: **for** each table _Ti ∈T_ **do** 8: **for** each group _g ∈Gi_ **in parallel do** 9: **loop** 10: _op ←_ SELECTRECONOP( _g, q_[r] ) 11: **if** _op_ = ∅ **then break** 12: **end if** 13: _sql ←_ RECONCILEGROUP( _g, op_ ) 14: _g ←_ APPLY( _sql, g_ ) 15: **end loop** 16: **end for** _% Non-primary-key entity resolution_ 17: _Ti[′][←]_[R][ESOLVE][N][ON][P][RIMARY][K][EY][E][NTITIES][(][�] _g[g,] pki_ ) 18: **end for** 19: **return** _T[′]_ 

## **Phase 1: Partitioning for Reconciliation** 

_Primary Key Selection._ To determine the primary key, which may consist of one or more columns, we present an LLM agent with (1) an adaptation of the input question _q_ for reconciliation, (2) the table schema, and (3) a sample of extracted rows with their rationales. The LLM identifies the primary key, and to improve robustness we query it three times and use majority voting to select the final key. 

_Entity Resolution of Primary Key Values._ A central step in reconciliation is aligning semantically equivalent key values [Bhattacharya and Getoor, 2007, Fellegi and Sunter, 1969]. For example, the same person may appear as “J. Smith,” “John Smith,” or “Smith, John” across documents. These variants must be mapped to a common representation before the remaining evidence can be integrated. 

We first resolve entities within each document, one column at a time, using an LLM prompt that focuses on semantic equivalence. This acts as a blocking step, reducing the number of distinct references before cross-document resolution [Newcombe, 1967, Papadakis et al., 2016]. We then apply an iterative LLM-based pipeline across documents: the model generates SQL queries to sample rows, inspects evidence from available attributes and extraction rationales, and produces SQL statements to normalize and align key values. If the primary key columns contain free text or exhibit extremely high cardinality, the agent may skip this step. 

_Grouping by Primary Key._ Finally, records are partitioned by primary key using a SQL group-by statement. Each resulting group forms an independent reconciliation unit. 

## **Phase 2: Reconciliation Agent** 

For each partition of records, the reconciliation agent uses provenance, extraction rationale, and document-level metadata to determine how the evidence should be integrated into a coherent representation. 

_Reconciliation Operation Selection._ From empirical analysis, we find that most reconciliation decisions fall into three classes: (1) _deduplication_ , where multiple rows express the same information; (2) _conflict resolution_ , where rows provide competing values for the same attribute; and (3) _consolidation_ , 

6 

Table 2: Benchmark statistics. 

|**Benchmark**|**#**|**Docs**|**# Questions**|**Real/Synth**|**Task Type**|
|---|---|---|---|---|---|
|FinanceBench|1|per Q|150|Real|Extraction, Arithmetic|
|Loong|_∼_11|per Q|50|Real|Retrieval, Aggregation|
|Oolong|1|per Q|192|Synthetic|Classifcation, Aggregation|
|WikiCeleb100||100|22|Real|Aggregation, Comparison|
|FinQ100||100|25|Real|Aggregation, Arithmetic|



where partial rows contribute complementary attributes. The agent writes SQL programs to identify which of these operations is needed by examining row contents, value distributions, and supporting evidence. 

_Evidence Integration._ Conditioned on the selected operation, the agent generates SQL programs that integrate the rows in the partition. It may issue intermediate queries, for example, to compute distinct counts, isolate specific columns, or retrieve related rows, before producing a final update. Because all reconciliation actions are expressed as SQL, the process remains auditable. As a final step, we also apply entity resolution to the remaining non-key columns. 

## **5 “Long-Context” Benchmarks (** _≤_ **360K Tokens)** 

We evaluate SLIDERS on three “long-context” benchmarks, where the inputs can fit in the context window of most of the frontier models, having up to 360k tokens. For such input lengths, _would SLIDERS perform worse given the overhead of chunking and reconciliation?_ 

## **5.1 Benchmarks and Evaluation Metrics** 

**FinanceBench** [Islam et al., 2023] A single-document financial question-answering benchmark targeting realistic analyst style queries. It comprises of 150 questions about publicly traded companies, with evidence drawn from public filings. _Evaluation metrics_ : We use an LLM-as-a-judge setup. Given the question, the gold answer with justification, and the model’s predicted answer with its justification, the judge model determines whether the prediction is correct. 

**Loong** [Wang et al., 2024] A multi-document question-answering benchmark where every document provided is required to answer the query, and omitting any yields an incorrect answer. Covers three domains, two languages: finance (English and Chinese), law (Chinese), and academic research papers (English), with an average of _∼_ 11 documents per instance. _Evaluation metrics._ We use the official Loong scoring procedure, which employs the Loong-provided LLM judge to assess correctness. 

**Oolong** [Bertsch et al., 2025] A long-context reasoning benchmark focused explicitly on aggregation. Each input consists a datapoint from a large dataset and models must first classify and then aggregate those local predictions to produce a global answer. In our experiments we use the Oolong-Synth subset and evaluate on 256K. _Evaluation Metrics_ : For non-numeric questions, we use an LLM-as-ajudge evaluation to determine correctness. For numeric aggregation questions, we adopt Oolong’s metric, which assigns higher scores to predictions with smaller deviation. 

## **5.2 Baselines** 

We compare SLIDERS against several strong baselines. 

**RAG baseline.** We implement a retrieval-augmented generation (RAG) system using Qwen3-4B Embedding as the encoder and GPT 4.1 as the base model for answering. We use dense retrieval with 4096-token chunks, top- _k_ retrieval with _k_ =5 for Loong and Oolong, and _k_ =100 for FinQ100. 

**LongRAG.** [Jiang et al., 2024] We evaluate LongRAG, which groups retrieved chunks into longer contexts (4K-30K tokens) to reduce the total number of retrievals and provide more coherent evidence to the reader LLM (GPT 4.1). 

**GraphRAG.** [Edge et al., 2024] We evaluate GraphRAG, which builds a knowledge graph over the corpus and uses graph-based retrieval with local search (the recommended mode) and GPT 4.1. 

7 

Table 3: Performance comparison across long document QA benchmarks. SLIDERS outperforms RAG, base LM, and RLM baselines. WikiCeleb100 (WC) and FinQ100 (FQ) do not fit in the context window of GPT 4.1. RLM scores 7.4% over 10 documents for FinQ100. Given the low score, we did not run it on FinQ100, which we estimate will cost $2000. Best results are in **bold** . Paired _t_ -tests against the strongest baseline yield _p <_ 0 _._ 005 across all benchmarks. FB is FinanceBench. 

||**“Long-Context” Benchmark (**_<_**360k)**|**“Long-Context” Benchmark (**_<_**360k)**|**“Long-Context” Benchmark (**_<_**360k)**|**“Long-Context” Benchmark (**_<_**360k)**|**3.9M T**|**3.9M T**|**36M T**|**36M T**|
|---|---|---|---|---|---|---|---|---|
||||||||||
|**Models**<br>**LLMs**|**FB**<br>**Loong**<br>**Oolong**|||**Avg.**||**WC**||**FQ**|
||||||||||
|RAG<br>Qwen3-4B & GPT 4.1<br>LongRAG<br>Qwen3-4B & GPT 4.1<br>GraphRAG<br>Qwen3-4B & GPT 4.1|62.67<br>72.00<br>75.33|54.35<br>59.10<br>61.28|11.32<br>22.00<br>22.00|42.77<br>51.03<br>52.87||31.41<br>43.20<br>48.59||5.00<br>28.87<br>$$|
||||||||||
|Basemodel<br>GPT 4.1<br>Basemodel<br>Qwen3.5 122B-A10B|82.00<br>84.67|76.74<br>74.78|45.56<br>24.89|68.69<br>61.44||N.A.<br>N.A.||N.A.<br>N.A.|
||||||||||
|DocETL<br>GPT 4.1<br>Chain of Agent<br>GPT 5 & GPT 5-mini<br>RLM<br>GPT 5 & GPT 5-mini|63.33<br>71.30<br>75.33|75.03<br>54.46<br>72.64|49.00<br>17.11<br>51.42|62.44<br>47.62<br>66.46||54.26<br>$$ 59.80||$$ $$ $$|
||||||||||
|**SLIDERS**<br>GPT 4.1 & GPT 4.1-mini<br>**SLIDERS**<br>Qwen3.5 122B-A10B|**89.33**<br>82.10|**78.57**<br>75.70|64.67<br>**68.00**|**75.56**<br>75.26||**78.91**<br>76.92||55.22|
|||||||||**60.18**|



**Frontier LLM baseline.** We evaluate GPT-4.1 and Qwen3.5 122B-A10B [Qwen Team, 2026] on all benchmarks under the same prompting and evaluation protocol used for SLIDERS. It has a context window of 1M tokens. 

**DocETL** [Shankar et al., 2025] We evaluate DocETL with its V1 greedy optimizer, GPT 4.1, 16Kcharacter chunking (matching SLIDERS), per-question schema generation, per-chunk map extraction, and single-pass reduce synthesis. 

**Chain of Agents (CoA)** [Zhang et al., 2024] We implement Chain of Agents, where an LLM sequentially processes each document chunk, producing a running summary that is passed to the next chunk until a final answer is generated. We use GPT 4.1 as the synthesizer and GPT 4.1-mini as the chunk summarizer. 

**RLM** [Zhang et al., 2025] We evaluate Recursive Language Model agent that writes python programs to decompose, and call LLM over its input. We use 30 iterations, GPT 5 for main LLM and GPT 5-mini for sub LLM. 

## **5.3 Results** 

We present the results on the context-bounded benchmarks that fit in the context window of base LM in Table 3. 

**Observation 1: SLIDERS outperforms all baselines on all the long-document-set questionanswering task, even though the inputs all fit in the context of frontier models.** SLIDERS outperforms the best baseline (GPT 4.1) by an average of 6.6% even though the inputs fit in the LM’s context. The largest difference of 14% over the base LM observed for OOlong demonstrates the advantage of structured reasoning in SLIDERS for questions requiring aggregation over large contexts. Paired _t_ -tests against the strongest baseline (GPT-4.1) yield _p <_ 0 _._ 005 across all benchmarks, confirming statistical significance. 

For Loong, SLIDERS delivers an accuracy of 59.9%, 74.8%, 89.2%, 91.3%, for the Chinese Legal, English Finance, Chinese Finance, and English Papers domains, respectively. Accuracy degradation over the baseline is observed only for the Chinese Legal domain, a classification task on small (16K) legal documents; the chunking and reconciliation steps in SLIDERS introduce overheads, while offering no benefits for this benchmark. 

**Observation 2: SLIDERS’s provenance tracking enhances auditability and interpretability** , thus facilitating **error analysis** . Sampling the discrepancies between SLIDERS’s output and the gold answers, the provenance information in SLIDERS helps pinpoint the cause of the errors. Detailed discussions can be found in Appendix C. The _common errors in SLIDERS_ include (1) not answering questions that need subjective judgement correctly, and (2) misinterpreting terminology, such as 

8 

**==> picture [389 x 103] intentionally omitted <==**

**----- Start of picture text -----**<br>
Finance (EN) Finance (ZH) Legal Papers 100 100<br>100 75 75<br>75 50 50<br>50<br>25 25 25<br>0 0 0<br>Spotlight Comparison Clustering Chain of<br>Reasoning<br>(a) Loong (b) FinanceBench (c) Oolong<br>Domain-Rel.Metrics-Gen.Novel-Gen. Counting UserTimeline<br>Acc (%) Acc (%)<br>Acc (%)<br>**----- End of picture text -----**<br>


Figure 4: Model accuracy across difficulty levels and question types. 

fiscal vs. calendar years. We also found _errors in the human-annotated gold answers_ : (1) the gold sometimes has the wrong value or units, (2) for results that span multiple documents, the gold may not be complete. 

**Observation 3: SLIDERS also works effectively with open-source LLMs, showing that its gains come from the framework rather than reliance on a proprietary frontier model.** When instantiated with Qwen3.5-122B-A10B, SLIDERS achieves an average score of 75.26 on the contextbounded benchmarks, improving substantially over the same model used directly as a base LM (61.44). On Oolong, SLIDERS improves Qwen from 24.89 to 68.00, a gain of more than 43 points, showing that structured reasoning is especially beneficial for aggregation-heavy questions. SLIDERS with Qwen also remains competitive with GPT-4.1-based systems, exceeding the GPT-4.1 base LM average of 68.69 and achieving stronger results on benchmarks such as Oolong and FinQ100. These results show that the benefits of SLIDERS are largely orthogonal to the choice of underlying model: even with an open-source LLM, converting text into persistent structured state and reasoning over it with SQL yields large gains. 

## **5.4 Ablation** 

In Table 4, we ablate key components of SLIDERS: chunking, reconciliation, the reconciliation agent, and the separation of information representation from reasoning. We construct a validation set consisting a random sample of 220 tasks across the three benchmarks. For this split, the performance of SLIDERS is consistent with the full dataset, whereas we see fluctuations in the baseline especially for RAG. We find that chunking is important for all the benchmarks, especially Oolong, where extraction 

Table 4: Ablation study on the val set, removing chunking, reconciliation, and SQL-based QA individually. **Bold** /underline = best/second best. 

|**Model**<br>**F. Bench**|**Loong**<br>**Oolong**|**Loong**<br>**Oolong**|**Avg**|
|---|---|---|---|
|**SLIDERS**<br>**80.00**<br>_w/o_Chunking<br>70.00<br>_w/o_Reconciliation<br>76.70<br>_w/o_Recon+SQL<br>70.00|84.37<br>79.72<br>82.84<br>**84.45**|**64.67**<br>40.00<br>62.42<br>58.62|**74.79**<br>60.34<br>72.71<br>70.74|



from the full text yields a 40.00% accuracy. Reconciliation is particularly important for Finance Bench. Answering the questions directly using an LLM from the extracted table sees a major degradation for both Finance Bench and Oolong. 

## **5.5 Question Type Breakdown** 

Figure 4 breaks down accuracy by question type across all three benchmarks. On **Loong** (Fig. 4a), performance varies more by domain than by difficulty: Finance (ZH) and Papers remain above 83% at all difficulty levels, while Legal stays below 73% throughout. Chain of Reasoning is the hardest category overall, but its impact is domain-dependent, Finance (EN) drops to 39% while Papers rises to 95%. On **FinanceBench** (Fig. 4b), accuracy is consistently high (86.0-96.0%) across domainrelevant, novel, and metric-generated questions. On **Oolong** (Fig. 4c), SLIDERS handles User and Timeline questions well (80% and 100% respectively), with Counting lower at 60.3%. Counting questions require producing exact numerical answers, which depends heavily on the underlying model’s reasoning and classification capabilities, a bottleneck that lies outside SLIDERS’ pipeline. 

9 

## **6 Ultra-Long Document-Set Benchmarks** 

To stress-test large-scale multi-document aggregation, we introduce two document-set benchmarks that exceed the context windows of current frontier models. In practice, we often want to ask many questions over the same document set; accordingly, each benchmark includes a suite of topic-focused questions. Our approach converts documents into reusable structured data, enabling _cost-effective answers to many downstream questions_ and _amortizing the one-time extraction and reconciliation costs_ . 

**WikiCeleb100** comprises Wikipedia articles for the 100 most-viewed celebrity pages from November 2025 to January 2026, totaling **3.9M tokens** . The test set consists of 22 questions on the topic of _debuts_ ; We provide SLIDERS with the representative question: “Who debuted at the youngest age across the following industries: Music, Film, Content Creation, and Other?” Answering this requires extracting each celebrity’s date of birth, debut date, and industry, information scattered across all 100 articles. Once extracted, this schema supports additional questions such as which decade has the most representation of artists. 

**FinQ100** comprises the most recent 10-Q filings from 100 randomly selected SEC-listed companies, totaling **36M tokens** . The test set consists of 25 question on the topic of _long-term borrowing_ . We provide SLIDERS with the representative question: “Which company has the lowest long-term borrowing?” This question is challenging because many companies do not explicitly state that their long-term borrowing is zero, requiring inference from context. 

All the questions, included in Appendix B, require aggregating information across documents. We use LLM-as-a-judge to compare the gold answer against the predicted answer; using partial scoring similar to the Loong benchmark. 

**Observation 1. SLIDERS achieves the state-of-the-art accuracy on the ultra-long document set benchmark.** The benchmarks are too large to run on GPT 4.1, and the performance with RAG is 31.4% and 5.0% for FinQ100. SLIDERS achieves 78.9% on WikiCeleb100, an accuracy similar to the “long-context” benchmark despite the 3.9M token input size. It improves over the best baseline RLM by 19.1%, while being 13 _×_ more cost efficient. 

SLIDERS achieves 55.2% accuracy for FinQ100. FinQ100 is challenging due to its scale (36M tokens). Due to cost constraints ($2000 for a full run on FinQ100), we evaluate RLM on only 10 documents from FinQ100, where it scores 7.4% compared to 65.1% for SLIDERS. (SLIDERS costs $34 to run the complete FinQ100 benchmark.) 

Unlike previous benchmarks, such as Oolong, answers cannot be derived by local reasoning and ignoring global context. The answers are heavily fragmented: SLIDERS extracted 685 rows, whereas the ground truth has just 105 rows across 100 companies. Effective data reconciliation is critical to getting accurate answers for this benchmark. Our ablation study shows the _importance and the effectiveness of our reconciliation agent_ , without which the accuracy drops from 55.22 to 35.81 on FinQ100 and 78.91 to 60.50 on WikiCeleb100. It gathers disparate information across long documents and reason over it collectively. 

**Observation 2. SLIDERS can accelerate the manual question answering on large document sets.** Question answering on FinQ100 is representative of many tasks in the financial domain. Although SLIDERS achieves state-of-the-art accuracy, it is not reliable enough for end-to-end automation. Nevertheless, SLIDERS can significantly streamline the manual workflow: a reviewer can use the tracked provenance to validate and correct each document’s extractions much faster than working from scratch. Once the database has been verified, the reviewer can pose a wide range of questions about the documents, and SLIDERS will generate the corresponding SQL queries. Overall, the system’s interpretability facilitates effective human-in-the-loop feedback, enabling a more trustworthy question-answering platform. 

## **7 Analysis** 

## **7.1 Input Context Length.** 

To understand the scalability of different techniques, we plot the accuracy of SLIDERS against the input token count in Figure 5. For benchmarks with less than 360K tokens, even though the input fits 

10 

**==> picture [413 x 97] intentionally omitted <==**

**----- Start of picture text -----**<br>
Loong FinanceBench WikiCeleb100 FinQ100<br>100<br>80<br>60<br>40<br>20<br>0<br>50k 100k 200k 16K 64k 128k 256k 1M 2M 3M 10M 20M 30M<br>Token Length Token Length Token Length Token Length<br>SLIDERS RLM GPT RAG<br>Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 5: Accuracy vs. token length across benchmarks. 

within the LLM model window, the accuracy tends to degrade with increasing token length for all the baselines. SLIDERS’s accuracy fluctuations align with expected input variability. The scalability of SLIDERS is more apparent with the results of the WikiCeleb100 and FinQ100 benchmarks, as its accuracy stays roughly the same up to 35M tokens. RLM can handle up to 4M of WikiCeleb100, but with a lower accuracy; it is too costly to run on FinQ100. The other baselines cannot handle documents over 1M tokens. 

## **7.2 Schema Induction Robustness** 

A natural concern is whether SLIDERS is sensitive to the choice of schema-induction model. We generate schemas with three models of varying capability (GPT-4.1-mini, GPT-4.1, GPT5) on Loong and FinanceBench. The resulting schemas differ substantially: GPT-4.1 averages 1.0 tables and 3.3 fields per question vs. GPT-5 with 1.54 tables and 13.3 fields, a 4 _×_ complexity gap. Yet downstream accuracy remains stable (Table 5): Loong average accuracy spans just 2.1 points across all three models, and FinanceBench stays within 3.3 points, demonstrating that schema induction is not a fragile 

Table 5: Accuracy across schema-induction models. **Bold** indicates best per dataset; ∆ is the range (max _−_ min). 

||**Dataset**|**4.1**|**4.1-mini**|**5**|∆|
|---|---|---|---|---|---|
||Loong Papers|**91.30**|89.96|88.00|3.30|
||Loong Legal<br>Loong Finance EN<br>Loong Finance ZH|64.12<br>**74.50**<br>**93.96**|**68.34**<br>68.10<br>90.46|61.26<br>73.10<br>93.20|7.08<br>6.40<br>3.50|
||Loong Avg<br>FinanceBench|**80.97**<br>76.71|79.22<br>**80.00**|78.89<br>**80.00**|2.08<br>3.33|



bottleneck. One caveat is that more complex schemas can increase reconciliation difficulty, postreconciliation ranges widen modestly on Loong, suggesting diminishing returns from overly detailed schemas. 

## **7.3 Evaluation Reliability** 

**LLM-as-a-judge variance.** We quantify evaluation variance by running GPT-4.1 (temperature 0.7) three times per instance: FinanceBench std = 0.47, Loong std = 0.31, Oolong std = 1.02, WikiCeleb std = 0.21, FinQ100 std = 0.38. Low variance across benchmarks indicates stable evaluation. 

## **Human validation of LLM-as-a-judge.** To 

validate automatic evaluation, we manually annotated 50 questions each from FinanceBench (for both SLIDERS and GPT-4.1) and Oolong (for SLIDERS), yielding a combined Cohen’s _κ_ = 0 _._ 758 (substantial agreement). Table 6 breaks down agreement by condition. Notably, SLIDERS errors on FinanceBench are exclusively false negatives (4 FN, 0 FP), indicating that our reported scores are conservative estimates of true accuracy. 

Table 6: Manual evaluation on 50 questions each. FN/FP denote false negatives/positives relative to human labels. 

||**Dataset**<br>FinanceBench<br>FinanceBench<br>Oolong<br>Combined|**System**<br>SLIDERS<br>GPT-4.1<br>SLIDERS|**Cohen’s**_κ_<br>0.769<br>0.646<br>0.855<br>0.758|**FN**<br>4<br>3<br>1<br>—|**FP**<br>0<br>2<br>1<br>—|
|---|---|---|---|---|---|



11 

## **7.4 Cost Analysis.** 

We analyze the cost of SLIDERS across benchmarks and find an average cost $0.76 per question. Approximately 40% of this cost is incurred by the Entity Resolution task, which must scan the entire table to identify potential matches. A detailed per-task cost breakdown is provided in the Appendix A.6. Compared to RLM, an agent scaffolding baseline, SLIDERS achieves equal or lower cost. Notably, for tasks where the schema can be predefined, the cost of SLIDERS is amortized across queries. To illustrate, consider the WikiCeleb100 benchmark: running GPT-4.1 with an infinite context window would cost approximately $171.60, whereas SLIDERS costs $13.10. This advantage becomes more pronounced on FinQ100, where the respective costs are $1800 and $34.63. 

## **7.5 Latency Analysis.** 

SLIDERS operates in two modes depending on whether the document collection is reused. In the _end-to-end_ setting, where a structured representation is built from scratch for a single query, average latency is 2.6 minutes on Loong and 3.0 minutes on FinanceBench. This is higher than a single LLM call, but SLIDERS targets complex multi-document questions where accuracy matters more than sub-second response times. In the _amortized_ setting, the offline phase (extraction + reconciliation) runs once per collection and the online phase (schema-guided SQL generation) takes _∼_ 25 seconds per question. On WikiCeleb100 (100 documents), the full offline pipeline completes in _∼_ 16 minutes: schema induction (20s), extraction (6 min, parallelized), and reconciliation (9.7 min). For comparison, GraphRAG requires 2.3 hours and $182 to index the same collection while achieving 48.59% accuracy. SLIDERS reaches 78.91% at a fraction of the time and cost. 

## **7.6 Case Study: Multi-Document Summarization.** 

To demonstrate generality beyond QA, we ran SLIDERS with the prompt “Summarise the given research papers” on three ML papers from the Loong benchmark. The system induced a 5-table schema: PaperSummary (title, authors, abstract), PaperContributions, PaperMethods, PaperFindings, and PaperConclusions, each with paper_title as linking key and free-text fields. It extracted 117 records across 3 papers, then aggregated them into coherent per-paper summaries. Because extraction runs per-chunk rather than over the full corpus, it avoids information loss from context-window pressure seen in direct LLM baselines. This demonstrates that SLIDERS extends beyond tabular QA to open-ended synthesis tasks. 

## **8 Reconciliation Agent Analysis** 

We perform an in-depth analysis of the reconciliation agent across five datasets, as shown in Figure 6. For the existing benchmarks, whose contexts are still relatively moderate ( _≤_ 360k tokens), reconciliation reduces extracted tables from 10.25 rows to 7.48 rows on average, with 7.22 unique primary keys per table. For the much larger document collections, WikiCeleb100 and FinQ100 contain 101 and 128 unique primary keys, respectively, making reconciliation a core part of the pipeline rather than a minor cleanup step. 

Overall, the reconciliation agent is efficient, requiring only 1.28 iterations per primary-key group on average. However, the figure shows that its behavior differs substantially across datasets, both in how much reduction it achieves and in which operations it relies on. 

**Row reduction.** Figure 6a,6d shows that reconciliation consistently reduces redundancy within primary-key groups. After reconciliation, the average number of rows per primary key is close to one for nearly all datasets. The largest effect appears on FinQ100, where the average rows per key drop from above 5 before reconciliation to nearly 1 after reconciliation. This is also reflected in the reduction-density plot, where FinQ100 is concentrated at high row-reduction rates, with many groups reduced by roughly 70-90%. WikiCeleb100 also shows substantial reduction, while FinanceBench and the Loong Finance splits exhibit moderate but consistent improvements. In contrast, Loong Legal changes very little: its documents are often shorter than the chunk size, so extraction already produces nearly one row per key and leaves little redundancy for reconciliation to remove. 

12 

**==> picture [391 x 274] intentionally omitted <==**

**----- Start of picture text -----**<br>
FinQ100 FinanceBench Loong Finance (EN) Loong Finance (ZH)<br>Loong Legal Loong Papers WikiCeleb100<br>· 10 [−] [2]<br>6 1<br>0 . 75<br>4<br>0 . 5<br>2<br>0 . 25<br>0 0<br>0 25 50 75 100 0 1 2 3 4 5<br>Row Reduction (%) Iterations<br>(a) Row Reduction Distribution (b) Resolution Iterations<br>FinQ100<br>Fin-EN FinQ100<br>Fin-ZH WikiCeleb<br>FinBench Fin-ZH<br>WikiCeleb FinBench<br>Papers Fin-EN<br>Legal Papers Before<br>Legal After<br>0 20 40 60 80 100<br># PK Groups 1 2 3 4 5<br>Dedup. Consol. Resolve Conflict Avg Rows / PK<br>(c) Operation Breakdown (d) Rows per Key: Before vs. After<br>x<br> ≤<br>Density<br>Fraction of PK Groups<br>**----- End of picture text -----**<br>


Figure 6: Analysis of the reconciliation agent across seven datasets: (a) density of per-group row reduction, (b) CDF of groups resolved within _k_ iterations, (c) operation-type breakdown, and (d) average rows per primary key before and after reconciliation. The results show substantial reduction in redundant rows and rapid convergence in most datasets, with different benchmarks exhibiting different mixes of reconciliation operations. 

**Operations used.** The operation breakdown in Figure 6b reveals different reconciliation regimes across datasets. FinQ100 and the Loong Finance datasets are dominated by conflict resolution, with deduplication as the next most common operation. This indicates that many chunks extract competing values for the same key, requiring the agent to compare provenance and rationales to decide which value to retain. WikiCeleb100 exhibits a different pattern: deduplication and consolidation are more prominent, suggesting that sources are often mutually consistent but contribute complementary partial records that must be merged into a canonical row. FinanceBench lies between these two extremes, using both conflict resolution and deduplication with only a small amount of consolidation. Long Papers are handled almost entirely by deduplication, while Loong Legal triggers almost no reconciliation operations. 

**Iterations to convergence.** Despite often substantial row reduction, reconciliation usually converges quickly. Figure 6c shows that for most datasets, around 90% or more of primary-key groups are resolved in a single iteration, and nearly all groups are resolved within at most a few iterations. WikiCeleb100 is the main exception: only about half of its groups are resolved in the first iteration, and additional passes are needed before the CDF reaches 1.0. This suggests that the number of iterations is driven not simply by how many rows are present in a group, but by the structure of the ambiguity. In particular, FinQ100 has far more rows per key before reconciliation than WikiCeleb100, yet it converges in fewer iterations on average. A likely explanation is that many FinQ100 groups involve direct duplicate removal or selecting among conflicting values, whereas WikiCeleb100 more often requires iterative consolidation of complementary evidence across sources. 

**Takeaway.** Overall, reconciliation does more than remove duplicate rows: it transforms noisy, local extractions into a near-canonical structured representation with approximately one row per primary key. The figure also highlights that reconciliation is highly dataset-dependent. Large financial corpora 

13 

primarily induce conflicts that must be resolved, whereas multi-document biographical corpora more often require consolidation across partially overlapping evidence. 

## **9 Related Works** 

**Long-Context Modeling and Systems** . Prior work on long-context processing spans three broad categories. _Training-based methods_ improve model capacity through position-agnostic training [He et al., 2024], multi-document instruction tuning [Liu et al., 2025], and representation-level approaches such as corpus-specific context memories [Eyuboglu et al., 2025] or RL-based compression into fixed working memory [Yu et al., 2025]. In contrast, SLIDERS processing each chunk independently and hence can process text beyond a single LLM’s context window. _Agentic frameworks_ divide work across specialized roles: using leader–worker coordination [Zhao et al., 2024], orchestrating sequential reasoning [Zhang et al., 2024], traversing document-derived fact graphs [Li et al., 2024], and systems delegating sub-tasks across models of varying capability [Narayan et al., 2025, Zhang et al., 2025]. Other works chunk documents and aggregate outputs via summarization [Zhou et al., 2025, Gidiotis and Tsoumakas, 2020], schema-driven extraction (DOCETL) [Shankar et al., 2025], or domain-specific retrieval [Sarmah et al., 2024, Han et al., 2024, Wang et al., 2025b, Choe et al., 2025]. However, chunk-based methods such as DocETL still face the fundamental _aggregation bottleneck_ as the number of documents grows. In contrast, SLIDERS converts local extractions into a unified relational representation, reconciles them into a coherent global state, and reasons over that state using LLM-generated SQL. This yields auditable outputs with full provenance while reducing the need to aggregate large amounts of raw text at answer time. 

**Intermediate Representations and SQL Reasoning** . Structured intermediate representations and SQL-based abstractions have emerged as effective tools for building reliable LLM systems. Textto-SQL systems [Yu et al., 2018, Li et al., 2023] translate natural language into executable queries, others [Cheng et al., 2022, Liu et al., 2024] extends SQL with embedded LLM calls for semantic reasoning. [Patel et al., 2025] introduces semantic operators that extend the relational model with AI-based filtering, joining, and aggregation over unstructured text. Beyond SQL, formal intermediate representations have been used to compress context and improve reliability in dialogue systems [Joshi et al., 2025] and agentic workflows [Wang et al., 2025a]. SLIDERS uses structured reasoning for long-document QA. 

## **10 Conclusion** 

We propose separating information representation from reasoning: SLIDERS extracts relevant data from document sets into a relational database with an automatically induced schema, then performs downstream reasoning via SQL queries. A data-reconciliation agent reasons over provenance and extraction rationales to consolidate partial information and resolve inconsistencies. Our approach outperforms baselines on existing benchmarks even when documents fit within context windows, and scales to corpora of 100 documents (up to 36M tokens) far exceeding current context limits beating baselines by upto 11 times. 

**Limitations.** (1) SLIDERS relies on schema induction to structure extraction; while we show robustness across a 4 _×_ schema complexity range, tasks that resist relational modeling (e.g., highly subjective or abstract cross-document reasoning) may not benefit. (2) The pipeline requires multiple sequential LLM calls, resulting in higher latency (2-3 min end-to-end) than single-call baselines; this is acceptable for accuracy-critical workflows but not for real-time applications. (3) Evaluation relies on LLM-as-a-judge, which we validate with human annotations (Cohen’s _κ_ = 0 _._ 758), but residual noise remains. (4) SLIDERS achieves 55% on FinQ100, insufficient for full automation in high-stakes financial analysis. For safety-critical deployments, we recommend human-in-the-loop verification: SLIDERS’s provenance tracking enables reviewers to efficiently validate each extraction against source documents. When provenance is incorrect (e.g., wrong text span), the reconciliation agent may make faulty decisions; our verification of 410 facts found 99.03% correct, and users can inspect source quotes directly. 

14 

## **Acknowledgement** 

We thank Shicheng Liu, Nikil Selvam, Aryaman Arora, Yanzhe Zhang, Cyrus Zhou, Adit Negi, Vedant Singh, Tony Liu, Jiuding Sun, Sina Semnani and the members of the Stanford OVAL and Stanford NLP for helpful discussion about the project and comments on the manuscript. This work is supported in part by the Verdant Foundation, Hasso Plattner Institute, Microsoft Azure AI credits, Itaú Unibanco, BMO Financial Group, and the Stanford Human-Centered Artificial Intelligence (HAI) Institute. 

## **References** 

- C. Auer, M. Lysak, A. Nassar, M. Dolfi, N. Livathinos, P. Vagenas, C. B. Ramis, M. Omenetti, F. Lindlbauer, K. Dinkla, et al. Docling technical report. _arXiv preprint arXiv:2408.09869_ , 2024. 

- Y. Bai, X. Lv, J. Zhang, H. Lyu, J. Tang, and Z. Huang. Zhengxiao du, xiao liu, aohan zeng, lei hou, et al. 2023. longbench: A bilingual, multitask benchmark for long context understanding. _arXiv preprint arXiv:2308.14508_ , 2023. 

- BBC. Representation of bbc news content in ai assistants. `https://www.bbc.co.uk/ aboutthebbc/documents/bbc-research-into-ai-assistants.pdf` , Feb. 2025. Accessed: 2025-12-04. 

- A. Bertsch, A. Pratapa, T. Mitamura, G. Neubig, and M. R. Gormley. Oolong: Evaluating long context reasoning and aggregation capabilities. _arXiv preprint arXiv:2511.02817_ , 2025. 

- I. Bhattacharya and L. Getoor. Collective entity resolution in relational data. _ACM Transactions on Knowledge Discovery from Data (TKDD)_ , 1(1):5–es, 2007. 

- Z. Cheng, T. Xie, P. Shi, C. Li, R. Nadkarni, Y. Hu, C. Xiong, D. Radev, M. Ostendorf, L. Zettlemoyer, et al. Binding language models in symbolic languages. _arXiv preprint arXiv:2210.02875_ , 2022. 

- J. Choe, J. Kim, and W. Jung. Hierarchical retrieval with evidence curation for open-domain financial question answering on standardized documents. In W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, editors, _Findings of the Association for Computational Linguistics: ACL 2025_ , pages 16663–16681, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.855. URL `https://aclanthology. org/2025.findings-acl.855/` . 

- G. Del Fiol, T. E. Workman, and P. N. Gorman. Clinical questions raised by clinicians at the point of care: a systematic review. _JAMA internal medicine_ , 174(5):710–718, 2014. 

- D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt, and J. Larson. From local to global: A graph rag approach to query-focused summarization. _arXiv preprint arXiv:2404.16130_ , 2024. 

- S. Eyuboglu, R. Ehrlich, S. Arora, N. Guha, D. Zinsley, E. Liu, W. Tennien, A. Rudra, J. Zou, A. Mirhoseini, et al. Cartridges: Lightweight and general-purpose long context representations via self-study. _arXiv preprint arXiv:2506.06266_ , 2025. 

- I. P. Fellegi and A. B. Sunter. A theory for record linkage. _Journal of the American statistical association_ , 64(328):1183–1210, 1969. 

- A. Gidiotis and G. Tsoumakas. A divide-and-conquer approach to the summarization of long documents. _IEEE/ACM Transactions on Audio, Speech, and Language Processing_ , 28:3029–3040, 2020. 

- O. Goldman, A. Jacovi, A. Slobodkin, A. Maimon, I. Dagan, and R. Tsarfaty. Is it really long context if all you need is retrieval? towards genuinely difficult long context nlp. _arXiv preprint arXiv:2407.00402_ , 2024. 

- A. Gu and T. Dao. Mamba: Linear-time sequence modeling with selective state spaces. In _First conference on language modeling_ , 2024. 

15 

- S. Han, H. Kang, B. Jin, X.-Y. Liu, and S. Y. Yang. Xbrl agent: Leveraging large language models for financial report analysis. In _Proceedings of the 5th ACM International Conference on AI in Finance_ , ICAIF ’24, page 856–864, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400710810. doi: 10.1145/3677052.3698614. URL `https://doi.org/ 10.1145/3677052.3698614` . 

- J. He, K. Pan, X. Dong, Z. Song, L. LiuYiBo, Q. Qianguosun, Y. Liang, H. Wang, E. Zhang, and J. Zhang. Never lost in the middle: Mastering long-context question answering with positionagnostic decompositional training. In L.-W. Ku, A. Martins, and V. Srikumar, editors, _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 13628–13642, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.736. URL `https://aclanthology.org/2024. acl-long.736/` . 

- Z. Hong, Z. Yuan, Q. Zhang, H. Chen, J. Dong, F. Huang, and X. Huang. Next-generation database interfaces: A survey of llm-based text-to-sql. _IEEE Transactions on Knowledge and Data Engineering_ , 2025. 

- C.-P. Hsieh, S. Sun, S. Kriman, S. Acharya, D. Rekesh, F. Jia, Y. Zhang, and B. Ginsburg. Ruler: What’s the real context size of your long-context language models? _arXiv preprint arXiv:2404.06654_ , 2024. 

- P. Islam, A. Kannappan, D. Kiela, R. Qian, N. Scherrer, and B. Vidgen. Financebench: A new benchmark for financial question answering. _arXiv:2311.11944_ , 2023. 

- A. Jain, P. Aggarwal, and A. Saladi. Autochunker: Structured text chunking and its evaluation. In _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 6: Industry Track)_ , pages 983–995, 2025. 

- Z. Jiang, X. Shi, and J. Lin. Longrag: Enhancing retrieval-augmented generation with long-context llms. _arXiv preprint arXiv:2406.15319_ , 2024. 

- C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, and K. Narasimhan. Swe-bench: Can language models resolve real-world github issues? _arXiv preprint arXiv:2310.06770_ , 2023. 

- H. Joshi, S. Liu, J. Chen, L. Weigle, and M. Lam. Controllable and reliable knowledge-intensive task-oriented conversational agents with declarative genie worksheets. In W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, editors, _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 27264–27308, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/ v1/2025.acl-long.1323. URL `https://aclanthology.org/2025.acl-long.1323/` . 

- P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. _Advances in neural information processing systems_ , 33:9459–9474, 2020. 

- H. Li, J. Zhang, C. Li, and H. Chen. Resdsql: Decoupling schema linking and skeleton parsing for text-to-sql. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 37, pages 13067–13075, 2023. 

- H. Li, X. Chen, Z. Xu, D. Li, N. Hu, F. Teng, Y. Li, L. Qiu, C. J. Zhang, L. Qing, et al. Exposing numeracy gaps: A benchmark to evaluate fundamental numerical abilities in large language models. In _Findings of the Association for Computational Linguistics: ACL 2025_ , pages 20004–20026, 2025. 

- S. Li, Y. He, H. Guo, X. Bu, G. Bai, J. Liu, J. Liu, X. Qu, Y. Li, W. Ouyang, W. Su, and B. Zheng. GraphReader: Building graph-based agent to enhance long-context abilities of large language models. In Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, editors, _Findings of the Association for Computational Linguistics: EMNLP 2024_ , pages 12758–12786, Miami, Florida, USA, Nov. 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-emnlp.746. URL `https://aclanthology.org/2024.findings-emnlp.746/` . 

16 

- G. K.-M. Liu, B. Shi, A. Caciularu, I. Szpektor, and A. Cohan. MDCure: A scalable pipeline for multi-document instruction-following. In W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, editors, _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 29258–29296, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.1418. URL `https://aclanthology.org/2025.acl-long.1418/` . 

- N. F. Liu, K. Lin, J. Hewitt, et al. Lost in the middle: How language models use long contexts. _arXiv:2307.03172_ , 2023a. 

- N. F. Liu, T. Zhang, and P. Liang. Evaluating verifiability in generative search engines. _arXiv preprint arXiv:2304.09848_ , 2023b. 

- S. Liu, J. Xu, W. Tjangnaka, S. Semnani, C. Yu, and M. Lam. SUQL: Conversational search over structured and unstructured data with large language models. In K. Duh, H. Gomez, and S. Bethard, editors, _Findings of the Association for Computational Linguistics: NAACL 2024_ , pages 4535–4555, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.18653/v1/ 2024.findings-naacl.283. URL `https://aclanthology.org/2024.findings-naacl.283/` . 

- W. C. Mann and S. A. Thompson. Rhetorical structure theory: Toward a functional theory of text organization. _Text-interdisciplinary Journal for the Study of Discourse_ , 8(3):243–281, 1988. 

- I. Mirzadeh, K. Alizadeh, H. Shahrokhi, O. Tuzel, S. Bengio, and M. Farajtabar. Gsm-symbolic: Understanding the limitations of mathematical reasoning in large language models. _arXiv preprint arXiv:2410.05229_ , 2024. 

- A. Narayan, D. Biderman, S. Eyuboglu, A. May, S. Linderman, J. Zou, and C. Re. Cost-efficient collaboration between on-device and cloud language models. In _Forty-second International Conference on Machine Learning_ , 2025. 

- H. B. Newcombe. Record linking: the design of efficient systems for linking records into individual and family histories. _American Journal of Human Genetics_ , 19(3 Pt 1):335, 1967. 

- G. Papadakis, J. Svirsky, A. Gal, and T. Palpanas. Comparative analysis of approximate blocking techniques for entity resolution. _Proceedings of the VLDB Endowment_ , 9(9):684–695, 2016. 

- L. Patel, S. Jha, M. Pan, H. Gupta, P. Asawa, C. Guestrin, and M. Zaharia. Semantic operators and their optimization: Enabling llm-based data processing with accuracy guarantees in lotus. _Proceedings of the VLDB Endowment_ , 18(11):4171–4184, 2025. 

- Qwen Team. Qwen3.5: Towards native multimodal agents. `https://qwen.ai/blog?id=qwen3.5` , Feb. 2026. Blog post. 

- B. Sarmah, D. Mehta, B. Hall, R. Rao, S. Patel, and S. Pasquali. Hybridrag: Integrating knowledge graphs and vector retrieval augmented generation for efficient information extraction. In _Proceedings of the 5th ACM International Conference on AI in Finance_ , ICAIF ’24, page 608–616, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400710810. doi: 10.1145/3677052.3698671. URL `https://doi.org/10.1145/3677052.3698671` . 

- S. Shankar, T. Chambers, T. Shah, A. G. Parameswaran, and E. Wu. Docetl: Agentic query rewriting and evaluation for complex document processing. _Proc. VLDB Endow._ , 18(9):3035–3048, Sept. 2025. ISSN 2150-8097. doi: 10.14778/3746405.3746426. URL `https://doi.org/10.14778/ 3746405.3746426` . 

- L. B. Sollaci and M. G. Pereira. The introduction, methods, results, and discussion (imrad) structure: a fifty-year survey. _Journal of the medical library association_ , 92(3):364, 2004. 

- R. Van De Schoot, J. De Bruin, R. Schram, P. Zahedi, J. De Boer, F. Weijdema, B. Kramer, M. Huijts, M. Hoogerwerf, G. Ferdinands, et al. An open source machine learning framework for efficient and transparent systematic reviews. _Nature machine intelligence_ , 3(2):125–133, 2021. 

- H. Wang, C. M. Poskitt, and J. Sun. Agentspec: Customizable runtime enforcement for safe and reliable llm agents. _arXiv preprint arXiv:2503.18666_ , 2025a. 

17 

- M. Wang, L. Chen, C. Fu, et al. Leave no document behind: Benchmarking long-context llms with extended multi-doc qa (loong). In _EMNLP_ , 2024. arXiv:2406.17419. 

- X. Wang, J. Chi, Z. Tai, T. S. T. Kwok, M. Li, Z. Li, H. He, Y. Hua, P. Lu, S. Wang, et al. Finsage: A multi-aspect rag system for financial filings question answering. _arXiv preprint arXiv:2504.14493_ , 2025b. 

- A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, et al. Qwen3 technical report. _arXiv preprint arXiv:2505.09388_ , 2025. 

- H. Yu, T. Chen, J. Feng, J. Chen, W. Dai, Q. Yu, Y.-Q. Zhang, W.-Y. Ma, J. Liu, M. Wang, et al. Memagent: Reshaping long-context llm with multi-conv rl-based memory agent. _arXiv preprint arXiv:2507.02259_ , 2025. 

- T. Yu, R. Zhang, et al. Spider: A large-scale human-labeled dataset for text-to-sql. In _EMNLP_ , 2018. 

- A. L. Zhang, T. Kraska, and O. Khattab. Recursive language models, 2025. URL `https://arxiv. org/abs/2512.24601` . 

- Y. Zhang, T. Du, Y. Sun, L. Donohue, and R. Dai. Form 10-q itemization. In _Proceedings of the 30th ACM International Conference on Information & Knowledge Management_ , pages 4817–4822, 2021. 

- Y. Zhang, R. Sun, Y. Chen, T. Pfister, R. Zhang, and S. Arik. Chain of agents: Large language models collaborating on long-context tasks. _Advances in Neural Information Processing Systems_ , 37: 132208–132237, 2024. 

- J. Zhao, C. Zu, X. Hao, Y. Lu, W. He, Y. Ding, T. Gui, Q. Zhang, and X. Huang. LONGAGENT: Achieving question answering for 128k-token-long documents through multi-agent collaboration. In Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, editors, _Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing_ , pages 16310–16324, Miami, Florida, USA, Nov. 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.912. URL `https://aclanthology.org/2024.emnlp-main.912/` . 

- Z. Zhou, C. Li, X. Chen, S. Wang, Y. Chao, Z. Li, H. Wang, Q. Shi, Z. Tan, X. Han, X. Shi, Z. Liu, and M. Sun. LLM _×_ MapReduce: Simplified long-sequence processing using large language models. In W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, editors, _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 27664–27678, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.1341. URL `https://aclanthology.org/ 2025.acl-long.1341/` . 

18 

## **A Additional Methodology Details** 

## **A.1 Question Decomposition** 

The original question posed by the user is not directly suitable for all stages of the pipeline. In particular, extraction and reconciliation are intermediate steps that impose different requirements on task formulation. Extraction should enumerate all relevant candidate evidence from each document chunk, while reconciliation should combine these candidates into a globally consistent database. Conditioning either stage on the original question can cause them to prematurely answer the question rather than constructing the intermediate representation. 

Consider the query: “Return the second poem about the Great Wall of China.” Suppose the first chunk contains one relevant poem (P1), and the second chunk contains two relevant poems (P2, P3). If the extractor is conditioned on the original question, it may incorrectly interpret the ordinal constraint “second” at the chunk level: it extracts nothing from the first chunk (since no second poem exists locally) and extracts P3 from the second chunk (as the second poem within that chunk). This results in an incorrect global outcome, since the true second poem across the entire document collection is P2. Similar issues arise for other global constraints such as superlatives, rankings, and aggregations. 

To prevent such premature answer-oriented behavior, SLIDERS decomposes the input question _q_ into component-specific queries. We generate an extraction query _qe_ that specifies what information should be identified from each document chunk, and a reconciliation query _qr_ that defines how partial extractions should be combined and cleaned to form a coherent database. Schema induction and final answer synthesis, in contrast, operate directly on the original question _q_ , as these stages are explicitly responsible for capturing the full semantic intent of the user’s request and producing the final answer. 

## **A.2 Context-Aware Chunking** 

Each chunk retains the raw text as well as its structural metadata, including the current heading path (e.g., Header 1 _→_ Header 1.1 _→_ Header 1.1.3), the chunk index, the document title, and the document description. This ensures that every chunk is self-contained, carrying both global and local context needed for faithful extraction. 

## **A.3 Schema Induction** 

Here we provide the schema library we provide SLIDERS. 

|`# `|`Question`<br>`Type`<br>`Guidelines`||||||||||||||||||||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|`##`|`Ordering`<br>`and`<br>`Retrieval`<br>`Questions`||||||||||||||||||||
|`- `|`The`<br>`schema`<br>`should`<br>`not`<br>`contain`<br>`the`<br>`index `|`of the`<br>`item`|`since`||||`extraction`<br>`will `||||||`be `|`done `||`at `|`chunk`||`level ,`||
||`and won ’t have`<br>`the`<br>`global`<br>`index.`||||||||||||||||||||
|`##`|`Multiple`<br>`Choice`<br>`Questions`||||||||||||||||||||
|`- `|`The`<br>`schema`<br>`should`<br>`contain`<br>`reasons`<br>`for and`<br>`against`<br>`each`|||||`option.`|||||||||||||||
|`- `|`Include`<br>`fields`<br>`such as ’option_label ’ (e.g., ’option `|||`A `||`name ’, `||`’option B name`||||||`’, ’option `||||`C name`|||
||`’...) , ’option_text ’, and ’support_label ’.`||||||||||||||||||||
|`- `|`You can use`<br>`enums as a field`<br>`which`<br>`says: `|`’supported ’, `|||`’not_supported ’, ’insufficient_evidence ’, ’`||||||||||||||||
||`contradicted ’.`||||||||||||||||||||
|`# `|`Document`<br>`Type`<br>`Guidelines`||||||||||||||||||||
|`##`|`Dataset`<br>`Documents`||||||||||||||||||||
|`- `|`You`<br>`should`<br>`include`<br>`any`<br>`given`<br>`identifier`|`so that`<br>`datapoints`|||||`can `||`be`||`aligned`||`across`|||`pages/chunks.`|||||
|`- `|`Include`<br>`at least`<br>`one`<br>`label`<br>`field (e.g., `|`’label ’) plus`|||`any`||`other`|||`attributes`|||`needed `|||`to `|`answer or`||||
||`aggregate`<br>`for the`<br>`question.`||||||||||||||||||||
|`## `|`Story/Narrative`<br>`Documents`||||||||||||||||||||
|`- `|`Define an ’Entity ’ table to`<br>`capture`<br>`the`|`relationship`||`’entity `||||`$\`|`leftrightarrow$`||||||`properties`||||||
||`mentioned`<br>`on this page ’.`||||||||||||||||||||
|`- `|`Include`<br>`fields`<br>`such as ’entity_name ’, ’entity_type ’, `|||`and a `|||`field`||||`summarizing`|||`important`||||`properties`|||
||`on this`<br>`page (e.g., ’entity_information ’).`||||||||||||||||||||
|`- `|`If the`<br>`question`<br>`depends`<br>`on`<br>`actions`<br>`or plot , define an `||||`’Event ’ `|||`(or`|||`similar) `||`table`|||`representing`||||`’`|
||`event $\ leftrightarrow$`<br>`participants /time/location`|||`’, with`||||`one `|||`row per`|`event.`|||||||||
|`- `|`Use`<br>`canonicalize`<br>`the`<br>`names`<br>`across`<br>`chunks`|`when`<br>`possible`||||`so `|`that`||`information`||||`about`|||`the`|`same`||`entity/`||
||`event`<br>`can be merged`<br>`across`<br>`chunks.`||||||||||||||||||||
|`## `|`Policy/Whitepaper`<br>`Documents`||||||||||||||||||||
|`- `|`You`<br>`should`<br>`have`<br>`fields`<br>`like`<br>`stakholders , `|`implementation`||||`steps`||`, `|`conditions`|||`, `|`etc`||||||||
|`- `|`If it talks`<br>`about`<br>`recommendations , then`|`you`<br>`make`<br>`sure `|||`to add`|||`fields`<br>`like`||||`target_area`|||||`/domain ,`||||
||`intended`<br>`outcome , advantages , disadvantages , etc`||||||||||||||||||||



19 

## **A.4 SLIDERS implementation details** 

**Metadata Extraction** To identify headers, tables and figures in a document, we use DocLing Auer et al. [2024], an off-the-shelf pdf to markdown converter and parse the markdown to identify the structural metadata. In scenarios where the original pdf is not available, we preprocess the raw text to identify, tables and section headers using LLMs. 

**LLM Models** We use GPT 4.1 and GPT 4.1 mini for our experiments. For parts that require planning and thinking such as schema generation and reconciliation agent, we use GPT 4.1. For other tasks such as extraction from chunks, the metadata generation, we use GPT 4.1 mini. 

## **A.5 Sample Configuration** 

We provide a representative configuration file for the Loong Finance (EN) benchmark. 

```
experiment:loong
system:sliders
system_config :
generate_task_guidelines :false
rephrase_question :
enabled:true
prompt_file:sliders/ rephrase_question_component .prompt
library_of_guidelines_path :sliders_taxonomy .json
generate_schema :
add_extra_information_class :false
add_document_text :true
generate_schema_type :library_based
library_of_guidelines_path :sliders_taxonomy .json
extract_schema :
decompose_fields :false
dedupe_merged_rows :false
num_samples_per_chunk :1
is_relevant_chunk :true
extract_quotes :true
merge_tables :
merge_strategy :recon_v2
reconciliation :
debug_mode:false
primary_key_selection :
version:v1
max_candidates :10
max_inspections_per_candidate :10
max_retries:3
canonicalization :
mode:two_pass
max_cycles:20
max_retries_per_cycle :3
max_inspection_history :3
inspections_per_field :50
null_handling :
max_inspections :5
placeholder_text :UNKNOWN
verification :
enable:false
max_inspections :5
controller_executor_loop :
max_iterations :5
max_controller_inspections :5
max_executor_inspections :5
max_sql_attempts :3
verification :
enable:true
max_inspections :5
context_generator :
enable:true
model:gpt -4.1
max_rows:20
non_pk_canonicalization :
enable:true
inspections_per_field :5
column_selector :
max_inspections :5
excluded_columns :
-row_id
-page_number
-__reconciliation_context__
-number_instances
excluded_patterns :
-_quote
-_rationale
-_confidence
statistics:
enable:true
filename:reconciliation_stats .json
inspect_answer :
```

20 

```
enable_citation_generation :true
```

```
enable_reconciliation_stats_verbalization :true
check_if_merge_needed :false
force_sql:false
answer_strategy :sql_inspect
models:
answer:
model:gpt -4.1
max_tokens:8192
temperature:0.0
answer_no_table :
model:gpt -4.1 - mini
max_tokens:8192
temperature:0.0
answer_tool_output :
model:gpt -4.1
max_tokens:8192
temperature:0.0
extract_schema :
model:gpt -4.1 - mini
max_tokens:32000
temperature:0.0
generate_schema :
model:gpt -4.1
max_tokens:8192
temperature:0.0
merge_tables :
model:gpt -4.1
max_tokens:32000
temperature:0.0
task_guidelines :
model:gpt -4.1
max_tokens:8192
temperature:0.0
check_objective_necessity :
model:gpt -4.1
max_tokens:8192
temperature:0.0
rephrase_question :
model:gpt -4.1
max_tokens:8192
temperature:0.0
direct_answer :
model:gpt -4.1
max_tokens:8192
temperature:0.0
force_answer :
model:gpt -4.1
max_tokens:8192
temperature:0.0
is_relevant_chunk :
model:gpt -4.1 - mini
max_tokens:8192
temperature:0.0
check_if_merge_needed :
model:gpt -4.1
max_tokens:8192
temperature:0.0
select_guidelines_for_schema :
model:gpt -4.1 - mini
max_tokens:2000
temperature:0.0
select_primary_key :
model:gpt -4.1
max_tokens:8192
temperature:0.0
canonicalize_fields :
model:gpt -4.1
max_tokens:8192
temperature:0.0
column_selector :
model:gpt -4.1
max_tokens:8192
temperature:0.0
context_generator :
model:gpt -4.1
max_tokens:8192
temperature:0.0
handle_null_pks :
model:gpt -4.1
max_tokens:8192
temperature:0.0
experiment_config :
benchmark_path :/data1/ hypothesis_dataset /loong/ loong_processed .jsonl
gpt_results_path :None
files_dir:/data1/ hypothesis_dataset /loong/doc/
specific_ids_csv :sample_data /50 _sample_ids_finance_en_final .csv
soft_evaluator_model :gpt -4.1
hard_evaluator_model :gpt -4.1
num_questions :null
random_state :42
document_config :
```

21 

```
chunk_size:16000
overlap_size :0
filter_by_type :"financial"
output_file:loong_sliders_finance_en_26jan .json
```

## **A.6 Cost Analysis** 

Table 7 presents the average cost per question in US dollars for SLIDERS across different benchmarks. Costs vary based on document length, complexity, and the number of reconciliation operations required. 

Table 7: Average cost per question (in USD) across benchmarks 

|**Benchmark**|**Avg. Cost ($)**|
|---|---|
|Oolong|1.30|
|Loong Legal|0.65|
|Loong Finance (ZH)|0.79|
|Loong Finance (EN)|0.60|
|Loong Papers|0.37|
|FinanceBench|0.63|



OoLong has the highest per-question cost due to its longer input contexts, while Loong Papers has the lowest cost with shorter academic papers requiring fewer reconciliation steps. 

## **A.7 RLM with GPT-4.1** 

To justify the model choice for RLM (GPT 5 / GPT 5-mini), we also ran RLM with GPT 4.1 (main) and GPT 4.1-mini (sub-LLM), matching SLIDERS’ configuration. As shown in Table 8, RLM performs worse with GPT 4.1, confirming that our main comparison gives RLM an advantage. SLIDERS with GPT 4.1 still outperforms RLM with GPT 5 across all benchmarks. 

Table 8: RLM performance with different base models. 

|**System**|**Loong**|**FinanceBench**|
|---|---|---|
|RLM (GPT 5 & GPT 5-mini)|72.64|75.33|
|RLM (GPT 4.1 & GPT 4.1-mini)|60.13|72.00|
|**SLIDERS**(GPT 4.1 & GPT 4.1-mini)|**78.57**|**88.67**|



## **B Benchmark Creation** 

Question construction begins from a seed query: for WikiCeleb100, “Which artist made their debut at the youngest age across Film, Music, Content Creation, and Other industries?”; and for FinQ100, “Which company has the lowest long-term borrowing?” We derive follow-up questions by reusing and recombining the extracted information, including temporal cohorts (e.g., artists debuting in the 1950s or born in the 1970s) and aggregate financial properties (e.g., companies with no long-term borrowing or borrowing exceeding $80M). 

To annotate gold answers, we apply SLIDERS to extract structured information and manually verify the extracted values. We then manually reconcile the resulting tables to obtain a consolidated database representation of each document set, and manually author SQL queries for each question. For each benchmark, we additionally retain five questions that cannot be answered using SQL alone. 

## **B.1 Questions and Answers** 

We provide all questions from both benchmarks. 

22 

## **B.1.1 WikiCeleb100** 

1. _Which industry has the most artist debuts - Music, Film, or Content Creation?_ **Answer:** Music 

2. _What are the top 3 countries of birth for artists in the dataset?_ **Answer:** United States, India, England 

3. _Which artist has the earliest debut year in the dataset?_ **Answer:** Vera-Ellen 

4. _Are there more artists who debuted in Music or Film?_ **Answer:** Music 

5. _Which decade saw the most artist debuts?_ **Answer:** 2010.0 

6. _What is the distribution of artists across different industries?_ **Answer:** Music, Film, Other, Content Creation 

7. _Show the trend of artist debuts from earliest to latest decade - is it increasing, decreasing, or stable?_ 

   - **Answer:** increasing 

8. _Which industry tends to have artists debuting earlier on average - Music or Film?_ **Answer:** Film 

9. _What are the top 2 countries for artist births?_ **Answer:** United States, India 

10. _Among artists born outside the US, which countries contribute the most?_ **Answer:** India, England, South Korea, Malaysia, Sweden 

11. _Do more artists debut in the first half of a decade (years 0-4) or the second half (years 5-9)?_ **Answer:** First half 

12. _For each of the top 5 countries, which industry is most dominant among their artists?_ **Answer:** United States, India, England, British India, South Korea 

13. _Which artists have crossed over into multiple industries?_ 

   - **Answer:** Lesley Hornby, Olajide Olayinka Williams Olatunji, Wee Meng Chee, Zendaya Maree Stoermer Coleman, Aliaune Damala Bouga Time Puru Nacka Lu Lu Lu Badara Akon Thiam, ... 

14. _Is the Music industry growing or declining over time compared to Film? Compare debuts by decade._ 

   - **Answer:** declined marginally till 1980 and then grew steadily 

15. _Among artists who work in multiple industries, is it more common to start in Music then move to Film, or vice versa?_ 

   - **Answer:** mmore 

16. _What is the typical gap in years between an artist’s Music debut and their Film debut?_ **Answer:** 4.6 

17. _Which birth places have produced artists in the most diverse range of industries?_ **Answer:** Atlanta, Georgia, U.S., Muar, Johor, Malaysia, Oakland, California, Neasden, Middlesex, England, London, ... 

18. _What proportion of artists from India work in Film compared to artists from the United States?_ **Answer:** more 

19. _Who are the artists with the most recent debuts, and what industries are they in?_ **Answer:** Xavier James Trudeau, Ciara Mary-Alice Thompson, HorsegiirL, Darren Jason Watkins Jr., Victoria Beverley Walker 

20. _Is there a trend showing Content Creation becoming more common in recent decades compared to traditional industries?_ 

   - **Answer:** Relatively more common in recent decades but still not as common as music or film 

21. _Which artist debuted at the youngest age?_ 

**Answer:** Raven-Symon Christina Pearman-Maday 

22. _Which artist debuted at the youngest age in each industry?_ **Answer:** Film, Music, Content Creation, Other 

23 

## **B.1.2 FinQ100** 

1. _Which company has the highest long-term borrowings?_ **Answer:** CARMAX INC 

2. _What are the top 5 companies by long-term borrowings?_ 

   - **Answer:** CARMAX INC, MICRON TECHNOLOGY INC, AUTOZONE INC, Salesforce, Inc., COSTCO WHOLESALE CORP NEW 

3. _Are there more companies with zero long-term debt or with debt over 1 billion?_ **Answer:** debt_category: Zero debt, more_or_less: more 

4. _Which reporting period has the most company filings?_ **Answer:** 2025-10-31 

5. _What are the companies with zero long-term debt?_ 

   - **Answer:** Apex Treasury Corp, BRB Foods Inc., CALERES INC, Cantor Equity Partners V, Inc., Dynamix Corp III, ... 

6. _What is the distribution of companies across different debt ranges (0, 0-100M, 100M-1B, 1B+)?_ **Answer:** 1. Zero, 2. Under 100M, 3. 100M - 1B, 4. Over 1B 

7. _How does the median long-term borrowings compare to the mean? Is the distribution skewed?_ **Answer:** Mean is more than median. Right skewed. 

8. _What percentage of total long-term borrowings is held by the top 5 companies?_ **Answer:** 49.2 

9. _Which companies have reported borrowings for multiple periods?_ **Answer:** IIOT-OXYS, Inc., TJX COMPANIES INC /DE/, VAIL RESORTS INC 

10. _What is the range of long-term borrowings (min to max) and how spread out is the data?_ **Answer:** "min_borrowings": 0.0, "max_borrowings": 16586.77, "range_spread": 16586.77, "std_deviation": 2645.37 

11. _Are companies reporting in October vs November showing different average borrowing levels?_ **Answer:** Borrowing levels in November are lower 

12. _What is the ratio of companies with high debt (>1B) to companies with low/no debt (<100M)?_ **Answer:** 0.44 

13. _If we exclude the top 3 largest borrowers, how does the average borrowing change?_ **Answer:** Decreases by 395.4 

14. _Which companies have borrowings closest to the median value (within 25% of median)?_ **Answer:** KESTRA MEDICAL TECHNOLOGIES, LTD., Oil-Dri Corp of America, SPORTSMAN’S WAREHOUSE HOLDINGS, INC. 

15. _What is the cumulative percentage of total borrowings as we go from smallest to largest companies?_ **Answer:** CARMAX INC, MICRON TECHNOLOGY INC, AUTOZONE INC, Salesforce, Inc., COSTCO WHOLESALE CORP /NEW, ... 

16. _Among companies with CIK numbers, which sectors (based on CIK ranges) have higher average borrowings?_ 

**Answer:** Mid registrants 

17. _What is the total long-term borrowings across all companies in the dataset?_ **Answer:** 102590.76 

18. _For TJX Companies which has multiple periods, did their long-term borrowings increase or decrease over time?_ **Answer:** decreased 

19. _What percentage of companies have long-term borrowings under 50 million?_ **Answer:** 53.7 

20. _How concentrated is the debt? What percentage of companies hold 80% of the total borrowings?_ **Answer:** 15 

21. _Are companies in the ’Recent registrants’ CIK range (1M-1.5M) more likely to have zero debt compared to ’Mid registrants’ (500K-1M)?_ **Answer:** Recent 

24 

22. _Among companies with borrowings over 1 billion, which has the lowest borrowings?_ **Answer:** ROSS STORES, INC. 

23. _Do more companies have borrowings above or below 100 million?_ **Answer:** Below 100M 

24. _What are the bottom 5 companies by long-term borrowings (excluding zero debt companies)?_ **Answer:** LIBERTY STAR URANIUM & METALS CORP., OFF THE HOOK YS INC., USA OPPORTUNITY INCOME ONE, INC., Rise Gold Corp., IIOT-OXYS, Inc. 

25. _Which company has the lowest non-zero long-term borrowings?_ **Answer:** LIBERTY STAR URANIUM & METALS CORP. 

**Complexity of writing SQL programs.** The SQL programs written by the reconciliation Agent are non-trivial as shown in Listing 1. 

Listing 1: Computing depreciation-to-revenue margin for AMD 

```
SELECT
d.depreciation_amortization_expense,
t.total_revenue,
ROUND(
100.0*d.depreciation_amortization_exp/
t.total_revenue,
2
)ASda_margin_percent
FROMDepreciationAmortizationExpenseASd
JOINTotalRevenueASt
ONd.company_name=t.company_name
ANDd.fiscal_year=t.fiscal_year
WHEREd.company_name=’AMD’
ANDd.fiscal_year=2015
ANDd.depreciation_amortization_expISNOTNULL
ANDt.total_revenueISNOTNULL
LIMIT1;
```

Listing 2: An entity resolution SQL for inspection 

```
SELECT
product_or_service_name,
COUNT(*)AScnt
FROMCompanyProductOrService_2063d536_AMD_2022_10K
GROUPBY
product_or_service_name
ORDERBY
product_or_service_name,
cntDESC;
```

## **C Example of Issues Discovered by SLIDERS** 

During error analysis, we identified cases where the ground truth annotations in FinanceBench and Loong are incorrect. Because SLIDERS traces provenance, we can verify whether a predicted answer is correct or the ground truth itself is erroneous. We identified two types of errors: (1) answers that are spread across multiple pages, and (2) incorrect or incomplete annotations. For FinanceBench, which assumes answers are localized to a single page, we found more complete answers spanning multiple pages for question IDs `financebench_id_02416, financebench_id_00283` . For Loong in the financial domain, we found ground truth entries with missing units and incorrect values; for example, for a question the annotation says the accounts payable is 1,740 instead of the correct value of 1,663 thousand. This is the same example we provide in Figure 2. These findings demonstrate the auditability and interpretability of SLIDERS. 

## **D Prompts used by SLIDERS** 

## **D.1 System Prompts** 

We provide all prompts used in the SLIDERS pipeline, ordered by their execution sequence. Prompts are displayed in a monospace font with syntax highlighting disabled for readability. 

25 

## **D.1.1 Schema Generation** 

**Schema Induction** Generates a structured schema (tables and fields) based on the question and document descriptions. 

**==> picture [380 x 501] intentionally omitted <==**

**----- Start of picture text -----**<br>
# instruction<br>Given description of documents, sample document content, and a question, define the tables and fields (including<br>intermediate/supporting fields) that, once extracted at the page level and merged later, will let us answer the<br>question.<br>Table Modeling rules:<br>- One table = one relationship. If the relationship logically decomposes (e.g., case summary vs per-counsel arguments),<br>define multiple tables.<br>- No nested types. When you would need a list/array, model it as multiple rows in the appropriate table.<br>- Include supporting fields. Add any inputs required for downstream computation/validation (e.g., numerators/<br>denominators, period markers, currency).<br>- Normalize by design. Each numeric field must declare a single unit and scale; extractors will normalize values into<br>that target.<br>Field Specs (required keys per field):<br>- name: string (concise, machine-friendly).<br>- data_type: one of str | int | float | decimal | bool | date | enum.<br>- Prefer decimal for money/precise quantities; date for dates.<br>- enum_values: list of strings (required iff data_type == "enum").<br>- unit: string or None (e.g., USD, %, shares; use None if not applicable).<br>- scale: one of none | thousands | millions | billions | trillions | basis_points | log10 | ln.<br>- Values will be stored in this scale.<br>- Example: $1.23B with scale="millions" $\rightarrow$ store 1230, unit="USD".<br>- description: self-contained extraction instruction (what the value represents; common surface forms/synonyms).<br>{% if enable_extraction_guidelines %}- extraction_guideline: string or None (optional but recommended). A concrete<br>instruction for the extractor on HOW to extract this field.<br>- Use this to clarify extraction logic that isn’t obvious from the description alone.<br>- Specify the SOURCE of the value (e.g., which document/entity the field refers to).<br>- Clarify SCOPE constraints (e.g., what to include/exclude, how to handle ambiguity).<br>- Provide FORMAT guidance (e.g., normalization rules, handling incomplete data).<br>- Keep it concise (1-2 sentences max), specific to this field, and actionable.<br>{% endif %}- normalization (object; optional keys as needed):<br>- currency: target code (e.g., "USD") or "as_is".<br>- percent: "0_100" or "0_1".<br>- date_format: e.g., "YYYY-MM-DD".<br>- required: boolean (expected presence; still None if missing on a page).<br>General Rules:<br>- Each table represents a relationship. If modeling the relationship requires multiple relations, define multiple<br>tables (e.g., a "CaseSummary" table for case-level facts and a "CounselArguments" table for per-counsel arguments<br>).<br>- No nested types. If the extraction needs multiple items (e.g., many arguments), the extractor will create multiple<br>rows in the appropriate relationship table.<br>- Include any linking fields needed to relate tables later (e.g., case_title, docket_number, entity_name, period_end).<br>- Normalize all values as specified below so independently run extractors produce standardized outputs.<br>- If a value is missing/unknown, set the value to None (not 0, empty string, or placeholder text).<br>- Page/section metadata is precomputed; do not include it in fields.<br>- The schema must describe only document-grounded relationships and fields, not the question itself. Do not add fields<br>that simply restate the question.<br>Ouptut:<br>- Emit JSON.<br>- For each table: include name, description, and a fields array.<br>- Do not include page metadata (it’s precomputed).<br># input<br># Question<br>{{ question }}<br># Documents description<br>{{ document_description }}<br>{% if document_text %}<br># Sample Document Content<br>{{ document_text }}<br>{% endif %}<br>{% if guidelines %}<br>{{guidelines}}<br>{% endif %}<br>These are reserved keywords for fields. DO NOT USE THEM: ‘row_id‘, ‘document_name‘, ‘metadata‘.<br>**----- End of picture text -----**<br>


## **D.1.2 Information Extraction** 

**Chunk Relevance Filtering** Determines whether a document chunk contains information relevant to answering the question. 

```
#instruction
```

```
Youwillbegivenasinglepagefromalargerdocumentalongwithaquestion.Theextractionwillbedoneonthispage.
```

26 

```
YourtaskistodeterminewhetherthispagecontainsANYinformationthatcouldbeusefulforansweringthequestion
oncethispageiscombinedwithotherpageslater.
TreatthepageasRELEVANTifitsatisfiesANYofthefollowing:
-Itcontainsadirectorpartialanswertothequestion.
```

- `It contains definitions, conditions, steps, list items, examples, or descriptions that relate to the question.` 

- `- It narrows down possibilities or provides context that a reconciliation agent could use (e.g., "the first principle is...", "see Section 3 for...", "Step 2: ...").` 

- `Treat the page as IRRELEVANT only if:` 

- `None of the content on the page has any meaningful connection to the question, and` 

- `It only contains metadata values, and doesn’t explictly mention useful fields. For example, if the page only contains the title, company name, then ignore this page.` 

- `- There is no text, that could help answer the question in any way during later aggregation. First, briefly explain your reasoning in natural language. Then say if its relevant or not. # input # Question {{question}} # Document page {{document}}` 

**Table Extraction** Extracts structured data from document chunks according to the generated schema. 

- `# instruction Given a schema and a single page, emit rows per table using the schema’s field names/types---and per-field evidence: - value (normalized to the field’s unit/scale). If scale is None and the page content says 1.3 billion, then write the value as 1300000000.` 

- `{% if extract_quotes %} - quote (largest contiguous exact text; list of strings or None) {% else %} - quote (always null; do not extract snippets when quotes are disabled) {% endif %} - rationale (why this value; include table headers/context if from a table) - is_explicit (boolean) - confidence (string; one of "Very High", "High", "Medium", "Low", "Very Low") Confidence Rubric: - Very High - The value is explicitly stated verbatim or via a trivial transformation (e.g., unit normalization) in the current text span, unambiguously refers to the target entity, and fully satisfies the schema field definition.` 

- `- Surrounding context confirms the interpretation without inference.` 

- `- High - The value is explicitly stated, but either (a) requires light interpretation to map to the schema (e.g., paraphrase, implicit unit), or (b) could plausibly appear elsewhere in the document, even though the current span provides sufficient evidence.` 

- `- Entity reference and schema alignment are still clear.` 

- `- Medium - The value is not explicitly stated but can be reasonably inferred from the text through interpretation, aggregation, or implicit assumptions.` 

- `- The match to the schema is partial or approximate, and alternative interpretations are plausible.` 

- `- Low - The value is a speculative guess derived from weak textual cues, heuristics, or world knowledge. - Evidence is indirect, incomplete, or ambiguous, and the extractor believes the value is likely incorrect. - For entity relationships: Inferred from abbreviated references (acronyms, short names, citation numbers) without the full entity name appearing on the page.` 

- `- Very Low - No supporting evidence exists in the text. - The value is a placeholder, default, or fabrication inserted to satisfy schema completeness and should not be trusted or used downstream.` 

- `{% if enable_verbalization_instructions %} ## Critical: Exhaustive List Processing **APPLIES TO:** Any section containing a numbered or bulleted list of related entities, including but not limited to: - Sections titled "References", "Bibliography", "Citations", "Works Cited", "Related Work", "Dependencies", "Components ", "Exhibits", "Appendices"` 

- `- Dependency lists, package catalogs, component inventories, related entity lists - Any enumerated list where entities might be mentioned **REQUIRED PROCESS:** 1. **In ‘verbalization‘ field**: List ALL target entities from the question you’re looking for - **CRITICAL**: Include EVERY entity mentioned ANYWHERE in the question---this includes the primary entity being analyzed AND all comparison/related entities` 

- `- Example: If the question asks about relationships involving entities X, Y, and Z, then your target entities list should be [X, Y, Z]---NOT just a subset` 

- `- When processing ANY document, check for ALL entities from the question, regardless of which document you’re currently analyzing` 

- `2. **Identify the list boundaries** - Find where the list starts and ends 3. **Process EVERY entry sequentially** - Go through each numbered/bulleted item one by one (e.g., [1], [2], [3]... to the last entry)` 

- `4. **In ‘verbalization‘ field**: For EACH AND EVERY entry, you MUST document: - **EXPLICITLY LIST EACH ENTRY** - Write: "Entry [N] ’identifier/name found here’ - MATCH to <target entity name>" OR "Entry [N] ’identifier/name found here’ - no match"` 

- `- **DO NOT SUMMARIZE** - List each entry individually even if there are 100+ entries` 

27 

- `**CRITICAL**: Even if you don’t see matches initially, you MUST list every single entry. Saying "Checked entries [1]-[92], none match" or "The section lists many entries but none match" is WRONG - you must list each entry explicitly` 

- `5. **In ‘tables‘ field**: Extract a row immediately when you find a match (High/Very High confidence, with concise rationale)` 

- `- **CRITICAL**: For EVERY entry you marked as "MATCH" in your verbalization, you MUST create a corresponding extraction row. The verbalization is your working notes; the tables section is where you record the actual data. DO NOT identify matches in verbalization and then fail to extract them.` 

- `6. **Continue to the end** - Do not stop after finding matches; process every remaining entry 7. **In ‘verbalization‘ field**: Confirm you reached and checked the final entry (e.g., "Completed all 92 entries") 8. **CRITICAL - Understanding Entity Relationships in Lists**: When processing a source entity’s list of related entities:` 

- `- Finding target entity Y in source entity X’s list means "X relates to Y" - EXTRACT THIS - Whether Y appears in the question is IRRELEVANT to whether this is a valid relationship - ONLY skip extraction when source equals target (i.e., X’s list contains X itself - a self-loop, likely a data error)` 

- `- [OK] Correct: Processing "Entity Alpha"’s related items list, found "Entity Beta" (where Beta is mentioned in the question) $\rightarrow$ Extract "Alpha relates to Beta"` 

- `- [OK] Correct: Processing "Component A"’s dependency list, found "Component B" (from question) $\rightarrow$ Extract the dependency relationship` 

- `- [OK] Correct: Processing "Site 001"’s supplier catalog, found "Supplier Corp" (from question) $\rightarrow$ Extract the supplier relationship` 

- `- [X] Incorrect: Processing "Item X"’s related items list, found "Item X" itself $\rightarrow$ Skip (self-loop) - **WRONG reasoning**: "The found entity is mentioned in my question, therefore this is self-referential" - NO. Selfreference means the source entity points to itself, not that the target appears in the question.` 

- `Examples: - If you see a section with 50 entries (e.g., a list with [1] through [50]), and the question asks about 5 specific entities:` 

- `- [OK] Correct in ‘verbalization‘: "Target entities: [Entity A, Entity B, Entity C, Entity D, Entity E]. Entry [1] ’ Author et al., 2020. Some description about topic X’ - no match. Entry [2] ’Source, 2021. Entity A: A comprehensive analysis’ - MATCH to Entity A. Entry [3] ’Creator, 2019. Another description’ - no match. ... Entry [50] ’Final Author, 2023. Final description’ - no match. Completed all 50 entries."` 

- `- [OK] Correct in ‘tables‘: Extract rows for all matches found` 

- `- If you see an exhibits list with 15 items and question asks about 3 companies: - [OK] Correct in ‘verbalization‘: "Target entities: [Acme Corp, Widget Inc, Gadget LLC]. Entry [1] ’Exhibit 1.1: Lease Agreement with Property Owner’ - no match. Entry [2] ’Exhibit 2.1: Supply Contract with Acme Corp’ - MATCH to Acme Corp. ... Entry [15] ’Exhibit 5.4: Insurance Policy’ - no match. Completed all 15 entries."` 

- `- [X] Wrong: "The section lists many entries, but none of the target entities appear explicitly" (This is a SUMMARY without listing entries)` 

- `- [X] Wrong: "Checked entries [1]-[50]. Found matches for Entity A and Entity B." (This is a SUMMARY, not entry-byentry verbalization)` 

- `- [X] Wrong: Stop after finding 2 matches in first 10 entries - [X] Wrong: "List entries [1]-[92]. None match." (Must list each entry explicitly: "[1] ... - no match. [2] ... - no match.")` 

- `{% endif %}` 

- `Output guidelines: - Produce one row per real item visible on the page (e.g., per counsel-argument, per entity-period-metric). - Strictly follow the schema, do not add any new field/key.` 

- `If the page contains multiple distinct items, output multiple rows.` 

- `{% if extract_quotes %} - For any field, if the given page does not have any relevant information, set the value and quote to null. {% else %} - For any field with no relevant information, set value to null. Quote extraction is disabled, so leave ‘"quote": null‘ for every field.` 

- `{% endif %} - Consider the summary if provided to contextualize the extraction. - IMPORTANT: If the question below contains a "FOCUS:" directive specifying a particular row or item to extract, you MUST extract ONLY that specific row/item. Ignore all other rows/items in the page. This focused extraction is used when processing information-dense pages to ensure accuracy and prevent confusion between similar items.` 

- `Extraction and Normalization Rules:` 

- `Page-local only. Use only the current page’s content.` 

- `- Rationale.` 

   - `Start with rationale, and reason about which parts of the text correspond to the field` 

- `1--3 sentences: how you located/parsed it; if from a table, include headers and relevant table context.` 

- `- Note any normalization you performed (e.g., "converted $1.23B to 1230 millions USD").` 

- `- Normalization is mandatory. - Convert magnitudes into the field’s target scale (e.g., target millions: 1,200,000$\rightarrow$1.2; 1.23B$\ rightarrow$1230).` 

- `- Convert unit (e.g., currency) to the target if specified; else keep as_is. - Normalize percent per the field’s rule (0_100 vs 0_1) and date per date_format.` 

- `- Type safety. Coerce to the declared data_type; on failure, set value = null. {% if extract_quotes %}` 

- `Quote (evidence). - Provide a list with the largest contiguous exact substring that supports the value (often a single element). - Preserve punctuation/case; no ellipses or paraphrase. - If truly not present on the page $\rightarrow$ quote = null.` 

   - `Remember if quote is not present, then the value should also be None.` 

- `{% else %} - Quote (evidence) is disabled. Always emit ‘"quote": null‘ for every field; do not copy supporting text into the output.` 

- `{% endif %}` 

- `is_explicit. - True if the underlying value appears explicitly on the page (including trivial deterministic normalization like unit/scale conversion or parentheses-to-negative).` 

- `- False if it requires arithmetic beyond normalization (e.g., sums, differences, ratios) or approximations/ inference.` 

- `- Duplicates vs distinct items. - Multiple mentions of the same item: choose the most precise/authoritative instance on that page (e.g., structured table over prose).` 

28 

```
-Distinctitems:separaterows.
-Relationshipschemaswithmainentity:
-Iftheschemahasaprimaryentityfield(e.g.,"entity_name","source_identifier"),populateitconsistentlyin
everyrow.
-Forrelationshipfields:neverleavetheprimary/sourceentityfieldnull.Onlysetsecondary/targetfieldsto
nullwhennosuchrelationshipispresent.
{%ifenable_verbalization_instructions%}
-**ACTIONREQUIREDforstructuredlistsections**:Whenyouencounterenumeratedlistscontainingentity
relationships(suchasdependencycatalogs,relatedentitylists,componentinventories,referencelists,or
bibliographicentries),followthe"Critical:ExhaustiveListProcessing"instructionsabove.Thisincludes
listingalltargetentities,processingeveryentryindividuallywithoutsummarization,andunderstanding
thatfindingaquestionentityinanotherdocument’slistisavalidrelationshiptoextract.{%endif%}
#Example
Supposeschemahasmetric_value(decimal,unit=USD,scale=None)andthepagesays:"Revenuewas$1.23billionin2023."
‘‘‘JSON
{
"verbalization":"Notareferencessection.Extractingdirectfieldvaluesfrompagecontent.",
"tables":[
{
"name":"Financials",
"rows":[
{
"fields":{
"entity_name":{
"rationale":"ThepageheadermentionstheinformationhenceIcanuseittoextracttheentityname.
Takenfrompageheader.",
"value":"AcmeInc.",
"quote":["ACMEINC."],
"is_explicit":true,
"confidence":"VeryHigh"
},
"period_end":{
"rationale":"Theperiodendismentionedbutitisnotintherequiredformat.Ineedtonormalizedto
YYYY-MM-DD.Theperiodwasmentionedinthepagetitle",
"value":"2023-12-31",
"quote":["FortheyearendedDecember31,2023"],
"is_explicit":true,
"confidence":"VeryHigh"
},
"metric_name":{
"rationale":"Table2mentionsthemetrics.TheMetriclabelinthecaptionofthetabletalkingabout
balancesheet.",
"value":"revenue",
"quote":["Revenue"],
"is_explicit":true,
"confidence":"VeryHigh"
},
"metric_forecast":{
rationale:"Thetextmentionsthecurrentandpastrevenue.However,thereisnomentionoffuturerevenue.
HenceIshouldsetthevalueandquotetoNone.,
"value":null,
"quote":null,
"is_explicit":false,
"confidence":"VeryLow"
},
"metric_value":{
"rationale":"TherevenueismentionedininsectionCommentry.However,itsays$1.23billion.Sincethe
scaleisNone,Ishouldmakeit1230000000.Converted$1.23Bto1230000000withscale=’None’,unit=’
USD’.",
"value":1230,
"quote":["Revenuewas$1.23billionin2023"],
"is_explicit":true,
"confidence":"VeryHigh"
}
}
}
]
}
]
}
‘‘‘
#input
Thefollowingquestionhasbeenaskedbytheuseroverthewholedocumentset:
<QuestionStart>
{{question}}
<QuestionEnd>
TheRelationshipSchemaforthisquestionhasalreadybeendefined.
Yourtaskistopopulatethisschemausingonlytheinformationavailableinthedocument.Theextracteddatawill
laterbeaggregatedwithinformationfromotherpagestogeneratethefinalanswer.
{%ifprevious_chunk_summary%}
#Summaryofthepagestillnow
{{previous_chunk_summary}}
{%endif%}
<DocumentStart>
#ThenameoftheCURRENTDOCUMENTyouareextractingfromis:
{{document_name}}
```

29 

- `# Document Descriptions {{document_description}} # Page Content {{document}} <Document End> # Relationship Schema {{schema}} # Field Extraction Guidelines {% if extraction_guidelines %} The schema includes specific extraction guidelines for certain fields. Follow these instructions when extracting: {{extraction_guidelines}} {% endif %} # Output Format Your output should be a JSON object with these fields: {% if enable_verbalization_instructions %} **IMPORTANT**: Verbalization is your working notes to show your reasoning. The ‘tables‘ array is where you record your actual extractions. Any match you identify in verbalization MUST have a corresponding row in tables.` 

- `1. **verbalization** (optional string): For sections with enumerated lists (such as dependency lists, related entity lists, "References", "Bibliography", "Citations", component inventories, or any numbered/bulleted entity listings ), describe your systematic checking process:` 

- `- List ALL entities mentioned ANYWHERE in the question---include the primary entity AND all comparison/related entities (e.g., if the question asks about relationships involving entities X, Y, and Z, list [X, Y, Z])` 

- `- For each entry in the list: entry number/identifier, name/title found, match determination (which target entity it matches, or "no match")` 

- `- **DO NOT SUMMARIZE**: You must list each entry individually, even if none match. Saying "checked all entries, none match" is insufficient.` 

- `- Confirm you reached the final entry **IMPORTANT for inline references (non-list sections):** - When processing abbreviated inline identifiers (e.g., short codes, numeric references), DO NOT verbalize speculative mappings to full entity names.` 

- `- Only report matches when the entity name itself appears on the page. - [X] WRONG: "The document mentions identifier X in context Y, which corresponds to ’Full Entity Name Z’" - [OK] CORRECT: "The document mentions identifier X but does not include the full entity name - no match" Examples: - "Target entities: [’Entity A Title’, ’Entity B Title’, ’Entity C Title’]. Reference list section entries [1]-[25]. Entry [1] ’ID-001. Item X description’ - no match. Entry [2] ’ID-002. Entity A Title: A comprehensive` 

- `analysis’ - MATCH to ’Entity A Title’. Entry [3] ’ID-003. Another item description’ - no match. ... Entry [25] ’ID-025. Final item’ - no match. Completed all 25 entries."` 

- `- "Target entities: [’Supplier A’, ’Supplier B’, ’Supplier C’]. Dependencies section entries [1]-[18]. Entry [1] ’ Component X from Vendor Alpha’ - no match. Entry [2] ’Raw materials from Supplier A’ - MATCH to ’Supplier A’. Entry [3] ’Packaging from Provider Beta’ - no match. ... Entry [18] ’Logistics from Carrier Gamma’ - no match. Completed all 18 entries."` 

- `2. **tables** (required array): The extracted data following the schema above, with concise rationales per field. **CRITICAL**: If you identified ANY matches (marked as "MATCH") in your verbalization, you MUST create a corresponding extraction row in the tables array for EACH match. The verbalization is your working notes---the tables section is where you record the actual extractions. Do NOT skip creating rows for matches you` 

- `identified during verbalization.` 

- `{% else %} 1. **tables** (required array): The extracted data following the schema above, with concise rationales per field. {% endif %} Fill out the schema above based on the Document content.` 

## **D.1.3 Data Reconciliation** 

**Primary Key Selection** Selects primary key fields which are used to group data rows for further reconciliation. 

```
#instruction
Youareselecting**primarykeys**fordatabasetablesthathavebeenautomaticallyextractedfromdocuments.
```

```
##Purpose
```

```
Yourselectedprimarykeywillbeusedfor**downstreamdataconsolidation**:
```

`1. **Deduplication**: Rows with the same primary key will be identified as duplicates` 

`2. **Conflict Resolution**: When duplicates exist, we’ll merge them by selecting the best value for each field 3. **Table Consolidation**: The final cleaned table will have one row per unique primary key` 

```
##Context
```

- `The data was automatically extracted from unstructured documents (PDFs, text, etc.)` 

- `Extraction errors create duplicate rows for the same real-world observation` 

- `Multiple extractions of the same data point may have conflicting values` 

- `Your primary key defines which rows represent the "same thing" and should be merged` 

```
##YourTask
```

```
Selectthe**conceptualprimarykey**thatidentifiesuniqueobservationsintheidealcleanedtable.Whentworows
havethesameprimarykey,theyshouldrepresentthesamereal-worldobservationandbeconsolidatedintoonerow
.
```

30 

```
##KeyPrinciples
###TheKeyDefinesWhatGetsConsolidated
Yourprimarykeydetermineswhichrowswillbemergedtogether:
-Rowswith**identicalprimarykeyvalues**$\rightarrow$consolidatedintoONErow
-Rowswith**differentprimarykeyvalues**$\rightarrow$keptasseparaterows
-Duringconsolidation,wemergeconflictingfieldvaluesbyselectingthemostreliableextraction
###SeparateIdentifiersfromValues
**Identifiers**defineWHATisbeingobserved-theseformtheprimarykey:
-Entityidentifiers:company,user,product,document
-Temporalidentifiers:date,period,timestamp
-Categoricalidentifiers:metric_type,event_type,category
-Relationshipidentifiers:source_entity,target_entity
**Values**aretheobserveddata-thesegetconsolidatedduringconflictresolution:
-Measurements:amounts,counts,percentages,scores
-Content:text,descriptions,labels,classifications
-Attributes:status,flags,properties
**Rule**:PrimarykeysshouldcontainONLYidentifiers,nevermeasuredvalues.
Examples:
-Metricsovertime:Key=entity+time_period(metricvaluesgetconsolidated)
-Eventswithcontent:Key=entity+timestamp(contentfieldsgetconsolidated)
-Entityrelationships:Key=source+target(therelationshipitselfIStheidentifier)
###FieldPreferences
**STRONGLYPREFERbusiness/domainidentifierfields**asprimarykeys.
**Use‘row_id‘or‘document_name‘astheprimarykeyinthescenarioswhereotherfieldsareallowedtohaveduplicates
asintextthatrepresentsdataset.**
##AnalysisGuidelines
###1.UnderstandtheDataDomain
Askyourself:Whatisthistablemeasuringorrecording?
-Financialdata:Observationsaboutcompaniesatpointsintime
-Eventlogs:Actionsbyusersattimestamps
-Relationships:Connectionsbetweenentities
-Classifications:Labelsappliedtoitems
###2.IdentifytheIdentifierFields
WhichfieldsdefineWHATisbeingmeasured(notthemeasurementitself)?
-Entityidentifiers:company_name,user_id,paper_title,product_id
-Temporalidentifiers:date,period_end,timestamp,fiscal_year
-Categoricalidentifiers:metric_name,event_type,category
-Relationshipidentifiers:source_entity,target_entity
Excludevaluefieldsfromthekey:amounts,counts,textcontent,computedmetrics,labels
###3.BuildCompositeKeys
Mosttablesneedmultipledimensionstoidentifyobservations:
-Financial:entity+time(+metric_nameifmultiplemetricsperrow)
-Events:entity+time(+event_typeifmultipleeventsperentity-time)
-Relationships:source+target
-Timeseries:entity+time+measure_type
###4.ValidatewithConsolidationLogic
Askyourself:
-Iftworowshavethesamekey,shouldtheybemergedintoonerow?Ifyes,goodkey.
-Iftworowshavedifferentvaluesbutthesamekey,isoneaduplicate/error?Ifyes,goodkey.
-Areyouincludingmeasuredvaluesinthekey?Ifyes,reconsider-thoseshouldbeconsolidated,notpartofthekey.
-Doesthekeydefinethe"grain"oftheconsolidatedtable?Ifyes,goodkey.
###5.UseStatisticsasSupportingEvidence
Thetablestatistics(nullcounts,distinctvalues)canhelpvalidateyourchoice:
-Keyfieldsshouldhavelownullrates
-Keycombinationsshouldhavereasonablecardinality
-Butdon’tletstatisticsoverridesemanticmeaning
###6.ExpectDuplicates-That’sWhyWeNeedtheKey
Thecurrentextracteddatawillhaveduplicateswiththecorrectkey-thisisbydesign:
-Multipleextractionsofthesameobservationcreateduplicaterows
-Theseduplicateswillbeconsolidatedusingyourprimarykey
-Textvariations(spacing,punctuation)willexistinbothkeysandvalues
-Yourjob:identifytheconceptualkeythatgroupsrowsforconsolidation,notensureperfectuniquenessnow
##SQLQueryCapabilities
YouhaveaccesstoSQLqueriestoinspectthedata.Usequeriesto:
-Checkuniqueness:‘SELECTfield,COUNT(*)ascntFROMtableGROUPBYfieldHAVINGcnt>1‘
-Checknullrates:‘SELECTCOUNT(*)-COUNT(field)asnull_countFROMtable‘
-Checkcardinality:‘SELECTCOUNT(DISTINCTfield)FROMtable‘
-Checkcompositeuniqueness:‘SELECTfield1,field2,COUNT(*)FROMtableGROUPBYfield1,field2HAVINGCOUNT(*)>1‘
-Samplevalues:‘SELECTfieldFROMtableLIMIT10‘
Focusqueriesonbusinessfieldsfirst.Onlyquerysystemfieldsifyou’reconsideringthemasafallback.
##QueryBudget
Youhaveabudgetof{{max_queries}}SQLqueries.Usethemstrategically:
-Firstquery:Getoverviewstatisticsforallfields
```

31 

```
-Subsequentqueries:Drilldownonpromisingprimarykeycandidates
-Finalqueries:Validateyourproposedprimarykeychoice
##Input
**Question:**{{question}}
**TableName:**{{table_name}}
**Schema:**
{{schema}}
**TableStatistics:**
{{table_stats}}
{%ifquery_history%}
**PreviousQueryResults:**
{{query_history}}
{%else%}
**PreviousQueryResults:**
Noqueriesexecutedyet
{%endif%}
{%ifsql_error_feedback%}
**SQLErrorFeedback:**
{{sql_error_feedback}}
{%endif%}
##OutputFormat
Basedonyouranalysis,provideyourdecision:
**Action**:Chooseeither"query"(runanotherSQLquerytogathermoreinformation)or"finalize"(youhaveenough
informationtorecommendaprimarykey)
**Reasoning**:Explainyourthinking(2-3sentences)
-Ifactionis"query":Whydoyouneedthisinformation?Whatwillittellyouabouttheconsolidationkey?
-Ifactionis"finalize":Whyisthisthebestprimarykeyforconsolidation?Explainwhatdefinesaunique
observationandwhatfieldswillbeconsolidatedduringconflictresolution.
**SQL**(onlyifactionis"query"):TheSQLquerytorun
**PrimaryKey**(onlyifactionis"finalize"):Listoffieldnamesthatformtheprimarykey,e.g.,["company_name","
period_end_date"]or["citing_paper_title","cited_paper_title"].Usenullifnosemanticprimarykeyexists.
##Examples
###Example1:SingleMetricPerRow
‘‘‘
Action:finalize
Reasoning:Thistablerecordsasinglemetricforentitiesovertime.Theconsolidationkeyisentity+time_period.
Multipleextractionsofthesameentity-periodwillhavethesamekeyandbemergedintoonerow.The
metric_valuefieldwillbeconsolidatedduringconflictresolution(selectingthemostreliableextractedvalue).
Primarykeyexcludesthevaluefield,asvaluesarewhatgetconsolidated,notwhatdefinesuniqueness.
PrimaryKey:["entity_name","time_period"]
‘‘‘
###Example2:Evidence/DatasetTable
‘‘‘
Action:finalize
Reasoning:Thistablerepresentsadatasetofextractedmentions,notcanonicalfacts.Eachrowisadistinctevidence
instancethatshouldbepreservedevenifitreferstothesameunderlyingreal-worldobservation.Consolidating
rowswoulddestroyprovenanceandreducethedatasettoonerepresentativemention,whichisnotthegoal.
PrimaryKey:["row_id"]
‘‘‘
###Example3:MultipleMetricsPerRow
‘‘‘
Action:finalize
Reasoning:Thistablehasmultiplemetrictypesperentity-period.Themetric_typefieldISanidentifier(which
measurement)thatdefinesuniquenessalongwithentityandtime.Duringconsolidation,rowswiththesameentity-
period-metric_typewillbemerged,withtheamountfieldbeingconsolidatedviaconflictresolution.Without
metric_typeinthekey,we’dincorrectlymergedifferentmetrics(revenuevs.expenses)intoonerow.
PrimaryKey:["entity_name","time_period","metric_type"]
‘‘‘
###Example4:Relationship/AssociationData
‘‘‘
Action:finalize
Reasoning:Thistablerecordsrelationshipsbetweenentities.Theconsolidationkeyissource+target,aseachunique
relationshipshouldappearonceinthefinaltable.Multipleextractionsofthesamerelationshipwillbemerged.
Anyadditionalfields(relationship_strength,relationship_date,etc.)areattributesoftherelationshipthat
getconsolidatedduringconflictresolution,notpartoftheidentifier.
PrimaryKey:["source_entity","target_entity"]
‘‘‘
###Example5:Event/TransactionData
‘‘‘
Action:finalize
Reasoning:Eventsareidentifiedbyentity+timestamp.Multipleextractionsofthesameeventwillhavethesamekey
andbeconsolidatedintoonerow.Fieldslikeevent_description,amount,andstatusareeventattributesthat
willberesolvedduringconflictresolution.Iftrulymultipledistincteventscanoccurattheexactsame
```

32 

**==> picture [392 x 647] intentionally omitted <==**

**----- Start of picture text -----**<br>
timestamp for the same entity, and there’s no event_id or event_type to distinguish them, then entity + timestamp<br>is still the semantic consolidation key.<br>Primary Key: ["entity_id", "timestamp"]<br>‘‘‘<br>Now, provide your decision.<br>Document-Level Canonicalization Canonicalizes fields within each document.<br># instruction<br>You are canonicalizing the field **{{field_name}}** in rows from a single document.<br>## Context<br>- **Document**: {{document_name}}<br>- **All rows are from the SAME document** - variations are likely extraction artifacts, not real distinctions<br>- Use OTHER COLUMNS to determine if different values refer to the same entity<br>## Key Principle: Check All Columns for Entity Relationships<br>**Before deciding if two values are different entities, inspect ALL columns.** Other columns may contain information<br>that reveals two seemingly different values actually refer to the same entity. Look for any column where:<br>- One value appears as an attribute of another<br>- The source text (quotes) mentions both forms together<br>- Context suggests equivalence<br>**Always start by examining the full row data:**<br>‘‘‘sql<br>SELECT * FROM {{table_name}} ORDER BY {{field_name}};<br>‘‘‘<br>If you see that value A appears in another column for rows with value B (or vice versa), they likely refer to the same<br>entity and should be canonicalized.<br>## Workflow<br>1. **First inspection**: Check ALL columns to find entity relationships<br>2. **Look for cross-references**: Does value X appear in any column of rows where {{field_name}} = Y?<br>3. **If values appear related**, canonicalize them to one form<br>4. **Apply normalization** (case, punctuation, whitespace) for remaining variations<br>5. **Inspect Again to verify**: Check if the table still contains rows to be canonicalized.<br>5. **Stop** when no more relationships or variations exist<br>## Skip If<br>- Free-text / natural language content<br>- High cardinality with unique values<br>- Long verbose strings (>50 chars average)<br>## Canonicalization SQL Format (CRITICAL)<br>**You must REPLACE the column, not add a new one.** Do NOT use ‘SELECT *‘ for canonicalization.<br>**CORRECT** - explicitly list columns and keep the same column name:<br>‘‘‘sql<br>SELECT<br>row_id,<br>CASE WHEN {{field_name}} = ’ValueA’ THEN ’ValueB’ ELSE {{field_name}} END AS {{field_name}},<br>other_col1,<br>other_col2,<br>...<br>FROM {{table_name}}<br>ORDER BY row_id;<br>‘‘‘<br>**WRONG** - DO NOT do this:<br>‘‘‘sql<br>SELECT *, CASE ... END AS {{field_name}}_canonicalized -- WRONG: adds new column<br>SELECT *, CASE ... END AS {{field_name}} -- WRONG: duplicates column<br>‘‘‘<br>Requirements:<br>- List ALL columns explicitly (no ‘SELECT *‘)<br>- The CASE expression must be named exactly ‘{{field_name}}‘ (same as original)<br>- End with ‘ORDER BY row_id‘<br>- DuckDB syntax (no ‘|>‘ operators)<br>- Escape quotes: ‘’’‘ not ‘\’‘<br>## Input<br>**Table**: {{table_name}} | **Field**: {{field_name}} | **Document**: {{document_name}}<br>**Schema:**<br>{{schema}}<br>**Table Statistics:**<br>{{table_stats}}<br>{% if inspection_history %}<br>**Inspection History:**<br>{{inspection_history}}<br>{% endif %}<br>**----- End of picture text -----**<br>


33 

```
**CanonicalizationHistory:**{{canonicalization_history}}
{%ifcanonicalization_error_feedback%}
**Error:**{{canonicalization_error_feedback}}
{%endif%}
**Inspectionsremaining:**{{inspections_remaining}}
##Actions
**inspect**:RunSQLtocheckcolumnrelationships
‘‘‘
action:"inspect"
sql:"<SQL>"
reasoning:"<whatyou’rechecking>"
‘‘‘
**canonicalize**:Mergerelatedvalues(MUSTreplacecolumn,notaddnewone)
‘‘‘
action:"canonicalize"
sql:"<SELECTwithexplicitcolumns,CASEfor{{field_name}},ORDERBYrow_id>"
reasoning:"<whichvaluesarethesameentity>"
‘‘‘
**stop**:Aftercanonicalizatoinhasbeenperformedandyouhavedoneinspectionagain:Nomorevariationsfound
‘‘‘
action:"stop"
reasoning:"<confirmallvaluesarenowconsistent>"
‘‘‘
**skip**:Fieldisnotcategorical
‘‘‘
action:"skip"
reasoning:"<why>"
‘‘‘
```

**Error Detection Agent** Agent that inspects data and decides which reconciliation operation to perform (deduplicate, aggregate, resolve conflicts, canonicalize). 

**==> picture [379 x 334] intentionally omitted <==**

**----- Start of picture text -----**<br>
# instruction<br>You are a data quality controller for a single primary key group during table reconciliation.<br>Your role is to analyze a subset of rows sharing the same primary key value and decide which reconciliation operation<br>to perform next.<br>You are processing rows for **one specific primary key value only**. This is a focused reconciliation task for a single<br>entity.<br>The table contains values extracted from different pages of documents. Each page was processed independently by<br>extraction workers.<br>## Pipeline Context<br>**Preprocessing completed**:<br>1. [OK] **Primary Key Selected**: A semantic primary key has been identified based on the schema and data domain<br>2. [OK] **Primary Key Fields Canonicalized**: Values in the primary key fields have been standardized (case<br>normalization, whitespace trimming, format consistency)<br>3. [OK] **Table Split by Primary Key**: Rows have been grouped by primary key values<br>**Primary Key for this table**: {{primary_key}}<br>**Your specific primary key value**: {{pk_value}}<br>You are seeing ONLY the rows with this specific primary key value (typically 2-5 rows, occasionally more).<br>## Your Position in the Loop<br>You are in iteration {{iteration}} of {{max_iterations}} for this primary key group.<br>**Operations performed so far**: {{operations_history}}<br>After each operation you chose, an executor agent:<br>- Inspected the rows<br>- Generated and executed SQL to perform the operation<br>- Returned updated rows (could be 0, 1, or all rows depending on the operation)<br>You now see the **current state** of rows for this primary key after those operations.<br>## Your Goal<br>**Primary objective**: Reduce this primary key group to **exactly 1 row** through reconciliation operations.<br>Multiple rows with the same primary key indicate extraction redundancy, complementary information, or conflicts that<br>need resolution. The ideal end state is a single consolidated row containing all valid information for this<br>entity.<br>**When to stop before reaching 1 row**:<br>**----- End of picture text -----**<br>


34 

```
-Onlyiftheexecutorhasattemptedanoperationanddetermined,afterdeepanalysisofthedata,thattheremaining
rowsrepresentgenuinelydistinctobservationsthatshouldNOTbemerged
-Theexecutor’srefusalmustbebasedonsemanticevidence(e.g.,differenttemporalcontexts,distinctaspectsthat
shouldn’tbecombined)
-Simplynoticingthatrows"lookdifferent"isNOTsufficientreasontostop-differencesareexactlywhat
reconciliationoperationsaredesignedtohandle
**Whenexecutordoesnotreduceto1row,diagnosethereason**:
1.**Legitimaterefusal**:Theexecutorreasonedthatrowsrepresentdistinctentities/eventsthatshouldnotbemerged
(e.g.,multipleseparateoccurrences,differentaspectsthatshouldn’tbecombined)$\rightarrow$routeto**
stop**
2.**Wrongoperationchosen**:Theexecutortriedtoapplytheoperationbutthecurrentstateactuallyrequiresa
differentoperation(e.g.,routedtoconsolidatebutthereareconflictstoresolve,orroutedtodeduplicatebut
there’scomplementaryinformationtoaggregate)$\rightarrow$routetothe**appropriateoperation**that
addressestheactualdatapattern
##YourDecisionProcess
Youiterate:**inspect**$\rightarrow$**inspect**$\rightarrow$...$\rightarrow$**route**
Ateachcall,youdecide:
1.**inspect**:RunaSQLquerytounderstandthecurrentstatebetter
2.**route**:Routetoaspecificreconciliationoperation(orstop)
##AvailableRouteActions
**Important**:Whendecidingwhichoperationtorouteto,focuson**maindatacolumns**(measuredvalueslikenames,
dates,amounts,descriptions).Usemetadatacolumns(‘*_quote‘,‘*_rationale‘,‘_confidence‘,‘_is_explicit‘)to
understandthecontextandsemanticsofthemaindata.However,doNOTroutetoanoperationiftheissueexists
ONLYinmetadatacolumnswhilethemaindatacolumnsarealreadyclean.Routebasedonproblemsinmaindata
columns;theexecutorwillthenhandlecorrespondingmetadatacolumnsappropriately.
1.**deduplicate**
-Triggerwhen:Multiplerowswithredundantinformationinnon-primary-keyfields
-Purpose:Removeredundantrows,keepingthebestrepresentative
-Executorwillselectthebestrowbasedonconfidence,completeness,anddataquality
-Note:Ifakeycolumnhasnon-redundantinformation(differentvalues,NULLstofill,orconflicts),routeto
aggregate,consolidate,orresolve_conflictsinstead
2.**aggregate**
-Triggerwhen:ThefieldsemanticsCLEARLYindicatemultiplevaluesshouldcoexist(e.g.,listofallproducts,sum
acrosscategories)
-Usesparingly:Onlywhenyou’reconfidentthefieldismeanttoholdmultiplevalues,notwhenyou’retryingto
avoidmakingachoice
-Purpose:Combinemultiplevalidvaluesintoaunifiedrepresentation(arrays,lists,sums,ranges)
-Executorwillapply:SUM(additivequantitiesacrosscategories),ARRAY_AGG(multipleitemsinacollection),MIN/
MAX(rangebounds),COALESCE(fillingdifferentfields)
-Examples:Multiplephonenumbers$\rightarrow$array,revenueacrossproductlines$\rightarrow$sum,multiple
categories$\rightarrow$array
3.**consolidate**
-Triggerwhen:Informationisscatteredacrossrowswithincompletedatathatcanbefilledbycopyingvaluesfrom
otherrows
-Purpose:Broadcastcompleteinformationtofillgapswheredataismissing
-Executorwillpropagateconsistentvaluesacrossrowswithoutcombiningortransformingthem
-Note:Returnsmultiplerows;usewheninformationshouldbecopied,notmerged
4.**resolve_conflicts**
-Triggerwhen:Differentvaluesexistforafield,andonlyONEiscorrectormostappropriate
-Defaultchoice:Whenuncertainwhethertoaggregateorresolve,chooseresolve_conflicts
-Purpose:Selectthesinglebestvaluebasedonevidence,sourcequality,semanticfit,andspecificity
-Executorwillevaluatewhichvaluetokeepanddiscardtheothers
-Examples:Differentdatesforsameevent,differentnames,differentstatuses,differentmeasurements,conflicting
attributes
5.**canonicalize**
-Triggerwhen:Non-primary-keyfieldshavedifferentformatsorrepresentations
-Example:"5M"vs"5000000","USA"vs"UnitedStates"
-Purpose:Standardizenon-primary-keycolumnvaluestoacanonicalform
-Note:Primarykeyfieldsarealreadycanonicalized.NULLvaluesshouldgenerallybepreserved.
6.**stop**
-Triggerwhen:
***Idealcase**:Successfullyreducedto1row
***Acceptablecase**:Afterdiagnosingthemostrecentexecutorresult,youdeterminetheexecutorlegitimately
refusedtomergebecauserowsrepresentgenuinelydistinctobservations(notbecausethewrongoperationwas
chosen)
-Beforestoppingwith>1rowremaining:Analyzetheexecutor’sreasoningfromthelastoperation.Diditrefuse
becauserowsaretrulydistinct,orbecausetheoperationchosenwasn’tsuitablefortheactualdatapattern?
-Iftheoperationwaswrong(e.g.,triedtodeduplicatewhenshouldaggregate,ortriedtoconsolidatewhenshould
resolveconflicts),routetothecorrectoperationinstead
-DoNOTstopjustbecause:Rowslookdifferent,havesomeNULLvalues,orseem"mostlydone"-theseareexactly
whatreconciliationaddresses
-Purpose:Endthereconciliationloopforthisprimarykeygroup
##CurrentState
Questiontoanswer:{{question}}
Tablename:{{table_name}}
Primarykeyvalue:{{pk_value}}
```

35 

```
Currentrowcount:{{current_row_count}}
##Schema
{{schema}}
##TableStatistics(foryourspecificPKgroup)
{{table_stats}}
##YourInspectionHistory(currentcontrollercall)
{{inspection_history}}
##InspectionBudget
{%ifmust_route_now%}
[WARNING]WARNING:Nomoreinspectionsallowed.YouMUSTroutetoanactionnow(action:"route").
{%else%}
Youcaninspect{{inspections_remaining}}moretime(s)beforeyoumustroute.
{%endif%}
#Task
Decide:shouldyouinspectmore,orareyoureadytoroutetoanaction?
**Ifyouneedmoreinformation**,output:
-action:"inspect"
-sql:YourinspectionSQLquery(e.g.,checkrowdifferences,compareconfidencescores)
-reasoning:Whyyouneedthisinspection
**Exampleinspections:**
‘‘‘sql
--ViewallrowsforthisPK
SELECT*FROM{{table_name}};
--Comparemeasuredvaluesacrossrows
SELECTuser_id,message_text,date,_confidenceFROM{{table_name}};
--Checkmetadata
SELECT_quote,_rationale,_is_explicitFROM{{table_name}};
‘‘‘
**Ifyou’rereadytoroute**,output:
-action:"route"
-route_to:Oneof[deduplicate,aggregate,consolidate,resolve_conflicts,canonicalize,stop]
-reasoning:Whyyou’reroutingtothisactionbasedonthecurrentstate
Thinkabout:
-**Target**:Areweat1rowyet?Ifnot,whatoperationwillmoveuscloser?
-Arerowsredundant(sameinformation),complementary(differentcolumnspopulated),orconflicting(samecolumnwith
differentvalues)?
-**Forconflictingvalues**:Defaultto**resolve_conflicts**topickthebestvalue.Onlyuse**aggregate**ifthe
fieldclearlyneedstoholdmultiplevalues
-Arethereformatinconsistenciestocanonicalize?
-Canconsolidateordeduplicatereducerowcount?Canresolve_conflictseliminatevalueconflicts?
-**Iflastoperationdidn’treduceto1row**:Wasitalegitimaterefusal(rowsaretrulydistinct)orwasthewrong
operationapplied?Ifwrongoperation,routetothecorrectone.
-**Onlystopearlyif**:Executorhaslegitimatelyrefusedtomergeafteranalysisbecauserowsrepresentdistinct
entities,NOTbecausethechosenoperationwasn’tsuitableforthedatapattern
**Important**:
-You’reonlyseeingrowsforONEprimarykeyvalue
-Afteryouroute,anexecutorwillperformthatONEoperation
-You’llbecalledagaintodecidethenextoperation(upto{{max_iterations}}totaliterations)
-Eachoperationshouldmakemeaningfulprogresstowardaclean,consolidatedresult
```

**Reconciliation agent** Agent that generates and executes SQL to perform the reconciliation operation chosen by the controller. 

- `# instruction You are an executor agent for reconciling rows that share the same primary key values. You are processing a small subset of the full table - specifically, **all rows that share a specific primary key value **.` 

- `{% if objective == "consolidate" %}Your task is to fill NULL values by broadcasting non-NULL values across these rows, returning **multiple rows** with gaps filled.{% elif objective == "aggregate" %}Your task is to aggregate values across these rows where appropriate, returning **multiple rows** with aggregated values broadcasted.{% elif objective == "deduplicate" %}Your task is to identify and remove redundant rows, returning **1 or more rows** depending on redundancy patterns.{% else %}Your task is to consolidate these rows into **1 output row** that best represents this entity.{% endif %}` 

- `The table contains values extracted from a document based on the provided schema. Each row was extracted independently by workers processing different document pages, which may have had partial context.` 

- `## Pipeline Context **Preprocessing completed**: 1. [OK] **Primary Key Selected**: A semantic primary key has been identified` 

`2. [OK] **Primary Key Fields Canonicalized**: Primary key values have been standardized` 

```
**PrimaryKeyforthistable**:{{primary_key}}
```

36 

- `**Your specific primary key value**: {{pk_value}} You are only seeing rows with this specific primary key value. Typically 2-5 rows, but occasionally more. ## Your Task You have been assigned the objective: **{{objective}}** You iterate: inspect $\rightarrow$ inspect $\rightarrow$ ... $\rightarrow$ generate_merge_sql {% if objective == "consolidate" %}Your merge SQL should return **multiple rows** (all input rows with NULLs filled where appropriate).{% elif objective == "aggregate" %}Your merge SQL should return **multiple rows** (all input rows with aggregated values broadcasted where appropriate).{% elif objective == "deduplicate" %}Your merge SQL should return **1 or more rows** (deduplicated set where redundant rows are reduced to representatives, distinct rows are preserved).{% else %}Your merge SQL should return **1 row** that consolidates all the rows in your subset.{% endif %} However:` 

- `- If you determine that the rows in your group are NOT actually duplicates/conflicts but represent distinct entities or events, return **all original rows** by using ‘SELECT * FROM {{table_name}}‘ to preserve them.` 

- `**Note on metadata fields**: Columns like _rationale, _quote are metadata about the extraction process, not measured values.` 

- `{% if objective == "deduplicate" %} ## Objective: Deduplicate **Goal**: Identify which rows contain redundant information in main data columns and deduplicate them, while preserving rows that contain distinct information. Return the deduplicated set.` 

- `### Phase 1: Inspect and Identify Redundancy Groups **Inspection strategy**: - Always examine all rows first: ‘SELECT * FROM {{table_name}}‘ - Read the _quote and _rationale fields to understand what each row represents - Compare ALL non-primary-key measured fields (date, amount, description, etc.) - Consider the semantic context: Is it possible for this entity type to have multiple instances? **Critical Decision: Can this event/entity occur multiple times in reality?** **Step 1: Identify redundancy patterns** Deduplication applies when rows have the same values across ALL (or nearly all) main data columns. You need to identify which specific rows share redundant information:` 

- `1. **All rows redundant**: All rows have the same values across main data columns - Example: 5 rows all describing the same entity with the same attributes (Company X, founded 2010, headquarters Chicago, CEO John Smith) extracted from 5 different pages` 

- `- All 5 rows form one redundancy group because they contain the same complete set of facts - **Action needed**: Select 1 best row from this group` 

- `2. **Partial redundancy**: Some rows share the same complete set of main data values (forming redundant subsets), while other rows differ in at least one main data column` 

- `- Example: 5 rows describing purchases where 3 have identical values (Item A, $50, 2024-01-10, Store X) and 2 have different values (Item B, $30, 2024-01-11, Store Y) and (Item A, $50, 2024-01-12, Store X)` 

- `- The 3 identical rows form one redundancy group - The other 2 rows are distinct (each differs in date or item from the redundancy group) - **Action needed**: Select 1 best row from the redundancy group, keep the 2 distinct rows unchanged` 

- `3. **No redundancy**: Each row has different values in at least one main data column - Example: 3 rows describing different product variants (Color: Red, Size: M), (Color: Blue, Size: L), (Color: Red, Size: L)` 

- `- No redundancy groups exist because each row has a unique combination of values - **Action needed**: Return all rows unchanged using ‘SELECT * FROM {{table_name}}‘` 

- `**Why grouping matters**: Redundant rows (those with the same complete set of main data values) should be reduced to one representative per group, but distinct rows (those with at least one differing main data value) must be preserved. Your SQL needs to handle both simultaneously.` 

- `**Step 2: For each redundancy group identified**, determine the semantic nature: **Important note on distinguishing metadata**: The absence of differing metadata (timestamps, quotes, rationale) between duplicate rows does NOT automatically indicate extraction redundancy. It may simply mean the extraction process was coarse-grained and didn’t capture fine-grained distinguishing details. Focus on the semantic nature of what the data represents.` 

- `**Case A: Multiple real-world occurrences are possible** - The entity/relationship represents actions, events, or transactions that can legitimately happen multiple times - Characteristics of multiple occurrences: - Behavioral or transactional data (purchases, interactions, activities) - Events occurring across time that can repeat - Primary key includes temporal dimensions without fine-grained precision (date without timestamp) - Data describes "what happened" rather than "what is"` 

- `- These rows represent DISTINCT REAL-WORLD EVENTS that share the same entity identifier - **Action**: Keep 1 best row, set ‘number_instances = <total_row_count>‘ - **Interpretation**: This event/transaction occurred N times in the real world **Case B: Only one instance exists in reality** - The fact/attribute is an intrinsic property or one-time event that cannot change or repeat - Characteristics of singular facts: - Definitional attributes (birth date, founding year, inherent properties) - One-time historical events that cannot recur - Data describes "what is" rather than "what happened" - Primary key is purely entity-based without temporal dimensions` 

37 

```
-TheserowsareEXTRACTIONREDUNDANCY:Thesamesingularfactcapturedmultipletimesfromdifferentsources
-**Action**:Keep1bestrow,set‘number_instances=1‘
-**Interpretation**:ThisfactexistsoncebutwasextractedNtimes
###Phase2:GenerateMergeSQL
Afteridentifyingredundancygroupsandtheirsemanticnature,generateSQLto:
1.Fromeachredundancygroup,selectthebestrepresentativerow(mostcomplete,bestquote)
2.Set‘number_instances‘appropriatelyforeachrepresentative
3.Includeallrowsthatarenotpartofanyredundancygroup(distinctrows)
4.Returnthededuplicatedset
**Example1:Allrowsredundant**(5rowsallcontainsameartistbirthdateextractedfromdifferentpages)
Inspectionreveals:All5rowshaveidenticalmaindata(artist_name,date_of_birth,etc.),onlydifferinmetadata
Decision:Extractionredundancy(birthdateisasingularfact)
Action:Return1rowwithnumber_instances=1
‘‘‘sql
--Afterinspection,choserow_id=102asbest(highestconfidence,clearestquote)
SELECT
*EXCLUDE(number_instances),
1ASnumber_instances--Singularfactextracted5times
FROM{{table_name}}
WHERErow_id=102
‘‘‘
**Example2:Subsetredundancy**(5rows:3containredundantalternatename"NameA",2containdistinctalternate
names"NameB"and"NameC")
Inspectionreveals:
-Rows201,202,203:allhavealternate_name="NameA"$\rightarrow$redundancygroup1
-Row204:hasalternate_name="NameB"$\rightarrow$distinct
-Row205:hasalternate_name="NameC"$\rightarrow$distinct
Decision:
-"NameA"appears3times(extractionredundancy,samenameextractedfrom3pages)
-"NameB"and"NameC"aredistinctvalues,shouldbepreserved
Action:Keep1rowfromthe"NameA"group,plusthe2distinctrows$\rightarrow$return3rowstotal
‘‘‘sql
--Afterinspection,choserow_id=201fromthe"NameA"groupasbestrepresentative
SELECT
*EXCLUDE(number_instances),
1ASnumber_instances--"NameA"issingular,extracted3times
FROM{{table_name}}
WHERErow_id=201--Bestofthe"NameA"redundancygroup
UNIONALL
--Preservedistinctrowsunchanged
SELECT*FROM{{table_name}}
WHERErow_idIN(204,205)--"NameB"and"NameC"aredistinct
‘‘‘
**Alternativepatternforpartialdeduplication**:
‘‘‘sql
--UseCASEtomarkwhichrowstokeep,thenfilter
WITHrankedAS(
SELECT*,
CASE
--Redundancygroup:rowswithsimilarmaindata
WHENmain_col_a=’duplicate_value’ANDmain_col_bISNULL
THENROW_NUMBER()OVER(PARTITIONBYmain_col_a,main_col_bORDERBY_confidenceDESC,row_id)
--Allotherrowsaredistinct,keepthem
ELSE1
ENDASkeep_rank
FROM{{table_name}}
)
SELECT
*EXCLUDE(keep_rank,number_instances),
CASE
WHEN<conditionforredundancygroup>THEN<1orcount>
ELSEnumber_instances--Preservefordistinctrows
ENDASnumber_instances
FROMranked
WHEREkeep_rank=1
‘‘‘
**Keypoints**:
-‘number_instances‘valuedependsonyoursemanticanalysis
-Return1ormorerowsdependingonredundancystructure
-Eachreturnedrowshouldhaveappropriate‘number_instances‘value
-Preserverowswithdistinctmaindatavalues
-Onlydeduplicaterowsthataretrulyredundantinmaindatacolumns
-Useinspectionresultstojustifyyourchoice
-Documentyourreasoningclearly
{%endif%}
{%ifobjective=="aggregate"%}
##Objective:Aggregate
```

38 

```
**Goal**:Forcolumnscontainingmultipledistinctvaluesthatshouldbecombined,computeanaggregatedrepresentation
(sum,collection,concatenation)andbroadcastittoallrows.
**Inspectionstrategy**:
-Examineallrows:‘SELECT*FROM{{table_name}}‘
-Foreachcolumn,identifyifithasmultipledistinctnon-NULLvalues
-Read_quoteand_rationaletoverifythesevaluesarecomplementaryandshouldbecombined
-Determinetheappropriateaggregationmethodforeachcolumn
**Decisionlogicforeachcolumn**:
Forcolumnswithmultipledistinctnon-NULLvalues:
-**Complementaryvaluesthatshouldcombine**$\rightarrow$applyaggregationacrossrowsandbroadcastresulttoall
rows
-Multiplevalidinstancesofamulti-valuedattribute:ARRAY_AGG,STRING_AGG
-Numericquantitiestosum:SUM
-Rangeboundaries:MIN/MAXpairs
-**Conflictingvalues**(mutuallyexclusive,onlyonecorrect)$\rightarrow$leaveunchanged,willbehandledby
resolve_conflicts
-**SinglevalueorallNULLs**$\rightarrow$leaveunchanged
**Keyprinciple**:Aggregationcombinessemanticallycompatiblevaluesthatshouldcoexist.Returnallrowswith
aggregatedvaluesbroadcastedwhereappropriate.
**MergeSQLstrategy**:
1.**Firstinspect**:Determinewhichcolumnshavemultiplevaluesthatshouldbecombined
2.**ThengenerateSQL**:Usewindowfunctionstocomputeaggregatedvaluesandbroadcasttoallrows
3.Columnsthatdon’tneedaggregationremainunchanged
4.ReturnALLrowswithaggregationsappliedwhereneeded
**SQLPatternusingwindowfunctions**:
‘‘‘sql
--Applyaggregationtospecificcolumnsthatneedcombining,broadcastresulttoallrows
--Use*EXCLUDEtoavoidlistingallcolumns
SELECT
*EXCLUDE(col_a,col_a_quote,col_b),
--Aggregatemulti-valuedcolumnacrossallrowsinthisPKgroup
STRING_AGG(DISTINCTcol_a,’|’)OVER(PARTITIONBYpk_col1,pk_col2)AScol_a,
STRING_AGG(DISTINCTcol_a_quote,’|’)OVER(PARTITIONBYpk_col1,pk_col2)AScol_a_quote,
--Sumnumericvaluesacrossrows
SUM(col_b)OVER(PARTITIONBYpk_col1,pk_col2)AScol_b
FROM{{table_name}}
WHEREpk_col1=’...’ANDpk_col2=’...’
‘‘‘
**Alternativepatternusingsubqueries**:
‘‘‘sql
--Computeaggregationsinsubqueries,joinbacktopreserveallrows
--Use*EXCLUDEforcleanerSQL
SELECT
t.*EXCLUDE(col_a,col_a_quote),
--Replacecol_awithaggregatedvalueacrossallrows
(SELECTSTRING_AGG(DISTINCTcol_a,’|’)FROM{{table_name}}WHEREpk_col1=t.pk_col1ANDpk_col2=t.pk_col2)AS
col_a,
(SELECTSTRING_AGG(DISTINCTcol_a_quote,’|’)FROM{{table_name}}WHEREpk_col1=t.pk_col1ANDpk_col2=t.pk_col2
)AScol_a_quote
FROM{{table_name}}t
WHEREt.pk_col1=’...’ANDt.pk_col2=’...’
‘‘‘
**Keypoints**:
-ReturnALLinputrowswiththesameschema(nocolumnsaddedordropped)
-Onlyaggregatecolumnswherecombiningvaluesmakessemanticsense
-Broadcastaggregatedvaluestoallrowsinthegroup
-Columnswithconflictsorsinglevaluesremainuntouchedforlateriterations
{%endif%}
{%ifobjective=="consolidate"%}
##Objective:Consolidate
**Goal**:FillNULLvaluesinnon-primary-keycolumnsbybroadcastingnon-NULLvaluesfromotherrowsinthesame
column,wheresemanticallyappropriate.
**Inspectionstrategy**:
-Examineallrows:‘SELECT*FROM{{table_name}}‘
-Foreachcolumn,identifyifithasNULLsinsomerowsandnon-NULLvaluesinothers
-Read_quoteand_rationaletodetermineifbroadcastingmakessemanticsense
-Checkifnon-NULLvaluesinthesamecolumnareconsistent(notconflicting)
**Decisionlogicforeachcolumn**:
ForeachcolumnwithmixedNULL/non-NULLvalues:
-**IfthereisexactlyONEdistinctnon-NULLvalue**$\rightarrow$broadcastittofillNULLsinthatcolumn
-**IfthereareMULTIPLEdistinctnon-NULLvalues**$\rightarrow$doNOTconsolidatethiscolumn;leaveitunchanged
forotheroperations(aggregateorresolve_conflicts)tohandle
-IfsemanticcontextsuggestsNULLsshouldremain(e.g.,trulymissingdata)$\rightarrow$preserveNULLs
**Keyprinciple**:Consolidationisafocusedoperationthatonlyfillsgapswhereinformationisunambiguous.Whena
columnhasmultipledistinctnon-NULLvalues,thosevaluesneeddifferenthandling(combiningviaaggregationor
selectingviaconflictresolution)whichwillbeaddressedinsubsequentiterations.
**MergeSQLstrategy**:
1.**Firstinspect**:Examineallrowstounderstandwhichrowhasthebestvalueforeachcolumn
```

39 

```
2.**ThengenerateSQL**:Foreachconsolidatablecolumn,choosethespecificrow_idwiththemostreliablevalue
3.Usesubqueriesorwindowfunctionstobroadcastfromthosechosenrows
4.ReturnALLinputrowswithNULLsfilledONLYincolumnsthatmeetconsolidationcriteria
5.Forcolumnswithmultipledistinctnon-NULLvalues,preservethemexactlyas-is
**Pattern1:Selectivebroadcastingwithsubqueries**
‘‘‘sql
--Afterinspecting,substituteactualrow_idsyouidentifiedashavingthebestvalues
--Use*EXCLUDEforcleanerSQLthatavoidslistingallcolumns
SELECT
t.*EXCLUDE(col_a,col_a_quote,col_b,col_b_quote),
--Broadcastcol_afromtherowyoudeterminedhasthebestvalue(e.g.,row117)
COALESCE(t.col_a,(SELECTcol_aFROM{{table_name}}WHERErow_id=<row_with_best_col_a>))AScol_a,
COALESCE(t.col_a_quote,(SELECTcol_a_quoteFROM{{table_name}}WHERErow_id=<row_with_best_col_a>))AS
col_a_quote,
--Broadcastcol_bfromadifferentrowyouchose(e.g.,row118)
COALESCE(t.col_b,(SELECTcol_bFROM{{table_name}}WHERErow_id=<row_with_best_col_b>))AScol_b,
COALESCE(t.col_b_quote,(SELECTcol_b_quoteFROM{{table_name}}WHERErow_id=<row_with_best_col_b>))AS
col_b_quote
FROM{{table_name}}t
‘‘‘
**Pattern2:Usingwindowfunctionswithspecificrowselection**
‘‘‘sql
--Use*EXCLUDEforsimplerSQL
SELECT
*EXCLUDE(col_a,col_b),
--Consolidatecol_abybroadcastingfromtherowwithbestmetadata
COALESCE(col_a,MAX(CASEWHENrow_id=<chosen_row_id>THENcol_aEND)OVER(PARTITIONBYpk_col1,pk_col2))AS
col_a,
--Consolidatecol_bfromadifferentsourcerow
COALESCE(col_b,MAX(CASEWHENrow_id=<other_row_id>THENcol_bEND)OVER(PARTITIONBYpk_col1,pk_col2))AS
col_b
FROM{{table_name}}
‘‘‘
**Keyapproach**:Afterinspectingtherows,determineforeachconsolidatablecolumnwhichspecificrowhasthemost
reliablenon-NULLvalue(basedonquotequality,rationaleclarity,orcompleteness),thenbroadcastthat
specificvalue.
-Keepmetadatacolumns(_quote,_rationale)fromoriginalrowsunchanged
**Keypoints**:
-Returnmultiplerows(allinputrowswithselectiveNULLsfilled)
```

- `Only modify columns where broadcasting a single value is appropriate` 

- `- Leave columns with multiple distinct non-NULL values untouched for subsequent operations - The reconciliation loop will handle remaining issues in later iterations {% endif %} {% if objective == "resolve_conflicts" %} ## Objective: Resolve Conflicts **Goal**: When rows have conflicting measured values, determine if it’s a true conflict to resolve or separate legitimate occurrences.` 

- `**Inspection strategy**: - Examine all rows: ‘SELECT * FROM {{table_name}}‘ - Identify which non-primary-key fields have conflicting (different distinct) values - Read _quote and _rationale fields to understand what each row represents **Important decision**: - **If conflicting field is a temporal/event field** (date, timestamp) and both values are equally valid $\rightarrow$ these represent separate events, keep all rows: ‘SELECT * FROM {{table_name}}‘` 

- `- **If conflict is due to measurement error** (different amounts, descriptions for same event) $\rightarrow$ resolve by selecting most reliable value` 

- `**Merge SQL strategy for true conflicts**: - Use metadata to decide: Compare _quote and _rationale for supporting evidence - **Decision process**: For each competing row, explicitly evaluate it against each criterion below. Document your assessment for each criterion before making the final selection.` 

- `- Selection criteria: 1. **Field definition alignment**: Prefer values that match the semantic scope of the field - Choose values that directly answer what the field asks for, not broader or narrower variants - Match the specificity: if the field has modifiers or constraints, select values that align with those constraints` 

- `- When the field name corresponds to a standard reporting item, the value extracted from where that item is formally reported should be strongly preferred over values described or explained elsewhere, even if explanatory sections provide additional context` 

- `2. **Extraction location relevance**: Prefer values extracted from locations that directly serve the field’s purpose - Consider which document sections or structures are authoritative for this type of information - Prioritize locations where this field would naturally appear in its primary context - When available, examine text_header or similar metadata to understand extraction context` 

- `3. **Source authority**: - Direct measurements or final values over constituent parts or supplementary details - Structured presentations (tables, statements, forms, labeled fields) over descriptive text (discussions, notes, commentary)` 

- `- Primary data sections over explanatory sections - Explicit field labels over derived mentions` 

- `4. **Quote quality**: Prefer direct quotes over inferred values 5. **Rationale clarity**: Prefer clear, specific rationale` 

- `- **Preferred approach**: After inspection, identify the specific row_id you want to keep and filter directly: ‘SELECT * FROM {{table_name}} WHERE row_id = <chosen_row_id>‘` 

40 

```
-Alternative:UseROW_NUMBER()withORDERBY,butbecarefulwithtiebreakers
{%endif%}
#Input
{%ifverification_mode%}
**Phase**:Verification
Primarykeyvalueyouwereprocessing:{{pk_value}}
##InitialTableSchema(beforeyourSQL)
{{initial_schema}}
##FinalTableSchema(afteryourSQL)
{{final_schema}}
###YourGeneratedSQL
‘‘‘sql
{{generated_sql}}
‘‘‘
###YourReasoningWhenGenerating
{{generation_reasoning}}
###AvailableTablesforInspection
TwotablesareregisteredinDuckDB:
-**‘initial_table‘**:{{initial_row_count}}rowsBEFOREyourSQLwasapplied
-**‘final_table‘**:{{final_row_count}}rowsAFTERyourSQLwasapplied
**IMPORTANT**:YoudoNOThavetheactualtablecontentsyet.YoumustinspectbothtablesusingSQLtoverifythe
transformationwasexecutedcorrectly.
###YourVerificationInspectionHistory
{{verification_inspection_history}}
{%else%}
Questiontoanswer:{{question}}
Tablename:{{table_name}}
Primarykeyvalueyou’reprocessing:{{pk_value}}
##Schema
{{schema}}
##TableStatistics(foryoursubset)
{{table_stats}}
##Controller’sReasoning
Thecontrollerroutedtothisobjectivewiththefollowingreasoning:
{{controller_reasoning}}
##YourInspectionHistory
{{inspection_history}}
##InspectionErrors(ifany)
{{inspection_error_feedback}}
##SQLGenerationErrors(ifany-forretry)
{{sql_error_feedback}}
##Status
{%ifinspections_remaining>0%}
**Phase**:Inspection(optional)
**Budget**:Youcaninspect{{inspections_remaining}}moretime(s),orchoosetogeneratemergeSQLnow.
Afterinspectionphaseends,you’llentertheSQLgenerationphasewith3attemptstogeneratevalidSQL.
{%else%}
**Phase**:SQLGeneration
[WARNING]YouarenowintheSQLgenerationphase.Youcannolongerinspect.
GeneratethemergeSQL.Ifitfails,you’llgeterrorfeedbackandupto2moreattempts(3total).
{%endif%}
{%endif%}
#Task
{%ifverification_mode%}
YourmergeSQLhasbeenexecuted.VerifythattheSQLcorrectlytransformedinitial_table$\rightarrow$final_tableas
intendedbyyourreasoning.
**VERIFICATIONSCOPE:**
YouarecheckingifyourSQLperformedthetransformationcorrectly,NOTwhetherthereconciliationapproachitselfwas
correct.Focuson:DidtheSQLexecutethelogicdescribedinyourreasoning?
**VERIFICATIONWORKFLOW:**
1.**First,youMUSTinspect**(especiallyifyouhaven’tinspectedyet):
-action:"inspect"
-sql:Querytocompareinitial_tableandfinal_table(e.g.,SELECT*fromboth,orJOINthem)
-reasoning:Whattransformationaspectsyou’reverifying(rowcountchanges,columntransformations,etc.)
```

41 

```
2.**Afterinspectingandverbalizingwhatyouobserved,thendecide:**
**OptionA:Approve**-TheSQLcorrectlyexecutedtheintendedtransformation
-action:"approve"
-reasoning:Confirmthetransformationmatchesyourgenerationreasoning:
*Doesinitial_table$\rightarrow$final_tablematchtheexpectedtransformation?
*Arerowcounts,aggregations,andcolumnvaluesasdescribedinyourreasoning?
**OptionB:Regenerate**-TheSQLdidnotexecutetheintendedtransformationcorrectly
-action:"regenerate"
-sql:CorrectedSQLthatproperlyimplementsyouroriginalreasoning
-reasoning:Whattransformationerrordidyouobserve?HowdoesthenewSQLfixit?
**RemainingInspections**:{{remaining_inspections}}
**CriticalRules:**
-Inspecttheactualdatabeforedeciding
-Compareinitial_tablevsfinal_tabletoverifythetransformation
-ApproveiftheSQLcorrectlyimplementedyourreasoning
-RegenerateiftheSQLfailedtoexecutetheintendedtransformation
-RegeneratedSQLisauto-accepted(nosecondverification)
{%else%}
Decide:shouldyouinspectmore,orareyoureadytogeneratethemergeSQL?
Ifyouneedmoreinformation,output:
-action:"inspect"
-sql:YourinspectionSQLquery(SELECTstatementtounderstandyoursubset)
-reasoning:Whyyouneedthisinspection
Ifyou’rereadytogeneratethemerge,output:
-action:"generate_merge_sql"
-sql:YourmergeSQLquerythat{%ifobjective=="consolidate"%}returns**multiplerows**withNULLsfilled{%elif
objective=="aggregate"%}returns**multiplerows**withaggregatedvaluesbroadcasted{%elifobjective=="
deduplicate"%}returns**1ormorerows**withredundancyremoved{%else%}consolidatesallrowsinyoursubset
to**exactly1row**{%endif%}
-reasoning:HowyourSQLaccomplishestheobjective
**Important**:
-**MergeSQLformat**:MustbeaSELECTstatement{%ifobjectivenotin["consolidate","aggregate","deduplicate"]%}
thatreturnsexactly1row{%endif%}(noCREATETABLE,INSERT,orDELETE)
-**Schemapreservation**:IncludeALLoriginalcolumns(row_id,page_number,*_quote,*_rationale)
-**number_instancescolumn**:{%ifobjective=="deduplicate"%}Setthistothetotalrowcount(iftrueduplicates)
or1(ifsemanticallydistinctinstances)foreachreturnedrow{%elifobjectivein["consolidate","aggregate"]
%}Preservetheexistingvalueforeachrow{%else%}Thiscolumnisnotvisibleinyourtable;itwillbe
preservedautomatically{%endif%}
-**Quote/rationalecolumns**:Whenmerging,updatethesecolumnsaccordingly(concatenatesourcesorkeepthebest)
{%ifobjectivein["consolidate","aggregate"]%}-**Multiplerowoutput**:YourSQLshouldreturnallinputrowswith
transformationsappliedwhereappropriate{%elifobjective=="deduplicate"%}-**Flexiblerowoutput**:Return
1ormorerowsdependingonredundancystructure(1representativeperredundancygroup+alldistinctrows){%
else%}-**Singlerowoutput**:YourSQLMUSTreturnexactly1row.UseLIMIT1,ROW_NUMBER()=1,orGROUPBYas
appropriate{%endif%}
-**Resultlimits**:InspectionSQLshoulduseLIMITtoavoidlargeresults
{%endif%}
```

## **D.1.4 Answer Generation** 

**SQL Query Generator** Iteratively generates SQL queries to inspect the reconciled data and gather information for answering. 

```
#instruction
YouareaSQLquerygeneratorforansweringquestionsaboutdatainDuckDBtables.
{%ifcitation_mode%}
Youhavecompletedthequerygenerationphase.Yourquerieshavebeenusedbythefinalanswergeneratortoproducethe
verbalizedanswer.
NowyouneedtogenerateasingleSQLquerythatselectstherowsusedtoanswerthequestion,includingallprovenance
columns(quote,rationale,text_headers).Thiswillenablepropercitationofthesourcedata.
{%else%}
Youcannotseefulltables-onlyschemaandstatistics.UseSQLqueriestoretrieveinformationneededtoanswerthe
question.
{%endif%}
{%ifnotcitation_mode%}
##Strategy
{%ifstrategy=="full_table"%}
Tableissmall(<=100rows).Startbysamplingthefulltablewith‘SELECT*FROMtableLIMIT{{row_limit}}‘toseeall
data,thenusefocusedqueriesifneeded.
{%else%}
Tableislarge(>100rows).UsefocusedSQLquerieswithfiltersandaggregations.Resultsarelimitedto{{row_limit}}
rows.Considersamplingfirstifyouneedtounderstandthestructure.
{%endif%}
##Guidelines
-Youhave{{queries_remaining}}quer{{"y"ifqueries_remaining==1else"ies"}}remaining
{%ifis_first_queryandstrategy!="sql_focused"%}
-**Firstquery**:Samplethetable(‘SELECT*FROMtableLIMIT{{row_limit}}‘)toseeactualdata
{%endif%}
```

42 

```
-Usefocusedqueries:filters,aggregations,ordering
```

```
-Verifyresultsthroughmultipleapproaches(sample,thenaggregate,orcross-checkwithdifferentfilters)
-**[Important]Emptyresults**:Ifaqueryreturnsnorows,verifythiswithalternativequeriesbeforedrawing
conclusions---emptyresultsoftenindicateSQLissuesratherthanmissingdata
-Checkdatatypesandsamplevaluesbeforefiltering
-Eachqueryresultistruncatedto{{row_limit}}rows
-Forcalculations,useSQLfunctionslikeROUND()toensureprecision
-Defaultordering:useORDERBYrow_idASCtopreserveorderinthedocumentunlessthequestionexplicitlyrequestsa
differentsort(e.g.,bydate).
-Onlythevaluesmightbewrong(schema/columnsareassumedcorrect).Userationale/evidencecolumnstovetrows;
preferrowswithclearrationalesthatmatchthequestionanddroprowswithweakorconflictingrationale.Feel
freetowritecleaningSQL(filters,trims,normalizations)beforeusingthedata.
#Workingwithdates
-IfacolumnisDATE/TIMESTAMP,donotuseLIKE,SUBSTR,LEFT,RIGHT,ILIKE,orregexonit.
-Usedate-nativepredicatesonly:EXTRACT,DATE_TRUNC,strftime,orrangefilterswithDATE’YYYY-MM-DD’.
{%ifsql_error_feedback!="NopreviousSQLerrors"%}
##PreviousSQLErrors
{{sql_error_feedback}}
FixtheSQLandretry.
{%endif%}
{%endif%}
#input
##Question
{{question}}
##Schema
{{schema}}
{%ifnotcitation_mode%}
##TableStatistics
{{table_stats}}
##PreviousQueries
{{query_history}}
##Status
{%ifqueries_remaining>0%}
{{queries_remaining}}quer{{"y"ifqueries_remaining==1else"ies"}}remaining.Either:
-‘action:"query"‘withSQLtoretrievemoreinformation
-‘action:"finalize"‘ifyouhavesufficientinformation
{%else%}
Noqueriesremaining.Mustfinalizenow(‘action:"finalize"‘).
{%endif%}
{%endif%}
{%ifcitation_mode%}
#CitationSQLGeneration
##PreviousInvestigationHistory
{{query_history}}
##FinalizationReasoning
{{finalization_reasoning}}
##FinalAnswerGenerated
{{final_answer}}
##YourCitationSQLAttempts
{{citation_attempts_history}}
##Task
GenerateaSQLquerythatselectstherowsusedtoanswerthequestion.Include:
-Allprimarykeycolumns
-Alldatacolumnsreferencedintheanswer
-**Provenancecolumns**:Anycolumnswithnamescontaining"quote","rationale",or"text_header"(case-insensitive)
-Filtertoonlythespecificrowsthatcontributedtotheanswer
**Workflow:**
1.**Ifyouhaven’ttriedyetORneedtorefine**:
-action:"execute"
-sql:YourcitationSQLquery
-reasoning:Whichrowsyou’reselectingandwhy
2.**IfyourpreviousSQLresultlooksgood**:
-action:"finalize"
-sql:""(leaveempty)
-reasoning:Confirmthatthepreviousresultcontainstherightcitationrows
```

```
**Attemptsremaining**:{{attempts_remaining}}
```

```
Provide:
```

- `‘reasoning‘: Explain your decision` 

- `‘action‘: "execute" (try SQL) or "finalize" (accept previous result)` 

- `- ‘sql‘: SQL query (required if action="execute", leave empty if action="finalize") {% else %} Provide:` 

- `‘reasoning‘: What you need or why you’re finalizing` 

43 

```
-‘action‘:"query"or"finalize"
```

```
-‘sql‘:SQLquery(requiredifactionis"query")
{%endif%}
```

**Answer Verbalization** Verbalizes SQL query results into a natural language answer. 

**==> picture [372 x 368] intentionally omitted <==**

**----- Start of picture text -----**<br>
# instruction<br>{% if citation_mode %}<br>Generate a citation paragraph based on the source data rows below.<br>You have already answered the question. Now you need to generate a citation paragraph that explains which source<br>documents and specific data points were used. This paragraph will be appended to your answer to provide proper<br>attribution.<br>**DO NOT re-answer the question.** Only generate the citation paragraph.<br>{% else %}<br>Answer the question using the SQL query results below.<br>You executed one or more SQL queries to gather information from the database. Based on the query results, provide a<br>comprehensive answer to the question.<br>{% endif %}<br># input<br># Question<br>{{question}}<br># Schema<br>{{classes}}<br>{% if citation_mode %}<br># Your Generated Answer<br>{{generated_answer}}<br># Citation SQL Query<br>The following SQL query was executed to retrieve source data rows with provenance information:<br>‘‘‘sql<br>{{citation_sql}}<br>‘‘‘<br># Source Data Rows (with Provenance)<br>{{citation_data}}<br># Task<br>Generate a concise citation paragraph (2-4 sentences) that:<br>- Mentions the specific source documents or data entries used<br>- References key provenance information (quotes, rationales, text headers) if available<br>- Explains how the source data supports the answer<br>This citation paragraph will be appended to your answer above. Format it to flow naturally after the answer.<br>{% else %}<br># SQL Queries Executed<br>{{tool_call}}<br># Query Results<br>{{tool_output}}<br>Based on these SQL query results, provide a clear and accurate answer to the question. Reference specific values from<br>the query results to support your answer.<br>{% endif %}<br>**----- End of picture text -----**<br>


## **E Example Execution Trajectories** 

We provide representative execution traces from the SLIDERS pipeline to illustrate key components in action. These logs show the reasoning and SQL operations performed by different agents during processing. 

## **E.1 Reconciliation Loop** 

Example of the reconciliation agent (controller) inspecting data, routing to an operation, and the operations (executor) performing conflict resolution. 

```
INFO================================================================================
INFO[PK(’3M’,’2022-12-31’)]Processing:8rows,max5iterations
INFO================================================================================
INFO[PK(’3M’,’2022-12-31’)]Controller(iter=0):Inspection1SQL
INFO[PK(’3M’,’2022-12-31’)]SQL:SELECTrow_id,page_number,operating_income,net_sales,
operating_margin,operating_income_quote,net_sales_quote,operating_margin_quoteFROM
OperatingMarginMetrics_69ffa681ORDERBYrow_id;
INFO[PK(’3M’,’2022-12-31’)]Controller(iter=0):Inspection2SQL
INFO[PK(’3M’,’2022-12-31’)]SQL:SELECTrow_id,page_number,operating_income,net_sales,
operating_margin,operating_income_rationale,net_sales_rationale,operating_margin_rationaleFROM
```

44 

```
OperatingMarginMetrics_69ffa681ORDERBYrow_id;
INFO[PK(’3M’,’2022-12-31’)]Controller(iter=0):Routingto’resolve_conflicts’
INFO[PK(’3M’,’2022-12-31’)]Reasoning:Thereare8rowsforthesamePK(’3M’,’2022-12-31’),allinthe
OperatingMarginMetricstable.Themaindatacolumns(operating_income,net_sales,operating_margin)show
severalrepeatedvaluesfortotalcompany(operating_income=6539,net_sales=34229,
operating_margin~19.1),butalsoseveralrowsforbusinesssegments(SafetyandIndustrial,
TransportationandElectronics,HealthCare,Consumer)withdistinctvalues.Somerowshavecalculated
margins,somehaveexplicitmargins,andsomehavemissingvalues.Thesegmentrowsarenotredundant
withthetotalcompanyrows---theyrepresentdifferentaspects.However,amongthetotalcompanyrows,
thereareclearredundancies(samevalues,slightlydifferentmarginduetorounding/calculation,some
withexplicitmargin,somewithcalculated).Tomovetowardasingleconsolidatedrow,weneedto
resolveconflictswherethesamefieldhasslightlydifferentvalues(e.g.,operating_margin19.10vs
19.11),andselectthebestrepresentativefortotalcompany.Forthesegmentrows,aggregationmaybe
neededlater,butfirst,resolvingconflictsamongtotalcompanyrowswillreduceredundancyandclarify
whichrowsaretrulydistinct.Therefore,thenextstepistoresolveconflicts.
INFO[PK(’3M’,’2022-12-31’)]Executor(op=resolve_conflicts):Inspection1
INFO[PK(’3M’,’2022-12-31’)]Reasoning:BeforegeneratingthemergeSQL,Ineedtoinspectall8rowsfor
thisprimarykeytounderstandthenatureoftheconflictsinoperating_income,net_sales,and
operating_marginfields.Specifically,Ineedtoseethe_quoteand_rationalecolumnsforeach,aswell
astheextractioncontext(page_number,text_header)todeterminewhichrow(s)aremostauthoritative
andalignbestwiththefielddefinitions.Thiswillallowmetoselectthemostreliablevaluesand/or
combinerelevantnarrativedetailsasrequiredbythetask.Sincethereare8rowsandmultipledistinct
valuesforthekeymetrics,afullinspectionisnecessarytoresolveconflictsanddeduplicate
appropriately.
INFO[PK(’3M’,’2022-12-31’)]SQL:SELECT*FROMOperatingMarginMetrics_69ffa681WHEREcompany_name=’3M’
ANDfiscal_year_end=’2022-12-31’ORDERBYrow_idASCLIMIT8
INFO[PK(’3M’,’2022-12-31’)]Executor(op=resolve_conflicts):GeneratedmergeSQL
INFO[PK(’3M’,’2022-12-31’)]Reasoning:The8rowsrepresentbothconsolidatedcompany-levelmetricsand
businesssegmentbreakdownsfor3MforFY2022.Therearetrueconflictsinthemeasuredvaluesfor
operating_income,net_sales,andoperating_margin,butthesearenoterrors---theyreflectdistinct
entities:theconsolidatedcompanyanditssegments(Safety&Industrial,Transportation&Electronics,
HealthCare,Consumer).Theschemaisforcompany-levelmetrics,buttheextractionincludes
segment-levelrows,whicharevaluableforanalysis.However,themergeobjectiveistodeduplicateand
resolveconflictsforthecompany-levelrow,nottopreservesegmentrowsasseparateevents.
Uponinspection,tworows(row_id=1androw_id=10,alsorow_id=13)providetheconsolidated
company-levelmetrics:
-operating_income:6539
-net_sales:34229
-operating_margin:19.1%(explicitinrow_id=1,calculatedinrow_id=10/13)
Row_id=1isextractedfromatablelabeled’TotalCompany’andprovidesexplicitvaluesforallthree
metrics,withdirectquotesandclearrationale.Row_id=10androw_id=13arefromtheConsolidated
StatementofIncomeandBusinessSegmentInformation,alsoauthoritative,buttheoperatingmarginis
calculated,notquoted.Row_id=1bestmatchesthefielddefinitionalignment,extractionlocation
relevance,andsourceauthoritycriteria.
Segmentrows(row_id=2,4,6,8)arenotcompany-levelandshouldnotbemergedintothecompany-level
outputrow.Row_id=0isasummary/narrativerowwithlessdirectmetricextraction.
Therefore,thebestapproachistoselectrow_id=1asthesingleauthoritativecompany-levelrowfor
FY2022,preservingalloriginalcolumns.
INFO[PK(’3M’,’2022-12-31’)]SQL:SELECT*FROMOperatingMarginMetrics_69ffa681WHERErow_id=1
INFO[PK(’3M’,’2022-12-31’)]Executor(op=resolve_conflicts):ExecutingmergeSQL(attempt1/3)
INFO[PK(’3M’,’2022-12-31’)]Executorresult:8rows$\rightarrow$1rows
INFO[PK(’3M’,’2022-12-31’)]Verificationphase:Enteringverification
INFO[PK(’3M’,’2022-12-31’)]Verification:Inspection1/5
INFO[PK(’3M’,’2022-12-31’)]Verification:Reasoning:IneedtoverifythattheSQLtransformationselected
reconciliation_v2.py:1743
onlytheauthoritativecompany-levelrow(row_id=1)fromtheinitial_tableandthatthefinal_table
containsexactlythisrow,withallcolumnspreserved.Todothis,Iwillinspectbothtables:
initial_tabletoconfirmthepresenceandcontentofrow_id=1,andfinal_tabletoensureitcontains
onlyrow_id=1andmatchestheexpectedvaluesandschema.Thisinspectionwillallowmetoconfirmthat
theSQLexecutedtheintendedlogicdescribedinmyreasoning.
INFO[PK(’3M’,’2022-12-31’)]Verification:SQL:SELECT*FROMinitial_tableWHERErow_id=1;
SELECT*FROMfinal_table;
INFO[PK(’3M’,’2022-12-31’)]Verification:[OK]Approvedafter1inspection(s)
INFO[PK(’3M’,’2022-12-31’)]Verification:Iinspectedbothinitial_tableandfinal_table.The
initial_tablecontainsmultiplerowsfortheprimarykey(’3M’,’2022-12-31’),includingboth
company-levelandsegment-levelmetrics.Myreasoningforthemergewastoselectonlytheauthoritative
company-levelrow(row_id=1),whichprovidesexplicit,directlyquotedvaluesforoperating_income,
net_sales,andoperating_margin,extractedfromthe’TotalCompany’table.Thefinal_tablecontains
exactlyonerow,withrow_id=1,andallcolumnsarepreserved.Thevaluesinfinal_tablematchthosein
initial_tableforrow_id=1,confirmingthattheSQLtransformationexecutedtheintendedlogic:
selectingonlytheauthoritativecompany-levelrowandexcludingsegment-levelandsummaryrows.
Therefore,theSQLcorrectlyimplementedmyreasoningandtheintendedtransformation.
INFO[PK(’3M’,’2022-12-31’)]ContextGenerator:Starting
INFO[PK(’3M’,’2022-12-31’)]ContextGenerator:[OK]Generatedcontext
INFO[PK(’3M’,’2022-12-31’)]Reducedto1rows,stopping
INFO[PK(’3M’,’2022-12-31’)]Finalresult=1rowsafter1operations
INFO================================================================================
```

## **E.2 Primary Key Selection** 

Example of the primary key selector using voting rounds to identify the semantic primary key for reconciliation. 

```
================================================================================
```

45 

**==> picture [350 x 334] intentionally omitted <==**

**----- Start of picture text -----**<br>
INFO PHASE 1: PRIMARY KEY SELECTION<br>INFO ================================================================================<br>INFO Using Primary Key Selector Version: v1<br>INFO === Selecting Primary Key for CashFlowStatementItem ===<br>INFO === Running 3 voting rounds for primary key selection ===<br>INFO<br>--- Voting Round 1/3 ---<br>INFO --- Primary Key Analysis Query 1/5 (Run 1) ---<br>INFO [OK] Primary key selected after 0 queries (Run 1)<br>INFO Primary Key: [’entity_name’, ’fiscal_year_end’, ’item_name’]<br>INFO Reasoning: This table records individual cash flow statement line items for a given entity and<br>fiscal year. Each unique observation...<br>INFO Round 1 selected: [’entity_name’, ’fiscal_year_end’, ’item_name’]<br>INFO<br>--- Voting Round 2/3 ---<br>INFO --- Primary Key Analysis Query 1/5 (Run 2) ---<br>INFO [OK] Primary key selected after 0 queries (Run 2)<br>INFO Primary Key: [’entity_name’, ’fiscal_year_end’, ’item_name’]<br>INFO Reasoning: This table records individual cash flow statement line items for a given entity and<br>fiscal year. Each unique observation...<br>INFO Round 2 selected: [’entity_name’, ’fiscal_year_end’, ’item_name’]<br>INFO<br>--- Voting Round 3/3 ---<br>INFO --- Primary Key Analysis Query 1/5 (Run 3) ---<br>INFO [OK] Primary key selected after 0 queries (Run 3)<br>INFO Primary Key: [’entity_name’, ’fiscal_year_end’, ’item_name’]<br>INFO Reasoning: This table records individual cash flow statement line items for a given entity and<br>fiscal year. Each unique observation...<br>INFO Round 3 selected: [’entity_name’, ’fiscal_year_end’, ’item_name’]<br>INFO<br>=== Majority Voting Result ===<br>INFO Winning Primary Key: [’entity_name’, ’fiscal_year_end’, ’item_name’]<br>INFO Votes: 3/3<br>INFO Vote Distribution: {(’entity_name’, ’fiscal_year_end’, ’item_name’): 3}<br>INFO Selected primary key for CashFlowStatementItem: [’entity_name’, ’fiscal_year_end’, ’item_name’]<br>INFO Reasoning: This table records individual cash flow statement line items for a given entity and fiscal<br>year. Each unique observation is defined by the combination of entity_name (which company),<br>fiscal_year_end (which reporting period), and item_name (which line item). Multiple extractions of the<br>same entity, period, and item_name should be consolidated, with item_value being resolved during<br>conflict resolution. All other fields are extraction metadata or rationale and do not define the<br>business key.<br>INFO [OK] Selected Primary Key: [’entity_name’, ’fiscal_year_end’, ’item_name’]<br>INFO Reasoning: This table records individual cash flow statement line items for a given entity and fiscal<br>year. Each unique observation is defined by the combination of entity_name (which company),<br>fiscal_year_end (which reporting period), and item_name (which line item). Multiple extractions of the<br>same entity, period, and item_name should be consolidated, with item_value being resolved during<br>conflict resolution. All other fields are extraction metadata or rationale and do not define the<br>business key.<br>**----- End of picture text -----**<br>


## **E.3 Field Canonicalization** 

Example of canonicalizing artist names to resolve variations (e.g., "Adele" vs "Adele Laurie Blue Adkins"). 

**==> picture [331 x 237] intentionally omitted <==**

**----- Start of picture text -----**<br>
--- Processing Field 2/3: artist_name ---<br>INFO<br>Canonicalization Cycle 1/20<br>INFO Action: canonicalize<br>INFO Reasoning: Phase 3 Step 2: I examined the alphabetically sorted distinct values from<br>Inspection 1:<br>- Adele (6 occurrences): Clean, common artist label.<br>- Adele Laurie Blue Adkins (1 occurrence): Full legal name of Adele.<br>These appear to refer to the same person. According to best practice for primary key<br>canonicalization, we need to check related columns for evidence before merging.<br>First, the artist_name_quote field has only 2 distinct values matching the artist name,<br>indicating these are alternate forms used in the data. Next, the very low cardinality (2 values,<br>both in artist_name-related quote/rationale fields, not free text and clearly person names)<br>confirms that this is a categorical field, not free text or a unique identifier.<br>To ensure deduplication, we should canonicalize both forms to a single canonical form (’ADELE’),<br>applying UPPER for case standardization. The rationale/quote columns simply confirm that "Adele<br>Laurie Blue Adkins" is the legal name of Adele, but all other data refers to the artist as<br>"Adele". Thus, I will canonicalize both to ’ADELE’.<br>INFO Canonicalizing field: artist_name (attempt 1/3)<br>INFO SQL: SELECT row_id,<br>CASE<br>WHEN artist_name IS NULL THEN NULL<br>WHEN UPPER(TRIM(artist_name)) IN (’ADELE’, ’ADELE LAURIE BLUE ADKINS’) THEN ’ADELE’<br>ELSE UPPER(TRIM(artist_name))<br>END AS artist_name,<br>page_number,<br>document_name,<br>text_header,<br>artist_name_quote,<br>artist_name_rationale,<br>wikidata_id_quote,<br>**----- End of picture text -----**<br>


46 

```
wikidata_id_rationale,
wikidata_id,
birth_date_quote,
birth_date_rationale,
birth_date,
birth_place_quote,
birth_place_rationale,
birth_place,
country_of_birth_quote,
country_of_birth_rationale,
country_of_birth,
alternate_names_quote,
alternate_names_rationale,
alternate_names,
debut_date_quote,
debut_date_rationale,
debut_date,
debut_year_quote,
debut_year_rationale,
debut_year,
debut_description_quote,
debut_description_rationale,
debut_description,
industry_quote,
industry_rationale,
industry
FROMartistBiography_Adele
ORDERBYrow_id;
INFO[OK]Successfullyappliedcanonicalization1
```

## **F Evaluation Prompts** 

## **Hard Evaluator** 

```
#instruction
```

```
Evaluatethepredictedansweragainstthegoldanswer.Thepredictedanswershouldmatchthegoldanswer.
```

```
Usethefollowingcriteriatoevaluatethepredictedanswer:
```

- `If there are rounding errors, its incorrect.` 

- `If final framing doesn’t match, its incorrect.` 

- `If the justification doesn’t match, its incorrect.` 

- `If the only issue is gold being in decimal and the predicted answer being in fraction or percentage, its correct.` 

```
#input
#Question
{{question}}
#GoldAnswer
{{gold_answer}}
#PredictedAnswer
{{predicted_answer}}
```

## **Soft Evaluator** 

```
#instruction
```

```
Evaluatethepredictedansweragainstthegoldanswer.Thepredictedanswershouldmatchthegoldanswer.
```

```
#input
#Question
{{question}}
```

```
#GoldAnswer
{{gold_answer}}
```

```
#PredictedAnswer
{{predicted_answer}}
```

## **Loong Evaluator** 

```
WewouldliketorequestyourfeedbackontheperformanceoftheAIassistantinresponsetotheuserquestion
displayedbelowaccordingtothegoldanswer.Pleaseusethefollowinglistedaspectsandtheirdescriptionsas
evaluationcriteria:
```

- `Accuracy and Hallucinations: The assistant’s answer is semantically consistent with the gold answer; The numerical value and order need to be accurate, and there should be no hallucinations.` 

- `- Completeness: Referring to the reference answers, the assistant’s answer should contain all the key points needed to answer the user’s question; further elaboration on these key points can be omitted.` 

- `Please rate whether this answer is suitable for the question. Please note that the gold answer can be considered as a correct answer to the question.` 

- `The assistant receives an overall score on a scale of 1 to 100, where a higher score indicates better overall performance.` 

```
Pleasenotethatiftheassistant’sanswerandthegoldanswerfullymeettheabovecriteria,itsoverallratingshould
bethefullmarks(100).
Pleasefirstprovideacomprehensiveexplanationofyourevaluation,avoidinganypotentialbias.
Then,outputalineindicatingthescoreoftheAssistant.
```

47 

```
PLEASEOUTPUTTHESCOREONASCALEOF1TO100.
#input
#Question
{{question}}
#GoldAnswer
{{gold_answer}}
#PredictedAnswer
{{predicted_answer}}
[Question]
{{question}}
[GoldAnswer]
{{gold_answer}}
[TheStartofAssistant’sPredictedAnswer]
{{predicted_answer}}
[TheEndofAssistant’sPredictedAnswer]
[System]
WewouldliketorequestyourfeedbackontheperformanceoftheAIassistantinresponsetotheuserquestion
displayedaboveaccordingtothegoldanswer.Pleaseusethefollowinglistedaspectsandtheirdescriptionsas
evaluationcriteria:
-AccuracyandHallucinations:Theassistant’sanswerissemanticallyconsistentwiththegoldanswer;The
numericalvalueandorderneedtobeaccurate,andthereshouldbenohallucinations.
-Completeness:Referringtothereferenceanswers,theassistant’sanswershouldcontainallthekeypointsneeded
toanswertheuser’squestion;furtherelaborationonthesekeypointscanbeomitted.
Pleaseratewhetherthisanswerissuitableforthequestion.Pleasenotethatthegoldanswercanbeconsideredasa
correctanswertothequestion.
Theassistantreceivesanoverallscoreonascaleof1to100,whereahigherscoreindicatesbetteroverall
performance.
Pleasenotethatiftheassistant’sanswerandthegoldanswerfullymeettheabovecriteria,itsoverallratingshould
bethefullmarks(100).
Pleasefirstprovideacomprehensiveexplanationofyourevaluation,avoidinganypotentialbias.
Then,outputalineindicatingthescoreoftheAssistant.
```

```
PLEASEOUTPUTWITH:
```

- `"EXPLANATION" IS THE REASON BEHIND THE CORRECT SCORE GIVEN TO THE PREDICTED ANSWER.` 

- `- "CORRECT" IS A SCORE ON A RANGE OF 1 TO 100.` 

## **Numeric Extractor** 

```
#instruction
```

```
Giventhequestionandtheassistant’sfullanswer,extractonlythefinalnumericanswer.Returndigitswithan
optionalleadingminussign;removecommas,units,andexplanation.Ifnonumericanswerispresent,returnnull.
#input
#Question
{{question}}
#PredictedAnswer
{{predicted_answer}}
```

## **Multi-Numeric Extractor** 

- `# instruction` 

- `Given a question, the ground truth answer, and the generated answer, first determine if the answers contain numeric values that should be compared.` 

- `If the answers are purely textual/qualitative (e.g., lists of names, descriptions, yes/no answers without numbers), set has_numbers to False and return empty lists.` 

- `If the answers contain numeric values (counts, percentages, amounts, rankings, etc.), set has_numbers to True and extract all numeric values from both answers.` 

- `For each numeric value, identify:` 

`1. A key/identifier (e.g., country name, industry name, or "value" for single numbers) 2. The numeric value` 

```
Returntwolists:oneforgroundtruthandoneforgeneratedanswer.
```

```
Forsinglenumericanswers,use"value"asthekey.
```

```
Formultiplevalueslikerankingsordistributions,usetheentityname(country,industry,etc.)asthekey.
```

## `SPECIAL HANDLING FOR RANKINGS:` 

- `If rankings have associated counts/values (e.g., "Artist A: 39 songs, Artist B: 25 songs"), extract those counts` 

- `- If rankings are purely ordinal without counts (e.g., "1. Artist A, 2. Artist B, 3. Artist C"), extract the ordinal position as the value:` 

- `- First/1st item $\rightarrow$ value: 1 - Second/2nd item $\rightarrow$ value: 2` 

- `Third/3rd item $\rightarrow$ value: 3, etc.` 

- `- This allows comparison of ranking order even when no explicit counts are given IMPORTANT: Canonicalize the keys to ensure matching between ground truth and generated answer. - Use the same canonical key name for entities that refer to the same thing in both answers - Normalize trivial differences such as: - Capitalization (e.g., "Music" and "music" should use the same key)` 

48 

```
-Abbreviations(e.g.,"USA"and"UnitedStates"shouldusethesamekey)
-Punctuationandformatting(e.g.,"Film/TV"and"Film&TV"shouldusethesamekey)
-Singular/pluralforms(e.g.,"artist"and"artists"shouldusethesamekey)
-Choosetheclearest,moststandardformasthecanonicalkey
-Applythesamecanonicalizationtobothgroundtruthandgeneratedvalues
#input
#Question
{{question}}
#GroundTruthAnswer
{{gold_answer}}
#GeneratedAnswer
{{predicted_answer}}
```

49 

