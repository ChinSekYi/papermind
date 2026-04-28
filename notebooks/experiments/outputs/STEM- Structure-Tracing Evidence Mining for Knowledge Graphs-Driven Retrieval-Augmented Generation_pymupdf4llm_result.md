# **STEM: Structure-Tracing Evidence Mining for Knowledge Graphs-Driven Retrieval-Augmented Generation** 

**Peng Yu, En Xu, Bin Chen**[*] **, Haibiao Chen, Yinfei Xu** AI Product Center, Kingsoft Corporation, Beijing, China {yupeng5,xuen,chenbin,chenhaibiao,xuyinfei}@kingsoft.com 

## **Abstract** 

Knowledge Graph-based Question Answering (KGQA) plays a pivotal role in complex reasoning tasks but remains constrained by two persistent challenges: the structural heterogeneity of Knowledge Graphs (KGs) often leads to semantic mismatch during retrieval, while existing reasoning path retrieval methods lack a global structural perspective. To address these issues, we propose Structure-Tracing Evidence Mining (STEM), a novel framework that reframes multi-hop reasoning as a schema-guided graph search task. First, we design a Semanticto-Structural Projection pipeline that leverages KG structural priors to decompose queries into atomic relational assertions and construct an adaptive query schema graph. Subsequently, we execute globally-aware node anchoring and subgraph retrieval to obtain the final evidence reasoning graph from KG. To more effectively integrate global structural information during the graph construction process, we design a Triple-Dependent GNN (Triple-GNN) to generate a Global Guidance Subgraph (Guidance Graph) that guides the construction. STEM significantly improves both the accuracy and evidence completeness of multi-hop reasoning graph retrieval, and achieves State-of-theArt performance on multiple multi-hop benchmarks. Our source code is available at https: //github.com/PennyYu123/STEM_RAG. 

## **1 Introduction** 

The research and development of large language models (LLMs) have spanned several years (OpenAI, 2023; Anthropic, 2024; Touvron et al., 2023; Chowdhery et al., 2023; Jiang et al., 2023a; Bai et al., 2023), however, LLMs suffer from the issue of hallucination, often leading to inaccuracies in responses to fact-based questions. To address this, Retrieval-Augmented Generation (RAG) was introduced as a promising paradigm to ground 

*Corresponding author. 

**==> picture [219 x 165] intentionally omitted <==**

**----- Start of picture text -----**<br>
Relation  Reasoning<br>Paths Retrieval Paths<br>Query →→ →→ Answer<br>Beam Search +<br>✕<br>Entity<br>Query Answer<br>Pattern  ✕<br>Graph<br>Query Answer<br>Schema<br>Assertions Graph<br>Query Answer<br>Decomposition  Graph<br>Agent Builder<br>Triple-GNN<br>**----- End of picture text -----**<br>


Figure 1: Different KG Retrieval Reasoning Frameworks. 

model outputs in verifiable external knowledge bases(Lewis et al., 2020; Trivedi et al., 2023; Guu et al., 2020; Borgeaud et al., 2022). By leveraging pre-existing knowledge bases, RAG enables LLMs to reference relevant contextual information when generating answers, thereby improving the accuracy and quality of responses. In recent years, knowledge graph-based question answering systems for LLMs have gained significant research attention (Yasunaga et al., 2021; Zhang et al., 2022; He et al., 2024; Edge et al., 2024). These systems utilize structured knowledge graphs to organize relationships among various real-world entities, providing the LLM with a hierarchical and logically clear knowledge network, thereby enabling more precise guidance for generating accurate answers. 

Knowledge graph-based multi-hop reasoning for question answering has also garnered extensive research attention (Yasunaga et al., 2021; Luo et al., 2024; Yu et al., 2023; Sui et al., 2025; Liu et al., 2026; Cai et al., 2025). Existing approaches for KG-based reasoning generally fall into three main paradigms: The first involves generating reasoning plans based on the query to retrieve evidence chains 

for answer generation (RoG) (Luo et al., 2024). The second employs step-wise path exploration, such as beam search, utilizing intelligent modules like LLMs to iteratively determine the optimal path (Path Decode) (Sui et al., 2025). The third focuses on structural matching, which involves constructing an aligned schema graph to guide step-by-step searches within the KG to build a minimal distance subgraph (Path Matching) (Cai et al., 2025). The retrieved subgraph is subsequently verbalized and fed into an LLM for final answer generation. An overview of the design of the above method is detailed in Figure 1. 

Despite the extensive research in multi-hop reasoning over Knowledge Graphs (KGs), existing methods still face significant impediments that limit their efficacy and robustness. We identify three primary challenges as follows: 

**First, the semantic-structural gap between LLM-generated plans and KG schemas hinders accurate retrieval.** KGs are characterized by inherent organizational complexity, containing millions of entities and diverse relation types. Current approaches typically rely on LLMs to decompose questions or generate reasoning plans in natural language. However, due to the LLM’s lack of prior knowledge regarding the specific KG schema, these generated plans often suffer from **schema hallucination** —predicting relations that are semantically plausible but topologically nonexistent in the target KG. For instance, consider the query “Which airport to fly into rome?” While the ground-truth path in the KG follows an hierarchilocation.location.nearby_airports cal structure like “Rome _−−−−−−−−−−−−−−−−−→_ Ciampino–G. B. Pastine International Airport”, but a schema-agnostic planner is likely to generate a forward-looking formulation such as “[ENT1] fly into _−−−−→_ Rome”. This creates a fundamental topological mismatch: the semantic implication of “fly into” often fails to align with the knowledge schema, resulting in retrieval failures. 

**Second, prevailing path-search paradigms suffer from lack of global view and evidence fragmentation.** Most existing methods employ stepwise search strategies, which rely on local semantic similarity or LLM-based decision-making to select the next hop. While some incorporate look-ahead scoring to anticipate future steps, they fundamentally lack a global structural blueprint. This leads to three critical issues: (1) Path Deviation: Without global guidance, the search process is 

highly sensitive to local spurious semantic correlations. (2)Hub Node Problem: The sheer volume of neighboring relations creates an overwhelming candidate space, which significantly increases the error rate of selection. This necessitates a schema graph for effective search space pruning. (3) Information Fragmentation: Since complex questions often require a subgraph structure rather than isolated reasoning paths, path-based methods frequently retrieve fragmented evidence, consequently the question cannot be adequately answered. 

**Finally, the reliance on interactive reasoning incurs prohibitive computational costs.** Many state-of-the-art methods adopt an “interleaved” approach, where the LLM is invoked at every step of the reasoning path to judge validity or refine the search direction, these approaches create a significant bottleneck in inference latency and resource consumption. This bottleneck is particularly exacerbated in multi-answer scenarios, where retrieving multiple answers necessitates traversing parallel reasoning paths, thereby leading to substantially higher LLM overhead. 

To address these challenges, we propose Structure-Tracing Evidence Mining (STEM), a novel architecture that reframes multi-hop KG reasoning from sequential path finding to holistic schema-guided subgraph matching. STEM distinguishes itself through two key innovations: First, we bridge the schema gap via building a Semanticto-Structural Projection pipeline. Different from existing works attempting to convert queries into logic forms (e.g., SPARQL) for KG retrieval (Sun et al., 2020; Lan and Jiang, 2020; Ye et al., 2022), we design two specialized modules: the SchemaGrounded Decomposition Agent (SGDA) and the Symbol-Aligned Graph Builder (SAGB) to ensure the retrieval plan aligns with the KG’s inherent topology. The SGDA first decomposes the complex query into a sequence of atomic relational assertions, stripping away linguistic ambiguity and align the semantic narratives with the logic of the KG. Subsequently, the SAGB grounds each assertion into a concrete triple structure, assembling them into a coherent schema graph. 

Second, we implement Structure-Tracing Subgraph Retrieval to retrieve related evidence. STEM employs a Triple-Dependent GNN (Triple-GNN) to produce a Global Guidance Subgraph (Guidance Graph). During the traversal phase, the retrieval is guided not only by local transition probabilities (semantic distance) but also by the global structural 

scores according to Guidance Graph. This mechanism ensures that search step is globally weighted, effectively correcting potential deviations and ensuring the retrieval of a complete, logically connected evidence subgraph. Based on the comprehensive introduction, the main contributions of this paper are summarized as follows: 

- We propose a novel Semantic-to-Structural Projection pipeline to bridge the inherent gap between natural language queries and KG schemas. By employing a Schema-Grounded Decomposition Agent and a Symbol-Aligned Graph Builder, we construct a precise schema graph that serves as a topological blueprint for subsequent retrieval. 

- We introduce Structure-Tracing Evidence Mining, a holistic subgraph matching paradigm for KGQA. By leveraging a TripleDependent GNN to construct a Global Guidance Subgraph, our method incorporates global structural priors into the search process, effectively ensuring both the accuracy and completeness of the retrieved evidence. 

- Our proposed method achieves State-of-theArt performance on complex multi-hop reasoning benchmarks including WebQSP and CWQ, demonstrating the effectiveness of our proposed method. 

## **2 Preliminaries** 

Typically, the structure of a knowledge graph can be defined as a tuple: _G_ = ( _V, E, T , N , R_ ), where _V_ represents the set of nodes, _E_ denotes the set of edges, _T_ = _{_ ( _h, r, t_ ) _| h, t ∈V, r ∈E}_ represents the set of triplets. _N_ corresponds to the textual descriptions of each entity node in _V_ , i.e., for each node _vi ∈V_ , there exists _ei ∈N_ representing the textual description of _vi_ . Likewise _R_ denotes the set of relations, corresponding to the descriptions of edge in _E_ , i.e., for each edge _di ∈E_ , there exists _ri ∈R_ representing the textual description of _di_ . 

The primary objective of KGQA framework is to identify and extract a relevant subgraph _G[′] ⊆G_ from the entire knowledge graph based on a given natural language query _Q_ , this retrieved subgraph serves as structured evidence to ground the reasoning process of a Large Language Model. We pre-index all entities mapped to nodes in graph _G_ within a high-speed indexing system for the rapid retrieval. 

## **3 Methodology** 

We present our methodology as follows. First, we describe how the SGDA module performs question decomposition. Next, we explain how the SAGB module builds a schema graph. We then introduce the Triple-GNN for Guidance Graph building. Subsequently, we present the complete workflow of Structure-Tracing Subgraph Retrieval. Finally, we introduce the answer generation process. The overall design framework of the approach in this paper is shown in Figure 2. 

## **3.1 Schema-Aligned Question Decomposition** 

The objective of the SGDA is to generate atomic relational assertions from the original query, these assertions will align the semantic narratives with the relations in KG. CoG (Zhao et al., 2025) proposed a “knowledge reciting” task, training an LLM with query-path pairs from datasets to equip with prior knowledge of paths, preventing plausible but irretrievable paths in the KG. Inspired by this, and considering the scale and complexity of nodes and paths in KGs, our SGDA module focuses on learning patterns rather than specific knowledge. For instance, given the training example (“What is the San Francisco Giants’ mascot?”, “San Francisco Giants’ mascot is [ENT1]”), the SGDA can construct assertions following the same pattern for similar queries, such as “What is the X’s mascot?”, successfully resolving cases associated with this type of problem. 

**Atomic Relational Assertion.** An atomic relational assertion refers to a text describing a single logical relationship—for instance, “Western Sahara contains Smara”—which can be directly mapped to a single triple (Western Sahara, contains, Smara). Formally, Given the prompt _P_ 1 and the multi-hop reasoning question _Q_ , SGDA decompose the original query into a structured sequence _S_ , where each element is represented as an assertion _s_ , the form is described below: 

**==> picture [158 x 12] intentionally omitted <==**

We present the specific content of _P_ 1 and all subsequent prompts in the Appendix G. To maintain logical connectivity across multi-hop reasoning steps, the SGDA employs a unified placeholder mechanism for entity linking. Since the answer to a preceding sub-question often serves as the bridging entity for the subsequent one, we standardize these intermediate variables using shared identifiers (e.g., 

**==> picture [455 x 276] intentionally omitted <==**

**----- Start of picture text -----**<br>
Musician<br>Decomposition<br>Agent Graph  Jeremy<br>Builder All Bad Bieber profession<br>justin bieber’s  Justin Bieber ENT1 ENT1 album parents children<br>What is the name of justin bieber’s brother？ children is [ENT2]parents is [ENT1] [ENT1]’s  ENT1 Triples parents children ENT2 Justin Bieber parents children ENT2 Triple-GNN artist BieberJustin  BieberJaxon  gender Male<br>Precision<br>Boyfriend<br>Assertions &<br>Top<br>Strategy<br>0.84 0.25<br>0.14<br>0.73 Triple Bias Entity Bias<br>0.96 0.81 Sim( T ， T ’ ) T ’ E ’ Sim( E ， E ’ )<br>0.16 T<br>LLM Triple Scoring  Entity Scoring<br>Jeremy Bieber Function with Bias Function with Bias<br>Instruction ：…<br>Jaxon  Reasoning Graph:  … Textualization<br>Bieber Query: ... parents children<br>Justin  Jaxon  T ’ E ’ E<br>Bieber Bieber Justin Bieber<br>**----- End of picture text -----**<br>


Figure 2: Overview of the STEM Framework. 

[ENTX]). For instance, given the multi-hop query “Where is the arena stadium of the team whose mascot is Clutch the Bear?”, the SGDA decomposes it into a coherent sequence of assertions sharing the bridging entity [ENT1]: 

**==> picture [158 x 26] intentionally omitted <==**

**Answer Strategy.** We consider multi-answer scenarios. For instance, the question “what are the four main languages spoken in Spain” corresponds to multiple potential answers, retrieving only a single evidence graph will result in incomplete answers. To address this, we categorize questions into two types: _Precision_ and _Breadth_ . The former corresponds to a definitive answer, while the latter involves multiple valid answers. The distinction between these two types will influence the behavior of subgraph matching, as detailed in subsequent sections. Since a single query may correspond to diverse potential KG logical structures, we employ beam search to enable the SGDA to generate _B_ multiple candidate results, finally constructing a list of candidate assertion-strategy pairs ( _SQ, σ_ ) corresponding to different planning results, _σ_ signifies the retrieval strategy assigned to _Q_ ,where _σ ∈{_ “ _Precision_ ” _,_ “ _Breadth_ ” _}_ . 

## **3.2 Symbol-Aligned Graph Construction** 

While the SGDA successfully decomposes complex queries into atomic relational assertions, the objective of the SAGB is to perform symbolic grounding: mapping these textual assertions into precise structural triples. For instance, consider the assertion: “Darryl Sutter’s hockey position is [ENT1].”, A naive keyword match might fail to identify the correct edge due to lexical divergence (e.g., (“Darryl Sutter”, “position”, “[ENT1]”)). However, possessing prior knowledge of the KG’s symbolic representation, the SAGB accurately grounds this assertion into the standardized triple: (“Darryl Sutter”, “ice_hockey.hockey_player.hockey_position”, “[ENT1]”). 

Formally, for a given set of assertions and the prompt _P_ 2 , denoted as _S_ = _{s_ 1 _, s_ 2 _, , ..., sN }_ , SAGB build a corresponding set of structural triples, represented as 

**==> picture [157 x 12] intentionally omitted <==**

After obtaining the set of triples _T_ , we construct the schema graph _Gsch_ by traversing each member of _T_ . 

To develop these two modules, we propose a specialized data construction and training framework. 

Furthermore, to enhance the KG knowledge injection and generalization capabilities of the SGDA and SAGB, we introduce Structure-to-Query Reverse Generation method for data augmentation. All details will be described in Appendix C. 

In Appendix F, we provide concrete running examples to illustrate the end-to-end processing workflow of the Semantic-to-Structural Projection pipeline and the underlying principles of pattern acquisition during training. 

## **3.3 Global Guidance Subgraph** 

Graph Neural Networks (GNNs) have been widely adopted for reasoning over KGs (Mavromatis and Karypis, 2025; Yasunaga et al., 2021; Liu et al., 2026). GFM-RAG (Luo et al., 2025) typically employs a Query-Dependent GNN, which incorporates query information to compute interactionaware representations, enabling the ability to capture relevant entity knowledge within the graph structure. Based on this approach, we propose the Triple-Dependent GNN (Triple-GNN), which leverages the explicit structural relationship inherent in triples. 

Formally, let _G_ denote the query-specific subgraph[1] corresponding to _Q_ , which serves as the knowledge graph for retrieval. _N_ and _R_ denote the set of entities and relations within _G_ , respectively, and _NQ_ denotes the set of all question entities in _Q_ . After obtaining the structural triples _T_ and schema graph _G_ sch. For each triple _t ∈T_ , we first employ a Pretrained Embedding Model (PEM) to generate its vector representation **E** _t ∈_ R _[d]_[GNN] . A pooling operation is then applied to integrate all **E** _ti_ ( _ti ∈T_ ) into a unified representation **E** _Q ∈_ R _[d]_[GNN] . In the message passing stage, **E** _Q_ is fed into the _L_ -layer Triple-GNN to derive the embedding representation **h** _[L] e_[for each entity] _[ e][∈N][Q]_[.][We consolidate] all entity representations into a single representation **H** _[L] Q[∈]_[R] _[|N|×][d]_[GNN][.][The][overall][computation] process is described as follows: 

**==> picture [179 x 33] intentionally omitted <==**

**==> picture [180 x 15] intentionally omitted <==**

**H**[0] _r[∈]_[R] _[|R|×][d]_[GNN][denotes the initialized feature rep-] resentations of relations in _R_ . **H**[0] _e[∈]_[R] _[|N|×][d]_[GNN] 

1We utilize the subgraph structures extracted by RoG (Luo et al., 2024). In their publicly available dataset, each query corresponds to a specific KG and a list of question entities, all of which are derived from Freebase. 

represents the initial feature input of entities, for any **h**[0] _ei[∈]_ **[H]** _e_[0][,][its][initialization][is][defined][as][fol-] lows: 

**==> picture [168 x 34] intentionally omitted <==**

Further details regarding the initialization of the entity and relation inputs ( **H**[0] _e_[and] **[H]**[0] _r_[),][as][well] as the subsequent computations within the TripleGNN, can be found in Appendix A. 

After obtaining entity embeddings **H** _[L] Q_[via] Triple-GNN, we pass them through a linear projection layer, followed by a Sigmoid operation to derive the node probability distribution[2] : 

**==> picture [197 x 16] intentionally omitted <==**

Based on the probability distribution of entities, we select the Top- _K_ entities with the highest probabilities and construct a candidate entity list _NQ′_[:] 

**==> picture [167 x 16] intentionally omitted <==**

where the value of _K_ is set to _| T | ∗_ 4, where _| T |_ is the number of triples in _T_ . 

Upon obtaining the candidate entity list _NQ′_[, we] anchor each entity _e′i[∈N] Q′_[within] _[ G]_[and construct] a subgraph connected by their existing relations _R_ , yielding the final Global Guidance Subgraph denoted as _G ′_[The training details of Triple-GNN] _Q_[.] will be described in Appendix C. 

## **3.4 Structure-Tracing Subgraph Retrieval** 

In this section, we introduce the Structure-Tracing Subgraph Retrieval module, the primary objective of this module is to identify a specific subgraph within _G_ that exhibits high structural and semantic isomorphism[3] (Cai et al., 2025) to the query schema graph _G_ sch. For each question entity ˆ _e ∈NQ_ , we first retrieve the Top- _N_ ( _N_ = 50) most similar entity nodes _Re_ ˆ from _G_ , along with their corresponding cosine similarity scores _**S** e_ ˆ: 

**==> picture [182 x 29] intentionally omitted <==**

> 2The subscript _ϕx_ indicates that this module contains trainable parameters proposed in this paper. 

> 3Following the approach of SimGRAG, we say that _P_ and _S_ are isomorphic if there exists a bijective mapping _f_ : _VP → VS_ such that an edge _⟨u, v⟩_ exists in _P_ if and only if the edge _⟨f_ ( _u_ ) _, f_ ( _v_ ) _⟩_ exists in _S_ . 

**Global Structural Consistency Bias.** Since **Sim** is a shallow entity-to-entity semantic similarity function, we design a global-prior score rectification mechanism by incorporating the structural priors from Guidance Graph _GQ′_[.][Specifically, for] the aforementioned score _**S** e_ ˆ, the score rectification is applied as follows: 

**==> picture [204 x 28] intentionally omitted <==**

where IEnt is **Entity-level Global Structural Consistency Bias** , and IEnt is defined as follows: 

**==> picture [171 x 34] intentionally omitted <==**

This means that if an entity _e_ exists in the Global Guidance Subgraph _GQ′_[, it is considered more im-] portant for the query reasoning structure. 

To initiate the subgraph matching search, we first establish a starting anchor mapping between the schema graph _G_ sch and the knowledge graph _G_ . Specifically, for the question entity ˆ _e_ , we identify its counterpart node _e[∗]_ in _G_ by selecting the highest-scoring candidate from _Ne_ ˆ based on _**S** e[∗]_ ˆ[.] Simultaneously, we locate the corresponding node _e_ within _G_ sch corresponding to the entity ˆ _e_ via fuzzy string matching. This establishes the starting pair ( _e_ , _e[∗]_ ). Proceeding from this anchor, we execute the structure-tracing matching: for a specific edge ( _e_ , _r_ , _e[′]_ )[4] defined in _G_ sch, we seek a structurally and semantically matching edge ( _e[∗]_ , _r[∗]_ , _e[∗][′]_ ) within _G_ . This matching process is guided by calculating the globally-aware triple score T-Score defined as follows: 

**==> picture [148 x 14] intentionally omitted <==**

**==> picture [199 x 33] intentionally omitted <==**

where Sim( _ti, tj_ ) = _∥_ **EE** _titi∥∥·_ **EE** _tjtj ∥_[.][Similar to entity] nodes retrieval, we incorporate a structural constraint based on the Global Guidance Subgraph _GQ′_ by integrating with a **Triple-level Global Structural Consistency Bias** ITri. The definition of ITri 

> 4 **Note:** For brevity and to explicitly delineate the constituent nodes of each edge, we uniformly use ( _e_ , _r_ , _e[′]_ ) to denote both the triple _t_ and the edge corresponding to relation _r_ . 

denotes an Triple-level structural constraint, defined as follows: 

**==> picture [170 x 33] intentionally omitted <==**

The above describes the method for obtaining edge mappings through single-step reasoning using T-Score. For the entire graph reasoning process, recursive matching is performed until a concrete subgraph _G_ sch _[∗]_[that is structurally isomorphic to] _[ G]_[sch][.] Specifically, at step _i_ of the matching process, the cumulative score being _Si_ , the score for the next step _Si_ +1 is computed as follows: 

**==> picture [184 x 28] intentionally omitted <==**

It is important to note that although the KG is directed, the matching process operates in an undirected context. In other words, we do not distinguish between incoming and outgoing edges in the matching selection. We will provide a detailed description of this algorithm in the Appendix H. 

## **3.5 Retrieval Behaviors of Different Strategies** 

As discussed in Section 3.1, multi-hop questions can be categorized into _Precision_ and _Breadth_ types. To accommodate these distinct reasoning requirements, STEM adopts an adaptive search strategy that dynamically adjusts the edge selection behavior during the Structure-Tracing Subgraph Retrieval process. Specifically, for _Precision_ strategy, we employ a Greedy Selection mechanism by selecting strictly the edge with the maximum score _S_ . For _Breadth_ strategy, we employ a ThresholdBased Selection mechanism by retaining all candidate edges whose scores _S_ exceed a pre-defined confidence threshold _θ_ . In fact, Threshold-Based Selection allows the structure tracing to branch out, transforming the linear search path into a search tree. 

We present detailed illustrations of both selection strategies and experiments to analyze the their impact on performance in the Appendix D.1. 

## **3.6 Generation** 

Upon completing the subgraph retrieval, we obtain a query-specific evidence subgraph _G_ reason by integrating the resulting subgraphs from all search processes. To linearize this graph structure into a format compatible with LLM prompting, we perform Depth-First Search starting from the question 

|**Model**|**WebQSP**|**CWQ**|
|---|---|---|
||Hit@1 _F_1 Score|Hit@1 _F_1 Score|
||**GPT-4o**||
|**GPT-4o**<br>**GPT-4o**+Fewshot<br>**GPT-4o**+CoT|61.8<br>43.6<br>71.68<br>63.7<br>74.12<br>64.25|38.2<br>32.9<br>57.59<br>44.72<br>59.36<br>48.24|
|**With Finetuning**|||
|**NSM**<br>**DeCAF**_F iD−_3_B_<br>**KD-CoT**_T_5_−large_<br>**RoG**_Llama_2_−Chat−_7_B_<br>**RoG**_Llama−_3_._1_−_70_B_<br>**RoG**_GP T −_4_o_<br>**LightProf**_Llama_3_−_8_B_<br>**GRAG**_LLaMA_2_−_7_B_<br>**GNN-RAG**_Llama_2_−_7_B_|74.31<br>-<br>82.1<br>-<br>73.7<br>50.2<br>83.15<br>69.81<br>86.1<br>68.87<br>88.09<br>70.12<br>83.8<br>-<br>72.75<br>50.41<br>86.4<br>69.0|53.92<br>-<br>70.42<br>-<br>50.5<br>-<br>61.39<br>56.17<br>67.43<br>60.3<br>69.61<br>61.97<br>59.3<br>-<br>-<br>-<br>67.3<br>59.1|
|**With Prompting**|||
|**Kaping**_gpt−_3_._5_−turbo_<br>**ToG**_gpt−_3_._5_−turbo_<br>**G-Ret**_Llama_2_−_7_B_<br>**PoG**_gpt−_3_._5<br>**ReKnoS**_gpt−_3_._5<br>**MFC**_gpt−_4_o−mini_<br>**SubgraphRAG**_gpt−_4_o_<br>**FiDeLiS**_gpt−_4_−turbo_<br>**ProgRAG**_gpt−_4_o−mini_|72.42<br>65.12<br>75.08<br>72.32<br>70.16<br>50.23<br>82.0<br>-<br>81.1<br>-<br>78.9<br>-<br>83.1<br>-<br>84.39<br>**78.32**<br>90.4<br>-|53.42<br>50.32<br>57.59<br>56.64<br>-<br>-<br>63.2<br>-<br>58.5<br>-<br>62.8<br>-<br>56.3<br>-<br>71.47<br>64.32<br>73.3<br>-|
|**Our Proposed Method**|||
|**STEM**_Llama−_3_._1_−_8_B_<br>**STEM**_Llama−_3_._1_−_70_B_<br>**STEM**_GP T −_4_o_|86.63<br>71.05<br>88.08<br>74.62<br>**90.94**<br>76.18|68.76<br>60.81<br>72.53<br>62.09<br>**74.09**<br>**65.33**|



Table 1: Comparison of different models on WebQSP and CWQ datasets. 

entity nodes within _G_ reason to flatten the subgraph into a set of coherent reasoning chains, finally obtain _C_ reason. We apply prompt _P_ 3 as an LLM instruction and take _P_ 3, _Q_ and _C_ reason as input, to infer the final answer: 

**==> picture [170 x 11] intentionally omitted <==**

## **4 Experiments** 

In this section, we present the experimental results and analysis. We conduct the experimental process from the following aspects: (1) How does STEM perform on multi-hop reasoning tasks compared to existing classical and state-of-the-art methods? (2) A fine-grained performance analysis across varying answer numbers, reasoning depths, and underlying reasoning models. (3) Ablation studies on the Semantic-to-Structural Projection pipeline and the Global Structural Consistency Bias. Detailed implementation regarding the training of SGDA, SAGB, and Triple-GNN—including data construc- 

tion processes and training configurations—is provided in Appendix C. 

Furthermore, we will describe the details of the experimental setup, test datasets, evaluation metrics, baselines, and other experiments and analyses in the Appendix B and Appendix D. Additionally, a systematic analysis of failure modes and error propagation across the STEM pipeline is detailed in Appendix E. 

## **4.1 Main Results** 

As described in Table 1, the comparative experimental results demonstrates that STEM achieves a significant performance improvement over other models across both datasets. First, compared to methods relying solely on GPT-4o (OpenAI, 2024), STEM exhibits an increase of over 10% in both Hit@1 and F1 Score on both datasets. When compared to fine-tuned models, STEM + Llama-3.1-8B outperforms other baselines with similar parameter scales, including RoG, LightProf and GNN-RAG. The lead is particularly notable on CWQ, where Hit@1 improves by approximately 6% compared to RoG. Remarkably, it even surpasses the RoG model utilizing the larger Llama-3.1-70B backbone. Meanwhile, STEM + Llama-3.1-70B achieves even greater margins, boosting the F1 score on WebQSP by about 6% and Hit@1 on CWQ by 5% compared to RoG + Llama-3.1-70B. When compared with prompting-based methods, STEM + GPT-4o demonstrates a significant advantage over other approaches utilizing the GPT series. Specifically, on WebQSP, it improves Hit@1 by approximately 7% over the highly competitive baseline, FiDeLiS, although its F1 score is slightly lower. On CWQ, STEM yields distinct improvements across all metrics. Ultimately, STEM + GPT-4o achieves SOTA performance on three out of the four evaluated metrics, with the exception of the F1 score on WebQSP. 

## **4.2 Performance Analysis** 

We partition the test set based on the number of answers and evaluate performance on each subset, with the results shown in Table 2. It is evident that STEM significantly outperforms both RoG and GNN-RAG across all answer count categories. Notably, STEM achieves an improvement of approximately 4% on the WebQSP subset with answers _≥_ 10 and a 9% gain on the CWQ subset with answers in [2, 4]. These results demonstrate STEM’s superior performance in ensuring comprehensive coverage for multi-answer queries. 

|**Method**|**Dataset**|**Ans**= 1 **Ans**_∈_[2,4] **Ans**_∈_[5,9] **Ans** _≥_10|
|---|---|---|
|RoG|**WebQSP**<br>**CWQ**|67.89<br>79.39<br>75.04<br>58.33<br>56.90<br>53.73<br>58.36<br>43.62|
|GNN-RAG|**WebQSP**<br>**CWQ**|71.24<br>76.30<br>74.06<br>56.28<br>60.40<br>55.52<br>61.49<br>50.08|
|STEM+GPT-4o|**WebQSP**<br>**CWQ**|75.26<br>81.87<br>78.38<br>62.46<br>65.32<br>64.35<br>66.37<br>53.86|



Table 2: Detailed results (F1) grouped by the number of answers. 

We stratify the performance by reasoning hop number, with the results presented in Table 3. It is evident that STEM generally maintains a strong competitive edge. However, a notable exception is observed on the CWQ (hop=2), where our method lags significantly behind GNN-RAG. 

We conducted a comparative analysis of RoG and STEM using different reasoning models while keeping the retrieval pipeline constant, the results are presented in Table 4. As the results demonstrate, although replacing the original LLaMA2Chat-7B used in RoG with Llama-3.1-70B-Ins or GPT-4o yields performance gains, STEM maintains a competitive edge under identical reasoning model configurations. 

However, we acknowledge a potential ambiguity in this analysis: the performance improvements might stem partially from the superior parametric knowledge of these advanced models, rather than solely from enhancements in reasoning capability. To investigate this further, we conducted an additional experiment to assess the coverage rate of evidence subgraph retrieval. Specifically, we calculated the proportion of the ground-truth reasoning path covered by the evidence subgraph obtained via Structure-Tracing Subgraph Retrieval. Specifically, given a question _Q_ , let _R_ denote the ground-truth reasoning path of _Q_ , and _G_ reason denote the evidence subgraph. Coverage rate is defined as: 

**==> picture [181 x 27] intentionally omitted <==**

The results are presented in Table 5, indicating that the coverage rate of the evidence subgraph gradually decreases as the number of answers increases, yet it remains at a relatively high level. Furthermore, the coverage on CWQ is consistently lower than on WebQSP due to the question complexity. 

## **4.3 Ablation study** 

**Semantic-to-Structural Projection Pipeline.** We conduct ablation studies by comparing our pro- 

|**Method**|**Dataset**|**Hop**= 1 **Hop**= 2 **Hop** _≥_3|
|---|---|---|
|RoG|**WebQSP**<br>**CWQ**|77.03<br>64.86<br>-<br>62.88<br>58.46<br>37.82|
|GNN-RAG|**WebQSP**<br>**CWQ**|72.0<br>69.8<br>-<br>47.4<br>69.4<br>51.8|
|STEM+GPT-4o|**WebQSP**<br>**CWQ**|81.49<br>75.35<br>-<br>67.46<br>66.73<br>52.15|



Table 3: Detailed results (F1) grouped by the maximum reasoning hop number. 

|**Method**|**Reasoning Model**|**WebQSP**|**CWQ**|
|---|---|---|---|
|||Hit@1 _F_1 Score|Hit@1 _F_1 Score|
|RoG|LLaMA2-Chat-7B<br>Llama-3.1-70B-Ins<br>GPT-4o|83.15<br>69.81<br>86.1<br>68.87<br>88.09<br>70.12|61.39<br>56.17<br>67.43<br>60.3<br>69.61<br>61.97|
|STEM|Llama-3.1-70B-Ins<br>GPT-4o|88.08<br>74.62<br>90.94<br>76.18|72.53<br>62.09<br>74.09<br>65.33|



Table 4: Impact of reasoning models on performance. 

posed Semantic-to-Structural Projection pipeline against powerful off-the-shelf LLMs to evaluate its effectiveness[5] . The comparative results are presented in Table 6. Our method demonstrates a significant performance advantage over the baseline models, surpassing the strongest competitor by over 23% on the CWQ dataset. Among the baselines, GPT-4o consistently outperforms Llama3.1-70B-Ins, reflecting its superior few-shot KG alignment. Our results validate the critical role of logic-aware projection in KG reasoning. 

**Global Structural Consistency Bias.** As the Global Guidance Subgraph serves as a critical structural prior, we conducted an ablation study by selectively removing its bias terms. We evaluated three variants: **w/o Entity Bias** (removing the calculation of the indicator term IEnt in Equation 10), **w/o Triple Bias** (removing the calculation of ITri from Equation 12), and **w/o Both** . 

The ablation results are presented in Table 7. We observe that incorporating both Entity-level and Triple-level Biases yields significant performance gains. Conversely, removing Triple-level Bias leads to a marked decline, particularly in the Hit@1 metric on WebQSP and across all metrics on CWQ, with performance drops reaching up to 10% on CWQ. Removing both biases causes further degradation, with the Hit@1 score on CWQ dropping by an additional 3% compared to the w/o Entity Bias setting, while the F1 score on 

5For baseline LLMs, we implement the pipeline using a few-shot prompting approach, adopting the identical prompt templates employed by the SGDA and SAGB. Regarding the answer strategy, if a valid strategy cannot be successfully extracted after 5 retries, we default to _Precision_ for the current query. 

|**Dataset**|**Ans**= 1 **Ans**_∈_[2,4] **Ans**_∈_[5,9] **Ans** _≥_10|
|---|---|
|**WebQSP**|81.9<br>76.64<br>71.45<br>58.23|
|**CWQ**|74.28<br>66.57<br>62.71<br>51.89|



Table 5: Coverage Rate (%) of ground-truth reasoning paths within evidence subgraphs. 

|**Pipelines**<br>**WebQSP**<br>**CWQ**<br>Hit@1 _F_1 Score Hit@1 _F_1 Score|**Pipelines**<br>**WebQSP**<br>**CWQ**<br>Hit@1 _F_1 Score Hit@1 _F_1 Score|**Pipelines**<br>**WebQSP**<br>**CWQ**<br>Hit@1 _F_1 Score Hit@1 _F_1 Score|
|---|---|---|
||Hit@1 _F_1 Score|Hit@1 _F_1 Score|
|**Llama-3.1-70B-Ins** <br>**GPT-4o**<br>**Our Pipeline**|77.74<br>61.21<br>83.14<br>65.77<br>90.94<br>76.18|46.68<br>41.83<br>50.43<br>43.2<br>74.09<br>65.33|



Table 6: Comparison of different Question Planing Pipelines. 

WebQSP plummets by nearly 5%. Furthermore, the w/o Triple-level setting yields inferior performance compared to the w/o Entity-level, indicating that the Triple-level bias plays a more critical role. These results revealing the limitations of relying solely on local semantic matching. Appendix D.3 details its error-correction analysis. 

We conducted a comprehensive set of additional experiments that are equally critical to validating the robustness of STEM; however, due to space constraints, these results are detailed in Appendix D. The supplementary evaluations encompass finegrained performance analyses and sensitivity studies on key hyperparameters, including the Answer Strategy _σ_ , Initial Entity Count _K_ , and SGDA Beam Size _B_ , as well as parameter tuning for the Global Structural Consistency Bias IEnt and ITri. Furthermore, we provide in-depth Case Studies, Efficiency Analysis, and Interpretability Analysis. 

## **5 Related Work** 

## **5.1 Retrieval-Augmented Generation (RAG)** 

Retrieval-Augmented Generation (RAG) has emerged as a dominant paradigm to mitigate the hallucination issues of Large Language Models (LLMs) (Lewis et al., 2020; Guu et al., 2020; Karpukhin et al., 2020; Izacard et al., 2022; Asai et al., 2024). Adaptive-RAG (Jeong et al., 2024) dynamically selects retrieval strategies based on query complexity. Similarly, Corrective RAG (CRAG) (Yan et al., 2024) incorporates a lightweight evaluator to filter irrelevant documents and trigger web searches as a fallback. 

## **5.2 Multi-hop Reasoning in RAG** 

For multi-step deduction, research has evolved from single-step retrieval to Iterative and Chain- 

|**Scoring Bias**|**WebQSP**|**CWQ**|
|---|---|---|
||Hit@1 _F_1 Score|Hit@1 _F_1 Score|
|**STEM**_GP T −_4_o_<br>**w/o**I**Ent** &I**Tri**<br>**w/o**I**Ent**<br>**w/o**I**Tri**|90.94<br>76.18<br>86.31<br>70.80<br>86.45<br>75.81<br>86.95<br>73.45|74.09<br>65.33<br>63.91<br>55.59<br>66.35<br>57.35<br>64.90<br>56.42|



Table 7: Ablation Study on Entity-level and Triple-level Scoring Biases. 

of-Thought (CoT) Retrieval (Trivedi et al., 2023). ReAct (Yao et al., 2023) further generalizes this by modeling LLMs as agents that can perform search actions. Chain-of-Note (Yu et al., 2024) generates sequential reading notes to evaluate document relevance before aggregation. Demonstrate-SearchPredict (DSP) (Khattab et al., 2022) uses frozen LMs to orchestrate sophisticated retrieval pipelines via natural language programs. Tree of Clarifications (Kim et al., 2023) constructs a tree of disambiguations to handle ambiguous questions recursively. 

## **5.3 Knowledge Graph-based RAG** 

Knowledge Graphs offer a promising solution to the reasoning disconnection problem to encode graph structures (Yasunaga et al., 2021; Zhang et al., 2022), but often struggled to scale or integrate flexibly with LLMs. GraphRAG (Edge et al., 2024) introduces a community-detection-based approach to generate hierarchical summaries of the graph for global query understanding. For multi-hop reasoning, GNN-RAG (Mavromatis and Karypis, 2025) combines GNN-based retrieval with LLM reasoning to handle complex graph topology. Other works like StructGPT (Jiang et al., 2023b) and KAPING (Baek et al., 2023) explore zero-shot prompting strategies to interface LLMs with structured data. 

## **6 Conclusion** 

We presented Structure-Tracing Evidence Mining, a novel framework that shifts multi-hop KG-RAG from sequential path finding to holistic subgraph matching. By synergizing a fine-tuned Semanticto-Structural Projection pipeline with a TripleDependent GNN, STEM effectively bridges the gap between natural language and KG schemas, retrieving logically connected evidence subgraphs and align with the knowledge structure. Extensive experiments on WebQSP and CWQ demonstrate that STEM significantly outperforms existing baselines, achieving new State-of-the-Art results. 

## **Limitations** 

Although STEM effectively bridges the semanticstructural gap, challenges persist due to the inherent diversity of KG topologies. First, in highly complex reasoning tasks, planning deviations may still occur, leading to scenarios where all generated candidate schema graphs fail to match the factual structure, thereby resulting in retrieval failure. Second, STEM relies on domain-specific fine-tuning and access to the target KG’s structure. While achieving strong performance on Freebase-based benchmarks (WebQSP, CWQ), we acknowledge it is not a general-purpose zero-shot method; this dependency limits its transferability to unseen KGs or novel schema types. Finally, regarding efficiency, the threshold-based expansion in the _Breadth_ strategy increases computational latency, we consider this a necessary trade-off for answer exhaustiveness. Moreover, this expansion is selective—occurring not at every step, but exclusively when the retrieval of multiple simultaneous answer paths is required. 

## **Ethical Considerations** 

We address the ethical considerations and potential risks as follows: 

**Data Provenance and Licensing.** The knowledge graph utilized in this study, is a widely adopted, publicly available database distributed under the CC-BY license. Our usage of WebQSP and CWQ datasets strictly adheres to their respective data usage policies and licenses. These datasets are standard benchmarks in the research community and do not contain private, personally identifiable information (PII), or offensive content that would require special redaction for this study. 

**Bias and Fairness.** We acknowledge that KGQA systems are susceptible to propagating biases inherent in their underlying knowledge sources. Specifically, the Freebase KG is known to exhibit significant demographic, cultural, and geographical skews. Since STEM is designed to be faithful to the retrieved subgraph, it inevitably reflects the distributional properties of the source KG. Therefore, users should interpret the outputs of STEM as a reflection of the facts stored in the specific Knowledge Graph, rather than as an unbiased representation of real-world truth. 

## **Acknowledgments** 

We would like to thank the Action Editors and the anonymous reviewers for their constructive feedback and insightful comments, which helped improve the quality of this paper. 

## **References** 

- Anthropic. 2024. The Claude 3 model family: Opus, Sonnet, and Haiku. _Anthropic Technical Report_ . 

- Tu Ao, Yanhua Yu, Yuling Wang, Yang Deng, Zirui Guo, Liang Pang, Pinghui Wang, Tat-Seng Chua, Xiao Zhang, and Zhen Cai. 2025. LightPROF: A lightweight reasoning framework for large language model on knowledge graph. In _Thirty-Ninth AAAI Conference on Artificial Intelligence, AAAI 2025_ , pages 23424–23432. AAAI Press. 

- Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024. Self-RAG: Learning to retrieve, generate, and critique through self-reflection. In _The Twelfth International Conference on Learning Representations, ICLR 2024_ . 

- Jinheon Baek, Alham Fikri Aji, and Amir Saffari. 2023. Knowledge-augmented language model prompting for zero-shot knowledge graph question answering. _CoRR_ , abs/2306.04136. 

- Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, and 29 others. 2023. Qwen technical report. _CoRR_ , abs/2309.16609. 

- Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George van den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, Diego de Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron Huang, Loren Maggiore, Chris Jones, Albin Cassirer, and 9 others. 2022. Improving language models by retrieving from trillions of tokens. In _International Conference on Machine Learning, ICML 2022_ , pages 2206–2240. PMLR. 

- Yuzheng Cai, Zhenyue Guo, YiWen Pei, WanRui Bian, and Weiguo Zheng. 2025. SimGRAG: Leveraging similar subgraphs for knowledge graphs driven retrieval-augmented generation. In _Findings of the Association for Computational Linguistics: ACL 2025_ , pages 3139–3158, Vienna, Austria. Association for Computational Linguistics. 

- Liyi Chen, Panrong Tong, Zhongming Jin, Ying Sun, Jieping Ye, and Hui Xiong. 2024. Plan-on-Graph: Self-correcting adaptive planning of large language model on knowledge graphs. In _Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024_ . 

- Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, and 48 others. 2023. PaLM: Scaling language modeling with pathways. _J. Mach. Learn. Res._ , 24:240:1–240:113. 

- Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, and Jonathan Larson. 2024. From Local to Global: A graph RAG approach to query-focused summarization. _CoRR_ , abs/2404.16130. 

- Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. 2020. REALM: retrievalaugmented language model pre-training. _CoRR_ , abs/2002.08909. 

- Gaole He, Yunshi Lan, Jing Jiang, Wayne Xin Zhao, and Ji-Rong Wen. 2021. Improving multi-hop knowledge base question answering by learning intermediate supervision signals. In _WSDM ’21, The Fourteenth ACM International Conference on Web Search and Data Mining_ , pages 553–561. ACM. 

- Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V. Chawla, Thomas Laurent, Yann LeCun, Xavier Bresson, and Bryan Hooi. 2024. G-Retriever: Retrievalaugmented generation for textual graph understanding and question answering. In _Advances in Neural Information Processing Systems (NeurIPS)_ . 

- Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen Ling, and Liang Zhao. 2025. GRAG: Graph retrievalaugmented generation. In _Findings of the Association for Computational Linguistics: NAACL 2025_ , pages 4145–4157, Albuquerque, New Mexico. Association for Computational Linguistics. 

- Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2022. Unsupervised dense information retrieval with contrastive learning. _Trans. Mach. Learn. Res._ , 2022. 

- Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C. Park. 2024. Adaptive-RAG: Learning to adapt retrieval-augmented large language models through question complexity. _CoRR_ , abs/2403.14403. 

- Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023a. Mistral 7B. _CoRR_ , abs/2310.06825. 

- Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Xin Zhao, and Ji-Rong Wen. 2023b. StructGPT: A general framework for large language model to reason 

   - over structured data. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pages 9237–9251, Singapore. Association for Computational Linguistics. 

- Dhiraj D. Kalamkar, Dheevatsa Mudigere, Naveen Mellempudi, Dipankar Das, Kunal Banerjee, Sasikanth Avancha, Dharma Teja Vooturi, Nataraj Jammalamadaka, Jianyu Huang, Hector Yuen, Jiyan Yang, Jongsoo Park, Alexander Heinecke, Evangelos Georganas, Sudarshan Srinivasan, Abhisek Kundu, Misha Smelyanskiy, Bharat Kaul, and Pradeep Dubey. 2019. A study of BFLOAT16 for deep learning training. _CoRR_ , abs/1905.12322. 

- Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for opendomain question answering. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 6769–6781, Online. Association for Computational Linguistics. 

- Omar Khattab, Keshav Santhanam, Xiang Lisa Li, David Hall, Percy Liang, Christopher Potts, and Matei Zaharia. 2022. Demonstrate-Search-Predict: Composing retrieval and language models for knowledge-intensive NLP. _CoRR_ , abs/2212.14024. 

- Gangwoo Kim, Sungdong Kim, Byeongguk Jeon, Joonsuk Park, and Jaewoo Kang. 2023. Tree of Clarifications: Answering ambiguous questions with retrievalaugmented large language models. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pages 996–1009, Singapore. Association for Computational Linguistics. 

- Yunshi Lan and Jing Jiang. 2020. Query graph generation for answering multi-hop complex questions from knowledge bases. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pages 969–974, Online. Association for Computational Linguistics. 

- Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledgeintensive NLP tasks. In _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020_ . 

- Mufei Li, Siqi Miao, and Pan Li. 2025. Simple is Effective: The roles of graphs and large language models in knowledge-graph-based retrieval-augmented generation. In _The Thirteenth International Conference on Learning Representations, ICLR 2025_ . 

- Yu Liu, Xixun Lin, Yanmin Shang, Yangxi Li, Shi Wang, and Yanan Cao. 2026. PathMind: A retrieveprioritize-reason framework for knowledge graph reasoning with large language models. In _Fortieth AAAI Conference on Artificial Intelligence, AAAI 2026_ , pages 15386–15393. AAAI Press. 

- Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In _7th International Conference on Learning Representations, ICLR 2019_ . 

- Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. 2024. Reasoning on Graphs: Faithful and interpretable large language model reasoning. In _The Twelfth International Conference on Learning Representations, ICLR 2024_ . 

- Linhao Luo, Zicheng Zhao, Gholamreza Haffari, Dinh Q. Phung, Chen Gong, and Shirui Pan. 2025. GFM-RAG: graph foundation model for retrieval augmented generation. _CoRR_ , abs/2502.01113. 

- Costas Mavromatis and George Karypis. 2025. GNNRAG: Graph neural retrieval for efficient large language model reasoning on knowledge graphs. In _Findings of the Association for Computational Linguistics: ACL 2025_ , pages 16682–16699, Vienna, Austria. Association for Computational Linguistics. 

- OpenAI. 2023. GPT-4 technical report. _CoRR_ , abs/2303.08774. 

- OpenAI. 2024. GPT-4o system card. _CoRR_ , abs/2410.21276. 

- Minbae Park, Hyemin Yang, Jeonghyun Kim, Kunsoo Park, and Hyunjoon Kim. 2026. ProgRAG: Hallucination-resistant progressive retrieval and reasoning over knowledge graphs. In _Fortieth AAAI Conference on Artificial Intelligence, AAAI 2026_ , pages 32674–32682. AAAI Press. 

- Yuan Sui, Yufei He, Nian Liu, Xiaoxin He, Kun Wang, and Bryan Hooi. 2025. FiDeLiS: Faithful reasoning in large language models for knowledge graph question answering. In _Findings of the Association for Computational Linguistics: ACL 2025_ , pages 8315–8330, Vienna, Austria. Association for Computational Linguistics. 

- Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel M. Ni, HeungYeung Shum, and Jian Guo. 2024. Think-on-Graph: Deep and responsible reasoning of large language model on knowledge graph. In _The Twelfth International Conference on Learning Representations, ICLR 2024_ . 

- Yawei Sun, Lingling Zhang, Gong Cheng, and Yuzhong Qu. 2020. SPARQA: Skeleton-based semantic parsing for complex questions over knowledge bases. In _The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020_ , pages 8952–8959. AAAI Press. 

- Alon Talmor and Jonathan Berant. 2018. The web as a knowledge-base for answering complex questions. In _Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)_ , pages 641–651, New Orleans, Louisiana. Association for Computational Linguistics. 

- Gemini Team. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. _CoRR_ , abs/2507.06261. 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. LLaMA: Open and efficient foundation language models. _CoRR_ , abs/2302.13971. 

- Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2023. Interleaving retrieval with chain-of-thought reasoning for knowledgeintensive multi-step questions. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 10014–10037, Toronto, Canada. Association for Computational Linguistics. 

- Junhong Wan, Tao Yu, Kunyu Jiang, Yao Fu, Weihao Jiang, and Jiang Zhu. 2025. Digest the knowledge: Large language models empowered message passing for knowledge graph question answering. In _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 15426–15442, Vienna, Austria. Association for Computational Linguistics. 

- Keheng Wang, Feiyu Duan, Sirui Wang, Peiguang Li, Yunsen Xian, Chuantao Yin, Wenge Rong, and Zhang Xiong. 2023. Knowledge-Driven CoT: Exploring faithful reasoning in LLMs for knowledge-intensive question answering. _CoRR_ , abs/2308.13259. 

- Song Wang, Junhong Lin, Xiaojie Guo, Julian Shun, Jundong Li, and Yada Zhu. 2025. Reasoning of large language models over knowledge graphs with superrelations. In _The Thirteenth International Conference on Learning Representations, ICLR 2025_ . 

- Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. 2024. Corrective retrieval augmented generation. _CoRR_ , abs/2401.15884. 

- Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. 2015. Embedding entities and relations for learning and inference in knowledge bases. In _3rd International Conference on Learning Representations, ICLR 2015_ . 

- Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing reasoning and acting in language models. In _The Eleventh International Conference on Learning Representations, ICLR 2023_ . 

- Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut, Percy Liang, and Jure Leskovec. 2021. QA-GNN: Reasoning with language models and knowledge graphs for question answering. In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 535–546, Online. Association for Computational Linguistics. 

- Xi Ye, Semih Yavuz, Kazuma Hashimoto, Yingbo Zhou, and Caiming Xiong. 2022. RNG-KBQA: Generation augmented iterative ranking for knowledge base question answering. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 6032–6043, Dublin, Ireland. Association for Computational Linguistics. 

- Wen-tau Yih, Matthew Richardson, Christopher Meek, Ming-Wei Chang, and Jina Suh. 2016. The value of semantic parse labeling for knowledge base question answering. In _Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, ACL 2016_ . The Association for Computer Linguistics. 

- Donghan Yu, Sheng Zhang, Patrick Ng, Henghui Zhu, Alexander Hanbo Li, Jun Wang, Yiqun Hu, William Yang Wang, Zhiguo Wang, and Bing Xiang. 2023. DecAF: Joint decoding of answers and logical forms for question answering over knowledge bases. In _The Eleventh International Conference on Learning Representations, ICLR 2023_ . 

- Wenhao Yu, Hongming Zhang, Xiaoman Pan, Peixin Cao, Kaixin Ma, Jian Li, Hongwei Wang, and Dong Yu. 2024. Chain-of-Note: Enhancing robustness in retrieval-augmented language models. In _Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing_ , pages 14672–14685, Miami, Florida, USA. Association for Computational Linguistics. 

- Bo Zhang, Jianghua Zhu, Chaozhuo Li, Hao Yu, Li Kong, Zhan Wang, Dezhuang Miao, Xiaoming Zhang, and Junsheng Zhou. 2025a. What is a good question? assessing question quality via meta-fact checking. In _Thirty-Ninth AAAI Conference on Artificial Intelligence, AAAI 2025_ , pages 15248–15256. AAAI Press. 

- Xikun Zhang, Antoine Bosselut, Michihiro Yasunaga, Hongyu Ren, Percy Liang, Christopher D. Manning, and Jure Leskovec. 2022. GreaseLM: Graph reasoning enhanced language models. In _The Tenth International Conference on Learning Representations, ICLR 2022_ . 

- Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. 2025b. Qwen3 Embedding: Advancing text embedding and reranking through foundation models. _CoRR_ , abs/2506.05176. 

- Ruilin Zhao, Feng Zhao, and Hong Zhang. 2025. Correcting on graph: Faithful semantic parsing over knowledge graphs with large language models. In _Findings of the Association for Computational Linguistics: ACL 2025_ , pages 5364–5376, Vienna, Austria. Association for Computational Linguistics. 

## **A Triple-GNN Implementation** 

In this section, we describe the overall execution process of the Triple-GNN. Specifically, we first employ an Pretrained Embedding Model (PEM) to obtain vector representations for both the triples and the entities: 

**==> picture [202 x 14] intentionally omitted <==**

**==> picture [200 x 15] intentionally omitted <==**

for all _x ∈{e, r, e[′] }_ , where the terms _e_ , _r_ and _e[′]_ represent the head entity, relation, and tail entity of the triple _t_ , respectively. The [;] operation denotes the concatenation of the embedding vectors, transforming the dimensionality from R _[d]_[PEM] _→_ R[3] _[d]_[PEM] . 

Subsequently, the MLP operation refers to a linear transformation that maps the concatenated vector from R[3] _[d]_[PEM] _→_ R _[d]_[GNN] . We finally obtain the embedding representation **E** _t_ of the triple _t_ . 

Given the structured triples _T_ = _{t_ 1 _, t_ 2 _, ..., tN }_ obtained from the Semantic-to-Structural Projection pipeline based on query _Q_ , with the corresponding knowledge graph _G_ , entity set _N_ and relation set _R_ , we first compute the embedding for each triple to yield **E** _T_ = _{_ **E** _t_ 1 _,_ **E** _t_ 2 _, ...,_ **E** _tN }_ , then aggregate the embedding set into a single representation **E** _Q ∈_ R _[d]_[GNN] via average pooling, and finally process it through the _L_ -layer Triple-GNN to obtain **H** _[L] Q[∈]_[R] _[|N|×][d]_[GNN][:] 

**==> picture [184 x 33] intentionally omitted <==**

**==> picture [186 x 15] intentionally omitted <==**

**H**[0] _e_[represents][the][initial][feature][input][of][entities] and **H**[0] _e[∈]_[R] _[|N|×][d]_[GNN][, for any] **[ h]**[0] _ei[∈]_ **[H]** _e_[0][, its initial-] ization method is defined as follows: 

**==> picture [168 x 34] intentionally omitted <==**

where _NQ_ denotes the set of question entities for question _Q_ . 

**H**[0] _r[∈]_[R] _[|R|×][d]_[GNN][denotes][the][initialized][fea-] ture representations of relations in _R_ . For each **h**[0] _ri[∈]_ **[H]** _r_[0][, we initialize it using the same Encoder] employed for the triple embeddings, followed by an MLP for linear projection, as defined below: 

**==> picture [208 x 33] intentionally omitted <==**

the MLP projects the encoded relation representation from R _[d]_[PEM] into the R _[d]_[GNN] space. 

We now describe the message passing computational flow of a single layer of Triple-GNN. Building upon the Query-Dependent GNN design proposed by GFM-RAG (Luo et al., 2025), for the embedding representation **h** _[L] e[−]_[1] of node _e_ obtained at the ( _L_ -1)-th GNN layer, the representation at the _L_ -th layer is computed as follows: 

**==> picture [203 x 14] intentionally omitted <==**

**==> picture [200 x 13] intentionally omitted <==**

where ( _e, r, e[′]_ ) _∈G_ , and _M[L] e_ = _{_ **m** _[L] e[′][|][e][′][∈]_ N _r_ ( _e_ ) _, r ∈R}_ . The set N _r_ ( _e_ ) denotes the collection of all neighboring nodes of entity _e_ under relation _r ∈R_ . In the context of a triple ( _e, r, e[′]_ ) associated with the entity _e_ , the notations **h** _[L] e[−]_[1] , **h** _[L] r[−]_[1] , and **h** _[L] e[′][−]_[1] represent the embedding representations of the head entity _e_ , the relation _r_ , and the tail entity _e[′]_ at layer _L_ -1, respectively. The operation denoted as MSG employs a DistMult (Yang et al., 2015) function to process the triple. The function _g[L]_ constitutes a 2-layer MLP operation at the subsequent layer _L_ . The Agg operation collects the states **m** _[L] e[′]_ of all neighbor nodes _e[′]_ of _e_ and performs a mean reduction: 

**==> picture [185 x 34] intentionally omitted <==**

The Update function performs the node update operation, achieved by fusing the aggregated neighbor features _M[L] e_[into the current node] **[ h]** _[L] e[−]_[1] . Specifically, the expression for Update is defined as follows: 

**==> picture [174 x 32] intentionally omitted <==**

the MLP is designed to map the concatenated R[2] _[d]_[GNN] intermediate state back to an R _[d]_[GNN] representation, acting as an effective fusing mechanism of the neighbor features. 

## **B Experiment Details** 

## **B.1 Datasets and Base KG** 

We conduct experiments on two publicly available datasets for multi-hop reasoning, including WebQuestionsSP (WebQSP) (Yih et al., 2016) and 

ComplexWebQuestions (CWQ) (Talmor and Berant, 2018). WebQSP is a large-scale multi-hop question-answering dataset, which comes with a knowledge graph in the Freebase[6] format. CWQ is a more difficult and challenging version of such datasets. Specifically, we utilized the open-source data format provided by RoG (Luo et al., 2024)[78] , as it contains complete queries paired with their corresponding subgraphs extracted from Freebase. To ensure fairness and consistency across experiments, we partitioned all datasets according to RoG, obtaining separate training and test splits. We present the statistics for all datasets in the Table 8. The distribution of answer counts in the dataset is presented in Table 9. 

## **B.2 Implementation Details** 

STEM involves three LLM-based modules: SGDA, SAGB, and the LLM reasoning model. For the first two modules, we fine-tune Qwen3-8B[9] respectively, and for reasoning model, we select Llama3.1-8B-Instruct[10] , Llama-3.1-70B-Instruct[11] , and GPT-4o[12] (OpenAI, 2024). To ensure experimental reproducibility, we set the temperature of the reasoning model to 0. Additionally, we configured the SGDA with a beam size _B_ of 4 and the SAGB with a temperature of 0. For feature embedding, we use the Qwen3-Embedding-0.6B[13] (Zhang et al., 2025b) model as the pretrained embedding model (i.e. _d_ PEM=1024), the Triple-GNN is configured with _L_ =6 layers, other configurations are consistent with the settings described in GFMRAG (Luo et al., 2025)(i.e. _d_ GNN=512). To determine the threshold _θ_ for the threshold-based search employed in the _Breadth_ strategy, we conducted a parameter search on the validation sets of WebQSP and CWQ, ultimately setting _θ_ =0.6. We performed three independent runs of the full experimental pipeline—encompassing both module training and retrieval-inference testing—and report the average values across all metrics. All models employed in this study were used in strict accordance with their respective licenses and terms of 

6https://github.com/microsoft/FastRDFStore 

7https://huggingface.co/datasets/rmanluo/RoG-webqsp 8https://huggingface.co/datasets/rmanluo/RoG-cwq 9https://huggingface.co/Qwen/Qwen3-8B 10https://huggingface.co/meta-llama/Llama-3. 

1-8B-Instruct 11https://huggingface.co/meta-llama/Llama-3. 1-70B-Instruct 

12https://chatgpt.com/ 13https://huggingface.co/Qwen/Qwen3-Embedding-0.6B 

|**Dataset**|**Train**<br>**Dev**<br>**Test**<br>**Hops**<br>**1 Hop**<br>**2 Hops**<br>_≥_**3 Hops**|
|---|---|
|**WebQSP**<br>**CWQ**|2,826<br>239<br>1,628<br>{1,2}<br>65.5%<br>34.5%<br>-<br>27,639<br>3297<br>3,531<br>{1,2,3,4}<br>41%<br>38.3%<br>20.7%|



Table 8: Statistics of the number of WebQSP and CWQ dataset splits along with the question hops. 

|**Dataset**|**Ans**= 1<br>**Ans**_∈_[2,4]<br>**Ans**_∈_[5,9]<br>**Ans**_≥_10|
|---|---|
|**WebQSP**<br>**CWQ**|51.8%<br>27.1%<br>8.1%<br>13.0%<br>71.4%<br>19.0%<br>5.9%<br>3.7%|



Table 9: Distribution statistics of answer counts in the two datasets. 

use: OpenAI Terms of Use for GPT-4o, Llama 3.1 Community License for Llama models, and Apache 2.0 License for Qwen models. 

## **B.3 Evaluation Metrics** 

Following established evaluation protocols, we assess model performance using two standard metrics: **Hits@1** and **F1 score** . Hits@1 quantifies the accuracy of the top-ranked answer prediction, while the F1 score provides a balanced assessment of answer coverage. 

## **B.4 Baselines** 

Our experimental baselines are categorized into four groups: **Pure LLM Reasoning** , referring to methods that utilize only the large language model’s inherent capabilities, **With Finetuning** , referring to methods that involve fine-tuning the reasoning model, **With Prompting**[14] , referring to methods that control the reasoning and answering behavior of large language models through prompting, and **Our Proposed Method** . Notably, our proposed STEM inherently belongs to the fine-tuning category, as it relies on fine-tuned upstream modules for retrieval. We now introduce each category as follows: 

**Pure LLM Reasoning.** We do not employ any retrieval components, but instead rely solely on the inherent reasoning capabilities, the model selected is GPT-4o, and experiments are conducted using three distinct settings: pure reasoning, few-shot learning, and CoT prompting. 

**With Finetuning.** We select the following methods for comparison: **NSM** (He et al., 2021) proposes a teacher network to supervise the intermedi- 

14Notably, comparing STEM against prompting baselines does not imply zero-shot parity. Rather, it demonstrates that when training is feasible, structural alignment provides substantial gains in factual precision and hallucination mitigation. 

ate reasoning process. **DeCAF** (Yu et al., 2023) performs question-relevant retrieval at the document level and constructs logic forms for answers to optimize responses. **KD-CoT** (Wang et al., 2023) proposes KG-guided intermediate reasoning verification to ensure a more reliable reasoning process. **RoG** (Luo et al., 2024) leverages LLM-generated reasoning paths and continuously refines them with KG before performing retrieval. **GRAG** (Hu et al., 2025) optimizes subgraph retrieval complexity and employs both text view and graph view to enhance question comprehension, and **LightProf** (Ao et al., 2025) retrieves the reasoning path, then integrate KG factual and structural information into embeddings for improved answering. 

**With Prompting.** We adopt the following approaches as baselines for comparison: **G-Ret** (GRetriever) (He et al., 2024) proposes a novel RAG framework that formulates subgraph retrieval as a Prize-Collecting Steiner Tree (PCST) problem. **ToG** (Sun et al., 2024) introduces a framework where an LLM acts as an agent to iteratively explore reasoning paths on a knowledge graph via beam search. **Kaping** (Baek et al., 2023) is a zeroshot framework that retrieves relevant facts from a knowledge graph and prepends them to the input prompt. **FiDeLiS** (Sui et al., 2025) proposes a training-free framework that combines step-wise beam search with a deductive scoring function and Path-RAG module. **PoG** (Chen et al., 2024) decomposes questions into sub-objectives and iteratively adapts reasoning paths through guidance, memory, and reflection mechanisms. **ReKnoS** (Wang et al., 2025) introduces a novel framework that enhances LLM reasoning by incorporating super-relations in knowledge graphs. **MFC** (Zhang et al., 2025a) transforms questions into knowledge graph triples using LLMs and quantifies question quality based on cognitive metrics. **SubgraphRAG** (Li et al., 

2025) decouples the roles of knowledge graphs and LLMs in RAG systems. **GNN-RAG** (Mavromatis and Karypis, 2025) leverages lightweight GNNs for efficient graph retrieval. **ProgRAG** (Park et al., 2026) introduces feedback-aware and evidenceaware mechanisms to progressively align LLM reasoning with factual knowledge from graphs. 

## **C Training Setup** 

## **C.1 Basic Training Configuration** 

Our work involves the training of three modules: Schema-Grounded Decomposition Agent, SymbolAligned Graph Builder, and Triple-GNN[15] . We will sequentially introduce the data construction and training methods for each of these modules. 

First, we introduce the training data construction method, we utilize the training splits of the WebQSP and CWQ datasets. For both datasets, we take a question _Q_ from the training split, along with its corresponding answer _A_ , question entities set _NQ_ and KG _G_ . We first extract reasoning chain _R_ from _G_ employing the method proposed in CoG (Zhao et al., 2025), then decompose the reasoning chain into individual triples _T_ : 

**==> picture [154 x 12] intentionally omitted <==**

Then, based on the question _Q_ , we mark all entities in _T_ that do not appear in _NQ_ using placeholders, this is because these entities are not question entities, but rather intermediate answer entities in the multi-hop reasoning process. This marking complies with the following principles: (1) when two triples share the same answer entity (indicating that they are connected in the graph), the same identifier “[ENTX]” is used; (2) different entities are distinguished by different identifiers (“[ENTX]” and “[ENTY]”). After formatting process, we obtain a new masked _T_ : 

**==> picture [156 x 14] intentionally omitted <==**

Based on the obtained _T ′_ , we generate an assertion for each _t′ ∈T ′_ by using a prompt _P_ 4 to instruct a large language model (for all prompt-based data generation tasks in this study, we consistently utilized **Gemini 2.5 Pro** (Team, 2025) API, with the temperature set to 1.0), thereby obtaining a 

15Here, the Triple-GNN encompasses not only the parameters of the GNN module itself but also the parameters of various associated projection layers: **MLP** _ϕ_ 1 , **MLP** _ϕ_ 2 and **MLP** _ϕ_ 4 , which have been denoted in their respective formula descriptions. 

set _S_ containing assertions corresponding to each _′_ triple in _T_ : 

**==> picture [160 x 15] intentionally omitted <==**

For **Symbol-Aligned Graph Builder Training** , we treat the generated assertions _Sj_ as the source input and the original structured triples _Tj′_[as the] target output to optimize the SAGB model: 

**==> picture [217 x 52] intentionally omitted <==**

where _D_ represents the training dataset, _P_ 2 denotes the instruction prompt used by the SAGB for schema graph building as introduced in Section 3.2. 

For question-answering strategy generation, we determine based on the number of ground-truth answers for _Q_ : if there is a single answer, _σ_ is set to “Precision”; if there are multiple answers, _σ_ is set to “Breadth”. Thus for **Schema-Grounded Decomposition Agent** training, we use _Qj_ as the source input and ( _Sj_ , _σj_ ) as the target output to optimize the SGDA model: 

**==> picture [212 x 54] intentionally omitted <==**

For Triple-GNN Training, we apply the method described in Appendix A to obtain the pooled embedding representation **E** _Q ∈_ R _[d]_[GNN] of _T ′_ , then fed into the _L_ -layer Triple-GNN to produce **H** _Q_ , and finally transformed via a mapping function and activated to obtain the entity probability: 

**==> picture [184 x 33] intentionally omitted <==**

The definition of **H**[0] _e[∈]_[R] _[|N|×][d]_[GNN][and] **[H]**[0] _r[∈]_ R _[|R|×][d]_[GNN] is the same as in Appendix A, _N_ and _R_ denote the set of entity nodes and relations in graph _G_ respectively. For the label of each entity _ei ∈N_ , we construct it as follows: 

**==> picture [164 x 34] intentionally omitted <==**

The final training loss employs the BCE loss (Luo et al., 2025), and is formulated as follows: 

**==> picture [200 x 30] intentionally omitted <==**

where _**p** ei ∈_ _**P** Q_ denotes the probability of entity _ei_ , the training details for each component are as follows: 

- For the SGDA, we utilize a training set of approximately 25k entries in the format ( _Q_ , ( _S_ , _σ_ )). The training configuration includes a learning rate of 1e-4, a batch size of 32, and 2 training epochs, we employing Bfloat16 (Kalamkar et al., 2019) precision and the AdamW optimizer (Loshchilov and Hutter, 2019). The maximum input length for the SGDA is set to 2048 tokens. 

- For the SAGB, the training set consists of about 29k entries in the format ( _S_ , _T ′_ ). It is trained for 2 epochs with a learning rate of 1e5[16] , using Bfloat16 precision and the AdamW optimizer. The maximum input length is set to 2048 tokens. 

- The Triple-GNN is trained for 2 epochs with a learning rate of 1e-5. 

- All training processes were conducted on a single machine with 4 _×_ NVIDIA H100 GPUs. 

**Statistics of Trainable Parameters.** The trainable components in our framework primarily include the SGDA and SAGB models, while each possesses a nominal size of 8B parameters. Additionally, the framework incorporates the TripleGNN module ( _**W** ϕ_ 3 _≈_ 8M parameters) and several auxiliary projection layers ( _**W** ϕ_ 1 _∈_ R[512] = 512, _**W** ϕ_ 2 _∈_ R[3072] _[×]_[512] _≈_ 1.5M and _**W** ϕ_ 4 _∈_ R[1024] _[×]_[512] _≈_ 0.5M), the aggregate number of above trainable parameters is about 10M parameters. 

## **C.2 Structure-to-Query Reverse Generation** 

The training datasets provide high-quality supervision, however they are limited in scale, which consequently restricts its coverage of the logic and schema diversification encapsulated in the KG. To equip our Semantic-to-Structural Projection pipeline with broader generalization capabilities and cover long-tail relations, we propose a novel Structure-to-Query Reverse Generation strategy, which constructs a large-scale, synthetic instruction-tuning dataset directly from the KG 

16The generation of triples constitutes a format-constrained task, which relies minimally on the prior knowledge of the text-based LLM, so we adopt a smaller learning rate to ensure more stable model training. 

topology. We elaborate on the procedure in two distinct phases. 

**Phase 1: Reasoning-Path Subgraph Sampling.** For each graph _G_ in the WebQSP and CWQ datasets, we employ a random walk strategy to obtain a corresponding subgraph _Gsub_ and its associated entity set _Nsub_ . Specifically, for subgraph _Gsub_ , we randomly mask a subset of nodes to serve as both the target answers and intermediate reasoning entities, replacing them with unified “[ENTX]” placeholders consistent with Section C.1, designating the remaining structure, denoted as _Gsub[∗]_[, as the] evidence subgraph required for reasoning. 

**Phase 2: LLM-Driven Reverse Generation.** We leverage a powerful LLM to generate natural language queries from the sampled subgraphs. Formally, sampled and masked graph _G_ sub _[∗]_ = _{_ ( _e_ 1 _, r_ 1 _, e_ 2) _,_ ( _e_ 2 _, r_ 2 _,_ “[ENT1]”) _}_ (assuming _e_ 3 is masked to serve as the answer entity), we instruct the large language model to generate multi-hop question _Q_ and declarative statements _S_ sub using prompt _P_ 6, _G_ sub and designated answer entity “[ENT1]”: 

**==> picture [206 x 12] intentionally omitted <==**

where _S_ sub = _{s_ 1 _, s_ 2 _, ..., sn}_ . Considering the complexity of this prompt, which entails a two-step task execution, we enabled the “thinking mode” throughout the LLM inference process. We then employ prompt _P_ 5, _Q_ and _S_ sub to generate corresponding answering strategy _σ_ , following the method described in Appendix C.1. 

Through the LLM-driven Reverse Generation method, we ultimately construct the knowledge graph reverse-generation dataset _D_ syn, for every _di ∈D_ syn: 

**==> picture [202 x 13] intentionally omitted <==**

where _A[i]_ denotes the answer to _Q[i]_ , corresponding to the masked entity _e_ 3 in the example of _G_ sub _[∗]_[.] 

For the training of the SGDA, the objective function is defined as follows: 

**==> picture [215 x 54] intentionally omitted <==**

For the training of the SAGB, the objective func- 

**==> picture [455 x 165] intentionally omitted <==**

**----- Start of picture text -----**<br>
World Series ... New York  New York  New York Giants<br>... Giants Giants<br>championship San Francisco Giants rival Lou Seal Sample San Francisco Giants rival Lou Seal Mask ENT1 rival team_ ENT2<br>... San Francisco2014 World Series containedby StadiumSeals  championstadiumarena_ Dodgers-Giants rivalmascotteam_ kind_of_rivalry Baseball StadiumSeals  stadiumarena_ mascotteam_ StadiumSeals  stadiumarena_ Selected answer entity mascot Select ENT2 ：<br>rivalry ENT1’s rival is New York<br>Giants<br>What is the mascot of the team whose opponent is the  ENT1’s arena stadium is Seals  Declarative<br>New York Giants and whose stadium is Seals Stadium? Stadium Sentences<br>Generated Multi-Hop  ENT1’s team mascot is ENT2<br>Question<br>**----- End of picture text -----**<br>


Figure 3: An illustrative example of the Structure-to-Query Reverse Generation pipeline. 

tion is defined as follows: 

**==> picture [209 x 53] intentionally omitted <==**

The training objective of the Triple-GNN is consistent with that described in C.1. Regarding labeling, we perform positive and negative sampling within the entity set _Nsub[i]_[, adhering to the following] labeling principles: 

**==> picture [161 x 34] intentionally omitted <==**

For clarity, we present a complete example of the construction process as shown in Figure 3. Ultimately, we constructed dataset _D_ syn comprising about 210k entries. Detailed statistical information regarding _D_ syn is presented in Table 10. This dataset is utilized to train the SGDA, SAGB, and Triple-GNN modules, following the same logic as previously described. We finally incorporate _D_ syn into the the original training set in Appendix C.1. We will validate the impact of the _D_ syn dataset by comparing performance with and without it in the Appendix D. The synthetic dataset _D_ syn will be released alongside the source code. 

## **C.3 Data ethics** 

We employ LLMs to generate intermediate assertions and multi-hop queries, a potential risk in such synthetic data generation is hallucination. To mitigate this, our Structure-to-Query Reverse Generation method is strictly grounded in sampled subgraphs from the KG. The reasoning paths are predetermined by the graph structure, ensuring that 

|**Statistic**|**Value**|
|---|---|
|Total Count|214,733|
|Avg Question Length|49|
|Hop=1|84,465|
|Hop=2|65,810|
|Hop=3|34,210|
|Hop_≥_4|30,248|
|_Precision_|132,391|
|_Breadth_|82,342|



Table 10: Statistics of the synthetic dataset _D_ syn, including total size, hop-count distribution, and strategy distribution. 

the generated questions and assertions are logically consistent with the underlying facts. While we have conducted manual quality checks on random samples, we acknowledge that minor semantic noise may persist. Furthermore, given that Knowledge Graphs and LLMs are known to exhibit inherent biases (e.g., geographical, gender, or cultural), our synthetic data—being derived from these sources—may inadvertently propagate such pre-existing biases. Furthermore, the synthetic data constructed via our reverse generation method is derived exclusively from public knowledge graph and standard LLM outputs, these public resources have already been anonymized and sanitized to remove personally identifiable information, thus all training datasets contain no PII or private data. 

## **D Additional Experimental Results and Analysis** 

## **D.1 The Impact of the Answer Strategy** _σ_ **on Performance** 

In this section, we investigate the impact of answer strategy differentiation on the experimental results. 

**==> picture [409 x 284] intentionally omitted <==**

**----- Start of picture text -----**<br>
Where is jamarcus russell<br>from? ... jamarcus<br>0.2<br>russell Mobile<br>jamarcus russell place_of jamarcus<br>_birth russell Mobile<br>place_of_birth ethnicity 0.43 0.9 Max Score place_of_birth Mobile<br>ENT1<br>African<br>American<br>(a) An illustrative example of Greedy Selection, corresponding to the  Precision  answering strategy.<br>What are the four main  Province of Badajoz Spanish Language Spanish Language<br>languages spoken in Spain? 0.88<br>0.39 Language_<br>Spain contained_by spoken 0.83 Language_ Spanish Language<br>Spain Language_spoken Occitan  Score≥0.6 spokenLanguage_ Occitan language<br>language_spoken ENT1 Language_spoken 0.86 Language_spoken 0.9 language Language_spoken Spain Language_spokenspoken languageOccitan  Galician LanguageCatalan language<br>Catalan  Galician<br>language Language Catalan  Galician<br>language<br>Language<br>**----- End of picture text -----**<br>


- (b) An illustrative example of Threshold-based Selection, corresponding to the _Breadth_ answering strategy. 

To illustrate the workflows and distinct behaviors of the search modes under the _Precision_ and _Breadth_ strategies, we first provide a representative example for Greedy Selection in Figure 4a and one for Threshold-based Selection in Figure 4b. 

We organize the experiments into three groups: the original Adaptive STEM Strategy, a purely Greedy Selection strategy, and a purely Thresholdbased Selection strategy. The experimental results are presented in Table 11. 

As shown in the results, the Only Greedy setting exhibits a significant performance decline in the F1 score, this is primarily because greedy search suffers from insufficient evidence recall in multianswer scenarios, leading to incomplete answers. In contrast, the Only Threshold-based setting does not show a drastic drop and even outperforms the original STEM configuration on both F1 metrics. This is because Threshold-based Selection ensures answer coverage. However, it still underperforms STEM in other metrics. This is attributed to the retrieval of excessive irrelevant evidence, which induces hallucinations. We discuss the efficiency of the two search modes in Appendix D.8. 

## **D.2 Initial Entity Count** _K_ **on Performance** 

During the construction of the Global Guidance Subgraph, we perform an initial retrieval of _K_ en- 

|**Strategy**|**WebQSP**|**CWQ**|
|---|---|---|
||Hit@1 _F_1 Score|Hit@1 _F_1 Score|
|Only Greedy<br>Only Threshold-based <br>STEM|88.50<br>60.75<br> 89.36<br>78.54<br>90.94<br>76.18|72.45<br>44.29<br>74.53<br>67.18<br>74.09<br>65.33|



Table 11: Performance comparison with different response strategies. 

tities. In this section, we conduct experiments to evaluate the impact of different _K_ values, to illus- _′_ trate this more clearly, we assume _K_ = _| T | ∗K_ and define _K′_ as the **Guidance Graph Scale Factor** , we investigate the impact of different values of _K′_ on performance. 

The comparison results of all parameters _K′_ across different metrics are shown in Figure 5a (Hit@1) and Figure 5b (F1). From the results in the tables, it can be seen that values of _K′_ smaller or larger than 4 both affect the performance. Particularly when _K′_ =1, there is a significant decline in metrics across both datasets. In terms of Hit@1 on both datasets, performance generally declines when _K′_ is less than 4, and gradually improves as _K′_ increases, with the CWQ dataset even showing a slightly better result at _K′_ =3 compared to the reported results with _K′_ =4. However, performance drops again when _K′_ exceeds 4. Regarding F1, a 

**==> picture [213 x 351] intentionally omitted <==**

**----- Start of picture text -----**<br>
90 . 94<br>88 . 56 88 . 64 89 . 59 89 . 43 89 . 66<br>90<br>80<br>72 . 41 73 . 05 74 . 47 74 . 09 74 . 05 73 . 84<br>70<br>WebQSP<br>CWQ<br>60<br>1 2 3 4 5 6<br>Guidance Graph Scale Factor ( K′ )<br>(a) Impact of Guidance Graph construction scale on per-<br>formance.<br>80 76 . 18 75 . 85 76 . 42<br>73 . 85 74 . 51 75 . 06<br>70<br>64 . 81 64 . 4 65 . 33 64 . 99<br>63 . 65<br>61 . 59<br>60<br>WebQSP<br>CWQ<br>50<br>1 2 3 4 5 6<br>Guidance Graph Scale Factor ( K′ )<br>Hit@1 (%)<br>F1 (%)<br>**----- End of picture text -----**<br>


(b) Impact of Guidance Graph construction scale on performance. 

similar declining trend is observed when _K′_ is less than 4, and simultaneously a moderate decline is observed when _K′_ exceeds 4, with the exception of the WebQSP dataset. Therefore, _K′_ =4 is selected as our final configuration. Our analysis suggests that a smaller _K′_ results in a smaller Guidance Graph, which risks missing key entities or question entities, while a larger _K′_ may introduce low-value entities and mislead the evidence search. 

## **D.3 Guidance Graph Correction Analysis** 

While Table 7 demonstrates that ablating the Guidance Graph significantly degrades overall QA performance, these end-to-end metrics are inevitably confounded by LLM generation stochasticity. To explicitly quantify the Guidance Graph’s errorcorrection capability independent of the LLM, we conduct a pure retrieval evaluation. Specifically, we measure the coverage of ground-truth reasoning paths within the retrieved subgraphs with and without Guidance Graph guidance (using Equation 17). Results are presented in Table 12. 

|**Scoring Bias**<br>**WebQSP CWQ**<br>**STEM** <br>7368<br>7039|**Scoring Bias**<br>**WebQSP CWQ**<br>**STEM** <br>7368<br>7039|**Scoring Bias**<br>**WebQSP CWQ**<br>**STEM** <br>7368<br>7039|
|---|---|---|
|_GP T −_4_o_<br>**w/o**I**Ent** &I**Tri**<br>**w/o**I**Ent**<br>**w/o**I**Tri**|.<br>65.07<br>70.25<br>68.49|.<br>59.8<br>66.68<br>65.04|



Table 12: Coverage rate of ground-truth reasoning paths within retrieved evidence subgraphs. We compare the pure retrieval quality under settings with and without the Guidance Graph. All values are reported as percentages (%). 

As shown in Table 12, the full Guidance Graph configuration (incorporating both Entity-level and Triple-level biases) achieves the highest retrieval coverage. Quantitatively, the inclusion of Guidance Graph increases coverage from 65.07% to 73.68% on WebQSP (an absolute improvement of over 8%) and from 59.8% to 70.39% on CWQ (an increase exceeding 10%). Conversely, removing both constraints yields the lowest coverage, particularly on the more complex CWQ dataset. This significant drop indicates that without global structural guidance, the subgraph search is highly susceptible to being misled by local semantic features, leading to severe error propagation. Furthermore, ablating either bias individually outperforms the unconstrained variant but remains inferior to the fully constrained variant, confirming their complementary roles. 

## **D.4 Impact of Global Structural Consistency Bias Values on Performance** 

In this section, we conduct experiments on bias constraints (IEnt and ITri). Specifically, we apply a multiplicative factor of 3 _/_ 2 to the scores during initial nodes selection in Equation 10 and 11, and an additive boost of 1 _/_ 2 to the scores during the edge search in Equation 12 and 13. Given that this involves the joint adjustment of two variables, we adopt a grid search strategy. Let _λ_ denote the multiplicative factor for the Entity-level bias, and _τ_ denote the boosting factor for the Triple-level bias. We define the search ranges for _λ_ and _τ_ as follows: 

**==> picture [198 x 28] intentionally omitted <==**

Since grid search involves a large number of experimental iterations, we randomly sample 200 examples from WebQSP and CWQ dataset to construct WebQSPsub and CWQsub, and conduct experiments on them. We fix one variable and search for 

the optimal setting of the other, all experimental results are presented in Figure 6a and Figure 6b. 

From the experimental results, we can conclude that when both _λ_ and _τ_ are relatively small, performance drops significantly. In particular, with _λ_ = 1.2, the WebQSP score decreases by about 3%, as _λ_ increases from 1.5 onward, scores generally improve and remain stable thereafter. A similar trend is observed for _τ_ : with _τ_ = 0.2, the performance also deteriorates. Starting from _τ_ = 0.5, the scores improve and stay stable. Since larger parameters do not bring significant further improvements, we select the relatively low values _λ_ = 1.5 and _τ_ = 0.5 as our final configuration. 

**Our Analysis** The results suggests that excessively high parameter values lead to an overreliance on the Guidance Graph. We posit that during query-based Guidance Graph building, the GNN may inadvertently incorporate edges with low relevance. These edges, while structurally connected, often contribute little to the actual reasoning process—a phenomenon we term “ **Structural Over-Interpretation** ”. This limitation elucidates why the Guidance Graph cannot serve as the final reasoning subgraph in isolation and necessitates a subsequent refinement via semantic search. Fundamentally, STEM represents a synergy between logical reasoning (structure) and semantic matching (content). It achieves an optimal equilibrium, avoiding over-reliance on either modality while leveraging the indispensable strengths of both. 

## **D.5 Impact of Reverse Generation Data** 

We evaluate the impact of Structure-to-Query training set reverse generation on model performance. We first designed two comparative settings: a standard dataset, denoted as _D_ std, constructed solely from the training splits of WebQSP and CWQ; and an augmented dataset, denoted as _D_ aug, which combines the _D_ std with the synthetic data _D_ syn produced in C.2. We trained two separate sets of SGDA, SAGB, and Triple-GNN modules using these respective datasets and conducted comparative experiments. Two specific experiments were designed: **1. Schema Generation Quality:** We employed the SGDA and SAGB modules to construct schema graphs for queries in the WebQSP and CWQ test sets. The quality of these graphs was then evaluated by measuring precision and recall against the ground-truth reasoning paths. **2. End-to-End QA Performance:** We integrated SGDA, SAGB, and Triple-GNN into the complete 

**==> picture [201 x 158] intentionally omitted <==**

**----- Start of picture text -----**<br>
80<br>70 . 18 70 . 3 70 . 54 70 . 12 70 . 35<br>70 67 . 15<br>60<br>52 . 71 54 . 22 54 . 16 53 . 19 54 . 1 53 . 98<br>50<br>WebQSP (sub)<br>CWQ (sub)<br>40<br>1 . 2 1 . 5 1 . 8 2 . 1 2 . 4 2 . 7<br>Multiplicative factor  λ<br>F1 (%)<br>**----- End of picture text -----**<br>


(a) Performance comparison with different _λ_ . Due to the constraints of the controlled variable method, the value of _τ_ is set to 0.2 for all experiments. 

**==> picture [201 x 158] intentionally omitted <==**

**----- Start of picture text -----**<br>
80<br>68 . 9 69 . 49 68 . 37 68 . 45 69 . 02 69 . 29<br>70 67 . 15<br>60 55 . 71 55 . 94 55 . 68 54 . 76 54 . 93 55 . 16<br>52 . 71<br>50<br>WebQSP (sub)<br>CWQ (sub)<br>40<br>0 . 2 0 . 5 0 . 8 1 . 1 1 . 4 1 . 7 2<br>Boosting factor  τ<br>F1 (%)<br>**----- End of picture text -----**<br>


(b) Performance comparison with different _τ_ . Due to the constraints of the controlled variable method, the value of _λ_ is set to 1.2 for all experiments. 

STEM retrieval framework and evaluated the overall question-answering performance on both test sets. 

## **D.5.1 Graph Evaluation** 

To evaluate schema graph generation quality, we first define the Precision and Recall metrics for the generated schema graphs. Given a question _Q_ , let _R_ denote the ground-truth reasoning path provided in the test set, and _Gsch_ denote the schema graph generated by our trained Semantic-to-Structural Projection pipeline. We calculate Precision, Recall, and F1 score as follows: 

**==> picture [165 x 57] intentionally omitted <==**

**==> picture [170 x 24] intentionally omitted <==**

The experimental results are presented in Figure 

|**Dataset**|**WebQSP**|**CWQ**|
|---|---|---|
||Hit@1 _F_1 Score|Hit@1 _F_1 Score|
|_D_std<br>_D_aug|86.35<br>74.17<br>90.94<br>76.18|67.03<br>58.78<br>74.09<br>65.33|



Table 13: Ablation study on reverse generation data: comparison of multi-hop QA results. 

**==> picture [215 x 228] intentionally omitted <==**

**----- Start of picture text -----**<br>
Precision Recall F1<br>100<br>80<br>60<br>40<br>20<br> D std  D aug  D std  D aug<br>WebQSP / WebQSP / CWQ / CWQ /<br>2589 . 8749 . 3688 .<br>7662 . 93 . 742 .<br>71<br>7858 . 5786 . 3158 .<br>4355 . 4532 . 4144 .<br>Precision / Recall / F1 (%)<br>**----- End of picture text -----**<br>


Figure 7: Ablation study on reverse generation data: comparison of schema graph P/R/F1 metrics. 

7. It is evident that incorporating the _D_ aug data leads to significant improvements in schema generation Precision, Recall, and F1 scores across both test sets. Notably, on WebQSP, the inclusion of _D_ aug yields a Recall increase of approximately 15% and an F1 improvement exceeding 14%. Similarly, the CWQ dataset witnesses a marked 15% rise in Precision and a 12% gain in Recall. Collectively, these results demonstrate that the incorporation of _D_ aug significantly bolsters the model’s capability to perform logical perception for complex queries. 

## **D.5.2 End-to-End QA Performance** 

We constructed complete STEM retrieval-QA pipelines using modules trained on the two respective datasets and conducted a comparative evaluation, with results shown in Table 13. The end-toend tests reveal that the pipeline trained with _D_ aug dataset consistently achieves higher overall scores than the one trained with _D_ aug dataset. Notably, the improvement margin on CWQ is more significant, with Hit@1 increasing by over 7%. This indicates that the performance benefits yielded by the Structure-to-Query Reverse Generation method 

|**Dataset**|**Ans**= 1 **Ans**_∈_[2,4] **Ans**_∈_[5,9] **Ans**_≥_10|
|---|---|
|**WebQSP**<br>**CWQ**|3.1<br>4.03<br>5.59<br>8.81<br>3.41<br>3.68<br>6.13<br>9.03|



Table 14: Detailed average reasoning time (s) partitioned by answer count intervals. 

|**Dataset**|**Ans**= 1 **Ans**_≥_2|
|---|---|
|**WebQSP**<br>**CWQ**|96.5<br>93.31<br>93.91<br>91.62|



Table 15: Strategy generation accuracy (%) of SGDA across different test set splits. We categorize questions into two groups based on answer cardinality: those with a single answer (count = 1) and those with answer counts _≥_ 2. For the former, accuracy is measured by the proportion of generated _Precision_ strategies, while for the latter, it is measured by the proportion of _Breadth_ strategies. 

are amplified in complex reasoning scenarios involving a higher number of hops, underscoring the critical importance of query planning in multi-hop reasoning tasks. 

## **D.6 Impact of SGDA Beam Size** _B_ **on Performance** 

Given the inherent structural complexity of KG schemas, the SGDA employs beam search to generate multiple candidate sequences of atomic relational assertions simultaneously. In this section, we investigate the impact of the beam size _B_ on model performance. The experimental results are illustrated in Figure 8. 

It is evident that with a small beam size, retrieval failures often arise from the insufficiency of generated assertions. Specifically, at _B_ = 1, the model yields only the single most probable assertion; however, since queries of the same semantic type do not invariably map to a single identical pattern, retrieving a diverse set of multi-pattern assertions significantly enhances the structural hit rate of the evidence subgraph. Consequently, performance improves significantly as _B_ = 1 gradually increases. This trend is particularly pronounced on CWQ. Conversely, increasing _B_ beyond 4 yields only marginal performance gains. Consequently, to balance retrieval accuracy with the computational complexity induced by processing excessive assertions, we adopt _B_ = 4 for this work. 

## **D.7 Typical Case Study** 

In this section, we present a qualitative analysis of STEM’s capabilities in query understanding, 

|**Dataset**|_Precision_<br>_Breadth_|
|---|---|
|**WebQSP**<br>**CWQ**|4.91<br>8.42<br>4.35<br>7.98|



Table 16: Latency comparison (s) of different answering strategies. 

**==> picture [203 x 153] intentionally omitted <==**

**----- Start of picture text -----**<br>
80 76 . 18 76 . 82 76 . 45 77 . 09<br>69 . 84<br>67 . 78 67 . 57<br>70 64 . 27 65 . 69 64 . 56 66 . 02<br>60 53 . 43<br>50 45 . 28 46 . 59<br>40 WebQSP<br>CWQ<br>30<br>1 2 3 4 5 6 7<br>Beam size  B<br>F1 (%)<br>**----- End of picture text -----**<br>


Figure 8: Performance comparison with different beam size _B_ . 

logical planning, schema graph construction, and retrieval through representative case studies. We selected diverse examples from both the WebQSP and CWQ datasets to illustrate various performance characteristics. 

We begin with the WebQSP dataset, examining cases C1 (Table 17), C2 (Table 18), and C3 (Table 19). Case C1 (Table 17) exemplifies the “schema hallucination” challenge discussed in Section 1. As shown in the table, the SGDA module successfully mapped the query semantic “airport to fly into” to the concept of “nearby airport”, guiding the SAGB to construct a schema graph consistent with KG facts. We also observe that all four sets of assertions were mapped to an identical schema graph, demonstrating the SAGB’s structural consistency in handling such queries. Furthermore, the SGDA correctly predicted the _Breadth_ strategy, facilitating successful retrieval of all subgraphs and comprehensive recall of the two answers. 

Regarding Case C2 (Table 18), based on diverse assertions, the SAGB constructed multiple candidate schema graphs, retrieving answerirrelevant yet structurally similar evidence subgraphs (e.g., (“Beech Street Historic District”, “location.location.containedby”, “Texarkana, Arkansas”)). However, leveraging the LLM’s precise discriminative capability, the system successfully identified and selected the correct ev- 

idence graph: (“Texarkana, Arkansas”, “location.hud_county_place.county”, “Miller County”). 

Finally, in Case C3 (Table 19), the SGDA successfully bridged the semantic gap by transforming the phrase “style of music” into the assertion term “music genre”. This alignment enabled the accurate construction of the schema graph and the subsequent retrieval of supporting evidence. 

Turning to the CWQ dataset, we analyze cases C4 (Table 20), C5 (Table 21), C6 (Table 22) and C7 (Table 23). The increased hop-count complexity inherent in CWQ inevitably exacerbates the potential for reasoning errors. In Case C4 (Table 20), although structurally similar schema graphs were constructed from different assertions, the relation names exhibited significant diversity, and the SAGB incorrectly mapped the relation to symbol “sports.school_sports_team.team”. As indicated in the “Retrieved” row, the ground-truth relation symbol is “sports.school_sports_team.school”. However, due to the high semantic similarity retained between the predicted and actual relations, the system still achieved accurate evidence retrieval. Simultaneously, the reasoning model successfully derived the final answer from the retrieved evidence subgraph. 

Case C5 (Table 21) demonstrates the SGDA’s robust divergent reasoning capability when decomposing multi-hop queries. While assertions (1) represents a generic logical form, the structural complexity of assertions (3) stems from the SGDA’s comprehensive grasp of KG schemas, exemplifying high-quality factual alignment. Consequently, the schema graph derived from assertions (3) accurately retrieved the target evidence subgraph. In contrast, the retrieval results for assertions (1) and (2) were rejected by the reasoning model due to the absence of critical information regarding the movie “Forrest Gump” (the question entity given in the test set) thereby ensuring the output of the correct evidence. 

Case C6 (Table 22) demonstrates a failure case of the SGDA. The assertions (1) (“Corfu’s official language is [ENT1]”) resulted in a retrieval failure due to the absence of corresponding relation within the KG. In contrast, assertions (3) (“Corfu is an administrative division of [ENT1]”, “[ENT1]’s official language is [ENT2]”) successfully aligned accurately with the underlying facts, ultimately retrieving all correct answers. Meanwhile, assertions (2), utilizing the relation “location.location.containedby”, retrieved an irrelevant 

subgraph. 

Finally, Case C7 (Table 23) presents an instance of assertion conversion error, where the relationship between “Brussels” and the “European Union” was misidentified as “location.location.containedby” instead of the correct “organization.organization.founders”. Nevertheless, the Triple-GNN successfully identified the “European Union” as a critical node during the construction of the Guidance Graph. Consequently, the correct triple (“European Union”, “organization.organization.founders”, “Belgium”) was included in the Guidance Graph, which applied a positive bias to this edge during the search phase. This structural prior effectively corrected the semantic deviation, ensuring the successful recall of the evidence subgraph[17] . Although the relation “organization.membership_organization.members” triggered the retrieval of an irrelevant subgraph, the presence of the entity “Brussels” successfully prevented the model from being misled. 

Collectively, these findings demonstrate STEM’s robust resilience against schema inconsistency, manifested in three key aspects: 

- **Multiple Planning Hypotheses** : Faced with diverse KG structures, the SGDA generates multiple candidate decomposition plans, significantly increasing the hit rate for the correct knowledge structure. 

- **Fuzzy Semantic Matching** : In cases of relation symbol mismatch caused by SAGB, the search mechanism compensates via semantic similarity, ensuring successful evidence recall provided that the semantics remain proximate. 

- **Global Structural Guidance** : For semantically distant relations between schema graph and true logic in KG, the Guidance Graphderived consistency bias incorporates global structural priors to prioritize potentially optimal edges, thereby safeguarding the correctness of each search step. 

> 17The correct triple (“European Union”, “organization.organization.founders”, “Belgium”) would not have been selected as the top-ranked edge under a pure semantic matching regime due to the significant semantic divergence between the relations “founders” and “containedby”, which yielded a low similarity score of 0.64. However, by incorporating the consistency bias, the score of this valid edge was successfully boosted to 1.14 and become the final selected edge. 

## **D.8 Efficiency Analysis** 

The STEM framework is designed to minimize reliance on heavy, iterative LLM calls. For a single query, the process involves exactly three distinct LLM inference steps: 

- **Projection** : For both the SGDA and SAGB module, we deploy a locally hosted model respectively. So the projection pipeline necessitates a single inference call of the 8B SGDA model to generate _B_ planning candidates, followed by _B_ forward passes of the 8B SAGB model, the total number of model invocations is (2 _× B_ ). Compared to FiDeLiS, which necessitates ( _BL_ + _L_ + 1) LLM calls (where _L_ denotes reasoning depth), our projection operation scales linearly with the beam size while maintaining optimal performance. This efficiency is comparable to existing highefficiency methods such as LMP (2 _L_ + 1) (Wan et al., 2025) and RoG ( _N_ + 1), where _N_ represents the number of generated relation paths. Moreover, due to the significantly shorter generation length of our assertions and schema graphs, the actual inference latency is further reduced. 

- **Triple-Dependent GNN** : We execute a single forward pass of a 6-layer Triple-Dependent GNN to compute the Guidance Graph. Since node and triple embeddings can be preindexed, the online computational cost is restricted to the propagation of query-specific interaction scores. 

- **Parallel Structure-Tracing Retrieval** : Since the entity selection phase yields multiple entry entities simultaneously, we employ a parallel execution strategy, where search threads originating from all the anchored nodes are conducted concurrently. By parallelizing the retrieval process and merging the resulting subgraphs, we significantly reduce the wall-clock time compared to sequential search methods. Consequently, the overall time cost of subgraph retrieval takes approximately 1.5 to 2 times longer than retrieval from a single entry node. 

- **Answer Generation** : Once the evidence subgraph is retrieved and linearized, the generator LLM is invoked once to produce the final response. 

|**Question**|which airport to fy into rome|
|---|---|
|**Assertions**|1. ("rome’s nearby airport is [ENT1]",)<br>2. ("the airport near rome is [ENT1].",)<br>3. ("rome is served by a nearby airport, [ENT1].",)<br>4. ("[ENT1] is a nearby airport for rome.",)|
|**Strategy**|Breadth|
|**Schema Graphs**|1. [("rome", "location.location.nearby_airports", "[ENT1]")]|
|**Retrieved**|1. [("Rome", "location.location.nearby_airports", "Ciampino–G. B. Pastine International<br>Airport")]<br>2. [("Rome","location.location.nearby_airports","Leonardo da Vinci–Fiumicino Airport")]|
|**Ground Truth (2**<br>**items)**|Ciampino–G. B. Pastine International Airport, Leonardo da Vinci–Fiumicino Airport|
|**Output Answer**|Ciampino - G. B. Pastine International Airport and Leonardo da Vinci – Fiumicino Airport.|



Table 17: Case study C1: Interpretability analysis on the WebQSP dataset. 

|**Question**|what county is texarkana arkansas in|
|---|---|
|**Assertions**|1. ("texarkana arkansas is a country of [ENT1]",)<br>2. ("texarkana, arkansas is a country within [ENT1].",)<br>3. ("texarkana arkansas is part of the country [ENT1].",)<br>4. ("the country to which texarkana arkansas belongs is [ENT1].",)|
|**Strategy**|Precision|
|**Schema Graphs**|1. [("texarkana arkansas", "location.location.containedby", "[ENT1]")]<br>2. [("texarkana arkansas", "location.hud_county_place.county", "[ENT1]")]<br>3. [("texarkana arkansas", "location.administrative_division", "[ENT1]")]|
|**Retrieved**|1. [("Beech Street Historic District", "location.location.containedby", "Texarkana,<br>Arkansas")]<br>2. [("texarkana, arkansas", "location.hud_county_place.county", "Miller County")]<br>3. [("Arkansas","location.administrative_division.country","United States of America")]|
|**Ground Truth**|Miller County|
|**Output Answer**|Miller County|



Table 18: Case study C2: Interpretability analysis on the WebQSP dataset. 

|**Question**|what style of music did bessie smith perform|
|---|---|
|**Assertions**|1. ("bessie smith’s music genre is [ENT1]",)<br>2. ("the music genre of bessie smith is [ENT1].",)<br>3. ("bessie smith’s genre of music is [ENT1].",)<br>4. ("[ENT1] is the music genre associated with bessie smith.",)|
|**Strategy**|Precision|
|**Schema Graphs**|1. [("bessie smith", "music.artist.genre", "[ENT1]")]|
|**Retrieved**|1. [("Bessie Smith", "music.artist.genre", "Jazz")]|
|**Ground Truth**|Jazz|
|**Output Answer**|Jazz|



Table 19: Case study C3: Interpretability analysis on the WebQSP dataset. 

|**Question**|What educational institution with men’s sports team named Wisconsin Badgers did Russell<br>Wilson go to?|
|---|---|
|**Assertions**|1. ("Wisconsin Badgers is a school sports team of [ENT1].", "Russell Wilson’s educational<br>institution is [ENT1].")<br>2. ("The school sports team known as the Wisconsin Badgers belongs to [ENT1].", "The<br>educational institution that Russell Wilson attended is [ENT1].")<br>3. ("[ENT1]’s offcial school sports team is called the Wisconsin Badgers.", "Russell Wilson’s<br>educational institution is [ENT1].")<br>4. ("[ENT1] is the institution that felds the Wisconsin Badgers sports team.", "Russell<br>Wilson received his education at [ENT1].")|
|**Strategy**|Precision|
|**Schema Graphs**|1.[("Wisconsin Badgers", "sports.sports_league.teams", "[ENT1]"), ("Russell Wilson", "edu-<br>cation.education.institution", "[ENT1]")]<br>2.[("Wisconsin Badgers", "sports.school_sports_team.team", "[ENT1]"), ("Russell Wilson",<br>"education.education.institution", "[ENT1]")]<br>3.[("Wisconsin Badgers", "sports.sports_league.teams", "[ENT1]"), ("[ENT1]", "edu-<br>cation.education.student","Russell Wilson")]|
|**Retrieved**|1.[("Wisconsin Badgers men’s basketball", "sports.school_sports_team.school", "University<br>of Wisconsin-Madison"), ("Russell Wilson", "education.education.institution",<br>"University of Wisconsin-Madison")]<br>2.[("Wisconsin Badgers", "education.athletics_brand.teams", "Wisconsin Badgers women’s<br>ice hockey") , ("University of Wisconsin-Madison",<br>"education.educational_institution.sports_teams", "Wisconsin Badgers women’s ice<br>hockey")]<br>3.[("m.0hpny0z", "education.education.student", "Russell Wilson"),<br>("m.0hpny0z","education.education.degree","Bachelor of Arts")]|
|**Ground Truth**|University of Wisconsin-Madison|
|**Output Answer**|University of Wisconsin-Madison|



Table 20: Case study C4: Interpretability analysis on the CWQ dataset. 

|**Question**|What actor played the a kid in the movie with a character named Jenny’s Father?|
|---|---|
|**Assertions**|1. ("Jenny’s father is a movie character in [ENT1].", "[ENT2] performs a role in the<br>production [ENT1].")<br>2. ("Jenny’s father is a character in [ENT1].", "[ENT2] appears as an actor in [ENT1].")<br>3. ("Jenny’s father is a character in movie [ENT1].", "[ENT2] is a character in [ENT1].",<br>"[ENT3] portrayed [ENT2] in the flm.")|
|**Strategy**|Precision|
|**Schema Graphs**|1.[("Jenny’s Father", "flm.performance.character", "[ENT1]"), ("[ENT2]",<br>"flm.performance.actor", "[ENT1]")]<br>2.[("Jenny’s Father", "flm.flm_character.portrayed_in_flms", "[ENT1]"), ("[ENT2]",<br>"flm.flm_character.portrayed_in_flms", "[ENT1]"), ("[ENT2]",<br>"flm.performance.actor","[ENT3]")]|
|**Retrieved**|1.[("m.0y54dnx", "flm.performance.character", "Jenny’s Father"),<br>("m.0y54dnx","flm.performance.actor","Kevin Mangan")]<br>2.[("Jenny’s Father", "flm.flm_character.portrayed_in_flms", "Forrest Gump"), ("Forrest<br>Gump", "flm.flm_character.portrayed_in_flms", "m.02xgww5"), ("m.02xgww5",<br>"flm.performance.actor","Michael Connor Humphreys")]|
|**Ground Truth**|Michael Connor Humphreys|
|**Output Answer**|Michael Connor Humphreys|



Table 21: Case study C5: Interpretability analysis on the CWQ dataset. 

|**Question**|People from the country that contains Corfu speak what language?|
|---|---|
|**Assertions**|1. ("Corfu’s offcial language is [ENT1].",)<br>2. ("Corfu is belong to [ENT1].", "[ENT1]’s offcial language is [ENT2].")<br>3. ("Corfu is an administrative division of [ENT1].","[ENT1]’s offcial language is [ENT2].")|
|**Strategy**|Breadth|
|**Schema Graphs**|1.[("Corfu", "location.country.offcial_language", "[ENT1]")]<br>2.[("Corfu", "location.location.containedby", "[ENT1]"), ("[ENT1]",<br>"location.country.offcial_language", "[ENT2]")]<br>3.[("Corfu", "location.administrative_division.country", "[ENT1]"), ("[ENT1]",<br>"location.country.offcial_language","[ENT2]")]|
|**Retrieved**|1.[("Corfu", "location.administrative_division.country", "Greece"), ("Greece",<br>"location.country.languages_spoken", "Albanian language")]<br>2.[("Corfu", "location.administrative_division.country", "Greece"), ("Greece",<br>"location.country.offcial_language", "Greek Language")]<br>3.[("Corfu", "location.location.containedby", "Corfu Island"), ("Corfu Island",<br>"common.topic.article","m.0cc3p")]|
|**Ground Truth (2**<br>**items)**|Albanian language, Greek Language|
|**Output Answer**|Albanian and Greek language|



Table 22: Case study C6: Interpretability analysis on the CWQ dataset. 

|**Question**|What European Union country is home to the capital city of Brussels?|
|---|---|
|**Assertions**|1. ("[ENT1]’s capital city is Brussels", "European Union contains [ENT1].")<br>2. ("The capital cities of [ENT1] are Brussels.", "The European Union is composed of<br>[ENT1].")<br>3. ("Brussels serves as the capital city for [ENT1].", "The member states of the European<br>Union are [ENT1].")<br>4. ("Brussels is the capital city of [ENT1]","European Union contains [ENT1].")|
|**Strategy**|Precision|
|**Schema Graphs**|1. [("Brussels", "location.administrative_division.capital", "[ENT1]"]), ("[ENT1]",<br>"location.location.containedby", "European Union")]<br>2. [("Brussels", "location.location.containedby", "[ENT1]"]), ("[ENT1]",<br>"location.location.containedby", "European Union")]<br>3. [("Brussels", "location.administrative_division.capital", "[ENT1]"]), ("[ENT1]",<br>"organization.membership_organization.members", "European Union")]<br>4. [("Brussels", "location.administrative_division.capital", "[ENT1]"]), ("[ENT1]",<br>"location.location.containedby","European Union")]|
|**Retrieved**|1. [("European Union", "organization.organization.founders", "Belgium"), ("Brussels",<br>"location.administrative_division.capital", "Belgium")]<br>2. [("European Union", "organization.membership_organization.members", "France"),<br>("Paris","location.administrative_division.capital","France")]|
|**Ground Truth**|Belgium|
|**Output Answer**|Belgium|



Table 23: Case study C7: Interpretability analysis on the CWQ dataset. 

A critical factor influencing the execution efficiency of STEM is the subgraph search mode, which is determined by the answer strategy. Consequently, we conduct a focused inference analysis of different search mode. All experiments were conducted on a single NVIDIA H100 GPU. 

Given the close correlation between answer number and search modes, we stratify the test samples by answer number and evaluate the inference efficiency for each group independently. We first present the strategy generation accuracy of the SGDA module in Table 15. The results demonstrate that the SGDA module exhibits strong performance in strategy discrimination, achieving an accuracy of over 90% across all dataset splits. 

Subsequently, we present a comparison of inference efficiency grouped by answer counts in Table 14, which indicates that inference latency increases significantly as the number of answers grows. Notably, when the answer count exceeds 10, the inference time rises to over 9 seconds. This is attributed to the fact that a higher volume of answers corresponds to a greater number of matched KG edges aligning with the schema graph. 

We further evaluated the test set by stratifying samples based on different answer strategies, with results shown in Table 16. Although inference latency is significantly higher in _Breadth_ mode than in _Precision_ mode, the substantial improvement in F1 score with _Breadth_ mode (Section D.1) justifies this trade-off for more comprehensive answers. We consider the moderate latency increase acceptable, especially given that real-world deployment efficiency is expected to improve with advancing technologies. 

**Pruning Strategy.** The threshold-based search mode incorporates a pruning mechanism to prevent computational explosion. Taking Case C6 (Table 22) as an example, the reasoning chain requires traversing from “Corfu” to “[ENT1]” and subsequently to “[ENT2]”. The retrieval results reveal that during the transition from “Corfu” to “[ENT1]”, the threshold filtering mechanism retained only a single unique edge, effectively converging to one path. Parallel branching emerged only during the subsequent expansion from “[ENT1]” to “[ENT2]” to capture all valid answers. This demonstrates that parallel paths are triggered specifically when necessary to cover multiple answers, thereby significantly preserving computational efficiency. 

**Efficiency Comparison with Baselines.** We 

|**Method**|**Type**|**WebQSP **|**CWQ**|
|---|---|---|---|
|**FiDeLiS**<br>**PoG**<br>**RoG**<br>**GNN-RAG**<br>**STEM(Ours)**|Interactive<br>Interactive<br>Generation<br>Retrieval<br> Generation|34.97<br>14.28<br>3.75<br>2.64<br>5.77|40.16<br>14.08<br>4.31<br>3.81<br>5.40|



Table 24: Comparison of average inference latency across different methods. Values denote the average wall-clock time (s) required to process a single query. 

compare our proposed method against existing baselines to demonstrate its advantages and superiority. To better contextualize the inference latency, we first categorize the existing approaches into three paradigms: (1) Interactive methods involve iterative LLM calls during the retrieval process. (2) Generation-based methods require constant number of LLM calls upfront for path or plan generation. (3) Retrieval-based methods rely entirely on GNNs for graph context encoding. The results are shown in Table 24. 

Empirical results demonstrate that STEM is substantially faster than interactive baselines. Specifically, it achieves a nearly a 6-fold speedup over FiDeLiS (5.77s vs. 34.97s) and operates 2.5 times faster than PoG (5.77s vs. 14.28s). This validates our claim that the offline Structure-Tracing paradigm effectively circumvents the computational bottleneck of repetitive LLM reasoning. While STEM is marginally slower than RoG (which generates linear relation paths) and GNNRAG (which relies on node-level scoring), it delivers vastly superior retrieval accuracy, representing a highly favorable trade-off between efficiency and effectiveness. 

## **D.9 Interpretability Analysis** 

STEM offers a transparent, interpretable workflow rooted in two key structural mechanisms: 

- **Explicit Logical Blueprinting** : By leveraging the linguistic and reasoning capabilities of LLMs within a KG-constrained framework, STEM explicitly visualizes the reasoning process. The SGDA transforms the ambiguity of natural language into a sequence of logical atomic relational assertions, which the SAGB then projects into a concrete schema graph. The resulting blueprint is not merely a semantic abstraction but a topologically valid plan aligned with the latent knowledge structures of the target KG. This allows us to directly 

inspect and verify the model’s decomposition logic before any retrieval occurs. 

- **Guided Structural Tracing** : Building on this blueprint, the Structure-Tracing Retrieval mechanism enables global matching based on the logical subgraph structure, directly mapping the query’s structural features onto the KG knowledge. The schema graph acts as a navigational chart, while the Guidance Graph derived from the Triple-GNN serves as a soft structural bias. This dual-guidance system ensures that the search process is not driven solely by one-sided semantic matching—which is prone to deviation—but is anchored by global structural hypotheses. 

## **E Analysis of Failure Modes and Error Propagation** 

Given the sequential architecture of STEM—where the outputs of the SGDA and SAGB modules serve as inputs for the Triple-GNN and subsequent retrieval—inaccuracies at any upstream stage can inevitably propagate downstream. To better understand this cascading effect, this section presents a systematic analysis of failure modes and error accumulation across the STEM pipeline. 

Specifically, we construct two error attribution matrices on WebQSP to reveal how errors at different stages affect the final QA performance. These matrices capture the cascading effects of two critical intermediate phases: (1) Schema Generation Correctness (SGDA & SAGB) versus Final QA Accuracy, and (2) Retrieved Evidence Subgraph Correctness versus Final QA Accuracy. We provide a detailed analysis of each dimension below. 

**Schema Generation Correctness vs. QA Accuracy** : We constructed an error attribution matrix to evaluate the correlation between Schema Generation Correctness (produced by the SGDA and SAGB pipeline) and the final QA Accuracy. The results are presented in Table 25. 

The results show that the highest proportion (85.24%) occurs when both planning and QA are correct, demonstrating the effectiveness of the query planning modules. The 8.67% where planning is correct but QA fails likely stems from retrieval errors or the LLM not utilizing the correct evidence. In 5.91% of cases, planning fails yet QA succeeds—almost entirely attributable to the LLM’s parametric knowledge. The matrix reveals that incorrect schema graphs account for 

|_Gsch_**match**|**Final QA Output**|
|---|---|
||Correct<br>Incorrect|
|**Valid**<br>**Invalid**|85.24<br>8.67<br>5.91<br>1.7|



Table 25: Schema Generation Correctness vs. QA Accuracy Matrix. Rows indicate whether the schema graph produced by the SGDA and SAGB pipeline successfully matches the ground-truth reasoning graph, while columns represent whether the final generated answer is correct. All values are reported as percentages (%). 

only 16.4% of total QA failures (0.017/0.1037). This indicates that the vast majority of errors originate in downstream stages—namely, during subgraph retrieval and the LLM’s final answer generation—rather than from upstream planning. 

**Retrieved Evidence Subgraph Correctness vs. QA Accuracy** : We analyze the correlation between retrieval correctness and final answer correctness. The results are shown in Table 26. 

The proportion of cases where retrieval is correct and the answer is correct reaches 83.01%, reflecting the effectiveness of the retrieval algorithm and its contribution to question answering. In 4.12% of cases, retrieval is correct but the answer is incorrect, which is primarily attributable to the LLM’s failure in contextual answer extraction. Cases where retrieval is incorrect yet the answer is correct account for 7.34%, largely due to the LLM’s parametric knowledge. Finally, instances where both retrieval and QA fail constitute 5.53%. From this analysis, we consider the 83.01%—where both retrieval and QA are correct—as the true reflection of STEM’s capability, demonstrating the robustness and accuracy of its retrieval mechanism and its positive impact on final answer generation. Further analysis reveals that retrieval failures account for 57.3% (0.0553/0.0965) of all incorrect answers. This indicates that flawed evidence retrieval is the primary source of downstream errors, with the remaining 42.7% attributable to inherent LLM hallucinations during the final generation phase. 

## **F Detailed Mechanism of the Semantic-to-Structural Projection and Structural Pattern Acquisition** 

To address potential ambiguities regarding how our method maps a natural language text to schema graph, we provide a detailed walkthrough of the Semantic-to-Structural Projection pipeline. This 

|_G_reason**match**|**Final QA Output**|
|---|---|
||Correct<br>Incorrect|
|**Valid**<br>**Invalid**|83.01<br>4.12<br>7.34<br>5.53|



Table 26: Retrieved Evidence Subgraph Correctness vs. QA Accuracy. The rows correspond to the correctness of the retrieved evidence subgraph, while the columns represent the correctness of the final answer. All values are reported as percentages (%). 

pipeline, comprising the SGDA and SAGB modules, operates as an end-to-end generative translation process, converting complex raw queries into a set of triples that constitute the schema graph. 

To illustrate this workflow concretely, consider the query: “where is the fukushima daiichi nuclear plant located”, it’s processed through the following two stages: 

**Stage 1: SGDA Decomposition** The SGDA module first decomposes the raw text query into a set of atomic relational assertions, ultimately yielding: 

"The fukushima daiichi nuclear power plant is contained by [ENT1]." 

Here, [ENT1] serves as a structural placeholder for the unknown answer or intermediate entity, simplifying the subsequent mapping task. 

**Stage 2: SAGB Alignment** To perform the symbolic mapping, the SAGB module autoregressively translates the input atomic assertions into the corresponding valid KG triples: 

- ("The fukushima daiichi nuclear power plant", "location.location.containedby", "[ENT1]") 

This process successfully translates highly variable natural language into strict KG-oriented structures by leveraging the LLM’s reasoning capabilities. Since this structural generalization capability is explicitly instilled through our training regimen, we next provide concrete examples to illustrate how it is achieved. 

**Generalization over Training** Importantly, this Semantic-to-Structural Projection learns underlying alignment patterns rather than merely memorizing specific facts. During the fine-tuning phase, the 

training data for SGDA and SAGB contain structurally similar alignments, such as those illustrated in Figures 9 and 10. 

## **Example (1)** 

**Query:** In which country is Kagoshima Prefecture located? **Atomic Relational Assertions:** ("Kagoshima Prefecture is contained by [ENT1]",) **Schema Graph:** [("Kagoshima Prefecture", "location.location.containedby", "[ENT1]")] **Answer:** Japan 

Figure 9: SDGA & SAGB Training Data Example (1). 

## **Example (2)** 

**Query:** Which city in Aomori Prefecture was affected by the 2011 Tohoku earthquake? **Atomic Relational Assertions:** ("[ENT1] is contained by Aomori Prefecture.", "[ENT1] experienced the event of the 2011 T¯ohoku earthquake and tsunami") **Schema Graph:** [("[ENT1]", "location.location.containedby", "Aomori Prefecture"), ("[ENT1]", "location.location.events", "2011 T¯ohoku earthquake and tsunami")] **Answer:** Tohoku 

Figure 10: SDGA & SAGB Training Data Example (2). 

After capturing these schema patterns, the pipeline can effectively generalize to structurally similar assertions (e.g., “X is located in Y”) across different entities. This generative design enables STEM to perform robust, structure-aware schema alignment, circumventing the rigidity and out-ofvocabulary issues typical of traditional step-wise path search or dictionary-based matching methods. 

## **G Prompt List** 

To ensure the robustness and reproducibility of STEM, we detail the core prompts utilized across the different stages of our pipeline. These encompass the processing prompts for the SGDA and SAGB modules, the generation prompt for the QA model, the assertion synthesis prompt for training data construction, the prompt for determining the response strategy, and notably, the core feature of our work: the Structure-to-Query Reverse Generation data synthesis prompt, which significantly enhances the model’s structural generalization capabilities. We present the complete instructional content of each prompt in Figures 11 through 16. 

## **G.1 Schema-Aligned Question Decomposition Prompt** _P_ 1 

## **Schema-Aligned Question Decomposition Prompt (** _P_ 1 **)** 

You are a multi-hop question decomposition expert specialized in knowledge graph-based question answering. Your task is to decompose an input multi-hop question into a sequence of single-hop assertions, and returns the answer strategy required to answer the query. The specific requirements are as follows: 

- 1. Decomposition must be entity-centric. Each single-hop assertion should correspond to a pair of entities and describe the relationship between them. 

- 2. Every single-hop assertion generated should contribute to answering the original multi-hop question. 

- 3. For entity references—including the final answer or any intermediate answer entities—you must label them with [ENT1], [ENT2], etc., and ensure that: 

- The same entity is consistently referred to with 

- the same label across all single-hop assertions. · Different entities are assigned distinct labels. 

- 4. Answering Strategy: If you determine that the answer to the current question is exclusive and deterministic, please return the strategy “Precision”. If you assess that the question involves multiple distinct answers (including final or intermediate answers), please return the strategy “Breadth”. You must select your strategy strictly from “Precision” and “Breadth”. 

Your response should be a single result consisting of the planned single-hop assertion (s) along with the strategy. **Your output format** : 

(Assertion_1, Assertion_2, Assertion_3), Strategy) 

## **G.2 Schema Graph Construction Prompt** _P_ 2 

## **Schema Graph Construction Prompt (** _P_ 2 **)** 

You are an entity-relationship construction expert who has memorized a rich and professional knowledge graphoriented semantic and logical structure. Based on your mastered graph structure data, you can construct appropriate entity-relationship triples for given single-hop assertions. 

Since these assertions are decomposed from a multi-hop query, you must fully consider their interdependencies when returning the triples. 

The specific requirements are as follows: 

- 1. Based on the given assertions and combined with your understanding of knowledge graph data, you must generate structural triples that best match the meaning expressed. The entities and relationship descriptions in the triples must be consistent with the meaning of the assertions. 

- 2. If an assertion contains an entity placeholder like [ENTX], you must copy it exactly as is when converting it into a triple. Do not alter the content of the placeholder. If there is no [ENTX] label in an assertion, generate the most relevant triple based on the described entities and relationship. 

## **Your output format** : 

[ (Entity_1, Relation_1, Entity_2), (Entity_2, Relation_2, Entity_3), ... 

] **Example:** [EXAMPLE_1] [EXAMPLE_2] 

**Input single-hop assertions:** [Assertion_1, Assertion_2, ...] 

## **Example:** 

[EXAMPLE_1] [EXAMPLE_2] [EXAMPLE_3] 

## **Input multi-hop question:** 

[Query] Please now plan the decomposition for the given question. You must strictly follow the requirements above. 

Figure 12: The prompt template for Schema Graph Construction ( _P_ 2). 

## **G.3 Generation Prompt** _P_ 3 

## **Generation Prompt (** _P_ 3 **)** 

Figure 11: The prompt template for Schema-Aligned Question Decomposition ( _P_ 1). 

Based on the knowledge structure graph, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list. 

**Knowledge Structure Graph** : [Knowledge Structure Graph] **Question** : [Question] **Answer** : [Answer] 

Figure 13: The prompt template for Generation ( _P_ 3). 

## **G.4 Path-based Assertions Generation Prompt** _P_ 4 

## **Path-based Assertions Generation Prompt (** _P_ 4 **)** 

Please construct appropriate declarative assertions for each given triple based on the provided logical triple list and original query. The requirements are as follows: 

- 1. The format of each given triple is (“entity1”, “relation”, “entity2”). You need to generate a suitable declarative assertion based on the meanings of entity1 and entity2 and the relationship between them. 

- 2. If an entity is in the label format such as [ENTX], the corresponding entity in the generated sentence should also be written in the same label format. If multiple triples contain the same label, for example, all are [ENT1], you must ensure consistency in the generated assertions and avoid modifying the label content arbitrarily. 

- 3. The number of assertions you return must match the number of triples in the given list, and they should correspond one-to-one. 

- 4. The assertions you generate must serve as important evidence for answering the given original query, meaning that answering the query requires referencing these assertions. 

## **Your output format** : 

[ Assertion_1, Assertion_2, ... ] **Example:** [EXAMPLE_1] [EXAMPLE_2] [EXAMPLE_3] 

## **G.5 Response Strategy Generation Prompt** _P_ 5 

## **Response Strategy Generation Prompt (** _P_ 5 **)** 

Please determine the appropriate retrieval strategy for the given question when used for multi-hop retrievalaugmented generation in a knowledge graph context. There are two available strategies: “Precision” and “Breadth”, described as follows: 

- 1. “Precision” is primarily used for answering questions that have a exclusive and deterministic answer, such as “In which year was Edison born?” Questions of this type typically have only one correct answer, and contradictions may arise if more than two answers are provided. 

- 2. “Breadth” is mainly used for answering questions involving multiple distinct answers, such as “What country is Russia close to?”, this type of question requires retrieving all qualifying answers to ensure comprehensive answer recall. 

- 3. You will be given a list of assertions representing the decomposed declarative statements derived from the given question. These assertions constitute the multi-hop logical decomposition of the question. You may reference them to get more in-depth guidance. 

Please remember, your response should contain only the word “Precision” or “Breadth”, with no additional explanatory content! 

**Example:** [EXAMPLE_1] [EXAMPLE_2] [EXAMPLE_3] 

**Question** : [Question] **Assertions** : [Assertion_1, Assertion_2, ...] 

**Original Query** : [Query] **Given Triple List** : [Triple_1, Triple_2, ...] 

Figure 15: The prompt template for Response Strategy Generation ( _P_ 5). 

Figure 14: The prompt template for Path-based Assertions Generation ( _P_ 4). 

## **G.6 Query & Assertions Generation based on Sampled-Graph Prompt** _P_ 6 

## **Query & Assertions Generation based on SampledGraph Prompt (** _P_ 6) 

Please construct appropriate declarative sentences for each given triple based on the provided triple list. The requirements are as follows: 

- 1. The format of each given triple is (“entity1”, “relation”, “entity2”). You need to generate a suitable declarative sentence based on the meanings of entity1 and entity2 and the relationship between them. 

- 2. If an entity is in the label format such as [ENTX], the corresponding entity in the generated sentence should also be written in the same label format. If multiple triples contain the same label, you must ensure consistency in the generated sentences and avoid modifying the label content arbitrarily. 

## **H Structure-Tracing Subgraph Retrieval** 

Algorithm 1 and 2 illustrate the overall execution and helper functions of Structure-Tracing Subgraph Retrieval, encompassing the execution procedures for two search modes. Descriptions of the key helper functions are provided below: 

   - CONTRADICT: Determines whether a specific triple _t_ already exists within the list of currently matched triples. 

   - GET_TAIL: Given a triple _t_ and one of its constituent entities _e_ , retrieves the complementary entity node. 

   - GET_N: Retrieves all incident edges (not differentiate edge direction) of entity _e_ within and _G_ returns them as a list of triples. 

   - BUILD_GRAPH: Constructs a graph structure from a list of triples. 

- 3. The number of sentences you return must match the number of triples in the given list, and they should correspond one-to-one. 

After constructing the declarative sentences, proceed to generate a multi-hop question based on them. You must adhere to the following requirements: 

- 1. Given the specific placeholder [ENTX] designated as the answer entity, your generated question must target [ENTX] as its final answer. 

- 2. If the sentences contain multiple distinct placeholders, treat all placeholders other than the target [ENTX] as intermediate answers. You must integrate them into the multi-hop question as modifiers, relative clauses, or nested constraints. 

- 3. The generated multi-hop question must be constructed such that answering it requires referencing all the provided declarative sentences. 

- 4. If you determine that the declarative sentences cannot yield an unambiguous multi-hop question that strictly relies on every sentence, simply output: “No Solution”. 

**Your output format** : [(Sentence_1, Sentence_2, ...), Multi-Hop Question] 

## **Example:** 

[EXAMPLE_1] [EXAMPLE_2] [EXAMPLE_3] 

**Given Triple List** : [Triple_1, Triple_2, ...] **Answer Entity** : [ENTX] 

Figure 16: The prompt template for Query & Assertions Generation based on Sampled-Graph ( _P_ 6). 

## **Algorithm 1** Structure-Tracing Subgraph Retrieval 

1: **Input:** Pattern graph _Gsch_ = ( _NQ, RQ_ ); Knowledge Graph _G_ ; Topic entity _e ∈Gsch_ ; Globally-aware entity score _Se[∗]_[; Retrieval strategy] _[ σ]_[; Confidence threshold] _[ θ]_[.] 

2: **Output:** Matched subgraph _Gsch[∗]_[.] 

3: _e[∗] ←_ Identify the matching entity node in _G_ corresponding to _e_ in _Gsch_ 4: _S_ 0 _← Se[∗]_[[] _[e][∗]_[];] _final_  list ←∅_ ; _last_  visit ←_ None _▷_ Initialization 5: _Tall ←_ MATCH( _G, Gsch, S_ 0 _, e, e[∗] , final_  list, last_  visit, σ, θ_ ) 6: _Gsch[∗][←]_[B][UILD][_G][RAPH][(] _[T][all]_[)] 7: **return** _Gsch[∗]_ 8: **function** CONTRADICT( _t, final_  list_ ) 9: **if** _t ∈ final_  list_ **then return** TRUE 10: **else return** FALSE 11: **function** GET_N( _e, G_ ) 12: _triples ←_ Fetch all incident edges for entity _e_ in _G_ as a triple list 13: **return** _triples_ 14: **function** MATCH( _G, Gsch, S, e, e[∗] , final_  list, last_  visit, σ, θ_ ) 15: **for each** _t ∈_ GET_N( _e, Gsch_ ) **do** 16: **if** _last_  visit_ = _t_ **then continue** 17: **if** _σ_ = "Precision" **then** 18: _step_  list ←_ STEP_PRECISION( _G, Gsch, S, e, e[∗] , t, final_  list, σ_ ) 19: **if** _σ_ = "Breadth" **then** 20: _step_  list ←_ STEP_BREADTH( _G, Gsch, S, e, e[∗] , t, final_  list, σ, θ_ ) 21: _final_  list ← final_  list ∪ step_  list_ 22: **return** _final_  list_ 

## **Algorithm 2** Subgraph Search Single Step Functions 

1: **function** STEP_PRECISION( _G, Gsch, S, e, e[∗] , t, final_  list, σ_ ) 2: _Ne[∗] ←_ GET_N( _e[∗] , G_ ) 3: _max_  score ←−_ 1 4: _max_  t ←_ None 5: **for each** _t[∗]_ in _Ne[∗]_ **do** 6: **if** CONTRADICT( _t[∗] , final_  list_ ) **then** 7: **continue** 8: _S[′] ← S_ + T-SCORE( _t[∗] , t_ ) 9: **if** _S[′] > max_  score_ **then** 10: _max_  score ← S[′]_ 11: _max_  t ← t[∗]_ 12: _final_  list ← final_  list ∪{max_  t}_ 13: _et ←_ GET_TAIL( _t, e_ ) 14: **if** _max_  t_ = None **then** 15: _e[∗] t[←]_[G][ET][_T][AIL][(] _[max]_[_] _[t][, e][∗]_[)] 16: _step_  list ←_ MATCH( _G, Gsch, max_  score, et, final_  list, t, σ,_ None) 17: _final_  list ← final_  list ∪ step_  list_ 18: **return** _final_  list_ 19: **function** STEP_BREADTH( _G, Gsch, S, e, e[∗] , t, final_  list, σ, θ_ ) 20: _Ne[∗] ←_ GET_N( _e[∗] , G_ ) 21: _candidates ←∅_ 22: **for each** _t[∗]_ in _Ne[∗]_ **do** 23: **if** CONTRADICT( _t[∗] , final_  list_ ) **then** 24: **continue** 25: **if** T-SCORE( _t[∗] , t_ ) _≥ θ_ **then** 26: _S[′] ← S_ + T-SCORE( _t[∗] , t_ ) 27: _candidates ← candidates ∪{_ ( _S[′] , t[∗]_ ) _}_ 28: _final_  list ← final_  list ∪{t[∗] }_ 29: _et ←_ GET_TAIL( _t, e_ ) 30: **for each** ( _score, t[∗]_ ) in _candidates_ **do** 31: _e[∗] t[←]_[G][ET][_T][AIL][(] _[t][∗][, e][∗]_[)] 32: _step_  list ←_ MATCH( _G, Gsch, score, et, et[∗][,][ final]_[_] _[list][, t, σ, θ]_[)] 33: _final_  list ← final_  list ∪ step_  list_ 34: **return** _final_  list_ 

