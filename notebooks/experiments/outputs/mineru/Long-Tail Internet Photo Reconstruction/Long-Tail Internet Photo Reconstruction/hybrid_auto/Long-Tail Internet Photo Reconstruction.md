# Long-Tail Internet Photo Reconstruction

Yuan Li $^{1}$ Yuanbo Xiangli $^{1\dagger}$ Hadar Averbuch-Elor $^{1}$ Noah Snavely $^{1}$ Ruojin Cai $^{2\dagger}$ $^{1}$ Cornell University $^{2}$ Kempner Institute, Harvard University

![](images/329388b7f29208e2945bcaea425ffd2aa1c8c70e35fb20db478526c20366b1c9.jpg)

<details>
<summary>line</summary>

| Scene | Registered images | Total images |
|-------|-------------------|--------------|
| Duomo (Cagliari) - Crypt | 7.0k | 10k |
| Ours | 82.7k | 10k |
| Pretrained π³ | 425.0k | 10k |
| Ours | 425.0k | 10k |
| Calvaire de Plougonven | 425.0k | 10k |
</details>

Figure 1. Long-tail Internet photo reconstruction. Internet photo collections follow a long-tailed distribution. In the top plot, the $x$ -axis represents scene index (sorted by image count) and the $y$ -axis shows images per scene (scenes are drawn from MegaScenes [36], a dataset of Internet photo collections). The light blue curve plots the total number of Internet photos per scene, while the steel blue curve shows the size of the subset of photos that were successfully registered using SfM. The head of this distribution of photo collections represents well-photographed scenes; here, there are 6,985 scenes with $>50$ registered images. However, most photo collections are in the long tail of this distribution; here, 418,056 scenes with fewer than 50 registered photos. State-of-the-art methods often fail on scenes in this tail. In the lower half of the figure, we show two examples from the long tail, along with representative input images and the corresponding reconstructions. On Calvaire de Plougonven, COLMAP doesn't register any image; on both Duomo (Cagliari)-Crypt and Calvaire de Plougonven, recent feed-forward reconstruction models like $\pi^3$ [44] produce poor results. We propose MegaDepth-X dataset and a strategy for mimicking long-tail camera distributions, on which fine-tuned models like $\pi^3$ exhibit better reconstruction robustness.

# Abstract

Internet photo collections exhibit an extremely long-tailed distribution: a few famous landmarks are densely photographed and easily reconstructed in 3D, while most real-world sites are represented with sparse, noisy, uneven imagery beyond the capabilities of both classical and learned 3D methods. We believe that tackling this long-tail regime represents one of the next frontiers for 3D foundation models. Although reliable ground-truth 3D supervision from sparse scenes is challenging to acquire, we observe that it can be effectively simulated by sampling sparse subsets from well-reconstructed Internet landmarks. To this end, we introduce MegaDepth-X, a large dataset of 3D reconstructions with clean, dense depth, together with a strategy for sampling sets of training images that mimic camera distributions in long-tail scenes. Finetuning 3D foundation models with these components yields robust reconstructions under extreme sparsity, and also enables more reliable reconstruction in symmetric and repetitive scenes, while preserving generalization to standard, dense 3D benchmark datasets. The dataset, finetuned models, and code are available at: https://megadepth-x.github.io/.

# 1. Introduction

Internet photo collections of real-world landmarks follow a long-tailed distribution. A small fraction of famous sites, such as the Colosseum or Notre Dame, are photographed from every conceivable angle and can be accurately reconstructed by standard Structure-from-Motion (SfM) pipelines. Yet the overwhelming majority of landmarks across the world are represented on the Internet with just a handful of sparse, noisy images (Fig. 1). We refer to this large body of scenes as the long-tail of online photo collections. Such scenes are the norm rather than the exception in real-world Internet imagery.

Reconstructing long-tail scenes is challenging. Classic methods, such as COLMAP $[28]$ , often fail because feature correspondence is hard to find across sparse, non-overlapping, or wide-baseline views. Modern learned feedforward models, like DUSt3R $[42]$ and VGGT $[39]$ , can learn powerful priors from millions of images that might help reconstruct long-tail collections. In practice, however, these models are primarily trained on controlled captures with clean, dense, and evenly sampled data. When applied to long-tail Internet scenes featuring sparse, diverse, and unevenly distributed imagery, we find that these models often fail to recover consistent geometry.

We believe that one of the next frontiers for 3D foundation models lies in tackling this long-tail regime of Internet photos. Better data is almost certainly key to this problem, but we cannot easily construct reliable 3D supervision from long-tail collections themselves, as most contain too few overlapping views for robust reconstruction. Instead, we propose to simulate such long-tailed sets by appropriate sampling of sparse images from the large, well-reconstructed Internet landmarks at the head of the distribution, inheriting ground truth from the full reconstruction.

This strategy requires drawing from large amounts of high-quality landmark reconstructions from Internet photos. Existing datasets fall short of this need: MegaDepth $[20]$ is clean but small, while MegaScenes $[36]$ is large but noisy and lacks depth maps. We therefore introduce MegaDepth-X (dubbed MD-X), a next-generation extension of MegaDepth in both scale (7× larger) and quality: a large-scale, clean, and dense-depth-enhanced dataset built from Internet photo reconstructions with consistent depth refinement and extensive manual verification against reliable references (e.g., Google Maps and satellite imagery). Equipped with MD-X, we propose a novel sparsity-aware sampling strategy that mimics the camera distributions of long-tail scenes, encouraging training batches to span wide baselines and partial overlap rather than clustered dense views.

Through extensive experiments, we show that models fine-tuned with MD-X and our sparsity-aware data sampling scheme are significantly more robust on long-tail Internet photo collections, including challenging doppelganger scenes with ambiguous or symmetric content, such as the Calvaire de Plougonven example in Fig. 1, where classical SfM and pretrained foundation models often fail. In summary, our contributions are:

- Defining the 3D long-tail regime: we formalize and characterize the long-tail distribution of Internet photo collections, highlighting this setting's distinct challenges.   
- MegaDepth-X, dubbed MD-X, a large-scale, clean, and depth-augmented dataset for finetuning 3D foundation models on real-world Internet scenes.   
- Sparsity-aware sampling strategies that simulate the distribution of long-tail Internet collections to improve generalization of 3D prediction models on real-world data.

# 2. Related Work

Feed-forward 3D reconstruction. Reconstructing 3D scene geometry from 2D images is a cornerstone of computer vision. Traditional structure from motion (SfM) [29] and multiview stereo (MVS) [30] methods were crowning achievements of the classic era of 3D vision, and were scaled to large Internet photo collections [1, 12, 34]. Recently, the new paradigm of feed-forward 3D reconstruction has emerged, which involves regressing 3D attributes directly from images in a single pass. Pioneering work in this area, such as DUSt3R, showed success at predicting pixel-aligned point maps from image pairs [42]. MASt3R extended this approach but still relied on pairwise processing [19]. Subsequent efforts focused on scaling these models to arbitrary numbers of views. VGGT [39], along with concurrent models like Fast3R [47] and FLARE [50], introduced large transformer architectures that can process hundreds of views simultaneously. By leveraging large-scale, diverse datasets and a multi-task learning objective, VGGT predicts a full suite of 3D attributes, including camera parameters, depth maps, and point maps. To eliminate reference-frame bias, $\pi^3$ [44] recently proposed a permutation-equivariant architecture that predicts affine-invariant camera poses and scale-invariant local point maps. ZipMap [16] and Scal3R [46] introduced test-time training approaches to process large image collections. These methods work well on densely-captured and well-conditioned scenes. However, we find that their performance on more sparse and noisy Internet photos remains suboptimal, particularly for long-tail scenes.

Long-tail challenges in 3D vision. Long-tailed problems are pervasive in computer vision. They occur when data for common scenarios (the head) are abundant, but examples of rare yet collectively frequent cases (the tail) are scarce. For instance, many object recognition problems involve a few dominant categories but many rarely seen ones, and in autonomous driving, routine driving scenes are plentiful while safety-critical events are hard to capture.

Recently, MegaScenes [36] introduced a large-scale scene-level dataset built from Internet photo collections, where long-tail effects are particularly pronounced. Many scenes in the dataset are either unreconstructed or incorrectly reconstructed. These failures stem from a combination of view sparsity, noisy imagery, and doppelganger issues [7]. Recent work has sought to address such challenges by developing stronger local features [10, 37] and matchers [15, 17, 21, 27], and by learning wide-baseline pose relationships from large-scale 3D datasets [3, 6]. The doppelganger problem was further addressed by Cai et al. [7, 45], who trained classifiers to prune false matches during the structure-from-motion phase of reconstruction.

While these advances have led to enhanced robustness, they do not yet work reliably at scale. Ideally, we'd mine ground truth 3D training data for long tail scenes and learn to reconstruct them, but that involves a chicken-and-egg problem, because the common practice of using available reconstructors (e.g. COLMAP [28, 31], VGGT [39]) to derive pseudo-ground-truth camera poses and point maps from natural data doesn't work. Instead, similar in spirit to approaches used in autonomous driving that augment training data by simulating rare events, our key idea is to take large, well-conditioned image collections and subsample them to simulate long-tailed photo collections, and use these to better balance training scene distributions for regression models in order to generalize to long-tailed scenes.

# 3. The MegaDepth-X Dataset

Learning in the long-tail regime requires high-quality 3D supervision derived from Internet photo collections. This involves two key challenges. First, reconstructions of Internet photo collections can be unreliable due to noise, dynamic content, and ambiguities $[7]$ . Second, most long-tail scenes lack any usable reconstructions, as classical SfM pipelines like COLMAP $[29]$ often fail on sparse or widely varying image sets. To address these issues, we construct MD-X, a large-scale, clean, and depth-refined dataset that provides reliable 3D supervision, built from well-reconstructed scenes in MegaScenes $[36]$ .

# 3.1. Filtering and Disambiguation

Our first step in constructing MD-X is to identify candidate Internet landmarks from which reliable supervision can be derived. We take as our starting pool the subset of MegaScenes with more than 100 registered images, which typically yields stable reconstructions. However, even these “well-reconstructed” scenes exhibit two common failure modes: (1) Many scenes contain dynamic events or crowded activities, causing feature matches to lock onto moving objects rather than static structures, leading to unreliable reconstructions. (2) The Doppelganger problem $[7, 45]$ , where visually similar but geographically distant images are mistakenly registered together. Both issues produce incorrect camera poses and fragmented, inconsistent point clouds as shown in Fig. 2.

To mitigate these issues, we first inspect the dataset and exclude scenes dominated by crowds or moving objects. Next, we address the doppelganger problem by replacing the default COLMAP SfM reconstruction with MASt3R-SfM [19], combined with Doppelganger classification [45]. Specifically, MASt3R-SfM constructs the scene graph using feature matches derived from MASt3R descriptors, after which the Doppelganger classifier identifies and prunes suspicious edges that may result from doppelganger-induced false correspondences. Finally, we manually verify the reconstructed scenes against external references such as Google Maps and satellite imagery, discarding any scenes that do not align with the corresponding bird's-eye view.

![](images/61a25d1d94262a491070c1c42404751d444e999ed9a0c449144c6119b509d2b2.jpg)

<details>
<summary>text_image</summary>

Diagram illustrating four different types of cultural or cultural scenes with labeled panels (a) through (d), each showing images of people, landmarks, and cultural landmarks.
</details>

Figure 2. Unreliable reconstructions in MegaScenes. Reconstructions are unreliable when feature matches are incorrectly established on salient, non-static objects (e.g., (a) humans, (b) statues, (c) airplanes) instead of the static scene structure. This results in fragmented and geometrically inconsistent point clouds. Example (d) illustrates a doppelganger failure, where images from opposite sides of the building are incorrectly registered together.

# 3.2. Dense Depth Refinement

After obtaining reliable sparse reconstructions, we seek to generate dense depth maps for supervision. We start by running a standard multi-view stereo (MVS) [31] pipeline. We observe, as in prior work [20], that the resulting geometric depth maps from in-the-wild collections often exhibit artifacts, including depth-bleeding effects (background depths leak into foreground regions) and inconsistent and noisy depths in areas with transient objects (e.g., people, cars).

To address these initial issues, we apply the full depth refinement strategy from MegaDepth $[20]$ , including a modified MVS procedure that conservatively retains the minimum depth value during propagation, stability filtering to remove flickering pixels, and semantic filtering to exclude transient objects. However, even after this pipeline, we still observe artifacts in the processed geometric depth maps: (1) the MegaDepth-modified MVS still leads to depth-bleeding artifacts, and (2) semantic filtering is not ideal as it relies on a manually designated list of object categories. Examples of such issues are shown in Fig. 3.

![](images/4108dcc56fd8b7db593b8d8e2f7fa5a842b571e4a2a07b74eae705ae5a79f8d2.jpg)

<details>
<summary>text_image</summary>

Image
MegaDepth Refinement
Our Refinement
Transient Object
Depth Bleeding
</details>

Figure 3. Depth refinement. MVS depth maps often suffer from artifacts like noise from transient objects (top row) and depth bleeding (bottom row). As shown in the middle column, the MegaDepth refinement pipeline (modified MVS, stability filtering, and semantic filtering) fails to fully remedy these issues. Our method (right column) introduces an additional monocular depth-guided filtering step, which effectively removes transient objects and significantly mitigates depth-bleeding artifacts.

Therefore, to augment MegaDepth's depth refinement procedure, we propose a monocular depth-guided filtering step. We use depth predictions from MoGe2 [41] as ordinal depth priors, and remove pixels in the processed geometric depth maps that are inconsistent with these priors. Specifically, we first align the processed geometric depths $D_{\mathrm{geom}}$ to the monocular predictions $D_{\mathrm{mono}}$ by matching their median values over valid pixels: $D_{\mathrm{geom}}^{\prime}(p) = s \cdot D_{\mathrm{geom}}(p)$ , where $s = \frac{\mathrm{med}\{D_{\mathrm{mono}}(p) | p \in P\}}{\mathrm{med}\{D_{\mathrm{geom}}(p) | p \in P\}}$ . After scale alignment, we compute the normalized depth discrepancy between the two maps: $\Delta(p) = \frac{|D_{\mathrm{geom}}^{\prime}(p) - D_{\mathrm{mono}}(p)|}{D_{\mathrm{geom}}^{\prime}(p)}$ , and discard pixels whose discrepancies exceed a predefined threshold $\tau_{\mathrm{depth}}$ . Moreover, to leverage $D_{\mathrm{mono}}$ for edge-aware filtering, we compute the discrepancies between the gradients of the two maps: $\Delta(p_{\mathrm{grad}}) = |\frac{|\nabla D_{\mathrm{mono}}|}{D_{\mathrm{mono}}} - \frac{|\nabla D_{\mathrm{geom}}^{\prime}|}{D_{\mathrm{geom}}^{\prime}}|$ and discard pixels whose discrepancies exceed a predefined threshold $\tau_{\mathrm{grad}}$ . This approach effectively filters both bleeding artifacts and noisy transient objects without relying on manual category lists, as depicted in Fig. 3.

# 3.3. Dataset Statistics

In summary, we identify 2,474 candidate scenes from MegaScenes with more than 100 registered images. Of these, 609 scenes are filtered out due to dynamic content, reconstruction errors, or geometric inconsistencies. Our final MD-X dataset comprises 1,865 reconstructions totaling 440k images. We reserve 127 scenes for testing, providing a novel set for evaluating both pretrained and fine-tuned methods. A comparison table with MegaDepth is provided in the supplementary.

# 4. Simulating Long-Tail Scenes

With MD-X providing reliable 3D supervision, the remaining challenge is a complementary supervision coverage problem: existing 3D foundation models are trained predominantly on the head of the Internet-photo distribution, where image collections are large, redundant, and visually well-connected. In this regime, models can rely on strong covisibility and abundant local correspondences. However, most real Internet photo collections lie in the long tail, where views are sparse, unevenly distributed, and only weakly connected. A more complete 3D prior should therefore be robust not only to diverse scene content, but also to this underrepresented observation regime. Rather than seeking unreliable supervision from true long-tail scenes, we start from well-reconstructed scenes in MD-X and sample subsets whose covisibility structure matches that of real long-tail collections. In this way, we expose the model to the missing part of the training distribution while inheriting trustworthy 3D supervision from the full reconstruction.

# 4.1. Defining Properties of Long-Tail Scenes

Common issues like transient occluders and motion blur affect Internet photos broadly, but they are not the primary bottleneck for long-tail scenes. The more fundamental challenge lies in their viewpoint distribution. In these scenes, sparse camera placements lead to limited mutual overlap between images. This results in fragmented, weakly connected clusters rather than a cohesive set, which poses a major hurdle for reliable 3D reconstruction. Because accurate camera poses are often unavailable for such scenes, we characterize this regime using statistics of the SfM view graph rather than absolute camera geometry. Our analysis reveals two consistent patterns: (1) sparser connectivity: scenes with low registration rates (e.g., only 20% of images registered) contain a substantially larger fraction of low-degree nodes, with 8% of cameras having degree two or less, compared with only 3% in well-reconstructed head scenes. This indicates that cameras in long-tail scenes are poorly connected, forming fragmented clusters with limited covisibility. (2) weaker connections: even among connected image pairs, the average number of geometrically verified feature matches is significantly lower in long-tail scenes than in head scenes (294.8 vs. 395.3), indicating reduced overlap and weaker geometric consistency. $^{1}$ Together, these observations show that the long tail is not simply a regime of fewer images, but one of sparse and weakly connected observation graphs.

Based on these findings, our sampling process should satisfy three requirements:

- Viewpoint Diversity: The sampled views should cover a wide range of viewing directions, ensuring that emulated scenes span diverse visual perspectives.   
- Sparsity: The selected views should be far enough apart to mimic the wide baselines typical of long-tail scenes, e.g. loosely connected views or views from disconnected scene components, encouraging the model to learn robust geo-

metric priors rather than relying on dense feature matches.

\- Local Reconstructability: Despite the sparsity, views within each sampled scene component should retain enough covisibility to remain locally reconstructable, since zero-overlap samples within a scene component can lead to unstable training signals and difficult optimization.

# 4.2. Sparsity-Aware Sampling Strategy

We therefore formulate the sampling task as sampling N views that form at most $N_{cc}$ connected components, in order to emulate a long-tail scene with multiple weakly connected or disconnected scene components. Specifically, components are allowed to be disconnected from one another, but within each sampled component we still require sufficient internal covisibility for local reconstructability. We find that naïve random or uniform subsampling often fails to satisfy this balance, producing either zero-overlap sets within scene components or clusters biased toward dense regions. We instead propose a structured sampling process. We first partition views into strongly connected communities and then select a minimal yet diverse subset that ensures both community coverage and global connectivity. This process is illustrated in Fig. 4.

Graph Communities. To promote viewpoint diversity in our sampling, we first identify the dominant “viewing areas” within each scene. We represent the SfM structure as a view graph $G = (V, E)$ , where each node $v_i \in V$ corresponds to a camera view and each edge $(v_i, v_j) \in E$ is weighted by the number of feature matches $w_{ij}$ . We prune edges with $w_{ij} < 50$ to remove minor overlaps, resulting in a filtered graph $G' = (V, E')$ that preserves only meaningful covisibility relationships. To reveal clusters of cameras with dense internal connectivity, we perform community detection (e.g., Louvain community detection [4]) on the view graph. This yields viewpoint groups $C_k$ that efficiently capture distinct visual regions and the dominant perspectives of the scene. We then randomly partition the graph into $N_{cc}$ connected components that span different communities and do the following steps within each graph partition. The partition algorithm is provided in the supplementary material.

Minimal Connectivity Subgraph. To preserve overall scene connectivity while maintaining sparsity and view diversity within limited nodes, we construct a minimal structure linking all identified communities without reintroducing dense redundancy within each partition. We then compute an approximate Steiner tree to link all of these nodes [18, 22]. $^{2}$ In particular, for each training batch for a given training scene, we first randomly select one representative view $v_{k} \in C_{k}$ from each community $C_{k}$ to form the terminal set $T = \{v_{k}\}$ . An approximate Steiner tree

![](images/b9d1a245e2e4b711b5b8b805eaabb5abd62ec6ce087c6c782f07e291f786470f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Graph Communities"] --> B["Run Louvain algorithm (based on #matches)"]
    B --> C["Build Steiner Tree to connect terminal nodes with minimal edges"]
    C --> D["Perform Greedy Search on the subgraph (e.g. after 4 iterations)"]
    D --> E["Example Results (N=24)"]
```
</details>

Figure 4. Sparsity-aware sampling strategy. Top: Our method follows a multi-stage process: (1) Apply the Louvain algorithm to the view graph to identify distinct viewpoint communities. (2) From each community, randomly select a terminal view and construct an approximate Steiner Tree to form a minimal, connected subgraph spanning these communities. (3) Perform a Greedy Search on this subgraph to select a sparse and diverse set of views. This procedure aims to cover as many communities as possible while ensuring a wide spatial distribution of cameras within each community. Bottom: A search depth parameter controls the final view coverage. In this example, we sample N = 24 views from the scene with $N_{cc} = 1$ . With search depth D = 24, all views are selected via greedy search, producing a more evenly spread distribution. With D = 12, 12 views come from greedy search and the remaining 12 are sampled locally from the neighborhoods of selected nodes, resulting in a more concentrated distribution.

algorithm then constructs a minimal connected subgraph $G_{\mathrm{sub}} = (V_{\mathrm{sub}}, E_{\mathrm{sub}})$ , $T \subseteq V_{sub} \subseteq V$ , that spans all terminal nodes using only the necessary intermediate nodes. This yields a compact subgraph connecting all communities using the fewest necessary nodes and edges, preserving global consistency while retaining sparsity. Since $G_{sub}$ can have an arbitrary number of nodes, we need to perform additional sampling to get desired number of views for the training and testing batches.

Greedy View Sampling. Inspired by skeletal sets $[35]$ , we perform greedy view sampling on the subgraph $G_{sub}$ to select a diverse subset of views for long-tail emulation. The objective is to iteratively expand the sampled set toward broad spatial coverage while maintaining sufficient covisibility among selected view pairs.

At each iteration, the algorithm aims to select the next view based on two criteria: (1) Community novelty: preferring cameras that belong to previously unseen communities, thereby introducing new viewing directions and reducing redundancy; and (2) Spatial distance: encouraging selection of cameras farther from the current viewpoint to promote wider baseline coverage. Specifically, the algorithm operates on a current node v and its connected neighborhood $N_{v}$ . Let S denote the set of already sampled nodes and M be the community map. We first determine which communities have already been reached in S, forming the set $S_{comm} = \{M[s] \mid s \in S\}$ . For each neighbor $u \in N_v$ , we then evaluate its community novelty by checking whether $M[u] \notin S_{comm}$ , and compute its spatial distance as $\|\operatorname{Pos}(u) - \operatorname{Pos}(v)\|_2$ , where $\operatorname{Pos}(\cdot)$ is camera position. Details for this algorithm are provided in the supplemental material. All candidate neighbors are ranked lexicographically by these two attributes, and the top-ranked neighbor $u^*$ is selected as the next sampled node. This procedure is repeated for D iterations (i.e., the search depth).

Implementation. In practice, we compute a fixed set of communities $C = \{C_k\}$ for each scene. To form a training batch of N images for a scene, we first randomly divide the N samples across all $N_{cc}$ partitions. In each partition, greedy view sampling stops once either a predefined search-depth limit D is reached or the target number of views assigned to that partition has been sampled. Here, D controls how far the search expands within a partition, hence the sparsity of the resulting set. If this process still produces fewer than N nodes in total, we fill the remaining slots by randomly sampling nodes from the local neighborhoods of the previously sampled nodes. Fig. 4 illustrates an example in which N = 24 and $N_{cc} = 1$ , and shows the different sparsities of the sampled set obtained under different values of D. Before training, we run the proposed sampling algorithm offline to generate mini-batches of 24 nodes, avoiding costly graph loading during training. We then perform depth-first search from random seed nodes to subsample 2 to 24 images for training batches.

# 5. Experiments

We evaluate how our approach improves 3D reconstruction in the long-tail regime of Internet photo collections. First, we show quantitative results on the proposed MD-X benchmark, demonstrating qualitative improvements on real-world long-tail and doppelganger scenes. We then analyze the effect of the proposed dataset and sampling strategy, and finally verify that our fine-tuned models preserve strong performance on standard, curated benchmarks. Further implementation details and additional results are in the supplementary material.

# 5.1. Experimental Setup

Backbones and variants. We finetune two feed-forward 3D foundation models, $\pi^{3}$ [44] and VGGT [39], on MD-X using our proposed sampling strategy. We adopt the loss functions from $\pi^{3}$ [44] and VGGT [39]. To preserve pre-trained geometric fidelity, we finetune only the Alternating-Attention modules and keep the point cloud and camera decoders frozen. More training details are in the supplementary. The resulting models are denoted as $\pi^{3}$ -FT and VGGT-FT.

To study how our proposed view sampling strategy affects performance, we finetune $\pi^{3}$ on clean Internet data using four sampling schemes:

Table 1. Quantitative results on MegaDepth-X for camera pose and point map estimation across two difficulty levels. Our finetuned models ( $\pi^{3}$ -FT and VGGT-FT) trained with the proposed dataset and sampling strategy consistently outperform pretrained baselines, especially on harder, sparser scenes. 

<table><tr><td rowspan="3" colspan="2">Method</td><td colspan="5">Camera Pose Estimation</td><td colspan="6">Point Map Estimation</td></tr><tr><td rowspan="2">RRA@5↑</td><td rowspan="2">RTA@5↑</td><td rowspan="2">AUC@5↑</td><td rowspan="2">MRE↓</td><td rowspan="2">MTE↓</td><td colspan="2">Acc↓</td><td colspan="2">Comp↓</td><td colspan="2">NC↑</td></tr><tr><td>Mean</td><td>Med.</td><td>Mean</td><td>Med.</td><td>Mean</td><td>Med.</td></tr><tr><td rowspan="4">easy</td><td> $\pi^3$ </td><td>88.97</td><td>68.79</td><td>45.84</td><td>4.12</td><td>7.82</td><td>0.055</td><td>0.030</td><td>0.039</td><td>0.019</td><td>0.712</td><td>0.822</td></tr><tr><td> $\pi^3$ -FT</td><td>95.64</td><td>76.85</td><td>55.58</td><td>1.64</td><td>5.50</td><td>0.035</td><td>0.020</td><td>0.024</td><td>0.012</td><td>0.724</td><td>0.837</td></tr><tr><td>VGGT</td><td>84.17</td><td>58.47</td><td>35.32</td><td>4.55</td><td>9.93</td><td>0.093</td><td>0.047</td><td>0.055</td><td>0.026</td><td>0.695</td><td>0.798</td></tr><tr><td>VGGT-FT</td><td>92.41</td><td>71.12</td><td>48.78</td><td>2.70</td><td>7.02</td><td>0.050</td><td>0.027</td><td>0.033</td><td>0.014</td><td>0.719</td><td>0.833</td></tr><tr><td rowspan="4">hard</td><td> $\pi^3$ </td><td>75.31</td><td>59.16</td><td>36.93</td><td>12.21</td><td>10.82</td><td>0.101</td><td>0.065</td><td>0.133</td><td>0.090</td><td>0.689</td><td>0.786</td></tr><tr><td> $\pi^3$ -FT</td><td>86.40</td><td>71.00</td><td>47.93</td><td>5.72</td><td>7.27</td><td>0.068</td><td>0.041</td><td>0.066</td><td>0.041</td><td>0.713</td><td>0.818</td></tr><tr><td>VGGT</td><td>70.98</td><td>52.98</td><td>29.10</td><td>13.20</td><td>13.34</td><td>0.149</td><td>0.092</td><td>0.151</td><td>0.104</td><td>0.675</td><td>0.764</td></tr><tr><td>VGGT-FT</td><td>81.07</td><td>65.59</td><td>41.49</td><td>7.22</td><td>9.05</td><td>0.089</td><td>0.053</td><td>0.084</td><td>0.055</td><td>0.709</td><td>0.814</td></tr></table>

- DENSE: training batches with densely overlapping views where $D = 5$ and $N_{cc} = 1$ ,   
- SPARSE: long-tail-like sampling emphasizing wide baselines where $D = 24$ and $N_{cc} = 4$ ,   
- MIXED: a combination of dense and sparse batches for balanced learning with $D \in [5, 24]$ and $N_{cc} \in [1, 4]$ ,   
• RANDOM: random view sampling.

Unless otherwise noted, FT (e.g., $\pi^{3}$ -FT) refers to the model finetuned on the cleaned dataset using the MIXED sampling strategy above. We additionally train a DIRTY variant on Internet data (using the same Mixed scheme) without the filtering strategy in Sec. 3.1, while keeping the same depth refinement pipeline in Sec. 3.2, to assess robustness to label noise and data contamination.

Evaluation Metrics. For camera pose estimation, we follow prior work $[39, 44]$ and report Relative Rotation Accuracy (RRA), Relative Translation Accuracy (RTA), and their combined Area Under Curve (AUC). We also report mean rotation and translation errors (MRE and MTE, in degrees). For point map evaluation, we follow prior work $[2, 38, 40, 42, 44]$ and report Accuracy (Acc), Completeness (Comp), and Normal Consistency (NC), each computed as the mean and median across test scenes.

# 5.2. Internet Photo Evaluation

We first evaluate models on the proposed MD-X benchmark, which contains Internet photo collections of varying sparsity and difficulty. For each test scene, we sample 24 images from the reconstructed scene graph using our sampling algorithm, and categorize them into easy $D = 5, N_{cc} = 1$ and hard $D = 24, N_{cc} = 4$ subsets according to the greedy search depth used for test data sampling.

Quantitative Results. Tab. 1 reports quantitative results for camera pose and point map estimation across three difficulty levels on MD-X. Finetuning markedly improves both $\pi^{3}$ and VGGT over their pretrained baselines, with larger gains observed in harder, sparser scenes. These improvements hold across metrics indicate that the fine-tuned models better capture global structure and maintain consistent 3D geometry

Table 2. Ablation study on MegaDepth-X. Finetuning on the cleaned dataset with MIXED dense–sparse sampling ( $\pi^{3}$ -FT) yields the best overall performance, while training on unfiltered data (DIRTY) degrades accuracy. 

<table><tr><td rowspan="3" colspan="2">Method</td><td colspan="5">Camera Pose Estimation</td><td colspan="6">Point Map Estimation</td></tr><tr><td rowspan="2">RRA@5↑</td><td rowspan="2">RTA@5↑</td><td rowspan="2">AUC@5↑</td><td rowspan="2">MRE↓</td><td rowspan="2">MTE↓</td><td colspan="2">Acc↓</td><td colspan="2">Comp↓</td><td colspan="2">NC↑</td></tr><tr><td>Mean</td><td>Med.</td><td>Mean</td><td>Med.</td><td>Mean</td><td>Med.</td></tr><tr><td rowspan="6">eury</td><td> $\pi^3$ </td><td>88.97</td><td>68.79</td><td>45.84</td><td>4.12</td><td>7.82</td><td>0.055</td><td>0.030</td><td>0.039</td><td>0.019</td><td>0.712</td><td>0.822</td></tr><tr><td> $\pi^3$ -FT</td><td>95.64</td><td>76.85</td><td>55.58</td><td>1.64</td><td>5.50</td><td>0.035</td><td>0.020</td><td>0.024</td><td>0.012</td><td>0.724</td><td>0.837</td></tr><tr><td> $\pi^3$ -DIRTY</td><td>91.25</td><td>72.80</td><td>51.77</td><td>5.16</td><td>7.28</td><td>0.075</td><td>0.052</td><td>0.081</td><td>0.051</td><td>0.710</td><td>0.818</td></tr><tr><td> $\pi^3$ -RANDOM</td><td>95.08</td><td>76.42</td><td>55.00</td><td>1.78</td><td>5.72</td><td>0.039</td><td>0.021</td><td>0.026</td><td>0.013</td><td>0.720</td><td>0.831</td></tr><tr><td> $\pi^3$ -DENSE</td><td>95.13</td><td>76.73</td><td>55.65</td><td>1.84</td><td>5.61</td><td>0.036</td><td>0.020</td><td>0.026</td><td>0.013</td><td>0.725</td><td>0.837</td></tr><tr><td> $\pi^3$ -SPARSE</td><td>96.27</td><td>76.46</td><td>55.12</td><td>1.61</td><td>5.59</td><td>0.038</td><td>0.020</td><td>0.026</td><td>0.013</td><td>0.723</td><td>0.835</td></tr><tr><td rowspan="6">hard</td><td> $\pi^3$ </td><td>75.31</td><td>59.16</td><td>36.93</td><td>12.21</td><td>10.82</td><td>0.101</td><td>0.065</td><td>0.133</td><td>0.090</td><td>0.689</td><td>0.786</td></tr><tr><td> $\pi^3$ -FT</td><td>86.40</td><td>71.00</td><td>47.93</td><td>5.72</td><td>7.27</td><td>0.068</td><td>0.041</td><td>0.066</td><td>0.041</td><td>0.713</td><td>0.818</td></tr><tr><td> $\pi^3$ -DIRTY</td><td>81.10</td><td>65.99</td><td>43.74</td><td>11.86</td><td>9.72</td><td>0.130</td><td>0.094</td><td>0.139</td><td>0.091</td><td>0.693</td><td>0.791</td></tr><tr><td> $\pi^3$ -RANDOM</td><td>85.93</td><td>69.84</td><td>47.17</td><td>6.53</td><td>7.78</td><td>0.071</td><td>0.040</td><td>0.073</td><td>0.045</td><td>0.708</td><td>0.812</td></tr><tr><td> $\pi^3$ -DENSE</td><td>85.82</td><td>70.06</td><td>47.47</td><td>6.04</td><td>7.64</td><td>0.071</td><td>0.042</td><td>0.062</td><td>0.035</td><td>0.713</td><td>0.817</td></tr><tr><td> $\pi^3$ -SPARSE</td><td>85.97</td><td>70.53</td><td>47.13</td><td>6.05</td><td>7.52</td><td>0.070</td><td>0.040</td><td>0.070</td><td>0.041</td><td>0.710</td><td>0.814</td></tr></table>

in sparse settings.

Ablation Analysis. We analyze the effects of data quality and sampling strategies, with results shown in Tab. 2. Training on unfiltered (DIRTY) data consistently reduces accuracy, even performing worse than the pretrained model in point-map estimation on both the easy and hard levels, highlighting the importance of clean supervision for robust generalization. Among sampling schemes, RANDOM sampling yields reasonable camera pose accuracy but provides limited improvement in point map reconstruction, emphasizing the importance of adequate covisibilities in training batches. DENSE sampling performs well on easier scenes but is less effective under sparse conditions. SPARSE sampling alone does not yield the best trade-off. Although it exposes the model to more challenging cases, MIXED sampling achieves slightly better overall performance across difficulty levels.

Qualitative Analysis. We show qualitative results for three settings: the MD-X test set, real-world long-tail Internet scenes, and doppelganger scenes.

MegaDepth-X Visualization. Fig. 5 shows reconstruction results on the MD-X test set across easy and hard levels. Our fine-tuned model produces more accurate camera poses, more dense and consistent 3D point maps compared to the pretrained baseline, especially on sparse (hard) scenes. It generalizes well across varying camera intrinsics and challenging appearance changes such as day-night shifts.

Real Long-Tail Scenes. Real long-tail Internet scenes often contain fewer than 100 usable photos captured from uneven viewpoints and mixed with transient or irrelevant content. Classical SfM pipelines, e.g., COLMAP, typically fail to register most images, producing extremely sparse geometry or incomplete reconstructions. Pretrained models struggle under these conditions, yielding low-confidence predictions and fragmented structures. Our finetuned model remains stable and reconstructs coherent global geometry. As shown in Fig. 6, our model successfully reconstructs dense geometry from very few views, and handles doppelganger ambiguities with higher confidence, demonstrating strong robustness and generalization to real-world long-tail scenes. In the supplementary material, we provide more results on doppelganger scenes.

![](images/5e369e385c500bafae12c5cb3ac92ba0e714400f6fe45a0f9e90502ad87eab73.jpg)

<details>
<summary>text_image</summary>

GT
Pretrained π³
Ours
easy
search depth
hard
wrong
correct
</details>

Figure 5. Reconstruction results on the MegaDepth-X test set across two difficulty levels. For each level, the top row shows the full 24-image input set, and the bottom row compares reconstructions from ground truth, pretrained $\pi^{3}$ , and our finetuned model with top-down views shown in the insets. Our model shows clearer improvements in the hard setting, where the inputs are more challenging. Note that hard was obtained using a deeper search depth than easy.

Table 3. Camera pose estimation on RealEstate10K [51] and CO3Dv2 [25]. We follow $\pi^{3}$ 's pose sampling conventions. Our fine-tuned models, trained on proposed Internet data dataset, remain comparable to pretrained baselines, demonstrating generalization to standard benchmarks. 

<table><tr><td rowspan="2">Method</td><td colspan="5">RealEstate10K</td><td colspan="5">CO3Dv2</td></tr><tr><td>RRA@5↑</td><td>RTA@5↑</td><td>AUC@5↑</td><td>MRE↓</td><td>MTE↓</td><td>RRA@5↑</td><td>RTA@5↑</td><td>AUC@5↑</td><td>MRE↓</td><td>MTE↓</td></tr><tr><td> $\pi^{3}$ </td><td>98.79</td><td>79.61</td><td>62.82</td><td>0.51</td><td>5.65</td><td>93.24</td><td>84.47</td><td>57.12</td><td>3.04</td><td>4.28</td></tr><tr><td> $\pi^{3}$ -FT</td><td>98.80</td><td>77.78</td><td>60.01</td><td>0.51</td><td>6.13</td><td>93.97</td><td>84.50</td><td>57.61</td><td>2.96</td><td>4.26</td></tr><tr><td>VGGT</td><td>97.49</td><td>62.32</td><td>38.09</td><td>1.03</td><td>8.66</td><td>96.97</td><td>86.19</td><td>67.84</td><td>2.33</td><td>3.95</td></tr><tr><td>VGGT-FT</td><td>98.23</td><td>71.88</td><td>48.23</td><td>0.82</td><td>6.85</td><td>97.11</td><td>86.27</td><td>67.81</td><td>2.29</td><td>3.92</td></tr></table>

# 5.3. Generalization to Standard Benchmarks

We next examine whether the finetuned models preserve generalization on standard, curated benchmarks.

Relative Pose Estimation. We evaluate on RealEstate-10K [51] and CO3Dv2 [25], following $\pi^{3}$ 's pose sampling conventions. As shown in Tab. 3, fine-tuning on Internet data generally maintains the performance of both backbones, and yields modest improvements for VGGT in particular. These results indicate that robustness learned from sparse, in-the-wild Internet photos does not compromise generalization to standard 3D benchmarks.

Point Map Estimation. Results on DTU [14], ETH3D [32], 7-Scenes [33], and NRGBD [2] (Tab. 4&5) show that our model maintains comparable reconstruction accuracy on

![](images/1cc9a9451ff62de0e8fda3c8a6f432c92aecc997056ce50b3ec0741fe8b49459.jpg)

<details>
<summary>text_image</summary>

Image Collection
Pretrained x²
Ours
COLMAP & Reference
</details>

Figure 6. Reconstruction results on real long-tail Internet scenes. Each scene contains only a handful of photos with uneven viewpoints and noisy content, where COLMAP fails to register most images and produces extremely sparse geometry. Pretrained $\pi^{3}$ makes low-confidence predictions and incomplete reconstructions, while our fine-tuned model discovers the correct large-scale layout (e.g., (1) Novo-Znamenka Manor, 66 images, 13 registered), handles very few-view inputs and recovers dense geometry ((2) Sobanski Palace in Guzow, 95 images, 11 registered), reconstructs more complete structures under sparse, long-tail settings ((3) Delizia del Verginese (Gambulaga, Portomaggiore), 69 images, 11 registered, (5) Chitharal Jain Monuments, 44 images, 15 registered), resolves doppelganger ambiguity ((4) Hoshang's Tomb, 85 images, 40 registered), and even works when COLMAP completely fails ((6) Chapel of Saint Andrew's cathedral (Saint Petersburg), 94 images, 0 registered). These results demonstrate that our model remains robust and confident under severe sparsity and ambiguity in real long-tail Internet scenes. For each scene, the confidence threshold is the same for pretrained $\pi^{3}$ and our method.

Table 4. Point map estimation on DTU [14] and ETH3D [32]. Fine-tuning on the proposed Internet photo dataset retain overall reconstruction quality on DTU, while performance on ETH3D decreases due to domain mismatch with Internet imagery. These results show that the model adapts to Internet photos without drifting too much on out-of-domain benchmarks. 

<table><tr><td rowspan="3">Method</td><td colspan="6">DTU</td><td colspan="6">ETH3D</td></tr><tr><td colspan="2">Acc. ↓</td><td colspan="2">Comp. ↓</td><td colspan="2">N.C. ↑</td><td colspan="2">Acc. ↓</td><td colspan="2">Comp. ↓</td><td colspan="2">N.C. ↑</td></tr><tr><td>Mean</td><td>Med.</td><td>Mean</td><td>Med.</td><td>Mean</td><td>Med.</td><td>Mean</td><td>Med.</td><td>Mean</td><td>Med.</td><td>Mean</td><td>Med.</td></tr><tr><td> $\pi^3$ </td><td>1.151</td><td>0.622</td><td>1.793</td><td>0.629</td><td>0.668</td><td>0.754</td><td>0.188</td><td>0.126</td><td>0.211</td><td>0.129</td><td>0.872</td><td>0.967</td></tr><tr><td> $\pi^3$ -FT</td><td>1.202</td><td>0.642</td><td>1.928</td><td>0.593</td><td>0.666</td><td>0.751</td><td>0.199</td><td>0.142</td><td>0.242</td><td>0.151</td><td>0.861</td><td>0.955</td></tr><tr><td>VGGT</td><td>1.308</td><td>0.761</td><td>1.929</td><td>1.015</td><td>0.665</td><td>0.750</td><td>0.270</td><td>0.174</td><td>0.304</td><td>0.180</td><td>0.841</td><td>0.942</td></tr><tr><td>VGGT-FT</td><td>1.283</td><td>0.759</td><td>1.900</td><td>0.953</td><td>0.669</td><td>0.756</td><td>0.282</td><td>0.205</td><td>0.394</td><td>0.225</td><td>0.838</td><td>0.927</td></tr></table>

Table 5. Point map estimation on 7-Scenes [33] and NRGBD [2] datasets. We evaluate both sparse-view and dense-view settings. Fine-tuning on Internet photos yields comparable performance to pretrained baselines with minor variations, indicating our method preserves generalization across diverse real world and synthetic datasets. 

<table><tr><td rowspan="3">View</td><td rowspan="3">Method</td><td colspan="6">7-Scenes</td><td colspan="6">NRGBD</td></tr><tr><td colspan="2">Acc. ↓</td><td colspan="2">Comp. ↓</td><td colspan="2">NC. ↑</td><td colspan="2">Acc. ↓</td><td colspan="2">Comp. ↓</td><td colspan="2">NC. ↑</td></tr><tr><td>Mean</td><td>Med.</td><td>Mean</td><td>Med.</td><td>Mean</td><td>Med.</td><td>Mean</td><td>Med.</td><td>Mean</td><td>Med.</td><td>Mean</td><td>Med.</td></tr><tr><td rowspan="4">sparse</td><td> $\pi^3$ </td><td>0.047</td><td>0.029</td><td>0.074</td><td>0.049</td><td>0.741</td><td>0.840</td><td>0.024</td><td>0.013</td><td>0.028</td><td>0.013</td><td>0.909</td><td>0.991</td></tr><tr><td> $\pi^3$ -FT</td><td>0.046</td><td>0.027</td><td>0.072</td><td>0.046</td><td>0.739</td><td>0.841</td><td>0.024</td><td>0.014</td><td>0.028</td><td>0.014</td><td>0.903</td><td>0.990</td></tr><tr><td>VGGT</td><td>0.044</td><td>0.024</td><td>0.056</td><td>0.033</td><td>0.733</td><td>0.846</td><td>0.049</td><td>0.027</td><td>0.066</td><td>0.037</td><td>0.882</td><td>0.979</td></tr><tr><td>VGGT-FT</td><td>0.062</td><td>0.046</td><td>0.097</td><td>0.070</td><td>0.738</td><td>0.844</td><td>0.071</td><td>0.046</td><td>0.071</td><td>0.041</td><td>0.875</td><td>0.959</td></tr><tr><td rowspan="4">dense</td><td> $\pi^3$ </td><td>0.016</td><td>0.007</td><td>0.022</td><td>0.011</td><td>0.689</td><td>0.792</td><td>0.013</td><td>0.007</td><td>0.014</td><td>0.006</td><td>0.874</td><td>0.981</td></tr><tr><td> $\pi^3$ -FT</td><td>0.016</td><td>0.007</td><td>0.023</td><td>0.011</td><td>0.686</td><td>0.789</td><td>0.013</td><td>0.007</td><td>0.014</td><td>0.005</td><td>0.864</td><td>0.978</td></tr><tr><td>VGGT</td><td>0.022</td><td>0.008</td><td>0.026</td><td>0.012</td><td>0.667</td><td>0.760</td><td>0.015</td><td>0.008</td><td>0.015</td><td>0.006</td><td>0.871</td><td>0.982</td></tr><tr><td>VGGT-FT</td><td>0.016</td><td>0.007</td><td>0.027</td><td>0.012</td><td>0.681</td><td>0.781</td><td>0.015</td><td>0.008</td><td>0.016</td><td>0.006</td><td>0.859</td><td>0.981</td></tr></table>

DTU, 7-Scenes and NRGBD. We observe a performance decrease on ETH3D and a mild drop for VGGT under sparse

NRGBD, likely reflecting the domain gap between these clean, controlled datasets and Internet imagery. Overall, the results indicate that training on diverse Internet photos preserves cross-dataset generalization without overfitting.

# 6. Conclusion

We presented a step towards robust, Internet-scale 3D reconstruction by defining and addressing the long-tail regime of Internet photo collections. Through the MegaDepth-X dataset and a sparsity-aware sampling strategy, we augment the ability of 3D foundation models to recover consistent geometry from sparse, noisy, and ambiguous imagery, where classical SfM and SOTA feed-forward 3D reconstruction models fail, and demonstrates disambiguation of doppelganger scenes while maintaining generalization across benchmarks.

Our dataset currently focuses on landmark-scale scenes, representing only a small fraction of the landscape of Internet photos. Bootstrapping on the current dataset and refining models for reconstructions of even more longed-tail data remains an important direction for future work. Extending this framework beyond landmarks to everyday objects, indoor scenes, and other Internet photo domains offers a promising path toward a truly universal 3D foundation model.

Acknowledgments This work was supported in part by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korean Government (MSIT) (No. RS-2024-00457882, National AI Research Lab Project). We thank Joseph Tung, Yiwen Zhang, Hanyu Chen and Haian Jin for discussion and help with MegaScenes dataset and depth post-processing.

# References

[1] Sameer Agarwal, Yasutaka Furukawa, Noah Snavely, Ian Simon, Brian Curless, Steven M Seitz, and Richard Szeliski. Building rome in a day. Communications of the ACM, 54(10):105–112, 2011. 2   
[2] Dejan Azinović, Ricardo Martin-Brualla, Dan B Goldman, Matthias Nießner, and Justus Thies. Neural rgb-d surface reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6290–6301, 2022. 6, 7, 8   
[3] Hana Bezalel, Dotan Ankri, Ruojin Cai, and Hadar Averbach-Elor. Extreme rotation estimation in the wild. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1061-1070, 2025. 3   
[4] Vincent D Blondel, Jean-Loup Guillaume, Renaud Lambiotte, and Etienne Lefebvre. Fast unfolding of communities in large networks. Journal of statistical mechanics: theory and experiment, 2008(10):P10008, 2008. 5   
[5] Daniel J. Butler, Jonas Wulff, Garrett B. Stanley, and Michael J. Black. A naturalistic open source movie for optical flow evaluation. In Proceedings of the 12th European Conference on Computer Vision - Volume Part VI, page 611–625, Berlin, Heidelberg, 2012. Springer-Verlag. 3, 4   
[6] Ruojin Cai, Bharath Hariharan, Noah Snavely, and Hadar Averbuch-Elor. Extreme rotation estimation using dense correlation volumes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14566–14575, 2021. 3   
[7] Ruojin Cai, Joseph Tung, Qianqian Wang, Hadar Averbuch-Elor, Bharath Hariharan, and Noah Snavely. Doppelgangers: Learning to disambiguate images of similar structures. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 34–44, 2023. 3   
[8] Filiberto Chiabrando, Loren Clark, John Driscoll, Scott McAvoy, Dominique Rissolo, Alessandra Spreafico, and Beatrice Tanduo. Salvation mountain - photogrammetry - terrestrial, photogrammetry - aerial, lidar - terrestrial, lidar - mobile, survey data, 2023. Distributed by Open Heritage 3D. 4   
[9] CyArk. Great mosque - kilwa kisiwani - lidar - terrestrial, photogrammetry - terrestrial, photogrammetry - aerial, 2020. Distributed by Open Heritage 3D. 4   
[10] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superpoint: Self-supervised interest point detection and description. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pages 224–236, 2018. 3   
[11] Bardienus Pieter Duisterhof, Lojze Zust, Philippe Weinzaepfel, Vincent Leroy, Yohann Cabon, and Jerome Revaud.

MASt3r-sfm: a fully-integrated solution for unconstrained structure-from-motion. In International Conference on 3D Vision 2025, 2025. 6   
[12] Jan-Michael Frahm, Pierre Fite-Georgel, David Gallup, Tim Johnson, Rahul Raguram, Changchang Wu, Yi-Hung Jen, Enrique Dunn, Brian Clipp, Svetlana Lazebnik, and Marc Pollefeys. Building Rome on a Cloudless Day. In ECCV, 2010. 2   
[13] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. International Journal of Robotics Research (IJRR), 2013. 3, 4   
[14] Rasmus Jensen, Anders Dahl, George Vogiatzis, Engil Tola, and Henrik Aanæs. Large scale multi-view stereopsis evaluation. In 2014 IEEE Conference on Computer Vision and Pattern Recognition, pages 406–413. IEEE, 2014. 7, 8   
[15] Hanwen Jiang, Hanwen Jiang, Arjun Karpur, Bingyi Cao, Qixing Huang, and Qi-Xing Huang. Omniglue: Generalizable feature matching with foundation model guidance. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19865–19875, 2024. 3   
[16] Haian Jin, Rundi Wu, Tianyuan Zhang, Ruiqi Gao, Jonathan T. Barron, Noah Snavely, and Aleksander Holynski. ZipMap: Linear-time stateful 3d reconstruction via test-time training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026. 2   
[17] Arjun Karpur, Guilherme Perrotta, Ricardo Martin-Brualla, Howard Zhou, and Andre F. de Araújo. Lfm-3d: Learnable feature matching across wide baselines using 3d signals. 2024 International Conference on 3D Vision (3DV), pages 11–20, 2023. 3   
[18] Lawrence Kou, George Markowsky, and Leonard Berman. A fast algorithm for steiner trees. Acta informatica, 15(2): 141–145, 1981. 5   
[19] Vincent Leroy, Yohann Cabon, and Jérôme Revaud. Grounding image matching in 3d with mast3r, 2024. 2, 3   
[20] Zhengqi Li and Noah Snavely. Megadepth: Learning single-view depth prediction from internet photos. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2041-2050, 2018. 2, 3, 1   
[21] Philipp Lindenberger, Paul-Edouard Sarlin, and Marc Pollefeys. Lightglue: Local feature matching atokens are frozen. arXiv preprint arXiv:2306.13643, 2023. 3   
[22] Kurt Mehlhorn. A faster approximation algorithm for the steiner problem in graphs. Information Processing Letters, 27(3):125-128, 1988. 5   
[23] Pushmeet Kohli Nathan Silberman, Derek Hoiem and Rob Fergus. Indoor segmentation and support inference from rgbd images. In ECCV, 2012. 3, 4   
[24] E. Palazzolo, J. Behley, P. Lottes, P. Giguère, and C. Stachniss. ReFusion: 3D Reconstruction in Dynamic Environments for RGB-D Cameras Exploiting Residuals. arXiv, 2019. 3, 4   
[25] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotný. Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 10881–10891, 2021. 7

[26] Ashley Richter, Michael Hess, Vid Petrovic, Falko Kuester, Cultural Heritage Engineering Initiative (CHEI), Architecture Center of Interdisciplinary Science for Art, and Archaeology (CISA3). Torre dei baldovinetti - florence - lidar - terrestrial, photogrammetry - terrestrial, 2023. Distributed by Open Heritage 3D. 4   
[27] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superglue: Learning feature matching with graph neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4938–4947, 2020. 3   
[28] Johannes Lutz Schönberger and Jan-Michael Frahm. Structure-from-motion revisited. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 2, 3   
[29] Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In CVPR, 2016. 2, 3   
[30] Johannes L. Schönberger, Enliang Zheng, Jan-Michael Frahm, and Marc Pollefeys. Pixelwise view selection for unstructured multi-view stereo. In European Conference on Computer Vision, 2016. 2   
[31] Johannes Lutz Schönberger, Enliang Zheng, Marc Pollefeys, and Jan-Michael Frahm. Pixelwise view selection for unstructured multi-view stereo. In European Conference on Computer Vision (ECCV), 2016. 3   
[32] Thomas Schöps, Johannes L. Schönberger, Silvano Galliani, Torsten Sattler, Konrad Schindler, Marc Pollefeys, and Andreas Geiger. A multi-view stereo benchmark with high-resolution images and multi-camera videos. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2538–2547, 2017. 7, 8   
[33] Jamie Shotton, Ben Glocker, Christopher Zach, Shahram Izadi, Antonio Criminisi, and Andrew Fitzgibbon. Scene coordinate regression forests for camera relocalization in rgb-d images. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2930–2937, 2013. 7, 8   
[34] Noah Snavely, Steven M Seitz, and Richard Szeliski. Photo tourism: exploring photo collections in 3d. In ACM siggraph 2006 papers, pages 835–846. 2006. 2   
[35] Noah Snavely, Steven M Seitz, and Richard Szeliski. Skeletal graphs for efficient structure from motion. In 2008 IEEE Conference on Computer Vision and Pattern Recognition, pages 1–8. IEEE, 2008. 5   
[36] Joseph Tung, Gene Chou, Ruojin Cai, Guandao Yang, Kai Zhang, Gordon Wetzstein, Bharath Hariharan, and Noah Snavely. Megascenes: Scene-level view synthesis at scale. arXiv preprint arXiv:2406.11819, 2024. 1, 2, 3   
[37] Michał Tyszkiewicz, Pascal Fua, and Eduard Trulls. Disk: Learning local features with policy gradient. Advances in Neural Information Processing Systems, 33:14254–14265, 2020. 3   
[38] Hengyi Wang and Lourdes Agapito. 3d reconstruction with spatial memory. arXiv preprint arXiv:2408.16061, 2024. 6   
[39] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025. 2, 3, 6

[40] Qianqian Wang\*, Yifei Zhang\*, Aleksander Holynski, Alexei A. Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. In CVPR, 2025. 6, 3   
[41] Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and Jiaolong Yang. Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5261–5271, 2025. 4   
[42] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. arXiv preprint arXiv:2312.14132, 2023. 2, 6   
[43] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer. Tartanair: A dataset to push the limits of visual slam. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 4909–4916. IEEE, 2020. 3   
[44] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. $\pi^{3}$ : Scalable permutation-equivariant visual geometry learning, 2025. 1, 2, 6, 3   
[45] Yuanbo Xiangli, Ruojin Cai, Hanyu Chen, Jeffrey Byrne, and Noah Snavely. Doppelgangers++: Improved visual disambiguation with geometric 3d features, 2025. 3, 6   
[46] Tao Xie, Peishan Yang, Yudong Jin, Yingfeng Cai, Wei Yin, Weiqiang Ren, Qian Zhang, Wei Hua, Sida Peng, Xiaoyang Guo, and Xiaowei Zhou. Scal3r: Scalable test-time training for large-scale 3d reconstruction, 2026. 2   
[47] Jianing Yang, Alexander Sax, Kevin J Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21924–21935, 2025. 2   
[48] Yao Yao, Zixin Luo, Shiwei Li, Jingyang Zhang, Yufan Ren, Lei Zhou, Tian Fang, and Long Quan. Blendedmvs: A large-scale dataset for generalized multi-view stereo networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1790–1799, 2020. 3   
[49] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and Ming-Hsuan Yang. Monst3r: A simple approach for estimating geometry in the presence of motion. arXiv preprint arxiv:2410.03825, 2024. 3   
[50] Shangzhan Zhang, Jianyuan Wang, Yinghao Xu, Nan Xue, Christian Rupprecht, Xiaowei Zhou, Yujun Shen, and Gordon Wetzstein. Flare: Feed-forward geometry, appearance and camera estimation from uncalibrated sparse views. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21936–21947, 2025. 2   
[51] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. arXiv preprint arXiv:1805.09817, 2018. 7

# Long-Tail Internet Photo Reconstruction

# Supplementary Material

# Visualization Webpage

Please refer to our project page for additional visualizations beyond this PDF. The webpage includes: (i) animations of our sparsity-aware sampling procedure on representative scenes; and (ii) comparisons of reconstructions from pretrained $\pi^{3}$ and our finetuned $\pi^{3}$ on long-tail scenes (where COLMAP registers 0 images). We also provide video fly-throughs of reconstructed point clouds and additional qualitative results on the webpage to help visualize performance on diverse, real-world scenes.

# A. The MegaDepth-X Dataset

# A.1. Data Processing

In this section, we compare COLMAP results with those produced by our proposed data-processing pipeline. Fig. 12 shows reconstructions from COLMAP and our MASt3R-SfM pipeline. COLMAP often fails on ambiguous scenes involving similar-looking objects, visually similar but distinct building facades, symmetric landmarks etc. In contrast, our reconstruction pipeline effectively mitigates these issues and recovers correct geometry. In Fig. 13, we show that our monocular depth-guided dense depthmap filtering strategy prevents background depths from leaking into foreground regions (i.e. the depth-bleeding issue [20]) and removes depth estimates on transient objects, which are often unreliable in COLMAP MVS. Note that we use monocular depth only as guidance, rather than warping it to align with the MVS depth. This is because we prioritize accurate depth maps over complete ones. Uncertainty in the relative depth predictions of monocular models can introduce additional noise and inconsistency across views. For example, in the last row of Fig. 13, COLMAP MVS fails to recover the depth of the foreground statue, and we opt to remove the depth values in that region. If we were to warp the monocular depth to match the MVS result, then any inaccuracy in the relative depth between the statue and the background building could produce erroneous and inconsistent cross-view depth estimates.

# A.2. Dataset Statistics

We provide an overall comparison between MegaDepth and MegaDepth-X in Tab. 6, including reconstruction statistics as well as several metrics that characterize the spatial distribution of viewpoints. Beyond basic dataset properties such as the number of intact reconstructions, image count, and whether doppelganger filtering or dense depth refinement is applied, we analyze how cameras are positioned and oriented in each scene, as scenes with broad viewpoint coverage allow our sampling strategy to construct more diverse and representative sparse-view subsets. The statistics are computed from Manhattan-aligned COLMAP reconstructions.

Positional coverage. To understand how cameras are placed in the horizontal plane, we compute each camera's azimuth angle relative to the scene centroid (that is, the angle of the direction from the scene centroid to the camera) and divide the full 0-360° range into 36 equal 10° bins. In practice, the scene centroid is derived from the average of the SFM point cloud. A scene with many occupied bins is one where cameras are well-distributed around the object. In the table, the columns “Positional Azimuth Coverage = 100% / ≥ 75% / ≥ 50% / ≥ 25%” report how many scenes achieve at least that percentage of bins(36/36, 27/36, 18/36, 9/36), with larger thresholds indicating closer to full 360° wrap-around coverage.

Rotational coverage. Position alone does not describe where cameras are looking. We therefore measure the coverage of camera orientations by mapping each camera's forward viewing direction to 36 azimuth bins similar to positional coverage. If cameras face more distinct directions, more bins are occupied; if they face similar directions, only few bins are occupied. We summarize this rotational azimuth coverage using the same percentage thresholds as positional azimuth coverage.

These statistics show that MegaDepth-X contains substantially more scenes with broad camera-position coverage and diverse viewing directions, making it better suited for robust sparse-view reconstruction than MegaDepth.

# B. Sparsity-aware Sampling

# B.1. Greedy Sampling Algorithm

We illustrate one iteration of the greedy view-sampling procedure in Alg. 1. At each step, the algorithm selects the next view based on two criteria:

1. Community novelty: prioritizing candidates whose camera-community has not yet been visited by the sampled set. This encourages the trajectory to enter unexplored regions of the view graph and reduces redundancy in viewpoint selection.   
2. Spatial distance: among candidates with equal novelty, preferring those that are farther from the current camera position. This promotes larger baselines and helps diversify the spatial coverage of the sampled views.

Table 6. Dataset statistics and viewpoint-distribution metrics. We report reconstruction statistics and metrics describing camera coverage. Positional Azimuth Coverage counts scenes whose camera positions occupy 9–36 (i.e. 25%-100%) of the 36 horizontal azimuth bins ( $10^{\circ}$ per bin, covering the full $360^{\circ}$ ). Rotational Azimuth Coverage represents scenes whose camera forwarding vectors occupy 9–36 (i.e. 25%-100%) of the 36 horizontal azimuth bins ( $10^{\circ}$ per bin, covering the full $360^{\circ}$ ). For each scene, the more bins covered, the wider the camera distribution is. $\dagger$ Dense depth refinement uses monocular depth–guided filtering. 

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">#Recons.</td><td rowspan="2">#Images</td><td rowspan="2">Doppelganger Check</td><td rowspan="2">Dense Depth Refinement</td><td colspan="4">Positional Azimuth Coverage</td><td colspan="4">Rotational Azimuth Coverage</td></tr><tr><td>= 100% ↑</td><td>≥ 75% ↑</td><td>≥ 50% ↑</td><td>≥ 25% ↑</td><td>= 100% ↑</td><td>≥ 75% ↑</td><td>≥ 50% ↑</td><td>≥ 25% ↑</td></tr><tr><td>MegaDepth [20]</td><td>266</td><td>119k</td><td>No</td><td>Yes</td><td>4</td><td>15</td><td>25</td><td>74</td><td>27</td><td>56</td><td>107</td><td>230</td></tr><tr><td>MegaDepth-X (Ours)</td><td>1,865</td><td>440k</td><td>Yes</td><td>Yes $^{\dagger}$ </td><td>6</td><td>80</td><td>223</td><td>752</td><td>76</td><td>490</td><td>1123</td><td>1816</td></tr></table>

Algorithm 1: One Step of Greedy View Sampling   
Input: Current node v
Neighborhood of v: $N_{v}$ Set of already sampled nodes S
Community map M (node → community)
Camera positions Pos(·)
Output: Next sampled node $u^{*}$ // Identify communities already covered $S_{comm} \leftarrow \{M[s] \mid s \in S\}$ ;
// Compute candidate list with
    community novelty and distance $C \leftarrow \emptyset$ ;
for each $u \in N_{v}$ do
    unreached $\leftarrow (M[u] \notin S_{comm})$ ;
    dist $\leftarrow \|\text{Pos}(u) - \text{Pos}(v)\|_{2}$ ; $C \leftarrow C \cup \{(u, \text{unreached}, \text{dist})\}$ ;
end
// Sort by unreached, then by distance
Sort C in descending lexicographic order by
(unreached, dist);
// Select the top-ranked candidate $(u^{*}, -, -) \leftarrow$ first element of C;
return $u^{*}$ ;

Candidates are lexicographically ranked according to these two criteria, and the highest-ranked node is chosen as the next sampled view.

# B.2. Graph Partition

Before sparsity-aware sampling, we partition COLMAP's view graph into $N_{cc}$ subgraphs. Specifically, we randomly select $N_{cc}$ seed nodes and treat each seed as the initial node of one partition. Starting from these seeds, we perform a parallel round-robin breadth-first search(BFS) over the view graph. During each iteration, every subgraph expands from its current frontier to its unassigned neighboring nodes, which are then incorporated into that subgraph. In this way, each node is assigned to the subgraph of the seed that first reaches it, until no further nodes can be expanded.

Algorithm 2: Round-Robin BFS Graph Partitioning   
Input: View graph $G = (V, E)$ Number of subgraphs $N_{cc}$ Output: Subgraphs $\{P_{1}, \ldots, P_{N_{cc}}\}$ Randomly select $N_{cc}$ seed nodes $\{s_{1}, \ldots, s_{N_{cc}}\} \subseteq V$ ;
Initialize each $P_{i}$ with seed $s_{i}$ ;
Initialize one BFS frontier for each subgraph;
while there exists a non-empty frontier do
    for each subgraph $P_{i}$ do
    Expand its frontier by one BFS step;
    Assign each newly reached unassigned node to $P_{i}$ ;
    end
end
return $\{P_{1}, \ldots, P_{N_{cc}}\}$ ;

# B.3. Graph Span vs. Search Depth

To understand how greedy search depth D affects the coverage and sparsity of the sampled views, we analyze several statistics on the view-graph. Let G denote the full view-graph of a scene and S the set of sampled nodes. The first two metrics quantify coverage with respect to the entire graph G, while the last two measure sparsity within the sampled subset S.

k-hop graph coverage. This metric measures how much of the view-graph is reached by the sampled views. Specifically, it computes the fraction of nodes in G that lie within k hops of any sampled node:

$$
\operatorname{Cov} _ {k} (G, S) = \frac {1}{| G |} | \{u \in G, v \in S \mid d _ {G} (u, v) \leq k \} |, \tag {1}
$$

where S is the subgraph of greedy sampled nodes and $d_{G}(u,v)$ is the shortest path from u to v on the graph G. A higher $Cov_{k}$ indicates broader topological coverage, i.e., the sampled set reaches many graph neighborhoods rather than remaining confined to a small region.

Nearest-sample distance. To evaluate spatial coverage in 3D, we compute the average Euclidean distance from each camera to its closest sampled camera:

![](images/947590527cf15572b0179e045d764adb7792a582e50d3d3233e593661e2ee4db.jpg)

<details>
<summary>line</summary>

| Search Depth | Cov_k |
|---|---|
| 2 | 0.475 |
| 4 | 0.525 |
| 6 | 0.550 |
| 8 | 0.575 |
| 10 | 0.600 |
| 12 | 0.620 |
| 14 | 0.635 |
| 16 | 0.650 |
| 18 | 0.655 |
| 20 | 0.660 |
| 22 | 0.665 |
| 24 | 0.670 |
</details>

(a) k-hop Coverage (k=2)

![](images/ae30caa4a6e1bf31df99093268d835a36daf0ff187e7d146237579c8d326ea3f.jpg)

<details>
<summary>line</summary>

| Search Depth | AvgNear |
| ------------ | ------- |
| 2            | 0.65    |
| 4            | 0.55    |
| 6            | 0.50    |
| 8            | 0.47    |
| 10           | 0.45    |
| 12           | 0.42    |
| 14           | 0.40    |
| 16           | 0.39    |
| 18           | 0.38    |
| 20           | 0.37    |
| 22           | 0.36    |
| 24           | 0.35    |
</details>

(b) Nearest-Sample Distance

![](images/1f532e6f77ce1928016cb28ca53301137e6a4e899176d66274edb4be82483a23.jpg)

<details>
<summary>line</summary>

| Search Depth | Disp_g |
| ------------ | ------ |
| 2            | 1.75   |
| 4            | 2.00   |
| 6            | 2.25   |
| 8            | 2.50   |
| 10           | 2.75   |
| 12           | 3.00   |
| 14           | 3.25   |
| 16           | 3.50   |
| 18           | 3.75   |
| 20           | 4.00   |
| 22           | 4.25   |
| 24           | 4.50   |
</details>

(c) Graph Dispersion (pairwise hops)

![](images/7649376585de6244f5d2e7877933a3b877970b709ea3da80f3197a16917eef3a.jpg)

<details>
<summary>line</summary>

| Search Depth | Disp_E |
| ------------ | ------ |
| 2            | 0.85   |
| 4            | 1.00   |
| 6            | 1.05   |
| 8            | 1.10   |
| 10           | 1.15   |
| 12           | 1.17   |
| 14           | 1.20   |
| 16           | 1.20   |
| 18           | 1.21   |
| 20           | 1.22   |
| 22           | 1.23   |
| 24           | 1.24   |
</details>

(d) Euclidean Dispersion (pairwise distance)   
Figure 7. Coverage and sparsity vs. search depth. Metrics in (a) and (b) evaluate coverage with respect to the full view-graph, while (c) and (d) measure the sparsity of the sampled subset. As the search depth increases, the sampled set reaches a larger portion of the view-graph, as shown by the rise in k-hop (graph-distance) coverage in (a). The average distance from each camera to its nearest sampled view decreases in (b), indicating broader spatial coverage. At the same time, both graph dispersion (average pairwise graph distance) in (c) and Euclidean dispersion (average pairwise 3D distance) in (d) increase with depth, showing that the sampled views become more widely separated across the graph and in 3D space.

$$
\operatorname{AvgNear} (G, S) = \frac {1}{| G |} \sum_ {u \in G} \min _ {v \in S} \| p _ {u} - p _ {v} \| _ {2}, \tag {2}
$$

where $p_{u}$ and $p_{v}$ are camera positions. Lower values mean the sampled views are spatially well-distributed and lie near many original cameras.

Graph dispersion and Euclidean dispersion. To understand the sparsity of the sampled views, we calculate the average pairwise distance among sampled views(dispersion) based on graph distances and Euclidean distances:

$$
\operatorname{Disp} _ {\mathrm{g}} (S) = \frac {1}{| S | (| S | - 1)} \sum_ {u, v \in S, u \neq v} d _ {G} (u, v), \tag {3}
$$

$$
\operatorname{Disp} _ {\mathrm{E}} (S) = \frac {1}{| S | (| S | - 1)} \sum_ {u, v \in S, u \neq v} \| p _ {u} - p _ {v} \| _ {2}. \tag {4}
$$

Higher dispersion values indicate that the sampled views are more sparsely distributed in both the graph and Euclidean space.

We compute these metrics for the top 100 scenes with the most registered images, evaluating 12 search depths and averaging over 8 sampling runs per depth. The number of sampled views is 24 for all samples. Results are shown in Fig. 7, indicating that deeper searches yield higher coverage on the full graph (a,b) and produce sparser, more widely distributed sampled subsets (c,d).

# C. Training Details and Additional Results

# C.1. Training Setup

We finetune both $\pi^{3}$ and VGGT using their released pre-trained checkpoints. All input images are first padded with white borders to a resolution of $518 \times 518$ . During training, we apply random crops to these padded images, sampling aspect ratios uniformly from [0.75, 1.0]. We also apply random color jittering on training images. Each mini-batch contains up to 24 images drawn from MegaDepth-X, with the number of views per batch randomly selected from $[2, 24]$ . We process at most 96 images on each GPU. We also augment image orientations during training by randomly rotating images $90^{\circ}$ clockwise or counterclockwise with a probability of 0.2.

We use the original loss functions from $\pi^{3}$ [44] and VGGT [39] to finetune the models. To preserve the geometric priors encoded in the pretrained models, we finetune only the Alternating-Attention modules, while keeping the point-cloud and camera decoders frozen. We further include BlendedMVS [48] and TartanAir [43] as additional training data for finetuning. Finetuning is performed for 100 epochs, where each epoch iterates over all scenes in the combined dataset. We use the AdamW optimizer with a peak learning rate of $1 \times 10^{-5}$ , scheduled with linear warm-up followed by cosine annealing. All experiments are conducted on 4 NVIDIA A6000 GPUs.

# C.2. Additional Depth-Estimation Results

We provide monocular and video depth results to complement the main paper. Following $[40, 44, 49]$ , we evaluate Absolute Relative Error (Abs Rel) and the accuracy at a threshold of $\delta < 1.25$ . For monocular depth, we report performance on Sintel $[5]$ , Bonn $[24]$ , KITTI $[13]$ , and NYU-v2 $[23]$ . For video depth, we evaluate on Sintel $[5]$ , Bonn $[24]$ , and KITTI $[13]$ under both scale and scale&shift alignment settings. Our finetuned models maintain competitive performance across all datasets, demonstrating that the adaptation to in-the-wild imagery does not degrade their depth-estimation ability.

# C.3. Results on Doppelganger Scenes

Doppelganger cases often cause both classical SfM pipelines and pretrained feed-forward models to fail, merging distinct structures into a single incorrect reconstruction. As shown in Fig. 8, our fine-tuned $\pi^{3}$ model correctly distinguishes visually similar but distinct structures within each landmark and recovers geometry consistent with reference aerial imagery, indicating improved reconstruction of global scene layout.

Table 7. Video Depth Estimation on Sintel [5], Bonn [24], and KITTI [13]. We report Absolute Relative Error (Abs Rel, lower is better) and the prediction accuracy at a threshold of $\delta < 1.25$ (higher is better). 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Align</td><td colspan="2">Sintel</td><td colspan="2">Bonn</td><td colspan="2">KITTI</td></tr><tr><td>Abs Rel↓</td><td> $\delta < 1.25 \uparrow$ </td><td>Abs Rel↓</td><td> $\delta < 1.25 \uparrow$ </td><td>Abs Rel↓</td><td> $\delta < 1.25 \uparrow$ </td></tr><tr><td> $\pi^3$ </td><td rowspan="4">scale</td><td>0.228</td><td>0.671</td><td>0.051</td><td>0.975</td><td>0.038</td><td>0.986</td></tr><tr><td> $\pi^3$ -FT</td><td>0.213</td><td>0.713</td><td>0.047</td><td>0.978</td><td>0.040</td><td>0.985</td></tr><tr><td>VGGT</td><td>0.294</td><td>0.649</td><td>0.055</td><td>0.971</td><td>0.072</td><td>0.965</td></tr><tr><td>VGGT-FT</td><td>0.242</td><td>0.707</td><td>0.061</td><td>0.969</td><td>0.065</td><td>0.966</td></tr><tr><td> $\pi^3$ </td><td rowspan="4">scale&amp;shift</td><td>0.207</td><td>0.735</td><td>0.045</td><td>0.976</td><td>0.036</td><td>0.986</td></tr><tr><td> $\pi^3$ -FT</td><td>0.188</td><td>0.739</td><td>0.043</td><td>0.978</td><td>0.038</td><td>0.985</td></tr><tr><td>VGGT</td><td>0.226</td><td>0.683</td><td>0.049</td><td>0.974</td><td>0.059</td><td>0.961</td></tr><tr><td>VGGT-FT</td><td>0.197</td><td>0.728</td><td>0.056</td><td>0.973</td><td>0.056</td><td>0.964</td></tr></table>

Table 8. Monocular Depth Estimation on Sintel [5], Bonn [24], KITTI [13], and NYU-v2 [23]. We report Absolute Relative Error (Abs Rel, lower is better) and threshold accuracy $\delta < 1.25$ (higher is better). 

<table><tr><td rowspan="2">Method</td><td colspan="2">Sintel</td><td colspan="2">Bonn</td><td colspan="2">KITTI</td><td colspan="2">NTU-v2</td></tr><tr><td>Abs Rel↓</td><td> $\delta < 1.25 \uparrow$ </td><td>Abs Rel↓</td><td> $\delta < 1.25 \uparrow$ </td><td>Abs Rel↓</td><td> $\delta < 1.25 \uparrow$ </td><td>Abs Rel↓</td><td> $\delta < 1.25 \uparrow$ </td></tr><tr><td> $\pi^3$ </td><td>0.277</td><td>0.621</td><td>0.052</td><td>0.971</td><td>0.059</td><td>0.972</td><td>0.054</td><td>0.956</td></tr><tr><td> $\pi^3$ -FT</td><td>0.284</td><td>0.629</td><td>0.049</td><td>0.977</td><td>0.056</td><td>0.972</td><td>0.052</td><td>0.958</td></tr><tr><td>VGGT</td><td>0.331</td><td>0.600</td><td>0.051</td><td>0.974</td><td>0.089</td><td>0.939</td><td>0.055</td><td>0.953</td></tr><tr><td>VGGT-FT</td><td>0.311</td><td>0.628</td><td>0.056</td><td>0.974</td><td>0.092</td><td>0.941</td><td>0.053</td><td>0.955</td></tr></table>

![](images/e98d7fc76604081a92e74d6fe6eb9ee61c386cbbffb812eb67d71642f63f14f8.jpg)

<details>
<summary>text_image</summary>

COLMAP
Pretrained π³
Ours
Reference
Palace of Rozumovský (Baturýn)
Church of the Saviour on the Blood
Radcliffe Camera
</details>

Figure 8. Disambiguation of doppelganger scenes. Each example shows a pair of visually similar structures that cause classical SfM (COLMAP) and pretrained $\pi^{3}$ to collapse into incorrect or merged reconstructions. In contrast, our finetuned model correctly distinguishes the symmetric or repetitive sides of the same building, reconstructing consistent geometry for each viewpoint. Reference views from Google Earth are provided for comparison, confirming that our model resolves these ambiguities and recovers accurate global structure under challenging visual similarity.

To evaluate the effectiveness of different sampling strategies on doppelganger scenes, we evaluate the pretrained $\pi^{3}$ and finetuned $\pi^{3}$ on doppelganger scenes and show results in

![](images/fd548a6f1e43a56f699991fce62f21af415cd10935cbf706f31bc8e2cc380f38.jpg)

<details>
<summary>text_image</summary>

Doppelganger pair
π³
π³-DENSE
π³-FT
π²-SPARSE
</details>

Figure 9. Comparison of ablated models on doppelganger scenes We show predictions from the pre-trained model and ablated models on two doppelganger scenes. Disambiguation behavior holds across fine-tuned variants with sparsity-aware sampling, while the pre-trained model and model finetuned with densely sampled views are less robust to doppelgangers.

fig.9. Results indicate that pretrained models and dense-only fine-tuning are less robust to ambiguity, while finetuning with sparsity-aware sampling (e.g., mixed or sparse) tends to improve disambiguation, suggesting sparsity-aware sampling helps.

# C.4. Quantitative results on Long-tail scenes

To enable quantitative evaluation on long-tail scenes, we augment MegaScenes with additional observations from external cultural heritage datasets $[8, 9, 26]$ and jointly register all images using COLMAP. The quantitative and qualitative results of this long-tail evaluation are shown in Fig. 10. Our model consistently reduces the mean relative rotation and

![](images/a42dcfc6516a39f6f3169c034b984a0b7f2fff5bad1504ae44a9e68a12f64174.jpg)  
MRE: 50.90 MTE: 25.90
RRA@5: 8.14 RTA@5: 22.39   
MRE: 10.85 MTE: 9.39
RRA@5: 51.35 RTA@5: 61.24   
MRE: 39.97 MTE: 43.18
RRA@5: 4.76 RTA@5: 23.81   
MRE: 16.04 MTE: 35.76
RRA@5: 23.81 RTA@5: 14.29   
MRE: 5.06 MTE: 6.13
RRA@5: 56.19 RTA@5: 69.52   
MRE: 2.87 MTE: 4.27
RRA@5: 86.67 RTA@5: 80.00

Figure 10. Quantitative results on Long-tail scenes. Our model performs better on scenes with strong ambiguities (first row) and on scenes with minimal overlap across different scene components (second row). For a more densely photographed scene that still exhibits large viewpoint variation (third row), our model not only reduces pose error but also reconstructs a more complete point cloud.

translation errors across all scenes, while also producing more complete point clouds.

# C.5. Limitations

Long-tail scenes often contain fragmented viewpoints, where different subsets of images capture disjoint parts of the scene (e.g., indoor and outdoor areas) without overlapping views to connect them. When such mixed collections are fed into the models at once, both pretrained and finetuned $\pi^{3}$ may blend these unrelated regions into a single 3D structure, as illustrated in Fig.11. While our finetuned model handles these mixtures more robustly than the pretrained baseline, enabling the model to reason robustly about disconnected components and produce reasonable overall layouts still remains a challenge.

![](images/6d1a2b03410d6304402b5d63887f72c5929329554c86ec0291c75a74aaec5b7b.jpg)

<details>
<summary>natural_image</summary>

Collage of black-and-white photos showing historical architecture and scenes, no visible text or symbols
</details>

Input Images

![](images/4fa526296579537f036bd1104255683b663bc10aa2ebbcba1ab3565546690f85.jpg)

<details>
<summary>natural_image</summary>

Sketch-style illustration of a bridge with red triangular markers at its base (no text or symbols)
</details>

Pretrained $\pi^3$

![](images/fc19ee41847aeee04ac28096e780f92f4a6dace96ef3fcd2142dee94a6109b28.jpg)

<details>
<summary>natural_image</summary>

Architectural sketch of a multi-story building with red triangular markers overlay (no text or symbols)
</details>

Ours

![](images/51f3cc9bd555e339bcee4c6b164e71c5ea4a944f5c40171f4fb85c749632544f.jpg)

<details>
<summary>natural_image</summary>

Aerial view of a historic building surrounded by trees and pathways (no visible text or signage)
</details>

Google Earth   
Figure 11. Limitations. This example contains images from two disjoint parts of the scene: indoor photos with warm lighting (producing a yellowish point cloud) and outdoor photos (producing a white point cloud). Pretrained $\pi^{3}$ struggles to handle such mixed inputs and produces inconsistent geometry. Our finetuned model is more robust in this setting, but both models still fuse the indoor and outdoor structures into a single reconstruction without separating them.

![](images/ae4232374f81a497e4d29ca6d3a3e24999bf3e47fa6240b2ff86771046d43fb7.jpg)

<details>
<summary>text_image</summary>

(a) Dragon Bridge (Ljubljana)
(b) Eletsky Monastery
(c) Schloss Linderhof
COLMAP
117 views
Ours
131 views
(d) Royal Albert Hall
(e) Sant'Andrea (Vercelli)
COLMAP
North
South
outsides
inside
Ours
</details>

Figure 12. Comparison of COLMAP and our reconstruction pipeline. We replace COLMAP with MASt3R-SfM $[11]$ combined with the doppelganger++ classifier $[45]$ to obtain sparse reconstructions, allowing effective disambiguation of doppelganger scenes. (a) The bridge has two similar dragon statues, one at each end. COLMAP incorrectly treats them as the same statue and registers them together, whereas our method correctly separates them. (b), (d), and (e) illustrate additional doppelganger cases, in which different sides or parts of a landmark are mistakenly merged. (c) In this low-texture scene, our pipeline also succeeds in registering more images.

![](images/0bdb62a021129ae5b3afb58856a9f4b6a123f602aadc7272f00ecace3faef78a.jpg)

<details>
<summary>text_image</summary>

Image
CHICAGO
COLMAP-MVS
Ours
</details>

Figure 13. Comparison of COLMAP MVS and our filtered dense depth results. COLMAP MVS suffers from depth bleeding and struggles to correctly estimate the depth of transient objects. Our strategy mitigates these issues by leveraging ordering priors from monocular depth predictions. Note that we prioritize accurate depth maps over complete ones. In the last row, COLMAP fails to recover the depth of the foreground statue, and we opt to remove the depth values in that region. If we were to warp the monocular depth to match the MVS result, then any inaccuracy in the relative depth between the statue and the background building could produce erroneous and inconsistent cross-view depth estimates.