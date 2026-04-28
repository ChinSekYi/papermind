#images per Scene

7.0k

82.7k Long Tail Scene (sorted by #images)

## Long-Tail Internet Photo Reconstruction

Pretrained n Pretrained n3

Yuan Li 1 Yuanbo Xiangli 1 † Hadar Averbuch-Elor 1 Noah Snavely 1 Ruojin Cai 2 † 1 Cornell University 2 Kempner Institute, Harvard University

Figure 1. Long-tail Internet photo reconstruction. Internet photo collections follow a long-tailed distribution. In the top plot, the x -axis represents scene index (sorted by image count) and the y -axis shows images per scene (scenes are drawn from MegaScenes [36], a dataset of Internet photo collections). The light blue curve plots the total number of Internet photos per scene, while the steel blue curve shows the size of the subset of photos that were successfully registered using SfM. The head of this distribution of photo collections represents well-photographed scenes; here, there are 6,985 scenes with &gt; 50 registered images. However, most photo collections are in the long tail of this distribution; here, 418,056 scenes with fewer than 50 registered photos. State-of-the-art methods often fail on scenes in this tail. In the lower half of the figure, we show two examples from the long tail, along with representative input images and the corresponding reconstructions. On Calvaire de Plougonven, COLMAP doesn't register any image; on both Duomo (Cagliari)-Crypt and Calvaire de Plougonven, recent feed-forward reconstruction models like π 3 [44] produce poor results. We propose MegaDepth-X dataset and a strategy for mimicking long-tail camera distributions, on which fine-tuned models like π 3 exhibit better reconstruction robustness.

<!-- image -->

## Abstract

Internet photo collections exhibit an extremely long-tailed distribution: a few famous landmarks are densely photographed and easily reconstructed in 3D, while most realworld sites are represented with sparse, noisy, uneven imagery beyond the capabilities of both classical and learned 3D methods. We believe that tackling this long-tail regime represents one of the next frontiers for 3D foundation models. Although reliable ground-truth 3D supervision from sparse scenes is challenging to acquire, we observe that it can be effectively simulated by sampling sparse subsets from well-reconstructed Internet landmarks. To this end, we introduce MegaDepth-X, a large dataset of 3D reconstruc- tions with clean, dense depth, together with a strategy for sampling sets of training images that mimic camera distributions in long-tail scenes. Finetuning 3D foundation models with these components yields robust reconstructions under extreme sparsity, and also enables more reliable reconstruction in symmetric and repetitive scenes, while preserving generalization to standard, dense 3D benchmark datasets. The dataset, finetuned models, and code are available at: https://megadepth-x.github.io/ .

† Corresponding authors.

## 1. Introduction

Internet photo collections of real-world landmarks follow a long-tailed distribution. A small fraction of famous sites, such as the Colosseum or Notre Dame, are photographed

— Registered images Total images

425.0k from every conceivable angle and can be accurately reconstructed by standard Structure-from-Motion (SfM) pipelines. Yet the overwhelming majority of landmarks across the world are represented on the Internet with just a handful of sparse, noisy images (Fig. 1). We refer to this large body of scenes as the long-tail of online photo collections. Such scenes are the norm rather than the exception in real-world Internet imagery.

Reconstructing long-tail scenes is challenging. Classic methods, such as COLMAP [28], often fail because feature correspondence is hard to find across sparse, nonoverlapping, or wide-baseline views. Modern learned feedforward models, like DUSt3R [42] and VGGT [39], can learn powerful priors from millions of images that might help reconstruct long-tail collections. In practice, however, these models are primarily trained on controlled captures with clean, dense, and evenly sampled data. When applied to long-tail Internet scenes featuring sparse, diverse, and unevenly distributed imagery, we find that these models often fail to recover consistent geometry.

We believe that one of the next frontiers for 3D foundation models lies in tackling this long-tail regime of Internet photos. Better data is almost certainly key to this problem, but we cannot easily construct reliable 3D supervision from long-tail collections themselves, as most contain too few overlapping views for robust reconstruction. Instead, we propose to simulate such long-tailed sets by appropriate sampling of sparse images from the large, well-reconstructed Internet landmarks at the head of the distribution, inheriting ground truth from the full reconstruction.

This strategy requires drawing from large amounts of high-quality landmark reconstructions from Internet photos. Existing datasets fall short of this need: MegaDepth [20] is clean but small, while MegaScenes [36] is large but noisy and lacks depth maps. We therefore introduce MegaDepth-X (dubbed MD-X), a next-generation extension of MegaDepth in both scale ( 7 × larger) and quality: a large-scale, clean, and dense-depth-enhanced dataset built from Internet photo reconstructions with consistent depth refinement and extensive manual verification against reliable references (e.g., Google Maps and satellite imagery). Equipped with MD-X, we propose a novel sparsity-aware sampling strategy that mimics the camera distributions of long-tail scenes, encouraging training batches to span wide baselines and partial overlap rather than clustered dense views.

Through extensive experiments, we show that models fine-tuned with MD-X and our sparsity-aware data sampling scheme are significantly more robust on long-tail Internet photo collections, including challenging doppelganger scenes with ambiguous or symmetric content, such as the Calvaire de Plougonven example in Fig. 1, where classical SfM and pretrained foundation models often fail. In summary, our contributions are:

- Defining the 3D long-tail regime : we formalize and characterize the long-tail distribution of Internet photo collections, highlighting this setting's distinct challenges.
- MegaDepth-X , dubbed MD-X, a large-scale, clean, and depth-augmented dataset for finetuning 3D foundation models on real-world Internet scenes.
- Sparsity-aware sampling strategies that simulate the distribution of long-tail Internet collections to improve generalization of 3D prediction models on real-world data.

## 2. Related Work

Feed-forward 3D reconstruction. Reconstructing 3D scene geometry from 2D images is a cornerstone of computer vision. Traditional structure from motion (SfM) [29] and multiview stereo (MVS) [30] methods were crowning achievements of the classic era of 3D vision, and were scaled to large Internet photo collections [1, 12, 34]. Recently, the new paradigm of feed-forward 3D reconstruction has emerged, which involves regressing 3D attributes directly from images in a single pass. Pioneering work in this area, such as DUSt3R, showed success at predicting pixel-aligned point maps from image pairs [42]. MASt3R extended this approach but still relied on pairwise processing [19]. Subsequent efforts focused on scaling these models to arbitrary numbers of views. VGGT [39], along with concurrent models like Fast3R [47] and FLARE [50], introduced large transformer architectures that can process hundreds of views simultaneously. By leveraging large-scale, diverse datasets and a multi-task learning objective, VGGT predicts a full suite of 3D attributes, including camera parameters, depth maps, and point maps. To eliminate reference-frame bias, π 3 [44] recently proposed a permutation-equivariant architecture that predicts affine-invariant camera poses and scaleinvariant local point maps. ZipMap [16] and Scal3R [46] introduced test-time training approaches to process large image collections. These methods work well on denselycaptured and well-conditioned scenes. However, we find that their performance on more sparse and noisy Internet photos remains suboptimal, particularly for long-tail scenes.

Long-tail challenges in 3D vision. Long-tailed problems are pervasive in computer vision. They occur when data for common scenarios (the head) are abundant, but examples of rare yet collectively frequent cases (the tail) are scarce. For instance, many object recognition problems involve a few dominant categories but many rarely seen ones, and in autonomous driving, routine driving scenes are plentiful while safety-critical events are hard to capture.

Recently, MegaScenes [36] introduced a large-scale scene-level dataset built from Internet photo collections, where long-tail effects are particularly pronounced. Many scenes in the dataset are either unreconstructed or incorrectly reconstructed. These failures stem from a combina- together.

tion of view sparsity, noisy imagery, and doppelganger issues [7]. Recent work has sought to address such challenges by developing stronger local features [10, 37] and matchers [15, 17, 21, 27], and by learning wide-baseline pose relationships from large-scale 3D datasets [3, 6]. The doppelganger problem was further addressed by Cai et al. [7, 45], who trained classifiers to prune false matches during the structure-from-motion phase of reconstruction.

While these advances have led to enhanced robustness, they do not yet work reliably at scale. Ideally, we'd mine ground truth 3D training data for long tail scenes and learn to reconstruct them, but that involves a chicken-and-egg problem, because the common practice of using available reconstructors (e.g. COLMAP [28, 31], VGGT [39]) to derive pseudo-ground-truth camera poses and point maps from natural data doesn't work. Instead, similar in spirit to approaches used in autonomous driving that augment training data by simulating rare events, our key idea is to take large, well-conditioned image collections and subsample them to simulate long-tailed photo collections, and use these to better balance training scene distributions for regression models in order to generalize to long-tailed scenes.

## 3. The MegaDepth-X Dataset

Learning in the long-tail regime requires high-quality 3D supervision derived from Internet photo collections. This involves two key challenges. First, reconstructions of Internet photo collections can be unreliable due to noise, dynamic content, and ambiguities [7]. Second, most long-tail scenes lack any usable reconstructions, as classical SfM pipelines like COLMAP [29] often fail on sparse or widely varying image sets. To address these issues, we construct MD-X, a large-scale, clean, and depth-refined dataset that provides reliable 3D supervision, built from well-reconstructed scenes in MegaScenes [36].

## 3.1. Filtering and Disambiguation

Our first step in constructing MD-X is to identify candidate Internet landmarks from which reliable supervision can be derived. We take as our starting pool the subset of MegaScenes with more than 100 registered images, which typically yields stable reconstructions. However, even these 'well-reconstructed' scenes exhibit two common failure modes: (1) Many scenes contain dynamic events or crowded activities, causing feature matches to lock onto moving objects rather than static structures, leading to unreliable reconstructions. (2) The Doppelganger problem [7, 45], where visually similar but geographically distant images are mistakenly registered together. Both issues produce incorrect camera poses and fragmented, inconsistent point clouds as shown in Fig. 2.

To mitigate these issues, we first inspect the dataset and exclude scenes dominated by crowds or moving objects.

Old Town Hall (Prague)

Figure 2. Unreliable reconstructions in MegaScenes. Reconstructions are unreliable when feature matches are incorrectly established on salient, non-static objects (e.g., (a) humans, (b) statues, (c) airplanes) instead of the static scene structure. This results in fragmented and geometrically inconsistent point clouds. Example (d) illustrates a doppelganger failure, where images from opposite sides of the building are incorrectly registered together.

<!-- image -->

Next, we address the doppelganger problem by replacing the default COLMAP SfM reconstruction with MASt3RSfM [19], combined with Doppelganger classification [45]. Specifically, MASt3R-SfM constructs the scene graph using feature matches derived from MASt3R descriptors, after which the Doppelganger classifier identifies and prunes suspicious edges that may result from doppelganger-induced false correspondences. Finally, we manually verify the reconstructed scenes against external references such as Google Maps and satellite imagery, discarding any scenes that do not align with the corresponding bird's-eye view.

## 3.2. Dense Depth Refinement

After obtaining reliable sparse reconstructions, we seek to generate dense depth maps for supervision. We start by running a standard multi-view stereo (MVS) [31] pipeline. We observe, as in prior work [20], that the resulting geometric depth maps from in-the-wild collections often exhibit artifacts, including depth-bleeding effects (background depths leak into foreground regions) and inconsistent and noisy depths in areas with transient objects (e.g., people, cars).

To address these initial issues, we apply the full depth refinement strategy from MegaDepth [20], including a modified MVS procedure that conservatively retains the minimum depth value during propagation, stability filtering to remove flickering pixels, and semantic filtering to exclude transient objects. However, even after this pipeline, we still observe artifacts in the processed geometric depth maps: (1) the MegaDepth-modified MVS still leads to depth-bleeding artifacts, and (2) semantic filtering is not ideal as it relies on a manually designated list of object categories. Examples of such issues are shown in Fig. 3.

Transient Object

Depth Bleeding Image MegaDepth Refinement Our Refinement Therefore, to augment MegaDepth's depth refinement procedure, we propose a monocular depth-guided filtering step. We use depth predictions from MoGe2 [41] as ordinal depth priors, and remove pixels in the processed geometric depth maps that are inconsistent with these priors. Specifically, we first align the processed geometric depths D geom to the monocular predictions D mono by matching their median values over valid pixels: D ′ geom ( p ) = s · D geom ( p ) , where s = med { D mono ( p ) | p ∈ P } med { D geom ( p ) | p ∈ P } . After scale alignment, we compute the normalized depth discrepancy between the two maps: ∆( p ) = | D ′ geom ( p ) -D mono ( p ) | D ′ geom ( p ) , and discard pixels whose discrepancies exceed a predefined threshold τ depth. Moreover, to leverage D mono for edge-aware filtering, we compute the discrepancies between the gradients of the two maps: ∆( p grad ) = | |∇ D mono | D mono -|∇ D ′ geom | D ′ geom | and discard pixels whose discrepancies exceed a predefined threshold τ grad . This approach effectively filters both bleeding artifacts and noisy transient objects without relying on manual category lists, as depicted in Fig. 3.

Figure 3. Depth refinement. MVS depth maps often suffer from artifacts like noise from transient objects (top row) and depth bleeding (bottom row). As shown in the middle column, the MegaDepth refinement pipeline (modified MVS, stability filtering, and semantic filtering) fails to fully remedy these issues. Our method (right column) introduces an additional monocular depth-guided filtering step, which effectively removes transient objects and significantly mitigates depth-bleeding artifacts.

<!-- image -->

## 3.3. Dataset Statistics

In summary, we identify 2,474 candidate scenes from MegaScenes with more than 100 registered images. Of these, 609 scenes are filtered out due to dynamic content, reconstruction errors, or geometric inconsistencies. Our final MD-X dataset comprises 1,865 reconstructions totaling 440k images. We reserve 127 scenes for testing, providing a novel set for evaluating both pretrained and fine-tuned methods. A comparison table with MegaDepth is provided in the supplementary.

## 4. Simulating Long-Tail Scenes

With MD-X providing reliable 3D supervision, the remaining challenge is a complementary supervision coverage problem: existing 3D foundation models are trained predominantly on the head of the Internet-photo distribution, where image collections are large, redundant, and visually well-connected. In this regime, models can rely on strong covisibility and abundant local correspondences. However, most real Internet photo collections lie in the long tail, where views are sparse, unevenly distributed, and only weakly connected. A more complete 3D prior should therefore be robust not only to diverse scene content, but also to this underrepresented observation regime. Rather than seeking unreliable supervision from true long-tail scenes, we start from well-reconstructed scenes in MD-X and sample subsets whose covisibility structure matches that of real long-tail collections. In this way, we expose the model to the missing part of the training distribution while inheriting trustworthy 3D supervision from the full reconstruction.

## 4.1. Defining Properties of Long-Tail Scenes

Common issues like transient occluders and motion blur affect Internet photos broadly, but they are not the primary bottleneck for long-tail scenes. The more fundamental challenge lies in their viewpoint distribution. In these scenes, sparse camera placements lead to limited mutual overlap between images. This results in fragmented, weakly connected clusters rather than a cohesive set, which poses a major hurdle for reliable 3D reconstruction. Because accurate camera poses are often unavailable for such scenes, we characterize this regime using statistics of the SfM view graph rather than absolute camera geometry. Our analysis reveals two consistent patterns: (1) sparser connectivity : scenes with low registration rates (e.g., only 20% of images registered) contain a substantially larger fraction of low-degree nodes, with 8% of cameras having degree two or less, compared with only 3% in well-reconstructed head scenes. This indicates that cameras in long-tail scenes are poorly connected, forming fragmented clusters with limited covisibility. (2) weaker connections : even among connected image pairs, the average number of geometrically verified feature matches is significantly lower in long-tail scenes than in head scenes (294.8 vs. 395.3), indicating reduced overlap and weaker geometric consistency. 1 Together, these observations show that the long tail is not simply a regime of fewer images, but one of sparse and weakly connected observation graphs.

Based on these findings, our sampling process should satisfy three requirements:

- Viewpoint Diversity: The sampled views should cover a wide range of viewing directions, ensuring that emulated scenes span diverse visual perspectives.
- Sparsity: The selected views should be far enough apart to mimic the wide baselines typical of long-tail scenes, e.g . loosely connected views or views from disconnected scene components, encouraging the model to learn robust geo-

1 To avoid statistics being dominated by severely noisy scenes, we compute these measurements only on long-tail subsets containing at least five registered images.

a relain

ole, since

can lead

ition.

pling N

, in order

onnected

nponents

ut within

t internal

at naive

tisfy this Run Louvain algorithm Build Steiner Tree to connect

(based on #matches)

terminal nodes with minimal edges

Covisibility Edge · Camera/View Terminal Node Perform Greedy Search on the subgraph (e.g, after 4 iterations)

· Unselected View

· Sampled View

metric priors rather than relying on dense feature matches.

- Local Reconstructability: Despite the sparsity, views within each sampled scene component should retain enough covisibility to remain locally reconstructable, since zero-overlap samples within a scene component can lead to unstable training signals and difficult optimization.

## 4.2. Sparsity-Aware Sampling Strategy

We therefore formulate the sampling task as sampling N views that form at most N cc connected components, in order to emulate a long-tail scene with multiple weakly connected or disconnected scene components. Specifically, components are allowed to be disconnected from one another, but within each sampled component we still require sufficient internal covisibility for local reconstructability. We find that na¨ ıve random or uniform subsampling often fails to satisfy this balance, producing either zero-overlap sets within scene components or clusters biased toward dense regions. We instead propose a structured sampling process. We first partition views into strongly connected communities and then select a minimal yet diverse subset that ensures both community coverage and global connectivity. This process is illustrated in Fig. 4.

Graph Communities. To promote viewpoint diversity in our sampling, we first identify the dominant 'viewing areas' within each scene. We represent the SfM structure as a view graph G = ( V, E ) , where each node v i ∈ V corresponds to a camera view and each edge ( v i , v j ) ∈ E is weighted by the number of feature matches w ij . We prune edges with w ij &lt; 50 to remove minor overlaps, resulting in a filtered graph G ′ = ( V, E ′ ) that preserves only meaningful covisibility relationships. To reveal clusters of cameras with dense internal connectivity, we perform community detection (e.g., Louvain community detection [4]) on the view graph. This yields viewpoint groups C k that efficiently capture distinct visual regions and the dominant perspectives of the scene. We then randomly partition the graph into N cc connected components that span different communities and do the following steps within each graph partition . The partition algorithm is provided in the supplementary material.

Minimal Connectivity Subgraph. To preserve overall scene connectivity while maintaining sparsity and view diversity within limited nodes, we construct a minimal structure linking all identified communities without reintroducing dense redundancy within each partition. We then compute an approximate Steiner tree to link all of these nodes [18, 22]. 2 In particular, for each training batch for a given training scene, we first randomly select one representative view v k ∈ C k from each community C k to form the terminal set T = { v k } . An approximate Steiner tree algorithm then constructs a minimal connected subgraph G sub = ( V sub , E sub ) , T ⊆ V sub ⊆ V , that spans all terminal nodes using only the necessary intermediate nodes. This yields a compact subgraph connecting all communities using the fewest necessary nodes and edges, preserving global consistency while retaining sparsity. Since G sub can have an arbitrary number of nodes, we need to perform additional sampling to get desired number of views for the training and testing batches.

2 ASteiner tree aims to span a specified set of terminal nodes while introducing only the minimal set of intermediate nodes required for connectivity.

Figure 4. Sparsity-aware sampling strategy. Top: Our method follows a multi-stage process: (1) Apply the Louvain algorithm to the view graph to identify distinct viewpoint communities. (2) From each community, randomly select a terminal view and construct an approximate Steiner Tree to form a minimal, connected subgraph spanning these communities. (3) Perform a Greedy Search on this subgraph to select a sparse and diverse set of views. This procedure aims to cover as many communities as possible while ensuring a wide spatial distribution of cameras within each community. Bottom: A search depth parameter controls the final view coverage. In this example, we sample N = 24 views from the scene with N cc = 1 . With search depth D = 24 , all views are selected via greedy search, producing a more evenly spread distribution. With D = 12 , 12 views come from greedy search and the remaining 12 are sampled locally from the neighborhoods of selected nodes, resulting in a more concentrated distribution.

<!-- image -->

Greedy View Sampling. Inspired by skeletal sets [35], we perform greedy view sampling on the subgraph G sub to select a diverse subset of views for long-tail emulation. The objective is to iteratively expand the sampled set toward broad spatial coverage while maintaining sufficient covisibility among selected view pairs.

At each iteration, the algorithm aims to select the next view based on two criteria: (1) Community novelty : preferring cameras that belong to previously unseen communities, thereby introducing new viewing directions and reducing redundancy; and (2) Spatial distance : encouraging selection of cameras farther from the current viewpoint to promote wider baseline coverage. Specifically, the algorithm operates on a current node v and its connected neighborhood N v . Let S denote the set of already sampled nodes and M be the community map. We first determine which communities have already been reached in S , form- ing the set S comm = { M [ s ] | s ∈ S } . For each neighbor u ∈ N v , we then evaluate its community novelty by checking whether M [ u ] / ∈ S comm, and compute its spatial distance as ∥ Pos( u ) -Pos( v ) ∥ 2 , where Pos( · ) is camera position. Details for this algorithm are provided in the supplemental material. All candidate neighbors are ranked lexicographically by these two attributes, and the top-ranked neighbor u ∗ is selected as the next sampled node. This procedure is repeated for D iterations (i.e., the search depth).

Implementation. In practice, we compute a fixed set of communities C = { C k } for each scene. To form a training batch of N images for a scene, we first randomly divide the N samples across all N cc partitions. In each partition, greedy view sampling stops once either a predefined search-depth limit D is reached or the target number of views assigned to that partition has been sampled. Here, D controls how far the search expands within a partition, hence the sparsity of the resulting set. If this process still produces fewer than N nodes in total, we fill the remaining slots by randomly sampling nodes from the local neighborhoods of the previously sampled nodes. Fig. 4 illustrates an example in which N = 24 and N cc = 1 , and shows the different sparsities of the sampled set obtained under different values of D . Before training, we run the proposed sampling algorithm offline to generate mini-batches of 24 nodes, avoiding costly graph loading during training. We then perform depth-first search from random seed nodes to subsample 2 to 24 images for training batches.

## 5. Experiments

We evaluate how our approach improves 3D reconstruction in the long-tail regime of Internet photo collections. First, we show quantitative results on the proposed MD-X benchmark, demonstrating qualitative improvements on real-world longtail and doppelganger scenes. We then analyze the effect of the proposed dataset and sampling strategy, and finally verify that our fine-tuned models preserve strong performance on standard, curated benchmarks. Further implementation details and additional results are in the supplementary material.

## 5.1. Experimental Setup

Backbones and variants. We finetune two feed-forward 3D foundation models, π 3 [44] and VGGT [39], on MD-X using our proposed sampling strategy. We adopt the loss functions from π 3 [44] and VGGT [39]. To preserve pretrained geometric fidelity, we finetune only the AlternatingAttention modules and keep the point cloud and camera decoders frozen. More training details are in the supplementary. The resulting models are denoted as π 3 -FT and VGGT-FT.

To study how our proposed view sampling strategy affects performance, we finetune π 3 on clean Internet data using four sampling schemes:

Table 1. Quantitative results on MegaDepth-X for camera pose and point map estimation across two difficulty levels. Our finetuned models ( π 3 -FT and VGGT-FT) trained with the proposed dataset and sampling strategy consistently outperform pretrained baselines, especially on harder, sparser scenes.

|              | Camera Pose Estimation   | Camera Pose Estimation   | Camera Pose Estimation   | Camera Pose Estimation   | Camera Pose Estimation   | Point Map Estimation   | Point Map Estimation   | Point Map Estimation   | Point Map Estimation   | Point Map Estimation   | Point Map Estimation   |
|--------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|
| Method       | RRA@5 ↑                  | RTA@5 ↑                  | AUC@5 ↑                  | MRE ↓                    | MTE ↓                    | Acc ↓                  | Acc ↓                  | Comp ↓                 | Comp ↓                 | NC ↑                   | NC ↑                   |
|              |                          |                          |                          |                          |                          | Mean                   | Med.                   | Mean                   | Med.                   | Mean                   | Med.                   |
| easy π 3 π 3 | 88.97                    | 68.79                    | 45.84                    | 4.12                     | 7.82                     | 0.055                  | 0.030                  | 0.039                  | 0.019                  | 0.712                  | 0.822                  |
| -FT          | 95.64                    | 76.85                    | 55.58                    | 1.64                     | 5.50                     | 0.035                  | 0.020                  | 0.024                  | 0.012                  | 0.724                  | 0.837                  |
| VGGT         | 84.17                    | 58.47                    | 35.32                    | 4.55                     | 9.93                     | 0.093                  | 0.047                  | 0.055                  | 0.026                  | 0.695                  | 0.798                  |
| VGGT-FT      | 92.41                    | 71.12                    | 48.78                    | 2.70                     | 7.02                     | 0.050                  | 0.027                  | 0.033                  | 0.014                  | 0.719                  | 0.833                  |
| π 3          | 75.31                    | 59.16                    | 36.93                    | 12.21                    | 10.82                    | 0.101                  | 0.065                  | 0.133                  | 0.090                  | 0.689                  | 0.786                  |
| hard π 3 -FT | 86.40                    | 71.00                    | 47.93                    | 5.72                     | 7.27                     | 0.068                  | 0.041                  | 0.066                  | 0.041                  | 0.713                  | 0.818                  |
| VGGT         | 70.98                    | 52.98                    | 29.10                    | 13.20                    | 13.34                    | 0.149                  | 0.092                  | 0.151                  | 0.104                  | 0.675                  | 0.764                  |
| VGGT-FT      | 81.07                    | 65.59                    | 41.49                    | 7.22                     | 9.05                     | 0.089                  | 0.053                  | 0.084                  | 0.055                  | 0.709                  | 0.814                  |

- DENSE: training batches with densely overlapping views where D = 5 and N cc = 1 ,
- SPARSE: long-tail-like sampling emphasizing wide baselines where D = 24 and N cc = 4 ,
- MIXED: a combination of dense and sparse batches for balanced learning with D ∈ [5 , 24] and N cc ∈ [1 , 4] ,
- RANDOM: random view sampling.

Unless otherwise noted, FT (e.g., π 3 -FT) refers to the model finetuned on the cleaned dataset using the MIXED sampling strategy above. We additionally train a DIRTY variant on Internet data (using the same Mixed scheme) without the filtering strategy in Sec. 3.1, while keeping the same depth refinement pipeline in Sec. 3.2, to assess robustness to label noise and data contamination.

Evaluation Metrics. For camera pose estimation, we follow prior work [39, 44] and report Relative Rotation Accuracy (RRA), Relative Translation Accuracy (RTA), and their combined Area Under Curve (AUC). We also report mean rotation and translation errors (MRE and MTE, in degrees). For point map evaluation, we follow prior work [2, 38, 40, 42, 44] and report Accuracy (Acc), Completeness (Comp), and Normal Consistency (NC), each computed as the mean and median across test scenes.

## 5.2. Internet Photo Evaluation

We first evaluate models on the proposed MD-X benchmark, which contains Internet photo collections of varying sparsity and difficulty. For each test scene, we sample 24 images from the reconstructed scene graph using our sampling algorithm, and categorize them into easy ( D = 5 , N cc = 1 ) and hard ( D = 24 , N cc = 4 ) subsets according to the greedy search depth used for test data sampling.

Quantitative Results. Tab. 1 reports quantitative results for camera pose and point map estimation across three difficulty levels on MD-X. Finetuning markedly improves both π 3 and VGGT over their pretrained baselines, with larger gains observed in harder, sparser scenes. These improvements hold across metrics indicate that the fine-tuned models better capture global structure and maintain consistent 3D geometry Search depth hard Diaura 5 1

easy GT

Pretrained n°

Ours

Table 2. Ablation study on MegaDepth-X. Finetuning on the cleaned dataset with MIXED dense-sparse sampling ( π 3 -FT) yields the best overall performance, while training on unfiltered data (DIRTY) degrades accuracy.

|                 | Camera Pose   | Camera Pose   | Camera Pose   | Camera Pose   | Camera Pose   | Point Map Estimation   | Point Map Estimation   | Point Map Estimation   | Point Map Estimation   | Point Map Estimation   | Point Map Estimation   |
|-----------------|---------------|---------------|---------------|---------------|---------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|
| Method          | RRA@5 ↑       | RTA@5 ↑       | AUC@5 ↑       | MRE ↓         | MTE ↓         | Acc ↓                  | Acc ↓                  | Comp ↓                 | Comp ↓                 | NC ↑                   | NC ↑                   |
|                 |               |               |               |               |               | Mean                   | Med.                   | Mean                   | Med.                   | Mean                   | Med.                   |
| π 3             | 88.97         | 68.79         | 45.84         | 4.12          | 7.82          | 0.055                  | 0.030                  | 0.039                  | 0.019                  | 0.712                  | 0.822                  |
| π 3 -FT         | 95.64         | 76.85         | 55.58         | 1.64          | 5.50          | 0.035                  | 0.020                  | 0.024                  | 0.012                  | 0.724                  | 0.837                  |
| easy π 3 -DIRTY | 91.25         | 72.80         | 51.77         | 5.16          | 7.28          | 0.075                  | 0.052                  | 0.081                  | 0.051                  | 0.710                  | 0.818                  |
| π 3 -RANDOM     | 95.08         | 76.42         | 55.00         | 1.78          | 5.72          | 0.039                  | 0.021                  | 0.026                  | 0.013                  | 0.720                  | 0.831                  |
| π 3 -DENSE      | 95.13         | 76.73         | 55.65         | 1.84          | 5.61          | 0.036                  | 0.020                  | 0.026                  | 0.013                  | 0.725                  | 0.837                  |
| π 3 -SPARSE     | 96.27         | 76.46         | 55.12         | 1.61          | 5.59          | 0.038                  | 0.020                  | 0.026                  | 0.013                  | 0.723                  | 0.835                  |
| π 3             | 75.31         | 59.16         | 36.93         | 12.21         | 10.82         | 0.101                  | 0.065                  | 0.133                  | 0.090                  | 0.689                  | 0.786                  |
| π 3 -FT         | 86.40         | 71.00         | 47.93         | 5.72          | 7.27          | 0.068                  | 0.041                  | 0.066                  | 0.041                  | 0.713                  | 0.818                  |
| hard π 3 -DIRTY | 81.10         | 65.99         | 43.74         | 11.86         | 9.72          | 0.130                  | 0.094                  | 0.139                  | 0.091                  | 0.693                  | 0.791                  |
| π 3 -RANDOM     | 85.93         | 69.84         | 47.17         | 6.53          | 7.78          | 0.071                  | 0.040                  | 0.073                  | 0.045                  | 0.708                  | 0.812                  |
| π 3 -DENSE      | 85.82         | 70.06         | 47.47         | 6.04          | 7.64          | 0.071                  | 0.042                  | 0.062                  | 0.035                  | 0.713                  | 0.817                  |
| π 3 -SPARSE     | 85.97         | 70.53         | 47.13         | 6.05          | 7.52          | 0.070                  | 0.040                  | 0.070                  | 0.041                  | 0.710                  | 0.814                  |

in sparse settings.

Ablation Analysis. We analyze the effects of data quality and sampling strategies, with results shown in Tab. 2. Training on unfiltered (DIRTY) data consistently reduces accuracy, even performing worse than the pretrained model in point-map estimation on both the easy and hard levels, highlighting the importance of clean supervision for robust generalization. Among sampling schemes, RANDOM sampling yields reasonable camera pose accuracy but provides limited improvement in point map reconstruction, emphasizing the importance of adequate covisibilities in training batches. DENSE sampling performs well on easier scenes but is less effective under sparse conditions. SPARSE sampling alone does not yield the best trade-off. Although it exposes the model to more challenging cases, MIXED sampling achieves slightly better overall performance across difficulty levels.

Qualitative Analysis. We show qualitative results for three settings: the MD-X test set, real-world long-tail Internet scenes, and doppelganger scenes.

MegaDepth-X Visualization. Fig. 5 shows reconstruction results on the MD-X test set across easy and hard levels. Our fine-tuned model produces more accurate camera poses, more dense and consistent 3D point maps compared to the pretrained baseline, especially on sparse ( hard ) scenes. It generalizes well across varying camera intrinsics and challenging appearance changes such as day-night shifts.

Real Long-Tail Scenes. Real long-tail Internet scenes often contain fewer than 100 usable photos captured from uneven viewpoints and mixed with transient or irrelevant content. Classical SfM pipelines, e.g., COLMAP, typically fail to register most images, producing extremely sparse geometry or incomplete reconstructions. Pretrained models struggle under these conditions, yielding low-confidence predictions and fragmented structures. Our finetuned model remains stable and reconstructs coherent global geometry. As shown in Fig. 6, our model successfully reconstructs dense geometry from very few views, and handles doppelganger ambiguities with higher confidence, demonstrating strong robustness and generalization to real-world long-tail scenes. In the supple- Search depth mentary material, we provide more results on doppelganger scenes.

Figure 5. Reconstruction results on the MegaDepth-X test set across two difficulty levels. For each level, the top row shows the full 24-image input set, and the bottom row compares reconstructions from ground truth, pretrained π 3 , and our finetuned model with top-down views shown in the insets. Our model shows clearer improvements in the hard setting, where the inputs are more challenging. Note that hard was obtained using a deeper search depth than easy.

<!-- image -->

Table 3. Camera pose estimation on RealEstate10K [51] and CO3Dv2 [25] . We follow π 3 's pose sampling conventions. Our fine-tuned models, trained on proposed Internet data dataset, remain comparable to pretrained baselines, demonstrating generalization to standard benchmarks.

|         | RealEstate10K   | RealEstate10K   | RealEstate10K   | RealEstate10K   | RealEstate10K   | CO3Dv2   | CO3Dv2   | CO3Dv2   | CO3Dv2   | CO3Dv2   |
|---------|-----------------|-----------------|-----------------|-----------------|-----------------|----------|----------|----------|----------|----------|
| Method  | RRA@5 ↑         | RTA@5 ↑         | AUC@5 ↑         | MRE ↓           | MTE ↓           | RRA@5 ↑  | RTA@5 ↑  | AUC@5 ↑  | MRE ↓    | MTE ↓    |
| π 3     | 98.79           | 79.61           | 62.82           | 0.51            | 5.65            | 93.24    | 84.47    | 57.12    | 3.04     | 4.28     |
| π 3 -FT | 98.80           | 77.78           | 60.01           | 0.51            | 6.13            | 93.97    | 84.50    | 57.61    | 2.96     | 4.26     |
| VGGT    | 97.49           | 62.32           | 38.09           | 1.03            | 8.66            | 96.97    | 86.19    | 67.84    | 2.33     | 3.95     |
| VGGT-FT | 98.23           | 71.88           | 48.23           | 0.82            | 6.85            | 97.11    | 86.27    | 67.81    | 2.29     | 3.92     |

## 5.3. Generalization to Standard Benchmarks

We next examine whether the finetuned models preserve generalization on standard, curated benchmarks.

Relative Pose Estimation. We evaluate on RealEstate10K [51] and CO3Dv2 [25], following π 3 's pose sampling conventions. As shown in Tab. 3, fine-tuning on Internet data generally maintains the performance of both backbones, and yields modest improvements for VGGT in particular. These results indicate that robustness learned from sparse, in-thewild Internet photos does not compromise generalization to standard 3D benchmarks.

Point Map Estimation. Results on DTU [14], ETH3D [32], 7-Scenes [33], and NRGBD [2] (Tab. 4&amp;5) show that our model maintains comparable reconstruction accuracy on tending this ects, indoor a promising del.

Image Collection

Pretrained 13

COLMAP &amp;

Reference

VGGT

View NRGBD, likely reflecting the domain gap between these clean, controlled datasets and Internet imagery. Overall, the results indicate that training on diverse Internet photos preserves cross-dataset generalization without overfitting.

Figure 6. Reconstruction results on real long-tail Internet scenes. Each scene contains only a handful of photos with uneven viewpoints and noisy content, where COLMAP fails to register most images and produces extremely sparse geometry. Pretrained π 3 makes low-confidence predictions and incomplete reconstructions, while our fine-tuned model discovers the correct large-scale layout (e.g., (1) Novo-Znamenka Manor , 66 images, 13 registered), handles very few-view inputs and recovers dense geometry ((2) Sobanski Palace in Guzow , 95 images, 11 registered), reconstructs more complete structures under sparse, long-tail settings ((3) Delizia del Verginese (Gambulaga, Portomaggiore) , 69 images, 11 registered, (5) Chitharal Jain Monuments , 44 images, 15 registered), resolves doppelganger ambiguity ((4) Hoshang's Tomb , 85 images, 40 registered), and even works when COLMAP completely fails ((6) Chapel of Saint Andrew's cathedral (Saint Petersburg) , 94 images, 0 registered). These results demonstrate that our model remains robust and confident under severe sparsity and ambiguity in real long-tail Internet scenes. For each scene, the confidence threshold is the same for pretrained π 3 and our method.

<!-- image -->

Table 4. Point map estimation on DTU [14] and ETH3D [32] . Finetuning on the proposed Internet photo dataset retain overall reconstruction quality on DTU, while performance on ETH3D decreases due to domain mismatch with Internet imagery. These results show that the model adapts to Internet photos without drifting too much on out-of-domain benchmarks.

struction by defining and addressing the long-tail regime

## 6. Conclusion

Image Collection We presented a step towards robust, Internet-scale 3D reconstruction by defining and addressing the long-tail regime of Internet photo collections. Through the MegaDepthX dataset and a sparsity-aware sampling strategy, we augment the ability of 3D foundation models to recover consistent geometry from sparse, noisy, and ambiguous imagery, where classical SfM and SOTA feed-forward 3D reconstruction models fail, and demonstrates disambiguation of doppelganger scenes while maintaining generalization across benchmarks.

Our dataset currently focuses on landmark-scale scenes, representing only a small fraction of the landscape of Internet photos. Bootstrapping on the current dataset and refining models for reconstructions of even more longed-tail data remains an important direction for future work. Extending this framework beyond landmarks to everyday objects, indoor scenes, and other Internet photo domains offers a promising path toward a truly universal 3D foundation model.

Table 5. Point map estimation on 7-Scenes [33] and NRGBD [2] datasets. We evaluate both sparse-view and dense-view settings. Finetuning on Internet photos yields comparable performance to pretrained baselines with minor variations, indicating our method preserves generalization across diverse real world and synthetic datasets.

|         | DTU    | DTU    | DTU     | DTU     | DTU    | DTU    | ETH3D   | ETH3D   | ETH3D   | ETH3D   | ETH3D   | ETH3D   |
|---------|--------|--------|---------|---------|--------|--------|---------|---------|---------|---------|---------|---------|
| Method  | Acc. ↓ | Acc. ↓ | Comp. ↓ | Comp. ↓ | N.C. ↑ | N.C. ↑ | Acc. ↓  | Acc. ↓  | Comp. ↓ | Comp. ↓ | N.C. ↑  | N.C. ↑  |
|         | Mean   | Med.   | Mean    | Med.    | Mean   | Med.   | Mean    | Med.    | Mean    | Med.    | Mean    | Med.    |
| π 3     | 1.151  | 0.622  | 1.793   | 0.629   | 0.668  | 0.754  | 0.188   | 0.126   | 0.211   | 0.129   | 0.872   | 0.967   |
| π 3 -FT | 1.202  | 0.642  | 1.928   | 0.593   | 0.666  | 0.751  | 0.199   | 0.142   | 0.242   | 0.151   | 0.861   | 0.955   |
| VGGT    | 1.308  | 0.761  | 1.929   | 1.015   | 0.665  | 0.750  | 0.270   | 0.174   | 0.304   | 0.180   | 0.841   | 0.942   |
| VGGT-FT | 1.283  | 0.759  | 1.900   | 0.953   | 0.669  | 0.756  | 0.282   | 0.205   | 0.394   | 0.225   | 0.838   | 0.927   |

DTU, 7-Scenes and NRGBD. We observe a performance decrease on ETH3D and a mild drop for VGGT under sparse

|        |              | 7-Scenes    | 7-Scenes    | 7-Scenes    | 7-Scenes    | 7-Scenes    | 7-Scenes    | NRGBD       | NRGBD       | NRGBD       | NRGBD       | NRGBD       | NRGBD       |
|--------|--------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| View   | Method       | Acc. ↓      | Acc. ↓      | Comp. ↓     | Comp. ↓     | NC. ↑       | NC. ↑       | Acc. ↓      | Acc. ↓      | Comp. ↓     | Comp. ↓     | NC. ↑       | NC. ↑       |
|        |              | Mean        | Med.        | Mean        | Med.        | Mean        | Med.        | Mean        | Med.        | Mean        | Med.        | Mean        | Med.        |
| sparse | π 3 π 3 -FT  | 0.047 0.046 | 0.029 0.027 | 0.074 0.072 | 0.049 0.046 | 0.741 0.739 | 0.840 0.841 | 0.024 0.024 | 0.013 0.014 | 0.028 0.028 | 0.013 0.014 | 0.909 0.903 | 0.991 0.990 |
| sparse | VGGT VGGT-FT | 0.044 0.062 | 0.024 0.046 | 0.056 0.097 | 0.033 0.070 | 0.733 0.738 | 0.846 0.844 | 0.049 0.071 | 0.027 0.046 | 0.066 0.071 | 0.037 0.041 | 0.882 0.875 | 0.979 0.959 |
| dense  | π 3 π 3 -FT  | 0.016 0.016 | 0.007 0.007 | 0.022 0.023 | 0.011 0.011 | 0.689 0.686 | 0.792 0.789 | 0.013 0.013 | 0.007 0.007 | 0.014 0.014 | 0.006 0.005 | 0.874 0.864 | 0.981 0.978 |
| dense  | VGGT         | 0.022 0.016 | 0.008 0.007 | 0.026 0.027 | 0.012       | 0.667 0.681 | 0.760 0.781 | 0.015       | 0.008       | 0.015       | 0.006       | 0.871       | 0.982       |
| dense  | VGGT-FT      |             |             |             | 0.012       |             |             | 0.015       | 0.008       | 0.016       | 0.006       | 0.859       | 0.981       |

Pretr Acknowledgments This work was supported in part by the Institute of Information &amp; Communications Technology Planning &amp; Evaluation (IITP) grant funded by the Korean Government (MSIT) (No. RS-2024-00457882, National AI Research Lab Project). We thank Joseph Tung, Yiwen Zhang, Hanyu Chen and Haian Jin for discussion and help with MegaScenes dataset and depth post-processing.

## References

- [1] Sameer Agarwal, Yasutaka Furukawa, Noah Snavely, Ian Simon, Brian Curless, Steven M Seitz, and Richard Szeliski. Building rome in a day. Communications of the ACM , 54(10): 105-112, 2011. 2
- [2] Dejan Azinovi´ c, Ricardo Martin-Brualla, Dan B Goldman, Matthias Nießner, and Justus Thies. Neural rgb-d surface reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pages 6290-6301, 2022. 6, 7, 8
- [3] Hana Bezalel, Dotan Ankri, Ruojin Cai, and Hadar AverbachElor. Extreme rotation estimation in the wild. In Proceedings of the Computer Vision and Pattern Recognition Conference , pages 1061-1070, 2025. 3
- [4] Vincent D Blondel, Jean-Loup Guillaume, Renaud Lambiotte, and Etienne Lefebvre. Fast unfolding of communities in large networks. Journal of statistical mechanics: theory and experiment , 2008(10):P10008, 2008. 5
- [5] Daniel J. Butler, Jonas Wulff, Garrett B. Stanley, and Michael J. Black. A naturalistic open source movie for optical flow evaluation. In Proceedings of the 12th European Conference on Computer Vision - Volume Part VI , page 611-625, Berlin, Heidelberg, 2012. Springer-Verlag. 3, 4
- [6] Ruojin Cai, Bharath Hariharan, Noah Snavely, and Hadar Averbuch-Elor. Extreme rotation estimation using dense correlation volumes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 14566-14575, 2021. 3
- [7] Ruojin Cai, Joseph Tung, Qianqian Wang, Hadar AverbuchElor, Bharath Hariharan, and Noah Snavely. Doppelgangers: Learning to disambiguate images of similar structures. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 34-44, 2023. 3
- [8] Filiberto Chiabrando, Loren Clark, John Driscoll, Scott McAvoy, Dominique Rissolo, Alessandra Spreafico, and Beatrice Tanduo. Salvation mountain - photogrammetry - terrestrial, photogrammetry - aerial, lidar - terrestrial, lidar - mobile, survey data, 2023. Distributed by Open Heritage 3D. 4
- [9] CyArk. Great mosque - kilwa kisiwani - lidar - terrestrial, photogrammetry - terrestrial, photogrammetry - aerial, 2020. Distributed by Open Heritage 3D. 4
- [10] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superpoint: Self-supervised interest point detection and description. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops , pages 224-236, 2018. 3
- [11] Bardienus Pieter Duisterhof, Lojze Zust, Philippe Weinzaepfel, Vincent Leroy, Yohann Cabon, and Jerome Revaud.
12. MASt3r-sfm: a fully-integrated solution for unconstrained structure-from-motion. In International Conference on 3D Vision 2025 , 2025. 6
- [12] Jan-Michael Frahm, Pierre Fite-Georgel, David Gallup, Tim Johnson, Rahul Raguram, Changchang Wu, Yi-Hung Jen, Enrique Dunn, Brian Clipp, Svetlana Lazebnik, and Marc Pollefeys. Building Rome on a Cloudless Day. In ECCV , 2010. 2
- [13] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. International Journal of Robotics Research (IJRR) , 2013. 3, 4
- [14] Rasmus Jensen, Anders Dahl, George Vogiatzis, Engil Tola, and Henrik Aanæs. Large scale multi-view stereopsis evaluation. In 2014 IEEE Conference on Computer Vision and Pattern Recognition , pages 406-413. IEEE, 2014. 7, 8
- [15] Hanwen Jiang, Hanwen Jiang, Arjun Karpur, Bingyi Cao, Qixing Huang, and Qi-Xing Huang. Omniglue: Generalizable feature matching with foundation model guidance. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pages 19865-19875, 2024. 3
- [16] Haian Jin, Rundi Wu, Tianyuan Zhang, Ruiqi Gao, Jonathan T. Barron, Noah Snavely, and Aleksander Holynski. ZipMap: Linear-time stateful 3d reconstruction via test-time training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2026. 2
- [17] Arjun Karpur, Guilherme Perrotta, Ricardo Martin-Brualla, Howard Zhou, and Andre F. de Ara´ ujo. Lfm-3d: Learnable feature matching across wide baselines using 3d signals. 2024 International Conference on 3D Vision (3DV) , pages 11-20, 2023. 3
- [18] Lawrence Kou, George Markowsky, and Leonard Berman. A fast algorithm for steiner trees. Acta informatica , 15(2): 141-145, 1981. 5
- [19] Vincent Leroy, Yohann Cabon, and J´ erˆ ome Revaud. Grounding image matching in 3d with mast3r, 2024. 2, 3
- [20] Zhengqi Li and Noah Snavely. Megadepth: Learning singleview depth prediction from internet photos. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 2041-2050, 2018. 2, 3, 1
- [21] Philipp Lindenberger, Paul-Edouard Sarlin, and Marc Pollefeys. Lightglue: Local feature matching atokens are frozen. arXiv preprint arXiv:2306.13643 , 2023. 3
- [22] Kurt Mehlhorn. A faster approximation algorithm for the steiner problem in graphs. Information Processing Letters , 27 (3):125-128, 1988. 5
- [23] Pushmeet Kohli Nathan Silberman, Derek Hoiem and Rob Fergus. Indoor segmentation and support inference from rgbd images. In ECCV , 2012. 3, 4
- [24] E. Palazzolo, J. Behley, P. Lottes, P. Gigu` ere, and C. Stachniss. ReFusion: 3D Reconstruction in Dynamic Environments for RGB-D Cameras Exploiting Residuals. arXiv , 2019. 3, 4
- [25] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotn´ y. Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. 2021 IEEE/CVF International Conference on Computer Vision (ICCV) , pages 10881-10891, 2021. 7
- [26] Ashley Richter, Michael Hess, Vid Petrovic, Falko Kuester, Cultural Heritage Engineering Initiative (CHEI), Architecture Center of Interdisciplinary Science for Art, and Archaeology (CISA3). Torre dei baldovinetti - florence - lidar - terrestrial, photogrammetry - terrestrial, 2023. Distributed by Open Heritage 3D. 4
- [27] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superglue: Learning feature matching with graph neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 4938-4947, 2020. 3
- [28] Johannes Lutz Sch¨ onberger and Jan-Michael Frahm. Structure-from-motion revisited. In Conference on Computer Vision and Pattern Recognition (CVPR) , 2016. 2, 3
- [29] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In CVPR , 2016. 2, 3
- [30] Johannes L. Sch¨ onberger, Enliang Zheng, Jan-Michael Frahm, and Marc Pollefeys. Pixelwise view selection for unstructured multi-view stereo. In European Conference on Computer Vision , 2016. 2
- [31] Johannes Lutz Sch¨ onberger, Enliang Zheng, Marc Pollefeys, and Jan-Michael Frahm. Pixelwise view selection for unstructured multi-view stereo. In European Conference on Computer Vision (ECCV) , 2016. 3
- [32] Thomas Sch¨ ops, Johannes L. Sch¨ onberger, Silvano Galliani, Torsten Sattler, Konrad Schindler, Marc Pollefeys, and Andreas Geiger. A multi-view stereo benchmark with highresolution images and multi-camera videos. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , pages 2538-2547, 2017. 7, 8
- [33] Jamie Shotton, Ben Glocker, Christopher Zach, Shahram Izadi, Antonio Criminisi, and Andrew Fitzgibbon. Scene coordinate regression forests for camera relocalization in rgbd images. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 2930-2937, 2013. 7, 8
- [34] Noah Snavely, Steven M Seitz, and Richard Szeliski. Photo tourism: exploring photo collections in 3d. In ACM siggraph 2006 papers , pages 835-846. 2006. 2
- [35] Noah Snavely, Steven M Seitz, and Richard Szeliski. Skeletal graphs for efficient structure from motion. In 2008 IEEE Conference on Computer Vision and Pattern Recognition , pages 1-8. IEEE, 2008. 5
- [36] Joseph Tung, Gene Chou, Ruojin Cai, Guandao Yang, Kai Zhang, Gordon Wetzstein, Bharath Hariharan, and Noah Snavely. Megascenes: Scene-level view synthesis at scale. arXiv preprint arXiv:2406.11819 , 2024. 1, 2, 3
- [37] Michał Tyszkiewicz, Pascal Fua, and Eduard Trulls. Disk: Learning local features with policy gradient. Advances in Neural Information Processing Systems , 33:14254-14265, 2020. 3
- [38] Hengyi Wang and Lourdes Agapito. 3d reconstruction with spatial memory. arXiv preprint arXiv:2408.16061 , 2024. 6
- [39] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference , pages 5294-5306, 2025. 2, 3, 6
- [40] Qianqian Wang*, Yifei Zhang*, Aleksander Holynski, Alexei A. Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. In CVPR , 2025. 6, 3
- [41] Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and Jiaolong Yang. Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision. In Proceedings of the Computer Vision and Pattern Recognition Conference , pages 5261-5271, 2025. 4
- [42] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. arXiv preprint arXiv:2312.14132 , 2023. 2, 6
- [43] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer. Tartanair: A dataset to push the limits of visual slam. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) , pages 4909-4916. IEEE, 2020. 3
- [44] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π 3 : Scalable permutation-equivariant visual geometry learning, 2025. 1, 2, 6, 3
- [45] Yuanbo Xiangli, Ruojin Cai, Hanyu Chen, Jeffrey Byrne, and Noah Snavely. Doppelgangers++: Improved visual disambiguation with geometric 3d features, 2025. 3, 6
- [46] Tao Xie, Peishan Yang, Yudong Jin, Yingfeng Cai, Wei Yin, Weiqiang Ren, Qian Zhang, Wei Hua, Sida Peng, Xiaoyang Guo, and Xiaowei Zhou. Scal3r: Scalable test-time training for large-scale 3d reconstruction, 2026. 2
- [47] Jianing Yang, Alexander Sax, Kevin J Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In Proceedings of the Computer Vision and Pattern Recognition Conference , pages 21924-21935, 2025. 2
- [48] Yao Yao, Zixin Luo, Shiwei Li, Jingyang Zhang, Yufan Ren, Lei Zhou, Tian Fang, and Long Quan. Blendedmvs: A largescale dataset for generalized multi-view stereo networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 1790-1799, 2020. 3
- [49] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and Ming-Hsuan Yang. Monst3r: A simple approach for estimating geometry in the presence of motion. arXiv preprint arxiv:2410.03825 , 2024. 3
- [50] Shangzhan Zhang, Jianyuan Wang, Yinghao Xu, Nan Xue, Christian Rupprecht, Xiaowei Zhou, Yujun Shen, and Gordon Wetzstein. Flare: Feed-forward geometry, appearance and camera estimation from uncalibrated sparse views. In Proceedings of the Computer Vision and Pattern Recognition Conference , pages 21936-21947, 2025. 2
- [51] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. arXiv preprint arXiv:1805.09817 , 2018. 7

## Visualization Webpage

Please refer to our project page for additional visualizations beyond this PDF. The webpage includes: (i) animations of our sparsity-aware sampling procedure on representative scenes; and (ii) comparisons of reconstructions from pretrained π 3 and our finetuned π 3 on long-tail scenes (where COLMAP registers 0 images). We also provide video flythroughs of reconstructed point clouds and additional qualitative results on the webpage to help visualize performance on diverse, real-world scenes.

## Long-Tail Internet Photo Reconstruction

## Supplementary Material

our sampling strategy to construct more diverse and representative sparse-view subsets. The statistics are computed from Manhattan-aligned COLMAP reconstructions.

## A. The MegaDepth-X Dataset

## A.1. Data Processing

In this section, we compare COLMAP results with those produced by our proposed data-processing pipeline. Fig. 12 shows reconstructions from COLMAP and our MASt3RSfM pipeline. COLMAP often fails on ambiguous scenes involving similar-looking objects, visually similar but distinct building facades, symmetric landmarks etc. In contrast, our reconstruction pipeline effectively mitigates these issues and recovers correct geometry. In Fig. 13, we show that our monocular depth-guided dense depthmap filtering strategy prevents background depths from leaking into foreground regions (i.e. the depth-bleeding issue [20]) and removes depth estimates on transient objects, which are often unreliable in COLMAP MVS. Note that we use monocular depth only as guidance, rather than warping it to align with the MVS depth. This is because we prioritize accurate depth maps over complete ones. Uncertainty in the relative depth predictions of monocular models can introduce additional noise and inconsistency across views. For example, in the last row of Fig. 13, COLMAP MVS fails to recover the depth of the foreground statue, and we opt to remove the depth values in that region. If we were to warp the monocular depth to match the MVS result, then any inaccuracy in the relative depth between the statue and the background building could produce erroneous and inconsistent cross-view depth estimates.

## A.2. Dataset Statistics

We provide an overall comparison between MegaDepth and MegaDepth-X in Tab. 6, including reconstruction statistics as well as several metrics that characterize the spatial distribution of viewpoints. Beyond basic dataset properties such as the number of intact reconstructions, image count, and whether doppelganger filtering or dense depth refinement is applied, we analyze how cameras are positioned and oriented in each scene, as scenes with broad viewpoint coverage allow Positional coverage. To understand how cameras are placed in the horizontal plane, we compute each camera's azimuth angle relative to the scene centroid (that is, the angle of the direction from the scene centroid to the camera) and divide the full 0-360° range into 36 equal 10° bins. In practice, the scene centroid is derived from the average of the SFM point cloud. A scene with many occupied bins is one where cameras are well-distributed around the object. In the table, the columns 'Positional Azimuth Coverage = 100% / ≥ 75% / ≥ 50% / ≥ 25% ' report how many scenes achieve at least that percentage of bins( 36 / 36 , 27 / 36 , 18 / 36 , 9 / 36 ), with larger thresholds indicating closer to full 360° wrap-around coverage.

Rotational coverage. Position alone does not describe where cameras are looking. We therefore measure the coverage of camera orientations by mapping each camera's forward viewing direction to 36 azimuth bins similar to positional coverage. If cameras face more distinct directions, more bins are occupied; if they face similar directions, only few bins are occupied. We summarize this rotational azimuth coverage using the same percentage thresholds as positional azimuth coverage.

These statistics show that MegaDepth-X contains substantially more scenes with broad camera-position coverage and diverse viewing directions, making it better suited for robust sparse-view reconstruction than MegaDepth.

## B. Sparsity-aware Sampling

## B.1. Greedy Sampling Algorithm

We illustrate one iteration of the greedy view-sampling procedure in Alg. 1. At each step, the algorithm selects the next view based on two criteria:

1. Community novelty : prioritizing candidates whose camera-community has not yet been visited by the sampled set. This encourages the trajectory to enter unexplored regions of the view graph and reduces redundancy in viewpoint selection.
2. Spatial distance : among candidates with equal novelty, preferring those that are farther from the current camera position. This promotes larger baselines and helps diversify the spatial coverage of the sampled views.

Table 6. Dataset statistics and viewpoint-distribution metrics. We report reconstruction statistics and metrics describing camera coverage. Positional Azimuth Coverage counts scenes whose camera positions occupy 9-36 (i.e. 25%-100%) of the 36 horizontal azimuth bins (10° per bin, covering the full 360°). Rotational Azimuth Coverage represents scenes whose camera forwarding vectors occupy 9-36 (i.e. 25%-100%) of the 36 horizontal azimuth bins (10° per bin, covering the full 360°). For each scene, the more bins covered, the wider the camera distribution is. † Dense depth refinement uses monocular depth-guided filtering.

| Dataset            |   #Recons. #Images |      | Doppelganger Dense Depth   | Doppelganger Dense Depth   | Positional Azimuth Coverage   | Positional Azimuth Coverage   | Positional Azimuth Coverage   | Positional Azimuth Coverage   | Rotational Azimuth Coverage   | Rotational Azimuth Coverage   | Rotational Azimuth Coverage   | Rotational Azimuth Coverage   |
|--------------------|--------------------|------|----------------------------|----------------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|
|                    |                    |      | Check                      | Refinement                 | = 100%                        | ↑ ≥ 75%                       | ≥ 50%                         | ≥ 25%                         | = 100%                        | ≥ 75% ↑                       | ≥ 50% ↑                       | ≥ 25% ↑                       |
| MegaDepth [20]     |                266 | 119k | No                         | Yes                        | 4                             | 15                            | 25                            | 74                            | 27                            | 56                            | 107                           | 230                           |
| MegaDepth-X (Ours) |              1,865 | 440k | Yes                        | Yes †                      | 6                             | 80                            | 223                           | 752                           | 76                            | 490                           | 1123                          | 1816                          |

```
Algorithm 1: One Step of Greedy View Sampling Input: Current node v Neighborhood of v : N v Set of already sampled nodes S Community map M (node → community) Camera positions Pos( · ) Output: Next sampled node u ∗ // Identify communities already covered S comm ←{ M [ s ] | s ∈ S } ; // Compute candidate list with community novelty and distance C ← ∅ ; for each u ∈ N v do unreached ← ( M [ u ] / ∈ S comm ) ; dist ←∥ Pos( u ) -Pos( v ) ∥ 2 ; C ← C ∪ { ( u, unreached , dist ) } ; end // Sort by unreached, then by distance Sort C in descending lexicographic order by ( unreached , dist ) ; // Select the top-ranked candidate ( u ∗ , , ) ←fi rst element of C ; return u ∗ ;
```

Candidates are lexicographically ranked according to these two criteria, and the highest-ranked node is chosen as the next sampled view.

## B.2. Graph Partition

Before sparsity-aware sampling, we partition COLMAP's view graph into N cc subgraphs. Specifically, we randomly select N cc seed nodes and treat each seed as the initial node of one partition. Starting from these seeds, we perform a parallel round-robin breadth-first search(BFS) over the view graph. During each iteration, every subgraph expands from its current frontier to its unassigned neighboring nodes, which are then incorporated into that subgraph. In this way, each node is assigned to the subgraph of the seed that first reaches it, until no further nodes can be expanded.

<!-- image -->

## B.3. Graph Span vs. Search Depth

To understand how greedy search depth D affects the coverage and sparsity of the sampled views, we analyze several statistics on the view-graph. Let G denote the full viewgraph of a scene and S the set of sampled nodes. The first two metrics quantify coverage with respect to the entire graph G , while the last two measure sparsity within the sampled subset S .

k-hop graph coverage. This metric measures how much of the view-graph is reached by the sampled views. Specifically, it computes the fraction of nodes in G that lie within k hops of any sampled node:

<!-- formula-not-decoded -->

where S is the subgraph of greedy sampled nodes and d G ( u, v ) is the shortest path from u to v on the graph G . A higher Cov k indicates broader topological coverage, i.e., the sampled set reaches many graph neighborhoods rather than remaining confined to a small region.

Nearest-sample distance. To evaluate spatial coverage in 3D, we compute the average Euclidean distance from each

0.650

0.0291

0.600

710.573

0.550

0.525

0.500

0.475

4 6

(a) L.hon Covergoe (h—2)

0.65

0.60

0.55

0.50

AvgNear

0.45

0.40

3.50

3.25

3.001

02.75

8 2.50 -

2.25

2.00

1.25

1.20-

1.15

w 1.10 -

1.00

8 1.05 -

0.95

0.90

<!-- image -->

tance)

Figure 7. Coverage and sparsity vs. search depth. Metrics in (a) and (b) evaluate coverage with respect to the full view-graph, while (c) and (d) measure the sparsity of the sampled subset. As the search depth increases, the sampled set reaches a larger portion of the view-graph, as shown by the rise in k -hop (graph-distance) coverage in (a). The average distance from each camera to its nearest sampled view decreases in (b), indicating broader spatial coverage. At the same time, both graph dispersion (average pairwise graph distance) in (c) and Euclidean dispersion (average pairwise 3D distance) in (d) increase with depth, showing that the sampled views become more widely separated across the graph and in 3D space.

camera to its closest sampled camera:

<!-- formula-not-decoded -->

where p u and p v are camera positions. Lower values mean the sampled views are spatially well-distributed and lie near many original cameras.

Graph dispersion and Euclidean dispersion. To understand the sparsity of the sampled views, we calculate the average pairwise distance among sampled views(dispersion) based on graph distances and Euclidean distances:

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

Higher dispersion values indicate that the sampled views are more sparsely distributed in both the graph and Euclidean space.

We compute these metrics for the top 100 scenes with the most registered images, evaluating 12 search depths and averaging over 8 sampling runs per depth. The number of sampled views is 24 for all samples. Results are shown in Fig. 7, indicating that deeper searches yield higher coverage on the full graph (a,b) and produce sparser, more widely distributed sampled subsets (c,d).

## C. Training Details and Additional Results

## C.1. Training Setup

We finetune both π 3 and VGGT using their released pretrained checkpoints. All input images are first padded with white borders to a resolution of 518 × 518 . During training, we apply random crops to these padded images, sampling aspect ratios uniformly from [0 . 75 , 1 . 0] . Wealso apply random color jittering on training images. Each mini-batch contains up to 24 images drawn from MegaDepth-X, with the number of views per batch randomly selected from [2 , 24] . We process at most 96 images on each GPU. We also augment image orientations during training by randomly rotating images 90 ◦ clockwise or counterclockwise with a probability of 0.2.

We use the original loss functions from π 3 [44] and VGGT [39] to finetune the models. To preserve the geometric priors encoded in the pretrained models, we finetune only the Alternating-Attention modules, while keeping the point-cloud and camera decoders frozen. We further include BlendedMVS [48] and TartanAir [43] as additional training data for finetuning. Finetuning is performed for 100 epochs, where each epoch iterates over all scenes in the combined dataset. We use the AdamW optimizer with a peak learning rate of 1 × 10 -5 , scheduled with linear warm-up followed by cosine annealing. All experiments are conducted on 4 NVIDIA A6000 GPUs.

## C.2. Additional Depth-Estimation Results

We provide monocular and video depth results to complement the main paper. Following [40, 44, 49], we evaluate Absolute Relative Error (Abs Rel) and the accuracy at a threshold of δ &lt; 1 . 25 . For monocular depth, we report performance on Sintel [5], Bonn [24], KITTI [13], and NYU-v2 [23]. For video depth, we evaluate on Sintel [5], Bonn [24], and KITTI [13] under both scale and scale&amp;shift alignment settings. Our finetuned models maintain competitive performance across all datasets, demonstrating that the adaptation to in-the-wild imagery does not degrade their depth-estimation ability.

## C.3. Results on Doppelganger Scenes

Doppelganger cases often cause both classical SfM pipelines and pretrained feed-forward models to fail, merging distinct structures into a single incorrect reconstruction. As shown in Fig. 8, our fine-tuned π 3 model correctly distinguishes visu-

—Avanean

- Disp\_g

→ Disp\_E|

· lo aoppergangers.

VGGT-FT

0.197

0.728

0.056

0.973

0.056

0.964

Table 8. Monocular Depth Estimation on Sintel [5], Bonn [24], KITTI [13], and NYU-v2 [23]. We report Absolute Relative Error (Abs Rel,

Is and dense-only

while finetuning

lower is better) and threshold accuracy &amp; &lt; 1.25 (higher is better).

I or sparse) tends Method Sintel Bonn KITTI

NTU-v2

Table 7. Video Depth Estimation on Sintel [5], Bonn [24], and KITTI [13]. We report Absolute Relative Error (Abs Rel, lower is better) and the prediction accuracy at a threshold of δ &lt; 1 . 25 (higher is better).

il scenes

g-tail scenes, we

rvations from ex-

nd jointly register

ve and qualitative

COLMAP

min Die 10 Ou

Palace of Rozumovskyi (Baturn)

0.054

0.052

A3-DENSE

0.059

0.972

0.956

| Method   | Align         | Sintel    | Sintel       | Bonn      | Bonn         | KITTI     | KITTI        |
|----------|---------------|-----------|--------------|-----------|--------------|-----------|--------------|
|          |               | Abs Rel ↓ | δ < 1 . 25 ↑ | Abs Rel ↓ | δ < 1 . 25 ↑ | Abs Rel ↓ | δ < 1 . 25 ↑ |
| π 3      | scale         | 0.228     | 0.671        | 0.051     | 0.975        | 0.038     | 0.986        |
| π 3 -FT  | scale         | 0.213     | 0.713        | 0.047     | 0.978        | 0.040     | 0.985        |
| VGGT     | scale         | 0.294     | 0.649        | 0.055     | 0.971        | 0.072     | 0.965        |
| VGGT-FT  |               | 0.242     | 0.707        | 0.061     | 0.969        | 0.065     | 0.966        |
| π 3      | scale & shift | 0.207     | 0.735        | 0.045     | 0.976        | 0.036     | 0.986        |
| π 3 -FT  |               | 0.188     | 0.739        | 0.043     | 0.978        | 0.038     | 0.985        |
| VGGT     |               | 0.226     | 0.683        | 0.049     | 0.974        | 0.059     | 0.961        |
| VGGT-FT  |               | 0.197     | 0.728        | 0.056     | 0.973        | 0.056     | 0.964        |

Doppelganger pair N3-FT

N3-SPARSE

Table 8. Monocular Depth Estimation on Sintel [5], Bonn [24], KITTI [13], and NYU-v2 [23]. We report Absolute Relative Error (Abs Rel, lower is better) and threshold accuracy δ &lt; 1 . 25 (higher is better). Pretrained 𝜋 3 Ours COLMAP

doppelganger scenes. Disambiguation behavior holds across fine-tuned vari-

Figure 8. Disambiguation of doppelganger scenes. Each example shows a pair of visually similar structures that cause classical SfM (COLMAP) and pretrained π 3 to collapse into incorrect or merged reconstructions. In contrast, our finetuned model correctly distinguishes the symmetric or repetitive sides of the same building, reconstructing consistent geometry for each viewpoint. Reference views from Google Earth are provided for comparison, confirming that our model resolves these ambiguities and recovers accurate global structure under challenging visual similarity.

| Method   | Sintel    | Sintel       | Bonn      | Bonn         | KITTI     | KITTI        | NTU-v2    | NTU-v2       |
|----------|-----------|--------------|-----------|--------------|-----------|--------------|-----------|--------------|
|          | Abs Rel ↓ | δ < 1 . 25 ↑ | Abs Rel ↓ | δ < 1 . 25 ↑ | Abs Rel ↓ | δ < 1 . 25 ↑ | Abs Rel ↓ | δ < 1 . 25 ↑ |
| π 3      | 0.277     | 0.621        | 0.052     | 0.971        | 0.059     | 0.972        | 0.054     | 0.956        |
| π 3 -FT  | 0.284     | 0.629        | 0.049     | 0.977        | 0.056     | 0.972        | 0.052     | 0.958        |
| VGGT     | 0.331     | 0.600        | 0.051     | 0.974        | 0.089     | 0.939        | 0.055     | 0.953        |
| VGGT-FT  | 0.311     | 0.628        | 0.056     | 0.974        | 0.092     | 0.941        | 0.053     | 0.955        |

<!-- image -->

ally similar but distinct structures within each landmark and recovers geometry consistent with reference aerial imagery, indicating improved reconstruction of global scene layout.

To evaluate the effectiveness of different sampling strategies on doppelganger scenes, we evaluate the pretrained π 3 and finetuned π 3 on doppelganger scenes and show results in fig.9. Results indicate that pretrained models and dense-only fine-tuning are less robust to ambiguity, while finetuning with sparsity-aware sampling (e.g., mixed or sparse) tends to improve disambiguation, suggesting sparsity-aware sampling helps.

Figure 9. Comparison of ablated models on doppelganger scenes We show predictions from the pre-trained model and ablated models on two doppelganger scenes. Disambiguation behavior holds across fine-tuned variants with sparsity-aware sampling, while the pre-trained model and model finetuned with densely sampled views are less robust to doppelgangers.

<!-- image -->

## C.4. Quantitative results on Long-tail scenes

To enable quantitative evaluation on long-tail scenes, we augment MegaScenes with additional observations from external cultural heritage datasets [8, 9, 26] and jointly register all images using COLMAP. The quantitative and qualitative results of this long-tail evaluation are shown in Fig. 10. Our model consistently reduces the mean relative rotation and Referen translation errors across all scenes, while also producing more complete point clouds.

Figure 10. Quantitative results on Long-tail scenes. Our model performs better on scenes with strong ambiguities (first row) and on scenes with minimal overlap across different scene components (second row). For a more densely photographed scene that still exhibits large viewpoint variation (third row), our model not only reduces pose error but also reconstructs a more complete point cloud.

<!-- image -->

## C.5. Limitations

Long-tail scenes often contain fragmented viewpoints, where different subsets of images capture disjoint parts of the scene (e.g., indoor and outdoor areas) without overlapping views to connect them. When such mixed collections are fed into the models at once, both pretrained and finetuned π 3 may blend these unrelated regions into a single 3D structure, as illustrated in Fig.11. While our finetuned model handles these mixtures more robustly than the pretrained baseline, enabling the model to reason robustly about disconnected components and produce reasonable overall layouts still remains a challenge.

'he bridge has two similar dragon statues, one at each end.

MAP incorrectly treats them as the same statue and registers

(a) Dragon Bridge (Ljubljana)

(b) Eletsky Monastery

(c) Schloss Linderhof

nd (e) illustrate additional doppelganger cases, in which differ-

together, whereas our method correctly separates them. (b),

ides or parts of a landmark are mistakenly merged. (c) In this

exture scene, our pipeline also succeeds in registering more

es.

COLMAP

Ours

COLMAP

North Input Images

(d) Royal Albert Hall Image COLMAP-MVS

and struggles to correctly estim

Our strategy mitigates these iss

from monocular depth predictior

depth maps over complete ones.

recover the depth of the foregrou

depth values in that region. If we

to match the MVS result, then a

Figure 11. Limitations. This example contains images from two disjoint parts of the scene: indoor photos with warm lighting (producing a yellowish point cloud) and outdoor photos (producing a white point cloud). Pretrained π 3 struggles to handle such mixed inputs and produces inconsistent geometry. Our finetuned model is more robust in this setting, but both models still fuse the indoor and outdoor structures into a single reconstruction without separating them.

<!-- image -->

(a) Dragon Bridge (Ljubljana)

(b) Eletsky Monastery

(c) Schloss Linderhof

Figure 13. Comparison of COLMAP MVS and our filtered

117 views

a) The bridge has two similar dragon statues, one at each end.

Ours

d), and (e) illustrate additional doppelganger cases, in which differ-

anger

131 views dense depth results. COLMAP MVS suffers from depth bleeding and struggles to correctly estimate the depth of transient objects.

Our strategy mitigates these issues by leveraging ordering priors

from monocular depth predictions. Note that we prioritize accurate

depth maps over complete ones. In the last row, COLMAP fails to

recover the depth of the foreground statue, and we opt to remove the

denth valies in that recion

(e) Sant'Andrea (Vercelli)

inside

North

(d) Royal Albert Hall

South outside

Red Square Hamburger Rathaus Cathedrale Notre-Dame de Chartres COLMAP Figure 12. Comparison of COLMAP and our reconstruction pipeline. We replace COLMAP with MASt3R-SfM [11] combined with the doppelganger++ classifier [45] to obtain sparse reconstructions, allowing effective disambiguation of doppelganger scenes. (a) The bridge has two similar dragon statues, one at each end. COLMAP incorrectly treats them as the same statue and registers them together, whereas our method correctly separates them. (b), (d), and (e) illustrate additional doppelganger cases, in which different sides or parts of a landmark are mistakenly merged. (c) In this low-texture scene, our pipeline also succeeds in registering more images.

Ours

LMAP

Dresden Castle York Minster Figure 13. Comparison of COLMAP MVS and our filtered dense depth results. COLMAP MVS suffers from depth bleeding and struggles to correctly estimate the depth of transient objects. Our strategy mitigates these issues by leveraging ordering priors from monocular depth predictions. Note that we prioritize accurate depth maps over complete ones. In the last row, COLMAP fails to recover the depth of the foreground statue, and we opt to remove the depth values in that region. If we were to warp the monocular depth to match the MVS result, then any inaccuracy in the relative depth between the statue and the background building could produce erroneous and inconsistent cross-view depth estimates.

<!-- image -->

Bath Abbey St. Vitus Cathedral

COLMAP

COLMAP

Ours Cathédrale Notre-Dame de Rouen Vienna State Opera Brandenburg Gate Ours