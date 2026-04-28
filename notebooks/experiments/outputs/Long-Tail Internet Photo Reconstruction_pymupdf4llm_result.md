## **Long-Tail Internet Photo Reconstruction** 

Yuan Li[1] 

Yuanbo Xiangli[1] _[†]_ Hadar Averbuch-Elor[1] Noah Snavely[1] Ruojin Cai[2] _[†]_ 1Cornell University 2Kempner Institute, Harvard University 

**==> picture [496 x 210] intentionally omitted <==**

**----- Start of picture text -----**<br>
Long Tail<br>Registered images Total images<br>Scene (sorted by #images)<br>Pretrained  𝜋 [3] Pretrained  𝜋 [3]<br>Duomo (Cagliari) - Crypt Ours Ours Calvaire de Plougonven<br>#images per Scene<br>**----- End of picture text -----**<br>


Figure 1. **Long-tail Internet photo reconstruction.** Internet photo collections follow a long-tailed distribution. In the top plot, the _x_ -axis represents scene index (sorted by image count) and the _y_ -axis shows images per scene (scenes are drawn from MegaScenes [36], a dataset of Internet photo collections). The light blue curve plots the total number of Internet photos per scene, while the steel blue curve shows the size of the subset of photos that were successfully registered using SfM. The _head_ of this distribution of photo collections represents well-photographed scenes; here, there are 6,985 scenes with _>_ 50 registered images. However, most photo collections are in the _long tail_ of this distribution; here, 418,056 scenes with fewer than 50 registered photos. State-of-the-art methods often fail on scenes in this tail. In the lower half of the figure, we show two examples from the long tail, along with representative input images and the corresponding reconstructions. On Calvaire de Plougonven, COLMAP doesn’t register any image; on both Duomo (Cagliari)-Crypt and Calvaire de Plougonven, recent feed-forward reconstruction models like _π_[3] [44] produce poor results. We propose MegaDepth-X dataset and a strategy for mimicking long-tail camera distributions, on which fine-tuned models like _π_[3] exhibit better reconstruction robustness. 

## **Abstract** 

_Internet photo collections exhibit an extremely long-tailed distribution: a few famous landmarks are densely photographed and easily reconstructed in 3D, while most realworld sites are represented with sparse, noisy, uneven imagery beyond the capabilities of both classical and learned 3D methods. We believe that tackling this long-tail regime represents one of the next frontiers for 3D foundation models. Although reliable ground-truth 3D supervision from sparse scenes is challenging to acquire, we observe that it can be effectively simulated by sampling sparse subsets from well-reconstructed Internet landmarks. To this end, we introduce MegaDepth-X, a large dataset of 3D reconstruc-_ 

- †Corresponding authors. 

_tions with clean, dense depth, together with a strategy for sampling sets of training images that mimic camera distributions in long-tail scenes. Finetuning 3D foundation models with these components yields robust reconstructions under extreme sparsity, and also enables more reliable reconstruction in symmetric and repetitive scenes, while preserving generalization to standard, dense 3D benchmark datasets. The dataset, finetuned models, and code are available at: https://megadepth-x.github.io/._ 

## **1. Introduction** 

Internet photo collections of real-world landmarks follow a long-tailed distribution. A small fraction of famous sites, such as the Colosseum or Notre Dame, are photographed 

from every conceivable angle and can be accurately reconstructed by standard Structure-from-Motion (SfM) pipelines. Yet the overwhelming majority of landmarks across the world are represented on the Internet with just a handful of sparse, noisy images (Fig. 1). We refer to this large body of scenes as the _long-tail_ of online photo collections. Such scenes are the norm rather than the exception in real-world Internet imagery. 

Reconstructing long-tail scenes is challenging. Classic methods, such as COLMAP [28], often fail because feature correspondence is hard to find across sparse, nonoverlapping, or wide-baseline views. Modern learned feedforward models, like DUSt3R [42] and VGGT [39], can learn powerful priors from millions of images that might help reconstruct long-tail collections. In practice, however, these models are primarily trained on controlled captures with clean, dense, and evenly sampled data. When applied to long-tail Internet scenes featuring sparse, diverse, and unevenly distributed imagery, we find that these models often fail to recover consistent geometry. 

We believe that one of the next frontiers for 3D foundation models lies in tackling this long-tail regime of Internet photos. Better data is almost certainly key to this problem, but we cannot easily construct reliable 3D supervision from long-tail collections themselves, as most contain too few overlapping views for robust reconstruction. Instead, we propose to _simulate_ such long-tailed sets by appropriate sampling of sparse images from the large, well-reconstructed Internet landmarks at the head of the distribution, inheriting ground truth from the full reconstruction. 

This strategy requires drawing from large amounts of high-quality landmark reconstructions from Internet photos. Existing datasets fall short of this need: MegaDepth [20] is clean but small, while MegaScenes [36] is large but noisy and lacks depth maps. We therefore introduce _MegaDepth-X_ (dubbed MD-X), a next-generation extension of MegaDepth in both scale (7 _×_ larger) and quality: a large-scale, clean, and dense-depth-enhanced dataset built from Internet photo reconstructions with consistent depth refinement and extensive manual verification against reliable references (e.g., Google Maps and satellite imagery). Equipped with MD-X, we propose a novel _sparsity-aware_ sampling strategy that mimics the camera distributions of long-tail scenes, encouraging training batches to span wide baselines and partial overlap rather than clustered dense views. 

Through extensive experiments, we show that models fine-tuned with MD-X and our sparsity-aware data sampling scheme are significantly more robust on long-tail Internet photo collections, including challenging doppelganger scenes with ambiguous or symmetric content, such as the Calvaire de Plougonven example in Fig. 1, where classical SfM and pretrained foundation models often fail. In summary, our contributions are: 

- **Defining the 3D long-tail regime** : we formalize and characterize the long-tail distribution of Internet photo collections, highlighting this setting’s distinct challenges. 

- **MegaDepth-X** , dubbed MD-X, a large-scale, clean, and depth-augmented dataset for finetuning 3D foundation models on real-world Internet scenes. 

- **Sparsity-aware sampling** strategies that simulate the distribution of long-tail Internet collections to improve generalization of 3D prediction models on real-world data. 

## **2. Related Work** 

**Feed-forward 3D reconstruction.** Reconstructing 3D scene geometry from 2D images is a cornerstone of computer vision. Traditional structure from motion (SfM) [29] and multiview stereo (MVS) [30] methods were crowning achievements of the classic era of 3D vision, and were scaled to large Internet photo collections [1, 12, 34]. Recently, the new paradigm of feed-forward 3D reconstruction has emerged, which involves regressing 3D attributes directly from images in a single pass. Pioneering work in this area, such as DUSt3R, showed success at predicting pixel-aligned point maps from image pairs [42]. MASt3R extended this approach but still relied on pairwise processing [19]. Subsequent efforts focused on scaling these models to arbitrary numbers of views. VGGT [39], along with concurrent models like Fast3R [47] and FLARE [50], introduced large transformer architectures that can process hundreds of views simultaneously. By leveraging large-scale, diverse datasets and a multi-task learning objective, VGGT predicts a full suite of 3D attributes, including camera parameters, depth maps, and point maps. To eliminate reference-frame bias, _π_[3] [44] recently proposed a permutation-equivariant architecture that predicts affine-invariant camera poses and scaleinvariant local point maps. ZipMap [16] and Scal3R [46] introduced test-time training approaches to process large image collections. These methods work well on denselycaptured and well-conditioned scenes. However, we find that their performance on more sparse and noisy Internet photos remains suboptimal, particularly for long-tail scenes. 

**Long-tail challenges in 3D vision.** Long-tailed problems are pervasive in computer vision. They occur when data for common scenarios (the head) are abundant, but examples of rare yet collectively frequent cases (the tail) are scarce. For instance, many object recognition problems involve a few dominant categories but many rarely seen ones, and in autonomous driving, routine driving scenes are plentiful while safety-critical events are hard to capture. 

Recently, MegaScenes [36] introduced a large-scale scene-level dataset built from Internet photo collections, where long-tail effects are particularly pronounced. Many scenes in the dataset are either unreconstructed or incorrectly reconstructed. These failures stem from a combina- 

tion of view sparsity, noisy imagery, and doppelganger issues [7]. Recent work has sought to address such challenges by developing stronger local features [10, 37] and matchers [15, 17, 21, 27], and by learning wide-baseline pose relationships from large-scale 3D datasets [3, 6]. The doppelganger problem was further addressed by Cai et al. [7, 45], who trained classifiers to prune false matches during the structure-from-motion phase of reconstruction. 

While these advances have led to enhanced robustness, they do not yet work reliably at scale. Ideally, we’d mine ground truth 3D training data for long tail scenes and learn to reconstruct them, but that involves a chicken-and-egg problem, because the common practice of using available reconstructors (e.g. COLMAP [28, 31], VGGT [39]) to derive pseudo-ground-truth camera poses and point maps from natural data doesn’t work. Instead, similar in spirit to approaches used in autonomous driving that augment training data by simulating rare events, our key idea is to take large, well-conditioned image collections and subsample them to simulate long-tailed photo collections, and use these to better balance training scene distributions for regression models in order to generalize to long-tailed scenes. 

## **3. The MegaDepth-X Dataset** 

Learning in the long-tail regime requires high-quality 3D supervision derived from Internet photo collections. This involves two key challenges. First, reconstructions of Internet photo collections can be unreliable due to noise, dynamic content, and ambiguities [7]. Second, most long-tail scenes lack any usable reconstructions, as classical SfM pipelines like COLMAP [29] often fail on sparse or widely varying image sets. To address these issues, we construct MD-X, a large-scale, clean, and depth-refined dataset that provides reliable 3D supervision, built from well-reconstructed scenes in MegaScenes [36]. 

## **3.1. Filtering and Disambiguation** 

Our first step in constructing MD-X is to identify candidate Internet landmarks from which reliable supervision can be derived. We take as our starting pool the subset of MegaScenes with more than 100 registered images, which typically yields stable reconstructions. However, even these “well-reconstructed” scenes exhibit two common failure modes: (1) Many scenes contain dynamic events or crowded activities, causing feature matches to lock onto moving objects rather than static structures, leading to unreliable reconstructions. (2) The Doppelganger problem [7, 45], where visually similar but geographically distant images are mistakenly registered together. Both issues produce incorrect camera poses and fragmented, inconsistent point clouds as shown in Fig. 2. 

To mitigate these issues, we first inspect the dataset and exclude scenes dominated by crowds or moving objects. 

**==> picture [237 x 184] intentionally omitted <==**

**----- Start of picture text -----**<br>
Old Town Hall (Prague)<br>(a) (b)<br>(c) (d)<br>**----- End of picture text -----**<br>


Figure 2. **Unreliable reconstructions in MegaScenes.** Reconstructions are unreliable when feature matches are incorrectly established on salient, non-static objects (e.g., (a) humans, (b) statues, (c) airplanes) instead of the static scene structure. This results in fragmented and geometrically inconsistent point clouds. Example (d) illustrates a doppelganger failure, where images from opposite sides of the building are incorrectly registered together. 

Next, we address the doppelganger problem by replacing the default COLMAP SfM reconstruction with MASt3RSfM [19], combined with Doppelganger classification [45]. Specifically, MASt3R-SfM constructs the scene graph using feature matches derived from MASt3R descriptors, after which the Doppelganger classifier identifies and prunes suspicious edges that may result from doppelganger-induced false correspondences. Finally, we manually verify the reconstructed scenes against external references such as Google Maps and satellite imagery, discarding any scenes that do not align with the corresponding bird’s-eye view. 

## **3.2. Dense Depth Refinement** 

After obtaining reliable sparse reconstructions, we seek to generate dense depth maps for supervision. We start by running a standard multi-view stereo (MVS) [31] pipeline. We observe, as in prior work [20], that the resulting geometric depth maps from in-the-wild collections often exhibit artifacts, including depth-bleeding effects (background depths leak into foreground regions) and inconsistent and noisy depths in areas with transient objects (e.g., people, cars). 

To address these initial issues, we apply the full depth refinement strategy from MegaDepth [20], including a modified MVS procedure that conservatively retains the minimum depth value during propagation, stability filtering to remove flickering pixels, and semantic filtering to exclude transient objects. However, even after this pipeline, we still observe artifacts in the processed geometric depth maps: (1) the MegaDepth-modified MVS still leads to depth-bleeding artifacts, and (2) semantic filtering is not ideal as it relies on a manually designated list of object categories. Examples of such issues are shown in Fig. 3. 

**==> picture [237 x 109] intentionally omitted <==**

**----- Start of picture text -----**<br>
Image MegaDepth Refinement Our Refinement<br>Transient Object<br>Depth Bleeding<br>**----- End of picture text -----**<br>


Figure 3. **Depth refinement.** MVS depth maps often suffer from artifacts like noise from transient objects (top row) and depth bleeding (bottom row). As shown in the middle column, the MegaDepth refinement pipeline (modified MVS, stability filtering, and semantic filtering) fails to fully remedy these issues. Our method (right column) introduces an additional monocular depth-guided filtering step, which effectively removes transient objects and significantly mitigates depth-bleeding artifacts. 

collections are large, redundant, and visually well-connected. In this regime, models can rely on strong covisibility and abundant local correspondences. However, most real Internet photo collections lie in the long tail, where views are sparse, unevenly distributed, and only weakly connected. A more complete 3D prior should therefore be robust not only to diverse scene content, but also to this underrepresented observation regime. Rather than seeking unreliable supervision from true long-tail scenes, we start from well-reconstructed scenes in MD-X and sample subsets whose covisibility structure matches that of real long-tail collections. In this way, we expose the model to the missing part of the training distribution while inheriting trustworthy 3D supervision from the full reconstruction. 

## **4.1. Defining Properties of Long-Tail Scenes** 

Therefore, to augment MegaDepth’s depth refinement procedure, we propose a monocular depth-guided filtering step. We use depth predictions from MoGe2 [41] as ordinal depth priors, and remove pixels in the processed geometric depth maps that are inconsistent with these priors. Specifically, we first align the processed geometric depths _D_ geom to the monocular predictions _D_ mono by matching their median values over valid pixels: _D_ geom _[′]_[(] _[p]_[)][=] _s · D_ geom( _p_ ) _,_ where _s_ =[med] med _[{] {[D] D_[mono] geom([(] _p[p]_ )[)] _|[|] p[p] ∈[∈] P[P] }[}]_[.][After scale align-] ment, we compute the normalized depth discrepancy between the two maps: ∆( _p_ ) = _|D_ g _[′]_ eom _D_[(] _[p]_ geom _[′]_[)] _[−][D]_ ( _p_[mono] )[(] _[p]_[)] _[|]_ , and discard pixels whose discrepancies exceed a predefined threshold _τ_ depth. Moreover, to leverage _D_ mono for edge-aware filtering, we compute the discrepancies between the gradients of the two maps: ∆( _p_ grad) = _|[|][∇] D[D]_ mono[mono] _[|] − |∇DD_ geom _[′]_ g _[′]_ eom _[|][|]_[ and discard pix-] els whose discrepancies exceed a predefined threshold _τ_ grad. This approach effectively filters both bleeding artifacts and noisy transient objects without relying on manual category lists, as depicted in Fig. 3. 

## **3.3. Dataset Statistics** 

In summary, we identify 2,474 candidate scenes from MegaScenes with more than 100 registered images. Of these, 609 scenes are filtered out due to dynamic content, reconstruction errors, or geometric inconsistencies. Our final MD-X dataset comprises 1,865 reconstructions totaling 440k images. We reserve 127 scenes for testing, providing a novel set for evaluating both pretrained and fine-tuned methods. A comparison table with MegaDepth is provided in the supplementary. 

## **4. Simulating Long-Tail Scenes** 

With MD-X providing reliable 3D supervision, the remaining challenge is a complementary supervision coverage problem: existing 3D foundation models are trained predominantly on the head of the Internet-photo distribution, where image 

Common issues like transient occluders and motion blur affect Internet photos broadly, but they are not the primary bottleneck for long-tail scenes. The more fundamental challenge lies in their viewpoint distribution. In these scenes, sparse camera placements lead to limited mutual overlap between images. This results in fragmented, weakly connected clusters rather than a cohesive set, which poses a major hurdle for reliable 3D reconstruction. Because accurate camera poses are often unavailable for such scenes, we characterize this regime using statistics of the SfM view graph rather than absolute camera geometry. Our analysis reveals two consistent patterns: (1) _sparser connectivity_ : scenes with low registration rates (e.g., only 20% of images registered) contain a substantially larger fraction of low-degree nodes, with 8% of cameras having degree two or less, compared with only 3% in well-reconstructed head scenes. This indicates that cameras in long-tail scenes are poorly connected, forming fragmented clusters with limited covisibility. (2) _weaker connections_ : even among connected image pairs, the average number of geometrically verified feature matches is significantly lower in long-tail scenes than in head scenes (294.8 vs. 395.3), indicating reduced overlap and weaker geometric consistency.[1] Together, these observations show that the long tail is not simply a regime of fewer images, but one of sparse and weakly connected observation graphs. 

Based on these findings, our sampling process should satisfy three requirements: 

- **Viewpoint Diversity:** The sampled views should cover a wide range of viewing directions, ensuring that emulated scenes span diverse visual perspectives. 

- **Sparsity:** The selected views should be far enough apart to mimic the wide baselines typical of long-tail scenes, _e.g_ . loosely connected views or views from disconnected scene components, encouraging the model to learn robust geo- 

- 1To avoid statistics being dominated by severely noisy scenes, we com- 

- pute these measurements only on long-tail subsets containing at least five registered images. 

metric priors rather than relying on dense feature matches. 

- **Local Reconstructability:** Despite the sparsity, views within each sampled scene component should retain enough covisibility to remain locally reconstructable, since zero-overlap samples within a scene component can lead to unstable training signals and difficult optimization. 

## **4.2. Sparsity-Aware Sampling Strategy** 

We therefore formulate the sampling task as sampling _N_ views that form at most _Ncc_ connected components, in order to emulate a long-tail scene with multiple weakly connected or disconnected scene components. Specifically, components are allowed to be disconnected from one another, but within each sampled component we still require sufficient internal covisibility for local reconstructability. We find that na¨ıve random or uniform subsampling often fails to satisfy this balance, producing either zero-overlap sets within scene components or clusters biased toward dense regions. We instead propose a structured sampling process. We first partition views into strongly connected communities and then select a minimal yet diverse subset that ensures both community coverage and global connectivity. This process is illustrated in Fig. 4. 

**Graph Communities.** To promote viewpoint diversity in our sampling, we first identify the dominant “viewing areas” within each scene. We represent the SfM structure as a view graph _G_ = ( _V, E_ ), where each node _vi ∈ V_ corresponds to a camera view and each edge ( _vi, vj_ ) _∈ E_ is weighted by the number of feature matches _wij_ . We prune edges with _wij <_ 50 to remove minor overlaps, resulting in a filtered graph _G[′]_ = ( _V, E[′]_ ) that preserves only meaningful covisibility relationships. To reveal clusters of cameras with dense internal connectivity, we perform community detection (e.g., Louvain community detection [4]) on the view graph. This yields viewpoint groups _Ck_ that efficiently capture distinct visual regions and the dominant perspectives of the scene. We then randomly partition the graph into _Ncc_ connected components that span different communities and do the following steps _within each graph partition_ . The partition algorithm is provided in the supplementary material. 

**Minimal Connectivity Subgraph.** To preserve overall scene connectivity while maintaining sparsity and view diversity within limited nodes, we construct a minimal structure linking all identified communities without reintroducing dense redundancy within each partition. We then compute an approximate Steiner tree to link all of these nodes [18, 22].[2] In particular, for each training batch for a given training scene, we first randomly select one representative view _vk ∈ Ck_ from each community _Ck_ to form the terminal set _T_ = _{vk}_ . An approximate Steiner tree 

> 2A Steiner tree aims to span a specified set of _terminal_ nodes while introducing only the minimal set of intermediate nodes required for connectivity. 

**==> picture [237 x 176] intentionally omitted <==**

**----- Start of picture text -----**<br>
Graph Communities Skeleton Subgraph View Sampling<br>Run  Louvain algorithm Build  Steiner Tree  to connect  Perform  Greedy Search  on the<br>(based on #matches)  terminal nodes  with minimal edges subgraph (e.g. after 4 iterations)<br>Covisibility Edge Camera/View Terminal Node Unselected View Sampled View<br>Reconstruction Top-down View Example Results (N=24)<br>Notre-Dame de Paris D= 24<br>Community2<br>D= 12<br>Community3<br>Community1<br>**----- End of picture text -----**<br>


Figure 4. **Sparsity-aware sampling strategy. Top:** Our method follows a multi-stage process: (1) Apply the _Louvain algorithm_ to the view graph to identify distinct viewpoint communities. (2) From each community, randomly select a terminal view and construct an approximate _Steiner Tree_ to form a minimal, connected subgraph spanning these communities. (3) Perform a _Greedy Search_ on this subgraph to select a sparse and diverse set of views. This procedure aims to cover as many communities as possible while ensuring a wide spatial distribution of cameras within each community. **Bottom:** A _search depth_ parameter controls the final view coverage. In this example, we sample _N_ = 24 views from the scene with _Ncc_ = 1. With search depth _D_ = 24, all views are selected via greedy search, producing a more evenly spread distribution. With _D_ = 12, 12 views come from greedy search and the remaining 12 are sampled locally from the neighborhoods of selected nodes, resulting in a more concentrated distribution. 

algorithm then constructs a minimal connected subgraph _G_ sub = ( _V_ sub _, E_ sub) _, T ⊆ V_ sub _⊆ V_ , that spans all terminal nodes using only the necessary intermediate nodes. This yields a compact subgraph connecting all communities using the fewest necessary nodes and edges, preserving global consistency while retaining sparsity. Since _G_ sub can have an arbitrary number of nodes, we need to perform additional sampling to get desired number of views for the training and testing batches. 

**Greedy View Sampling.** Inspired by skeletal sets [35], we perform greedy view sampling on the subgraph _G_ sub to select a diverse subset of views for long-tail emulation. The objective is to iteratively expand the sampled set toward broad spatial coverage while maintaining sufficient covisibility among selected view pairs. 

At each iteration, the algorithm aims to select the next view based on two criteria: (1) _Community novelty_ : preferring cameras that belong to previously unseen communities, thereby introducing new viewing directions and reducing redundancy; and (2) _Spatial distance_ : encouraging selection of cameras farther from the current viewpoint to promote wider baseline coverage. Specifically, the algorithm operates on a current node _v_ and its connected neighborhood _Nv_ . Let _S_ denote the set of already sampled nodes and _M_ be the community map. We first determine which communities have already been reached in _S_ , form- 

ing the set _S_ comm = _{M_ [ _s_ ] _| s ∈ S}_ . For each neighbor _u ∈ Nv_ , we then evaluate its community novelty by checking whether _M_ [ _u_ ] _∈/ S_ comm, and compute its spatial distance as _∥_ Pos( _u_ ) _−_ Pos( _v_ ) _∥_ 2, where Pos( _·_ ) is camera position. Details for this algorithm are provided in the supplemental material. All candidate neighbors are ranked lexicographically by these two attributes, and the top-ranked neighbor _u[∗]_ is selected as the next sampled node. This procedure is repeated for _D_ iterations (i.e., the search depth). 

**Implementation.** In practice, we compute a fixed set of communities _C_ = _{Ck}_ for each scene. To form a training batch of _N_ images for a scene, we first randomly divide the _N_ samples across all _Ncc_ partitions. In each partition, greedy view sampling stops once either a predefined search-depth limit _D_ is reached or the target number of views assigned to that partition has been sampled. Here, _D_ controls how far the search expands within a partition, hence the sparsity of the resulting set. If this process still produces fewer than _N_ nodes in total, we fill the remaining slots by randomly sampling nodes from the local neighborhoods of the previously sampled nodes. Fig. 4 illustrates an example in which _N_ = 24 and _Ncc_ = 1, and shows the different sparsities of the sampled set obtained under different values of _D_ . Before training, we run the proposed sampling algorithm offline to generate mini-batches of 24 nodes, avoiding costly graph loading during training. We then perform depth-first search from random seed nodes to subsample 2 to 24 images for training batches. 

## **5. Experiments** 

We evaluate how our approach improves 3D reconstruction in the long-tail regime of Internet photo collections. First, we show quantitative results on the proposed MD-X benchmark, demonstrating qualitative improvements on real-world longtail and doppelganger scenes. We then analyze the effect of the proposed dataset and sampling strategy, and finally verify that our fine-tuned models preserve strong performance on standard, curated benchmarks. Further implementation details and additional results are in the supplementary material. 

## **5.1. Experimental Setup** 

**Backbones and variants.** We finetune two feed-forward 3D foundation models, _π_[3] [44] and VGGT [39], on MD-X using our proposed sampling strategy. We adopt the loss functions from _π_[3] [44] and VGGT [39]. To preserve pretrained geometric fidelity, we finetune only the AlternatingAttention modules and keep the point cloud and camera decoders frozen. More training details are in the supplementary. The resulting models are denoted as _π_[3] -FT and VGGT-FT. 

To study how our proposed view sampling strategy affects performance, we finetune _π_[3] on clean Internet data using 

Table 1. **Quantitative results on MegaDepth-X** for camera pose and point map estimation across two difficulty levels. Our finetuned models ( _π_[3] -FT and VGGT-FT) trained with the proposed dataset and sampling strategy consistently outperform pretrained baselines, especially on harder, sparser scenes. 

||**Method**|**Camera Pose Estimation**<br>RRA@5_↑_RTA@5_↑_AUC@5_↑_MRE_↓_MTE_↓_|**Poin**|**t Map Estimat**|**ion**<br>NC_↑_<br>Mean<br>Med.|
|---|---|---|---|---|---|
||||Acc_↓_<br>Mean<br>Med.|Comp_↓_<br>Mean<br>Med.||
|_easy_|_π_3<br>_π_3-FT|88.97<br>68.79<br>45.84<br>4.12<br>7.82<br>**95.64**<br>**76.85**<br>**55.58**<br>**1.64**<br>**5.50**|0.055 0.030 <br>**0.035 0.020 **|0.039 0.019 <br> **0.024 0.012 **|0.712 0.822<br> **0.724 0.837**<br> 0.695 0.798<br> **0.719 0.833**|
||VGGT<br>VGGT-FT|84.17<br>58.47<br>35.32<br>4.55<br>9.93<br>**92.41**<br>**71.12**<br>**48.78**<br>**2.70**<br>**7.02**|0.093 0.047 <br>**0.050 0.027 **|0.055 0.026 <br> **0.033 0.014 **||
|_hard_|_π_3<br>_π_3-FT|75.31<br>59.16<br>36.93<br>12.21<br>10.82<br>**86.40**<br>**71.00**<br>**47.93**<br>**5.72**<br>**7.27**|0.101 0.065 <br>**0.068 0.041 **|0.133 0.090 <br> **0.066 0.041 **|0.689 0.786<br> **0.713 0.818**<br> 0.675 0.764<br> **0.709 0.814**|
||VGGT<br>VGGT-FT|70.98<br>52.98<br>29.10<br>13.20<br>13.34<br>**81.07**<br>**65.59**<br>**41.49**<br>**7.22**<br>**9.05**|0.149 0.092 <br>**0.089 0.053 **|0.151 0.104 <br> **0.084 0.055 **||



four sampling schemes: 

- DENSE: training batches with densely overlapping views where _D_ = 5 and _Ncc_ = 1, 

- SPARSE: long-tail–like sampling emphasizing wide baselines where _D_ = 24 and _Ncc_ = 4, 

- MIXED: a combination of dense and sparse batches for balanced learning with _D ∈_ [5 _,_ 24] and _Ncc ∈_ [1 _,_ 4], 

• RANDOM: random view sampling. Unless otherwise noted, FT (e.g., _π_[3] -FT) refers to the model finetuned on the cleaned dataset using the MIXED sampling strategy above. We additionally train a DIRTY variant on Internet data (using the same Mixed scheme) without the filtering strategy in Sec. 3.1, while keeping the same depth refinement pipeline in Sec. 3.2, to assess robustness to label noise and data contamination. 

**Evaluation Metrics.** For camera pose estimation, we follow prior work [39, 44] and report Relative Rotation Accuracy (RRA), Relative Translation Accuracy (RTA), and their combined Area Under Curve (AUC). We also report mean rotation and translation errors (MRE and MTE, in degrees). For point map evaluation, we follow prior work [2, 38, 40, 42, 44] and report Accuracy (Acc), Completeness (Comp), and Normal Consistency (NC), each computed as the mean and median across test scenes. 

## **5.2. Internet Photo Evaluation** 

We first evaluate models on the proposed MD-X benchmark, which contains Internet photo collections of varying sparsity and difficulty. For each test scene, we sample 24 images from the reconstructed scene graph using our sampling algorithm, and categorize them into _easy_ ( _D_ = 5, _Ncc_ = 1) and _hard_ ( _D_ = 24, _Ncc_ = 4) subsets according to the greedy search depth used for test data sampling. 

**Quantitative Results.** Tab. 1 reports quantitative results for camera pose and point map estimation across three difficulty levels on MD-X. Finetuning markedly improves both _π_[3] and VGGT over their pretrained baselines, with larger gains observed in harder, sparser scenes. These improvements hold across metrics indicate that the fine-tuned models better capture global structure and maintain consistent 3D geometry 

|Table 2. **Ablation study on MegaDepth-X.**Finetuning on the cleaned<br>dataset with MIXEDdense–sparse sampling (_π_3-FT) yields the best overall<br>performance, while training on unfltered data (DIRTY) degrades accuracy.<br>**Camera Pose Estimation**<br>**Point Map Estimation**<br>**Method**<br>RRA@5_↑_RTA@5_↑_AUC@5_↑_MRE_↓_MTE_↓_<br>Acc_↓_<br>Comp_↓_<br>NC_↑_<br>Mean<br>Med.<br>Mean<br>Med.<br>Mean<br>Med.<br>_easy_<br>_π_3<br>88.97<br>68.79<br>45.84<br>4.12<br>7.82<br>0.055 0.030 0.039 0.019 0.712 0.822<br>_π_3-FT<br>95.64<br>**76.85**<br>55.58<br>1.64<br>**5.50**<br>**0.035 0.020 0.024 0.012** 0.724<br>**0.837**<br>_π_3-DIRTY<br>91.25<br>72.80<br>51.77<br>5.16<br>7.28<br>0.075 0.052 0.081 0.051 0.710 0.818<br>_π_3-RANDOM<br>95.08<br>76.42<br>55.00<br>1.78<br>5.72<br>0.039 0.021 0.026<br>0.013<br>0.720 0.831<br>_π_3-DENSE<br>95.13<br>76.73<br>**55.65**<br>1.84<br>5.61<br>0.036<br>**0.020** 0.026<br>0.013<br>**0.725 0.837**<br>_π_3-SPARSE<br>**96.27**<br>76.46<br>55.12<br>**1.61**<br>5.59<br>0.038 **0.020** 0.026<br>0.013<br>0.723 0.835<br>_hard_<br>_π_3<br>75.31<br>59.16<br>36.93<br>12.21<br>10.82<br>0.101 0.065 0.133 0.090 0.689 0.786<br>_π_3-FT<br>**86.40**<br>**71.00**<br>**47.93**<br>**5.72**<br>**7.27**<br>**0.068** 0.041<br>0.066<br>0.041<br>**0.713 0.818**<br>_π_3-DIRTY<br>81.10<br>65.99<br>43.74<br>11.86<br>9.72<br>0.130 0.094 0.139 0.091 0.693 0.791<br>_π_3-RANDOM<br>85.93<br>69.84<br>47.17<br>6.53<br>7.78<br>0.071 **0.040** 0.073 0.045 0.708 0.812<br>_π_3-DENSE<br>85.82<br>70.06<br>47.47<br>6.04<br>7.64<br>0.071 0.042 **0.062 0.035 0.713** 0.817<br>_π_3-SPARSE<br>85.97<br>70.53<br>47.13<br>6.05<br>7.52<br>0.070<br>**0.040** 0.070 0.041<br>0.710<br>0.814<br>in sparse settings.<br>**Ablation Analysis.** We analyze the effects of data qual-<br>ity and sampling strategies, with results shown in Tab. 2.<br>Training on unfltered (DIRTY) data consistently reduces<br>accuracy, even performing worse than the pretrained model<br>in point-map estimation on both the_easy_and_hard_ levels,<br>highlighting the importance of clean supervision for robust<br>**GT**<br>**Pretrained π3**<br>**Ours**<br>**_easy_**<br>**_hard_**<br>**Search depth**<br>**wrong**<br>**correct**|Table 2. **Ablation study on MegaDepth-X.**Finetuning on the cleaned<br>dataset with MIXEDdense–sparse sampling (_π_3-FT) yields the best overall<br>performance, while training on unfltered data (DIRTY) degrades accuracy.<br>**Camera Pose Estimation**<br>**Point Map Estimation**<br>**Method**<br>RRA@5_↑_RTA@5_↑_AUC@5_↑_MRE_↓_MTE_↓_<br>Acc_↓_<br>Comp_↓_<br>NC_↑_<br>Mean<br>Med.<br>Mean<br>Med.<br>Mean<br>Med.<br>_easy_<br>_π_3<br>88.97<br>68.79<br>45.84<br>4.12<br>7.82<br>0.055 0.030 0.039 0.019 0.712 0.822<br>_π_3-FT<br>95.64<br>**76.85**<br>55.58<br>1.64<br>**5.50**<br>**0.035 0.020 0.024 0.012** 0.724<br>**0.837**<br>_π_3-DIRTY<br>91.25<br>72.80<br>51.77<br>5.16<br>7.28<br>0.075 0.052 0.081 0.051 0.710 0.818<br>_π_3-RANDOM<br>95.08<br>76.42<br>55.00<br>1.78<br>5.72<br>0.039 0.021 0.026<br>0.013<br>0.720 0.831<br>_π_3-DENSE<br>95.13<br>76.73<br>**55.65**<br>1.84<br>5.61<br>0.036<br>**0.020** 0.026<br>0.013<br>**0.725 0.837**<br>_π_3-SPARSE<br>**96.27**<br>76.46<br>55.12<br>**1.61**<br>5.59<br>0.038 **0.020** 0.026<br>0.013<br>0.723 0.835<br>_hard_<br>_π_3<br>75.31<br>59.16<br>36.93<br>12.21<br>10.82<br>0.101 0.065 0.133 0.090 0.689 0.786<br>_π_3-FT<br>**86.40**<br>**71.00**<br>**47.93**<br>**5.72**<br>**7.27**<br>**0.068** 0.041<br>0.066<br>0.041<br>**0.713 0.818**<br>_π_3-DIRTY<br>81.10<br>65.99<br>43.74<br>11.86<br>9.72<br>0.130 0.094 0.139 0.091 0.693 0.791<br>_π_3-RANDOM<br>85.93<br>69.84<br>47.17<br>6.53<br>7.78<br>0.071 **0.040** 0.073 0.045 0.708 0.812<br>_π_3-DENSE<br>85.82<br>70.06<br>47.47<br>6.04<br>7.64<br>0.071 0.042 **0.062 0.035 0.713** 0.817<br>_π_3-SPARSE<br>85.97<br>70.53<br>47.13<br>6.05<br>7.52<br>0.070<br>**0.040** 0.070 0.041<br>0.710<br>0.814<br>in sparse settings.<br>**Ablation Analysis.** We analyze the effects of data qual-<br>ity and sampling strategies, with results shown in Tab. 2.<br>Training on unfltered (DIRTY) data consistently reduces<br>accuracy, even performing worse than the pretrained model<br>in point-map estimation on both the_easy_and_hard_ levels,<br>highlighting the importance of clean supervision for robust<br>**GT**<br>**Pretrained π3**<br>**Ours**<br>**_easy_**<br>**_hard_**<br>**Search depth**<br>**wrong**<br>**correct**|Table 2. **Ablation study on MegaDepth-X.**Finetuning on the cleaned<br>dataset with MIXEDdense–sparse sampling (_π_3-FT) yields the best overall<br>performance, while training on unfltered data (DIRTY) degrades accuracy.<br>**Camera Pose Estimation**<br>**Point Map Estimation**<br>**Method**<br>RRA@5_↑_RTA@5_↑_AUC@5_↑_MRE_↓_MTE_↓_<br>Acc_↓_<br>Comp_↓_<br>NC_↑_<br>Mean<br>Med.<br>Mean<br>Med.<br>Mean<br>Med.<br>_easy_<br>_π_3<br>88.97<br>68.79<br>45.84<br>4.12<br>7.82<br>0.055 0.030 0.039 0.019 0.712 0.822<br>_π_3-FT<br>95.64<br>**76.85**<br>55.58<br>1.64<br>**5.50**<br>**0.035 0.020 0.024 0.012** 0.724<br>**0.837**<br>_π_3-DIRTY<br>91.25<br>72.80<br>51.77<br>5.16<br>7.28<br>0.075 0.052 0.081 0.051 0.710 0.818<br>_π_3-RANDOM<br>95.08<br>76.42<br>55.00<br>1.78<br>5.72<br>0.039 0.021 0.026<br>0.013<br>0.720 0.831<br>_π_3-DENSE<br>95.13<br>76.73<br>**55.65**<br>1.84<br>5.61<br>0.036<br>**0.020** 0.026<br>0.013<br>**0.725 0.837**<br>_π_3-SPARSE<br>**96.27**<br>76.46<br>55.12<br>**1.61**<br>5.59<br>0.038 **0.020** 0.026<br>0.013<br>0.723 0.835<br>_hard_<br>_π_3<br>75.31<br>59.16<br>36.93<br>12.21<br>10.82<br>0.101 0.065 0.133 0.090 0.689 0.786<br>_π_3-FT<br>**86.40**<br>**71.00**<br>**47.93**<br>**5.72**<br>**7.27**<br>**0.068** 0.041<br>0.066<br>0.041<br>**0.713 0.818**<br>_π_3-DIRTY<br>81.10<br>65.99<br>43.74<br>11.86<br>9.72<br>0.130 0.094 0.139 0.091 0.693 0.791<br>_π_3-RANDOM<br>85.93<br>69.84<br>47.17<br>6.53<br>7.78<br>0.071 **0.040** 0.073 0.045 0.708 0.812<br>_π_3-DENSE<br>85.82<br>70.06<br>47.47<br>6.04<br>7.64<br>0.071 0.042 **0.062 0.035 0.713** 0.817<br>_π_3-SPARSE<br>85.97<br>70.53<br>47.13<br>6.05<br>7.52<br>0.070<br>**0.040** 0.070 0.041<br>0.710<br>0.814<br>in sparse settings.<br>**Ablation Analysis.** We analyze the effects of data qual-<br>ity and sampling strategies, with results shown in Tab. 2.<br>Training on unfltered (DIRTY) data consistently reduces<br>accuracy, even performing worse than the pretrained model<br>in point-map estimation on both the_easy_and_hard_ levels,<br>highlighting the importance of clean supervision for robust<br>**GT**<br>**Pretrained π3**<br>**Ours**<br>**_easy_**<br>**_hard_**<br>**Search depth**<br>**wrong**<br>**correct**|Table 2. **Ablation study on MegaDepth-X.**Finetuning on the cleaned<br>dataset with MIXEDdense–sparse sampling (_π_3-FT) yields the best overall<br>performance, while training on unfltered data (DIRTY) degrades accuracy.<br>**Camera Pose Estimation**<br>**Point Map Estimation**<br>**Method**<br>RRA@5_↑_RTA@5_↑_AUC@5_↑_MRE_↓_MTE_↓_<br>Acc_↓_<br>Comp_↓_<br>NC_↑_<br>Mean<br>Med.<br>Mean<br>Med.<br>Mean<br>Med.<br>_easy_<br>_π_3<br>88.97<br>68.79<br>45.84<br>4.12<br>7.82<br>0.055 0.030 0.039 0.019 0.712 0.822<br>_π_3-FT<br>95.64<br>**76.85**<br>55.58<br>1.64<br>**5.50**<br>**0.035 0.020 0.024 0.012** 0.724<br>**0.837**<br>_π_3-DIRTY<br>91.25<br>72.80<br>51.77<br>5.16<br>7.28<br>0.075 0.052 0.081 0.051 0.710 0.818<br>_π_3-RANDOM<br>95.08<br>76.42<br>55.00<br>1.78<br>5.72<br>0.039 0.021 0.026<br>0.013<br>0.720 0.831<br>_π_3-DENSE<br>95.13<br>76.73<br>**55.65**<br>1.84<br>5.61<br>0.036<br>**0.020** 0.026<br>0.013<br>**0.725 0.837**<br>_π_3-SPARSE<br>**96.27**<br>76.46<br>55.12<br>**1.61**<br>5.59<br>0.038 **0.020** 0.026<br>0.013<br>0.723 0.835<br>_hard_<br>_π_3<br>75.31<br>59.16<br>36.93<br>12.21<br>10.82<br>0.101 0.065 0.133 0.090 0.689 0.786<br>_π_3-FT<br>**86.40**<br>**71.00**<br>**47.93**<br>**5.72**<br>**7.27**<br>**0.068** 0.041<br>0.066<br>0.041<br>**0.713 0.818**<br>_π_3-DIRTY<br>81.10<br>65.99<br>43.74<br>11.86<br>9.72<br>0.130 0.094 0.139 0.091 0.693 0.791<br>_π_3-RANDOM<br>85.93<br>69.84<br>47.17<br>6.53<br>7.78<br>0.071 **0.040** 0.073 0.045 0.708 0.812<br>_π_3-DENSE<br>85.82<br>70.06<br>47.47<br>6.04<br>7.64<br>0.071 0.042 **0.062 0.035 0.713** 0.817<br>_π_3-SPARSE<br>85.97<br>70.53<br>47.13<br>6.05<br>7.52<br>0.070<br>**0.040** 0.070 0.041<br>0.710<br>0.814<br>in sparse settings.<br>**Ablation Analysis.** We analyze the effects of data qual-<br>ity and sampling strategies, with results shown in Tab. 2.<br>Training on unfltered (DIRTY) data consistently reduces<br>accuracy, even performing worse than the pretrained model<br>in point-map estimation on both the_easy_and_hard_ levels,<br>highlighting the importance of clean supervision for robust<br>**GT**<br>**Pretrained π3**<br>**Ours**<br>**_easy_**<br>**_hard_**<br>**Search depth**<br>**wrong**<br>**correct**|
|---|---|---|---|
||**_easy_**<br>**_hard_**<br>**wrong**<br>**correct**|**_easy_**||
|||||



Table 2. **Ablation study on MegaDepth-X.** Finetuning on the cleaned dataset with MIXED dense–sparse sampling ( _π_[3] -FT) yields the best overall performance, while training on unfiltered data (DIRTY) degrades accuracy. 

in sparse settings. 

**Ablation Analysis.** We analyze the effects of data quality and sampling strategies, with results shown in Tab. 2. Training on unfiltered (DIRTY) data consistently reduces accuracy, even performing worse than the pretrained model in point-map estimation on both the _easy_ and _hard_ levels, highlighting the importance of clean supervision for robust generalization. Among sampling schemes, RANDOM sampling yields reasonable camera pose accuracy but provides limited improvement in point map reconstruction, emphasizing the importance of adequate covisibilities in training batches. DENSE sampling performs well on easier scenes but is less effective under sparse conditions. SPARSE sampling alone does not yield the best trade-off. Although it exposes the model to more challenging cases, MIXED sampling achieves slightly better overall performance across difficulty levels. 

Figure 5. **Reconstruction results on the MegaDepth-X test set across two difficulty levels.** For each level, the top row shows the full 24-image input set, and the bottom row compares reconstructions from ground truth, pretrained _π_[3] , and our finetuned model with top-down views shown in the insets. Our model shows clearer improvements in the _hard_ setting, where the inputs are more challenging. Note that _hard_ was obtained using a deeper search depth than _easy._ 

Table 3. **Camera pose estimation on RealEstate10K [51] and CO3Dv2 [25]** . We follow _π_[3] ’s pose sampling conventions. Our fine-tuned models, trained on proposed Internet data dataset, remain comparable to pretrained baselines, demonstrating generalization to standard benchmarks. 

|**Method**|**RealEstate10K**<br>RRA@5_↑_<br>RTA@5_↑_<br>AUC@5_↑_<br>MRE_↓_<br>MTE_↓_|**CO3Dv2**<br>RRA@5_↑_<br>RTA@5_↑_<br>AUC@5_↑_<br>MRE_↓_<br>MTE_↓_|
|---|---|---|
|_π_3<br>_π_3-FT<br>VGGT<br>VGGT-FT|98.79<br>**79.61**<br>**62.82**<br>**0.51**<br>**5.65**<br>**98.80**<br>77.78<br>60.01<br>**0.51**<br>6.13|93.24<br>84.47<br>57.12<br>3.04<br>4.28<br>**93.97**<br>**84.50**<br>**57.61**<br>**2.96**<br>**4.26**<br>96.97<br>86.19<br>**67.84**<br>2.33<br>3.95<br>**97.11**<br>**86.27**<br>67.81<br>**2.29**<br>**3.92**|
||97.49<br>62.32<br>38.09<br>1.03<br>8.66<br>**98.23**<br>**71.88**<br>**48.23**<br>**0.82**<br>**6.85**||



**Qualitative Analysis.** We show qualitative results for three settings: the MD-X test set, real-world long-tail Internet scenes, and doppelganger scenes. 

_MegaDepth-X Visualization._ Fig. 5 shows reconstruction results on the MD-X test set across _easy_ and _hard_ levels. Our fine-tuned model produces more accurate camera poses, more dense and consistent 3D point maps compared to the pretrained baseline, especially on sparse ( _hard_ ) scenes. It generalizes well across varying camera intrinsics and challenging appearance changes such as day-night shifts. 

mentary material, we provide more results on doppelganger scenes. 

## **5.3. Generalization to Standard Benchmarks** 

We next examine whether the finetuned models preserve generalization on standard, curated benchmarks. 

_Real Long-Tail Scenes._ Real long-tail Internet scenes often contain fewer than 100 usable photos captured from uneven viewpoints and mixed with transient or irrelevant content. Classical SfM pipelines, e.g., COLMAP, typically fail to register most images, producing extremely sparse geometry or incomplete reconstructions. Pretrained models struggle under these conditions, yielding low-confidence predictions and fragmented structures. Our finetuned model remains stable and reconstructs coherent global geometry. As shown in Fig. 6, our model successfully reconstructs dense geometry from very few views, and handles doppelganger ambiguities with higher confidence, demonstrating strong robustness and generalization to real-world long-tail scenes. In the supple- 

**Relative Pose Estimation.** We evaluate on RealEstate10K [51] and CO3Dv2 [25], following _π_[3] ’s pose sampling conventions. As shown in Tab. 3, fine-tuning on Internet data generally maintains the performance of both backbones, and yields modest improvements for VGGT in particular. These results indicate that robustness learned from sparse, in-thewild Internet photos does not compromise generalization to standard 3D benchmarks. 

**Point Map Estimation.** Results on DTU [14], ETH3D [32], 7-Scenes [33], and NRGBD [2] (Tab. 4&5) show that our model maintains comparable reconstruction accuracy on 

**==> picture [486 x 255] intentionally omitted <==**

**----- Start of picture text -----**<br>
Image Collection<br>3 𝜋<br>Pretrained<br>Ours<br>COLMAP & Reference<br>**----- End of picture text -----**<br>


Figure 6. **Reconstruction results on real long-tail Internet scenes.** Each scene contains only a handful of photos with uneven viewpoints and noisy content, where COLMAP fails to register most images and produces extremely sparse geometry. Pretrained _π_[3] makes low-confidence predictions and incomplete reconstructions, while our fine-tuned model discovers the correct large-scale layout (e.g., (1) _Novo-Znamenka Manor_ , 66 images, 13 registered), handles very few-view inputs and recovers dense geometry ((2) _Sobanski Palace in Guzow_ , 95 images, 11 registered), reconstructs more complete structures under sparse, long-tail settings ((3) _Delizia del Verginese (Gambulaga, Portomaggiore)_ , 69 images, 11 registered, (5) _Chitharal Jain Monuments_ , 44 images, 15 registered), resolves doppelganger ambiguity ((4) _Hoshang’s Tomb_ , 85 images, 40 registered), and even works when COLMAP completely fails ((6) _Chapel of Saint Andrew’s cathedral (Saint Petersburg)_ , 94 images, 0 registered). These results demonstrate that our model remains robust and confident under severe sparsity and ambiguity in real long-tail Internet scenes. **For each scene, the confidence threshold is the same for pretrained** _π_[3] **and our method.** 

Table 4. **Point map estimation on DTU [14] and ETH3D [32]** . Finetuning on the proposed Internet photo dataset retain overall reconstruction quality on DTU, while performance on ETH3D decreases due to domain mismatch with Internet imagery. These results show that the model adapts to Internet photos without drifting too much on out-of-domain benchmarks. 

|**Method**||**DTU**|N.C._↑_<br>Mean<br>Med.||**ETH3D**|N.C._↑_<br>Mean<br>Med.|
|---|---|---|---|---|---|---|
||Acc. _↓_|Comp. _↓_<br>Mean<br>Med.||Acc. _↓_<br>Mean<br>Med.|Comp. _↓_<br>Mean<br>Med.||
||Mean<br>Med.||||||
|_π_3<br>_π_3-FT<br>VGGT<br>VGGT-FT|**1.151**<br>**0.622**<br>1.202<br>0.642|**1.793**<br>0.629<br>1.928<br>**0.593**|**0.668**<br>**0.754**<br>0.666<br>0.751|**0.188**<br>**0.126**<br>0.199<br>0.142|**0.211**<br>**0.129**<br>0.242<br>0.151|**0.872**<br>**0.967**<br>0.861<br>0.955<br>**0.841**<br>**0.942**<br>0.838<br>0.927|
||1.308<br>0.761<br>**1.283**<br>**0.759**|1.929<br>1.015<br>**1.900**<br>**0.953**|0.665<br>0.750<br>**0.669**<br>**0.756**|**0.270**<br>**0.174**<br>0.282<br>0.205|**0.304**<br>**0.180**<br>0.394<br>0.225||



Table 5. **Point map estimation on 7-Scenes [33] and NRGBD [2] datasets.** We evaluate both sparse-view and dense-view settings. Finetuning on Internet photos yields comparable performance to pretrained baselines with minor variations, indicating our method preserves generalization across diverse real world and synthetic datasets. 

|**View**|**Method**||**7-Scenes**|NC._↑_<br>Mean<br>Med.||**NRGBD**|NC._↑_<br>Mean<br>Med.|
|---|---|---|---|---|---|---|---|
|||Acc. _↓_<br>Mean<br>Med.|Comp. _↓_<br>Mean<br>Med.||Acc. _↓_<br>Mean<br>Med.|Comp. _↓_<br>Mean<br>Med.||
|_sparse_|_π_3<br>_π_3-FT|0.047<br>0.029<br>**0.046**<br>**0.027**|0.074<br>0.049<br>**0.072**<br>**0.046**|**0.741**<br>0.840<br>0.739<br>**0.841**|**0.024**<br>**0.013**<br>**0.024**<br>0.014|**0.028**<br>**0.013**<br>**0.028**<br>0.014|**0.909**<br>**0.991**<br>0.903<br>0.990<br>**0.882**<br>**0.979**<br>0.875<br>0.959|
||VGGT<br>VGGT-FT|**0.044**<br>**0.024**<br>0.062<br>0.046|**0.056**<br>**0.033**<br>0.097<br>0.070|0.733<br>**0.846**<br>**0.738**<br>0.844|**0.049**<br>**0.027**<br>0.071<br>0.046|**0.066**<br>**0.037**<br>0.071<br>0.041||
|_dense_|_π_3<br>_π_3-FT|**0.016**<br>**0.007**<br>**0.016**<br>**0.007**|**0.022**<br>**0.011**<br>0.023<br>**0.011**|**0.689**<br>**0.792**<br>0.686<br>0.789|**0.013**<br>**0.007**<br>**0.013**<br>**0.007**|**0.014**<br>0.006<br>**0.014**<br>**0.005**|**0.874**<br>**0.981**<br>0.864<br>0.978<br>**0.871**<br>**0.982**<br>0.859<br>0.981|
||VGGT<br>VGGT-FT|0.022<br>0.008<br>**0.016**<br>**0.007**|**0.026**<br>**0.012**<br>0.027<br>**0.012**|0.667<br>0.760<br>**0.681**<br>**0.781**|**0.015**<br>**0.008**<br>**0.015**<br>0.008|**0.015**<br>**0.006**<br>0.016<br>**0.006**||



DTU, 7-Scenes and NRGBD. We observe a performance decrease on ETH3D and a mild drop for VGGT under sparse 

NRGBD, likely reflecting the domain gap between these clean, controlled datasets and Internet imagery. Overall, the results indicate that training on diverse Internet photos preserves cross-dataset generalization without overfitting. 

## **6. Conclusion** 

We presented a step towards robust, Internet-scale 3D reconstruction by defining and addressing the long-tail regime of Internet photo collections. Through the MegaDepthX dataset and a sparsity-aware sampling strategy, we augment the ability of 3D foundation models to recover consistent geometry from sparse, noisy, and ambiguous imagery, where classical SfM and SOTA feed-forward 3D reconstruction models fail, and demonstrates disambiguation of doppelganger scenes while maintaining generalization across benchmarks. 

Our dataset currently focuses on landmark-scale scenes, representing only a small fraction of the landscape of Internet photos. Bootstrapping on the current dataset and refining models for reconstructions of even more longed-tail data remains an important direction for future work. Extending this framework beyond landmarks to everyday objects, indoor scenes, and other Internet photo domains offers a promising path toward a truly universal 3D foundation model. 

**Acknowledgments** This work was supported in part by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korean Government (MSIT) (No. RS-2024-00457882, National AI Research Lab Project). We thank Joseph Tung, Yiwen Zhang, Hanyu Chen and Haian Jin for discussion and help with MegaScenes dataset and depth post-processing. 

## **References** 

- [1] Sameer Agarwal, Yasutaka Furukawa, Noah Snavely, Ian Simon, Brian Curless, Steven M Seitz, and Richard Szeliski. Building rome in a day. _Communications of the ACM_ , 54(10): 105–112, 2011. 2 

- [2] Dejan Azinovic,´ Ricardo Martin-Brualla, Dan B Goldman, Matthias Nießner, and Justus Thies. Neural rgb-d surface reconstruction. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 6290–6301, 2022. 6, 7, 8 

- [3] Hana Bezalel, Dotan Ankri, Ruojin Cai, and Hadar AverbachElor. Extreme rotation estimation in the wild. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ , pages 1061–1070, 2025. 3 

- [4] Vincent D Blondel, Jean-Loup Guillaume, Renaud Lambiotte, and Etienne Lefebvre. Fast unfolding of communities in large networks. _Journal of statistical mechanics: theory and experiment_ , 2008(10):P10008, 2008. 5 

- [5] Daniel J. Butler, Jonas Wulff, Garrett B. Stanley, and Michael J. Black. A naturalistic open source movie for optical flow evaluation. In _Proceedings of the 12th European Conference on Computer Vision - Volume Part VI_ , page 611–625, Berlin, Heidelberg, 2012. Springer-Verlag. 3, 4 

- [6] Ruojin Cai, Bharath Hariharan, Noah Snavely, and Hadar Averbuch-Elor. Extreme rotation estimation using dense correlation volumes. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 14566–14575, 2021. 3 

- [7] Ruojin Cai, Joseph Tung, Qianqian Wang, Hadar AverbuchElor, Bharath Hariharan, and Noah Snavely. Doppelgangers: Learning to disambiguate images of similar structures. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 34–44, 2023. 3 

- [8] Filiberto Chiabrando, Loren Clark, John Driscoll, Scott McAvoy, Dominique Rissolo, Alessandra Spreafico, and Beatrice Tanduo. Salvation mountain - photogrammetry - terrestrial, photogrammetry - aerial, lidar - terrestrial, lidar - mobile, survey data, 2023. Distributed by Open Heritage 3D. 4 

- [9] CyArk. Great mosque - kilwa kisiwani - lidar - terrestrial, photogrammetry - terrestrial, photogrammetry - aerial, 2020. Distributed by Open Heritage 3D. 4 

- [10] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superpoint: Self-supervised interest point detection and description. In _Proceedings of the IEEE conference on computer vision and pattern recognition workshops_ , pages 224–236, 2018. 3 

- [11] Bardienus Pieter Duisterhof, Lojze Zust, Philippe Weinzaepfel, Vincent Leroy, Yohann Cabon, and Jerome Revaud. 

MASt3r-sfm: a fully-integrated solution for unconstrained structure-from-motion. In _International Conference on 3D Vision 2025_ , 2025. 6 

- [12] Jan-Michael Frahm, Pierre Fite-Georgel, David Gallup, Tim Johnson, Rahul Raguram, Changchang Wu, Yi-Hung Jen, Enrique Dunn, Brian Clipp, Svetlana Lazebnik, and Marc Pollefeys. Building Rome on a Cloudless Day. In _ECCV_ , 2010. 2 

- [13] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. _International Journal of Robotics Research (IJRR)_ , 2013. 3, 4 

- [14] Rasmus Jensen, Anders Dahl, George Vogiatzis, Engil Tola, and Henrik Aanæs. Large scale multi-view stereopsis evaluation. In _2014 IEEE Conference on Computer Vision and Pattern Recognition_ , pages 406–413. IEEE, 2014. 7, 8 

- [15] Hanwen Jiang, Hanwen Jiang, Arjun Karpur, Bingyi Cao, Qixing Huang, and Qi-Xing Huang. Omniglue: Generalizable feature matching with foundation model guidance. _2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 19865–19875, 2024. 3 

- [16] Haian Jin, Rundi Wu, Tianyuan Zhang, Ruiqi Gao, Jonathan T. Barron, Noah Snavely, and Aleksander Holynski. ZipMap: Linear-time stateful 3d reconstruction via test-time training. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2026. 2 

- [17] Arjun Karpur, Guilherme Perrotta, Ricardo Martin-Brualla, Howard Zhou, and Andre F. de Araujo.´ Lfm-3d: Learnable feature matching across wide baselines using 3d signals. _2024 International Conference on 3D Vision (3DV)_ , pages 11–20, 2023. 3 

- [18] Lawrence Kou, George Markowsky, and Leonard Berman. A fast algorithm for steiner trees. _Acta informatica_ , 15(2): 141–145, 1981. 5 

- [19] Vincent Leroy, Yohann Cabon, and Jer´ ome Revaud.ˆ Grounding image matching in 3d with mast3r, 2024. 2, 3 

- [20] Zhengqi Li and Noah Snavely. Megadepth: Learning singleview depth prediction from internet photos. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 2041–2050, 2018. 2, 3, 1 

- [21] Philipp Lindenberger, Paul-Edouard Sarlin, and Marc Pollefeys. Lightglue: Local feature matching atokens are frozen. _arXiv preprint arXiv:2306.13643_ , 2023. 3 

- [22] Kurt Mehlhorn. A faster approximation algorithm for the steiner problem in graphs. _Information Processing Letters_ , 27 (3):125–128, 1988. 5 

- [23] Pushmeet Kohli Nathan Silberman, Derek Hoiem and Rob Fergus. Indoor segmentation and support inference from rgbd images. In _ECCV_ , 2012. 3, 4 

- [24] E. Palazzolo, J. Behley, P. Lottes, P. Giguere, and C. Stachniss.` ReFusion: 3D Reconstruction in Dynamic Environments for RGB-D Cameras Exploiting Residuals. _arXiv_ , 2019. 3, 4 

- [25] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny.´ Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. _2021 IEEE/CVF International Conference on Computer Vision (ICCV)_ , pages 10881–10891, 2021. 7 

- [26] Ashley Richter, Michael Hess, Vid Petrovic, Falko Kuester, Cultural Heritage Engineering Initiative (CHEI), Architecture Center of Interdisciplinary Science for Art, and Archaeology (CISA3). Torre dei baldovinetti - florence - lidar - terrestrial, photogrammetry - terrestrial, 2023. Distributed by Open Heritage 3D. 4 

- [27] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superglue: Learning feature matching with graph neural networks. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 4938–4947, 2020. 3 

- [28] Johannes Lutz Schonberger¨ and Jan-Michael Frahm. Structure-from-motion revisited. In _Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2016. 2, 3 

- [29] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In _CVPR_ , 2016. 2, 3 

- [30] Johannes L. Schonberger, Enliang Zheng, Jan-Michael Frahm,¨ and Marc Pollefeys. Pixelwise view selection for unstructured multi-view stereo. In _European Conference on Computer Vision_ , 2016. 2 

- [31] Johannes Lutz Schonberger, Enliang Zheng, Marc Pollefeys,¨ and Jan-Michael Frahm. Pixelwise view selection for unstructured multi-view stereo. In _European Conference on Computer Vision (ECCV)_ , 2016. 3 

- [32] Thomas Schops, Johannes L. Sch¨ onberger, Silvano Galliani,¨ Torsten Sattler, Konrad Schindler, Marc Pollefeys, and Andreas Geiger. A multi-view stereo benchmark with highresolution images and multi-camera videos. In _2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 2538–2547, 2017. 7, 8 

- [33] Jamie Shotton, Ben Glocker, Christopher Zach, Shahram Izadi, Antonio Criminisi, and Andrew Fitzgibbon. Scene coordinate regression forests for camera relocalization in rgbd images. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ , pages 2930–2937, 2013. 7, 8 

- [34] Noah Snavely, Steven M Seitz, and Richard Szeliski. Photo tourism: exploring photo collections in 3d. In _ACM siggraph 2006 papers_ , pages 835–846. 2006. 2 

- [35] Noah Snavely, Steven M Seitz, and Richard Szeliski. Skeletal graphs for efficient structure from motion. In _2008 IEEE Conference on Computer Vision and Pattern Recognition_ , pages 1–8. IEEE, 2008. 5 

- [36] Joseph Tung, Gene Chou, Ruojin Cai, Guandao Yang, Kai Zhang, Gordon Wetzstein, Bharath Hariharan, and Noah Snavely. Megascenes: Scene-level view synthesis at scale. _arXiv preprint arXiv:2406.11819_ , 2024. 1, 2, 3 

- [37] Michał Tyszkiewicz, Pascal Fua, and Eduard Trulls. Disk: Learning local features with policy gradient. _Advances in Neural Information Processing Systems_ , 33:14254–14265, 2020. 3 

- [38] Hengyi Wang and Lourdes Agapito. 3d reconstruction with spatial memory. _arXiv preprint arXiv:2408.16061_ , 2024. 6 

- [39] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ , pages 5294–5306, 2025. 2, 3, 6 

- [40] Qianqian Wang*, Yifei Zhang*, Aleksander Holynski, Alexei A. Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. In _CVPR_ , 2025. 6, 3 

- [41] Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and Jiaolong Yang. Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ , pages 5261–5271, 2025. 4 

- [42] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. _arXiv preprint arXiv:2312.14132_ , 2023. 2, 6 

- [43] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer. Tartanair: A dataset to push the limits of visual slam. In _2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)_ , pages 4909–4916. IEEE, 2020. 3 

- [44] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. _π_[3] : Scalable permutation-equivariant visual geometry learning, 2025. 1, 2, 6, 3 

- [45] Yuanbo Xiangli, Ruojin Cai, Hanyu Chen, Jeffrey Byrne, and Noah Snavely. Doppelgangers++: Improved visual disambiguation with geometric 3d features, 2025. 3, 6 

- [46] Tao Xie, Peishan Yang, Yudong Jin, Yingfeng Cai, Wei Yin, Weiqiang Ren, Qian Zhang, Wei Hua, Sida Peng, Xiaoyang Guo, and Xiaowei Zhou. Scal3r: Scalable test-time training for large-scale 3d reconstruction, 2026. 2 

- [47] Jianing Yang, Alexander Sax, Kevin J Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ , pages 21924–21935, 2025. 2 

- [48] Yao Yao, Zixin Luo, Shiwei Li, Jingyang Zhang, Yufan Ren, Lei Zhou, Tian Fang, and Long Quan. Blendedmvs: A largescale dataset for generalized multi-view stereo networks. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 1790–1799, 2020. 3 

- [49] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and Ming-Hsuan Yang. Monst3r: A simple approach for estimating geometry in the presence of motion. _arXiv preprint arxiv:2410.03825_ , 2024. 3 

- [50] Shangzhan Zhang, Jianyuan Wang, Yinghao Xu, Nan Xue, Christian Rupprecht, Xiaowei Zhou, Yujun Shen, and Gordon Wetzstein. Flare: Feed-forward geometry, appearance and camera estimation from uncalibrated sparse views. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ , pages 21936–21947, 2025. 2 

- [51] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. _arXiv preprint arXiv:1805.09817_ , 2018. 7 

## **Long-Tail Internet Photo Reconstruction** 

## Supplementary Material 

## **Visualization Webpage** 

Please refer to our project page for additional visualizations beyond this PDF. The webpage includes: (i) animations of our sparsity-aware sampling procedure on representative scenes; and (ii) comparisons of reconstructions from pretrained _π_[3] and our finetuned _π_[3] on long-tail scenes (where COLMAP registers 0 images). We also provide video flythroughs of reconstructed point clouds and additional qualitative results on the webpage to help visualize performance on diverse, real-world scenes. 

## **A. The MegaDepth-X Dataset** 

## **A.1. Data Processing** 

In this section, we compare COLMAP results with those produced by our proposed data-processing pipeline. Fig. 12 shows reconstructions from COLMAP and our MASt3RSfM pipeline. COLMAP often fails on ambiguous scenes involving similar-looking objects, visually similar but distinct building facades, symmetric landmarks etc. In contrast, our reconstruction pipeline effectively mitigates these issues and recovers correct geometry. In Fig. 13, we show that our monocular depth–guided dense depthmap filtering strategy prevents background depths from leaking into foreground regions (i.e. the depth-bleeding issue [20]) and removes depth estimates on transient objects, which are often unreliable in COLMAP MVS. Note that we use monocular depth only as guidance, rather than warping it to align with the MVS depth. This is because we prioritize _accurate_ depth maps over complete ones. Uncertainty in the relative depth predictions of monocular models can introduce additional noise and inconsistency across views. For example, in the last row of Fig. 13, COLMAP MVS fails to recover the depth of the foreground statue, and we opt to remove the depth values in that region. If we were to warp the monocular depth to match the MVS result, then any inaccuracy in the relative depth between the statue and the background building could produce erroneous and inconsistent cross-view depth estimates. 

## **A.2. Dataset Statistics** 

We provide an overall comparison between MegaDepth and MegaDepth-X in Tab. 6, including reconstruction statistics as well as several metrics that characterize the spatial distribution of viewpoints. Beyond basic dataset properties such as the number of intact reconstructions, image count, and whether doppelganger filtering or dense depth refinement is applied, we analyze how cameras are positioned and oriented in each scene, as scenes with broad viewpoint coverage allow 

our sampling strategy to construct more diverse and representative sparse-view subsets. The statistics are computed from Manhattan-aligned COLMAP reconstructions. 

**Positional coverage.** To understand how cameras are placed in the horizontal plane, we compute each camera’s azimuth angle relative to the scene centroid (that is, the angle of the direction from the scene centroid to the camera) and divide the full 0-360° range into 36 equal 10° bins. In practice, the scene centroid is derived from the average of the SFM point cloud. A scene with many occupied bins is one where cameras are well-distributed around the object. In the table, the columns “Positional Azimuth Coverage = 100% / _≥_ 75% / _≥_ 50% / _≥_ 25%” report how many scenes achieve at least that percentage of bins(36 _/_ 36, 27 _/_ 36, 18 _/_ 36, 9 _/_ 36), with larger thresholds indicating closer to full 360° wrap-around coverage. 

**Rotational coverage.** Position alone does not describe where cameras are looking. We therefore measure the coverage of camera orientations by mapping each camera’s forward viewing direction to 36 azimuth bins similar to positional coverage. If cameras face more distinct directions, more bins are occupied; if they face similar directions, only few bins are occupied. We summarize this rotational azimuth coverage using the same percentage thresholds as positional azimuth coverage. 

These statistics show that MegaDepth-X contains substantially more scenes with broad camera-position coverage and diverse viewing directions, making it better suited for robust sparse-view reconstruction than MegaDepth. 

## **B. Sparsity-aware Sampling** 

## **B.1. Greedy Sampling Algorithm** 

We illustrate one iteration of the greedy view-sampling procedure in Alg. 1. At each step, the algorithm selects the next view based on two criteria: 

1. _Community novelty_ : prioritizing candidates whose camera-community has not yet been visited by the sampled set. This encourages the trajectory to enter unexplored regions of the view graph and reduces redundancy in viewpoint selection. 

2. _Spatial distance_ : among candidates with equal novelty, preferring those that are farther from the current camera position. This promotes larger baselines and helps diversify the spatial coverage of the sampled views. 

Table 6. **Dataset statistics and viewpoint-distribution metrics.** We report reconstruction statistics and metrics describing camera coverage. _Positional Azimuth Coverage_ counts scenes whose camera positions occupy 9–36 (i.e. 25%-100%) of the 36 horizontal azimuth bins (10° per bin, covering the full 360°). _Rotational Azimuth Coverage_ represents scenes whose camera forwarding vectors occupy 9–36 (i.e. 25%-100%) of the 36 horizontal azimuth bins (10° per bin, covering the full 360°). For each scene, the more bins covered, the wider the camera distribution is. _[†]_ Dense depth refinement uses monocular depth–guided filtering. 

|**Dataset**<br>**#Recons. #Images Doppelganger**<br>**Check**<br>**Dense Depth**<br>**Refnement**<br>MegaDepth [20]<br>266<br>119k<br>No<br>Yes<br>MegaDepth-X (Ours)<br>1,865<br>440k<br>Yes<br>Yes_†_|**Positional Azimuth Coverage**<br>= 100%_↑≥_75%_↑≥_50%_↑≥_25%_↑_|**Positional Azimuth Coverage**<br>= 100%_↑≥_75%_↑≥_50%_↑≥_25%_↑_|**Rotational Azimuth Coverage**<br>= 100%_↑≥_75%_↑≥_50%_↑≥_25%_↑_|
|---|---|---|---|
||4<br>6|15<br>25<br>74<br>80<br>223<br>752|27<br>56<br>107<br>230<br>76<br>490<br>1123<br>1816|
|||||
|**Algorithm 1:**One Stepof GreedyView Sampling||**Algorithm 2:**Round-Robin BFS Graph Partitioning||
|**Input:** Current node_v_<br>Neighborhood of_v_: _Nv_<br>Set of already sampled nodes_S_<br>Community map_M_ (node_→_community)<br>Camera positionsPos(_·_)<br>**Output:** Next sampled node_u∗_||**Input:** View graph_G_= (_V, E_)<br>Number of subgraphs_Ncc_<br>**Output:** Subgraphs_{P_1_, . . . , PNcc}_<br>Randomly select_Ncc_seed nodes<br>_{s_1_, . . . , sNcc} ⊆V_;<br>Initialize each_P_with seed_s_||



**Input:** View graph _G_ = ( _V, E_ ) Number of subgraphs _Ncc_ **Output:** Subgraphs _{P_ 1 _, . . . , PNcc }_ Randomly select _Ncc_ seed nodes _{s_ 1 _, . . . , sNcc } ⊆ V_ ; Initialize each _Pi_ with seed _si_ ; Initialize one BFS frontier for each subgraph; **while** _there exists a non-empty frontier_ **do for** _**each** subgraph Pi_ **do** Expand its frontier by one BFS step; Assign each newly reached unassigned node to _Pi_ ; **end end return** _{P_ 1 _, . . . , PNcc }_ ; 

// Identify communities already covered _S_ comm _←{M_ [ _s_ ] _| s ∈ S}_ ; // Compute candidate list with community novelty and distance _C ←∅_ ; **for** _**each** u ∈ Nv_ **do** _unreached ←_ ( _M_ [ _u_ ] _∈/ S_ comm); _dist ←∥_ Pos( _u_ ) _−_ Pos( _v_ ) _∥_ 2; _C ←C ∪{_ ( _u, unreached, dist_ ) _}_ ; **end** // Sort by unreached, then by distance Sort _C_ in descending lexicographic order by ( _unreached, dist_ ); // Select the top-ranked candidate ( _u[∗] ,_ ~~_,_ )~~ _←_ first element of _C_ ; **return** _u[∗]_ ; 

## **B.3. Graph Span vs. Search Depth** 

To understand how greedy search depth _D_ affects the coverage and sparsity of the sampled views, we analyze several statistics on the view-graph. Let _G_ denote the full viewgraph of a scene and _S_ the set of sampled nodes. The first two metrics quantify coverage with respect to the _entire_ graph _G_ , while the last two measure sparsity _within_ the sampled subset _S_ . 

Candidates are lexicographically ranked according to these two criteria, and the highest-ranked node is chosen as the next sampled view. 

**k-hop graph coverage.** This metric measures how much of the view-graph is reached by the sampled views. Specifically, it computes the fraction of nodes in _G_ that lie within _k_ hops of any sampled node: 

## **B.2. Graph Partition** 

**==> picture [231 x 23] intentionally omitted <==**

Before sparsity-aware sampling, we partition COLMAP’s view graph into _Ncc_ subgraphs. Specifically, we randomly select _Ncc_ seed nodes and treat each seed as the initial node of one partition. Starting from these seeds, we perform a parallel round-robin breadth-first search(BFS) over the view graph. During each iteration, every subgraph expands from its current frontier to its unassigned neighboring nodes, which are then incorporated into that subgraph. In this way, each node is assigned to the subgraph of the seed that first reaches it, until no further nodes can be expanded. 

where _S_ is the subgraph of greedy sampled nodes and _dG_ ( _u, v_ ) is the shortest path from _u_ to _v_ on the graph _G_ . A higher Cov _k_ indicates broader topological coverage, i.e., the sampled set reaches many graph neighborhoods rather than remaining confined to a small region. 

**Nearest-sample distance.** To evaluate spatial coverage in 3D, we compute the average Euclidean distance from each 

**==> picture [122 x 72] intentionally omitted <==**

**==> picture [122 x 72] intentionally omitted <==**

**==> picture [122 x 72] intentionally omitted <==**

**==> picture [122 x 72] intentionally omitted <==**

**==> picture [475 x 15] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) k-hop Coverage ( k =2) (b) Nearest-Sample Distance (c) Graph Dispersion (pairwise hops) (d) Euclidean Dispersion (pairwise dis-<br>tance)<br>**----- End of picture text -----**<br>


Figure 7. **Coverage and sparsity vs. search depth.** Metrics in (a) and (b) evaluate coverage with respect to the _full_ view-graph, while (c) and (d) measure the sparsity of the _sampled_ subset. As the search depth increases, the sampled set reaches a larger portion of the view-graph, as shown by the rise in _k_ -hop (graph-distance) coverage in (a). The average distance from each camera to its nearest sampled view decreases in (b), indicating broader spatial coverage. At the same time, both graph dispersion (average pairwise graph distance) in (c) and Euclidean dispersion (average pairwise 3D distance) in (d) increase with depth, showing that the sampled views become more widely separated across the graph and in 3D space. 

camera to its closest sampled camera: 

**==> picture [209 x 27] intentionally omitted <==**

where _pu_ and _pv_ are camera positions. Lower values mean the sampled views are spatially well-distributed and lie near many original cameras. 

**Graph dispersion and Euclidean dispersion.** To understand the sparsity of the sampled views, we calculate the average pairwise distance among sampled views(dispersion) based on graph distances and Euclidean distances: 

**==> picture [223 x 60] intentionally omitted <==**

Higher dispersion values indicate that the sampled views are more sparsely distributed in both the graph and Euclidean space. 

We compute these metrics for the top 100 scenes with the most registered images, evaluating 12 search depths and averaging over 8 sampling runs per depth. The number of sampled views is 24 for all samples. Results are shown in Fig. 7, indicating that deeper searches yield higher coverage on the full graph (a,b) and produce sparser, more widely distributed sampled subsets (c,d). 

## **C. Training Details and Additional Results** 

## **C.1. Training Setup** 

We finetune both _π_[3] and VGGT using their released pretrained checkpoints. All input images are first padded with white borders to a resolution of 518 _×_ 518. During training, we apply random crops to these padded images, sampling aspect ratios uniformly from [0 _._ 75 _,_ 1 _._ 0]. We also apply random 

color jittering on training images. Each mini-batch contains up to 24 images drawn from MegaDepth-X, with the number of views per batch randomly selected from [2 _,_ 24]. We process at most 96 images on each GPU. We also augment image orientations during training by randomly rotating images 90 _[◦]_ clockwise or counterclockwise with a probability of 0.2. 

We use the original loss functions from _π_[3] [44] and VGGT [39] to finetune the models. To preserve the geometric priors encoded in the pretrained models, we finetune only the Alternating-Attention modules, while keeping the point-cloud and camera decoders frozen. We further include BlendedMVS [48] and TartanAir [43] as additional training data for finetuning. Finetuning is performed for 100 epochs, where each epoch iterates over all scenes in the combined dataset. We use the AdamW optimizer with a peak learning rate of 1 _×_ 10 _[−]_[5] , scheduled with linear warm-up followed by cosine annealing. All experiments are conducted on 4 NVIDIA A6000 GPUs. 

## **C.2. Additional Depth-Estimation Results** 

We provide monocular and video depth results to complement the main paper. Following [40, 44, 49], we evaluate Absolute Relative Error (Abs Rel) and the accuracy at a threshold of _δ <_ 1 _._ 25. For monocular depth, we report performance on Sintel [5], Bonn [24], KITTI [13], and NYU-v2 [23]. For video depth, we evaluate on Sintel [5], Bonn [24], and KITTI [13] under both _scale_ and _scale&shift_ alignment settings. Our finetuned models maintain competitive performance across all datasets, demonstrating that the adaptation to in-the-wild imagery does not degrade their depth-estimation ability. 

## **C.3. Results on Doppelganger Scenes** 

Doppelganger cases often cause both classical SfM pipelines and pretrained feed-forward models to fail, merging distinct structures into a single incorrect reconstruction. As shown in Fig. 8, our fine-tuned _π_[3] model correctly distinguishes visu- 

Table 7. Video Depth Estimation on Sintel [5], Bonn [24], and KITTI [13]. We report Absolute Relative Error (Abs Rel, lower is better) and the prediction accuracy at a threshold of _δ <_ 1 _._ 25 (higher is better). 

||Method<br>Align|Sintel<br>Abs Rel_↓_<br>_δ <_1_._25_↑_|Bonn<br>Abs Rel_↓_<br>_δ <_1_._25_↑_|KITTI<br>Abs Rel_↓_<br>_δ <_1_._25_↑_|
|---|---|---|---|---|
||_π_3<br>_scale_<br>_π_3-FT<br>VGGT<br>VGGT-FT|0.228<br>0.671<br>**0.213**<br>**0.713**|0.051<br>0.975<br>**0.047**<br>**0.978**|**0.038**<br>**0.986**<br>0.040<br>0.985<br>0.072<br>0.965<br>**0.065**<br>**0.966**|
|||0.294<br>0.649<br>**0.242**<br>**0.707**|**0.055**<br>**0.971**<br>0.061<br>0.969||
||_π_3<br>_scale_&_shift_<br>_π_3-FT<br>VGGT<br>VGGT-FT|0.207<br>0.735<br>**0.188**<br>**0.739**|0.045<br>0.976<br>**0.043**<br>**0.978**|**0.036**<br>**0.986**<br>0.038<br>0.985<br>0.059<br>0.961<br>**0.056**<br>**0.964**|
|||0.226<br>0.683<br>**0.197**<br>**0.728**|**0.049**<br>**0.974**<br>0.056<br>0.973||



Table 8. Monocular Depth Estimation on Sintel [5], Bonn [24], KITTI [13], and NYU-v2 [23]. We report Absolute Relative Error (Abs Rel, lower is better) and threshold accuracy _δ <_ 1 _._ 25 (higher is better). 

||Method|Sintel<br>Abs Rel_↓_<br>_δ <_1_._25_↑_|Bonn<br>Abs Rel_↓_<br>_δ <_1_._25_↑_|KITTI<br>Abs Rel_↓_<br>_δ <_1_._25_↑_|NTU-v2<br>Abs Rel_↓_<br>_δ <_1_._25_↑_|
|---|---|---|---|---|---|
||_π_3<br>_π_3-FT|**0.277**<br>0.621<br>0.284<br>**0.629**|0.052<br>0.971<br>**0.049**<br>**0.977**|0.059<br>**0.972**<br>**0.056**<br>**0.972**|0.054<br>0.956<br>**0.052**<br>**0.958**<br>0.055<br>0.953<br>**0.053**<br>**0.955**|
||VGGT<br>VGGT-FT|0.331<br>0.600<br>**0.311**<br>**0.628**|**0.051**<br>**0.974**<br>0.056<br>**0.974**|**0.089**<br>0.939<br>0.092<br>**0.941**||



**==> picture [237 x 165] intentionally omitted <==**

**----- Start of picture text -----**<br>
COLMAP Pretrained  𝜋 [3] Ours Reference<br>Palace of Rozumovskyi (Baturyn)<br>Church of the Saviour on the Blood<br>Radcliffe Camera<br>**----- End of picture text -----**<br>


Figure 8. **Disambiguation of doppelganger scenes.** Each example shows a pair of visually similar structures that cause classical SfM (COLMAP) and pretrained _π_[3] to collapse into incorrect or merged reconstructions. In contrast, our finetuned model correctly distinguishes the symmetric or repetitive sides of the same building, reconstructing consistent geometry for each viewpoint. Reference views from Google Earth are provided for comparison, confirming that our model resolves these ambiguities and recovers accurate global structure under challenging visual similarity. 

**==> picture [237 x 91] intentionally omitted <==**

**----- Start of picture text -----**<br>
Doppelganger pair 𝜋 [3] 𝜋 [3] -DENSE 𝜋 [3] -FT 𝜋 [3] -SPARSE<br>**----- End of picture text -----**<br>


Figure 9. **Comparison of ablated models on doppelganger scenes** We show predictions from the pre-trained model and ablated models on two doppelganger scenes. Disambiguation behavior holds across fine-tuned variants with sparsity-aware sampling, while the pre-trained model and model finetuned with densely sampled views are less robust to doppelgangers. 

fig.9. Results indicate that pretrained models and dense-only fine-tuning are less robust to ambiguity, while finetuning with sparsity-aware sampling (e.g., mixed or sparse) tends to improve disambiguation, suggesting sparsity-aware sampling helps. 

## **C.4. Quantitative results on Long-tail scenes** 

ally similar but distinct structures within each landmark and recovers geometry consistent with reference aerial imagery, indicating improved reconstruction of global scene layout. 

To evaluate the effectiveness of different sampling strategies on doppelganger scenes, we evaluate the pretrained _π_[3] and finetuned _π_[3] on doppelganger scenes and show results in 

To enable quantitative evaluation on long-tail scenes, we augment MegaScenes with additional observations from external cultural heritage datasets [8, 9, 26] and jointly register all images using COLMAP. The quantitative and qualitative results of this long-tail evaluation are shown in Fig. 10. Our model consistently reduces the mean relative rotation and 

**==> picture [496 x 396] intentionally omitted <==**

**----- Start of picture text -----**<br>
Image Collection COLMAP Pretrained  𝜋 [3] Ours<br>MRE:     50.90   MTE:      25.90 MRE:       10.85   MTE:        9.39<br>RRA@5:  8.14   RTA@5: 22.39 RRA@5:  51.35   RTA@5: 61.24<br>MRE:      39.97   MTE:      43.18 MRE:       16.04   MTE:      35.76<br>RRA@5:   4.76   RTA@5: 23.81 RRA@5:  23.81   RTA@5: 14.29<br>MRE:        5.06  MTE:        6.13 MRE:         2.87   MTE:        4.27<br>RRA@5: 56.19  RTA@5: 69.52 RRA@5:  86.67   RTA@5: 80.00<br>**----- End of picture text -----**<br>


Figure 10. **Quantitative results on Long-tail scenes.** Our model performs better on scenes with strong ambiguities (first row) and on scenes with minimal overlap across different scene components (second row). For a more densely photographed scene that still exhibits large viewpoint variation (third row), our model not only reduces pose error but also reconstructs a more complete point cloud. 

translation errors across all scenes, while also producing more complete point clouds. 

## **C.5. Limitations** 

Long-tail scenes often contain fragmented viewpoints, where different subsets of images capture disjoint parts of the scene (e.g., indoor and outdoor areas) without overlapping views to connect them. When such mixed collections are fed into the models at once, both pretrained and finetuned _π_[3] may blend these unrelated regions into a single 3D structure, as illustrated in Fig.11. While our finetuned model handles these mixtures more robustly than the pretrained baseline, enabling the model to reason robustly about disconnected components and produce reasonable overall layouts still remains a challenge. 

**==> picture [496 x 118] intentionally omitted <==**

**----- Start of picture text -----**<br>
Input Images Pretrained 𝜋 [3] Ours Google Earth<br>**----- End of picture text -----**<br>


Figure 11. **Limitations.** This example contains images from two disjoint parts of the scene: indoor photos with warm lighting (producing a yellowish point cloud) and outdoor photos (producing a white point cloud). Pretrained _π_[3] struggles to handle such mixed inputs and produces inconsistent geometry. Our finetuned model is more robust in this setting, but both models still fuse the indoor and outdoor structures into a single reconstruction without separating them. 

**==> picture [496 x 324] intentionally omitted <==**

**----- Start of picture text -----**<br>
Image COLMAP-MVS Ours<br>(a) Dragon Bridge (Ljubljana) (b) Eletsky Monastery (c) Schloss Linderhof<br>117 views<br>131 views<br>(d) Royal Albert Hall (e) Sant'Andrea (Vercelli)<br>North South outside inside<br>Figure 12. Comparison of COLMAP and our reconstruction<br>COLMAP<br>Ours<br>COLMAP<br>Ours<br>**----- End of picture text -----**<br>


Figure 12. **Comparison of COLMAP and our reconstruction pipeline.** We replace COLMAP with MASt3R-SfM [11] combined with the doppelganger++ classifier [45] to obtain sparse reconstructions, allowing effective disambiguation of doppelganger scenes. (a) The bridge has two similar dragon statues, one at each end. COLMAP incorrectly treats them as the same statue and registers them together, whereas our method correctly separates them. (b), (d), and (e) illustrate additional doppelganger cases, in which different sides or parts of a landmark are mistakenly merged. (c) In this low-texture scene, our pipeline also succeeds in registering more images. 

Figure 13. **Comparison of COLMAP MVS and our filtered dense depth results.** COLMAP MVS suffers from depth bleeding and struggles to correctly estimate the depth of transient objects. Our strategy mitigates these issues by leveraging ordering priors from monocular depth predictions. Note that we prioritize _accurate_ depth maps over complete ones. In the last row, COLMAP fails to recover the depth of the foreground statue, and we opt to remove the depth values in that region. If we were to warp the monocular depth to match the MVS result, then any inaccuracy in the relative depth between the statue and the background building could produce erroneous and inconsistent cross-view depth estimates. 

