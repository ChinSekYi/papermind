## Generative Modeling of Neurodegenerative Brain Anatomy with 4D Longitudinal Diffusion Model

Nivetha Jayakumar ∗

Department of Electrical and Computer Engineering, University of Virginia, Charlottesville, 22903, VA, USA

Swakshar Deb *

Department of Electrical and Computer Engineering, University of Virginia, Charlottesville, 22903, VA, USA

Bahram Jafrasteh

Department of Radiology, Weill Cornell Medical College, Cornell University, Ithaca, 14850, NY, USA

Qingyu Zhao

Department of Radiology, Weill Cornell Medical College, Cornell University, Ithaca, 14850, NY, USA

Miaomiao Zhang

Department of Electrical and Computer Engineering, Department of Computer Science, University of Virginia, Charlottesville, 22903, VA, USA

## Abstract

Understanding and predicting the progression of neurodegenerative diseases remains a major challenge in medical AI, with significant implications for early diagnosis, disease monitoring, and treatment planning. However, most available longitudinal neuroimaging datasets are temporally sparse with a few follow-up scans per subject. This scarcity of temporal data limits our ability to model and accurately capture the continuous anatomical changes related to disease progression in individual subjects. To address this problem, we propose a novel 4D (3D × T) diffusion-based generative framework that effectively models and synthesizes longitudinal brain anatomy over time, con- ditioned on available clinical variables such as health status, age, sex, and other relevant factors. Moreover, while most current approaches focus on manipulating image intensity or texture, our method explicitly learns the data distribution of topology-preserving spatiotemporal deformations to effectively capture the geometric changes of brain structures over time. This design enables the realistic generation of future anatomical states and the reconstruction of anatomically consistent disease trajectories, providing a more faithful representation of longitudinal brain changes. We validate our model through both synthetic sequence generation and downstream longitudinal disease classification, as well as brain segmentation. Experiments on two large-scale longitudinal neuroimage datasets demonstrate that our method outperforms state-of-the-art baselines in generating anatomically accurate, temporally consistent, and clinically meaningful brain trajectories. Our code is available on Github.

∗ These authors contribute equally to this work.

Keywords: Longitudinal Neurodegeneration, Generative Modeling, Deformation-Based Morphometry

## 1. Introduction

Neurodegenerative diseases such as Alzheimer's disease are characterized by gradual, spatially heterogeneous atrophy of brain structures that precede clinical symptoms by years [50, 22, 51]. Understanding the morphometric and geometric changes of brain structures is essential for early diagnosis, individualized prognosis, and timely therapeutic intervention [16, 36]. However, longitudinal neuroimaging data are often temporally sparse, incomplete, or entirely unavailable due to challenges such as high imaging costs, patient dropout, and the difficulty of conducting repeated scans over extended study periods [37, 39]. This raises a clinically and technically significant question: can we predict an individual's future anatomical trajectory from a single baseline scan? Solving such a task would benefit early identification of high-risk individuals, simulate future disease progression, and better support clinical trials aimed at an early-stage intervention.

Recent advances in generative modeling have opened new possibilities for synthesizing future or missing brain imaging scans from limited longitudinal observations [64, 61, 58]. Early generative approaches, such as generative adversarial networks (GANs) [25, 41], demonstrated initial promise but often suffered from instability, limited sample diversity, and mode collapse. Such issues hinder their ability to model the full variability of neurodegenerative processes. In contrast, denoising diffusion probabilistic models (DDPMs) [18] have recently emerged as powerful alternatives capable of capturing complex, high-dimensional distributions of brain morphology with remarkable stability and fidelity [64, 26, 27]. By conditioning on auxiliary variables such as age, cognitive scores, disease status, and anatomical priors, diffusion-based frameworks have demonstrated promising results in generating high-quality, high-resolution 2D brain scans to model disease progression and structural degeneration. These advances represent a significant step toward data-driven, predictive modeling of neurodegeneration in clinical neuroscience.

## 1.1. Related Works

Current generative diffusion models synthesize brain anatomy by directly manipulating image intensities or textures, without explicitly modeling the underlying geometry of anatomical structures [64, 27]. As a result, these methods may produce anatomically implausible samples, including unrealistic topology and structural distortions [60, 19]. To address this, recent work has incorporated deformation-based morphometry (DBM) into generative frameworks [43, 26], representing brain changes as smooth transformations between pairwise images rather than direct intensity edits. Such transformations maintain one-to-one correspondences across time and subjects, preventing folding, tearing, or discontinuous warping [15, 19]. This property is particularly critical for studying progressive neurodegenerative diseases, where capturing subtle localized atrophy requires maintaining anatomical plausibility. Despite these advantages, existing DBM-based generative methods remain limited as they primarily model deformations between pairwise images while intermediate time points are generated via temporal interpolation [26]. These approaches do not explicitly model the true longitudinal progression trajectory, thus failing to synthesize subsequent scans that reflect realistic, temporally consistent anatomical changes over several time points.

More importantly, existing generative models are fundamentally limited in their ability to process full 4D (3D × T) neuroimaging data. Most current approaches can synthesize only sequential 2D slices (2D x T) [58], or a single 3D follow-up volume [19], but they cannot generate anatomically coherent 3D sequences that evolve continuously over time. Therefore, they are unable to produce full 4D longitudinal trajectories from a single baseline scan. To alleviate this issue, several diffusion-based methods propose to synthesize follow-up scans by conditioning on a sequence of prior scans [32, 60] or rely on auxiliary information such as image-derived features, radiomics [12, 11], and regional atrophy measurements [44, 46]. However, these methods assume access to multiple longitudinal scans, costly brain segmentations, or additional imaging modalities at inference time, all of which are rarely available in routine clinical workflows; hence substantially limiting their real-world applicability. Meanwhile, recent 4D diffusion models emerging in computer vision focus on synthesizing dynamic 3D scenes from fusing multiple camera viewpoints [62, 31, 56]. These methods are inherently tailored to multi-view consistency and camera pose estimation, making them fundamentally incompatible with the challenge of modeling biological shape changes over time in longitudinal neuroimaging.

In this paper, we propose a novel 4D longitudinal diffusion model in the time-sequential deformation space of brain images. In contrast to existing approaches [43, 20] that are limited to 2D or 3D architectures -thereby sacrificing either spatial or temporal fidelity - our framework fully captures spatiotemporal anatomical dynamics by modeling a diffusion process over a sequence of 3D deformation fields. These deformation fields are parameterized by stationary velocity fields, enabling smooth, invertible transformations that preserve anatomical topology [53]. To support this 4D generative modeling, we introduce a new frame-wise volumetric patch embedding strategy that tokenizes each 3D volume independently while maintaining temporal consistency across the sequence. This allows us to explicitly learn the temporal evolution of brain structures without compromising spatial detail or anatomical plausibility. Our contributions are summarized below:

- To the best of our knowledge, we are first to develop a full 4D diffusion network for longitudinal brain modeling that jointly learns spatial and temporal features, which extends current architectures beyond 3D without compromising either dimension.
- Develop a novel diffusion model in spatiotemporal deformation spaces to ensure smooth, topology-preserving transformations and anatomically consistent results.
- Demonstrate utility of the generated samples in two downstream tasks: AD classification and brain segmentation via augmentation with missing data from longitudinal neuroimage repositories. It is worthy to note that our designed network architecture is flexible and can operate in both the intensity and deformation spaces.

We evaluate the effectiveness of our proposed framework on longitudinal brain MRI data from the ADNI repository [42]. Experimental results demonstrate that our method excels at generating longitudinal sequences with preserved anatomical structure and shape, outperforming state-of-theart diffusion models that operate in the pixel space, as well as recursive models that synthesize missing time points using multi-frame guidance.

## 2. Background: Deformation-based Brain Morphometry

This section outlines the formulation of topological constraints within the framework of deformation-based brain morphometry by leveraging diffeomorphic transformations between source and target brain images [35, 17]. In the context of longitudinal neuroimaging, it is typically assumed that any pair of scans acquired from the same subject over time can be related through a continuous deformation field [4, 47, 23]. To ensure anatomical fidelity and preserve the underlying brain topology, these deformation fields are constrained to lie within the space of diffeomorphisms, i.e., smooth, invertible mappings with smooth inverses [6, 2, 59]. These topological constraints are essential in DBM, as they prevent non-physical artifacts such as folding, tearing, or self-intersections in the warped anatomy, enabling reliable quantification of structural changes over time.

Given a template image S and a fixed image F defined on a d -dimensional torus domain Ω = R d / Z d ( S ( x ) , F ( x ) : x ∈ Ω → R ), a diffeomorphic transformation, ϕ t , for t ∈ [0 , 1] , is defined as a smooth flow over time to deform a template image to a fixed image by a composite function, S ◦ ϕ -1 t . Here, the ◦ denotes an interpolation operator. Such a transformation is typically parameterized by time-dependent velocity fields under a large diffeomorphic deformation metric mapping (LDDMM) [6], or a stationary velocity field (SVF), which remains constant over time and is obtained using the scalingand-squaring algorithm [53, 3]. While we employ SVF in this paper, our framework is easily applicable to the other. For a stationary velocity field v , the diffeomorphisms, ϕ t , are generated as solutions to the equation:

<!-- formula-not-decoded -->

The solution of Eq. (1) is identified as a group exponential map using a scaling and squaring scheme [3]. More details are included in [3].

The diffeomorphic transformation, ϕ at t = 1 , that reflects geometric changes between images can be solved by minimizing the energy function:

<!-- formula-not-decoded -->

Here Dist(·,·) is a distance function that measures the dissimilarity between images, Reg( · ) is a regularization term that enforces the smoothness of transformation fields, and η is a positive weighting parameter. Widely used distance functions include the sum-of-squared intensity differences ( L 2 -norm) [6], normalized cross correlation (NCC) [4], and mutual information (MI) [57]. In this paper, we utilize a sum-of-squared distance function.

## 3. Our Method

This section introduces a novel 4D longitudinal diffusion model, which comprises of two main components: (i) a diffeomorphic registration network U φ [5] that extracts DBM from the longitudinal brain images via velocity fields; and (ii) a 4D diffusion model ψ θ that learns to synthesize a time sequence of velocity fields conditioned on clinical and anatomical context. An overall network architecture is presented in Figure 1.

Problem Setup. Different from 4D video or motion data [26, 33], longitudinal neuroimaging data consists of discrete 3D volumetric brain scans acquired at irregular and subject-specific time points. Consider a set of N longitudinal images, for each subject n ∈ [1 , · · · , N ] , let a t,n ∈ { a 0 ,n , a 1 ,n , . . . , a T,n } denote the age at the t -th scan, where T is the total number of time frames and a 0 ,n corresponds to the baseline age. The corresponding baseline brain volume is denoted by I a 0 ,n ∈ R H × W × L , where H,W,L are the spatial dimensions of the 3D MRI scan. Given this baseline scan and associated clinical/demographic information such as the subject's disease label y n , our objective is to model a sequence of anatomically plausible follow-up brain volumes, { I a 1 ,n , I a 2 ,n , . . . , I a T,n } , corresponding to subject-specific, non-uniform age intervals ∆ a t,n = a t +1 ,n -a t,n , which may vary across individuals.

We first learn the velocity fields from the longitudinal MRIs using a diffeomorphic registration network [5]. Specifically, the baseline scan, I a 0 ,n , is independently registered to each follow-up scan, I a t ,n , resulting in a set of initial velocity fields, i.e., { v a 1 ,n , v a 2 ,n , ...v a T ,n } . These velocity fields are then integrated by Eq. (1) to obtain smooth, invertible deformation maps { ϕ -1 a 1 ,n , ϕ -1 a 2 ,n , ...ϕ -1 a T ,n } that capture the anatomical transformations from the Baseline Scan Follow-up Scans

74

Spatial Gradient

Registration Net Velocity Fields

4D Longitudinal Diffusion Model

т ~ (0,1000)

"Alzheimer's"

LVPE

baseline to each follow-up time point. Let φ denote the parameters of the registration network, we define the associated loss function as

APPE

<!-- formula-not-decoded -->

where ||∇ v a t ,n ( φ ) || 2 is a regularizer enforcing smoothness of the transformations with λ being a positive weighting parameter. These predicted velocities serve as training inputs for the longitudinal 4D diffusion model.

[[0.85, 0.23,

.., 0.556]]

## 3.1. 4D Longitudinal Diffusion Model

0.33, 0.978,

Inspired by denoising diffusion probabilistic models [18], we develop a diffusion transformer that explicitly models longitudinal sequences within each

~

Figure 1: The overall architecture of our framework.

<!-- image -->

Timestep

Embedder

Class

Embedder

Patch MLP

(LVPE)

76

Reverse Diffusion

Add &amp; Norm step of the Markov chain. More specifically, our model learns to approximate the data distribution defined over temporal sequences of velocity fields, i.e., z n ≜ { v a 0 ,n , v a 1 ,n , ...v a T ,n } . Specifically, each input sequence is a 4D tensor z ∈ R C × T × H × W × L , where C is the number of channels (i.e., C = 3 ). For simplicity, we drop the batch notation n . The forward process progressively perturbs the input data over a fixed number of timesteps, transforming it into a distribution that approximates a standard Gaussian. At an intermediate timestep, τ ∈ { 1 , · · · , T } , the diffusion process can be formulated as

<!-- formula-not-decoded -->

where β τ controls the noise variance. After re-parametrization, the last timestep of this Markov chain can be obtained as a single step process using the formulation z T = √ ¯ α τ z 0 + √ 1 -¯ α τ ϵ where ϵ ∼ N (0 , I ) , α τ = 1 -β τ and ¯ α τ = ∏ τ s =1 α s .

The reverse process reconstructs the signal by iteratively sampling from a Gaussian distribution using

<!-- formula-not-decoded -->

where µ θ and Σ θ represent the mean and variance of the process at timestep τ , estimated by a network ψ θ parameterized by θ . As shown in [18], we can directly estimate the reverse process mean function estimator by training a neural network to predict ϵ from z τ based on a set of conditional signals C .

To effectively model full 4D longitudinal sequences, we propose a new transformer-based architecture to replace the conventional U-Net denoising network, ψ θ , parameterized by θ . Drawing inspiration from the video vision transformer [1], we redesign the denoising backbone with dedicated components optimized for spatiotemporal modeling, as described below:

Longitudinal Volumetric Patch Embedding (LVPE). Existing methods generate longitudinal data by processing sequences of 3D volumes [60, 33]; however, they often compromise either spatial or temporal coherence by flattening or reshaping the input sequence to make it compatible to current transformer architectures. To address this, we introduce a patch extraction mechanism that operates in the 4D spatiotemporal domain. More specifically, we first apply a 3D convolutional-based patch embedder κ independently to each temporal frame, as brain anatomical topology remains fundamentally stable across time points with minor deformations occurring between scan intervals.

We then partition each 3D volume into non-overlapping patches and project them into a fixed-dimensional embedding space R d of dimension d , yielding a patch token tensor of shape m ∈ R T × N d × d , where N d = ( H × W × L ) /P 3 is the number of spatial patches per frame with a patch size of P . Formally, the embedding process is defined as

<!-- formula-not-decoded -->

Note that our proposed operation differs from current works [40, 34] by incorporating 4D data through frame-wise 3D patch embeddings. This approach preserves the fine-grained spatial structure within each 3D volume while maintaining temporal correspondence across time points, enabling effective decoupling of spatial and temporal modeling in subsequent transformer layers. In contrast to tubelet-based embeddings [63, 49] that extract spatiotemporal tubes assuming uniform temporal spacing, our frame-wise patch extraction is naturally suited to longitudinal medical data with irregular age intervals ∆ a t . This design choice avoids the need for temporal interpolation or padding, reducing computational overhead while improving modeling flexibility.

Age-aligned Patchwise Position Encoding (APPE). The patches extracted from the input sequence are enriched with both spatial and temporal positional encodings to preserve temporal alignment and to capture global spatiotemporal dependencies. A key innovation of our design is a two-step temporal alignment strategy. First , we introduce a temporally aware encoding that embeds age information directly into the sinusoidal positional functions, enabling the model to reason about continuous biological time. Second , we incorporate a complementary fixed 1D temporal embedding assigned to each volumetric patch, providing a stable temporal reference across the sequence. To ensure that the ages are accurately incorporated with their respective 3D volumes, we define linear temporal position encoding using sinusoidal embeddings, followed by a non-linear multilayer perceptron (MLP) transformation. Each linear age embedding f ( a t ) is defined as a continuous function, i.e.,

<!-- formula-not-decoded -->

where ω i ′ = 1 / 10000 2 i ′ D . The age-aligned temporal encoding, MLP ( f ( a t )) , is then added to the extracted patches along the temporal axis. In addition, we introduce a second set of fixed temporal sinusoidal encodings, which are injected before each temporal attention block in the transformer. This dualencoding strategy provides both biologically grounded temporal context and a stable temporal reference to improve the model's ability to learn coherent spatiotemporal dynamics.

We then employ a fixed 3D sinusoidal spatial positional encoding that assigns each token a unique embedding based on its location in a 4D spatiotemporal grid. Inspired by [34], we construct independent 1D sinusoidal encodings along each spatial axis and combine them into a full 3D positional embedding. This design guarantees a unique representation for every spatial coordinate in the volume while enabling the transformer to capture longrange, global spatial dependencies. Beyond being transformer-agnostic, this approach also overcomes the limitations of relative or axis-decoupled encodings, which cannot fully encode multidimensional spatial context and may inadvertently assign identical embeddings to distinct patches with the same relative index.

## Adaptive Spatio-Temporal Contextualization using Anatomical Em-

beddings. Our design is motivated by the need to incorporate both subjectspecific anatomical context and temporally coherent disease progression, which standard transformer architectures and existing conditioning schemes fail to fully capture. We propose a multimodal conditioning mechanism that integrates both disease class and anatomical priors directly into the transformer. Specifically, the anatomical prior is defined as the spatial gradient of the initial image scan, ∇ I a 0 , which aligns with the directionality of learned transformations [6, 24]. The disease label y and diffusion timestep τ are separately embedded through learnable non-linear layers, and the resulting embeddings are fused and injected into the normalization layers of each DiT block via adaptive layer normalization. This avoids the overhead and complexity of voxel-wise cross-attention mechanisms, which often require explicit spatial alignment-an unrealistic assumption in longitudinal synthesis where anatomical structure evolves across time. To jointly capture anatomical detail and temporal consistency, we employ a factorized space-time attention scheme [1] to alternate between spatial and temporal transformer blocks. Spatial blocks attend to 3D patches within each timepoint, preserving intraframe anatomical structure, while temporal blocks attend across frames at fixed spatial locations, modeling age-dependent progression. This enables anatomy-aware attention for individual frames and temporally coherent syn- thesis across the sequence, which are crucial for realistic 4D generation.

## 3.2. Training Objective

Given a sequence of clean initial latent features ( z 0 ) and a randomly sampled timestep τ ∈ { 1 , . . . , T } , we train the model to predict the added noise, ϵ , from z τ a t = √ ¯ α τ z 0 a t + √ 1 -¯ α τ ϵ . The denoiser ψ θ ( z τ , τ, C ) is conditioned on diffusion timestep τ and additional condition C (i.e., age, baseline scan and disease), and is trained to minimize the L1 error

<!-- formula-not-decoded -->

This formulation encourages the model to learn a conditional score function that reverses the forward noise process, generating consistent longitudinal velocity fields. These velocity fields are then integrated to deform the baseline scan, producing a temporally ordered 4D trajectory of brain anatomy. The pseudocode for training and sampling are outlined in Alg.1 and Alg.2.

## 4. Experiments

## 4.1. Experimental Setup

This section highlights our experimental setup to validate our method. We evaluate our framework by assessing the quality, fidelity, and anatomical consistency of the synthesized longitudinal MRI scans. This includes comparison against state-of-the-art baselines, analyses across multiple quantitative metrics, and ablation studies to study the effect of model capacity, temporal conditioning, and autoencoders for various synthesis strategies. We further assess the utility of the generated scans for downstream tasks, including classification and segmentation, to demonstrate their clinical applicability and value in longitudinal MRI analysis.

Evaluation Metrics. Conditioned on baseline scan, age, and diagnosis, we evaluated the generated followup MRI scans using multiple metrics to assess both accuracy and realism. For accuracy, we employed Peak Signal-to-Noise Ratio (PSNR) [14] and Structural Similarity Index Measure (SSIM) [55], both standard in image quality assessment. To measure realism, we computed the Fréchet Inception Distance (FID) and Kernel Inception Distance (KID) in the feature space, following established protocols and using a standard pre-trained model [10]. We also evaluate the anatomical consistency of

## Algorithm 1 LDT Training Inputs: Scans { I a 0 , I a 1 , . . . , I a T } , ages [ a 0 , a 1 , ..., a T ] , disease class label y Stage 1: Registration Net U φ Training 1: for n ∈ [1 , N ] do 2: Template ← I a 0 ,n ; Fixed ←{ I a 1 ,n , . . . , I a T ,n } . 3: for a t,n ∈ [ a 1 ,n , a T,n ] do 4: v a t ,n ← U ψ ( I a 0 ,n , I a t ,n ) 5: end for 6: Minimize l ( φ ) 7: end for Stage 2: 4D Longitudinal Diffusion Model ψ θ Training 1: z 0 n ← [ v a 1 ,n , ..., v a T ,n ] (dropping ′ n ′ ) 2: z τ ← √ ¯ α τ z 0 + √ 1 -¯ α τ ϵ 3: m = [ κ ( z τ a 1 ) , κ ( z τ a 2 ) , ..., κ ( z τ a T )] 4: m ← [( m a 1 + f ( a 1 )) , ..., ( m a T + f ( a T ))] 5: m ← m + spatial encoding. 6: for ˆ l ∈ # layers do 7: m ← Spatial Block(m) 8: m ← m + temporal encoding. 9: m ← Temporal Block(m) 10: m ← Normalization ( m,κ ( ∇ I a 0 ) , τ embed , y embed ) 11: end for 12: ϵ θ ← Einsum ( m ) 13: Minimize L diff Algorithm 2 LDT Sampling Inputs: Predictor step, T = 1000 , Corrector step, M = 2 1: Initialize z T ∼ P T ( x ) 2: for τ = T 1 , . . . , 0 do 3: z τ ← Predictor ( z τ +1 ) 4: for ˆ τ = 1 , . . . , M do 5: z τ ← Corrector ( z τ ) 6: end for 7: end for 8: return z 0

our generated deformation fields by analyzing the Determinant of Jacobian (DetJac) across frames and LDT model variants. Low percentages of negative DetJac values indicate preserved topology, providing a measure of how reliably the synthesized scans maintain anatomically plausible deformations.

Baselines. We compare our method (of both intensity- and deformationbased variants) against three state-of-the-art diffusion-based baselines for longitudinal MRI generation: Sequence-Aware Diffusion Model (SADM) [60], BrLP [44] and CounterSynth [43]. Due to SADM's high computational demands, we implement a latent version using features from a pretrained autoencoder. To ensure fair comparison in the same experimental setting, BrLP is trained without its auxiliary segmentation component since we do not include ground-truth segmentation labels in our training dataset. While many prior works focus on interpolating between two scans, our evaluation focuses on comparison with models that extrapolate future volumes to promote a fair benchmark given that our method relies solely on the baseline scan.

Downstream Task. We evaluate the utility of our synthesized longitudinal scans as a data augmentation strategy for downstream classification and segmentation. For classification, missing scans are generated for each subject and incorporated into the training set using the VGG3D network [45], allowing us to study how synthetic images supplement real data, particularly when labeled samples are limited, and improve performance across all disease stages (CN, MCI, AD). For segmentation, a UNet model [48] is trained on the OASIS dataset using FreeSurfer-generated labels, and the same synthesized deformations are applied to generate paired, anatomically consistent images and segmentation maps. This allows us to assess how the generated scans enhance anatomical fidelity and support accurate downstream predictions.

Ablation Studies. To evaluate our framework, we perform ablation studies focusing on model architecture, temporal conditioning, and synthesis strategies. We experiment with three transformer variants-LDT-S, LDT-L, and LDT-XL-which differ in hidden embedding dimensions, number of transformer blocks, and number of attention heads, to study the scalability of our deformation-based model (Ours-Def.). We also compare two age-conditioning strategies: age-wise temporal position encoding (APPE) applied to extracted volumetric patches, and linear age vectors incorporated through adaptive normalization alongside diffusion timestep, disease class, and anatomical prior. Finally, we implement an intensity-based version (Ours-Int.) by replacing the registration network with a latent-space VAE-GAN and compare it against our deformation-based model, which learns the distribution of velocity fields from a pre-trained registration network. These studies allow us to analyze the effect of model capacity, temporal encoding, and synthesis approach on generating anatomically plausible longitudinal scans.

## 4.2. Dataset.

This section outlines the datasets, preprocessing steps, and associated metadata used for training and evaluation. We utilize longitudinal brain MRI data from two public repositories to train our framework and validate its performance via downstream classification and segmentation tasks. Details regarding dataset composition, temporal characteristics, and preprocessing steps involved are provided below.

Alzheimer's Disease Neuroimaging Initiative (ADNI) [42]. We use T1-weighted MRIs of 1021 participants with at least 4 longitudinal visits from the ADNI repository [37]. Scans are skull-stripped, intensity normalized, and affine-registered to a common template space [8]. Based on the disease diagnosis at the time of visit, the subjects (aged 55 -92 ) are divided into three classes - Cognitively Normal (CN), Alzheimer's Disease (AD), and Mild Cognitive Impairment (MCI). Each sample is resized to ( 128 × 128 × 128 × 4 ), i.e., 3D volumes at 4 time points. Metadata includes age and diagnosis at each individual visit. The dataset is split into 85% training and 15% testing, with each subject having at least 4 time-points. The baseline classification models are trained with the same dataset, and augmented with subjects having less than 4 time-points, where the longitudinal sequence is completed using synthesized samples from our framework. For the augmented data, we use 176 subjects with only one scan, 188 subjects with two time-points and 148 subjects with three available time-points. Note that all these subjects have a disease diagnosis and ages available for the respective time-points. We sample ages for the synthesized scans from normal distributions computed with disease-wise means and standard deviations from the training dataset.

Open Access Series of Imaging Studies(OASIS) [28]. We use the OASIS dataset solely for evaluating the downstream segmentation task. To ensure compatibility with our registration framework trained on ADNI, all OASIS MRI scans are aligned to the ADNI template space using affine registration via the ANTs toolkit [52]. Following affine registration and alignment, the segmentation labels are obtained using the SynthSeg tool under the FreeSurfer framework [7]. The dataset has a total of 53 subjects, 20% of which are used for testing. A baseline segmentation model is trained using the initial scan of each subject, and the synthesized frames are incrementally used to augment the training set. To obtain segmentation labels for the augmented samples, we propagate the ground truth labels from the initial scans using the velocity fields generated by our deformation model, as we do with the initial scans.

## 4.3. Implementation Details.

Similar to DDPM [18], we set the total number of diffusion timesteps as 1000 . A cosine noise schedule [38] is used in the diffusion process. All networks are trained with a learning rate of 10 -4 , effective batch size of 48 and the Adam optimizer. We train the registration network for 1500 epochs and the diffusion model for approximately 200 K training steps. While the final brain MRIs are at a resolution of 128 3 , our diffusion model synthesizes velocity fields at a lower resolution of 32 3 for computational efficiency. Since velocities exhibit a smooth, band-limited structure [54, 21], we up-sample the synthesized velocity field back to the original resolution using trilinear interpolation. All experiments were performed with NVIDIA A100 GPUs.

## 5. Results

## 5.1. Evaluation of Sample Fidelity.

Figures 2, 3, and 4 illustrate examples of anatomical changes of CN, MCI, and AD captured by our framework compared to SOTA methods. Visually, Latent-SADM [60] fails to preserve anatomical structures as the age gap increases, while BrLP [44] and CounterSynth [43] fail to model the progression of cortical degeneration seen in subjects with AD. In contrast, both our models (intensity/deformation-based) generate realistic changes in brain volume, conditioned on ages.

While our intensity-based variant occasionally introduces anatomically implausible regions, the deformation model consistently preserves brain topology and accurately models longitudinal progression. These results also highlight that our model better preserves structural integrity and more effectively captures the anatomical progression patterns across the cognitive spectrum Initial Scan

77yrs

Ours-Def.

Left- Deformations

Right- DetJacs Synthesized Volumes

80.06yrs

83.02yrs

&lt; X

85.04yrs Morphological Difference

43.06yrs

46.02yrs

48.04yrs Initial Scan

Figure 2: Left to right: Comparison of synthesized follow-up volumes across all methods, along with morphological difference maps that highlight longitudinal changes from the initial scan for a subject with Cognitively Normal (shown in the axial view ). For our proposed deformation-based model, we additionally visualize the estimated deformation field and the corresponding Jacobian determinant (DetJac) that reflect topological structure of the brain changes over time.

<!-- image -->

68.2yrs

Ours-Def.

eft- Deformations

Right- DetJacs Synthesized Volumes

69.33yrs

74.75yrs

77.78yrs Morphological Difference

41.13yrs

46.55yrs

49.58yrs

8

-

Figure 3: Left to right: Comparison of synthesized follow-up volumes across all methods, along with morphological difference maps that highlight longitudinal changes from the initial scan for a subject with Mild Cognitive Impairment (shown in the axial view ). For our proposed deformation-based model, we additionally visualize the estimated deformation field and the corresponding Jacobian determinant (DetJac) that reflect topological structure of the brain changes over time.

<!-- image -->

Initial Scan

62.8yrs

Ours-Def.

Left- Deformations

Right- DetJacs Synthesized Volumes

63.84yrs

67.08yrs

70.5yrs Morphological Difference

41.04yrs

44.28yrs

47.7yrs of CN, MCI, and AD. Notably, it reflects the accelerated atrophy characteristic of AD, including prominent ventricular enlargement and visible shrinkage of both gray and white matter structures. In contrast, MCI shows more gradual structural changes, while CN remains largely stable over time. The generated sequences preserve fine anatomical details and temporal coherence, demonstrating the model's ability to synthesize realistic neurodegenerative trajectories consistent with clinical observations.

Figure 4: Left to right: Comparison of synthesized follow-up volumes across all methods, along with morphological difference maps that highlight longitudinal changes from the initial scan for a subject with Alzheimer's Disease (shown in the axial view ). For our proposed deformation-based model, we additionally visualize the estimated deformation field and the corresponding Jacobian determinant (DetJac) that reflect topological structure of the brain changes over time.

<!-- image -->

Table 1 presents the quantitative comparison of all methods. SADM requires multiple follow-up scans as input to synthesize a single volume, making it less practical. Overall, it shows that our proposed models (intensity or deformation-based) achieve superior fidelity across all metrics. To further quantitatively verify anatomical consistency, we also compute the determinant of Jacobian (DetJac) distributions for our deformation-based model (last row of Figure 4). DetJac serves as a standard measure of topological preservation, where the values of 1 indicate local volume preservation, values &lt; 1 denote shrinkage, and values &gt; 1 indicate expansion. Crucially, negative DetJac values correspond to anatomically implausible transformations, such as folding or singularities, that violate topology. DetJac of our deformation model accurately captures tissue atrophy and cerebrospinal fluid expansion with merely 3 . 5 × 10 -4 % negative values across all samples, demonstrating its strong ability to maintain anatomical consistency.

Table 1: Quality of generated MRI scans by different methods based on ADNI.

| Metric       | SADM                       | BrLP              | CounterSynth      | Ours-Int.              | Ours-Def.              |
|--------------|----------------------------|-------------------|-------------------|------------------------|------------------------|
| Input Output | { I a i } T - 1 i =0 I a T | I a 0 I a t       | I a t - 1 I a t   | I a 0 { I a i } T i =1 | I a 0 { I a i } T i =1 |
| FID ↓        | 23 . 53 ± 0 . 00           | 62 . 13 ± 16 . 44 | 0 . 78 ± 0 . 57   | 1 . 014 ± 0 . 812      | 0 . 241 ± 0 . 2        |
| PSNR ↑       | 14 . 00 ± 0 . 00           | 18 . 04 ± 0 . 45  | 25 . 39 ± 2 . 05  | 24 . 72 ± 1 . 73       | 25 . 90 ± 1 . 30       |
| SSIM ↑       | 0 . 73 ± 0 . 00            | 0 . 32 ± 0 . 05   | 0 . 92 ± 0 . 02   | 0 . 89 ± 0 . 01        | 0 . 93 ± 0 . 01        |
| KID ↓        | 6 . 42 ± 0 . 00            | 9 . 88 ± 4 . 30   | 0 . 008 ± 0 . 008 | 0 . 037 ± 0 . 04       | 0 . 003 ± 0 . 005      |

Table 2 reports the mean, standard deviation, and percentage of negative values in the Determinant of Jacobian (DetJac) of the deformation fields across frames and LDT model variants. A low or zero percentage of negative values indicates preservation of topology in the generated deformation fields. As seen in our results, all variants of our model exhibit a notably low percentage of negative DetJac values, indicating consistent preservation of topology in the synthesized scans.

Table 2: DetJac statistics across transformer model configurations and time-points.

| Model   | Frame                   | Mean                    | Std.Dev.                | - DetJac % ↓                              |
|---------|-------------------------|-------------------------|-------------------------|-------------------------------------------|
| LDT-S   | Frame 1 Frame 2 Frame 3 | 1 . 000 0 . 999 0 . 999 | 0 . 043 0 . 047 0 . 058 | 1 . 892 e - 6 0 . 0000 0 . 0000           |
| LDT-L   | Frame 1 Frame 2 Frame 3 | 1 . 000 1 . 000 0 . 999 | 0 . 045 0 . 050 0 . 055 | 1 . 298 e - 4 7 . 466 e - 4 1 . 869 e - 4 |
| LDT-XL  | Frame 1 Frame 2 Frame 3 | 0 . 999 0 . 999 0 . 999 | 0 . 102 0 . 109 0 . 137 | 8 . 250 e - 5 1 . 211 e - 5 3 . 405 e - 5 |

## 5.2. Evaluation of Sample Reliability via Downstream Tasks.

We first evaluate the utility of our synthesized longitudinal scans as a data augmentation strategy for downstream classification using the VGG3D network [45] as the backbone. A key advantage of our framework is that it synthesizes deformations, which can be applied to both the input images and their ground-truth segmentation maps. This allows us to generate new samples of paired longitudinal image and segmentation labels. More specifically, we generate missing scans for each subject in the dataset and incorporate these synthetic images to augment the training set. We then evaluate the anatomical fidelity of the synthesized longitudinal scans by training a hippocampal segmentation model based on UNet backbone [48] on the OASIS dataset [29]. Here, we use 53 longitudinal sequences with the corresponding segmentation labels generated by FreeSurfer [7] as ground truth.

We then train the segmentation model with various numbers of synthesized longitudinal frames added to the original training set to further evaluate

As shown in Figure 5, we first gradually increase the proportion of original training data while keeping the synthesized sample fixed. The consistent improvements in classification accuracy across all training sizes demonstrate that our generated scans effectively supplement the real data, especially when labeled data is limited. Furthermore, Figure 6 presents class-wise improvements for CN, MCI, and AD, demonstrating that the synthesized scans boost performance across all disease stages.

al MRI augmentation.

Dice Scores for Hippocampus by Augmentation

0.85 - longitudinal MRI scans.

Ground Truth

Hippocampus Mask Predicted Hippocampus Mask - Baseline Predicted Hippocampus Mask - Augmented 3x

eir potential impact on downstream performance. As shown in Figure

eft), the inclusion of 1-3 synthetic volumes progressively improves the se

entation dice score (a metric that quantifies the overlap between the pr

0.84-

0.83 -

cted segmentation and the ground truth) [13], yielding up to a 3% increa.

Prediction Error

er the baseline. Figure 7 (Right) presents qualitative comparisons: tl

seline model under-segments the hippocampal tail and produces over

0.82

1oothed boundaries, while the augmented model generates segmentatic

+2

+3

Number of Frames Augmented

0.81

asks that better align with the ground truth, more accurately capturir

Baseline +1

e full hippocampal anatomy with sharper contours. These results hig

Figure 5: Improved ADNI classification accuracy via synthetic longitudinal MRI augmentation.

<!-- image -->

Figure 6: Per class performance improvement on ADNI with data augmentation using synthesized longitudinal MRI scans. Ground Truth Predicted Hippocampus Predicted Hippocampus

<!-- image -->

Hippocampus Mask Mask - Baseline Mask - Augmented 3x

their potential impact on downstream performance. As shown in Figure 7 (Left), the inclusion of 1-3 synthetic volumes progressively improves the segmentation dice score (a metric that quantifies the overlap between the predicted segmentation and the ground truth) [13], yielding up to a 3% increase over the baseline. Figure 7 (Right) presents qualitative comparisons: the baseline model under-segments the hippocampal tail and produces overly smoothed boundaries, while the augmented model generates segmentation masks that better align with the ground truth, more accurately capturing the full hippocampal anatomy with sharper contours. These results highlight the anatomical consistency of our generated scans and their utility in enhancing downstream segmentation performance. Prediction Error

Figure 7: Left: hippocampi segmentation results trained with frame-wise longitudinal augmentation from OASIS. Right (top): exemplary visualization of groundtruth hippocampi segmentations vs. predictions without/with augmentation. Right (bottom): segmentation error maps between groundtruth vs.predictions without/with augmentation.

<!-- image -->

## 5.3. Ablation Studies

Transformer scalability. To evaluate the scalability of our proposed model in the deformation space (Ours-Def.), we experiment with three transformer variants, LDT-S, LDT-L, and LDT-XL, which differ in hidden embedding dimensions, depth and the number of attention heads.

Table 3 summarizes the corresponding model configurations and performance metrics. As the model capacity increases, we observe a consistent improvement in both the visual fidelity and the quantitative performance of the generated scans. This suggests that our architecture benefits from scaling, effectively using increased quality of learned representations to model temporally complex anatomical changes. The results further show that larger transformer backbones can more accurately capture longitudinal brain dynamics while maintaining structural coherence.

Table 3: Model architecture scaling performance metrics for different transformer configurations written as (Hidden Dimension, Number of Heads, Number of Layers).

| Model   | FID ↓          | PSNR ↑          | SSIM ↑          | KID ↓           | Configuration   |
|---------|----------------|-----------------|-----------------|-----------------|-----------------|
| LDT-S   | 0 . 499 ± 0.17 | 25 . 434 ± 1.83 | 0 . 924 ± 0.02  | 0 . 033 ± 0.04  | (384, 6, 12)    |
| LDT-L   | 0 . 241 ± 0.2  | 25 . 897 ± 1.30 | 0 . 934 ± 0.01  | 0 . 003 ± 0.005 | (768, 12, 12)   |
| LDT-XL  | 0 . 160 ± 0.09 | 25 . 379 ± 1.35 | 0 . 927 ± 0.005 | 0 . 002 ± 0.001 | (960, 12, 16)   |

APPE's Age-wise Temporal Position Encoding. We provide qualitative results comparing two age-conditioning strategies: age-wise temporal position encoding applied to extracted volumetric patches, and linear age vectors incorporated through adaptive normalization alongside diffusion timestep τ , disease class y , and anatomical prior ∇ I a 0 .

Figure 8 indicates that temporal position encoding leads to better integration of age information, resulting in more realistic and anatomically plausible brain changes over time as compared to the linear additive approach.

Intensity vs. Deformation Variants. We implement an intensity-based version (Ours-Int.) by replacing the registration network with a latent-space VAE-GAN from MONAI [9, 30], trained on our dataset to ensure accurate reconstruction. We present results from both the intensity-based model and the deformation-based model, which learns the distribution of velocity fields from a pre-trained registration network.

Synthesized Volumes

63.84yrs

67.08yrs

70.5yrs Morphological Difference

41.04yrs

44.28yrs

47.7yrs

1.00

Figure 8: Visualization of axial, coronal and sagittal views of scans synthesized by our intensity framework with and without age-specific temporal position encoding. Scans are generated for a subject with Alzheimer's Disease.

<!-- image -->

Figure 9 shows that the intensity model could produce unrealistic, hallucinated anatomical structures due to direct manipulation of voxel intensities. In contrast, the deformation model generates structure-preserving transfor-

-Del.

Halluc

mations by learning plausible velocity fields that deform the initial scan. This leads to more anatomically consistent and clinically relevant samples, making the deformation-based approach better suited for downstream tasks.

Figure 9: Visualization of sagittal views of scans synthesized by our intensity framework vs. our deformation framework for a subject with Alzheimer's Disease.

<!-- image -->

## 6. Conclusion

In this work, we introduced a novel framework for synthesizing a complete 4D longitudinal brain anatomy from a single baseline scan, leveraging a transformer-based diffusion model in the space of diffeomorphic velocity fields. Our model jointly learns spatial and temporal dynamics, which ensures anatomically consistent and topology-preserving trajectories across time. Extensive experiments demonstrate the superiority of our approach over the state-of-the-art in terms of synthesis quality and downstream clinical utility, including neurodegenerative disease classification and brain segmentation. Notably, our flexible network architecture supports both intensityand deformation-space modeling. By filling in gaps in sparse longitudinal datasets and predicting brain changes over time, our method represents a significant step toward data-driven modeling of neurodegenerative progression in clinical and research settings. Future work will unify intensity-based and deformation-based models to better capture longitudinal brain changes involving both structural deformations and appearance variations.

Acknowledgments. This work was supported by NSF CAREER Grant 2239977 and NIH 1R21EB032597.

Declaration of generative AI and AI-assisted technologies in the manuscript preparation process. During the preparation of this work the author(s) used Copilot in order to perform a grammar check and make the content concise. After using this tool/service, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the published article.

## References

- [1] Arnab, A., Dehghani, M., Heigold, G., Sun, C., Lučić, M., Schmid, C., 2021. Vivit: A video vision transformer, in: Proceedings of the IEEE/CVF international conference on computer vision, pp. 6836-6846.
- [2] Arnold, V., 1966. Sur la géométrie différentielle des groupes de lie de dimension infinie et ses applications à l'hydrodynamique des fluides parfaits, in: Annales de l'institut Fourier, pp. 319-361.
- [3] Arsigny, V., Commowick, O., Pennec, X., Ayache, N., 2006. A logeuclidean framework for statistics on diffeomorphisms, in: Medical Image Computing and Computer-Assisted Intervention-MICCAI 2006: 9th International Conference, Copenhagen, Denmark, October 1-6, 2006. Proceedings, Part I 9, Springer. pp. 924-931.
- [4] Avants, B.B., Epstein, C.L., Grossman, M., Gee, J.C., 2008. Symmetric diffeomorphic image registration with cross-correlation: evaluating automated labeling of elderly and neurodegenerative brain. Medical image analysis 12, 26-41.
- [5] Balakrishnan, G., Zhao, A., Sabuncu, M.R., Guttag, J., Dalca, A.V., 2018. An unsupervised learning model for deformable medical image registration, in: Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 9252-9260.
- [6] Beg, M.F., Miller, M.I., Trouvé, A., Younes, L., 2005. Computing large deformation metric mappings via geodesic flows of diffeomorphisms. International journal of computer vision 61, 139-157.
- [7] Billot, B., Greve, D.N., Puonti, O., Thielscher, A., Van Leemput, K., Fischl, B., Dalca, A.V., Iglesias, J.E., et al., 2023. Synthseg: Segmentation of brain mri scans of any contrast and resolution without retraining. Medical image analysis 86, 102789.
- [8] Brett, M., Johnsrude, I.S., Owen, A.M., 2002. The problem of functional localization in the human brain. Nature reviews neuroscience 3, 243-249.
- [9] Cardoso, M.J., Li, W., Brown, R., Ma, N., Kerfoot, E., Wang, Y., Murrey, B., Myronenko, A., Zhao, C., Yang, D., et al., 2022. Monai: An

- open-source framework for deep learning in healthcare. arXiv preprint arXiv:2211.02701 .
- [10] Chen, S., Ma, K., Zheng, Y., 2019. Med3d: Transfer learning for 3d medical image analysis. arXiv preprint arXiv:1904.00625 .
- [11] Chintapalli, S.S., Wang, R., Yang, Z., Tassopoulou, V., Yu, F., Bashyam, V., Erus, G., Chaudhari, P., Shou, H., Davatzikos, C., 2024. Generative models of mri-derived neuroimaging features and associated dataset of 18,000 samples. Scientific Data 11, 1330.
- [12] Cho, H., Wei, Z., Lee, S., Dan, T., Wu, G., Kim, W.H., 2025. Conditional diffusion with ordinal regression: Longitudinal data generation for neurodegenerative disease studies, in: The Thirteenth International Conference on Learning Representations.
- [13] Dice, L.R., 1945. Measures of the amount of ecologic association between species. Ecology 26, 297-302.
- [14] Eskicioglu, A.M., Fisher, P.S., 2002. Image quality measures and their performance. IEEE Transactions on communications 43, 2959-2965.
- [15] Fu, J., Zheng, Y., Dey, N., Ferreira, D., Moreno, R., 2025. Synthesizing individualized aging brains in health and disease with generative models and parallel transport. Medical Image Analysis , 103669.
- [16] Gao, Z., Zhu, W., Li, Y., Ye, W., Chen, X., Zhou, S., Li, X., Li, X., Yu, Y., Initiative, A.D.N., 2024. Identification and cognitive function prediction of alzheimer's disease based on multivariate pattern analysis of hippocampal volumes. Journal of Alzheimer's Disease 102, 1111-1120.
- [17] Grenander, U., Miller, M.I., 1998. Computational anatomy: An emerging discipline. Quarterly of applied mathematics 56, 617-694.
- [18] Ho, J., Jain, A., Abbeel, P., 2020. Denoising diffusion probabilistic models. Advances in neural information processing systems 33, 68406851.
- [19] Jayakumar, N., Gadila, S.R., Hossain, T., Ji, Y., Zhang, M., 2024. Tpie: Topology-preserved image editing with text instructions. arXiv preprint arXiv:2411.16714 .

- [20] Jayakumar, N., Hossain, T., Zhang, M., 2023. Sadir: shape-aware diffusion models for 3d image reconstruction, in: International workshop on shape in medical imaging, Springer. pp. 287-300.
- [21] Jia, X., Bartlett, J., Chen, W., Song, S., Zhang, T., Cheng, X., Lu, W., Qiu, Z., Duan, J., 2023. Fourier-net: Fast image registration with band-limited deformation, in: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 1015-1023.
- [22] Jones, D., Lowe, V., Graff-Radford, J., Botha, H., Barnard, L., Wiepert, D., Murphy, M.C., Murray, M., Senjem, M., Gunter, J., et al., 2022. A computational model of neurodegeneration in alzheimer's disease. Nature communications 13, 1643.
- [23] Joshi, S., Davis, B., Jomier, M., Gerig, G., 2004. Unbiased diffeomorphic atlas construction for computational anatomy. NeuroImage 23, S151S160.
- [24] Joshi, S.C., Miller, M.I., 2000. Landmark matching via large deformation diffeomorphisms. IEEE transactions on image processing 9, 13571370.
- [25] Jung, E., Luna, M., Park, S.H., 2023. Conditional gan with 3d discriminator for mri generation of alzheimer's disease progression. Pattern Recognition 133, 109061.
- [26] Kim, B., Ye, J.C., 2022. Diffusion deformable model for 4d temporal medical image generation, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 539-548.
- [27] Kim, J., Yoon, H., Park, G., Kim, K., Yang, E., 2024. Data-efficient unsupervised interpolation without any intermediate frame for 4d medical images, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11353-11364.
- [28] LaMontagne, P.J., Benzinger, T.L., Morris, J.C., Keefe, S., Hornbeck, R., Xiong, C., Grant, E., Hassenstab, J., Moulder, K., Vlassenko, A.G., Raichle, M.E., Cruchaga, C., Marcus, D., 2019a. Oasis-3: Longitudinal neuroimaging, clinical, and cognitive dataset for normal aging and alzheimer disease. medRxiv .

- [29] LaMontagne, P.J., Benzinger, T.L., Morris, J.C., Keefe, S., Hornbeck, R., Xiong, C., Grant, E., Hassenstab, J., Moulder, K., Vlassenko, A.G., et al., 2019b. Oasis-3: longitudinal neuroimaging, clinical, and cognitive dataset for normal aging and alzheimer disease. medrxiv , 2019-12.
- [30] Larsen, A.B.L., Sønderby, S.K., Larochelle, H., Winther, O., 2016. Autoencoding beyond pixels using a learned similarity metric, in: International conference on machine learning, PMLR. pp. 1558-1566.
- [31] Liang, H., Yin, Y., Xu, D., Liang, H., Wang, Z., Plataniotis, K.N., Zhao, Y., Wei, Y., 2024. Diffusion4d: Fast spatial-temporal consistent 4d generation via video diffusion models. arXiv preprint arXiv:2405.16645 .
- [32] Litrico, M., Guarnera, F., Giuffrida, M.V., Ravì, D., Battiato, S., 2024. Tadm: Temporally-aware diffusion model for neurodegenerative progression on brain mri, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 444-453.
- [33] Liu, C., Yuan, X., Yu, Z., Wang, Y., 2024. Texdc: Text-driven diseaseaware 4d cardiac cine mri images generation, in: Proceedings of the Asian Conference on Computer Vision, pp. 3005-3021.
- [34] Ma, X., Wang, Y., Jia, G., Chen, X., Liu, Z., Li, Y.F., Chen, C., Qiao, Y., 2024. Latte: Latent diffusion transformer for video generation. CoRR .
- [35] Miller, M.I., 2004. Computational anatomy: shape, growth, and atrophy comparison via diffeomorphisms. NeuroImage 23, S19-S33.
- [36] Mofrad, S.A., Lundervold, A., Lundervold, A.S., Initiative, A.D.N., et al., 2021. A predictive framework based on brain volume trajectories enabling early detection of alzheimer's disease. Computerized Medical Imaging and Graphics 90, 101910.
- [37] Mueller, S.G., Weiner, M.W., Thal, L.J., Petersen, R.C., Jack, C., Jagust, W., Trojanowski, J.Q., Toga, A.W., Beckett, L., 2005. The alzheimer's disease neuroimaging initiative. Neuroimaging Clinics 15, 869-877.

- [38] Nichol, A.Q., Dhariwal, P., 2021. Improved denoising diffusion probabilistic models, in: International conference on machine learning, PMLR. pp. 8162-8171.
- [39] Ortner, M., Drost, R., Heddderich, D., Goldhardt, O., MüllerSarnowski, F., Diehl-Schmid, J., Förstl, H., Yakushev, I., Grimmer, T., 2019. Amyloid pet, fdg-pet or mri?-the power of different imaging biomarkers to detect progression of early alzheimer's disease. BMC neurology 19, 264.
- [40] Peebles, W., Xie, S., 2023. Scalable diffusion models with transformers, in: Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195-4205.
- [41] Peng, L., Lin, L., Lin, Y., Chen, Y.w., Mo, Z., Vlasova, R.M., Kim, S.H., Evans, A.C., Dager, S.R., Estes, A.M., et al., 2021. Longitudinal prediction of infant mr images with multi-contrast perceptual adversarial learning. Frontiers in neuroscience 15, 653213.
- [42] Petersen, R.C., Aisen, P.S., Beckett, L.A., Donohue, M.C., Gamst, A.C., Harvey, D.J., Jack Jr, C., Jagust, W.J., Shaw, L.M., Toga, A.W., et al., 2010. Alzheimer's disease neuroimaging initiative (adni) clinical characterization. Neurology 74, 201-209.
- [43] Pombo, G., Gray, R., Cardoso, M.J., Ourselin, S., Rees, G., Ashburner, J., Nachev, P., 2023. Equitable modelling of brain imaging by counterfactual augmentation with morphologically constrained 3d deep generative models. Medical Image Analysis 84, 102723.
- [44] Puglisi, L., Alexander, D.C., Ravì, D., 2024. Enhancing spatiotemporal disease progression models via latent diffusion and prior knowledge, in: International Conference on Medical Image Computing and ComputerAssisted Intervention, Springer. pp. 173-183.
- [45] Rahman, A.U., Ali, S., Saqia, B., Halim, Z., Al-Khasawneh, M., AlHammadi, D.A., Khan, M.Z., Ullah, I., Alharbi, M., 2025. Alzheimer's disease prediction using 3d-cnns: Intelligent processing of neuroimaging data. SLAS technology 32, 100265.
- [46] Ravi, D., Alexander, D.C., Oxtoby, N.P., Initiative, A.D.N., 2019. Degenerative adversarial neuroimage nets: generating images that mimic

- disease progression, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 164-172.
- [47] Reuter, M., Schmansky, N.J., Rosas, H.D., Fischl, B., 2012. Withinsubject template estimation for unbiased longitudinal image analysis. Neuroimage 61, 1402-1418.
- [48] Ronneberger, O., Fischer, P., Brox, T., 2015. U-net: Convolutional networks for biomedical image segmentation, in: International Conference on Medical image computing and computer-assisted intervention, Springer. pp. 234-241.
- [49] Singh, S., Dewangan, S., Krishna, G.S., Tyagi, V., Reddy, S., Medi, P.R., 2022. Video vision transformers for violence detection. arXiv preprint arXiv:2209.03561 .
- [50] Tahami Monfared, A.A., Byrnes, M.J., White, L.A., Zhang, Q., 2022. Alzheimer's disease: epidemiology and clinical progression. Neurology and therapy 11, 553-569.
- [51] Thompson, P., Apostolova, L., 2007. Computational anatomical methods as applied to ageing and dementia. The British journal of radiology 80, S78-S91.
- [52] Tustison, N.J., Cook, P.A., Holbrook, A.J., Johnson, H.J., Muschelli, J., Devenyi, G.A., Duda, J.T., Das, S.R., Cullen, N.C., Gillen, D.L., Yassa, M.A., Stone, J.R., Gee, J.C., Avants, B.B., 2021. The ANTsX ecosystem for quantitative biological and medical imaging. Scientific Reports 11, 9068. URL: https://doi.org/10.1038/s41598-021-87564-6 , doi: 10. 1038/s41598-021-87564-6 .
- [53] Vercauteren, T., Pennec, X., Perchant, A., Ayache, N., 2008. Symmetric log-domain diffeomorphic registration: A demons-based approach, in: International conference on medical image computing and computerassisted intervention, Springer. pp. 754-761.
- [54] Wang, J., Zhang, M., 2020. Deepflash: An efficient network for learningbased medical image registration, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4444-4452.

- [55] Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P., 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13, 600-612.
- [56] Watson, D., Saxena, S., Li, L., Tagliasacchi, A., Fleet, D.J., 2025. Controlling space and time with diffusion models, in: The Thirteenth International Conference on Learning Representations.
- [57] Wells III, W.M., Viola, P., Atsumi, H., Nakajima, S., Kikinis, R., 1996. Multi-modal volume registration by maximization of mutual information. Medical image analysis 1, 35-51.
- [58] Wu, N., Jayakumar, N., Xing, J., Zhang, M., 2025. Igg: Image generation informed by geodesic dynamics in deformation spaces, in: International Conference on Information Processing in Medical Imaging, Springer. pp. 232-246.
- [59] Wu, N., Zhang, M., 2023. Neurepdiff: Neural operators to predict geodesics in deformation spaces, in: International Conference on Information Processing in Medical Imaging, Springer. pp. 588-600.
- [60] Yoon, J.S., Zhang, C., Suk, H.I., Guo, J., Li, X., 2023. Sadm: Sequenceaware diffusion model for longitudinal medical image generation, in: International Conference on Information Processing in Medical Imaging, Springer. pp. 388-400.
- [61] Yuan, C., Duan, J., Xu, K., Tustison, N.J., Hubbard, R.A., Linn, K.A., 2024. Remind: recovery of missing neuroimaging using diffusion models with application to alzheimer's disease. Imaging Neuroscience 2, 1-14.
- [62] Zhang, H., Chen, X., Wang, Y., Liu, X., Wang, Y., Qiao, Y., 2024. 4diffusion: Multi-view video diffusion model for 4d generation. Advances in Neural Information Processing Systems 37, 15272-15295.
- [63] Zhao, J., Zhang, Y., Li, X., Chen, H., Shuai, B., Xu, M., Liu, C., Kundu, K., Xiong, Y., Modolo, D., et al., 2022. Tuber: Tubelet transformer for video action detection, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13598-13607.

- [64] Zhu, Z., Tao, T., Tao, Y., Deng, H., Cai, X., Wu, G., Wang, K., Tang, H., Zhu, L., Gu, Z., et al., 2024. Loci-diffcom: Longitudinal consistencyinformed diffusion model for 3d infant brain image completion, in: International Conference on Medical Image Computing and ComputerAssisted Intervention, Springer. pp. 249-258.