# Relaxation-Informed Training of Neural Network Surrogate Models 

Calvin Tsay 

Department of Computing, Imperial College London, South Kensington, SW7 2AZ, United Kingdom. 

Contributing authors: c.tsay@imperial.ac.uk; 

## **Abstract** 

ReLU neural networks trained as surrogate models can be embedded exactly in mixed-integer linear programs (MILPs), enabling global optimization over the learned function. The tractability of the resulting MILP depends on structural properties of the network, i.e., the number of binary variables in associated formulations and the tightness of the continuous LP relaxation. These properties are determined during training, yet standard training objectives (prediction loss with classical weight regularization) offer no mechanism to directly control them. This work studies training regularizers that directly target downstream MILP tractability. Specifically, we propose simple bound-based regularizers that penalize the big-M constants of MILP formulations and/or the number of unstable neurons. Moreover, we introduce an LP relaxation gap regularizer that explicitly penalizes the per-sample gap of the continuous relaxation at training points. We derive its associated gradient and provide an implementation from LP dual variables without custom automatic differentiation tools. We show that combining the above regularizers can approximate the full total derivative of the LP gap with respect to the network parameters, capturing both direct and indirect sensitivities. Experiments on non-convex benchmark functions and a two-stage stochastic programming problem with quantile neural network surrogates demonstrate that the proposed regularizers can reduce MILP solve times by up to four orders of magnitude relative to an unregularized baseline, while maintaining competitive surrogate model accuracy. 

## **1 Introduction** 

Neural network surrogate models have become a popular tool in mathematical optimization, enabling complicated or unknown functions to be replaced by trained parametric approximations that can then be embedded in optimization formulations [1–3]. Feedforward neural networks with rectified linear unit (ReLU) activations are particularly attractive for this purpose: the piecewiselinear structure of the ReLU function (and thus the combined network) allows the trained model to be encoded exactly in a mixed-integer linear program (MILP), enabling branch-and-bound global optimization [4–6]. Machine learning applications include NN verification/certification [7–10], counterfactual explanations [11, 12], reinforcement learning [13, 14], and model compression [15, 16]. Optimization applications include optimizing over black-box objectives [17–19], constraint learning [20, 21], and stochastic programming [22–24]. Toolkits such as JANOS [25], OMLT [26], and PySCIPOpt-ML [27] have helped popularize this approach across a range of engineering domains, including process design, energy systems, and planning [28–31]. We refer the reader to Huchette et al. [5] for a comprehensive overview of the intersection between ReLU neural networks and MILP. 

1 

For a given trained network, the complexity and tractability of associated MILP formulations depends is linked to its structural properties. In the standard big-M formulation [32–34], each hidden neuron with unknown activation state is encoded by introducing a binary variable. The number of these variables effectively dictates the combinatorial search space of the branch-andbound search. Equally important is the strength of the continuous LP relaxation, obtained by relaxing each binary variable to a continuous variable in [0 _,_ 1]. The LP provides a bound on the MILP optimum at every node of the branch-and-bound tree, and a loose relaxation forces the solver to explore more nodes before certifying global optimality. Both the number of binary variables and the LP relaxation gap remain important even in more sophisticated formulations [35, 36], and can be controlled by the variable bounds, which in turn depend on the network parameters _θ_ (obtained during training). Tight bounds can reduce big-M constants, stabilize neurons, and strengthen the LP relaxation, effectively resulting in more manageable MILP problems [37–39]. However, computational approaches for tightening bounds often scale poorly with network size (e.g., requiring solving optimization problems for each neuron), and these strategies are applied _after training_ , with no mechanism to guide the network towards tractable structures during model training. 

Standard neural network training minimizes a prediction loss and may include classical weight regularization such as _ℓ_ 1 or _ℓ_ 2 penalties. While inclusion of these regularizers can improve downstream MILP tractability [6], neither term in the training objective directly accounts for the downstream application(s) of the resulting surrogate model. A model trained to high accuracy may have many unstable neurons or loose LP relaxation bounds, making subsequent MILP-based optimization intractable. This decoupling of training and optimization is an important, but largely unexplored, source of inefficiency in the surrogate modeling pipeline. 

This paper proposes a family of regularization terms that can be added to the standard training loss to explicitly target the factors governing MILP tractability. The key observation is that the pre-activation bounds are often (sub)differentiable functions of the network parameters _θ_ and can therefore be incorporated into regularizers for gradient-based training. We derive the form and gradient of each proposed regularizer, and we establish formal relationships between them and the full derivative of the LP relaxation gap with respect to _θ_ . Furthermore, we show that the gradient of the LP relaxation itself can be computed efficiently using sensitivity of parametric linear programs [40, 41] and incorporated into regularizers. 

The main contributions of this paper are as follows. 

1. We derive two bound propagation-based regularizers: _R_ BW (bound-width) and _R_ SN (stableneuron). We provide closed-form expressions for their subgradients via automatic differentiation through the bound propagation. The bound widths prescribe the big-M constants of the MILP, and their recursive structure through the network depth can be exploited for gradient computations at the cost of a single additional forward pass per training step. 

2. We introduce the LP relaxation gap regularizer _R_ LP, which directly penalizes the per-sample continuous relaxation of the MILP at each training point. We derive and express its gradient in terms of the LP dual variables and the standard backpropagation gradient. An exact implementation via a straight-through estimator avoids the need for custom differentiation tools. 

3. We establish a gradient decomposition (Proposition 3) showing that the combined regularizer _R_ BW + _R_ LP approximates the full total derivative of the LP gap with respect to _θ_ , capturing both the direct sensitivity through the constraint right-hand sides and the indirect sensitivity through the big-M constants via IBP. 

4. We evaluate all regularizers on benchmark surrogate functions and on large-scale stochastic programming tasks, measuring their effect on the number of unstable neurons, LP relaxation gap, MILP node count, and solve time across a range of network architectures and regularization strengths. 

The remainder of the paper is organised as follows. We first contextualize our contribution in relation to existing work in Section 1.1. Section 2 then introduces neural network models, the training objective, the big-M MILP formulation, and bound propagation. Sections 3 and 4 derive the bound-based regularizers and the LP gap regularizer, respectively. Section 5 briefly analyzes the combined regularizer and its relation to the total derivative. Computational experiments are reported in Section 6, and conclusions are drawn in Section 7. 

2 

## **1.1 Relation to existing work** 

The idea of training for downstream tractability MILP has parallels in the adversarial machine learning literature, where networks trained for certified robustness have been shown to exhibit fewer unstable neurons and tighter bounded output domains than their unregularized counterparts [42– 45]. Here, ‘robustness’ refers to provable immunity to adversarial perturbations within a norm ball around each input at inference [8] or training [9, 46] time. More generally, the motivation of training a model with its downstream optimization use in mind is shared with the _decision-focused learning_ literature [47], e.g., ‘smart predict-then-optimize’ [48] and task-based learning [49]. 

While the regularizers developed in Sections 3–4 share some technical components with the verification literature (notably differentiable bounds and penalties that target neuron stability), our motivation, formulation, and context differ in several important respects detailed below. On the other hand, while we consider a similar application as the decision-focused learning literature, the proposed regularization strategies purely target downstream MILP tractability, rather than improving the quality of decisions. 

**Application domain.** Certified training methods target _adversarial robustness_ of classifiers, typically defined as classification accuracy under worst-case perturbation in an _ϵ_ -ball around each test image. Note that this problem can often be solved without finding the worst-case perturbation (i.e., completely solving a MILP): it requires only a successful worst-case perturbation or safe bound [8, 36]. On the other hand, the proposed methods primarily target _surrogate models for optimization_ . In this setting, the network approximates a continuous function over a known box domain _X_ , and the downstream task is to solve a MILP over/involving the trained surrogate [4]. In contrast to the verification literature, the relevant metrics therefore include optimization properties such as LP relaxation gap and MILP solve time. 

**Training for certification.** In certified training, intermediate variable bounds are used to compute an output-level bound, rather than as targets themselves [43, 45]. Nevertheless, in MILP intermediate bounds directly influence the tightness of the overall formulation, and we therefore quantify and regularize with total bound width across all hidden neurons (an objective with no analog in the certified training literature). Furthermore, we propose regularizers directly targeting the relaxation gap, which is entirely specific to the MILP surrogate setting. Most similar to the present work methodologically are regularizers targeting neuron stability, which can accelerate verification of classification networks [42, 50]. The present work studies stability regularizers for the MILP setting, where the network is embedded as an objective or constraint in a downstream optimization problem rather than verified for a fixed property. Finally, we note the certified training literature typically studies multi-class classifiers with cross-entropy loss and specification-based margins. Our setting involves scalar regression surrogates trained with MSE loss, where there is no class margin to certify, and the relevant relaxation is the continuous relaxation of the surrogate MILP, not the convex outer adversarial polytope. This definition of model quality gives a fundamentally different perspective to the tradeoff between accuracy and tractability. 

**Decision-focused learning.** Many decision-focused learning methods modify the training loss of the surrogate model completely [48, 51], e.g., minimizing task loss or regret rather than prediction error alone. These modifications aim to improve the quality of decisions produced in downstream applications. Most similar to the present work in motivation are regularizers targeting gradients of the learned surrogate model, with the purpose of improving _optimization performance_ in downstream (gradient-based) optimization [52–54]. The present paper takes an orthogonal approach within this broader family. Here, our objective is not to improve the quality of the solution found (which is also an important problem), but to reduce the computational cost of finding it via MILP. In other words, decision-focused learning assumes the downstream problem is solvable and asks how to improve solution quality; this paper assumes a useful surrogate and asks how to make it tractable. 

3 

## **2 Background** 

## **2.1 Neural network notation** 

We consider feed-forward neural networks (NNs), which are directed acyclic graphs comprising nodes/neurons structured into _L_ hidden layers. At each layer _l_ = 1 _, ..., L_ +1, the NN contains nodes that receive the outputs of nodes in the preceding layer ( _l −_ 1) as inputs. Each node then computes a weighted sum of its inputs (known as the preactivation), and applies a nonlinear activation function to this computed term. While many options for activation function are available, we focus on the ReLU activation function 

_y_ = max(0 _, w[T] x_ + _b_ ) _,_ 

which is amenable to mixed-integer linear programming (MILP) formulations, given its piecewiselinear form [5]. 

Mathematically, we denote a feedforward neural network model as _fθ_ : R _[n]_[0] _→_ R with _L_ hidden layers, ReLU activations, and a linear output layer. Denote the weight matrix and bias vector at layer _ℓ_ by _W_[(] _[ℓ]_[)] _∈_ R _[n][ℓ][×][n][ℓ][−]_[1] and _b_[(] _[ℓ]_[)] _∈_ R _[n][ℓ]_ , respectively, for _ℓ_ = 1 _, . . . , L_ +1. The collective parameter vector for the model is expressed as _θ_ = _{_ ( _W_[(] _[ℓ]_[)] _, b_[(] _[ℓ]_[)] ) _}[L] ℓ_ =1[+1][. Finding the values for] _[ θ]_[, e.g.,] to best fit a given dataset, is referred to as _training_ the NN model. 

For an input _x ∈_ R _[n]_[0] , evaluation of the neural network _fθ_ ( _x_ ) is referred to as a _forward pass_ . In particular, the forward pass computes: 

**==> picture [379 x 39] intentionally omitted <==**

ˆ with _x_[(0)] = _x_ being the input and _fθ_ ( _x_ ) = _z_[(] _[L]_[+1)] being the scalar output (note the lack of nonlinear activation at the output layer). We consider the input domain as a box _X_ = _{x_ : _x_[lb] _≤ x ≤ x_[ub] _}_ , noting that in general, other constraints on _x_ could be added in later steps. 

## **2.2 Training the neural network** 

The neural network parameters are trained using training data _{_ ( _xi, yi_ ) _}[N] i_ =1[,][e.g.,][samples][drawn] from a function we wish to approximate _g_ : _X →_ R. Generally, we find values of _θ_ by minimizing empirical loss over the training data. For regression tasks, this is often taken using the mean squared error (MSE): 

**==> picture [129 x 30] intentionally omitted <==**

To avoid overfitting (or otherwise guide the training process), regularization terms can be appended to the loss function: 

_L_ ( _θ_ ) = _L_ MSE + _λ R_ ( _θ_ ) _,_ (3) 

where _R_ ( _θ_ ) is a regularization term and _λ >_ 0 controls the trade-off between accuracy and regularization. We refer the reader to Goodfellow et al. [55] for a comprehensive overview of this training paradigm. 

Training the neural network is typically performed using gradient-based optimization methods, requiring the computation of _∇θL_ , e.g., using back-propagation. We observe that gradients are therefore required for both terms in (3), _∇θL_ MSE and _∇θR_ ( _θ_ ). This study precisely aims to introduce regularizers _R_ ( _θ_ ) targeting the MILP tractability of the resulting NN surrogate. In Section 3, we explicitly derive the form and gradient of each regularizer _R_ we consider. 

## **2.3 Mixed-integer optimization formulation** 

In contrast to the training of NNs (where the parameters _θ_ are decision variables), optimization over a NN surrogate seeks to compute extreme cases for an _already trained_ model. In other words, the parameters _θ_ are fixed, and we optimize over _fθ_ as a fixed function (or embed it within constraints in a larger problem). This step therefore requires formulating _fθ_ over _X_ as optimization constraints. 

4 

In MILP formulations, each ReLU unit is commonly modelled with a binary variable _a_[(] _j[ℓ]_[)] _∈ {_ 0 _,_ 1 _}_ indicating whether neuron _j_ at layer _ℓ_ is active ( _a_[(] _j[ℓ]_[)] = 1) or inactive ( _a_[(] _j[ℓ]_[)] = 0). Let ˆ _L_[(] _j[ℓ]_[)] _≤ zj_[(] _[ℓ]_[)] _≤ Uj_[(] _[ℓ]_[)] be valid pre-activation bounds. The big-M formulation of _x_[(] _j[ℓ]_[)] = ReLU( _zj_[(] _[ℓ]_[)][)] is [32–34]: 

**==> picture [273 x 71] intentionally omitted <==**

for each hidden neuron ( _ℓ, j_ ), _ℓ_ = 1 _, . . . , L_ . The tractability of this formulation hinges on the values used for the bounds _L_[(] _j[ℓ]_[)] and _Uj_[(] _[ℓ]_[)][,][i.e.,][the][big-M][coefficients][[][38][].][Notice][that][smaller][values][for] these coefficients yield tighter constraints (4c)–(4d). Ideally, these bounds are taken to be as tight as possible such that they remain valid, with _zj_[(] _[ℓ]_[)] _∈_ [ _L_[(] _j[ℓ]_[)] _[, U] j_[ (] _[ℓ]_[)][].][We][note][that][several][interesting] alternatives to the big-M formulation have been proposed [35, 36], but it remains popular given its simplicity. 

_Remark 1_ (Strength of continuous relaxation) MILP is often solved using branch-and-bound algorithms, which leverage a cheaper, continuous relaxation to bound the objective at each node of the search tree. The solver then explores the domain over decision variables by ‘branching’ until the gap between the best feasible objective value found and the tightest relaxation found falls below a given tolerance. A tighter, or _stronger_ , relaxation can reduce this search tree considerably. The continuous relaxation of (4) is obtained by relaxing _a_[(] _j[ℓ]_[)] _∈{_ 0 _,_ 1 _}_ to _a_[(] _j[ℓ]_[)] _∈_ [0 _,_ 1], yielding a linear program (LP) whose optimal value bounds the MILP optimum. The _LP relaxation gap_ is the difference between this LP bound and the MILP optimum. Since the LP relaxation of (4c)–(4d) tightens as _|L_[(] _j[ℓ]_[)] _[|]_[ and] _[ U] j_[(] _[ℓ]_[)] decrease, the choice of bounds is a primary determinant of relaxation strength, and therefore MILP solve efficiency. Figure 1 illustrates the predictions and continuous relaxations for several NN models. 

**==> picture [384 x 251] intentionally omitted <==**

**----- Start of picture text -----**<br>
None L1 L2<br>val MSE=0.0028 val MSE=0.0038 val MSE=0.0032<br>0 0 0<br>3 f 3 f 3<br>6 6 6<br>9 9 9<br>1.5 1.5 1.5<br>1.5 0.0 1.5 0.0 1.5 0.0<br>0.0 0.0 0.0<br>1.5 1.5 1.5 1.5 1.5 1.5<br>BoundWidth StableNeuron LP gap<br>val MSE=0.0020 val MSE=0.0032 val MSE=0.0032<br>0 0 0<br>3 f 3 f 3<br>6 6 6<br>9 9 9<br>1.5 1.5 1.5<br>1.5 0.0 1.5 0.0 1.5 0.0<br>0.0 0.0 0.0<br>1.5 1.5 1.5 1.5 1.5 1.5<br>f (x) [NN output] V [min] ( , x) [LP relaxation]<br>x1 x1 x1<br>x1 x1 x1<br>x2 x2 x2<br>x2 x2 x2<br>**----- End of picture text -----**<br>


**Fig. 1** Predictions and LP lower bounds for NN models with _{_ 32,32 _}_ hidden layers trained on the scaled Peaks function with two inputs ( _x_ 1, _x_ 2), with output _f_ ( _x_ ). Different regularizers are applied during training, with weights chosen to maintain validation MSE of similar scale. 

5 

**Definition 1** (Neuron stability) A neuron is _stable active_ if _L_[(] _j[ℓ]_[)] _≥_ 0 (the ReLU never turns off), in which case _x_ ˆ[(] _j[ℓ]_[)] = _zj_[(] _[ℓ]_[)] and _a_[(] _j[ℓ]_[)] = 1 can be fixed. Likewise, a neuron is _stable inactive_ if _Uj_[(] _[ℓ]_[)] _≤_ 0 (ˆ _x_[(] _j[ℓ]_[)] = 0, _a_[(] _j[ℓ]_[)] = 0). A neuron is said to be _stable_ if it is either stable active or inactive; for stable neurons, the value of _a_[(] _j[ℓ]_[)] is fixed, and no binary variable is required. The set of _unstable_ neurons requiring a binary variable to formulate using (4) is therefore defined: 

**==> picture [277 x 15] intentionally omitted <==**

The number of (unfixed) binary variables in the MILP resulting from applying (4) to all neurons equals _|U|_ . 

## **2.4 Obtaining and tightening bounds** 

Given the input domain _X_ = [ _x_[lb] _, x_[ub] ], simple valid pre-activation bounds _L_[(] _j[ℓ]_[)] _[, U] j_[ (] _[ℓ]_[)] can be computed by applying interval arithmetic layer by layer. This is also referred to as interval bound propagation, or IBP. Let[ˆ] _l_[(] _[ℓ]_[)] _,_ ˆ _u_[(] _[ℓ]_[)] denote the post-activation (post-ReLU) bounds at layer _ℓ_ , with ˆ the input layer defined by given bounds[ˆ] _l_[(0)] = _x_[lb] , _u_[(0)] = _x_[ub] . 

Valid pre-activation bounds for a layer can be computed using interval arithmetic: 

**==> picture [309 x 16] intentionally omitted <==**

**==> picture [310 x 15] intentionally omitted <==**

where the operators [ _v_ ][+] = max( _v,_ 0) and [ _v_ ] _[−]_ = min( _v,_ 0) are applied element-wise. The ReLU function output is nonnegative, and the post-ReLU bounds for hidden layers can be further tightened: 

**==> picture [309 x 16] intentionally omitted <==**

Interval arithmetic methods do not provide the tightest valid bounds in general, as dependencies between the input nodes are ignored. Propagating the resulting over-approximated bounds through the layers of a neural network leads to increasingly large over-approximations; in other words, propagating weak bounds through layers results in a model with significantly weaker continuous relaxation. Tighter bounds could potentially be obtained using optimization-based bound tightening (OBBT), i.e., solving an optimization problem with the objective set to minimize/maximize a particular pre-activation term to provide its bounds [37, 39]. To reduce the computational cost of OBBT problems, OBBT can be performed using relaxations or problem-based decompositions [38]. In contrast to interval arithmetic, bounds obtained using OBBT can incorporate variable dependencies. In this work, we focus on IBP bounds given their popularity. 

A key observation is that the operations in the IBP recursion (6)–(7) are compositions of affine maps and element-wise max( _·,_ 0), similar to the ReLU forward pass. The IBP process is therefore subdifferentiable with respect to the NN parameters _θ_ . The subgradients are well-defined almost everywhere and can be computed by automatic differentiation (e.g., in PyTorch or JAX). This property enables the IBP bounds to be incorporated directly into gradient-based training as differentiable regularization terms, as developed in the following sections. 

## **3 MILP-informed regularization** 

We now introduce regularizers _R_ ( _θ_ ) for use in the training objective (3) that target MILP tractability. For instance, Figure 1 shows the predictions and LP lower bounds for trained NN models on the Peaks function. We begin with standard shrinkage penalties, which serve as baselines, and then present two IBP-based regularizers that directly target the mechanisms governing MILP difficulty. 

## **3.1 Shrinkage regularization** 

Shrinkage regularization is a strategy that aims to improve model generalizability and reduce overfitting by penalizing large parameter values, effectively ‘shrinking’ them towards zero. Shrinking the parameter values manages the bias-variance trade-off by introducing a small amount of bias to (significantly) reduce model variance. Common methods here include Ridge (L2) and Lasso (L1) 

6 

regression. These methods may produce models with tighter bounds, as shrinking the values of _Wj_[(] _[ℓ]_[)] can directly improve the bounds obtained using (6). Plate et al. [6] find that increasing shrinkage regularization can produce neural networks with a lower number of linear regions, improving performance in downstream MILP. Manng˚ard et al. [56] study methods using these regularizers to explicitly induce weight sparsity. 

## _**L1 regularization**_ 

**==> picture [288 x 30] intentionally omitted <==**

where _∥·∥_ 1 denotes the entry-wise _ℓ_ 1 norm. This promotes weight sparsity and indirectly reduces IBP bound widths, as _Uj_[(] _[ℓ]_[)] _−L_[(] _j[ℓ]_[)] scales with _∥Wj_[(] _[ℓ]_[)] _∥_ 1 (see (11) below). However, it does not account for the layered, recursive structure of bound propagation and treats all parameters uniformly regardless of their role in the MILP formulation. 

## _**L2 regularization**_ 

**==> picture [289 x 31] intentionally omitted <==**

where _∥·∥F_ denotes the Frobenius norm. Again, this can indirectly shrink big-M values but does not directly target bound widths or neuron stability. 

## **3.2 Bound-width regularization** 

Define the _width_ of the IBP pre-activation bound at neuron ( _ℓ, j_ ) as ∆[(] _j[ℓ]_[)] = _Uj_[(] _[ℓ]_[)] _− L_[(] _j[ℓ]_[)][.][We] introduce a bound-width regularizer, which simply penalizes the mean bound width obtained across all hidden neurons: 

**==> picture [347 x 31] intentionally omitted <==**

Subtracting (6a) from (6b) gives the bound width at layer _ℓ_ as: 

**==> picture [351 x 16] intentionally omitted <==**

where _|Wj_[(] _[ℓ]_[)] _|_ denotes the element-wise absolute value of the _j_ -th row, and ∆[(] post _[ℓ][−]_[1)] is the vector of post-ReLU bound widths at layer _ℓ−_ 1. The post-ReLU bound widths satisfy ∆[(] post _[ℓ]_[)] _,j[≤]_[∆][(] _j[ℓ]_[)] by (7), so the layer-wise recursion (11) shows that bound widths compound multiplicatively through the network depth. 

## _**Gradient.**_ 

Since the post-ReLU bound widths ∆[(] post _[ℓ][−]_[1)] depend in turn on earlier layers through the recursion (6)–(7), the total gradient _∂R_ BW _/∂θ_ captures the full chain of IBP bound propagation through the NN. In our experiments we directly implement (11) in PyTorch, and its gradient is computed automatically by PyTorch’s reverse-mode automatic differentiation. Note that this requires implementing the (subdifferentiable) ‘IBP forward pass,’ i.e., propagating bounds through the layers of the neural network using (6)–(7). The computational cost is one IBP forward pass per training step. Note that more advanced OBBT propagation schemes may be incorporated as regularizers following the differentiable optimization procedures in Section 4. 

## _**Interpretation.**_ 

Including _R_ BW as a regularization term in (3) explicitly penalizes the magnitude of big-M constants in the downstream MILP formulation. For an unstable neuron, we observe that _|L_[(] _j[ℓ]_[)] _[|][≤]_[∆][(] _j[ℓ]_[)] 

7 

and _Uj_[(] _[ℓ]_[)] _≤_ ∆[(] _j[ℓ]_[)][.][Reducing][∆][(] _j[ℓ]_[)] therefore simultaneously shrinks both big-M values in (4c)–(4d), tightening the LP relaxation. When the bounds are both positive or negative, the neuron is stable, and no binary variable is required (Definition 1). 

_Remark 2_ An alternative view of _R_ BW is the direct penalization of the magnitude of big-M constants, i. e., the product of weight magnitudes and input bound ranges. In this case, this is exactly represented by (11), as IBP bound widths (composed recursively through the layers) are precisely the big-M constants. Nevertheless, in more sophisticated MILP formulations without big-M constants, corresponding regularization terms can still be derived based on the width of the involved bounds. 

## **3.3 Stability regularization** 

While the inclusion of _R_ BW can strengthen the continuous LP relaxation by tightening bounds involved, the _combinatorial_ difficulty of the MILP is governed by the number of binary variables, and therefore unstable neurons _|U|_ in (5). Both the relaxation tightness and the number of discrete combinations in a search tree impact the efficiency of branch-and-bound algorithms. As given in Definition 1, neuron ( _ℓ, j_ ) is unstable when its pre-activation bounds straddle zero: _L_[(] _j[ℓ]_[)] _<_ 0 _< Uj_[(] _[ℓ]_[)][.] A naive approach could directly penalize the number of unstable neurons _|U|_ . 

Nevertheless, knowing bounds _L_[(] _j[ℓ]_[)] and _Uj_[(] _[ℓ]_[)] also informs us how ‘close’ a neuron is to being stable, e.g., how close the bounds are to zero. Based on this idea, we introduce a regularization term that penalizes the mean “distance to stability” to encourage stable nodes during training: 

**==> picture [319 x 32] intentionally omitted <==**

where [ _v_ ][+] = max( _v,_ 0). For a stable neuron ( _L_[(] _j[ℓ]_[)] _≥_ 0 or _Uj_[(] _[ℓ]_[)] _≤_ 0), at least one of [ _−L_[(] _j[ℓ]_[)][]][+][or] [ _Uj_[(] _[ℓ]_[)][]][+][is][zero,][so][the][contribution][to] _[R]_[SN][is][zero.][For][an][unstable][neuron,][[] _[−][L]_[(] _j[ℓ]_[)][]][+][=] _[|][L]_[(] _j[ℓ]_[)] _[|]_[and] [ _Uj_[(] _[ℓ]_[)][]][+][=] _[ U] j_[ (] _[ℓ]_[)][, and the contribution to] _[ R]_[SN][is min(] _[|][L]_[(] _j[ℓ]_[)] _[|][, U] j_[ (] _[ℓ]_[)][)] _[ >]_[ 0. In other words, the regularizer] pushes either _L_[(] _j[ℓ]_[)] upward toward zero (making the neuron stably active) or _Uj_[(] _[ℓ]_[)] downward toward zero (making it stably inactive), whichever requires the smaller change. We note that, even if this does not force the neuron to be stable, pusing one of the bounds closer to zero may still produce a tighter continuous relaxation (Figure 2). 

**Proposition 1** (Subgradient of _R_ SN) _The subgradient of_ (12) _with respect to θ is_ 

**==> picture [307 x 56] intentionally omitted <==**

_where ∂L_[(] _j[ℓ]_[)] _[/∂θ][and][∂U] j_[(] _[ℓ]_[)] _/∂θ are obtained from automatic differentiation through the IBP recursion. At the non-differentiable point |L_[(] _j[ℓ]_[)] _[|]_[ =] _[ U] j_[(] _[ℓ]_[)] _, any convex combination of the two cases is a valid subgradient._ 

In our implementation, we use the PyTorch `torch.minimum` function, which handles the subgradient at the tie point _|L_[(] _j[ℓ]_[)] _[|]_[ =] _[ U] j_[ (] _[ℓ]_[)] automatically. 

A related line of work aims to produce networks that are not only robust, but also _easy to verify exactly_ using MILP-based solvers. Xiao et al. [42] identify weight sparsity and ReLU stability as two network properties that reduce exact verification time. They employ _ℓ_ 1 regularization and small-weight pruning to promote sparsity, and introduce an alternative regularizer (termed RS loss in [42]) targeting ReLU stability. We denote this as an alternative stability regularizer: 

**==> picture [314 x 31] intentionally omitted <==**

8 

where _Uj_[(] _[ℓ]_[)] and _L_[(] _j[ℓ]_[)] are again upper and lower bounds on the pre-activation of neuron ( _ℓ, j_ ). When both bounds have the same sign (stable neuron), the product _Uj_[(] _[ℓ]_[)] _· L_[(] _j[ℓ]_[)] is positive and the penalty is small; when the bounds straddle zero (unstable neuron), the product is negative and the penalty increases. The authors found that adding this regularizer to the adversarial training objective reduces unstable neuron counts and yields considerable speedups in MILP-based verification time. 

The stability regularizer _R_ SN (12) is conceptually related to the RS Loss (14) of Xiao et al. [42]: both encourage neurons to be stably active or inactive. Nevertheless, the formulations differ practically. Our proposed regularizer _R_ SN in (12) uses min([ _−L_ ][+] _,_ [ _U_ ][+] ), to directly measures the distance to stability and has a piecewise-linear structure. On the other hand, the RS Loss in (14), denoted here as _R_ SN2, uses a smooth surrogate for sign agreement, which does not account for distance to stability. Moreover, in our setting _R_ SN and _R_ SN2 can be combined with other regularizers to target complementary aspects of MILP difficulty. 

## _**Interpretation.**_ 

The regularizers _R_ SN and _R_ BW target different mechanisms of MILP. The former _R_ BW shrinks all bound widths uniformly, improving the strength of the continuous LP relaxation. On the other hand, _R_ SN concentrates its effect on the boundary at zero, aiming to eliminate (fix) binary variables from the formulation entirely. A trained NN could have tight bounds (small ∆[(] _j[ℓ]_[)][) that still straddle] zero on many neurons, or wide bounds that happen to be one-sided (i.e., stable neurons). The two regularizers address related and complementary aspects of MILP difficulty and can be combined. 

## **4 Relaxation-informed regularization** 

The regularization methods introduced in Section 3 all may help improve the strength of MILP reformulations of a NN surrogate model, albeit indirectly. In other words, they are heuristics aimed at producing tighter bounds or reducing the number of binary variables. Figure 2 illustrates the relaxation-related properties targeted by each regularizer. In this section, we consider a direct measure of relaxation quality: the LP relaxation gap itself. 

For a given input _xi ∈X_ , the true network output is _fθ_ ( _xi_ ), which is uniquely determined by the forward pass (1)–(2). This unique solution is exactly encoded (for fixed input _xi_ ) by the MILP constraints (4) when integrality is enforced. The LP relaxation, however, admits different output values because the relaxed binary variables _a_[(] _j[ℓ]_[)] _∈_ [0 _,_ 1] allow intermediate neurons to deviate from their true ReLU outputs. 

## **4.1 Pointwise LP relaxation gap** 

As mentioned above, the continuous relaxation of (4) obtained by relaxing integral constraints yields an LP that effectively provides a bound on the MILP optimum. We now derive a regularizer using the LP relaxation gap, i.e., the difference between the LP bound and the MILP solution (which gives the true NN output _fθ_ ). For a fixed NN input _xi_ , we denote the LP relaxation value: 

**==> picture [301 x 69] intentionally omitted <==**

The _pointwise LP gap_ in the minimization direction is: 

**==> picture [302 x 13] intentionally omitted <==**

since the LP relaxation can only under-estimate the minimum, i.e., _V_[min] _≤ fθ_ ( _xi_ ). An analogous quantity _δ_[max] ( _θ, xi_ ) = _V_[max] ( _θ, xi_ ) _− fθ_ ( _xi_ ) _≥_ 0 measures the gap in the maximization direction. 

9 

**==> picture [327 x 100] intentionally omitted <==**

**----- Start of picture text -----**<br>
Reduce Bound Width Reduce Instability Reduce LP Gap<br>x ˆ [(] [ℓ] [)] x ˆ [(] [ℓ] [)] x ˆ [(] [ℓ] [)]<br>j j j<br>V [max]<br>V [max] V [max]<br>fθ fθ fθ<br>R LP<br>z [(] [ℓ] [)] z [(] [ℓ] [)] z [(] [ℓ] [)]<br>j R SN j j<br>R BW<br>L [(] j [l] [)] Uj [(] [l] [)] L [(] j [l] [)] Uj [(] [l] [)] L [(] j [l] [)] Uj [(] [l] [)]<br>**----- End of picture text -----**<br>


**Fig. 2** Conceptual depiction of the various goals of MILP-related regularization. 

The LP gap regularizer penalizes the average gap over a mini-batch _B_ : 

**==> picture [277 x 27] intentionally omitted <==**

where _Bs ⊆ B_ is an optional subsample of the training mini-batch to limit the number of LP solves per training step. In practice, we find that even _|Bs|_ = 1 can achieve the desired effect. We use _δi_ to denote _δi_[min] , _δi_[max] , or their sum (total LP gap), depending on the optimization context. For example, in surrogate-based problems where the NN output must be minimized, penalizing _δi_[min] is the natural choice, as it targets the gap relevant to the downstream MILP objective. 

We note that in some settings surrogate models can have multiple output neurons, e.g., classification models or quantile neural networks [57, 58], complicating the definition of the LP relaxation value (15). For these models we propose quantifying the LP relaxation gap for a surrogate objective by projecting the vector of outputs _z_[(] _[L]_[+1)] onto a random vector, analogous to stochastic Sobolev training [54]. Following this approach, the objective function for (15) is replaced by _ω[⊤] z_[(] _[L]_[+1)] , where _ω_ is a normalized randomly sampled vector. Averaging over many mini-batches would naturally encourage tightening in all possible output directions. 

## _**Interpretation**_ 

_R_ LP measures relaxation looseness at individual training points _xi_ , while the global LP gap (minimizing/maximizing over all _x ∈X_ ) measures the worst-case looseness over the domain _X_ . Including pointwise estimates at many training points is expected to generally tighten the relaxation over regions of the search space, e.g., sub-domains of a branch-and-bound search. Intuitively, the global relaxation may be tightened as well, though this is not guaranteed. 

## **4.2 Differentiating through the LP solution** 

Computing the gradient _∂R_ LP _/∂θ_ requires differentiating the solution to the LP (15) with respect to the network parameters _θ_ . The LP is a parametric linear program whose constraint data depend on _θ_ . Writing this LP (15) in standard form: 

**==> picture [306 x 68] intentionally omitted <==**

where _y_ collects all primal variables ( _z_[(] _[ℓ]_[)] _,_ ˆ _x_[(] _[ℓ]_[)] _, a_[(] _[ℓ]_[)] ) across NN layers, _c_ is the objective vector (in this case selecting the output neuron), the equality constraints encode the pre-activation definitions (1), and the inequality constraints encode the big-M constraints (4) with _a_[(] _j[ℓ]_[)] _∈_ [0 _,_ 1]. At the LP solution, let _ν[∗] ∈_ R _[m]_[eq] and _µ[∗] ∈_ R _[m]_[ineq] denote the optimal dual variables for the equality and inequality constraints, respectively, with _µ[∗] ≥_ 0. 

10 

**Proposition 2** (Sensitivity for parametric LP) _Suppose the LP_ (18) _has a unique, non-degenerate optimal basis. Then the optimal value V_[min] _is differentiable with respect to θ, and_ 

**==> picture [356 x 22] intentionally omitted <==**

_Proof_ To obtain these derivatives, we follow the approach of [59] and differentiate the KKT conditions. A similar analysis is also provided in Fiacco [41, Chapter 3.4]. In particular, the Lagrangian of (18) is given by: 

**==> picture [332 x 72] intentionally omitted <==**

The KKT conditions for stationarity, primal feasibility, and complementary slackness are: 

where the _D_ ( _·_ ) operator forms a diagonal matrix from a vector. To obtain a derivative, we assume (or approximate) the active-constraint set is locally constant, i.e., at a non-degenerate optimal basis, so the solution [ _y[∗]_ ( _θ_ ) _, ν[∗]_ ( _θ_ ) _, µ[∗]_ ( _θ_ )] is a smooth function of _θ_ by the implicit function theorem applied to (21). We refer the reader to Fiacco [41, Chapter 2.4] for an overview of relevant implicit function theorem results. Since the objective vector _c_ is fixed, we can first differentiate the objective _V_[min] = _c[⊤] y[∗]_ , giving 

**==> picture [251 x 21] intentionally omitted <==**

We then substitute the stationarity condition from (21), giving 

**==> picture [299 x 22] intentionally omitted <==**

Now, differentiating the primal feasibility conditions, _A_ eq( _θ_ ) _y[∗]_ ( _θ_ ) = _b_ eq( _θ, xi_ ), gives: 

**==> picture [276 x 20] intentionally omitted <==**

For the inequality constraints, complementary slackness (21) gives _µ[∗] i[>]_[0][only][when] _[G][i]_[(] _[θ]_[)] _[ y][∗]_[=] _[h][i]_[(] _[θ]_[),] i.e., the _i_ -th constraint is active. Differentiating the active inequality constraints (noting that _µ[∗] i_[=][0][for] inactive constraints) therefore gives: 

**==> picture [426 x 35] intentionally omitted <==**

The result applies the familiar LP shadow-price interpretation (the rate of change of the optimum with respect to the right-hand side equals the dual variable) to perturbations in both the constraint matrix and the inequality right-hand side. In other words, the sensitivity of the optimal value to perturbations in constraint data can be computed from the optimal dual variables, without requiring differentiation through the `min` operator itself. The dual variables _ν[∗]_ and _µ[∗]_ are a standard output of LP solvers (e.g., as shadow prices from HiGHS or Gurobi). We refer the reader to [60, Chapter 5] for a more comprehensive treatment of parametric LPs and global LP sensitivity analysis. 

While Proposition 2 gives a simple avenue to obtain sensitivities for simple, LP-based relaxations, more complicated formulations may also be used to produce bounds, e.g., convex NLP relaxations. Note that recent works [31, 61] study relaxations for nonlinear activation functions, another direction for future generalization beyond LP relaxations. The proposed regularizer may be generalized to these settings, e.g., following approaches to differentiate through nonlinear programs [59, 62, 63]. Moreover, traditional envelope theorems describe conditions for the value of a parameterized (nonlinear) optimization problem to be differentiable in the parameter and provide formulas for their derivatives [40, 41]. We note there is also a growing literature on software frameworks for differentiable optimization [64, 65]. 

11 

_Remark 3_ (Envelope theorem versus KKT differentiation) An alternative route to _∂V_[min] _/∂θ_ is to differentiate the KKT stationarity conditions implicitly. At the optimal basis, differentiating the stationarity conditions (21) with respect to _θ_ yields a linear system involving the Jacobian _∂y[∗] /∂θ_ of the optimal primal solution. Computing this Jacobian requires solving an _ny × |θ|_ system at every training step, where _ny_ is the number of primal LP variables (scaling with network width and depth), and _|θ|_ is the number of network parameters. The proposed formulation avoids this scaling entirely. Because _∂L/∂y_ = 0 at optimality (primal stationarity), the terms involving _∂y[∗] /∂θ_ cancel in the total derivative, and the gradient collapses to the dual-weighted expression (19). In practice, we require only the dual variables _ν[∗]_ and _µ[∗]_ , which are already a standard solver output, (with no additional linear system to solve). 

## **4.3 Application to the big-M LP** 

We observe that the LP constraints depend on the value of _θ_ through two channels, as written in (18): 

- ˆ 

- 1. **Equality constraints (direct):** The pre-activation definitions _z_[(] _[ℓ]_[)] = _W_[(] _[ℓ]_[)] _x_[(] _[ℓ]_[)] [ _ℓ −_ 1] + _b_[(] _[ℓ]_[)] contribute _b_[(] _[ℓ]_[)] to the right-hand side _b_ eq and _W_[(] _[ℓ]_[)] to the constraint matrix _A_ eq. 

2. **Inequality constraints (indirect):** The big-M values _L_[(] _j[ℓ]_[)] _[, U] j_[ (] _[ℓ]_[)] appearing in (4c)–(4d) contribute to both _G_ and _h_ , and their values depend on _θ_ indirectly. For example, the bounds may be computed through IBP recursion (6). 

Noting the similarity of channel (2) to the bound widths discussed in Section 3, in our implementation, we treat the big-M values _L_[(] _j[ℓ]_[)] _[, U] j_[ (] _[ℓ]_[)] as constants when differentiating through the LP, retaining only channel (1). This simplification is further motivated in Section 5, where we explicitly show that the omitted big-M sensitivity can be recovered by the bound-width regularizer when the two are combined. 

Following this simplification, _L_[(] _j[ℓ]_[)] _[, U] j_[ (] _[ℓ]_[)] are treated as constants during differentiation, and the inequality constraint data _G_ and _h_ are independent of _θ_ . The sensitivity (19) therefore reduces to: 

**==> picture [309 x 27] intentionally omitted <==**

The equality constraints can be grouped by layer for ease of notation. At layer _ℓ_ , the constraint ˆ _zj_[(] _[ℓ]_[)] _− Wj_[(] _[ℓ]_[)] _x_[(] _[ℓ]_[)] [ _ℓ −_ 1] = _b_[(] _j[ℓ]_[)] has dual variable _νj_[(] _[ℓ]_[)][.][Differentiating][and][plugging][into][(][26][)][gives:] 

**==> picture [271 x 29] intentionally omitted <==**

**==> picture [271 x 30] intentionally omitted <==**

ˆ where _x_[(] _k[ℓ][−]_[1)] _[∗]_ is the LP primal value of the post-activation variable at layer _ℓ−_ 1 (for _ℓ_ = 1, these correspond to elements of the fixed input component _xi_ ). The gradient of the per-sample LP gap (16) is therefore approximated as: 

**==> picture [287 x 27] intentionally omitted <==**

where the first term is the standard backpropagation gradient. 

12 

## **4.4 Proxy implementation** 

Rather than implementing a custom backward pass for optimization problems as in [59], we construct a differentiable proxy tensor that has a gradient matching (27)–(28): 

**==> picture [303 x 30] intentionally omitted <==**

where _ν_[(] _[ℓ]_[)] and _x_ ˆ[(] _[ℓ][−]_[1)] _[∗]_ are treated as fixed constants (detached from the computation graph) obtained from the LP solution, while the network parameters _W_[(] _[ℓ]_[)] and _b_[(] _[ℓ]_[)] remain in the computation graph. Observe that, by construction, the derivatives _∂P/∂θ_ reproduce (27)–(28) exactly. Nevertheless, the forward-pass value of _P_ ( _θ_ ) does not match the true LP value _V_[min] , which we would like to include in the regularizer (17). Therefore, we apply the idea of a ‘straight-through estimator,’ i.e., a proxy derivative that is used in the backward pass only [66, 67]: 

**==> picture [294 x 14] intentionally omitted <==**

where sg[ _·_ ] denotes the _stop-gradient operator_ . Specifically, sg[ _u_ ] returns the same numerical value as _u_ , but is treated as a constant during differentiation, i.e., _∂_ sg[ _u_ ] _/∂θ ≡_ 0. In automatic differentiation frameworks this is implemented by detaching the tensor from the computation graph (e.g. `u.detach()` in PyTorch). 

The two passes of (31) behave differently by design. In the _forward pass_ , _P_ ( _θ_ ) and sg[ _P_ ( _θ_ )] evaluate to the same scalar _p_ , so _V_[�][min] = _p − p_ + _V_[min] = _V_[min] . In other words, the forward pass returns the desired LP optimal value from the solver. In the _backward pass_ , the stop-gradient removes the second term and the solver output _V_[min] is a constant (solving the relaxation using an LP solver is not included in the computation graph), giving _∂V_[�][min] _/∂θ_ = _∂P/∂θ−_ 0+0 = _∂P/∂θ_ . In other words, the backward pass returns the desired gradient (27)–(28). This proxy implementation avoids having to implement a custom backward pass for the proposed regularizer, while preserving both the correct function value and the correct gradient. 

Figure 3 illustrates the pointwise LP relaxation gap _δ_[min] ( _θ, xi_ ) in (16) for NN models trained with the various regularizers on a simple benchmark function. We observe that the LP gap regularizer can produce surrogate models with much tighter pointwise relaxations over the function domain. 

## **5 Combining regularization strategies** 

Sections 3 and 4 introduce several regularization strategies that target different aspects of MILP difficulty when used downstream as surrogate models (Figure 2). A summary of the various proposed regularizers is given in Table 1. Computational costs for the various regularizers are given in Table 2. 

**Table 1** Summary of MILP- and relaxation-informed regularization strategies. 

|Regularizer|What it targets|Gradient w.r.t. _θ_|
|---|---|---|
|_R_BW|Bound widths (big-M values)|_dL/dθ, dU/dθ_ via autodif|
|_R_SN|Number of binary variables|_dL/dθ_ or _dU/dθ_ via autodif|
||(combinatorial difculty)|(for unstable neurons)|
|_R_LP|LP relaxation gap|LP dual variables|
||(_L, U_ treated as constants)||
|_R_BW +_R_LP|Both big-M values and LP gap|Approximates full _dV_ max_/dθ_|
|||(see Proposition 3)|



Here _|Bs|_ is the number of LP samples per batch (a tunable parameter to manage training overhead). The IBP forward pass has the same cost as a standard network forward pass (one 

13 

**==> picture [384 x 228] intentionally omitted <==**

**----- Start of picture text -----**<br>
None L1 L2<br>mean 2 gap=9.49;  num unstable=64 mean 2 gap=2.92;  num unstable=64 mean 2 gap=2.89;  num unstable=64<br>10 [2]<br>1 1 1<br>0 0 0<br>1 1 1<br>10 [1]<br>2 2 2<br>2 1 0 1 2 2 1 0 1 2 2 1 0 1 2<br>x1 x1 x1<br>BoundWidth StableNeuron LP gap<br>mean 2 gap=2.29;  num unstable=60 mean 2 gap=3.69;  num unstable=59 mean 2 gap=0.72;  num unstable=64<br>10 [0]<br>1 1 1<br>0 0 0<br>1 1 1<br>10 1<br>2 2 2<br>2 1 0 1 2 2 1 0 1 2 2 1 0 1 2<br>x1 x1 x1<br>x2 x2 x2<br>)  [log scale]<br> x,<br>min(<br>V<br>)(x<br>fLP gap:<br>x2 x2 x2<br>**----- End of picture text -----**<br>


**Fig. 3** LP gap for NN models with _{_ 32,32 _}_ hidden layers trained on the scaled Peaks function with two inputs ( _x_ 1, _x_ 2), with output _f_ ( _x_ ). Different regularizers are applied during training, with weights chosen to maintain validation MSE of similar scale. 

**Table 2** Per-step computational cost of each regularizer. 

|regularizer|Extra cost per training step|Gradient source|
|---|---|---|
|_R_L1_, R_L2|Negligible|Autograd|
|_R_BW|1 IBP forward pass|Autograd through IBP|
|_R_SN, _R_SN2|1 IBP forward pass|Autograd through IBP|
|_R_LP|_|Bs|_ LP solves|LP duals + autograd|
|_R_BW +_R_LP|1 IBP pass + _|Bs|_ LP solves|Both channels of (32)|



matrix–vector product per layer). The LP solves are the dominant cost for _R_ LP; they can be parallelized across samples and potentially accelerated by warm-starting from the previous iterate. 

## **5.1 Approximation of the total derivative** 

As observed in Section 4.3, the LP optimal value _V_[max] depends on _θ_ through two channels: 

**==> picture [367 x 50] intentionally omitted <==**

The first term is what _R_ LP effectively computes via (26), as the bound widths (big-M values) are assumed constant. The second (omitted) term captures how changing _θ_ alters the big-M constants _L_[(] _j[ℓ]_[)] _[, U] j_[ (] _[ℓ]_[)][,][which][in][turn][affect][the][LP][feasible][region][and][hence][the][tightness][of][the][LP][relaxation.] This second term factors as: 

- _∂V_[max] _/∂L_[(] _j[ℓ]_[)] and _∂V_[max] _/∂Uj_[(] _[ℓ]_[)][: the sensitivity of the LP value to the big-M constants, given by] the dual variables _µ[∗]_ of the inequality constraints (4c)–(4d); 

- _dL_[(] _j[ℓ]_[)] _[/dθ]_[and] _[dU] j_[ (] _[ℓ]_[)] _[/dθ]_[:][the][gradients][of][the][bound][w.r.t.] _[θ]_[,][e.g.,][obtained][using][IBP.] 

14 

**==> picture [412 x 61] intentionally omitted <==**

Comparing with the second (indirect) term in (32), we see that _R_ BW effectively provides a _surrogate_ for the indirect big-M sensitivity path, with the LP dual multipliers _∂V_[max] _/∂L_[(] _j[ℓ]_[)] and _∂V_[max] _/∂Uj_[(] _[ℓ]_[)] replaced by the uniform weights _−_ 1 and +1, respectively. 

**Proposition 3** (Combining _R_ BW and _R_ LP regularizers approximates the full gradient) _The combined regularizer R_ LP + _α R_ BW _produces the gradient:_ 

**==> picture [350 x 49] intentionally omitted <==**

_which approximates the total derivative_ (32) _with the LP dual weights ∂V_[max] _/∂L_[(] _j[ℓ]_[)] _and ∂V_[max] _/∂Uj_[(] _[ℓ]_[)] _replaced by α and −α._ 

_Remark 4_ (Why not differentiate through the big-M values directly?) Computing the exact second term in (32) would require the LP dual variables _µ[∗]_ for the inequality constraints as well as the full IBP Jacobian _dL/dθ, dU/dθ_ . While feasible in principle, this doubles the information needed from each LP solve and couples the LP backward pass to the IBP backward pass. The combined _R_ LP + _α R_ BW avoids this coupling while still capturing both sensitivity paths, with _α_ serving as a tunable proxy for the (unknown, sampledependent) LP dual weights. The scalar _α_ can be interpreted as a uniform “importance weight” for big-M tightness relative to constraint-RHS sensitivity. 

## **6 Computational Results** 

To evaluate the regularization techniques proposed in Sections 3–4, we first consider the experimental settings of Plate et al. [6] and train NNs as surrogates for standard non-convex benchmark functions. We furthermore study quantile NNs as surrogates in stochastic programming applications, following Alc´antara et al. [57]. We compare training and MILP performance on downstream optimization problems with different (combinations of) regularizers added during training. 

## **6.1 Implementation** 

All experiments were run on a server equipped with AMD EPYC 7742 64-Core Processors. Each training and optimization run was allocated 8 CPU cores and 16 GB of memory. NN surrogate models and regularizers were implemented using PyTorch [68], and MILP optimization problems were solved using Gurobi v13.0.1 [69]. The LPs for the relaxation-based regularizer are implemented using `scipy.optimize` and solved using HiGHS [70]. The author acknowledges the use of Anthropic’s Claude (v4.6 models) to assist with setting-up the server experimental environments. The content was reviewed by the author, who takes full responsibility for the final manuscript. 

Although Gurobi can solve LPs, we use a HiGHS implementation for two reasons: first, it avoids the per-call overhead of constructing Gurobi model objects inside each training batch, which dominates runtime for the relaxed LPs encountered; and second, it keeps the entire training pipeline within open-source Python dependencies following the convention of machine learning software. The LP instances encountered during training consist of one LP per regularized sample, with the number of variables and constraints growing linearly in the total number of neurons. We found that HiGHS solves each these LP in milliseconds, but the cost accumulates over many minibatches, which is reflected in the computational costs reported in Table 3. While our experiments are limited to CPU servers, an interesting direction for future work is to exploit GPU-based LP solvers [71, 72] during training, which could substantially reduce this overhead and integrate more naturally with GPU-based model training pipelines. 

15 

**==> picture [384 x 255] intentionally omitted <==**

**----- Start of picture text -----**<br>
None L1 L2<br>test MSE=0.0032 test MSE=0.0037 test MSE=0.0033<br>1.0 1.0 1.0<br>0.8 0.8 0.8<br>0.6 0.6 0.6<br>0.4 0.4 0.4<br>0.2 0.2 0.2<br>0.0 0.0 0.0<br>0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0<br>True (normalized) True (normalized) True (normalized)<br>BoundWidth StableNeuron LP gap<br>test MSE=0.0020 test MSE=0.0032 test MSE=0.0034<br>1.0 1.0 1.0<br>0.8 0.8 0.8<br>0.6 0.6 0.6<br>0.4 0.4 0.4<br>0.2 0.2 0.2<br>0.0 0.0 0.0<br>0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0<br>True (normalized) True (normalized) True (normalized)<br>Predicted Predicted Predicted<br>Predicted Predicted Predicted<br>**----- End of picture text -----**<br>


**Fig. 4** Parity plots for NN models with _{_ 32,32 _}_ hidden layers trained on the scaled Peaks function with two inputs ( _x_ 1, _x_ 2), with output _f_ ( _x_ ). Different regularizers are applied during training, with weights chosen to maintain validation MSE of similar scale. 

## **6.2 Direct Optimization over Surrogates** 

## **6.2.1 Benchmark Functions** 

We first study the direct minimization over a surrogate model output, i.e., solving the straightforward problem to minimize _fθ_ ( _x_ ). We consider the benchmark functions and training settings used by Plate et al. [6] to facilitate comparison: 

1. The Himmelblau function _f_ himmelblau : [ _−_ 5 _,_ 5][2] _→_ R, denoted as `himmelblau` , which has a global minimum of 0 at four points: 

**==> picture [215 x 13] intentionally omitted <==**

2. The Peaks function _f_ peaks : [ _−_ 2 _,_ 2][2] _→_ R, denoted as `peaks` , which is multimodal with a unique global minimum of _−_ 6 _._ 551 [6]: 

**==> picture [394 x 38] intentionally omitted <==**

3. The _d_ -dimensional Ackley function _f_ ackley _, d_ : [ _−_ 3 _._ 5 _,_ 3 _._ 5] _[d] →_ R, denoted as `ackley-d` . The function is multimodal with a unique global minimum of 0: 

**==> picture [394 x 39] intentionally omitted <==**

As an illustrative example, we first train feedforward neural networks with two hidden layers of 32 neurons each with the various proposed regularizers, tuning the regularization penalties to ensure a similar validation MSE. The continuous relaxations are shown in Figure 1, the pointwise relaxation gaps in Figure 3, and prediction parity plots in Figure 4. This simple example allows 

16 

us to visually verify that the proposed regularizers may be tuned to improve relaxation tightness without significantly worsening prediction accuracy and/or generalization ability. 

For the optimization studies below, we now partially follow the training setting of [6] and consider feedforward neural networks of with _{_ 2 _,_ 3 _,_ 5 _}_ hidden layers of 25 neurons each. The models are trained on 100,000 samples for `peaks` and `himmelblau` and 150,000 samples for `ackley-2` , with all samples generated using Latin Hypercube sampling. We test some larger models including hidden layers of 50 neurons on the 5-dimensional Ackley function, `ackley-5` , where the number of samples is doubled to 300,000. Data are normalized, and 30% of the data are used as a test set to measure generalization ability. Networks are trained for 200 epochs using the Adam optimizer. 

## **6.2.2 Training Results** 

## _**Computational overhead.**_ 

Table 3 shows training-time ratios relative to the unregularized baseline. Shrinkage regularizers ( _R_ L1, _R_ L2) add modest overhead (generally 1–2 _×_ ), as they require only element-wise weight penalties. The bound-width and stability regularizers _R_ BW, _R_ SN and _R_ SN2 incur similar overheads to each other and slightly more than the shrinkage regularizers. We also observe their computational costs scale with network depth due to the layer-by-layer IBP propagation. The LP-based regularizer _R_ LP is the most expensive, at approximately 5–10 _×_ , reflecting the cost of solving one full LP relaxation of the network per regularization sample; the combined regularizer _R_ BW + _R_ LP similarly reflects computational costs dominated by the LP component. Note these overheads are incurred only once during model training time and can potentially be amortized over usage in many downstream optimization instances. 

**Table 3** Training-time overhead of each regularizer relative to the unregularized baseline,¯ averaged across `ackley-2` , `himmelblau` , `peaks` . Each cell shows _r_ ¯ = _|B|_ 1 � _b∈B t_ ¯ _tb,b_ none _,_ reg[,][where] _[t]_[¯][is][the][mean][training][time][over][seeds.] 

||regularizer<br>_λ_<br>`2-25-25-1`<br>`2-25-25-25-1`<br>`2-25-25-25-25-25-1`|
|---|---|
||Baseline (_t_0)<br>—<br>70_._0_±_12_._3 s<br>83_._9_±_13_._6 s<br>107_._6_±_22_._2 s|
||None<br>—<br>1_._00<br>1_._00<br>1_._00|
||_R_L1<br>10_−_4<br>1_._58_±_0_._31<br>1_._62_±_0_._31<br>1_._74_±_0_._30<br>10_−_3<br>1_._19_±_0_._02<br>1_._23_±_0_._02<br>1_._25_±_0_._04<br>10_−_2<br>1_._21_±_0_._04<br>1_._23_±_0_._04<br>1_._24_±_0_._04<br>_R_L2<br>10_−_4<br>2_._06_±_0_._73<br>1_._55_±_0_._30<br>1_._86_±_0_._65<br>10_−_3<br>1_._28_±_0_._06<br>1_._25_±_0_._06<br>1_._30_±_0_._07<br>10_−_2<br>1_._29_±_0_._12<br>1_._23_±_0_._08<br>1_._35_±_0_._07<br>_R_BW<br>10_−_4<br>1_._65_±_0_._26<br>1_._77_±_0_._25<br>2_._45_±_0_._40<br>10_−_3<br>1_._42_±_0_._03<br>1_._49_±_0_._06<br>1_._70_±_0_._09<br>10_−_2<br>1_._41_±_0_._04<br>1_._52_±_0_._04<br>1_._76_±_0_._05<br>_R_SN<br>10_−_4<br>1_._75_±_0_._20<br>2_._08_±_0_._22<br>3_._01_±_0_._77<br>10_−_3<br>1_._59_±_0_._04<br>1_._67_±_0_._07<br>1_._91_±_0_._08<br>10_−_2<br>1_._58_±_0_._01<br>1_._64_±_0_._09<br>1_._92_±_0_._10<br>_R_SN2<br>10_−_4<br>2_._38_±_0_._57<br>2_._02_±_0_._41<br>2_._19_±_0_._40<br>10_−_3<br>1_._53_±_0_._03<br>1_._57_±_0_._05<br>1_._79_±_0_._08<br>10_−_2<br>1_._51_±_0_._05<br>1_._59_±_0_._03<br>1_._83_±_0_._11<br>_R_LP<br>10_−_4<br>6_._06_±_1_._21<br>6_._64_±_0_._64<br>10_._92_±_0_._34<br>10_−_3<br>4_._55_±_0_._09<br>6_._38_±_0_._21<br>9_._74_±_0_._43<br>10_−_2<br>4_._59_±_0_._07<br>6_._07_±_0_._23<br>9_._09_±_0_._55<br>_R_BW +_R_LP<br>10_−_4<br>5_._92_±_0_._57<br>7_._11_±_1_._26<br>10_._28_±_0_._45<br>10_−_3<br>4_._37_±_0_._03<br>5_._14_±_0_._24<br>7_._28_±_0_._38<br>10_−_2<br>4_._01_±_0_._08<br>4_._41_±_0_._14<br>5_._43_±_0_._50|



17 

**Table 4** Test MSE of each regularizer relative to the unregularized¯ baseline, averaged across `ackley-2` , `himmelblau` , `peaks` . Cells show _r_ ¯ = _|B|_ 1 � _b∈B m_ ¯ _mb,b_ none _,_ reg _[±]_[std,][where] _[m]_[¯] is the mean test MSE over seeds. Raw MSE values differ across benchmarks, so only the normalised ratio is shown. Values _<_ 1 indicate lower test error than the baseline. 

||regularizer<br>_λ_<br>`2-25-25-1`<br>`2-25-25-25-1`<br>`2-25-25-25-25-25-1`|
|---|---|
||None<br>—<br>1_._00<br>1_._00<br>1_._00|
||_R_L1<br>10_−_4<br>4_._51_±_3_._36<br>10_._55_±_7_._11<br>32_._86_±_13_._38<br>10_−_3<br>45_._48_±_43_._78<br>106_._86_±_94_._31<br>352_._35_±_235_._47<br>10_−_2<br>148_._46_±_147_._78<br>415_._74_±_416_._71<br>1493_._12_±_1290_._10<br>_R_L2<br>10_−_4<br>1_._92_±_1_._16<br>3_._71_±_1_._52<br>12_._09_±_4_._73<br>10_−_3<br>18_._72_±_19_._17<br>35_._29_±_30_._76<br>96_._81_±_58_._84<br>10_−_2<br>84_._11_±_71_._10<br>212_._63_±_170_._74<br>760_._82_±_525_._09<br>_R_BW<br>10_−_4<br>0_._54_±_0_._22<br>0_._40_±_0_._13<br>0_._78_±_0_._06<br>10_−_3<br>0_._89_±_0_._38<br>0_._99_±_0_._31<br>1_._75_±_0_._23<br>10_−_2<br>3_._15_±_2_._06<br>4_._77_±_2_._06<br>11_._04_±_4_._53<br>_R_SN<br>10_−_4<br>0_._74_±_0_._17<br>0_._61_±_0_._21<br>0_._80_±_0_._20<br>10_−_3<br>0_._63_±_0_._26<br>0_._68_±_0_._27<br>1_._34_±_0_._10<br>10_−_2<br>2_._28_±_1_._16<br>3_._11_±_1_._04<br>6_._47_±_2_._30<br>_R_SN2<br>10_−_4<br>0_._97_±_0_._02<br>0_._97_±_0_._03<br>1_._01_±_0_._02<br>10_−_3<br>0_._93_±_0_._04<br>0_._95_±_0_._03<br>1_._00_±_0_._05<br>10_−_2<br>1_._10_±_0_._12<br>0_._99_±_0_._07<br>0_._95_±_0_._02<br>_R_LP<br>10_−_4<br>0_._77_±_0_._35<br>0_._65_±_0_._27<br>0_._90_±_0_._14<br>10_−_3<br>0_._98_±_0_._58<br>1_._02_±_0_._47<br>2_._17_±_0_._97<br>10_−_2<br>3_._69_±_3_._18<br>7_._20_±_5_._45<br>33_._13_±_28_._23<br>_R_BW +_R_LP<br>10_−_4<br>0_._51_±_0_._25<br>0_._47_±_0_._21<br>0_._89_±_0_._16<br>10_−_3<br>1_._45_±_0_._89<br>1_._87_±_0_._83<br>3_._75_±_1_._54<br>10_−_2<br>9_._51_±_9_._58<br>12_._47_±_10_._34<br>42_._91_±_28_._30|



## _**Accuracy vs tractability tradeoff.**_ 

Table 4 shows normalized test MSE ratios relative to the unregularized baseline. Across all regularization methods, increasing the regularization penalty induces a reduced test accuracy as expected. Shrinkage regularizers ( _R_ L1, _R_ L2) impose a steep accuracy penalty even at the moderate regularization strengths considered, with ratios exceeding 100 _×_ at _λ_ = 10 _[−]_[3] for the deepest architectures. Note that, following convention, these shrinkage regularizers are not normalized by the number of model parameters, while the per-sample regularizers are averaged over number of samples. 

On the other hand, the proposed bound-width ( _R_ BW) and stability ( _R_ SN) regularizers achieve ratios near, or even below unity at _λ_ = 10 _[−]_[4] , indicating that mild regularization of the IBP bound widths may even provide a beneficial implicit regularization that simultaneously improves generalization and MILP tractability (though we do not claim this in general). The combined _R_ BW + _R_ LP regularizer shows the same property at _λ_ = 10 _[−]_[4] . Nevertheless, for these regularizers, accuracy again degrades as _λ_ increases, most notably for the LP and combined regularizers. These results suggest that, in general regularizers provide a handle for tuning the tradeoff between surrogate model quality and the tractability of downstream optimization applications. 

## **6.2.3 Optimization Results** 

Tables 5–7 report results for surrogate models trained with the various regularization strategies on two-dimensional benchmark functions. Four MILP tractability metrics are reported: number of unstable neurons _|U|_ , LP relaxation gap, MILP node count, and wall-clock MILP solve time. The unregularized baseline is computationally intractable on the deepest architectures for all benchmarks, consistently exceeding the 1800 s MILP time limit. 

Overall, models trained with the proposed regularizers exhibit reduced MILP solve times, by up to four orders of magnitude relative to the baseline. Recall that, stronger regularization can further reduce MILP solve time at the expense of accuracy (Table 4). To show this performance tradeoff, rows are shaded in grey when the mean objective found in the downstream problem is 

18 

at least 5% higher than the objective found using the unregularized surrogate model. On the simple `himmelblau` function (Table 5), surrogate model training appears especially sensitive to the shrinkage regularizers, where regularization degrades downstream performance in all cases except the weakest L2 regularizer. Models trained with the bound-width regularizer weakly included considerably accelerate the downstream MILP solution time without affecting solution quality. For the slightly more complicated `peaks` function (Table 6), the shrinkage regularizers again generally degrade decision performance for smaller surrogate models. This trend is mitigated for the deepest model (five hidden layers), which is more over-parameterized. In this setting, we found that regularization with the combined bound-width and LP regularizer to greatly accelerate downstream MILP solution, including many settings where the MILP can be solved in a single node. 

Overall, on these simple functions, _R_ BW at _λ_ = 10 _[−]_[3] reduces unstable neurons by roughly 50% on `himmelblau` and `peaks` (e.g., from 75 to 32–43 for the 3-layer architecture) and drives MILP times below 1 s across most architectures on the simpler benchmarks. The stability regularizer _R_ SN achieves similar reductions in _|U|_ and MILP times, with slightly less aggressive compression of the LP gap as expected. The second stability regularizer _R_ SN2 consistently produces worse models compared to _R_ SN, suggesting the weaker gradient signal can make it less effective in practice. The combined _R_ BW + _R_ LP regularizer is particularly effective at the lowest regularization strength ( _λ_ = 10 _[−]_[4] ), where it simultaneously achieves the smallest _|U|_ and near-zero LP gap, resulting in the best MILP solve times in most settings where solution quality is not degraded. For the challenging `ackley-2` function (Table 7), the combined regularizer again generally results in shortest downstream MILP solve times. Note that the shrinkage regularizers can also reduce MILP solve times in this setting without affecting solution quality, albeit to a lesser extent. 

Interestingly, inclusion of the LP regularizer _R_ LP alone nearly eliminates the LP relaxation gap (often to _<_ 0 _._ 01), but does not reduce the number of unstable neurons, since it only tightens the continuous relaxation without encouraging neuron stability. As a result, Gurobi must still branch on a large number of binary ReLU variables, and solve times remain significant on the harder instances. This complementarity motivates the combined regularizer: _R_ BW drives neurons toward stability (reducing _|U|_ ), while _R_ LP tightens the LP gap, and together they compound to produce substantially smaller B&B trees (and MILP solve times). In summary, results on these two-dimensional benchmark functions show that, for simpler functions (where models are more overparameterized), downstream performance is more sensitive to regularization weights, especially shrinkage regularizers, and weak relaxation-informed regularization can greatly accelerate downstream performance. For the challenging Ackley function, (where models are less overparametrized), most regularization techniques accelerate downstream solution, with the powerful combined regularizer producing models with the best performance across most architectures. 

## _**Effect of function complexity and architecture depth.**_ 

The relative benefit of the proposed regularizers grows with both function complexity and network depth. On `himmelblau` (the simplest benchmark), _R_ BW at _λ_ = 10 _[−]_[3] is sufficient to reduce the 5-layer MILP from infeasible within the time limit to 0.23 s on average. On `ackley-2` , higher LP gaps in the baseline (e.g., 17.84 vs. 13.39 for `peaks` with two hidden layers) mean that reducing _|U|_ alone is not sufficient at low _λ_ , and the combined regularizer is needed for consistently fast solves. 

Table 8 shows results for some larger NNs trained on the five-dimensional Ackley function. The L2 regularizer was omitted from these experiments, as we found it to be dominated by the L1 regularizer in previous experiments, similar to literature observations for this setting [6]. On `ackley-5` function, even with more neurons (up to 250 in the 50-wide architecture), the combined _R_ BW + _R_ LP regularizer at _λ_ = 10 _[−]_[4] reduces the 5-layer solve time from _>_ 1800 s to under 1 s while maintaining competitive surrogate accuracy. The standard L1 shrinkage regularizers offers less noticeable tractability improvements on these harder instances (larger NN models), though they are effective on the smaller models, e.g., three hidden layers (middle column of Table 8). This observation suggests that directly targeting the structure of the MILP embedding is essential for large gains in more challenging settings, where surrogate models are larger and less overparametrized. 

19 

|**Table 5** Results on the `himmelblau` benchmark across architectures and regularizers. Each regularizer family shows three rows for<br>_λ ∈{_10_−_4_,_10_−_3_,_10_−_2_}_. _|U|_: mean number of unstable neurons; LP gap: mean LP relaxation gap; MILP nodes/time: mean branch-and-bound<br>nodes and wall-clock time (s). **Bold** values indicate the best (lowest) result across all regularizers at each _λ_ level per architecture; ties are all<br>bolded.<br>Shaded entries mark when the mean objective value found is worse (higher) than the unregularized baseline; such rows are excluded<br>from best-value consideration.|regularizer<br>_λ_<br>`2-25-25-1`<br>`2-25-25-25-1`<br>`2-25-25-25-25-25-1`<br>_|U|_<br>LP gap<br>MILP<br>_|U|_<br>LP gap<br>MILP<br>_|U|_<br>LP gap<br>MILP<br>nodes<br>time<br>nodes<br>time<br>nodes<br>time|None<br>—<br>50_._0<br>23_._93<br>18_,_530<br>4_._08<br>75_._0<br>53_._91<br>841_,_316<br>243_._87<br>125_._0<br>350_._81<br>_>_2_,_321_,_554<br>_>_1800|_R_L1<br>10_−_4<br>49_._5<br>1_._12<br>49<br>0_._14<br>71_._2<br>0_._37<br>67<br>0_._25<br>105_._0<br>0_._28<br>778<br>2_._02<br>10_−_3<br>46_._5<br>0_._51<br>19<br>0_._07<br>61_._5<br>0_._04<br>1<br>0_._06<br>85_._2<br>0_._51<br>1_,_412<br>2_._9<br>10_−_2<br>49_._9<br>6_._85<br>3_,_400<br>1_._44<br>75_._0<br>15_._15<br>742_,_272<br>295_._94<br>125_._0<br>65_._26<br>_>_2_,_088_,_555<br>_>_1800<br>_R_L2<br>10_−_4<br>39_._0<br>0_._29<br>32<br>0_._18<br>53_._8<br>0_._07<br>148<br>0_._39<br>**70**_._**5**<br>0_._01<br>205<br>**0**_._**50**<br>10_−_3<br>43_._5<br>0_._21<br>1<br>0_._04<br>65_._0<br>0_._13<br>416<br>0_._48<br>94_._6<br>1_._14<br>112_,_459<br>73_._40<br>10_−_2<br>49_._4<br>6_._12<br>5_,_610<br>1_._64<br>73_._5<br>10_._84<br>207_,_382<br>70_._27<br>123_._8<br>72_._41<br>2_,_443_,_711<br>1649_._91<br>_R_BW<br>10_−_4<br>39_._3<br>1_._64<br>231<br>0_._38<br>52_._1<br>0_._31<br>275<br>0_._46<br>79_._2<br>0_._24<br>1_,_280<br>1_._73<br>10_−_3<br>**23**_._**6**<br>0_._26<br>**1**<br>**0**_._**03**<br>**32**_._**7**<br>**0**_._**13**<br>**14**<br>**0**_._**07**<br>**51**_._**2**<br>**0**_._**02**<br>**41**<br>**0**_._**23**<br>10_−_2<br>13_._2<br>0_._13<br>1<br>0_._01<br>18_._4<br>0_._04<br>1<br>0_._02<br>**28**_._**8**<br>**0**_._**01**<br>**1**<br>**0**_._**03**<br>_R_SN<br>10_−_4<br>45_._3<br>5_._61<br>1_,_148<br>0_._65<br>61_._8<br>1_._55<br>2_,_327<br>1_._21<br>88_._4<br>0_._54<br>9_,_531<br>8_._06<br>10_−_3<br>25_._5<br>0_._64<br>**1**<br>0_._04<br>34_._9<br>0_._40<br>23<br>0_._15<br>53_._7<br>0_._11<br>312<br>0_._74<br>10_−_2<br>**13**_._**4**<br>**0**_._**35**<br>**1**<br>**0**_._**01**<br>17_._0<br>0_._30<br>1<br>0_._02<br>28_._0<br>0_._06<br>1<br>0_._06<br>_R_SN2<br>10_−_4<br>50_._0<br>24_._15<br>15_,_001<br>3_._32<br>74_._8<br>52_._72<br>1_,_028_,_957<br>232_._30<br>124_._0<br>334_._55<br>_>_3_,_090_,_349<br>_>_1800<br>10_−_3<br>49_._6<br>24_._09<br>18_,_414<br>4_._36<br>73_._7<br>52_._28<br>978_,_297<br>276_._49<br>123_._0<br>347_._01<br>_>_2_,_574_,_826<br>_>_1800<br>10_−_2<br>48_._4<br>23_._70<br>18_,_444<br>5_._11<br>72_._5<br>51_._46<br>822_,_372<br>284_._34<br>122_._5<br>329_._29<br>_>_2_,_483_,_279<br>_>_1800<br>_R_LP<br>10_−_4<br>50_._0<br>0_._51<br>159<br>0_._46<br>75_._0<br>0_._11<br>1_,_906<br>1_._07<br>125_._0<br>0_._01<br>693_,_059<br>582_._14<br>10_−_3<br>49_._9<br>**0**_._**02**<br>60<br>0_._22<br>75_._0<br>0_._01<br>1_,_316<br>0_._98<br>125_._0<br>0_._00<br>35_,_418<br>33_._46<br>10_−_2<br>49_._5<br>0_._00<br>1<br>0_._06<br>74_._9<br>0_._00<br>163<br>0_._27<br>124_._8<br>0_._00<br>7_,_184<br>7_._38<br>_R_BW +_R_LP<br>10_−_4<br>**32**_._**2**<br>**0**_._**08**<br>**1**<br>**0**_._**03**<br>**46**_._**6**<br>**0**_._**00**<br>**1**<br>**0**_._**07**<br>77_._2<br>**0**_._**00**<br>**204**<br>0_._60<br>10_−_3<br>18_._9<br>0_._00<br>1<br>0_._01<br>28_._6<br>0_._00<br>1<br>0_._02<br>46_._4<br>0_._00<br>1<br>0_._05<br>10_−_2<br>11_._8<br>0_._00<br>1<br>0_._01<br>14_._1<br>0_._00<br>1<br>0_._01<br>20_._8<br>0_._00<br>1<br>0_._01|
|---|---|---|---|



20 

|**Table 6** Results on the `peaks` benchmark across architectures and regularizers. Each regularizer family shows three rows for<br>_λ ∈{_10_−_4_,_10_−_3_,_10_−_2_}_. _|U|_: mean number of unstable neurons; LP gap: mean LP relaxation gap; MILP nodes/time: mean branch-and-bound<br>nodes and wall-clock time (s). **Bold** values indicate the best (lowest) result across all regularizers at each _λ_ level per architecture; ties are all<br>bolded.<br>Shaded entries mark when the mean objective value found is worse (higher) than the unregularized baseline; such rows are excluded<br>from best-value consideration.|regularizer<br>_λ_<br>`2-25-25-1`<br>`2-25-25-25-1`<br>`2-25-25-25-25-25-1`<br>_|U|_<br>LP gap<br>MILP<br>_|U|_<br>LP gap<br>MILP<br>_|U|_<br>LP gap<br>MILP<br>nodes<br>time<br>nodes<br>time<br>nodes<br>time|None<br>—<br>50_._0<br>13_._39<br>10_,_547<br>2_._37<br>74_._9<br>32_._07<br>462_,_620<br>117_._38<br>124_._9<br>232_._06<br>_>_2_,_337_,_016<br>_>_1800|_R_L1<br>10_−_4<br>47_._8<br>1_._22<br>34<br>0_._08<br>69_._2<br>0_._44<br>107<br>0_._23<br>100_._8<br>1_._58<br>133_,_770<br>90_._61<br>10_−_3<br>47_._8<br>2_._31<br>456<br>0_._36<br>64_._5<br>3_._79<br>6_,_404<br>3_._22<br>75_._8<br>4_._07<br>118_,_823<br>125_._17<br>10_−_2<br>49_._5<br>2_._61<br>566<br>0_._43<br>74_._0<br>2_._95<br>5_,_058<br>2_._48<br>124_._7<br>10_._64<br>858_,_981<br>613_._41<br>_R_L2<br>10_−_4<br>42_._1<br>1_._00<br>86<br>0_._23<br>60_._5<br>0_._29<br>622<br>0_._51<br>**79**_._**0**<br>0_._02<br>649<br>0_._89<br>10_−_3<br>44_._0<br>0_._29<br>42<br>0_._07<br>59_._8<br>0_._81<br>2_,_680<br>1_._73<br>72_._2<br>0_._24<br>1_,_606<br>5_._25<br>10_−_2<br>50_._0<br>4_._01<br>1_,_726<br>0_._80<br>75_._0<br>7_._74<br>36_,_476<br>17_._60<br>125_._0<br>32_._88<br>2_,_308_,_125<br>1341_._62<br>_R_BW<br>10_−_4<br>44_._0<br>2_._07<br>396<br>0_._49<br>58_._5<br>0_._83<br>671<br>0_._72<br>91_._6<br>0_._39<br>990<br>2_._11<br>10_−_3<br>29_._8<br>0_._82<br>**1**<br>0_._04<br>43_._1<br>0_._41<br>**1**<br>0_._08<br>64_._7<br>0_._10<br>30<br>0_._17<br>10_−_2<br>**17**_._**1**<br>0_._50<br>**1**<br>**0**_._**02**<br>23_._1<br>0_._27<br>**1**<br>0_._02<br>37_._8<br>0_._02<br>**1**<br>**0**_._**03**<br>_R_SN<br>10_−_4<br>45_._4<br>4_._03<br>799<br>0_._60<br>62_._4<br>1_._86<br>1_,_999<br>1_._46<br>92_._7<br>0_._88<br>2_,_406<br>4_._11<br>10_−_3<br>34_._1<br>1_._00<br>**1**<br>0_._06<br>44_._0<br>0_._61<br>**1**<br>0_._10<br>67_._4<br>0_._27<br>450<br>0_._57<br>10_−_2<br>19_._6<br>0_._83<br>**1**<br>0_._02<br>24_._3<br>0_._42<br>**1**<br>0_._03<br>37_._8<br>0_._16<br>**1**<br>0_._06<br>_R_SN2<br>10_−_4<br>50_._0<br>12_._26<br>8_,_454<br>2_._15<br>74_._6<br>27_._77<br>297_,_886<br>87_._00<br>123_._3<br>200_._84<br>_>_3_,_083_,_439<br>_>_1800<br>10_−_3<br>46_._7<br>10_._59<br>5_,_319<br>1_._43<br>68_._7<br>25_._24<br>171_,_206<br>52_._98<br>115_._0<br>179_._48<br>3_,_168_,_132<br>1773_._63<br>10_−_2<br>40_._2<br>9_._38<br>3_,_967<br>1_._18<br>62_._5<br>21_._51<br>94_,_206<br>32_._22<br>111_._2<br>158_._83<br>2_,_735_,_933<br>1653_._35<br>_R_LP<br>10_−_4<br>50_._0<br>0_._24<br>**1**<br>0_._05<br>74_._9<br>0_._01<br>568<br>0_._39<br>124_._9<br>**0**_._**00**<br>51_,_325<br>70_._36<br>10_−_3<br>50_._0<br>0_._00<br>**1**<br>0_._05<br>75_._0<br>0_._00<br>11<br>0_._13<br>124_._8<br>**0**_._**00**<br>107_,_905<br>123_._91<br>10_−_2<br>50_._0<br>**0**_._**00**<br>**1**<br>0_._04<br>74_._9<br>0_._00<br>**1**<br>0_._09<br>125_._0<br>0_._00<br>40_,_729<br>47_._21<br>_R_BW +_R_LP<br>10_−_4<br>**38**_._**9**<br>**0**_._**12**<br>**1**<br>**0**_._**05**<br>**55**_._**6**<br>**0**_._**00**<br>**1**<br>**0**_._**10**<br>90_._3<br>0_._00<br>**189**<br>**0**_._**60**<br>10_−_3<br>**26**_._**3**<br>**0**_._**00**<br>**1**<br>**0**_._**01**<br>**41**_._**1**<br>**0**_._**00**<br>**1**<br>**0**_._**03**<br>**59**_._**4**<br>0_._00<br>**1**<br>**0**_._**12**<br>10_−_2<br>14_._4<br>0_._00<br>1<br>0_._01<br>**21**_._**6**<br>**0**_._**00**<br>**1**<br>**0**_._**01**<br>**29**_._**3**<br>**0**_._**00**<br>**1**<br>0_._03|
|---|---|---|---|



21 

|**Table 7** Results on the `ackley-2` benchmark across architectures and regularizers. Each regularizer family shows three rows for<br>_λ ∈{_10_−_4_,_10_−_3_,_10_−_2_}_. _|U|_: mean number of unstable neurons; LP gap: mean LP relaxation gap; MILP nodes/time: mean branch-and-bound<br>nodes and wall-clock time (s). **Bold** values indicate the best (lowest) result across all regularizers at each _λ_ level per architecture; ties are all<br>bolded.<br>Shaded entries mark when the mean objective value found is worse (higher) than the unregularized baseline; such rows are excluded<br>from best-value consideration.|regularizer<br>_λ_<br>`2-25-25-1`<br>`2-25-25-25-1`<br>`2-25-25-25-25-25-1`<br>_|U|_<br>LP gap<br>MILP<br>_|U|_<br>LP gap<br>MILP<br>_|U|_<br>LP gap<br>MILP<br>nodes<br>time<br>nodes<br>time<br>nodes<br>time|None<br>—<br>50_._0<br>17_._84<br>14_,_683<br>3_._82<br>75_._0<br>94_._92<br>1_,_642_,_697<br>462_._94<br>125_._0<br>777_._85<br>_>_2_,_558_,_663<br>_>_1800|_R_L1<br>10_−_4<br>50_._0<br>6_._15<br>2_,_963<br>1_._11<br>74_._8<br>17_._13<br>474_,_665<br>115_._66<br>123_._3<br>88_._96<br>1_,_752_,_605<br>1304_._37<br>10_−_3<br>47_._4<br>2_._34<br>1_,_012<br>0_._52<br>69_._9<br>6_._14<br>45_,_800<br>15_._65<br>101_._8<br>32_._29<br>1_,_250_,_287<br>900_._52<br>10_−_2<br>49_._6<br>2_._13<br>803<br>0_._51<br>69_._1<br>1_._62<br>7_,_001<br>3_._45<br>113_._2<br>10_._66<br>694_,_491<br>648_._10<br>_R_L2<br>10_−_4<br>48_._0<br>6_._56<br>4_,_978<br>1_._34<br>71_._2<br>19_._20<br>498_,_437<br>98_._04<br>118_._5<br>107_._00<br>3_,_377_,_901<br>1721_._21<br>10_−_3<br>46_._5<br>3_._03<br>1_,_223<br>0_._61<br>71_._8<br>8_._88<br>78_,_488<br>21_._36<br>117_._0<br>48_._85<br>1_,_934_,_706<br>967_._87<br>10_−_2<br>48_._7<br>2_._54<br>1_,_072<br>0_._82<br>73_._3<br>5_._50<br>37_,_752<br>15_._01<br>124_._7<br>40_._25<br>2_,_450_,_682<br>1443_._08<br>_R_BW<br>10_−_4<br>41_._2<br>4_._94<br>612<br>0_._69<br>62_._5<br>9_._36<br>15_,_001<br>5_._01<br>**99**_._**3**<br>8_._54<br>486_,_829<br>470_._16<br>10_−_3<br>22_._3<br>0_._90<br>**1**<br>0_._03<br>**38**_._**8**<br>**2**_._**03**<br>**102**<br>**0**_._**25**<br>66_._5<br>1_._78<br>3_,_712<br>2_._46<br>10_−_2<br>9_._9<br>0_._12<br>6<br>0_._02<br>12_._2<br>0_._05<br>1<br>0_._01<br>21_._7<br>0_._08<br>1<br>0_._02<br>_R_SN<br>10_−_4<br>45_._5<br>6_._73<br>1_,_220<br>0_._79<br>64_._5<br>11_._36<br>19_,_685<br>9_._29<br>101_._2<br>12_._68<br>357_,_570<br>492_._44<br>10_−_3<br>34_._5<br>2_._33<br>32<br>0_._15<br>51_._8<br>3_._65<br>1_,_261<br>0_._87<br>83_._3<br>3_._54<br>33_,_285<br>35_._72<br>10_−_2<br>10_._2<br>0_._16<br>1<br>0_._01<br>17_._8<br>0_._34<br>1<br>0_._02<br>35_._2<br>0_._55<br>14<br>0_._11<br>_R_SN2<br>10_−_4<br>49_._7<br>16_._86<br>16_,_744<br>4_._14<br>74_._9<br>87_._16<br>1_,_170_,_921<br>271_._89<br>124_._5<br>740_._97<br>_>_2_,_145_,_526<br>_>_1800<br>10_−_3<br>48_._8<br>16_._66<br>11_,_933<br>2_._83<br>73_._2<br>84_._83<br>1_,_157_,_158<br>273_._73<br>122_._1<br>759_._84<br>_>_2_,_670_,_470<br>_>_1800<br>10_−_2<br>45_._9<br>16_._72<br>12_,_411<br>3_._04<br>70_._5<br>84_._66<br>1_,_078_,_148<br>218_._33<br>120_._3<br>707_._98<br>_>_2_,_667_,_471<br>_>_1800<br>_R_LP<br>10_−_4<br>50_._0<br>**0**_._**81**<br>49<br>0_._21<br>75_._0<br>**0**_._**11**<br>3_,_438<br>1_._71<br>125_._0<br>0_._02<br>601_,_896<br>592_._19<br>10_−_3<br>50_._0<br>0_._00<br>1<br>0_._07<br>75_._0<br>0_._01<br>308<br>0_._35<br>125_._0<br>0_._00<br>59_,_792<br>74_._28<br>10_−_2<br>49_._0<br>**0**_._**00**<br>6<br>0_._06<br>74_._5<br>0_._00<br>4<br>0_._12<br>124_._8<br>0_._00<br>1_,_834<br>1_._73<br>_R_BW +_R_LP<br>10_−_4<br>**37**_._**2**<br>0_._88<br>**20**<br>**0**_._**10**<br>**58**_._**0**<br>0_._21<br>**269**<br>**0**_._**51**<br>99_._8<br>**0**_._**00**<br>**2**_,_**772**<br>**2**_._**53**<br>10_−_3<br>**19**_._**6**<br>**0**_._**07**<br>5<br>**0**_._**02**<br>33_._2<br>0_._00<br>1<br>0_._02<br>**64**_._**2**<br>**0**_._**00**<br>**25**<br>**0**_._**20**<br>10_−_2<br>**7**_._**7**<br>0_._01<br>**1**<br>**0**_._**00**<br>**8**_._**3**<br>**0**_._**00**<br>**1**<br>**0**_._**00**<br>**10**_._**8**<br>**0**_._**00**<br>**1**<br>**0**_._**01**|
|---|---|---|---|



22 

|**Table 8** Results on the `ackley-5` benchmark across architectures and regularizers. Each regularizer family shows three rows for _λ ∈{_10_−_4_,_10_−_3_,_10_−_2_}_.<br>_|U|_: mean number of unstable neurons; LP gap: mean LP relaxation gap; MILP nodes/time: mean branch-and-bound nodes and wall-clock time (s).<br>**Bold** values indicate the best (lowest) result across all regularizers at each _λ_ level per architecture; ties are all bolded.<br>Shaded entries mark when the<br>mean objective value found is worse (higher) than the unregularized baseline; such rows are excluded from best-value consideration.|regularizer<br>_λ_<br>`5-25-25-25-25-25-1`<br>`5-50-50-50-1`<br>`5-50-50-50-50-50-1`<br>_|U|_<br>LP gap<br>MILP<br>_|U|_<br>LP gap<br>MILP<br>_|U|_<br>LP gap<br>MILP<br>nodes<br>time<br>nodes<br>time<br>nodes<br>time|None<br>—<br>125_._0<br>376_._81<br>2_,_851_,_019<br>1541_._25<br>149_._6<br>32_._70<br>2_,_044_,_480<br>1363_._88<br>249_._9<br>1955_._61<br>_>_468_,_135<br>_>_1800|_R_L1<br>10_−_4<br>84_._8<br>2_._06<br>166_,_609<br>180_._14<br>122_._8<br>0_._97<br>152_,_175<br>204_._79<br>204_._2<br>32_._71<br>241_,_319<br>1042_._25<br>10_−_3<br>81_._8<br>0_._08<br>339<br>0_._47<br>130_._4<br>**1**_._**55**<br>**474**_,_**476**<br>**611**_._**46**<br>181_._0<br>7_._20<br>207_,_418<br>698_._49<br>10_−_2<br>123_._0<br>9_._77<br>872_,_461<br>554_._50<br>**148**_._**3**<br>**5**_._**13**<br>**566**_,_**103**<br>**788**_._**14**<br>229_._2<br>12_._83<br>277_,_162<br>1153_._38<br>_R_BW<br>10_−_4<br>67_._8<br>0_._56<br>523<br>0_._95<br>**79**_._**8**<br>**1**_._**84**<br>**1**_,_**506**<br>**1**_._**76**<br>94_._2<br>1_._10<br>2_,_089<br>3_._81<br>10_−_3<br>34_._6<br>0_._08<br>1<br>0_._02<br>43_._2<br>0_._32<br>1<br>0_._05<br>49_._6<br>0_._13<br>1<br>**0**_._**05**<br>10_−_2<br>16_._3<br>0_._00<br>0<br>0_._00<br>16_._8<br>0_._00<br>1<br>0_._00<br>22_._6<br>0_._00<br>0<br>0_._01<br>_R_SN<br>10_−_4<br>73_._9<br>1_._03<br>1_,_379<br>1_._62<br>86_._8<br>1_._99<br>1_,_950<br>2_._30<br>107_._7<br>1_._93<br>34_,_844<br>25_._64<br>10_−_3<br>**43**_._**5**<br>0_._14<br>**2**<br>**0**_._**06**<br>54_._4<br>0_._60<br>46<br>0_._20<br>64_._8<br>0_._18<br>103<br>0_._35<br>10_−_2<br>18_._6<br>0_._02<br>1<br>0_._01<br>21_._1<br>0_._01<br>1<br>0_._01<br>29_._1<br>0_._01<br>1<br>0_._02<br>_R_SN2<br>10_−_4<br>124_._3<br>358_._06<br>2_,_667_,_570<br>1439_._55<br>134_._4<br>24_._52<br>1_,_201_,_306<br>728_._03<br>246_._8<br>1732_._69<br>_>_591_,_537<br>_>_1800<br>10_−_3<br>123_._2<br>361_._40<br>2_,_899_,_113<br>1401_._15<br>**117**_._**6**<br>19_._23<br>744_,_149<br>377_._74<br>243_._7<br>1809_._90<br>_>_503_,_628<br>_>_1800<br>10_−_2<br>**122**_._**1**<br>349_._86<br>2_,_418_,_068<br>1253_._44<br>119_._8<br>21_._80<br>356_,_002<br>221_._30<br>240_._5<br>1475_._18<br>_>_637_,_046<br>_>_1800<br>_R_LP<br>10_−_4<br>125_._0<br>0_._00<br>13_,_136<br>8_._33<br>149_._9<br>0_._00<br>13_,_686<br>9_._17<br>250_._0<br>**0**_._**00**<br>10_,_110<br>20_._54<br>10_−_3<br>124_._9<br>**0**_._**00**<br>96_,_528<br>41_._03<br>149_._9<br>0_._00<br>13_,_889<br>6_._00<br>249_._8<br>0_._00<br>200_,_495<br>231_._59<br>10_−_2<br>124_._9<br>**0**_._**00**<br>**614**_,_**044**<br>**224**_._**40**<br>149_._8<br>0_._00<br>48_,_391<br>30_._20<br>249_._8<br>**0**_._**00**<br>**5**_,_**465**<br>**12**_._**43**<br>_R_BW +_R_LP<br>10_−_4<br>**66**_._**2**<br>**0**_._**01**<br>**5**<br>**0**_._**14**<br>71_._9<br>0_._04<br>3<br>0_._31<br>**87**_._**6**<br>0_._01<br>**155**<br>**0**_._**52**<br>10_−_3<br>30_._9<br>0_._00<br>1<br>0_._01<br>34_._9<br>0_._00<br>1<br>0_._04<br>**45**_._**0**<br>**0**_._**00**<br>**1**<br>0_._06<br>10_−_2<br>13_._0<br>0_._00<br>1<br>0_._00<br>14_._4<br>0_._00<br>1<br>0_._01<br>**31**_._**0**<br>0_._60<br>18_,_290<br>90_._03|
|---|---|---|---|



23 

## **6.3 Optimization over Quantile Neural Networks** 

The previous section studied the proposed regularizers on single-output surrogate models, where the downstream MILP simply minimizes the predicted output. However, many practical applications involve more complicated settings, e.g., surrogate models can have multiple outputs and the downstream optimization problem can involve a more complex objective. To encompass some of these elements, we consider quantile neural networks (QNNs) applied to two-stage stochastic programming (2SP), an important setting in which QNN surrogates have been used to approximate the distribution of the second-stage ‘recourse’ function [57, 58]. QNNs are multi-output models that predict specific quantiles of a target distribution, enabling them to estimate uncertainty and produce prediction intervals instead of only point predictions. 

We consider the capacitated facility location problem (CFLP), a classical 2SP in which binary first-stage decisions _y ∈{_ 0 _,_ 1 _}[n][f]_ determine which of _nf_ facilities to open, and the second-stage recourse allocates facility capacity to _nc_ customer demands that are revealed as random scenarios _ξ_ . Patel et al. [24] study the problem using standard NN surrogates for the second-stage objective, and Alc´antara et al. [57] use QNN surrogates to enable risk-aware, distributional modeling. Interestingly, Liu et al. [73] train input-convex NN surrogates to accelerate the downstream 2SP problem, which can then be formulated as an LP. Following the setup of Alc´antara et al. [57], we train a QNN surrogate _fθ_ : _{_ 0 _,_ 1 _}[n][f] →_ R _[K]_ that, given a first-stage decision _y_ , predicts _K_ = 50 quantiles of the second-stage cost distribution at equally spaced intervals. The QNN is trained by minimizing the pinball (quantile regression) loss: 

**==> picture [372 x 31] intentionally omitted <==**

where _vi_ is the realized second-stage cost for the _i_ -th training sample, generated by solving the recourse problem for a random demand scenario. 

Once trained, the QNN surrogate replaces the expensive recourse computation in the first-stage optimization and enables distributional predictions such as Conditional Value-at-Risk (CVaR). The resulting 2SP is formulated as a MILP that minimizes a mean and/or CVaR-based objective over the predicted quantile outputs: 

**==> picture [351 x 30] intentionally omitted <==**

where _cf_ is the vector of first-stage costs, _λ_ 2SP is the risk-aversion parameter, _T_ = _{k_ : _τk ≥ α}_ is the set of tail quantile indices, and _α_ is the CVaR confidence level. This formulation includes both binary decision variables and the MILP encoding of the trained QNN, making tractability of the overall problem highly dependent on the structural properties of the surrogate model. We refer the reader to Alc´antara et al. [57] for further details on the QNN-based 2SP framework. 

## **6.3.1 Training Results** 

We study the `CFLP 50` ~~`5`~~ `0` instance ( _nf_ = _nc_ = 50), following the instance generation procedure of Patel et al. [24], which is based in turn on Cornu´ejols et al. [74]. We also consider an extended `CFLP` ~~`7`~~ `5` ~~`7`~~ `5` instance ( _nf_ = _nc_ = 75), generated using the same procedure. The training dataset for each consists of 20,000 samples, with samples generated by drawing a random first-stage decision _y_ and solving the second-stage recourse for a random demand scenario. For the `CFLP` ~~`5`~~ `0` ~~`5`~~ `0` and `CFLP` ~~`7`~~ `5` ~~`7`~~ `5` problems, 250 and 570 samples respectively reached the 600 s time limit, but feasible solutions were found for all of them. Data are split 80/20 into training and validation sets, normalized, and models are trained for 200 epochs with Adam. 

We consider three architectures with _{_ 2 _,_ 3 _,_ 5 _}_ hidden layers of 25 neurons each, denoted respectively as `X-25-25-50` , `X-25-25-25-50` , `X-25-25-25-25-25-50` . The input dimension equals _nf_ , and output dimensions is 50, corresponding to 50 quantiles. Each configuration is trained over 20 random seeds and tested in downstream MILPs as in (36). We omit the _R_ SN2 regularizer for brevity, as it failed to produce tractable models in nearly all of the settings above, but reintroduce 

24 

**Table 9** Pinball loss (mean _±_ std over seeds, scaled _×_ 10[2] ) on `cflp` ~~`5`~~ `0` ~~`5`~~ `0` for each regularizer and architecture. Values are computed on the held-out test set. **Bold** indicates the lowest pinball loss within each _λ_ level per architecture, excluding shaded rows. Shaded entries exceed the unregularized baseline by more than 10%, indicating that the regularizer has degraded predictive quality. 

||regularizer<br>_λ_<br>`50-25-25-50`<br>`50-25-25-25-50`<br>`50-25-25-25-25-25-50`<br>(_×_102)<br>(_×_102)<br>(_×_102)|
|---|---|
||None<br>—<br>3_._03_±_0_._02<br>3_._02_±_0_._03<br>3_._06_±_0_._00|
||_R_L1<br>10_−_4<br>**2**_._**94**_±_**0**_._**01**<br>**2**_._**89**_±_**0**_._**01**<br>**2**_._**87**_±_**0**_._**01**<br>10_−_3<br>3_._43_±_0_._14<br>3_._26_±_0_._15<br>3_._20_±_0_._15<br>10_−_2<br>30_._77_±_6_._80<br>32_._54_±_7_._27<br>34_._60_±_7_._07|
||_R_L2<br>10_−_4<br>2_._99_±_0_._02<br>3_._00_±_0_._02<br>2_._97_±_0_._02<br>10_−_3<br>3_._15_±_0_._02<br>3_._06_±_0_._01<br>3_._02_±_0_._01<br>10_−_2<br>22_._82_±_3_._54<br>14_._38_±_9_._06<br>4_._77_±_0_._02|
||_R_BW<br>10_−_4<br>3_._01_±_0_._02<br>3_._00_±_0_._03<br>2_._98_±_0_._02<br>10_−_3<br>**2**_._**97**_±_**0**_._**02**<br>**2**_._**94**_±_**0**_._**02**<br>2_._93_±_0_._02<br>10_−_2<br>**2**_._**92**_±_**0**_._**01**<br>**2**_._**88**_±_**0**_._**01**<br>2_._86_±_0_._01<br>_R_SN<br>10_−_4<br>3_._02_±_0_._02<br>3_._01_±_0_._03<br>2_._99_±_0_._03<br>10_−_3<br>3_._01_±_0_._02<br>2_._99_±_0_._02<br>2_._95_±_0_._02<br>10_−_2<br>2_._93_±_0_._02<br>2_._90_±_0_._02<br>**2**_._**86**_±_**0**_._**01**<br>_R_LP<br>10_−_4<br>3_._04_±_0_._02<br>3_._09_±_0_._03<br>3_._07_±_0_._04<br>10_−_3<br>3_._08_±_0_._03<br>3_._11_±_0_._04<br>3_._33_±_0_._11<br>10_−_2<br>3_._08_±_0_._03<br>3_._12_±_0_._05<br>8_._76_±_5_._59|
||_R_BW +_R_LP<br>10_−_4<br>3_._03_±_0_._02<br>3_._04_±_0_._02<br>3_._00_±_0_._02<br>10_−_3<br>2_._99_±_0_._02<br>2_._98_±_0_._02<br>**2**_._**91**_±_**0**_._**03**<br>10_−_2<br>2_._92_±_0_._01<br>23_._05_±_8_._71<br>23_._18_±_10_._39|



the L2 regularizer, as its performance in multi-output settings has not been studied. The multioutput LP relaxation gap regularizer _R_ LP uses random projections with non-negative directions, consistent with the non-negative weights in the downstream objective. The computational overheads of the proposed regularizers are independent of the specific training data; we observe costs largely similar to what was reported in Table 3 for the benchmark functions. 

Tables 9–10 report the test pinball loss (scaled _×_ 10[2] ) for each regularizer and architecture combination. The unregularized baseline achieves a pinball loss of approximately 3 _._ 0 and 2 _._ 5 for the two problems respectively (scaled) across all architectures. The proposed regularizers _R_ BW and _R_ SN preserve or even slightly improve prediction quality at all regularization strengths considered; on both problems _R_ BW at _λ_ = 10 _[−]_[2] achieves the lowest pinball loss across architectures. In contrast, the shrinkage regularizers again suffer severe prediction degradation at higher regularization strengths, supporting our observation that these regularizers must be tuned and deployed more cautiously in practice. For example, _R_ L1 at _λ_ = 10 _[−]_[2] inflates the pinball loss by an order of magnitude (to 30–35 on both problems), indicating that the network has collapsed to a near-constant function. _R_ L2 exhibits similar degradation at _λ_ = 10 _[−]_[2] . Any downstream MILP tractability “improvements” at these settings are therefore not attributable to the regularizer, but rather to the trivialization of the surrogate model. 

In this setting a trained QNN surrogate is usable across downstream instances, e.g., Alc´antara et al. [57] perform sensitivity studies across various levels of _λ_ 2SP and _α_ , while Ghilardi et al. [58] consider varying first-stage costs _cf_ . Given the lack of an obvious single objective, we consider the pinball loss (35) as a general metric of model decision quality across downstream optimization tasks: rows where the test pinball loss exceeds the baseline by more than 10% are shaded in grey in Tables 9–10. Moreover, they are excluded from best-value comparisons in Tables 11–12. 

## **6.3.2 Optimization Results** 

Tables 11–12 report the same four MILP tractability metrics for the 2SP formulations (36): unstable neuron count _|U|_ , LP relaxation gap, branch-and-bound nodes, and wall-clock MILP solve time. We arbitrarily select _λ_ 2SP = 0 _._ 1 for the risk-aversion parameter and _α_ = 0 _._ 9 is the CVaR confidence 

25 

**Table 10** Pinball loss (mean _±_ std over seeds, scaled _×_ 10[2] ) on `cflp` ~~`7`~~ `5 75` for each regularizer and architecture. Values are computed on the held-out test set. **Bold** indicates the lowest pinball loss within each _λ_ level per architecture, excluding shaded rows. Shaded entries exceed the unregularized baseline by more than 10%, indicating that the regularizer has degraded predictive quality. 

||regularizer<br>_λ_<br>`75-25-25-50`<br>`75-25-25-25-50`<br>`75-25-25-25-25-25-50`<br>(_×_102)<br>(_×_102)<br>(_×_102)|
|---|---|
||None<br>—<br>2_._53_±_0_._04<br>2_._49_±_0_._03<br>2_._47_±_0_._02|
||_R_L1<br>10_−_4<br>**2**_._**40**_±_**0**_._**01**<br>**2**_._**35**_±_**0**_._**00**<br>**2**_._**33**_±_**0**_._**01**<br>10_−_3<br>2_._88_±_0_._12<br>2_._75_±_0_._18<br>2_._58_±_0_._05<br>10_−_2<br>30_._51_±_6_._23<br>30_._81_±_6_._63<br>34_._69_±_6_._43|
||_R_L2<br>10_−_4<br>2_._49_±_0_._02<br>2_._48_±_0_._02<br>2_._44_±_0_._02<br>10_−_3<br>2_._60_±_0_._02<br>2_._50_±_0_._01<br>2_._46_±_0_._01<br>10_−_2<br>24_._11_±_0_._11<br>22_._10_±_5_._75<br>15_._82_±_9_._60|
||_R_BW<br>10_−_4<br>2_._51_±_0_._03<br>2_._46_±_0_._02<br>2_._43_±_0_._02<br>10_−_3<br>**2**_._**44**_±_**0**_._**02**<br>**2**_._**40**_±_**0**_._**01**<br>2_._37_±_0_._02<br>10_−_2<br>**2**_._**37**_±_**0**_._**01**<br>**2**_._**34**_±_**0**_._**01**<br>2_._33_±_0_._01<br>_R_SN<br>10_−_4<br>2_._52_±_0_._03<br>2_._48_±_0_._03<br>2_._45_±_0_._03<br>10_−_3<br>2_._50_±_0_._03<br>2_._44_±_0_._03<br>2_._40_±_0_._02<br>10_−_2<br>2_._39_±_0_._02<br>2_._34_±_0_._02<br>**2**_._**31**_±_**0**_._**01**<br>_R_LP<br>10_−_4<br>2_._55_±_0_._03<br>2_._56_±_0_._02<br>2_._50_±_0_._03<br>10_−_3<br>2_._61_±_0_._04<br>2_._60_±_0_._04<br>2_._79_±_0_._04<br>10_−_2<br>2_._60_±_0_._06<br>2_._65_±_0_._05<br>7_._50_±_4_._08|
||_R_BW +_R_LP<br>10_−_4<br>2_._53_±_0_._02<br>2_._50_±_0_._03<br>2_._47_±_0_._03<br>10_−_3<br>2_._47_±_0_._02<br>2_._44_±_0_._02<br>**2**_._**36**_±_**0**_._**05**<br>10_−_2<br>3_._55_±_5_._23<br>26_._95_±_0_._76<br>28_._69_±_0_._75|



level. The unregularized baseline again becomes increasingly intractable with network depth, with mean solve times in the `cflp 50 50` case study ranging from 2.3 s (2 hidden layers) to 702 s (5 hidden layers) and node counts growing from 5 _,_ 235 to 270 _,_ 161. These are magnified in the `cflp` ~~`7`~~ `5` ~~`7`~~ `5` case study to 4.6 s (2 hidden layers) and 1395 s (5 hidden layers), with node counts growing from 9 _,_ 061 to 1 _,_ 799 _,_ 004, reflecting the more challenging response surface learned by the surrogate. 

In these two settings, we again observe that the L1 regularizer can be effective at low weights ( _λ_ = 10 _[−]_[4] ), but must be deployed cautiously as it can quickly degrade model prediction performance. In contrast, the bound-width regularizer _R_ BW improves downstream performance with less sensitivity to tuning. For example, at _λ_ = 10 _[−]_[2] _R_ BW reduces MILP solve times by 1–3 orders of magnitude across all architectures while maintaining the lowest pinball loss in both problem settings (Tables 9–10). On the deepest architectures, the MILP solution time drops several orders of magnitude, from 702 s and 1395 s, on the two problems respectively, to less than 1 s, with unstable neurons reduced from 125 to approximately 20. The stability regularizer _R_ SN at _λ_ = 10 _[−]_[2] achieves similar improvements without sensitivity to regularization weight tuning, reducing the 5-layer solve time to _≈_ 0.35 s with 25–30 unstable neurons, though we again observe its LP gap reduction is slightly less aggressive. 

We again observe the LP-based regularizer _R_ LP effectively reduces the LP gap but, as in the benchmark experiments, does not reduce the number of unstable neurons by itself. On the 2- and 3-layer architectures, this still yields meaningful acceleration of the downstream MILP (e.g., 0.56 s and 1.47 s vs. baselines of 2.3 s and 18 s in the `cflp` ~~`5`~~ `0` ~~`5`~~ `0` setting). However, on the deepest architectures _R_ LP alone is insufficient, matching our observations from the Ackley function above. Here, despite reducing the LP gap from 2 _._ 5 _×_ 10[6] to 1 _._ 4 _×_ 10[4] at _λ_ = 10 _[−]_[2] in the `cflp` ~~`5`~~ `0` ~~`5`~~ `0` setting), the network retains all 125 unstable neurons and still requires _>_ 100 s to solve. Notably, _R_ LP at _λ_ = 10 _[−]_[2] also degrades prediction quality on the 5-layer architecture (pinball loss 8 _._ 76 vs. baseline 3 _._ 06), perhaps reflecting the difficulty of the multi-output LP regularization at strong regularization strengths. The same trends can be seen in the larger problem. Alternatively, this could simply be an effect of the regularizer strength, as the shrinkage regularizers also degrade prediction quality at this weight. 

26 

The combined regularizer _R_ BW + _R_ LP at _λ_ = 10 _[−]_[3] achieves a particularly effective balance of the above effects in both problem settings. For the 5-layer surrogate model architecture, it reduces _|U|_ from 125 to _>_ 50 at a weight of _λ_ = 10 _[−]_[3] , reduces the LP gap from _O_ (10[6] ) to approximately 400 on both problems, and produces a MILP solvable in under 1 s without degrading (even improving) prediction quality. Across almost all architectures the combined regularizer consistently matches or improves upon the individual components, confirming the complementarity strengths of targeting both neuron stability (via _R_ BW) and relaxation tightness (via _R_ LP). 

**==> picture [384 x 193] intentionally omitted <==**

**----- Start of picture text -----**<br>
3-layer  (50-25-25-25-50) 5-layer  (50-25-25-25-25-25-50)<br>1000<br>100<br>10<br>1<br>0<br>10 20 30 40 10 20 30 40<br>Pinball loss (×10 [2] ) Pinball loss (×10 [2] )<br>None ℓ1 ℓ2 RBW RSN RLP RBW + RLP Degraded (pinball  > 10% above baseline)<br>2SP MILP solve time (s)<br>**----- End of picture text -----**<br>


**Fig. 5** Tradeoff between solution time of downstream MILP performance and pinball loss for QNNs in stochastic programming setting. Each plotted marker shows mean performance for a different regularizer and weight. 

## _**Comparison at equal prediction quality**_ 

The shading convention in Tables 9–10 and Tables 11–12 exposes a critical confound in na¨ıve comparisons. As trends are largely consistent between the two problem settings, we focus our discussion on the `cflp 50` ~~`5`~~ `0` setting here. We observe that _R_ L1 at _λ_ = 10 _[−]_[2] achieves the fastest MILP solve times in raw numbers (0.03 s), but this is only because the network has collapsed. In fact, Table 9 shows the pinball loss is 10 _×_ worse than the baseline. At comparable prediction quality, e.g., comparing _R_ L1 at _λ_ = 10 _[−]_[4] (pinball 2 _._ 89, MILP time 0.54 s on the 3-layer model) against _R_ BW at _λ_ = 10 _[−]_[2] (pinball 2 _._ 88, MILP time 0.10 s), the proposed relaxation-informed regularizer achieves a 5 _×_ faster solve with 3 _×_ fewer unstable neurons for a QNN surrogate model the same predictive quality. This highlights the importance of accounting for prediction quality when evaluating MILP tractability improvements. 

Figure 5 compares the pinball loss achieved by various regularizer configurations against the downstream MILP solve times in this setting, allowing us to visualize this tradeoff. Many points are clustered on the lefthand side of both plots (pinball loss close to the unnormalized baseline of _≈_ 3). In this region of regularizers that maintain prediction quality, training with relaxation-aware regularizers _R_ BW, _R_ SN, and the combined regularizer sit at the bottom (lowest MILP solve times), forming the Pareto front. Improvements are most dramatic for the deeper QNN surrogate models. The shrinkage regularizers (and sometimes the combined regularizer) can produce model-collapse configurations where the regularizer dominates, resulting in a higher pinball loss values. These configurations are indicated using faded markers. 

## _**Limitations.**_ 

This work focuses on the big-M MILP formulation of ReLU networks; tailored regularizers for other formulations (e.g., ideal formulations [35] or partition-based formulations [36]) and other activation functions remain to be established. The bound-based regularizers _R_ BW and _R_ SN rely on IBP, which can produce increasingly loose bounds in deeper networks due to the recursive 

27 

over-approximation discussed in Section 2.4. The LP-based regularizer _R_ LP incurs a non-negligible training overhead (5–20 _×_ , Table 3), which scales with network size and limits its practicality for very large models, though this is less relevant for the surrogate model setting and cost is incurred only once during training. Furthermore, _R_ LP targets pointwise relaxation gaps, and global tightening is not guaranteed. 

## **7 Conclusions** 

This paper introduced a family of relaxation-informed regularization strategies that target the downstream tractability of neural network surrogate models during training. Two bound propagation-based regularizers, _R_ BW (bound-width) and _R_ SN (stable-neuron), penalize the big-M constants and the number of unstable neurons, respectively, through automatic differentiation of the bounds computation. An LP relaxation gap regularizer _R_ LP directly targets the continuous relaxation tightness, with gradients derived from dual variables via the envelope theorem for parametric linear programs. Proposition 3 shows that combining _R_ BW and _R_ LP approximates the full total derivative of the LP gap with respect to the network parameters, capturing both the direct constraint sensitivity and the indirect big-M sensitivity. 

Computational experiments on benchmark surrogate functions demonstrated that the proposed regularizers can reduce MILP solve times by up to four orders of magnitude while maintaining competitive modeling performance. On the two-stage stochastic programming case study with quantile neural networks, training with the bound-width regularizer _R_ BW could reduce the MILP solve time on the deepest architecture from over 700 s to under 1 s without degrading prediction quality, and the combined regularizer achieved similar acceleration, with complementary reductions in both unstable neuron count and LP gap. These results highlight an important practical consideration: classical shrinkage regularizers ( _R_ L1, _R_ L2) can improve tractability at strong regularization weights, but this often requires tuning to avoid a collapse in predictive quality (rather than genuine improvement of the MILP formulation). 

Several directions for future work are worth noting. The LP-based regularizer incurs a training overhead of approximately 5–20 _×_ due to repeated LP solves; exploiting GPU-based LP solvers [71, 72] could substantially reduce this cost and integrate with neural network training pipelines. Moreover, incorporating tighter bound propagation schemes (e.g., optimization-based bound tightening) as differentiable regularizers could yield further improvements, particularly for deeper architectures. Finally, extending the framework to other activation functions and to more complex downstream formulations beyond the big-M MILP remains an open direction. 

## **Acknowledgements** 

Funding from the BASF/Royal Academy of Engineering Senior Research Fellowship is gratefully acknowledged. 

## **Data Availability Statement** 

No new data were created for synthetic benchmarks. Data for stochastic programming applications were generated following `https://github.com/khalil-research/Neur2SP` . 

28 

|**Table 11** Results on the `cflp`<br>`50`<br>`50` QNN surrogate 2SP across architectures and regularizers. Each regularizer family shows three rows for<br>_λ ∈{_10_−_4_,_10_−_3_,_10_−_2_}_. _|U|_: mean number of unstable neurons; LP gap: mean 2SP LP relaxation gap; MILP nodes/time: mean<br>branch-and-bound nodes and wall-clock time (s) for the 2SP mean+CVaR MILP. **Bold** values indicate the best (lowest) result across all<br>regularizers at each _λ_ level per architecture; ties are all bolded.<br>Shaded entries indicate regularizers whose test pinball loss exceeds the<br>unregularized baseline by more than 10%, refecting degraded predictive quality; such confgurations are excluded from best-value consideration.|regularizer<br>_λ_<br>`50-25-25-50`<br>`50-25-25-25-50`<br>`50-25-25-25-25-25-50`<br>_|U|_<br>LP gap<br>MILP<br>_|U|_<br>LP gap<br>MILP<br>_|U|_<br>LP gap<br>MILP<br>nodes<br>time<br>nodes<br>time<br>nodes<br>time|None<br>—<br>50_._0<br>44005_._06<br>5_,_235<br>2_._31<br>75_._0<br>196857_._98<br>11_,_982<br>18_._16<br>125_._0<br>2469369_._64<br>270_,_161<br>701_._74|_R_L1<br>10_−_4<br>**32**_._**2**<br>**319**_._**71**<br>**1**<br>**0**_._**39**<br>**55**_._**0**<br>**593**_._**19**<br>**4**<br>**0**_._**54**<br>103_._7<br>**7634**_._**47**<br>**44**<br>**0**_._**61**<br>10_−_3<br>32_._0<br>110_._63<br>12<br>0_._29<br>**46**_._**8**<br>**404**_._**72**<br>145<br>0_._29<br>75_._7<br>829_._68<br>52<br>0_._54<br>10_−_2<br>32_._4<br>37_._42<br>1<br>0_._03<br>37_._0<br>102_._94<br>1<br>0_._03<br>50_._0<br>222_._49<br>384<br>1_._39<br>_R_L2<br>10_−_4<br>46_._1<br>11756_._43<br>456<br>0_._63<br>70_._9<br>41684_._35<br>7_,_079<br>5_._14<br>117_._5<br>237213_._23<br>36_,_561<br>45_._49<br>10_−_3<br>40_._2<br>2326_._55<br>**1**<br>**0**_._**25**<br>54_._3<br>6566_._90<br>**1**<br>**0**_._**20**<br>79_._8<br>11557_._11<br>56<br>0_._68<br>10_−_2<br>44_._5<br>744_._48<br>1<br>0_._06<br>60_._9<br>1639_._93<br>313<br>0_._62<br>97_._1<br>3470_._08<br>4_,_049<br>2_._41<br>_R_BW<br>10_−_4<br>48_._1<br>26486_._03<br>3_,_621<br>1_._71<br>69_._8<br>75976_._64<br>9_,_747<br>10_._53<br>108_._2<br>217288_._14<br>30_,_437<br>66_._05<br>10_−_3<br>38_._9<br>4776_._56<br>193<br>0_._49<br>57_._4<br>17001_._69<br>1_,_645<br>1_._45<br>77_._2<br>41766_._83<br>5_,_948<br>5_._52<br>10_−_2<br>16_._4<br>207_._46<br>**1**<br>**0**_._**08**<br>**17**_._**8**<br>**425**_._**11**<br>**1**<br>**0**_._**10**<br>**18**_._**9**<br>**756**_._**10**<br>4<br>**0**_._**19**<br>_R_SN<br>10_−_4<br>46_._9<br>27429_._07<br>3_,_504<br>1_._43<br>69_._5<br>97827_._09<br>9_,_981<br>12_._62<br>107_._6<br>411523_._37<br>41_,_510<br>64_._14<br>10_−_3<br>40_._7<br>9861_._86<br>1_,_458<br>1_._14<br>59_._0<br>30793_._00<br>4_,_244<br>3_._22<br>85_._0<br>84640_._76<br>7_,_524<br>11_._94<br>10_−_2<br>21_._3<br>935_._90<br>**1**<br>0_._17<br>25_._4<br>1349_._28<br>13<br>0_._23<br>29_._9<br>1201_._37<br>**1**<br>0_._34<br>_R_LP<br>10_−_4<br>49_._8<br>4410_._45<br>124<br>0_._65<br>74_._5<br>18490_._36<br>2_,_325<br>2_._10<br>125_._0<br>22936_._09<br>84_,_133<br>104_._38<br>10_−_3<br>48_._8<br>673_._95<br>746<br>0_._60<br>74_._7<br>1171_._38<br>5_,_972<br>3_._47<br>125_._0<br>49816_._20<br>365_,_632<br>239_._22<br>10_−_2<br>49_._1<br>347_._71<br>1_,_057<br>0_._56<br>74_._8<br>522_._99<br>3_,_189<br>1_._47<br>125_._0<br>14303_._72<br>64_,_183<br>49_._79<br>_R_BW +_R_LP<br>10_−_4<br>47_._7<br>2390_._24<br>26<br>0_._58<br>68_._6<br>10001_._84<br>1_,_525<br>1_._60<br>**101**_._**6**<br>27172_._28<br>8_,_130<br>14_._94<br>10_−_3<br>**35**_._**4**<br>**355**_._**61**<br>311<br>0_._52<br>48_._8<br>429_._38<br>146<br>0_._55<br>**48**_._**4**<br>**397**_._**85**<br>**1**<br>**0**_._**39**<br>10_−_2<br>**11**_._**8**<br>**78**_._**19**<br>**1**<br>0_._15<br>8_._2<br>83_._49<br>1<br>0_._02<br>22_._9<br>182_._73<br>1<br>0_._06|
|---|---|---|---|



29 

|**Table 12** Results on the `cflp`<br>`75`<br>`75` QNN surrogate 2SP across architectures and regularizers. Each regularizer family shows three rows for<br>_λ ∈{_10_−_4_,_10_−_3_,_10_−_2_}_. _|U|_: mean number of unstable neurons; LP gap: mean 2SP LP relaxation gap; MILP nodes/time: mean branch-and-bound<br>nodes and wall-clock time (s) for the 2SP mean+CVaR MILP. **Bold** values indicate the best (lowest) result across all regularizers at each _λ_ level<br>per architecture; ties are all bolded.<br>Shaded entries indicate regularizers whose test pinball loss exceeds the unregularized baseline by more than<br>10%, refecting degraded predictive quality; such confgurations are excluded from best-value consideration.|regularizer<br>_λ_<br>`75-25-25-50`<br>`75-25-25-25-50`<br>`75-25-25-25-25-25-50`<br>_|U|_<br>LP gap<br>MILP<br>_|U|_<br>LP gap<br>MILP<br>_|U|_<br>LP gap<br>MILP<br>nodes<br>time<br>nodes<br>time<br>nodes<br>time|None<br>—<br>50_._0<br>76961_._06<br>9_,_061<br>4_._64<br>75_._0<br>332696_._67<br>13_,_755<br>20_._16<br>125_._0<br>3562051_._76<br>1_,_799_,_004<br>1395_._20|_R_L1<br>10_−_4<br>**32**_._**2**<br>**1131**_._**16**<br>**84**<br>**0**_._**27**<br>**53**_._**9**<br>**2322**_._**02**<br>**121**<br>**0**_._**48**<br>105_._0<br>**12883**_._**05**<br>**500**<br>**1**_._**05**<br>10_−_3<br>34_._2<br>193_._47<br>1<br>0_._34<br>47_._9<br>1211_._57<br>316<br>0_._50<br>76_._8<br>2946_._37<br>396<br>0_._81<br>10_−_2<br>31_._2<br>0_._00<br>1<br>0_._03<br>38_._2<br>41_._94<br>1<br>0_._03<br>27_._2<br>0_._00<br>1<br>0_._02<br>_R_L2<br>10_−_4<br>48_._4<br>26889_._45<br>1_,_904<br>1_._27<br>72_._9<br>85918_._82<br>9_,_295<br>8_._19<br>120_._0<br>495568_._32<br>53_,_254<br>62_._58<br>10_−_3<br>39_._7<br>4500_._89<br>**1**<br>**0**_._**28**<br>55_._4<br>10438_._69<br>**1**<br>**0**_._**24**<br>83_._5<br>25823_._26<br>967<br>1_._34<br>10_−_2<br>44_._9<br>1245_._53<br>1<br>0_._09<br>63_._9<br>1895_._88<br>147<br>0_._24<br>92_._8<br>2950_._60<br>862<br>0_._92<br>_R_BW<br>10_−_4<br>48_._5<br>46063_._34<br>8_,_071<br>4_._28<br>71_._0<br>140223_._45<br>9_,_534<br>11_._02<br>107_._6<br>358786_._38<br>71_,_025<br>91_._14<br>10_−_3<br>40_._8<br>10063_._15<br>869<br>0_._84<br>57_._0<br>29386_._24<br>3_,_820<br>2_._06<br>69_._5<br>59873_._45<br>5_,_001<br>4_._95<br>10_−_2<br>**15**_._**6**<br>1342_._98<br>**7**<br>**0**_._**17**<br>**16**_._**2**<br>1549_._56<br>**1**<br>**0**_._**20**<br>**20**_._**9**<br>**1689**_._**45**<br>**1**<br>**0**_._**21**<br>_R_SN<br>10_−_4<br>48_._1<br>54656_._72<br>8_,_502<br>3_._67<br>68_._5<br>177112_._92<br>9_,_151<br>11_._40<br>107_._0<br>649873_._14<br>82_,_863<br>135_._28<br>10_−_3<br>41_._6<br>24314_._73<br>3_,_984<br>2_._18<br>57_._7<br>55394_._37<br>7_,_255<br>4_._54<br>81_._2<br>118008_._83<br>6_,_244<br>16_._25<br>10_−_2<br>20_._1<br>1697_._08<br>395<br>0_._36<br>23_._2<br>2063_._29<br>45<br>0_._38<br>26_._9<br>2162_._87<br>1<br>0_._35<br>_R_LP<br>10_−_4<br>49_._8<br>7710_._15<br>1_,_626<br>1_._20<br>74_._9<br>34157_._12<br>6_,_657<br>5_._73<br>125_._0<br>30153_._51<br>146_,_887<br>134_._11<br>10_−_3<br>48_._6<br>1132_._86<br>4_,_988<br>2_._15<br>74_._7<br>2150_._41<br>50_,_828<br>16_._63<br>125_._0<br>27486_._72<br>285_,_447<br>260_._60<br>10_−_2<br>49_._5<br>**363**_._**43**<br>11_,_305<br>1_._91<br>74_._9<br>**1112**_._**30**<br>29_,_829<br>8_._05<br>125_._0<br>23664_._74<br>162_,_029<br>109_._02<br>_R_BW +_R_LP<br>10_−_4<br>47_._6<br>4314_._56<br>1_,_324<br>1_._13<br>68_._5<br>20778_._39<br>3_,_851<br>2_._63<br>**100**_._**8**<br>37881_._30<br>7_,_448<br>13_._69<br>10_−_3<br>**35**_._**9**<br>**556**_._**51**<br>1_,_262<br>0_._90<br>**50**_._**1**<br>**720**_._**99**<br>491<br>0_._86<br>**43**_._**0**<br>**432**_._**37**<br>**8**<br>**0**_._**33**<br>10_−_2<br>10_._1<br>67_._38<br>43<br>0_._23<br>8_._2<br>0_._00<br>0<br>0_._00<br>16_._0<br>0_._00<br>0<br>0_._01|
|---|---|---|---|



30 

## **References** 

- [1] Bertsimas, D., Margaritis, G.: Global optimization: a machine learning approach. Journal of Global Optimization **91** (1), 1–37 (2025) 

- [2] Bradley, W., Kim, J., Kilwein, Z., Blakely, L., Eydenberg, M., Jalvin, J., Laird, C., Boukouvala, F.: Perspectives on the integration between first-principles and data-driven modeling. Computers & Chemical Engineering **166** , 107898 (2022) 

- [3] Misener, R., Biegler, L.: Formulating data-driven surrogate models for process optimization. Computers & Chemical Engineering **179** , 108411 (2023) 

- [4] Grimstad, B., Andersson, H.: ReLU networks as surrogate models in mixed-integer linear programs. Computers & Chemical Engineering **131** , 106580 (2019) 

- [5] Huchette, J., Mu˜noz, G., Serra, T., Tsay, C.: When deep learning meets polyhedral theory: A survey. INFORMS Journal on Computing (2026) 

- [6] Plate, C., Hahn, M., Klimek, A., Ganzer, C., Sundmacher, K., Sager, S.: An analysis of optimization problems involving relu neural networks. Optimization and Engineering, 1–33 (2026) 

- [7] Botoeva, E., Kouvaros, P., Kronqvist, J., Lomuscio, A., Misener, R.: Efficient verification of relu-based neural networks via dependency analysis. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, pp. 3291–3299 (2020) 

- [8] R¨ossig, A., Petkovic, M.: Advances in verification of ReLU neural networks. Journal of Global Optimization **81** (1), 109–152 (2021) 

- [9] Sosnin, P., M¨uller, M.N., Baader, M., Tsay, C., Wicker, M.: Certified robustness to data poisoning in gradient-based training. arXiv preprint arXiv:2406.05670 (2024) 

- [10] Sosnin, P., Knapp, J., Kennedy, F., Collyer, J., Tsay, C.: Exact certification of data-poisoning attacks using mixed-integer programming. arXiv preprint arXiv:2602.16944 (2026) 

- [11] Kanamori, K., Takagi, T., Kobayashi, K., Ike, Y., Uemura, K., Arimura, H.: Ordered counterfactual explanation by mixed-integer linear optimization. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, pp. 11564–11574 (2021) 

- [12] Tsiourvas, A., Sun, W., Perakis, G.: Manifold-aligned counterfactual explanations for neural networks. In: International Conference on Artificial Intelligence and Statistics, pp. 3763–3771 (2024). PMLR 

- [13] Burtea, R., Tsay, C.: Constrained continuous-action reinforcement learning for supply chain inventory management. Computers & Chemical Engineering **181** , 108518 (2024) 

- [14] Ryu, M., Chow, Y., Anderson, R., Tjandraatmadja, C., Boutilier, C.: CAQL: Continuous action Q-learning. arXiv preprint arXiv:1909.12397 (2019) 

- [15] Benbaki, R., Chen, W., Meng, X., Hazimeh, H., Ponomareva, N., Zhao, Z., Mazumder, R.: Fast as chita: Neural network pruning with combinatorial optimization. In: International Conference on Machine Learning, pp. 2031–2049 (2023). PMLR 

- [16] Serra, T., Yu, X., Kumar, A., Ramalingam, S.: Scaling up exact neural network compression by ReLU stability. Advances in Neural Information Processing Systems **34** , 27081–27093 (2021) 

- [17] Perakis, G., Tsiourvas, A.: Optimizing objective functions from trained ReLU neural networks via sampling. arXiv preprint arXiv:2205.14189 (2022) 

31 

- [18] Tong, J., Cai, J., Serra, T.: Optimization over trained neural networks: Taking a relaxing walk. In: International Conference on the Integration of Constraint Programming, Artificial Intelligence, and Operations Research, pp. 221–233 (2024). Springer 

- [19] Tong, J., Zhu, Y., Serra, T., Burer, S.: Optimization over trained neural networks: Going large with gradient-based algorithms. arXiv preprint arXiv:2512.24295 (2025) 

- [20] Fajemisin, A.O., Maragno, D., Hertog, D.: Optimization with constraint learning: A framework and survey. European Journal of Operational Research **314** (1), 1–14 (2024) 

- [21] Maragno, D., Wiberg, H., Bertsimas, D., Birbil, S¸.I.,[˙] Hertog, D., Fajemisin, A.O.: Mixedinteger optimization with constraint learning. Operations Research **73** (2), 1011–1028 (2025) 

- [22] Dumouchelle, J., Julien, E., Kurtz, J., Khalil, E.B.: Neur2RO: Neural two-stage robust optimization. In: International Conference on Learning Representations (2023) 

- [23] Kronqvist, J., Li, B., Rolfes, J., Zhao, S.: Alternating mixed-integer programming and neural network training for approximating stochastic two-stage problems. In: International Conference on Machine Learning, Optimization, and Data Science, pp. 124–139 (2023). Springer 

- [24] Patel, R.M., Dumouchelle, J., Khalil, E., Bodur, M.: Neur2SP: Neural two-stage stochastic programming. Advances in neural information processing systems **35** , 23992–24005 (2022) 

- [25] Bergman, D., Huang, T., Brooks, P., Lodi, A., Raghunathan, A.U.: JANOS: an integrated predictive and prescriptive modeling framework. INFORMS Journal on Computing **34** (2), 807–816 (2022) 

- [26] Ceccon, F., Jalving, J., Haddad, J., Thebelt, A., Tsay, C., Laird, C.D., Misener, R.: OMLT: Optimization & machine learning toolkit. Journal of Machine Learning Research **23** (349), 1–8 (2022) 

- [27] Turner, M., Chmiela, A., Koch, T., Winkler, M.: PySCIPOpt-ML: Embedding trained machine learning models into mixed-integer programs. In: International Conference on the Integration of Constraint Programming, Artificial Intelligence, and Operations Research, pp. 218–234 (2025). Springer 

- [28] Jalving, J., Ghouse, J., Cortes, N., Gao, X., Knueven, B., Agi, D., Martin, S., Chen, X., Guittet, D., Tumbalam-Gooty, R., _et al._ : Beyond price taker: Conceptual design and optimization of integrated energy systems using machine learning market surrogates. Applied Energy **351** , 121767 (2023) 

- [29] L´opez-Flores, F.J., Ram´ırez-M´arquez, C., Ponce-Ortega, J.M.: Process systems engineering tools for optimization of trained machine learning models: Comparative and perspective. Industrial & Engineering Chemistry Research **63** (32), 13966–13979 (2024) 

- [30] McDonald, T., Tsay, C., Schweidtmann, A.M., Yorke-Smith, N.: Mixed-integer optimisation of graph neural networks for computer-aided molecular design. Computers & Chemical Engineering **185** , 108660 (2024) 

- [31] Schweidtmann, A.M., Mitsos, A.: Deterministic global optimization with artificial neural networks embedded. Journal of Optimization Theory and Applications **180** (3), 925–948 (2019) 

- [32] Fischetti, M., Jo, J.: Deep neural networks and mixed integer linear optimization. Constraints **23** (3), 296–309 (2018) 

- [33] Lomuscio, A., Maganti, L.: An approach to reachability analysis for feed-forward relu neural 

32 

networks. arXiv preprint arXiv:1706.07351 (2017) 

- [34] Tjeng, V., Xiao, K.Y., Tedrake, R.: Evaluating robustness of neural networks with mixed integer programming. In: International Conference on Learning Representations (2017) 

- [35] Anderson, R., Huchette, J., Ma, W., Tjandraatmadja, C., Vielma, J.P.: Strong mixed-integer programming formulations for trained neural networks. Mathematical Programming **183** (1), 3–39 (2020) 

- [36] Tsay, C., Kronqvist, J., Thebelt, A., Misener, R.: Partition-based formulations for mixedinteger optimization of trained ReLU neural networks. Advances in neural information processing systems **34** , 3068–3080 (2021) 

- [37] Badilla, F., Goycoolea, M., Mu˜noz, G., Serra, T.: Computational tradeoffs of optimizationbased bound tightening in relu networks. arXiv preprint arXiv:2312.16699 (2023) 

- [38] Sosnin, P., Tsay, C.: Scaling mixed-integer programming for certification of neural network controllers using bounds tightening. In: 2024 IEEE 63rd Conference on Decision and Control (CDC), pp. 1645–1650 (2024). IEEE 

- [39] Zhao, H., Hijazi, H., Jones, H., Moore, J., Tanneau, M., Van Hentenryck, P.: Bound tightening using rolling-horizon decomposition for neural network verification. In: International Conference on the Integration of Constraint Programming, Artificial Intelligence, and Operations Research, pp. 289–303 (2024). Springer 

- [40] Milgrom, P., Segal, I.: Envelope theorems for arbitrary choice sets. Econometrica **70** (2), 583– 601 (2002) https://doi.org/10.1111/1468-0262.00296 

- [41] Fiacco, A.V.: Introduction to Sensitivity and Stability Analysis in Nonlinear Programming. Academic Press, New York (1983) 

- [42] Xiao, K., Tjeng, V., Shafiullah, N.M., Madry, A.: Training for faster adversarial robustness verification via inducing ReLU stability. In: International Conference on Learning Representations (2019) 

- [43] Gowal, S., Dvijotham, K., Stanforth, R., Bunel, R., Qin, C., Uesato, J., Arandjelovic, R., Mann, T., Kohli, P.: On the effectiveness of interval bound propagation for training verifiably robust models. arXiv preprint arXiv:1810.12715 (2018) 

- [44] Mirman, M., Gehr, T., Vechev, M.: Differentiable abstract interpretation for provably robust neural networks. In: International Conference on Machine Learning, pp. 3578–3586 (2018). PMLR 

- [45] Zhang, H., Chen, H., Xiao, C., Gowal, S., Stanforth, R., Li, B., Boning, D., Hsieh, C.J.: Towards stable and efficient training of verifiably robust neural networks. In: International Conference on Learning Representations (2020) 

- [46] Sosnin, P., Wicker, M., Collyer, J., Tsay, C.: Abstract gradient training: A unified certification framework for data poisoning, unlearning, and differential privacy. arXiv preprint arXiv:2511.09400 (2025) 

- [47] Mandi, J., Kotary, J., Berden, S., Mulamba, M., Bucarey, V., Guns, T., Fioretto, F.: Decisionfocused learning: Foundations, state of the art, benchmark and future opportunities. Journal of Artificial Intelligence Research **80** , 1623–1701 (2024) 

- [48] Elmachtoub, A.N., Grigas, P.: Smart “predict, then optimize”. Management Science **68** (1), 9–26 (2022) 

33 

- [49] Donti, P., Amos, B., Kolter, J.Z.: Task-based end-to-end model learning in stochastic optimization. Advances in neural information processing systems **30** (2017) 

- [50] Dvijotham, K., Gowal, S., Stanforth, R., Arandjelovic, R., O’Donoghue, B., Uesato, J., Kohli, P.: Training verified learners with learned verifiers. arXiv preprint arXiv:1805.10265 (2018) 

- [51] Tang, B., Khalil, E.B.: PyEPO: a PyTorch-based end-to-end predict-then-optimize library for linear and integer programming. Mathematical Programming Computation **16** (3), 297–335 (2024) https://doi.org/10.1007/s12532-024-00255-x 

- [52] Amos, B., Xu, L., Kolter, J.Z.: Input convex neural networks. In: International Conference on Machine Learning, pp. 146–155 (2017). PMLR 

- [53] Rosemberg, A.W., Garcia, J.D., Bent, R., Van Hentenryck, P.: Sobolev training of end-to-end optimization proxies. arXiv preprint arXiv:2505.11342 (2025) 

- [54] Tsay, C.: Sobolev trained neural network surrogate models for optimization. Computers & Chemical Engineering **153** , 107419 (2021) 

- [55] Goodfellow, I., Bengio, Y., Courville, A., Bengio, Y.: Deep Learning vol. 1. MIT press Cambridge, ??? (2016) 

- [56] Manng˚ard, M., Kronqvist, J., B¨oling, J.M.: Structural learning in artificial neural networks using sparse optimization. Neurocomputing **272** , 660–667 (2018) 

- [57] Alc´antara, A., Ruiz, C., Tsay, C.: A quantile neural network framework for two-stage stochastic optimization. Expert Systems with Applications **284** , 127876 (2025) 

- [58] Ghilardi, L.M., Patr´on, G.D., Alc´antara, A., Tsay, C.: Integrated design and scheduling of hydrogen processes under uncertainty: A quantile neural network approach. Industrial & Engineering Chemistry Research **64** (44), 21235–21250 (2025) 

- [59] Amos, B., Kolter, J.Z.: Optnet: Differentiable optimization as a layer in neural networks. In: International Conference on Machine Learning, pp. 136–145 (2017). PMLR 

- [60] Bertsimas, D., Tsitsiklis, J.N.: Introduction to Linear Optimization. Athena Scientific, Belmont, MA (1997) 

- [61] Wilhelm, M.E., Wang, C., Stuber, M.D.: Convex and concave envelopes of artificial neural network activation functions for deterministic global optimization. Journal of Global Optimization **85** (3), 569–594 (2023) 

- [62] Agrawal, A., Amos, B., Barratt, S., Boyd, S., Diamond, S., Kolter, J.Z.: Differentiable convex optimization layers. Advances in neural information processing systems **32** (2019) 

- [63] Pineda, L., Fan, T., Monge, M., Venkataraman, S., Sodhi, P., Chen, R.T., Ortiz, J., DeTone, D., Wang, A., Anderson, S., _et al._ : Theseus: A library for differentiable nonlinear optimization. Advances in Neural Information Processing Systems **35** , 3801–3818 (2022) 

- [64] Besan¸con, M., Dias Garcia, J., Legat, B., Sharma, A.: Flexible differentiable optimization via model transformations. INFORMS Journal on Computing **36** (2), 456–478 (2024) 

- [65] Rosemberg, A.W., Garcia, J.D., Pacaud, F., Parker, R.B., Legat, B., Sundar, K., Bent, R., Van Hentenryck, P.: A general and streamlined differentiable optimization framework. arXiv preprint arXiv:2510.25986 (2025) 

- [66] Bengio, Y., L´eonard, N., Courville, A.: Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432 (2013) 

34 

- [67] Yin, P., Lyu, J., Zhang, S., Osher, S., Qi, Y., Xin, J.: Understanding straight-through estimator in training activation quantized neural nets. In: International Conference on Learning Representations (2019) 

- [68] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al.: Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems **32** (2019) 

- [69] Gurobi Optimization, LLC: Gurobi Optimizer Reference Manual (2026). https://www.gurobi. com 

- [70] Huangfu, Q., Hall, J.J.: Parallelizing the dual revised simplex method. Mathematical Programming Computation **10** (1), 119–142 (2018) 

- [71] Applegate, D., D´ıaz, M., Hinder, O., Lu, H., Lubin, M., O’Donoghue, B., Schudy, W.: Practical large-scale linear programming using primal-dual hybrid gradient. Advances in Neural Information Processing Systems **34** , 20243–20257 (2021) 

- [72] Applegate, D., Hinder, O., Lu, H., Lubin, M.: Faster first-order primal-dual methods for linear programming using restarts and sharpness. Mathematical Programming **201** (1), 133– 184 (2023) 

- [73] Liu, Y., Oliveira, F., Kronqvist, J.: ICNN-enhanced 2SP: Leveraging input convex neural networks for solving two-stage stochastic programming. arXiv preprint arXiv:2505.05261 (2025) 

- [74] Cornu´ejols, G., Sridharan, R., Thizy, J.-M.: A comparison of heuristics and relaxations for the capacitated plant location problem. European Journal of Operational Research **50** (3), 280–297 (1991) 

35 

