# Ternary Weight Networks

- **Authors:** Fengfu Li, Bin Liu, Xiaoxing Wang, Bo Zhang, Junchi Yan (Institute of Applied Math. AMSS CAS; MOE Key Lab of Artificial Intelligence, Shanghai Jiao Tong University)
- **Year:** 2016 (arXiv v1); this local copy is v3, revised 2022-11-20
- **Source:** https://arxiv.org/abs/1605.04711
- **Local PDF:** `../../../../ignored/papers/twn-1605.04711.pdf` (gitignored, not checked in)
- **MORPH uses:** The least-squares scale rule (§2.3, Eq. 5: α*_Δ = mean of |W_i| over the
  entries whose |W_i| exceeds the threshold Δ) is the ancestor of `norm_match` mode in
  `morph/model/ternary_qat.py`. TWN's rule-of-thumb threshold Δ* ≈ 0.75·E(|W|) (§2.3) is
  close to but NOT the same number as BitNet b1.58's fixed threshold 0.5·mean|W|
  (`docs/references/training-objectives/bitnet-b1.58/bitnet-b1.58.md`), which MORPH's
  shipped `symmetric` mode uses. See `docs/references.md` §9 for the full comparison.

---

# TERNARY WEIGHT NETWORKS

Fengfu Li¹†, Bin Liu²†, Xiaoxing Wang², Bo Zhang¹*, Junchi Yan²*

¹Institute of Applied Math., AMSS, CAS, Beijing, China
lifengfu12@mails.ucas.ac.cn, b.zhang@amt.ac.cn
²MOE Key Lab of Artificial Intelligence, Shanghai Jiao Tong University, Shanghai, China
{binliu_sjtu, figure1_wxx, yanjunchi}@sjtu.edu.cn

†: Equal contribution. *: Correspondence authors.

arXiv:1605.04711v3 [cs.CV] 20 Nov 2022

## ABSTRACT

We present a memory and computation efficient ternary weight networks (TWNs) — with
weights constrained to +1, 0 and -1. The Euclidian distance between full (float or double)
precision weights and the ternary weights along with a scaling factor is minimized in
training stage. Besides, a threshold-based ternary function is optimized to get an
approximated solution which can be fast and easily computed. TWNs have shown better
expressive abilities than binary precision counterparts. Meanwhile, TWNs achieve up to
16× model compression rate and need fewer multiplications compared with the float32
precision counterparts. Extensive experiments on MNIST, CIFAR-10, and ImageNet datasets
show that the TWNs achieve much better result than the Binary-Weight-Networks (BWNs) and
the classification performance on MNIST and CIFAR-10 is very close to the full precision
networks. We also verify our method on object detection task and show that TWNs
significantly outperforms BWN by more than 10% mAP on PASCAL VOC dataset. The pytorch
version of source code is available at: https://github.com/Thinklab-SJTU/twns.

## 1. INTRODUCTION AND RELATED WORK

Deep neural networks (DNN) have made significant improvements in lots of computer vision
tasks such as object recognition [1, 2, 3, 4] and object detection [5, 6]. This motivates
interests to deploy the state-of-the-art DNN models to real world applications like smart
phones, wearable embedded devices or other edge computing devices. However, these models
often need considerable storage and computational power [7], and can easily overburden the
limited storage, battery power, and computer capabilities of the smart wearable embedded
devices. As a result, it remains a challenge for the deployment.

To mitigate the storage and computational problem [8, 9], methods that seek to binarize
weights or activations in DNN models have been proposed. BinaryConnect [10] uses a single
sign function to binarize the weights. Binary Weight Networks [7] adopts the same
binarization function but adds an extra scaling factor. The extensions of the previous
methods are BinaryNet [11] and XNOR-Net [7] where both weights and activations are
binary-valued. These models eliminate most of the multiplications in the forward and
backward propagations, and thus own the potential of gaining significant benefits with
specialized deep learning (DL) hardware by replacing many multiply-accumulate operations
by simple accumulation [12]. Besides, binary weight networks achieve up to 32× model
compression rate. Despite the binary techniques, some other compression methods focus on
identifying models with few parameters while preserving accuracy by compressing existing
state-of-the-art DNN models in a lossy way. SqueezeNet [13] is such a model that has 50×
fewer parameters than AlexNet [2] but maintains AlexNet-level accuracy on ImageNet.
MobileNet [14] and ShuffleNet [15] propose lightweight architectures to reduce the
parameters and computation cost. Other methods propose to search efficient architectures
and achieves great performance on both classification [16, 17] and object detection [18].
Deep Compression [9] is another most recently proposed method that uses pruning, trained
quantization and huffman coding for compressing neural networks. It reduced the storage
requirement of AlexNet and VGG-16 [3] by 35× and 49×, respectively, without loss of
accuracy. This paper has the following contributions:

1) To our best knowledge, this was the first (at least at its debut in arxiv) ternary
   weight quantization scheme to reduce storage and computational cost for deep neural
   networks.
2) We propose an approximated and universal solution with threshold-based ternary
   function for calculating the ternary weights of the raw neural networks.
3) Experiments show the efficacy of our approach on public benchmarks for both image
   classification and detection.

## 2. TERNARY WEIGHT NETWORKS

### 2.1. Advantage Overview

We address the limited storage and computational resources issues by introducing ternary
weight networks (TWNs), which constrain the weights to be ternary-valued: +1, 0 and -1.
TWNs seek to make a balance between the full precision weight networks (FPWNs)
counterparts and the binary precision weight networks (BPWNs) counterparts. The detailed
features are listed as follows.

**Expressive ability** In most recent network architectures such as VGG [3], GoogLeNet [4]
and ResNet [1], a most commonly used convolutional filter is of size 3×3. With binary
precision, there is only 2^(3×3) = 512 templates. However, a ternary filter with the same
size owns 3^(3×3) = 19683 templates, which gains 38× more stronger expressive abilities
than the binary counterpart.

**Model compression** In TWNs, 2-bit storage requirement is needed for a unit of weight.
Thus, TWNs achieve up to 16× model compression rate compared with the float32 precision
counterparts. Take VGG-19 [3] as an example, float version of the model needs ~500M
storage requirement, which can be reduced to ~32M with ternary precision. Thus, although
the compression rate of TWNs is 2× less than that of BPWNs, it is fair enough for
compressing most of the existing state-of-the-art DNN models.

**Computational requirement** Compared with the BPWNs, TWNs own an extra zero state.
However, the zero terms need not be accumulated for any multiple operations. Thus, the
multiply-accumulate operations in TWNs keep unchanged compared with binary precision
counterparts. As a result, it is also hardware-friendly for training large-scale networks
with specialized DL hardware.

In the following parts, we will give detailed descriptions about the ternary weight
networks problem and an approximated but efficient solution. After that, a simple training
algorithm with error back-propagation is introduced and the run time usage is described
at last.

### 2.2. Problem Formulation

To make the ternary weight networks perform well, we seek to minimize the Euclidian
distance between the full precision weights W and the ternary-valued weights W̃ along with
a nonnegative scaling factor α [7]. The optimization problem is formulated as follows,

**Eq. 1:**

    α*, W̃* = argmin_{α, W̃} J(α, W̃) = ‖W − αW̃‖²₂
    s.t.  α ≥ 0,  W̃_i ∈ {−1, 0, +1},  i = 1, 2, ..., n

Here n is the number of the filter. With the approximation W ≈ αW̃, a basic block of
forward propagation in ternary weight networks is as follows,

**Eq. 2:**

    Z = X * W ≈ X * (αW̃) = (αX) ⊕ W̃
    X_next = g(Z)

where X is the input of the block; * is a convolution or inner product operation; g is a
nonlinear activation function; ⊕ indicates a convolution or an inner product operation
without multiplication; Z is the output feature map of the neural network block. It can
also be used as input of the next block.

### 2.3. Threshold-based Ternary Function

One way to solve the optimization Eq. 1 is to expand the cost function J(α, W̃) and take
the derivative w.r.t. α and W̃_i respectively. However, this would get interdependent α*
and W̃*_i. Thus, there is no deterministic solution in this way [19]. To overcome this, we
try to find an approximated optimal solution with a threshold-based ternary function,

**Eq. 3:**

    W̃_i = f(W_i | Δ) =
        +1   if  W_i > Δ
         0   if  |W_i| ≤ Δ
        −1   if  W_i < −Δ

Here Δ is an positive threshold parameter. With Eq. 3, the original problem can be
transformed to

**Eq. 4:**

    α*, Δ* = argmin_{α≥0, Δ>0} ( |W_Δ| α² − 2( Σ_{i∈I_Δ} |W_i| ) α + c_Δ )

where I_Δ = {i | |W_i| > Δ} and |I_Δ| denotes the number of elements in I_Δ; c_Δ =
Σ_{i∈I_Δ^c} W_i²; W_i² is a α-independent constant. Thus, for any given Δ, the optimal α
can be computed as follows,

**Eq. 5 (the scale rule MORPH cites):**

    α*_Δ = (1 / |I_Δ|) · Σ_{i∈I_Δ} |W_i|

By substituting α*_Δ into Eq. 4, we get a Δ-dependent equation, which can be simplified as
follows,

**Eq. 6:**

    Δ* = argmin_{Δ>0} (1 / |I_Δ|) · ( Σ_{i∈I_Δ} |W_i| )²

The above equation has no straightforward solutions. Though discrete optimization can be
made to solve the problem (due to states of W_i is finite), it should be very time
consuming. As a viable alternative, we make a single assumption that W_i are generated
from uniform or normal distribution. In case of W_i are uniformly distributed in [−α, α]
and Δ lies in (0, α], the approximated Δ* is α/3, which equals to (2/3)E(|W|). When W_i is
generated from normal distributions N(0, σ²), the approximated Δ* is 0.6σ which equals to
0.75E(|W|). Thus, we can use a rule of thumb that

**the threshold rule MORPH cites:**

    Δ* ≈ 0.75·E(|W|) ≈ (0.75/n) · Σ_{i=1}^{n} |W_i|

for simplicity.

> **Verbatim from the PDF, §2.3, last paragraph** — "In case of Wi are uniformly
> distributed in [−α, α] and ∆ lies in (0, α], the approximated ∆∗is α/3, which equals
> to 2/3E(|W|). When Wi is generated from normal distributions N(0, σ2), the approximated
> ∆∗is 0.6σ which equals to 0.75E(|W|). Thus, we can use a rule of thumb that
> ∆∗ ≈ 0.75E(|W|) ≈ 0.75/n Σn i=1 |Wi| for simplicity." The constant is **0.75**, not
> 0.7 — see the correction note in `docs/references.md` §9.

### 2.4. Training of Ternary-Weight-Networks

CNNs typically includes Convolution layer, Fully-Connected layer, Pooling layer (e.g.,
Max-Pooling, Avg-Pooling), Batch-Normalization (BN) layer [20] and Activation layer (e.g.,
ReLU, Sigmoid), in TWNs, we also follow the traditional neural network block design
philosophy, the order of layers in a typical ternary block of TWNs is shown in Fig. 1
(TernaryConv2d → BatchNorm → ReLU).

We borrow the parameter optimization strategy which successfully applied from
BinaryConncet [10] and XNOR-Net [7], in our design, ternarization only happens at the
forward and backward pass in convolution and fully-connected layers, but in the parameters
update stage, we still keep a copy of the full-precision parameters. In addition, two
effective tricks, Batch-Normalization and learning rate step decay that drops the learning
rate by a factor every few epochs, are adopted. We use stochastic gradient descent (SGD)
with momentum to update the the parameters when training TWNs, the detailed training
strategy show in Table 1.

**Algorithm 1: Train a M-layers CNN w/ ternary weights**

Inputs: A minibatch of inputs and targets (I, Y), loss function L(Y, Ŷ) and current weight
W^t. Hyper-parameter: current learning rate η^t. Outputs: updated weight W^(t+1), updated
learning rate η^(t+1).

```
1  Make the float32 weight filters as ternary ones:
2  for m = 1 to M do
3      for k-th filter in m-th layer do
4          Δ_mk = (0.75/n) ‖W^t_mk‖_l1
5          W̃_mk = {-1, 0, +1}, refer to Eq. 3
6          α_mk = (W^t_mk · W̃_mk) / (W̃_mk · W̃_mk)
7          T_mk = α_mk W̃_mk
8  Ŷ = TernaryForward(I, W̃, α)          // standard forward propagation
9  ∂L/∂T̃ = TernaryBackward(∂L/∂Ŷ, T̃)   // standard backward propagation except that
                                          // gradients are computed using T instead of W^t
10 W^(t+1) = UpdateParameters(W^t, ∂L/∂T, η^t)   // we use SGD in this paper
11 η^(t+1) = UpdateLearningrate(η^t, t)           // we use learning rate step decay
```

Note line 6's dot-product form `α_mk = (W·W̃)/(W̃·W̃)` is algebraically the same
quantity as Eq. 5 once W̃ ∈ {−1, 0, +1}: the numerator sums `|W_i|` over the nonzero
(selected) entries and the denominator counts them, so it reduces to the mean of
`|W_i|` over `I_Δ`.

### 2.5. Inference of Ternary-Weight-Networks

In the forward pass, the scaling factor α could be transformed to the inputs according to
Eq. 2. Thus, we only need to keep the ternary-valued weights and the scaling factors for
deployment. This would results up to 16× model compression rate for deployment compared
with the float32 precision counterparts.

## 3. EXPERIMENTS AND DISCUSSION

We benchmark Ternary Weight Networks (TWNs) with Binary Weight Networks (BPWNs) and Full
Precision Networks (FPWNs) on both classification task (MNIST, CIFAR-10 and ImageNet) and
object detection task (PASCAL VOC).

**Table 1. Backbones and hyperparameters setting for different datasets used by our method
on three benchmarks.**

| | MNIST | CIFAR-10 | ImageNet |
|---|---|---|---|
| backbone architecture | LeNet-5 | VGG-7 | ResNet18B |
| weight decay | 1e-4 | 1e-4 | 1e-4 |
| mini-batch size | 50 | 100 | 64(x4) |
| initial learning rate | 0.01 | 0.1 | 0.1 |
| learning rate adjust step | 15, 25 | 80, 120 | 30, 40, 50 |
| momentum | 0.9 | 0.9 | 0.9 |

For a fair comparison, we keep the following configures to be same: network architecture,
regularization method (L2 weight decay), learning rate scaling procedure (multi-step) and
optimization method (SGD with momentum). BPWNs use sign function to binarize the weights
and FPWNs use float-valued weights. See Table 1 for training configurations.

### 3.1. Experiments of Classification

MNIST is a collection of handwritten digits. It is a very popular dataset in the field of
image processing. The LeNet-5 [21] architecture we used in MNIST experiment is "32-C5 +
MP2 + 64-C5 + MP2 + 512 FC + SVM" which starts with a 5x5 convolutional block that includes
a convolution layer, a BN layer and a relu layer. Then a max-pooling layer is followed with
stride 2. The "FC" is a fully connect block with 512 nodes. The top layer is a SVM
classifier with 10 labels. Finally, hinge loss is minimized with SGD.

CIFAR-10 consists of 10 classes with 6K color images of 32x32 resolution for each class. It
is divided into 50K training and 10K testing images. We define a VGG inspired
architecture, denoted as VGG-7, by "2×(128-C3) + MP2 + 2×(256-C3) + MP2 + 2×(512-C3) + MP2
+ 1024-FC + Softmax". Compared with the architecture in [10], we ignore the last fully
connected layer. We follow the data augmentation in [1, 22] for training: 4 pixels are
padded on each side, and a 32×32 crop is randomly sampled from the padded image or its
horizontal flip. At testing time, we only evaluate the single view of the original 32×32
image.

ImageNet consists of about 1.2 million train images from 1000 categories and 50,000
validation images. ImageNet has higher resolution and greater diversity, is more close to
real life than MNIST and CIFAR-10. We adopt the popular ResNet18 architecture [1] as
backbone. Besides, we also benchmark another enlarged counterpart whose number of filters
in each block is 1.5× of the original one which is termed as ResNet18B. In each training
iteration, images are randomly cropped with 224×224 size. We do not use any resize tricks
[7] or any color augmentation.

**Table 2. Classification accuracy (%) on ImageNet with ResNet18 (or ResNet18B in
bracket) as backbones.**

| | MNIST | CIFAR-10 | ImageNet (top-1) | ImageNet (top-5) |
|---|---|---|---|---|
| TWNs (our main approach) | 99.35 | 92.56 | 61.80 (65.3) | 84.20 (86.2) |
| BPWNs (binary precision counterpart) | 99.05 | 90.18 | 57.50 (61.6) | 81.20 (83.9) |
| FPWNs (full precision counterpart) | 99.41 | 92.88 | 65.4 (67.6) | 86.76 (88.0) |
| BinaryConnect [10] | 98.82 | 91.73 | - | - |
| Binarized Neural Networks [11] | 98.60 | 89.85 | - | - |
| Binary Weight Networks [7] | - | - | 60.8 | 83.0 |
| XNOR-Net [7] | - | - | 51.2 | 73.2 |

Table 2 shows the classification results. On the small datasets (MNIST and CIFAR-10), TWNs
achieve similar performance as FPWNs, while beats BPWNs. On the large-scale ImageNet
dataset, BPWNs and TWNs both get poorer performance than FPWNs. However, the accuracy gap
between TWNs and FPWNs is smaller than the gap between BPWNs and TWNs. In addition, when
we change the backbone from ResNet18 to ResNet18B, as the model size is larger, the
performance gap between TWNs (or BPWNs) and FPWNs has been reduced. This indicates low
precision networks gain more merits from larger models than the full precision
counterparts.

The validation accuracy curves of different approaches across all training epochs on
MNIST, CIFAR-10 and ImageNet datasets illustrate in Fig. 2 (accuracy vs. epoch, omitted
here). As we can see in the figure, obviously, BPWNs converge slowly and the training loss
is not stable compared with TWNs and FPWNs. However, TWNs converge almost as fast and
stably as FPWNs.

### 3.2. Experiments of Detection

PASCAL VOC [23] consists of 20 classes with 11540 images and 27450 labeled objects. We
adopt the popular YOLOv5 (small) [24] architecture and compare the performance of full
precision, binary precision and ternary precision in Table 3. Specifically, we initialize
each model by the weights trained on MS-COCO dataset [25] (provided by YOLOv5) and
fine-tune each model by 150 epochs. We observe that TWNs significantly outperforms BPWNs
by more than 10% mAP, showing the great effectiveness of our method.

**Table 3. Detection performance (%) on PASCAL VOC with YOLOv5 (small) as detector on
Pascal VOC.**

| | Precision | Recall | mAP_50 | mAP_50:95 |
|---|---|---|---|---|
| TWNs (our main approach) | 78.0% | 69.1% | 76.8% | 51.5% |
| BPWNs (binary precision counterpart) | 69.8% | 56.7% | 62.9% | 39.4% |
| FPWNs (full precision counterpart) | 83.3% | 80.8% | 86.7% | 63.7% |

## 4. CONCLUSION

In this paper, we have introduced the simple, efficient, and accurate ternary weight
networks for real world AI application which can reduce the memory usage about 16x and the
the computation about 2x. We present the optimization problem of TWNs and give an
approximated solution with a simple but effective ternary function. The proposed TWNs
achieve a balance between accuracy and model compression rate as well as potentially low
computational requirements of BPWNs. Empirical results on public benchmarks show the
superior performance of the proposed method.

## 5. REFERENCES

Full reference list omitted here — see the PDF (`ignored/papers/twn-1605.04711.pdf`) or the
arXiv abstract page for the 25 numbered citations [1]-[25].
