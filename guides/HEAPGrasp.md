# HEAPGrasp: Hand-Eye Active Perception to Grasp Objects With Diverse Optical Properties
**Ginga Kennis and Shogo Arai, Member, IEEE**

**Abstract**—Autonomous robotic handling requires accurate 3-D scene measurement followed by grasp planning. Conventional systems struggle with transparent or specular objects. Additionally, in hand-eye setups, moving through multiple viewpoints increases handling execution time. In this paper, we propose HEAPGrasp—Hand-Eye Active Perception to Grasp objects with diverse optical properties. To measure such objects, we focus on the ability to segment objects regardless of their optical properties in RGB images. We employ Shape from Silhouette based on the segmented images for 3-D measurement. To shorten the time required for multi-view capture with a hand-eye camera, we plan its trajectory using a cost function that balances 3-D measurement accuracy against its trajectory length. Real-robot experiments achieve a 96.0% grasp success rate on transparent, specular, and opaque objects, while reducing the hand-eye camera's trajectory length by 52% and handling execution time by 19% relative to a baseline that circles around the scene for 3-D measurement.

**Index Terms**—Grasping, perception for grasping and manipulation, transparent objects, hand-eye camera.

---

## I. INTRODUCTION
Robot handling—grasping objects and transporting them to target locations—constitutes a large share of robotic applications and is one of the most critical operations. Demands range from automotive parts and logistics packages to food ingredients and restaurant dishes. Autonomous handling requires accurate 3-D measurement of the scene followed by grasp planning. However, the difficulty of 3-D measurement depends significantly on the object's optical properties. Measuring opaque objects governed by diffuse reflection is relatively easy, whereas measurement becomes increasingly challenging as an object's transparency increases. Measuring specular objects is also challenging. High-power laser-scanning sensors can measure specular surfaces, but their high price prevents them from being widely adopted.

To address these issues, we propose HEAPGrasp—Hand-Eye Active Perception to Grasp objects with diverse optical properties. An overview of the proposed method is shown in Fig. 1. To measure objects with diverse optical properties, we focus on the ability to segment objects regardless of their optical properties in RGB images. The hand-eye camera captures multi-view RGB images, and semantic segmentation extracts object silhouettes. We then employ Shape from Silhouette (SfS) using the segmented images for 3-D measurement. In SfS, increasing the diversity of viewpoints improves 3-D measurement accuracy, leading to a higher grasp success rate. However, moving the hand-eye camera to multiple viewpoints is time-consuming. Therefore, there is a trade-off between 3-D measurement accuracy and the time required for multi-view capture. To resolve this trade-off, we generate the hand-eye camera's trajectory using a cost function that balances 3-D measurement accuracy against its trajectory length. While the hand-eye camera moves along this trajectory, it continuously captures the scene and updates the 3-D measurement result.

We validate the proposed method on a real robotic system. Results show that the proposed method achieves a grasp success rate of 96.0% for objects with diverse optical properties. Additionally, it reduces the hand-eye camera's trajectory by 52% and the handling execution time by 19% compared to a baseline that circles around the scene for 3-D measurement.

In summary, the contributions of this work are as follows:
* A 3-D measurement and grasping method for objects with diverse optical properties using segmentation and Shape from Silhouette,
* An active perception planning for a hand-eye camera to resolve the trade-off between 3-D measurement accuracy and the time required for multi-view capture,
* Experimental results showing 96.0% grasp success rate on objects with diverse optical properties, 52% reduction in the hand-eye camera's trajectory, and 19% decrease in handling execution time compared to a baseline that circles around the scene for 3-D measurement.

This paper uses the following notations. Let $\mathbb{Z}_{+}$ denote the set of positive integers, $\mathbb{R}$ the set of real numbers, and $\mathbb{G}$ the set of grasp candidates. For a vector $a\in\mathbb{R}^{n}$, Euclidean norm is denoted by $||a||$. All poses are represented as $p := [t^{\top}, q^{\top}]^{\top} \in \mathbb{R}^{7}$, where $t := [x, y, z]^{\top} \in \mathbb{R}^{3}$ is position and $q := [q_{x}, q_{y}, q_{z}, q_{w}]^{\top} \in \mathbb{R}^{4}$ is orientation in quaternion.

---

## II. RELATED WORK
Object handling is often classified by sensing modality into tactile-based approaches, image-based methods, and 3-D measurement-based techniques. Here, we focus on image-based and 3-D measurement-based techniques.

### A. Handling Objects With Diverse Optical Properties
Prior work on handling opaque objects typically relies on depth images. Dex-Net and TossingBot use a single top-down depth image, and VGN fuses multi-view depth images into a TSDF. Such depth-based methods cannot handle transparent or specular objects.

For transparent objects, earlier studies used model-based refraction analysis, while recent work applies deep learning. ClearGrasp predicts normals, boundaries, and masks of transparent objects and restores depth via refinement. However, because these methods are specialized for the optical characteristics of transparent objects or trained on datasets containing only transparent objects, they do not generalize to objects with diverse optical properties.

More recent studies fully exploit deep learning to handle objects with diverse optical properties. ASGrasp predicts visible and occluded depth from RGB and IR stereo images, and GraspNeRF performs end-to-end 3-D reconstruction and grasp planning using Generalizable NeRF. By training on datasets that include objects with diverse optical properties, these models generalize to transparent, specular, and opaque objects.

### B. Active Perception to Reduce Handling Execution Time
Accurate 3-D measurement benefits from capturing the scene from multiple viewpoints. However, obtaining multi-view images by moving a hand-eye camera increases the handling execution time. To address this issue, recent studies have explored active perception using hand-eye cameras to reduce multi-view acquisition time. Evo-NeRF performs 3-D measurement from multi-view RGB images using implicit NeRF representation, executing a full capture trajectory for the first grasp and updating the NeRF with shorter camera trajectories for subsequent grasps. However, NeRF optimization in Evo-NeRF still requires several seconds, which limits its applicability to real-time viewpoint optimization. In contrast, Breyer et al. propose a closed-loop next-best-view strategy to grasp partially occluded objects efficiently. The method performs explicit TSDF integration for 3-D measurement, as in VGN, which can be executed in tens of milliseconds. It then computes the next viewpoint based on information gain over the TSDF volume, enabling real-time viewpoint optimization.

### C. Positioning of This Work
Although significant progress has been made in both directions, no existing method achieves both simultaneously. We propose a method that enables grasping objects with diverse optical properties using a monocular camera, while reducing the handling execution time through active perception. To enable real-time viewpoint optimization while measuring such objects, we adopt a hybrid approach combining deep-learning-based segmentation with explicit Shape from Silhouette (SfS) measurement.

---

## III. PROBLEM FORMULATION
We consider the problem of performing a pick and place task of $n_{ob}$ objects $O_{1}, O_{2}, ..., O_{n_{ob}}$ in space $\mathbb{L} \in \mathbb{W}$ using a hand-eye robot system with a monocular camera, where $\mathbb{W}$ denotes the robot's workspace. The true 3-D shape of the scene is defined as $S \in \mathbb{S}$. We change the pose of the hand-eye camera $p_{cam} \in \mathbb{R}^{7}$ and capture the scene N times. A 3-D measurement algorithm $\mathcal{R}$ processes the captured images and outputs a 3-D measurement result $\hat{S}$. Subsequently, a grasp planning module $\mathcal{G}$ takes $\hat{S}$ as input and computes a set of grasp candidates. Finally, the robot system picks up the object with the optimal grasp and places it at the desired pose $p_{place} \in \mathbb{R}^{7}$.

First, by using the images $I := \{I^{1}, I^{2}, ..., I^{N}\}$, the 3-D measurement result can be expressed as
$\hat{S} = \mathcal{R}(S, I)$. (1)

We assume that the images I depend only on the poses of the hand-eye camera $P_{cam} := \{p_{cam}^{1}, p_{cam}^{2}, ..., p_{cam}^{N}\}$ when they are captured. Therefore, the 3-D measurement result can be rewritten as $\hat{S} = \mathcal{R}(S, P_{cam})$. For convenience, we sometimes write this as $\hat{S}(P_{cam})$.

Then, the grasping module $\mathcal{G}$ computes the grasp candidates $G := \{g^{1}, g^{2}, ..., g^{M}\}$ and their associated grasp scores $S_{grasp} := \{s_{grasp}^{1}, s_{grasp}^{2}, ..., s_{grasp}^{M}\}$ from the 3-D measurement result, which can be expressed as
$\mathcal{G}(\hat{S}) = (\mathcal{G}_{cd}(\hat{S}), \mathcal{G}_{score}(\hat{S})) : \mathbb{S} \rightarrow \mathbb{G}^{M} \times \mathbb{R}^{M}$. (2)

To simplify notation, we expand the expression of the grasp planning module $\mathcal{G}$ as follows
$\mathcal{G} = (\mathcal{G}_{cd}, \mathcal{G}_{score}, \mathcal{G}_{cd}^{*}, \mathcal{G}_{ob}^{*}) : \mathbb{S} \rightarrow \mathbb{G}^{M} \times \mathbb{R}_{+}$, (3)
where $\mathcal{G}_{cd}^{*}$ represents the optimal grasp and $\mathcal{G}_{ob}^{*}$ denotes the label of the object that $\mathcal{G}_{cd}^{*}$ targets. Here, we have omitted the input S. Accordingly, the target object of the optimal grasp $\mathcal{G}_{cd}^{*}$ is denoted by $O_{\mathcal{G}_{ob}^{*}}$.

The grasp success rate of the target object $O_{\mathcal{G}_{ob}^{*}}$ depends on the grasp planning module and the true 3-D shape of the scene, which can be expressed as
$s_{rate} = s_{rate}(\mathcal{G}(S), S)$. (4)

Given the initial pose of the hand-eye camera $p_{cam}^{0} \in \mathbb{R}^{7}$, the problem of minimizing the total time to complete a pick and place task of all objects while ensuring the grasp success rate is above a threshold $\epsilon_{rate} \in \mathbb{R}$ can be written as
$\min_{P_{cam,i}} \sum_{i=1}^{n_{ob}} T(p_{cam}^{0}, P_{cam,i}, \mathcal{G}_{cd}^{*}(\hat{S}(P_{cam,i})), p_{place})$ (5)
s.t. $s_{rate}(\mathcal{G}(S(P_{cam,i})), S_{i}) \ge \epsilon_{rate}$ (6)

Here, $P_{cam,i} := \{p_{cam,i}^{1}, p_{cam,i}^{2}, ..., p_{cam,i}^{N_{i}}\}$ denotes the hand-eye camera poses at which images in phase $i \in \{1, 2, ..., n_{ob}\}$ are captured, where $p_{cam,i}^{n}$ is the pose at which the n-th image is captured, and $N_{i}$ is the total number of images in phase i. $S_{i}$ is the true 3-D shape of the scene in phase i.

Under the following assumptions:
* Task execution time T is determined mainly by the length of the hand-eye camera's trajectory,
* Grasp success rate increases with 3-D measurement accuracy,

the optimization problem we consider can be written as
$\min_{P_{cam,i}} \sum_{i=1}^{n_{ob}} \alpha d_{space}(p_{cam,i}^{0}, P_{cam,i}, \mathcal{G}_{cd}^{*}(\hat{S}(P_{cam,i})), p_{place}) + \beta d_{measure}(\hat{S}(P_{cam,i}), S_{i})$. (7)

Here, $d_{space}$ represents the hand-eye camera's trajectory length as it moves from the initial pose $p_{cam}^{0}$ through viewpoints $P_{cam,i}$, optimal grasp $\mathcal{G}_{cd}^{*}(S(P_{cam,i}))$ to the place pose $p_{place}$. $d_{measure}$ is the error between the true 3-D shape of the scene $S_{i}$ and the 3-D measurement result $\hat{S}(P_{cam,i})$.

Finally, to reduce computational time, we do not optimize the entire set of viewpoints $P_{cam,i}$. Instead, we optimize only a small subset of $L \ll N_{i}$ view waypoints $\overline{P}_{cam,i} := \{\overline{p}_{cam,i}^{1}, ..., \overline{p}_{cam,i}^{L}\}$ and interpolate between consecutive view waypoints to generate the full set of viewpoints
$P_{cam,i} = (f(p_{cam}^{0}, \overline{p}_{cam,i}^{1}), ..., f(\overline{p}_{cam,i}^{L-1}, \overline{p}_{cam,i}^{L}))$ (8)
Here, $f(p_{A}, p_{B})$ is an interpolation function that generates a sequence of poses connecting poses $p_{A} \in \mathbb{R}^{7}$ and $p_{B} \in \mathbb{R}^{7}$.

---

## IV. METHOD
This section focuses on the i-th phase, and we drop the subscript i. The block diagram of the proposed method is shown in Fig. 2. Let $\Sigma_{r}$ and $\Sigma_{w}$ denote the robot base and workspace frames, respectively. We assume the relative pose between $\Sigma_{r}$ and $\Sigma_{w}$ is given. Unless stated otherwise, all poses are expressed in the workspace frame $\Sigma_{w}$.

### A. Perception
When the hand-eye camera arrives at the n-th viewpoint $(n=1, 2, ..., N)$, denoted by $p_{cam}^{n}$, it captures an RGB image $I_{rgb}^{n}$. We apply semantic segmentation on $I_{rgb}^{n}$ to extract object silhouettes and generate a segmented image $I_{seg}^{n}$. Given the hand-eye camera pose $p_{cam}^{n}$, the segmented image $I_{seg}^{n}$ and the 3-D measurement result up to the $n-1$-th viewpoint $S^{n-1} := S(p_{cam}^{1}, p_{cam}^{2}, ..., p_{cam}^{n-1})$, we perform Shape from Silhouette (SfS) to update the 3-D measurement result.

We define scene S as a cubic volume of side length l. The cube is discretized into $N_{x}$, $N_{y}$, and $N_{z}$ voxels along the x-, y-, and z-axes, respectively, and the 3-D measurement result at the n-th viewpoint is represented as an occupancy grid
$\hat{S}_{0-1}^{n} \in [0, 1]^{N_{x} \times N_{y} \times N_{z}}$ (9)

For each voxel $a \in \{1, ..., N_{x}\}$, $b \in \{1, ..., N_{y}\}$, $c \in \{1, ..., N_{z}\}$, we compute the occupancy as
$\hat{S}_{0-1, a, b, c}^{n} := \begin{cases} \frac{|I_{seg, a, b, c}^{n}|}{|I_{rgb, a, b, c}^{n}|}, & \text{when } |I_{rgb, a, b, c}^{n}| > 0 \\ 0, & \text{when } |I_{rgb, a, b, c}^{n}| = 0 \end{cases}$ (10)
where $|I_{seg, a, b, c}^{n}|$ is the number of images up to the n-th viewpoint in which the voxel is segmented, and $|I_{rgb, a, b, c}^{n}|$ is the number of images up to the n-th viewpoint in which the voxel projects inside the image.

Hereafter, let $\hat{S}_{\ge\lambda}^{n}$ be referred to as the high-confidence occupancy grid, the occupancy grid with values below $\lambda \in [0, 1]$ set to zero.

### B. Next Pose Planning
When the hand-eye camera reaches the $l(n)$-th view waypoint $(l(n)=1, 2, ..., L)$, denoted by $\overline{p}_{cam}^{l(n)}$, we perform grasp planning, view planning, and next pose selection to compute the $l(n)+1$-th view waypoint $\overline{p}_{cam}^{l(n)+1}$. Here, $l(n)$ denotes the index of the view waypoint that the hand-eye camera passed just before capturing the n-th image. The Next Pose Planning block is executed only for those indices n that satisfy $l(n) - l(n-1) = 1$, i.e., immediately after the hand-eye camera arrives at a view waypoint. We repeat this process until the hand-eye camera reaches the final view waypoint $\overline{p}_{cam}^{L}$. After reaching the final view waypoint, we perform grasp planning and next pose selection once more to compute the optimal grasp $\mathcal{G}_{cd}^{*}(S^{N})$.

**Grasp Planning:** We input the high-confidence occupancy grid $\hat{S}_{\ge\lambda}^{n}$ into an encoder-decoder 3D CNN that predicts the grasp orientation $R_{grasp}^{l(n)} \in \mathbb{R}^{N_{x} \times N_{y} \times N_{z} \times 4}$, the gripper width $W_{grasp}^{l(n)} \in \mathbb{R}^{N_{x} \times N_{y} \times N_{z} \times 1}$, and the grasp score $S_{grasp}^{l(n)} \in [0, 1]^{N_{x} \times N_{y} \times N_{z} \times 1}$ at every voxel. Let $T_{grasp} \in \mathbb{R}^{N_{x} \times N_{y} \times N_{z} \times 3}$ denote the voxel center positions. Using the grasp poses $P_{grasp}^{l(n)} := [T_{grasp}, R_{grasp}^{l(n)}] \in \mathbb{R}^{N_{x} \times N_{y} \times N_{z} \times 7}$ and the gripper width $W_{grasp}^{l(n)}$, we define the grasp candidates computed at the $l(n)$-th view waypoint as
$G^{l(n)} := [P_{grasp}^{l(n)}, W_{grasp}^{l(n)}] \in \mathbb{R}^{N_{x} \times N_{y} \times N_{z} \times 8}$ (11)

**View Planning (VP):** We predefine a spherical coordinate system of radius r, whose origin is at the bottom center of the scene. By uniformly dividing the azimuthal angle $\phi \in [0, 2\pi)$ into U intervals and the polar angle $\theta \in [0, \theta_{max})$ into V intervals, we obtain UV view waypoint candidates
$\overline{P}_{cam,cd} := [\overline{p}_{cam}^{u,v}]_{u=1, 2, ..., U; v=1, 2, ..., V} \in \mathbb{R}^{U \times V \times 7}$ (12)
where candidates are oriented toward the center of the scene.

We first input the occupancy grid $\hat{S}_{0-1}^{n}$ into a 3-D CNN encoder to produce a 3-D feature map. Next, we compress this 3-D feature map along the z-axis (height) to obtain a 2-D feature map. We then apply a sequence of 2-D convolution, ReLU activation, and upsampling multiple times to the 2-D feature map. Finally, we once again apply a 2-D convolution followed by a sigmoid activation to output the view scores
$S_{cam}^{l(n)+1} := [s_{cam}^{l(n)+1,u,v}]_{u=1, 2, ..., U; v=1, 2, ..., V} \in [0, 1]^{U \times V}$ (13)

Each view score $s_{cam}^{l(n)+1,u,v}$ corresponds to the view waypoint candidate $\overline{p}_{cam}^{u,v}$ whose spherical coordinates are given by $\phi := 2\pi(u-1)/U$ and $\theta := \theta_{max}(v-1)/V$.

In SfS, smaller intersections of visual hulls indicate less uncertainty in the 3-D measurement result and thus higher measurement accuracy. Based on this property, the view score $s_{cam}^{l(n)+1,u,v}$ is interpreted as the expected reduction in occupied voxels and is defined as
$s_{cam}^{l(n)+1,u,v} := 1 - \frac{\#\hat{S}_{\ge\lambda}(P_{cam}^{l(n)}, f(\overline{p}_{cam}^{l(n)}, \overline{p}_{cam}^{u,v}))}{\#\hat{S}_{\ge\lambda}(P_{cam}^{l(n)})}$ (14)

Here, $P_{cam}^{l(n)}$ denotes the viewpoints up to the $l(n)$-th view waypoint, $\#\hat{S}_{\ge\lambda}(P_{cam}^{l(n)})$ and $\#\hat{S}_{\ge\lambda}(P_{cam}^{l(n)}, f(\overline{p}_{cam}^{l(n)}, \overline{p}_{cam}^{u,v}))$ represent the number of non-zero voxels in the current high-confidence occupancy grid and in the high-confidence occupancy grid obtained by hypothetically moving the hand-eye camera to the view waypoint candidate $\overline{p}_{cam}^{u,v}$, respectively.

**Next Pose Selection:** We first smooth the grasp scores $S_{grasp}^{l(n)}$ with a 3-D Gaussian filter. We then remove grasp candidates whose scores fall below a threshold $\epsilon \in \mathbb{R}$. Finally, we apply non-maximum suppression to retain candidates with the highest score among their neighbors. The grasp poses retained after this process are defined as
$P_{grasp,\epsilon}^{l(n)} := \{p_{grasp}^{l(n),a,b,c} | s_{grasp}^{l(n),a,b,c} \ge \epsilon\}$. (15)

We select the target grasp pose at the $l(n)$-th view waypoint as
$p_{grasp}^{l(n)} := \arg\min_{p \in P_{grasp,\epsilon}^{l(n)}} dist(p_{grasp}^{l(n)-1}, p)$ (16)
thereby ensuring that the target grasp pose changes minimally between consecutive view waypoints. Here, $dist(p_{a}, p_{b}) \in \mathbb{R}$ is the distance between two poses $p_{a} := [t_{a}^{\top}, q_{a}^{\top}]^{\top}, p_{b} := [t_{b}^{\top}, q_{b}^{\top}]^{\top} \in \mathbb{R}^{7}$, computed as
$dist(p_{a}, p_{b}) := ||t_{a} - t_{b}|| + \gamma_{1}\min(||q_{a} + q_{b}||, ||q_{a} - q_{b}||)$, (17)
where $t_{a}, t_{b} \in \mathbb{R}^{3}$ denote the positions, and $q_{a}, q_{b}$ denotes the orientation in quaternion. $\gamma_{1}$ is a hyperparameter.

Finally, we select the $l(n)+1$-th view waypoint as $\overline{p}_{cam}^{l(n)+1} := \overline{p}_{cam}^{u^{*},v^{*}}$ where
$(u^{*}, v^{*}) := \arg\max_{1 \le u \le U, 1 \le v \le V} s_{cam}^{l(n)+1,u,v} + \gamma_{2}h(u, v)$. (18)

Here, $h(u, v)$ is the rate of reduction in distance to the target grasp pose, defined as
$h(u, v) := 1 - \frac{dist(\overline{p}_{cam}^{u,v}, p_{grasp}^{l(n)})}{dist(\overline{p}_{cam}^{l(n)}, p_{grasp}^{l(n)})}$. (19)
where $\gamma_{2}$ is a hyperparameter that balances 3-D measurement accuracy and distance reduction to the target grasp pose.

---

## V. TRAINING
For semantic segmentation, we use DeepLabv3+ with a ResNet-50 backbone pretrained on ImageNet. We fine-tune the model on 427 real-world images captured by our research group to segment images into four classes: transparent objects, specular objects, opaque objects, and background. We evaluate the segmentation performance on 40 validation images and obtain IoU values of 0.93, 0.95, and 0.94 for transparent, specular, and opaque objects, respectively, with a mIoU of 0.94. These results demonstrate robust segmentation performance regardless of the optical properties of the objects.

We adopt the Volumetric Grasping Network (VGN) as the grasp planning model. Unlike the original VGN, our method represents its input as a high-confidence occupancy grid. Following the same dataset generation scheme as VGN, we use the PyBullet physics engine to generate approximately $2.0 \times 10^{6}$ grasp candidates.

The encoder in the view planning model shares the same architecture as that in the grasp planning model. We use the PyBullet physics engine to generate the view planning dataset, with the same objects used in the grasp planning dataset, as described in Algorithm 1. In this algorithm, SfS (lines 12 and 17) maps hand-eye camera poses and segmented images to an occupancy grid. After generating the dataset in this manner and removing invalid samples, we obtain approximately $5.5 \times 10^{6}$ view scores.

```text
Algorithm 1: View Planning Dataset Generation Scheme.
1: for i = 1 to N_scene do
2:   n_ob <- UniformRandom(1, N_ob)
3:   Randomly place n_ob objects in the scene
4:   k <- UniformRandom(0, L-1)
5:   if k = 0 then
6:     P_cam^k := (p_cam^0)
7:   else
8:     {p_cam^1, ..., p_cam^k} <- Randomly select k view waypoints from P_cam,cd
9:     P_cam^k := (f(p_cam^0, p_cam^1), ..., f(p_cam^{k-1}, p_cam^k))
10:  end if
11:  I_seg^k <- GetSegmentedImage(P_cam^k)
12:  S(P_cam^k) = SfS(P_cam^k, I_seg^k)
13:  for (u, v) in {1, ..., U} x {1, ..., V} do
14:    p_cam^{u,v} <- GetPoseFromIndex(P_cam,cd, u, v)
15:    P_cam^{k+1,u,v} := (P_cam^k, f(p_cam^k, p_cam^{u,v}))
16:    I_seg^{k+1,u,v} <- GetSegmentedImage(P_cam^{k+1,u,v})
17:    S(P_cam^{k+1,u,v}) = SfS(P_cam^{k+1,u,v}, I_seg^{k+1,u,v})
18:    s_cam^{k+1,u,v} := 1 - (#S_{>=lambda}(P_cam^{k+1,u,v}) / #S_{>=lambda}(P_cam^k))
19:  end for
20: end for
```

---

## VI. EXPERIMENTS

### A. Experimental Setup
We conduct experiments in both simulation and real-world environments. The simulation experiments are performed in the same environment as VGN, and do not model the optical properties of objects. Therefore, both depth and segmentation images are assumed to be ideal. This simulation aims to compare the algorithms of each method under ideal conditions, while the robustness to optical properties is verified in real-robot experiments.

The real-robot system consists of a Universal Robots A/S UR5e, an Intel Corporation RealSense Depth Camera D415, and a ROBOTIQ Inc. 2F140 Adaptive Gripper. Note that depth images are used only by the baseline method.

### B. Baseline Methods
We compare the proposed method (HEAPGrasp and HEAPGrasp w/o VP) with baseline methods (VGN, GraspNeRF):
* **VGN:** We predefine L view waypoints evenly spaced around the scene. As the hand-eye camera moves from its initial pose through each view waypoint, it captures depth images (640×480 pixels at 30 fps) and integrates them into a TSDF. After reaching the final view waypoint, we execute the top-scoring grasp.
* **GraspNeRF:** We uniformly sample six view waypoints on a hemisphere centered in the workspace. All viewpoints point toward the workspace center and are represented in spherical coordinates with radius $r=0.5$ m, polar angle $\theta=\pi/6$, and azimuthal angle $\phi \in U(0, 2\pi)$ where $U(a, b)$ denotes a uniform distribution over the interval [a, b]. After capturing six RGB images (640×360 pixels) from these view waypoints, we execute the grasp with the highest grasp score.
* **HEAPGrasp w/o VP:** As in VGN, we use the same L predefined view waypoints and trajectory to capture RGB images (320×240 pixels at 30 fps) and perform 3-D measurement with the proposed perception module. After reaching the final view waypoint, we input the high-confidence occupancy grid to the proposed grasp planning module and execute the top-scoring grasp.

**TABLE I: Parameters Used in the Experiments**

| Description | Symbol | Value |
|---|---|---|
| Number of objects | $N_{ob}$ | 5 |
| Side length of the cubic volume | L | 0.30 m |
| Voxel grid resolution | $N_{x}, N_{y}, N_{z}$ | 40, 40, 40 |
| Radius | r | 0.33 m |
| Maximum polar angle | $\theta_{max}$ | $\pi/5$ rad |
| Azimuth and polar divisions | U, V | 40, 20 |
| Number of view waypoints | L | 3 |
| Occupancy threshold | $\lambda$ | 0.9 |
| Grasp score threshold | $\epsilon$ | 0.8 |
| Weight in (17) | $\gamma_{1}$ | 0.2 |
| Weight in (18) | $\gamma_{2}$ | 0.0, 0.35 |

In HEAPGrasp, when computing the first view waypoint $\overline{p}_{cam}^{1}$ from the initial pose $p_{cam}^{0}$, the high uncertainty in the 3D measurement result makes grasp candidates unreliable. To address this, we set the weighting parameter $\gamma_{2}$ to 0.0 for the first view waypoint calculation. After reaching $\overline{p}_{cam}^{1}$, we set $\gamma_{2}$ to 0.35 for subsequent view waypoint calculations.

### C. Experimental Protocol
We evaluate each method across 20 scenes, each containing 5 objects. The 20 scenes are evenly divided into 4 groups of 5 scenes: transparent objects only, opaque objects only, specular objects only, and mixed scenes with all three categories. For each scene, we execute the handling task until one of the following conditions is met: all objects have been successfully picked and placed, two consecutive grasp attempts fail, or no grasp candidates are detected.

### D. Evaluation Metrics
We evaluate each method using the following metrics:
* **Grasp success rate (%):** the number of successful grasps divided by the total number of grasp attempts.
* **Trajectory length (m):** the average length of the hand-eye camera's trajectory per phase.
* **Shortest trajectory ratio (%):** the average ratio of the hand-eye camera's trajectory length to the Euclidean distance between the initial position and the grasp position per phase.
* **Execution time (s):** the average execution time from the start to the end of each phase.

### E. Results
Table II summarizes the results in the simulation. Although HEAPGrasp w/o VP uses only RGB images, it achieves a grasp success rate of 92.7%, which is higher than that of VGN (87.1%) relying on depth images. Furthermore, HEAPGrasp shortens the trajectory length by 50% and the shortest trajectory ratio by 55% compared with HEAPGrasp w/o VP, while maintaining an 86.4% success rate.

**TABLE II: Results of the Simulation Experiments**

| Method | Grasp success rate (%) | Trajectory length (m) | Shortest trajectory ratio (%) |
|---|---|---|---|
| VGN | 87.1 (433/497) | $1.87 \pm 0.09$ | $325 \pm 35$ |
| HEAPGrasp w/o VP | 92.7 (458/494) | $1.87 \pm 0.08$ | $337 \pm 32$ |
| HEAPGrasp | 86.4 (432/500) | $0.94 \pm 0.12$ | $151 \pm 27$ |

Table III summarizes the real-robot experimental results for VGN, GraspNeRF, HEAPGrasp w/o VP, and HEAPGrasp. VGN achieves a grasp success rate of 88.5% on opaque objects, but this rate falls to 72.0% for specular and 53.8% for transparent objects. GraspNeRF also achieves a high success rate of 91.7% on opaque objects, but this also drops to 52.2% for specular and 68.2% for transparent objects. By contrast, both HEAPGrasp w/o VP and HEAPGrasp achieve grasp success rates above 92.6% across all categories.

**TABLE III: Results of the Real Robot Experiments**

| Method | Scene | Grasp success rate (%) Seen | Grasp success rate (%) Unseen | Grasp success rate (%) Overall | Trajectory length (m) | Shortest trajectory ratio (%) | Execution time (s) |
|---|---|---|---|---|---|---|---|
| VGN [6] | Opaque | - | - | 88.5 (23/26) | 1.95 ± 0.10 | 340 ± 40 | 10.06 ± 0.35 |
| | Specular | - | - | 72.0 (18/25) | 1.97 ± 0.09 | 343 ± 37 | 10.00 ± 0.38 |
| | Transparent | - | - | 53.8 (14/26) | 1.99 ± 0.11 | 366 ± 30 | 10.12 ± 0.31 |
| | Mixed | - | - | 74.1 (20/27) | 1.99 ± 0.13 | 349 ± 46 | 10.13 ± 0.38 |
| | Overall | - | - | 72.1 (75/104) | 1.98 ± 0.11 | 348 ± 41 | 10.07 ± 0.36 |
| GraspNeRF [13] | Opaque | - | - | 91.7 (22/24) | 2.31 ± 0.06 | 419 ± 48 | 18.50 ± 0.62 |
| | Specular | - | - | 52.2 (12/23) | 2.36 ± 0.10 | 439 ± 33 | 19.00 ± 0.64 |
| | Transparent | - | - | 68.2 (15/22) | 2.36 ± 0.11 | 446 ± 50 | 19.11 ± 0.99 |
| | Mixed | - | - | 73.1 (19/26) | 2.30 ± 0.06 | 433 ± 41 | 18.79 ± 0.43 |
| | Overall | - | - | 71.6 (68/95) | 2.33 ± 0.09 | 432 ± 45 | 18.80 ± 0.72 |
| HEAPGrasp w/o VP | Opaque | 92.9 (13/14) | 100.0 (12/12) | 96.2 (25/26) | 2.03 ± 0.08 | 315 ± 27 | 9.98 ± 0.37 |
| | Specular | 100.0 (13/13) | 100.0 (12/12) | 100.0 (25/25) | 2.03 ± 0.07 | 322 ± 25 | 9.88 ± 0.25 |
| | Transparent | 100.0 (13/13) | 91.7 (11/12) | 96.0 (24/25) | 2.03 ± 0.07 | 332 ± 34 | 9.85 ± 0.18 |
| | Mixed | 100.0 (11/11) | 100.0 (14/14) | 100.0 (25/25) | 2.03 ± 0.08 | 323 ± 25 | 9.94 ± 0.34 |
| | Overall | 98.0 (50/51) | 98.0 (49/50) | 98.0 (99/101) | 2.03 ± 0.07 | 323 ± 29 | 9.91 ± 0.30 |
| HEAPGrasp | Opaque | 92.9 (13/14) | 92.3 (12/13) | 92.6 (25/27) | 0.98 ± 0.13 | 153 ± 32 | 8.23 ± 1.10 |
| | Specular | 100.0 (13/13) | 100.0 (12/12) | 100.0 (25/25) | 0.95 ± 0.11 | 151 ± 27 | 7.90 ± 0.81 |
| | Transparent | 92.3 (12/13) | 100.0 (10/10) | 95.7 (22/23) | 0.92 ± 0.08 | 146 ± 23 | 7.83 ± 0.53 |
| | Mixed | 100.0 (11/11) | 93.3 (14/15) | 96.2 (25/26) | 1.01 ± 0.17 | 159 ± 33 | 8.07 ± 0.97 |
| | Overall | 96.1 (49/51) | 96.0 (48/50) | 96.0 (97/101) | 0.97 ± 0.13 | 152 ± 29 | 8.01 ± 0.90 |

This gap in grasp success rates is mainly attributable to differences in the 3-D measurement results. Since VGN relies on depth images, its 3-D measurements are incomplete for transparent objects and contain many outliers for specular objects. Although GraspNeRF produces more accurate 3-D measurements for transparent and specular objects, some regions remain incomplete or contain outliers. Compared with VGN and GraspNeRF, HEAPGrasp w/o VP produces cleaner 3-D measurements. Furthermore, Table III shows that HEAPGrasp w/o VP and HEAPGrasp exhibit strong generalization capability to unseen objects, achieving grasp success rates of 98.0% and 96.0%, respectively.

Table III shows that GraspNeRF requires the longest trajectory at 2.33 m, with a shortest trajectory ratio of 432% since it requires six viewpoints around the scene. HEAPGrasp w/o VP records a shorter trajectory of 2.03 m and a shortest trajectory ratio of 323%. In contrast, HEAPGrasp further reduces the trajectory length by 52% to 0.97 m and the shortest trajectory ratio by 53% to 152% compared to HEAPGrasp w/o VP. Finally, HEAPGrasp reduces the handling execution time by 19% compared with HEAPGrasp w/o VP, achieving an execution time of 8.01 s.

### F. Ablation Studies and Discussion
We first analyze the effect of object geometry on grasp performance, particularly the limitation of SfS in measuring concave objects. The grasp-planning model generates grasp candidates when (1) the candidate lies on the visual hull with high grasp stability and (2) its opening width is smaller than the maximum opening width of the gripper used during grasp-planning model training.

Next, we analyze the effect of scene complexity by comparing HEAPGrasp w/o VP and HEAPGrasp (w/ VP) in cluttered scenes. As shown in Table IV, their success rates are 95% and 80%, decreasing by 3% and 16% compared with Table III.

**TABLE IV: Results of the Real-Robot Experiments in Cluttered Scenes With Mixed Optical Properties**

| Number of objects | Grasp success rate (%) w/o VP | Grasp success rate (%) w/ VP | Trajectory length (m) w/o VP | Trajectory length (m) w/ VP | Shortest trajectory ratio (%) w/o VP | Shortest trajectory ratio (%) w/ VP | Execution time (s) w/o VP | Execution time (s) w/ VP |
|---|---|---|---|---|---|---|---|---|
| 6 | 80 (4/5) | 80 (4/5) | 2.15 ± 0.10 | 1.02 ± 0.08 | 308 ± 27 | 139 ± 13 | 8.36 ± 0.72 | 9.99 ± 0.23 |
| 8 | 100 (5/5) | 100 (5/5) | 2.13 ± 0.14 | 1.06 ± 0.11 | 281 ± 18 | 138 ± 15 | 8.24 ± 0.40 | 9.92 ± 0.13 |
| 10 | 100 (5/5) | 80 (4/5) | 2.07 ± 0.13 | 0.93 ± 0.11 | 281 ± 14 | 123 ± 13 | 9.84 ± 0.14 | 7.58 ± 0.48 |
| 12 | 60 (3/5) | 100 (5/5) | 2.07 ± 0.10 | 1.12 ± 0.19 | 292 ± 39 | 148 ± 23 | 9.86 ± 0.22 | 8.36 ± 0.95 |
| Overall | 95 (19/20) | 80 (16/20) | 2.10 ± 0.13 | 1.03 ± 0.14 | 290 ± 28 | 137 ± 18 | 9.90 ± 0.19 | 8.13 ± 0.71 |

This drop mainly results from the degradation of measurement accuracy in cluttered scenes, yet HEAPGrasp still achieves 80% success by grasping objects near scene boundaries, as SfS reconstructs the visual hull.

Finally, we examine the effect of segmentation accuracy on 3-D measurement and grasping performance. We evaluate HEAPGrasp w/o VP by varying the noise level—the probability that each pixel in a binary mask is randomly flipped—in simulation, and evaluate the mean squared error (MSE) and intersection over union (IoU) between the high-confidence occupancy grids obtained from noisy and noise-free masks, as well as the grasp success rate.

**TABLE V: Effect of Segmentation Noise on Measurement and Grasping**

| Noise level | MSE | IoU | Successfully grasped objects (%) |
|---|---|---|---|
| 0.01 | 0.02 | 0.99 | 90.2 (451/500) |
| 0.05 | 0.12 | 0.92 | 92.8 (464/500) |
| 0.10 | 0.93 | 0.44 | 92.8 (464/500) |
| 0.15 | 1.63 | 0.03 | 68.6 (343/500) |

As shown in Table V, the IoU remains 0.44 even with 10% noise, while the success rate stays high at 92.8%, demonstrating robustness to segmentation errors.

---

## VII. CONCLUSION
This paper proposes HEAPGrasp—Hand-Eye Active Perception to Grasp objects with diverse optical properties. HEAPGrasp segments objects in multi-view RGB images and measures their shape via Shape from Silhouette. The view planning module reduces multi-view capture time by optimizing a cost function balancing 3-D measurement accuracy and hand-eye camera's trajectory length. Experiments show that HEAPGrasp achieves a 96.0% grasp success rate, while reducing the hand-eye camera trajectory length by 52% and handling execution time by 19% compared to a baseline that circles around the scene for 3-D measurement.
```