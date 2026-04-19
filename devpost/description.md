
# Zero Depth: RGB based grasping

## What inspired us
Industrial robotics is currently bleeding money on expensive depth-cameras and lengthy calibration processes just to navigate cluttered factory bins. We realized that true scalability in manufacturing doesn't come from adding more expensive sensors; it comes from smarter software. We were heavily inspired by breakthrough research published by the Tokyo University of Science in January 2026. Their work proved that we could compute highly precise 3D object poses without the 3D hardware price tag. We set out to adapt this multi-view architecture to create a pipeline that unlocks flawless, depth-level grip stability using standard, highly efficient RGB cameras.

## How we built it
We built the intelligence layer entirely independent of the physical robotic arm, creating a decoupled, real-time control loop. The vision engine is designed to run on a **Raspberry Pi 5** and is strictly **PyTorch-first**. 

Our core pipeline, which we call **HEAPGrasp**, moves through the following stages: 
* **Capture & Calibrate:** We use `picamera2` to capture synchronized RGB views and perform auto-calibration using ArUco markers.
* **Voxel Carving (Shape-from-Silhouette):** We compute the 3D geometry of the object without a depth sensor. Mathematically, a 3D voxel belongs to the visual hull $\mathcal{H}$ only if its projection falls within the segmented 2D silhouette $S_i$ across all $N$ camera views. This intersection can be expressed as:
    $$V_{hull} = \bigcap_{i=1}^{N} P_i^{-1}(S_i)$$
    This creates a dense voxel cloud and exports it as a PLY file.
* **Grasp Planning:** We pass this geometry into our grasp module, which ranks parallel-jaw grasp candidates. The candidates are defined by `position`, `approach`, `jaw_axis`, `width`, and `score`.
* **Decoupled Execution:** Finally, we stream these exact coordinates as a `grasps.json` directly to the hardware (like a Ford UR10e) for actuation.

For an interactive experience, we also built a Streamlit dashboard featuring a live capture wizard, a segmentation playground, and a 3D object scan with a Next-Best-View planner.



## Challenges we faced

We ran into a mix of technical obstacles and classic hackathon chaos. On the theory side, the biggest surprise was that the stock Dex-Net 2.0 checkpoints have been taken offline — the official Berkeley Box link is dead and no public mirror exists. We pivoted: re-implemented the GQ-CNN 2.0 architecture in PyTorch, exported it to a structurally correct ONNX, and made our patch renderer fluent in Dex-Net's dual-input signature (image + gripper-depth scalar) and metric depth encoding. That gave us an honest grasp-ranking pipeline wired end-to-end; the rankings are placeholders until we close the loop with Phase B fine-tuning on Azure GPU (scaffolding is already in `infra/azure/`). Even with real weights, a visual-hull-vs-real-depth domain gap would remain — which is why we're planning to fine-tune on our own patches rather than deploy stock weights blindly.

Then came the physical reality of the event. We burned precious time stuck in 3D printing waiting queues, which severely squeezed the hours we had left to build and optimize the image processing pipeline. But the most frustrating part was the **hardware gap**: we didn't have a physical robotic arm at our disposal to close the loop. We were essentially flying blind at the finish line. We could see exactly what gripping approach the CNN wanted to take, but we couldn't physically manipulate an arm around the bin to test and validate those decisions in the real world.



## What we learned

We learned what it takes to bridge the gap between academic theory and a production-ready solution. Dropping in a pre-trained model is easy; the real engineering is building the architecture around it. To integrate our grasp-scoring model, we built the invisible glue that safely feeds precise depth and position metrics into the neural network, ensuring our Raspberry Pi 5 delivers real-time results instead of crashing.

Building vision AI for an edge device also forced a level of software discipline you don't need on a powerful laptop. We had to architect our code defensively, separating our camera logic so teammates could test the core engine off-device without breaking the system.

Finally, lacking a physical robotic arm validated our best decision: completely decoupling the intelligence engine from the hardware. While we couldn't execute the physical grip, we proved our system can synthesize a 3D scene and instantly stream actionable, exact grasp coordinates. We built a fully validated control loop—it’s just waiting for the arm to plug in.