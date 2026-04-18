 # Using Grippers to Lead Next Gen Manufacturing
Purdue StarkHacks Hackathon
Ford ATP

## Overview: The Difficulties of Grasping
* Robotic grasping is a major remaining hurdle to deploying robots in diverse, real-world applications
* Grasping is difficult because it combines uncertainty in hardware and perception, among other challenges:
* **Gripper Versatility**
  * A single, universal gripper matching human dexterity and capabilities does not yet exist. Grippers are often application-specific to balance tradeoffs (precision, robustness, speed, cost, versatility), which can lead to recurring development costs or suboptimal performance in new tasks and when conditions slightly change
* **Perception**
  * Variability in object properties and environmental conditions (clutter, occlusion, lighting, reflective/transparent surfaces, deformable objects) makes it hard to reliably localize objects and infer stable grasp points/poses
  * Sensor noise and calibration error further increase uncertainty in estimated grasp poses
* Because these multifaceted uncertainties compound, grasping is difficult to generalize and to make robust and repeatable

## Challenge Details
* Teams will be tasked with developing a solution that improves a robot's grasping performance
* The solution can be hardware and/or software based
* There are many ways that you could approach the challenge, including:
  * Developing a more versatile/universal gripper
  * Developing a quick-swap mechanism for a more modular gripper attachment
  * Developing a solution to effectively grasp parts regardless of clutter/orientation
  * Developing a solution to determine when a currently grasped part is unstable/the grip needs to be adjusted
* The solution can also target a specific grasp related task, such as:
  * Bolt alignment and fastening
  * Operation of different handheld tools
* These examples are meant to be used as a starting point and are by no means comprehensive. Identify impactful areas for grasping improvement and get creative!
* Teams should record a video or prepare a demo that demonstrates the benefits of their solution and showcases the functionality of their prototype

## Scoring
* Teams will be scored on a combination of the following factors:
* **Impact (15%)** - Is there a clear value in the team's solution, and does it solve a real challenge associated with grasping?
* **Innovation (30%)** - How creative and novel is the team's solution? Is it fundamentally approaching the problem in a new way, or finding a unique application of something that already exists?
* **Performance (20%)** - Is the prototype functional? How effective is the solution at completing its intended task?
* **Technical Execution (20%)** - What is the overall impression of the quality and robustness of the solution? Does the solution prove a strong understanding of engineering principles?
* **Feasibility (15%)** - How practical is the solution? Could it be developed and scaled to be implemented/deployed in real world applications?

## Provided Resources
* A hardware components kit has been provided and will be distributed by the hardware booth to aid in the development of hardware grasping solutions.
  * This parts kit includes vision sensors, servo motors, suction cups, vacuum pumps, electromagnets, etc. For simplicity, distribution of our components will follow the same process as the hardware booth.
* Teams will be given the option to mount/test their solutions with a Universal Robots UR10e cobot to demonstrate the capability of their solution.
  * Teams are not required to use the UR cobot, but if successfully integrated this may improve the team's score.
  * For safety reasons, teams will not be permitted to wire in their solution to the UR cobot, or have the UR cobot interact with their solution in any form other than physical mounting. Teams may only program/manipulate the pathing of the cobot using the designated software, and operation of their solution must be done independently of the UR.
* A variety of Ford/automotive related parts will be provided as sample grasping objects for teams to test their solutions with.
  * These components will encompass a variety of different shapes, sizes, weights, materials, and textures.
* Teams will be asked to check out parts individually to track possession and ensure there is enough supply for all. Come visit our booth during the hackathon for details.

## UR10e Cobot Specifications

| Specification | Value |
| :--- | :--- |
| Payload | 12.5 kg (27.5 lbs) |
| Reach | 1300 mm (51.2 in) |
| Degrees of freedom | 6 rotating joints |
| Programming | 12 inch touchscreen with PolyScope graphical user interface |
| Power consumption (average) | 615 W |
| Maximum power | 350 W |
| Moderate operating settings | Ambient temperature: $0-50^{\circ}C$ (32-122°F) |
| Safety functions | 17 configurable safety functions |
| In compliance with | EN ISO 13849-1 (PLd Category 3) and EN ISO 10218-1 |

### Force sensing, tool flange/torque sensor
| Performance | Force, x-y-z | Torque, x-y-z |
| :--- | :--- | :--- |
| Range | 100.0 N | ± 10.0 Nm |
| Precision | ± 5.0 N | ±0.2 Nm |
| Accuracy | ± 5.5 N | ± 0.5 Nm |

*Note: UR10e payload drops from 12.5 kg at 0-100 mm offset to approximately 5 kg at 800 mm center of gravity offset.*

## UR10e Mounting Specifications
* Specifications for the UR10e mounting interface:
* Lumberg RKMW 8-354 connector
* 4x M6-6H8 holes at 90° intervals on a 50 ±0.1 mm diameter
* Pin hole: 6 H7 6.20 ±0.20 at 45°
* Inner/Outer dimensions: 31.50 H7, 63 h8, 90 ±0.1 mm
* Depths: 6.50, 35.65, 48.75, 6.20 mm
* The tool output flange (ISO 9409-1-50-4-M6) is where the tool is mounted at the tip of the robot. All measures are in mm.

## Potential Parts
*Note: this list is not exclusive, and the final parts we bring may be different!*
* Item: Tapered Roller Bearing | Dimensions: $3^{\prime\prime}\times3^{\prime\prime}\times1/2^{\prime\prime}$
* Item: Helical Planetary Gear | Dimensions: $2^{\prime\prime}\times2^{\prime\prime}\times1^{\prime\prime}$
* Item: Ford Emblem | Dimensions: $9"\times3\frac{1}{2}"\times\frac{1}{4}"$
* Item: Reinforcement Plate | Dimensions: $12"\times8"\times1/8"$
* Item: Spark Plug | Dimensions: $1^{\prime\prime}\times1^{\prime\prime}\times3\frac{1}{2}"$
* Item: Oil Dipstick | Dimensions: 30" (flexible) x $0.050"\times0.050"$
* Item: Weatherstrip Segment | Dimensions: $12"\times1"\times1"\times1"$
* Item: Piston Rod | Dimensions: $9^{\prime\prime}\times4^{\prime\prime}\times4^{\prime\prime}$
* Item: Door Handle | Dimensions: $10"\times2"\times1"\times1"$
* Item: Wire Harness Segment | Dimensions: $8"\times1/2"\times1/2"\times1/2"$
* Item: Cabin Air Filter | Dimensions: $10^{\prime\prime}\times8^{\prime\prime}\times1^{\prime\prime}$
* Item: Gear Shift Selector | Dimensions: $5^{\prime\prime}\times5^{\prime\prime}\times4^{\prime\prime}$
* Item: Shims | Dimensions: $2"\times11/2"\times1/8"$
* Item: Engine Oil Filter | Dimensions: $4"\times3"\times3"\times3"$
* Item: Key Fob | Dimensions: $3"\times11/2"\times1/2"$
* Item: Oil Funnel | Dimensions: $6^{\prime\prime}\times6^{\prime\prime}\times8^{\prime\prime}$
* Item: Brake Pad | Dimensions: $6~1/2"\times2"\times1/2"$
* Item: Sheet Metal | Dimensions: $12"\times12"\times1/8"$
* Item: Lug Nut | Dimensions: $1"\times1"\times2"$
* Item: Interior Trim Retainer Clip | Dimensions: $3/4"\times1/2"\times1/4"\times1/4"$

## Common Gripper Types for Grasping

* **Simple Mechanical Grippers**
  * Pros: Fast, Low Cost, Reliable, Simple to design
  * Cons: Individual designs result in lack of versatility to complex shapes
* **Complex Mechanical Grippers**
  * Pros: Versatile and more functionality/capability, High precision and repeatability
  * Cons: Expensive, Complex to design, Often inefficient compared to grippers specialized for a task
* **Magnetic**
  * Pros: Fast grasping speed, Minimal maintenance, Versatility among magnetic materials
  * Cons: Relies on material being magnetic, Can induce magnetism in target part, leading the part to attract debris
* **Suction**
  * Pros: Cost effective, Applicable to a wide range of object shapes/materials, Lightweight
  * Cons: Surface texture, porosity, and contamination are important, More maintenance and higher wear
* **Conformable**
  * Pros: High level of flexibility for different object geometries, Gentle on fragile items
  * Cons: Limited payload, Durability, Slow, Limited precision
* **Adhesive (Gecko)**
  * Pros: Functions in vacuum, Energy efficient, Gentle/leaves no residue
  * Cons: Limited payload, May be difficult to detach, Sensitivity to surface texture/contamination, Durability

## Common Vision Sensors for Grasping

* **RGB Camera (2D Only)**
  * Pros: Cheap, High resolution, Fast frame rates, Effective for handling classification, color recognition, barcodes/labels/fiducial scanning
  * Cons: No direct depth sensing, Sensitive to lighting and shadows, Hard to resolve clutter/occlusion in 3D
* **Stereo Vision**
  * Pros: Passive depth sensing, Works outdoors/bright environments, Effective in scenes with good texture + baseline distance, Great at medium range
  * Cons: Calibration between cameras is critical, Struggles with low-texture, repetitive patterns, reflective/transparent objects, Noisy depth at edges and low light conditions
* **RGB-D Structured Light / Active IR**
  * Pros: Dense depth maps at short to medium range, Great in controlled indoor lighting, Simplifies 3D grasp heuristics for point cloud based grasping
  * Cons: Can fail in harsh sunlight/strong IR conditions, Ineffective on shiny, reflective, transparent, very dark materials
* **RGB-D Time of Flight (ToF)**
  * Pros: Effective for real-time applications (robust and fast), Works well with most surface conditions
  * Cons: Usually lower resolution than structured light, Struggles with shiny objects, foggy/dusty air conditions