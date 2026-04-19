# Ford Gripper Track: Master Script Backbone
**Current Total Word Count:** [ 232 / 300 MAX ]

*DIRECTOR'S NOTE: Fill in the blockquotes below with your spoken narration. Do not exceed the allocated word limits per section. Zero filler. Technical terminology only.*

---

## 1. Problem
**Section Limit:** [ 36 / 40 Words ]
**Visual:** Locked tripod shot. Medium-close up on the exact failure point or operational constraint (e.g., object slippage, extreme clutter). Ensure heavy task lighting.
**Pacing:** Establish the failure state immediately. Pause 1 second before transitioning.

**Audio Script:**
> Industrial depth cameras add cost, calibration overhead, and integration complexity for bin-picking. With 2D RGB alone, grasping in clutter fails due to occlusion and missing depth cues. We target reliable low-cost grasping without dedicated depth hardware.


---

## 2. Innovation
**Section Limit:** [ 58 / 80 Words ]
**Visual:** Full-screen capture using dedicated screen recording software. Show the computer vision software architecture (node graphs, terminal logs, or live bounding boxes). No camera-pointed-at-screen footage will be accepted.
**Pacing:** Deliberate. Let the UI elements breathe so judges can read the outputs.

**Audio Script:**
> We implement HEAP Grasp (University of Tokyo, 2023) with active multi-view perception. The UR10e captures synchronized RGB frames from several viewpoints around the object. A fusion module aggregates cross-view keypoints and pose hypotheses, then ranks grasp candidates by collision risk, reachability, and gripper alignment. The planner selects the highest-confidence grasp and outputs a Cartesian approach vector for execution.

---

## 3. Performance Demo
**Section Limit:** [ 57 / 80 Words ]
**Visual:** 50/50 Split Screen. 
* **Left:** Physical hardware executing the grasp (Gripper must fill 70% of the frame, zero shadows). 
* **Right:** Synchronized screen recording of the AI model output.
**Pacing:** Synchronize your audio beats with the physical gripper actuation on screen.

**Audio Script:**
> On the right, the model streams segmented object masks, grasp scores, and the selected end-effector pose in real time. On the left, UR10e follows the planned trajectory, aligns our gripper attachment to the predicted contact normal, and closes at the commanded depth. When confidence remains above threshold through final approach, the lift succeeds without slip or regrasp.

---

## 4. Evaluation Criteria
**Section Limit:** [ 81 / 100 Words ]
**Visual:** Continuous B-roll footage of successful grasping from multiple, well-lit angles. Overlay crisp, sans-serif text corresponding to each rubric criterion as you speak it.
**Pacing:** Steady, factual delivery. 

**Audio Script:**
> Impact: This pipeline reduces sensing bill-of-materials while enabling robust picking in cluttered industrial cells. Innovation: We combine HEAP Grasp with active multi-view capture on a standard UR10e workflow, replacing single-view heuristics with view-fused grasp ranking. Performance: In repeated trials, the system achieves high first-attempt success with stable cycle time. Technical Execution: Perception, planning, and control are synchronized with deterministic handoff and collision-aware constraints. Feasibility: The architecture uses commodity RGB cameras and manufacturable gripper hardware, supporting rapid deployment and low maintenance cost.