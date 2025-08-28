# 3D-Tumor-Bounding-Box-Prediction-from-3D-Images-with-YOLO-and-Volumetric-Evaluation-Metrics


This repository contains code for predicting **3D bounding boxes** of objects of interest (particularly tumors) from Medical Images (particularly MRI Volumes) using a **slice-by-slice YOLO detector** and evaluating the results with multiple 3D metrics.

The pipeline demonstrates how 2D object detection can be extended into 3D by stacking slice-level predictions, applying robust post-processing, and using metrics designed for volumetric comparisons.

---

## 🚀 Frameworks and Libraries

The following frameworks were used:

* **Ultralytics YOLO** → 2D object detection on MRI slices.
* **PyTorch** → deep learning backend.
* **OpenCV** → image preprocessing and RGB conversion.
* **SimpleITK** → medical image I/O (`.nii` volumes).
* **NumPy** → numerical operations.
* **SciPy (ConvexHull)** → polygon clipping and overlap area in BEV (XY plane).
* **Matplotlib** → visualization of slices and 3D boxes.

---

## 📂 Workflow Overview

1. **Data loading**

   * MRI scans are loaded as 3D volumes (`.nii` format).
   * Each scan consists of a sequence of 2D slices.

2. **Slice-wise YOLO detection**

   * Each slice is normalized to 0–255 and converted to RGB.
   * YOLO predicts bounding boxes per slice.
   * Predictions below a confidence threshold are discarded.

3. **3D box construction**

   * Predictions are collected across slices.
   * The **longest consecutive run of slices** with detections is chosen to ensure continuity.
   * Percentiles (10th, 90th) are used for x and y boundaries to reduce noise.
   * The z-axis is slightly expanded (±1) to avoid underestimation.
   * A single 3D predicted box is formed with `{xmin, ymin, zmin, xmax, ymax, zmax}`.

4. **Evaluation metrics**
   The predicted 3D box is compared against the ground truth box using:

   * **3D IoU (Intersection over Union):** measures volumetric overlap. Very strict; small z-errors significantly reduce the score.
   * **Dice coefficient:** smoother version of IoU, widely used in medical imaging.
   * **BEV IoU:** overlap in the XY plane (bird’s-eye view). Useful to check localization when z thickness differs.
   * **Centroid distance:** Euclidean distance between centers of GT and predicted boxes. Reflects localization accuracy.
   * **Dimension errors (Δx, Δy, Δz):** differences in box size along each axis.

---

## 📊 Why IoU Can Be Unforgiving

IoU in 3D is **very sensitive**. Even if the XY overlap is excellent, small errors in predicted slice range (`zmin`, `zmax`) reduce the overlap drastically.

Since the 3D box is created by **stacking 2D predictions**, small inconsistencies across slices accumulate and hurt IoU.
That is why additional metrics (Dice, BEV IoU, centroid distance, dimension errors) are critical for a complete performance picture.

---

## 🧮 Metric Implementations

```python
import numpy as np
from scipy.spatial import ConvexHull

# ------------------------
# 3D IoU and Dice
# ------------------------
def iou3d_axis(boxA, boxB):
    xA, yA, zA = max(boxA["xmin"], boxB["xmin"]), max(boxA["ymin"], boxB["ymin"]), max(boxA["zmin"], boxB["zmin"])
    xB, yB, zB = min(boxA["xmax"], boxB["xmax"]), min(boxA["ymax"], boxB["ymax"]), min(boxA["zmax"], boxB["zmax"])
    inter = max(0, xB-xA) * max(0, yB-yA) * max(0, zB-zA)
    volA = (boxA["xmax"]-boxA["xmin"]) * (boxA["ymax"]-boxA["ymin"]) * (boxA["zmax"]-boxA["zmin"])
    volB = (boxB["xmax"]-boxB["xmin"]) * (boxB["ymax"]-boxB["ymin"]) * (boxB["zmax"]-boxB["zmin"])
    return inter / (volA+volB-inter+1e-6)

def dice3d(boxA, boxB):
    iou = iou3d_axis(boxA, boxB)
    return (2*iou) / (1+iou)

# ------------------------
# Localization metrics
# ------------------------
def centroid_distance(boxA, boxB):
    cA = [(boxA["xmin"]+boxA["xmax"])/2, (boxA["ymin"]+boxA["ymax"])/2, (boxA["zmin"]+boxA["zmax"])/2]
    cB = [(boxB["xmin"]+boxB["xmax"])/2, (boxB["ymin"]+boxB["ymax"])/2, (boxB["zmin"]+boxB["zmax"])/2]
    return np.linalg.norm(np.array(cA)-np.array(cB))

def dimension_errors(boxA, boxB):
    dimsA = [boxA["xmax"]-boxA["xmin"], boxA["ymax"]-boxA["ymin"], boxA["zmax"]-boxA["zmin"]]
    dimsB = [boxB["xmax"]-boxB["xmin"], boxB["ymax"]-boxB["ymin"], boxB["zmax"]-boxB["zmin"]]
    return [dimsB[i]-dimsA[i] for i in range(3)]

# ------------------------
# BEV IoU (XY overlap)
# ------------------------
def poly_area(x,y): return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def polygon_clip(subjectPolygon, clipPolygon):
    def inside(p): return (cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
    def computeIntersection():
        dc = [cp1[0]-cp2[0], cp1[1]-cp2[1]]
        dp = [s[0]-e[0], s[1]-e[1]]
        n1, n2 = cp1[0]*cp2[1]-cp1[1]*cp2[0], s[0]*e[1]-s[1]*e[0]
        n3 = 1.0/(dc[0]*dp[1]-dc[1]*dp[0])
        return [(n1*dp[0]-n2*dc[0])*n3, (n1*dp[1]-n2*dc[1])*n3]
    outputList, cp1 = subjectPolygon, clipPolygon[-1]
    for clipVertex in clipPolygon:
        cp2, inputList, outputList, s = clipVertex, outputList, [], subjectPolygon[-1]
        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s): outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s): outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if not outputList: return None
    return outputList

def convex_hull_intersection(p1, p2):
    inter_p = polygon_clip(p1, p2)
    if inter_p is None: return None, 0.0
    hull_inter = ConvexHull(inter_p)
    return inter_p, hull_inter.volume

def bev_iou(boxA, boxB):
    rect1 = [(boxA["xmin"], boxA["ymin"]), (boxA["xmax"], boxA["ymin"]),
             (boxA["xmax"], boxA["ymax"]), (boxA["xmin"], boxA["ymax"])]
    rect2 = [(boxB["xmin"], boxB["ymin"]), (boxB["xmax"], boxB["ymin"]),
             (boxB["xmax"], boxB["ymax"]), (boxB["xmin"], boxB["ymax"])]
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    if inter is None: return 0.0
    return inter_area/(area1+area2-inter_area+1e-6)
```

---

## 🧪 Example Usage

```python
gt_box   = {'xmin': 89,'ymin':96,'zmin':53,'xmax':124,'ymax':129,'zmax':64}
pred_box = {'xmin': 85,'ymin':96,'zmin':55,'xmax':122,'ymax':128,'zmax':63}

print("3D IoU:", iou3d_axis(gt_box,pred_box))
print("Dice:", dice3d(gt_box,pred_box))
print("Centroid distance:", centroid_distance(gt_box,pred_box))
print("Dimension errors:", dimension_errors(gt_box,pred_box))
print("BEV IoU:", bev_iou(gt_box,pred_box))
```

---

## 📌 Key Takeaways

* Slice-level YOLO predictions can be extended into 3D by smart aggregation (longest run + percentiles + z expansion).
* 3D IoU is very strict and often harsh when stacking 2D predictions; other metrics are essential to capture the full picture.
* Dice and BEV IoU provide more forgiving and clinically meaningful assessments.
* Centroid distance and dimension errors help explain where the prediction went wrong: mislocalization vs. under/over-estimation of size.




