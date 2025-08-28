import numpy as np
from scipy.spatial import ConvexHull

# ===============================
# Polygon helpers (for BEV IoU)
# ===============================
def poly_area(x, y):
    """Compute polygon area from coordinates"""
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def polygon_clip(subjectPolygon, clipPolygon):
    """Sutherland–Hodgman polygon clipping"""
    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])
    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]
    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]
        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return outputList

def convex_hull_intersection(p1, p2):
    """Intersection polygon + area between two convex polygons"""
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  

# ===============================
# Metric helpers
# ===============================
def iou3d_axis(boxA, boxB):
    """Axis-aligned 3D IoU for min/max boxes"""
    xA = max(boxA["xmin"], boxB["xmin"])
    yA = max(boxA["ymin"], boxB["ymin"])
    zA = max(boxA["zmin"], boxB["zmin"])
    xB = min(boxA["xmax"], boxB["xmax"])
    yB = min(boxA["ymax"], boxB["ymax"])
    zB = min(boxA["zmax"], boxB["zmax"])
    interVol = max(0, xB - xA) * max(0, yB - yA) * max(0, zB - zA)
    volA = (boxA["xmax"] - boxA["xmin"]) * (boxA["ymax"] - boxA["ymin"]) * (boxA["zmax"] - boxA["zmin"])
    volB = (boxB["xmax"] - boxB["xmin"]) * (boxB["ymax"] - boxB["ymin"]) * (boxB["zmax"] - boxB["zmin"])
    return interVol / (volA + volB - interVol + 1e-6)

def dice3d(boxA, boxB):
    """3D Dice score based on IoU"""
    iou = iou3d_axis(boxA, boxB)
    return (2 * iou) / (1 + iou)

def centroid_distance(boxA, boxB):
    """Euclidean distance between box centroids"""
    cA = [(boxA["xmin"] + boxA["xmax"]) / 2,
          (boxA["ymin"] + boxA["ymax"]) / 2,
          (boxA["zmin"] + boxA["zmax"]) / 2]
    cB = [(boxB["xmin"] + boxB["xmax"]) / 2,
          (boxB["ymin"] + boxB["ymax"]) / 2,
          (boxB["zmin"] + boxB["zmax"]) / 2]
    return np.linalg.norm(np.array(cA) - np.array(cB))

def dimension_errors(boxA, boxB):
    """Difference in box dimensions along each axis"""
    dimsA = [boxA["xmax"] - boxA["xmin"], 
             boxA["ymax"] - boxA["ymin"], 
             boxA["zmax"] - boxA["zmin"]]
    dimsB = [boxB["xmax"] - boxB["xmin"], 
             boxB["ymax"] - boxB["ymin"], 
             boxB["zmax"] - boxB["zmin"]]
    return [dimsB[i] - dimsA[i] for i in range(3)]

def bev_iou(boxA, boxB):
    """2D BEV IoU in XY-plane"""
    rect1 = [(boxA["xmin"], boxA["ymin"]), (boxA["xmax"], boxA["ymin"]),
             (boxA["xmax"], boxA["ymax"]), (boxA["xmin"], boxA["ymax"])]
    rect2 = [(boxB["xmin"], boxB["ymin"]), (boxB["xmax"], boxB["ymin"]),
             (boxB["xmax"], boxB["ymax"]), (boxB["xmin"], boxB["ymax"])]
    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    if inter is None: 
        return 0.0
    return inter_area / (area1 + area2 - inter_area + 1e-6)

# ===============================
# Example usage
# ===============================
if __name__ == "__main__":
    gt_box   = {'xmin': 89,'ymin': 96,'zmin': 53,'xmax':124,'ymax':129,'zmax':64}
    pred_box = {'xmin': 85,'ymin': 96,'zmin': 55,'xmax':122,'ymax':128,'zmax':63}

    print("Axis-aligned IoU:", iou3d_axis(gt_box, pred_box))
    print("Dice:", dice3d(gt_box, pred_box))
    print("Centroid distance:", centroid_distance(gt_box, pred_box))
    print("Dim errors (dx,dy,dz):", dimension_errors(gt_box, pred_box))
    print("BEV IoU:", bev_iou(gt_box, pred_box))
