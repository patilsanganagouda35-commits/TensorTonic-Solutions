import numpy as np

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    stride = image_size / feature_size
    
    # Compute grid cell centers
    offsets = (np.arange(feature_size) + 0.5) * stride  # (feature_size,)
    cx, cy = np.meshgrid(offsets, offsets)  # both (feature_size, feature_size)
    cx = cx.flatten()  # (feature_size^2,)
    cy = cy.flatten()  # (feature_size^2,)
    
    anchors = []
    
    # Iterate: row-major (grid cells), then scales, then aspect ratios
    for i in range(len(cx)):
        for s in scales:
            for r in aspect_ratios:
                w = s * np.sqrt(r)
                h = s / np.sqrt(r)
                
                x1 = cx[i] - w / 2
                y1 = cy[i] - h / 2
                x2 = cx[i] + w / 2
                y2 = cy[i] + h / 2
                
                anchors.append([x1, y1, x2, y2])
    
    return anchors