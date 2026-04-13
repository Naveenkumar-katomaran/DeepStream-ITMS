import numpy as np

def interpolate_bboxes(bbox_list, num_intermediate=2):
    """
    Densify a list of N bounding boxes by linearly interpolating
    `num_intermediate` extra boxes between every consecutive pair.

    Args:
        bbox_list       : list of [(x1,y1),(x2,y2)] in full-frame pixel coords
        num_intermediate: how many new boxes to insert between each adjacent pair

    Returns:
        Densified list of [(x1,y1),(x2,y2)] — originals preserved in-order.
    """
    n = len(bbox_list)
    if n <= 1 or num_intermediate <= 0:
        return list(bbox_list)  # nothing to interpolate

    # Convert to a (N, 4) float32 array  [x1, y1, x2, y2]
    arr = np.array(
        [[b[0][0], b[0][1], b[1][0], b[1][1]] for b in bbox_list],
        dtype=np.float32
    )

    result = []
    for i in range(n - 1):
        start = arr[i]      # shape (4,)
        end   = arr[i + 1]  # shape (4,)

        # Append the anchor box
        result.append(bbox_list[i])

        # Generate `num_intermediate` evenly-spaced boxes (excluding endpoints)
        for t in np.linspace(0.0, 1.0, num_intermediate + 2)[1:-1]:
            interp = start + t * (end - start)
            x1, y1, x2, y2 = int(interp[0]), int(interp[1]), int(interp[2]), int(interp[3])
            result.append([(x1, y1), (x2, y2)])

    # Append the final anchor
    result.append(bbox_list[-1])
    return result
