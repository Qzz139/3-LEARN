"""Translate YOLO boxes to vision_msgs without depending on a specific ROS release."""


CLASS_NAMES = {32: 'sports ball', 39: 'bottle', 75: 'vase'}


def _center(box):
    return box.center.position if hasattr(box.center, 'position') else box.center


def to_detection_array(header, predictions, array_type, detection_type, hypothesis_type):
    message = array_type()
    message.header = header
    for item in predictions:
        detection = detection_type()
        detection.header = header
        x1, y1, x2, y2 = item['xyxy']
        center = _center(detection.bbox)
        center.x, center.y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        detection.bbox.size_x, detection.bbox.size_y = x2 - x1, y2 - y1
        hypothesis = hypothesis_type()
        if hasattr(hypothesis, 'hypothesis'):
            hypothesis.hypothesis.class_id = str(item['class_id'])
            hypothesis.hypothesis.score = item['confidence']
        else:
            hypothesis.id = str(item['class_id'])
            hypothesis.score = item['confidence']
        detection.results.append(hypothesis)
        message.detections.append(detection)
    return message


def from_detection_array(message, width, height):
    results = []
    for detection in message.detections:
        if not detection.results:
            continue
        hypothesis = detection.results[0]
        value = hypothesis.hypothesis if hasattr(hypothesis, 'hypothesis') else hypothesis
        try:
            class_id = int(value.class_id if hasattr(value, 'class_id') else value.id)
        except (ValueError, TypeError):
            continue
        center = _center(detection.bbox)
        half_x, half_y = detection.bbox.size_x / 2.0, detection.bbox.size_y / 2.0
        results.append({'class_id': class_id, 'class_name': CLASS_NAMES.get(class_id, str(class_id)),
                        'confidence': float(value.score),
                        'xyxy': [center.x - half_x, center.y - half_y,
                                 center.x + half_x, center.y + half_y],
                        'image_width': width, 'image_height': height})
    return results
