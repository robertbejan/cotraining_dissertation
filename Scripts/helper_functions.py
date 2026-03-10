import numpy


def serialize_confusion_matrix(cm):
    flat_cm = cm.flatten()
    cm_string = ",".join(str(x) for x in flat_cm)
    return cm_string


def deserialize_confusion_matrix(cm_string, shape):
    values = [int(x) for x in cm_string.split(",")]
    cm = numpy.array(values).reshape(shape)
    return cm


def load_and_analyze_confusion_matrix(csv_row):
    cm_string = csv_row["rgb_confusion_matrix"]
    num_classes = int(csv_row["num_classes"])
    shape = (num_classes, num_classes)

    cm = deserialize_confusion_matrix(cm_string, shape)

    per_class_accuracy = []
    per_class_precision = []
    per_class_recall = []

    for i in range(num_classes):
        true_positives = cm[i, i]
        total_for_class = sum(cm[i, :])
        accuracy = true_positives / total_for_class if total_for_class > 0 else 0
        per_class_accuracy.append(accuracy)

        predicted_as_i = sum(cm[:, i])
        precision = true_positives / predicted_as_i if predicted_as_i > 0 else 0
        per_class_precision.append(precision)

        recall = accuracy
        per_class_recall.append(recall)

    return {"confusion_matrix": cm,
            "per_class_accuracy": per_class_accuracy,
            "per_class_precision": per_class_precision,
            "per_class_recall": per_class_recall}
