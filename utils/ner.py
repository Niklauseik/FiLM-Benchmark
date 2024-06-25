def get_entities(seq):
    """Extract entities from a sequence of labels"""
    entities = []
    current_entity = None
    current_type = None
    for idx, label in enumerate(seq):
        if label.startswith("B-"):
            if current_entity is not None:
                entities.append((tuple(current_entity), current_type))
            current_entity = [idx]
            current_type = label[2:]
        elif label.startswith("I-") and current_entity is not None and label[2:] == current_type:
            current_entity.append(idx)
        else:
            if current_entity is not None:
                entities.append((tuple(current_entity), current_type))
                current_entity = None
                current_type = None
    if current_entity is not None:
        entities.append((tuple(current_entity), current_type))
    return entities

def compute_metrics(true_labels, pred_labels):
    true_entities = [get_entities(seq) for seq in true_labels]
    pred_entities = [get_entities(seq) for seq in pred_labels]

    true_entities_flat = [entity for sublist in true_entities for entity in sublist]
    pred_entities_flat = [entity for sublist in pred_entities for entity in sublist]

    true_entities_set = set(true_entities_flat)
    pred_entities_set = set(pred_entities_flat)

    nb_correct = len(true_entities_set & pred_entities_set)
    nb_pred = len(pred_entities_set)
    nb_true = len(true_entities_set)

    precision = nb_correct / nb_pred if nb_pred > 0 else 0
    recall = nb_correct / nb_true if nb_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
