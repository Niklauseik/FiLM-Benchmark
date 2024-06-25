import torch
from transformers import AutoTokenizer

B_HEAD = 1
I_HEAD = 2
B_TAIL = 3
I_TAIL = 4
RELATION_OFFSET = 5

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

def create_label_ids(parsed_labels, tokenized_inputs, relation_to_id):
    max_len = tokenized_inputs['input_ids'].size(1)
    all_labels = torch.zeros((len(tokenized_inputs['input_ids']), max_len), dtype=torch.long)

    # Example inputs
    parsed_labels = [[('Facebook', 'Mark Zuckerberg', 'founded_by')]]
    tokenized_inputs = tokenizer(['Facebook was founded by Mark Zuckerberg.'], padding=True, truncation=True, return_tensors="pt")
    relation_to_id = {'founded_by': 0}

    print("Parsed Labels:", parsed_labels)
    print("Tokenized Inputs:", tokenized_inputs)
    print("Relation to ID:", relation_to_id)

    max_len = tokenized_inputs['input_ids'].size(1)
    all_labels = torch.zeros((len(tokenized_inputs['input_ids']), max_len), dtype=torch.long)
    print("Max Length:", max_len)
    print("Initialized Labels Tensor:", all_labels)

    for i, triplets in enumerate(parsed_labels):
        print(f"Sentence {i}: {triplets}")
        for head, tail, relation in triplets:
            if relation not in relation_to_id:
                print(f"Relation '{relation}' not found in relation_to_id mapping!")
                continue

            head_tokens = tokenizer.tokenize(head)
            tail_tokens = tokenizer.tokenize(tail)
            head_ids = tokenizer.convert_tokens_to_ids(head_tokens)
            tail_ids = tokenizer.convert_tokens_to_ids(tail_tokens)

            print(f"Head: {head}, Tail: {tail}, Relation: {relation}")
            print(f"Head Tokens: {head_tokens}, Tail Tokens: {tail_tokens}")
            print(f"Head IDs: {head_ids}, Tail IDs: {tail_ids}")

            head_start = torch.where(tokenized_inputs['input_ids'][i] == head_ids[0])[0]
            tail_start = torch.where(tokenized_inputs['input_ids'][i] == tail_ids[0])[0]
            print(f"Head Start Positions: {head_start}, Tail Start Positions: {tail_start}")

            if len(head_start) == 0 or len(tail_start) == 0:
                print(f"Tokens for '{head}' or '{tail}' not found in the tokenized input.")
                continue

            head_start = head_start[0].item()
            tail_start = tail_start[0].item()
            print(f"Head Start: {head_start}, Tail Start: {tail_start}")

            all_labels[i, head_start] = B_HEAD
            all_labels[i, head_start + 1:head_start + len(head_ids)] = I_HEAD

            all_labels[i, tail_start] = B_TAIL
            all_labels[i, tail_start + 1:tail_start + len(tail_ids)] = I_TAIL

            for pos in range(head_start + len(head_ids), tail_start):
                all_labels[i, pos] = relation_to_id[relation] + RELATION_OFFSET

            print("Updated Labels Tensor:", all_labels)

    return all_labels

# Run the example
parsed_labels = [[('Facebook', 'Mark Zuckerberg', 'founded_by')]]
tokenized_inputs = tokenizer(['Facebook was founded by Mark Zuckerberg.'], padding=True, truncation=True, return_tensors="pt")
relation_to_id = {'founded_by': 0}
create_label_ids(parsed_labels, tokenized_inputs, relation_to_id)
