import numpy as np
from transformers import AutoTokenizer, TFBertForMaskedLM
import tensorflow as tf
import sys

# -------------------------------------------------------------------------
# My three functions – no top‑level matplotlib import
# -------------------------------------------------------------------------


def get_mask_token_index(mask_token_id, inputs):
    """
    Find the index (0‑based) of the [MASK] token in the input sequence.
    Return None if it's not there.
    """
    input_ids = inputs['input_ids'][0]
    for idx, token_id in enumerate(input_ids):
        if token_id == mask_token_id:
            return idx
    return None


def get_color_for_attention_score(score):
    """
    Convert an attention score (0..1) into an RGB gray triple.
    0.0 → (0,0,0) black, 1.0 → (255,255,255) white.
    Using truncation (floor) – accepted by the spec.
    """
    gray = int(score * 255)
    return (gray, gray, gray)


def visualize_attentions(tokens, attentions):
    """
    Generate one attention diagram per layer per head.
    """
    num_layers = len(attentions)
    for layer_idx, layer_attentions in enumerate(attentions):
        num_heads = layer_attentions.shape[1]
        for head_idx in range(num_heads):
            attention_matrix = layer_attentions[0, head_idx].numpy()
            generate_diagram(
                layer_idx + 1,
                head_idx + 1,
                tokens,
                attention_matrix
            )


# -------------------------------------------------------------------------
# generate_diagram – now imports matplotlib only when called
# -------------------------------------------------------------------------

def generate_diagram(layer_num, head_num, tokens, attention):
    import matplotlib.pyplot as plt   # <-- safe import: only if we actually plot

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    im = ax.imshow(attention, cmap='gray', vmin=0, vmax=1)

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)

    ax.set_xlabel('Token Attended To', fontsize=12)
    ax.set_ylabel('Token Attending From', fontsize=12)
    ax.set_title(f'Layer {layer_num}, Head {head_num}', fontsize=14)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f'attention_layer{layer_num}_head{head_num}.png', dpi=150)
    plt.close()


# -------------------------------------------------------------------------
# Original main (unchanged)
# -------------------------------------------------------------------------

def main():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFBertForMaskedLM.from_pretrained(model_name, output_attentions=True)

    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            text = f.read().strip()
    else:
        text = input("Text: ")

    if "[MASK]" not in text:
        print('The input must include exactly one "[MASK]" token.')
        return

    inputs = tokenizer(text, return_tensors="tf")
    token_ids = inputs["input_ids"].numpy()[0]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        print('Could not find "[MASK]" token in the input.')
        return

    outputs = model(inputs)
    predictions = outputs.logits[0, mask_token_index]
    top_k_indices = np.argsort(predictions)[-5:][::-1]

    print("\nTop 5 predictions:")
    for i in range(5):
        pred_token = tokenizer.decode([top_k_indices[i]]).strip()
        print(f"  {text.replace('[MASK]', pred_token)}")

    visualize_attentions(tokens, outputs.attentions)


if __name__ == "__main__":
    main()
