# %% [markdown]
# # Replication of Figures in the Paper

# %% [markdown]
# ### Setup


# %%
import math
import os
import pickle
import io


import imgkit
import matplotlib.pyplot as plt
import torch
from IPython.display import HTML, display
from transformer_lens import HookedTransformer, HookedTransformerConfig

from src import *


# Select best available device (CUDA, MPS on Apple Silicon, or CPU)
DEVICE = (
    "cuda" if torch.cuda.is_available() else (
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    )
)

# %%
file_name = "dataset.txt"
n_examples = 300_000
n_states = 16

dataset = GraphDataset(n_states, file_name, n_examples)
pred = generate_example(16, 0, order="backward")

cfg = HookedTransformerConfig(
    n_layers=6,
    d_model=128,
    n_ctx=dataset.max_seq_length - 1,
    n_heads=1,
    d_mlp=512,
    d_head=128,
    #attn_only=True,
    d_vocab=len(dataset.idx2tokens),
    device=DEVICE,
    attention_dir= "causal",
    act_fn="gelu",
)
model = HookedTransformer(cfg)


# Load in the model if weights are in the directory, else train new model
if os.path.exists("model.pt"):
    # Load checkpoint to CPU to avoid CUDA deserialization issues, then move to DEVICE
    state_dict = torch.load("model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(DEVICE)

# %%
pred = generate_example(16, 1234, order="backward") #"10>3,10>14,3>7,3>1,7>6,6>13,7>5,1>15,15>9,14>4,14>8,4>0,0>2,0>11,8>12|9:10>3>1>15>9"
parse_example(pred)

# %%
assert is_model_correct(model, dataset, pred)
labels, cache = get_example_cache(pred, model, dataset)

for l in range(model.cfg.n_layers):
    for h in range(model.cfg.n_heads):
        fig = display_head(cache, labels, l, h, show=True)

# %% [markdown]
# ### Tuned Lens
# * 4th lens score: 0.7257897283418372
# * 5th: 0.9673407840982938
# * 6th: 1.0

# %%
def load_or_compute_tuned_lenses(model, dataset, filename='tuned_lenses.pkl'):
    # Check if the tuned lenses are already saved in the directory
    if os.path.exists(filename):
        # Load the tuned lenses from the pickle file
        with open(filename, 'rb') as file:
            try:
                lenses = pickle.load(file)
            except RuntimeError:
                # If loading fails (e.g. CUDA tensors on CPU), try mapping to current device
                file.seek(0)
                class DeviceUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module == 'torch.storage' and name == '_load_from_bytes':
                            return lambda b: torch.load(io.BytesIO(b), map_location=DEVICE)
                        return super().find_class(module, name)
                lenses = DeviceUnpickler(file).load()
        
        # Ensure lenses are on the correct device
        lenses = {k: v.to(DEVICE) for k, v in lenses.items()}
        print(f'Loaded tuned lenses from {filename}')
    else:
        # Calculate the tuned lenses
        lenses = calculate_tuned_lens(model, dataset)
        # Save the computed tuned lenses to the directory in pickle format
        # Save as CPU tensors to ensure compatibility across devices
        lenses_cpu = {k: v.cpu() for k, v in lenses.items()}
        with open(filename, 'wb') as file:
            pickle.dump(lenses_cpu, file)
        print(f'Saved tuned lenses to {filename}')
    
    return lenses

def backtracking_viz_proj(pred, model, dataset, lenses=None, pos=47):
    # Get labels and cache
    labels, cache = get_example_cache(pred, model, dataset)
    # Calculate end idx of the labels
    end = num_last(labels, ",")
    # Get the logit lens for each layer's resid_post
    data = []
    for layer in range(1, model.cfg.n_layers+1):
        if layer < model.cfg.n_layers:
            act_name = tl_util.get_act_name("normalized", layer, "ln1")
        else:
            act_name = "ln_final.hook_normalized"
        res_stream = cache[act_name][0]
        if lenses is not None:
            out_proj = res_stream @ lenses[act_name]
        else:
            out_proj = res_stream @ model.W_U
        out_proj = (out_proj*6).softmax(dim=-1)[pos]
        lens_out = [(dataset.idx2tokens[i], out_proj[i].item()) for i in range(9, 19)] + [(dataset.idx2tokens[i], out_proj[i].item()) for i in range(3, 9)]
        data.append([f"Layer {layer} Lens"] + lens_out)
    
    # Generate table
    # Initialize the rows for text and background color
    colors = [[] for _ in data]  # The first column will be white for layer names

    # Go through each layer's data, and populate the text and colors
    for i, layer in enumerate(data):
        for cond, val in layer[1:]:
            
            # Calculate a color based on the value, interpolating between white (0) and deep blue (1)
            colors[i].append(val)
    
    # Set up a matplotlib figure with 3x2 subplots
    fig, axs = plt.subplots(3, 2, figsize=(10, 15), constrained_layout=True)  # 3 rows, 2 columns for 6 graphs

    for i, layer in enumerate(colors):
        highlighted = []
        for node, logit in enumerate(layer):
            logit = 1-1*math.exp(-4*logit)
            highlighted.append((node, logit))
        ax = axs.flatten()[i]
        parse_example(pred, highlight_nodes=highlighted, ax=ax)
        ax.set_title(f"Layer {i+1}", fontsize=14, weight='bold')        
    
    # Adjust layout for better spacing
    #plt.tight_layout()
    return fig


# %%
lenses = load_or_compute_tuned_lenses(model, dataset)

# %%
os.makedirs("images", exist_ok=True)
pred = generate_example(16, 89342568, path_length=5, order="backward")
parse_example(pred)
fig = backtracking_viz_proj(pred, model, dataset, lenses=lenses)
fig.savefig("images/length_5.png")

# %%
pred = generate_example(16, 89342568, path_length=8, order="backward")
parse_example(pred)
fig = backtracking_viz_proj(pred, model, dataset, lenses=lenses)
fig.savefig("images/length_8.png")

# %%
pred = generate_example(16, 89342568, path_length=10, order="backward")
parse_example(pred)
fig = backtracking_viz_proj(pred, model, dataset, lenses=lenses)
fig.savefig("images/length_10.png")

# %%
pred = generate_example(16, 89342568, path_length=13, order="backward")
parse_example(pred)
fig = backtracking_viz_proj(pred, model, dataset, lenses=lenses)
fig.savefig("images/length_13.png")

# %%
pred = generate_example(16, 89342568, path_length=10, order="backward")
parse_example(pred)
fig = backtracking_viz_proj(pred, model, dataset, lenses=lenses, pos=41)
fig.savefig("images/length_10.png")

# %% [markdown]
# ### Attention Patterns

# %%
def create_table(table_data, header, file_name):
    # Formatting for the HTML
    css = """
    <style type="text/css">
        .content-wrapper {
            border: 1px solid #000;
            padding: 10px;
            display: inline-block;
            border-radius: 15px;
            font-family: Arial, sans-serif; /* Change font-family */
        }
        .header {
            font-weight: bold;
            text-align: center;
            background-color: #333;
            color: white; 
            margin: -10px;
            padding: 10px;
            border-top-left-radius: 14px;
            border-top-right-radius: 14px;
            font-size: 20px; /* Increase font size */
        }
        .header + .row {
            padding-top: 15px;
        }
        .row {
            padding: 5px 0;
            line-height: 1.5; /* Increase line height */
            font-size: 16px; /* Increase font size */
        }
    </style>
    """

    # Convert the table data to HTML with colored substrings
    html_content = '<div class="content-wrapper">'
    html_content += f'<div class="header">{header}</div>'

    for row in table_data:
        html_row = '<div class="row">'
        for text, color in row:
            if color:
                html_row += f"<span style='background-color: {color};'>{text}</span>"
            else:
                html_row += text
        html_row += '</div>'
        html_content += html_row

    html_content += '</div>'

    # Create the complete HTML with CSS
    html_str = f"<html><head>{css}</head><body>{html_content}</body></html>"
    display(HTML(html_str))

    # Generate the image using imgkit (assuming imgkit is properly configured)
    options = {
        'format': 'png',
        'quality': '100',
        'width': '140',
    }

    try:
        imgkit.from_string(html_str, file_name, options=options)
    except Exception as e:
        print(f"[warn] Could not render image via imgkit: {e}")
        print("[hint] Install wkhtmltoimage (wkhtmltopdf suite) and ensure it is on PATH.\n"
              "macOS: download .pkg from https://wkhtmltopdf.org/downloads.html and reopen the terminal,\n"
              "or set IMGKIT_CONFIG to the installed binary path.")


def generate_att_table(example, header, file_path, pos=47, norm=True):
    table_data = []
    labels, cache = get_example_cache(example, model, dataset)
    for i in range(6):
        pattern = cache[f"blocks.{i}.attn.hook_pattern"][0][0][pos].tolist()[:pos+1]
        if norm:
            pattern = [p / max(pattern) for p in pattern]
        pattern = [f"rgba(220, 20, 60, {min(1, p):.2f})" for p in pattern]
        row = list(zip(labels, pattern))
        table_data.append(row)
    create_table(table_data, header, file_path)

# %%
model.reset_hooks()

ex = "14>15,13>14,12>13,11>12,10>11,9>10,0>9,7>8,6>7,5>6,4>5,3>4,2>3,1>2,0>1|15:0"
pred, _ = eval_model(model, dataset, ex)
assert is_model_correct(model, dataset, pred)
parse_example(ex)
plt.show()
generate_att_table(ex, "Attention from the First Path Token", "images/backchaining_main.png", norm=True)

# %%
generate_att_table(ex, "Attention from the First Path Token", "images/backchaining_rp36.png", pos=36, norm=False)

# %%
generate_att_table(ex, "Attention from the First Path Token", "images/backchaining_rp38.png", pos=38, norm=False)

# %%
generate_att_table(ex, "Attention from the First Path Token", "images/backchaining_rp42.png", pos=42, norm=False)

# %% [markdown]
# ##### Causal Analysis of Subpaths
# 

# %%
def attention_knockout_registers(model, dataset, test_graph, threshold=0.7):
    # Evaluate model on test_graph
    model.reset_hooks()
    correct, base_probs = is_model_correct(model, dataset, test_graph, return_probs=True)
    assert correct
    labels, cache = get_example_cache(test_graph, model, dataset)
    
    # Iterate over heads in each layer
    important = []
    ablated = []
    
    register_tokens = [36, 38, 39, 41, 42, 44, 45]
    
    for register in register_tokens:
        model.reset_hooks()
        
        for abl in ablated:
            # add_attention_blockout(model, 3, 0, 47, abl)
            add_attention_blockout(model, 4, 0, 47, abl)
            add_attention_blockout(model, 5, 0, 47, abl)

        # add_attention_blockout(model, 3, 0, 47, register)
        add_attention_blockout(model, 4, 0, 47, register)
        add_attention_blockout(model, 5, 0, 47, register)
    
        correct, new_probs = is_model_correct(model, dataset, test_graph, return_probs=True)
        score = kl_divergence(base_probs[0:1], new_probs[0:1])
                    
        if correct and (threshold is None or score < threshold):
            ablated.append(register)
        else:
            important.append(register)
                        
    return important


def find_largest_excluding_indices(tensor, exclude_indices):
    """
    Returns the index of the largest value in the tensor that is not in the specified indices to exclude.

    Parameters:
    - tensor (torch.Tensor): The input tensor.
    - exclude_indices (list or torch.Tensor): Indices to exclude from consideration.

    Returns:
    - int: The index of the largest value in the tensor excluding specified indices.
    """
    # Clone the tensor to avoid modifying the original tensor
    modified_tensor = tensor.clone()

    # Set the values at the excluded indices to negative infinity
    if isinstance(exclude_indices, list):
        exclude_indices = torch.tensor(exclude_indices, dtype=torch.long)
    modified_tensor[exclude_indices] = float('-inf')

    # Find the index of the largest value excluding the specified indices
    return modified_tensor.argmax().item()


def is_valid_path(dataset, graph, subpath):
    # Extract the adjacency matrix from the graph
    mat = extract_adj_matrix(graph)
    
    # Iterate through pairs of consecutive nodes in subpath
    for i in range(len(subpath) - 1):
        start, end = subpath[i], subpath[i + 1]
        
        # Check if these nodes are directly connected in the graph
        # Adjust the condition based on how your adjacency matrix is structured
        if mat[start][end] == 0:
            return False  # Early return if any pair is not directly connected
    
    # If all pairs are directly connected, return True
    return True


def view_subpath(dataset, graph, pos, max_layer=5):
    # Calculate the children of the root node
    tokens = dataset.tokenize(graph)[:-1]
    start_idx = np.where(tokens == dataset.start_token)[0].item() + 2
    labels = [dataset.idx2tokens[idx] for idx in tokens]
    mat = extract_adj_matrix(graph)
    current_node = int( labels[start_idx-1].replace(">", "") )
    children = mat[current_node]
    # Calculate subpath from attention pattern
    node_order = []
    register_tokens = [36, 38, 39, 41, 42, 44, 45, 46]
    labels, cache = get_example_cache(graph, model, dataset)
    for i in range(max_layer):
        pattern = cache[f"blocks.{i}.attn.hook_pattern"][0][0][pos].tolist()[:pos+1]
        pattern_argmax = find_largest_excluding_indices(torch.tensor(pattern), [x for x in register_tokens if x < pos])
        if i == 0:
            try:
                node = int(labels[pattern_argmax])
            except:
                continue
        else:
            node = int(labels[pattern_argmax - 1])
            node_order.append(node)
        if children[node]:
            break
    # Create one hot encoding
    t = torch.zeros(16)
    for node in node_order:
        t[node] = 1
    return node_order[::-1], t

# %%
finding_cases = []
for i in range(1000):
    model.reset_hooks()
    path_length = np.random.randint(7, 12)
    examples = generate_example_secondary_path(
        n_states=16,
        seed=np.random.randint(0, 1_000_000_000),
        path_length=9,
        second_path_min_length=6,
        order="random",
        return_all_leafs=True
    )
    for example in examples:
        try:
            important = attention_knockout_registers(model, dataset, example)
            #print(important)
            if len(important) > 0:
                finding_cases.append((example, important))
        except:
            pass
           # print("Failed")
model.reset_hooks()


# %%
for idx, (example, important) in enumerate(finding_cases):
    #parse_example(example)
    plt.show()
    subpath_pos = [47,]
    try:
        for i in important:
            subpath, _ = view_subpath(dataset, example, i)
            if is_valid_path(dataset, example, subpath):
                #print(i, subpath)
                if len(subpath) >= 3:
                    subpath_pos.append(i)
        if len(subpath_pos) > 2:
            print(idx, len(subpath_pos))
    except:
        pass

# %%
example, important = finding_cases[123]
parse_example(example)
plt.show()
subpath_pos = [47,]
for i in important:
    subpath, _ = view_subpath(dataset, example, i)
    if is_valid_path(dataset, example, subpath):
        print(i, subpath)
        subpath_pos.append(i)

# %%
example, important = finding_cases[123]
parse_example(example)
plt.show()
subpath_pos = [47,]
for i in important:
    subpath, _ = view_subpath(dataset, example, i)
    if is_valid_path(dataset, example, subpath):
        if len(subpath) > 2:
            print(i, subpath)
            subpath_pos.append(i)


def convert_to_latex(example, positions, norm=True):
    labels, cache = get_example_cache(example, model, dataset)
    colors = ["red", "blue", "green", "orange", "pink"]

    table = [[ [None, 0] for _ in range(max(positions)+1)] for _ in range(model.cfg.n_layers)]
    for pos in positions[::-1]:
        for layer in range(6):
            if layer >=4 and pos != 47:
                continue
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0][0][pos].tolist()[:pos+1]
            if norm:
                pattern = [p / max(pattern) for p in pattern]
            if pos!=47 and layer == 0:
                pattern = [int(p*100*0.8) for p in pattern]
            else:
                pattern = [int(p*100*2) for p in pattern]
            for ctx_pos, p in enumerate(pattern):
                if p >= table[layer][ctx_pos][1]:
                    table[layer][ctx_pos][0] = colors[positions.index(pos)]
                    table[layer][ctx_pos][1] = p
    
    latex_str = ""
    with open("latex.txt", "w") as f:
        for layer in range(len(table)):
            f.write("\n\hspace{-0.1cm}\n")
            for idx in range(len(table[layer])):
                label_name = labels[idx].replace(">", r"$\rightarrow$\,")
                if idx == len(table[layer]) - 1:
                    f.write("\hlfancy{" + table[layer][idx][0] + "!" + str(table[layer][idx][1]) + "}{" +  label_name + "}\\\\\n")
                else:
                    f.write("\hlfancy{" + table[layer][idx][0] + "!" + str(table[layer][idx][1]) + "}{" +  label_name + "}\n")
                if (idx + 2) % 3 == 0:
                    f.write("\hspace{-0.15cm}\n")

    
convert_to_latex(example, subpath_pos)

# %% [markdown]
# ### Linear Probes/Concept Erasure

# %%
def linear_probe_all(X_train, y_train, X_test, y_test, mult=False):
    results = []
    for xkey in X_train.keys():    
        for ykey in y_train.keys():
            if mult:
                probe = LinearMultiClsProbe(max_iter=1000, learning_rate_init=1e-3, verbose=False)
                probe.fit(X_train[xkey].astype(np.float32), y_train[ykey].astype(np.float32))
                score = probe.score(X_test[xkey].astype(np.float32), y_test[ykey].astype(np.float32))
            else:            
                probe = LinearClsProbe(max_iter=100, learning_rate_init=1e-2, verbose=False)
                probe.fit(X_train[xkey].astype(np.float32), y_train[ykey].astype(np.int64))
                score = probe.score(X_test[xkey].astype(np.float32), y_test[ykey].astype(np.int64))
            print(f"{xkey} {ykey}", score)
            results.append( (xkey, ykey, score) )
    return results

# %%
def nonlinear_probe_all(X_train, y_train, X_test, y_test, mult=False):
    results = []
    for xkey in X_train.keys():    
        for ykey in y_train.keys():
            if mult:
                probe = NonlinearMultiClsProbe(max_iter=1000, learning_rate_init=1e-3, verbose=False)
                probe.fit(X_train[xkey].astype(np.float32), y_train[ykey].astype(np.float32))
                score = probe.score(X_test[xkey].astype(np.float32), y_test[ykey].astype(np.float32))
            else:            
                probe = NonlinearClsProbe(max_iter=100, learning_rate_init=1e-2, verbose=False)
                probe.fit(X_train[xkey].astype(np.float32), y_train[ykey].astype(np.int64))
                score = probe.score(X_test[xkey].astype(np.float32), y_test[ykey].astype(np.int64))
            print(f"{xkey} {ykey}", score)
            results.append( (xkey, ykey, score) )
    return results

# %%
def block_registers(registers=[36, 38, 39, 41, 42, 44, 46]):
    for pos in range(47, model.cfg.n_ctx):
        for layer in range(1, model.cfg.n_layers):
            for register in registers:
                add_attention_blockout(model, layer, 0, pos, register)
    
def block_leaves():
    for pos in range(0, 46):
        for layer in range(0, model.cfg.n_layers):
            add_attention_blockout(model, layer, 0, 46, pos)

# %% [markdown]
# ##### Edge Aggregation in layer 0

# %%
from sklearn import preprocessing


def generate_dataset():
    # Sample completions from model
    X, graphs = aggregate_activations(
        model=model,
        dataset=dataset,
        activation_keys=["blocks.0.hook_resid_pre", "blocks.0.hook_resid_mid"],
        n_samples=1024
    )
    # Generate data
    newX = {}
    for key in X.keys():
        tensor_list = [ x[0, [i for i in range(45) if (i) % 3 == 0]] for x in X[key] ]
        newX[key.split('.')[2] + " outgoing"] = torch.cat(tensor_list, dim=0).detach().cpu().numpy()
        
        tensor_list = [ x[0, [i for i in range(45) if (i-1) % 3 == 0]] for x in X[key] ]
        newX[key.split('.')[2] + " incoming"] = torch.cat(tensor_list, dim=0).detach().cpu().numpy()
    # Generate labels
    y = {"incoming": [], "outgoing": []}
    for graph in graphs:
        tokens = dataset.tokenize(graph)[:-1]
        labels = [dataset.idx2tokens[idx] for idx in tokens]
        y["outgoing"].append([labels[i-1] for i in range(45) if (i-1)%3 == 0])
        y["incoming"].append([labels[ i ] for i in range(45) if (i-1)%3 == 0])
    y["incoming"] = preprocessing.LabelEncoder().fit_transform( np.array(y["incoming"]).flatten() )
    y["outgoing"] = preprocessing.LabelEncoder().fit_transform( np.array(y["outgoing"]).flatten() )

    return newX, y


import warnings
warnings.filterwarnings('always')

X_train, y_train = generate_dataset()
X_test, y_test = generate_dataset()

results = linear_probe_all(X_train, y_train, X_test, y_test)

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create DataFrame
df = pd.DataFrame(results, columns=['source', 'target', 'value'])
df["target"] = df["target"].str.replace("incoming", r"$B_i$").str.replace("outgoing", "$A_i$")
df["source"] = df["source"].str.replace("incoming", r"$B_i$").str.replace("outgoing", "$A_i$").str.replace("hook_resid_", "")


# Separate 'pre' and 'post' data
df_pre = df[df['source'].str.contains('pre')].copy()
df_post = df[df['source'].str.contains('mid')].copy()

df_pre["source"]= df_pre["source"].str.replace("pre ", "")
df_post["source"]= df_post["source"].str.replace("mid ", "")

# Pivot DataFrames to create a 2D structure for each one
heatmap_df_pre = df_pre.pivot(index="source", columns="target", values="value")
heatmap_df_post = df_post.pivot(index="source", columns="target", values="value")

# Set the style of the visualization
sns.set_style("white")

# Setup the color palette to be a gradient of blues
blue_cmap = sns.light_palette("#a275ac", as_cmap=True)

# Create subplots for side-by-side heatmaps with adjustable width_ratios
fig, ax = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.5})

# Plot heatmaps and maintain aspect ratio to make plots square
sns.heatmap(heatmap_df_pre, annot=True, fmt=".2f", cmap=blue_cmap, cbar=False, annot_kws={"size": 12}, ax=ax[0], square=True)
sns.heatmap(heatmap_df_post, annot=True, fmt=".2f", cmap=blue_cmap, cbar=False, annot_kws={"size": 12}, ax=ax[1], square=True)

# Improve the visibility of the heatmap
ax[0].set_title('Before Layer 1', pad=20, fontsize=18)
ax[1].set_title('After Layer 1', pad=20, fontsize=18)

# Rotate x-axis tick labels for both heatmaps (axis labeling is removed from here)
for axis in ax:
    
    axis.tick_params(axis='x', rotation=0, which='both', labelsize=16)
    axis.tick_params(axis='y', rotation=0, which='both', labelsize=16)

    # Remove the axis labels for a cleaner look
    axis.set_xlabel('')  # Remove the x-axis label
    axis.set_ylabel('')  # Remove the y-axis label 

# Set common axis labels for the entire figure instead of individual plots
fig.text(0.5, 0.04, 'Labels', ha='center', va='center', fontsize=18)
fig.text(0.04, 0.5, 'Inputs', ha='center', va='center', rotation='vertical', fontsize=18) 

# Remove axis ticks and spines for a cleaner look
sns.despine(left=True, bottom=True)

plt.tight_layout()  # Adjust the layout to fit better
plt.show()

fig.savefig("images/heatmaps1.png")


# %%
print("A->(A,B) Before", results[1][-1] * results[0][-1])
print("B->(A,B) Before", results[3][-1] * results[2][-1])


print("A->(A,B) After", results[5][-1] * results[4][-1])
print("B->(A,B) Before", results[7][-1] * results[6][-1])


# %%
X_train.keys()

# %%
# Validate the causal importance of the linear probe
model.reset_hooks()
eraser = ConceptErasure(X_train["hook_resid_mid incoming"], y_train["outgoing"])
def leace_hook(activation, hook):
    scrubbed = eraser.predict(activation[:, 1:45:3].cpu())
    activation[:, 1:45:3] = scrubbed.to(DEVICE)
    return activation
model.blocks[0].hook_resid_mid.add_hook(leace_hook)


start_seed = 250_000
num_samples = 100

for path_length in range(1, 15):
  total_correct = 0
  for seed in range(start_seed, start_seed + num_samples):
      graph = generate_example(16, seed, order="backward", path_length=path_length)
      correct = is_model_correct(model, dataset, graph)
      if correct:
        total_correct += 1
      else:
        pred, _ = eval_model(model, dataset, graph)
        plt.show()

  print(f"Length {path_length}: {100* total_correct / num_samples:.4f}%")

# %%
ex = "14>15,13>14,12>13,11>12,10>11,9>10,0>9,7>8,6>7,5>6,4>5,3>4,2>3,1>2,0>1|15:0"
pred, _ = eval_model(model, dataset, ex)
print(pred)
parse_example(ex)
plt.show()
generate_att_table(ex, "Attention From First Path Token", "images/leaced_edge_backchaining.png", norm=False)

# %% [markdown]
# ##### Goal aggregation in layer 0

# %%
def generate_dataset():
    # Sample completions from model
    X, graphs = aggregate_activations(
        model=model,
        dataset=dataset,
        activation_keys=["blocks.0.hook_resid_pre", "blocks.0.hook_resid_mid"],
        n_samples=1024
    )
    # Generate data
    newX = {}
    for key in X.keys():
        tensor_list = []
        for idx, graph in enumerate(graphs):
            tokens = dataset.tokenize(graph)[:-1]
            start_idx = np.where(tokens == dataset.start_token)[0].item() + 2
            labels = [dataset.idx2tokens[idx] for idx in tokens]
            end_idx = num_last(labels, ",") + 1
            tensor_list.append(X[key][idx][0, start_idx-1:end_idx-1])
        newX[key] = torch.cat(tensor_list, dim=0).detach().cpu().numpy()
        # Generate labels
    y = {"goal": []}
    for idx, graph in enumerate(graphs):
        # Tokenize graph
        tokens = dataset.tokenize(graph)[:-1]
        start_idx = np.where(tokens == dataset.start_token)[0].item() + 2
        labels = [dataset.idx2tokens[idx] for idx in tokens]
        end_idx = num_last(labels, ",") + 1
        for label_idx in range(start_idx-1, end_idx - 1):
            y["goal"].append(int(labels[45]))
    y["goal"] = np.vstack(y["goal"]).flatten().astype(np.int64)
    return newX, y


X_train, y_train = generate_dataset()
X_test, y_test = generate_dataset()

results = linear_probe_all(X_train, y_train, X_test, y_test)

# %%
# Create DataFrame
df = pd.DataFrame(results, columns=['source', 'target', 'value'])

df["source"] = df["source"].str.replace("path", "").str.replace("hook_resid_post ", "Path After \n Layer 1").str.replace("hook_resid_pre ", "Path Before \n Layer 1")
df["target"] = df["target"].str.replace("goal", "Goal")
heatmap_df = df.pivot(index="source", columns="target", values="value")


# Set the style of the visualization
sns.set_style("white")

# Define the color map as a gradient of blues
blue_cmap = sns.light_palette("#a275ac", as_cmap=True)

# Plot heatmap; increase figure size for a larger display
fig = plt.figure(figsize=(4.8, 4.8))  # You can adjust the dimensions as needed
ax = sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap=blue_cmap, cbar=False, square=True, linewidths=.5)

# Improve the visibility of the heatmap
ax.set_xticklabels(ax.get_xticklabels(), horizontalalignment='right')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.tick_params(axis='y', which='both', length=0, labelsize=14)  # Hide y tick marks for cleanliness
ax.tick_params(axis='x', which='both', length=0, labelsize=14)  # Hide y tick marks for cleanliness

# Set common axis labels for the entire figure instead of individual plots
ax.text(0.42, 2.2, 'Labels', ha='center', va='center', fontsize=16)
ax.text(-0.7, 0.95, 'Inputs', ha='center', va='center', rotation='vertical', fontsize=16) 


# Remove the axis labels for a cleaner look
ax.set_xlabel('')  # Remove the x-axis label
ax.set_ylabel('')  # Remove the y-axis label  

# Remove axis ticks and spines for a cleaner look
sns.despine(left=True, bottom=True)

sns.despine(left=True, bottom=True)  # Remove borders for a cleaner look
plt.tight_layout()  # Adjust the layout to fit better
plt.show()

fig.savefig("images/heatmaps2.png")


# %%
# Validate the causal importance of the linear probe
model.reset_hooks()
eraser = ConceptErasure(X_train["blocks.0.hook_resid_mid"], y_train["goal"])
def leace_hook(activation, hook):
    scrubbed = eraser.predict(activation[:, 47:].cpu())
    activation[:, 47:] = scrubbed.to(DEVICE)
    return activation
model.blocks[0].hook_resid_mid.add_hook(leace_hook)

start_seed = 250_000
num_samples = 100

for path_length in range(1, 15):
  total_correct = 0
  for seed in range(start_seed, start_seed + num_samples):
      graph = generate_example(16, seed, order="backward", path_length=path_length)
      correct = is_model_correct(model, dataset, graph)
      if correct:
        total_correct += 1
      else:
        pred, _ = eval_model(model, dataset, graph)
        plt.show()

  print(f"Length {path_length}: {100* total_correct / num_samples:.4f}%")

# %%
ex = "14>15,13>14,12>13,11>12,10>11,9>10,0>9,7>8,6>7,5>6,4>5,3>4,2>3,1>2,0>1|15:0"
pred, _ = eval_model(model, dataset, ex)
print(pred)
parse_example(ex)
plt.show()
generate_att_table(ex, "Attention From First Path Token", "images/leaced_goal_backchaining.png", norm=False)

# %% [markdown]
# ##### Probe for children of current node

# %%
from sklearn import preprocessing

model.reset_hooks()
block_registers()

def generate_dataset(n_samples, endshift=4):
    # Sample completions from model
    X, graphs = aggregate_activations(
        model=model,
        dataset=dataset,
        activation_keys=["blocks.4.hook_resid_pre", "blocks.4.hook_resid_mid", "blocks.5.hook_resid_mid"],
        n_samples=n_samples,
        path_length=10
    )
    # Generate data
    newX = {}
    for key in X.keys():
        tensor_list = []
        for idx, graph in enumerate(graphs):
            tokens = dataset.tokenize(graph)[:-1]
            start_idx = np.where(tokens == dataset.start_token)[0].item() + 2
            labels = [dataset.idx2tokens[idx] for idx in tokens]
            end_idx = num_last(labels, ",") + 1 - endshift
            if len(X[key][0].shape)==3:
                tensor_list.append(X[key][idx][0, start_idx-1:end_idx-1])
                #tensor_list = [ x[0, 47:48] for x in X[key] ]
            elif len(X[key][0].shape)==4:
                tensor_list.append(X[key][idx][0, start_idx-1:end_idx-1, 0])
                # tensor_list = [ x[0, 47:48, 0] for x in X[key] ]
        newX[key] = torch.cat(tensor_list, dim=0).detach().cpu().numpy()

    # Generate labels
    y = {"leaves": [], "children":[]}
    for idx, graph in enumerate(graphs):
        # Tokenize graph
        tokens = dataset.tokenize(graph)[:-1]
        start_idx = np.where(tokens == dataset.start_token)[0].item() + 2
        labels = [dataset.idx2tokens[idx] for idx in tokens]
        end_idx = num_last(labels, ",") + 1 - endshift
        mat = extract_adj_matrix(graph)
        leaves = find_leaf_nodes(graph)
        not_leaves = np.logical_not(leaves).astype(int)
        # Add all the children for each node in the path
        for label_idx in range(start_idx-1, end_idx - 1):
            current_node = int( labels[label_idx].replace(">", "") )
            children = mat[current_node]
            y["children"].append(children)
            y["leaves"].append(children & not_leaves) # Add the nodes that are both children and leaves

    y["leaves"] = np.vstack(y["leaves"])
    y["children"] = np.vstack(y["children"])

    return newX, y, graphs

X_train, y_train, train_graphs = generate_dataset(16384)
X_test, y_test, test_graphs = generate_dataset(4096)

results = linear_probe_all(X_train, y_train, X_test, y_test, mult=True)

# %%
model.reset_hooks()

eraser = ConceptErasure(X_train["blocks.5.hook_resid_mid"], y_train["leaves"])
def leace_hook(activation, hook):
    scrubbed = eraser.predict(activation[:, 47:].cpu())
    activation[:, 47:] = scrubbed.to(DEVICE)
    return activation
model.blocks[5].hook_resid_mid.add_hook(leace_hook)

start_seed = 250_000
num_samples = 100

for path_length in range(1, 15):
  total_correct = 0
  for seed in range(start_seed, start_seed + num_samples):
      graph = generate_example(16, seed, order="backward", path_length=path_length)
      correct = is_model_correct(model, dataset, graph)
      if correct:
        total_correct += 1
      else:
        pred, _ = eval_model(model, dataset, graph)
        plt.show()

  print(f"Length {path_length}: {100* total_correct / num_samples:.4f}%")

# %% [markdown]
# ##### Probing for Subpaths
# 

# %%
finding_cases = []
for i in range(5000):
    model.reset_hooks()
    examples = generate_example_secondary_path(16, np.random.randint(0, 1_000_000_000), order="backward", return_all_leafs=True)
    for example in examples:
        try:
            important = attention_knockout_registers(model, dataset, example)
            #print(important)
            if len(important) > 0:
                finding_cases.append((example, important))
        except:
            pass
           # print("Failed")
model.reset_hooks()


# %%

X = []
y = []

for example, important in finding_cases:
    try:
        for i in important:
            subpath, encoding = view_subpath(dataset, example, i)
            if is_valid_path(dataset, example, subpath):
                _, cache = get_example_cache(example, model, dataset)
                X.append(cache["blocks.5.hook_resid_pre"][0, i])
                y.append(encoding)
    except ValueError:
        pass

X = torch.stack(X)
y = torch.stack(y)

# %%
from sklearn import preprocessing

split = int(X.shape[0]*0.8)
X_train = {"blocks.5.hook_resid_pre": X[:split].cpu().detach().numpy()}
y_train = {"subpath": y[:split].cpu().detach().numpy()}

X_test = {"blocks.5.hook_resid_pre": X[split:].cpu().detach().numpy()}
y_test = {"subpath": y[split:].cpu().detach().numpy()}

results = linear_probe_all(X_train, y_train, X_test, y_test, mult=True)
results = nonlinear_probe_all(X_train, y_train, X_test, y_test, mult=True)

# %% [markdown]
# ### Causal Scrubbing
# 

# %%
def sample_bifurcated(path_length):
 # Sample random trees where the root node has at least 2 children
    while True:
        example = generate_example(
            n_states=16,
            seed=np.random.randint(0, 1_000_000_000),
            order='backward',
            path_length=path_length
        )
        if not is_model_correct(model, dataset, example):
            continue
        tokens = dataset.tokenize(example)
        labels = [dataset.idx2tokens[idx] for idx in tokens]
        current_node = int( labels[47].replace(">", "") )
        mat = extract_adj_matrix(example)
        n_children = mat[current_node].sum()
        if n_children > 1:
            return example
        
def block_registers(registers=[36, 38, 39, 41, 42, 44, 46]):
    for pos in range(47, model.cfg.n_ctx):
        for layer in range(1, model.cfg.n_layers):
            for register in registers:
                add_attention_blockout(model, layer, 0, pos, register)

def block_goal(registers=[45]):
    for pos in range(47, model.cfg.n_ctx):
        for layer in range(1, model.cfg.n_layers):
            for register in registers:
                add_attention_blockout(model, layer, 0, pos, register)

def block_leaves():
    for pos in range(0, 46):
        for layer in range(0, model.cfg.n_layers):
            add_attention_blockout(model, layer, 0, 46, pos)

def get_inverse_path(graph):
    path = [int(x) for x in graph.split(":")[1].split(">")][1:]
    return path[::-1] + [path[0]] * 6


def get_root_children(graph):
    tokens = dataset.tokenize(graph)[:-1]
    labels = [dataset.idx2tokens[idx] for idx in tokens]
    mat = extract_adj_matrix(graph)
    current_node = int( labels[47].replace(">", "") )
    return np.where(mat[current_node])[0].tolist()

def get_matching_graph(node, dist_from_goal, childs=None):
    while True:
        graph = generate_example(16, np.random.randint(0, 1_000_000_000), order="backward", path_length=np.random.randint(dist_from_goal, 15))
        path = get_inverse_path(graph)
        if childs is not None and get_root_children(graph) != childs:
            continue
        if path[dist_from_goal-1] == node:
            return graph
    
def layer_specific(layer, hypothesis, n_samples=100):
    recovered = []
    variances = []
    
    for pl in range(1, 15):
        
        orig_losses = []
        new_losses = []
        rand_losses = []
                
        for i in range(n_samples):
            orig_loss, new_loss, rand_loss = hypothesis(pl, layer=layer)
            orig_losses.append(orig_loss)
            new_losses.append(new_loss)
            rand_losses.append(rand_loss)
        
        orig_losses = np.array(orig_losses)
        new_losses = np.array(new_losses)
        rand_losses = np.array(rand_losses)
        rec = (new_losses - rand_losses) / (orig_losses - rand_losses) * 100
                
        #loss_rec = ((sum(new_losses) / n_samples) - (sum(rand_losses) / n_samples)) / ((sum(orig_losses) / n_samples) - (sum(rand_losses) / n_samples)) * 100
        recovered.append(np.mean(rec))
        variances.append(np.std(rec))
    
    return recovered, variances

# %%
def test_hypothesis_1(path_length, layer=None, position=47):
    model.reset_hooks()

    graph = generate_example_secondary_path(
        n_states=16,
        seed=np.random.randint(0, 1_000_000_000),
        path_length=path_length,
        order="backward",
        second_path_min_length=2
    )
    path = get_inverse_path(graph)
    if layer is None:
        layer = np.random.randint(0, 6)
        
    resampled_graph = get_matching_graph(path[layer], layer+1)
    
    clean_labels, clean_cache  = get_example_cache(graph, model, dataset)
    corrupt_labels, corrupt_cache  = get_example_cache(resampled_graph, model, dataset)
    
    # Patch attention_out at layer, position and do the forward pass again
    def patching_hook(activation, hook):
        activation[:, position, :] = corrupt_cache[hook.name][:, position, :]
        return activation
    model.blocks[layer].hook_attn_out.add_hook(patching_hook)
    #if visualize:
    #    generate_att_table(clean_graph, "Patching Result", "images/test.png", norm=False)
    new_labels, new_cache  = get_example_cache(graph, model, dataset)
    
    original = (clean_cache["ln_final.hook_normalized"] @ model.W_U)[0, position].log_softmax(dim=0)
    new = (new_cache["ln_final.hook_normalized"] @ model.W_U)[0, position].log_softmax(dim=0)
    
    random_dist = torch.zeros_like(new)
    #random_dist[3:19] = 100
    random_dist = random_dist.log_softmax(dim=0)
    
    orig_loss = -original[original.argmax()]
    new_loss = -new[original.argmax()]
    rand_loss = -random_dist[original.argmax()]
    #print(original.argmax() == new.argmax())
    
    return orig_loss.item(), new_loss.item(), rand_loss.item()

loss_rec = {}
loss_rec_var = {}

for layer in range(6):
    loss_rec[layer], loss_rec_var[layer] = layer_specific(layer, test_hypothesis_1, n_samples=150)

# Setup figure and axis
fig, ax = plt.subplots()

# Plotting each list in the dictionary
for label, values in loss_rec.items():
    # Adjusting the x-values to start at 1 instead of 0
    x_values = range(1, len(values) + 1)
    ax.plot(x_values, values, label=f"Layer {label}")

# Setting the legend
ax.legend()

# Setting axis names
ax.set_xlabel('Path Length')
ax.set_ylabel(f'% of Loss Recovered')

# Setting ticks on the x-axis to match your data (starting at 1)
plt.xticks(range(1, len(list(loss_rec.values())[0]) + 1))

# Show plot
plt.show()

# %%
def test_hypothesis_2(path_length, layer=None, position=47):
    model.reset_hooks()

    graph = generate_example_secondary_path(
        n_states=16,
        seed=np.random.randint(0, 1_000_000_000),
        path_length=path_length,
        order="backward",
        second_path_min_length=2
    ) # generate_example(16, np.random.randint(0, 1_000_000_000), path_length=path_length, order="backward")
    path = get_inverse_path(graph)

    if layer is None:
        layer = np.random.randint(0, 6)
        
    resampled_graph = get_matching_graph(path[layer], layer+1)
    
    clean_labels, clean_cache  = get_example_cache(graph, model, dataset)
    corrupt_labels, corrupt_cache  = get_example_cache(resampled_graph, model, dataset)
    
    # Patch attention_out at layer, position and do the forward pass again
    def patching_hook(activation, hook):
        layer_name = int(hook.name.split(".")[1])
        register_tokens = [36,38,39,41,42,44,45]
        # Recompute the attn_hook_out without the register tokens from a different input
        corrupt_values = corrupt_cache[f"blocks.{layer}.attn.hook_v"] # B x P x H x D
        corrupt_pattern = corrupt_cache[f"blocks.{layer}.attn.hook_pattern"]
        corrupt_vp = torch.einsum("bqhd,bhpq->bpqhd", corrupt_values, corrupt_pattern)
        
        clean_values = clean_cache[f"blocks.{layer}.attn.hook_v"] # B x P x H x D
        clean_pattern = clean_cache[f"blocks.{layer}.attn.hook_pattern"]
        clean_vp = torch.einsum("bqhd,bhpq->bpqhd", clean_values, clean_pattern)
        
        corrupt_vp[:, :, register_tokens] = clean_vp[:, :, register_tokens]
        corrupt_z = corrupt_vp.sum(dim=2)
        # Compute the B x H x P x P x D s.t. summing over second p gives you v
        activation[:, position] = corrupt_z[:, position]
        return activation

    model.blocks[layer].attn.hook_z.add_hook(patching_hook)
    #if visualize:
    #    generate_att_table(clean_graph, "Patching Result", "images/test.png", norm=False)
    new_labels, new_cache  = get_example_cache(graph, model, dataset)
    
    original = (clean_cache["ln_final.hook_normalized"] @ model.W_U)[0, position].log_softmax(dim=0)
    new = (new_cache["ln_final.hook_normalized"] @ model.W_U)[0, position].log_softmax(dim=0)

    random_dist = torch.zeros_like(new)
    #random_dist[3:19] = 100
    random_dist = random_dist.log_softmax(dim=0)
    
    orig_loss = -original[original.argmax()]
    new_loss = -new[original.argmax()]
    rand_loss = -random_dist[original.argmax()]
    #print(original.argmax() == new.argmax())
    
    return orig_loss.item(), new_loss.item(), rand_loss.item()


loss_rec = {}
loss_rec_var = {}

for layer in range(6):
    loss_rec[layer], loss_rec_var[layer] = layer_specific(layer, test_hypothesis_2, n_samples=150)
# Setup figure and axis
fig, ax = plt.subplots()

# Plotting each list in the dictionary
for label, values in loss_rec.items():
    # Adjusting the x-values to start at 1 instead of 0
    x_values = range(1, len(values) + 1)
    ax.plot(x_values, values, label=f"Layer {label}")

# Setting the legend
ax.legend()

# Setting axis names
ax.set_xlabel('Path Length')
ax.set_ylabel(f'% of Loss Recovered')

# Setting ticks on the x-axis to match your data (starting at 1)
plt.xticks(range(1, len(list(loss_rec.values())[0]) + 1))

# Show plot
plt.show()
    

# %%
def test_hypothesis_3(path_length, layer=None, position=47):
    model.reset_hooks()

    graph = generate_example_secondary_path(
        n_states=16,
        seed=np.random.randint(0, 1_000_000_000),
        path_length=path_length,
        order="backward",
        second_path_min_length=2
    ) # generate_example(16, np.random.randint(0, 1_000_000_000), path_length=path_length, order="backward")
    path = get_inverse_path(graph)
    if layer is None:
        layer = np.random.randint(0, 6)
        
    resampled_graph = get_matching_graph(path[layer], layer+1)
    
    clean_labels, clean_cache  = get_example_cache(graph, model, dataset)
    corrupt_labels, corrupt_cache  = get_example_cache(resampled_graph, model, dataset)
    
    # Patch attention_out at layer, position and do the forward pass again
    def patching_hook(activation, hook):
        layer_name = int(hook.name.split(".")[1])
        register_tokens = [i for i in range(46) if (i+2)%3 == 0] + [36,38,39,41,42,44,45]
        # Recompute the attn_hook_out without the register tokens from a different input
        corrupt_values = corrupt_cache[f"blocks.{layer}.attn.hook_v"] # B x P x H x D
        corrupt_pattern = corrupt_cache[f"blocks.{layer}.attn.hook_pattern"]
        corrupt_vp = torch.einsum("bqhd,bhpq->bpqhd", corrupt_values, corrupt_pattern)
        
        clean_values = clean_cache[f"blocks.{layer}.attn.hook_v"] # B x P x H x D
        clean_pattern = clean_cache[f"blocks.{layer}.attn.hook_pattern"]
        clean_vp = torch.einsum("bqhd,bhpq->bpqhd", clean_values, clean_pattern)
        
        corrupt_vp[:, :, register_tokens] = clean_vp[:, :, register_tokens]
        corrupt_z = corrupt_vp.sum(dim=2)
        # Compute the B x H x P x P x D s.t. summing over second p gives you v
        activation[:, position] = corrupt_z[:, position]
        return activation

    model.blocks[layer].attn.hook_z.add_hook(patching_hook)
    #if visualize:
    #    generate_att_table(clean_graph, "Patching Result", "images/test.png", norm=False)
    new_labels, new_cache  = get_example_cache(graph, model, dataset)
    
    original = (clean_cache["ln_final.hook_normalized"] @ model.W_U)[0, position].log_softmax(dim=0)
    new = (new_cache["ln_final.hook_normalized"] @ model.W_U)[0, position].log_softmax(dim=0)

    random_dist = torch.zeros_like(new)
    #random_dist[3:19] = 100
    random_dist = random_dist.log_softmax(dim=0)
    
    orig_loss = -original[original.argmax()]
    new_loss = -new[original.argmax()]
    rand_loss = -random_dist[original.argmax()]
    #print(original.argmax() == new.argmax())
    
    return orig_loss.item(), new_loss.item(), rand_loss.item()

loss_rec = {}
loss_rec_var = {}

for layer in range(6):
    loss_rec[layer], loss_rec_var[layer] = layer_specific(layer, test_hypothesis_3, n_samples=150)
# Setup figure and axis
fig, ax = plt.subplots()

# Plotting each list in the dictionary
for label, values in loss_rec.items():
    # Adjusting the x-values to start at 1 instead of 0
    x_values = range(1, len(values) + 1)
    ax.plot(x_values, values, label=f"Layer {label}")

# Setting the legend
ax.legend()

# Setting axis names
ax.set_xlabel('Path Length')
ax.set_ylabel(f'% of Loss Recovered')

# Setting ticks on the x-axis to match your data (starting at 1)
plt.xticks(range(1, len(list(loss_rec.values())[0]) + 1))

# Show plot
plt.show()

# %% [markdown]
# ### Head Compositions
# 

# %%
scores = torch.zeros(model.cfg.n_ctx, model.cfg.n_ctx)
for q in range(model.cfg.n_ctx):
    for k in range(model.cfg.n_ctx):
        q_out = model.blocks[0].ln1( model.W_pos[q:q+1] ) @ model.blocks[0].attn.W_Q[0] + model.blocks[0].attn.b_Q
        k_out = model.blocks[0].ln1( model.W_pos[k:k+1] ) @ model.blocks[0].attn.W_K[0] + model.blocks[0].attn.b_K
        
        if k < q:
            scores[q, k] = q_out @ k_out.T


plt.imshow(scores.softmax(dim=1).detach(), cmap="RdBu", vmin=-1, vmax=1)
plt.ylabel("Position Embedding in Query")
plt.xlabel("Position Embedding in Key")

plt.savefig('images/M_qk1.png', dpi=300)


# %%
scores2 = torch.zeros(16, 16)

for i in range(16):
    goal_rep = model.W_E[dataset.tokens2idx[str(i)]][None, :] #+ model.W_pos[45:46]
    v1 = goal_rep @ model.blocks[0].attn.W_V[0] # + model.blocks[0].attn.b_V
    o1 = v1 @ model.blocks[0].attn.W_O[0]  + model.blocks[0].attn.b_O
    
    
    o1 = model.blocks[0].ln2(o1)
    o1 = o1 + model.blocks[0].mlp(o1[None, :])[0]

    for j in range(16):
        key = model.W_E[dataset.tokens2idx[">"+str(j)]][None, :] @ model.blocks[1].attn.W_K[0] # + model.blocks[1].attn.b_K
        query = o1 @ model.blocks[1].attn.W_Q[0] #+ model.blocks[1].attn.b_Q
        scores2[i, j] = query @ key.T


plt.imshow((scores2).softmax(-1).detach())
plt.title("K-Compositon Backtracking Mechanism")

plt.xticks(list(range(16)), [str(i) for i in range(16)])
plt.yticks(list(range(16)), [">" + str(i) for i in range(16)])

plt.xlabel("Query Token (Goal Representation)")
plt.ylabel("Key Token")

plt.savefig('images/M_qk2.png', dpi=300)


# %%
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

scores = torch.zeros(model.cfg.n_ctx, model.cfg.d_vocab)
for q in range(model.cfg.n_ctx):
    for k in range(model.cfg.d_vocab):
        
        q_out = model.blocks[0].ln1( model.W_pos[q:q+1] ) @ model.blocks[0].attn.W_Q[0] + model.blocks[0].attn.b_Q
        k_out = model.blocks[0].ln1( model.W_E[k:k+1] ) @ model.blocks[0].attn.W_K[0] + model.blocks[0].attn.b_K
        
        scores[q, k] = q_out @ k_out.T

scores = torch.cat([
    scores.softmax(dim=1)[:, 19+6:19+16].detach(),
    scores.softmax(dim=1)[:, 19:19+6].detach()
], dim=1)

fig, ax = plt.subplots()  # create figure and axes objects

position_list=[36, 38, 39, 41, 42, 44, 45]
scores = scores[position_list]

ax.imshow(scores, cmap="Blues", vmin=0, vmax=1)

ax.hlines(np.arange(0.5, scores.shape[0]), *ax.get_xlim(), color='grey', linestyle=':', linewidth=0.5, alpha = 0.7)
ax.vlines(np.arange(0.5, scores.shape[1]), *ax.get_ylim(), color='grey', linestyle=':', linewidth=0.5, alpha = 0.7)

ax.set_xticks(list(range(scores.shape[1]))) 
ax.set_xticklabels(list(range(scores.shape[1])), rotation=45) 
ax.set_yticks(list(range(scores.shape[0]))) 
ax.set_yticklabels(position_list, rotation=0)

ax.set_ylabel("Position Embedding in Query")
ax.set_xlabel("Token Embedding in Key")

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(ax.get_images()[0], ax=ax,cax=cax)  # create colorbar  

plt.tight_layout()

plt.savefig('images/register_comp.png', dpi=300)


# %% [markdown]
# ### Understanding Backup Mechanism

# %%
model.reset_hooks()
ex = generate_example_secondary_path(16, np.random.randint(0, 1_000_000_000), path_length=10, second_path_min_length=2, order="backward")
#print(eval_model(model, dataset, ex))
while not is_model_correct(model, dataset, ex):
    ex = generate_example_secondary_path(16, np.random.randint(0, 1_000_000_000), path_length=10, second_path_min_length=2, order="backward")
parse_example(ex)

labels, cache = get_example_cache(ex, model, dataset)


# %%
important_pos = [1 + 3*i for i in range(13)] # + [36, 38, 39, 41, 42, 44, 46] # + [47]
M = model.blocks[5].attn.W_O @ model.W_U
x = cache["blocks.5.attn.hook_v"][0, important_pos, 0] @ M
dataset.idx2tokens[x.sum(1).argmax()]


# %%
model.reset_hooks()
ex = generate_example_secondary_path(16, np.random.randint(0, 1_000_000_000), path_length=10, second_path_min_length=2, order="backward")
#print(eval_model(model, dataset, ex))
#assert is_model_correct(model, dataset, ex)
parse_example(ex)

labels, cache = get_example_cache(ex, model, dataset)


important_pos = [1 + 3*i for i in range(15)] + [47] # + [36, 38, 39, 41, 42, 44, 46] # + [47]
M_4 = model.blocks[4].attn.W_O @ model.W_U
M_5 = model.blocks[5].attn.W_O @ model.W_U
X_4 = cache["blocks.4.attn.hook_v"][0, important_pos, 0] * cache['blocks.4.attn.hook_pattern'][0,0,47, important_pos][:, None].clip(0,0.07)
X_5 = cache["blocks.5.attn.hook_v"][0, important_pos, 0] * cache['blocks.5.attn.hook_pattern'][0,0,47, important_pos][:, None].clip(0,0.07)
out = (X_4 @ M_4) + (X_5 @ M_5)
out = (out - out.median())[0, :, 3:19]
x_labels = dataset.idx2tokens[3:19]
y_labels = [labels[i-1]+labels[i] for i in important_pos]

# Rearrange x axis
out = torch.cat([out[:, 6:], out[:, :6]], dim=-1)
x_labels = x_labels[6:] + x_labels[:6]

# Create the heatmap using imshow
plt.figure(figsize=(20, 8))    # Set the figure size as you like

plt.imshow(out.cpu().detach(), cmap='RdBu', vmax=10, vmin=-10, aspect='auto')  # Choose a colormap that suits your preference

# Configure the ticks
plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels, fontsize="14")  # Set x-axis labels
plt.yticks(ticks=np.arange(len(important_pos)), labels=y_labels, fontsize="14")  # Optionally, set y-axis labels if needed

# Adding a colorbar to show the scale
plt.colorbar()

# Optional enhancements
plt.xlabel('Output Token')
plt.ylabel('Edge in Context')
#plt.title('Direct Logit A of Edges to the Unembed through Layers 4 & 5')

# Show the plot
plt.savefig('images/dla_backup.png', dpi=300)


# %%
for l in range(model.cfg.n_layers):
    top_t = (cache[f"blocks.{l}.hook_attn_out"][:, 47] @ model.W_U).argmax()
    print(dataset.idx2tokens[top_t])
    
M = (model.blocks[5].attn.W_O @ model.W_U)[0]
important_pos = [1 + 3*i for i in range(16)] + [36, 38, 39, 41, 42, 44, 46]
contribs = cache["blocks.5.attn.hook_v"][0, important_pos, 0] @ M
