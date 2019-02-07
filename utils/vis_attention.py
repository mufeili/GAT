import errno
import math
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import random
import torch
from PIL import Image

matplotlib.rcParams.update({'font.size': 22})

def vis_atten(g, nodes_labels, N, attn_coefs, num_heads):
    """
    Visualize edge attentions for all heads.

    g: nx.DiGraph
        Directed networkx graph
    nodes_labels: list, numpy.array or torch.tensor
        nodes_labels[i] specifies the label of the ith node, which will
        decide the node color on the plot. Default to be None. If None,
        all nodes will have the same canonical label. The nodes_labels
        should contain labels for all nodes to be plot.
    N: int
        Number of nodes to sample
    attn_coefs: list
        List of attention coefficients where each element is an
        array of shape (1, num_nodes, num_nodes), where num_nodes
        is the number of nodes in the graph
    num_heads: list
        List of integers where list[i] specifies the number of heads
        in the i-th layer.
    """
    assert N > 0, 'Expect the number of samples to be positive.'
    nx_sub, nodes_sampled, nodes_to_plot = sample_subgraph(g, N)
    atten_coefs = [torch.tensor(w).squeeze(0) for w in attn_coefs]

    # Get the subgraph based on the nodes selected
    pos = nx.spring_layout(nx_sub)

    # Prepare directories
    dir_list = ['./real', './hist', './normalized']
    for dir in dir_list:
        try:
            os.makedirs(dir, exist_ok=True)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    for normalize_edge_color in [True, False]:
        curr_head_id = 0
        titles = []
        for layer, l_heads in enumerate(num_heads):
            if layer < len(num_heads) - 1:
                prefix = 'layer {:d}'.format(layer + 1)
            else:
                prefix = 'final layer'
            for i in range(l_heads):
                title = ', '.join([prefix, 'head {:d}'.format(i + 1)])
                titles.append(title)
                atten_coefs_head = atten_coefs[curr_head_id + i]
                plot(g, atten_coefs_head, title,
                     nodes_to_plot=nodes_to_plot, nodes_labels=nodes_labels,
                     edges_to_plot=g.in_edges(nodes_sampled),
                     nodes_pos=pos,
                     normalize_edge_color=normalize_edge_color)
                max_discrepancy = 0
                for j in range(atten_coefs_head.size(0)):
                    atten = atten_coefs_head[j]
                    atten = atten[atten.nonzero()]
                    discrepancy = (atten.max() - atten.min()).item()
                    if max_discrepancy < discrepancy:
                        max_discrepancy = discrepancy
                print('Attention max discrepancy for ' + title + ': {:.4f}'.format(max_discrepancy))
            curr_head_id += l_heads
        dir = './normalized/' if normalize_edge_color else './real/'
        merge(dir, titles)

    entropies = []
    for i in range(len(atten_coefs)):
        dist = torch.distributions.Categorical(atten_coefs[i])
        entropy = dist.entropy().detach().numpy()
        entropies.append(entropy)
    min_entropy = min([np.min(entro) for entro in entropies])
    max_entropy = max([np.max(entro) for entro in entropies])

    for head in range(len(entropies)):
        plot_hist(entropies[head], titles[head], min_entropy, max_entropy)
    merge('./hist/', titles)

    # Uniform distribution entropy for comparison
    degree_atten = torch.zeros(g.number_of_nodes(), g.number_of_nodes())
    for i in range(g.number_of_nodes()):
        in_degree = g.in_degree(i)
        degree_atten[i, 0:in_degree] = 1. / in_degree
    dist =  torch.distributions.Categorical(degree_atten)
    uni_entropy = dist.entropy().detach().numpy()
    plot_hist(uni_entropy, 'uniform', min_entropy, max_entropy)

def sample_subgraph(g, N):
    """
    Parameters
    ----------
    g: dgl.DGLGraph or nx.DiGraph
        The original giant directed graph from which we want to sample
        a subgraph.
    N: int
        Number of nodes to sample

    Returns
    -------
    nx_sub: nx.DiGraph
        A directed subgraph of g in networkx format
    nodes_sampled: list
        The list of nodes sampled
    nodes_to_lot: list
        The one hop neighborhood of the sampled nodes
    """
    nx_g = g if isinstance(g, nx.DiGraph) else g.to_networkx()
    nodes_sampled = []
    nodes_to_plot = []

    if N == 1:
        node_sampled = 0
        for v in range(1, g.number_of_nodes()):
            if g.in_degree(v) == 18:
                node_sampled = v

        nodes_sampled.append(node_sampled)
        nodes_to_plot.append(node_sampled)
        nodes_to_plot.extend(list(nx_g.neighbors(node_sampled)))
    else:
        for i in range(N):
            # We don't want too many disjoint connected components.
            # When possible, simply expand existing connected components.
            if len(nodes_to_plot) > i + 1:
                node_sampled = random.choice(nodes_to_plot)
            else:
                node_sampled = random.randint(0, g.number_of_nodes())

            nodes_sampled.append(node_sampled)
            nodes_to_plot.append(node_sampled)
            nodes_to_plot.extend(list(nx_g.neighbors(node_sampled)))
    # Remove repeated nodes
    nodes_sampled = list(set(nodes_sampled))
    nodes_to_plot = list(set(nodes_to_plot))

    # Get the subgraph based on the nodes selected
    nx_sub = nx_g.subgraph(nodes_to_plot)
    return nx_sub, nodes_sampled, nodes_to_plot

def plot(g, attention, title, nodes_to_plot=None, nodes_labels=None,
         edges_to_plot=None, nodes_pos=None, nodes_colors=None,
         edge_colormap=plt.cm.Reds, normalize_edge_color=False):
    """
    Visualize edge attentions by coloring edges on the graph.

    g: nx.DiGraph
        Directed networkx graph
    attention: torch.tensor
        attention[i, j] gives the attention for edge (j, i)
    title: str
        Title for the plot to be made
    nodes_to_plot: list
        List of node ids specifying which nodes to plot. Default to
        be None. If None, all nodes will be plot.
    nodes_labels: list, numpy.array or torch.tensor
        nodes_labels[i] specifies the label of the ith node, which will
        decide the node color on the plot. Default to be None. If None,
        all nodes will have the same canonical label. The nodes_labels
        should contain labels for all nodes to be plot.
    edges_to_plot: list of 2-tuples (i, j)
        List of edges represented as (source, destination). Default to
        be None. If None, all edges will be plot.
    nodes_pos: dictionary mapping int to numpy.array of size 2
        Default to be None. Specifies the layout of nodes on the plot.
    nodes_colors: list
        Specifies node color for each node class. Its length should be
        bigger than number of node classes in nodes_labels.
    edge_colormap: plt.cm
        Specifies the colormap to be used for coloring edges.
    normalize_edge_color: bool
        If True, the edge weights will be normalized to [0, 1] on the
        plot, otherwise the true edge weights will be used for coloring.
    """
    if nodes_to_plot is None:
        nodes_to_plot = g.nodes()
    if edges_to_plot is None:
        assert isinstance(g, nx.DiGraph), 'Expected g to be an networkx.DiGraph' \
                                          'object, got {}.'.format(type(g))
        edges_to_plot = g.in_edges(nodes_to_plot)
    edges_atten = []
    for edge in edges_to_plot:
        edges_atten.append(attention[edge[1], edge[0]].item())
    if nodes_pos is None:
        nodes_pos = nx.spring_layout(g)

    fig, ax = plt.subplots()
    if normalize_edge_color:
        nx.draw_networkx_edges(g, nodes_pos, edgelist=edges_to_plot,
                               edge_color=edges_atten, edge_cmap=edge_colormap,
                               width=2, alpha=0.5, ax=ax)
    else:
        nx.draw_networkx_edges(g, nodes_pos, edgelist=edges_to_plot,
                               edge_color=edges_atten, edge_cmap=edge_colormap,
                               width=2, alpha=0.5, ax=ax, edge_vmin=0,
                               edge_vmax=1)

    if nodes_colors is None:
        nodes_colors = ['lightskyblue', 'g', 'salmon', 'c', 'm', 'yellow',
                        'mediumaquamarine', '#FEBD69']
    if nodes_labels is None:
        nodes_labels = [0] * len(nodes_to_plot)
    nx.draw_networkx_nodes(g, nodes_pos, nodelist=nodes_to_plot, ax=ax, node_size=30,
                           node_color=[nodes_colors[nodes_labels[v]] for v in nodes_to_plot],
                           with_labels=False, alpha=0.9)
    ax.set_axis_off()
    ax.set_title(title)
    if normalize_edge_color:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=min(edges_atten),
                                                                        vmax=max(edges_atten)))
    else:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm)
    fig.tight_layout()
    if normalize_edge_color:
        fig.savefig('./normalized/' + title + '.png', dpi=fig.dpi)
    else:
        fig.savefig('./real/' + title + '.png', dpi=fig.dpi)
    plt.close()

def merge(dir, title_list):
    assert len(title_list) <= 12, 'The list is too long to make the plot easy to see.'
    n_rows = math.ceil(len(title_list) / 3)
    for row in range(n_rows):
        im_list = [dir + title_list[i] + '.png' for i in range(
            3 * row, min(3 * (row + 1), len(title_list)))]
        imgs = [Image.open(i) for i in im_list]
        # pick the image which is the smallest, and resize the others to match it
        # (can be arbitrary image shape here)
        min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
        imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))

        # Save the image
        imgs_comb = Image.fromarray(imgs_comb)
        imgs_comb.save(dir + 'row_{:d}.png'.format(row))
    # Stack all rows into a single plot
    list_im = [dir + 'row_{:d}.png'.format(row) for row in range(n_rows)]
    imgs = [Image.open(i) for i in list_im]
    """
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    min_shape = (int(1.1 * min_shape[0]), int(1.1 * min_shape[1]))
    imgs_comb = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs))
    """
    imgs_comb = np.vstack((np.asarray(i) for i in imgs))
    imgs_comb = Image.fromarray(imgs_comb)
    imgs_comb.save(dir + 'merged.png')

def plot_hist(data, title, min_entropy, max_entropy, n_bins=20):
    fig, ax = plt.subplots()
    plt.hist(data, bins=n_bins, range=(min_entropy, max_entropy))
    plt.title(title)
    plt.xlabel('entropy', fontsize=22)
    plt.ylabel('# of nodes', fontsize=22)
    fig.tight_layout()
    fig.savefig('./hist/' + title + '.png', dpi=fig.dpi)
    plt.close()
