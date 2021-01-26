import matplotlib
matplotlib.rc("xtick", labelsize = 24)
matplotlib.rc("ytick", labelsize = 24)
matplotlib.rc("axes", titlesize = 48)
matplotlib.rc("axes", labelsize = 48)
matplotlib.rc("lines", markersize = 16)

from matplotlib.path import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from eldr.explain.explain_cs import TGT
import numpy as np
import torch

from eldr.misc import truncate

def plot_polys(data_rep, vertices):

    num_clusters = len(vertices)

    fig, ax = plt.subplots(figsize=(20, 10))
    patches = []

    for i in range(num_clusters):
        line = plt.Polygon(vertices[i], closed = False, color="blue", alpha=0.3)
        ax.add_line(line)

    plt.scatter(data_rep[:, 0], data_rep[:, 1])

    #plt.show()
    plt.close()

def plot_groups(x, data_rep, num_clusters, labels, contour = None, name = "plot_groups.png"):

    n = x.shape[0]
    cluster = -1.0 * np.ones((n))
    
    indices = [[]] * num_clusters
    centers = [[]] * num_clusters
    means = [[]] * num_clusters
    for i in range(num_clusters):
        indices[i] = []
        for j in range(n):
            if labels[j] == i:
                cluster[j] = i
                indices[i].append(j)
        means[i] = np.mean(x[indices[i], :], axis = 0)
        centers[i] = np.mean(data_rep[indices[i], :], axis = 0)
        
    centers = np.array(centers)
    means = np.array(means)

    fig, ax = plt.subplots(figsize=(20, 10))
    
    patches = []
    
    plt.scatter(data_rep[:, 0], data_rep[:, 1], c = cluster, cmap = plt.cm.coolwarm)

    for i in range(num_clusters):
        plt.text(centers[i, 0], centers[i, 1], str(i), fontsize = 72)
        
    if contour is not None:
        feature_0 = contour[0]
        feature_1 = contour[1]
        map = contour[2]
        plt.contour(feature_0, feature_1, map)
        plt.colorbar()

    plt.savefig(name)
    #plt.show()
    plt.close()

    return means, centers, indices
    
def plot_metrics(a, b, name = "plot_metrics.png", fontsize = 55, labelsize = 40):

    # Set up figure and image grid
    fig = plt.figure(figsize=(20, 10))
        
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(1,2),
                     axes_pad=0.75,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="7%",
                     cbar_pad=0.25,
                     )

    # Add data to image grid
    c = 0
    for ax in grid:
        ax.tick_params(axis = "both", which = "major", labelsize = labelsize)
        
        if c == 0:
            im = ax.imshow(a, cmap = "RdYlGn", interpolation = "none", vmin = 0.0, vmax = 1.0)
            ax.set_title("Correctness - " + str(np.round(np.mean(a), 3)), fontsize = fontsize)
            ax.set_ylabel("Initial Group", fontsize = fontsize)
        elif c == 1:
            im = ax.imshow(b, cmap = "RdYlGn", interpolation = "none", vmin = 0.0, vmax = 1.0)
            ax.set_title("Coverage - "  + str(np.round(np.mean(b), 3)), fontsize = fontsize)
        ax.set_xlabel("Target Group", fontsize = fontsize)
        c += 1

    # Colorbar
    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)
    ax.cax.tick_params(labelsize = labelsize)

    plt.savefig(name)
    #plt.show()
    plt.close()

def plot_explanation(model, x, data_rep, indices, deltas, a, b, c1, c2,  k = None, num_points = 50, name = "plot_explanation.png", feature_names = None,
                    logit_gammas = None):
    
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if not torch.is_tensor(deltas):
        deltas = torch.tensor(deltas)
    if not torch.is_tensor(data_rep):
        data_rep = torch.tensor(data_rep)
    if (logit_gammas is not None) and not torch.is_tensor(logit_gammas):
        logit_gammas = torch.tensor(logit_gammas)

    # Find the explanation from c1 to c2
    
    # if logit_gammas is not None:
    #     g_intermediate = torch.exp(-logit_gammas[c1 - 1])
    #     logit_g_scaling = logit_gammas[c2] - logit_gammas[c1]
    # else:
    #     g_intermediate = torch.ones_like(deltas[0])
    #     logit_g_scaling = torch.zeros_like(deltas[0])

    # if c1 == 0:
    #     d = deltas[c2 - 1]
    # elif c2 == 0:
    #     d = -1.0 * g_intermediate * deltas[c1 - 1]
    # else:
    #     d = -1.0 * g_intermediate * deltas[c1 - 1] + deltas[c2 - 1]
    
    # if k is not None:
    #     d = truncate(d, k)
    #     if logit_gammas is not None:
    #         logit_g_scaling = truncate(logit_g_scaling, k)
    
    # g_scaling = torch.exp(logit_g_scaling)
    # g_scaling = torch.reshape(g_scaling, (1, g_scaling.shape[0]))
    # d = torch.reshape(d, (1, d.shape[0]))

    n_dim = deltas.shape[1]
    n_clusters = deltas.shape[0] + 1
    use_scaling = (logit_gammas is not None)
    tgt = TGT(n_dim,n_clusters,init_deltas=deltas, use_scaling=use_scaling, init_gammas_logit=logit_gammas)
   
    # Visualize the data
    fig, ax = plt.subplots(figsize=(20, 30))
    fig.subplots_adjust(hspace = .3)

    for i in range(2):
        if i == 0:
            initial = c1
            target = c2
            sign = 1.0
        elif i == 1:
            initial = c2
            target = c1
            sign = -1.0

        # Plot the full representation
        ax = plt.subplot(3, 1, i + 1)
        
        plt.scatter(data_rep[:, 0], data_rep[:, 1])
    
        # Sample num_points in initial group
        indices_initial = np.random.choice(indices[initial], num_points, replace = False)
        points_initial = x[indices_initial, :]
    
        # Load the model
        #sess, rep, X, D = load_model()
        # d_zeros = np.zeros(d.shape)
    
        # Plot the chosen points before perturbing them
        y_initial = model.Encode(points_initial)#sess.run(rep, feed_dict={X: points_initial, D: d_zeros})
        plt.scatter(y_initial[:,0], y_initial[:,1], marker = "v", c = "magenta")
    
        # Plot the chosen points after perturbing them
        if use_scaling:
            points_after, d, logit_g = tgt(points_initial, initial, target, k)
        else:
            points_after, d = tgt(points_initial, initial, target, k)
        y_after = model.Encode(points_after) #sess.run(rep, feed_dict={X: points_initial, D: sign * d})
        plt.scatter(y_after[:,0], y_after[:,1], marker = "v", c = "red")
    
        plt.title("Mapping from Group " + str(initial) + " to Group " + str(target) + "\n Correctness - " + str(np.round(a[initial, target], 3)) + ", Coverage - " + str(np.round(b[initial, target], 3)))
    
    ax = plt.subplot(3, 1, 3)

    feature_index = np.array(range(n_dim))
    d = torch.reshape(d, (1, d.shape[0])).detach()
    plt.scatter(feature_index, d, label='delta')
    if use_scaling:
        logit_g = torch.reshape(logit_g, (1, logit_g.shape[0])).detach()
        plt.scatter(feature_index, logit_g, label='gamma')
    plt.legend()
    plt.title("Explanation for Group " + str(c1) + " to Group " + str(c2))
    plt.ylabel("Change applied")
    if feature_names is None:
        plt.xlabel("Feature Index")
    else:
        plt.xlabel("Feature")
        plt.xticks(range(n_dim), feature_names, rotation=90, fontsize = 40)

    plt.savefig(name)
    plt.show()
    plt.close()


def plot_change(deltas, deltas_original, name = "plot_similarity.png", feature_names = None):

    num_clusters = deltas_original.shape[0] + 1

    print(np.round(deltas_original, 2))
    print(np.round(deltas[:num_clusters - 1, ], 2))
    
    diff = np.abs(deltas_original - deltas[:num_clusters - 1, ])
    
    plt.figure(figsize=(20, 10))
    
    plt.ylabel("Basis Explanation")
    plt.yticks(np.arange(0, num_clusters + 1, dtype=np.int), labels = 1 + np.arange(0, num_clusters + 1, dtype=np.int))
    if feature_names is None:
        plt.xlabel("Feature Index")
        plt.xticks(np.arange(0, deltas.shape[1] + 1, dtype=np.int))
    else:
        plt.xlabel("Feature")
        plt.xticks(np.arange(0, deltas.shape[1] + 1, dtype=np.int), feature_names, rotation=90, fontsize = 40)
    
    plt.title("Change in Explanation (Normalized)")
    
    plt.imshow(diff, vmin = 0.0, vmax = np.max(np.abs(deltas_original)))

    plt.colorbar()
    
    plt.savefig(name)
    #plt.show()
    plt.close()