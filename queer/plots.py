import matplotlib.pyplot as plt


def plot_susceptibility(suscep,sym,labels,show=False,save=False,path=False):
    plt.figure(figsize=(3,4))
    plt.plot(suscep[0],label='Re(χ)')
    plt.plot(suscep[1],label='Im(χ)')
    plt.xticks(ticks=sym, labels=labels, fontsize=15)
    plt.xlim(sym[0], sym[-1])
    for i in sym[1:-1]:
        plt.axvline(i, c="black")
    plt.ylabel("1/eV")
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
    if save:
        plt.savefig('susceptibility.pdf')
    elif path:
        plt.savefig(f'{path}/susceptibility.pdf')
        

def plot_electron_mesh(band, mesh, metallic_band_index, fig_size=[5,5], save=None, temp=None, cmap='jet',s=1):
    x, y,z = mesh

    plt.figure(figsize=(fig_size[0],fig_size[1]))
    plt.scatter(x, y, c=band[metallic_band_index], cmap=cmap,s=s)
    plt.colorbar()
    # plt.xlim(xlim[0], xlim[1])
    # plt.ylim(ylim[0], ylim[1])
    plt.axis("equal")
    # plt.xticks([])
    # plt.yticks([])
    plt.xlabel(r"$k_x$ [$\AA^{-1}$]")
    plt.ylabel(r"$k_y$ [$\AA^{-1}$]")
    plt.axis("equal")
    if temp == None:
        plt.title("")
    else:
        plt.title(f"σ = {temp}")
    if save != None:
        plt.savefig(save)
