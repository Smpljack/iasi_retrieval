import matplotlib.pyplot as plt


def plot_retrieval_profiles(a_priori, retrieved, true, z, quantity_str):
    plt.figure()
    plt.plot(true, z, label="Truth", c="blue")
    plt.plot(retrieved, z, label="Retrieved", c="red")
    plt.plot(a_priori, z, label="A Priori", c="grey", ls="--")
    plt.xlabel(f"{quantity_str}")
    plt.ylabel("z [km]")
    plt.legend()
    return plt.gcf()