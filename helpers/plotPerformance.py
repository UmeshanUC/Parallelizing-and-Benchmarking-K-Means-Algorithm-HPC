# Plot the performance curves of sequential and parallel execution times vs sizes of the sample sizes
import matplotlib.pyplot as plt


def plotPerformance(seq, paralell, sizes):

    if (paralell):
        # Plotting the parallel line
        plt.plot(sizes, paralell, label='Parallel')

    if (seq):
        # Plotting the sequential line
        plt.plot(sizes, seq, label='Sequential')

    # Adding legends
    plt.legend()
    plt.grid()

    # Adding labels and title
    plt.xlabel('Size of dataset')
    plt.ylabel('Execution time (ms)')
    plt.title("Excecution time performance")

    # Display the plot
    plt.show()
