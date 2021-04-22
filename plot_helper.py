import matplotlib.pyplot as plt


class PlotHelp(object):
    def __init__(self):
        pass

    def plot_examples(self, examples, potential_labels=None):

        """
        examples : List of Mel Spectrogrames of audio data
        potential_labels : If not None, we take the supplied examples as
        Positive cases where 'Activate' is Present
        """

        if potential_labels:
            assert len(examples) == len(potential_labels)

        nrows = len(examples)
        ncols = 1

        fig, ax = plt.subplots(nrows, ncols, figsize=(24 * nrows, 12))

        if nrows == 1:
            ax.imshow(examples[0])
            if potential_labels:
                start, end = potential_labels[0]
                ax.axvline(start, color="r")
                ax.axvline(end, color="r")
            plt.show()

        else:
            for row in range(nrows):
                ax[row].imshow(examples[row])
                if potential_labels:
                    start, end = potential_labels[row]
                    ax[row].axvline(start, color="r")
                    ax[row].axvline(end, color="r")

            plt.title("Examples")
            plt.show()
