import numpy as np
import pandas as pd


def mutate(base):
    "Mutate a nucleobase to any different nucleobase"
    diff = set("ACTG") - set(base)
    return np.random.choice(list(diff))


def filter_missing(arr, maxfreq):
    """
    Function to remove sites with a high frequency of missing reads from an array of DNA seqs

    Args:
        arr: The array of DNA sequences, with missing reads designated as N
        maxfreq: The critical frequency for missing reads. If the frequency of N at a site
                 exceeds maxfreq, that site is excluded from the result

    Returns:
        A view of arr including only columns in which the frequency of N <= maxfreq
    """
    # determine the frequency of missing reads at a site
    freqmissing = np.sum(arr == "N", axis=0) / arr.shape[0]

    # return arr, sliced to only include cols with freqmissing <= maxfreq
    return arr[:, freqmissing <= maxfreq]


def filter_maf(arr, minfreq):
    """
    Filters an array of seqs to only include sites at which a minimum
    proportion of alleles differ from the majority

    Args:
        arr: the array of sequence data
        minfreq: the minimum proportion of alleles which must differ from
                 the most common allele at a site

    Returns:
        A view of the numpy array arr, filtered to include only columns in
        which the minor allele frequency exceeds minfreq
    """
    # determine proportion of alleles in a col that do not match the first row
    freqs = np.sum(arr != arr[0], axis=0) / arr.shape[0]

    # create a copy of freqs with all values > 0.5 changed to their complement
    maf = freqs.copy()
    maf[maf > 0.5] = 1 - maf[maf > 0.5]

    return arr[:, maf > minfreq]


class Seqlib:
    def __init__(self, ninds, nsites):
        self.ninds = ninds
        self.nsites = nsites
        self.seqs = self.simulate()

    def simulate(self):
        """
        Simulates ninds DNA sequences of length nsites
        Adds mutations and then missing reads (probability 0.1 for each)

        Returns:
            A numpy array of dimensions (ninds, nsites) containing the sequences
        """
        # generate a random sequence of length nsites
        oseq = np.random.choice(list("ACGT"), size=self.nsites)

        # initialize an array with oseq repeated ninds times
        arr = np.array([oseq for i in range(self.ninds)])

        # create a random mask for array - prob of mutation for any site is 0.1
        muts = np.random.binomial(1, 0.1, (self.ninds, self.nsites))

        # mutate sequences
        for col in range(self.nsites):
            # pick a mutation
            newbase = mutate(arr[0, col])

            # create mask for col from muts
            mask = muts[:, col].astype(bool)

            # mutate randomly selected rows
            arr[:, col][mask] = newbase

        # create a random mask for missing bases and modify arr
        missing = np.random.binomial(1, 0.1, (self.ninds, self.nsites))
        arr[missing.astype(bool)] = "N"

        return arr

    def filter(self, minmaf, maxmissing):
        "Wrapper to combine filter_missing and filter_maf into one function"
        return(filter_missing(arr=filter_maf(arr=self.seqs,
                                             minfreq=minmaf),
                              maxfreq=maxmissing))

    def calculate_statistics(self):
        "Return several statistics for a numpy array of DNA sequences"
        # mean nucleotide diversity
        nd = np.var(self.seqs == self.seqs[0], axis=0).mean()
        # mean frequency of minor allele at each site
        mf = np.mean(np.sum(self.seqs != self.seqs[0], axis=0) / self.seqs.shape[0])
        # number of sites with no mutations or missing reads
        inv = np.any(self.seqs != self.seqs[0], axis=0).sum()
        # number of sites with mutations and missing reads
        var = self.seqs.shape[1] - inv

        return pd.Series(
            {"mean nucleotide diversity": nd,
             "mean minor allele frequency": mf,
             "invariant sites": inv,
             "variable sites": var,
             })
