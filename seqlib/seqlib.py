import copy
import numpy as np
import pandas as pd


class Seqlib:
    def __init__(self, ninds, nsites):
        self.ninds = ninds
        self.nsites = nsites

        self.seqs = self._simulate()

        self.maf = self._get_maf()

    # private functions
    def _mutate(self, base):
        """Mutate a nucleobase to any different nucleobase"""
        diff = set("ACTG") - set(base)
        return np.random.choice(list(diff))

    def _simulate(self):
        """
        Simulates ninds DNA sequences of length nsites
        Adds mutations and then missing reads (probability 0.1 for each)

        Returns:
            A numpy array of dimensions (ninds, nsites) containing the seqs
        """
        # generate a random sequence of length nsites
        oseq = np.random.choice(list("ACGT"), size=self.nsites)

        # initialize an array with oseq repeated ninds times
        arr = np.array([oseq for i in range(self.ninds)])

        # create a random mask for array - prob of mutation for any site is 0.1
        muts = np.random.binomial(1, 0.1, (self.ninds, self.nsites))

        # _mutate sequences
        for col in range(self.nsites):
            # pick a mutation
            newbase = self._mutate(arr[0, col])

            # create mask for col from muts
            mask = muts[:, col].astype(bool)

            # _mutate randomly selected rows
            arr[:, col][mask] = newbase

        # create a random mask for missing bases and modify arr
        missing = np.random.binomial(1, 0.1, (self.ninds, self.nsites))
        arr[missing.astype(bool)] = "N"

        return arr

    def _get_maf(self):
        """Get an array of minor allele frequencies"""
        # determine proportion of alleles in a col that do not match row 1
        freqs = np.sum(self.seqs != self.seqs[0], axis=0) / self.seqs.shape[0]

        # create a copy of freqs with all values > 0.5 changed to complements
        maf = freqs.copy()
        maf[maf > 0.5] = 1 - maf[maf > 0.5]

        return maf

    def _filter_missing(self, maxfreq):
        """
        Remove sites with many missing reads from seqs

        Args:
            maxfreq: The critical frequency for missing reads.
                     If the frequency of N at a site exceeds maxfreq, that site
                     is excluded from the result

        Returns:
            A boolean filter for seqs indicating cols with freq of N <= maxfreq
        """
        # determine the frequency of missing reads at a site
        freqmissing = np.sum(self.seqs == "N", axis=0) / self.seqs.shape[0]

        # return arr, sliced to only include cols with freqmissing <= maxfreq
        return freqmissing <= maxfreq

    def _filter_maf(self, minfreq):
        """
        Filters seqs to only include sites at which a minimum proportion of
        alleles differ from the most common allele

        Args:
            minfreq: the minimum proportion of alleles which must differ from
                     the most common allele at a site

        Returns:
            A boolean filter for seqs indicating cols with maf > minfreq
        """

        return self.maf > minfreq

    # public functions
    def filter(self, minmaf=0, maxmissing=1):
        """
        Public wrapper to combine filter_missing and filter_maf

        Args:
            minmaf: the minimum minor allele frequency
            maxmissing: the maximum frequency of missing reads

        Returns:
            A filtered view of self.seqs, as defined by minmaf and maxmissing
            Default values return self.seqs in full
        """

        maf_filter = self._filter_maf(minmaf)
        miss_filter = self._filter_missing(maxmissing)

        all_filter = maf_filter + miss_filter

        return self.seqs[:, all_filter]

    def filter_seqlib(self, minmaf=0, maxmissing=1):
        """
        Version of filter that returns a new Seqlib object

        Args:
            minmaf: the minimum minor allele frequency
            maxmissing: the maximum frequency of missing reads

        Returns:
            A new Seqlib object with a filtered version of seqs as new.seqs
            Default values return a copy of self in full
        """
        # apply filters to get new seqs
        newseqs = self.filter(minmaf, maxmissing)

        # make a new copy of the Seqlib object
        newself = copy.deepcopy(self)
        newself.__init__(newseqs.shape[0], newseqs.shape[1])

        # overwrite the seqs array
        newself.seqs = newseqs

        # call _get_maf to match maf to array
        newself._get_maf()

        return newself

    def calculate_stats(self):
        """Return several statistics for a numpy array of DNA sequences"""
        # mean nucleotide diversity
        nd = np.var(self.seqs == self.seqs[0], axis=0).mean()
        # mean frequency of minor allele at each site
        mf = np.mean(np.sum(self.seqs != self.seqs[0], axis=0)
                     / self.seqs.shape[0])
        # number of sites with no mutations or missing reads
        inv = np.all(self.seqs == self.seqs[0], axis=0).sum()
        # number of sites with mutations and missing reads
        var = self.seqs.shape[1] - inv

        return pd.Series(
            {"mean nucleotide diversity": nd,
             "mean minor allele frequency": mf,
             "invariant sites": inv,
             "variable sites": var,
             })
