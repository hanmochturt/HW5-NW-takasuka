# Importing Dependencies
import pytest
import sys
import pathlib

PARENT_PARENT_FOLDER = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PARENT_PARENT_FOLDER))
from align import NeedlemanWunsch, read_fasta
import numpy as np


def test_nw_alignment():
    """
    Unit test for NW alignment,
    using test_seq1.fa and test_seq2.fa,
    asserting that you have correctly filled out
    your 3 alignment matrices.
    BLOSUM62 matrix and a gap open penalty
    of -10 and a gap extension penalty of -1
    """
    seq1, _ = read_fasta("./data/test_seq1.fa")
    seq2, _ = read_fasta("./data/test_seq2.fa")
    nw = NeedlemanWunsch("./substitution_matrices/BLOSUM62.mat", -10, -1)
    assert nw.align(seq1, seq2) == (4.0, 'MYQR', 'M-QR')

    assert np.all(nw._align_matrix == [[0., float("-inf"), float("-inf"), float("-inf"),
                                        float("-inf")],
                                       [float("-inf"), 5., -12., -12., -14.],
                                       [float("-inf"), -11., 4., -1., -6.],
                                       [float("-inf"), -13., -8., 5., 4.]])

    assert np.all(nw._gapA_matrix == [[-10., float("-inf"), float("-inf"), float("-inf"),
                                       float("-inf")],
                                      [-11., float("-inf"), float("-inf"), float("-inf"),
                                       float("-inf")],
                                      [-12., -6., -23., -23., -25.],
                                      [-13., -7., -7., -12., -17.]])

    assert np.all(nw._gapB_matrix == [[-10., -11., -12., -13., -14.],
                                      [float("-inf"), float("-inf"), -6., -7., -8.],
                                      [float("-inf"), float("-inf"), -22., -7., -8.],
                                      [float("-inf"), float("-inf"), -24., -19., -6.]])


def test_nw_backtrace():
    """
    Unit test for NW backtracing
    using test_seq3.fa and test_seq4.fa,
    asserting that the backtrace is correct.
    Use the BLOSUM62 matrix. Gap open
    penalty of -10 and a gap extension penalty of -1.
    """
    seq3, _ = read_fasta("./data/test_seq3.fa")
    seq4, _ = read_fasta("./data/test_seq4.fa")
    nw = NeedlemanWunsch("./substitution_matrices/BLOSUM62.mat", -10, -1)
    assert nw.align(seq3, seq4) == (17.0, 'MAVHQLIRRP', 'M---QLIRHP')
