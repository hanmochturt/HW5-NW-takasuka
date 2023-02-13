# Importing Dependencies
import numpy as np
from typing import Tuple


# Defining class for Needleman-Wunsch Algorithm for Global pairwise alignment
class NeedlemanWunsch:
    """ Class for NeedlemanWunsch Alignment

    Parameters:
        sub_matrix_file: str
            Path/filename of substitution matrix
        gap_open: float
            Gap opening penalty
        gap_extend: float
            Gap extension penalty

    Attributes:
        seqA_align: str
            seqA alignment
        seqB_align: str
            seqB alignment
        alignment_score: float
            Score of alignment from algorithm
        gap_open: float
            Gap opening penalty
        gap_extend: float
            Gap extension penalty
    """

    def __init__(self, sub_matrix_file: str, gap_open: float, gap_extend: float):
        # Init alignment and gap matrices
        self._align_matrix = None
        self._gapA_matrix = None
        self._gapB_matrix = None

        # Init matrices for backtrace procedure
        self._back = None
        self._back_A = None
        self._back_B = None

        # Init alignment_score
        self.alignment_score = 0

        # Init empty alignment attributes
        self.seqA_align = ""
        self.seqB_align = ""

        # Init empty sequences
        self._seqA = ""
        self._seqB = ""

        # Setting gap open and gap extension penalties
        self.gap_open = gap_open
        assert gap_open < 0, "Gap opening penalty must be negative."
        self.gap_extend = gap_extend
        assert gap_extend < 0, "Gap extension penalty must be negative."

        # Generating substitution matrix
        self.sub_dict = self._read_sub_matrix(sub_matrix_file)  # substitution dictionary

    def _read_sub_matrix(self, sub_matrix_file):
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This function reads in a scoring matrix from any matrix like file.
        Where there is a line of the residues followed by substitution matrix.
        This file also saves the alphabet list attribute.

        Parameters:
            sub_matrix_file: str
                Name (and associated path if not in current working directory)
                of the matrix file that contains the scoring matrix.

        Returns:
            dict_sub: dict
                Substitution matrix dictionary with tuple of the two residues as
                the key and score as value e.g. {('A', 'A'): 4} or {('A', 'D'): -8}
        """
        with open(sub_matrix_file, 'r') as f:
            dict_sub = {}  # Dictionary for storing scores from sub matrix
            residue_list = []  # For storing residue list
            start = False  # trigger for reading in score values
            res_2 = 0  # used for generating substitution matrix
            # reading file line by line
            for line_num, line in enumerate(f):
                # Reading in residue list
                if '#' not in line.strip() and start is False:
                    residue_list = [k for k in line.strip().upper().split(' ') if k != '']
                    start = True
                # Generating substitution scoring dictionary
                elif start is True and res_2 < len(residue_list):
                    line = [k for k in line.strip().split(' ') if k != '']
                    # reading in line by line to create substitution dictionary
                    assert len(residue_list) == len(
                        line), "Score line should be same length as residue list"
                    for res_1 in range(len(line)):
                        dict_sub[(residue_list[res_1], residue_list[res_2])] = float(line[res_1])
                    res_2 += 1
                elif start is True and res_2 == len(residue_list):
                    break
        return dict_sub

    def align(self, seqA: str, seqB: str) -> Tuple[float, str, str]:
        """
        This function performs global sequence alignment of two strings
        using the Needleman-Wunsch Algorithm
        
        Parameters:
        	seqA: str
         		the first string to be aligned
         	seqB: str
         		the second string to be aligned with seqA
         
        Returns:
         	(alignment score, seqA alignment, seqB alignment) : Tuple[float, str, str]
         		the score and corresponding strings for the alignment of seqA and seqB
        """
        # Resetting alignment in case method is called more than once
        self.seqA_align = ""
        self.seqB_align = ""

        # Resetting alignment score in case method is called more than once
        self.alignment_score = 0

        # Initializing sequences for use in backtrace method
        self._seqA = seqA
        self._seqB = seqB

        # Initialize matrix private attributes for use in alignment
        # create matrices for alignment scores, gaps, and backtracing
        p_score = np.zeros((len(seqB) + 1, len(seqA) + 1))
        for seq_a_index, seq_a_value in enumerate(seqA):
            for seq_b_index, seq_b_value in enumerate(seqB):
                p_score[seq_b_index + 1, seq_a_index + 1] = self.sub_dict[
                    (seq_a_value, seq_b_value)]

        n_rows, n_columns = p_score.shape
        alignment_matrix = np.zeros(p_score.shape)
        gap_score_columns = np.zeros(p_score.shape)
        gap_score_columns[0, 0] = self.gap_open
        gap_score_rows = np.zeros(p_score.shape)
        gap_score_rows[0, 0] = self.gap_open
        for index in range(1, n_columns):
            gap_score_columns[0, index] = float('-inf')
            gap_score_rows[0, index] = gap_score_rows[0, index - 1] + self.gap_extend
            alignment_matrix[0, index] = float('-inf')

        for index in range(1, n_rows):
            gap_score_rows[index, 0] = float('-inf')
            gap_score_columns[index, 0] = gap_score_columns[index - 1, 0] + self.gap_extend
            alignment_matrix[index, 0] = float('-inf')

        # Implement global alignment
        for row_index in range(1, n_rows):
            for column_index in range(1, n_columns):
                alignment = alignment_matrix[row_index - 1, column_index - 1] + p_score[row_index,
                                                                                        column_index]
                gap_column = gap_score_columns[row_index - 1, column_index - 1] + p_score[row_index,
                                                                                          column_index]
                gap_row = gap_score_rows[row_index - 1, column_index - 1] + p_score[row_index,
                                                                                    column_index]
                alignment_matrix[row_index, column_index] = max(alignment, gap_column, gap_row)

                gap_column_1 = alignment_matrix[row_index - 1, column_index] + self.gap_extend + \
                               self.gap_open
                gap_column_2 = gap_score_columns[row_index - 1, column_index] + self.gap_extend
                gap_score_columns[row_index, column_index] = max(gap_column_1, gap_column_2)

                gap_row_1 = alignment_matrix[row_index, column_index - 1] + self.gap_extend + \
                            self.gap_open
                gap_row_2 = gap_score_rows[row_index, column_index - 1] + self.gap_extend
                gap_score_rows[row_index, column_index] = max(gap_row_1, gap_row_2)

        self._align_matrix = alignment_matrix
        self._gapA_matrix = gap_score_columns
        self._gapB_matrix = gap_score_rows

        return self._backtrace()

    def _backtrace(self) -> Tuple[float, str, str]:
        """
        Trace back through the back matrix created with the
        align function in order to return the final alignment score and strings.
        
        Parameters:
        	None
        
        Returns:
         	(alignment score, seqA alignment, seqB alignment) : Tuple[float, str, str]
         		the score and corresponding strings for the alignment of seqA and seqB
        """
        self.alignment_score = max(self._align_matrix[-1, -1], self._gapA_matrix[-1, -1],
                                   self._gapB_matrix[-1, -1])
        score = self.alignment_score
        current_row = -1
        current_column = -1
        num_rows, num_cols = self._align_matrix.shape

        while current_row != -1 * num_rows or current_column != -1 * num_cols:
            # if it is favorable to match, add the next character to each sequence
            if score == self._align_matrix[current_row, current_column]:
                self.seqA_align = f"{self._seqA[current_column]}{self.seqA_align}"
                self.seqB_align = f"{self._seqB[current_row]}{self.seqB_align}"
                current_row -= 1
                current_column -= 1
            # if favorable to insert a gap in sequence A over matching, then insert a gap in A
            # and add the next character to seqB
            elif score == self._gapA_matrix[current_row, current_column]:
                self.seqA_align = f"-{self.seqA_align}"
                self.seqB_align = f"{self._seqB[current_row]}{self.seqB_align}"
                current_row -= 1
            # if favorable to insert a gap in sequence B over matching or inserting a gap in A,
            # then insert a gap in B and add the next character to seqA
            else:
                self.seqA_align = f"{self._seqA[current_column]}{self.seqA_align}"
                self.seqB_align = f"-{self.seqB_align}"
                current_column -= 1
            score = max(self._align_matrix[current_row, current_column],
                        self._gapA_matrix[current_row, current_column],
                        self._gapB_matrix[current_row, current_column])

        return (self.alignment_score, self.seqA_align, self.seqB_align)


def read_fasta(fasta_file: str) -> Tuple[str, str]:
    """
    DO NOT MODIFY THIS FUNCTION! IT IS ALREADY COMPLETE!

    This function reads in a FASTA file and returns the associated
    string of characters (residues or nucleotides) and the header.
    This function assumes a single protein or nucleotide sequence
    per fasta file and will only read in the first sequence in the
    file if multiple are provided.

    Parameters:
        fasta_file: str
            name (and associated path if not in current working directory)
            of the Fasta file.

    Returns:
        seq: str
            String of characters from FASTA file
        header: str
            Fasta header
    """
    assert fasta_file.endswith(".fa"), "Fasta file must be a fasta file with the suffix .fa"
    with open(fasta_file) as f:
        seq = ""  # initializing sequence
        first_header = True
        for line in f:
            is_header = line.strip().startswith(">")
            # Reading in the first header
            if is_header and first_header:
                header = line.strip()  # reading in fasta header
                first_header = False
            # Reading in the sequence line by line
            elif not is_header:
                seq += line.strip().upper()  # generating full sequence
            # Breaking if more than one header is provided in the fasta file
            elif is_header and not first_header:
                break
    return seq, header
