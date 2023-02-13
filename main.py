# Import NeedlemanWunsch class and read_fasta function
from align import read_fasta, NeedlemanWunsch


def main():
    """
    This function should
    (1) Align all species to humans and print species in order of most similar to human BRD
    (2) Print all alignment scores between each species BRD2 and human BRD2
    """
    hs_seq, hs_header = read_fasta("./data/Homo_sapiens_BRD2.fa")
    gg_seq, gg_header = read_fasta("./data/Gallus_gallus_BRD2.fa")
    mm_seq, mm_header = read_fasta("./data/Mus_musculus_BRD2.fa")
    br_seq, br_header = read_fasta("./data/Balaeniceps_rex_BRD2.fa")
    tt_seq, tt_header = read_fasta("./data/tursiops_truncatus_BRD2.fa")

    nw = NeedlemanWunsch("./substitution_matrices/BLOSUM62.mat", -10, -1)

    # Align all species to humans and print species in order of most similar to human BRD
    # using gap opening penalty of -10 and a gap extension penalty of -1 and BLOSUM62 matrix
    # Print all of the alignment score between each species BRD2 and human BRD2
    # using gap opening penalty of -10 and a gap extension penalty of -1 and BLOSUM62 matrix
    scores = {}
    for seq, header in ((gg_seq, gg_header), (mm_seq, mm_header),
                        (br_seq, br_header), (tt_seq, tt_header)):
        score, _, _ = nw.align(hs_seq, seq)
        scores[score] = extract_species_from_header(header)
    sorted_scores = sorted(list(scores.keys()), reverse=True)
    print("Most similar species to humans rankings:")
    for ranking, score in enumerate(sorted_scores):
        print(f"{ranking + 1}. {scores[score]} \t\t(score: {score})")


def extract_species_from_header(header: str) -> str:
    """
    Find species name from header

    :param header: header string from FASTA file

    :returns: species name
    """
    start_marker = "OS="
    end_marker = " OX"
    species_name_start = header.find(start_marker) + len(start_marker)
    species_name_end = header.find(end_marker)
    return header[species_name_start:species_name_end]


if __name__ == "__main__":
    main()
