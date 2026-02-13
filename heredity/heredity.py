import csv
import itertools
import sys

# Correct probabilities according to project specifications
PROBS = {
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },
    "trait": {
        2: {
            True: 0.65,
            False: 0.35
        },
        1: {
            True: 0.56,
            False: 0.44
        },
        0: {
            True: 0.01,
            False: 0.99
        }
    },
    "mutation": 0.01
}


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")

    people = load_data(sys.argv[1])
    probabilities = {
        person: {
            "gene": {2: 0, 1: 0, 0: 0},
            "trait": {True: 0, False: 0}
        }
        for person in people
    }

    names = set(people)
    for have_gene_2 in powerset(names):
        for have_gene_1 in powerset(names - have_gene_2):
            have_gene_0 = names - have_gene_2 - have_gene_1
            for have_trait in powerset(names):
                fails_evidence = False
                for person in names:
                    if people[person]["trait"] is not None:
                        if (people[person]["trait"] == 1) != (person in have_trait):
                            fails_evidence = True
                            break
                if fails_evidence:
                    continue

                p = joint_probability(people, have_gene_1, have_gene_2, have_trait)
                update(probabilities, have_gene_1, have_gene_2, have_trait, p)

    normalize(probabilities)

    for person in people:
        print(f"{person}:")
        for field in ["gene", "trait"]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] if row["mother"] else None,
                "father": row["father"] if row["father"] else None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    s = list(s)
    return [
        set(subset) for i in range(len(s) + 1)
        for subset in itertools.combinations(s, i)
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute the joint probability of a specific configuration of genes and traits.
    """
    joint_p = 1.0

    for person in people:
        # Determine number of genes for the person in current scenario
        n_genes = 2 if person in two_genes else 1 if person in one_gene else 0
        has_trait = person in have_trait

        mother = people[person]["mother"]
        father = people[person]["father"]

        if not mother:
            # No parents: use population distribution
            gene_p = PROBS["gene"][n_genes]
        else:
            # Child with parents: inheritance logic
            # Probability of each parent passing a gene
            passing_probs = {}
            for parent in [mother, father]:
                if parent in two_genes:
                    passing_probs[parent] = 1 - PROBS["mutation"]
                elif parent in one_gene:
                    # 50% chance to pass the gene (mutation cancels out: 0.5*(1-m) + 0.5*m = 0.5)
                    passing_probs[parent] = 0.5
                else:
                    passing_probs[parent] = PROBS["mutation"]

            m_pass = passing_probs[mother]
            f_pass = passing_probs[father]

            # Calculate gene_p based on child's gene count (n_genes)
            if n_genes == 2:
                gene_p = m_pass * f_pass
            elif n_genes == 1:
                gene_p = m_pass * (1 - f_pass) + f_pass * (1 - m_pass)
            else:
                gene_p = (1 - m_pass) * (1 - f_pass)

        # Final person probability: gene probability * trait probability given genes
        joint_p *= gene_p * PROBS["trait"][n_genes][has_trait]

    return joint_p


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add a newly computed joint probability to the existing probability distribution.
    """
    for person in probabilities:
        n_genes = 2 if person in two_genes else 1 if person in one_gene else 0
        probabilities[person]["gene"][n_genes] += p
        probabilities[person]["trait"][person in have_trait] += p


def normalize(probabilities):
    """
    Normalize each probability distribution so that values sum to 1.
    """
    for person in probabilities:
        for field in ["gene", "trait"]:
            total = sum(probabilities[person][field].values())
            if total > 0:
                for val in probabilities[person][field]:
                    probabilities[person][field][val] /= total


if __name__ == "__main__":
    main()
