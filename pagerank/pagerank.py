import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are a list of pages.
    """
    pages = dict()
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?\s+)?href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}
    return pages


def transition_model(corpus, page, damping_factor):
    """
    Returns a probability distribution over which page to visit next,
    given a current page and a damping factor.
    """
    probs = {}
    links = corpus[page]
    total_pages = len(corpus)

    # If the page has no outgoing links, treat it as if it links to everywhere
    if not links:
        for p in corpus:
            probs[p] = 1 / total_pages
        return probs

    # Probability of picking a page at random (1 - d)
    random_prob = (1 - damping_factor) / total_pages

    # Probability of following a specific link (d / number of links)
    link_prob = damping_factor / len(links)

    for p in corpus:
        probs[p] = random_prob
        if p in links:
            probs[p] += link_prob

    return probs


def sample_pagerank(corpus, damping_factor, n):
    """
    Estimates PageRank for each page by sampling n pages from a Markov Chain.
    """
    # Initialize page counts
    counts = {page: 0 for page in corpus}

    # First sample is completely random
    current_page = random.choice(list(corpus.keys()))
    counts[current_page] += 1

    # Generate the remaining n-1 samples
    for _ in range(n - 1):
        model = transition_model(corpus, current_page, damping_factor)
        pages = list(model.keys())
        weights = list(model.values())

        # Pick next page based on transition probabilities
        current_page = random.choices(pages, weights=weights, k=1)[0]
        counts[current_page] += 1

    # Normalize counts to get probabilities (summing to 1)
    return {page: count / n for page, count in counts.items()}


def iterate_pagerank(corpus, damping_factor):
    """
    Calculates PageRank using the iterative formula until convergence.
    """
    N = len(corpus)
    # Start with every page having 1/N rank
    ranks = {page: 1 / N for page in corpus}

    while True:
        new_ranks = {}
        for page in corpus:
            # First part of the formula: (1 - d) / N
            rank_sum = (1 - damping_factor) / N

            # Second part: d * sum of (PR(i) / NumLinks(i))
            link_contribution = 0
            for possible_linker in corpus:
                links = corpus[possible_linker]

                # Rule: page with no links links to everyone
                if not links:
                    link_contribution += ranks[possible_linker] / N
                elif page in links:
                    link_contribution += ranks[possible_linker] / len(links)

            new_ranks[page] = rank_sum + (damping_factor * link_contribution)

        # Check for convergence (threshold = 0.001)
        converged = True
        for page in ranks:
            if abs(new_ranks[page] - ranks[page]) > 0.001:
                converged = False
                break

        if converged:
            break

        ranks = new_ranks.copy()

    return ranks


if __name__ == "__main__":
    main()
