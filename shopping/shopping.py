import csv
import sys

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Mapping month abbreviations to integers 0‑11
MONTHS = {
    'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5,
    'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11
}


def load_data(filename):
    """
    Load shopping data from a CSV file and convert it into a format suitable
    for a k‑nearest neighbours classifier.

    Returns a tuple (evidence, labels):
      - evidence: list of lists, each inner list contains the 17 numeric features
      - labels:   list of ints (1 if the user made a purchase, else 0)
    """
    evidence = []
    labels = []

    with open(filename, mode='r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header row

        for row in reader:
            # 1. Administrative (int)
            administrative = int(row[0])
            # 2. Administrative_Duration (float)
            administrative_duration = float(row[1])
            # 3. Informational (int)
            informational = int(row[2])
            # 4. Informational_Duration (float)
            informational_duration = float(row[3])
            # 5. ProductRelated (int)
            product_related = int(row[4])
            # 6. ProductRelated_Duration (float)
            product_related_duration = float(row[5])
            # 7. BounceRates (float)
            bounce_rates = float(row[6])
            # 8. ExitRates (float)
            exit_rates = float(row[7])
            # 9. PageValues (float)
            page_values = float(row[8])
            # 10. SpecialDay (float)
            special_day = float(row[9])
            # 11. Month -> int (0‑11)
            month = MONTHS[row[10]]
            # 12. OperatingSystems (int)
            operating_systems = int(row[11])
            # 13. Browser (int)
            browser = int(row[12])
            # 14. Region (int)
            region = int(row[13])
            # 15. TrafficType (int)
            traffic_type = int(row[14])
            # 16. VisitorType -> 1 if "Returning_Visitor", else 0
            visitor_type = 1 if row[15] == 'Returning_Visitor' else 0
            # 17. Weekend -> 1 if TRUE, else 0
            weekend = 1 if row[16] == 'TRUE' else 0

            # Assemble the evidence list for this user
            evidence_row = [
                administrative,
                administrative_duration,
                informational,
                informational_duration,
                product_related,
                product_related_duration,
                bounce_rates,
                exit_rates,
                page_values,
                special_day,
                month,
                operating_systems,
                browser,
                region,
                traffic_type,
                visitor_type,
                weekend
            ]

            evidence.append(evidence_row)

            # Label: 1 if Revenue == 'TRUE', else 0
            labels.append(1 if row[17] == 'TRUE' else 0)

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Train a 1‑nearest neighbour classifier on the provided evidence and labels.

    Returns the fitted KNeighborsClassifier.
    """
    # k = 1 as per the specification
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Calculate sensitivity (true positive rate) and specificity (true negative rate).

    Returns a tuple (sensitivity, specificity) as floats.
    """
    # Count true positives, true negatives, actual positives, actual negatives
    tp = 0
    tn = 0
    actual_pos = 0
    actual_neg = 0

    for true, pred in zip(labels, predictions):
        if true == 1:
            actual_pos += 1
            if pred == 1:
                tp += 1
        else:  # true == 0
            actual_neg += 1
            if pred == 0:
                tn += 1

    sensitivity = tp / actual_pos if actual_pos > 0 else 0.0
    specificity = tn / actual_neg if actual_neg > 0 else 0.0

    return (sensitivity, specificity)


def main():
    # ... (the provided main() remains untouched) ...
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data.csv")

    evidence, labels = load_data(sys.argv[1])

    # Split into training and testing sets (75% / 25%)
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=0.25, random_state=42
    )

    # Train model
    model = train_model(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {sensitivity * 100:.2f}%")
    print(f"True Negative Rate: {specificity * 100:.2f}%")


if __name__ == "__main__":
    main()
