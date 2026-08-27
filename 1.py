import transformers
from transformers import pipeline

print("Loading NLP Models... Please Wait.")

# -------------------------------
# Load Models
# -------------------------------

# Text Classification
text_classification = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Zero-Shot Classification
zero_shot = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# Question Answering
qa = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)

# Mask Filling
mask_fill = pipeline(
    "fill-mask",
    model="bert-base-uncased"
)

print("\nAll NLP Models Loaded Successfully!")

# -------------------------------
# Menu Driven Program
# -------------------------------

while True:

    print("\n" + "=" * 60)
    print("            NLP MENU DRIVEN PROGRAM")
    print("=" * 60)
    print("1. Text Classification")
    print("2. Zero-Shot Classification")
    print("3. Question Answering")
    print("4. Mask Filling")
    print("5. Exit")
    print("=" * 60)

    choice = input("Enter your choice (1-5): ").strip()

    if not choice.isdigit():
        print("\nInvalid Choice!")
        print("Please Enter a Number Between 1 and 5.")
        continue

    choice = int(choice)

    # ---------------------------
    # Text Classification
    # ---------------------------
    if choice == 1:

        text = input("\nEnter a Sentence:\n")

        result = text_classification(text)

        print("\nText Classification")
        print("-" * 40)
        print("Label      :", result[0]["label"])
        print("Confidence :", round(result[0]["score"], 4))

    # ---------------------------
    # Zero-Shot Classification
    # ---------------------------
    elif choice == 2:

        text = input("\nEnter a Sentence:\n")
        labels = input("Enter labels separated by commas:\n")

        candidate_labels = [label.strip() for label in labels.split(",")]

        result = zero_shot(text, candidate_labels)

        print("\nZero-Shot Classification")
        print("-" * 40)
        print("Predicted Label :", result["labels"][0])
        print("Confidence      :", round(result["scores"][0], 4))

    # ---------------------------
    # Question Answering
    # ---------------------------
    elif choice == 3:

        context = input("\nEnter the Context:\n")
        question = input("Enter the Question:\n")

        result = qa(question=question, context=context)

        print("\nQuestion Answering")
        print("-" * 40)
        print("Answer      :", result["answer"])
        print("Confidence  :", round(result["score"], 4))

    # ---------------------------
    # Mask Filling
    # ---------------------------
    elif choice == 4:

        print("\nUse [MASK] where the missing word should be.")

        sentence = input("\nEnter a Sentence:\n")

        result = mask_fill(sentence)

        print("\nMask Filling")
        print("-" * 40)

        for r in result:
            print(f"{r['token_str']} : {round(r['score'], 4)}")

    # ---------------------------
    # Exit
    # ---------------------------
    elif choice == 5:

        print("\nThank You!")
        print("Exiting NLP Menu Program...")
        break

    # ---------------------------
    # Invalid Choice
    # ---------------------------
    else:

        print("\nInvalid Choice!")
        print("Please Enter a Number Between 1 and 5.")