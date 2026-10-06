from rag import IndustrialRAG


# Questions designed to test each of our 4 documents
TEST_QUESTIONS = [

    # Motor document
    "What is the preventive maintenance trigger for the IM-45 motor?",

    "What are the common causes of motor overheating?",

    # PLC document
    "Which PLC input indicates motor overload?",

    "What does Q0.0 control?",

    # CNC document
    "What should be verified before running a CNC program?",

    "What is a work coordinate system?",

    # Safety document
    "Why must stored energy be considered after electrical isolation?",

    "What is the purpose of lockout tagout?"
]


def main():

    print("=" * 70)
    print("AI INDUSTRIAL CHATBOT")
    print("RETRIEVAL TEST")
    print("=" * 70)

    # IMPORTANT:
    # We only want to test document retrieval here.
    # We do NOT need to load the large language model yet.
    rag = IndustrialRAG(
        load_llm=False
    )

    for number, question in enumerate(
        TEST_QUESTIONS,
        start=1
    ):

        print("\n")
        print("=" * 70)
        print(f"QUESTION {number}")
        print("=" * 70)

        print(question)

        # Search the FAISS database
        results = rag.retrieve(
            question,
            top_k=3
        )

        print("\nTOP RETRIEVED DOCUMENT CHUNKS")
        print("-" * 70)

        if not results:

            print(
                "No relevant documents found."
            )

            continue

        for rank, (item, score) in enumerate(
            results,
            start=1
        ):

            print(
                f"\nResult {rank}"
            )

            print(
                f"Similarity score: {score:.3f}"
            )

            print(
                f"Source: {item['source']}"
            )

            print(
                f"Page: {item['page']}"
            )

            print(
                f"Chunk: {item['chunk_id']}"
            )

            # Show only the first part of the
            # retrieved text so the terminal
            # doesn't become extremely long.
            print(
                f"Text: {item['text'][:300]}..."
            )


if __name__ == "__main__":
    main()