from pathlib import Path
import json
from typing import List, Dict, Tuple

import faiss
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# CONFIGURATION
# ============================================================

VECTOR_DIR = Path("vector_db")

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

LLM_MODEL = (
    "Qwen/Qwen2.5-1.5B-Instruct"
)

TOP_K = 4

MAX_NEW_TOKENS = 384


# ============================================================
# INDUSTRIAL RAG SYSTEM
# ============================================================

class IndustrialRAG:

    def __init__(self, load_llm=True):

        # ----------------------------------------------------
        # Check FAISS files
        # ----------------------------------------------------

        index_path = (
            VECTOR_DIR / "index.faiss"
        )

        metadata_path = (
            VECTOR_DIR / "metadata.json"
        )

        if not index_path.exists():

            raise FileNotFoundError(
                "FAISS index not found. "
                "Run python build_index.py first."
            )

        if not metadata_path.exists():

            raise FileNotFoundError(
                "metadata.json not found. "
                "Run python build_index.py first."
            )

        # ----------------------------------------------------
        # Load FAISS
        # ----------------------------------------------------

        print("Loading FAISS index...")

        self.index = faiss.read_index(
            str(index_path)
        )

        # ----------------------------------------------------
        # Load metadata
        # ----------------------------------------------------

        print("Loading document metadata...")

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as f:

            self.metadata = json.load(f)

        # ----------------------------------------------------
        # Load embedding model
        # ----------------------------------------------------

        print(
            "Loading embedding model..."
        )

        self.embedder = SentenceTransformer(
            EMBEDDING_MODEL
        )

        # ----------------------------------------------------
        # LLM variables
        # ----------------------------------------------------

        self.model = None

        self.tokenizer = None

        self.device = None

        # ----------------------------------------------------
        # Load LLM only if requested
        # ----------------------------------------------------

        if load_llm:

            self._load_llm()

        else:

            print(
                "Retrieval-only mode: "
                "LLM not loaded."
            )

    # ========================================================
    # LOAD HUGGING FACE LLM
    # ========================================================

    def _load_llm(self):

        if (
            self.model is not None
            and
            self.tokenizer is not None
        ):

            return

        print(
            "\nLoading Hugging Face LLM..."
        )

        print(
            f"Model: {LLM_MODEL}"
        )

        # ----------------------------------------------------
        # Detect GPU / CPU
        # ----------------------------------------------------

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Device: {self.device}"
        )

        # ----------------------------------------------------
        # Load tokenizer
        # ----------------------------------------------------

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                LLM_MODEL
            )
        )

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        if self.device == "cuda":

            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    LLM_MODEL,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            )

        else:

           self.model = (
    AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
)
        self.model.eval()

        print(
            "Hugging Face LLM loaded successfully."
        )

    # ========================================================
    # RETRIEVAL
    # ========================================================

    def retrieve(
        self,
        question: str,
        top_k: int = TOP_K
    ) -> List[Tuple[Dict, float]]:

        # ----------------------------------------------------
        # Convert question to embedding
        # ----------------------------------------------------

        question_embedding = (
            self.embedder.encode(
                [question],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            .astype("float32")
        )

        # ----------------------------------------------------
        # Search FAISS
        # ----------------------------------------------------

        scores, indices = (
            self.index.search(
                question_embedding,
                top_k
            )
        )

        results = []

        # ----------------------------------------------------
        # Collect matching document chunks
        # ----------------------------------------------------

        for score, index_number in zip(
            scores[0],
            indices[0]
        ):

            if index_number < 0:
                continue

            if index_number >= len(
                self.metadata
            ):
                continue

            item = self.metadata[
                index_number
            ]

            results.append(
                (
                    item,
                    float(score)
                )
            )

        return results

    # ========================================================
    # BUILD PROMPT
    # ========================================================

    def build_prompt(
        self,
        question: str,
        retrieved
    ) -> str:

        context_parts = []

        for number, (
            item,
            score
        ) in enumerate(
            retrieved,
            start=1
        ):

            context_parts.append(

                f"""
[SOURCE {number}]

Document:
{item["source"]}

Page:
{item["page"]}

Content:
{item["text"]}
"""
            )

        context = "\n".join(
            context_parts
        )

        # ----------------------------------------------------
        # Prompt Engineering
        # ----------------------------------------------------

        prompt = f"""
You are an AI assistant for industrial
technical documentation.

Answer the user's question using ONLY
the information contained in the provided
industrial documents.

Rules:

1. Do not invent technical information.

2. Do not make up specifications,
   procedures, measurements or values.

3. If the answer is not available in
   the provided documents, say:

"I could not find this information
in the provided industrial documents."

4. Give a clear and concise answer.

5. When possible, mention the document
   and page containing the information.

--------------------------------------------------
INDUSTRIAL DOCUMENT CONTEXT
--------------------------------------------------

{context}

--------------------------------------------------
USER QUESTION
--------------------------------------------------

{question}

--------------------------------------------------
ANSWER
--------------------------------------------------
"""

        return prompt

    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    def generate_answer(
        self,
        question: str,
        retrieved
    ) -> str:

        if not retrieved:

            return (
                "I could not find this information "
                "in the provided industrial documents."
            )

        # Load the LLM only now.
        self._load_llm()

        prompt = self.build_prompt(
            question,
            retrieved
        )

        # ----------------------------------------------------
        # Chat messages
        # ----------------------------------------------------

        messages = [

            {
                "role": "system",
                "content":
                    "You are a careful industrial "
                    "technical assistant."
            },

            {
                "role": "user",
                "content": prompt
            }

        ]

        # ----------------------------------------------------
        # Convert messages to model input
        # ----------------------------------------------------

        text = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        )

        model_inputs = (
            self.tokenizer(
                [text],
                return_tensors="pt"
            )
        )

        model_inputs = {
            key: value.to(
                self.model.device
            )
            for key, value
            in model_inputs.items()
        }

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        with torch.no_grad():

            generated_ids = (
                self.model.generate(
                    **model_inputs,

                    max_new_tokens=
                        MAX_NEW_TOKENS,

                    do_sample=False,

                    pad_token_id=
                        self.tokenizer.eos_token_id
                )
            )

        # ----------------------------------------------------
        # Remove prompt tokens
        # ----------------------------------------------------

        input_length = (
            model_inputs[
                "input_ids"
            ].shape[-1]
        )

        output_ids = (
            generated_ids[0][
                input_length:
            ]
        )

        # ----------------------------------------------------
        # Decode answer
        # ----------------------------------------------------

        answer = (
            self.tokenizer.decode(
                output_ids,
                skip_special_tokens=True
            )
            .strip()
        )

        return answer

    # ========================================================
    # COMPLETE QUESTION → ANSWER PIPELINE
    # ========================================================

    def ask(
        self,
        question: str,
        top_k: int = TOP_K
    ):

        # Step 1: Retrieve
        retrieved = self.retrieve(
            question,
            top_k
        )

        # Step 2: Generate answer
        answer = self.generate_answer(
            question,
            retrieved
        )

        # Step 3: Prepare sources
        sources = []

        for item, score in retrieved:

            sources.append({

                "source":
                    item["source"],

                "page":
                    item["page"],

                "score":
                    score,

                "chunk_id":
                    item["chunk_id"]

            })

        return {

            "question":
                question,

            "answer":
                answer,

            "sources":
                sources

        }


# ============================================================
# COMMAND-LINE CHATBOT
# ============================================================

if __name__ == "__main__":

    rag = IndustrialRAG()

    print("\n")
    print("=" * 70)
    print("AI INDUSTRIAL CHATBOT")
    print("=" * 70)

    while True:

        question = input(
            "\nAsk an industrial question "
            "(or type 'exit'): "
        )

        if question.lower().strip() == "exit":

            print(
                "\nExiting chatbot..."
            )

            break

        if not question.strip():

            continue

        result = rag.ask(
            question
        )

        print("\n")
        print("=" * 70)
        print("ANSWER")
        print("=" * 70)

        print(
            result["answer"]
        )

        print("\n")
        print("=" * 70)
        print("SOURCES")
        print("=" * 70)

        for source in result["sources"]:

            print(
                f"{source['source']} | "
                f"Page {source['page']} | "
                f"Similarity "
                f"{source['score']:.3f}"
            )