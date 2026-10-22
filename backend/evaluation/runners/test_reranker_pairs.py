
from app.rag.reranker import get_reranker

query = "When does Apple's fiscal year end?"

relevant_text = """
The Company's fiscal year is the 52- or 53-week period
that ends on the last Saturday of September. An additional
week is included in the first fiscal quarter every five or
six years to realign fiscal quarters with calendar quarters.
"""

irrelevant_text = """
Services 30,739 27,423 91,728 80,408.
Total net sales $109,417 $94,036 $364,357 $313,695.
The Company's proportion of net sales by disaggregated
revenue source was generally consistent.
"""

model = get_reranker()

pairs = [
    (query, relevant_text),
    (query, irrelevant_text),
]

scores = model.predict(pairs)

print("\nRERANKER SANITY CHECK")
print("-" * 50)
print(f"Relevant fiscal-year definition: {float(scores[0]):.6f}")
print(f"Unrelated revenue table:         {float(scores[1]):.6f}")

if scores[0] > scores[1]:
    print("\nPASS: Relevant text receives the higher score.")
else:
    print("\nFAIL: Investigate model behavior before changing the pipeline.")
