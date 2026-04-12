import torch
from rag import RAG
from cross_transformer import MultiChunkTransformer


context = """
Renewable energy refers to energy generated from natural sources that are constantly replenished,
such as solar, wind, hydroelectric, geothermal, and biomass energy.

Solar power systems convert sunlight into electricity using photovoltaic (PV) panels.
These systems are widely used in residential rooftops and large-scale solar farms.

Wind energy is generated using wind turbines that convert kinetic energy from wind into electrical power.
Countries with strong wind infrastructure include the United States, Germany, and China.

Hydroelectric power is produced by using flowing water to spin turbines, typically in dams.
It is one of the oldest and most widely used renewable energy sources.

Geothermal energy harnesses heat from beneath the Earth's surface to generate electricity and heating.

Battery storage systems are critical for renewable energy integration because they store excess energy
for use when production is low (e.g., at night for solar power).

Electric vehicles (EVs) are closely linked to renewable energy adoption since they reduce dependence on fossil fuels.

Smart grids use AI and IoT systems to optimize electricity distribution and balance energy demand in real time.

Climate change mitigation strategies include reducing carbon emissions, increasing renewable adoption,
and improving energy efficiency across industries.

Many governments are investing heavily in renewable infrastructure to achieve net-zero emissions targets.
"""



rag = RAG()
rag.add_context(context)


query = "what are the different types of renewable energy"


top_k = rag.retrieve_top_k(query, k=3)

print("\n===== TOP-K RETRIEVED CHUNKS =====\n")
chunks = []
for i, item in enumerate(top_k):
    print(f"[{i}] score={item['score']:.4f}")
    print(item["text"])
    print("-" * 80)
    chunks.append(item["text"])

model = MultiChunkTransformer()

importance = model(query, chunks)

print("\n===== CHUNK IMPORTANCE SCORES =====\n")

for i, score in enumerate(importance.squeeze(0).tolist()):
    print(f"Chunk {i}: {score:.4f}")


best_chunk_idx = torch.argmax(importance).item()

print("\n===== BEST CHUNK =====\n")
print(f"Chunk {best_chunk_idx}")
print(chunks[best_chunk_idx])