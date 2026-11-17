

from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent

load_dotenv()


# 1. Gewünschte Datenstruktur festlegen
class ProductSearch(BaseModel):
    category: str
    max_price: int | None = None
    min_ram_gb: int | None = None
    purpose: str


# 2. LLM mit strukturiertem Output konfigurieren
agent = Agent(
    "openai:gpt-4.1-mini",
    output_type=ProductSearch,
    instructions=(
        "Extrahiere die Suchkriterien aus der Benutzeranfrage. "
        "Erfinde keine fehlenden Angaben."
    ),
)


# 3. Natürlichsprachliche Anfrage
result = agent.run_sync(
    "Ich suche einen Laptop für Python-Entwicklung "
    "mit mindestens 32 GB RAM für maximal 1500 Euro."
)


# 4. Validiertes Pydantic-Objekt
product = result.output

print(product.category)
print(product.max_price)
print(product.min_ram_gb)


# 5. Als JSON ausgeben
print(product.model_dump_json(indent=2))
