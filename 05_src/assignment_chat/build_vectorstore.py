import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

docs = [
    "When humidity is high and temperatures are hot, wear light, breathable fabrics like linen or cotton.",
    "Freezing rain creates black ice. Avoid driving if possible; if you must, drive slowly and increase following distance.",
    "A UV index of 8 or above is very high. Apply SPF 30+ sunscreen, wear a hat, and avoid sun exposure between 10am-4pm.",
    "Wind chill makes air feel colder than the actual temperature. At 0°C with 30 km/h winds, it can feel like -10°C.",
    "A barometric pressure drop usually signals incoming rain or storms, while rising pressure indicates clearing skies.",
    "Fog forms when the air temperature and dew point are within 2–3 degrees of each other, usually overnight or at dawn.",
    "Lightning can strike up to 16 km away from the center of a storm. If you can hear thunder, you are already within striking range.",
    "Snow is actually a better insulator than rain. A heavy snowfall can keep ground temperatures several degrees warmer than a cold, clear night.",
    "Heat exhaustion can occur when the humidex (heat index) exceeds 40°C, even if the actual air temperature is lower.",
    "A wind speed above 60 km/h is considered a storm-force wind and can make walking difficult and cause minor structural damage."    
]

embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("weather_knowledge", embedding_function=embedding_fn)

collection.add(
    documents=docs,
    ids=[f"doc_{i}" for i in range(len(docs))]
)

print(f"Stored {len(docs)} documents.")