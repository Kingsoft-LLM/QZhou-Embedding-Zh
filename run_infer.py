from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

def get_prompteol_input(text: str) -> str:
    return f"This sentence: <|im_start|>“{text}” means in one word: “"

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

model = SentenceTransformer(
    "Kingsoft-LLM/QZhou-Embedding-Zh",
    model_kwargs={"device_map": "cuda", "trust_remote_code": True},
    tokenizer_kwargs={"padding_side": "left", "trust_remote_code": True},
    trust_remote_code=True
)

task= "Given a web search query, retrieve relevant passages that answer the query"
queries = [
    get_prompteol_input(get_detailed_instruct(task, "光合作用是什么？")),
    get_prompteol_input(get_detailed_instruct(task, "电话是谁发明的？"))
]

documents = [
    get_prompteol_input("光合作用是绿色植物利用阳光、二氧化碳和水生成葡萄糖和氧气的过程。这一生化反应发生在叶绿体中。"),
    get_prompteol_input("亚历山大·格拉汉姆·贝尔（Alexander Graham Bell）因于1876年发明了第一台实用电话而广受认可，并为此设备获得了美国专利第174,465号。")
]
breakpoint()
query_embeddings = model.encode(queries, normalize_embeddings=False)
document_embeddings = model.encode(documents, normalize_embeddings=False)

dim=1792 # 128, 256, 512, 768, 1024, 1280, 1536, 1792
query_embeddings = normalize(query_embeddings[:, :dim])
document_embeddings = normalize(document_embeddings[:, :dim])

similarity = model.similarity(query_embeddings, document_embeddings)
print(similarity)