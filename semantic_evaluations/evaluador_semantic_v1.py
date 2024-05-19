from semantic_splitters import LLMTextSplitter, split_text_semantic_langchain


text_v1 = "La cocina mediterránea es muy saludable, sobre todo por el uso de aceite, vegetales y otros productos. \
           La estación espacial tiene un problema con algunas bacterias ya que están mutando de forma diferente a como lo hacen en la Tierra. \
           El Real Madrid juega hoy la semifinal de la copa de europa. \
           La familia que come reunida se mantiene unida. \
           El coche eléctrico tiene grandes ventajas pero no queda claro si será el vehículo del futuro si sigue indirectamente alimentado por energías fósiles en lugar de las renovables."

text_v2 = "La cocina mediterránea es muy saludable, sobre todo por el uso de aceite de oliva, vegetales y otros productos. \
           La estación espacial tiene un problema con algunas bacterias ya que están mutando de forma diferente a como lo hacen en la Tierra. \
           El Real Madrid juega hoy la semifinal de la copa de Europa. El otro equipo que juega es el Bayern de Munich. \
           La familia que come reunida se mantiene unida. Y si es grande mucho mejor y más divertido. \
           El coche eléctrico tiene grandes ventajas pero no queda claro si será el vehículo del futuro si sigue indirectamente alimentado por energías fósiles en lugar de las renovables."

texts = [text_v1, text_v2]

for text in texts:
    with open("semantic_evaluations/langchain_chunks.txt", 'a', encoding='utf-8') as f:
        chunks = split_text_semantic_langchain(text, False, 95)

        for chunk in chunks:
            f.writelines(chunk)
            f.write("\n\n")
        f.write("\n---\n")

    with open("semantic_evaluations/gpt_chunks.txt", 'a', encoding='utf-8') as f:
        llm_splitter = LLMTextSplitter(count_tokens=True)
        chunks = llm_splitter.split_text(text)

        for chunk in chunks:
            f.writelines(chunk)
            f.write("\n\n")
        f.write("\n---\n")