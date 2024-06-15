# TFG-LLMDocs

Este sistema es fruto del desarrollo del Trabajo de Fin de Grafo del grado de Ingeniería Informática en la Universidad de La Laguna "Modelos de lenguaje grandes para el tratamiento de datos documentales".

Memoria del proyecto: 

## Características principales

Consiste en un sistema de preguntas y respuestas basado en la arquitectura de generación con recuperación aumentada (RAG, por sus siglas en inglés). Utiliza un modelo de lenguaje grande para generar de respuestas en base a una pregunta sobre un conjunto de datos documentales, además de otras tareas dentro del flujo del sistema.

## Uso

Ejecución: chainlit run qa_rag_app.py

El uso de RAG v2 requiere tener instalado Neo4j: https://neo4j.com/docs/operations-manual/current/installation/

## Estructura del proyecto

qa_rag_app.py: Aplicación principal
preprocess_functions.py: Funciones de preprocesado de documentos
split_functions: Funciones de troceado
label_functions.py: Funciones de etiquetado
my_embedding_function.py: Función de embeddings personalizada 
prompts.py: Prompts utilizados en todo el código

Directorios /pdf_*: Documentos para la ejecución
Directorio /semantic_evaluations: Funciones de troceado semántico y scripts de evaluación