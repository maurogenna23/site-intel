# Comparación · https://arniechat.com/

El mismo pipeline en varios modelos. **Campos** es cuántos de los nueve
campos estructurados volvieron con contenido: un modelo barato que devuelve
cuatro nulls no hizo el trabajo, y eso no se ve mirando tokens.

| Modelo | Páginas | Inventadas | Campos | Tokens | Costo | Seg | Palabras |
|---|---|---|---|---|---|---|---|
| GPT-4.1 mini · OpenAI | 6 | 0 | 7/9 | 7,177 | 0.3268 ¢ | 17.8 | 487 |
| Gemini 3.1 Flash Lite · Google | 6 | 0 | 7/9 | 7,660 | 0.3454 ¢ | 34.8 | 407 |
| Llama 3.2 3B · local | 1 | 3 | 7/9 | 2,901 | 0.0000 ¢ | 16.9 | 249 |
| GPT-4.1 nano · OpenAI | 5 | 1 | 6/9 | 6,292 | 0.0981 ¢ | 9.2 | 442 |

## GPT-4.1 mini · OpenAI

Páginas que eligió leer:

- <https://arniechat.com/>
- <https://arniechat.com/automatizar-reservas>
- <https://arniechat.com/chatbot-whatsapp>
- <https://arniechat.com/asistente-de-voz>
- <https://arniechat.com/base-de-conocimiento>
- <https://arniechat.com/conversaciones-en-vivo>

Campos que no pudo extraer: size_hint, hiring_detail

## GPT-4.1 nano · OpenAI

Páginas que eligió leer:

- <https://arniechat.com/>
- <https://arniechat.com/automatizar-reservas>
- <https://arniechat.com/chatbot-whatsapp>
- <https://arniechat.com/asistente-de-voz>
- <https://arniechat.com/chatbot-hoteles>

URLs que devolvió y no estaban en la lista (descartadas): https://arniechat.com/

Campos que no pudo extraer: sector, size_hint, hiring_detail

## Gemini 3.1 Flash Lite · Google

Páginas que eligió leer:

- <https://arniechat.com/>
- <https://arniechat.com/chatbot-whatsapp>
- <https://arniechat.com/asistente-de-voz>
- <https://arniechat.com/base-de-conocimiento>
- <https://arniechat.com/chatbot-hoteles>
- <https://arniechat.com/chatbot-clinicas>

Campos que no pudo extraer: size_hint, hiring_detail

## Llama 3.2 3B · local

Páginas que eligió leer:

- <https://arniechat.com/>

URLs que devolvió y no estaban en la lista (descartadas): https://arniechat.com/nosotros, https://arniechat.com/trabaja-con-nosotros, No encontrado

Campos que no pudo extraer: size_hint, hiring_detail
