# ArcheologyGuide
This is the repository for the Archeology Guide for the Palimpsisto Project.
This project is supported by the PALIMPSISTO project cofinanced by the European Regional Development Fund of the European Union and Greek national funds through the Operational Program Competitiveness, Entrepreneurship and Innovation, under the call RESEARCH–CREATE– INNOVATE (project code: T2EDK-01894).



## Technologies behind the Archeology Guide
The Archeology Guide is a back end service that is based on the following technologies:
* [FastAPI](https://fastapi.tiangolo.com/) - The  backend framework used.
* [Dolly-V2-3b](https://huggingface.co/databricks/dolly-v2-3b) - The base LLM that was fine-tuned and was used to answer questions about the archeological sites to the users.
* [Parrot Paraphaser](https://huggingface.co/prithivida/parrot_paraphraser_on_T5) - The paraphraser that was fine tuned to change some context so the returned info are not always the same.
* [Roberta Base Squad2](https://huggingface.co/deepset/roberta-base-squad2) - The QA model that was used to answer questions about the archeological sites to the users if we are of high confidence for the answer.
* [T5 Small Base](https://huggingface.co/allenai/t5-small-squad2-question-generation) - The base model that was fine tuned to generate questions about the archeological sites.
* [MiniLM-L6 v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - The similarity model that was used to retrieve the most relative context to the user's question.
