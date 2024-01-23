**Question Answering and Text Generation Repository**

This repository contains implementations of various Natural Language Processing (NLP) models and applications for question-answering, image description generation, stable diffusion, PDF question answering, and CSV question answering. These models leverage Mistral API, IDEFICS 9B, Stable Diffusion, Google Gemini API, RAG, and Google TAPAS.

**Models and Implementations:**

**Mistral API for General Question Answering:**

We the Mistral API as given in the competition for handling a wide range of general question-answering tasks.

**IDEFICS 9B for Generating Image Descriptions:**

We've implemented a solution utilizing the IDEFICS-9B model, a variant designed for vision-to-text tasks, particularly in the context of Hugging Face's Transformers library. This powerful model is adept at generating descriptive textual content from images, demonstrating its capability to bridge the gap between visual data and natural language understanding.

The code begins by setting up the necessary environment, checking for GPU availability, and loading the pre-trained IDEFICS-9B model checkpoint from Hugging Face's model hub. Additionally, it employs the BitsAndBytesConfig to enable quantization, optimizing the model for efficient inference while maintaining performance. The AutoProcessor class handles the tokenization of prompts, converting them into a format suitable for input to the model.

The script then prepares a sample prompt consisting of an image file path and a textual description. The model is fed this prompt using the processor, and the generated text is obtained. The use of bad_words_ids helps guide the generation process, preventing the model from outputting certain undesired tokens.

The implementation is flexible, supporting both batched and single-sample modes. The generated text is then decoded, excluding any special tokens, and printed for further analysis.

Real-world applications of this code could be found in industries requiring automated image captioning, such as content indexing for large image databases or assisting visually impaired individuals in understanding image content. It can also be employed in e-commerce for enhancing product descriptions, in social media for auto-generating captions, or even in medical imaging for generating textual reports based on diagnostic images. The utilization of quantization ensures that the model is deployable in resource-constrained environments, making it applicable in a wide array of scenarios where bridging the visual-textual gap is crucial.


**Stable Diffusion using Model "CompVis/stable-diffusion-v1-4":**

A Stable Diffusion Pipeline is implemented using the Diffusers library and the Hugging Face Transformers library. The Stable Diffusion model, identified by the model_id "CompVis/stable-diffusion-v1-4", is loaded into the pipeline, and a scheduler, specifically a PNDMScheduler, is set up to control the diffusion process during generation. The code leverages mediaPy for displaying and saving the generated images.

Before executing the code, several Python packages, including diffusers, transformers, scipy, mediapy, and accelerate, are installed and upgraded. Additionally, the Hugging Face CLI login is performed (hugging face token is required for this), ensuring authentication for model retrieval.

The StableDiffusionPipeline is initialized with the pre-trained model, specifying the PNDMScheduler for diffusion control. The pipeline is configured to use mixed-precision inference (torch.float16) for improved performance. If desired, a safety checker can be removed to disable certain safety measures in the pipeline.

The code then generates images based on a given prompt ("photo of human hands") using the pipeline. The autocast context manager is utilized to enable automatic mixed-precision inference, optimizing computation on CUDA-enabled GPUs. The resulting images are displayed using mediaPy and saved to an output.jpg file.

The Stable Diffusion model, with its ability to generate high-quality images, finds applications in various domains, including art generation, content creation, and data augmentation for computer vision tasks. The PNDMScheduler enhances the flexibility of the diffusion process, allowing users to finely control the trade-off between image quality and computational resources.

Overall, this code showcases the integration of Stable Diffusion models into a pipeline for image generation, providing a versatile tool for users interested in harnessing state-of-the-art generative models for creative or practical purposes.

**PDF Question Answering using Google Gemini API and RAG:**

A comprehensive pipeline for question-answering from a PDF document is implemented, leveraging various libraries and services. Let's break down the key components:

1. **Google API Key Setup:**
   - A Google API key is configured to access Google's GenerativeAI service.
   - Steps to get API Key:
   - Go to the website:  https://makersuite.google.com/app/apikey  ( First sign in and then you will be able to create API key )

2. **Installation of Dependencies:**
   - Requirements from a specified file are installed using the pip package manager. This requirements.txt is provided in the respected folder.

3. **PDF Processing:**
   - The PyPDF2 library is used to extract text from a PDF file. The extracted text is then divided into manageable chunks using a RecursiveCharacterTextSplitter.

4. **Embeddings and Vector Store:**
   - GoogleGenerativeAIEmbeddings is employed to generate embeddings for the text chunks.
   - The embeddings are used to create a FAISS vector store, enabling efficient similarity search.

5. **Conversational Chain Setup:**
   - A conversational chain is established for question-answering using the ChatGoogleGenerativeAI model, specifically the "gemini-pro" variant. A template for the QA prompt is defined.

6. **User Input Processing:**
   - User input in the form of a question is embedded using GoogleGenerativeAIEmbeddings.
   - Similar documents are retrieved from the vector store based on the user's question.
   - The conversational chain is utilized to generate a detailed answer to the user's question.

7. **Application to PDF Document:**
   - A sample PDF document ("Music_Generation_using_Deep_Learning.pdf") is processed using the defined pipeline.
   - The pipeline involves extracting text, creating embeddings and a vector store, setting up a conversational chain, and finally responding to a user's input question.

**Industry-Level Applications:**
   - **Document Understanding and Retrieval:** This code facilitates efficient searching and retrieval of information from large documents, a crucial task in industries dealing with extensive documentation and research.

   - **Automated Question-Answering Systems:** The conversational chain and generative model can be deployed in customer support systems, helping industries streamline responses to user queries.

   - **Knowledge Extraction and Summarization:** The pipeline can be adapted for knowledge extraction, summarization, and quick retrieval of relevant information, beneficial in sectors like legal, healthcare, and research.

   - **Educational Resources:** The ability to answer questions from documents makes this code valuable for creating educational platforms and tools, aiding students and professionals in accessing relevant information.

Overall, this provides a versatile and powerful solution for extracting, organizing, and retrieving information from documents, demonstrating its applicability in various industry scenarios.


**CSV Question Answering using Google TAPAS:**

We've taken a strategic approach to seamlessly integrate cutting-edge technologies for efficient table-based question-answering tasks. Let's break down the key components:

1. **Library Installation and Setup:**
   - The Transformers library version 4.4.2 is installed, ensuring compatibility with the subsequent code.
   - The torch library is imported, establishing a foundation for PyTorch-based computations.

2. **PyTorch Geometric Library Installation:**
   - The torch-scatter library is installed using a specific URL, enhancing the PyTorch environment with geometric operations. This is particularly useful for scenarios requiring advanced graph-based computations.

3. **Transformers Pipeline for Table Question-Answering:**
   - The code utilizes the Transformers library's pipeline module to set up a Table Question-Answering (TQA) task.
   - Specifically, the Google TAPAS (Tabular Pretrained Language Model) model, fine-tuned on the WTQ (Web Table Questions) dataset, is employed. This model is optimized for accurately answering questions related to tabular data.

4. **Data Preparation:**
   - A tabular dataset is loaded into a Pandas DataFrame (`canned-consumption.csv`).
   - The dataset is converted to a string format to ensure consistent processing.

5. **Question-Answering Execution:**
   - A sample query, "What is the amount of Tuna fish at 2012?" is formulated.
   - The TQA pipeline is executed on the provided table and query, producing a detailed answer.

6. **Industry-Level Relevance:**
   - **Data Analytics and Business Intelligence:** This code demonstrates a streamlined approach to extracting insights from tabular data, a crucial task in industries relying on data analytics and business intelligence.

   - **Financial Analysis:** In finance, this technique can be employed to quickly retrieve specific financial metrics or historical data points from large datasets, aiding in financial decision-making.

   - **Healthcare Data Processing:** For healthcare professionals dealing with extensive patient records, this approach facilitates swift extraction of relevant information, enhancing patient care and research.

   - **Supply Chain Management:** Industries managing complex supply chains can leverage this code to extract critical information from tables, enabling better decision-making and optimization.

   - **Automated Reporting:** Implementing such table-based question-answering systems is instrumental in automating report generation processes across various domains.

In essence, this code showcases the integration of advanced language models for precise question-answering in the context of tabular data, offering a versatile solution with significant implications for industry-level applications.

**Gradio Applications**
We deployed interactive Gradio applications for - 

Mistral API-based question answering
Stable Diffusion image processing
CSV question answering with Google TAPAS

The video demos are uploaded as mp4 format, you need to download it for watching.
