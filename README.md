# AkashlinuxGPT

This project was inspired by the original privateGPT (https://github.com/imartinez/privateGPT) and LocalGPT (https://github.com/PromtEngineer/localGPT)

In this model, I have replaced the Vicuna-7B model with Google Flan T5 Xxl and we are using the InstructorEmbeddings instead of LlamaEmbeddings as used in the original privateGPT. Both Embeddings as well as LLM will run on GPU instead of CPU. It also has CPU support if you do not have a GPU (see below for instruction). 



# Environment Setup
In order to set your environment up to run the code here, first install all requirements:

```shell
pip install -r requirements.txt
```

## Test dataset
This repo uses a [Constitution of USA ] as an example.

## Instructions for ingesting your own dataset

Put any and all of your .txt, .pdf, or .csv files into the SOURCE_DOCUMENTS directory
in the load_documents() function, replace the docs_path with the absolute path of your source_documents directory. 

Run the following command to ingest all the data.

```shell
python ingest.py
```

It will create an index containing the local vectorstore. Will take time, depending on the size of your documents.
You can ingest as many documents as you want, and all will be accumulated in the local embeddings database. 
If you want to start from an empty database, delete the `index`.

Note: When you run this for the first time, it will download take time as it has to download the embedding model. In the subseqeunt runs, no data will leave your local enviroment and can be run without internet connection.

## Ask questions to your documents, locally!
In order to ask a question, run a command like:

```shell
python run_localGPT.py
```

And wait for the script to require your input. 

```shell
> Enter a query:
```

Hit enter. Wait while the LLM model consumes the prompt and prepares the answer. Once done, it will print the answer and the 4 sources it used as context from your documents; you can then ask another question without re-running the script, just wait for the prompt again. 

Note: When you run this for the first time, it will need internet connection to download the vicuna-7B model. After that you can turn off your internet connection, and the script inference would still work. No data gets out of your local environment.

Type `exit` to finish the script.

# Run it on CPU
By default, localGPT will use your GPU to run both the `ingest.py` and `run_localGPT.py` scripts. But if you do not have a GPU and want to run this on CPU, now you can do that (Warning: Its going to be slow!). You will need to use `--device_type cpu`flag with both scripts. 

For Ingestion run the following: 
```shell
python ingest.py --device_type cpu
```
In order to ask a question, run a command like:

```shell
python run_localGPT.py --device_type cpu
```

# How does it work?
Selecting the right local models and the power of `LangChain` you can run the entire pipeline locally, without any data leaving your environment, and with reasonable performance.

- `ingest.py` uses `LangChain` tools to parse the document and create embeddings locally using `InstructorEmbeddings`. It then stores the result in a local vector database using `Chroma` vector store. 
- `run_localGPT.py` uses a local LLM (Vicuna-7B in this case) to understand questions and create answers. The context for the answers is extracted from the local vector store using a similarity search to locate the right piece of context from the docs.
- You can replace this local LLM with any other LLM from the HuggingFace. Make sure whatever LLM you select is in the HF format.

# System Requirements

## Python Version
To use this software, you must have Python 3.10 or later installed. Earlier versions of Python will not compile.

## C++ Compiler
If you encounter an error while building a wheel during the `pip install` process, you may need to install a C++ compiler on your computer.

### For Windows 10/11
To install a C++ compiler on Windows 10/11, follow these steps:

1. Install Visual Studio 2022.
2. Make sure the following components are selected:
   * Universal Windows Platform development
   * C++ CMake tools for Windows
3. Download the MinGW installer from the [MinGW website](https://sourceforge.net/projects/mingw/).
4. Run the installer and select the "gcc" component.

### NVIDIA Driver's Issues:
Follow this [page](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-22-04) to install NVIDIA Drivers. 
