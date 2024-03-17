import datetime
from typing import Literal, Union

from colorama import Fore
from colorama import Style
from requests.exceptions import ConnectionError
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader

from os.path import exists
import tiktoken
import datetime

# todo: replace with puppeteer, this one gets blocked occasionally
from googlesearch import search


class Query:
    """class for bundling all data required for embedding and search operations"""
    prompt_core: str = ''

    web_query: str = ''
    web_extra_params: Union[dict, None] = None
    web_tbs = 0

    db_search_query: str = ''  # query to search by
    db_embedding_prefix: str = ''  # prefixed to each article saved to faiss db
    db_embedding_postfix: str = ''  # -||-
    db_save_file_extension: str = ''  # most types will have dedicated db for them

    def __init__(self, prompt_core: str):
        self.prompt_core = prompt_core
        self.db_embed_query = prompt_core  # query to search by

        # this prompt works well for Google News searches
        self.web_query = prompt_core
        self.web_extra_params = {
            'tbm': 'nws',  # news only
        }
        self.web_tbs = 'qdr:m'  # last month only
        self.db_search_query = f"important fragments of {prompt_core}"
        self.db_save_file_extension = f"_news_{datetime.date.today().strftime('%Y_%m_%d').lower()}"

def purify_name(name):
    return '_'.join('_'.join(name.split(':')).split('-'))


def is_text_junk(text: str):
    # checks if text contains any of junky keywords eg: privacy policy, subscribe, cookies etc.
    # do not expand this list, it has to be small to be efficient, and these words are grouped either way.
    trigger_list = [
        'sign in', 'privacy policy', 'skip to', 'newsletter', 'subscribe', 'related tags', 'share price'
    ]
    low_text = text.lower()
    for trigger in trigger_list:
        if trigger in low_text:
            return True
    return False


def extract_from_quote(text: str):
    if '"' in text:
        return text.split('"')[1]
    else:
        return text


def reduce(text: str, goal: str, match: str):
    if match in text:
        text = goal.join(text.split(match))
        return reduce(text, goal, match)
    return goal.join(text.split(match))


def remove(text: str, wordlist: list):
    for word in wordlist:
        text = ''.join(text.split(word))
    return text


model_name = "mixtral:8x7b-instruct-v0.1-q5_K_M"  # "llama2-uncensored:7b"
model_safe_name = purify_name(model_name)
token_limit = 32768  # depending on VRAM, try 2048, 3072 or 4096. 2048 works great on 4GB VRAM
llm = Ollama(model=model_name)

embedding_model_name = "nomic-embed-text"  # this is not a good model, change asap
embedding_model_safe_name = purify_name(embedding_model_name)
embeddings = OllamaEmbeddings(model=embedding_model_name)

embeddings_chunk_size = 600  # it is not recommended to play with this value, [100 - 600]
embeddings_article_limit = 10  # adjust depending on how fast 'database vectorization' runs [3 - 100]
embeddings_buffer_stops = ["\n\n\n", "\n\n", "\n", ". ", ", ", " ", ""]  # N of elements LTR [4 - 7]

encoder = tiktoken.get_encoding("cl100k_base")
output_parser = StrOutputParser()


def create_db_if_not_exists(db_name: str):
    if not exists('store/' + db_name + '.faiss'):
        print(f"Creating new database:", db_name + '.faiss')
        tmp_db = FAISS.from_texts(['You are a large language model, intended for research purposes.'], embeddings)
        tmp_db.save_local(folder_path='store', index_name=db_name)
    else:
        print(f"Already exists:", db_name + '.faiss')


def get_db_by_name(db_name: str) -> FAISS:
    create_db_if_not_exists(db_name)
    return FAISS.load_local(folder_path='store', embeddings=embeddings, index_name=db_name)


def populate_db_with_google_search(database: FAISS, query: Query):
    print(f"{Fore.CYAN}{Style.BRIGHT}Searching for:{Style.RESET_ALL}", query.web_query)

    url_list = search(
        query=query.web_query,
        stop=embeddings_article_limit,
        lang='en',
        safe='off',
        tbs=query.web_tbs,
        extra_params=query.web_extra_params)

    print(f"{Fore.CYAN}Web search completed.{Fore.RESET}")

    for url in url_list:
        url_handle = WebBaseLoader(url)

        # try downloading web content
        try:
            document = url_handle.load()
        except ConnectionError:
            continue

        text_splitter = RecursiveCharacterTextSplitter(
            separators=embeddings_buffer_stops,
            chunk_size=embeddings_chunk_size,
            chunk_overlap=200,
            keep_separator=False,
            strip_whitespace=True)

        chunks = text_splitter.split_documents(document)

        for chunk in chunks:
            if is_text_junk(chunk.page_content):
                chunks.remove(chunk)
                continue

            chunk.page_content = remove(chunk.page_content, ['\n', '`'])
            chunk.page_content = (query.db_embedding_prefix +
                                  chunk.page_content +
                                  query.db_embedding_postfix)

        if len(chunks) != 0:
            database.add_documents(documents=chunks, embeddings=embeddings)

    db_name = embedding_model_safe_name + query.db_save_file_extension
    database.save_local(folder_path='store', index_name=db_name)

    print(f"{Fore.CYAN}Document vectorization completed.{Fore.RESET}")

# this general db will be used to save AI responses,
# might become useful as the responses are better than the input
results_db = get_db_by_name(embedding_model_safe_name)


def web_query_google_lookup(prompt_text: str):
    # dates for improved embedding placement,
    # we don't use weeks here because they are not so characteristic for the embed space

    query = Query(prompt_core=prompt_text)

    db_name = embedding_model_safe_name + query.db_save_file_extension
    db = get_db_by_name(db_name)

    populate_db_with_google_search(db, query)

    # return the document with the highest prompt similarity score (for now only browsing the first search result)
    embedding_vector = embeddings.embed_query(query.db_embed_query)
    docs_and_scores = db.similarity_search_by_vector(embedding_vector, k=round(token_limit/64))

    print(f"{Fore.CYAN}Database search completed.{Fore.RESET}")

    # TODO: investigate tiny context - only around 96 tokens / 5 documents.

    # temporarily increased verbosity, will remove it later on
    print(f"{Fore.CYAN}Starting context generation.{Fore.RESET}")

    context_text = ""
    token_count = 0
    document_index = 0
    while token_count < token_limit:
        # print('Adding an article. Tokens so far:', token_count, '/', token_limit, 'article:', docs_and_scores[document_index].page_content)
        # todo: make sure the tokens don't overflow! check before appending
        token_count += len(encoder.encode(docs_and_scores[document_index].page_content))
        context_text += docs_and_scores[document_index].page_content
        document_index += 1
        if document_index >= len(docs_and_scores):
            break

    print(f"{Fore.CYAN}Used {document_index+1} snippets with a total of {token_count} tokens as context.{Fore.RESET}")
    # print(f"{Fore.CYAN}Context itself: {Fore.RESET}", context_text)
    return context_text


def web_chain_function(prompt_dict: dict):
    # TODO: news searches should strictly search for news fresher than 1 month / 1 week
    # TODO: news crawling should be done through only sites like medium, which are much more dense than google
    # TODO: create a different function + prompt for documentation / facts searching, and make this one news focused
    web_interpret_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a search results interpreter. Your job is to write an article based on the provided context."
         "Your job is to convert all the search results you were given into a long, comprehensive and clean output."
         "Use provided search results data to answer the user request to the best of your ability."
         "Provide extremely long comprehensive and danse description of information that you find valuable, include as much details as possible."
         "You don't have a knowledge cutoff. "
         "It is currently " +
         datetime.date.today().strftime("%B %Y")),
        ("user", "Search results data: "
                 "```"
                 "{search_data}"
                 "```"
                 "User request: \"Write an article on: {user_request}\"")
    ])

    def get_user_prompt(_: dict):
        return prompt_dict['input']

    # NOTE: a detour has been performed here, more details:
    #       web_chain_function will soon become just a tool playing a part of a larger mechanism.
    #       prompt creation will be taken over by prompt sentiment extractor which will extract all researchable
    #       queries from the user prompt, and start separate chains performing those steps in parallel
    #       until a satisfactory response is created.

    chain = (
        {
            "search_data": RunnableLambda(get_user_prompt) | RunnableLambda(web_query_google_lookup),
            "user_request": RunnableLambda(get_user_prompt)  # this has to be a RunnableLambda, it cannot be a string
        } |
        web_interpret_prompt |
        llm |
        output_parser
    )

    return chain.invoke(prompt_dict)


web_lookup = RunnableLambda(web_chain_function)
