import yaml
import os


def load_openai_config():
    dir = os.path.dirname(os.path.abspath(__file__))
    OPENAI_CONFIG = yaml.load(open(dir + '/config.yaml'), Loader=yaml.FullLoader)
    if OPENAI_CONFIG['OPENAI_API_TYPE'] == 'azure':
        os.environ["OPENAI_API_TYPE"] = OPENAI_CONFIG['OPENAI_API_TYPE']
        os.environ["OPENAI_API_VERSION"] = OPENAI_CONFIG['OPENAI_API_VERSION']
        os.environ["OPENAI_API_BASE"] = OPENAI_CONFIG['OPENAI_API_BASE']
        os.environ["OPENAI_API_KEY"] = OPENAI_CONFIG['OPENAI_API_KEY']
        os.environ["EMBEDDING_MODEL"] = OPENAI_CONFIG['EMBEDDING_MODEL']
    elif OPENAI_CONFIG['OPENAI_API_TYPE'] == 'openai':
        os.environ["OPENAI_API_TYPE"] = OPENAI_CONFIG['OPENAI_API_TYPE']
        os.environ["OPENAI_API_KEY"] = OPENAI_CONFIG['REAL_OPENAI_KEY']
    else:
        raise ValueError("Unknown OPENAI_API_TYPE")

    if "LANGCHAIN_TRACING_V2" in OPENAI_CONFIG and OPENAI_CONFIG["LANGCHAIN_TRACING_V2"]:
        os.environ["LANGCHAIN_TRACING_V2"] = "1"
        os.environ["LANGCHAIN_ENDPOINT"] = OPENAI_CONFIG["LANGCHAIN_ENDPOINT"]
        os.environ["LANGCHAIN_API_KEY"] = OPENAI_CONFIG["LANGCHAIN_API_KEY"]
        os.environ["LANGCHAIN_PROJECT"] = OPENAI_CONFIG["LANGCHAIN_PROJECT"]
        print("LANGCHAIN_TRACING_V2 is enabled")

if __name__ == "__main__":
    pass