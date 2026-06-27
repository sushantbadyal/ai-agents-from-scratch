from configs.app_config import config


def main():
    print("HELLO....")
    print("SECRETS:", config.SECRETS)
    print("LLM_CONFIGS:", config.LLM_CONFIGS)


if __name__ == "__main__":
    main()
