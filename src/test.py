from AI.persistent_memory import SQLDatabase, Memory


if __name__ == '__main__':
    mem = Memory()
    # mem.add_utterance(0, 'hello world!', 0)
    # mem.add_utterance(0, 'nice to meet you!', 1)
    # mem.add_utterance(0, 'I love to code.', 0)

    res = mem.search_memory(0, 'card game')

    print(mem.create_chat_string(res))
