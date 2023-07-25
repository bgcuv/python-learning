def dictionary(dict):
    for key, value in dict.items():
        print(key, value)


def main():
    dict = {"a": 1, "b": 2, "c": 3}
    dictionary(dict)


if __name__ == "__main__":
    main()
