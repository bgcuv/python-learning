def conditionals():
    if 1 > 0:
        print("1 is greater than 0")
    else:
        print("1 is not greater than 0")


# Change this function to print the values of x and y instead of the strings "x is greater than y" and "x is not greater than y"
def gt(x, y):
    if x > y:
        print("x is greater than y")
    else:
        print("x is not greater than y")


def main():
    conditionals()
    gt(10, 0)


if __name__ == "__main__":
    main()
