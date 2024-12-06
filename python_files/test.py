from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    chain,
)

add_1 = RunnableLambda(lambda x: x + 1)
add_3 = RunnableLambda(lambda x: x + 3)


@chain
def mult_2(x: int):
    return x * 2


sequence = mult_2 | add_1
parallel = mult_2 | {"add_1": add_1, "add_3": add_3}

# same as above:
parallel1 = RunnableParallel(add_1=add_1, add_3=add_3)

print(sequence.invoke(1))  # 3